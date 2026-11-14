"""One compute kernel, written once, encoded for both GPU backends.

The frame's passes (render/gpu_raster.mojo, render/vulkan_shaders.mojo) are two
hand-written sets that must be kept in step. GPU meshing (package 1.3) has four
passes with far more arithmetic, so they are written once against this builder,
which emits Radeon ISA through `ShaderAssembler` or SPIR-V through `SpirvModule`
from the same calls in the same order: the two backends run the same steps by
construction, and `test-gpu-mesher` checks that they agree.

The model is deliberately small:

- A kernel runs in groups of 32 invocations along x (one wave32 on Radeon). `gid_x`
  is the global x index; `gid_y` is the group's y index (groups are one row high).
- Every value is a 32-bit `KVal`, an unsigned integer or a float. Booleans are
  integers that are 0 or 1, so `select` and `begin_if` take integers.
- Buffers are flat arrays of words, numbered from 0. On Radeon their addresses
  sit in a table whose address is in s[0:1]; on Vulkan buffer n is binding n.
- Push words are user SGPRs s2.. on Radeon and push constants on Vulkan.
- Mutable values are `KVar`s: a register on Radeon (lanes outside the exec mask
  keep theirs), a function variable on Vulkan. `get` copies, so a value read from a
  variable never changes behind the reader's back.
- Control flow is `begin_if`/`end_if` only; loops are unrolled when the kernel is
  built. On Radeon an if is the exec mask, saved in s20.. and skipped when empty.
- Arithmetic is IEEE single precision with no fused multiply-add, on both sides,
  so a kernel can reproduce CPU arithmetic exactly when it avoids `rsq`. On
  Vulkan that holds for sums of products only when the kernel sets
  `exact_floats` (NoContraction on every add, subtract and multiply): without
  it RADV may fuse a multiply into the add after it.
- `imul` is a 24-bit multiply on Radeon: operands must stay below 2^24.

Radeon registers: s0:1 buffer table, s2.. push words, s16 group x, s18 group y,
s17/s19 scratch, s20-s31 saved exec masks, s32-s47 buffer addresses, v1 thread x,
v3 the constant 1, values from v8 up (scoped with `mark` and `release`).
"""
from flint.platform.amd_isa import (
    ISA_GFX12, CommandBuilder, ShaderAssembler, SRC_EXEC_LO, SRC_LITERAL, SRC_TTMP7, SRC_TTMP9,
    SRC_VCC_LO, INPUT_GROUP_X, INPUT_GROUP_Y, INPUT_THREAD_XY, src_int, src_vgpr,
)
from flint.render.spirv import SpirvModule, emit, OP_DECORATE, OP_SELECT

comptime TARGET_ISA: Int = 0
comptime TARGET_SPIRV: Int = 1
comptime KERNEL_GROUP: Int = 32
comptime KERNEL_MAX_BUFFERS: Int = 8
comptime KERNEL_MAX_PUSH: Int = 14
comptime KERNEL_INPUTS: UInt32 = INPUT_GROUP_X | INPUT_GROUP_Y | INPUT_THREAD_XY
comptime FIRST_VALUE_VGPR: UInt32 = 8
comptime EXEC_STACK_SGPR: UInt32 = 20
comptime EXEC_STACK_DEPTH: Int = 12
comptime BUFFER_SGPR: UInt32 = 32
comptime DECORATION_NO_CONTRACTION: UInt32 = 42


def kernel_float_bits(value: Float32) -> UInt32:
    var copy = value
    return Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]


@fieldwise_init
struct KVal(Copyable, ImplicitlyCopyable, Movable):
    """A register on Radeon, an id on Vulkan."""
    var id: UInt32
    var is_float: Bool


@fieldwise_init
struct KVar(Copyable, ImplicitlyCopyable, Movable):
    var id: UInt32
    var is_float: Bool


struct Kernel(Movable):
    var target: Int
    var isa: Int
    var asm: ShaderAssembler
    var spv: SpirvModule
    var buffers: List[UInt32]  # Vulkan: the variable of each binding
    var push_words: Int
    var next_vgpr: UInt32
    var top_vgpr: UInt32
    var exec_labels: List[Int]
    var waits_deferred: Bool
    var gx: KVal
    var gy: KVal
    var error: String
    # Vulkan: mark every float add, subtract and multiply NoContraction, so the
    # driver's compiler neither fuses nor reorders them and the kernel rounds
    # as the Radeon encoding and the CPU do (package 3.4's skinning asks for it).
    var exact_floats: Bool

    def __init__(out self, target: Int, buffers: Int, push_words: Int, isa: Int = ISA_GFX12):
        self.target = target
        self.isa = isa
        self.asm = ShaderAssembler(isa)
        self.spv = SpirvModule(KERNEL_GROUP, 1, push_words)
        self.buffers = List[UInt32]()
        self.push_words = push_words
        self.next_vgpr = FIRST_VALUE_VGPR
        self.top_vgpr = FIRST_VALUE_VGPR
        self.exec_labels = List[Int]()
        self.waits_deferred = False
        self.gx = KVal(0, False)
        self.gy = KVal(0, False)
        self.error = String()
        self.exact_floats = False
        if buffers > KERNEL_MAX_BUFFERS or push_words > KERNEL_MAX_PUSH:
            self.error = "a kernel takes at most 8 buffers and 14 push words"
        if target == TARGET_SPIRV:
            for binding in range(buffers):
                self.buffers.append(self.spv.buffer(binding))
            self.gx = KVal(self.spv.invocation_axis(0), False)
            self.gy = KVal(self.spv.invocation_axis(1), False)
            return
        # Radeon prologue: ids, the constant 1, and the buffer addresses.
        var user = UInt32(2 + push_words)
        if isa == ISA_GFX12:
            self.asm.v_and_b32(1, SRC_LITERAL, 0)
            self.asm.literal(0x3FF)  # x id, bits 0-9 of v0
            self.asm.s_mov_b32(16, SRC_TTMP9)
            self.asm.s_and_b32(18, SRC_TTMP7, SRC_LITERAL)
            self.asm.literal(0xFFFF)
        else:
            self.asm.v_mov_b32(1, src_vgpr(0))
            self.asm.s_mov_b32(16, user)
            self.asm.s_mov_b32(18, user + 1)
        self.asm.v_mov_b32(3, src_int(1))
        self.asm.s_mov_b32(17, src_int(0))
        self.asm.s_load_b512(BUFFER_SGPR, 0, 17)
        self.asm.s_lshl_b32(17, 16, src_int(5))
        var x = self.alloc()
        self.asm.v_add_nc_u32(x, 17, 1)
        var y = self.alloc()
        self.asm.v_mov_b32(y, 18)
        self.gx = KVal(x, False)
        self.gy = KVal(y, False)
        self.asm.s_wait_kmcnt()

    def user_sgprs(self) -> UInt32:
        return UInt32(2 + self.push_words)

    def alloc(mut self) -> UInt32:
        var register = self.next_vgpr
        self.next_vgpr += 1
        if self.next_vgpr > self.top_vgpr:
            self.top_vgpr = self.next_vgpr
        if self.next_vgpr > 256:
            self.error = "kernel ran out of vector registers"
        return register

    def mark(self) -> UInt32:
        """Registers taken after this point go back with `release`."""
        return self.next_vgpr

    def release(mut self, mark: UInt32):
        if self.target == TARGET_ISA:
            self.next_vgpr = mark

    def vgprs(self) -> UInt32:
        return self.top_vgpr

    # Values.

    def gid_x(self) -> KVal:
        return self.gx

    def gid_y(self) -> KVal:
        return self.gy

    def const_u(mut self, value: UInt32) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.uint(value), False)
        var register = self.alloc()
        if value <= 64:
            self.asm.v_mov_b32(register, src_int(value))
        else:
            self.asm.v_mov_b32(register, SRC_LITERAL)
            self.asm.literal(value)
        return KVal(register, False)

    def const_f(mut self, value: Float32) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.float(value), True)
        var register = self.alloc()
        self.asm.v_mov_b32(register, SRC_LITERAL)
        self.asm.literal(kernel_float_bits(value))
        return KVal(register, True)

    def push(mut self, index: Int) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.push_word(index), False)
        var register = self.alloc()
        self.asm.v_mov_b32(register, UInt32(2 + index))
        return KVal(register, False)

    # Variables.

    def var_u(mut self, initial: KVal) -> KVar:
        if self.target == TARGET_SPIRV:
            return KVar(self.spv.uvar(initial.id), False)
        var register = self.alloc()
        self.asm.v_mov_b32(register, src_vgpr(initial.id))
        return KVar(register, False)

    def var_f(mut self, initial: KVal) -> KVar:
        if self.target == TARGET_SPIRV:
            return KVar(self.spv.fvar(initial.id), True)
        var register = self.alloc()
        self.asm.v_mov_b32(register, src_vgpr(initial.id))
        return KVar(register, True)

    def get(mut self, variable: KVar) -> KVal:
        if self.target == TARGET_SPIRV:
            if variable.is_float:
                return KVal(self.spv.load_f(variable.id), True)
            return KVal(self.spv.load_u(variable.id), False)
        var register = self.alloc()
        self.asm.v_mov_b32(register, src_vgpr(variable.id))
        return KVal(register, variable.is_float)

    def set(mut self, variable: KVar, value: KVal):
        if self.target == TARGET_SPIRV:
            self.spv.store(variable.id, value.id)
            return
        self.asm.v_mov_b32(variable.id, src_vgpr(value.id))

    # Memory. Indices count words (or bytes for load_byte) from the buffer's start.

    def byte_address(mut self, index: KVal) -> UInt32:
        var address = self.alloc()
        self.asm.v_lshlrev_b32(address, src_int(2), index.id)
        return address

    def load(mut self, buffer: Int, index: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.load_word(self.buffers[buffer], index.id), False)
        var address = self.byte_address(index)
        var register = self.alloc()
        self.asm.global_load_b32(register, address, BUFFER_SGPR + UInt32(2 * buffer))
        if not self.waits_deferred:
            self.asm.s_wait_loadcnt()
        return KVal(register, False)

    def load_float(mut self, buffer: Int, index: KVal) -> KVal:
        var word = self.load(buffer, index)
        return self.as_float(word)

    def load_byte(mut self, buffer: Int, byte_index: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            var word = self.spv.load_word(self.buffers[buffer], self.spv.shr(byte_index.id, self.spv.uint(2)))
            var shift = self.spv.shl(self.spv.bit_and(byte_index.id, self.spv.uint(3)), self.spv.uint(3))
            return KVal(self.spv.bit_and(self.spv.shr(word, shift), self.spv.uint(255)), False)
        var register = self.alloc()
        self.asm.global_load_u8(register, byte_index.id, BUFFER_SGPR + UInt32(2 * buffer))
        if not self.waits_deferred:
            self.asm.s_wait_loadcnt()
        return KVal(register, False)

    def load_byte_at(mut self, buffer: Int, byte_index: KVal, offset: Int) -> KVal:
        """The byte at byte_index + offset. On Radeon a small offset rides in the
        instruction (12 signed bits fit both generations); others take an add."""
        if self.target == TARGET_SPIRV or offset < -2048 or offset > 2047:
            if offset == 0:
                return self.load_byte(buffer, byte_index)
            # No `keep` here: with waits deferred the value is not there yet to move.
            var address = self.iadd_c(byte_index, UInt32(offset & 0xFFFFFFFF))
            return self.load_byte(buffer, address)
        var register = self.alloc()
        var encoded = UInt32(offset & 0xFFFFFF) if self.isa == ISA_GFX12 else UInt32(offset & 0xFFF)
        self.asm.global_load(0xEE040000, 8, register, byte_index.id, BUFFER_SGPR + UInt32(2 * buffer), encoded)
        if not self.waits_deferred:
            self.asm.s_wait_loadcnt()
        return KVal(register, False)

    def defer_waits(mut self):
        """Issue the loads that follow without waiting for each; `wait_loads` before
        any of them is read. Radeon only: SPIR-V loads are values at once."""
        self.waits_deferred = True

    def wait_loads(mut self):
        self.waits_deferred = False
        if self.target == TARGET_ISA:
            self.asm.s_wait_loadcnt()

    def store(mut self, buffer: Int, index: KVal, value: KVal):
        if self.target == TARGET_SPIRV:
            var word = self.spv.bitcast_uint(value.id) if value.is_float else value.id
            self.spv.store_word(self.buffers[buffer], index.id, word)
            return
        var mark = self.mark()
        var address = self.byte_address(index)
        self.asm.global_store_b32(address, value.id, BUFFER_SGPR + UInt32(2 * buffer))
        self.release(mark)

    def atomic_add(mut self, buffer: Int, index: KVal, value: KVal) -> KVal:
        """Returns the word before the add. Device scope, relaxed."""
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.atomic_add_word(self.buffers[buffer], index.id, value.id), False)
        var address = self.byte_address(index)
        var register = self.alloc()
        self.asm.global_atomic_add_u32(register, address, value.id, BUFFER_SGPR + UInt32(2 * buffer))
        self.asm.s_wait_loadcnt()
        return KVal(register, False)

    # Integer arithmetic.

    def vop2(mut self, a: KVal, b: KVal, operation: Int, is_float: Bool) -> KVal:
        var d = self.alloc()
        var x = src_vgpr(a.id)
        if operation == 0:
            self.asm.v_add_nc_u32(d, x, b.id)
        elif operation == 1:
            self.asm.v_sub_nc_u32(d, x, b.id)
        elif operation == 2:
            self.asm.v_mul_u32_u24(d, x, b.id)
        elif operation == 3:
            self.asm.v_and_b32(d, x, b.id)
        elif operation == 4:
            self.asm.v_or_b32(d, x, b.id)
        elif operation == 5:
            self.asm.v_xor_b32(d, x, b.id)
        elif operation == 6:
            self.asm.v_min_u32(d, x, b.id)
        elif operation == 7:
            self.asm.v_max_u32(d, x, b.id)
        elif operation == 8:
            self.asm.v_min_i32(d, x, b.id)
        elif operation == 9:
            self.asm.v_max_i32(d, x, b.id)
        elif operation == 10:
            self.asm.v_add_f32(d, x, b.id)
        elif operation == 11:
            self.asm.v_sub_f32(d, x, b.id)
        elif operation == 12:
            self.asm.v_mul_f32(d, x, b.id)
        elif operation == 13:
            self.asm.v_min_num_f32(d, x, b.id)
        elif operation == 14:
            self.asm.v_max_num_f32(d, x, b.id)
        return KVal(d, is_float)

    def iadd(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.iadd(a.id, b.id), False)
        return self.vop2(a, b, 0, False)

    def isub(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.isub(a.id, b.id), False)
        return self.vop2(a, b, 1, False)

    def imul(mut self, a: KVal, b: KVal) -> KVal:
        """Both operands below 2^24."""
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.imul(a.id, b.id), False)
        return self.vop2(a, b, 2, False)

    def band(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.bit_and(a.id, b.id), False)
        return self.vop2(a, b, 3, False)

    def bor(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.bit_or(a.id, b.id), False)
        return self.vop2(a, b, 4, False)

    def bxor(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.binary(198, self.spv.type_uint, a.id, b.id), False)  # OpBitwiseXor
        return self.vop2(a, b, 5, False)

    def umin(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.umin(a.id, b.id), False)
        return self.vop2(a, b, 6, False)

    def umax(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.umax(a.id, b.id), False)
        return self.vop2(a, b, 7, False)

    def smin(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.smin(a.id, b.id), False)
        return self.vop2(a, b, 8, False)

    def smax(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.smax(a.id, b.id), False)
        return self.vop2(a, b, 9, False)

    def shl(mut self, a: KVal, bits: Int) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.shl(a.id, self.spv.uint(UInt32(bits))), False)
        var d = self.alloc()
        self.asm.v_lshlrev_b32(d, src_int(UInt32(bits)), a.id)
        return KVal(d, False)

    def shr(mut self, a: KVal, bits: Int) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.shr(a.id, self.spv.uint(UInt32(bits))), False)
        var d = self.alloc()
        self.asm.v_lshrrev_b32(d, src_int(UInt32(bits)), a.id)
        return KVal(d, False)

    def sar(mut self, a: KVal, bits: Int) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.sar(a.id, self.spv.uint(UInt32(bits))), False)
        var d = self.alloc()
        self.asm.v_ashrrev_i32(d, src_int(UInt32(bits)), a.id)
        return KVal(d, False)

    def shr_by(mut self, a: KVal, bits: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.shr(a.id, bits.id), False)
        var d = self.alloc()
        self.asm.v_lshrrev_b32(d, src_vgpr(bits.id), a.id)
        return KVal(d, False)

    def iadd_c(mut self, a: KVal, value: UInt32) -> KVal:
        var mark = self.mark()
        var c = self.const_u(value)
        var result = self.iadd(a, c)
        return self.keep(mark, result)

    def imul_c(mut self, a: KVal, value: UInt32) -> KVal:
        var mark = self.mark()
        var c = self.const_u(value)
        var result = self.imul(a, c)
        return self.keep(mark, result)

    def band_c(mut self, a: KVal, value: UInt32) -> KVal:
        var mark = self.mark()
        var c = self.const_u(value)
        var result = self.band(a, c)
        return self.keep(mark, result)

    def compact(mut self, mark: UInt32, values: List[KVal]) -> List[KVal]:
        """Release everything after `mark` except `values`, moved down to it in order.
        The values must have been taken after `mark`, in increasing order."""
        if self.target == TARGET_SPIRV:
            return values.copy()
        var moved = List[KVal]()
        self.next_vgpr = mark
        for value in values:
            var register = self.alloc()
            if register != value.id:
                self.asm.v_mov_b32(register, src_vgpr(value.id))
            moved.append(KVal(register, value.is_float))
        return moved^

    def keep(mut self, mark: UInt32, value: KVal) -> KVal:
        """Release everything after `mark` except `value`, which moves down to it.
        Never on a value whose load is still in flight (see `defer_waits`)."""
        if self.target == TARGET_SPIRV:
            return value
        self.next_vgpr = mark
        var register = self.alloc()
        if register != value.id:
            self.asm.v_mov_b32(register, src_vgpr(value.id))
        return KVal(register, value.is_float)

    # Float arithmetic.

    def exact(mut self, id: UInt32) -> KVal:
        if self.exact_floats:
            emit(self.spv.decorations, OP_DECORATE, id, DECORATION_NO_CONTRACTION)
        return KVal(id, True)

    def fadd(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return self.exact(self.spv.fadd(a.id, b.id))
        return self.vop2(a, b, 10, True)

    def fsub(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return self.exact(self.spv.fsub(a.id, b.id))
        return self.vop2(a, b, 11, True)

    def fmul(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return self.exact(self.spv.fmul(a.id, b.id))
        return self.vop2(a, b, 12, True)

    def fmin(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.nmin(a.id, b.id), True)
        return self.vop2(a, b, 13, True)

    def fmax(mut self, a: KVal, b: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.nmax(a.id, b.id), True)
        return self.vop2(a, b, 14, True)

    def fmul_c(mut self, a: KVal, value: Float32) -> KVal:
        var mark = self.mark()
        var c = self.const_f(value)
        var result = self.fmul(a, c)
        return self.keep(mark, result)

    def fadd_c(mut self, a: KVal, value: Float32) -> KVal:
        var mark = self.mark()
        var c = self.const_f(value)
        var result = self.fadd(a, c)
        return self.keep(mark, result)

    def fclamp01(mut self, a: KVal) -> KVal:
        """max(0, min(1, a)), in the CPU's order."""
        var mark = self.mark()
        var one = self.const_f(1.0)
        var zero = self.const_f(0.0)
        var low = self.fmin(one, a)
        var result = self.fmax(zero, low)
        return self.keep(mark, result)

    def rsq(mut self, a: KVal) -> KVal:
        """Approximate 1/sqrt(a): not bit-exact between backends or with the CPU."""
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.inverse_sqrt(a.id), True)
        var d = self.alloc()
        self.asm.v_rsq_f32(d, src_vgpr(a.id))
        return KVal(d, True)

    def fabs(mut self, a: KVal) -> KVal:
        var mark = self.mark()
        var bits = self.as_uint(a)
        var cleared = self.band_c(bits, 0x7FFFFFFF)
        var result = self.as_float(cleared)
        return self.keep(mark, result)

    def float_of_uint(mut self, a: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.float_from_uint(a.id), True)
        var d = self.alloc()
        self.asm.v_cvt_f32_u32(d, src_vgpr(a.id))
        return KVal(d, True)

    def float_of_int(mut self, a: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.float_from_int(a.id), True)
        var d = self.alloc()
        self.asm.v_cvt_f32_i32(d, src_vgpr(a.id))
        return KVal(d, True)

    def uint_of_float(mut self, a: KVal) -> KVal:
        """Truncates toward zero; negatives and NaN give 0 on both backends."""
        if self.target == TARGET_SPIRV:
            return KVal(self.spv.saturate_uint(a.id), False)
        var d = self.alloc()
        self.asm.v_cvt_u32_f32(d, src_vgpr(a.id))
        return KVal(d, False)

    def as_float(mut self, a: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            if a.is_float:
                return a
            return KVal(self.spv.bitcast_float(a.id), True)
        return KVal(a.id, True)

    def as_uint(mut self, a: KVal) -> KVal:
        if self.target == TARGET_SPIRV:
            if not a.is_float:
                return a
            return KVal(self.spv.bitcast_uint(a.id), False)
        return KVal(a.id, False)

    # Comparisons give 0 or 1.

    def compare(mut self, a_in: KVal, b_in: KVal, operation: Int) -> KVal:
        # Integer comparisons see a float's bits (SPIR-V needs integer operands;
        # Radeon registers are just bits).
        var a = self.as_uint(a_in) if operation <= 3 else a_in
        var b = self.as_uint(b_in) if operation <= 3 else b_in
        if self.target == TARGET_SPIRV:
            var condition: UInt32
            if operation == 0:
                condition = self.spv.ult(a.id, b.id)
            elif operation == 1:
                condition = self.spv.ieq(a.id, b.id)
            elif operation == 2:
                condition = self.spv.ine(a.id, b.id)
            elif operation == 3:
                condition = self.spv.binary(177, self.spv.type_bool, a.id, b.id)  # OpSLessThan
            elif operation == 4:
                condition = self.spv.flt(a.id, b.id)
            elif operation == 5:
                condition = self.spv.fgt(a.id, b.id)
            else:
                condition = self.spv.fge(a.id, b.id)
            return KVal(self.spv.select_u(condition, self.spv.uint(1), self.spv.uint(0)), False)
        var x = src_vgpr(a.id)
        if operation == 0:
            self.asm.v_cmp_lt_u32(x, b.id)
        elif operation == 1:
            self.asm.v_cmp_eq_u32(x, b.id)
        elif operation == 2:
            self.asm.v_cmp_ne_u32(x, b.id)
        elif operation == 3:
            self.asm.v_cmp_lt_i32(x, b.id)
        elif operation == 4:
            self.asm.v_cmp_lt_f32(x, b.id)
        elif operation == 5:
            self.asm.v_cmp_gt_f32(x, b.id)
        else:
            self.asm.v_cmp_ge_f32(x, b.id)
        var d = self.alloc()
        self.asm.v_cndmask_b32(d, src_int(0), 3)
        return KVal(d, False)

    def ult(mut self, a: KVal, b: KVal) -> KVal:
        return self.compare(a, b, 0)

    def ieq(mut self, a: KVal, b: KVal) -> KVal:
        return self.compare(a, b, 1)

    def ine(mut self, a: KVal, b: KVal) -> KVal:
        return self.compare(a, b, 2)

    def slt(mut self, a: KVal, b: KVal) -> KVal:
        return self.compare(a, b, 3)

    def flt(mut self, a: KVal, b: KVal) -> KVal:
        return self.compare(a, b, 4)

    def fgt(mut self, a: KVal, b: KVal) -> KVal:
        return self.compare(a, b, 5)

    def fge(mut self, a: KVal, b: KVal) -> KVal:
        return self.compare(a, b, 6)

    def ult_c(mut self, a: KVal, value: UInt32) -> KVal:
        var mark = self.mark()
        var c = self.const_u(value)
        var result = self.ult(a, c)
        return self.keep(mark, result)

    def ine_c(mut self, a: KVal, value: UInt32) -> KVal:
        var mark = self.mark()
        var c = self.const_u(value)
        var result = self.ine(a, c)
        return self.keep(mark, result)

    def ieq_c(mut self, a: KVal, value: UInt32) -> KVal:
        var mark = self.mark()
        var c = self.const_u(value)
        var result = self.ieq(a, c)
        return self.keep(mark, result)

    def select(mut self, condition: KVal, yes: KVal, no: KVal) -> KVal:
        """yes where condition is nonzero, otherwise no."""
        if self.target == TARGET_SPIRV:
            var flag = self.spv.ine(condition.id, self.spv.uint(0))
            if yes.is_float:
                var id = self.spv.fresh()
                emit(self.spv.code, OP_SELECT, self.spv.type_float, id, flag, yes.id, no.id)
                return KVal(id, True)
            return KVal(self.spv.select_u(flag, yes.id, no.id), False)
        self.asm.v_cmp_ne_u32(src_int(0), condition.id)
        var d = self.alloc()
        self.asm.v_cndmask_b32(d, src_vgpr(no.id), yes.id)
        return KVal(d, yes.is_float)

    # Control flow.

    def begin_if(mut self, condition: KVal):
        if self.target == TARGET_SPIRV:
            self.spv.begin_if(self.spv.ine(condition.id, self.spv.uint(0)))
            return
        var depth = len(self.exec_labels)
        if depth >= EXEC_STACK_DEPTH:
            self.error = "kernel ifs nest too deep"
            return
        self.asm.v_cmp_ne_u32(src_int(0), condition.id)
        self.asm.s_mov_b32(EXEC_STACK_SGPR + UInt32(depth), SRC_EXEC_LO)
        self.asm.s_and_b32(SRC_EXEC_LO, SRC_EXEC_LO, SRC_VCC_LO)
        var label = self.asm.new_label()
        self.asm.s_cbranch_execz(label)
        self.exec_labels.append(label)

    def end_if(mut self):
        if self.target == TARGET_SPIRV:
            self.spv.end_if()
            return
        var label = self.exec_labels.pop()
        self.asm.bind(label)
        self.asm.s_mov_b32(SRC_EXEC_LO, EXEC_STACK_SGPR + UInt32(len(self.exec_labels)))

    # Output.

    def finish_isa(mut self) raises -> ShaderAssembler:
        if len(self.exec_labels) != 0:
            raise Error("kernel has an unclosed if")
        if self.error.byte_length() > 0:
            raise Error(self.error)
        self.asm.s_endpgm()
        self.asm.resolve()
        if self.asm.far_branches > 0:
            raise Error("kernel has " + String(self.asm.far_branches) + " branches longer than 32767 dwords (" + String(len(self.asm.words)) + " dwords in all)")
        var result = ShaderAssembler(self.isa)
        swap(result, self.asm)
        return result^

    def finish_spirv(mut self) raises -> List[UInt32]:
        if self.error.byte_length() > 0:
            raise Error(self.error)
        return self.spv.finish()


def kernel_compute_state(mut builder: CommandBuilder, shader_address: UInt64, kernel_vgprs: UInt32, push_words: Int):
    """Program a dispatch of a kernel built here: 32 threads a group."""
    builder.compute_state(shader_address, UInt32(KERNEL_GROUP), 1, 1, UInt32(2 + push_words), kernel_vgprs, KERNEL_INPUTS)

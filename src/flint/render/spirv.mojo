"""A SPIR-V emitter for compute shaders, in the spirit of platform/amd_isa.mojo.

Shaders are written as Mojo calls that append instructions, with no SDK and no
compiler in between. A module targets SPIR-V 1.3 and Vulkan 1.2 with only the
`Shader` capability. Everything is scalar: 32-bit unsigned integers, 32-bit
floats and booleans. Storage buffers are flat arrays of uint, so a shader
addresses memory the way the amdgpu shaders do, in words from a binding's start.

Values are ids, as in SPIR-V itself: `s.fadd(a, b)` emits an instruction and
returns its result. There is one integer type, because SPIR-V's signed
operations do not care about the signedness of the type. Mutable values live in
function variables (`uvar`, `fvar`, `load_u`, `load_f`, `store`), which keeps the
emitter free of phi nodes; the driver promotes them.

Control flow is structured, which SPIR-V requires:

    s.begin_if(condition) ... s.begin_else() ... s.end_if()
    s.begin_loop()
    s.break_if(done)
    s.continue_if(skip)
    s.end_loop()
    s.return_if(condition)
"""
from std.collections import Dict

comptime SPIRV_MAGIC: UInt32 = 0x07230203
comptime SPIRV_VERSION_1_3: UInt32 = 0x00010300

comptime OP_EXT_INST_IMPORT: UInt32 = 11
comptime OP_EXT_INST: UInt32 = 12
comptime OP_MEMORY_MODEL: UInt32 = 14
comptime OP_ENTRY_POINT: UInt32 = 15
comptime OP_EXECUTION_MODE: UInt32 = 16
comptime OP_CAPABILITY: UInt32 = 17
comptime OP_TYPE_VOID: UInt32 = 19
comptime OP_TYPE_BOOL: UInt32 = 20
comptime OP_TYPE_INT: UInt32 = 21
comptime OP_TYPE_FLOAT: UInt32 = 22
comptime OP_TYPE_VECTOR: UInt32 = 23
comptime OP_TYPE_ARRAY: UInt32 = 28
comptime OP_TYPE_RUNTIME_ARRAY: UInt32 = 29
comptime OP_TYPE_STRUCT: UInt32 = 30
comptime OP_TYPE_POINTER: UInt32 = 32
comptime OP_TYPE_FUNCTION: UInt32 = 33
comptime OP_CONSTANT_TRUE: UInt32 = 41
comptime OP_CONSTANT_FALSE: UInt32 = 42
comptime OP_CONSTANT: UInt32 = 43
comptime OP_SPEC_CONSTANT_TRUE: UInt32 = 48
comptime OP_SPEC_CONSTANT_FALSE: UInt32 = 49
comptime OP_SPEC_CONSTANT: UInt32 = 50
comptime OP_FUNCTION: UInt32 = 54
comptime OP_FUNCTION_END: UInt32 = 56
comptime OP_VARIABLE: UInt32 = 59
comptime OP_LOAD: UInt32 = 61
comptime OP_STORE: UInt32 = 62
comptime OP_ACCESS_CHAIN: UInt32 = 65
comptime OP_DECORATE: UInt32 = 71
comptime OP_MEMBER_DECORATE: UInt32 = 72
comptime OP_CONVERT_F_TO_U: UInt32 = 109
comptime OP_CONVERT_F_TO_S: UInt32 = 110
comptime OP_CONVERT_S_TO_F: UInt32 = 111
comptime OP_CONVERT_U_TO_F: UInt32 = 112
comptime OP_BITCAST: UInt32 = 124
comptime OP_CONTROL_BARRIER: UInt32 = 224
comptime OP_MEMORY_BARRIER: UInt32 = 225
comptime OP_ATOMIC_I_ADD: UInt32 = 234
comptime OP_ATOMIC_U_MAX: UInt32 = 239
comptime OP_I_ADD: UInt32 = 128
comptime OP_F_ADD: UInt32 = 129
comptime OP_I_SUB: UInt32 = 130
comptime OP_F_SUB: UInt32 = 131
comptime OP_I_MUL: UInt32 = 132
comptime OP_F_MUL: UInt32 = 133
comptime OP_F_DIV: UInt32 = 136
comptime OP_LOGICAL_OR: UInt32 = 166
comptime OP_LOGICAL_AND: UInt32 = 167
comptime OP_LOGICAL_NOT: UInt32 = 168
comptime OP_SELECT: UInt32 = 169
comptime OP_I_EQUAL: UInt32 = 170
comptime OP_I_NOT_EQUAL: UInt32 = 171
comptime OP_U_GREATER_THAN_EQUAL: UInt32 = 174
comptime OP_U_LESS_THAN: UInt32 = 176
comptime OP_F_ORD_LESS_THAN: UInt32 = 184
comptime OP_F_ORD_GREATER_THAN: UInt32 = 186
comptime OP_F_ORD_GREATER_THAN_EQUAL: UInt32 = 190
comptime OP_SHIFT_RIGHT_LOGICAL: UInt32 = 194
comptime OP_SHIFT_RIGHT_ARITHMETIC: UInt32 = 195
comptime OP_SHIFT_LEFT_LOGICAL: UInt32 = 196
comptime OP_BITWISE_OR: UInt32 = 197
comptime OP_BITWISE_XOR: UInt32 = 198
comptime OP_BITWISE_AND: UInt32 = 199
comptime OP_LOOP_MERGE: UInt32 = 246
comptime OP_SELECTION_MERGE: UInt32 = 247
comptime OP_LABEL: UInt32 = 248
comptime OP_BRANCH: UInt32 = 249
comptime OP_BRANCH_CONDITIONAL: UInt32 = 250
comptime OP_KILL: UInt32 = 252
comptime OP_RETURN: UInt32 = 253

comptime GLSL_F_ABS: UInt32 = 4
comptime GLSL_FLOOR: UInt32 = 8
comptime GLSL_FRACT: UInt32 = 10
comptime GLSL_EXP2: UInt32 = 29
comptime GLSL_LOG2: UInt32 = 30
comptime GLSL_SQRT: UInt32 = 31
comptime GLSL_INVERSE_SQRT: UInt32 = 32
comptime GLSL_U_MIN: UInt32 = 38
comptime GLSL_S_MIN: UInt32 = 39
comptime GLSL_U_MAX: UInt32 = 41
comptime GLSL_S_MAX: UInt32 = 42
comptime GLSL_FMA: UInt32 = 50
comptime GLSL_FIND_I_LSB: UInt32 = 73
comptime GLSL_N_MIN: UInt32 = 79
comptime GLSL_N_MAX: UInt32 = 80
comptime GLSL_N_CLAMP: UInt32 = 81

comptime CAPABILITY_SHADER: UInt32 = 1
comptime ADDRESSING_LOGICAL: UInt32 = 0
comptime MEMORY_GLSL450: UInt32 = 1
comptime MODEL_GL_COMPUTE: UInt32 = 5
comptime MODE_LOCAL_SIZE: UInt32 = 17

comptime STORAGE_INPUT: UInt32 = 1
comptime STORAGE_FUNCTION: UInt32 = 7
comptime STORAGE_PUSH_CONSTANT: UInt32 = 9
comptime STORAGE_BUFFER: UInt32 = 12

comptime DEC_SPEC_ID: UInt32 = 1
comptime DECORATION_SPEC_ID: UInt32 = 1
comptime DECORATION_BLOCK: UInt32 = 2
comptime DECORATION_ARRAY_STRIDE: UInt32 = 6
comptime DECORATION_BUILT_IN: UInt32 = 11
comptime DECORATION_BINDING: UInt32 = 33
comptime DECORATION_DESCRIPTOR_SET: UInt32 = 34
comptime DECORATION_OFFSET: UInt32 = 35
comptime BUILT_IN_GLOBAL_INVOCATION_ID: UInt32 = 28


def string_words(text: String) -> List[UInt32]:
    """A nul-terminated UTF-8 string packed little-endian into words."""
    var bytes = text.as_bytes()
    var words = List[UInt32](length=(len(bytes) + 4) // 4, fill=0)
    for index in range(len(bytes)):
        words[index // 4] |= UInt32(bytes[index]) << UInt32((index % 4) * 8)
    return words^


def emit(mut target: List[UInt32], opcode: UInt32, *operands: UInt32):
    target.append(opcode | (UInt32(len(operands) + 1) << 16))
    for operand in operands:
        target.append(operand)


struct LoopLabels(Copyable, Movable):
    var header: UInt32
    var next: UInt32  # the continue target
    var merge: UInt32

    def __init__(out self, header: UInt32, next: UInt32, merge: UInt32):
        self.header = header
        self.next = next
        self.merge = merge


struct IfLabels(Copyable, Movable):
    var merge: UInt32
    var otherwise: UInt32

    def __init__(out self, merge: UInt32, otherwise: UInt32):
        self.merge = merge
        self.otherwise = otherwise


struct SpirvModule(Movable):
    """One compute entry point called main, under construction."""
    var bound: UInt32
    var decorations: List[UInt32]
    var globals: List[UInt32]
    var variables: List[UInt32]
    var code: List[UInt32]
    var constants: Dict[Int, UInt32]
    var loops: List[LoopLabels]
    var ifs: List[IfLabels]
    var type_void: UInt32
    var type_bool: UInt32
    var type_uint: UInt32
    var type_float: UInt32
    var type_uvec3: UInt32
    var type_function: UInt32
    var pointer_buffer_uint: UInt32
    var pointer_function_uint: UInt32
    var pointer_function_float: UInt32
    var pointer_push_uint: UInt32
    var pointer_input_uvec3: UInt32
    var buffer_pointer: UInt32
    var glsl: UInt32
    var main: UInt32
    var invocation: UInt32
    var local_invoc_var: UInt32
    var workgroup_id_var: UInt32
    var push: UInt32
    var group_width: UInt32
    var group_height: UInt32

    def __init__(out self, group_width: Int, group_height: Int, push_words: Int):
        self.bound = 1
        self.decorations = List[UInt32]()
        self.globals = List[UInt32]()
        self.variables = List[UInt32]()
        self.code = List[UInt32]()
        self.constants = Dict[Int, UInt32]()
        self.loops = List[LoopLabels]()
        self.ifs = List[IfLabels]()
        self.type_void = 0
        self.type_bool = 0
        self.type_uint = 0
        self.type_float = 0
        self.type_uvec3 = 0
        self.type_function = 0
        self.pointer_buffer_uint = 0
        self.pointer_function_uint = 0
        self.pointer_function_float = 0
        self.pointer_push_uint = 0
        self.pointer_input_uvec3 = 0
        self.buffer_pointer = 0
        self.glsl = 0
        self.main = 0
        self.invocation = 0
        self.local_invoc_var = 0
        self.workgroup_id_var = 0
        self.push = 0
        self.group_width = UInt32(group_width)
        self.group_height = UInt32(group_height)
        self.declare_types(push_words)
        var first = self.fresh()
        emit(self.code, OP_LABEL, first)

    def fresh(mut self) -> UInt32:
        var id = self.bound
        self.bound += 1
        return id

    def declare_types(mut self, push_words: Int):
        self.glsl = self.fresh()
        self.main = self.fresh()
        self.type_void = self.fresh()
        self.type_function = self.fresh()
        self.type_bool = self.fresh()
        self.type_uint = self.fresh()
        self.type_float = self.fresh()
        self.type_uvec3 = self.fresh()
        emit(self.globals, OP_TYPE_VOID, self.type_void)
        emit(self.globals, OP_TYPE_FUNCTION, self.type_function, self.type_void)
        emit(self.globals, OP_TYPE_BOOL, self.type_bool)
        emit(self.globals, OP_TYPE_INT, self.type_uint, 32, 0)
        emit(self.globals, OP_TYPE_FLOAT, self.type_float, 32)
        emit(self.globals, OP_TYPE_VECTOR, self.type_uvec3, self.type_uint, 3)

        # One buffer type for every binding: struct { uint words[]; }.
        var runtime = self.fresh()
        var block = self.fresh()
        self.buffer_pointer = self.fresh()
        self.pointer_buffer_uint = self.fresh()
        emit(self.globals, OP_TYPE_RUNTIME_ARRAY, runtime, self.type_uint)
        emit(self.globals, OP_TYPE_STRUCT, block, runtime)
        emit(self.globals, OP_TYPE_POINTER, self.buffer_pointer, STORAGE_BUFFER, block)
        emit(self.globals, OP_TYPE_POINTER, self.pointer_buffer_uint, STORAGE_BUFFER, self.type_uint)
        emit(self.decorations, OP_DECORATE, runtime, DECORATION_ARRAY_STRIDE, 4)
        emit(self.decorations, OP_DECORATE, block, DECORATION_BLOCK)
        emit(self.decorations, OP_MEMBER_DECORATE, block, 0, DECORATION_OFFSET, 0)

        self.pointer_function_uint = self.fresh()
        self.pointer_function_float = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, self.pointer_function_uint, STORAGE_FUNCTION, self.type_uint)
        emit(self.globals, OP_TYPE_POINTER, self.pointer_function_float, STORAGE_FUNCTION, self.type_float)

        self.pointer_input_uvec3 = self.fresh()
        self.invocation = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, self.pointer_input_uvec3, STORAGE_INPUT, self.type_uvec3)
        emit(self.globals, OP_VARIABLE, self.pointer_input_uvec3, self.invocation, STORAGE_INPUT)
        emit(self.decorations, OP_DECORATE, self.invocation, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID)

        if push_words > 0:
            var array = self.fresh()
            var push_block = self.fresh()
            var push_pointer = self.fresh()
            self.pointer_push_uint = self.fresh()
            self.push = self.fresh()
            var length = self.uint(UInt32(push_words))
            emit(self.globals, OP_TYPE_ARRAY, array, self.type_uint, length)
            emit(self.globals, OP_TYPE_STRUCT, push_block, array)
            emit(self.globals, OP_TYPE_POINTER, push_pointer, STORAGE_PUSH_CONSTANT, push_block)
            emit(self.globals, OP_TYPE_POINTER, self.pointer_push_uint, STORAGE_PUSH_CONSTANT, self.type_uint)
            emit(self.globals, OP_VARIABLE, push_pointer, self.push, STORAGE_PUSH_CONSTANT)
            emit(self.decorations, OP_DECORATE, array, DECORATION_ARRAY_STRIDE, 4)
            emit(self.decorations, OP_DECORATE, push_block, DECORATION_BLOCK)
            emit(self.decorations, OP_MEMBER_DECORATE, push_block, 0, DECORATION_OFFSET, 0)

    # Constants. Both kinds share a dictionary, told apart by bit 32.

    def uint(mut self, value: UInt32) -> UInt32:
        var key = Int(value)
        var known = self.constants.get(key)
        if known:
            return known.value()
        var id = self.fresh()
        emit(self.globals, OP_CONSTANT, self.type_uint, id, value)
        self.constants[key] = id
        return id

    def float_bits(mut self, bits: UInt32) -> UInt32:
        var key = Int(bits) | (1 << 32)
        var known = self.constants.get(key)
        if known:
            return known.value()
        var id = self.fresh()
        emit(self.globals, OP_CONSTANT, self.type_float, id, bits)
        self.constants[key] = id
        return id

    def float(mut self, value: Float32) -> UInt32:
        var copy = value
        return self.float_bits(Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0])

    def spec_constant_u32(mut self, spec_id: Int, default_val: UInt32 = 0) -> UInt32:
        """Emits `OpSpecConstant` (`UInt32`) decorated with `SpecId = spec_id`."""
        var id = self.fresh()
        emit(self.decorations, OP_DECORATE, id, DEC_SPEC_ID, UInt32(spec_id))
        emit(self.globals, OP_SPEC_CONSTANT, self.type_uint, id, default_val)
        return id

    def spec_constant_bool(mut self, spec_id: Int, default_val: Bool = False) -> UInt32:
        """Emits `OpSpecConstantTrue` / `OpSpecConstantFalse` decorated with `SpecId = spec_id`."""
        var id = self.fresh()
        var op = OP_SPEC_CONSTANT_TRUE if default_val else OP_SPEC_CONSTANT_FALSE
        emit(self.decorations, OP_DECORATE, id, DEC_SPEC_ID, UInt32(spec_id))
        emit(self.globals, op, self.type_bool, id)
        return id

    def spec_constant_f32(mut self, spec_id: Int, default_val: Float32 = 0.0) -> UInt32:
        """Emits `OpSpecConstant` (`Float32`) decorated with `SpecId = spec_id`."""
        var id = self.fresh()
        var copy = default_val
        var bits = Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]
        emit(self.decorations, OP_DECORATE, id, DEC_SPEC_ID, UInt32(spec_id))
        emit(self.globals, OP_SPEC_CONSTANT, self.type_float, id, bits)
        return id

    # Resources.

    def buffer(mut self, binding: Int) -> UInt32:
        var id = self.fresh()
        emit(self.globals, OP_VARIABLE, self.buffer_pointer, id, STORAGE_BUFFER)
        emit(self.decorations, OP_DECORATE, id, DECORATION_DESCRIPTOR_SET, 0)
        emit(self.decorations, OP_DECORATE, id, DECORATION_BINDING, UInt32(binding))
        return id

    def load_word(mut self, buffer: UInt32, index: UInt32) -> UInt32:
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, pointer)
        return id

    def load_float(mut self, buffer: UInt32, index: UInt32) -> UInt32:
        return self.bitcast_float(self.load_word(buffer, index))

    def store_word(mut self, buffer: UInt32, index: UInt32, value: UInt32):
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        emit(self.code, OP_STORE, pointer, value)

    def atomic_add_word(mut self, buffer: UInt32, index: UInt32, value: UInt32) -> UInt32:
        """Add to a word at device scope with relaxed ordering; returns the old value."""
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        var device_scope = self.uint(1)
        var relaxed = self.uint(0)
        var id = self.fresh()
        emit(self.code, OP_ATOMIC_I_ADD, self.type_uint, id, pointer, device_scope, relaxed, value)
        return id

    def atomic_umax_word(mut self, buffer: UInt32, index: UInt32, value: UInt32) -> UInt32:
        """Atomic unsigned max on a word at device scope with relaxed ordering; returns the old value."""
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        var device_scope = self.uint(1)
        var relaxed = self.uint(0)
        var id = self.fresh()
        emit(self.code, OP_ATOMIC_U_MAX, self.type_uint, id, pointer, device_scope, relaxed, value)
        return id

    def push_word(mut self, index: Int) -> UInt32:
        var pointer = self.fresh()
        var zero = self.uint(0)
        var word = self.uint(UInt32(index))
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_push_uint, pointer, self.push, zero, word)
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, pointer)
        return id

    def push_float(mut self, index: Int) -> UInt32:
        return self.bitcast_float(self.push_word(index))

    def invocation_axis(mut self, axis: Int) -> UInt32:
        var vector = self.fresh()
        emit(self.code, OP_LOAD, self.type_uvec3, vector, self.invocation)
        var id = self.fresh()
        emit(self.code, 81, self.type_uint, id, vector, UInt32(axis))  # OpCompositeExtract
        return id

    # Mutable values.

    def uvar(mut self, initial: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.variables, OP_VARIABLE, self.pointer_function_uint, id, STORAGE_FUNCTION)
        self.store(id, initial)
        return id

    def fvar(mut self, initial: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.variables, OP_VARIABLE, self.pointer_function_float, id, STORAGE_FUNCTION)
        self.store(id, initial)
        return id

    def load_u(mut self, variable: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, variable)
        return id

    def load_f(mut self, variable: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_float, id, variable)
        return id

    def store(mut self, variable: UInt32, value: UInt32):
        emit(self.code, OP_STORE, variable, value)

    # Arithmetic.

    def binary(mut self, opcode: UInt32, type: UInt32, a: UInt32, b: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, opcode, type, id, a, b)
        return id

    def unary(mut self, opcode: UInt32, type: UInt32, a: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, opcode, type, id, a)
        return id

    def ext1(mut self, operation: UInt32, type: UInt32, a: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a)
        return id

    def ext2(mut self, operation: UInt32, type: UInt32, a: UInt32, b: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a, b)
        return id

    def ext3(mut self, operation: UInt32, type: UInt32, a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a, b, c)
        return id

    def iadd(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_ADD, self.type_uint, a, b)

    def isub(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_SUB, self.type_uint, a, b)

    def imul(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_MUL, self.type_uint, a, b)

    def bit_and(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_AND, self.type_uint, a, b)

    def bit_or(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_OR, self.type_uint, a, b)

    def bit_xor(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_XOR, self.type_uint, a, b)

    def shl(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_LEFT_LOGICAL, self.type_uint, a, b)

    def shr(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_RIGHT_LOGICAL, self.type_uint, a, b)

    def sar(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_RIGHT_ARITHMETIC, self.type_uint, a, b)

    def umin(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_U_MIN, self.type_uint, a, b)

    def umax(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_U_MAX, self.type_uint, a, b)

    def smin(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_S_MIN, self.type_uint, a, b)

    def smax(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_S_MAX, self.type_uint, a, b)

    def fadd(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ADD, self.type_float, a, b)

    def fsub(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_SUB, self.type_float, a, b)

    def fmul(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_MUL, self.type_float, a, b)

    def fdiv(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_DIV, self.type_float, a, b)

    def fma(mut self, a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        return self.ext3(GLSL_FMA, self.type_float, a, b, c)

    def fract(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_FRACT, self.type_float, a)

    def fabs(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_F_ABS, self.type_float, a)

    def ffloor(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_FLOOR, self.type_float, a)

    def exp2(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_EXP2, self.type_float, a)

    def log2(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_LOG2, self.type_float, a)

    def fsqrt(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_SQRT, self.type_float, a)

    def inverse_sqrt(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_INVERSE_SQRT, self.type_float, a)

    def nmax(mut self, a: UInt32, b: UInt32) -> UInt32:
        """Maximum that ignores a NaN operand, as v_max_num_f32 does."""
        return self.ext2(GLSL_N_MAX, self.type_float, a, b)

    def nmin(mut self, a: UInt32, b: UInt32) -> UInt32:
        """Minimum that ignores a NaN operand, as v_min_num_f32 does."""
        return self.ext2(GLSL_N_MIN, self.type_float, a, b)

    def nclamp(mut self, a: UInt32, low: UInt32, high: UInt32) -> UInt32:
        return self.ext3(GLSL_N_CLAMP, self.type_float, a, low, high)

    def find_lsb(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_FIND_I_LSB, self.type_uint, a)

    def float_from_uint(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_U_TO_F, self.type_float, a)

    def float_from_int(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_S_TO_F, self.type_float, a)

    def uint_from_float(mut self, a: UInt32) -> UInt32:
        """Truncates. The operand must already be in range: see `saturate_uint`."""
        return self.unary(OP_CONVERT_F_TO_U, self.type_uint, a)

    def int_from_float(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_F_TO_S, self.type_uint, a)

    def saturate_uint(mut self, a: UInt32) -> UInt32:
        """Like v_cvt_u32_f32: NaN and negatives give 0. A conversion out of range is
        undefined in SPIR-V, so the clamp is what makes this portable."""
        return self.uint_from_float(self.nclamp(a, self.float(0), self.float(4294967040.0)))

    def bitcast_float(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_BITCAST, self.type_float, a)

    def bitcast_uint(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_BITCAST, self.type_uint, a)

    def select_u(mut self, condition: UInt32, yes: UInt32, no: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_SELECT, self.type_uint, id, condition, yes, no)
        return id

    def select_f(mut self, condition: UInt32, yes: UInt32, no: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_SELECT, self.type_float, id, condition, yes, no)
        return id

    # Comparisons give booleans.

    def ieq(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_EQUAL, self.type_bool, a, b)

    def ine(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_NOT_EQUAL, self.type_bool, a, b)

    def ult(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_U_LESS_THAN, self.type_bool, a, b)

    def uge(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_U_GREATER_THAN_EQUAL, self.type_bool, a, b)

    def flt(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_LESS_THAN, self.type_bool, a, b)

    def fgt(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_GREATER_THAN, self.type_bool, a, b)

    def fge(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_GREATER_THAN_EQUAL, self.type_bool, a, b)

    def logical_and(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_LOGICAL_AND, self.type_bool, a, b)

    def logical_or(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_LOGICAL_OR, self.type_bool, a, b)

    def logical_not(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_LOGICAL_NOT, self.type_bool, a)

    # Structured control flow.

    def label(mut self, id: UInt32):
        emit(self.code, OP_LABEL, id)

    def begin_if(mut self, condition: UInt32):
        var yes = self.fresh()
        var otherwise = self.fresh()
        var merge = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, merge, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, yes, otherwise)
        self.label(yes)
        self.ifs.append(IfLabels(merge, otherwise))

    def begin_else(mut self):
        var top = self.ifs[len(self.ifs) - 1].copy()
        emit(self.code, OP_BRANCH, top.merge)
        self.label(top.otherwise)
        self.ifs[len(self.ifs) - 1].otherwise = 0

    def end_if(mut self):
        var top = self.ifs.pop()
        emit(self.code, OP_BRANCH, top.merge)
        if top.otherwise != 0:
            self.label(top.otherwise)
            emit(self.code, OP_BRANCH, top.merge)
        self.label(top.merge)

    def begin_loop(mut self):
        var header = self.fresh()
        var body = self.fresh()
        var next = self.fresh()
        var merge = self.fresh()
        emit(self.code, OP_BRANCH, header)
        self.label(header)
        emit(self.code, OP_LOOP_MERGE, merge, next, 0)
        emit(self.code, OP_BRANCH, body)
        self.label(body)
        self.loops.append(LoopLabels(header, next, merge))

    def leave_if(mut self, condition: UInt32, target: UInt32):
        var carry_on = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, carry_on, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, target, carry_on)
        self.label(carry_on)

    def break_if(mut self, condition: UInt32):
        self.leave_if(condition, self.loops[len(self.loops) - 1].merge)

    def continue_if(mut self, condition: UInt32):
        self.leave_if(condition, self.loops[len(self.loops) - 1].next)

    def end_loop(mut self):
        var top = self.loops.pop()
        emit(self.code, OP_BRANCH, top.next)
        self.label(top.next)
        emit(self.code, OP_BRANCH, top.header)
        self.label(top.merge)

    def return_if(mut self, condition: UInt32):
        var leave = self.fresh()
        var carry_on = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, carry_on, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, leave, carry_on)
        self.label(leave)
        emit(self.code, OP_RETURN)
        self.label(carry_on)

    def discard_if(mut self, condition: UInt32):
        """Fragment-stage alpha-cutout discard (`OpKill`) inside structured selection merge."""
        var kill_blk = self.fresh()
        var carry_on = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, carry_on, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, kill_blk, carry_on)
        self.label(kill_blk)
        emit(self.code, OP_KILL)
        self.label(carry_on)

    def finish(mut self) -> List[UInt32]:
        """Close the entry point and lay the module out in the order SPIR-V requires."""
        emit(self.code, OP_RETURN)
        emit(self.code, OP_FUNCTION_END)
        var words = List[UInt32]()
        words.append(SPIRV_MAGIC)
        words.append(SPIRV_VERSION_1_3)
        words.append(0)
        words.append(self.bound)
        words.append(0)
        emit(words, OP_CAPABILITY, CAPABILITY_SHADER)
        var import_name = string_words("GLSL.std.450")
        words.append(OP_EXT_INST_IMPORT | (UInt32(len(import_name) + 2) << 16))
        words.append(self.glsl)
        words.extend(import_name^)
        emit(words, OP_MEMORY_MODEL, ADDRESSING_LOGICAL, MEMORY_GLSL450)
        var entry_name = string_words("main")
        # SPIR-V 1.3 lists only Input and Output variables on the entry point.
        words.append(OP_ENTRY_POINT | (UInt32(len(entry_name) + 4) << 16))
        words.append(MODEL_GL_COMPUTE)
        words.append(self.main)
        words.extend(entry_name^)
        words.append(self.invocation)
        emit(words, OP_EXECUTION_MODE, self.main, MODE_LOCAL_SIZE, self.group_width, self.group_height, 1)
        words.extend(self.decorations.copy())
        words.extend(self.globals.copy())
        emit(words, OP_FUNCTION, self.type_void, self.main, 0, self.type_function)
        # A function's variables sit at the top of its first block.
        words.append(self.code[0])
        words.append(self.code[1])
        words.extend(self.variables.copy())
        for index in range(2, len(self.code)):
            words.append(self.code[index])
        return words^


comptime SpirvBuilder = SpirvModule


# ============================================================================
# AAA SPIR-V 1.5 Emitter: Graphics (Vertex/Fragment), Bindless BC7/BC5/BC6H
# Sampled Images (`Grad` / `Lod` / `Fetch`), Cubemaps & `SPV_KHR_ray_query`
# ============================================================================

comptime SPIRV_VERSION_1_5: UInt32 = 0x00010500

comptime OP_EXTENSION: UInt32 = 10
comptime OP_TYPE_IMAGE: UInt32 = 25
comptime OP_TYPE_SAMPLER: UInt32 = 26
comptime OP_TYPE_SAMPLED_IMAGE: UInt32 = 27
comptime OP_CONSTANT_COMPOSITE: UInt32 = 44
comptime OP_COMPOSITE_CONSTRUCT: UInt32 = 80
comptime OP_COMPOSITE_EXTRACT: UInt32 = 81
comptime OP_IMAGE_SAMPLE_IMPLICIT_LOD: UInt32 = 87
comptime OP_IMAGE_SAMPLE_EXPLICIT_LOD: UInt32 = 88
comptime OP_IMAGE_FETCH: UInt32 = 95
comptime OP_IMAGE_READ: UInt32 = 98
comptime OP_IMAGE_WRITE: UInt32 = 99
comptime OP_IMAGE: UInt32 = 100
comptime OP_F_NEGATE: UInt32 = 127
comptime OP_U_DIV: UInt32 = 134
comptime OP_U_MOD: UInt32 = 139
comptime OP_DPDX: UInt32 = 207
comptime OP_DPDY: UInt32 = 208

comptime OP_TYPE_ACCELERATION_STRUCTURE_KHR: UInt32 = 5341
comptime OP_TYPE_RAY_QUERY_KHR: UInt32 = 4472
comptime OP_RAY_QUERY_INITIALIZE_KHR: UInt32 = 4473
comptime OP_RAY_QUERY_TERMINATE_KHR: UInt32 = 4474
comptime OP_RAY_QUERY_PROCEED_KHR: UInt32 = 4477
comptime OP_RAY_QUERY_CONFIRM_INTERSECTION_KHR: UInt32 = 4478
comptime OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR: UInt32 = 4479
comptime OP_RAY_QUERY_GET_INTERSECTION_T_KHR: UInt32 = 6018
comptime OP_RAY_QUERY_GET_INTERSECTION_INSTANCE_CUSTOM_INDEX_KHR: UInt32 = 6019
comptime OP_RAY_QUERY_GET_INTERSECTION_INSTANCE_ID_KHR: UInt32 = 6020
comptime OP_RAY_QUERY_GET_INTERSECTION_SBT_OFFSET_KHR: UInt32 = 6021
comptime OP_RAY_QUERY_GET_INTERSECTION_GEOMETRY_INDEX_KHR: UInt32 = 6022
comptime OP_RAY_QUERY_GET_INTERSECTION_PRIMITIVE_INDEX_KHR: UInt32 = 6023
comptime OP_RAY_QUERY_GET_INTERSECTION_BARYCENTRICS_KHR: UInt32 = 6024
comptime OP_RAY_QUERY_GET_INTERSECTION_FRONT_FACE_KHR: UInt32 = 6025

comptime GLSL_SIN: UInt32 = 13
comptime GLSL_COS: UInt32 = 14
comptime GLSL_ATAN2: UInt32 = 25
comptime GLSL_POW: UInt32 = 26
comptime GLSL_PACK_HALF2X16: UInt32 = 58
comptime GLSL_UNPACK_HALF2X16: UInt32 = 62

comptime IMAGE_OPERANDS_LOD: UInt32 = 2
comptime IMAGE_OPERANDS_GRAD: UInt32 = 4

comptime MODEL_VERTEX: UInt32 = 0
comptime MODEL_FRAGMENT: UInt32 = 4

comptime MODE_ORIGIN_UPPER_LEFT: UInt32 = 7
comptime MODE_EARLY_FRAGMENT_TESTS: UInt32 = 9

comptime STORAGE_UNIFORM_CONSTANT: UInt32 = 0
comptime STORAGE_OUTPUT: UInt32 = 3

comptime DECORATION_FLAT: UInt32 = 14
comptime DECORATION_LOCATION: UInt32 = 30
comptime DECORATION_NON_UNIFORM: UInt32 = 5300

comptime BUILT_IN_POSITION: UInt32 = 0
comptime BUILT_IN_FRAG_COORD: UInt32 = 15
comptime BUILT_IN_VERTEX_INDEX: UInt32 = 42
comptime BUILT_IN_INSTANCE_INDEX: UInt32 = 43

comptime CAPABILITY_IMAGE_QUERY: UInt32 = 50
comptime CAPABILITY_SHADER_NON_UNIFORM: UInt32 = 5301
comptime CAPABILITY_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING: UInt32 = 5307
comptime CAPABILITY_RAY_QUERY_KHR: UInt32 = 4472
comptime CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS: UInt32 = 49
comptime CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC: UInt32 = 63
comptime CAPABILITY_GROUP_NON_UNIFORM_BALLOT: UInt32 = 64
comptime CAPABILITY_GEOMETRY: UInt32 = 6

comptime STORAGE_WORKGROUP: UInt32 = 4

comptime OP_GROUP_NON_UNIFORM_ELECT: UInt32 = 333
comptime OP_GROUP_NON_UNIFORM_BALLOT: UInt32 = 339
comptime OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT: UInt32 = 344
comptime OP_GROUP_NON_UNIFORM_I_ADD: UInt32 = 349
comptime OP_GROUP_NON_UNIFORM_F_ADD: UInt32 = 350

comptime GROUP_OP_REDUCE: UInt32 = 0
comptime GROUP_OP_INCLUSIVE_SCAN: UInt32 = 1
comptime GROUP_OP_EXCLUSIVE_SCAN: UInt32 = 2
comptime GROUP_OP_CLUSTERED_REDUCE: UInt32 = 3

comptime SCOPE_DEVICE: UInt32 = 1
comptime SCOPE_WORKGROUP: UInt32 = 2
comptime SCOPE_SUBGROUP: UInt32 = 3

comptime IMAGE_FORMAT_RGBA16F: UInt32 = 2
comptime IMAGE_FORMAT_R11F_G11F_B10F: UInt32 = 8

comptime BUILT_IN_PRIMITIVE_ID: UInt32 = 7
comptime BUILT_IN_WORKGROUP_ID: UInt32 = 26
comptime BUILT_IN_LOCAL_INVOCATION_ID: UInt32 = 27

comptime DIM_2D: UInt32 = 1
comptime DIM_CUBE: UInt32 = 3


struct AaaSpirvModule(Movable):
    """SPIR-V 1.5 emitter for Vulkan 1.3 hardware Vertex, Fragment, Bindless & Ray-Query shaders."""
    var bound: UInt32
    var exec_model: UInt32
    var capabilities: List[UInt32]
    var extensions: List[String]
    var interface_vars: List[UInt32]
    var decorations: List[UInt32]
    var globals: List[UInt32]
    var variables: List[UInt32]
    var code: List[UInt32]
    var constants: Dict[Int, UInt32]
    var loops: List[LoopLabels]
    var ifs: List[IfLabels]
    var type_void: UInt32
    var type_bool: UInt32
    var type_uint: UInt32
    var type_float: UInt32
    var type_vec2: UInt32
    var type_vec3: UInt32
    var type_vec4: UInt32
    var type_uvec2: UInt32
    var type_uvec3: UInt32
    var type_uvec4: UInt32
    var type_function: UInt32
    var type_image_2d: UInt32
    var type_sampled_2d: UInt32
    var type_image_cube: UInt32
    var type_sampled_cube: UInt32
    var type_uimage_2d: UInt32
    var type_sampled_u2d: UInt32
    var type_accel_khr: UInt32
    var type_ray_query_khr: UInt32
    var type_storage_image_rgba16f: UInt32
    var type_storage_image_r11g11b10f: UInt32
    var primitive_id_var: UInt32
    var pointer_buffer_uint: UInt32
    var pointer_function_uint: UInt32
    var pointer_function_float: UInt32
    var pointer_function_ray_query: UInt32
    var pointer_push_uint: UInt32
    var buffer_pointer: UInt32
    var glsl: UInt32
    var main: UInt32
    var invocation: UInt32
    var local_invoc_var: UInt32
    var workgroup_id_var: UInt32
    var vert_index_var: UInt32
    var inst_index_var: UInt32
    var position_out_var: UInt32
    var frag_coord_var: UInt32
    var push: UInt32
    var group_width: UInt32
    var group_height: UInt32
    var early_depth: Bool
    var sota_mat_layered: Bool  # --- sota:mat hook --- emit the layered-material code (pipeline variant)

    def __init__(
        out self,
        exec_model: UInt32,
        group_width: Int = 8,
        group_height: Int = 8,
        push_words: Int = 0,
        enable_bindless: Bool = False,
        enable_ray_query: Bool = False,
        early_depth: Bool = True,
        enable_subgroups: Bool = False,
        enable_storage_image_formats: Bool = False,
    ):
        self.bound = 1
        self.exec_model = exec_model
        self.capabilities = List[UInt32]()
        self.capabilities.append(CAPABILITY_SHADER)
        self.extensions = List[String]()
        if enable_bindless:
            self.capabilities.append(CAPABILITY_SHADER_NON_UNIFORM)
            self.capabilities.append(CAPABILITY_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING)
        if enable_ray_query:
            self.capabilities.append(CAPABILITY_RAY_QUERY_KHR)
            self.extensions.append("SPV_KHR_ray_query")
        if enable_subgroups:
            self.capabilities.append(CAPABILITY_GROUP_NON_UNIFORM_BALLOT)
            self.capabilities.append(CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC)
        if enable_storage_image_formats:
            self.capabilities.append(CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS)
        self.interface_vars = List[UInt32]()
        self.decorations = List[UInt32]()
        self.globals = List[UInt32]()
        self.variables = List[UInt32]()
        self.code = List[UInt32]()
        self.constants = Dict[Int, UInt32]()
        self.loops = List[LoopLabels]()
        self.ifs = List[IfLabels]()
        self.type_void = 0
        self.type_bool = 0
        self.type_uint = 0
        self.type_float = 0
        self.type_vec2 = 0
        self.type_vec3 = 0
        self.type_vec4 = 0
        self.type_uvec2 = 0
        self.type_uvec3 = 0
        self.type_uvec4 = 0
        self.type_function = 0
        self.type_image_2d = 0
        self.type_sampled_2d = 0
        self.type_image_cube = 0
        self.type_sampled_cube = 0
        self.type_uimage_2d = 0
        self.type_sampled_u2d = 0
        self.type_accel_khr = 0
        self.type_ray_query_khr = 0
        self.type_storage_image_rgba16f = 0
        self.type_storage_image_r11g11b10f = 0
        self.primitive_id_var = 0
        self.pointer_buffer_uint = 0
        self.pointer_function_uint = 0
        self.pointer_function_float = 0
        self.pointer_function_ray_query = 0
        self.pointer_push_uint = 0
        self.buffer_pointer = 0
        self.glsl = 0
        self.main = 0
        self.invocation = 0
        self.local_invoc_var = 0
        self.workgroup_id_var = 0
        self.vert_index_var = 0
        self.inst_index_var = 0
        self.position_out_var = 0
        self.frag_coord_var = 0
        self.push = 0
        self.group_width = UInt32(group_width)
        self.group_height = UInt32(group_height)
        self.early_depth = early_depth
        self.sota_mat_layered = False  # --- sota:mat hook ---
        self._declare_core_types(push_words, enable_ray_query)
        var first = self.fresh()
        emit(self.code, OP_LABEL, first)

    def fresh(mut self) -> UInt32:
        var id = self.bound
        self.bound += 1
        return id

    def _declare_core_types(mut self, push_words: Int, enable_ray_query: Bool):
        self.glsl = self.fresh()
        self.main = self.fresh()
        self.type_void = self.fresh()
        self.type_function = self.fresh()
        self.type_bool = self.fresh()
        self.type_uint = self.fresh()
        self.type_float = self.fresh()
        self.type_vec2 = self.fresh()
        self.type_vec3 = self.fresh()
        self.type_vec4 = self.fresh()
        self.type_uvec2 = self.fresh()
        self.type_uvec3 = self.fresh()
        self.type_uvec4 = self.fresh()
        emit(self.globals, OP_TYPE_VOID, self.type_void)
        emit(self.globals, OP_TYPE_FUNCTION, self.type_function, self.type_void)
        emit(self.globals, OP_TYPE_BOOL, self.type_bool)
        emit(self.globals, OP_TYPE_INT, self.type_uint, 32, 0)
        emit(self.globals, OP_TYPE_FLOAT, self.type_float, 32)
        emit(self.globals, OP_TYPE_VECTOR, self.type_vec2, self.type_float, 2)
        emit(self.globals, OP_TYPE_VECTOR, self.type_vec3, self.type_float, 3)
        emit(self.globals, OP_TYPE_VECTOR, self.type_vec4, self.type_float, 4)
        emit(self.globals, OP_TYPE_VECTOR, self.type_uvec2, self.type_uint, 2)
        emit(self.globals, OP_TYPE_VECTOR, self.type_uvec3, self.type_uint, 3)
        emit(self.globals, OP_TYPE_VECTOR, self.type_uvec4, self.type_uint, 4)

        # Sampled 2D float image + Cube float image + Sampled 2D uint image (for R32G32_UINT VisBuffer fetch!)
        self.type_image_2d = self.fresh()
        self.type_sampled_2d = self.fresh()
        self.type_image_cube = self.fresh()
        self.type_sampled_cube = self.fresh()
        self.type_uimage_2d = self.fresh()
        self.type_sampled_u2d = self.fresh()
        # OpTypeImage: Result, SampledType, Dim, Depth=0, Arrayed=0, MS=0, Sampled=1, Format=0(Unknown)
        emit(self.globals, OP_TYPE_IMAGE, self.type_image_2d, self.type_float, DIM_2D, 0, 0, 0, 1, 0)
        emit(self.globals, OP_TYPE_SAMPLED_IMAGE, self.type_sampled_2d, self.type_image_2d)
        emit(self.globals, OP_TYPE_IMAGE, self.type_image_cube, self.type_float, DIM_CUBE, 0, 0, 0, 1, 0)
        emit(self.globals, OP_TYPE_SAMPLED_IMAGE, self.type_sampled_cube, self.type_image_cube)
        emit(self.globals, OP_TYPE_IMAGE, self.type_uimage_2d, self.type_uint, DIM_2D, 0, 0, 0, 1, 0)
        emit(self.globals, OP_TYPE_SAMPLED_IMAGE, self.type_sampled_u2d, self.type_uimage_2d)

        # Hardware 2D Storage Images (Rgba16f and R11fG11fB10f, Sampled=2)
        self.type_storage_image_rgba16f = self.fresh()
        emit(self.globals, OP_TYPE_IMAGE, self.type_storage_image_rgba16f, self.type_float, DIM_2D, 0, 0, 0, 2, IMAGE_FORMAT_RGBA16F)
        self.type_storage_image_r11g11b10f = self.fresh()
        emit(self.globals, OP_TYPE_IMAGE, self.type_storage_image_r11g11b10f, self.type_float, DIM_2D, 0, 0, 0, 2, IMAGE_FORMAT_R11F_G11F_B10F)

        if enable_ray_query:
            self.type_accel_khr = self.fresh()
            self.type_ray_query_khr = self.fresh()
            self.pointer_function_ray_query = self.fresh()
            emit(self.globals, OP_TYPE_ACCELERATION_STRUCTURE_KHR, self.type_accel_khr)
            emit(self.globals, OP_TYPE_RAY_QUERY_KHR, self.type_ray_query_khr)
            emit(self.globals, OP_TYPE_POINTER, self.pointer_function_ray_query, STORAGE_FUNCTION, self.type_ray_query_khr)

        # Flat uint[] storage buffer block
        var runtime = self.fresh()
        var block = self.fresh()
        self.buffer_pointer = self.fresh()
        self.pointer_buffer_uint = self.fresh()
        emit(self.globals, OP_TYPE_RUNTIME_ARRAY, runtime, self.type_uint)
        emit(self.globals, OP_TYPE_STRUCT, block, runtime)
        emit(self.globals, OP_TYPE_POINTER, self.buffer_pointer, STORAGE_BUFFER, block)
        emit(self.globals, OP_TYPE_POINTER, self.pointer_buffer_uint, STORAGE_BUFFER, self.type_uint)
        emit(self.decorations, OP_DECORATE, runtime, DECORATION_ARRAY_STRIDE, 4)
        emit(self.decorations, OP_DECORATE, block, DECORATION_BLOCK)
        emit(self.decorations, OP_MEMBER_DECORATE, block, 0, DECORATION_OFFSET, 0)

        self.pointer_function_uint = self.fresh()
        self.pointer_function_float = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, self.pointer_function_uint, STORAGE_FUNCTION, self.type_uint)
        emit(self.globals, OP_TYPE_POINTER, self.pointer_function_float, STORAGE_FUNCTION, self.type_float)

        if self.exec_model == MODEL_GL_COMPUTE:
            var ptr_in_uvec3 = self.fresh()
            self.invocation = self.fresh()
            emit(self.globals, OP_TYPE_POINTER, ptr_in_uvec3, STORAGE_INPUT, self.type_uvec3)
            emit(self.globals, OP_VARIABLE, ptr_in_uvec3, self.invocation, STORAGE_INPUT)
            emit(self.decorations, OP_DECORATE, self.invocation, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID)
            self.interface_vars.append(self.invocation)

            self.local_invoc_var = self.fresh()
            emit(self.globals, OP_VARIABLE, ptr_in_uvec3, self.local_invoc_var, STORAGE_INPUT)
            emit(self.decorations, OP_DECORATE, self.local_invoc_var, DECORATION_BUILT_IN, BUILT_IN_LOCAL_INVOCATION_ID)
            self.interface_vars.append(self.local_invoc_var)

            self.workgroup_id_var = self.fresh()
            emit(self.globals, OP_VARIABLE, ptr_in_uvec3, self.workgroup_id_var, STORAGE_INPUT)
            emit(self.decorations, OP_DECORATE, self.workgroup_id_var, DECORATION_BUILT_IN, BUILT_IN_WORKGROUP_ID)
            self.interface_vars.append(self.workgroup_id_var)
        elif self.exec_model == MODEL_VERTEX:
            var ptr_in_uint = self.fresh()
            self.vert_index_var = self.fresh()
            self.inst_index_var = self.fresh()
            emit(self.globals, OP_TYPE_POINTER, ptr_in_uint, STORAGE_INPUT, self.type_uint)
            emit(self.globals, OP_VARIABLE, ptr_in_uint, self.vert_index_var, STORAGE_INPUT)
            emit(self.globals, OP_VARIABLE, ptr_in_uint, self.inst_index_var, STORAGE_INPUT)
            emit(self.decorations, OP_DECORATE, self.vert_index_var, DECORATION_BUILT_IN, BUILT_IN_VERTEX_INDEX)
            emit(self.decorations, OP_DECORATE, self.inst_index_var, DECORATION_BUILT_IN, BUILT_IN_INSTANCE_INDEX)
            var ptr_out_vec4 = self.fresh()
            self.position_out_var = self.fresh()
            emit(self.globals, OP_TYPE_POINTER, ptr_out_vec4, STORAGE_OUTPUT, self.type_vec4)
            emit(self.globals, OP_VARIABLE, ptr_out_vec4, self.position_out_var, STORAGE_OUTPUT)
            emit(self.decorations, OP_DECORATE, self.position_out_var, DECORATION_BUILT_IN, BUILT_IN_POSITION)
            self.interface_vars.append(self.vert_index_var)
            self.interface_vars.append(self.inst_index_var)
            self.interface_vars.append(self.position_out_var)
        elif self.exec_model == MODEL_FRAGMENT:
            var ptr_in_vec4 = self.fresh()
            self.frag_coord_var = self.fresh()
            emit(self.globals, OP_TYPE_POINTER, ptr_in_vec4, STORAGE_INPUT, self.type_vec4)
            emit(self.globals, OP_VARIABLE, ptr_in_vec4, self.frag_coord_var, STORAGE_INPUT)
            emit(self.decorations, OP_DECORATE, self.frag_coord_var, DECORATION_BUILT_IN, BUILT_IN_FRAG_COORD)
            self.interface_vars.append(self.frag_coord_var)

        if push_words > 0:
            var array = self.fresh()
            var push_block = self.fresh()
            var push_pointer = self.fresh()
            self.pointer_push_uint = self.fresh()
            self.push = self.fresh()
            var length = self.uint(UInt32(push_words))
            emit(self.globals, OP_TYPE_ARRAY, array, self.type_uint, length)
            emit(self.globals, OP_TYPE_STRUCT, push_block, array)
            emit(self.globals, OP_TYPE_POINTER, push_pointer, STORAGE_PUSH_CONSTANT, push_block)
            emit(self.globals, OP_TYPE_POINTER, self.pointer_push_uint, STORAGE_PUSH_CONSTANT, self.type_uint)
            emit(self.globals, OP_VARIABLE, push_pointer, self.push, STORAGE_PUSH_CONSTANT)
            emit(self.decorations, OP_DECORATE, array, DECORATION_ARRAY_STRIDE, 4)
            emit(self.decorations, OP_DECORATE, push_block, DECORATION_BLOCK)
            emit(self.decorations, OP_MEMBER_DECORATE, push_block, 0, DECORATION_OFFSET, 0)
            self.interface_vars.append(self.push)

    # Constants & Vectors
    def uint(mut self, value: UInt32) -> UInt32:
        var key = Int(value)
        var known = self.constants.get(key)
        if known:
            return known.value()
        var id = self.fresh()
        emit(self.globals, OP_CONSTANT, self.type_uint, id, value)
        self.constants[key] = id
        return id

    def float_bits(mut self, bits: UInt32) -> UInt32:
        var key = Int(bits) | (1 << 32)
        var known = self.constants.get(key)
        if known:
            return known.value()
        var id = self.fresh()
        emit(self.globals, OP_CONSTANT, self.type_float, id, bits)
        self.constants[key] = id
        return id

    def float(mut self, value: Float32) -> UInt32:
        var copy = value
        return self.float_bits(Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0])

    def spec_constant_u32(mut self, spec_id: Int, default_val: UInt32 = 0) -> UInt32:
        """Emits `OpSpecConstant` (`UInt32`) decorated with `SpecId = spec_id`."""
        var id = self.fresh()
        emit(self.decorations, OP_DECORATE, id, DEC_SPEC_ID, UInt32(spec_id))
        emit(self.globals, OP_SPEC_CONSTANT, self.type_uint, id, default_val)
        return id

    def spec_constant_bool(mut self, spec_id: Int, default_val: Bool = False) -> UInt32:
        """Emits `OpSpecConstantTrue` / `OpSpecConstantFalse` decorated with `SpecId = spec_id`."""
        var id = self.fresh()
        var op = OP_SPEC_CONSTANT_TRUE if default_val else OP_SPEC_CONSTANT_FALSE
        emit(self.decorations, OP_DECORATE, id, DEC_SPEC_ID, UInt32(spec_id))
        emit(self.globals, op, self.type_bool, id)
        return id

    def spec_constant_f32(mut self, spec_id: Int, default_val: Float32 = 0.0) -> UInt32:
        """Emits `OpSpecConstant` (`Float32`) decorated with `SpecId = spec_id`."""
        var id = self.fresh()
        var copy = default_val
        var bits = Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0]
        emit(self.decorations, OP_DECORATE, id, DEC_SPEC_ID, UInt32(spec_id))
        emit(self.globals, OP_SPEC_CONSTANT, self.type_float, id, bits)
        return id

    def vec2(mut self, x: UInt32, y: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_CONSTRUCT, self.type_vec2, id, x, y)
        return id

    def vec3(mut self, x: UInt32, y: UInt32, z: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_CONSTRUCT, self.type_vec3, id, x, y, z)
        return id

    def vec4(mut self, x: UInt32, y: UInt32, z: UInt32, w: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_CONSTRUCT, self.type_vec4, id, x, y, z, w)
        return id

    def uvec2(mut self, x: UInt32, y: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_CONSTRUCT, self.type_uvec2, id, x, y)
        return id

    def uvec4(mut self, x: UInt32, y: UInt32, z: UInt32, w: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_CONSTRUCT, self.type_uvec4, id, x, y, z, w)
        return id

    def extract_f(mut self, vec_id: UInt32, idx: Int) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_EXTRACT, self.type_float, id, vec_id, UInt32(idx))
        return id

    def extract_u(mut self, vec_id: UInt32, idx: Int) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_EXTRACT, self.type_uint, id, vec_id, UInt32(idx))
        return id

    # Stage IO & Resources
    def vertex_index(mut self) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, self.vert_index_var)
        return id

    def instance_index(mut self) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, self.inst_index_var)
        return id

    def write_clip_position(mut self, x: UInt32, y: UInt32, z: UInt32, w: UInt32):
        var v = self.vec4(x, y, z, w)
        emit(self.code, OP_STORE, self.position_out_var, v)

    def frag_coord(mut self) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_vec4, id, self.frag_coord_var)
        return id

    def add_stage_io(mut self, storage_class: UInt32, vec_type: UInt32, location: Int, flat: Bool = False) -> Tuple[UInt32, UInt32]:
        """Declares an Input or Output stage variable at `location`, returning `(ptr_type, var_id)`."""
        var ptr_t = self.fresh()
        var var_id = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_t, storage_class, vec_type)
        emit(self.globals, OP_VARIABLE, ptr_t, var_id, storage_class)
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_LOCATION, UInt32(location))
        if flat:
            emit(self.decorations, OP_DECORATE, var_id, DECORATION_FLAT)
        self.interface_vars.append(var_id)
        return (ptr_t, var_id)

    def buffer(mut self, binding: Int, set: Int = 0) -> UInt32:
        var id = self.fresh()
        emit(self.globals, OP_VARIABLE, self.buffer_pointer, id, STORAGE_BUFFER)
        emit(self.decorations, OP_DECORATE, id, DECORATION_DESCRIPTOR_SET, UInt32(set))
        emit(self.decorations, OP_DECORATE, id, DECORATION_BINDING, UInt32(binding))
        self.interface_vars.append(id)
        return id

    def sampled_image_2d(mut self, binding: Int, set: Int = 0, is_uint: Bool = False, is_cube: Bool = False) -> Tuple[UInt32, UInt32]:
        """Declares a single combined `sampler2D` / `usampler2D` / `samplerCube`, returning `(sampled_type, var_id)`."""
        var s_type = self.type_sampled_u2d if is_uint else (self.type_sampled_cube if is_cube else self.type_sampled_2d)
        var ptr_t = self.fresh()
        var var_id = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_t, STORAGE_UNIFORM_CONSTANT, s_type)
        emit(self.globals, OP_VARIABLE, ptr_t, var_id, STORAGE_UNIFORM_CONSTANT)
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_DESCRIPTOR_SET, UInt32(set))
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_BINDING, UInt32(binding))
        self.interface_vars.append(var_id)
        return (s_type, var_id)

    def bindless_sampled_array_2d(mut self, binding: Int, array_len: Int, set: Int = 0) -> Tuple[UInt32, UInt32]:
        """Declares `sampler2D textures[array_len]` for bindless BC7/BC5/BC6H sampling, returning `(elem_ptr_type, array_var_id)`."""
        var len_c = self.uint(UInt32(array_len))
        var arr_t = self.fresh()
        var arr_ptr_t = self.fresh()
        var elem_ptr_t = self.fresh()
        var var_id = self.fresh()
        emit(self.globals, OP_TYPE_ARRAY, arr_t, self.type_sampled_2d, len_c)
        emit(self.globals, OP_TYPE_POINTER, arr_ptr_t, STORAGE_UNIFORM_CONSTANT, arr_t)
        emit(self.globals, OP_TYPE_POINTER, elem_ptr_t, STORAGE_UNIFORM_CONSTANT, self.type_sampled_2d)
        emit(self.globals, OP_VARIABLE, arr_ptr_t, var_id, STORAGE_UNIFORM_CONSTANT)
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_DESCRIPTOR_SET, UInt32(set))
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_BINDING, UInt32(binding))
        self.interface_vars.append(var_id)
        return (elem_ptr_t, var_id)

    def sample_bindless_grad(
        mut self,
        bindless_pair: Tuple[UInt32, UInt32],
        tex_idx: UInt32,
        uv: UInt32,
        ddx_uv: UInt32,
        ddy_uv: UInt32,
    ) -> UInt32:
        """Hardware 16x Anisotropic BC7/BC5/RGBA sample using explicit analytic VisBuffer UV gradients (`Grad`)."""
        var ptr = self.fresh()
        emit(self.code, OP_ACCESS_CHAIN, bindless_pair[0], ptr, bindless_pair[1], tex_idx)
        emit(self.decorations, OP_DECORATE, ptr, DECORATION_NON_UNIFORM)
        var samp = self.fresh()
        emit(self.code, OP_LOAD, self.type_sampled_2d, samp, ptr)
        emit(self.decorations, OP_DECORATE, samp, DECORATION_NON_UNIFORM)
        var out_v4 = self.fresh()
        emit(self.code, OP_IMAGE_SAMPLE_EXPLICIT_LOD, self.type_vec4, out_v4, samp, uv, IMAGE_OPERANDS_GRAD, ddx_uv, ddy_uv)
        return out_v4

    def sample_bindless_lod(
        mut self,
        bindless_pair: Tuple[UInt32, UInt32],
        tex_idx: UInt32,
        uv: UInt32,
        lod: UInt32,
    ) -> UInt32:
        var ptr = self.fresh()
        emit(self.code, OP_ACCESS_CHAIN, bindless_pair[0], ptr, bindless_pair[1], tex_idx)
        emit(self.decorations, OP_DECORATE, ptr, DECORATION_NON_UNIFORM)
        var samp = self.fresh()
        emit(self.code, OP_LOAD, self.type_sampled_2d, samp, ptr)
        emit(self.decorations, OP_DECORATE, samp, DECORATION_NON_UNIFORM)
        var out_v4 = self.fresh()
        emit(self.code, OP_IMAGE_SAMPLE_EXPLICIT_LOD, self.type_vec4, out_v4, samp, uv, IMAGE_OPERANDS_LOD, lod)
        return out_v4

    def sample_2d_lod(mut self, samp_pair: Tuple[UInt32, UInt32], uv: UInt32, lod: UInt32) -> UInt32:
        var samp = self.fresh()
        emit(self.code, OP_LOAD, samp_pair[0], samp, samp_pair[1])
        var out_v4 = self.fresh()
        emit(self.code, OP_IMAGE_SAMPLE_EXPLICIT_LOD, self.type_vec4, out_v4, samp, uv, IMAGE_OPERANDS_LOD, lod)
        return out_v4

    def sample_cube_lod(mut self, cube_pair: Tuple[UInt32, UInt32], dir_vec3: UInt32, lod: UInt32) -> UInt32:
        var samp = self.fresh()
        emit(self.code, OP_LOAD, cube_pair[0], samp, cube_pair[1])
        var out_v4 = self.fresh()
        emit(self.code, OP_IMAGE_SAMPLE_EXPLICIT_LOD, self.type_vec4, out_v4, samp, dir_vec3, IMAGE_OPERANDS_LOD, lod)
        return out_v4

    def texel_fetch_u2d(mut self, usamp_pair: Tuple[UInt32, UInt32], coord_uvec2: UInt32, lod: UInt32) -> UInt32:
        """Exact integer `texelFetch(usampler2D, ivec2, lod)` via `OpImage` + `OpImageFetch`."""
        var samp = self.fresh()
        emit(self.code, OP_LOAD, usamp_pair[0], samp, usamp_pair[1])
        var raw_img = self.fresh()
        emit(self.code, OP_IMAGE, self.type_uimage_2d, raw_img, samp)
        var out_uv4 = self.fresh()
        emit(self.code, OP_IMAGE_FETCH, self.type_uvec4, out_uv4, raw_img, coord_uvec2, IMAGE_OPERANDS_LOD, lod)
        return out_uv4

    def texel_fetch_f2d(mut self, fsamp_pair: Tuple[UInt32, UInt32], coord_uvec2: UInt32, lod: UInt32) -> UInt32:
        """Exact float `texelFetch(sampler2D, ivec2, lod)` via `OpImage` + `OpImageFetch`."""
        var samp = self.fresh()
        emit(self.code, OP_LOAD, fsamp_pair[0], samp, fsamp_pair[1])
        var raw_img = self.fresh()
        emit(self.code, OP_IMAGE, self.type_image_2d, raw_img, samp)
        var out_v4 = self.fresh()
        emit(self.code, OP_IMAGE_FETCH, self.type_vec4, out_v4, raw_img, coord_uvec2, IMAGE_OPERANDS_LOD, lod)
        return out_v4

    def acceleration_structure(mut self, binding: Int, set: Int = 0) -> UInt32:
        var ptr_t = self.fresh()
        var var_id = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_t, STORAGE_UNIFORM_CONSTANT, self.type_accel_khr)
        emit(self.globals, OP_VARIABLE, ptr_t, var_id, STORAGE_UNIFORM_CONSTANT)
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_DESCRIPTOR_SET, UInt32(set))
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_BINDING, UInt32(binding))
        self.interface_vars.append(var_id)
        return var_id

    def ray_query_var(mut self) -> UInt32:
        var id = self.fresh()
        emit(self.variables, OP_VARIABLE, self.pointer_function_ray_query, id, STORAGE_FUNCTION)
        return id

    def ray_query_init(
        mut self,
        rq_var: UInt32,
        tlas_var: UInt32,
        ray_flags: UInt32,
        cull_mask: UInt32,
        origin_v3: UInt32,
        t_min: UInt32,
        dir_v3: UInt32,
        t_max: UInt32,
    ):
        var tlas_val = self.fresh()
        emit(self.code, OP_LOAD, self.type_accel_khr, tlas_val, tlas_var)
        emit(
            self.code,
            OP_RAY_QUERY_INITIALIZE_KHR,
            rq_var,
            tlas_val,
            ray_flags,
            cull_mask,
            origin_v3,
            t_min,
            dir_v3,
            t_max,
        )

    def ray_query_proceed(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        emit(self.code, OP_RAY_QUERY_PROCEED_KHR, self.type_bool, res, rq_var)
        return res

    def ray_query_committed_type(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var committed = self.uint(1)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR, self.type_uint, res, rq_var, committed)
        return res

    def ray_query_committed_t(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var committed = self.uint(1)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_T_KHR, self.type_float, res, rq_var, committed)
        return res

    def ray_query_committed_custom_index(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var committed = self.uint(1)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_INSTANCE_CUSTOM_INDEX_KHR, self.type_uint, res, rq_var, committed)
        return res

    def ray_query_committed_instance_id(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var committed = self.uint(1)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_INSTANCE_ID_KHR, self.type_uint, res, rq_var, committed)
        return res

    def ray_query_committed_prim_index(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var committed = self.uint(1)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_PRIMITIVE_INDEX_KHR, self.type_uint, res, rq_var, committed)
        return res

    def ray_query_committed_barycentrics(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var committed = self.uint(1)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_BARYCENTRICS_KHR, self.type_vec2, res, rq_var, committed)
        return res

    def ray_query_committed_front_face(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var committed = self.uint(1)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_FRONT_FACE_KHR, self.type_bool, res, rq_var, committed)
        return res

    def ray_query_candidate_type(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var candidate = self.uint(0)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR, self.type_uint, res, rq_var, candidate)
        return res

    def ray_query_candidate_t(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var candidate = self.uint(0)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_T_KHR, self.type_float, res, rq_var, candidate)
        return res

    def ray_query_candidate_instance_id(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var candidate = self.uint(0)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_INSTANCE_ID_KHR, self.type_uint, res, rq_var, candidate)
        return res

    def ray_query_candidate_prim_index(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var candidate = self.uint(0)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_PRIMITIVE_INDEX_KHR, self.type_uint, res, rq_var, candidate)
        return res

    def ray_query_candidate_barycentrics(mut self, rq_var: UInt32) -> UInt32:
        var res = self.fresh()
        var candidate = self.uint(0)
        emit(self.code, OP_RAY_QUERY_GET_INTERSECTION_BARYCENTRICS_KHR, self.type_vec2, res, rq_var, candidate)
        return res

    def ray_query_confirm_intersection(mut self, rq_var: UInt32):
        emit(self.code, OP_RAY_QUERY_CONFIRM_INTERSECTION_KHR, rq_var)

    # Buffer & Push Ops
    def load_word(mut self, buffer: UInt32, index: UInt32) -> UInt32:
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, pointer)
        return id

    def load_float(mut self, buffer: UInt32, index: UInt32) -> UInt32:
        return self.bitcast_float(self.load_word(buffer, index))

    def store_word(mut self, buffer: UInt32, index: UInt32, value: UInt32):
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        emit(self.code, OP_STORE, pointer, value)

    def store_float(mut self, buffer: UInt32, index: UInt32, value: UInt32):
        self.store_word(buffer, index, self.bitcast_uint(value))

    def atomic_add_word(mut self, buffer: UInt32, index: UInt32, value: UInt32) -> UInt32:
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        var id = self.fresh()
        emit(self.code, OP_ATOMIC_I_ADD, self.type_uint, id, pointer, self.uint(1), self.uint(0), value)
        return id

    def atomic_umax_word(mut self, buffer: UInt32, index: UInt32, value: UInt32) -> UInt32:
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        var id = self.fresh()
        emit(self.code, OP_ATOMIC_U_MAX, self.type_uint, id, pointer, self.uint(1), self.uint(0), value)
        return id

    def push_word(mut self, index: Int) -> UInt32:
        var pointer = self.fresh()
        var zero = self.uint(0)
        var word = self.uint(UInt32(index))
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_push_uint, pointer, self.push, zero, word)
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, pointer)
        return id

    def push_float(mut self, index: Int) -> UInt32:
        return self.bitcast_float(self.push_word(index))

    def invocation_axis(mut self, axis: Int) -> UInt32:
        var vector = self.fresh()
        emit(self.code, OP_LOAD, self.type_uvec3, vector, self.invocation)
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_EXTRACT, self.type_uint, id, vector, UInt32(axis))
        return id

    def local_axis(mut self, axis: Int) -> UInt32:
        var vector = self.fresh()
        emit(self.code, OP_LOAD, self.type_uvec3, vector, self.local_invoc_var)
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_EXTRACT, self.type_uint, id, vector, UInt32(axis))
        return id

    def workgroup_axis(mut self, axis: Int) -> UInt32:
        var vector = self.fresh()
        emit(self.code, OP_LOAD, self.type_uvec3, vector, self.workgroup_id_var)
        var id = self.fresh()
        emit(self.code, OP_COMPOSITE_EXTRACT, self.type_uint, id, vector, UInt32(axis))
        return id

    # Mutable Function Variables
    def uvar(mut self, initial: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.variables, OP_VARIABLE, self.pointer_function_uint, id, STORAGE_FUNCTION)
        self.store(id, initial)
        return id

    def fvar(mut self, initial: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.variables, OP_VARIABLE, self.pointer_function_float, id, STORAGE_FUNCTION)
        self.store(id, initial)
        return id

    def load_u(mut self, variable: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, variable)
        return id

    def load_f(mut self, variable: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_float, id, variable)
        return id

    def store(mut self, variable: UInt32, value: UInt32):
        emit(self.code, OP_STORE, variable, value)

    # Arithmetic & Intrinsics
    def binary(mut self, opcode: UInt32, type: UInt32, a: UInt32, b: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, opcode, type, id, a, b)
        return id

    def unary(mut self, opcode: UInt32, type: UInt32, a: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, opcode, type, id, a)
        return id

    def ext1(mut self, operation: UInt32, type: UInt32, a: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a)
        return id

    def ext2(mut self, operation: UInt32, type: UInt32, a: UInt32, b: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a, b)
        return id

    def ext3(mut self, operation: UInt32, type: UInt32, a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a, b, c)
        return id

    def iadd(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_ADD, self.type_uint, a, b)

    def isub(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_SUB, self.type_uint, a, b)

    def imul(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_MUL, self.type_uint, a, b)

    def udiv(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_U_DIV, self.type_uint, a, b)

    def umod(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_U_MOD, self.type_uint, a, b)

    def bit_and(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_AND, self.type_uint, a, b)

    def bit_or(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_OR, self.type_uint, a, b)

    def bit_xor(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_XOR, self.type_uint, a, b)

    def shl(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_LEFT_LOGICAL, self.type_uint, a, b)

    def shr(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_RIGHT_LOGICAL, self.type_uint, a, b)

    def sar(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_RIGHT_ARITHMETIC, self.type_uint, a, b)

    def umin(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_U_MIN, self.type_uint, a, b)

    def umax(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_U_MAX, self.type_uint, a, b)

    def fneg(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_F_NEGATE, self.type_float, a)

    def fadd(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ADD, self.type_float, a, b)

    def fsub(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_SUB, self.type_float, a, b)

    def fmul(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_MUL, self.type_float, a, b)

    def fdiv(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_DIV, self.type_float, a, b)

    def fma(mut self, a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        return self.ext3(GLSL_FMA, self.type_float, a, b, c)

    def fract(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_FRACT, self.type_float, a)

    def fabs(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_F_ABS, self.type_float, a)

    def ffloor(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_FLOOR, self.type_float, a)

    def fsin(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_SIN, self.type_float, a)

    def fcos(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_COS, self.type_float, a)

    def fatan2(mut self, y: UInt32, x: UInt32) -> UInt32:
        return self.ext2(GLSL_ATAN2, self.type_float, y, x)

    def fpow(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_POW, self.type_float, a, b)

    def pack_half2x16(mut self, v2_id: UInt32) -> UInt32:
        return self.ext1(GLSL_PACK_HALF2X16, self.type_uint, v2_id)

    def unpack_half2x16(mut self, u32_id: UInt32) -> UInt32:
        return self.ext1(GLSL_UNPACK_HALF2X16, self.type_vec2, u32_id)

    @always_inline
    def pack_f16x2(mut self, x_f32: UInt32, y_f32: UInt32) -> UInt32:
        var cx = self.nclamp(x_f32, self.float(-65504.0), self.float(65504.0))
        var cy = self.nclamp(y_f32, self.float(-65504.0), self.float(65504.0))
        return self.pack_half2x16(self.vec2(cx, cy))

    @always_inline
    def unpack_f16x2(mut self, packed_u32: UInt32) -> Tuple[UInt32, UInt32]:
        var v2 = self.unpack_half2x16(packed_u32)
        return (self.extract_f(v2, 0), self.extract_f(v2, 1))

    def exp2(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_EXP2, self.type_float, a)

    def log2(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_LOG2, self.type_float, a)

    def fsqrt(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_SQRT, self.type_float, a)

    def inverse_sqrt(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_INVERSE_SQRT, self.type_float, a)

    def nmax(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_N_MAX, self.type_float, a, b)

    def nmin(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_N_MIN, self.type_float, a, b)

    def nclamp(mut self, a: UInt32, low: UInt32, high: UInt32) -> UInt32:
        return self.ext3(GLSL_N_CLAMP, self.type_float, a, low, high)

    def dpdx(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_DPDX, self.type_float, a)

    def dpdy(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_DPDY, self.type_float, a)

    def float_from_uint(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_U_TO_F, self.type_float, a)

    def float_from_int(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_S_TO_F, self.type_float, a)

    def uint_from_float(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_F_TO_U, self.type_uint, a)

    def saturate_uint(mut self, a: UInt32) -> UInt32:
        return self.uint_from_float(self.nclamp(a, self.float(0.0), self.float(4294967040.0)))

    def bitcast_float(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_BITCAST, self.type_float, a)

    def bitcast_uint(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_BITCAST, self.type_uint, a)

    def select_u(mut self, condition: UInt32, yes: UInt32, no: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_SELECT, self.type_uint, id, condition, yes, no)
        return id

    def select_f(mut self, condition: UInt32, yes: UInt32, no: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_SELECT, self.type_float, id, condition, yes, no)
        return id

    # Comparisons & Logic
    def ieq(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_EQUAL, self.type_bool, a, b)

    def ine(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_NOT_EQUAL, self.type_bool, a, b)

    def ult(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_U_LESS_THAN, self.type_bool, a, b)

    def uge(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_U_GREATER_THAN_EQUAL, self.type_bool, a, b)

    def flt(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_LESS_THAN, self.type_bool, a, b)

    def fgt(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_GREATER_THAN, self.type_bool, a, b)

    def fge(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_GREATER_THAN_EQUAL, self.type_bool, a, b)

    def logical_and(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_LOGICAL_AND, self.type_bool, a, b)

    def logical_or(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_LOGICAL_OR, self.type_bool, a, b)

    def logical_not(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_LOGICAL_NOT, self.type_bool, a)

    # Forwarders for alternative DSL conventions
    @always_inline
    def load_uint(mut self, buffer: UInt32, index: UInt32) -> UInt32:
        return self.load_word(buffer, index)

    @always_inline
    def store_uint(mut self, buffer: UInt32, index: UInt32, value: UInt32):
        self.store_word(buffer, index, value)

    @always_inline
    def iand(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.bit_and(a, b)

    @always_inline
    def ior(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.bit_or(a, b)

    @always_inline
    def ishr(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.shr(a, b)

    @always_inline
    def ishl(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.shl(a, b)

    @always_inline
    def fle(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.fge(b, a)

    # Control Flow
    def label(mut self, id: UInt32):
        emit(self.code, OP_LABEL, id)

    def begin_if(mut self, condition: UInt32):
        var yes = self.fresh()
        var otherwise = self.fresh()
        var merge = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, merge, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, yes, otherwise)
        self.label(yes)
        self.ifs.append(IfLabels(merge, otherwise))

    def begin_else(mut self):
        var top = self.ifs[len(self.ifs) - 1].copy()
        emit(self.code, OP_BRANCH, top.merge)
        self.label(top.otherwise)
        self.ifs[len(self.ifs) - 1].otherwise = 0

    def end_if(mut self):
        var top = self.ifs.pop()
        emit(self.code, OP_BRANCH, top.merge)
        if top.otherwise != 0:
            self.label(top.otherwise)
            emit(self.code, OP_BRANCH, top.merge)
        self.label(top.merge)

    def begin_loop(mut self):
        var header = self.fresh()
        var body = self.fresh()
        var next = self.fresh()
        var merge = self.fresh()
        emit(self.code, OP_BRANCH, header)
        self.label(header)
        emit(self.code, OP_LOOP_MERGE, merge, next, 0)
        emit(self.code, OP_BRANCH, body)
        self.label(body)
        self.loops.append(LoopLabels(header, next, merge))

    def leave_if(mut self, condition: UInt32, target: UInt32):
        var carry_on = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, carry_on, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, target, carry_on)
        self.label(carry_on)

    def break_if(mut self, condition: UInt32):
        self.leave_if(condition, self.loops[len(self.loops) - 1].merge)

    def continue_if(mut self, condition: UInt32):
        self.leave_if(condition, self.loops[len(self.loops) - 1].next)

    def end_loop(mut self):
        var top = self.loops.pop()
        emit(self.code, OP_BRANCH, top.next)
        self.label(top.next)
        emit(self.code, OP_BRANCH, top.header)
        self.label(top.merge)

    def return_if(mut self, condition: UInt32):
        var leave = self.fresh()
        var carry_on = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, carry_on, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, leave, carry_on)
        self.label(leave)
        emit(self.code, OP_RETURN)
        self.label(carry_on)

    def discard_if(mut self, condition: UInt32):
        """Fragment-stage alpha-cutout discard (`OpKill`) inside structured selection merge."""
        var kill_blk = self.fresh()
        var carry_on = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, carry_on, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, kill_blk, carry_on)
        self.label(kill_blk)
        emit(self.code, OP_KILL)
        self.label(carry_on)

    # --- sota:shadow hook --- gl_ClipDistance[n] output (vertex stage; needs the shaderClipDistance feature)
    def write_clip_distances(mut self, values: List[UInt32]):
        """Declares `float gl_ClipDistance[len(values)]` (BuiltIn ClipDistance, Capability ClipDistance) and stores
        `values` into it. Call once per vertex shader. A primitive is clipped where a distance is negative."""
        var n = len(values)
        if n == 0:
            return
        self.capabilities.append(UInt32(32))  # Capability ClipDistance
        var arr_t = self.fresh()
        emit(self.globals, OP_TYPE_ARRAY, arr_t, self.type_float, self.uint(UInt32(n)))
        var ptr_arr = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_arr, STORAGE_OUTPUT, arr_t)
        var var_id = self.fresh()
        emit(self.globals, OP_VARIABLE, ptr_arr, var_id, STORAGE_OUTPUT)
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_BUILT_IN, UInt32(3))  # BuiltIn ClipDistance
        self.interface_vars.append(var_id)
        var ptr_f = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_f, STORAGE_OUTPUT, self.type_float)
        for i in range(n):
            var elem = self.fresh()
            emit(self.code, OP_ACCESS_CHAIN, ptr_f, elem, var_id, self.uint(UInt32(i)))
            emit(self.code, OP_STORE, elem, values[i])
    # --- end sota:shadow hook ---

    def primitive_id(mut self) -> UInt32:
        """Loads the BuiltIn PrimitiveId (gl_PrimitiveID) for fragment stage primitive indexing."""
        if self.primitive_id_var == 0:
            var has_geom = False
            for cap in self.capabilities:
                if cap == CAPABILITY_GEOMETRY:
                    has_geom = True
            if not has_geom:
                self.capabilities.append(CAPABILITY_GEOMETRY)
            var ptr_in_uint = self.fresh()
            self.primitive_id_var = self.fresh()
            emit(self.globals, OP_TYPE_POINTER, ptr_in_uint, STORAGE_INPUT, self.type_uint)
            emit(self.globals, OP_VARIABLE, ptr_in_uint, self.primitive_id_var, STORAGE_INPUT)
            emit(self.decorations, OP_DECORATE, self.primitive_id_var, DECORATION_BUILT_IN, BUILT_IN_PRIMITIVE_ID)
            self.interface_vars.append(self.primitive_id_var)
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, self.primitive_id_var)
        return id

    # Workgroup LDS arrays & operations
    def workgroup_array_f32(mut self, size: Int) -> Tuple[UInt32, UInt32]:
        var len_id = self.uint(UInt32(size))
        var arr_t = self.fresh()
        emit(self.globals, OP_TYPE_ARRAY, arr_t, self.type_float, len_id)
        var ptr_arr_t = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_arr_t, STORAGE_WORKGROUP, arr_t)
        var ptr_elem_t = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_elem_t, STORAGE_WORKGROUP, self.type_float)
        var var_id = self.fresh()
        emit(self.globals, OP_VARIABLE, ptr_arr_t, var_id, STORAGE_WORKGROUP)
        self.interface_vars.append(var_id)
        return (ptr_elem_t, var_id)

    def workgroup_array_u32(mut self, size: Int) -> Tuple[UInt32, UInt32]:
        var len_id = self.uint(UInt32(size))
        var arr_t = self.fresh()
        emit(self.globals, OP_TYPE_ARRAY, arr_t, self.type_uint, len_id)
        var ptr_arr_t = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_arr_t, STORAGE_WORKGROUP, arr_t)
        var ptr_elem_t = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_elem_t, STORAGE_WORKGROUP, self.type_uint)
        var var_id = self.fresh()
        emit(self.globals, OP_VARIABLE, ptr_arr_t, var_id, STORAGE_WORKGROUP)
        self.interface_vars.append(var_id)
        return (ptr_elem_t, var_id)

    def load_workgroup_f32(mut self, ptr_elem_t: UInt32, var_id: UInt32, index: UInt32) -> UInt32:
        var ptr = self.fresh()
        emit(self.code, OP_ACCESS_CHAIN, ptr_elem_t, ptr, var_id, index)
        var res = self.fresh()
        emit(self.code, OP_LOAD, self.type_float, res, ptr)
        return res

    def store_workgroup_f32(mut self, ptr_elem_t: UInt32, var_id: UInt32, index: UInt32, value: UInt32):
        var ptr = self.fresh()
        emit(self.code, OP_ACCESS_CHAIN, ptr_elem_t, ptr, var_id, index)
        emit(self.code, OP_STORE, ptr, value)

    def load_workgroup_u32(mut self, ptr_elem_t: UInt32, var_id: UInt32, index: UInt32) -> UInt32:
        var ptr = self.fresh()
        emit(self.code, OP_ACCESS_CHAIN, ptr_elem_t, ptr, var_id, index)
        var res = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, res, ptr)
        return res

    def store_workgroup_u32(mut self, ptr_elem_t: UInt32, var_id: UInt32, index: UInt32, value: UInt32):
        var ptr = self.fresh()
        emit(self.code, OP_ACCESS_CHAIN, ptr_elem_t, ptr, var_id, index)
        emit(self.code, OP_STORE, ptr, value)

    def atomic_add_workgroup_u32(mut self, ptr_elem_t: UInt32, var_id: UInt32, index: UInt32, value: UInt32) -> UInt32:
        var ptr = self.fresh()
        emit(self.code, OP_ACCESS_CHAIN, ptr_elem_t, ptr, var_id, index)
        var s_workgroup = self.uint(2)
        var relaxed = self.uint(0)
        var res = self.fresh()
        emit(self.code, OP_ATOMIC_I_ADD, self.type_uint, res, ptr, s_workgroup, relaxed, value)
        return res

    def atomic_umax_workgroup_u32(mut self, ptr_elem_t: UInt32, var_id: UInt32, index: UInt32, value: UInt32) -> UInt32:
        var ptr = self.fresh()
        emit(self.code, OP_ACCESS_CHAIN, ptr_elem_t, ptr, var_id, index)
        var s_workgroup = self.uint(2)
        var relaxed = self.uint(0)
        var res = self.fresh()
        emit(self.code, OP_ATOMIC_U_MAX, self.type_uint, res, ptr, s_workgroup, relaxed, value)
        return res

    # Barriers
    def control_barrier(mut self, exec_scope: UInt32, mem_scope: UInt32, mem_semantics: UInt32):
        var s_exec = self.uint(exec_scope)
        var s_mem = self.uint(mem_scope)
        var sem = self.uint(mem_semantics)
        emit(self.code, OP_CONTROL_BARRIER, s_exec, s_mem, sem)

    def memory_barrier(mut self, mem_scope: UInt32, mem_semantics: UInt32):
        var s_mem = self.uint(mem_scope)
        var sem = self.uint(mem_semantics)
        emit(self.code, OP_MEMORY_BARRIER, s_mem, sem)

    def workgroup_barrier(mut self):
        self.control_barrier(2, 2, 776)

    # Subgroups
    def subgroup_elect(mut self) -> UInt32:
        var res = self.fresh()
        var s_subgroup = self.uint(3)
        emit(self.code, OP_GROUP_NON_UNIFORM_ELECT, self.type_bool, res, s_subgroup)
        return res

    def subgroup_ballot(mut self, predicate: UInt32) -> UInt32:
        var res = self.fresh()
        var s_subgroup = self.uint(3)
        emit(self.code, OP_GROUP_NON_UNIFORM_BALLOT, self.type_uvec4, res, s_subgroup, predicate)
        return res

    def subgroup_fadd(mut self, value: UInt32, group_op: UInt32 = GROUP_OP_REDUCE) -> UInt32:
        var res = self.fresh()
        var s_subgroup = self.uint(3)
        emit(self.code, OP_GROUP_NON_UNIFORM_F_ADD, self.type_float, res, s_subgroup, group_op, value)
        return res

    def subgroup_iadd(mut self, value: UInt32, group_op: UInt32 = GROUP_OP_REDUCE) -> UInt32:
        var res = self.fresh()
        var s_subgroup = self.uint(3)
        emit(self.code, OP_GROUP_NON_UNIFORM_I_ADD, self.type_uint, res, s_subgroup, group_op, value)
        return res

    # Hardware 2D Storage Images
    def storage_image_2d(mut self, binding: Int, set: Int = 0, format: UInt32 = IMAGE_FORMAT_RGBA16F) -> Tuple[UInt32, UInt32]:
        var img_t = self.type_storage_image_rgba16f if format == IMAGE_FORMAT_RGBA16F else self.type_storage_image_r11g11b10f
        var ptr_t = self.fresh()
        var var_id = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, ptr_t, STORAGE_UNIFORM_CONSTANT, img_t)
        emit(self.globals, OP_VARIABLE, ptr_t, var_id, STORAGE_UNIFORM_CONSTANT)
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_DESCRIPTOR_SET, UInt32(set))
        emit(self.decorations, OP_DECORATE, var_id, DECORATION_BINDING, UInt32(binding))
        self.interface_vars.append(var_id)
        return (img_t, var_id)

    def image_read(mut self, img_pair: Tuple[UInt32, UInt32], coord: UInt32) -> UInt32:
        var img = self.fresh()
        emit(self.code, OP_LOAD, img_pair[0], img, img_pair[1])
        var res = self.fresh()
        emit(self.code, OP_IMAGE_READ, self.type_vec4, res, img, coord)
        return res

    def image_write(mut self, img_pair: Tuple[UInt32, UInt32], coord: UInt32, texel: UInt32):
        var img = self.fresh()
        emit(self.code, OP_LOAD, img_pair[0], img, img_pair[1])
        emit(self.code, OP_IMAGE_WRITE, img, coord, texel)

    def finish(mut self) -> List[UInt32]:
        emit(self.code, OP_RETURN)
        emit(self.code, OP_FUNCTION_END)
        var words = List[UInt32]()
        words.append(SPIRV_MAGIC)
        words.append(SPIRV_VERSION_1_5)
        words.append(0)
        words.append(self.bound)
        words.append(0)
        for cap in self.capabilities:
            emit(words, OP_CAPABILITY, cap)
        for ext in self.extensions:
            var ext_w = string_words(ext)
            words.append(OP_EXTENSION | (UInt32(len(ext_w) + 1) << 16))
            words.extend(ext_w^)
        var import_name = string_words("GLSL.std.450")
        words.append(OP_EXT_INST_IMPORT | (UInt32(len(import_name) + 2) << 16))
        words.append(self.glsl)
        words.extend(import_name^)
        emit(words, OP_MEMORY_MODEL, ADDRESSING_LOGICAL, MEMORY_GLSL450)
        var entry_name = string_words("main")
        # SPIR-V 1.5 requires ALL statically referenced global OpVariables in OpEntryPoint
        var ep_words = 3 + len(entry_name) + len(self.interface_vars)
        words.append(OP_ENTRY_POINT | (UInt32(ep_words) << 16))
        words.append(self.exec_model)
        words.append(self.main)
        words.extend(entry_name^)
        for iv in self.interface_vars:
            words.append(iv)
        if self.exec_model == MODEL_GL_COMPUTE:
            emit(words, OP_EXECUTION_MODE, self.main, MODE_LOCAL_SIZE, self.group_width, self.group_height, 1)
        elif self.exec_model == MODEL_FRAGMENT:
            emit(words, OP_EXECUTION_MODE, self.main, MODE_ORIGIN_UPPER_LEFT)
            if self.early_depth:
                emit(words, OP_EXECUTION_MODE, self.main, MODE_EARLY_FRAGMENT_TESTS)
        words.extend(self.decorations.copy())
        words.extend(self.globals.copy())
        emit(words, OP_FUNCTION, self.type_void, self.main, 0, self.type_function)
        words.append(self.code[0])
        words.append(self.code[1])
        words.extend(self.variables.copy())
        for index in range(2, len(self.code)):
            words.append(self.code[index])
        return words^

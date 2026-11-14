"""Radeon shader assembler and compute command builder.

Shaders are written once against the methods here, and each GPU generation supplies
only its encodings: an ISA value selects them instruction by instruction. Encodings
are checked against LLVM's gfx12_asm_*.s and gfx10_asm_*.s tests, and register offsets
come from Mesa's gfx12.json and gfx103.json, all used as reference text only.
"""
from flint.platform.amdgpu import GpuBuffer, pkt3

comptime ISA_GFX103: Int = 103
comptime ISA_GFX12: Int = 120

comptime PKT3_NOP_DISPATCH_DIRECT: UInt32 = 0x15
comptime PKT3_DISPATCH_INDIRECT: UInt32 = 0x16
comptime PKT3_COPY_DATA: UInt32 = 0x40
comptime PKT3_WAIT_REG_MEM: UInt32 = 0x3C
comptime PKT3_EVENT_WRITE: UInt32 = 0x46
comptime PKT3_ACQUIRE_MEM: UInt32 = 0x58
comptime PKT3_SET_SH_REG: UInt32 = 0x76
comptime SHADER_TYPE_COMPUTE: UInt32 = 2
comptime SH_REG_BASE: UInt32 = 0xB000

comptime REG_COMPUTE_START_X: UInt32 = 0xB810
comptime REG_COMPUTE_NUM_THREAD_X: UInt32 = 0xB81C
comptime REG_COMPUTE_PERFCOUNT_ENABLE: UInt32 = 0xB82C
comptime REG_COMPUTE_PGM_LO: UInt32 = 0xB830
comptime REG_COMPUTE_DISPATCH_PKT_ADDR_LO: UInt32 = 0xB838
comptime REG_COMPUTE_PGM_RSRC1: UInt32 = 0xB848
comptime REG_COMPUTE_RESOURCE_LIMITS: UInt32 = 0xB854
comptime REG_COMPUTE_STATIC_THREAD_MGMT_SE0: UInt32 = 0xB858
comptime REG_COMPUTE_TMPRING_SIZE: UInt32 = 0xB860
comptime REG_COMPUTE_STATIC_THREAD_MGMT_SE2: UInt32 = 0xB864
comptime REG_COMPUTE_STATIC_THREAD_MGMT_SE8: UInt32 = 0xB88C
comptime REG_COMPUTE_USER_ACCUM_0: UInt32 = 0xB890
comptime REG_COMPUTE_PGM_RSRC3: UInt32 = 0xB8A0
comptime REG_COMPUTE_STATIC_THREAD_MGMT_SE4: UInt32 = 0xB8AC
comptime REG_COMPUTE_DISPATCH_INTERLEAVE: UInt32 = 0xB8BC
comptime REG_COMPUTE_USER_DATA_0: UInt32 = 0xB900
comptime REG_COMPUTE_DISPATCH_TUNNEL: UInt32 = 0xB9F4

comptime INITIATOR_ENABLE: UInt32 = 1
comptime INITIATOR_FORCE_START_AT_000: UInt32 = 1 << 2
comptime INITIATOR_WAVE32: UInt32 = 1 << 15
# COMPUTE_PGM_RSRC2 entry inputs: group ids in SGPRs and packed x/y ids in v0.
comptime INPUT_GROUP_X: UInt32 = 1 << 7
comptime INPUT_GROUP_Y: UInt32 = 1 << 8
comptime INPUT_THREAD_XY: UInt32 = 1 << 11
comptime EVENT_CS_PARTIAL_FLUSH: UInt32 = 7 | (4 << 8)
# COPY_DATA from the GPU clock to memory: 64 bits, confirmed before the next packet.
comptime COPY_DATA_SRC_TIMESTAMP: UInt32 = 9
comptime COPY_DATA_SRC_IMMEDIATE: UInt32 = 5
comptime WAIT_REG_MEM_EQUAL: UInt32 = 3
comptime WAIT_REG_MEM_MEMORY: UInt32 = 1 << 4
comptime COPY_DATA_DST_MEMORY: UInt32 = 5 << 8
comptime COPY_DATA_64_BITS: UInt32 = 1 << 16
comptime COPY_DATA_WRITE_CONFIRM: UInt32 = 1 << 20
# GCR_CNTL: invalidate the scalar and vector first-level caches.
comptime GCR_GLK_INV: UInt32 = 1 << 7
comptime GCR_GLV_INV: UInt32 = 1 << 8
comptime GCR_GL1_INV: UInt32 = 1 << 9
comptime GCR_GL2_WB: UInt32 = 1 << 15  # write dirty L2 lines back to memory

# Operand encodings shared by VOP1 and VOP2 source 0.
comptime SRC_LITERAL: UInt32 = 255
comptime SRC_VGPR_BASE: UInt32 = 256
comptime SRC_VCC_LO: UInt32 = 106
comptime SRC_TTMP7: UInt32 = 115
comptime SRC_TTMP9: UInt32 = 117
comptime SRC_EXEC_LO: UInt32 = 126
comptime SRC_FLOAT_HALF: UInt32 = 240
comptime SRC_FLOAT_MINUS_HALF: UInt32 = 241
comptime SRC_FLOAT_ONE: UInt32 = 242


def src_int(value: UInt32) -> UInt32:
    """Inline integer constant 0-64 as a source operand."""
    return 128 + value


def src_vgpr(index: UInt32) -> UInt32:
    return SRC_VGPR_BASE + index


struct ShaderAssembler(Movable):
    var isa: Int
    var words: List[UInt32]
    var label_targets: List[Int]
    var fixup_words: List[Int]
    var fixup_labels: List[Int]
    var far_branches: Int  # branches `resolve` found beyond the 16-bit offset's reach

    def __init__(out self, isa: Int = ISA_GFX12):
        self.isa = isa
        self.words = List[UInt32]()
        self.label_targets = List[Int]()
        self.fixup_words = List[Int]()
        self.fixup_labels = List[Int]()
        self.far_branches = 0

    def v_mov_b32(mut self, vdst: UInt32, src0: UInt32):
        self.words.append(0x7E000200 | (vdst << 17) | src0)

    def v_add_nc_u32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x4A000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def pick(self, gfx12: UInt32, gfx103: UInt32) -> UInt32:
        return gfx12 if self.isa == ISA_GFX12 else gfx103

    def v_lshlrev_b32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(self.pick(0x30000000, 0x34000000) | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_mul_u32_u24(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x16000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_and_b32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x36000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_or_b32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x38000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_xor_b32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x3A000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_lshrrev_b32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(self.pick(0x32000000, 0x2C000000) | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_ashrrev_i32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(self.pick(0x34000000, 0x30000000) | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_cndmask_b32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        """vdst = vsrc1 where vcc is set, otherwise src0."""
        self.words.append(0x02000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_add_f32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x06000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_sub_f32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x08000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_mul_f32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x10000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_max_num_f32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(self.pick(0x2C000000, 0x20000000) | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_min_num_f32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(self.pick(0x2A000000, 0x1E000000) | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_mul_i32_i24(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x12000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_min_i32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x22000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_max_i32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x24000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_min_u32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x26000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_max_u32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x28000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def v_sub_nc_u32(mut self, vdst: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x4C000000 | (vdst << 17) | (vsrc1 << 9) | src0)

    def vop1(mut self, opcode: UInt32, vdst: UInt32, src0: UInt32):
        self.words.append(0x7E000000 | (vdst << 17) | (opcode << 9) | src0)

    def v_cvt_f32_u32(mut self, vdst: UInt32, src0: UInt32):
        self.vop1(6, vdst, src0)

    def v_cvt_f32_i32(mut self, vdst: UInt32, src0: UInt32):
        self.vop1(5, vdst, src0)

    def v_cvt_u32_f32(mut self, vdst: UInt32, src0: UInt32):
        self.vop1(7, vdst, src0)

    def v_cvt_i32_f32(mut self, vdst: UInt32, src0: UInt32):
        self.vop1(8, vdst, src0)

    def v_fract_f32(mut self, vdst: UInt32, src0: UInt32):
        self.vop1(0x20, vdst, src0)

    def v_rcp_f32(mut self, vdst: UInt32, src0: UInt32):
        self.vop1(0x2A, vdst, src0)

    def v_rsq_f32(mut self, vdst: UInt32, src0: UInt32):
        self.vop1(0x2E, vdst, src0)

    def vopc(mut self, opcode: UInt32, src0: UInt32, vsrc1: UInt32):
        self.words.append(0x7C000000 | (opcode << 17) | (vsrc1 << 9) | src0)

    def v_cmp_lt_f32(mut self, src0: UInt32, vsrc1: UInt32):
        self.vopc(self.pick(0x11, 0x01), src0, vsrc1)

    def v_cmp_gt_f32(mut self, src0: UInt32, vsrc1: UInt32):
        self.vopc(self.pick(0x14, 0x04), src0, vsrc1)

    def v_cmp_ge_f32(mut self, src0: UInt32, vsrc1: UInt32):
        self.vopc(self.pick(0x16, 0x06), src0, vsrc1)

    def v_cmp_lt_u32(mut self, src0: UInt32, vsrc1: UInt32):
        self.vopc(self.pick(0x49, 0xC1), src0, vsrc1)

    def v_cmp_lt_i32(mut self, src0: UInt32, vsrc1: UInt32):
        self.vopc(self.pick(0x41, 0x81), src0, vsrc1)

    def v_cmp_gt_u32(mut self, src0: UInt32, vsrc1: UInt32):
        self.vopc(self.pick(0x4C, 0xC4), src0, vsrc1)

    def v_cmp_eq_u32(mut self, src0: UInt32, vsrc1: UInt32):
        self.vopc(self.pick(0x4A, 0xC2), src0, vsrc1)

    def v_cmp_ne_u32(mut self, src0: UInt32, vsrc1: UInt32):
        self.vopc(self.pick(0x4D, 0xC5), src0, vsrc1)

    def vop3(mut self, head: UInt32, vdst: UInt32, src0: UInt32, src1: UInt32, src2: UInt32):
        self.words.append(head | vdst)
        self.words.append(src0 | (src1 << 9) | (src2 << 18))

    def v_fma_f32(mut self, vdst: UInt32, src0: UInt32, src1: UInt32, src2: UInt32):
        self.vop3(self.pick(0xD6130000, 0xD54B0000), vdst, src0, src1, src2)

    def v_mad_u32_u24(mut self, vdst: UInt32, src0: UInt32, src1: UInt32, src2: UInt32):
        self.vop3(self.pick(0xD60B0000, 0xD5430000), vdst, src0, src1, src2)

    def v_lshl_add_u32(mut self, vdst: UInt32, src0: UInt32, src1: UInt32, src2: UInt32):
        self.vop3(self.pick(0xD6460000, 0xD7460000), vdst, src0, src1, src2)

    def s_mov_b32(mut self, sdst: UInt32, src0: UInt32):
        self.words.append(self.pick(0xBE800000, 0xBE800300) | (sdst << 16) | src0)

    def sop2(mut self, opcode: UInt32, sdst: UInt32, src0: UInt32, src1: UInt32):
        self.words.append((opcode << 24) | (sdst << 16) | (src1 << 8) | src0)

    def s_add_co_u32(mut self, sdst: UInt32, src0: UInt32, src1: UInt32):
        self.sop2(0x80, sdst, src0, src1)

    def s_lshl_b32(mut self, sdst: UInt32, src0: UInt32, src1: UInt32):
        self.sop2(self.pick(0x84, 0x8F), sdst, src0, src1)

    def s_lshr_b32(mut self, sdst: UInt32, src0: UInt32, src1: UInt32):
        self.sop2(self.pick(0x85, 0x90), sdst, src0, src1)

    def s_and_b32(mut self, sdst: UInt32, src0: UInt32, src1: UInt32):
        self.sop2(self.pick(0x8B, 0x87), sdst, src0, src1)

    def s_mul_i32(mut self, sdst: UInt32, src0: UInt32, src1: UInt32):
        self.sop2(self.pick(0x96, 0x93), sdst, src0, src1)

    def s_cmp_eq_u32(mut self, src0: UInt32, src1: UInt32):
        self.words.append(0xBF060000 | (src1 << 8) | src0)

    def s_cmp_ge_u32(mut self, src0: UInt32, src1: UInt32):
        self.words.append(0xBF090000 | (src1 << 8) | src0)

    def s_nop(mut self, count: UInt32):
        self.words.append(0xBF800000 | count)

    def s_wait_loadcnt(mut self):
        # On GFX10.3, s_waitcnt with vmcnt(0) and the other counters left at maximum.
        self.words.append(self.pick(0xBFC00000, 0xBF8C3F70))

    def s_wait_kmcnt(mut self):
        # On GFX10.3, s_waitcnt with lgkmcnt(0) and the other counters left at maximum.
        self.words.append(self.pick(0xBFC70000, 0xBF8CC07F))

    def smem_load(mut self, opcode: UInt32, sdata: UInt32, sbase: UInt32, soffset: UInt32):
        var shift = UInt32(13) if self.isa == ISA_GFX12 else UInt32(18)
        self.words.append(0xF4000000 | (opcode << shift) | (sdata << 6) | (sbase >> 1))
        self.words.append(soffset << 25)

    def s_load_b32(mut self, sdata: UInt32, sbase: UInt32, soffset: UInt32):
        self.smem_load(0, sdata, sbase, soffset)

    def s_load_b64(mut self, sdata: UInt32, sbase: UInt32, soffset: UInt32):
        self.smem_load(1, sdata, sbase, soffset)

    def s_load_b128(mut self, sdata: UInt32, sbase: UInt32, soffset: UInt32):
        self.smem_load(2, sdata, sbase, soffset)

    def s_load_b256(mut self, sdata: UInt32, sbase: UInt32, soffset: UInt32):
        self.smem_load(3, sdata, sbase, soffset)

    def s_load_b512(mut self, sdata: UInt32, sbase: UInt32, soffset: UInt32):
        self.smem_load(4, sdata, sbase, soffset)

    def global_load(mut self, gfx12_head: UInt32, gfx103_opcode: UInt32, vdata: UInt32, vaddr: UInt32, sbase: UInt32, offset: UInt32):
        if self.isa == ISA_GFX12:
            self.words.append(gfx12_head | sbase)
            self.words.append(vdata)
            self.words.append(vaddr | (offset << 8))
        else:
            self.words.append(0xDC008000 | (gfx103_opcode << 18) | offset)
            self.words.append(vaddr | (sbase << 16) | (vdata << 24))

    def global_load_b32(mut self, vdata: UInt32, vaddr: UInt32, sbase: UInt32, offset: UInt32 = 0):
        self.global_load(0xEE050000, 12, vdata, vaddr, sbase, offset)

    def global_load_u8(mut self, vdata: UInt32, vaddr: UInt32, sbase: UInt32):
        self.global_load(0xEE040000, 8, vdata, vaddr, sbase, 0)

    def new_label(mut self) -> Int:
        self.label_targets.append(-1)
        return len(self.label_targets) - 1

    def bind(mut self, label: Int):
        self.label_targets[label] = len(self.words)

    def branch(mut self, opcode: UInt32, label: Int):
        self.fixup_words.append(len(self.words))
        self.fixup_labels.append(label)
        self.words.append(0xBF800000 | (opcode << 16))

    def s_branch(mut self, label: Int):
        self.branch(self.pick(0x20, 0x02), label)

    def s_cbranch_scc1(mut self, label: Int):
        self.branch(self.pick(0x22, 0x05), label)

    def s_cbranch_scc0(mut self, label: Int):
        self.branch(self.pick(0x21, 0x04), label)

    def s_cbranch_execz(mut self, label: Int):
        self.branch(self.pick(0x25, 0x08), label)

    def resolve(mut self):
        """Patch branch offsets, which count dwords from the next instruction.
        A signed 16-bit offset reaches 32767 dwords either way; a branch past
        that is counted in `far_branches` (it would land in the wrong place:
        package 2.7 found a kernel that had one and hung the GPU)."""
        for index in range(len(self.fixup_words)):
            var position = self.fixup_words[index]
            var offset = self.label_targets[self.fixup_labels[index]] - position - 1
            if offset < -32768 or offset > 32767:
                self.far_branches += 1
            self.words[position] = self.words[position] | UInt32(offset & 0xFFFF)

    def literal(mut self, value: UInt32):
        self.words.append(value)

    def global_store_b32(mut self, vaddr: UInt32, vdata: UInt32, sbase: UInt32, offset: UInt32 = 0):
        """Store vdata at s[sbase:sbase+1] + vaddr + offset."""
        if self.isa == ISA_GFX12:
            self.words.append(0xEE068000 | sbase)
            self.words.append(vdata << 23)
            self.words.append(vaddr | (offset << 8))
        else:
            self.words.append(0xDC708000 | offset)
            self.words.append(vaddr | (vdata << 8) | (sbase << 16))

    def global_atomic_add_u32(mut self, vdst: UInt32, vaddr: UInt32, vdata: UInt32, sbase: UInt32):
        """vdst = old value; memory at s[sbase:sbase+1] + vaddr += vdata. The old value
        arrives like a load: wait with s_wait_loadcnt before reading vdst."""
        if self.isa == ISA_GFX12:
            self.words.append(0xEE0D4000 | sbase)  # opcode 53, then TH_ATOMIC_RETURN below
            self.words.append(vdst | (UInt32(1) << 20) | (vdata << 23))
            self.words.append(vaddr)
        else:
            self.words.append(0xDC008000 | (UInt32(0x32) << 18) | (UInt32(1) << 16))  # GLC returns
            self.words.append(vaddr | (vdata << 8) | (sbase << 16) | (vdst << 24))

    def s_endpgm(mut self):
        self.words.append(self.pick(0xBFB00000, 0xBF810000))

    def read_thread_position(mut self, user_sgprs: UInt32):
        """Leave a thread's id in v1 and v2, and its group's in s16 and s18.

        GFX12 packs thread ids into v0 and delivers group ids in trap registers.
        GFX10.3 gives each id its own VGPR and puts group ids after the user SGPRs.
        """
        if self.isa == ISA_GFX12:
            self.v_and_b32(1, src_int(7), 0)
            self.v_lshrrev_b32(2, src_int(10), 0)
            self.v_and_b32(2, src_int(3), 2)
            self.s_mov_b32(16, SRC_TTMP9)
        else:
            self.v_mov_b32(2, src_vgpr(1))
            self.v_mov_b32(1, src_vgpr(0))
            self.s_mov_b32(18, user_sgprs + 1)
            self.s_mov_b32(16, user_sgprs)

    def write_to(self, buffer: GpuBuffer, first_dword: Int = 0) -> Int:
        for index in range(len(self.words)):
            buffer.write32(first_dword + index, self.words[index])
        return len(self.words)


struct CommandBuilder(Movable):
    var isa: Int
    var words: List[UInt32]

    def __init__(out self, isa: Int = ISA_GFX12):
        self.isa = isa
        self.words = List[UInt32]()

    def set_sh(mut self, register: UInt32, values: List[UInt32]):
        self.words.append(pkt3(PKT3_SET_SH_REG, UInt32(len(values)) + 1) | SHADER_TYPE_COMPUTE)
        self.words.append((register - SH_REG_BASE) >> 2)
        for value in values:
            self.words.append(value)

    def set_sh_repeat(mut self, register: UInt32, count: Int, value: UInt32):
        var values = List[UInt32](length=count, fill=value)
        self.set_sh(register, values)

    def set_sh_one(mut self, register: UInt32, value: UInt32):
        self.set_sh_repeat(register, 1, value)

    def compute_state(mut self, shader_address: UInt64, threads_x: UInt32, threads_y: UInt32, threads_z: UInt32, user_sgprs: UInt32, vgprs: UInt32, entry_inputs: UInt32 = 0):
        """Program every compute register a dispatch reads, in wave32 mode."""
        var all_compute_units = UInt32(0xFFFFFFFF)
        self.set_sh_repeat(REG_COMPUTE_START_X, 3, 0)
        var threads = List[UInt32]()
        threads.append(threads_x)
        threads.append(threads_y)
        threads.append(threads_z)
        self.set_sh(REG_COMPUTE_NUM_THREAD_X, threads)
        self.set_sh_one(REG_COMPUTE_PERFCOUNT_ENABLE, 0)
        var program = List[UInt32]()
        program.append(UInt32((shader_address >> 8) & 0xFFFFFFFF))
        program.append(UInt32(shader_address >> 40))
        self.set_sh(REG_COMPUTE_PGM_LO, program)
        self.set_sh_repeat(REG_COMPUTE_DISPATCH_PKT_ADDR_LO, 2, 0)

        var rsrc = List[UInt32]()
        # Wave32 allocates VGPRs in blocks of eight. FLOAT_MODE 0xF0 keeps denormals.
        rsrc.append(((max(vgprs, 1) - 1) // 8) | (UInt32(0xF0) << 12) | (UInt32(1) << 30))
        rsrc.append((user_sgprs << 1) | entry_inputs)
        self.set_sh(REG_COMPUTE_PGM_RSRC1, rsrc)
        self.set_sh_one(REG_COMPUTE_PGM_RSRC3, 0)
        self.set_sh_one(REG_COMPUTE_RESOURCE_LIMITS, 0)

        self.set_sh_repeat(REG_COMPUTE_STATIC_THREAD_MGMT_SE0, 2, all_compute_units)
        self.set_sh_one(REG_COMPUTE_TMPRING_SIZE, 0)
        self.set_sh_repeat(REG_COMPUTE_STATIC_THREAD_MGMT_SE2, 2, all_compute_units)
        self.set_sh_repeat(REG_COMPUTE_STATIC_THREAD_MGMT_SE4, 4, 0)
        self.set_sh_repeat(REG_COMPUTE_USER_ACCUM_0, 4, 0)
        self.set_sh_one(REG_COMPUTE_DISPATCH_TUNNEL, 0)
        if self.isa == ISA_GFX12:
            self.set_sh_one(REG_COMPUTE_STATIC_THREAD_MGMT_SE8, 0)
            self.set_sh_one(REG_COMPUTE_DISPATCH_INTERLEAVE, 256)

    def user_address(mut self, slot: UInt32, address: UInt64):
        """Put a 64-bit address in s[slot:slot+1] at shader entry."""
        var values = List[UInt32]()
        values.append(UInt32(address & 0xFFFFFFFF))
        values.append(UInt32(address >> 32))
        self.set_sh(REG_COMPUTE_USER_DATA_0 + slot * 4, values)

    def user_values(mut self, slot: UInt32, values: List[UInt32]):
        """Put 32-bit values in s[slot], s[slot+1], ... at shader entry."""
        self.set_sh(REG_COMPUTE_USER_DATA_0 + slot * 4, values)

    def dispatch(mut self, groups_x: UInt32, groups_y: UInt32, groups_z: UInt32):
        self.words.append(pkt3(PKT3_NOP_DISPATCH_DIRECT, 4) | SHADER_TYPE_COMPUTE)
        self.words.append(groups_x)
        self.words.append(groups_y)
        self.words.append(groups_z)
        self.words.append(INITIATOR_ENABLE | INITIATOR_FORCE_START_AT_000 | INITIATOR_WAVE32)

    def dispatch_indirect(mut self, address: UInt64):
        """Dispatch with the group counts (x, y, z dwords) at `address`, written
        earlier on the GPU. On the compute queue (the MEC) the packet carries the
        address itself, as Mesa's radv emits it; the counts must be visible to the
        command processor, which a pass_barrier after their writer makes sure of."""
        self.words.append(pkt3(PKT3_DISPATCH_INDIRECT, 3) | SHADER_TYPE_COMPUTE)
        self.words.append(UInt32(address & 0xFFFFFFFF))
        self.words.append(UInt32(address >> 32))
        self.words.append(INITIATOR_ENABLE | INITIATOR_FORCE_START_AT_000 | INITIATOR_WAVE32)

    def wait_for_shaders(mut self):
        self.words.append(pkt3(PKT3_EVENT_WRITE, 1))
        self.words.append(EVENT_CS_PARTIAL_FLUSH)

    def pass_barrier(mut self):
        """Let finished shader writes reach the next pass, which may hold stale cache lines."""
        self.wait_for_shaders()
        self.words.append(pkt3(PKT3_ACQUIRE_MEM, 7))
        self.words.append(0)  # CP_COHER_CNTL
        self.words.append(0xFFFFFFFF)  # CP_COHER_SIZE
        self.words.append(0x00FFFFFF)  # CP_COHER_SIZE_HI
        self.words.append(0)  # CP_COHER_BASE
        self.words.append(0)  # CP_COHER_BASE_HI
        self.words.append(0x0000000A)  # POLL_INTERVAL
        self.words.append(GCR_GLK_INV | GCR_GLV_INV | (UInt32(0) if self.isa == ISA_GFX12 else GCR_GL1_INV))

    def arguments_barrier(mut self):
        """A pass_barrier that also writes the L2 back to memory, for a pass
        whose output the command processor itself reads next (indirect
        dispatch counts in system memory): the CP does not read through the
        shaders' L2, so without this it sees the counts of the frame before
        (package 2.7 measured exactly that)."""
        self.wait_for_shaders()
        self.words.append(pkt3(PKT3_ACQUIRE_MEM, 7))
        self.words.append(0)  # CP_COHER_CNTL
        self.words.append(0xFFFFFFFF)  # CP_COHER_SIZE
        self.words.append(0x00FFFFFF)  # CP_COHER_SIZE_HI
        self.words.append(0)  # CP_COHER_BASE
        self.words.append(0)  # CP_COHER_BASE_HI
        self.words.append(0x0000000A)  # POLL_INTERVAL
        self.words.append(GCR_GLK_INV | GCR_GLV_INV | GCR_GL2_WB | (UInt32(0) if self.isa == ISA_GFX12 else GCR_GL1_INV))

    def write_timestamp(mut self, address: UInt64):
        """Store the GPU clock at this point in the queue. After `wait_for_shaders`
        or `pass_barrier` that is the moment the pass before it finished."""
        self.words.append(pkt3(PKT3_COPY_DATA, 5))
        self.words.append(COPY_DATA_SRC_TIMESTAMP | COPY_DATA_DST_MEMORY | COPY_DATA_64_BITS | COPY_DATA_WRITE_CONFIRM)
        self.words.append(0)
        self.words.append(0)
        self.words.append(UInt32(address & 0xFFFFFFFF))
        self.words.append(UInt32(address >> 32))

    def write_marker(mut self, address: UInt64, value: UInt32):
        """Store `value` at `address` when the command processor gets here. After a
        hang, the markers say which pass was running (render/gpu_raster.mojo)."""
        self.words.append(pkt3(PKT3_COPY_DATA, 5))
        self.words.append(COPY_DATA_SRC_IMMEDIATE | COPY_DATA_DST_MEMORY | COPY_DATA_WRITE_CONFIRM)
        self.words.append(value)
        self.words.append(0)
        self.words.append(UInt32(address & 0xFFFFFFFF))
        self.words.append(UInt32(address >> 32))

    def wait_memory(mut self, address: UInt64, value: UInt32):
        """Stall the queue until the dword at `address` equals `value`. The fault
        injection uses it to hold a frame on the GPU until the CPU lets go, which
        times a fence out without a real hang."""
        self.words.append(pkt3(PKT3_WAIT_REG_MEM, 6))
        self.words.append(WAIT_REG_MEM_EQUAL | WAIT_REG_MEM_MEMORY)
        self.words.append(UInt32(address & 0xFFFFFFFC))
        self.words.append(UInt32(address >> 32))
        self.words.append(value)
        self.words.append(0xFFFFFFFF)  # mask
        self.words.append(4)  # poll interval

    def write_to(self, buffer: GpuBuffer) -> Int:
        for index in range(len(self.words)):
            buffer.write32(index, self.words[index])
        return len(self.words)

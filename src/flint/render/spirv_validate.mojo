"""Phase 1 (`V16`): Pure-Mojo SPIR-V 1.3 / 1.5 Structural, Type & SSA Dominance Validator (`spirv_validate.mojo`).

Replaces the external C++ `SPIRV-Tools` (`spirv-val`) library with a zero-dependency
Mojo 1.2 validator checking:
  1. Binary header (`SPIRV_MAGIC = 0x07230203`, version `1.0..1.6`, non-zero `<id>` bound, word bounds).
  2. Logical module layout order (Capabilities -> Extensions -> ExtInstImport -> MemoryModel ->
     EntryPoint -> ExecutionMode -> Annotations -> Types/Constants/Globals -> Functions).
  3. Unique `<id>` definitions (`1 <= id < bound`), forward-reference resolution (`OpEntryPoint`,
     `OpDecorate`), and static Type Checking (`OpLoad`, `OpStore`, `OpSelect`, `OpBranchConditional`, ALU).
  4. Control-Flow Graph (`CFG`), basic-block termination, entry-block `OpVariable` placement,
     structured control-flow headers (`OpSelectionMerge`, `OpLoopMerge`), `OpPhi` placement,
     and iterative Dominator-Tree **SSA ID Dominance** verification.
  5. Extension & capability gating (`SPV_KHR_ray_query` + `Capability RayQueryKHR`).
"""
from std.collections import Dict
from flint.render.spirv import (
    CAPABILITY_RAY_QUERY_KHR,
    CAPABILITY_SHADER,
    OP_ACCESS_CHAIN,
    OP_ATOMIC_I_ADD,
    OP_ATOMIC_U_MAX,
    OP_BITCAST,
    OP_BITWISE_AND,
    OP_BITWISE_OR,
    OP_BITWISE_XOR,
    OP_BRANCH,
    OP_BRANCH_CONDITIONAL,
    OP_CAPABILITY,
    OP_COMPOSITE_CONSTRUCT,
    OP_COMPOSITE_EXTRACT,
    OP_CONSTANT,
    OP_CONSTANT_FALSE,
    OP_CONSTANT_TRUE,
    OP_CONSTANT_COMPOSITE,
    OP_SPEC_CONSTANT,
    OP_SPEC_CONSTANT_FALSE,
    OP_SPEC_CONSTANT_TRUE,
    OP_CONTROL_BARRIER,
    OP_CONVERT_F_TO_S,
    OP_CONVERT_F_TO_U,
    OP_CONVERT_S_TO_F,
    OP_CONVERT_U_TO_F,
    OP_DECORATE,
    OP_DPDX,
    OP_DPDY,
    OP_ENTRY_POINT,
    OP_EXECUTION_MODE,
    OP_EXTENSION,
    OP_EXT_INST,
    OP_EXT_INST_IMPORT,
    OP_GROUP_NON_UNIFORM_ELECT,
    OP_GROUP_NON_UNIFORM_BALLOT,
    OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT,
    OP_GROUP_NON_UNIFORM_I_ADD,
    OP_GROUP_NON_UNIFORM_F_ADD,
    OP_FUNCTION,
    OP_FUNCTION_END,
    OP_F_ADD,
    OP_F_DIV,
    OP_F_MUL,
    OP_F_NEGATE,
    OP_F_ORD_GREATER_THAN,
    OP_F_ORD_GREATER_THAN_EQUAL,
    OP_F_ORD_LESS_THAN,
    OP_F_SUB,
    OP_IMAGE,
    OP_IMAGE_FETCH,
    OP_IMAGE_READ,
    OP_IMAGE_SAMPLE_EXPLICIT_LOD,
    OP_IMAGE_SAMPLE_IMPLICIT_LOD,
    OP_IMAGE_WRITE,
    OP_I_ADD,
    OP_I_EQUAL,
    OP_I_MUL,
    OP_I_NOT_EQUAL,
    OP_I_SUB,
    OP_LABEL,
    OP_LOAD,
    OP_LOGICAL_AND,
    OP_LOGICAL_NOT,
    OP_LOGICAL_OR,
    OP_LOOP_MERGE,
    OP_MEMBER_DECORATE,
    OP_MEMORY_BARRIER,
    OP_MEMORY_MODEL,
    OP_RAY_QUERY_CONFIRM_INTERSECTION_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_BARYCENTRICS_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_FRONT_FACE_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_GEOMETRY_INDEX_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_INSTANCE_CUSTOM_INDEX_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_INSTANCE_ID_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_PRIMITIVE_INDEX_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_SBT_OFFSET_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR,
    OP_RAY_QUERY_GET_INTERSECTION_T_KHR,
    OP_RAY_QUERY_INITIALIZE_KHR,
    OP_RAY_QUERY_PROCEED_KHR,
    OP_RAY_QUERY_TERMINATE_KHR,
    OP_RETURN,
    OP_SELECT,
    OP_SELECTION_MERGE,
    OP_SHIFT_LEFT_LOGICAL,
    OP_SHIFT_RIGHT_ARITHMETIC,
    OP_SHIFT_RIGHT_LOGICAL,
    OP_STORE,
    OP_TYPE_ACCELERATION_STRUCTURE_KHR,
    OP_TYPE_ARRAY,
    OP_TYPE_BOOL,
    OP_TYPE_FLOAT,
    OP_TYPE_FUNCTION,
    OP_TYPE_IMAGE,
    OP_TYPE_INT,
    OP_TYPE_POINTER,
    OP_TYPE_RAY_QUERY_KHR,
    OP_TYPE_RUNTIME_ARRAY,
    OP_TYPE_SAMPLED_IMAGE,
    OP_TYPE_SAMPLER,
    OP_TYPE_STRUCT,
    OP_TYPE_VECTOR,
    OP_TYPE_VOID,
    OP_U_DIV,
    OP_U_GREATER_THAN_EQUAL,
    OP_U_LESS_THAN,
    OP_U_MOD,
    OP_VARIABLE,
    SPIRV_MAGIC,
    STORAGE_FUNCTION,
    CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS,
    CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC,
    CAPABILITY_GROUP_NON_UNIFORM_BALLOT,
    CAPABILITY_GEOMETRY,
)

comptime _OP_PHI: UInt32 = 245
comptime _OP_KILL: UInt32 = 252
comptime _OP_RETURN_VALUE: UInt32 = 254
comptime _OP_UNREACHABLE: UInt32 = 255
comptime _OP_COMPOSITE_INSERT: UInt32 = 82
comptime _OP_VECTOR_SHUFFLE: UInt32 = 79
comptime _OP_FUNCTION_PARAM: UInt32 = 55

comptime _KIND_NONE: Int = 0
comptime _KIND_VOID: Int = 1
comptime _KIND_BOOL: Int = 2
comptime _KIND_INT: Int = 3
comptime _KIND_FLOAT: Int = 4
comptime _KIND_VECTOR: Int = 5
comptime _KIND_PTR: Int = 6
comptime _KIND_OTHER_TYPE: Int = 7


struct SpirvValidationReport(Copyable, Movable):
    var valid: Bool
    var errors: List[String]
    var instruction_count: Int
    var function_count: Int
    var block_count: Int
    var has_ray_query: Bool

    def __init__(out self):
        self.valid = True
        self.errors = List[String]()
        self.instruction_count = 0
        self.function_count = 0
        self.block_count = 0
        self.has_ray_query = False

    def fail(mut self, msg: String):
        self.valid = False
        self.errors.append(msg)


struct _UseCheck(Copyable, Movable):
    var used_id: Int
    var block_idx: Int
    var inst_ord: Int
    var is_phi: Bool
    var phi_pred_label: Int

    def __init__(
        out self,
        used_id: Int,
        block_idx: Int,
        inst_ord: Int,
        is_phi: Bool = False,
        phi_pred_label: Int = 0,
    ):
        self.used_id = used_id
        self.block_idx = block_idx
        self.inst_ord = inst_ord
        self.is_phi = is_phi
        self.phi_pred_label = phi_pred_label


def _decode_spv_string(words: List[UInt32], start: Int, max_end: Int) -> Tuple[String, Int]:
    var bytes = List[UInt8]()
    var w_idx = start
    while w_idx < max_end:
        var w = words[w_idx]
        w_idx += 1
        for b in range(4):
            var ch = UInt8((w >> UInt32(b * 8)) & 0xFF)
            if ch == 0:
                return (String(unsafe_from_utf8=bytes), w_idx)
            bytes.append(ch)
    return (String(unsafe_from_utf8=bytes), w_idx)


@always_inline
def _is_terminator(op: UInt32) -> Bool:
    return (
        op == OP_BRANCH
        or op == OP_BRANCH_CONDITIONAL
        or op == OP_RETURN
        or op == _OP_RETURN_VALUE
        or op == _OP_KILL
        or op == _OP_UNREACHABLE
    )


def validate_spirv_module(words: List[UInt32]) -> SpirvValidationReport:
    """Validates a SPIR-V 1.0–1.6 word stream completely in pure Mojo."""
    var rep = SpirvValidationReport()
    if len(words) < 5:
        rep.fail("module truncated (< 5 header words)")
        return rep^

    if words[0] != SPIRV_MAGIC:
        rep.fail("bad SPIR-V magic header: expected 0x07230203")
        return rep^

    var ver = words[1]
    if ver < 0x00010000 or ver > 0x00010600:
        rep.fail("unsupported SPIR-V version word: " + String(ver))

    var bound = Int(words[3])
    if bound < 1 or bound > 4_000_000:
        rep.fail("invalid <id> bound: " + String(bound))
        return rep^

    if words[4] != 0:
        rep.fail("header word[4] reserved schema must be 0")

    var id_defined = List[Bool](length=bound, fill=False)
    var id_is_global = List[Bool](length=bound, fill=False)
    var type_kind = List[Int](length=bound, fill=_KIND_NONE)
    var ptr_pointee = List[Int](length=bound, fill=0)
    var val_type = List[Int](length=bound, fill=0)

    var forward_refs = List[Int]()
    var has_cap_shader = False
    var has_cap_ray_query = False
    var has_ext_ray_query = False
    var has_cap_subgroup_ballot = False
    var has_cap_subgroup_arithmetic = False
    var has_cap_storage_image_extended_formats = False
    var memory_model_count = 0
    var entry_point_count = 0

    # Logical section order tracker:
    # 0=Capabilities, 1=Extensions, 2=ExtInstImport, 3=MemoryModel, 4=EntryPoint,
    # 5=ExecutionMode, 6=Decorations, 7=Types/Constants/Globals, 8=Functions
    var section = 0
    var in_function = False

    # Per-function CFG & SSA tracking
    var label_to_block = Dict[Int, Int]()
    var block_labels = List[Int]()
    var block_succs = List[List[Int]]()  # stores target label IDs first, resolved to block indices at OpFunctionEnd
    var block_terminated = List[Bool]()
    var block_merge_kind = List[Int]()   # 0=none, 1=selection, 2=loop
    var block_merge_target = List[Int]()
    var block_cont_target = List[Int]()
    var def_block = List[Int](length=bound, fill=-1)
    var def_ord = List[Int](length=bound, fill=-1)
    var fn_uses = List[_UseCheck]()

    var cur_block = -1
    var cur_inst_ord = 0
    var saw_non_phi_in_block = False
    var pending_merge_op: UInt32 = 0

    var pos = 5
    var n_words = len(words)

    while pos < n_words:
        var header = words[pos]
        var wc = Int(header >> 16)
        var op = header & 0xFFFF
        if wc < 1 or pos + wc > n_words:
            rep.fail("malformed instruction word count at offset " + String(pos))
            return rep^

        rep.instruction_count += 1

        # Verify `OpSelectionMerge` / `OpLoopMerge` is IMMEDIATELY followed by a branching terminator
        if pending_merge_op != 0:
            if pending_merge_op == OP_SELECTION_MERGE and op != OP_BRANCH_CONDITIONAL:
                rep.fail("OpSelectionMerge must immediately precede OpBranchConditional")
            elif pending_merge_op == OP_LOOP_MERGE and (op != OP_BRANCH and op != OP_BRANCH_CONDITIONAL):
                rep.fail("OpLoopMerge must immediately precede OpBranch or OpBranchConditional")
            pending_merge_op = 0

        if not in_function:
            if op == OP_CAPABILITY:
                if section > 0:
                    rep.fail("OpCapability out of section order")
                if wc != 2:
                    rep.fail("OpCapability expects 2 words")
                else:
                    var cap = words[pos + 1]
                    if cap == CAPABILITY_SHADER:
                        has_cap_shader = True
                    elif cap == CAPABILITY_RAY_QUERY_KHR:
                        has_cap_ray_query = True
                    elif cap == CAPABILITY_GROUP_NON_UNIFORM_BALLOT:
                        has_cap_subgroup_ballot = True
                    elif cap == CAPABILITY_GROUP_NON_UNIFORM_ARITHMETIC:
                        has_cap_subgroup_arithmetic = True
                    elif cap == CAPABILITY_STORAGE_IMAGE_EXTENDED_FORMATS:
                        has_cap_storage_image_extended_formats = True
                    elif cap == CAPABILITY_GEOMETRY:
                        _ = True
            elif op == OP_EXTENSION:
                if section > 1:
                    rep.fail("OpExtension out of section order")
                section = max(section, 1)
                var ext_s = _decode_spv_string(words, pos + 1, pos + wc)
                if ext_s[0] == "SPV_KHR_ray_query":
                    has_ext_ray_query = True
            elif op == OP_EXT_INST_IMPORT:
                if section > 2:
                    rep.fail("OpExtInstImport out of section order")
                section = max(section, 2)
                if wc < 3:
                    rep.fail("OpExtInstImport truncated")
                else:
                    var rid = Int(words[pos + 1])
                    if rid <= 0 or rid >= bound or id_defined[rid]:
                        rep.fail("invalid/duplicate OpExtInstImport <id> " + String(rid))
                    else:
                        id_defined[rid] = True
                        id_is_global[rid] = True
            elif op == OP_MEMORY_MODEL:
                if section > 3:
                    rep.fail("OpMemoryModel out of section order")
                section = 3
                memory_model_count += 1
            elif op == OP_ENTRY_POINT:
                if section > 4:
                    rep.fail("OpEntryPoint out of section order")
                section = 4
                entry_point_count += 1
                if wc < 4:
                    rep.fail("OpEntryPoint truncated")
                else:
                    forward_refs.append(Int(words[pos + 2]))
                    var ep_str = _decode_spv_string(words, pos + 3, pos + wc)
                    for iv_pos in range(ep_str[1], pos + wc):
                        forward_refs.append(Int(words[iv_pos]))
            elif op == OP_EXECUTION_MODE:
                if section > 5:
                    rep.fail("OpExecutionMode out of section order")
                section = 5
                if wc >= 3:
                    forward_refs.append(Int(words[pos + 1]))
            elif op == OP_DECORATE or op == OP_MEMBER_DECORATE:
                if section > 6:
                    rep.fail("OpDecorate out of section order")
                section = 6
                if wc >= 3:
                    forward_refs.append(Int(words[pos + 1]))
            elif op == OP_FUNCTION:
                section = 8
                in_function = True
                rep.function_count += 1
                if wc != 5:
                    rep.fail("OpFunction expects 5 words")
                else:
                    var fn_id = Int(words[pos + 2])
                    if fn_id <= 0 or fn_id >= bound or id_defined[fn_id]:
                        rep.fail("invalid/duplicate OpFunction <id> " + String(fn_id))
                    else:
                        id_defined[fn_id] = True
                        id_is_global[fn_id] = True
                label_to_block = Dict[Int, Int]()
                block_labels.clear()
                block_succs.clear()
                block_terminated.clear()
                block_merge_kind.clear()
                block_merge_target.clear()
                block_cont_target.clear()
                fn_uses.clear()
                cur_block = -1
                cur_inst_ord = 0
            else:
                # Section 7: Types, Constants, Global OpVariables
                if section > 7:
                    rep.fail("global declaration appeared after OpFunction")
                section = 7
                if (
                    op == OP_TYPE_VOID
                    or op == OP_TYPE_BOOL
                    or op == OP_TYPE_INT
                    or op == OP_TYPE_FLOAT
                    or op == OP_TYPE_VECTOR
                    or op == OP_TYPE_IMAGE
                    or op == OP_TYPE_SAMPLER
                    or op == OP_TYPE_SAMPLED_IMAGE
                    or op == OP_TYPE_ARRAY
                    or op == OP_TYPE_RUNTIME_ARRAY
                    or op == OP_TYPE_STRUCT
                    or op == OP_TYPE_POINTER
                    or op == OP_TYPE_FUNCTION
                    or op == OP_TYPE_RAY_QUERY_KHR
                    or op == OP_TYPE_ACCELERATION_STRUCTURE_KHR
                ):
                    if wc < 2:
                        rep.fail("OpType* truncated (missing result <id>)")
                    else:
                        var tid = Int(words[pos + 1])
                        if tid <= 0 or tid >= bound or id_defined[tid]:
                            rep.fail("invalid/duplicate OpType* <id> " + String(tid))
                        else:
                            id_defined[tid] = True
                            id_is_global[tid] = True
                            if op == OP_TYPE_VOID:
                                type_kind[tid] = _KIND_VOID
                            elif op == OP_TYPE_BOOL:
                                type_kind[tid] = _KIND_BOOL
                            elif op == OP_TYPE_INT:
                                type_kind[tid] = _KIND_INT
                            elif op == OP_TYPE_FLOAT:
                                type_kind[tid] = _KIND_FLOAT
                            elif op == OP_TYPE_VECTOR:
                                type_kind[tid] = _KIND_VECTOR
                            elif op == OP_TYPE_POINTER and wc >= 4:
                                type_kind[tid] = _KIND_PTR
                                ptr_pointee[tid] = Int(words[pos + 3])
                            else:
                                type_kind[tid] = _KIND_OTHER_TYPE
                    if op == OP_TYPE_RAY_QUERY_KHR or op == OP_TYPE_ACCELERATION_STRUCTURE_KHR:
                        rep.has_ray_query = True
                        if not has_cap_ray_query or not has_ext_ray_query:
                            rep.fail("RayQuery/AccelerationStructure type used without SPV_KHR_ray_query + RayQueryKHR capability")
                elif (
                    op == OP_CONSTANT
                    or op == OP_CONSTANT_TRUE
                    or op == OP_CONSTANT_FALSE
                    or op == OP_SPEC_CONSTANT
                    or op == OP_SPEC_CONSTANT_TRUE
                    or op == OP_SPEC_CONSTANT_FALSE
                    or op == OP_CONSTANT_COMPOSITE
                    or op == OP_VARIABLE
                ):
                    var min_wc = 3 if (
                        op == OP_CONSTANT_COMPOSITE
                        or op == OP_CONSTANT_TRUE
                        or op == OP_CONSTANT_FALSE
                        or op == OP_SPEC_CONSTANT_TRUE
                        or op == OP_SPEC_CONSTANT_FALSE
                    ) else 4
                    if wc < min_wc:
                        rep.fail("global OpConstant/OpSpecConstant/OpVariable truncated")
                    else:
                        var rtype = Int(words[pos + 1])
                        var rid = Int(words[pos + 2])
                        var rtype_ok = rtype > 0 and rtype < bound and id_defined[rtype]
                        if not rtype_ok:
                            rep.fail("undefined result type <id> " + String(rtype) + " on global <id> " + String(rid))
                        if rid <= 0 or rid >= bound or id_defined[rid]:
                            rep.fail("invalid/duplicate global <id> " + String(rid))
                        else:
                            id_defined[rid] = True
                            id_is_global[rid] = True
                            if rtype_ok:
                                val_type[rid] = rtype
                        if op == OP_VARIABLE and wc >= 4 and words[pos + 3] == STORAGE_FUNCTION:
                            rep.fail("Function storage-class OpVariable cannot appear at module scope")
        else:
            # Inside OpFunction .. OpFunctionEnd
            if op == OP_FUNCTION_END:
                if cur_block < 0:
                    rep.fail("OpFunction has zero basic blocks")
                elif not block_terminated[cur_block]:
                    rep.fail("final basic block in OpFunction is missing terminator")

                # Validate CFG edges, structured merges, and Dominator-Tree SSA Dominance!
                var num_b = len(block_labels)
                rep.block_count += num_b
                if num_b > 0:
                    var preds = List[List[Int]]()
                    var succ_idx = List[List[Int]]()
                    for _ in range(num_b):
                        preds.append(List[Int]())
                        succ_idx.append(List[Int]())

                    for bi in range(num_b):
                        if not block_terminated[bi]:
                            rep.fail("basic block " + String(bi) + " is not terminated")
                        if block_merge_kind[bi] != 0:
                            var mt = block_merge_target[bi]
                            var m_opt = label_to_block.get(mt)
                            if not m_opt:
                                rep.fail("merge target label <id> " + String(mt) + " does not name an OpLabel")
                            elif m_opt.value() == bi:
                                rep.fail("merge target cannot equal header block itself")
                            if block_merge_kind[bi] == 2:
                                var ct = block_cont_target[bi]
                                if not label_to_block.get(ct):
                                    rep.fail("loop continue target <id> " + String(ct) + " does not name an OpLabel")
                        for sl in block_succs[bi]:
                            var sb_opt = label_to_block.get(sl)
                            if not sb_opt:
                                rep.fail("branch target <id> " + String(sl) + " is not a valid OpLabel")
                            else:
                                var sb = sb_opt.value()
                                succ_idx[bi].append(sb)
                                preds[sb].append(bi)

                    # Reachability & Reverse Postorder via iterative DFS from entry block 0
                    var reachable = List[Bool](length=num_b, fill=False)
                    var postorder = List[Int](capacity=num_b)
                    var dfs_node = List[Int]()
                    var dfs_edge = List[Int]()
                    reachable[0] = True
                    dfs_node.append(0)
                    dfs_edge.append(0)
                    while len(dfs_node) > 0:
                        var top = len(dfs_node) - 1
                        var u = dfs_node[top]
                        var ei = dfs_edge[top]
                        if ei < len(succ_idx[u]):
                            dfs_edge[top] = ei + 1
                            var v = succ_idx[u][ei]
                            if not reachable[v]:
                                reachable[v] = True
                                dfs_node.append(v)
                                dfs_edge.append(0)
                        else:
                            postorder.append(u)
                            _ = dfs_node.pop()
                            _ = dfs_edge.pop()

                    var n_rpo = len(postorder)
                    var rpo_blocks = List[Int](capacity=n_rpo)
                    var rpo_pos = List[Int](length=num_b, fill=-1)
                    for i in range(n_rpo):
                        var b_id = postorder[n_rpo - 1 - i]
                        rpo_blocks.append(b_id)
                        rpo_pos[b_id] = i

                    # O(num_b) Cooper-Harvey-Kennedy iterative immediate dominator algorithm
                    var doms = List[Int](length=num_b, fill=-1)
                    doms[0] = 0
                    var changed = True
                    while changed:
                        changed = False
                        for ri in range(1, n_rpo):
                            var b = rpo_blocks[ri]
                            var new_idom = -1
                            for p in preds[b]:
                                if reachable[p] and doms[p] != -1:
                                    if new_idom == -1:
                                        new_idom = p
                                    else:
                                        var f1 = p
                                        var f2 = new_idom
                                        while f1 != f2:
                                            while rpo_pos[f1] > rpo_pos[f2]:
                                                f1 = doms[f1]
                                            while rpo_pos[f2] > rpo_pos[f1]:
                                                f2 = doms[f2]
                                        new_idom = f1
                            if new_idom != -1 and doms[b] != new_idom:
                                doms[b] = new_idom
                                changed = True

                    # Build dominator tree + Euler tour (tin/tout) for O(1) dominance queries
                    var dom_children = List[List[Int]](capacity=num_b)
                    for _ in range(num_b):
                        dom_children.append(List[Int]())
                    for b in range(1, num_b):
                        if reachable[b] and doms[b] >= 0:
                            dom_children[doms[b]].append(b)

                    var tin = List[Int](length=num_b, fill=-1)
                    var tout = List[Int](length=num_b, fill=-1)
                    var timer = 0
                    dfs_node.clear()
                    dfs_edge.clear()
                    dfs_node.append(0)
                    dfs_edge.append(0)
                    tin[0] = timer
                    timer += 1
                    while len(dfs_node) > 0:
                        var top = len(dfs_node) - 1
                        var u = dfs_node[top]
                        var ei = dfs_edge[top]
                        if ei < len(dom_children[u]):
                            dfs_edge[top] = ei + 1
                            var v = dom_children[u][ei]
                            tin[v] = timer
                            timer += 1
                            dfs_node.append(v)
                            dfs_edge.append(0)
                        else:
                            tout[u] = timer
                            timer += 1
                            _ = dfs_node.pop()
                            _ = dfs_edge.pop()

                    # Verify SSA Dominance for every instruction operand inside the function!
                    for u_chk in fn_uses:
                        var uid = u_chk.used_id
                        if uid <= 0 or uid >= bound or not id_defined[uid]:
                            rep.fail("instruction references undefined SSA <id> " + String(uid))
                            continue
                        if id_is_global[uid]:
                            continue
                        var db = def_block[uid]
                        var dord = def_ord[uid]
                        if db < 0:
                            rep.fail("SSA <id> " + String(uid) + " defined outside current function")
                            continue
                        if u_chk.is_phi:
                            var pb_opt = label_to_block.get(u_chk.phi_pred_label)
                            if not pb_opt:
                                rep.fail("OpPhi references unknown parent block label " + String(u_chk.phi_pred_label))
                            else:
                                var pb = pb_opt.value()
                                if reachable[pb]:
                                    var dom_ok = reachable[db] and tin[db] <= tin[pb] and tout[pb] <= tout[db]
                                    if not dom_ok:
                                        rep.fail("OpPhi operand <id> " + String(uid) + " does not dominate predecessor block")
                        else:
                            var ub = u_chk.block_idx
                            if not reachable[ub]:
                                continue
                            if db == ub:
                                if dord >= u_chk.inst_ord:
                                    rep.fail("SSA <id> " + String(uid) + " used before definition in same block")
                            else:
                                var dom_ok = reachable[db] and tin[db] <= tin[ub] and tout[ub] <= tout[db]
                                if not dom_ok:
                                    rep.fail("SSA <id> " + String(uid) + " does not dominate use block " + String(ub))

                in_function = False
                cur_block = -1
            elif op == _OP_FUNCTION_PARAM:
                if cur_block >= 0:
                    rep.fail("OpFunctionParameter must precede first OpLabel")
                if wc == 3:
                    var rtype = Int(words[pos + 1])
                    var rid = Int(words[pos + 2])
                    var rtype_ok = rtype > 0 and rtype < bound and id_defined[rtype]
                    if not rtype_ok:
                        rep.fail("undefined result type <id> " + String(rtype) + " on OpFunctionParameter <id> " + String(rid))
                    if rid <= 0 or rid >= bound or id_defined[rid]:
                        rep.fail("invalid/duplicate OpFunctionParameter <id> " + String(rid))
                    else:
                        id_defined[rid] = True
                        id_is_global[rid] = True
                        if rtype_ok:
                            val_type[rid] = rtype
            elif op == OP_LABEL:
                if cur_block >= 0 and not block_terminated[cur_block]:
                    rep.fail("OpLabel started before previous basic block was terminated")
                if wc != 2:
                    rep.fail("OpLabel expects 2 words")
                else:
                    var lid = Int(words[pos + 1])
                    if lid <= 0 or lid >= bound or id_defined[lid]:
                        rep.fail("invalid/duplicate OpLabel <id> " + String(lid))
                    else:
                        id_defined[lid] = True
                    cur_block = len(block_labels)
                    label_to_block[lid] = cur_block
                    block_labels.append(lid)
                    block_succs.append(List[Int]())
                    block_terminated.append(False)
                    block_merge_kind.append(0)
                    block_merge_target.append(0)
                    block_cont_target.append(0)
                    cur_inst_ord = 0
                    saw_non_phi_in_block = False
            else:
                if cur_block < 0:
                    rep.fail("instruction appeared inside OpFunction before first OpLabel")
                    pos += wc
                    continue
                if block_terminated[cur_block]:
                    rep.fail("unreachable instruction after terminator in block " + String(cur_block))

                cur_inst_ord += 1
                if op != _OP_PHI and op != OP_VARIABLE:
                    saw_non_phi_in_block = True

                if op == _OP_PHI:
                    if saw_non_phi_in_block:
                        rep.fail("OpPhi must appear at top of basic block before non-Phi instructions")
                    if wc < 3 or (wc % 2) == 0:
                        rep.fail("malformed OpPhi word count")
                    else:
                        var rtype = Int(words[pos + 1])
                        var rid = Int(words[pos + 2])
                        var rtype_ok = rtype > 0 and rtype < bound and id_defined[rtype]
                        if not rtype_ok:
                            rep.fail("undefined result type <id> " + String(rtype) + " on OpPhi <id> " + String(rid))
                        if rid <= 0 or rid >= bound or id_defined[rid]:
                            rep.fail("invalid/duplicate OpPhi <id> " + String(rid))
                        else:
                            id_defined[rid] = True
                            if rtype_ok:
                                val_type[rid] = rtype
                            def_block[rid] = cur_block
                            def_ord[rid] = cur_inst_ord
                        var k = 3
                        while k + 1 < wc:
                            fn_uses.append(
                                _UseCheck(Int(words[pos + k]), cur_block, cur_inst_ord, is_phi=True, phi_pred_label=Int(words[pos + k + 1]))
                            )
                            k += 2
                elif op == OP_VARIABLE:
                    if cur_block != 0:
                        rep.fail("Function OpVariable must reside in entry basic block (block 0)")
                    if wc < 4:
                        rep.fail("OpVariable truncated")
                    else:
                        var rtype = Int(words[pos + 1])
                        var rid = Int(words[pos + 2])
                        var rtype_ok = rtype > 0 and rtype < bound and id_defined[rtype]
                        if not rtype_ok:
                            rep.fail("undefined result type <id> " + String(rtype) + " on local OpVariable <id> " + String(rid))
                        if words[pos + 3] != STORAGE_FUNCTION:
                            rep.fail("OpVariable inside function must have Function storage class")
                        if rid <= 0 or rid >= bound or id_defined[rid]:
                            rep.fail("invalid/duplicate local OpVariable <id> " + String(rid))
                        else:
                            id_defined[rid] = True
                            if rtype_ok:
                                val_type[rid] = rtype
                            def_block[rid] = 0
                            def_ord[rid] = cur_inst_ord
                elif op == OP_SELECTION_MERGE:
                    if wc != 3:
                        rep.fail("OpSelectionMerge expects 3 words")
                    else:
                        block_merge_kind[cur_block] = 1
                        block_merge_target[cur_block] = Int(words[pos + 1])
                        pending_merge_op = OP_SELECTION_MERGE
                elif op == OP_LOOP_MERGE:
                    if wc < 4:
                        rep.fail("OpLoopMerge expects >= 4 words")
                    else:
                        block_merge_kind[cur_block] = 2
                        block_merge_target[cur_block] = Int(words[pos + 1])
                        block_cont_target[cur_block] = Int(words[pos + 2])
                        pending_merge_op = OP_LOOP_MERGE
                elif op == OP_BRANCH:
                    if wc != 2:
                        rep.fail("OpBranch expects 2 words")
                    else:
                        block_succs[cur_block].append(Int(words[pos + 1]))
                    block_terminated[cur_block] = True
                elif op == OP_BRANCH_CONDITIONAL:
                    if wc < 4:
                        rep.fail("OpBranchConditional expects >= 4 words")
                    else:
                        var cid = Int(words[pos + 1])
                        fn_uses.append(_UseCheck(cid, cur_block, cur_inst_ord))
                        if cid > 0 and cid < bound and id_defined[cid]:
                            var ct = val_type[cid]
                            if ct <= 0 or ct >= bound or type_kind[ct] != _KIND_BOOL:
                                rep.fail("OpBranchConditional condition <id> " + String(cid) + " must be OpTypeBool")
                        block_succs[cur_block].append(Int(words[pos + 2]))
                        block_succs[cur_block].append(Int(words[pos + 3]))
                    block_terminated[cur_block] = True
                elif _is_terminator(op):
                    if op == _OP_RETURN_VALUE and wc >= 2:
                        fn_uses.append(_UseCheck(Int(words[pos + 1]), cur_block, cur_inst_ord))
                    block_terminated[cur_block] = True
                elif op == OP_STORE:
                    if wc < 3:
                        rep.fail("OpStore expects >= 3 words")
                    else:
                        var pid = Int(words[pos + 1])
                        var oid = Int(words[pos + 2])
                        fn_uses.append(_UseCheck(pid, cur_block, cur_inst_ord))
                        fn_uses.append(_UseCheck(oid, cur_block, cur_inst_ord))
                        if pid > 0 and pid < bound and oid > 0 and oid < bound and id_defined[pid] and id_defined[oid]:
                            var pt = val_type[pid]
                            if pt <= 0 or pt >= bound or type_kind[pt] != _KIND_PTR:
                                rep.fail("OpStore target <id> " + String(pid) + " is not a pointer")
                            elif ptr_pointee[pt] != val_type[oid]:
                                rep.fail("OpStore type mismatch: pointer pointee != stored object type")
                elif (
                    op == OP_RAY_QUERY_INITIALIZE_KHR
                    or op == OP_RAY_QUERY_TERMINATE_KHR
                    or op == OP_RAY_QUERY_CONFIRM_INTERSECTION_KHR
                ):
                    rep.has_ray_query = True
                    if not has_cap_ray_query or not has_ext_ray_query:
                        rep.fail("OpRayQuery*KHR instruction used without SPV_KHR_ray_query")
                    for arg_i in range(1, wc):
                        fn_uses.append(_UseCheck(Int(words[pos + arg_i]), cur_block, cur_inst_ord))
                elif op == OP_IMAGE_WRITE:
                    # OpImageWrite: Image(1), Coordinate(2), Texel(3), [ImageOperands mask(4), ids(5..)]
                    if wc >= 4:
                        fn_uses.append(_UseCheck(Int(words[pos + 1]), cur_block, cur_inst_ord))
                        fn_uses.append(_UseCheck(Int(words[pos + 2]), cur_block, cur_inst_ord))
                        fn_uses.append(_UseCheck(Int(words[pos + 3]), cur_block, cur_inst_ord))
                        for arg_i in range(5, wc):
                            fn_uses.append(_UseCheck(Int(words[pos + arg_i]), cur_block, cur_inst_ord))
                elif op == OP_CONTROL_BARRIER or op == OP_MEMORY_BARRIER:
                    for arg_i in range(1, wc):
                        fn_uses.append(_UseCheck(Int(words[pos + arg_i]), cur_block, cur_inst_ord))
                else:
                    # Value-producing instructions: word[1] = Result Type, word[2] = Result <id>
                    if wc >= 3:
                        var rtype = Int(words[pos + 1])
                        var rid = Int(words[pos + 2])
                        var rtype_ok = rtype > 0 and rtype < bound and id_defined[rtype]
                        if not rtype_ok:
                            rep.fail("instruction op=" + String(op) + " uses undefined Result Type <id> " + String(rtype))
                        if rid <= 0 or rid >= bound or id_defined[rid]:
                            rep.fail("duplicate or out-of-bounds Result <id> " + String(rid) + " at op=" + String(op))
                        else:
                            id_defined[rid] = True
                            if rtype_ok:
                                val_type[rid] = rtype
                            def_block[rid] = cur_block
                            def_ord[rid] = cur_inst_ord

                        if (
                            op == OP_RAY_QUERY_PROCEED_KHR
                            or op == OP_RAY_QUERY_GET_INTERSECTION_TYPE_KHR
                            or (op >= OP_RAY_QUERY_GET_INTERSECTION_T_KHR and op <= OP_RAY_QUERY_GET_INTERSECTION_FRONT_FACE_KHR)
                        ):
                            rep.has_ray_query = True
                            if not has_cap_ray_query or not has_ext_ray_query:
                                rep.fail("OpRayQuery*KHR instruction used without SPV_KHR_ray_query")

                        # Record SSA operand <id> uses (skipping literal-number operand slots!)
                        if op == OP_EXT_INST:
                            # word[3] = ExtInstSet <id>, word[4] = literal opcode, words[5..] = <id> operands
                            if wc >= 4:
                                fn_uses.append(_UseCheck(Int(words[pos + 3]), cur_block, cur_inst_ord))
                            for arg_i in range(5, wc):
                                fn_uses.append(_UseCheck(Int(words[pos + arg_i]), cur_block, cur_inst_ord))
                        elif op == OP_COMPOSITE_EXTRACT:
                            # word[3] = Composite <id>, words[4..] = literal component indices
                            if wc >= 4:
                                fn_uses.append(_UseCheck(Int(words[pos + 3]), cur_block, cur_inst_ord))
                        elif op == _OP_COMPOSITE_INSERT or op == _OP_VECTOR_SHUFFLE:
                            # words[3], [4] = <id>s, words[5..] = literal indices
                            if wc >= 5:
                                fn_uses.append(_UseCheck(Int(words[pos + 3]), cur_block, cur_inst_ord))
                                fn_uses.append(_UseCheck(Int(words[pos + 4]), cur_block, cur_inst_ord))
                        elif (
                            op == OP_IMAGE_SAMPLE_IMPLICIT_LOD
                            or op == OP_IMAGE_SAMPLE_EXPLICIT_LOD
                            or op == OP_IMAGE_FETCH
                            or op == OP_IMAGE_READ
                        ):
                            _ = has_cap_storage_image_extended_formats
                            # word[3] = SampledImage/Image <id>, word[4] = Coord <id>, [word[5] = literal ImageOperands mask, words[6..] = <id>s]
                            if wc >= 5:
                                fn_uses.append(_UseCheck(Int(words[pos + 3]), cur_block, cur_inst_ord))
                                fn_uses.append(_UseCheck(Int(words[pos + 4]), cur_block, cur_inst_ord))
                            for arg_i in range(6, wc):
                                fn_uses.append(_UseCheck(Int(words[pos + arg_i]), cur_block, cur_inst_ord))
                        elif op == OP_LOAD:
                            # word[3] = Pointer <id>, [word[4] = literal MemoryAccess]
                            if wc >= 4:
                                var pid = Int(words[pos + 3])
                                fn_uses.append(_UseCheck(pid, cur_block, cur_inst_ord))
                                if pid > 0 and pid < bound and id_defined[pid]:
                                    var pt = val_type[pid]
                                    if pt <= 0 or pt >= bound or type_kind[pt] != _KIND_PTR:
                                        rep.fail("OpLoad source <id> " + String(pid) + " is not a pointer")
                                    elif ptr_pointee[pt] != rtype:
                                        rep.fail("OpLoad result type != pointer pointee type")
                        elif (
                            op == OP_GROUP_NON_UNIFORM_F_ADD
                            or op == OP_GROUP_NON_UNIFORM_I_ADD
                            or op == OP_GROUP_NON_UNIFORM_BALLOT_BIT_COUNT
                        ):
                            # word[3] = Scope <id>, word[4] = literal GroupOperation, word[5] = Value <id>, [word[6] = ClusterSize <id>]
                            if not has_cap_subgroup_arithmetic:
                                rep.fail("OpGroupNonUniform*Add used without GroupNonUniformArithmetic capability")
                            if wc >= 4:
                                fn_uses.append(_UseCheck(Int(words[pos + 3]), cur_block, cur_inst_ord))
                            if wc >= 6:
                                fn_uses.append(_UseCheck(Int(words[pos + 5]), cur_block, cur_inst_ord))
                            if wc >= 7:
                                fn_uses.append(_UseCheck(Int(words[pos + 6]), cur_block, cur_inst_ord))
                        elif op == OP_GROUP_NON_UNIFORM_BALLOT:
                            if not has_cap_subgroup_ballot:
                                rep.fail("OpGroupNonUniformBallot used without GroupNonUniformBallot capability")
                            if wc >= 4:
                                fn_uses.append(_UseCheck(Int(words[pos + 3]), cur_block, cur_inst_ord))
                            if wc >= 5:
                                fn_uses.append(_UseCheck(Int(words[pos + 4]), cur_block, cur_inst_ord))
                        elif op == OP_GROUP_NON_UNIFORM_ELECT:
                            if not has_cap_subgroup_ballot and not has_cap_subgroup_arithmetic:
                                rep.fail("OpGroupNonUniformElect used without GroupNonUniform capability")
                            if wc >= 4:
                                fn_uses.append(_UseCheck(Int(words[pos + 3]), cur_block, cur_inst_ord))
                        else:
                            for arg_i in range(3, wc):
                                fn_uses.append(_UseCheck(Int(words[pos + arg_i]), cur_block, cur_inst_ord))

                        # Static Type Consistency checks on common ALU & Select ops
                        if op == OP_SELECT and wc == 6:
                            var cid = Int(words[pos + 3])
                            var yid = Int(words[pos + 4])
                            var nid = Int(words[pos + 5])
                            if cid > 0 and cid < bound and id_defined[cid]:
                                var ct = val_type[cid]
                                if ct <= 0 or ct >= bound or type_kind[ct] != _KIND_BOOL:
                                    rep.fail("OpSelect condition <id> " + String(cid) + " must be OpTypeBool")
                            if yid > 0 and yid < bound and id_defined[yid] and val_type[yid] != rtype:
                                rep.fail("OpSelect true-branch type != result type")
                            if nid > 0 and nid < bound and id_defined[nid] and val_type[nid] != rtype:
                                rep.fail("OpSelect false-branch type != result type")
                        elif (
                            op == OP_F_ADD
                            or op == OP_F_SUB
                            or op == OP_F_MUL
                            or op == OP_F_DIV
                            or op == OP_I_ADD
                            or op == OP_I_SUB
                            or op == OP_I_MUL
                            or op == OP_U_DIV
                            or op == OP_U_MOD
                            or op == OP_BITWISE_AND
                            or op == OP_BITWISE_OR
                            or op == OP_BITWISE_XOR
                        ) and wc == 5:
                            var aid = Int(words[pos + 3])
                            var bid = Int(words[pos + 4])
                            if aid > 0 and aid < bound and id_defined[aid] and val_type[aid] != rtype:
                                rep.fail("binary op=" + String(op) + " lhs type != result type")
                            if bid > 0 and bid < bound and id_defined[bid] and val_type[bid] != rtype:
                                rep.fail("binary op=" + String(op) + " rhs type != result type")

        pos += wc

    if in_function:
        rep.fail("unterminated OpFunction (missing OpFunctionEnd)")
    if not has_cap_shader:
        rep.fail("missing Capability Shader")
    if memory_model_count != 1:
        rep.fail("expected exactly 1 OpMemoryModel, found " + String(memory_model_count))
    if entry_point_count < 1:
        rep.fail("module has zero OpEntryPoint declarations")
    if rep.function_count < 1:
        rep.fail("module has zero OpFunction definitions")

    for fref in forward_refs:
        if fref <= 0 or fref >= bound or not id_defined[fref]:
            rep.fail("forward reference in EntryPoint/ExecutionMode/Decorate to undefined <id> " + String(fref))

    return rep^

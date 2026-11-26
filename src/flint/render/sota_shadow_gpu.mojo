"""Area G (shadows) render SOTA, round 2: GPU-driven VSM page allocation, caster culling and page raster
(`sota_shadow_gpu`, cvar `ren.sota_shadow.gpu_pages`, default off).

v5 reads the page requests back to the CPU, allocates physical pages in `VsmPageManager.process_requests`, culls
every instance against every dirty page in `_pack_vsm_indirect_commands` and records one render pass per dirty page.
On Forest a cold frame costs 22 ms of CPU and 673 render passes, and a camera move that dirties pages needs a
synchronous submit + wait, because the requests are only known after the GPU marked them.

Here it all stays on the GPU, in the frame's command buffer, right after `VsmMarkPages`:

1. `alloc0`  (1 group)     zero the per-frame counters, write the tile-clear draw header.
2. `alloc1`  (16384 thr)   per virtual page: apply the CPU invalidation bitmap of this frame slot, consume the
                           request flag, keep a cached page (touch its LRU stamp), queue a mapped-but-stale page for
                           re-render, queue an unmapped requested page for allocation.
3. `alloc2`  (1024 thr)    per physical page: free list (no owner), eviction lists (unused >= 2 frames, unused 1 frame).
4. `alloc3`  (16384 thr)   needs in cascade order (finest first, so an oversubscribed pool drops far pages, not
                           near ones); allocation i takes free[i], else old[...], else recent[...]; the victim's
                           entry clears.
5. `cull`    (64 thr/item) per (instance, 64 meshlets) item x cascade: project the meshlet's world sphere into the
                           cascade window, walk the pages it covers (toroidal slots) and append one draw per page that
                           renders this frame: `{tris*3, 1, meshlet_hi<<24 | phys_page<<10, inst<<17 | meshlet_lo}`.
6. raster    one dynamic-rendering pass over the whole 4096^2 atlas: an instanced quad per dirty page clears it to
             depth 1 (compare ALWAYS), then one `vkCmdDrawIndirectCount` for opaque and one for alpha-masked casters.
             The vertex shader decodes the physical page from `VertexIndex`, looks up its virtual page / cascade /
             window position and confines the triangle to its 128^2 tile with four `gl_ClipDistance` planes (no
             per-page viewport, no per-page render pass).

Everything lives in one buffer that replaces `buf_vsm_table` at `BIND_AAA_VSM_TABLE` while the mode is on, so the
v5 descriptor sets (with their bindless textures, needed by the alpha-tested pass) are reused unchanged:

    [0, 16384)        page entries, v5 format `0x10000 | py << 8 | px` (+ `0x20000` valid, `0x40000` render now;
                      samplers mask with 0xFF / test 0x10000, so the extra bits are invisible to them)
    [16384, 32768)    request flags (written by `VsmMarkPages`, consumed by `alloc1`)
    EXT  (4096)       P2V[1024] owner virtual page (~0 = free), LAST[1024] epoch, INVAL[4][512] CPU invalidations
    HDR  (48)         counters, per-cascade dirty-page boxes + the clear draw (see `SOTA_SHADOW_H_*`)
    ST   (20480)      dirty[1024], need[4][4096] (per cascade), free[1024], old[1024], recent[1024]
    ITEMS             per frame slot: `item_cap` x [inst | alpha << 31, first meshlet, count <= 64]
    CMDS              opaque `cmd_cap` x 4 words, then alpha `cmd_cap` x 4 words

CPU invalidations (sun rotation, moving casters, toroidal scroll, uploads) still go through
`VsmPageManager.virtual_valid`; each frame the renderer folds the cleared flags into this frame slot's bitmap.
Overflowing the command lists clears the page's valid bit (it re-renders next frame) and the CPU grows the lists
from the `WANTED` counter it reads one frame later.

Fallbacks: without `shaderClipDistance` or `drawIndirectCount`, outside cascade mode, or when a pipeline fails, the
v5 CPU path (with the Area G fixes) runs.
"""
from std.collections import List
from std.ffi import external_call
from std.math import max, min
from std.memory import bitcast
from flint.assets.meshlet import MeshLodSlice
from flint.core.math import Vec3
from flint.platform.amdgpu import GpuBuffer
from flint.platform.vulkan import (
    ACCESS_DEPTH_ATTACHMENT_READ,
    ACCESS_DEPTH_ATTACHMENT_WRITE,
    ACCESS_INDIRECT_READ,
    ACCESS_SHADER_READ,
    ACCESS_SHADER_WRITE,
    COMPARE_ALWAYS,
    COMPARE_LESS_OR_EQUAL,
    CULL_MODE_BACK,
    CULL_MODE_NONE,
    FORMAT_D32_SFLOAT,
    FRONT_FACE_CW,
    LAYOUT_DEPTH_ATTACHMENT,
    LAYOUT_SHADER_READ_ONLY,
    PIPELINE_COMPUTE,
    PIPELINE_DRAW_INDIRECT,
    PIPELINE_EARLY_FRAGMENT_TESTS,
    PIPELINE_LATE_FRAGMENT_TESTS,
    PIPELINE_VERTEX_SHADER,
    VulkanGpu,
    VulkanImage,
    get32,
    put32,
)
from flint.render.spirv import (
    AaaSpirvModule,
    MODEL_GL_COMPUTE,
    MODEL_VERTEX,
    OP_STORE,
    STORAGE_OUTPUT,
    emit,
)
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_PUSH_WORDS,
    BIND_AAA_GLOBALS,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_MESHLETS,
    BIND_AAA_VERTICES,
    BIND_AAA_VSM_TABLE,
    emit_instance_point,
    emit_mat4_mul_point,
)
from flint.render.aaa_vsm import (
    SOTA_SHADOW_GLOBALS_FLAGS_WORD,
    SOTA_SHADOW_GLOBALS_TOROIDAL_OFFSETS_WORD,
    SOTA_SHADOW_GLOBALS_TOROIDAL_WORD,
    VSM_GLOBALS_CASCADE_1_WORD,
    VSM_GLOBALS_CASCADE_2_WORD,
    VSM_GLOBALS_CASCADE_3_WORD,
    VSM_PHYS_TILE_COUNT,
    VSM_TABLE_WORDS,
    VSM_VIRTUAL_PAGE_COUNT,
    vsm_alphamask_fragment_shader,
)

comptime PIPELINE_ALL_COMMANDS: Int = 0x10000
comptime ACCESS_MEMORY_READ: Int = 0x8000
comptime ACCESS_MEMORY_WRITE: Int = 0x10000

# ---- buffer layout (words) -----------------------------------------------------------------
comptime SOTA_SHADOW_EXT: Int = VSM_TABLE_WORDS  # 32768
comptime SOTA_SHADOW_P2V: Int = SOTA_SHADOW_EXT
comptime SOTA_SHADOW_LAST: Int = SOTA_SHADOW_EXT + 1024
comptime SOTA_SHADOW_INVAL: Int = SOTA_SHADOW_EXT + 2048
comptime SOTA_SHADOW_INVAL_WORDS: Int = 512
comptime SOTA_SHADOW_MAX_SLOTS: Int = 4
comptime SOTA_SHADOW_HDR: Int = SOTA_SHADOW_EXT + 4096
comptime SOTA_SHADOW_H_OPAQUE: Int = SOTA_SHADOW_HDR + 0
comptime SOTA_SHADOW_H_ALPHA: Int = SOTA_SHADOW_HDR + 1
comptime SOTA_SHADOW_H_WANTED: Int = SOTA_SHADOW_HDR + 2  # draws the cull wanted (incl. those that did not fit)
comptime SOTA_SHADOW_H_OVERFLOW: Int = SOTA_SHADOW_HDR + 3  # page x draw overflows
comptime SOTA_SHADOW_H_CLEAR: Int = SOTA_SHADOW_HDR + 4  # {6, dirty pages, 0, 0}
comptime SOTA_SHADOW_H_DIRTY: Int = SOTA_SHADOW_HDR + 5
comptime SOTA_SHADOW_H_NEED: Int = SOTA_SHADOW_HDR + 8  # 4 counters, one per cascade (finest allocates first)
comptime SOTA_SHADOW_H_FREE: Int = SOTA_SHADOW_HDR + 12
comptime SOTA_SHADOW_H_OLD: Int = SOTA_SHADOW_HDR + 13
comptime SOTA_SHADOW_H_RECENT: Int = SOTA_SHADOW_HDR + 14
comptime SOTA_SHADOW_H_FINE: Int = SOTA_SHADOW_HDR + 15  # r3: requested resident cascade-0 tiles (pool pressure only)
comptime SOTA_SHADOW_H_CDIRTY: Int = SOTA_SHADOW_HDR + 16  # 4 cascades x [dirty pages, max(63-lx), max lx, max(63-ly), max ly]
comptime SOTA_SHADOW_HDR_WORDS: Int = 48
comptime SOTA_SHADOW_ST: Int = SOTA_SHADOW_HDR + SOTA_SHADOW_HDR_WORDS
comptime SOTA_SHADOW_ST_DIRTY: Int = SOTA_SHADOW_ST
comptime SOTA_SHADOW_ST_NEED: Int = SOTA_SHADOW_ST + 1024  # 4 lists of 4096 (one per cascade)
comptime SOTA_SHADOW_ST_FREE: Int = SOTA_SHADOW_ST_NEED + 16384
comptime SOTA_SHADOW_ST_OLD: Int = SOTA_SHADOW_ST_FREE + 1024
comptime SOTA_SHADOW_ST_RECENT: Int = SOTA_SHADOW_ST_OLD + 1024
comptime SOTA_SHADOW_ST_FINE: Int = SOTA_SHADOW_ST_RECENT + 1024  # r3
comptime SOTA_SHADOW_ITEMS: Int = SOTA_SHADOW_ST_FINE + 1024  # 58384
# r3: [inst | alpha << 31, first meshlet, count <= 64, cascade mask, instance world sphere cx, cy, cz, r]. The mask
# lets an instance cast with LOD 0 into the fine cascades and with a coarse LOD into the coarse ones.
comptime SOTA_SHADOW_ITEM_WORDS: Int = 8
comptime SOTA_SHADOW_ITEM_MESHLETS: Int = 64
comptime SOTA_SHADOW_H_WANTED_ALPHA: Int = SOTA_SHADOW_HDR + 36  # r3: alpha draws wanted (H_WANTED = opaque)
comptime SOTA_SHADOW_H_CULL_ARGS: Int = SOTA_SHADOW_HDR + 40  # r3: {items that survived the pre-cull, 1, 1}
comptime SOTA_SHADOW_H_OVERFLOW_RETRY: Int = SOTA_SHADOW_HDR + 37  # r3: draws that did not fit in the retry pass either
comptime SOTA_SHADOW_H_RETRY_PAGES: Int = SOTA_SHADOW_HDR + 45  # r3: pages each retry round re-culls (3 rounds max)
comptime SOTA_SHADOW_H_CULLED_A: Int = SOTA_SHADOW_HDR + 39  # r3: pass A's pre-cull survivors (CULL_ARGS is reused)
comptime SOTA_SHADOW_H_DRAWS_A: Int = SOTA_SHADOW_HDR + 43  # r3: pass A's draws (the list counters are reused)
comptime SOTA_SHADOW_H_RETRY_RAN: Int = SOTA_SHADOW_HDR + 44  # r3: 1 when the retry shader ran this frame

comptime SOTA_SHADOW_ENTRY_MAPPED: UInt32 = 0x10000
comptime SOTA_SHADOW_ENTRY_VALID: UInt32 = 0x20000
comptime SOTA_SHADOW_ENTRY_RENDER: UInt32 = 0x40000
comptime SOTA_SHADOW_ENTRY_RETRY: UInt32 = 0x80000  # r3: page a draw of which overflowed the round before
comptime SOTA_SHADOW_ITEM_RETRY_SHIFT: Int = 24  # r3: item word 3 bit 24 + pass = "a draw of this item overflowed"

# push words
comptime SOTA_SHADOW_PUSH_EPOCH: Int = 0
comptime SOTA_SHADOW_PUSH_SLOT: Int = 1
comptime SOTA_SHADOW_PUSH_ITEMS: Int = 2  # item count
comptime SOTA_SHADOW_PUSH_CAP: Int = 3  # draws in the opaque list
comptime SOTA_SHADOW_PUSH_ITEM_BASE: Int = 4
comptime SOTA_SHADOW_PUSH_CMD_BASE: Int = 5
comptime SOTA_SHADOW_PUSH_CAP_ALPHA: Int = 6  # r3: draws in the alpha list (the lists are sized separately)
comptime SOTA_SHADOW_PUSH_CLIST: Int = 7  # r3: word offset of the pre-culled item list
comptime SOTA_SHADOW_PUSH_PASS: Int = 8  # r3: 0 = main cull, 1 = retry of the pages that overflowed the lists


def sota_shadow_clist_base(item_cap: Int) -> Int:
    """Pre-culled item indices (one list, rebuilt every frame on the GPU)."""
    return SOTA_SHADOW_ITEMS + SOTA_SHADOW_MAX_SLOTS * item_cap * SOTA_SHADOW_ITEM_WORDS


def sota_shadow_cmd_base(item_cap: Int) -> Int:
    return sota_shadow_clist_base(item_cap) + item_cap


def sota_shadow_buffer_words(item_cap: Int, cap_opaque: Int, cap_alpha: Int) -> Int:
    return sota_shadow_cmd_base(item_cap) + (cap_opaque + cap_alpha) * 4


def sota_shadow_draw_estimate(meshlets: Int, factor: Float32) -> Int:
    """Pre-size of a draw list: `factor` page draws per caster meshlet (a teleport / sun change renders every
    requested page; each meshlet lands in ~1-4 pages per cascade it overlaps). Forest 1080p wants 2.0-2.3 per meshlet."""
    return Int(Float32(meshlets) * factor)


def sota_shadow_list_target(need: Int, have: Int) -> Int:
    """Capacity for `need` draws: 25 % headroom, 16k granules, never shrinks."""
    if need <= have:
        return have
    var t = need + need // 4
    return max(have, ((t + 16383) // 16384) * 16384)


def sota_shadow_append_items(
    mut items: List[UInt32], inst_word: UInt32, sl: MeshLodSlice, mesh_first_ml: Int, lod0_count: Int, cmask: Int,
    wc: Vec3, wr: Float32,
) -> Int:
    """Appends the caster items (64 meshlets each) of one instance LOD that casts into the cascades in `cmask`.
    Returns the meshlet count."""
    var first_ml = sl.first_meshlet if sl.meshlet_count > 0 else mesh_first_ml
    var ml_cnt = sl.meshlet_count if sl.meshlet_count > 0 else lod0_count
    var m0 = 0
    while m0 < ml_cnt:
        items.append(inst_word)
        items.append(UInt32(first_ml + m0))
        items.append(UInt32(min(SOTA_SHADOW_ITEM_MESHLETS, ml_cnt - m0)))
        items.append(UInt32(cmask))
        items.append(bitcast[DType.uint32](wc.x))
        items.append(bitcast[DType.uint32](wc.y))
        items.append(bitcast[DType.uint32](wc.z))
        items.append(bitcast[DType.uint32](wr))
        m0 += SOTA_SHADOW_ITEM_MESHLETS
    return ml_cnt


def _u(mut s: AaaSpirvModule, v: Int) -> UInt32:
    return s.uint(UInt32(v))


def _cascade_mat_word(mut s: AaaSpirvModule, c: UInt32) -> UInt32:
    return s.select_u(
        s.ieq(c, s.uint(0)),
        s.uint(56),
        s.select_u(
            s.ieq(c, s.uint(1)),
            _u(s, VSM_GLOBALS_CASCADE_1_WORD),
            s.select_u(s.ieq(c, s.uint(2)), _u(s, VSM_GLOBALS_CASCADE_2_WORD), _u(s, VSM_GLOBALS_CASCADE_3_WORD)),
        ),
    )


def _emit_slot_offsets(mut s: AaaSpirvModule, b_globals: UInt32, c: UInt32) -> Tuple[UInt32, UInt32]:
    """Toroidal slot offset (ox, oy) of cascade `c` (0 when toroidal addressing is off)."""
    var on = s.ine(s.load_word(b_globals, _u(s, SOTA_SHADOW_GLOBALS_TOROIDAL_WORD)), s.uint(0))
    var w = s.load_word(b_globals, s.iadd(_u(s, SOTA_SHADOW_GLOBALS_TOROIDAL_OFFSETS_WORD), c))
    var ox = s.select_u(on, s.bit_and(w, s.uint(63)), s.uint(0))
    var oy = s.select_u(on, s.bit_and(s.shr(w, s.uint(8)), s.uint(63)), s.uint(0))
    return (ox, oy)


def _emit_append(mut s: AaaSpirvModule, b: UInt32, count_word: Int, list_base: Int, cap: Int, value: UInt32):
    var idx = s.atomic_add_word(b, _u(s, count_word), s.uint(1))
    s.begin_if(s.ult(idx, _u(s, cap)))
    s.store_word(b, s.iadd(_u(s, list_base), idx), value)
    s.end_if()


def _emit_mark_dirty(mut s: AaaSpirvModule, b: UInt32, b_globals: UInt32, v: UInt32):
    """Appends virtual page `v` to the dirty list and grows its cascade's dirty box (window-local pages)."""
    _emit_append(s, b, SOTA_SHADOW_H_DIRTY, SOTA_SHADOW_ST_DIRTY, VSM_PHYS_TILE_COUNT, v)
    var c = s.iadd(s.bit_and(s.shr(v, s.uint(6)), s.uint(1)), s.imul(s.shr(v, s.uint(13)), s.uint(2)))
    var offs = _emit_slot_offsets(s, b_globals, c)
    var lx = s.bit_and(s.isub(s.iadd(s.bit_and(v, s.uint(63)), s.uint(64)), offs[0]), s.uint(63))
    var ly = s.bit_and(s.isub(s.iadd(s.bit_and(s.shr(v, s.uint(7)), s.uint(63)), s.uint(64)), offs[1]), s.uint(63))
    var hb = s.iadd(_u(s, SOTA_SHADOW_H_CDIRTY), s.imul(c, s.uint(5)))
    _ = s.atomic_add_word(b, hb, s.uint(1))
    _ = s.atomic_umax_word(b, s.iadd(hb, s.uint(1)), s.isub(s.uint(63), lx))
    _ = s.atomic_umax_word(b, s.iadd(hb, s.uint(2)), lx)
    _ = s.atomic_umax_word(b, s.iadd(hb, s.uint(3)), s.isub(s.uint(63), ly))
    _ = s.atomic_umax_word(b, s.iadd(hb, s.uint(4)), ly)


def _entry_phys(mut s: AaaSpirvModule, e: UInt32) -> UInt32:
    return s.iadd(s.imul(s.bit_and(s.shr(e, s.uint(8)), s.uint(0xFF)), s.uint(32)), s.bit_and(e, s.uint(0xFF)))


# ---------------------------------------------------------------------------------------------
# Allocation passes
# ---------------------------------------------------------------------------------------------
def sota_shadow_alloc0_shader() -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var t = s.invocation_axis(0)
    s.begin_if(s.ult(t, _u(s, SOTA_SHADOW_HDR_WORDS)))
    # clear draw header {6, n, 0, 0} and the cull dispatch {n, 1, 1}
    var one = s.logical_or(
        s.ieq(t, _u(s, SOTA_SHADOW_H_CULL_ARGS + 1 - SOTA_SHADOW_HDR)), s.ieq(t, _u(s, SOTA_SHADOW_H_CULL_ARGS + 2 - SOTA_SHADOW_HDR))
    )
    s.store_word(b, s.iadd(_u(s, SOTA_SHADOW_HDR), t), s.select_u(s.ieq(t, s.uint(4)), s.uint(6), s.select_u(one, s.uint(1), s.uint(0))))
    s.end_if()
    return s.finish()


def sota_shadow_alloc1_shader() -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var v = s.invocation_axis(0)
    s.return_if(s.uge(v, _u(s, VSM_VIRTUAL_PAGE_COUNT)))
    var epoch = s.push_word(SOTA_SHADOW_PUSH_EPOCH)
    var slot = s.push_word(SOTA_SHADOW_PUSH_SLOT)
    var entry = s.uvar(s.bit_and(s.load_word(b, v), s.uint(~(SOTA_SHADOW_ENTRY_RENDER | SOTA_SHADOW_ENTRY_RETRY))))
    var inval_w = s.load_word(
        b, s.iadd(s.iadd(_u(s, SOTA_SHADOW_INVAL), s.imul(slot, _u(s, SOTA_SHADOW_INVAL_WORDS))), s.shr(v, s.uint(5)))
    )
    s.begin_if(s.ine(s.bit_and(s.shr(inval_w, s.bit_and(v, s.uint(31))), s.uint(1)), s.uint(0)))
    s.store(entry, s.bit_and(s.load_u(entry), s.uint(~SOTA_SHADOW_ENTRY_VALID)))
    s.end_if()
    var req_i = s.iadd(_u(s, VSM_VIRTUAL_PAGE_COUNT), v)
    s.begin_if(s.ine(s.load_word(b, req_i), s.uint(0)))
    s.store_word(b, req_i, s.uint(0))
    var e = s.load_u(entry)
    s.begin_if(s.ine(s.bit_and(e, s.uint(SOTA_SHADOW_ENTRY_MAPPED)), s.uint(0)))
    s.store_word(b, s.iadd(_u(s, SOTA_SHADOW_LAST), _entry_phys(s, e)), epoch)
    s.begin_if(s.ieq(s.bit_and(e, s.uint(SOTA_SHADOW_ENTRY_VALID)), s.uint(0)))
    s.store(entry, s.bit_or(e, s.uint(SOTA_SHADOW_ENTRY_VALID | SOTA_SHADOW_ENTRY_RENDER)))
    _emit_mark_dirty(s, b, b_globals, v)
    s.begin_else()
    # r3: under pool pressure the cached cascade-0 tiles are the coarser cascades' last resort (alloc3)
    var pressure = s.ine(s.bit_and(s.load_word(b_globals, _u(s, SOTA_SHADOW_GLOBALS_FLAGS_WORD)), s.uint(8)), s.uint(0))
    s.begin_if(s.logical_and(pressure, s.ieq(s.bit_and(v, s.uint(64 | 8192)), s.uint(0))))
    _emit_append(s, b, SOTA_SHADOW_H_FINE, SOTA_SHADOW_ST_FINE, VSM_PHYS_TILE_COUNT, _entry_phys(s, e))
    s.end_if()
    s.end_if()
    s.begin_else()
    # one need list per cascade: when the pool is oversubscribed the finest cascades get their pages first
    var c = s.iadd(s.bit_and(s.shr(v, s.uint(6)), s.uint(1)), s.imul(s.shr(v, s.uint(13)), s.uint(2)))
    var ni = s.atomic_add_word(b, s.iadd(_u(s, SOTA_SHADOW_H_NEED), c), s.uint(1))
    s.store_word(b, s.iadd(s.iadd(_u(s, SOTA_SHADOW_ST_NEED), s.imul(c, s.uint(4096))), ni), v)
    s.end_if()
    s.end_if()
    s.store_word(b, v, s.load_u(entry))
    return s.finish()


def sota_shadow_alloc2_shader() -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var p = s.invocation_axis(0)
    s.return_if(s.uge(p, _u(s, VSM_PHYS_TILE_COUNT)))
    var epoch = s.push_word(SOTA_SHADOW_PUSH_EPOCH)
    var owner = s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_P2V), p))
    s.begin_if(s.ieq(owner, s.uint(0xFFFFFFFF)))
    _emit_append(s, b, SOTA_SHADOW_H_FREE, SOTA_SHADOW_ST_FREE, VSM_PHYS_TILE_COUNT, p)
    s.begin_else()
    var age = s.isub(epoch, s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_LAST), p)))
    s.begin_if(s.uge(age, s.uint(2)))
    _emit_append(s, b, SOTA_SHADOW_H_OLD, SOTA_SHADOW_ST_OLD, VSM_PHYS_TILE_COUNT, p)
    s.begin_else()
    s.begin_if(s.ieq(age, s.uint(1)))
    _emit_append(s, b, SOTA_SHADOW_H_RECENT, SOTA_SHADOW_ST_RECENT, VSM_PHYS_TILE_COUNT, p)
    s.end_if()
    s.end_if()
    s.end_if()
    return s.finish()


def sota_shadow_alloc3_shader() -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var i = s.invocation_axis(0)
    # r3: while the pool is oversubscribed (globals 234 bit 3: parent pages are requested too) the coarsest cascade
    # allocates first, so the pool loses fine pages and their pixels fall back to the resident parent; else finest first.
    var coarse_first = s.ine(s.bit_and(s.load_word(b_globals, _u(s, SOTA_SHADOW_GLOBALS_FLAGS_WORD)), s.uint(8)), s.uint(0))
    var k0 = s.select_u(coarse_first, s.uint(3), s.uint(0))
    var k1 = s.select_u(coarse_first, s.uint(2), s.uint(1))
    var k2 = s.select_u(coarse_first, s.uint(1), s.uint(2))
    var k3 = s.select_u(coarse_first, s.uint(0), s.uint(3))
    var n0 = s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_H_NEED), k0))
    var n1 = s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_H_NEED), k1))
    var n2 = s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_H_NEED), k2))
    var n3 = s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_H_NEED), k3))
    var e1 = s.iadd(n0, n1)
    var e2 = s.iadd(e1, n2)
    s.return_if(s.uge(i, s.iadd(e2, n3)))
    # i -> (cascade list, index) in priority order k0, k1, k2, k3
    var c = s.select_u(s.ult(i, n0), k0, s.select_u(s.ult(i, e1), k1, s.select_u(s.ult(i, e2), k2, k3)))
    var c_start = s.select_u(s.ult(i, n0), s.uint(0), s.select_u(s.ult(i, e1), n0, s.select_u(s.ult(i, e2), e1, e2)))
    var epoch = s.push_word(SOTA_SHADOW_PUSH_EPOCH)
    var n_free = s.umin(s.load_word(b, _u(s, SOTA_SHADOW_H_FREE)), _u(s, VSM_PHYS_TILE_COUNT))
    var n_old = s.umin(s.load_word(b, _u(s, SOTA_SHADOW_H_OLD)), _u(s, VSM_PHYS_TILE_COUNT))
    var n_rec = s.umin(s.load_word(b, _u(s, SOTA_SHADOW_H_RECENT)), _u(s, VSM_PHYS_TILE_COUNT))
    var n_fo = s.iadd(n_free, n_old)
    var n_for = s.iadd(n_fo, n_rec)
    # r3: a coarser need may take a cached cascade-0 tile (listed by alloc1 under pressure only); fine needs never do,
    # so the split is stable across frames. Pool exhausted: unmapped this frame (samplers fall back coarser).
    var n_fine = s.select_u(s.ieq(c, s.uint(0)), s.uint(0), s.umin(s.load_word(b, _u(s, SOTA_SHADOW_H_FINE)), _u(s, VSM_PHYS_TILE_COUNT)))
    s.return_if(s.uge(i, s.iadd(n_for, n_fine)))
    var v = s.load_word(b, s.iadd(s.iadd(_u(s, SOTA_SHADOW_ST_NEED), s.imul(c, s.uint(4096))), s.isub(i, c_start)))
    var is_free = s.ult(i, n_free)
    var p = s.select_u(
        is_free,
        s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_ST_FREE), i)),
        s.select_u(
            s.ult(i, n_fo),
            s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_ST_OLD), s.isub(i, n_free))),
            s.select_u(
                s.ult(i, n_for),
                s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_ST_RECENT), s.isub(i, n_fo))),
                s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_ST_FINE), s.isub(i, n_for))),
            ),
        ),
    )
    s.begin_if(s.logical_not(is_free))
    var old = s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_P2V), p))
    s.store_word(b, s.bit_and(old, _u(s, VSM_VIRTUAL_PAGE_COUNT - 1)), s.uint(0))
    s.end_if()
    s.store_word(b, s.iadd(_u(s, SOTA_SHADOW_P2V), p), v)
    s.store_word(b, s.iadd(_u(s, SOTA_SHADOW_LAST), p), epoch)
    var entry = s.bit_or(
        s.uint(SOTA_SHADOW_ENTRY_MAPPED | SOTA_SHADOW_ENTRY_VALID | SOTA_SHADOW_ENTRY_RENDER),
        s.bit_or(s.shl(s.shr(p, s.uint(5)), s.uint(8)), s.bit_and(p, s.uint(31))),
    )
    s.store_word(b, v, entry)
    _emit_mark_dirty(s, b, b_globals, v)
    return s.finish()


# ---------------------------------------------------------------------------------------------
# r3 instance-level pre-cull: one thread per caster item. Its instance's world sphere is tested against every
# cascade's dirty-page box; the survivors go to a compact list whose length is the cull pass's indirect group count.
# A static frame (no dirty page) or a small update (a few pages) therefore launches no / few cull workgroups instead
# of one per item.
# ---------------------------------------------------------------------------------------------
def _emit_sphere_page_window(
    mut s: AaaSpirvModule, b_globals: UInt32, c: Int, wx: UInt32, wy: UInt32, wz: UInt32, wr: UInt32
) -> Tuple[UInt32, UInt32, UInt32, UInt32, UInt32]:
    """World sphere -> cascade `c` window: (page x, page y, radius in pages incl. the PCF margin, z low, z high)."""
    var mo = _cascade_mat_word(s, _u(s, c))
    var lc = emit_mat4_mul_point(s, b_globals, mo, wx, wy, wz)
    var m0 = s.load_float(b_globals, mo)
    var m4 = s.load_float(b_globals, s.iadd(mo, s.uint(4)))
    var m8 = s.load_float(b_globals, s.iadd(mo, s.uint(8)))
    var m2 = s.load_float(b_globals, s.iadd(mo, s.uint(2)))
    var m6 = s.load_float(b_globals, s.iadd(mo, s.uint(6)))
    var m10 = s.load_float(b_globals, s.iadd(mo, s.uint(10)))
    var sx = s.fsqrt(s.fma(m0, m0, s.fma(m4, m4, s.fmul(m8, m8))))
    var sz = s.fsqrt(s.fma(m2, m2, s.fma(m6, m6, s.fmul(m10, m10))))
    var fr = s.fma(s.fmul(wr, sx), s.float(32.0), s.float(0.30))
    var fx = s.fmul(s.fma(lc[0], s.float(0.5), s.float(0.5)), s.float(64.0))
    var fy = s.fmul(s.fma(s.fneg(lc[1]), s.float(0.5), s.float(0.5)), s.float(64.0))
    var rz = s.fmul(wr, sz)
    return (fx, fy, fr, s.fsub(lc[2], rz), s.fadd(lc[2], rz))


def sota_shadow_precull_shader() -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var item = s.invocation_axis(0)
    s.return_if(s.uge(item, s.push_word(SOTA_SHADOW_PUSH_ITEMS)))
    # static frame (pass A) / nothing overflowed (pass B): nothing to draw
    var retry = s.ine(s.push_word(SOTA_SHADOW_PUSH_PASS), s.uint(0))
    var retry_word = s.iadd(_u(s, SOTA_SHADOW_H_RETRY_PAGES - 1), s.push_word(SOTA_SHADOW_PUSH_PASS))
    var n_pages = s.load_word(b, s.select_u(retry, retry_word, _u(s, SOTA_SHADOW_H_DIRTY)))
    s.return_if(s.ieq(n_pages, s.uint(0)))
    var ib = s.iadd(s.push_word(SOTA_SHADOW_PUSH_ITEM_BASE), s.imul(item, _u(s, SOTA_SHADOW_ITEM_WORDS)))
    var wx = s.load_float(b, s.iadd(ib, s.uint(4)))
    var wy = s.load_float(b, s.iadd(ib, s.uint(5)))
    var wz = s.load_float(b, s.iadd(ib, s.uint(6)))
    var wr = s.load_float(b, s.iadd(ib, s.uint(7)))
    var cmask = s.load_word(b, s.iadd(ib, s.uint(3)))
    # retry round r: only items a draw of which overflowed in the round before
    var prev_bit = s.shl(s.uint(1), s.iadd(_u(s, SOTA_SHADOW_ITEM_RETRY_SHIFT - 1), s.push_word(SOTA_SHADOW_PUSH_PASS)))
    s.return_if(s.logical_and(retry, s.ieq(s.bit_and(cmask, prev_bit), s.uint(0))))
    var hit = s.uvar(s.uint(0))
    for c in range(4):
        var win = _emit_sphere_page_window(s, b_globals, c, wx, wy, wz, wr)
        var hb = _u(s, SOTA_SHADOW_H_CDIRTY + c * 5)
        var has_dirty = s.ine(s.load_word(b, hb), s.uint(0))
        var bx0 = s.float_from_uint(s.isub(s.uint(63), s.load_word(b, s.iadd(hb, s.uint(1)))))
        var bx1 = s.float_from_uint(s.load_word(b, s.iadd(hb, s.uint(2))))
        var by0 = s.float_from_uint(s.isub(s.uint(63), s.load_word(b, s.iadd(hb, s.uint(3)))))
        var by1 = s.float_from_uint(s.load_word(b, s.iadd(hb, s.uint(4))))
        # page p covers [p, p + 1): the sphere's page span [f - r, f + r] meets the dirty box [b0, b1 + 1)
        var ox = s.logical_and(s.fge(s.fadd(win[0], win[2]), bx0), s.flt(s.fsub(win[0], win[2]), s.fadd(bx1, s.float(1.0))))
        var oy = s.logical_and(s.fge(s.fadd(win[1], win[2]), by0), s.flt(s.fsub(win[1], win[2]), s.fadd(by1, s.float(1.0))))
        var oz = s.logical_and(s.fle(win[3], s.float(1.5)), s.fge(win[4], s.float(-0.5)))
        var in_mask = s.ine(s.bit_and(cmask, s.uint(UInt32(1 << c))), s.uint(0))
        s.begin_if(s.logical_and(s.logical_and(in_mask, has_dirty), s.logical_and(oz, s.logical_and(ox, oy))))
        s.store(hit, s.uint(1))
        s.end_if()
    s.begin_if(s.ine(s.load_u(hit), s.uint(0)))
    var idx = s.atomic_add_word(b, _u(s, SOTA_SHADOW_H_CULL_ARGS), s.uint(1))
    s.store_word(b, s.iadd(s.push_word(SOTA_SHADOW_PUSH_CLIST), idx), item)
    s.end_if()
    return s.finish()


# ---------------------------------------------------------------------------------------------
# r3 retry: when a page x draw pair does not fit the lists, the cull clears the page's VALID bit and flags the item
# (item word 3, bit 24 + pass). After the raster a retry round re-validates those pages (flag RETRY), empties the lists
# and culls only the flagged items against only the RETRY pages, drawing on top without a clear (a caster drawn twice
# gives the same depth). An undersized list costs a retry round in the same frame instead of a frame of lit pages.
# ---------------------------------------------------------------------------------------------
def sota_shadow_retry_shader() -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var v = s.invocation_axis(0)
    s.return_if(s.uge(v, _u(s, VSM_VIRTUAL_PAGE_COUNT)))
    s.begin_if(s.ieq(v, s.uint(0)))
    # first round: keep pass A's stats (CULL_ARGS and the list counters are reused); later rounds add their draws
    var first = s.ieq(s.load_word(b, _u(s, SOTA_SHADOW_H_RETRY_RAN)), s.uint(0))
    s.begin_if(first)
    s.store_word(b, _u(s, SOTA_SHADOW_H_CULLED_A), s.load_word(b, _u(s, SOTA_SHADOW_H_CULL_ARGS)))
    s.end_if()
    s.store_word(b, _u(s, SOTA_SHADOW_H_RETRY_RAN), s.uint(1))
    s.store_word(
        b, _u(s, SOTA_SHADOW_H_DRAWS_A),
        s.iadd(
            s.load_word(b, _u(s, SOTA_SHADOW_H_DRAWS_A)),
            s.iadd(s.load_word(b, _u(s, SOTA_SHADOW_H_OPAQUE)), s.load_word(b, _u(s, SOTA_SHADOW_H_ALPHA))),
        ),
    )
    s.store_word(b, _u(s, SOTA_SHADOW_H_OVERFLOW_RETRY), s.uint(0))  # only the last round's overflow stays incomplete
    s.store_word(b, _u(s, SOTA_SHADOW_H_OPAQUE), s.uint(0))
    s.store_word(b, _u(s, SOTA_SHADOW_H_ALPHA), s.uint(0))
    s.store_word(b, _u(s, SOTA_SHADOW_H_CULL_ARGS), s.uint(0))
    s.end_if()
    var e = s.load_word(b, v)
    var redo = s.logical_and(
        s.ine(s.bit_and(e, s.uint(SOTA_SHADOW_ENTRY_RENDER)), s.uint(0)),
        s.ieq(s.bit_and(e, s.uint(SOTA_SHADOW_ENTRY_VALID)), s.uint(0)),
    )
    s.begin_if(redo)
    s.store_word(b, v, s.bit_or(e, s.uint(SOTA_SHADOW_ENTRY_VALID | SOTA_SHADOW_ENTRY_RETRY)))
    _ = s.atomic_add_word(b, s.iadd(_u(s, SOTA_SHADOW_H_RETRY_PAGES - 1), s.push_word(SOTA_SHADOW_PUSH_PASS)), s.uint(1))
    s.begin_else()
    s.begin_if(s.ine(s.bit_and(e, s.uint(SOTA_SHADOW_ENTRY_RETRY)), s.uint(0)))
    s.store_word(b, v, s.bit_and(e, s.uint(~SOTA_SHADOW_ENTRY_RETRY)))  # finished in an earlier round
    s.end_if()
    s.end_if()
    return s.finish()


# ---------------------------------------------------------------------------------------------
# Caster cull: one workgroup per pre-culled caster item (indirect), one thread per meshlet.
# ---------------------------------------------------------------------------------------------
def sota_shadow_cull_shader() -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var b_ml = s.buffer(BIND_AAA_MESHLETS)
    var b_inst = s.buffer(BIND_AAA_INSTANCES)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var wg = s.workgroup_axis(0)
    var lt = s.local_axis(0)
    s.return_if(s.uge(wg, s.load_word(b, _u(s, SOTA_SHADOW_H_CULL_ARGS))))
    var item = s.load_word(b, s.iadd(s.push_word(SOTA_SHADOW_PUSH_CLIST), wg))
    var ib = s.iadd(s.push_word(SOTA_SHADOW_PUSH_ITEM_BASE), s.imul(item, _u(s, SOTA_SHADOW_ITEM_WORDS)))
    var w0 = s.load_word(b, ib)
    var cmask = s.load_word(b, s.iadd(ib, s.uint(3)))  # r3: cascades this item casts into (per-cascade LOD)
    var first_ml = s.load_word(b, s.iadd(ib, s.uint(1)))
    var cnt = s.load_word(b, s.iadd(ib, s.uint(2)))
    s.return_if(s.uge(lt, cnt))
    var inst = s.bit_and(w0, s.uint(0x7FFFFFFF))
    var is_alpha = s.ine(s.shr(w0, s.uint(31)), s.uint(0))
    var ml = s.iadd(first_ml, lt)
    var mb = s.imul(ml, s.uint(12))
    var tri_cnt = s.load_word(b_ml, s.iadd(mb, s.uint(9)))
    s.return_if(s.ieq(tri_cnt, s.uint(0)))
    var ibase = s.imul(inst, _u(s, AAA_INSTANCE_WORDS))
    var wc = emit_mat4_mul_point(
        s, b_inst, ibase,
        s.load_float(b_ml, mb), s.load_float(b_ml, s.iadd(mb, s.uint(1))), s.load_float(b_ml, s.iadd(mb, s.uint(2))),
    )
    var sc2 = List[UInt32]()
    for col in range(3):
        var x = s.load_float(b_inst, s.iadd(ibase, _u(s, col * 4)))
        var y = s.load_float(b_inst, s.iadd(ibase, _u(s, col * 4 + 1)))
        var z = s.load_float(b_inst, s.iadd(ibase, _u(s, col * 4 + 2)))
        sc2.append(s.fma(x, x, s.fma(y, y, s.fmul(z, z))))
    var max_sc = s.fsqrt(s.nmax(s.float(1e-6), s.nmax(sc2[0], s.nmax(sc2[1], sc2[2]))))
    var wr = s.fmul(s.load_float(b_ml, s.iadd(mb, s.uint(3))), max_sc)
    var cmd0 = s.imul(tri_cnt, s.uint(3))
    var inst_word = s.bit_or(s.shl(inst, s.uint(17)), s.bit_and(ml, s.uint(0x1FFFF)))
    var ml_hi = s.shl(s.shr(ml, s.uint(17)), s.uint(24))
    var count_word = s.select_u(is_alpha, _u(s, SOTA_SHADOW_H_ALPHA), _u(s, SOTA_SHADOW_H_OPAQUE))
    var wanted_word = s.select_u(is_alpha, _u(s, SOTA_SHADOW_H_WANTED_ALPHA), _u(s, SOTA_SHADOW_H_WANTED))
    var retry = s.ine(s.push_word(SOTA_SHADOW_PUSH_PASS), s.uint(0))
    var page_bit = s.select_u(retry, s.uint(SOTA_SHADOW_ENTRY_RETRY), s.uint(SOTA_SHADOW_ENTRY_RENDER))
    var my_bit = s.shl(s.uint(1), s.iadd(_u(s, SOTA_SHADOW_ITEM_RETRY_SHIFT), s.push_word(SOTA_SHADOW_PUSH_PASS)))
    var wanted_inc = s.select_u(retry, s.uint(0), s.uint(1))  # the need is what pass A wanted
    var overflow_word = s.select_u(retry, _u(s, SOTA_SHADOW_H_OVERFLOW_RETRY), _u(s, SOTA_SHADOW_H_OVERFLOW))
    var cap_op = s.push_word(SOTA_SHADOW_PUSH_CAP)
    var cap = s.select_u(is_alpha, s.push_word(SOTA_SHADOW_PUSH_CAP_ALPHA), cap_op)
    var list_base = s.iadd(s.push_word(SOTA_SHADOW_PUSH_CMD_BASE), s.select_u(is_alpha, s.imul(cap_op, s.uint(4)), s.uint(0)))

    for c in range(4):
        var cu = _u(s, c)
        var mo = _cascade_mat_word(s, cu)
        var lc = emit_mat4_mul_point(s, b_globals, mo, wc[0], wc[1], wc[2])
        var m0 = s.load_float(b_globals, mo)
        var m4 = s.load_float(b_globals, s.iadd(mo, s.uint(4)))
        var m8 = s.load_float(b_globals, s.iadd(mo, s.uint(8)))
        var sx = s.fsqrt(s.fma(m0, m0, s.fma(m4, m4, s.fmul(m8, m8))))
        var fr = s.fma(s.fmul(wr, sx), s.float(32.0), s.float(0.30))  # radius in pages (+ PCF margin, as v5)
        var fx = s.fmul(s.fma(lc[0], s.float(0.5), s.float(0.5)), s.float(64.0))
        var fy = s.fmul(s.fma(s.fneg(lc[1]), s.float(0.5), s.float(0.5)), s.float(64.0))
        var z_ok = s.logical_and(s.fge(lc[2], s.float(-0.5)), s.fle(lc[2], s.float(1.5)))
        var in_x = s.logical_and(s.fge(s.fadd(fx, fr), s.float(0.0)), s.flt(s.fsub(fx, fr), s.float(64.0)))
        var in_y = s.logical_and(s.fge(s.fadd(fy, fr), s.float(0.0)), s.flt(s.fsub(fy, fr), s.float(64.0)))
        # only the box of this cascade's dirty pages (v5's packer does the same on the CPU)
        var hb = _u(s, SOTA_SHADOW_H_CDIRTY + c * 5)
        var has_dirty = s.ine(s.load_word(b, hb), s.uint(0))
        var bx0 = s.isub(s.uint(63), s.load_word(b, s.iadd(hb, s.uint(1))))
        var bx1 = s.load_word(b, s.iadd(hb, s.uint(2)))
        var by0 = s.isub(s.uint(63), s.load_word(b, s.iadd(hb, s.uint(3))))
        var by1 = s.load_word(b, s.iadd(hb, s.uint(4)))
        var in_mask = s.ine(s.bit_and(cmask, s.uint(UInt32(1 << c))), s.uint(0))
        s.begin_if(s.logical_and(s.logical_and(in_mask, has_dirty), s.logical_and(z_ok, s.logical_and(in_x, in_y))))
        var x0 = s.umax(bx0, s.umin(s.uint(63), s.saturate_uint(s.fsub(fx, fr))))
        var x1 = s.umin(bx1, s.umin(s.uint(63), s.saturate_uint(s.fadd(fx, fr))))
        var y0 = s.umax(by0, s.umin(s.uint(63), s.saturate_uint(s.fsub(fy, fr))))
        var y1 = s.umin(by1, s.umin(s.uint(63), s.saturate_uint(s.fadd(fy, fr))))
        var offs = _emit_slot_offsets(s, b_globals, cu)
        var qx = _u(s, (c & 1) * 64)
        var qy = _u(s, (c >> 1) * 64)
        var yv = s.uvar(y0)
        s.begin_loop()
        var yy = s.load_u(yv)
        s.break_if(s.logical_not(s.ult(yy, s.iadd(y1, s.uint(1)))))
        var row = s.imul(s.iadd(qy, s.bit_and(s.iadd(yy, offs[1]), s.uint(63))), s.uint(128))
        var xv = s.uvar(x0)
        s.begin_loop()
        var xx = s.load_u(xv)
        s.break_if(s.logical_not(s.ult(xx, s.iadd(x1, s.uint(1)))))
        var vpage = s.iadd(row, s.iadd(qx, s.bit_and(s.iadd(xx, offs[0]), s.uint(63))))
        var entry = s.load_word(b, vpage)
        s.begin_if(s.ine(s.bit_and(entry, page_bit), s.uint(0)))
        _ = s.atomic_add_word(b, wanted_word, wanted_inc)
        var idx = s.atomic_add_word(b, count_word, s.uint(1))
        s.begin_if(s.ult(idx, cap))
        var dst = s.iadd(list_base, s.imul(idx, s.uint(4)))
        s.store_word(b, dst, cmd0)
        s.store_word(b, s.iadd(dst, s.uint(1)), s.uint(1))
        s.store_word(b, s.iadd(dst, s.uint(2)), s.bit_or(ml_hi, s.shl(_entry_phys(s, entry), s.uint(10))))
        s.store_word(b, s.iadd(dst, s.uint(3)), inst_word)
        s.begin_else()
        # did not fit: the page must not be cached as complete, and the item is flagged for the next retry round
        # (after the last round the page re-renders next frame; the lists grow). Every writer here only clears VALID
        # (and keeps RENDER) / sets the same item bit, so the plain stores race benignly.
        s.store_word(b, vpage, s.bit_and(entry, s.uint(~SOTA_SHADOW_ENTRY_VALID)))
        s.store_word(b, s.iadd(ib, s.uint(3)), s.bit_or(cmask, my_bit))
        _ = s.atomic_add_word(b, overflow_word, s.uint(1))
        s.end_if()
        s.end_if()
        s.store(xv, s.iadd(xx, s.uint(1)))
        s.end_loop()
        s.store(yv, s.iadd(yy, s.uint(1)))
        s.end_loop()
        s.end_if()
    return s.finish()


# ---------------------------------------------------------------------------------------------
# Raster vertex shaders (whole-atlas viewport with flipped y, like the v5 per-page viewports)
# ---------------------------------------------------------------------------------------------
def _emit_page_transform(mut s: AaaSpirvModule, b: UInt32, b_globals: UInt32, page: UInt32, wx: UInt32, wy: UInt32, wz: UInt32):
    """World point -> atlas clip position inside physical page `page` + 4 clip distances confining it to the tile."""
    var v = s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_P2V), page))
    var vx = s.bit_and(v, s.uint(127))
    var vy = s.bit_and(s.shr(v, s.uint(7)), s.uint(127))
    var c = s.iadd(s.shr(vx, s.uint(6)), s.imul(s.shr(vy, s.uint(6)), s.uint(2)))
    var offs = _emit_slot_offsets(s, b_globals, c)
    var lx = s.bit_and(s.isub(s.iadd(s.bit_and(vx, s.uint(63)), s.uint(64)), offs[0]), s.uint(63))
    var ly = s.bit_and(s.isub(s.iadd(s.bit_and(vy, s.uint(63)), s.uint(64)), offs[1]), s.uint(63))
    var lc = emit_mat4_mul_point(s, b_globals, _cascade_mat_word(s, c), wx, wy, wz)
    var flx = s.float_from_uint(lx)
    var fly = s.float_from_uint(ly)
    # the v5 page zoom: x_p = clip.x * 64 + (63 - 2 lx), y_p = clip.y * 64 - (63 - 2 ly)
    var xp = s.fma(lc[0], s.float(64.0), s.fsub(s.float(63.0), s.fmul(flx, s.float(2.0))))
    var yp = s.fsub(s.fmul(lc[1], s.float(64.0)), s.fsub(s.float(63.0), s.fmul(fly, s.float(2.0))))
    # page clip -> atlas clip (32 x 32 tiles; the viewport flips y, so tile row py sits at clip y 1 - (2py+1)/32)
    var px = s.float_from_uint(s.bit_and(page, s.uint(31)))
    var py = s.float_from_uint(s.shr(page, s.uint(5)))
    var ax = s.fsub(s.fmul(s.fadd(s.fma(px, s.float(2.0), s.float(1.0)), xp), s.float(1.0 / 32.0)), s.float(1.0))
    var ay = s.fsub(s.float(1.0), s.fmul(s.fsub(s.fma(py, s.float(2.0), s.float(1.0)), yp), s.float(1.0 / 32.0)))
    var az = s.nclamp(lc[2], s.float(0.0), s.float(1.0))
    s.write_clip_position(ax, ay, az, s.float(1.0))
    var cd = List[UInt32]()
    cd.append(s.fsub(s.float(1.0), xp))
    cd.append(s.fadd(s.float(1.0), xp))
    cd.append(s.fsub(s.float(1.0), yp))
    cd.append(s.fadd(s.float(1.0), yp))
    s.write_clip_distances(cd)


def sota_shadow_raster_vertex_shader(alpha: Bool) -> List[UInt32]:
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_meshlets = s.buffer(BIND_AAA_MESHLETS)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var b_materials: UInt32 = 0
    var out_uv = (UInt32(0), UInt32(0))
    var out_talb = (UInt32(0), UInt32(0))
    if alpha:
        b_materials = s.buffer(BIND_AAA_MATERIALS)
        out_uv = s.add_stage_io(STORAGE_OUTPUT, s.type_vec2, location=0, flat=False)
        out_talb = s.add_stage_io(STORAGE_OUTPUT, s.type_uint, location=1, flat=True)
    var vi = s.vertex_index()
    var ii = s.instance_index()
    var local = s.bit_and(vi, s.uint(0x3FF))
    var page = s.bit_and(s.shr(vi, s.uint(10)), s.uint(0x3FF))
    var inst = s.shr(ii, s.uint(17))
    var ml = s.bit_or(s.shl(s.shr(vi, s.uint(24)), s.uint(17)), s.bit_and(ii, s.uint(0x1FFFF)))
    var local_prim = s.udiv(local, s.uint(3))
    var corner = s.umod(local, s.uint(3))
    var ml_base = s.imul(ml, s.uint(12))
    var first_tri = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(8)))
    var real_vtx = s.load_word(b_indices, s.iadd(s.imul(s.iadd(first_tri, local_prim), s.uint(3)), corner))
    var v_base = s.imul(real_vtx, s.uint(12))
    var lpx = s.load_float(b_verts, v_base)
    var lpy = s.load_float(b_verts, s.iadd(v_base, s.uint(1)))
    var lpz = s.load_float(b_verts, s.iadd(v_base, s.uint(2)))
    var inst_base = s.imul(inst, _u(s, AAA_INSTANCE_WORDS))
    var wp = emit_instance_point(s, b_instances, b_globals, b_verts, inst_base, v_base, lpx, lpy, lpz)
    _emit_page_transform(s, b, b_globals, page, wp[0], wp[1], wp[2])
    if alpha:
        var ml_mat = s.load_word(b_meshlets, s.iadd(ml_base, s.uint(11)))
        var mat_ov = s.bit_and(s.load_word(b_instances, s.iadd(inst_base, s.uint(34))), s.uint(0xFFFF))
        var final_mat = s.select_u(s.ieq(mat_ov, s.uint(0xFFFF)), ml_mat, mat_ov)
        var tex_pk = s.load_word(b_materials, s.iadd(s.imul(final_mat, s.uint(24)), s.uint(23)))
        var uv_u = s.load_float(b_verts, s.iadd(v_base, s.uint(9)))
        var uv_v = s.load_float(b_verts, s.iadd(v_base, s.uint(10)))
        emit(s.code, OP_STORE, out_uv[1], s.vec2(uv_u, uv_v))
        emit(s.code, OP_STORE, out_talb[1], s.bit_and(tex_pk, s.uint(0xFF)))
    return s.finish()


def sota_shadow_clear_vertex_shader() -> List[UInt32]:
    """Instanced tile clear: instance i = dirty list entry i, 6 vertices = its tile quad at depth 1."""
    var s = AaaSpirvModule(MODEL_VERTEX, push_words=AAA_PUSH_WORDS)
    var b = s.buffer(BIND_AAA_VSM_TABLE)
    var vi = s.vertex_index()
    var ii = s.instance_index()
    var v = s.load_word(b, s.iadd(_u(s, SOTA_SHADOW_ST_DIRTY), ii))
    var e = s.load_word(b, s.bit_and(v, _u(s, VSM_VIRTUAL_PAGE_COUNT - 1)))
    var px = s.float_from_uint(s.bit_and(e, s.uint(0xFF)))
    var py = s.float_from_uint(s.bit_and(s.shr(e, s.uint(8)), s.uint(0xFF)))
    # corners (0,0) (1,0) (0,1) | (1,0) (1,1) (0,1)
    var cx = s.select_u(s.logical_or(s.ieq(vi, s.uint(1)), s.logical_or(s.ieq(vi, s.uint(3)), s.ieq(vi, s.uint(4)))), s.uint(1), s.uint(0))
    var cy = s.select_u(s.logical_or(s.ieq(vi, s.uint(2)), s.logical_or(s.ieq(vi, s.uint(4)), s.ieq(vi, s.uint(5)))), s.uint(1), s.uint(0))
    var ax = s.fsub(s.fmul(s.fadd(px, s.float_from_uint(cx)), s.float(1.0 / 16.0)), s.float(1.0))
    var ay = s.fsub(s.float(1.0), s.fmul(s.fadd(py, s.float_from_uint(cy)), s.float(1.0 / 16.0)))
    s.write_clip_position(ax, ay, s.float(1.0), s.float(1.0))
    return s.finish()


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------
struct SotaShadowGpu(Movable):
    """Pipelines + the combined table/state/command buffer of the GPU page path."""

    var ready: Bool
    var pipe_alloc: List[Int]  # alloc0..3
    var pipe_precull: Int
    var pipe_cull: Int
    var pipe_retry: Int
    var pipe_clear: Int
    var pipe_opaque: Int
    var pipe_alpha: Int
    var buf: GpuBuffer
    var item_cap: Int
    var cap_opaque: Int  # draws the opaque list holds
    var cap_alpha: Int  # draws the alpha-masked list holds
    var presize_factor: Float32  # r3: page draws reserved per caster meshlet before any frame has run
    var epoch: Int
    var grows: Int
    # counters of the newest finished frame (read one frame late from the shared header)
    var dirty_last: Int
    var draws_last: Int
    var wanted_last: Int
    var wanted_max: Int
    var wanted_opaque_max: Int
    var wanted_alpha_max: Int
    var overflow_total: Int  # page x draw pairs that did not fit pass A (the retry pass redraws them)
    var overflow_retry_total: Int  # ... that did not fit the retry pass either (those pages stay incomplete a frame)
    var retry_pages_last: Int
    var retry_frames: Int  # frames the retry pass was recorded
    var retry_pass: Bool  # r3: record the same-frame retry pass (bug fix: on by default)
    var retry_rounds: Int  # retry passes per frame (each redraws what still did not fit)
    var unmapped_last: Int  # requested pages that found no physical page (pool oversubscribed)
    var requested_last: Int  # pages requested (mapped + needing allocation) in the newest finished frame
    var culled_items_last: Int  # items that survived the instance pre-cull
    var items_last: Int
    var meshlets_opaque_last: Int  # caster meshlets in this frame's items
    var meshlets_alpha_last: Int

    def __init__(out self):
        self.ready = False
        self.pipe_alloc = List[Int]()
        self.pipe_precull = 0
        self.pipe_cull = 0
        self.pipe_retry = 0
        self.pipe_clear = 0
        self.pipe_opaque = 0
        self.pipe_alpha = 0
        self.buf = GpuBuffer()
        self.item_cap = 0
        self.cap_opaque = 0
        self.cap_alpha = 0
        self.presize_factor = Float32(3.0)
        self.epoch = 1
        self.grows = 0
        self.dirty_last = 0
        self.draws_last = 0
        self.wanted_last = 0
        self.wanted_max = 0
        self.wanted_opaque_max = 0
        self.wanted_alpha_max = 0
        self.overflow_total = 0
        self.overflow_retry_total = 0
        self.retry_pages_last = 0
        self.retry_frames = 0
        self.retry_pass = True
        self.retry_rounds = 2
        self.unmapped_last = 0
        self.requested_last = 0
        self.culled_items_last = 0
        self.items_last = 0
        self.meshlets_opaque_last = 0
        self.meshlets_alpha_last = 0

    def init(mut self, mut vk: VulkanGpu, pipe_layout: Int) -> Bool:
        """Creates the pipelines (once). False when the device lacks clip distances / draw-indirect-count."""
        if self.ready:
            return True
        if not vk.has_clip_distance or not vk.has_draw_indirect_count:
            return False
        self.pipe_alloc = List[Int]()
        self.pipe_alloc.append(vk.create_aaa_compute_pipeline(pipe_layout, sota_shadow_alloc0_shader()))
        self.pipe_alloc.append(vk.create_aaa_compute_pipeline(pipe_layout, sota_shadow_alloc1_shader()))
        self.pipe_alloc.append(vk.create_aaa_compute_pipeline(pipe_layout, sota_shadow_alloc2_shader()))
        self.pipe_alloc.append(vk.create_aaa_compute_pipeline(pipe_layout, sota_shadow_alloc3_shader()))
        self.pipe_precull = vk.create_aaa_compute_pipeline(pipe_layout, sota_shadow_precull_shader())
        self.pipe_cull = vk.create_aaa_compute_pipeline(pipe_layout, sota_shadow_cull_shader())
        self.pipe_retry = vk.create_aaa_compute_pipeline(pipe_layout, sota_shadow_retry_shader())
        self.pipe_clear = vk.create_graphics_pipeline(
            pipe_layout, sota_shadow_clear_vertex_shader(), List[UInt32](), has_color=False, color_format=0,
            depth_format=FORMAT_D32_SFLOAT, depth_test=True, depth_write=True, depth_compare=COMPARE_ALWAYS,
            cull_mode=CULL_MODE_NONE, front_face=FRONT_FACE_CW,
        )
        self.pipe_opaque = vk.create_graphics_pipeline(
            pipe_layout, sota_shadow_raster_vertex_shader(False), List[UInt32](), has_color=False, color_format=0,
            depth_format=FORMAT_D32_SFLOAT, depth_test=True, depth_write=True, depth_compare=COMPARE_LESS_OR_EQUAL,
            cull_mode=CULL_MODE_BACK, front_face=FRONT_FACE_CW,
        )
        self.pipe_alpha = vk.create_graphics_pipeline(
            pipe_layout, sota_shadow_raster_vertex_shader(True), vsm_alphamask_fragment_shader(), has_color=False,
            color_format=0, depth_format=FORMAT_D32_SFLOAT, depth_test=True, depth_write=True,
            depth_compare=COMPARE_LESS_OR_EQUAL, cull_mode=CULL_MODE_NONE, front_face=FRONT_FACE_CW,
        )
        var ok = self.pipe_cull != 0 and self.pipe_clear != 0 and self.pipe_opaque != 0 and self.pipe_alpha != 0
        ok = ok and self.pipe_precull != 0 and self.pipe_retry != 0
        for p in self.pipe_alloc:
            ok = ok and p != 0
        if not ok:
            self.destroy(vk)
            return False
        self.ready = True
        return True

    def list_needs(self) -> Tuple[Int, Int]:
        """Draws each list should hold: what any frame so far wanted, and at least the pre-size estimate from this
        frame's caster meshlets (so the first teleport / sun change does not overflow)."""
        var op = max(self.wanted_opaque_max, sota_shadow_draw_estimate(self.meshlets_opaque_last, self.presize_factor))
        var al = max(self.wanted_alpha_max, sota_shadow_draw_estimate(self.meshlets_alpha_last, self.presize_factor))
        return (max(16384, op), max(16384, al))

    def ensure_buffer(mut self, mut vk: VulkanGpu, item_need: Int, op_need: Int, al_need: Int) -> Bool:
        """(Re)creates the buffer so it holds `item_need` items per slot and the draw lists, keeping the page table
        and allocator state. True when the buffer changed (the caller rebinds `BIND_AAA_VSM_TABLE`)."""
        if self.buf.size > 0 and item_need <= self.item_cap and op_need <= self.cap_opaque and al_need <= self.cap_alpha:
            return False
        var new_items = self.item_cap
        if item_need > self.item_cap:
            new_items = max(self.item_cap, ((item_need + item_need // 4 + 4095) // 4096) * 4096)
        var new_op = sota_shadow_list_target(op_need, self.cap_opaque)
        var new_al = sota_shadow_list_target(al_need, self.cap_alpha)
        var words = sota_shadow_buffer_words(new_items, new_op, new_al)
        var nb = vk.create_buffer(words * 4, local=True)
        if nb.size == 0:
            print("[sota_shadow] GPU page buffer of", words * 4, "bytes failed")
            return False
        if self.buf.size > 0:
            vk.wait_idle()
            _ = external_call["memcpy", Int](nb.cpu_address, self.buf.cpu_address, SOTA_SHADOW_ITEMS * 4)
            vk.destroy_buffer(self.buf)
            self.grows += 1
        else:
            sota_shadow_reset_tables(nb)
        self.buf = nb^
        self.item_cap = new_items
        self.cap_opaque = new_op
        self.cap_alpha = new_al
        return True

    def read_stats(mut self):
        """Counters of the newest frame that finished (the header is shared by the frames in flight)."""
        if self.buf.size == 0:
            return
        var a = self.buf.cpu_address
        self.dirty_last = Int(get32(a + SOTA_SHADOW_H_DIRTY * 4))
        self.draws_last = Int(get32(a + SOTA_SHADOW_H_OPAQUE * 4)) + Int(get32(a + SOTA_SHADOW_H_ALPHA * 4))
        self.retry_pages_last = Int(get32(a + SOTA_SHADOW_H_RETRY_PAGES * 4))
        var ran_retry = get32(a + SOTA_SHADOW_H_RETRY_RAN * 4) != 0
        var w_op = Int(get32(a + SOTA_SHADOW_H_WANTED * 4))
        var w_al = Int(get32(a + SOTA_SHADOW_H_WANTED_ALPHA * 4))
        self.wanted_last = w_op + w_al
        self.wanted_max = max(self.wanted_max, self.wanted_last)
        self.wanted_opaque_max = max(self.wanted_opaque_max, w_op)
        self.wanted_alpha_max = max(self.wanted_alpha_max, w_al)
        var ov = Int(get32(a + SOTA_SHADOW_H_OVERFLOW * 4))
        self.overflow_total += ov
        self.culled_items_last = Int(get32(a + SOTA_SHADOW_H_CULL_ARGS * 4))
        if ran_retry:
            self.draws_last += Int(get32(a + SOTA_SHADOW_H_DRAWS_A * 4))
            self.culled_items_last = Int(get32(a + SOTA_SHADOW_H_CULLED_A * 4))
            self.overflow_retry_total += Int(get32(a + SOTA_SHADOW_H_OVERFLOW_RETRY * 4))
        elif ov > 0:
            self.overflow_retry_total += ov  # no retry pass recorded: every overflowed page stayed incomplete
        var need = 0
        for c in range(4):
            need += Int(get32(a + (SOTA_SHADOW_H_NEED + c) * 4))
        var pool = Int(get32(a + SOTA_SHADOW_H_FREE * 4)) + Int(get32(a + SOTA_SHADOW_H_OLD * 4))
        pool += Int(get32(a + SOTA_SHADOW_H_RECENT * 4))
        self.unmapped_last = max(0, need - pool)

    def upload(mut self, slot: Int, items: List[UInt32], mut virtual_valid: List[Bool]):
        """Caster items and the invalidation bitmap of frame slot `slot` (the slot's last frame has finished)."""
        var a = self.buf.cpu_address
        var n = min(len(items), self.item_cap * SOTA_SHADOW_ITEM_WORDS)
        if n > 0:
            var dst = a + (SOTA_SHADOW_ITEMS + slot * self.item_cap * SOTA_SHADOW_ITEM_WORDS) * 4
            _ = external_call["memcpy", Int](dst, Int(items.unsafe_ptr()), n * 4)
        self.items_last = n // SOTA_SHADOW_ITEM_WORDS
        var inval = a + (SOTA_SHADOW_INVAL + slot * SOTA_SHADOW_INVAL_WORDS) * 4
        for w in range(SOTA_SHADOW_INVAL_WORDS):
            var bits = UInt32(0)
            for k in range(32):
                if not virtual_valid[w * 32 + k]:
                    bits |= UInt32(1) << UInt32(k)
                    virtual_valid[w * 32 + k] = True
            put32(inval + w * 4, bits)

    def record(
        mut self, mut vk: VulkanGpu, frame: Int, slot: Int, pipe_layout: Int, desc_set: Int, mut vsm_img: VulkanImage,
        atlas_texels: Int,
    ):
        """Allocation + pre-cull + cull + raster, recorded after `VsmMarkPages`."""
        self.epoch += 1
        var push = List[UInt32](length=AAA_PUSH_WORDS, fill=0)
        push[SOTA_SHADOW_PUSH_EPOCH] = UInt32(self.epoch)
        push[SOTA_SHADOW_PUSH_SLOT] = UInt32(slot)
        push[SOTA_SHADOW_PUSH_ITEMS] = UInt32(self.items_last)
        push[SOTA_SHADOW_PUSH_CAP] = UInt32(self.cap_opaque)
        push[SOTA_SHADOW_PUSH_ITEM_BASE] = UInt32(SOTA_SHADOW_ITEMS + slot * self.item_cap * SOTA_SHADOW_ITEM_WORDS)
        push[SOTA_SHADOW_PUSH_CMD_BASE] = UInt32(sota_shadow_cmd_base(self.item_cap))
        push[SOTA_SHADOW_PUSH_CAP_ALPHA] = UInt32(self.cap_alpha)
        push[SOTA_SHADOW_PUSH_CLIST] = UInt32(sota_shadow_clist_base(self.item_cap))
        var rw = Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
        # the previous frame's raster / shading and this frame's mark pass are done with the table
        vk.memory_barrier2(frame, PIPELINE_ALL_COMMANDS, ACCESS_MEMORY_READ | ACCESS_MEMORY_WRITE, PIPELINE_COMPUTE, rw)
        vk.dispatch_aaa(frame, self.pipe_alloc[0], pipe_layout, desc_set, push, 1, 1)
        vk.memory_barrier2(frame, PIPELINE_COMPUTE, rw, PIPELINE_COMPUTE, rw)
        vk.dispatch_aaa(frame, self.pipe_alloc[1], pipe_layout, desc_set, push, VSM_VIRTUAL_PAGE_COUNT // 64, 1)
        vk.memory_barrier2(frame, PIPELINE_COMPUTE, rw, PIPELINE_COMPUTE, rw)
        vk.dispatch_aaa(frame, self.pipe_alloc[2], pipe_layout, desc_set, push, VSM_PHYS_TILE_COUNT // 64, 1)
        vk.memory_barrier2(frame, PIPELINE_COMPUTE, rw, PIPELINE_COMPUTE, rw)
        vk.dispatch_aaa(frame, self.pipe_alloc[3], pipe_layout, desc_set, push, VSM_VIRTUAL_PAGE_COUNT // 64, 1)
        vk.memory_barrier2(frame, PIPELINE_COMPUTE, rw, PIPELINE_COMPUTE, rw)
        if self.items_last > 0:
            vk.dispatch_aaa(frame, self.pipe_precull, pipe_layout, desc_set, push, (self.items_last + 63) // 64, 1)
            vk.memory_barrier2(
                frame, PIPELINE_COMPUTE, rw, PIPELINE_COMPUTE | PIPELINE_DRAW_INDIRECT, rw | Int(ACCESS_INDIRECT_READ)
            )
            vk.dispatch_aaa_indirect(frame, self.pipe_cull, pipe_layout, desc_set, push, self.buf, SOTA_SHADOW_H_CULL_ARGS * 4)
        vk.memory_barrier2(
            frame, PIPELINE_COMPUTE, rw, PIPELINE_DRAW_INDIRECT | PIPELINE_VERTEX_SHADER | PIPELINE_COMPUTE,
            Int(ACCESS_INDIRECT_READ | ACCESS_SHADER_READ),
        )
        vk.transition_image(
            frame, vsm_img, LAYOUT_DEPTH_ATTACHMENT, PIPELINE_COMPUTE, ACCESS_SHADER_READ, PIPELINE_EARLY_FRAGMENT_TESTS,
            ACCESS_DEPTH_ATTACHMENT_WRITE,
        )
        vk.begin_rendering(frame, 0, 0, atlas_texels, atlas_texels, False, 0, False, 0, 0, True, vsm_img.view, False, 1.0)
        vk.set_viewport_and_scissor(frame, 0, 0, atlas_texels, atlas_texels, flip_y=True)
        var cmd_base = sota_shadow_cmd_base(self.item_cap)
        vk.bind_graphics_pipeline(frame, self.pipe_clear, pipe_layout, desc_set, push)
        vk.draw_indirect(frame, self.buf, SOTA_SHADOW_H_CLEAR * 4, 1, stride=16)
        vk.bind_graphics_pipeline(frame, self.pipe_opaque, pipe_layout, desc_set, push)
        vk.draw_indirect_count(frame, self.buf, cmd_base * 4, self.buf, SOTA_SHADOW_H_OPAQUE * 4, self.cap_opaque, stride=16)
        vk.bind_graphics_pipeline(frame, self.pipe_alpha, pipe_layout, desc_set, push)
        vk.draw_indirect_count(
            frame, self.buf, (cmd_base + self.cap_opaque * 4) * 4, self.buf, SOTA_SHADOW_H_ALPHA * 4, self.cap_alpha, stride=16
        )
        vk.end_rendering(frame)
        if self.retry_pass and self.items_last > 0:
            self.retry_frames += 1
            for r in range(min(self.retry_rounds, 3)):
                self._record_retry(vk, frame, pipe_layout, desc_set, push, vsm_img, atlas_texels, r + 1)
        vk.transition_image(
            frame, vsm_img, LAYOUT_SHADER_READ_ONLY, PIPELINE_LATE_FRAGMENT_TESTS, ACCESS_DEPTH_ATTACHMENT_WRITE,
            PIPELINE_COMPUTE, ACCESS_SHADER_READ,
        )

    def _record_retry(
        mut self, mut vk: VulkanGpu, frame: Int, pipe_layout: Int, desc_set: Int, push: List[UInt32],
        mut vsm_img: VulkanImage, atlas_texels: Int, round: Int,
    ):
        """r3 pass B: pages whose draws overflowed pass A's lists are re-culled into the emptied lists and drawn
        on top (no clear) in the same frame. With nothing overflowed the pre-cull returns at once and the cull
        dispatch is {0, 1, 1}, so pass B costs a few empty dispatches and an empty render pass."""
        var rw = Int(ACCESS_SHADER_READ | ACCESS_SHADER_WRITE)
        var pb = push.copy()
        pb[SOTA_SHADOW_PUSH_PASS] = UInt32(round)
        # pass A's indirect reads (list counters, cull args) and its cull writes are done before the reset
        vk.memory_barrier2(
            frame, PIPELINE_DRAW_INDIRECT | PIPELINE_VERTEX_SHADER | PIPELINE_COMPUTE,
            Int(ACCESS_INDIRECT_READ | ACCESS_SHADER_READ | ACCESS_SHADER_WRITE), PIPELINE_COMPUTE, rw,
        )
        vk.dispatch_aaa(frame, self.pipe_retry, pipe_layout, desc_set, pb, VSM_VIRTUAL_PAGE_COUNT // 64, 1)
        vk.memory_barrier2(frame, PIPELINE_COMPUTE, rw, PIPELINE_COMPUTE, rw)
        vk.dispatch_aaa(frame, self.pipe_precull, pipe_layout, desc_set, pb, (self.items_last + 63) // 64, 1)
        vk.memory_barrier2(
            frame, PIPELINE_COMPUTE, rw, PIPELINE_COMPUTE | PIPELINE_DRAW_INDIRECT, rw | Int(ACCESS_INDIRECT_READ)
        )
        vk.dispatch_aaa_indirect(frame, self.pipe_cull, pipe_layout, desc_set, pb, self.buf, SOTA_SHADOW_H_CULL_ARGS * 4)
        vk.memory_barrier2(
            frame, PIPELINE_COMPUTE, rw, PIPELINE_DRAW_INDIRECT | PIPELINE_VERTEX_SHADER | PIPELINE_COMPUTE,
            Int(ACCESS_INDIRECT_READ | ACCESS_SHADER_READ),
        )
        vk.memory_barrier2(
            frame, PIPELINE_LATE_FRAGMENT_TESTS, Int(ACCESS_DEPTH_ATTACHMENT_WRITE), PIPELINE_EARLY_FRAGMENT_TESTS,
            Int(ACCESS_DEPTH_ATTACHMENT_WRITE | ACCESS_DEPTH_ATTACHMENT_READ),
        )
        vk.begin_rendering(frame, 0, 0, atlas_texels, atlas_texels, False, 0, False, 0, 0, True, vsm_img.view, False, 1.0)
        vk.set_viewport_and_scissor(frame, 0, 0, atlas_texels, atlas_texels, flip_y=True)
        var cmd_base = sota_shadow_cmd_base(self.item_cap)
        vk.bind_graphics_pipeline(frame, self.pipe_opaque, pipe_layout, desc_set, pb)
        vk.draw_indirect_count(frame, self.buf, cmd_base * 4, self.buf, SOTA_SHADOW_H_OPAQUE * 4, self.cap_opaque, stride=16)
        vk.bind_graphics_pipeline(frame, self.pipe_alpha, pipe_layout, desc_set, pb)
        vk.draw_indirect_count(
            frame, self.buf, (cmd_base + self.cap_opaque * 4) * 4, self.buf, SOTA_SHADOW_H_ALPHA * 4, self.cap_alpha, stride=16
        )
        vk.end_rendering(frame)

    def destroy(mut self, mut vk: VulkanGpu):
        for p in self.pipe_alloc:
            if p != 0:
                vk.destroy_pipeline(p)
        self.pipe_alloc = List[Int]()
        if self.pipe_precull != 0:
            vk.destroy_pipeline(self.pipe_precull)
        if self.pipe_cull != 0:
            vk.destroy_pipeline(self.pipe_cull)
        if self.pipe_retry != 0:
            vk.destroy_pipeline(self.pipe_retry)
        self.pipe_retry = 0
        if self.pipe_clear != 0:
            vk.destroy_pipeline(self.pipe_clear)
        if self.pipe_opaque != 0:
            vk.destroy_pipeline(self.pipe_opaque)
        if self.pipe_alpha != 0:
            vk.destroy_pipeline(self.pipe_alpha)
        self.pipe_precull = 0
        self.pipe_cull = 0
        self.pipe_clear = 0
        self.pipe_opaque = 0
        self.pipe_alpha = 0
        if self.buf.size > 0:
            vk.destroy_buffer(self.buf)
            self.buf = GpuBuffer()
        self.item_cap = 0
        self.cap_opaque = 0
        self.cap_alpha = 0
        self.ready = False


def sota_shadow_reset_tables(buf: GpuBuffer):
    """Empty table, no requests, every physical page free."""
    _ = external_call["memset", Int](buf.cpu_address, Int32(0), SOTA_SHADOW_ITEMS * 4)
    _ = external_call["memset", Int](buf.cpu_address + SOTA_SHADOW_P2V * 4, Int32(-1), VSM_PHYS_TILE_COUNT * 4)

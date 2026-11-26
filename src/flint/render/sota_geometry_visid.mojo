"""Area A (`sota:geometry`): the wide (24-bit) global meshlet id shared by the draw commands and the VisBuffer.

v5 packed `inst << 17 | meshlet` into one 32-bit word everywhere (indirect `firstInstance`, occlusion
candidates, VisBuffer word 0), so a scene with more than 131070 global meshlets (Forest.glb without the
cluster DAG) aliased or was refused. The id is now 24 bits (16.7M meshlets); instances keep 15 bits
(`max_instances` is 32768) and every word stays where it was, so ids below 131071 encode bit-for-bit as before.

Draw commands (indexed `VkDrawIndexedIndirectCommand` and the VSM's non-indexed `VkDrawIndirectCommand`):
  firstInstance            = inst << 17 | (meshlet & 0x1FFFF)
  vertexOffset/firstVertex = (meshlet >> 17) << 24
  The vertex shader sees `VertexIndex = index + vertexOffset` (no vertex input bindings, programmable
  pulling), so `vertex = VertexIndex & 0xFFFFFF` and `meshlet hi = VertexIndex >> 24`. This needs no
  `shaderDrawParameters` (BaseVertex) and limits the geometry to 2^24 vertices (`SOTA_MAX_GLOBAL_VERTICES`).

VisBuffer (`R32G32_UINT`, 0 = sky):
  word0 = inst << 17 | L          L = meshlet % 131071 + 1, in 1..131071 (never 0, so word0 == 0 stays "empty")
  word1 = mat << 16 | H << 8 | prim   H = meshlet / 131071 (0..128), prim < 124 (`MESHLET_MAX_TRIS`)
  meshlet = H * 131071 + L - 1.

Occlusion candidates (phase-1 -> phase-2 queue): two words per entry, `[4 + 2 i] = inst`, `[5 + 2 i] = meshlet`.
"""
from flint.render.spirv import AaaSpirvModule

comptime SOTA_VIS_LO_MOD: Int = 131071                 # L = meshlet % 131071 + 1 (17 bits, never 0)
comptime SOTA_MAX_GLOBAL_MESHLETS_WIDE: Int = 16777215  # 24-bit meshlet id (draw path), H <= 128 (VisBuffer)
comptime SOTA_MAX_GLOBAL_VERTICES: Int = 16777216       # VertexIndex low 24 bits carry the vertex
comptime SOTA_CANDIDATE_WORDS: Int = 2                  # words per occlusion candidate entry


# ---------------------------------------------------------------- CPU reference / packers
def sota_draw_pack(inst: Int, meshlet: Int) -> Tuple[UInt32, UInt32]:
    """`(firstInstance, vertexOffset)` of a meshlet draw."""
    var fi = (UInt32(inst) << 17) | (UInt32(meshlet) & UInt32(0x1FFFF))
    var vo = (UInt32(meshlet) >> 17) << 24
    return (fi, vo)


def sota_draw_unpack(vertex_index: UInt32, instance_index: UInt32) -> Tuple[Int, Int, Int]:
    """`(vertex, inst, meshlet)` as the vertex shader decodes them."""
    var vtx = Int(vertex_index & UInt32(0xFFFFFF))
    var inst = Int(instance_index >> 17)
    var ml = Int(((vertex_index >> 24) << 17) | (instance_index & UInt32(0x1FFFF)))
    return (vtx, inst, ml)


def sota_vis_encode(inst: Int, meshlet: Int, mat: Int, prim: Int) -> Tuple[UInt32, UInt32]:
    var lo = UInt32(meshlet % SOTA_VIS_LO_MOD + 1)
    var hi = UInt32(meshlet // SOTA_VIS_LO_MOD)
    var w0 = (UInt32(inst) << 17) | lo
    var w1 = (UInt32(mat) << 16) | (hi << 8) | (UInt32(prim) & UInt32(0xFF))
    return (w0, w1)


def sota_vis_decode(w0: UInt32, w1: UInt32) -> Tuple[Int, Int, Int, Int]:
    """`(inst, meshlet, mat, prim)`; meshlet is -1 for an empty (sky) texel."""
    if w0 == 0:
        return (-1, -1, -1, -1)
    var lo = Int(w0 & UInt32(0x1FFFF))
    var hi = Int((w1 >> 8) & UInt32(0xFF))
    return (Int(w0 >> 17), hi * SOTA_VIS_LO_MOD + lo - 1, Int(w1 >> 16), Int(w1 & UInt32(0xFF)))


# ---------------------------------------------------------------- SPIR-V emitters
def emit_sota_draw_decode(mut s: AaaSpirvModule, vertex_index: UInt32, instance_index: UInt32) -> Tuple[UInt32, UInt32, UInt32]:
    """`(vertex, inst, meshlet)` from `VertexIndex` / `InstanceIndex` (see the module doc)."""
    var vtx = s.bit_and(vertex_index, s.uint(0xFFFFFF))
    var inst = s.shr(instance_index, s.uint(17))
    var ml = s.bit_or(s.shl(s.shr(vertex_index, s.uint(24)), s.uint(17)), s.bit_and(instance_index, s.uint(0x1FFFF)))
    return (vtx, inst, ml)


def emit_sota_draw_vertex_offset(mut s: AaaSpirvModule, meshlet: UInt32) -> UInt32:
    """`vertexOffset` / `firstVertex` word of a GPU-written meshlet draw."""
    return s.shl(s.shr(meshlet, s.uint(17)), s.uint(24))


def emit_sota_draw_first_instance(mut s: AaaSpirvModule, inst: UInt32, meshlet: UInt32) -> UInt32:
    return s.bit_or(s.shl(inst, s.uint(17)), s.bit_and(meshlet, s.uint(0x1FFFF)))


def emit_sota_vis_payload(mut s: AaaSpirvModule, inst: UInt32, meshlet: UInt32, mat: UInt32) -> Tuple[UInt32, UInt32]:
    """`(word0, word1_base)`; the fragment shader ORs the primitive id (< 256) into word1."""
    var m = s.uint(UInt32(SOTA_VIS_LO_MOD))
    var w0 = s.bit_or(s.shl(inst, s.uint(17)), s.iadd(s.umod(meshlet, m), s.uint(1)))
    var w1 = s.bit_or(s.shl(mat, s.uint(16)), s.shl(s.udiv(meshlet, m), s.uint(8)))
    return (w0, w1)


def emit_sota_vis_meshlet(mut s: AaaSpirvModule, vis0: UInt32, vis1: UInt32) -> UInt32:
    """Global meshlet id of a non-empty VisBuffer texel."""
    var lo = s.bit_and(vis0, s.uint(0x1FFFF))
    var hi = s.bit_and(s.shr(vis1, s.uint(8)), s.uint(0xFF))
    return s.isub(s.iadd(s.imul(hi, s.uint(UInt32(SOTA_VIS_LO_MOD))), lo), s.uint(1))


def emit_sota_vis_prim(mut s: AaaSpirvModule, vis1: UInt32) -> UInt32:
    return s.bit_and(vis1, s.uint(0xFF))

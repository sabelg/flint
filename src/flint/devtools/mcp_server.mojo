"""Game-agnostic Flint MCP & LiveLink introspection handler.

Allows AI agents and automated tools to query any running Flint game's build stamp,
tick counter, memory budgets, perceptual FLIP image diff, and screenshots with
zero game-specific coupling.
"""
from std.atomic import Atomic
from std.ffi import external_call
from std.math import abs, cos, max, min, sin
from std.memory import bitcast
from flint.assets.image import decode_png
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote
from flint.core.build_info import build_stamp
from flint.core.cvars import ConsoleVars, CVAR_COUNT, cvar_entry
from flint.core.math import Vec3
from flint.core.memory import W_CPU_LIVE, W_VRAM_LIVE, mem_total
from flint.core.serial import fnv64
from flint.core.transform import Quat, Mat4, trs
from flint.platform.os import read_bytes, close_fd, last_errno
from flint.render.aaa_clustered_lights import AAA_LIGHT_SPOT, AAA_MAX_CLUSTERED_LIGHTS, AaaClusterLight
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.camera_basis import view_right
from flint.render.flip import flip_error_map, flip_mean

comptime AF_UNIX: Int32 = 1
comptime SOCK_STREAM: Int32 = 1
comptime SOCK_NONBLOCK: Int32 = 2048
comptime SOCK_CLOEXEC: Int32 = 524288
comptime MSG_NOSIGNAL: Int32 = 0x4000
comptime EINTR: Int32 = 4
comptime EAGAIN: Int32 = 11
comptime MCP_MAX_CLIENTS: Int = 16
comptime MCP_MAX_REQ_BYTES: Int = 4096
comptime MCP_CLIENT_IDLE_POLLS: Int = 300

comptime MCP_CVAR_BASE: Int = 0x5D5E00000000
comptime MCP_CVAR_BYTES: Int = 4096
comptime MCP_CVAR_MAGIC: Int64 = 0x4D43504356415231  # "MCPCVAR1"

comptime MCP_SCENE_BASE: Int = 0x5D5E00010000
comptime MCP_SCENE_BYTES: Int = 16384
comptime MCP_SCENE_MAGIC: Int64 = 0x4D435053434E3031  # "MCPSCN01"
comptime MCP_SCENE_MAX_QUEUE: Int = 64
comptime MCP_SCENE_REC_BYTES: Int = 128

comptime SCENE_OP_SET_TRANSFORM: Int = 1
comptime SCENE_OP_SET_LIGHT: Int = 2
comptime SCENE_OP_SET_MATERIAL: Int = 3
comptime SCENE_OP_SET_CAMERA: Int = 4

comptime SMASK_V0: UInt32 = 1
comptime SMASK_V1: UInt32 = 2
comptime SMASK_QUAT: UInt32 = 4
comptime SMASK_F0: UInt32 = 8
comptime SMASK_F1: UInt32 = 16
comptime SMASK_F2: UInt32 = 32
comptime SMASK_F3: UInt32 = 64


def _mcp_name_hash(name: String) -> UInt64:
    var bs = List[UInt8]()
    for b in name.as_bytes():
        bs.append(b)
    return fnv64(bs)


def _mcp_cvar_ensure() -> Bool:
    if external_call["msync", Int32](MCP_CVAR_BASE, 4096, Int32(1)) == 0:
        var magic = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_CVAR_BASE)[unsafe_offset=0].load()
        if magic == MCP_CVAR_MAGIC:
            return True
    var got = external_call["mmap", Int](
        MCP_CVAR_BASE, MCP_CVAR_BYTES, Int32(3), Int32(0x22 | 0x100000), Int32(-1), Int(0)
    )
    if got != MCP_CVAR_BASE:
        return external_call["msync", Int32](MCP_CVAR_BASE, 4096, Int32(1)) == 0
    var p_u32 = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=MCP_CVAR_BASE + 64)
    for i in range(CVAR_COUNT):
        p_u32[unsafe_offset=i] = bitcast[DType.uint32](cvar_entry(i).default)
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_CVAR_BASE)[unsafe_offset=0].store(MCP_CVAR_MAGIC)
    return True


def _mcp_cvar_set(name: String, val: Float32) -> Float32:
    var cv = ConsoleVars()
    var idx = cv.find(name)
    var eff = (Float32(1.0) if val != Float32(0.0) else Float32(0.0)) if (idx >= 0 and cvar_entry(idx).flag) else val
    if not _mcp_cvar_ensure():
        return eff
    if idx >= 0 and idx < 256:
        Pointer[Atomic[Int32], MutAnyOrigin](unsafe_from_address=MCP_CVAR_BASE + 64 + idx * 4)[unsafe_offset=0].store(
            bitcast[DType.int32](eff)
        )
        return eff
    var h = _mcp_name_hash(name)
    var cnt_p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_CVAR_BASE + 8)
    var cnt = Int(cnt_p[unsafe_offset=0].load())
    for k in range(min(64, cnt)):
        var slot_addr = MCP_CVAR_BASE + 1024 + k * 16
        var sh = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=slot_addr)[unsafe_offset=0]
        if sh == h:
            Pointer[Atomic[Int32], MutAnyOrigin](unsafe_from_address=slot_addr + 8)[unsafe_offset=0].store(
                bitcast[DType.int32](eff)
            )
            return eff
    if cnt < 64:
        var slot_addr = MCP_CVAR_BASE + 1024 + cnt * 16
        Pointer[UInt64, MutAnyOrigin](unsafe_from_address=slot_addr)[unsafe_offset=0] = h
        Pointer[Atomic[Int32], MutAnyOrigin](unsafe_from_address=slot_addr + 8)[unsafe_offset=0].store(
            bitcast[DType.int32](eff)
        )
        cnt_p[unsafe_offset=0].store(Int64(cnt + 1))
    return eff


def _mcp_cvar_get(name: String) -> Float32:
    var cv = ConsoleVars()
    var idx = cv.find(name)
    if not _mcp_cvar_ensure():
        return cv.get(idx) if idx >= 0 else Float32(0.0)
    if idx >= 0 and idx < 256:
        var bits = Pointer[Atomic[Int32], MutAnyOrigin](unsafe_from_address=MCP_CVAR_BASE + 64 + idx * 4)[unsafe_offset=0].load()
        return bitcast[DType.float32](bits)
    var h = _mcp_name_hash(name)
    var cnt = Int(Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_CVAR_BASE + 8)[unsafe_offset=0].load())
    for k in range(min(64, cnt)):
        var slot_addr = MCP_CVAR_BASE + 1024 + k * 16
        var sh = Pointer[UInt64, MutAnyOrigin](unsafe_from_address=slot_addr)[unsafe_offset=0]
        if sh == h:
            var bits = Pointer[Atomic[Int32], MutAnyOrigin](unsafe_from_address=slot_addr + 8)[unsafe_offset=0].load()
            return bitcast[DType.float32](bits)
    return Float32(0.0)


def _json_node_to_string(doc: JsonDocument, node: Int) -> String:
    if node < 0 or node >= len(doc.nodes):
        return "null"
    var k = doc.nodes[node].kind
    if k == 1:
        return "true" if doc.nodes[node].boolean else "false"
    if k == 2:
        return doc.nodes[node].text
    if k == 3:
        return json_quote(doc.nodes[node].text)
    if k == 4:
        var s = String("[")
        for i in range(len(doc.nodes[node].children)):
            if i > 0:
                s += ","
            s += _json_node_to_string(doc, doc.nodes[node].children[i])
        return s + "]"
    if k == 5:
        var s = String("{")
        for i in range(len(doc.nodes[node].keys)):
            if i > 0:
                s += ","
            s += json_quote(doc.nodes[node].keys[i]) + ":" + _json_node_to_string(doc, doc.nodes[node].children[i])
        return s + "}"
    return "null"


struct FlintSceneMutation(Copyable, Movable):
    var op: Int
    var target_id: Int
    var mask: UInt32
    var seq: Int
    var v0: Vec3
    var v1: Vec3
    var quat: Quat
    var f0: Float32
    var f1: Float32
    var f2: Float32
    var f3: Float32
    var target_name: String

    def __init__(out self):
        self.op = 0
        self.target_id = -1
        self.mask = 0
        self.seq = 0
        self.v0 = Vec3()
        self.v1 = Vec3()
        self.quat = Quat()
        self.f0 = Float32(0.0)
        self.f1 = Float32(0.0)
        self.f2 = Float32(0.0)
        self.f3 = Float32(0.0)
        self.target_name = String()


def _mcp_scene_ensure() -> Bool:
    if external_call["msync", Int32](MCP_SCENE_BASE, 4096, Int32(1)) == 0:
        var magic = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE)[unsafe_offset=0].load()
        if magic == MCP_SCENE_MAGIC:
            return True
    var got = external_call["mmap", Int](
        MCP_SCENE_BASE, MCP_SCENE_BYTES, Int32(3), Int32(0x22 | 0x100000), Int32(-1), Int(0)
    )
    if got != MCP_SCENE_BASE:
        return external_call["msync", Int32](MCP_SCENE_BASE, 4096, Int32(1)) == 0
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 8)[unsafe_offset=0].store(Int64(0))
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 16)[unsafe_offset=0].store(Int64(0))
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 24)[unsafe_offset=0].store(Int64(0))
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE)[unsafe_offset=0].store(MCP_SCENE_MAGIC)
    return True


def _mcp_scene_lock():
    var lp = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 8)
    while True:
        var expected = Int64(0)
        if lp[unsafe_offset=0].compare_exchange(expected, Int64(1)):
            return
        _ = external_call["sched_yield", Int32]()


def _mcp_scene_unlock():
    Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 8)[unsafe_offset=0].store(Int64(0))


def _mcp_scene_bump_seq() -> Int:
    if not _mcp_scene_ensure():
        return -1
    _mcp_scene_lock()
    var seq_p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 16)
    var next_seq = Int(seq_p[unsafe_offset=0].load()) + 1
    seq_p[unsafe_offset=0].store(Int64(next_seq))
    _mcp_scene_unlock()
    return next_seq


def _mcp_scene_push(mut m: FlintSceneMutation) -> Tuple[Int, Int]:
    if not _mcp_scene_ensure():
        return (-1, 0)
    _mcp_scene_lock()
    var seq_p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 16)
    var cnt_p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 24)
    var cnt = Int(cnt_p[unsafe_offset=0].load())
    if cnt >= MCP_SCENE_MAX_QUEUE:
        _mcp_scene_unlock()
        return (-2, cnt)
    var next_seq = Int(seq_p[unsafe_offset=0].load()) + 1
    m.seq = next_seq
    var rec = MCP_SCENE_BASE + 64 + cnt * MCP_SCENE_REC_BYTES
    var p_i32 = Pointer[Int32, MutAnyOrigin](unsafe_from_address=rec)
    p_i32[unsafe_offset=0] = Int32(m.op)
    p_i32[unsafe_offset=1] = Int32(m.target_id)
    Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rec + 8)[unsafe_offset=0] = m.mask
    p_i32[unsafe_offset=3] = Int32(m.seq)
    var p_f32 = Pointer[Float32, MutAnyOrigin](unsafe_from_address=rec + 16)
    p_f32[unsafe_offset=0] = m.v0.x
    p_f32[unsafe_offset=1] = m.v0.y
    p_f32[unsafe_offset=2] = m.v0.z
    p_f32[unsafe_offset=3] = m.v1.x
    p_f32[unsafe_offset=4] = m.v1.y
    p_f32[unsafe_offset=5] = m.v1.z
    p_f32[unsafe_offset=6] = m.quat.x
    p_f32[unsafe_offset=7] = m.quat.y
    p_f32[unsafe_offset=8] = m.quat.z
    p_f32[unsafe_offset=9] = m.quat.w
    p_f32[unsafe_offset=10] = m.f0
    p_f32[unsafe_offset=11] = m.f1
    p_f32[unsafe_offset=12] = m.f2
    p_f32[unsafe_offset=13] = m.f3
    var nbytes = m.target_name.as_bytes()
    var nlen = min(48, len(nbytes))
    Pointer[Int32, MutAnyOrigin](unsafe_from_address=rec + 72)[unsafe_offset=0] = Int32(nlen)
    var p_u8 = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=rec + 76)
    for i in range(nlen):
        p_u8[unsafe_offset=i] = nbytes[i]
    seq_p[unsafe_offset=0].store(Int64(next_seq))
    cnt_p[unsafe_offset=0].store(Int64(cnt + 1))
    _mcp_scene_unlock()
    return (next_seq, cnt + 1)


def mcp_scene_mutation_seq() -> Int:
    if not _mcp_scene_ensure():
        return 0
    return Int(Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 16)[unsafe_offset=0].load())


def mcp_scene_pending_count() -> Int:
    if not _mcp_scene_ensure():
        return 0
    return Int(Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 24)[unsafe_offset=0].load())


def mcp_scene_reset() -> Tuple[Int, Int]:
    if not _mcp_scene_ensure():
        return (0, -1)
    _mcp_scene_lock()
    var seq_p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 16)
    var cnt_p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 24)
    var cleared = Int(cnt_p[unsafe_offset=0].load())
    var next_seq = Int(seq_p[unsafe_offset=0].load()) + 1
    cnt_p[unsafe_offset=0].store(Int64(0))
    seq_p[unsafe_offset=0].store(Int64(next_seq))
    _mcp_scene_unlock()
    return (cleared, next_seq)


def pop_mcp_scene_mutations() -> List[FlintSceneMutation]:
    var out = List[FlintSceneMutation]()
    if not _mcp_scene_ensure():
        return out^
    _mcp_scene_lock()
    var cnt_p = Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=MCP_SCENE_BASE + 24)
    var cnt = Int(cnt_p[unsafe_offset=0].load())
    for i in range(min(MCP_SCENE_MAX_QUEUE, cnt)):
        var rec = MCP_SCENE_BASE + 64 + i * MCP_SCENE_REC_BYTES
        var p_i32 = Pointer[Int32, MutAnyOrigin](unsafe_from_address=rec)
        var p_f32 = Pointer[Float32, MutAnyOrigin](unsafe_from_address=rec + 16)
        var m = FlintSceneMutation()
        m.op = Int(p_i32[unsafe_offset=0])
        m.target_id = Int(p_i32[unsafe_offset=1])
        m.mask = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=rec + 8)[unsafe_offset=0]
        m.seq = Int(p_i32[unsafe_offset=3])
        m.v0 = Vec3(p_f32[unsafe_offset=0], p_f32[unsafe_offset=1], p_f32[unsafe_offset=2])
        m.v1 = Vec3(p_f32[unsafe_offset=3], p_f32[unsafe_offset=4], p_f32[unsafe_offset=5])
        m.quat = Quat(p_f32[unsafe_offset=6], p_f32[unsafe_offset=7], p_f32[unsafe_offset=8], p_f32[unsafe_offset=9])
        m.f0 = p_f32[unsafe_offset=10]
        m.f1 = p_f32[unsafe_offset=11]
        m.f2 = p_f32[unsafe_offset=12]
        m.f3 = p_f32[unsafe_offset=13]
        var nlen = max(0, min(48, Int(Pointer[Int32, MutAnyOrigin](unsafe_from_address=rec + 72)[unsafe_offset=0])))
        if nlen > 0:
            var nb = List[UInt8](capacity=nlen)
            var p_u8 = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=rec + 76)
            for k in range(nlen):
                nb.append(p_u8[unsafe_offset=k])
            m.target_name = String(unsafe_from_utf8=nb)
        out.append(m^)
    cnt_p[unsafe_offset=0].store(Int64(0))
    _mcp_scene_unlock()
    return out^


@always_inline
def _is_finite_f64(v: Float64) -> Bool:
    return (v == v) and (abs(v) <= Float64(1.0e30))


def _parse_nonneg_int_node(doc: JsonDocument, node: Int) -> Tuple[Bool, Int]:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 2:
        return (False, -1)
    var v = doc.nodes[node].number
    if not _is_finite_f64(v) or v < 0.0 or v > Float64(2147483647.0):
        return (False, -1)
    var iv = Int(v)
    if Float64(iv) != v:
        return (False, -1)
    return (True, iv)


def _parse_vec3_node(doc: JsonDocument, node: Int) -> Tuple[Bool, Vec3]:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 4:
        return (False, Vec3())
    if doc.array_size(node) != 3:
        return (False, Vec3())
    var c0 = doc.at(node, 0)
    var c1 = doc.at(node, 1)
    var c2 = doc.at(node, 2)
    if doc.nodes[c0].kind != 2 or doc.nodes[c1].kind != 2 or doc.nodes[c2].kind != 2:
        return (False, Vec3())
    var x = doc.nodes[c0].number
    var y = doc.nodes[c1].number
    var z = doc.nodes[c2].number
    if not (_is_finite_f64(x) and _is_finite_f64(y) and _is_finite_f64(z)):
        return (False, Vec3())
    return (True, Vec3(Float32(x), Float32(y), Float32(z)))


def _parse_quat_or_euler(doc: JsonDocument, node: Int) -> Tuple[Bool, Quat]:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 4:
        return (False, Quat())
    var n = doc.array_size(node)
    if n == 4:
        var c0 = doc.at(node, 0)
        var c1 = doc.at(node, 1)
        var c2 = doc.at(node, 2)
        var c3 = doc.at(node, 3)
        if doc.nodes[c0].kind != 2 or doc.nodes[c1].kind != 2 or doc.nodes[c2].kind != 2 or doc.nodes[c3].kind != 2:
            return (False, Quat())
        var x = doc.nodes[c0].number
        var y = doc.nodes[c1].number
        var z = doc.nodes[c2].number
        var w = doc.nodes[c3].number
        if not (_is_finite_f64(x) and _is_finite_f64(y) and _is_finite_f64(z) and _is_finite_f64(w)):
            return (False, Quat())
        var len2 = x * x + y * y + z * z + w * w
        if len2 <= 1.0e-12:
            return (False, Quat())
        return (True, Quat(Float32(x), Float32(y), Float32(z), Float32(w)).normalized())
    if n == 3:
        var ev = _parse_vec3_node(doc, node)
        if not ev[0]:
            return (False, Quat())
        var deg2rad = Float32(0.0174532925) * Float32(0.5)
        var hx = ev[1].x * deg2rad
        var hy = ev[1].y * deg2rad
        var hz = ev[1].z * deg2rad
        var qx = Quat(sin(hx), Float32(0.0), Float32(0.0), cos(hx))
        var qy = Quat(Float32(0.0), sin(hy), Float32(0.0), cos(hy))
        var qz = Quat(Float32(0.0), Float32(0.0), sin(hz), cos(hz))
        return (True, qz.multiply(qy).multiply(qx).normalized())
    return (False, Quat())


def _handle_flint_mutate_scene(args_json: String) -> String:
    var raw = List[UInt8]()
    for b in args_json.as_bytes():
        raw.append(b)
    var doc = parse_json(raw)
    if not doc.valid or doc.root < 0 or doc.nodes[doc.root].kind != 5:
        return '{"ok":false,"error":"invalid json: expected object"}'
    var op_node = doc.object_get(doc.root, "op")
    if op_node < 0:
        op_node = doc.object_get(doc.root, "action")
    if op_node < 0 or doc.nodes[op_node].kind != 3:
        return '{"ok":false,"error":"missing or invalid op"}'
    var op = doc.text(op_node)
    if op.byte_length() == 0:
        return '{"ok":false,"error":"empty op"}'

    if op == "set_transform" or op == "entity_transform":
        var id_node = doc.object_get(doc.root, "entity_id")
        if id_node < 0:
            id_node = doc.object_get(doc.root, "instance_id")
        if id_node < 0:
            id_node = doc.object_get(doc.root, "id")
        var name_node = doc.object_get(doc.root, "name")
        var mut_rec = FlintSceneMutation()
        mut_rec.op = SCENE_OP_SET_TRANSFORM
        if id_node >= 0:
            var parsed_id = _parse_nonneg_int_node(doc, id_node)
            if not parsed_id[0]:
                return '{"ok":false,"error":"invalid entity_id: expected non-negative integer"}'
            mut_rec.target_id = parsed_id[1]
        elif name_node >= 0 and doc.nodes[name_node].kind == 3 and doc.text(name_node).byte_length() > 0:
            mut_rec.target_name = doc.text(name_node)
        else:
            return '{"ok":false,"error":"missing entity_id or name"}'

        var pos_node = doc.object_get(doc.root, "position")
        if pos_node >= 0:
            var pv = _parse_vec3_node(doc, pos_node)
            if not pv[0]:
                return '{"ok":false,"error":"invalid position: expected [x,y,z]"}'
            mut_rec.v0 = pv[1]
            mut_rec.mask |= SMASK_V0
        var rot_node = doc.object_get(doc.root, "rotation")
        if rot_node >= 0:
            var rq = _parse_quat_or_euler(doc, rot_node)
            if not rq[0]:
                return '{"ok":false,"error":"invalid rotation: expected non-zero [x,y,z,w] or [rx,ry,rz]"}'
            mut_rec.quat = rq[1]
            mut_rec.mask |= SMASK_QUAT
        var scl_node = doc.object_get(doc.root, "scale")
        if scl_node >= 0:
            if doc.nodes[scl_node].kind == 2:
                var sv = doc.nodes[scl_node].number
                if not _is_finite_f64(sv) or sv <= 0.0:
                    return '{"ok":false,"error":"invalid scale: expected positive number"}'
                var fs = Float32(sv)
                mut_rec.v1 = Vec3(fs, fs, fs)
                mut_rec.mask |= SMASK_V1
            else:
                var sv3 = _parse_vec3_node(doc, scl_node)
                if not sv3[0] or sv3[1].x <= Float32(0.0) or sv3[1].y <= Float32(0.0) or sv3[1].z <= Float32(0.0):
                    return '{"ok":false,"error":"invalid scale: expected positive [sx,sy,sz]"}'
                mut_rec.v1 = sv3[1]
                mut_rec.mask |= SMASK_V1
        if mut_rec.mask == 0:
            return '{"ok":false,"error":"set_transform requires position, rotation, or scale"}'
        var pushed = _mcp_scene_push(mut_rec)
        if pushed[0] == -1:
            return '{"ok":false,"error":"scene state backing unavailable"}'
        if pushed[0] == -2:
            return '{"ok":false,"error":"scene mutation queue full"}'
        return (
            '{"ok":true,"mutated":true,"op":"set_transform","seq":'
            + String(pushed[0])
            + ',"entity_id":'
            + String(mut_rec.target_id)
            + ',"pending":'
            + String(pushed[1])
            + "}"
        )

    if op == "set_light" or op == "light":
        var mut_rec = FlintSceneMutation()
        mut_rec.op = SCENE_OP_SET_LIGHT
        var tgt_node = doc.object_get(doc.root, "target")
        if tgt_node >= 0 and doc.nodes[tgt_node].kind == 3 and doc.text(tgt_node) == "sun":
            mut_rec.target_id = -1
        else:
            var lid_node = doc.object_get(doc.root, "light_id")
            if lid_node < 0:
                lid_node = doc.object_get(doc.root, "id")
            if lid_node < 0:
                return '{"ok":false,"error":"missing light_id or target=sun"}'
            var parsed_id = _parse_nonneg_int_node(doc, lid_node)
            if not parsed_id[0]:
                return '{"ok":false,"error":"invalid light_id: expected non-negative integer"}'
            mut_rec.target_id = parsed_id[1]

        var pos_node = doc.object_get(doc.root, "position")
        if pos_node >= 0:
            var pv = _parse_vec3_node(doc, pos_node)
            if not pv[0]:
                return '{"ok":false,"error":"invalid light position"}'
            mut_rec.v0 = pv[1]
            mut_rec.mask |= SMASK_V0
        var dir_node = doc.object_get(doc.root, "direction")
        if dir_node >= 0:
            var dv = _parse_vec3_node(doc, dir_node)
            if not dv[0] or dv[1].length() <= Float32(1.0e-5):
                return '{"ok":false,"error":"invalid light direction: expected non-zero [x,y,z]"}'
            var dn = dv[1].normalized()
            if mut_rec.target_id == -1:
                mut_rec.v0 = dn
                mut_rec.mask |= SMASK_V0
            else:
                mut_rec.quat = Quat(dn.x, dn.y, dn.z, Float32(1.0))
                mut_rec.mask |= SMASK_QUAT
        var col_node = doc.object_get(doc.root, "color")
        if col_node < 0:
            col_node = doc.object_get(doc.root, "radiance")
        if col_node >= 0:
            var cv = _parse_vec3_node(doc, col_node)
            if not cv[0] or cv[1].x < Float32(0.0) or cv[1].y < Float32(0.0) or cv[1].z < Float32(0.0):
                return '{"ok":false,"error":"invalid light color/radiance: expected non-negative [r,g,b]"}'
            mut_rec.v1 = cv[1]
            mut_rec.mask |= SMASK_V1
        var int_node = doc.object_get(doc.root, "intensity")
        if int_node >= 0:
            if doc.nodes[int_node].kind != 2 or not _is_finite_f64(doc.nodes[int_node].number) or doc.nodes[int_node].number < 0.0:
                return '{"ok":false,"error":"invalid light intensity"}'
            mut_rec.f0 = Float32(doc.nodes[int_node].number)
            mut_rec.mask |= SMASK_F0
        var rad_node = doc.object_get(doc.root, "radius")
        if rad_node >= 0:
            if doc.nodes[rad_node].kind != 2 or not _is_finite_f64(doc.nodes[rad_node].number) or doc.nodes[rad_node].number <= 0.0:
                return '{"ok":false,"error":"invalid light radius"}'
            mut_rec.f1 = Float32(doc.nodes[rad_node].number)
            mut_rec.mask |= SMASK_F1
        if mut_rec.mask == 0:
            return '{"ok":false,"error":"set_light requires position, direction, color, intensity, or radius"}'
        var pushed = _mcp_scene_push(mut_rec)
        if pushed[0] == -1:
            return '{"ok":false,"error":"scene state backing unavailable"}'
        if pushed[0] == -2:
            return '{"ok":false,"error":"scene mutation queue full"}'
        return (
            '{"ok":true,"mutated":true,"op":"set_light","seq":'
            + String(pushed[0])
            + ',"light_id":'
            + String(mut_rec.target_id)
            + ',"pending":'
            + String(pushed[1])
            + "}"
        )

    if op == "set_material" or op == "material":
        var mid_node = doc.object_get(doc.root, "material_id")
        if mid_node < 0:
            mid_node = doc.object_get(doc.root, "id")
        var parsed_id = _parse_nonneg_int_node(doc, mid_node)
        if not parsed_id[0]:
            return '{"ok":false,"error":"missing or invalid material_id"}'
        var mut_rec = FlintSceneMutation()
        mut_rec.op = SCENE_OP_SET_MATERIAL
        mut_rec.target_id = parsed_id[1]
        var bc_node = doc.object_get(doc.root, "base_color")
        if bc_node >= 0:
            var bv = _parse_vec3_node(doc, bc_node)
            if (
                not bv[0]
                or bv[1].x < Float32(0.0) or bv[1].x > Float32(1.0)
                or bv[1].y < Float32(0.0) or bv[1].y > Float32(1.0)
                or bv[1].z < Float32(0.0) or bv[1].z > Float32(1.0)
            ):
                return '{"ok":false,"error":"invalid base_color: expected [r,g,b] in 0..1"}'
            mut_rec.v0 = bv[1]
            mut_rec.mask |= SMASK_V0
        var r_node = doc.object_get(doc.root, "roughness")
        if r_node >= 0:
            if doc.nodes[r_node].kind != 2 or not _is_finite_f64(doc.nodes[r_node].number):
                return '{"ok":false,"error":"invalid roughness"}'
            var rv = doc.nodes[r_node].number
            if rv < 0.0 or rv > 1.0:
                return '{"ok":false,"error":"roughness out of range [0,1]"}'
            mut_rec.f0 = Float32(rv)
            mut_rec.mask |= SMASK_F0
        var m_node = doc.object_get(doc.root, "metallic")
        if m_node >= 0:
            if doc.nodes[m_node].kind != 2 or not _is_finite_f64(doc.nodes[m_node].number):
                return '{"ok":false,"error":"invalid metallic"}'
            var mv = doc.nodes[m_node].number
            if mv < 0.0 or mv > 1.0:
                return '{"ok":false,"error":"metallic out of range [0,1]"}'
            mut_rec.f1 = Float32(mv)
            mut_rec.mask |= SMASK_F1
        var em_node = doc.object_get(doc.root, "emissive")
        if em_node >= 0:
            var ev = _parse_vec3_node(doc, em_node)
            if not ev[0] or ev[1].x < Float32(0.0) or ev[1].y < Float32(0.0) or ev[1].z < Float32(0.0):
                return '{"ok":false,"error":"invalid emissive: expected non-negative [r,g,b]"}'
            mut_rec.v1 = ev[1]
            mut_rec.mask |= SMASK_V1
        if mut_rec.mask == 0:
            return '{"ok":false,"error":"set_material requires base_color, roughness, metallic, or emissive"}'
        var pushed = _mcp_scene_push(mut_rec)
        if pushed[0] == -1:
            return '{"ok":false,"error":"scene state backing unavailable"}'
        if pushed[0] == -2:
            return '{"ok":false,"error":"scene mutation queue full"}'
        return (
            '{"ok":true,"mutated":true,"op":"set_material","seq":'
            + String(pushed[0])
            + ',"material_id":'
            + String(mut_rec.target_id)
            + ',"pending":'
            + String(pushed[1])
            + "}"
        )

    if op == "set_camera" or op == "camera":
        var mut_rec = FlintSceneMutation()
        mut_rec.op = SCENE_OP_SET_CAMERA
        mut_rec.target_id = 0
        var pos_node = doc.object_get(doc.root, "position")
        if pos_node >= 0:
            var pv = _parse_vec3_node(doc, pos_node)
            if not pv[0]:
                return '{"ok":false,"error":"invalid camera position"}'
            mut_rec.v0 = pv[1]
            mut_rec.mask |= SMASK_V0
        var tgt_node = doc.object_get(doc.root, "target")
        if tgt_node >= 0:
            var tv = _parse_vec3_node(doc, tgt_node)
            if not tv[0]:
                return '{"ok":false,"error":"invalid camera target"}'
            mut_rec.v1 = tv[1]
            mut_rec.mask |= SMASK_V1
        if (mut_rec.mask & SMASK_V0) != 0 and (mut_rec.mask & SMASK_V1) != 0:
            if (mut_rec.v1 - mut_rec.v0).length() <= Float32(1.0e-5):
                return '{"ok":false,"error":"camera target must differ from position"}'
        var fov_node = doc.object_get(doc.root, "fov_deg")
        if fov_node >= 0:
            if doc.nodes[fov_node].kind != 2 or not _is_finite_f64(doc.nodes[fov_node].number):
                return '{"ok":false,"error":"invalid fov_deg"}'
            var fv = doc.nodes[fov_node].number
            if fv < 10.0 or fv > 160.0:
                return '{"ok":false,"error":"fov_deg out of range [10,160]"}'
            mut_rec.f0 = Float32(fv)
            mut_rec.mask |= SMASK_F0
        var exp_node = doc.object_get(doc.root, "exposure")
        if exp_node >= 0:
            if doc.nodes[exp_node].kind != 2 or not _is_finite_f64(doc.nodes[exp_node].number):
                return '{"ok":false,"error":"invalid exposure"}'
            var ex = doc.nodes[exp_node].number
            if ex < 0.01 or ex > 100.0:
                return '{"ok":false,"error":"exposure out of range [0.01,100]"}'
            mut_rec.f1 = Float32(ex)
            mut_rec.mask |= SMASK_F1
        if mut_rec.mask == 0:
            return '{"ok":false,"error":"set_camera requires position, target, fov_deg, or exposure"}'
        var pushed = _mcp_scene_push(mut_rec)
        if pushed[0] == -1:
            return '{"ok":false,"error":"scene state backing unavailable"}'
        if pushed[0] == -2:
            return '{"ok":false,"error":"scene mutation queue full"}'
        return (
            '{"ok":true,"mutated":true,"op":"set_camera","seq":'
            + String(pushed[0])
            + ',"pending":'
            + String(pushed[1])
            + "}"
        )

    if op == "set_cvar":
        var name_node = doc.object_get(doc.root, "name")
        if name_node < 0:
            name_node = doc.object_get(doc.root, "cvar")
        var cvar_name = doc.text(name_node)
        if cvar_name.byte_length() == 0 or ConsoleVars().find(cvar_name) < 0:
            return '{"ok":false,"error":"unknown or missing cvar name"}'
        var val_node = doc.object_get(doc.root, "value")
        if val_node < 0 or doc.nodes[val_node].kind != 2 or not _is_finite_f64(doc.nodes[val_node].number):
            return '{"ok":false,"error":"invalid cvar value"}'
        var eff = _mcp_cvar_set(cvar_name, Float32(doc.nodes[val_node].number))
        var seq = _mcp_scene_bump_seq()
        return (
            '{"ok":true,"mutated":true,"op":"set_cvar","seq":'
            + String(seq)
            + ',"name":'
            + json_quote(cvar_name)
            + ',"value":'
            + String(eff)
            + "}"
        )

    if op == "reset" or op == "clear":
        var rst = mcp_scene_reset()
        return (
            '{"ok":true,"mutated":true,"op":"reset","cleared":'
            + String(rst[0])
            + ',"seq":'
            + String(rst[1])
            + "}"
        )

    return '{"ok":false,"error":' + json_quote("unsupported op: " + op) + "}"


def apply_mcp_scene_mutations(mut ren: FlintAaaVulkanRenderer) -> Int:
    """Drains pending MCP scene mutations and applies them directly to `FlintAaaVulkanRenderer`."""
    var muts = pop_mcp_scene_mutations()
    var applied = 0
    for i in range(len(muts)):
        ref m = muts[i]
        if m.op == SCENE_OP_SET_TRANSFORM:
            var idx = m.target_id
            if idx >= 0 and idx < len(ren.frame_instances):
                var cur = ren.frame_instances[idx].copy()
                cur.prev_world_mat = cur.world_mat.copy()
                var pos = m.v0 if (m.mask & SMASK_V0) != 0 else Vec3(
                    cur.world_mat.values[12], cur.world_mat.values[13], cur.world_mat.values[14]
                )
                var sx = Vec3(cur.world_mat.values[0], cur.world_mat.values[1], cur.world_mat.values[2]).length()
                var sy = Vec3(cur.world_mat.values[4], cur.world_mat.values[5], cur.world_mat.values[6]).length()
                var sz = Vec3(cur.world_mat.values[8], cur.world_mat.values[9], cur.world_mat.values[10]).length()
                var scl = m.v1 if (m.mask & SMASK_V1) != 0 else Vec3(
                    max(Float32(1e-4), sx), max(Float32(1e-4), sy), max(Float32(1e-4), sz)
                )
                if (m.mask & SMASK_QUAT) != 0:
                    cur.world_mat = trs(pos, m.quat, scl)
                elif (m.mask & SMASK_V1) != 0:
                    var inv_sx = scl.x / max(Float32(1e-4), sx)
                    var inv_sy = scl.y / max(Float32(1e-4), sy)
                    var inv_sz = scl.z / max(Float32(1e-4), sz)
                    cur.world_mat.values[0] *= inv_sx; cur.world_mat.values[1] *= inv_sx; cur.world_mat.values[2] *= inv_sx
                    cur.world_mat.values[4] *= inv_sy; cur.world_mat.values[5] *= inv_sy; cur.world_mat.values[6] *= inv_sy
                    cur.world_mat.values[8] *= inv_sz; cur.world_mat.values[9] *= inv_sz; cur.world_mat.values[10] *= inv_sz
                    cur.world_mat.values[12] = pos.x; cur.world_mat.values[13] = pos.y; cur.world_mat.values[14] = pos.z
                else:
                    cur.world_mat.values[12] = pos.x; cur.world_mat.values[13] = pos.y; cur.world_mat.values[14] = pos.z
                cur.uniform_scale = max(Float32(0.001), max(scl.x, max(scl.y, scl.z)))
                ren.frame_instances[idx] = cur^
                applied += 1
        elif m.op == SCENE_OP_SET_LIGHT:
            if m.target_id == -1:
                if (m.mask & SMASK_V0) != 0:
                    ren.sun_dir = m.v0.normalized()
                if (m.mask & SMASK_V1) != 0:
                    ren.sun_radiance = m.v1
                if (m.mask & SMASK_F0) != 0:
                    ren.sun_radiance = ren.sun_radiance * m.f0
                applied += 1
            elif m.target_id >= 0 and m.target_id < AAA_MAX_CLUSTERED_LIGHTS:
                if m.target_id == len(ren.frame_cluster_lights):
                    var l_pos = m.v0 if (m.mask & SMASK_V0) != 0 else Vec3(0.0, 0.0, 0.0)
                    var l_rad_rgb = m.v1 if (m.mask & SMASK_V1) != 0 else Vec3(10.0, 10.0, 10.0)
                    if (m.mask & SMASK_F0) != 0:
                        l_rad_rgb = l_rad_rgb * m.f0
                    var l_r = m.f1 if (m.mask & SMASK_F1) != 0 else Float32(10.0)
                    if (m.mask & SMASK_QUAT) != 0:
                        var l_dir = Vec3(m.quat.x, m.quat.y, m.quat.z).normalized()
                        var nl = AaaClusterLight.spot(l_pos, l_dir, l_rad_rgb, l_r)
                        ren.frame_cluster_lights.append(nl^)
                    else:
                        var nl = AaaClusterLight.point(l_pos, l_rad_rgb, l_r)
                        ren.frame_cluster_lights.append(nl^)
                    applied += 1
                elif m.target_id < len(ren.frame_cluster_lights):
                    var cl = ren.frame_cluster_lights[m.target_id].copy()
                    if (m.mask & SMASK_V0) != 0:
                        cl.position = m.v0
                    if (m.mask & SMASK_QUAT) != 0:
                        cl.axis_u = Vec3(m.quat.x, m.quat.y, m.quat.z).normalized()
                        if cl.kind != AAA_LIGHT_SPOT or cl.param1 <= Float32(0.0):
                            cl.param0 = Float32(0.81915206)  # cos(35 deg)
                            cl.param1 = Float32(8.295963)    # 1.0 / (cos(20 deg) - cos(35 deg))
                        cl.kind = AAA_LIGHT_SPOT
                    if (m.mask & SMASK_V1) != 0:
                        cl.radiance = m.v1
                    if (m.mask & SMASK_F0) != 0:
                        cl.radiance = cl.radiance * m.f0
                    if (m.mask & SMASK_F1) != 0:
                        cl.radius = max(Float32(0.05), m.f1)
                    ren.frame_cluster_lights[m.target_id] = cl^
                    applied += 1
        elif m.op == SCENE_OP_SET_MATERIAL:
            if m.target_id >= 0 and m.target_id < len(ren.materials):
                var mat = ren.materials[m.target_id].copy()
                if (m.mask & SMASK_V0) != 0:
                    mat.base_color = m.v0
                if (m.mask & SMASK_F0) != 0:
                    mat.roughness = max(Float32(0.04), min(Float32(1.0), m.f0))
                if (m.mask & SMASK_F1) != 0:
                    mat.metallic = max(Float32(0.0), min(Float32(1.0), m.f1))
                if (m.mask & SMASK_V1) != 0:
                    mat.emissive = m.v1
                if ren.buf_materials.cpu_address != 0:
                    ren.update_material(m.target_id, mat)
                else:
                    ren.materials[m.target_id] = mat^
                applied += 1
        elif m.op == SCENE_OP_SET_CAMERA:
            if (m.mask & SMASK_V0) != 0:
                ren.cam_pos = m.v0
            if (m.mask & SMASK_V1) != 0:
                var fwd = (m.v1 - ren.cam_pos).normalized()
                if fwd.length() > Float32(1e-5):
                    var right = view_right(fwd)
                    var up = fwd.cross(right).normalized()
                    ren.cam_fwd = fwd
                    ren.cam_right = right
                    ren.cam_up = up
            if (m.mask & SMASK_F0) != 0:
                ren.fov_y_deg = m.f0
            if (m.mask & SMASK_F1) != 0:
                ren.exposure = m.f1
            applied += 1
    return applied


struct FlintMcpInfo(Copyable, Movable):
    var app_name: String
    var tick: Int
    var backend: String
    var custom_state_json: String

    def __init__(
        out self,
        app_name: String = "flint-app",
        tick: Int = 0,
        backend: String = "SIMD-CPU",
        custom_state_json: String = "{}",
    ):
        self.app_name = app_name
        self.tick = tick
        self.backend = backend
        self.custom_state_json = custom_state_json


def handle_flint_mcp_query(method: String, args_json: String, info: FlintMcpInfo) -> String:
    """Pure JSON responder callable over LiveLink Unix sockets, stdio MCP, or unit tests."""
    if method == "ping":
        return '{"ok":true,"pong":true}'
    if method == "help":
        return '{"ok":true,"tools":["ping","help","flint_hello","flint_memory","flint_state","flint_flip_diff","flint_set_cvar","flint_get_cvar","flint_mutate_scene"]}'
    if method == "flint_hello":
        return (
            '{"engine":"flint","version":"0.1.0","app":'
            + json_quote(info.app_name)
            + ',"build":'
            + json_quote(build_stamp())
            + ',"backend":'
            + json_quote(info.backend)
            + ',"tick":'
            + String(info.tick)
            + "}"
        )
    if method == "flint_memory":
        return (
            '{"cpu_live_bytes":'
            + String(mem_total(W_CPU_LIVE))
            + ',"vram_live_bytes":'
            + String(mem_total(W_VRAM_LIVE))
            + "}"
        )
    if method == "flint_state":
        return (
            '{"tick":'
            + String(info.tick)
            + ',"state":'
            + info.custom_state_json
            + "}"
        )
    if method == "flint_flip_diff":
        var raw = List[UInt8]()
        for b in args_json.as_bytes():
            raw.append(b)
        var doc = parse_json(raw)
        if not doc.valid:
            return '{"ok":false,"error":"invalid json"}'
        var path_a = doc.text(doc.object_get(doc.root, "a"))
        var path_b = doc.text(doc.object_get(doc.root, "b"))
        var img_a = decode_png(read_bytes(path_a), 4096)
        var img_b = decode_png(read_bytes(path_b), 4096)
        if not img_a.valid or not img_b.valid or img_a.width != img_b.width or img_a.height != img_b.height:
            return '{"ok":false,"error":"mismatched or unreadable pngs"}'
        var err = flip_mean(flip_error_map(img_a.pixels, img_b.pixels, img_a.width, img_a.height))
        return '{"ok":true,"flip_mean":' + String(err) + "}"
    if method == "flint_set_cvar":
        var raw = List[UInt8]()
        for b in args_json.as_bytes():
            raw.append(b)
        var doc = parse_json(raw)
        if not doc.valid:
            return '{"ok":false,"error":"invalid json"}'
        var name_node = doc.object_get(doc.root, "name")
        if name_node < 0:
            name_node = doc.object_get(doc.root, "cvar")
        var cvar_name = doc.text(name_node)
        if cvar_name.byte_length() == 0:
            return '{"ok":false,"error":"missing cvar name"}'
        var val_node = doc.object_get(doc.root, "value")
        var cvar_val = Float32(doc.number(val_node, 0.0))
        var stored_val = _mcp_cvar_set(cvar_name, cvar_val)
        return '{"ok":true,"name":' + json_quote(cvar_name) + ',"value":' + String(stored_val) + "}"
    if method == "flint_get_cvar":
        var raw = List[UInt8]()
        for b in args_json.as_bytes():
            raw.append(b)
        var doc = parse_json(raw)
        if not doc.valid:
            return '{"ok":false,"error":"invalid json"}'
        var name_node = doc.object_get(doc.root, "name")
        if name_node < 0:
            name_node = doc.object_get(doc.root, "cvar")
        var cvar_name = doc.text(name_node)
        if cvar_name.byte_length() == 0:
            return '{"ok":false,"error":"missing cvar name"}'
        var cur_val = _mcp_cvar_get(cvar_name)
        return '{"ok":true,"name":' + json_quote(cvar_name) + ',"value":' + String(cur_val) + "}"
    if method == "flint_mutate_scene":
        return _handle_flint_mutate_scene(args_json)
    return '{"ok":false,"error":' + json_quote("unknown method: " + method) + "}"


def _make_sockaddr_un(path: String) -> List[UInt8]:
    """Builds a 110-byte Linux `sockaddr_un` (`AF_UNIX = 1` + NUL-terminated path)."""
    var addr = List[UInt8](length=110, fill=0)
    addr[0] = UInt8(1)
    addr[1] = UInt8(0)
    var pbytes = path.as_bytes()
    var limit = min(107, len(pbytes))
    for i in range(limit):
        addr[2 + i] = pbytes[i]
    return addr^


def _unlink_cpath(path: String):
    var cstr = List[UInt8]()
    for b in path.as_bytes():
        cstr.append(b)
    cstr.append(0)
    _ = external_call["unlink", Int32](cstr.unsafe_ptr())


struct FlintMcpLiveEndpoint(Movable):
    """Non-blocking Unix domain socket LiveLink server (`{"method":"flint_state","args":"{}"}`).

    Polled once per frame via `poll_once(info)` with zero blocking or heap churn when idle.
    """
    var socket_path: String
    var listen_fd: Int
    var active: Bool
    var client_fds: List[Int]
    var client_bufs: List[List[UInt8]]
    var client_idle: List[Int]
    var _scratch: List[UInt8]

    def __init__(out self, socket_path: String = ""):
        self.socket_path = socket_path
        self.listen_fd = -1
        self.active = False
        self.client_fds = List[Int]()
        self.client_bufs = List[List[UInt8]]()
        self.client_idle = List[Int]()
        self._scratch = List[UInt8](length=MCP_MAX_REQ_BYTES, fill=0)
        if socket_path.byte_length() > 0:
            _ = self.start(socket_path)

    def __deinit__(deinit self):
        for i in range(len(self.client_fds)):
            close_fd(self.client_fds[i])
        if self.listen_fd >= 0:
            close_fd(self.listen_fd)
        if self.active and self.socket_path.byte_length() > 0:
            _unlink_cpath(self.socket_path)

    def start(mut self, socket_path: String) -> Bool:
        self.close()
        self.socket_path = socket_path
        _unlink_cpath(socket_path)
        var fd = Int(external_call["socket", Int32](AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, Int32(0)))
        if fd < 0:
            return False
        var addr = _make_sockaddr_un(socket_path)
        if external_call["bind", Int32](Int32(fd), addr.unsafe_ptr(), UInt32(110)) != 0:
            close_fd(fd)
            return False
        if external_call["listen", Int32](Int32(fd), Int32(8)) != 0:
            close_fd(fd)
            _unlink_cpath(socket_path)
            return False
        self.listen_fd = fd
        self.active = True
        return True

    def _drop_client(mut self, idx: Int):
        close_fd(self.client_fds[idx])
        _ = self.client_fds.pop(idx)
        _ = self.client_bufs.pop(idx)
        _ = self.client_idle.pop(idx)

    def _answer_client(mut self, idx: Int, info: FlintMcpInfo) -> Bool:
        var method = String("flint_hello")
        var args = String("{}")
        var doc = parse_json(self.client_bufs[idx])
        if doc.valid:
            var m_txt = doc.text(doc.object_get(doc.root, "method"))
            if m_txt.byte_length() > 0:
                method = m_txt
            var a_node = doc.object_get(doc.root, "args")
            if a_node >= 0:
                if doc.nodes[a_node].kind == 3:
                    var a_txt = doc.text(a_node)
                    if a_txt.byte_length() > 0:
                        args = a_txt
                elif doc.nodes[a_node].kind == 5:
                    args = _json_node_to_string(doc, a_node)
        var resp = handle_flint_mcp_query(method, args, info) + "\n"
        var rbytes = List[UInt8]()
        for b in resp.as_bytes():
            rbytes.append(b)
        _ = external_call["send", Int64](
            Int32(self.client_fds[idx]), rbytes.unsafe_ptr(), Int64(len(rbytes)), MSG_NOSIGNAL
        )
        self._drop_client(idx)
        return True

    def poll_once(mut self, info: FlintMcpInfo) -> Int:
        """Accepts and answers any queued non-blocking MCP requests. Returns count answered."""
        if not self.active or self.listen_fd < 0:
            return 0
        var answered = 0
        for _ in range(4):
            if len(self.client_fds) >= MCP_MAX_CLIENTS:
                break
            var cfd = Int(
                external_call["accept4", Int32](
                    Int32(self.listen_fd), Int(0), Int(0), SOCK_CLOEXEC | SOCK_NONBLOCK,
                )
            )
            if cfd < 0:
                break
            self.client_fds.append(cfd)
            self.client_bufs.append(List[UInt8]())
            self.client_idle.append(0)

        var idx = 0
        while idx < len(self.client_fds):
            var cfd = self.client_fds[idx]
            var eof = False
            var fatal = False
            while len(self.client_bufs[idx]) < MCP_MAX_REQ_BYTES:
                var room = MCP_MAX_REQ_BYTES - len(self.client_bufs[idx])
                var n = Int(external_call["read", Int64](Int32(cfd), self._scratch.unsafe_ptr(), Int64(room)))
                if n > 0:
                    for k in range(n):
                        self.client_bufs[idx].append(self._scratch[k])
                    self.client_idle[idx] = 0
                elif n == 0:
                    eof = True
                    break
                else:
                    var err = last_errno()
                    if err == EINTR:
                        continue
                    if err == EAGAIN:
                        break
                    fatal = True
                    break

            if fatal:
                self._drop_client(idx)
                continue

            var blen = len(self.client_bufs[idx])
            if blen == 0:
                if eof:
                    self._drop_client(idx)
                    continue
                self.client_idle[idx] += 1
                if self.client_idle[idx] >= MCP_CLIENT_IDLE_POLLS:
                    self._drop_client(idx)
                    continue
                idx += 1
                continue

            var complete = eof or blen >= MCP_MAX_REQ_BYTES or self.client_bufs[idx][blen - 1] == 10
            if not complete:
                var probe_doc = parse_json(self.client_bufs[idx])
                if probe_doc.valid:
                    complete = True

            if complete:
                if self._answer_client(idx, info):
                    answered += 1
                continue

            self.client_idle[idx] += 1
            if self.client_idle[idx] >= MCP_CLIENT_IDLE_POLLS:
                self._drop_client(idx)
                continue
            idx += 1

        return answered

    def close(mut self):
        for i in range(len(self.client_fds)):
            close_fd(self.client_fds[i])
        self.client_fds.clear()
        self.client_bufs.clear()
        self.client_idle.clear()
        if self.listen_fd >= 0:
            close_fd(self.listen_fd)
            self.listen_fd = -1
        if self.active and self.socket_path.byte_length() > 0:
            _unlink_cpath(self.socket_path)
        self.active = False


def query_flint_mcp_unix(socket_path: String, method: String, args_json: String = "{}") -> String:
    """Connects to a running `FlintMcpLiveEndpoint` Unix socket and returns its JSON response."""
    var fd = Int(external_call["socket", Int32](AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, Int32(0)))
    if fd < 0:
        return '{"ok":false,"error":"socket failed"}'
    var addr = _make_sockaddr_un(socket_path)
    if external_call["connect", Int32](Int32(fd), addr.unsafe_ptr(), UInt32(110)) != 0:
        close_fd(fd)
        return '{"ok":false,"error":"connect failed"}'
    var payload = '{"method":' + json_quote(method) + ',"args":' + json_quote(args_json) + "}"
    var pbytes = List[UInt8]()
    for b in payload.as_bytes():
        pbytes.append(b)
    _ = external_call["send", Int64](Int32(fd), pbytes.unsafe_ptr(), Int64(len(pbytes)), MSG_NOSIGNAL)
    _ = external_call["shutdown", Int32](Int32(fd), Int32(1))  # SHUT_WR
    var buf = List[UInt8](length=4096, fill=0)
    var n = Int(external_call["read", Int64](Int32(fd), buf.unsafe_ptr(), Int64(4095)))
    close_fd(fd)
    if n <= 0:
        return '{"ok":false,"error":"empty response"}'
    var out = List[UInt8](capacity=n)
    for i in range(n):
        if buf[i] != 10:
            out.append(buf[i])
    return String(unsafe_from_utf8=out)

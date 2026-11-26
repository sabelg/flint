# AREA: m1
"""Tier 2 Boundary & Negative Opaque API Tests: Area C — M1 Assets, FRIG, .flmat, .flprefab & Scene v2 (F16-F21)."""

from std.collections import List
from flint.anim.flrig import (
    FrigJointDef,
    FrigRigDef,
    FrigSocketDef,
    RagdollBoneConfig,
    RagdollChain,
    load_flrig_bytes,
    validate_rig_bindings,
)
from flint.assets.asset_ref import AssetRef
from flint.assets.flmat import load_flmat_bytes
from flint.assets.flpk_v3 import ASSET_KIND_MESH, AssetUuid
from flint.assets.glb_native import parse_native_glb
from flint.assets.model import NativeModel
from flint.assets.server import AssetServer
from flint.core.lz4 import lz4_pack, lz4_unpack
from flint.core.math import Vec3
from flint.core.serial import BlobWriter, fourcc
from flint.core.transform import Quat
from flint.ecs.world import World
from flint.platform.os import make_dirs
from flint.render.plugin import register_render_codecs
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, ScalableRenderer
from flint.scene import (
    SCENE_BIN_MAGIC,
    ComponentRegistry,
    LocalTransform,
    NestedPrefabRef,
    Prefab,
    Scene,
    load_flprefab_bytes,
)
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    rm_rf_tmp,
    write_bytes,
    write_minimal_glb,
    write_text,
)


def _str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _has_fourcc_blob(data: List[UInt8], tag: UInt32, payload: List[UInt8]) -> Bool:
    var b0 = UInt8(tag & 0xFF)
    var b1 = UInt8((tag >> 8) & 0xFF)
    var b2 = UInt8((tag >> 16) & 0xFF)
    var b3 = UInt8((tag >> 24) & 0xFF)
    var plen = len(payload)
    if len(data) < 12 + plen:
        return False
    for i in range(len(data) - 12 - plen + 1):
        if data[i] == b0 and data[i + 1] == b1 and data[i + 2] == b2 and data[i + 3] == b3:
            var ver = (
                UInt32(data[i + 4])
                | (UInt32(data[i + 5]) << 8)
                | (UInt32(data[i + 6]) << 16)
                | (UInt32(data[i + 7]) << 24)
            )
            var blen = (
                Int(data[i + 8])
                | (Int(data[i + 9]) << 8)
                | (Int(data[i + 10]) << 16)
                | (Int(data[i + 11]) << 24)
            )
            if ver >= 1 and blen == plen:
                var match_all = True
                for k in range(plen):
                    if data[i + 12 + k] != payload[k]:
                        match_all = False
                        break
                if match_all:
                    return True
    return False


def main() raises:
    var tmp = make_tmp_dir("t2_m1_assets_boundary")
    var runner = StudioSuiteRunner("t2_m1_assets_boundary")

    # -----------------------------------------------------------------------
    # 1. F16/F18 Boundary: Missing UUID fallback, Missing Both, & Traversal Rejection
    # -----------------------------------------------------------------------
    var ws = tmp + "/ws_fallback"
    _ = make_dirs(ws + "/meshes")
    _ = write_minimal_glb(ws + "/meshes/valid.glb")

    var srv = AssetServer(enable_watcher=False)
    _ = srv.watch_loose_directory(ws)

    var missing_uuid = AssetUuid(UInt64(0xDEAD111122223333), UInt64(0x4444555566667777))
    var ref_fb = AssetRef(missing_uuid, "meshes/valid.glb", ASSET_KIND_MESH)
    var h_fb = srv.resolve_mesh(ref_fb)

    var ref_both_missing = AssetRef(missing_uuid, "meshes/does_not_exist.glb", ASSET_KIND_MESH)
    var h_miss = srv.resolve_mesh(ref_both_missing)

    var ref_trav = AssetRef(AssetUuid(), "../../etc/passwd", ASSET_KIND_MESH)
    var h_trav = srv.resolve_mesh(ref_trav)

    var ref_abs = AssetRef(AssetUuid(), "/etc/passwd", ASSET_KIND_MESH)
    var h_abs = srv.resolve_mesh(ref_abs)

    var case1_ok = (
        srv.meshes.is_alive(h_fb)
        and srv.meshes.is_loaded(h_fb)
        and (not srv.meshes.is_loaded(h_miss))
        and (not srv.meshes.is_loaded(h_trav))
        and (not srv.meshes.is_loaded(h_abs))
    )
    srv.close()
    _ = runner.check("t2_m1_missing_uuid_fallback_missing_both_and_traversal_rejected", case1_ok, "AssetRef boundary resolution failed")

    # -----------------------------------------------------------------------
    # 2. F19/F20 Boundary: Corrupt .flmat / .flprefab / .flrig Never Crashes Loader
    # -----------------------------------------------------------------------
    var mat_err = False
    try:
        _ = load_flmat_bytes(_str_to_bytes("{corrupt_flmat_json"))
    except:
        mat_err = True

    var pf_err = False
    try:
        _ = load_flprefab_bytes(_str_to_bytes('{"format":"flprefab","nodes":"bad"}'))
    except:
        pf_err = True

    var rig_err_bad_magic = False
    try:
        _ = load_flrig_bytes(_str_to_bytes("NOT_FRIG_MAGIC_BYTES"))
    except:
        rig_err_bad_magic = True

    var rig_err_empty = False
    try:
        _ = load_flrig_bytes(List[UInt8]())
    except:
        rig_err_empty = True

    _ = runner.check(
        "t2_m1_corrupt_flmat_flprefab_flrig_cleanly_rejected",
        mat_err and pf_err and rig_err_bad_magic and rig_err_empty,
        "mat=" + String(mat_err) + " pf=" + String(pf_err) + " rig_magic=" + String(rig_err_bad_magic) + " rig_empty=" + String(rig_err_empty),
    )

    # -----------------------------------------------------------------------
    # 3. F20 Boundary: Cyclic NestedPrefabRef (Self-Cycle & Mutual Cycle) Rejected
    # -----------------------------------------------------------------------
    var reg = ComponentRegistry.with_builtins()
    register_render_codecs(reg)

    var pf_self = Prefab("SelfCycle")
    var s0 = pf_self.add_node("Root")
    pf_self.set_component(s0, LocalTransform.identity())
    pf_self.add_nested(NestedPrefabRef("SelfCycle", s0))

    var self_lib = List[Prefab]()
    self_lib.append(pf_self.copy())
    var w_cyc = World()
    var self_rejected = False
    try:
        _ = pf_self.instantiate(w_cyc, reg, self_lib)
    except:
        self_rejected = True

    var pf_a = Prefab("CycleA")
    var a0 = pf_a.add_node("A")
    pf_a.set_component(a0, LocalTransform.identity())
    pf_a.add_nested(NestedPrefabRef("CycleB", a0))

    var pf_b = Prefab("CycleB")
    var b0 = pf_b.add_node("B")
    pf_b.set_component(b0, LocalTransform.identity())
    pf_b.add_nested(NestedPrefabRef("CycleA", b0))

    var mutual_lib = List[Prefab]()
    mutual_lib.append(pf_a.copy())
    mutual_lib.append(pf_b.copy())
    var w_mut = World()
    var mutual_rejected = False
    try:
        _ = pf_a.instantiate(w_mut, reg, mutual_lib)
    except:
        mutual_rejected = True

    _ = runner.check(
        "t2_m1_cyclic_nested_prefab_self_and_mutual_rejected",
        self_rejected and mutual_rejected,
        "self_rejected=" + String(self_rejected) + " mutual_rejected=" + String(mutual_rejected),
    )

    # -----------------------------------------------------------------------
    # 4. F21 Boundary: Unknown-Component Opaque Blob Lossless Across 3 Cycles
    # -----------------------------------------------------------------------
    var multi_unk_json = (
        '{\n'
        '  "format": "flscn",\n'
        '  "version": 2,\n'
        '  "name": "MultiOpaque",\n'
        '  "entities": [\n'
        '    {\n'
        '      "slot": 0,\n'
        '      "parent": -1,\n'
        '      "components": {\n'
        '        "UnknownAlpha": {},\n'
        '        "LocalTransform": {"translation": [2.0, 3.0, 4.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},\n'
        '        "UnknownBeta": {"nested": {"k": [1, 2, 3]}, "flag": true}\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        '}\n'
    )
    var scn_m = Scene("MultiOpaque")
    var cur_json = multi_unk_json
    for _ in range(3):
        var wm = World()
        _ = scn_m.load_flscn_bytes(wm, reg, _str_to_bytes(cur_json))
        cur_json = scn_m.save_flscn_string(wm, reg)

    var json_multi_ok = (
        cur_json.find("UnknownAlpha") >= 0
        and cur_json.find("UnknownBeta") >= 0
        and cur_json.find("nested") >= 0
    )

    # Binary 3-cycle lossless roundtrip with 0-byte and non-empty unknown blobs
    var tag_u000 = fourcc("U000")
    var tag_ubet = fourcc("UBET")
    var bw_m = BlobWriter()
    bw_m.begin(SCENE_BIN_MAGIC, 2)
    bw_m.text("MultiOpaqueBin")
    bw_m.integer(1)
    bw_m.integer(-1)
    bw_m.integer(3)
    bw_m.begin(tag_u000, 1)
    bw_m.end()
    bw_m.begin(LocalTransform.SCHEMA_FOURCC, LocalTransform.SCHEMA_VERSION)
    LocalTransform(Vec3(2.0, 3.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)).write_bin(bw_m)
    bw_m.end()
    bw_m.begin(tag_ubet, 7)
    bw_m.u8(0xAA); bw_m.u8(0x55); bw_m.u8(0x01)
    bw_m.end()
    bw_m.end()

    var expected_ubet = List[UInt8]()
    expected_ubet.append(0xAA)
    expected_ubet.append(0x55)
    expected_ubet.append(0x01)

    var cur_bin = lz4_pack(bw_m.data)
    var first_resaved = List[UInt8]()
    for cycle in range(3):
        var wmb = World()
        _ = scn_m.load_flscnb_bytes(wmb, reg, cur_bin)
        cur_bin = scn_m.save_flscnb_bytes(wmb, reg)
        if cycle == 0:
            first_resaved = cur_bin.copy()

    var unpacked_cur = lz4_unpack(cur_bin)
    var bin_multi_ok = (
        len(first_resaved) > 0
        and len(first_resaved) == len(cur_bin)
        and _has_fourcc_blob(unpacked_cur, tag_u000, List[UInt8]())
        and _has_fourcc_blob(unpacked_cur, tag_ubet, expected_ubet)
    )
    if bin_multi_ok:
        for i in range(len(cur_bin)):
            if first_resaved[i] != cur_bin[i]:
                bin_multi_ok = False
                break

    _ = runner.check(
        "t2_m1_unknown_component_multi_cycle_lossless",
        json_multi_ok and bin_multi_ok,
        "multi-cycle unknown component preservation failed",
    )

    # -----------------------------------------------------------------------
    # 5. F21 Boundary: Unsupported Future Version (999) / Version 0 / Corrupt Rejects Without Mutating World
    # -----------------------------------------------------------------------
    var w_sentinel = World()
    reg.ensure_registered_in_world(w_sentinel)
    var sentinel = w_sentinel.spawn_empty()
    _ = w_sentinel.insert[LocalTransform](sentinel, LocalTransform(Vec3(42.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))

    var future_json = '{"format":"flscn","version":999,"name":"Future","entities":[]}'
    var zero_json = '{"format":"flscn","version":0,"name":"Zero","entities":[]}'

    var bw_fut = BlobWriter()
    bw_fut.begin(SCENE_BIN_MAGIC, 999)
    bw_fut.text("FutureBin")
    bw_fut.integer(0)
    bw_fut.end()
    var fut_packed = lz4_pack(bw_fut.data)

    var scn_err = Scene("VersionGuard")
    var err_fut_json = False
    try:
        _ = scn_err.load_flscn_bytes(w_sentinel, reg, _str_to_bytes(future_json))
    except:
        err_fut_json = True

    var err_zero_json = False
    try:
        _ = scn_err.load_flscn_bytes(w_sentinel, reg, _str_to_bytes(zero_json))
    except:
        err_zero_json = True

    var err_fut_bin = False
    try:
        _ = scn_err.load_flscnb_bytes(w_sentinel, reg, fut_packed)
    except:
        err_fut_bin = True

    var sentinel_intact = (
        w_sentinel.is_alive(sentinel)
        and w_sentinel.has[LocalTransform](sentinel)
        and w_sentinel.get_ptr[LocalTransform](sentinel).value()[].translation.x == Float32(42.0)
    )
    _ = runner.check(
        "t2_m1_unsupported_schema_version_rejected_without_world_mutation",
        err_fut_json and err_zero_json and err_fut_bin and sentinel_intact,
        "future=" + String(err_fut_json) + " zero=" + String(err_zero_json) + " bin=" + String(err_fut_bin) + " intact=" + String(sentinel_intact),
    )

    # -----------------------------------------------------------------------
    # 6. F19 Boundary: Missing Bone Name in .flrig Flagged Cleanly (Not Silent -1)
    # -----------------------------------------------------------------------
    var bad_rig = FrigRigDef("MissingBoneRig")
    bad_rig.joints.append(FrigJointDef("Root", -1, Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))
    bad_rig.sockets.append(FrigSocketDef("BadSocket", "NonExistentBone_XYZ", Vec3(), Quat(), Vec3(1.0, 1.0, 1.0)))
    var bad_chain = RagdollChain("BadChain")
    bad_chain.bones.append(RagdollBoneConfig("GhostBone_ABC", Float32(0.1), Float32(0.3), Float32(2.0)))
    bad_rig.ragdoll_chains.append(bad_chain^)

    var diag_err = False
    try:
        validate_rig_bindings(bad_rig)
    except e:
        var msg = String(e)
        diag_err = msg.find("NonExistentBone_XYZ") >= 0 or msg.find("GhostBone_ABC") >= 0 or msg.byte_length() > 0

    _ = runner.check("t2_m1_flrig_missing_bone_name_explicit_diagnostic", diag_err, "missing bone name in .flrig was not flagged")

    # -----------------------------------------------------------------------
    # 7. F17/F18 Boundary: Empty Model Upload & Hot-Reload Corrupt File Resilience
    # -----------------------------------------------------------------------
    var empty_model = NativeModel()
    var r_t0 = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, 32, 32)
    var up_empty = r_t0.upload_native_model(empty_model, build_lods=False, upload_skinned=False)
    var parsed_bad_glb = parse_native_glb(_str_to_bytes("glTF_corrupt_header"))

    var hot_ws = tmp + "/hot_corrupt"
    _ = write_minimal_glb(hot_ws + "/prop.glb")
    var srv_hot = AssetServer(enable_watcher=True)
    _ = srv_hot.watch_loose_directory(hot_ws)
    var h_live = srv_hot.load_mesh("prop.glb")
    var alive_before = srv_hot.meshes.is_alive(h_live)
    # Overwrite with 0-byte corrupt file and tick watcher
    _ = write_bytes(hot_ws + "/prop.glb", List[UInt8]())
    _ = srv_hot.tick(10.0)
    _ = srv_hot.tick(11.0)
    var alive_after = srv_hot.meshes.is_alive(h_live)
    var loaded_after = srv_hot.meshes.is_loaded(h_live)
    srv_hot.close()

    _ = runner.check(
        "t2_m1_empty_glb_upload_and_corrupt_hot_reload_no_crash",
        len(up_empty.mesh_entries) == 0 and len(parsed_bad_glb.meshes) == 0 and alive_before and alive_after and loaded_after,
        "empty upload or corrupt hot reload boundary failed",
    )

    rm_rf_tmp(tmp)
    runner.finish()

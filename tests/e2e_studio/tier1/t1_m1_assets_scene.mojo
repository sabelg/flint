# AREA: m1
"""Tier 1 Opaque API Tests: Area C — M1 AssetRef, Dynamic Upload, AssetPlugin & Scene v2 (F16, F17, F18, F21)."""

from std.collections import List
from flint.app.engine_app import AppConfig, EngineApp
from flint.assets.asset_plugin import AssetPlugin, SceneAssetResolver
from flint.assets.asset_ref import AssetRef
from flint.assets.flpk_v3 import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    AssetMeta,
    AssetUuid,
    write_flmeta,
)
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_writer import write_native_glb
from flint.assets.image import decode_png
from flint.assets.jpeg import decode_image_auto
from flint.assets.json import parse_json
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.model import NativeMesh, NativeModel
from flint.assets.server import ASSET_STATE_READY, AssetServer
from flint.anim.plugin import AnimPlugin, AnimatorComponent, RigComponent, register_anim_codecs
from flint.audio.plugin import AudioEmitterComponent, AudioPlugin, register_audio_codecs
from flint.core.lz4 import lz4_pack, lz4_unpack
from flint.core.math import Mat4, Vec3
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.core.transform import Quat
from flint.ecs.world import World
from flint.platform.audio import decode_wav
from flint.platform.os import get_time, make_dirs
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.plugin import (
    MaterialOverrideComponent,
    MeshRendererComponent,
    RenderPlugin,
    register_render_codecs,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    RenderCameraPacket,
    RenderInstancePacket,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene import (
    SCENE_BIN_MAGIC,
    SCENE_SCHEMA_VERSION,
    ComponentRegistry,
    LocalTransform,
    Scene,
    TransformPlugin,
)
from flint.ui.context import UiDrawList
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    read_bytes,
    read_text,
    rm_rf_tmp,
    write_bytes,
    write_minimal_glb,
    write_minimal_png,
    write_minimal_wav,
    write_text,
)


def _str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _write_modified_glb(path: String) -> Bool:
    var mb = MeshBuilder(
        base_color=0xFF40C0FF,
        roughness=0.25,
        metallic=0.8,
    )
    mb.add_beveled_box(Vec3(-0.9, -0.9, -0.9), Vec3(0.9, 0.9, 0.9), 0.1)
    var mesh = mb^.finish()
    var prims = List[NativeMesh]()
    prims.append(mesh^)
    var model = NativeModel()
    _ = append_node_meshes(model, "studio_test_box_modified", prims)
    return write_native_glb(path, model)


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
    var tmp = make_tmp_dir("t1_m1_assets_scene")
    var runner = StudioSuiteRunner("t1_m1_assets_scene")

    # -----------------------------------------------------------------------
    # 1. F16: AssetRef JSON/Binary Round-Trip & AssetServer UUID-First + Fallback
    # -----------------------------------------------------------------------
    var u_crate = AssetUuid.from_path("assets/meshes/crate_canonical.glb")
    var ref_orig = AssetRef(u_crate, "assets/meshes/crate_canonical.glb", ASSET_KIND_MESH)

    var bw = BlobWriter()
    bw.begin(fourcc("AREF"), 1)
    ref_orig.write_bin(bw)
    bw.end()
    var br = BlobReader(bw.data.copy())
    var aref_ver = br.enter(fourcc("AREF"), 1)
    var ref_bin = AssetRef.read_bin(br, aref_ver)
    br.leave()

    var ref_json_bytes = _str_to_bytes(ref_orig.write_json())
    var ref_doc = parse_json(ref_json_bytes)
    var ref_json = AssetRef.read_json(ref_doc, ref_doc.root)

    var codec_ok = (
        ref_bin.uuid == u_crate
        and ref_bin.rel_path == "assets/meshes/crate_canonical.glb"
        and ref_bin.kind == ASSET_KIND_MESH
        and ref_json.uuid == u_crate
        and ref_json.rel_path == "assets/meshes/crate_canonical.glb"
        and ref_json.kind == ASSET_KIND_MESH
    )
    _ = runner.check("t1_m1_asset_ref_json_and_binary_roundtrip", codec_ok, "AssetRef codec roundtrip mismatch")

    var ws_res = tmp + "/ws_resolve"
    _ = make_dirs(ws_res + "/a")
    _ = make_dirs(ws_res + "/b")
    _ = write_minimal_glb(ws_res + "/a/moved_crate.glb")
    _ = write_minimal_glb(ws_res + "/b/fallback_crate.glb")
    var meta_a = AssetMeta()
    meta_a.uuid = AssetUuid.from_path("custom/moved_crate_uuid")
    meta_a.asset_kind = ASSET_KIND_MESH
    meta_a.source_path = "a/moved_crate.glb"
    _ = write_flmeta(ws_res + "/a/moved_crate.glb.flmeta", meta_a)

    var srv = AssetServer(enable_watcher=False)
    _ = srv.watch_loose_directory(ws_res)

    var ref_uuid_first = AssetRef(meta_a.uuid, "b/fallback_crate.glb", ASSET_KIND_MESH)
    var h_uuid = srv.resolve_mesh(ref_uuid_first)
    var ref_fallback = AssetRef(AssetUuid(), "b/fallback_crate.glb", ASSET_KIND_MESH)
    var h_fb = srv.resolve_mesh(ref_fallback)

    var res_ok = (
        srv.meshes.is_alive(h_uuid)
        and srv.meshes.is_loaded(h_uuid)
        and srv.meshes.is_alive(h_fb)
        and srv.meshes.is_loaded(h_fb)
        and h_uuid.slot() != h_fb.slot()
    )
    srv.close()
    _ = runner.check("t1_m1_asset_server_uuid_first_then_rel_path_fallback", res_ok, "UUID-first or rel_path fallback resolution failed")

    # -----------------------------------------------------------------------
    # 2. F17: Dynamic GLB/Texture/Audio Import + ScalableRenderer.upload_native_model
    # -----------------------------------------------------------------------
    var imp_dir = tmp + "/import_assets"
    var glb_ok = write_minimal_glb(imp_dir + "/prop.glb")
    var png_ok = write_minimal_png(imp_dir + "/albedo.png", 16, 16, 0xFFCC8844)
    var wav_ok = write_minimal_wav(imp_dir + "/sfx.wav", 0.04, 48000, 2)

    var model = load_native_glb(imp_dir + "/prop.glb")
    var img = decode_image_auto(read_bytes(imp_dir + "/albedo.png"))
    var png_dec = decode_png(read_bytes(imp_dir + "/albedo.png"))
    var wav = decode_wav(read_bytes(imp_dir + "/sfx.wav"))

    var r_t0 = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, 64, 64)
    var up_t0 = r_t0.upload_native_model(model, build_lods=False, upload_skinned=False)

    var model_t2 = load_native_glb(imp_dir + "/prop.glb")
    var r_t2 = ScalableRenderer(RENDER_TIER_2_AAA_VISBUFFER_RT, 64, 64)
    var up_t2 = r_t2.upload_native_model(model_t2, build_lods=False, upload_skinned=False)

    var ui = UiDrawList()
    var rw_bg = RenderWorld()
    rw_bg.camera = RenderCameraPacket(Vec3(0.0, 0.5, 3.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 55.0, 0.1, 100.0, 1.0)
    _ = r_t0.render_world(rw_bg, ui, capture_readback=True, force_readback=True)
    var px_bg = r_t0.readback_pixels()

    var rw = RenderWorld()
    rw.camera = RenderCameraPacket(Vec3(0.0, 0.5, 3.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 55.0, 0.1, 100.0, 1.0)
    rw.cluster_lights.append(AaaClusterLight.point(Vec3(1.5, 2.0, 2.0), Vec3(5.0, 5.0, 5.0), 10.0))
    if len(up_t0.mesh_entries) > 0 and len(up_t0.material_ids) > 0:
        rw.instances.append(
            RenderInstancePacket(
                UInt64(1),
                up_t0.mesh_entries[0].mesh_id,
                Mat4(),
                Mat4(),
                up_t0.material_ids[0],
            )
        )
    var drew = r_t0.render_world(rw, ui, capture_readback=True, force_readback=True)
    var px = r_t0.readback_pixels()
    var non_bg_px = 0
    if len(px) == len(px_bg):
        for pi in range(len(px)):
            if px[pi] != px_bg[pi]:
                non_bg_px += 1

    var f17_ok = (
        glb_ok and png_ok and wav_ok
        and len(model.meshes) >= 1
        and img.width == 16 and img.height == 16
        and png_dec.width == 16
        and wav.sample_rate == 48000 and len(wav.samples) > 0
        and len(up_t0.mesh_entries) == len(model.meshes)
        and len(up_t0.material_ids) >= 1
        and len(up_t2.mesh_entries) == len(model_t2.meshes)
        and drew and len(px) == 64 * 64 and non_bg_px >= 16
    )
    _ = runner.check("t1_m1_dynamic_import_and_upload_native_model_t0_t2", f17_ok, "GLB/PNG/WAV import or upload_native_model failed")

    # -----------------------------------------------------------------------
    # 3. F18: AssetPlugin + SceneAssetResolver Hot-Reload Wiring in EngineApp
    # -----------------------------------------------------------------------
    var hot_ws = tmp + "/hot_ws"
    _ = write_minimal_glb(hot_ws + "/assets/prop.glb")

    var app = EngineApp(AppConfig(title="M1AssetPluginTest", headless=True, worker_threads=0, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AssetPlugin(loose_root=hot_ws, enable_watcher=True))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))

    var prop_ref = AssetRef(AssetUuid.from_path("assets/prop.glb"), "assets/prop.glb", ASSET_KIND_MESH)
    var e_prop = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](e_prop, LocalTransform(Vec3(0.0, 0.0, -3.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    var mr_prop = MeshRendererComponent()
    mr_prop.mesh_ref = prop_ref.copy()
    _ = app.world.insert[MeshRendererComponent](e_prop, mr_prop^)

    app.startup()
    app.step_one_frame(Float32(1.0 / 60.0))

    var has_srv = app.resources.has[AssetServer]()
    var has_resolver = app.resources.has[SceneAssetResolver]()
    var mr_resolved_ok = (
        app.world.has[MeshRendererComponent](e_prop)
        and app.world.get_ptr[MeshRendererComponent](e_prop).value()[].mesh_id >= 0
    )
    var resolved_ok = False
    var epoch_before: UInt64 = 0
    var reloads_before: Int = 0
    if has_srv:
        var asrv_ptr = app.resources.get_ptr_mut[AssetServer]().value()
        var h_chk = asrv_ptr[unsafe_offset=0].resolve_mesh(prop_ref)
        resolved_ok = asrv_ptr[unsafe_offset=0].meshes.is_alive(h_chk) and asrv_ptr[unsafe_offset=0].meshes.is_loaded(h_chk)
        epoch_before = asrv_ptr[unsafe_offset=0].meshes.reload_epoch(h_chk)
        reloads_before = asrv_ptr[unsafe_offset=0].hot_reloads_applied

    # Modify asset on disk with genuinely different geometry and advance EngineApp + watcher clock
    var mod_ok = _write_modified_glb(hot_ws + "/assets/prop.glb")
    app.step_one_frame(Float32(1.0 / 60.0))
    if has_srv:
        var asrv2_ptr = app.resources.get_ptr_mut[AssetServer]().value()
        var t_now = get_time()
        _ = asrv2_ptr[unsafe_offset=0].tick(t_now + 1.0)
        _ = asrv2_ptr[unsafe_offset=0].tick(t_now + 2.0)
    app.step_one_frame(Float32(1.0 / 60.0))

    var hot_ok = False
    var mr_after_ok = (
        app.world.has[MeshRendererComponent](e_prop)
        and app.world.get_ptr[MeshRendererComponent](e_prop).value()[].mesh_id >= 0
    )
    if has_srv and resolved_ok and has_resolver:
        var asrv3_ptr = app.resources.get_ptr_mut[AssetServer]().value()
        var h_after = asrv3_ptr[unsafe_offset=0].resolve_mesh(prop_ref)
        var epoch_after = asrv3_ptr[unsafe_offset=0].meshes.reload_epoch(h_after)
        hot_ok = (asrv3_ptr[unsafe_offset=0].hot_reloads_applied > reloads_before) or (epoch_after > epoch_before)
    app.shutdown()
    _ = runner.check(
        "t1_m1_asset_plugin_resolver_and_hot_reload_wiring",
        has_srv and has_resolver and mr_resolved_ok and resolved_ok and mod_ok and mr_after_ok and hot_ok,
        "AssetPlugin/SceneAssetResolver hot-reload failed",
    )

    # -----------------------------------------------------------------------
    # 4. F16 + F21: SCENE_SCHEMA_VERSION == 2 & Migrated Component Round-Trip
    # -----------------------------------------------------------------------
    var schema_ok = SCENE_SCHEMA_VERSION == 2
    _ = runner.check("t1_m1_scene_schema_version_is_2", schema_ok, "expected SCENE_SCHEMA_VERSION == 2, got " + String(SCENE_SCHEMA_VERSION))

    var w_save = World()
    var reg = ComponentRegistry.with_builtins()
    register_render_codecs(reg)
    register_audio_codecs(reg)
    register_anim_codecs(reg)
    reg.ensure_registered_in_world(w_save)

    var e_hero = w_save.spawn_empty()
    _ = w_save.insert[LocalTransform](e_hero, LocalTransform(Vec3(1.0, 2.0, 3.0), Quat(), Vec3(1.0, 1.0, 1.0)))

    var u_hero_mesh = AssetUuid.from_path("assets/meshes/hero.glb")
    var u_hero_mat = AssetUuid.from_path("assets/materials/hero.flmat")
    var u_ov_mat = AssetUuid.from_path("assets/materials/override.flmat")
    var u_step_aud = AssetUuid.from_path("assets/audio/step.wav")
    var u_hero_anim = AssetUuid.from_path("assets/anims/hero.flanim")
    var u_hero_rig = AssetUuid.from_path("assets/rigs/hero.flrig")

    var mesh_c = MeshRendererComponent()
    mesh_c.mesh_ref = AssetRef(u_hero_mesh, "assets/meshes/hero.glb", ASSET_KIND_MESH)
    mesh_c.material_ref = AssetRef(u_hero_mat, "assets/materials/hero.flmat", ASSET_KIND_RAW)
    _ = w_save.insert[MeshRendererComponent](e_hero, mesh_c^)

    var mat_c = MaterialOverrideComponent()
    mat_c.material_ref = AssetRef(u_ov_mat, "assets/materials/override.flmat", ASSET_KIND_RAW)
    mat_c.base_color = Vec3(0.9, 0.25, 0.1)
    mat_c.enabled = True
    _ = w_save.insert[MaterialOverrideComponent](e_hero, mat_c^)

    var aud_c = AudioEmitterComponent()
    aud_c.audio_ref = AssetRef(u_step_aud, "assets/audio/step.wav", ASSET_KIND_AUDIO)
    aud_c.volume = Float32(0.75)
    _ = w_save.insert[AudioEmitterComponent](e_hero, aud_c^)

    var anim_c = AnimatorComponent()
    anim_c.graph_ref = AssetRef(u_hero_anim, "assets/anims/hero.flanim", ASSET_KIND_RAW)
    _ = w_save.insert[AnimatorComponent](e_hero, anim_c^)

    var rig_c = RigComponent()
    rig_c.rig_ref = AssetRef(u_hero_rig, "assets/rigs/hero.flrig", ASSET_KIND_RAW)
    _ = w_save.insert[RigComponent](e_hero, rig_c^)

    var scn = Scene("M1ContentScene")
    var flscn_path = tmp + "/scene_v2.flscn"
    var flscnb_path = tmp + "/scene_v2.flscnb"
    var saved_json = scn.save_flscn_file(flscn_path, w_save, reg)
    var saved_bin = scn.save_flscnb_file(flscnb_path, w_save, reg)

    var w_json = World()
    var w_bin = World()
    var ents_json = scn.load_flscn_file(flscn_path, w_json, reg)
    var ents_bin = scn.load_flscnb_file(flscnb_path, w_bin, reg)

    var rt_ok = saved_json and saved_bin and len(ents_json) == 1 and len(ents_bin) == 1
    if rt_ok:
        var ej = ents_json[0]
        var eb = ents_bin[0]
        rt_ok = (
            w_json.has[MeshRendererComponent](ej)
            and w_json.has[MaterialOverrideComponent](ej)
            and w_json.has[AudioEmitterComponent](ej)
            and w_json.has[AnimatorComponent](ej)
            and w_json.has[RigComponent](ej)
            and w_bin.has[MeshRendererComponent](eb)
            and w_bin.has[MaterialOverrideComponent](eb)
            and w_bin.has[AudioEmitterComponent](eb)
            and w_bin.has[AnimatorComponent](eb)
            and w_bin.has[RigComponent](eb)
            and w_json.get_ptr[MeshRendererComponent](ej).value()[].mesh_ref.uuid == u_hero_mesh
            and w_json.get_ptr[MeshRendererComponent](ej).value()[].mesh_ref.rel_path == "assets/meshes/hero.glb"
            and w_json.get_ptr[MeshRendererComponent](ej).value()[].mesh_ref.kind == ASSET_KIND_MESH
            and w_json.get_ptr[MeshRendererComponent](ej).value()[].material_ref.uuid == u_hero_mat
            and w_json.get_ptr[MeshRendererComponent](ej).value()[].material_ref.rel_path == "assets/materials/hero.flmat"
            and w_json.get_ptr[MeshRendererComponent](ej).value()[].material_ref.kind == ASSET_KIND_RAW
            and w_bin.get_ptr[MeshRendererComponent](eb).value()[].mesh_ref.uuid == u_hero_mesh
            and w_bin.get_ptr[MeshRendererComponent](eb).value()[].mesh_ref.rel_path == "assets/meshes/hero.glb"
            and w_bin.get_ptr[MeshRendererComponent](eb).value()[].mesh_ref.kind == ASSET_KIND_MESH
            and w_bin.get_ptr[MeshRendererComponent](eb).value()[].material_ref.uuid == u_hero_mat
            and w_bin.get_ptr[MeshRendererComponent](eb).value()[].material_ref.rel_path == "assets/materials/hero.flmat"
            and w_bin.get_ptr[MeshRendererComponent](eb).value()[].material_ref.kind == ASSET_KIND_RAW
            and w_json.get_ptr[MaterialOverrideComponent](ej).value()[].material_ref.uuid == u_ov_mat
            and w_json.get_ptr[MaterialOverrideComponent](ej).value()[].material_ref.rel_path == "assets/materials/override.flmat"
            and w_json.get_ptr[MaterialOverrideComponent](ej).value()[].material_ref.kind == ASSET_KIND_RAW
            and w_bin.get_ptr[MaterialOverrideComponent](eb).value()[].material_ref.uuid == u_ov_mat
            and w_bin.get_ptr[MaterialOverrideComponent](eb).value()[].material_ref.rel_path == "assets/materials/override.flmat"
            and w_bin.get_ptr[MaterialOverrideComponent](eb).value()[].material_ref.kind == ASSET_KIND_RAW
            and w_json.get_ptr[AudioEmitterComponent](ej).value()[].audio_ref.uuid == u_step_aud
            and w_json.get_ptr[AudioEmitterComponent](ej).value()[].audio_ref.rel_path == "assets/audio/step.wav"
            and w_json.get_ptr[AudioEmitterComponent](ej).value()[].audio_ref.kind == ASSET_KIND_AUDIO
            and w_bin.get_ptr[AudioEmitterComponent](eb).value()[].audio_ref.uuid == u_step_aud
            and w_bin.get_ptr[AudioEmitterComponent](eb).value()[].audio_ref.rel_path == "assets/audio/step.wav"
            and w_bin.get_ptr[AudioEmitterComponent](eb).value()[].audio_ref.kind == ASSET_KIND_AUDIO
            and w_json.get_ptr[AnimatorComponent](ej).value()[].graph_ref.uuid == u_hero_anim
            and w_json.get_ptr[AnimatorComponent](ej).value()[].graph_ref.rel_path == "assets/anims/hero.flanim"
            and w_json.get_ptr[AnimatorComponent](ej).value()[].graph_ref.kind == ASSET_KIND_RAW
            and w_bin.get_ptr[AnimatorComponent](eb).value()[].graph_ref.uuid == u_hero_anim
            and w_bin.get_ptr[AnimatorComponent](eb).value()[].graph_ref.rel_path == "assets/anims/hero.flanim"
            and w_bin.get_ptr[AnimatorComponent](eb).value()[].graph_ref.kind == ASSET_KIND_RAW
            and w_json.get_ptr[RigComponent](ej).value()[].rig_ref.uuid == u_hero_rig
            and w_json.get_ptr[RigComponent](ej).value()[].rig_ref.rel_path == "assets/rigs/hero.flrig"
            and w_json.get_ptr[RigComponent](ej).value()[].rig_ref.kind == ASSET_KIND_RAW
            and w_bin.get_ptr[RigComponent](eb).value()[].rig_ref.uuid == u_hero_rig
            and w_bin.get_ptr[RigComponent](eb).value()[].rig_ref.rel_path == "assets/rigs/hero.flrig"
            and w_bin.get_ptr[RigComponent](eb).value()[].rig_ref.kind == ASSET_KIND_RAW
        )
    _ = runner.check("t1_m1_scene_v2_json_and_binary_asset_ref_roundtrip", rt_ok, "Scene v2 .flscn/.flscnb AssetRef roundtrip failed")

    # -----------------------------------------------------------------------
    # 5. F21: Scene v1 Backward Compatibility Load & Migration (.flscn & .flscnb)
    # -----------------------------------------------------------------------
    var v1_json = (
        '{\n'
        '  "format": "flscn",\n'
        '  "version": 1,\n'
        '  "name": "LegacyV1",\n'
        '  "entities": [\n'
        '    {\n'
        '      "slot": 0,\n'
        '      "parent": -1,\n'
        '      "components": {\n'
        '        "LocalTransform": {"translation": [4.0, 5.0, 6.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},\n'
        '        "MeshRendererComponent": {"mesh_id": 2, "material_id": 1, "cast_shadows": true, "visible": true},\n'
        '        "AudioEmitterComponent": {"asset_id": -1, "asset_path": "assets/audio/legacy.wav", "volume": 0.6}\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        '}\n'
    )
    var w_v1 = World()
    var scn_v1 = Scene("V1Load")
    var ents_v1 = scn_v1.load_flscn_bytes(w_v1, reg, _str_to_bytes(v1_json))
    var v1_ok = len(ents_v1) == 1
    if v1_ok:
        var ev1 = ents_v1[0]
        v1_ok = (
            w_v1.has[LocalTransform](ev1)
            and w_v1.has[MeshRendererComponent](ev1)
            and w_v1.has[AudioEmitterComponent](ev1)
            and w_v1.get_ptr[LocalTransform](ev1).value()[].translation.x == Float32(4.0)
            and w_v1.get_ptr[MeshRendererComponent](ev1).value()[].mesh_id == 2
            and w_v1.get_ptr[AudioEmitterComponent](ev1).value()[].audio_ref.rel_path == "assets/audio/legacy.wav"
            and abs(w_v1.get_ptr[AudioEmitterComponent](ev1).value()[].volume - Float32(0.6)) < Float32(0.001)
        )

    # Binary v1 .flscnb read_bin(..., version=1) migration check
    var bw_v1 = BlobWriter()
    bw_v1.begin(SCENE_BIN_MAGIC, 1)
    bw_v1.text("LegacyV1Bin")
    bw_v1.integer(1)
    bw_v1.integer(-1)
    bw_v1.integer(3)
    bw_v1.begin(LocalTransform.SCHEMA_FOURCC, 1)
    LocalTransform(Vec3(4.0, 5.0, 6.0), Quat(), Vec3(1.0, 1.0, 1.0)).write_bin(bw_v1)
    bw_v1.end()
    bw_v1.begin(MeshRendererComponent.SCHEMA_FOURCC, 1)
    bw_v1.integer(2); bw_v1.integer(1); bw_v1.boolean(True); bw_v1.boolean(True)
    bw_v1.end()
    bw_v1.begin(AudioEmitterComponent.SCHEMA_FOURCC, 1)
    bw_v1.integer(-1)
    bw_v1.text("assets/audio/legacy_bin.wav")
    bw_v1.f32(Float32(0.55))
    bw_v1.f32(Float32(1.0))
    bw_v1.integer(1)
    bw_v1.boolean(False)
    bw_v1.boolean(True)
    bw_v1.integer(1)
    bw_v1.f32(Float32(15.0))
    bw_v1.f32(Float32(350.0))
    bw_v1.boolean(False)
    bw_v1.boolean(False)
    bw_v1.boolean(False)
    bw_v1.boolean(False)
    bw_v1.f32(Float32(1.0))
    bw_v1.f32(Float32(20000.0))
    bw_v1.f32(Float32(0.0))
    bw_v1.vec3(Vec3())
    bw_v1.boolean(False)
    bw_v1.end()
    bw_v1.end()

    var w_v1_bin = World()
    var scn_v1_bin = Scene("V1BinLoad")
    var ents_v1_bin = scn_v1_bin.load_flscnb_bytes(w_v1_bin, reg, lz4_pack(bw_v1.data))
    var v1_bin_ok = len(ents_v1_bin) == 1
    if v1_bin_ok:
        var ev1b = ents_v1_bin[0]
        v1_bin_ok = (
            w_v1_bin.has[LocalTransform](ev1b)
            and w_v1_bin.has[MeshRendererComponent](ev1b)
            and w_v1_bin.has[AudioEmitterComponent](ev1b)
            and w_v1_bin.get_ptr[LocalTransform](ev1b).value()[].translation.x == Float32(4.0)
            and w_v1_bin.get_ptr[MeshRendererComponent](ev1b).value()[].mesh_id == 2
            and w_v1_bin.get_ptr[AudioEmitterComponent](ev1b).value()[].audio_ref.rel_path == "assets/audio/legacy_bin.wav"
            and abs(w_v1_bin.get_ptr[AudioEmitterComponent](ev1b).value()[].volume - Float32(0.55)) < Float32(0.001)
        )
    _ = runner.check("t1_m1_scene_v1_backward_compat_load_and_migration", v1_ok and v1_bin_ok, "Scene v1 backward compatibility load failed")

    # -----------------------------------------------------------------------
    # 6. F21: Unknown-Component Opaque Blob Preservation (.flscn & .flscnb)
    # -----------------------------------------------------------------------
    var unk_json = (
        '{\n'
        '  "format": "flscn",\n'
        '  "version": 2,\n'
        '  "name": "OpaqueBlobScene",\n'
        '  "entities": [\n'
        '    {\n'
        '      "slot": 0,\n'
        '      "parent": -1,\n'
        '      "components": {\n'
        '        "LocalTransform": {"translation": [1.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [1.0, 1.0, 1.0]},\n'
        '        "ThirdPartyPluginComponent": {"power": 9001, "tags": ["boss", "shielded"]}\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        '}\n'
    )
    var w_unk = World()
    var scn_unk = Scene("OpaqueBlobScene")
    var ents_unk = scn_unk.load_flscn_bytes(w_unk, reg, _str_to_bytes(unk_json))
    var re_saved_json = scn_unk.save_flscn_string(w_unk, reg)
    var unk_json_ok = (
        len(ents_unk) == 1
        and re_saved_json.find("ThirdPartyPluginComponent") >= 0
        and re_saved_json.find("9001") >= 0
        and re_saved_json.find("shielded") >= 0
    )

    # Binary unknown-component FourCC blob preservation
    var bw_unk = BlobWriter()
    bw_unk.begin(SCENE_BIN_MAGIC, 2)
    bw_unk.text("OpaqueBinScene")
    bw_unk.integer(1)  # entity count
    bw_unk.integer(-1) # parent idx
    bw_unk.integer(2)  # component count
    bw_unk.begin(LocalTransform.SCHEMA_FOURCC, LocalTransform.SCHEMA_VERSION)
    LocalTransform(Vec3(7.0, 8.0, 9.0), Quat(), Vec3(1.0, 1.0, 1.0)).write_bin(bw_unk)
    bw_unk.end()
    var unk_fourcc = fourcc("XUNK")
    bw_unk.begin(unk_fourcc, 1)
    bw_unk.u8(0xDE); bw_unk.u8(0xAD); bw_unk.u8(0xBE); bw_unk.u8(0xEF); bw_unk.u8(0x42)
    bw_unk.end()
    bw_unk.end()

    var expected_unk_payload = List[UInt8]()
    expected_unk_payload.append(0xDE)
    expected_unk_payload.append(0xAD)
    expected_unk_payload.append(0xBE)
    expected_unk_payload.append(0xEF)
    expected_unk_payload.append(0x42)

    var w_unk_bin = World()
    var scn_unk_bin = Scene("OpaqueBinScene")
    var ents_unk_bin = scn_unk_bin.load_flscnb_bytes(w_unk_bin, reg, lz4_pack(bw_unk.data))
    var re_saved_bin = scn_unk_bin.save_flscnb_bytes(w_unk_bin, reg)
    var w_unk_bin2 = World()
    var ents_unk_bin2 = scn_unk_bin.load_flscnb_bytes(w_unk_bin2, reg, re_saved_bin)
    var re_saved_bin2 = scn_unk_bin.save_flscnb_bytes(w_unk_bin2, reg)
    var unpacked_bin = lz4_unpack(re_saved_bin)
    var unk_bin_ok = (
        len(ents_unk_bin) == 1
        and len(ents_unk_bin2) == 1
        and len(re_saved_bin) > 0
        and len(re_saved_bin) == len(re_saved_bin2)
        and _has_fourcc_blob(unpacked_bin, unk_fourcc, expected_unk_payload)
    )
    if unk_bin_ok:
        for i in range(len(re_saved_bin)):
            if re_saved_bin[i] != re_saved_bin2[i]:
                unk_bin_ok = False
                break

    _ = runner.check(
        "t1_m1_unknown_component_opaque_blob_preservation",
        unk_json_ok and unk_bin_ok,
        "unknown component opaque blob lost across .flscn/.flscnb roundtrip",
    )

    rm_rf_tmp(tmp)
    runner.finish()

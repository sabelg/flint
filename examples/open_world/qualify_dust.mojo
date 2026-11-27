"""Verify pack reopening, one-shot foot markers and authored world-space dust at native resolutions."""
from bridge_vfx import BridgeDust, load_bridge_dust, extract_bridge_dust
from flint.anim.plugin import AnimAssetCache, AnimEventsResource, AnimEventRecord, SkeletonComponent
from flint.anim.sync import FiredEvent
from flint.app import AppConfig, EngineApp
from flint.assets import ASSET_KIND_RAW, AssetPlugin, AssetServer
from flint.assets.cooker_v2 import cook_single_source_bytes
from flint.assets.flpk_v3 import AssetMeta, AssetUuid, write_flpk_v3
from flint.assets.glb_native import parse_native_glb
from flint.core.math import Vec3
from flint.platform.os import read_bytes, remove_file
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.particle_effect import load_particle_burst
from flint.render.render_world import RenderWorld, ScalableRenderer, RenderCameraPacket, RENDER_TIER_0_HEADLESS_SIMD
from flint.render.simd_canvas import Camera3D
from flint.scene import GlobalTransform, LocalTransform
from flint.ui.draw_list import UiDrawList
from std.ffi import external_call
from std.testing import assert_equal, assert_true


def check_pack(path: String, data: List[UInt8], count: Int) raises:
    var meta = AssetMeta()
    meta.asset_kind = ASSET_KIND_RAW
    meta.uuid = AssetUuid.from_path("bridge_dust.json")
    var content = cook_single_source_bytes(data, meta, "bridge_dust.json")
    assert_true(content.valid and write_flpk_v3(path, content.chunks))
    var app = EngineApp(AppConfig(headless=True, worker_threads=0))
    app.add_plugin(AssetPlugin(enable_watcher=False))
    assert_true(app.resources.get_ptr_mut[AssetServer]().value()[].mount_flpk(path))
    load_bridge_dust(app)
    var actor = app.world.spawn()
    var model = parse_native_glb(read_bytes("examples/character_course/assets/performer.glb"))
    var sk = SkeletonComponent()
    sk.model_id = 1
    sk.skeleton.reset_to_rest(model)
    var foot = model.find_node("Foot.L")
    assert_true(foot >= 0)
    var local = LocalTransform()
    local.translation = Vec3(2, 3, 4)
    var gt = GlobalTransform.from_local(local)
    var expected = gt.transform_point(sk.skeleton.world_matrices[foot].transform_point(Vec3()))
    _ = app.world.insert[GlobalTransform](actor, gt^)
    _ = app.world.insert[SkeletonComponent](actor, sk^)
    var cache = AnimAssetCache()
    cache.register_model(1, model^)
    app.insert_resource(cache^)
    app.insert_resource(AnimEventsResource())
    ref events = app.resources.get_ptr_mut[AnimEventsResource]().value()[].events
    _ = events.send(AnimEventRecord(actor, FiredEvent("footstep_L", "sole", 0, 1)))
    var frame = RenderWorld()
    extract_bridge_dust(app.world, app.resources, frame, actor, True)
    extract_bridge_dust(app.world, app.resources, frame, actor, True)
    assert_equal(len(frame.particle_emitters), 1)
    assert_equal(frame.particle_emitters[0].spawn_count, count)
    assert_true((frame.particle_emitters[0].origin - expected).length() < 1e-5)
    assert_equal(app.resources.get_ptr[BridgeDust]().value()[].bursts, 1)
    frame.clear_frame()
    assert_equal(len(frame.particle_emitters), 0)


def check_tier0(width: Int, height: Int) raises:
    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, width, height)
    var frame = RenderWorld()
    var ui = UiDrawList()
    var eye = Vec3(0, 1, 4)
    var target = Vec3(0, 1, 0)
    frame.camera = RenderCameraPacket(eye, target, fov_y_deg=40)
    var cam = Camera3D(width, height, eye, target, 40)
    assert_true(ren.render_world(frame, ui))
    var baseline = ren.readback_pixels()
    var dust = load_particle_burst(read_bytes("examples/character_course/assets/bridge_dust.json"))
    for y in [Float32(0.2), Float32(1.8)]:
        dust.origin = Vec3(0, y, 0)
        frame.particle_emitters.append(dust.copy())
    assert_true(ren.render_world(frame, ui))
    var after = ren.readback_pixels()
    for y in [Float32(0.2), Float32(1.8)]:
        var p = cam.project(Vec3(0, y, 0))
        var changed = 0
        var control = 0
        var r = height // 35
        for py in range(Int(p[1])-r, Int(p[1])+r):
            for px in range(Int(p[0])-r, Int(p[0])+r):
                var idx = py*width + px
                if baseline[idx] != after[idx]:
                    changed += 1
                if baseline[idx+width//4] != after[idx+width//4]:
                    control += 1
        print("Tier 0", width, height, "row", Int(p[1]), "changed", changed, "control", control)
        assert_true(changed > 12 and control == 0)
    frame.particle_emitters.clear()
    frame.delta_time = 0.1
    for _ in range(5):
        assert_true(ren.render_world(frame, ui))
    assert_true(not len(ren.cpu_particles.particles))
    ren.shutdown()


def check_vulkan(width: Int, height: Int) raises:
    var ren = FlintAaaVulkanRenderer(width=width, height=height, tsr_scale=1.0, max_draws=32, max_instances=32, env_cube_dim=16)
    assert_true(ren.active)
    ren.tsr_history_weight = 0
    ren.enable_auto_exposure = False
    ren.enable_froxel_fog = False
    ren.enable_bokeh_dof = False
    ren.enable_motion_blur = False
    ren.bloom_intensity = 0
    var eye = Vec3(0, 1, 4)
    var target = Vec3(0, 1, 0)
    var cam = Camera3D(width, height, eye, target, 40)
    var baseline = List[Vec3]()
    var dust = load_particle_burst(read_bytes("examples/character_course/assets/bridge_dust.json"))
    for frame in range(2):
        ren.begin_frame(eye, target, fov_y_deg=40)
        if frame == 1:
            for y in [Float32(0.2), Float32(1.8)]:
                dust.origin = Vec3(0, y, 0)
                ren.submit_particle_emitter(dust)
        assert_true(ren.render_frame(capture_readback=True))
        var hdr = ren.readback_resolved_hdr()
        if frame == 0:
            baseline = hdr^
            continue
        for y in [Float32(0.2), Float32(1.8)]:
            var projected = cam.project(Vec3(0, y, 0))
            var cx = Int(projected[0])
            var cy = Int(projected[1])
            var changed = 0
            var control = 0
            var radius = height // 35
            for py in range(cy-radius, cy+radius):
                for px in range(cx-radius, cx+radius):
                    var idx = py*width + px
                    if (hdr[idx] - baseline[idx]).length() > 0.012:
                        changed += 1
                    var ref_idx = idx + width // 4
                    if (hdr[ref_idx] - baseline[ref_idx]).length() > 0.012:
                        control += 1
            print("Vulkan native", width, height, "row", cy, "changed", changed, "control", control)
            assert_true(changed > 12 and changed > control * 3)
    ren.shutdown()


def main() raises:
    var path = "/tmp/flint-test-dust-" + String(external_call["getpid", Int32]()) + ".flpk"
    var data = read_bytes("examples/character_course/assets/bridge_dust.json")
    check_pack(path, data, 14)
    var text = String(unsafe_from_utf8=data).replace('"count": 14', '"count": 23')
    var revised = List[UInt8]()
    for byte in text.as_bytes():
        revised.append(byte)
    check_pack(path, revised, 23)
    _ = remove_file(path)
    print("[PASS] edited packed dust reopened, 14 -> 23 particles, one burst at left foot, no replay")
    for width in [1920, 3840]:
        var height = width * 9 // 16
        check_tier0(width, height)
        check_vulkan(width, height)

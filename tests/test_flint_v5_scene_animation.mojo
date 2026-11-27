"""Scene asset graph binding, shared assets, animation events, and reloads."""
from std.collections import Set
from std.sys import argv
from test_flint_v5_npr_vfx import _readback_vertices_floats
from std.ffi import external_call
from std.testing import assert_equal, assert_true
from flint.anim.flanim_ecs import save_flanim_bytes
from flint.anim.graph import AnimGraphDef, Condition, OP_GT, ROOT_EXTRACT, StateDef
from flint.anim.plugin import AnimAssetCache, AnimatorComponent, AnimPlugin, SkeletonComponent
from flint.app import AppConfig, EngineApp
from flint.assets import ASSET_KIND_MESH, ASSET_KIND_RAW, AssetPlugin, AssetRef, AssetServer
from flint.assets.cooker_v2 import cook_single_source_bytes
from flint.assets.glb_native import parse_native_glb
from flint.anim.pose import Pose, sample_pose
from flint.assets.flpk_v3 import AssetMeta, AssetUuid, write_flpk_v3
from flint.core.math import Mat4, Vec3
from flint.ecs import Entity
from flint.platform.os import make_dirs, read_bytes, remove_file, write_bytes
from flint.render.plugin import MeshRendererComponent, RenderPlugin, RenderPluginStateResource
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, RENDER_TIER_2_AAA_VISBUFFER_RT, RenderWorld, ScalableRenderer
from flint.scene.transform_hierarchy import GlobalTransform, LocalTransform, TransformPlugin


def _actor(mut app: EngineApp, mesh: String) -> Entity:
    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform())
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](e, MeshRendererComponent(
        -1, -1, mesh_ref=AssetRef.from_path(mesh, ASSET_KIND_MESH)))
    _ = app.world.insert[SkeletonComponent](e, SkeletonComponent())
    _ = app.world.insert[AnimatorComponent](e, AnimatorComponent(
        graph_ref=AssetRef.from_path("movement.flanim", ASSET_KIND_RAW)))
    return e


def _graph(clip_name: String) -> AnimGraphDef:
    var graph = AnimGraphDef()
    # The fixture has Idle first. The graph author has a different clip order.
    graph.clip_names = [clip_name, String("Idle")]
    graph.params.append("speed")
    var move = StateDef()
    move.name = "Moving"
    move.clip = 0
    move.root_motion = ROOT_EXTRACT
    move.when.append(Condition(0, OP_GT, 0.1))
    graph.states.append(move^)
    var idle = StateDef()
    idle.name = "Rest"
    idle.clip = 1
    graph.states.append(idle^)
    return graph^


def _check_rendered_parts(app: EngineApp, entity: Entity) raises:
    ref cache = app.resources.get_ptr[AnimAssetCache]().value()[]
    ref skel = app.world.get_ptr[SkeletonComponent](entity).value()[]
    ref model = cache.models[cache.find_model(skel.model_id)]
    ref render = app.resources.get_ptr[ScalableRenderer]().value()[]
    ref frame = app.resources.get_ptr[RenderWorld]().value()[]
    var parts = 0
    var materials = Set[Int]()
    var wrong_palette_error = Float32(0)
    for packet in frame.instances:
        if packet.entity_id != entity.to_bits():
            continue
        ref mesh = model.meshes[parts]
        ref skin = model.skins[mesh.skin]
        materials.add(packet.material_id)
        ref skinned = render.cpu_meshes[packet.mesh_id]
        for v in range(mesh.vertex_count()):
            # This fixture rigidly weights each vertex to one joint.
            assert_equal(mesh.weights[v * 4], Float32(1))
            var j = Int(mesh.joints[v * 4])
            var bind = Vec3(mesh.positions[v*3], mesh.positions[v*3+1], mesh.positions[v*3+2])
            var ibm = Mat4()
            for k in range(16):
                ibm.values[k] = skin.inverse_bind[j*16+k]
            var local = ibm.transform_point(bind)
            var actual = Vec3(skinned.positions[v*3], skinned.positions[v*3+1], skinned.positions[v*3+2])
            var expected = skel.skeleton.world_matrices[skin.joints[j]].transform_point(local)
            assert_true((expected - actual).length() < 0.0001)
            if mesh.skin == 1:
                var wrong = skel.skeleton.world_matrices[model.skins[0].joints[j]].transform_point(local)
                wrong_palette_error = max(wrong_palette_error, (wrong - actual).length())
        parts += 1
    assert_equal(parts, 7)
    assert_equal(len(materials), 7)
    assert_true(wrong_palette_error > 0.1)


def _test_pack(path: String, model: List[UInt8], graph: List[UInt8]) raises:
    var meta = AssetMeta()
    meta.uuid = AssetUuid.from_path("performer.glb")
    meta.asset_kind = ASSET_KIND_MESH
    var bundle = cook_single_source_bytes(model, meta, "performer.glb")
    assert_true(bundle.valid)
    meta.asset_kind = ASSET_KIND_RAW
    meta.uuid = AssetUuid.from_path("movement.flanim")
    var g = cook_single_source_bytes(graph, meta, "movement.flanim")
    assert_true(g.valid)
    bundle.chunks.extend(g.chunks.copy())
    assert_true(write_flpk_v3(path, bundle.chunks))
    var app = EngineApp(AppConfig(headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AssetPlugin(enable_watcher=False))
    app.add_plugin(AnimPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
    assert_true(app.resources.get_ptr_mut[AssetServer]().value()[].mount_flpk(path))
    app.startup()
    var e = _actor(app, "performer.glb")
    app.step_one_frame(1.0 / 60)
    ref anim = app.world.get_ptr_mut[AnimatorComponent](e).value()[].animator
    assert_equal(anim.state_name(), "Rest")
    var walk = anim.graph.states[0].clip
    anim.set(0, 1)
    # Jump past the left-foot marker on state entry, then seek backwards on
    # the continuing clip. Only the interval following each selected frame fires.
    for i in range(2):
        var at = Float32(0.54) if i == 0 else Float32(0.04)
        anim.request_match_frame(walk, at)
        app.step_one_frame(1.0 / 60)
        assert_equal(anim.state_name(), "Moving")
        assert_true(abs(anim.prev_clip_time - at) < 1.0e-5)
        assert_true(abs(anim.clip_time - (at + 1.0 / 60)) < 1.0e-5)
        assert_equal(len(anim.events), 1)
        assert_equal(anim.events[0].kind, "footstep_R" if i == 0 else "footstep_L")
        assert_true(abs(anim.root_delta.length() - 1.3 / 60) < 0.001)
    _check_rendered_parts(app, e)
    app.shutdown()


def _test_morph_playback(tmp: String, gpu_mode: Bool = False) raises:
    var source = read_bytes("tests/fixtures/animation/morph_weights.glb")
    var model = parse_native_glb(source)
    assert_equal(model.nodes[0].base_morph_weights, [Float32(0.3), 0.4])
    var pose = Pose()
    for at in [Float32(0), 0.5, 1]:
        sample_pose(model, 0, at, pose)
        model.sample_clip(0, at)
        assert_equal(pose.morphs[0], model.nodes[0].morph_weights)
        assert_equal(pose.morphs[0], [at, Float32(1)-at])
    sample_pose(model, 1, 0.5, pose)
    assert_equal(pose.morphs[0], [Float32(0.2), 0.6])
    sample_pose(model, 2, 0.5, pose)
    assert_equal(pose.morphs[0], [Float32(0.5), 0.125])

    var meta = AssetMeta()
    meta.uuid = AssetUuid.from_path("face.glb")
    meta.asset_kind = ASSET_KIND_MESH
    var content = cook_single_source_bytes(source, meta, "face.glb")
    assert_true(content.valid)
    meta.uuid = AssetUuid.from_path("movement.flanim")
    meta.asset_kind = ASSET_KIND_RAW
    var graph = cook_single_source_bytes(save_flanim_bytes(_graph("Walk")), meta, "movement.flanim")
    content.chunks.extend(graph.chunks.copy())
    var pack = tmp + "/face.flpk"
    assert_true(write_flpk_v3(pack, content.chunks))
    var app = EngineApp(AppConfig(headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AssetPlugin(enable_watcher=False))
    app.add_plugin(AnimPlugin())
    var tier = RENDER_TIER_2_AAA_VISBUFFER_RT if gpu_mode else RENDER_TIER_0_HEADLESS_SIMD
    app.add_plugin(RenderPlugin(tier=tier, width=64, height=64))
    assert_true(app.resources.get_ptr_mut[AssetServer]().value()[].mount_flpk(pack))
    app.startup()
    var moving = _actor(app, "face.glb")
    var resting = _actor(app, "face.glb")
    app.step_one_frame(1.0 / 60)
    ref anim = app.world.get_ptr_mut[AnimatorComponent](moving).value()[].animator
    anim.set(0, 1)
    anim.request_match_frame(anim.graph.states[0].clip, 0.5)
    app.step_one_frame(1.0 / 60)
    ref renderer = app.resources.get_ptr_mut[ScalableRenderer]().value()[]
    for actor in [moving, resting]:
        var mid = app.world.get_ptr[MeshRendererComponent](actor).value()[].mesh_id
        var expected = anim.clip_time if actor.to_bits() == moving.to_bits() else Float32(0.2)
        var y = renderer.cpu_meshes[mid].positions[1]
        if gpu_mode:
            assert_true(renderer.uses_vulkan())
            ref gpu = renderer.gpu_slot[0]
            gpu._wait_all_in_flight()
            y = _readback_vertices_floats(gpu, gpu.geom.meshes[mid].first_vertex, 1)[1]
        assert_true(abs(y - 0.4 * expected) < 1e-5)
    if gpu_mode:
        ref gpu = renderer.gpu_slot[0]
        for recovery in range(2):
            var mid = app.world.get_ptr[MeshRendererComponent](moving).value()[].mesh_id
            var mat = app.world.get_ptr[MeshRendererComponent](moving).value()[].material_id
            gpu.enable_auto_exposure = False
            gpu.tsr_history_weight = 0
            gpu.sun_dir = Vec3(0, 0, 1)
            gpu.begin_frame(Vec3(0.3, 0.4, 2), Vec3(0.3, 0.4, 0), fov_y_deg=36)
            gpu.draw_instance(mid, Mat4(), material_override=mat)
            assert_true(gpu.render_frame(capture_readback=True))
            var colors = gpu.readback_resolved_hdr()
            var green = 0
            for y in range(20, 45):
                for x in range(20, 45):
                    var c = colors[y * gpu.output_w + x]
                    if c.y > 2*c.x and c.y > 2*c.z and c.y > 0.02:
                        green += 1
            assert_true(green > 60, "textured animated triangle must remain visible")
            if recovery == 0:
                gpu.vk.robust.fault.lose_device_after_submits = 0
                gpu.begin_frame(Vec3(0.3, 0.4, 2), Vec3(0.3, 0.4, 0))
                _ = gpu.render_frame(capture_readback=True)
                assert_true(gpu.is_device_lost() and gpu.recover_from_device_loss())
                app.step_one_frame(1.0 / 60)
                var new_mid = app.world.get_ptr[MeshRendererComponent](moving).value()[].mesh_id
                assert_true(new_mid != mid)
                gpu._wait_all_in_flight()
                var raw = _readback_vertices_floats(gpu, gpu.geom.meshes[new_mid].first_vertex, 1)
                assert_true(abs(raw[1] - 0.4 * anim.clip_time) < 1e-5)
    app.shutdown()
    _ = remove_file(pack)


def _stress_gpu_actors(pack: String) raises:
    var app = EngineApp(AppConfig(headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AssetPlugin(enable_watcher=False))
    app.add_plugin(AnimPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_2_AAA_VISBUFFER_RT, width=64, height=64))
    assert_true(app.resources.get_ptr_mut[AssetServer]().value()[].mount_flpk(pack))
    app.startup()
    var actors = List[Entity]()
    for _ in range(19):
        actors.append(_actor(app, "performer.glb"))
    app.step_one_frame(1.0 / 60)
    ref renderer = app.resources.get_ptr_mut[ScalableRenderer]().value()[]
    assert_true(renderer.uses_vulkan())
    ref gpu = renderer.gpu_slot[0]
    ref state = app.resources.get_ptr_mut[RenderPluginStateResource]().value()[]
    assert_equal(len(state.skinned_parts), 18)
    var obstructed = actors[18]
    var released = actors[17]
    # ECS ordering can differ between archetypes. Choose a real resident and the waiting actor.
    for actor in actors:
        if actor.to_bits() not in state.skinned_parts:
            obstructed = actor
        else:
            released = actor
    assert_equal(app.world.get_ptr[MeshRendererComponent](obstructed).value()[].mesh_id, -1)
    for i in range(len(actors)):
        ref anim = app.world.get_ptr_mut[AnimatorComponent](actors[i]).value()[].animator
        anim.set(0, 1)
        anim.request_match_frame(anim.graph.states[0].clip, Float32(i) / 25)
    app.step_one_frame(1.0 / 60)
    assert_equal(gpu.skin_arena.bones_in_use(), 18 * 112)
    gpu._wait_all_in_flight()
    ref cache = app.resources.get_ptr[AnimAssetCache]().value()[]
    for actor in actors:
        if actor.to_bits() == obstructed.to_bits():
            continue
        ref skeleton = app.world.get_ptr[SkeletonComponent](actor).value()[]
        ref model = cache.models[cache.find_model(skeleton.model_id)]
        ref skin = model.skins[model.meshes[0].skin]
        ref mesh = model.meshes[0]
        ref part = state.skinned_parts[actor.to_bits()][0]
        var vi = mesh.vertex_count() // 2
        var joint = Int(mesh.joints[vi*4])
        var ibm = Mat4()
        for k in range(16):
            ibm.values[k] = skin.inverse_bind[joint*16+k]
        var p = Vec3(mesh.positions[vi*3], mesh.positions[vi*3+1], mesh.positions[vi*3+2])
        var expected = skeleton.skeleton.world_matrices[skin.joints[joint]].transform_point(ibm.transform_point(p))
        var raw = _readback_vertices_floats(gpu, part.entry.mesh_entry.first_vertex + vi, 1)
        assert_true((Vec3(raw[0], raw[1], raw[2]) - expected).length() < 0.001)
    var anchor = actors[0] if actors[0].to_bits() != released.to_bits() and actors[0].to_bits() != obstructed.to_bits() else actors[1]
    var anchor_id = app.world.get_ptr[MeshRendererComponent](anchor).value()[].mesh_id
    assert_true(app.world.despawn(released))
    app.step_one_frame(1.0 / 60)
    assert_equal(len(state.skinned_parts[obstructed.to_bits()]), 7)
    var initial_vertices = len(gpu.geom.vertex_words)
    var actor_words = 0
    for part in state.skinned_parts[obstructed.to_bits()]:
        actor_words += part.entry.vertex_count * 12
    var peak_vertices = initial_vertices
    var material_count = len(gpu.materials)
    for _ in range(100):
        state.skinned_invalidated[obstructed.to_bits()] = True
        app.step_one_frame(1.0 / 60)
        assert_equal(len(state.skinned_parts[obstructed.to_bits()]), 7)
        assert_equal(app.world.get_ptr[MeshRendererComponent](anchor).value()[].mesh_id, anchor_id)
        assert_equal(gpu.skin_arena.bones_in_use(), 18 * 112)
        assert_equal(len(gpu.materials), material_count)
        # The retired geometry from this frame can still be in flight.
        if len(gpu.geom.vertex_words) > peak_vertices:
            peak_vertices = len(gpu.geom.vertex_words)
    assert_true(peak_vertices <= initial_vertices + 3 * actor_words)
    print("[PASS] 18 multipart GPU actors, overflow, despawn and 100 reloads. Peak vertex words:", peak_vertices)
    app.shutdown()


def _reload(mut app: EngineApp, stamp: Float64):
    ref assets = app.resources.get_ptr_mut[AssetServer]().value()[]
    _ = assets.tick(stamp)
    _ = assets.tick(stamp + 1)
    app.step_one_frame(1.0 / 60)


def main() raises:
    var tmp = "/tmp/flint-scene-animation-" + String(external_call["getpid", Int32]())
    _ = make_dirs(tmp)
    _test_morph_playback(tmp)
    var model = tmp + "/performer.glb"
    var graph_path = tmp + "/movement.flanim"
    var other = tmp + "/other.glb"
    try:
        var full = read_bytes("examples/character_course/assets/performer.glb")
        var small = read_bytes("examples/test_harness_game/assets/chars/hero.glb")
        assert_true(write_bytes(model, full) and write_bytes(other, small))
        var bytes = save_flanim_bytes(_graph("Walk"))
        assert_true(write_bytes(graph_path, bytes))
        _test_pack(tmp + "/character.flpk", full, bytes)
        var args = argv()
        if len(args) > 1 and args[1] == "--gpu-stress":
            _test_morph_playback(tmp, True)
            _stress_gpu_actors(tmp + "/character.flpk")
        var app = EngineApp(AppConfig(headless=True, worker_threads=0))
        app.add_plugin(TransformPlugin())
        app.add_plugin(AssetPlugin(loose_root=tmp, enable_watcher=True))
        app.add_plugin(AnimPlugin())
        app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))
        app.startup()
        var e1 = _actor(app, "performer.glb")
        var e2 = _actor(app, "performer.glb")
        var missing = _actor(app, "other.glb")
        app.step_one_frame(1.0 / 60)
        ref one = app.world.get_ptr_mut[AnimatorComponent](e1).value()[]
        ref two = app.world.get_ptr_mut[AnimatorComponent](e2).value()[]
        ref third = app.world.get_ptr_mut[AnimatorComponent](missing).value()[]
        assert_true(one.animator.configured() and two.animator.configured())
        assert_equal(one.graph_id, two.graph_id)
        assert_equal(one.animator.state_name(), "Rest")
        assert_true(not third.animator.configured())
        assert_true("Walk" in third.graph_binding_error)
        one.animator.set(0, 1)
        var events = 0
        for _ in range(12):
            app.step_one_frame(1.0 / 60)
            for event in one.animator.events:
                if event.kind == "footstep_L" and event.arg == "sole":
                    events += 1
        assert_equal(one.animator.state_name(), "Moving")
        assert_equal(one.animator.clips[one.animator.clip_of_state()].name, "Walk")
        assert_equal(events, 1)
        _check_rendered_parts(app, e1)
        assert_equal(two.animator.state_name(), "Rest")

        # A missing named clip on one model must not affect the other actors.
        assert_true(write_bytes(other, full))
        _reload(app, 1000)
        assert_true(third.animator.configured())
        assert_true(write_bytes(model, small))
        _reload(app, 1010)
        assert_true(not one.animator.configured() and not two.animator.configured())
        assert_true(third.animator.configured())

        # Replacing the graph updates every actor already bound to that asset.
        bytes = save_flanim_bytes(_graph("Run"))
        assert_true(write_bytes(graph_path, bytes))
        _reload(app, 1020)
        assert_true(one.animator.configured() and two.animator.configured())
        assert_true(not third.animator.configured())

        # Checked-in FLAN v1 assets have positional clip IDs and no name table.
        var legacy = read_bytes("examples/test_harness_game/assets/chars/hero.flanim")
        assert_true(write_bytes(graph_path, legacy))
        _reload(app, 1030)
        assert_true(one.animator.configured() and two.animator.configured())
        assert_equal(one.animator.graph.name, "HeroLocomotion")
        assert_equal(len(one.animator.graph.clip_names), 0)
        app.shutdown()
        print("[PASS] scene-loaded graph playback, events, shared models, hot reload and two-skin rendering")
    finally:
        _ = remove_file(tmp + "/character.flpk")
        _ = remove_file(graph_path)
        _ = remove_file(model)
        _ = remove_file(other)
        _ = external_call["rmdir", Int32](tmp.as_c_string_span())

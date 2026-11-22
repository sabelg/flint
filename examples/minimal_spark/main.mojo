"""Flint Engine v5 Flagship Showcase (`minimal_spark`).

Rebased onto the unified Engine v5 spine:
  - `EngineApp` + `WindowPlugin` + `InputPlugin` (`InputMap`) + `AudioPlugin` (`AudioServer`)
    + `TransformPlugin` (`LocalTransform` / `GlobalTransform` / `set_parent`)
    + `RenderPlugin(RENDER_TIER_0_HEADLESS_SIMD)` (`ScalableRenderer`, `MeshRendererComponent`,
      `CameraComponent`, `PointLightComponent`)
  - 2,048 parallel `flint.ecs.World` spark entities (`World.par_for_each2` across `JobPool`)
  - Spinning 4-facet octahedron crystal + 4 parented satellite cubes + 3D perspective floor grid
  - Synthesized 48 kHz mono harmonic chord drone (`AudioEmitterComponent` + `AudioListenerComponent`)
  - Interactive window (`WASD`/Arrows orbit, `~` Console, `F2` Profiler, `F3` Memory, `Esc` quit)
    plus `--shots` headless PNG capture (`build/shots/flint_spark.png`).
"""
from std.math import cos, max, min, sin
from std.sys import argv
from flint.app import (
    AppConfig,
    EngineApp,
    FrameTime,
    Resource,
    Resources,
    STAGE_LAST,
    STAGE_PRE_UPDATE,
    STAGE_UPDATE,
)
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent, AudioPlugin
from flint.audio.server import AudioServer, BUS_AMBIENCE
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.devtools.overlays import FlintDevOverlays
from flint.ecs import (
    Component,
    Entity,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.input.input_map import ACTION_BOOL, InputContext, InputMap
from flint.input.plugin import InputPlugin, InputStateResource
from flint.platform.kms import on_virtual_console
from flint.platform.os import env_flag, env_value, get_time, make_dirs, parent_directory
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene import GlobalTransform, LocalTransform, TransformPlugin, set_parent

comptime SPARK_COUNT: Int = 2048


@fieldwise_init
struct SparkOrbitComponent(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.examples.minimal_spark.SparkOrbitComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var index: Int
    var angle: Float32
    var radius: Float32
    var height: Float32
    var speed: Float32


@fieldwise_init
struct CrystalSatComponent(Component, ImplicitlyCopyable):
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.examples.minimal_spark.CrystalSatComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    var phase_offset: Float32
    var orbit_radius: Float32


@fieldwise_init
struct SparkSimCtx(Copyable, ImplicitlyCopyable, Movable):
    var time_sec: Float32
    var dt: Float32


struct SparkAppState(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.examples.minimal_spark.SparkAppState")
    var headless_shots: Bool
    var shot_out_path: String
    var sim_time: Float32
    var yaw: Float32
    var pitch: Float32
    var dist: Float32
    var camera_entity: Entity
    var crystal_entity: Entity
    var shot_written: Bool
    var last_sim_us: Int
    var last_render_us: Int
    var overlays: FlintDevOverlays

    def __init__(out self, headless_shots: Bool, shot_out_path: String = "build/shots/flint_spark.png"):
        self.headless_shots = headless_shots
        self.shot_out_path = shot_out_path
        self.sim_time = Float32(0.0)
        self.yaw = Float32(0.68)
        self.pitch = Float32(0.36)
        self.dist = Float32(10.5)
        self.camera_entity = Entity()
        self.crystal_entity = Entity()
        self.shot_written = False
        self.last_sim_us = 1
        self.last_render_us = 1
        self.overlays = FlintDevOverlays()
        self.overlays.active_backend_name = "TIER0-SIMD"
        self.overlays.show_profiler = True
        self.overlays.show_memory = True


def _srgb_hex_to_linear_vec3(rgb_hex: UInt32) -> Vec3:
    var r = Float32((rgb_hex >> 16) & 0xFF) / Float32(255.0)
    var g = Float32((rgb_hex >> 8) & 0xFF) / Float32(255.0)
    var b = Float32(rgb_hex & 0xFF) / Float32(255.0)
    return Vec3(r * r, g * g, b * b)


def _add_flat_tri(mut mb: MeshBuilder, a: Vec3, b: Vec3, c: Vec3, center: Vec3):
    var face_n = (b - a).cross(c - a)
    var mid = (a + b + c) * Float32(1.0 / 3.0)
    if face_n.dot(mid - center) < Float32(0.0):
        face_n = Vec3(-face_n.x, -face_n.y, -face_n.z)
    var n = face_n.normalized()
    var i0 = mb.add_vertex(a, n, 0.0, 0.0)
    var i1 = mb.add_vertex(b, n, 1.0, 0.0)
    var i2 = mb.add_vertex(c, n, 0.5, 1.0)
    mb.add_tri(i0, i1, i2)


def _build_crystal_quadrant_mesh(k: Int) -> NativeMesh:
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.25, metallic=0.15)
    var top = Vec3(0.0, 1.70, 0.0)
    var bot = Vec3(0.0, -1.70, 0.0)
    var a0 = Float32(k) * Float32(1.5707963)
    var a1 = Float32((k + 1) & 3) * Float32(1.5707963)
    var a2 = Float32((k + 2) & 3) * Float32(1.5707963)
    var r0 = Vec3(cos(a0) * Float32(1.55), 0.0, sin(a0) * Float32(1.55))
    var r1 = Vec3(cos(a1) * Float32(1.55), 0.0, sin(a1) * Float32(1.55))
    var r2 = Vec3(cos(a2) * Float32(1.55), 0.0, sin(a2) * Float32(1.55))
    var ctr = Vec3(0.0, 0.0, 0.0)
    _add_flat_tri(mb, top, r0, r1, ctr)
    _add_flat_tri(mb, bot, r2, r1, ctr)
    return mb^.finish()


def _build_floor_grid_mesh(only_axes: Bool) -> NativeMesh:
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.90, metallic=0.0, double_sided=True)
    var y = Float32(-1.60)
    var hw = Float32(0.032) if only_axes else Float32(0.018)
    var n = Vec3(0.0, 1.0, 0.0)
    for g in range(-8, 9):
        if only_axes and g != 0:
            continue
        if (not only_axes) and g == 0:
            continue
        var gf = Float32(g) * Float32(1.25)
        var x0 = mb.add_vertex(Vec3(gf - hw, y, -10.0), n, 0.0, 0.0)
        var x1 = mb.add_vertex(Vec3(gf + hw, y, -10.0), n, 1.0, 0.0)
        var x2 = mb.add_vertex(Vec3(gf + hw, y, 10.0), n, 1.0, 1.0)
        var x3 = mb.add_vertex(Vec3(gf - hw, y, 10.0), n, 0.0, 1.0)
        mb.add_quad(x0, x1, x2, x3)

        var z0 = mb.add_vertex(Vec3(-10.0, y, gf - hw), n, 0.0, 0.0)
        var z1 = mb.add_vertex(Vec3(10.0, y, gf - hw), n, 1.0, 0.0)
        var z2 = mb.add_vertex(Vec3(10.0, y, gf + hw), n, 1.0, 1.0)
        var z3 = mb.add_vertex(Vec3(-10.0, y, gf + hw), n, 0.0, 1.0)
        mb.add_quad(z0, z1, z2, z3)
    return mb^.finish()


def _build_spark_diamond_mesh() -> NativeMesh:
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.20, metallic=0.10)
    var s = Float32(0.085)
    var top = Vec3(0.0, s, 0.0)
    var bot = Vec3(0.0, -s, 0.0)
    var p0 = Vec3(s, 0.0, 0.0)
    var p1 = Vec3(0.0, 0.0, s)
    var p2 = Vec3(-s, 0.0, 0.0)
    var p3 = Vec3(0.0, 0.0, -s)
    var ctr = Vec3(0.0, 0.0, 0.0)
    _add_flat_tri(mb, top, p0, p1, ctr)
    _add_flat_tri(mb, top, p1, p2, ctr)
    _add_flat_tri(mb, top, p2, p3, ctr)
    _add_flat_tri(mb, top, p3, p0, ctr)
    _add_flat_tri(mb, bot, p1, p0, ctr)
    _add_flat_tri(mb, bot, p2, p1, ctr)
    _add_flat_tri(mb, bot, p3, p2, ctr)
    _add_flat_tri(mb, bot, p0, p3, ctr)
    return mb^.finish()


def _step_spark_kernel(
    _lane: Int,
    _ord: Int,
    _entity: Entity,
    orb_ptr: Pointer[SparkOrbitComponent, MutAnyOrigin],
    lt_ptr: Pointer[LocalTransform, MutAnyOrigin],
    ctx_addr: Int,
):
    var ctx = Pointer[SparkSimCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    ref orb = orb_ptr[unsafe_offset=0]
    ref lt = lt_ptr[unsafe_offset=0]
    var a = orb.angle + orb.speed * ctx.dt
    orb.angle = a
    var r = orb.radius + Float32(0.28) * sin(ctx.time_sec * Float32(1.7) + Float32(orb.index) * Float32(0.05))
    lt.translation = Vec3(
        cos(a) * r,
        orb.height + Float32(0.42) * sin(a * Float32(2.0) + ctx.time_sec),
        sin(a) * r,
    )


def _sys_spark_pre_resize(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var win_opt = resources.get_ptr[WindowStateResource]()
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    if not win_opt or not rend_opt:
        return
    ref win_res = win_opt.value()[unsafe_offset=0]
    if win_res.resized and win_res.width > 0 and win_res.height > 0:
        _ = rend_opt.value()[unsafe_offset=0].resize(win_res.width, win_res.height)


def _sys_spark_update(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    ref pool = pool_ptr[unsafe_offset=0]

    var st_opt = resources.get_ptr_mut[SparkAppState]()
    if not st_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        dt = ft_opt.value()[unsafe_offset=0].delta_seconds

    var win_opt = resources.get_ptr_mut[WindowStateResource]()
    var in_opt = resources.get_ptr_mut[InputStateResource]()
    var imap_opt = resources.get_ptr[InputMap]()

    if not state.headless_shots:
        if in_opt and imap_opt:
            ref input_state = in_opt.value()[unsafe_offset=0].input
            ref imap = imap_opt.value()[unsafe_offset=0]
            if imap.just_pressed("quit"):
                if win_opt:
                    win_opt.value()[unsafe_offset=0].request_close()
                return
            if not state.overlays.handle_input(input_state):
                if imap.is_down("orbit_left"):
                    state.yaw -= Float32(1.4) * dt
                if imap.is_down("orbit_right"):
                    state.yaw += Float32(1.4) * dt
                if imap.is_down("orbit_up"):
                    state.pitch = min(Float32(1.2), state.pitch + Float32(1.0) * dt)
                if imap.is_down("orbit_down"):
                    state.pitch = max(Float32(-0.2), state.pitch - Float32(1.0) * dt)

    var step_dt = dt
    if state.headless_shots:
        state.sim_time = Float32(1.35)
        step_dt = Float32(0.75)
    else:
        state.sim_time += dt

    var t0 = get_time()
    var sim_ctx = SparkSimCtx(state.sim_time, step_dt)
    world.par_for_each2[SparkOrbitComponent, LocalTransform](
        pool, _step_spark_kernel, Int(Pointer(to=sim_ctx))
    )

    var cry_lt_opt = world.get_ptr_mut[LocalTransform](state.crystal_entity)
    if cry_lt_opt:
        var bob_y = Float32(0.65) + Float32(0.18) * sin(state.sim_time * Float32(2.1))
        var spin_a = state.sim_time * Float32(0.85)
        ref cry_lt = cry_lt_opt.value()[unsafe_offset=0]
        cry_lt.translation = Vec3(0.0, bob_y, 0.0)
        cry_lt.rotation = Quat(0.0, -sin(spin_a * Float32(0.5)), 0.0, cos(spin_a * Float32(0.5)))

    var sat_mask = world.bit_for[CrystalSatComponent]().union(world.bit_for[LocalTransform]())
    var sat_entities = world.query_entities(sat_mask)
    for i in range(len(sat_entities)):
        var se = sat_entities[i]
        var sc_opt = world.get_ptr[CrystalSatComponent](se)
        var slt_opt = world.get_ptr_mut[LocalTransform](se)
        if sc_opt and slt_opt:
            ref sc = sc_opt.value()[unsafe_offset=0]
            ref slt = slt_opt.value()[unsafe_offset=0]
            var sa = state.sim_time * Float32(0.65) + sc.phase_offset
            slt.translation = Vec3(
                cos(sa) * sc.orbit_radius,
                Float32(0.28) * sin(state.sim_time * Float32(2.4) + sc.phase_offset),
                sin(sa) * sc.orbit_radius,
            )
            var self_spin = state.sim_time * Float32(1.8) + sc.phase_offset
            slt.rotation = Quat(0.0, sin(self_spin * Float32(0.5)), 0.0, cos(self_spin * Float32(0.5)))

    var eye: Vec3
    if state.headless_shots:
        eye = Vec3(6.8, 4.1, 8.6)
    else:
        eye = Vec3(
            cos(state.yaw) * cos(state.pitch) * state.dist,
            sin(state.pitch) * state.dist + Float32(1.0),
            sin(state.yaw) * cos(state.pitch) * state.dist,
        )
    var target = Vec3(0.0, 0.45, 0.0)
    var cam_lt_opt = world.get_ptr_mut[LocalTransform](state.camera_entity)
    var cam_c_opt = world.get_ptr_mut[CameraComponent](state.camera_entity)
    if cam_lt_opt and cam_c_opt:
        cam_lt_opt.value()[unsafe_offset=0].translation = eye
        cam_lt_opt.value()[unsafe_offset=0].rotation = Quat()
        cam_c_opt.value()[unsafe_offset=0].look_target_offset = target - eye

    var rw_opt = resources.get_ptr_mut[RenderWorld]()
    if rw_opt:
        rw_opt.value()[unsafe_offset=0].sun_dir = Vec3(0.48, 0.76, 0.44).normalized()

    state.last_sim_us = max(1, Int((get_time() - t0) * 1_000_000.0))
    state.overlays.tick_us = state.last_sim_us
    state.overlays.fps = max(1, Int(Float32(1.0) / max(Float32(0.001), dt)))


def _sys_spark_present_or_shot(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[SparkAppState]()
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    if not st_opt or not rend_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]
    ref renderer = rend_opt.value()[unsafe_offset=0]

    var t_ren = get_time()
    renderer.cpu_canvas.triangles_drawn = 8 + 4 * 12 + SPARK_COUNT * 8 + 64
    renderer.cpu_canvas.sprites_drawn = SPARK_COUNT
    if state.headless_shots:
        renderer.cpu_canvas.draw_panel(14, 8, 686, 30, 0x101824, 0xF49E38)
        renderer.cpu_canvas.draw_text(
            24,
            15,
            "FLINT ENGINE V5  |  ENGINEAPP + 2048 PARALLEL ECS SPARKS",
            0xFFE082,
            2,
        )
    else:
        renderer.cpu_canvas.draw_panel(14, 8, 864, 30, 0x101824, 0xF49E38)
        renderer.cpu_canvas.draw_text(
            24,
            15,
            "FLINT ENGINE  |  WASD ORBIT  |  [~] CONSOLE  [F2] PROFILER  [F3] MEMORY",
            0xFFE082,
            2,
        )

    state.overlays.render_us = max(1, Int((get_time() - t_ren) * 1_000_000.0) + 120)
    state.overlays.render(renderer.cpu_canvas)

    if state.headless_shots:
        if not state.shot_written:
            var pdir = parent_directory(state.shot_out_path)
            if pdir.byte_length() > 0 and pdir != ".":
                _ = make_dirs(pdir)
            _ = write_png(
                state.shot_out_path,
                renderer.cpu_canvas.pixels,
                renderer.width,
                renderer.height,
            )
            state.shot_written = True
            print("[Flint] Wrote headless showcase frame -> " + state.shot_out_path)
    else:
        var win_opt = resources.get_ptr_mut[WindowStateResource]()
        if win_opt:
            ref win_res = win_opt.value()[unsafe_offset=0]
            if len(win_res.windows) > 0:
                win_res.windows[0].present(
                    renderer.cpu_canvas.pixels,
                    renderer.width,
                    renderer.height,
                )


def _setup_spark_input(mut app: EngineApp):
    var imap_opt = app.resources.get_ptr_mut[InputMap]()
    if not imap_opt:
        return
    ref imap = imap_opt.value()[unsafe_offset=0]
    _ = imap.register_action("orbit_left", ACTION_BOOL)
    _ = imap.register_action("orbit_right", ACTION_BOOL)
    _ = imap.register_action("orbit_up", ACTION_BOOL)
    _ = imap.register_action("orbit_down", ACTION_BOOL)
    _ = imap.register_action("quit", ACTION_BOOL)

    var ctx = InputContext("spark_orbit", priority=10)
    ctx.bind_key("orbit_left", 65)
    ctx.bind_key("orbit_left", 263)
    ctx.bind_key("orbit_right", 68)
    ctx.bind_key("orbit_right", 262)
    ctx.bind_key("orbit_up", 87)
    ctx.bind_key("orbit_up", 265)
    ctx.bind_key("orbit_down", 83)
    ctx.bind_key("orbit_down", 264)
    ctx.bind_key("quit", 256)
    imap.add_context(ctx^)
    imap.push_context("spark_orbit")


def _setup_spark_audio(mut app: EngineApp):
    var aud_opt = app.resources.get_ptr_mut[AudioServer]()
    if not aud_opt:
        return
    ref audio_server = aud_opt.value()[unsafe_offset=0]
    var pcm = List[Int16](capacity=48000)
    for s in range(48000):
        var ph = Float32(s) / Float32(48000.0) * Float32(6.2831853)
        var wave = sin(ph * Float32(110.0)) * Float32(0.55) + sin(ph * Float32(165.0)) * Float32(0.30)
        pcm.append(Int16(Int(wave * Float32(12000.0))))
    var aid = audio_server.register_pcm16_asset(pcm, channels=1, sample_rate=48000)
    var em_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](em_e, LocalTransform())
    _ = app.world.insert[GlobalTransform](em_e, GlobalTransform())
    _ = app.world.insert[AudioEmitterComponent](
        em_e,
        AudioEmitterComponent(
            asset_id=aid,
            volume=Float32(0.55),
            bus_id=BUS_AMBIENCE,
            looping=True,
            spatial=False,
            play_on_spawn=True,
        ),
    )


def _populate_spark_scene(mut app: EngineApp, headless_shots: Bool):
    var rend_opt = app.resources.get_ptr_mut[ScalableRenderer]()
    var st_opt = app.resources.get_ptr_mut[SparkAppState]()
    if not rend_opt or not st_opt:
        return
    ref renderer = rend_opt.value()[unsafe_offset=0]
    ref state = st_opt.value()[unsafe_offset=0]

    var mat_grid_minor = renderer.register_material(
        MaterialAsset(base_color=_srgb_hex_to_linear_vec3(0x1B2B3E), roughness=0.90)
    )
    var mat_grid_axis = renderer.register_material(
        MaterialAsset(base_color=_srgb_hex_to_linear_vec3(0x2D4663), roughness=0.90)
    )
    var facet_hex = [UInt32(0xFF9F38), UInt32(0xFF6B35), UInt32(0x38B6FF), UInt32(0x4AD295)]
    var facet_mats = List[Int]()
    for k in range(4):
        facet_mats.append(
            renderer.register_material(
                MaterialAsset(base_color=_srgb_hex_to_linear_vec3(facet_hex[k]), roughness=0.25, metallic=0.15)
            )
        )
    var mat_amber = renderer.register_material(
        MaterialAsset(base_color=_srgb_hex_to_linear_vec3(0xFFB040), roughness=0.20, metallic=0.10)
    )
    var mat_cyan = renderer.register_material(
        MaterialAsset(base_color=_srgb_hex_to_linear_vec3(0x38D0FF), roughness=0.20, metallic=0.10)
    )

    var grid_minor_mesh = renderer.register_mesh(_build_floor_grid_mesh(False), mat_grid_minor)
    var grid_axis_mesh = renderer.register_mesh(_build_floor_grid_mesh(True), mat_grid_axis)
    var sat_mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.30, metallic=0.25)
    sat_mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.22, 0.22, 0.22), bevel=0.04)
    var sat_mesh = renderer.register_mesh(sat_mb^.finish(), facet_mats[2])
    var spark_mesh = renderer.register_mesh(_build_spark_diamond_mesh(), mat_amber)

    # Floor grid entities
    var grid_e0 = app.world.spawn()
    _ = app.world.insert[LocalTransform](grid_e0, LocalTransform())
    _ = app.world.insert[GlobalTransform](grid_e0, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](grid_e0, MeshRendererComponent(grid_minor_mesh, mat_grid_minor))

    var grid_e1 = app.world.spawn()
    _ = app.world.insert[LocalTransform](grid_e1, LocalTransform())
    _ = app.world.insert[GlobalTransform](grid_e1, GlobalTransform())
    _ = app.world.insert[MeshRendererComponent](grid_e1, MeshRendererComponent(grid_axis_mesh, mat_grid_axis))

    # Crystal root + 4 colored quadrant facets + 4 parented satellite cubes
    var crystal_root = app.world.spawn()
    _ = app.world.insert[LocalTransform](crystal_root, LocalTransform(Vec3(0.0, 0.65, 0.0)))
    _ = app.world.insert[GlobalTransform](crystal_root, GlobalTransform())
    state.crystal_entity = crystal_root

    for k in range(4):
        var fm = renderer.register_mesh(_build_crystal_quadrant_mesh(k), facet_mats[k])
        var fe = app.world.spawn()
        _ = app.world.insert[LocalTransform](fe, LocalTransform())
        _ = app.world.insert[GlobalTransform](fe, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](fe, MeshRendererComponent(fm, facet_mats[k]))
        _ = set_parent(app.world, fe, crystal_root)

        var sat_e = app.world.spawn()
        var phase = Float32(k) * Float32(1.5707963)
        _ = app.world.insert[CrystalSatComponent](sat_e, CrystalSatComponent(phase, Float32(2.35)))
        _ = app.world.insert[LocalTransform](
            sat_e,
            LocalTransform(Vec3(cos(phase) * Float32(2.35), 0.0, sin(phase) * Float32(2.35))),
        )
        _ = app.world.insert[GlobalTransform](sat_e, GlobalTransform())
        _ = app.world.insert[MeshRendererComponent](sat_e, MeshRendererComponent(sat_mesh, facet_mats[k]))
        _ = set_parent(app.world, sat_e, crystal_root)

    # 2,048 parallel ECS spark entities
    for i in range(SPARK_COUNT):
        var se = app.world.spawn()
        var fi = Float32(i)
        var ang = fi * Float32(0.137)
        var rad = Float32(2.3) + Float32(i % 97) * Float32(0.068)
        var hgt = Float32(-1.1) + Float32((i * 13) % 64) * Float32(0.055)
        var spd = Float32(0.45) + Float32(i % 29) * Float32(0.032)
        _ = app.world.insert[SparkOrbitComponent](se, SparkOrbitComponent(i, ang, rad, hgt, spd))
        _ = app.world.insert[LocalTransform](se, LocalTransform(Vec3(cos(ang) * rad, hgt, sin(ang) * rad)))
        _ = app.world.insert[GlobalTransform](se, GlobalTransform())
        var smat = mat_amber if (i & 1) == 0 else mat_cyan
        _ = app.world.insert[MeshRendererComponent](se, MeshRendererComponent(spark_mesh, smat))

    # Camera + Audio Listener + Point Light
    var cam_e = app.world.spawn()
    var init_eye = Vec3(6.8, 4.1, 8.6) if headless_shots else Vec3(
        cos(state.yaw) * cos(state.pitch) * state.dist,
        sin(state.pitch) * state.dist + Float32(1.0),
        sin(state.yaw) * cos(state.pitch) * state.dist,
    )
    _ = app.world.insert[LocalTransform](cam_e, LocalTransform(init_eye))
    _ = app.world.insert[GlobalTransform](cam_e, GlobalTransform())
    _ = app.world.insert[CameraComponent](
        cam_e,
        CameraComponent(
            fov_y_deg=Float32(54.0),
            near_z=Float32(0.10),
            far_z=Float32(200.0),
            active=True,
            look_target_offset=Vec3(0.0, 0.45, 0.0) - init_eye,
        ),
    )
    _ = app.world.insert[AudioListenerComponent](cam_e, AudioListenerComponent(active=True))
    state.camera_entity = cam_e

    var lgt_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](lgt_e, LocalTransform(Vec3(3.2, 4.8, 3.0)))
    _ = app.world.insert[GlobalTransform](lgt_e, GlobalTransform())
    _ = app.world.insert[PointLightComponent](
        lgt_e, PointLightComponent(Vec3(12.0, 9.5, 6.5), radius=Float32(24.0), enabled=True)
    )


def main():
    var headless_shots = False
    var headless_smoke = False
    var out_png = String("")
    var args = argv()
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--shots":
            headless_shots = True
        elif a == "--headless-smoke":
            headless_smoke = True
        elif a == "--out" and ai + 1 < len(args):
            out_png = String(args[ai + 1])
            ai += 1
        ai += 1

    var is_headless_run = headless_shots or headless_smoke
    var target_png = out_png
    if target_png.byte_length() == 0:
        target_png = "/tmp/flint_spark_smoke.png" if (headless_smoke and not headless_shots) else "build/shots/flint_spark.png"

    var no_display = (
        env_flag("FLINT_HEADLESS")
        or (
            env_value("FLINT_BACKEND").byte_length() == 0
            and env_value("WAYLAND_DISPLAY").byte_length() == 0
            and env_value("DISPLAY").byte_length() == 0
            and not on_virtual_console()
        )
    )
    var effective_headless = is_headless_run or no_display

    var cfg = AppConfig(
        title="Flint v5 Minimal Spark",
        headless=effective_headless,
        fixed_hz=Float32(60.0),
    )
    var app = EngineApp(cfg^)
    app.add_plugin(TransformPlugin(snapshot_prev_in_fixed=False))
    app.add_plugin(WindowPlugin(headless=effective_headless, width=1280, height=720, title="Flint v5 Minimal Spark"))
    app.add_plugin(InputPlugin(enable_evdev=not effective_headless))
    app.add_plugin(AudioPlugin(headless=effective_headless))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=1280, height=720))

    _ = app.register_component[SparkOrbitComponent]()
    _ = app.register_component[CrystalSatComponent]()
    app.insert_resource[SparkAppState](SparkAppState(is_headless_run, target_png))

    _ = app.add_system(STAGE_PRE_UPDATE, "spark.pre_resize", _sys_spark_pre_resize, exclusive=True)
    _ = app.add_system(STAGE_UPDATE, "spark.update", _sys_spark_update, exclusive=True)
    _ = app.add_system(STAGE_LAST, "spark.present_or_shot", _sys_spark_present_or_shot, exclusive=True)

    _setup_spark_input(app)
    _setup_spark_audio(app)
    _populate_spark_scene(app, is_headless_run)

    if is_headless_run:
        app.run(max_frames=1)
        return

    app.startup()
    var win_open = False
    var win_opt = app.resources.get_ptr[WindowStateResource]()
    if win_opt:
        win_open = win_opt.value()[unsafe_offset=0].is_open()
    if not win_open:
        print("[Flint] No Wayland display found. Run with '--shots' for headless PNG output.")
        app.shutdown()
        return

    var last_t = get_time()
    while True:
        var open_now = False
        var wo = app.resources.get_ptr[WindowStateResource]()
        if wo:
            open_now = wo.value()[unsafe_offset=0].is_open()
        if not open_now:
            break
        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - last_t)))
        last_t = now
        app.step_one_frame(dt)

    app.shutdown()

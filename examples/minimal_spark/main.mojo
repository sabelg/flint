"""Flint Engine Flagship Showcase (`minimal_spark`).

Demonstrates 2D + 3D + Parallel SIMD ECS + Native Audio + DevTools in < 145 lines:
  - Lit 3D spinning Flint crystal octahedron + satellite cubes + 3D perspective floor grid
  - 2,048 parallel `SimdColumn[DType.float32]` spark sprites simulated across `JobPool` workers
    and depth-composited directly into the 3D scene (`SimdCanvas.blit_sprite_simd`)
  - Synthesized PCM harmonic chord + seamless raised-cosine ambient drone (`NativeAudio.set_drone`)
  - Interactive Wayland window (`WASD`/Arrows orbit, `~` Console, `F2` Profiler, `F3` Memory)
    plus `--shots` headless PNG capture (`build/shots/flint_spark.png`).
"""
from std.sys import argv
from std.math import sin, cos
from flint.core.math import Vec3
from flint.core.jobs import JobPool, default_worker_count
from flint.core.ecs import ArchetypeRegistry, ArchetypeSlots, SimdColumn, run_parallel_ctx
from flint.assets.image import write_png
from flint.platform.os import get_time
from flint.platform.input import InputState
from flint.platform.wayland import WaylandWindow
from flint.platform.audio import NativeAudio, WavClip
from flint.render.simd_canvas import Camera3D, SimdCanvas, pack_argb
from flint.devtools.overlays import FlintDevOverlays

comptime SPARK_COUNT: Int = 2048
comptime SPARK_CHUNK: Int = 256


@fieldwise_init
struct SparkSimCtx(Copyable, ImplicitlyCopyable, Movable):
    var angle_addr: Int
    var radius_addr: Int
    var height_addr: Int
    var speed_addr: Int
    var out_x_addr: Int
    var out_y_addr: Int
    var out_z_addr: Int
    var time_sec: Float32
    var dt: Float32


def step_spark_chunk(ctx_addr: Int, chunk_idx: Int):
    var ctx = Pointer[SparkSimCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var start = chunk_idx * SPARK_CHUNK
    var end = min(SPARK_COUNT, start + SPARK_CHUNK)
    var ang_p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.angle_addr)
    var rad_p = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=ctx.radius_addr)
    var hgt_p = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=ctx.height_addr)
    var spd_p = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=ctx.speed_addr)
    var ox_p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.out_x_addr)
    var oy_p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.out_y_addr)
    var oz_p = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.out_z_addr)

    for i in range(start, end):
        var a = ang_p[unsafe_offset=i] + spd_p[unsafe_offset=i] * ctx.dt
        ang_p[unsafe_offset=i] = a
        var r = rad_p[unsafe_offset=i] + Float32(0.28) * sin(ctx.time_sec * Float32(1.7) + Float32(i) * Float32(0.05))
        ox_p[unsafe_offset=i] = cos(a) * r
        oz_p[unsafe_offset=i] = sin(a) * r
        oy_p[unsafe_offset=i] = hgt_p[unsafe_offset=i] + Float32(0.42) * sin(a * Float32(2.0) + ctx.time_sec)


def make_spark_sprite(core_col: UInt32, glow_col: UInt32) -> List[UInt32]:
    var px = List[UInt32](length=64, fill=0)
    for y in range(8):
        for x in range(8):
            var dx = abs(Float32(x) - Float32(3.5))
            var dy = abs(Float32(y) - Float32(3.5))
            var manhattan = dx + dy
            if manhattan < Float32(1.6):
                px[y * 8 + x] = core_col | 0xFF000000
            elif manhattan < Float32(3.4):
                px[y * 8 + x] = glow_col | 0xFF000000
    return px^


def draw_crystal_scene(mut canvas: SimdCanvas, cam: Camera3D, t: Float32):
    var light_dir = Vec3(0.48, 0.76, 0.44).normalized()
    for g in range(-8, 9):
        var gf = Float32(g) * Float32(1.25)
        var col = UInt32(0x2D4663) if g == 0 else UInt32(0x1B2B3E)
        canvas.draw_line_3d(cam, Vec3(gf, -1.6, -10.0), Vec3(gf, -1.6, 10.0), col, 1)
        canvas.draw_line_3d(cam, Vec3(-10.0, -1.6, gf), Vec3(10.0, -1.6, gf), col, 1)

    var top = Vec3(0.0, 2.35 + Float32(0.18) * sin(t * Float32(2.1)), 0.0)
    var bot = Vec3(0.0, -1.05 + Float32(0.18) * sin(t * Float32(2.1)), 0.0)
    var ring = List[Vec3]()
    for k in range(4):
        var ang = t * Float32(0.85) + Float32(k) * Float32(1.5707963)
        ring.append(Vec3(cos(ang) * Float32(1.55), Float32(0.65) + Float32(0.18) * sin(t * Float32(2.1)), sin(ang) * Float32(1.55)))
    var facet_cols = [UInt32(0xFF9F38), UInt32(0xFF6B35), UInt32(0x38B6FF), UInt32(0x4AD295)]
    for k in range(4):
        var nk = (k + 1) & 3
        canvas.draw_tri_3d_lit(cam, top, ring[k], ring[nk], facet_cols[k], light_dir)
        canvas.draw_tri_3d_lit(cam, bot, ring[nk], ring[k], facet_cols[(k + 1) & 3], light_dir)
        canvas.draw_line_3d(cam, top, ring[k], 0xFFE6A3, 1)
        canvas.draw_line_3d(cam, bot, ring[k], 0x8BE0FF, 1)


def main():
    var headless_shots = False
    for i in range(1, len(argv())):
        if String(argv()[i]) == "--shots":
            headless_shots = True

    var reg = ArchetypeRegistry()
    var spark_arch = reg.register("flint_spark")
    var slots = ArchetypeSlots(spark_arch, SPARK_COUNT)
    var col_angle = SimdColumn[DType.float32](SPARK_COUNT)
    var col_radius = SimdColumn[DType.float32](SPARK_COUNT)
    var col_height = SimdColumn[DType.float32](SPARK_COUNT)
    var col_speed = SimdColumn[DType.float32](SPARK_COUNT)
    var col_x = SimdColumn[DType.float32](SPARK_COUNT)
    var col_y = SimdColumn[DType.float32](SPARK_COUNT)
    var col_z = SimdColumn[DType.float32](SPARK_COUNT)

    for i in range(SPARK_COUNT):
        _ = slots.spawn()
        var fi = Float32(i)
        col_angle.set(i, fi * Float32(0.137))
        col_radius.set(i, Float32(2.3) + Float32(i % 97) * Float32(0.068))
        col_height.set(i, Float32(-1.1) + Float32((i * 13) % 64) * Float32(0.055))
        col_speed.set(i, Float32(0.45) + Float32(i % 29) * Float32(0.032))

    var amber_sp = make_spark_sprite(0xFFF3C4, 0xFF8C2B)
    var cyan_sp = make_spark_sprite(0xE0FFFF, 0x2EB5FF)
    var pool = JobPool()
    _ = pool.start(max(1, default_worker_count()))

    var w = 1280
    var h = 720
    var canvas = SimdCanvas(w, h)
    var overlays = FlintDevOverlays()
    overlays.show_profiler = True
    overlays.show_memory = True

    if headless_shots:
        var t_sim = Float32(1.35)
        var sim_ctx = SparkSimCtx(
            col_angle.base_addr(), col_radius.base_addr(), col_height.base_addr(), col_speed.base_addr(),
            col_x.base_addr(), col_y.base_addr(), col_z.base_addr(), t_sim, Float32(0.75),
        )
        var t0 = get_time()
        run_parallel_ctx(pool, SPARK_COUNT // SPARK_CHUNK, sim_ctx, step_spark_chunk)
        col_angle.keep_alive()
        col_radius.keep_alive()
        col_height.keep_alive()
        col_speed.keep_alive()
        var t1 = get_time()
        var cam = Camera3D(w, h, Vec3(6.8, 4.1, 8.6), Vec3(0.0, 0.45, 0.0), 54.0)
        canvas.clear(0x0D1726, 0x1B2B44)
        draw_crystal_scene(canvas, cam, t_sim)
        for i in range(SPARK_COUNT):
            var proj = cam.project(Vec3(col_x.get(i), col_y.get(i), col_z.get(i)))
            if proj[3]:
                var sp_addr = Int(amber_sp.unsafe_ptr()) if (i & 1) == 0 else Int(cyan_sp.unsafe_ptr())
                canvas.blit_sprite_simd(sp_addr, 8, 8, Int(proj[0]) - 4, Int(proj[1]) - 4, proj[2])
        overlays.tick_us = max(1, Int((t1 - t0) * 1_000_000.0))
        overlays.render_us = max(1, Int((get_time() - t1) * 1_000_000.0))
        canvas.draw_panel(14, 8, 686, 30, 0x101824, 0xF49E38)
        canvas.draw_text(24, 15, "FLINT ENGINE 0.1  |  3D LIT MESH + 2048 SIMD DEPTH SPRITES", 0xFFE082, 2)
        overlays.render(canvas)
        _ = write_png("build/shots/flint_spark.png", canvas.pixels, w, h)
        pool.stop()
        print("[Flint] Wrote headless showcase frame -> build/shots/flint_spark.png")
        return

    var audio = NativeAudio()
    if audio.initialize():
        var drone = WavClip()
        drone.valid = True
        drone.channels = 1
        drone.sample_rate = 48000
        for s in range(48000):
            var ph = Float32(s) / Float32(48000.0) * Float32(6.2831853)
            var wave = sin(ph * Float32(110.0)) * Float32(0.55) + sin(ph * Float32(165.0)) * Float32(0.30)
            drone.samples.append(Int16(Int(wave * Float32(12000.0))))
        audio.clips.append(drone^)
        audio.drone_clip = len(audio.clips) - 1
        audio.set_drone(0.55)

    var win = WaylandWindow()
    var input = InputState()
    if not win.initialize(w, h):
        pool.stop()
        audio.close()
        print("[Flint] No Wayland display found. Run with '--shots' for headless PNG output.")
        return

    var yaw = Float32(0.68)
    var pitch = Float32(0.36)
    var dist = Float32(10.5)
    var start_time = get_time()
    var last_time = start_time

    while not win.close_requested:
        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - last_time)))
        last_time = now
        var elapsed = Float32(now - start_time)

        input.begin_frame()
        win.poll(input)
        if input.key_pressed(256):
            break
        if not overlays.handle_input(input):
            if input.key_down(65) or input.key_down(263):
                yaw -= Float32(1.4) * dt
            if input.key_down(68) or input.key_down(262):
                yaw += Float32(1.4) * dt
            if input.key_down(87) or input.key_down(265):
                pitch = min(Float32(1.2), pitch + Float32(1.0) * dt)
            if input.key_down(83) or input.key_down(264):
                pitch = max(Float32(-0.2), pitch - Float32(1.0) * dt)

        var sim_t0 = get_time()
        var sim_ctx = SparkSimCtx(
            col_angle.base_addr(), col_radius.base_addr(), col_height.base_addr(), col_speed.base_addr(),
            col_x.base_addr(), col_y.base_addr(), col_z.base_addr(), elapsed, dt,
        )
        run_parallel_ctx(pool, SPARK_COUNT // SPARK_CHUNK, sim_ctx, step_spark_chunk)
        col_angle.keep_alive()
        col_radius.keep_alive()
        col_height.keep_alive()
        col_speed.keep_alive()
        var ren_t0 = get_time()

        var eye = Vec3(cos(yaw) * cos(pitch) * dist, sin(pitch) * dist + Float32(1.0), sin(yaw) * cos(pitch) * dist)
        var cam = Camera3D(w, h, eye, Vec3(0.0, 0.45, 0.0), 54.0)
        canvas.clear(0x0D1726, 0x1B2B44)
        draw_crystal_scene(canvas, cam, elapsed)
        for i in range(SPARK_COUNT):
            var proj = cam.project(Vec3(col_x.get(i), col_y.get(i), col_z.get(i)))
            if proj[3]:
                var sp_addr = Int(amber_sp.unsafe_ptr()) if (i & 1) == 0 else Int(cyan_sp.unsafe_ptr())
                canvas.blit_sprite_simd(sp_addr, 8, 8, Int(proj[0]) - 4, Int(proj[1]) - 4, proj[2])

        overlays.tick_us = Int((ren_t0 - sim_t0) * 1_000_000.0)
        overlays.render_us = Int((get_time() - ren_t0) * 1_000_000.0)
        overlays.fps = Int(Float32(1.0) / dt)
        canvas.draw_panel(14, 8, 864, 30, 0x101824, 0xF49E38)
        canvas.draw_text(24, 15, "FLINT ENGINE  |  WASD ORBIT  |  [~] CONSOLE  [F2] PROFILER  [F3] MEMORY", 0xFFE082, 2)
        overlays.render(canvas)
        win.present(canvas.pixels, w, h)
        audio.update(0.45)

    audio.close()
    win.close()
    pool.stop()

"""Flint Engine v3: `64K` GPU Compute Particle & Ribbon Pipeline (`aaa_gpu_particles.mojo`).

Implements a zero-CPU-readback, two-pass `16 x 16` tile-binned GPU particle system stored in
`BIND_AAA_OIT_NODES` (preserving the unified 23-binding AAA descriptor layout):
  1. `gpu_particle_clear_tiles_shader`: Clears `16 x 16` screen-tile bin counts.
  2. `gpu_particle_sim_and_bin_shader` (`65,536` persistent particles):
     - Deterministic ring-buffer spawning from up to `64` `AaaParticleEmitter`s per frame
     - Drag + gravity + 3D curl-noise turbulence + ground/screen-depth restitution bounce
     - Frustum culling & atomic binning into `16 x 16` screen tiles
  3. `gpu_particle_tile_composite_shader` (`group = 16 x 16` with LDS shared-memory preload):
     - `PARTICLE_SPARK_STREAK (0)`: Velocity-stretched blackbody capsule SDF + firework glitter flicker
     - `PARTICLE_WATER_SPRAY (1)`: FBM-eroded wet-tire rooster-tail mist lit live from the `160x90x64` Froxel Prefix-LUT
     - `PARTICLE_DRIFT_SMOKE (2)`: Billowing FBM tire/coaster smoke with soft depth & SH9/sun lighting
     - `PARTICLE_EXHAUST_PLASMA (3)`: Supersonic Mach shock-diamond exhaust plasma cones
     - `PARTICLE_GENERIC (4)`: Soft round (or velocity-stretched) sprite whose color, size and alpha
       blend from start to end over its life; additive or alpha blended, optionally sky/sun lit
     - Writes disocclusion-reactive weights to `b_oit` so `FlintTSR v3` never ghosts moving particles!
"""
from std.math import max, min
from std.memory import bitcast
from flint.core.math import Vec3
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_PUSH_WORDS,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_HISTORY,
    BIND_AAA_OIT_NODES,
    emit_vec3_dot,
)
from flint.render.aaa_ibl import emit_eval_sh9_irradiance
from flint.render.aaa_oit_volumetrics import AAA_OIT_REACTIVE_BASE_WORDS
from flint.render.aaa_froxel_volumetrics import emit_sample_froxel_prefix_lut_spv

comptime AAA_MAX_PARTICLES: Int = 65536
comptime MAX_PARTICLES: Int = 65536
comptime AAA_PARTICLE_WORDS: Int = 16
comptime AAA_MAX_PARTICLE_EMITTERS: Int = 64
comptime AAA_EMITTER_WORDS: Int = 24
comptime AAA_MAX_PARTICLES_PER_TILE: Int = 48

# Particle visual & physical archetypes (exported under both `PARTICLE_*` and `ARCHETYPE_*` names)
comptime PARTICLE_SPARK_STREAK: UInt32 = 0
comptime PARTICLE_WATER_SPRAY: UInt32 = 1
comptime PARTICLE_DRIFT_SMOKE: UInt32 = 2
comptime PARTICLE_EXHAUST_PLASMA: UInt32 = 3
comptime PARTICLE_GENERIC: UInt32 = 4

comptime ARCHETYPE_SPARK: UInt32 = 0
comptime ARCHETYPE_WATER_SPRAY: UInt32 = 1
comptime ARCHETYPE_DRIFT_SMOKE: UInt32 = 2
comptime ARCHETYPE_EXHAUST_PLASMA: UInt32 = 3
comptime ARCHETYPE_GENERIC: UInt32 = 4

# `PARTICLE_GENERIC` flags (`AaaParticleEmitter.flags`)
comptime PARTICLE_FLAG_ADDITIVE: UInt32 = 1  # add light (sparks, glows); otherwise alpha blend over the scene
comptime PARTICLE_FLAG_LIT: UInt32 = 2  # alpha-blended particles take sky + sun light (dust, smoke); otherwise emissive
comptime PARTICLE_FLAG_STRETCH: UInt32 = 4  # stretch along the velocity (streaks)
# Low mantissa bits of the bounce height that carry the archetype (word 14 of an emitter, 11 of a particle)
comptime _ARCH_MASK: UInt32 = 7
comptime _BOUNCE_MASK: UInt32 = 0xFFFFFFF8


@always_inline
def gpu_particles_base_word_offset(max_pix: Int) -> Int:
    """Base word offset in `BIND_AAA_OIT_NODES` after OIT plumes (`3072`) and reactive mask (`max_pix`)."""
    return 4096 + max_pix


@always_inline
def gpu_particles_emitters_word_offset(max_pix: Int) -> Int:
    return gpu_particles_base_word_offset(max_pix) + 16


@always_inline
def gpu_particles_pool_word_offset(max_pix: Int) -> Int:
    return gpu_particles_emitters_word_offset(max_pix) + AAA_MAX_PARTICLE_EMITTERS * AAA_EMITTER_WORDS


@always_inline
def gpu_particles_tile_counts_word_offset(max_pix: Int) -> Int:
    return gpu_particles_pool_word_offset(max_pix) + AAA_MAX_PARTICLES * AAA_PARTICLE_WORDS


@always_inline
def gpu_particles_tile_count(width: Int, height: Int) -> Int:
    return ((width + 15) // 16) * ((height + 15) // 16)


@always_inline
def gpu_particles_tile_bins_word_offset(max_pix: Int, num_tiles: Int) -> Int:
    return gpu_particles_tile_counts_word_offset(max_pix) + num_tiles


@always_inline
def gpu_particles_total_oit_words(max_pix: Int, num_tiles: Int) -> Int:
    return gpu_particles_tile_bins_word_offset(max_pix, num_tiles) + num_tiles * AAA_MAX_PARTICLES_PER_TILE


struct AaaParticleEmitter(Copyable, Movable):
    """CPU submission descriptor for a GPU compute particle burst or continuous stream."""
    var origin: Vec3
    var spawn_radius: Float32
    var velocity: Vec3
    var velocity_spread: Float32
    var color: Vec3
    var size: Float32
    var lifetime: Float32
    var gravity_scale: Float32
    var drag: Float32
    var bounce_ground_y: Float32
    var spawn_count: Int
    var archetype: UInt32
    # `PARTICLE_GENERIC` only: color, size and alpha blend from start to end over each particle's life
    var end_color: Vec3
    var end_size: Float32
    var alpha: Float32
    var end_alpha: Float32
    var flags: UInt32

    def __init__(
        out self,
        origin: Vec3 = Vec3(0.0, 0.0, 0.0),
        velocity: Vec3 = Vec3(0.0, 2.0, 0.0),
        color: Vec3 = Vec3(1.0, 0.85, 0.55),
        size: Float32 = 0.08,
        lifetime: Float32 = 0.75,
        spawn_count: Int = 32,
        archetype: UInt32 = PARTICLE_SPARK_STREAK,
        spawn_radius: Float32 = 0.08,
        velocity_spread: Float32 = 2.5,
        gravity_scale: Float32 = 1.0,
        drag: Float32 = 1.2,
        bounce_ground_y: Float32 = -1e6,
        end_color: Vec3 = Vec3(-1.0, -1.0, -1.0),
        end_size: Float32 = -1.0,
        alpha: Float32 = 1.0,
        end_alpha: Float32 = 1.0,
        flags: UInt32 = 0,
    ):
        """`end_color` / `end_size` below zero mean "same as the start"."""
        self.origin = origin
        self.spawn_radius = max(Float32(0.0), spawn_radius)
        self.velocity = velocity
        self.velocity_spread = max(Float32(0.0), velocity_spread)
        self.color = color
        self.size = max(Float32(0.008), size)
        self.lifetime = max(Float32(0.02), lifetime)
        self.gravity_scale = gravity_scale
        self.drag = max(Float32(0.0), drag)
        self.bounce_ground_y = bounce_ground_y
        self.spawn_count = max(0, min(8191, spawn_count))
        self.archetype = min(archetype, PARTICLE_GENERIC)
        self.end_color = color if end_color.x < 0 or end_color.y < 0 or end_color.z < 0 else end_color
        self.end_size = self.size if end_size < 0 else max(Float32(0.004), end_size)
        self.alpha = max(Float32(0.0), min(Float32(1.0), alpha))
        self.end_alpha = max(Float32(0.0), min(Float32(1.0), end_alpha))
        self.flags = flags & UInt32(7)

    @staticmethod
    def generic(
        origin: Vec3,
        velocity: Vec3,
        spawn_count: Int,
        color: Vec3 = Vec3(1.0, 1.0, 1.0),
        end_color: Vec3 = Vec3(-1.0, -1.0, -1.0),
        size: Float32 = 0.05,
        end_size: Float32 = -1.0,
        alpha: Float32 = 1.0,
        end_alpha: Float32 = 0.0,
        lifetime: Float32 = 1.0,
        spawn_radius: Float32 = 0.0,
        velocity_spread: Float32 = 1.0,
        gravity_scale: Float32 = 1.0,
        drag: Float32 = 0.0,
        additive: Bool = True,
        lit: Bool = False,
        stretch: Bool = False,
        bounce_ground_y: Float32 = -1e6,
    ) -> AaaParticleEmitter:
        """A `PARTICLE_GENERIC` emitter: `spawn_count` particles this frame, each living about
        `lifetime` s (+-20%), starting at `velocity` plus a random spherical kick of up to
        `velocity_spread` m/s; `gravity_scale` 1 = Earth gravity, negative rises; `drag` 1/s."""
        var fl = UInt32(0)
        if additive:
            fl |= PARTICLE_FLAG_ADDITIVE
        if lit:
            fl |= PARTICLE_FLAG_LIT
        if stretch:
            fl |= PARTICLE_FLAG_STRETCH
        return AaaParticleEmitter(
            origin=origin,
            velocity=velocity,
            color=color,
            size=size,
            lifetime=lifetime,
            spawn_count=spawn_count,
            archetype=PARTICLE_GENERIC,
            spawn_radius=spawn_radius,
            velocity_spread=velocity_spread,
            gravity_scale=gravity_scale,
            drag=drag,
            bounce_ground_y=bounce_ground_y,
            end_color=end_color,
            end_size=end_size,
            alpha=alpha,
            end_alpha=end_alpha,
            flags=fl,
        )

    @staticmethod
    def spark_burst(
        origin: Vec3,
        velocity: Vec3,
        color: Vec3 = Vec3(8.5, 4.2, 1.1),
        spawn_count: Int = 48,
        spread: Float32 = 5.5,
        lifetime: Float32 = 0.85,
        ground_y: Float32 = 0.0,
    ) -> AaaParticleEmitter:
        return AaaParticleEmitter(
            origin=origin,
            velocity=velocity,
            color=color,
            size=Float32(0.045),
            lifetime=lifetime,
            spawn_count=spawn_count,
            archetype=PARTICLE_SPARK_STREAK,
            spawn_radius=Float32(0.06),
            velocity_spread=spread,
            gravity_scale=Float32(1.0),
            drag=Float32(0.85),
            bounce_ground_y=ground_y,
        )

    @staticmethod
    def sparks(
        origin: Vec3,
        velocity: Vec3,
        color: Vec3 = Vec3(8.5, 4.2, 1.1),
        spawn_count: Int = 48,
        spread: Float32 = 5.5,
        lifetime: Float32 = 0.85,
        ground_y: Float32 = 0.0,
    ) -> AaaParticleEmitter:
        return AaaParticleEmitter.spark_burst(origin, velocity, color, spawn_count, spread, lifetime, ground_y)

    @staticmethod
    def water_spray(
        origin: Vec3,
        velocity: Vec3,
        spawn_count: Int = 24,
        size: Float32 = 0.42,
        lifetime: Float32 = 0.55,
        tint: Vec3 = Vec3(0.78, 0.86, 0.95),
    ) -> AaaParticleEmitter:
        return AaaParticleEmitter(
            origin=origin,
            velocity=velocity,
            color=tint,
            size=size,
            lifetime=lifetime,
            spawn_count=spawn_count,
            archetype=PARTICLE_WATER_SPRAY,
            spawn_radius=Float32(0.16),
            velocity_spread=Float32(2.2),
            gravity_scale=Float32(0.45),
            drag=Float32(2.6),
            bounce_ground_y=Float32(-1e6),
        )

    @staticmethod
    def drift_smoke(
        origin: Vec3,
        velocity: Vec3,
        spawn_count: Int = 18,
        size: Float32 = 0.65,
        lifetime: Float32 = 0.95,
        tint: Vec3 = Vec3(0.72, 0.74, 0.78),
    ) -> AaaParticleEmitter:
        return AaaParticleEmitter(
            origin=origin,
            velocity=velocity,
            color=tint,
            size=size,
            lifetime=lifetime,
            spawn_count=spawn_count,
            archetype=PARTICLE_DRIFT_SMOKE,
            spawn_radius=Float32(0.20),
            velocity_spread=Float32(1.6),
            gravity_scale=Float32(-0.18),
            drag=Float32(2.2),
            bounce_ground_y=Float32(-1e6),
        )

    @staticmethod
    def exhaust_plasma(
        origin: Vec3,
        velocity: Vec3,
        color: Vec3 = Vec3(2.4, 4.8, 12.0),
        spawn_count: Int = 16,
        size: Float32 = 0.14,
        lifetime: Float32 = 0.14,
    ) -> AaaParticleEmitter:
        return AaaParticleEmitter(
            origin=origin,
            velocity=velocity,
            color=color,
            size=size,
            lifetime=lifetime,
            spawn_count=spawn_count,
            archetype=PARTICLE_EXHAUST_PLASMA,
            spawn_radius=Float32(0.03),
            velocity_spread=Float32(0.65),
            gravity_scale=Float32(0.0),
            drag=Float32(0.5),
            bounce_ground_y=Float32(-1e6),
        )

    def append_words(self, mut out: List[UInt32], ring_start: Int):
        out.append(bitcast[DType.uint32](self.origin.x))
        out.append(bitcast[DType.uint32](self.origin.y))
        out.append(bitcast[DType.uint32](self.origin.z))
        out.append(bitcast[DType.uint32](self.spawn_radius))
        out.append(bitcast[DType.uint32](self.velocity.x))
        out.append(bitcast[DType.uint32](self.velocity.y))
        out.append(bitcast[DType.uint32](self.velocity.z))
        out.append(bitcast[DType.uint32](self.velocity_spread))
        out.append(bitcast[DType.uint32](self.color.x))
        out.append(bitcast[DType.uint32](self.color.y))
        out.append(bitcast[DType.uint32](self.color.z))
        out.append(bitcast[DType.uint32](self.size))
        out.append(bitcast[DType.uint32](self.lifetime))
        out.append(bitcast[DType.uint32](self.gravity_scale))
        var clean_by = bitcast[DType.uint32](self.bounce_ground_y) & _BOUNCE_MASK
        out.append(clean_by | (self.archetype & _ARCH_MASK))
        var drag_cl = max(Float32(0.0), min(Float32(15.0), self.drag))
        var drag_u4 = UInt32(Int(drag_cl * Float32(16.0) + Float32(0.5))) & UInt32(0xFF)
        var cnt_u8 = UInt32(min(255, max(0, self.spawn_count)))
        # Word 14 holds bounce_y | archetype; Word 15 holds `(ring_start << 16) | (spawn_cnt_12 << 4) | drag_hi`
        var cnt_u12 = UInt32(min(4095, max(0, self.spawn_count)))
        var drag_nib = UInt32(min(15, Int(drag_cl * Float32(2.0) + Float32(0.5))))
        _ = drag_u4
        _ = cnt_u8
        var meta15 = ((UInt32(ring_start) & UInt32(0xFFFF)) << 16) | (cnt_u12 << 4) | drag_nib
        out.append(meta15)
        # Words 16..23: end color, end size, start / end alpha, generic flags, (reserved)
        out.append(bitcast[DType.uint32](self.end_color.x))
        out.append(bitcast[DType.uint32](self.end_color.y))
        out.append(bitcast[DType.uint32](self.end_color.z))
        out.append(bitcast[DType.uint32](self.end_size))
        out.append(bitcast[DType.uint32](self.alpha))
        out.append(bitcast[DType.uint32](self.end_alpha))
        out.append(self.flags)
        out.append(UInt32(0))


@always_inline
def _emit_pcg_f01(mut s: AaaSpirvModule, seed_in: UInt32) -> Tuple[UInt32, UInt32]:
    var st = s.iadd(s.imul(seed_in, s.uint(747796405)), s.uint(2891336453))
    var wd = s.imul(s.bit_xor(s.shr(st, s.iadd(s.shr(st, s.uint(28)), s.uint(4))), st), s.uint(277803737))
    var res = s.bit_xor(s.shr(wd, s.uint(22)), wd)
    var f01 = s.fmul(s.float_from_uint(s.bit_and(res, s.uint(0x00FFFFFF))), s.float(1.0 / 16777216.0))
    return (res, f01)


def gpu_particle_clear_tiles_shader() -> List[UInt32]:
    """Clears `16 x 16` screen-tile particle counters before simulation (`group = 64 x 1`).
    Push[0] = tile_counts_off, Push[1] = num_tiles.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)
    var tid = s.invocation_axis(0)
    var counts_off = s.push_word(0)
    var num_tiles = s.push_word(1)
    s.return_if(s.uge(tid, num_tiles))
    s.store_word(b_oit, s.iadd(counts_off, tid), s.uint(0))
    return s.finish()


def gpu_particle_sim_and_bin_shader() -> List[UInt32]:
    """Pass 1 (`65,536` threads, `group = 64 x 1`, `1024` workgroups):
    Spawns ring-buffered particles from up to `64` CPU emitters, integrates drag + gravity +
    3D curl turbulence + wet-road/depth restitution bounce, and bins into `16 x 16` screen tiles!
    Push[0]=render_w, Push[1]=render_h, Push[2]=dt_sec, Push[3]=emitter_count,
    Push[4]=emitters_off, Push[5]=pool_off, Push[6]=tile_counts_off, Push[7]=tile_bins_off.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=64, group_height=1, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)

    var pid = s.invocation_axis(0)
    s.return_if(s.uge(pid, s.uint(UInt32(AAA_MAX_PARTICLES))))

    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var dt = s.nclamp(s.push_float(2), s.float(0.0005), s.float(0.10))
    var em_cnt = s.umin(s.push_word(3), s.uint(UInt32(AAA_MAX_PARTICLE_EMITTERS)))
    var em_base_off = s.push_word(4)
    var pool_off = s.push_word(5)
    var counts_off = s.push_word(6)
    var bins_off = s.push_word(7)

    var frame_idx = s.load_word(b_globals, s.uint(100))
    var p_off = s.iadd(pool_off, s.imul(pid, s.uint(UInt32(AAA_PARTICLE_WORDS))))

    # 1. Check if `pid` is newly spawned by any active emitter `[0 .. em_cnt - 1]` this frame
    var ei_v = s.uvar(s.uint(0))
    s.begin_loop()
    var ei = s.load_u(ei_v)
    s.break_if(s.uge(ei, em_cnt))
    var e_off = s.iadd(em_base_off, s.imul(ei, s.uint(UInt32(AAA_EMITTER_WORDS))))
    var meta15 = s.load_word(b_oit, s.iadd(e_off, s.uint(15)))
    var ring_st = s.shr(meta15, s.uint(16))
    var sp_cnt = s.bit_and(s.shr(meta15, s.uint(4)), s.uint(0xFFF))
    var rel_slot = s.bit_and(s.isub(pid, ring_st), s.uint(0xFFFF))

    s.begin_if(s.ult(rel_slot, sp_cnt))
    var seed0 = s.bit_xor(s.imul(pid, s.uint(747796405)), s.imul(s.iadd(frame_idx, s.uint(1)), s.uint(19349663)))
    var r0 = _emit_pcg_f01(s, seed0)
    var r1 = _emit_pcg_f01(s, r0[0])
    var r2 = _emit_pcg_f01(s, r1[0])
    var r3 = _emit_pcg_f01(s, r2[0])
    var r4 = _emit_pcg_f01(s, r3[0])
    var r5 = _emit_pcg_f01(s, r4[0])

    var ox = s.load_float(b_oit, e_off)
    var oy = s.load_float(b_oit, s.iadd(e_off, s.uint(1)))
    var oz = s.load_float(b_oit, s.iadd(e_off, s.uint(2)))
    var srad = s.load_float(b_oit, s.iadd(e_off, s.uint(3)))
    var vx0 = s.load_float(b_oit, s.iadd(e_off, s.uint(4)))
    var vy0 = s.load_float(b_oit, s.iadd(e_off, s.uint(5)))
    var vz0 = s.load_float(b_oit, s.iadd(e_off, s.uint(6)))
    var vsp = s.load_float(b_oit, s.iadd(e_off, s.uint(7)))
    var cr_e = s.load_float(b_oit, s.iadd(e_off, s.uint(8)))
    var cg_e = s.load_float(b_oit, s.iadd(e_off, s.uint(9)))
    var cb_e = s.load_float(b_oit, s.iadd(e_off, s.uint(10)))
    var sz_e = s.load_float(b_oit, s.iadd(e_off, s.uint(11)))
    var life_e = s.load_float(b_oit, s.iadd(e_off, s.uint(12)))
    var grav_e = s.load_float(b_oit, s.iadd(e_off, s.uint(13)))
    var w14_e = s.load_word(b_oit, s.iadd(e_off, s.uint(14)))
    var drag_e = s.fmul(s.float_from_uint(s.bit_and(meta15, s.uint(0xF))), s.float(0.5))

    # Uniform spherical jitter for natural non-boxy radial bursts!
    var z_sp = s.fma(r0[1], s.float(2.0), s.float(-1.0))
    var phi_sp = s.fmul(r1[1], s.float(6.2831853))
    var r_xy = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fmul(z_sp, z_sp))))
    var dx_s = s.fmul(r_xy, s.fcos(phi_sp))
    var dy_s = z_sp
    var dz_s = s.fmul(r_xy, s.fsin(phi_sp))
    var rad_j = s.fmul(srad, s.fsqrt(r2[1]))
    var spd_j = s.fmul(vsp, s.fma(r3[1], s.float(0.75), s.float(0.25)))

    var sp_px = s.fma(dx_s, rad_j, ox)
    var sp_py = s.fma(dy_s, rad_j, oy)
    var sp_pz = s.fma(dz_s, rad_j, oz)
    var sp_vx = s.fma(dx_s, spd_j, vx0)
    var sp_vy = s.fma(dy_s, spd_j, vy0)
    var sp_vz = s.fma(dz_s, spd_j, vz0)
    var life_j = s.fmul(life_e, s.fma(r4[1], s.float(0.40), s.float(0.80)))
    var sz_jit = s.fma(r5[1], s.float(0.36), s.float(0.82))
    var sz_j = s.fmul(sz_e, sz_jit)
    # Sub-frame staggered initial age `[0 .. 0.5*dt]` prevents discrete emission banding at high speed
    var init_age = s.fmul(r0[1], s.fmul(dt, s.float(0.5)))

    s.store_float(b_oit, p_off, sp_px)
    s.store_float(b_oit, s.iadd(p_off, s.uint(1)), sp_py)
    s.store_float(b_oit, s.iadd(p_off, s.uint(2)), sp_pz)
    s.store_float(b_oit, s.iadd(p_off, s.uint(3)), sp_vx)
    s.store_float(b_oit, s.iadd(p_off, s.uint(4)), sp_vy)
    s.store_float(b_oit, s.iadd(p_off, s.uint(5)), sp_vz)
    s.store_float(b_oit, s.iadd(p_off, s.uint(6)), init_age)
    s.store_float(b_oit, s.iadd(p_off, s.uint(7)), life_j)
    s.store_word(b_oit, s.iadd(p_off, s.uint(8)), s.pack_f16x2(cr_e, cg_e))
    s.store_word(b_oit, s.iadd(p_off, s.uint(9)), s.pack_f16x2(cb_e, sz_j))
    s.store_word(b_oit, s.iadd(p_off, s.uint(10)), s.pack_f16x2(grav_e, drag_e))
    s.store_word(b_oit, s.iadd(p_off, s.uint(11)), w14_e)
    # Generic over-life words: end color (12, 13), end size (13), start / end alpha (14), flags (15)
    var ecr = s.load_float(b_oit, s.iadd(e_off, s.uint(16)))
    var ecg = s.load_float(b_oit, s.iadd(e_off, s.uint(17)))
    var ecb = s.load_float(b_oit, s.iadd(e_off, s.uint(18)))
    var esz = s.fmul(s.load_float(b_oit, s.iadd(e_off, s.uint(19))), sz_jit)
    s.store_word(b_oit, s.iadd(p_off, s.uint(12)), s.pack_f16x2(ecr, ecg))
    s.store_word(b_oit, s.iadd(p_off, s.uint(13)), s.pack_f16x2(ecb, esz))
    s.store_word(b_oit, s.iadd(p_off, s.uint(14)), s.pack_f16x2(s.load_float(b_oit, s.iadd(e_off, s.uint(20))), s.load_float(b_oit, s.iadd(e_off, s.uint(21)))))
    s.store_word(b_oit, s.iadd(p_off, s.uint(15)), s.load_word(b_oit, s.iadd(e_off, s.uint(22))))
    s.end_if()

    s.store(ei_v, s.iadd(ei, s.uint(1)))
    s.end_loop()

    # 2. Simulate active particles (`lifetime > 0 && age < lifetime`)
    var life = s.load_float(b_oit, s.iadd(p_off, s.uint(7)))
    s.return_if(s.fle(life, s.float(1e-4)))
    var age = s.fadd(s.load_float(b_oit, s.iadd(p_off, s.uint(6))), dt)
    s.begin_if(s.fge(age, life))
    s.store_float(b_oit, s.iadd(p_off, s.uint(7)), s.float(0.0))
    s.end_if()
    s.return_if(s.fge(age, life))

    var px_w = s.fvar(s.load_float(b_oit, p_off))
    var py_w = s.fvar(s.load_float(b_oit, s.iadd(p_off, s.uint(1))))
    var pz_w = s.fvar(s.load_float(b_oit, s.iadd(p_off, s.uint(2))))
    var vx_w = s.fvar(s.load_float(b_oit, s.iadd(p_off, s.uint(3))))
    var vy_w = s.fvar(s.load_float(b_oit, s.iadd(p_off, s.uint(4))))
    var vz_w = s.fvar(s.load_float(b_oit, s.iadd(p_off, s.uint(5))))

    var p9 = s.unpack_f16x2(s.load_word(b_oit, s.iadd(p_off, s.uint(9))))
    var p10 = s.unpack_f16x2(s.load_word(b_oit, s.iadd(p_off, s.uint(10))))
    var w11 = s.load_word(b_oit, s.iadd(p_off, s.uint(11)))
    var p_size = p9[1]
    var grav_sc = p10[0]
    var drag_k = p10[1]
    var arch = s.bit_and(w11, s.uint(_ARCH_MASK))
    var bounce_y = s.bitcast_float(s.bit_and(w11, s.uint(_BOUNCE_MASK)))
    var is_gen = s.ieq(arch, s.uint(PARTICLE_GENERIC))
    var gen_flags = s.load_word(b_oit, s.iadd(p_off, s.uint(15)))

    var damp = s.exp2(s.fmul(s.fmul(drag_k, dt), s.float(-1.442695)))
    var nvx = s.fvar(s.fmul(s.load_f(vx_w), damp))
    var nvy = s.fvar(s.fma(s.float(-9.81), s.fmul(grav_sc, dt), s.fmul(s.load_f(vy_w), damp)))
    var nvz = s.fvar(s.fmul(s.load_f(vz_w), damp))

    # Ambient Wind Advection (`b_globals[128..130]`) + Divergence-Free 3D Curl Turbulence on spray/smoke/contrails
    var is_mist = s.logical_or(s.ieq(arch, s.uint(1)), s.ieq(arch, s.uint(2)))
    s.begin_if(is_mist)
    var wind_x = s.load_float(b_globals, s.uint(128))
    var wind_y = s.load_float(b_globals, s.uint(129))
    var wind_z = s.load_float(b_globals, s.uint(130))
    var wind_couple = s.fsub(s.float(1.0), damp)
    s.store(nvx, s.fma(wind_x, wind_couple, s.load_f(nvx)))
    s.store(nvy, s.fma(wind_y, wind_couple, s.load_f(nvy)))
    s.store(nvz, s.fma(wind_z, wind_couple, s.load_f(nvz)))

    var time_s = s.load_float(b_globals, s.uint(123))
    var wx_c = s.fsub(s.load_f(px_w), s.fmul(wind_x, s.fmul(time_s, s.float(0.25))))
    var wy_c = s.fsub(s.load_f(py_w), s.fmul(wind_y, s.fmul(time_s, s.float(0.25))))
    var wz_c = s.fsub(s.load_f(pz_w), s.fmul(wind_z, s.fmul(time_s, s.float(0.25))))
    var cx_t = s.fsin(s.fma(wy_c, s.float(1.7), s.fma(wz_c, s.float(1.3), s.fmul(time_s, s.float(2.4)))))
    var cy_t = s.fcos(s.fma(wx_c, s.float(1.5), s.fsub(s.fmul(wz_c, s.float(1.9)), s.fmul(time_s, s.float(2.1)))))
    var cz_t = s.fsin(s.fma(wx_c, s.float(1.8), s.fma(wy_c, s.float(-1.4), s.fmul(time_s, s.float(2.6)))))
    var turb_dt = s.fmul(dt, s.float(4.5))
    s.store(nvx, s.fma(cx_t, turb_dt, s.load_f(nvx)))
    s.store(nvy, s.fma(cy_t, s.fmul(turb_dt, s.float(0.65)), s.load_f(nvy)))
    s.store(nvz, s.fma(cz_t, turb_dt, s.load_f(nvz)))
    s.end_if()

    var npx = s.fvar(s.fma(s.load_f(nvx), dt, s.load_f(px_w)))
    var npy = s.fvar(s.fma(s.load_f(nvy), dt, s.load_f(py_w)))
    var npz = s.fvar(s.fma(s.load_f(nvz), dt, s.load_f(pz_w)))

    # Ground-plane restitution bounce for sparks and generic particles (`bounce_y > -1e5`)
    s.begin_if(s.logical_and(s.logical_or(s.ieq(arch, s.uint(0)), is_gen), s.fgt(bounce_y, s.float(-1e5))))
    var g_lim = s.fadd(bounce_y, s.float(0.02))
    s.begin_if(s.logical_and(s.flt(s.load_f(npy), g_lim), s.flt(s.load_f(nvy), s.float(0.0))))
    s.store(npy, g_lim)
    s.store(nvy, s.fmul(s.fneg(s.load_f(nvy)), s.float(0.58)))
    s.store(nvx, s.fmul(s.load_f(nvx), s.float(0.82)))
    s.store(nvz, s.fmul(s.load_f(nvz), s.float(0.82)))
    s.end_if()
    s.end_if()

    # Camera projection & screen-depth spark bounce + `16 x 16` tile binning
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var near_z = s.nmax(s.float(0.10), s.load_float(b_globals, s.uint(35)))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var tan_x = s.nmax(s.float(0.1), s.load_float(b_globals, s.uint(39)))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var tan_y = s.nmax(s.float(0.1), s.load_float(b_globals, s.uint(43)))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))

    var rx_c = s.fsub(s.load_f(npx), cam_x)
    var ry_c = s.fsub(s.load_f(npy), cam_y)
    var rz_c = s.fsub(s.load_f(npz), cam_z)
    var vz_c = emit_vec3_dot(s, rx_c, ry_c, rz_c, fwd_x, fwd_y, fwd_z)

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)

    s.begin_if(s.fgt(vz_c, near_z))
    var vx_c = emit_vec3_dot(s, rx_c, ry_c, rz_c, rgt_x, rgt_y, rgt_z)
    var vy_c = emit_vec3_dot(s, rx_c, ry_c, rz_c, up_x, up_y, up_z)
    var u_sc = s.fma(s.fdiv(vx_c, s.fmul(vz_c, tan_x)), s.float(0.5), s.float(0.5))
    var v_sc = s.fma(s.fneg(s.fdiv(vy_c, s.fmul(vz_c, tan_y))), s.float(0.5), s.float(0.5))

    # Screen-depth restitution bounce for sparks hitting walls/tracks/wet road (`arch == 0`)
    var in_scr = s.logical_and(
        s.logical_and(s.fgt(u_sc, s.float(0.01)), s.flt(u_sc, s.float(0.99))),
        s.logical_and(s.fgt(v_sc, s.float(0.01)), s.flt(v_sc, s.float(0.99))),
    )
    s.begin_if(s.logical_and(s.ieq(arch, s.uint(0)), in_scr))
    var spx_u = s.umin(s.isub(w_u, s.uint(1)), s.saturate_uint(s.fmul(u_sc, fw)))
    var spy_u = s.umin(s.isub(h_u, s.uint(1)), s.saturate_uint(s.fmul(v_sc, fh)))
    var h_b = s.imul(s.iadd(s.imul(spy_u, w_u), spx_u), s.uint(8))
    var z_s = s.load_float(b_hdr, s.iadd(h_b, s.uint(3)))
    var hit_depth = s.logical_and(s.fgt(vz_c, z_s), s.flt(vz_c, s.fadd(z_s, s.float(0.55))))
    s.begin_if(hit_depth)
    var snx = s.load_float(b_hdr, s.iadd(h_b, s.uint(6)))
    var sny = s.load_float(b_hdr, s.iadd(h_b, s.uint(7)))
    var v_dot_n = s.fma(s.load_f(nvx), snx, s.fmul(s.load_f(nvy), sny))
    s.begin_if(s.flt(v_dot_n, s.float(0.0)))
    s.store(nvx, s.fmul(s.fsub(s.load_f(nvx), s.fmul(s.float(1.6), s.fmul(v_dot_n, snx))), s.float(0.75)))
    s.store(nvy, s.fadd(s.fabs(s.fmul(s.load_f(nvy), s.float(0.52))), s.float(0.85)))
    s.store(nvz, s.fmul(s.load_f(nvz), s.float(0.75)))
    s.end_if()
    s.end_if()
    s.end_if()

    # Persist updated state
    s.store_float(b_oit, p_off, s.load_f(npx))
    s.store_float(b_oit, s.iadd(p_off, s.uint(1)), s.load_f(npy))
    s.store_float(b_oit, s.iadd(p_off, s.uint(2)), s.load_f(npz))
    s.store_float(b_oit, s.iadd(p_off, s.uint(3)), s.load_f(nvx))
    s.store_float(b_oit, s.iadd(p_off, s.uint(4)), s.load_f(nvy))
    s.store_float(b_oit, s.iadd(p_off, s.uint(5)), s.load_f(nvz))
    s.store_float(b_oit, s.iadd(p_off, s.uint(6)), age)

    # Expand puff size slightly over lifetime for mist/smoke (`arch == 1 || arch == 2`)
    var life_t = s.nclamp(s.fdiv(age, s.nmax(s.float(1e-3), life)), s.float(0.0), s.float(1.0))
    var gen_rad = s.fma(s.fsub(s.unpack_f16x2(s.load_word(b_oit, s.iadd(p_off, s.uint(13))))[1], p_size), life_t, p_size)
    var eff_rad_m = s.select_f(is_gen, gen_rad, s.select_f(is_mist, s.fmul(p_size, s.fma(life_t, s.float(1.15), s.float(0.65))), p_size))
    var px_per_m = s.fdiv(fh, s.nmax(s.float(0.05), s.fmul(s.float(2.0), s.fmul(vz_c, tan_y))))
    var rad_px = s.nclamp(s.fmul(eff_rad_m, px_per_m), s.float(1.5), s.float(30.0))

    # Include velocity streak tail in screen-space bounding box for `SPARK_STREAK (0)` & `EXHAUST_PLASMA (3)`
    var cx_px = s.fmul(u_sc, fw)
    var cy_px = s.fmul(v_sc, fh)
    var tx_px = s.fvar(cx_px)
    var ty_px = s.fvar(cy_px)
    var gen_stretch = s.logical_and(is_gen, s.ine(s.bit_and(gen_flags, s.uint(PARTICLE_FLAG_STRETCH)), s.uint(0)))
    var is_streak = s.logical_or(s.logical_or(s.ieq(arch, s.uint(0)), s.ieq(arch, s.uint(3))), gen_stretch)
    s.begin_if(is_streak)
    var tail_dt = s.select_f(s.ieq(arch, s.uint(3)), s.float(0.024), s.float(0.036))
    var tpx_w = s.fsub(s.load_f(npx), s.fmul(s.load_f(nvx), tail_dt))
    var tpy_w = s.fsub(s.load_f(npy), s.fmul(s.load_f(nvy), tail_dt))
    var tpz_w = s.fsub(s.load_f(npz), s.fmul(s.load_f(nvz), tail_dt))
    var trx = s.fsub(tpx_w, cam_x); var try_v = s.fsub(tpy_w, cam_y); var trz = s.fsub(tpz_w, cam_z)
    var tvz = s.nmax(near_z, emit_vec3_dot(s, trx, try_v, trz, fwd_x, fwd_y, fwd_z))
    var tvx = emit_vec3_dot(s, trx, try_v, trz, rgt_x, rgt_y, rgt_z)
    var tvy = emit_vec3_dot(s, trx, try_v, trz, up_x, up_y, up_z)
    var tu_s = s.fma(s.fdiv(tvx, s.fmul(tvz, tan_x)), s.float(0.5), s.float(0.5))
    var tv_s = s.fma(s.fneg(s.fdiv(tvy, s.fmul(tvz, tan_y))), s.float(0.5), s.float(0.5))
    # Cap screen tail length to 42 px so tile span stays compact
    var dx_tl = s.nclamp(s.fsub( s.fmul(tu_s, fw), cx_px), s.float(-42.0), s.float(42.0))
    var dy_tl = s.nclamp(s.fsub( s.fmul(tv_s, fh), cy_px), s.float(-42.0), s.float(42.0))
    s.store(tx_px, s.fadd(cx_px, dx_tl))
    s.store(ty_px, s.fadd(cy_px, dy_tl))
    s.end_if()

    var min_x = s.fsub(s.nmin(cx_px, s.load_f(tx_px)), rad_px)
    var max_x = s.fadd(s.nmax(cx_px, s.load_f(tx_px)), rad_px)
    var min_y = s.fsub(s.nmin(cy_px, s.load_f(ty_px)), rad_px)
    var max_y = s.fadd(s.nmax(cy_px, s.load_f(ty_px)), rad_px)

    var vis_scr = s.logical_and(
        s.logical_and(s.flt(min_x, fw), s.fgt(max_x, s.float(0.0))),
        s.logical_and(s.flt(min_y, fh), s.fgt(max_y, s.float(0.0))),
    )
    s.begin_if(vis_scr)
    var tiles_x = s.udiv(s.iadd(w_u, s.uint(15)), s.uint(16))
    var tiles_y = s.udiv(s.iadd(h_u, s.uint(15)), s.uint(16))
    var tx0 = s.umin(s.isub(tiles_x, s.uint(1)), s.saturate_uint(s.fmul(s.nmax(s.float(0.0), min_x), s.float(1.0 / 16.0))))
    var ty0 = s.umin(s.isub(tiles_y, s.uint(1)), s.saturate_uint(s.fmul(s.nmax(s.float(0.0), min_y), s.float(1.0 / 16.0))))
    var tx1 = s.umin(s.isub(tiles_x, s.uint(1)), s.umin(s.iadd(tx0, s.uint(4)), s.saturate_uint(s.fmul(s.nmax(s.float(0.0), max_x), s.float(1.0 / 16.0)))))
    var ty1 = s.umin(s.isub(tiles_y, s.uint(1)), s.umin(s.iadd(ty0, s.uint(4)), s.saturate_uint(s.fmul(s.nmax(s.float(0.0), max_y), s.float(1.0 / 16.0)))))

    var ty_i = s.uvar(ty0)
    s.begin_loop()
    var b_ty = s.load_u(ty_i)
    s.break_if(s.uge(b_ty, s.iadd(ty1, s.uint(1))))
    var tx_i = s.uvar(tx0)
    s.begin_loop()
    var b_tx = s.load_u(tx_i)
    s.break_if(s.uge(b_tx, s.iadd(tx1, s.uint(1))))
    var tile_id = s.iadd(s.imul(b_ty, tiles_x), b_tx)
    var slot = s.atomic_add_word(b_oit, s.iadd(counts_off, tile_id), s.uint(1))
    s.begin_if(s.ult(slot, s.uint(UInt32(AAA_MAX_PARTICLES_PER_TILE))))
    var b_idx = s.iadd(bins_off, s.iadd(s.imul(tile_id, s.uint(UInt32(AAA_MAX_PARTICLES_PER_TILE))), slot))
    s.store_word(b_oit, b_idx, pid)
    s.end_if()
    s.store(tx_i, s.iadd(b_tx, s.uint(1)))
    s.end_loop()
    s.store(ty_i, s.iadd(b_ty, s.uint(1)))
    s.end_loop()
    s.end_if()
    s.begin_else()
    # Behind camera: still persist updated world state so particles behind camera continue living!
    s.store_float(b_oit, p_off, s.load_f(npx))
    s.store_float(b_oit, s.iadd(p_off, s.uint(1)), s.load_f(npy))
    s.store_float(b_oit, s.iadd(p_off, s.uint(2)), s.load_f(npz))
    s.store_float(b_oit, s.iadd(p_off, s.uint(3)), s.load_f(nvx))
    s.store_float(b_oit, s.iadd(p_off, s.uint(4)), s.load_f(nvy))
    s.store_float(b_oit, s.iadd(p_off, s.uint(5)), s.load_f(nvz))
    s.store_float(b_oit, s.iadd(p_off, s.uint(6)), age)
    s.end_if()

    return s.finish()


def gpu_particle_tile_composite_shader() -> List[UInt32]:
    """Pass 2 (`group = 16 x 16`, cooperative LDS preload):
    Composites `PARTICLE_SPARK_STREAK`, Froxel-lit `PARTICLE_WATER_SPRAY`, `PARTICLE_DRIFT_SMOKE`,
    and Mach shock-diamond `PARTICLE_EXHAUST_PLASMA` onto `b_hdr` + writes `FlintTSR v3` reactive mask!
    Push[0]=render_w, Push[1]=render_h, Push[2]=pool_off, Push[3]=tile_counts_off,
    Push[4]=tile_bins_off, Push[5]=froxel_prefix_off, Push[6]=has_froxel_u32.
    """
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=16, group_height=16, push_words=AAA_PUSH_WORDS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_hist = s.buffer(BIND_AAA_HISTORY)
    var b_oit = s.buffer(BIND_AAA_OIT_NODES)

    # Cooperative LDS cache: 48 particles * 10 floats (`cx, cy, tx, ty, vz, rad_px, r, g, b, life_t`) + 48 meta words
    # (`arch | flags<<3 | pid<<6`); a `PARTICLE_GENERIC` particle stores its over-life color and its alpha for `life_t`
    var lds_f = s.workgroup_array_f32(AAA_MAX_PARTICLES_PER_TILE * 10)
    var lds_u = s.workgroup_array_u32(AAA_MAX_PARTICLES_PER_TILE)

    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var lx = s.local_axis(0)
    var ly = s.local_axis(1)
    var local_tid = s.iadd(s.imul(ly, s.uint(16)), lx)
    var tx_wg = s.workgroup_axis(0)
    var ty_wg = s.workgroup_axis(1)

    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var pool_off = s.push_word(2)
    var counts_off = s.push_word(3)
    var bins_off = s.push_word(4)
    var frox_prefix_off = s.push_word(5)
    var has_froxel = s.ine(s.push_word(6), s.uint(0))

    var tiles_x = s.udiv(s.iadd(w_u, s.uint(15)), s.uint(16))
    var tile_id = s.iadd(s.imul(ty_wg, tiles_x), tx_wg)
    var raw_cnt = s.load_word(b_oit, s.iadd(counts_off, tile_id))
    var tile_cnt = s.umin(raw_cnt, s.uint(UInt32(AAA_MAX_PARTICLES_PER_TILE)))
    # Uniform workgroup early-out when tile has 0 particles!
    s.return_if(s.ieq(tile_cnt, s.uint(0)))

    var fw = s.float_from_uint(w_u)
    var fh = s.float_from_uint(h_u)
    var cam_x = s.load_float(b_globals, s.uint(32))
    var cam_y = s.load_float(b_globals, s.uint(33))
    var cam_z = s.load_float(b_globals, s.uint(34))
    var near_z = s.nmax(s.float(0.10), s.load_float(b_globals, s.uint(35)))
    var fwd_x = s.load_float(b_globals, s.uint(36))
    var fwd_y = s.load_float(b_globals, s.uint(37))
    var fwd_z = s.load_float(b_globals, s.uint(38))
    var tan_x = s.nmax(s.float(0.1), s.load_float(b_globals, s.uint(39)))
    var rgt_x = s.load_float(b_globals, s.uint(40))
    var rgt_y = s.load_float(b_globals, s.uint(41))
    var rgt_z = s.load_float(b_globals, s.uint(42))
    var tan_y = s.nmax(s.float(0.1), s.load_float(b_globals, s.uint(43)))
    var up_x = s.load_float(b_globals, s.uint(44))
    var up_y = s.load_float(b_globals, s.uint(45))
    var up_z = s.load_float(b_globals, s.uint(46))

    # Cooperative preload: threads `0 .. tile_cnt - 1` project particle once into LDS!
    s.begin_if(s.ult(local_tid, tile_cnt))
    var b_idx = s.iadd(bins_off, s.iadd(s.imul(tile_id, s.uint(UInt32(AAA_MAX_PARTICLES_PER_TILE))), local_tid))
    var pid = s.load_word(b_oit, b_idx)
    var p_off = s.iadd(pool_off, s.imul(pid, s.uint(UInt32(AAA_PARTICLE_WORDS))))

    var pwx = s.load_float(b_oit, p_off)
    var pwy = s.load_float(b_oit, s.iadd(p_off, s.uint(1)))
    var pwz = s.load_float(b_oit, s.iadd(p_off, s.uint(2)))
    var pvx = s.load_float(b_oit, s.iadd(p_off, s.uint(3)))
    var pvy = s.load_float(b_oit, s.iadd(p_off, s.uint(4)))
    var pvz = s.load_float(b_oit, s.iadd(p_off, s.uint(5)))
    var age = s.load_float(b_oit, s.iadd(p_off, s.uint(6)))
    var life = s.nmax(s.float(1e-3), s.load_float(b_oit, s.iadd(p_off, s.uint(7))))
    var c0 = s.unpack_f16x2(s.load_word(b_oit, s.iadd(p_off, s.uint(8))))
    var c1 = s.unpack_f16x2(s.load_word(b_oit, s.iadd(p_off, s.uint(9))))
    var arch = s.bit_and(s.load_word(b_oit, s.iadd(p_off, s.uint(11))), s.uint(_ARCH_MASK))
    var is_gen = s.ieq(arch, s.uint(PARTICLE_GENERIC))
    var gen_flags = s.bit_and(s.load_word(b_oit, s.iadd(p_off, s.uint(15))), s.uint(7))

    var life_t = s.nclamp(s.fdiv(age, life), s.float(0.0), s.float(1.0))
    var is_mist = s.logical_or(s.ieq(arch, s.uint(1)), s.ieq(arch, s.uint(2)))
    # Generic: color, size and alpha from start to end over the particle's life
    var e0 = s.unpack_f16x2(s.load_word(b_oit, s.iadd(p_off, s.uint(12))))
    var e1 = s.unpack_f16x2(s.load_word(b_oit, s.iadd(p_off, s.uint(13))))
    var al = s.unpack_f16x2(s.load_word(b_oit, s.iadd(p_off, s.uint(14))))
    var gen_r = s.fma(s.fsub(e0[0], c0[0]), life_t, c0[0])
    var gen_g = s.fma(s.fsub(e0[1], c0[1]), life_t, c0[1])
    var gen_b = s.fma(s.fsub(e1[0], c1[0]), life_t, c1[0])
    var gen_rad = s.fma(s.fsub(e1[1], c1[1]), life_t, c1[1])
    var gen_a = s.fma(s.fsub(al[1], al[0]), life_t, al[0])
    var eff_rad_m = s.select_f(is_gen, gen_rad, s.select_f(is_mist, s.fmul(c1[1], s.fma(life_t, s.float(1.15), s.float(0.65))), c1[1]))

    var rx_p = s.fsub(pwx, cam_x); var ry_p = s.fsub(pwy, cam_y); var rz_p = s.fsub(pwz, cam_z)
    var vz_p = s.nmax(near_z, emit_vec3_dot(s, rx_p, ry_p, rz_p, fwd_x, fwd_y, fwd_z))
    var vx_p = emit_vec3_dot(s, rx_p, ry_p, rz_p, rgt_x, rgt_y, rgt_z)
    var vy_p = emit_vec3_dot(s, rx_p, ry_p, rz_p, up_x, up_y, up_z)
    var cx_s = s.fmul(s.fma(s.fdiv(vx_p, s.fmul(vz_p, tan_x)), s.float(0.5), s.float(0.5)), fw)
    var cy_s = s.fmul(s.fma(s.fneg(s.fdiv(vy_p, s.fmul(vz_p, tan_y))), s.float(0.5), s.float(0.5)), fh)

    var px_per_m = s.fdiv(fh, s.nmax(s.float(0.05), s.fmul(s.float(2.0), s.fmul(vz_p, tan_y))))
    var rad_px = s.nclamp(s.fmul(eff_rad_m, px_per_m), s.float(1.35), s.float(30.0))

    var tx_s = s.fvar(cx_s)
    var ty_s = s.fvar(cy_s)
    var gen_stretch = s.logical_and(is_gen, s.ine(s.bit_and(gen_flags, s.uint(PARTICLE_FLAG_STRETCH)), s.uint(0)))
    var is_streak = s.logical_or(s.logical_or(s.ieq(arch, s.uint(0)), s.ieq(arch, s.uint(3))), gen_stretch)
    s.begin_if(is_streak)
    var tail_dt = s.select_f(s.ieq(arch, s.uint(3)), s.float(0.024), s.float(0.036))
    var tpx = s.fsub(pwx, s.fmul(pvx, tail_dt))
    var tpy = s.fsub(pwy, s.fmul(pvy, tail_dt))
    var tpz = s.fsub(pwz, s.fmul(pvz, tail_dt))
    var trx = s.fsub(tpx, cam_x); var try_v = s.fsub(tpy, cam_y); var trz = s.fsub(tpz, cam_z)
    var tvz = s.nmax(near_z, emit_vec3_dot(s, trx, try_v, trz, fwd_x, fwd_y, fwd_z))
    var tvx = emit_vec3_dot(s, trx, try_v, trz, rgt_x, rgt_y, rgt_z)
    var tvy = emit_vec3_dot(s, trx, try_v, trz, up_x, up_y, up_z)
    var tu_p = s.fmul(s.fma(s.fdiv(tvx, s.fmul(tvz, tan_x)), s.float(0.5), s.float(0.5)), fw)
    var tv_p = s.fmul(s.fma(s.fneg(s.fdiv(tvy, s.fmul(tvz, tan_y))), s.float(0.5), s.float(0.5)), fh)
    var dx_tl = s.nclamp(s.fsub(tu_p, cx_s), s.float(-42.0), s.float(42.0))
    var dy_tl = s.nclamp(s.fsub(tv_p, cy_s), s.float(-42.0), s.float(42.0))
    s.store(tx_s, s.fadd(cx_s, dx_tl))
    s.store(ty_s, s.fadd(cy_s, dy_tl))
    s.end_if()

    var fb = s.imul(local_tid, s.uint(10))
    s.store_workgroup_f32(lds_f[0], lds_f[1], fb, cx_s)
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(1)), cy_s)
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(2)), s.load_f(tx_s))
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(3)), s.load_f(ty_s))
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(4)), vz_p)
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(5)), rad_px)
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(6)), s.select_f(is_gen, gen_r, c0[0]))
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(7)), s.select_f(is_gen, gen_g, c0[1]))
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(8)), s.select_f(is_gen, gen_b, c1[0]))
    s.store_workgroup_f32(lds_f[0], lds_f[1], s.iadd(fb, s.uint(9)), s.select_f(is_gen, gen_a, life_t))
    s.store_workgroup_u32(lds_u[0], lds_u[1], local_tid, s.bit_or(s.bit_or(arch, s.shl(gen_flags, s.uint(3))), s.shl(pid, s.uint(6))))
    s.end_if()
    s.workgroup_barrier()

    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))

    var pix_idx = s.iadd(s.imul(py, w_u), px)
    var h_base = s.imul(pix_idx, s.uint(8))
    var z_surf = s.load_float(b_hdr, s.iadd(h_base, s.uint(3)))
    var fpx = s.fadd(s.float_from_uint(px), s.float(0.5))
    var fpy = s.fadd(s.float_from_uint(py), s.float(0.5))

    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_r = s.load_float(b_globals, s.uint(52))
    var sun_g = s.load_float(b_globals, s.uint(53))
    var sun_b = s.load_float(b_globals, s.uint(54))
    var ibl_int = s.load_float(b_globals, s.uint(55))
    var time_s = s.load_float(b_globals, s.uint(123))
    var far_frox = s.nmax(s.fadd(near_z, s.float(5.0)), s.load_float(b_globals, s.uint(131)))
    var log_ratio = s.log2(s.fdiv(far_frox, near_z))

    var sh_up = emit_eval_sh9_irradiance(s, b_globals, 72, s.float(0.0), s.float(1.0), s.float(0.0))
    var sun_top = s.fmul(s.nmax(s.float(0.0), sun_y), s.float(0.28))
    var env_lit_r = s.fma(sh_up[0], s.fmul(ibl_int, s.float(0.55)), s.fma(sun_r, sun_top, s.float(0.04)))
    var env_lit_g = s.fma(sh_up[1], s.fmul(ibl_int, s.float(0.55)), s.fma(sun_g, sun_top, s.float(0.04)))
    var env_lit_b = s.fma(sh_up[2], s.fmul(ibl_int, s.float(0.55)), s.fma(sun_b, sun_top, s.float(0.05)))

    # Sample local Froxel Prefix-LUT illumination once at pixel `(u, v)` for `PARTICLE_WATER_SPRAY`!
    var frox_lr = s.fvar(s.float(0.0))
    var frox_lg = s.fvar(s.float(0.0))
    var frox_lb = s.fvar(s.float(0.0))
    s.begin_if(has_froxel)
    var fx_f = s.nclamp(s.fmul(s.fdiv(fpx, fw), s.float(160.0)), s.float(0.0), s.float(159.0))
    var fy_f = s.nclamp(s.fmul(s.fdiv(fpy, fh), s.float(90.0)), s.float(0.0), s.float(89.0))
    var z_probe = s.nmin(z_surf, s.float(28.0))
    var fpr = emit_sample_froxel_prefix_lut_spv(s, b_hist, frox_prefix_off, fx_f, fy_f, z_probe, near_z, log_ratio)
    var frox_gain = s.fdiv(s.float(1.0), s.nmax(s.float(0.08), s.fsub(s.float(1.0), fpr[3])))
    s.store(frox_lr, s.nmin(s.float(8.0), s.fmul(fpr[0], frox_gain)))
    s.store(frox_lg, s.nmin(s.float(8.0), s.fmul(fpr[1], frox_gain)))
    s.store(frox_lb, s.nmin(s.float(8.0), s.fmul(fpr[2], frox_gain)))
    s.end_if()

    var cur_r = s.fvar(s.load_float(b_hdr, h_base))
    var cur_g = s.fvar(s.load_float(b_hdr, s.iadd(h_base, s.uint(1))))
    var cur_b = s.fvar(s.load_float(b_hdr, s.iadd(h_base, s.uint(2))))
    var react_acc = s.fvar(s.float(0.0))

    var ki_v = s.uvar(s.uint(0))
    s.begin_loop()
    var ki = s.load_u(ki_v)
    s.break_if(s.uge(ki, tile_cnt))
    var k_fb = s.imul(ki, s.uint(10))
    var k_cx = s.load_workgroup_f32(lds_f[0], lds_f[1], k_fb)
    var k_cy = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(1)))
    var k_tx = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(2)))
    var k_ty = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(3)))
    var k_vz = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(4)))
    var k_rad = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(5)))
    var cr_p = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(6)))
    var cg_p = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(7)))
    var cb_p = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(8)))
    var k_life_t = s.load_workgroup_f32(lds_f[0], lds_f[1], s.iadd(k_fb, s.uint(9)))
    var meta_u = s.load_workgroup_u32(lds_u[0], lds_u[1], ki)
    var k_arch = s.bit_and(meta_u, s.uint(_ARCH_MASK))
    var k_flags = s.bit_and(s.shr(meta_u, s.uint(3)), s.uint(7))
    var pid_f = s.float_from_uint(s.bit_and(s.shr(meta_u, s.uint(6)), s.uint(1023)))

    s.begin_if(s.flt(k_vz, s.fadd(z_surf, s.float(0.20))))
    var soft_z = s.nclamp(s.fmul(s.fsub(s.fadd(z_surf, s.float(0.20)), k_vz), s.float(2.2)), s.float(0.0), s.float(1.0))
    var fade_age = s.fsub(s.float(1.0), s.fmul(k_life_t, k_life_t))

    # Archetype 4 (`GENERIC`): soft round sprite (a capsule when stretched along its velocity)
    s.begin_if(s.ieq(k_arch, s.uint(PARTICLE_GENERIC)))
    var g_sx = s.fsub(k_tx, k_cx)
    var g_sy = s.fsub(k_ty, k_cy)
    var g_l2 = s.fma(g_sx, g_sx, s.fmul(g_sy, g_sy))
    var g_qx = s.fsub(fpx, k_cx)
    var g_qy = s.fsub(fpy, k_cy)
    var g_h = s.select_f(
        s.fgt(g_l2, s.float(0.01)),
        s.nclamp(s.fdiv(s.fma(g_qx, g_sx, s.fmul(g_qy, g_sy)), g_l2), s.float(0.0), s.float(1.0)),
        s.float(0.0),
    )
    var g_dx = s.fsub(g_qx, s.fmul(g_sx, g_h))
    var g_dy = s.fsub(g_qy, s.fmul(g_sy, g_h))
    var g_d2 = s.fdiv(s.fma(g_dx, g_dx, s.fmul(g_dy, g_dy)), s.nmax(s.float(0.25), s.fmul(k_rad, k_rad)))
    s.begin_if(s.flt(g_d2, s.float(1.0)))
    var g_fall = s.fsub(s.float(1.0), g_d2)
    # `k_life_t` holds the particle's alpha; a stretched tail thins toward its end
    var g_w = s.fmul(s.fmul(g_fall, g_fall), s.fmul(k_life_t, s.fmul(soft_z, s.fsub(s.float(1.0), s.fmul(g_h, s.float(0.6))))))
    s.begin_if(s.ine(s.bit_and(k_flags, s.uint(PARTICLE_FLAG_ADDITIVE)), s.uint(0)))
    s.store(cur_r, s.fma(cr_p, g_w, s.load_f(cur_r)))
    s.store(cur_g, s.fma(cg_p, g_w, s.load_f(cur_g)))
    s.store(cur_b, s.fma(cb_p, g_w, s.load_f(cur_b)))
    s.store(react_acc, s.nmax(s.load_f(react_acc), s.nmin(s.float(0.85), g_w)))
    s.begin_else()
    var g_a = s.nmin(s.float(1.0), g_w)
    var g_lit = s.ine(s.bit_and(k_flags, s.uint(PARTICLE_FLAG_LIT)), s.uint(0))
    var g_lr = s.select_f(g_lit, s.fmul(cr_p, s.fadd(env_lit_r, s.load_f(frox_lr))), cr_p)
    var g_lg = s.select_f(g_lit, s.fmul(cg_p, s.fadd(env_lit_g, s.load_f(frox_lg))), cg_p)
    var g_lb = s.select_f(g_lit, s.fmul(cb_p, s.fadd(env_lit_b, s.load_f(frox_lb))), cb_p)
    var g_om = s.fsub(s.float(1.0), g_a)
    s.store(cur_r, s.fma(s.load_f(cur_r), g_om, s.fmul(g_lr, g_a)))
    s.store(cur_g, s.fma(s.load_f(cur_g), g_om, s.fmul(g_lg, g_a)))
    s.store(cur_b, s.fma(s.load_f(cur_b), g_om, s.fmul(g_lb, g_a)))
    s.store(react_acc, s.nmax(s.load_f(react_acc), s.nmin(s.float(0.85), g_a)))
    s.end_if()
    s.end_if()
    s.begin_else()

    # Archetype 0 (`SPARK_STREAK`) & Archetype 3 (`EXHAUST_PLASMA`): 2D Velocity Capsule SDF
    var k_is_streak = s.logical_or(s.ieq(k_arch, s.uint(0)), s.ieq(k_arch, s.uint(3)))
    s.begin_if(k_is_streak)
    var seg_x = s.fsub(k_tx, k_cx)
    var seg_y = s.fsub(k_ty, k_cy)
    var seg_l2 = s.fma(seg_x, seg_x, s.fmul(seg_y, seg_y))
    var qx = s.fsub(fpx, k_cx)
    var qy = s.fsub(fpy, k_cy)
    var h_seg = s.select_f(
        s.fgt(seg_l2, s.float(0.01)),
        s.nclamp(s.fdiv(s.fma(qx, seg_x, s.fmul(qy, seg_y)), seg_l2), s.float(0.0), s.float(1.0)),
        s.float(0.0),
    )
    var dx_c = s.fsub(qx, s.fmul(seg_x, h_seg))
    var dy_c = s.fsub(qy, s.fmul(seg_y, h_seg))
    var d2_c = s.fma(dx_c, dx_c, s.fmul(dy_c, dy_c))
    var r2_c = s.fmul(k_rad, k_rad)

    s.begin_if(s.flt(d2_c, r2_c))
    var d_norm = s.fsqrt(s.fdiv(d2_c, s.nmax(s.float(0.25), r2_c)))
    var core = s.fpow(s.nmax(s.float(0.0), s.fsub(s.float(1.0), d_norm)), s.float(2.2))
    var tail_fade = s.fsub(s.float(1.0), s.fmul(h_seg, s.float(0.72)))

    s.begin_if(s.ieq(k_arch, s.uint(0)))
    # Blackbody spark + firework glitter flicker (`hot white center -> ember orange tail`)
    var glitter = s.fma(s.float(0.38), s.fsin(s.fma(time_s, s.float(54.0), pid_f)), s.float(0.72))
    var w_sp = s.fmul(core, s.fmul(tail_fade, s.fmul(fade_age, s.fmul(soft_z, glitter))))
    var hot = s.fmul(core, core)
    s.store(cur_r, s.fma(s.fma(hot, s.float(3.0), cr_p), w_sp, s.load_f(cur_r)))
    s.store(cur_g, s.fma(s.fma(hot, s.float(2.6), cg_p), w_sp, s.load_f(cur_g)))
    s.store(cur_b, s.fma(s.fma(hot, s.float(2.0), cb_p), w_sp, s.load_f(cur_b)))
    s.store(react_acc, s.nmax(s.load_f(react_acc), s.nmin(s.float(0.85), w_sp)))
    s.begin_else()
    # Mach Shock-Diamond Exhaust Plasma (`periodic standing-wave nodes along h_seg`)
    var diamond = s.fma(s.float(0.65), s.fpow(s.fabs(s.fcos(s.fma(h_seg, s.float(12.566), s.fmul(time_s, s.float(-28.0))))), s.float(3.0)), s.float(0.45))
    var w_pl = s.fmul(core, s.fmul(diamond, s.fmul(fade_age, soft_z)))
    s.store(cur_r, s.fma(cr_p, w_pl, s.load_f(cur_r)))
    s.store(cur_g, s.fma(cg_p, w_pl, s.load_f(cur_g)))
    s.store(cur_b, s.fma(cb_p, w_pl, s.load_f(cur_b)))
    s.store(react_acc, s.nmax(s.load_f(react_acc), s.nmin(s.float(0.80), w_pl)))
    s.end_if()
    s.end_if()

    s.begin_else()
    # Archetype 1 (`WATER_SPRAY`) & Archetype 2 (`DRIFT_SMOKE`): Billowing FBM Soft Volumetric Puff
    var dx_p = s.fsub(fpx, k_cx)
    var dy_p = s.fsub(fpy, k_cy)
    var d2_p = s.fma(dx_p, dx_p, s.fmul(dy_p, dy_p))
    var r2_p = s.fmul(k_rad, k_rad)
    s.begin_if(s.flt(d2_p, r2_p))
    var inv_rp = s.fdiv(s.float(1.0), s.nmax(s.float(1.0), k_rad))
    var ux_p = s.fmul(dx_p, inv_rp)
    var uy_p = s.fmul(dy_p, inv_rp)
    var rn = s.fsqrt(s.nclamp(s.fma(ux_p, ux_p, s.fmul(uy_p, uy_p)), s.float(0.0), s.float(1.0)))

    var fbm_p = s.fma(
        s.float(0.32),
        s.fsin(s.fma(ux_p, s.float(5.2), s.fma(uy_p, s.float(4.7), pid_f))),
        s.fma(s.float(0.18), s.fcos(s.fma(ux_p, s.float(11.3), s.fmul(uy_p, s.float(-9.7)))), s.float(0.50)),
    )
    var puff_sh = s.nclamp(s.fsub(s.fsub(s.float(1.0), rn), s.fmul(s.fsub(s.float(1.0), fbm_p), s.float(0.38))), s.float(0.0), s.float(1.0))
    var env_age = s.fmul(s.nclamp(s.fmul(k_life_t, s.float(5.0)), s.float(0.0), s.float(1.0)), fade_age)

    s.begin_if(s.ieq(k_arch, s.uint(1)))
    # Wet-tire rooster-tail water spray illuminated live by Froxel Prefix-LUT + Sky!
    var sp_a = s.nmin(s.float(0.52), s.fmul(puff_sh, s.fmul(env_age, s.fmul(soft_z, s.float(0.46)))))
    var lr_sp = s.fmul(cr_p, s.fadd(env_lit_r, s.load_f(frox_lr)))
    var lg_sp = s.fmul(cg_p, s.fadd(env_lit_g, s.load_f(frox_lg)))
    var lb_sp = s.fmul(cb_p, s.fadd(env_lit_b, s.load_f(frox_lb)))
    var om_sp = s.fsub(s.float(1.0), sp_a)
    s.store(cur_r, s.fma(s.load_f(cur_r), om_sp, s.fmul(lr_sp, sp_a)))
    s.store(cur_g, s.fma(s.load_f(cur_g), om_sp, s.fmul(lg_sp, sp_a)))
    s.store(cur_b, s.fma(s.load_f(cur_b), om_sp, s.fmul(lb_sp, sp_a)))
    s.store(react_acc, s.nmax(s.load_f(react_acc), sp_a))
    s.begin_else()
    # Billowing drift/coaster smoke with top-hemisphere self-shadowing (`0.55 - 0.35 * uy_p`)
    var sm_a = s.nmin(s.float(0.62), s.fmul(puff_sh, s.fmul(env_age, s.fmul(soft_z, s.float(0.54)))))
    var top_sh = s.fma(s.fneg(uy_p), s.float(0.28), s.float(0.76))
    var lr_sm = s.fmul(cr_p, s.fmul(s.fadd(env_lit_r, s.fmul(s.load_f(frox_lr), s.float(0.5))), top_sh))
    var lg_sm = s.fmul(cg_p, s.fmul(s.fadd(env_lit_g, s.fmul(s.load_f(frox_lg), s.float(0.5))), top_sh))
    var lb_sm = s.fmul(cb_p, s.fmul(s.fadd(env_lit_b, s.fmul(s.load_f(frox_lb), s.float(0.5))), top_sh))
    var om_sm = s.fsub(s.float(1.0), sm_a)
    s.store(cur_r, s.fma(s.load_f(cur_r), om_sm, s.fmul(lr_sm, sm_a)))
    s.store(cur_g, s.fma(s.load_f(cur_g), om_sm, s.fmul(lg_sm, sm_a)))
    s.store(cur_b, s.fma(s.load_f(cur_b), om_sm, s.fmul(lb_sm, sm_a)))
    s.store(react_acc, s.nmax(s.load_f(react_acc), sm_a))
    s.end_if()
    s.end_if()
    s.end_if()
    s.end_if()  # generic / built-in archetypes
    s.end_if()

    s.store(ki_v, s.iadd(ki, s.uint(1)))
    s.end_loop()

    s.store_float(b_hdr, h_base, s.load_f(cur_r))
    s.store_float(b_hdr, s.iadd(h_base, s.uint(1)), s.load_f(cur_g))
    s.store_float(b_hdr, s.iadd(h_base, s.uint(2)), s.load_f(cur_b))

    s.begin_if(s.fgt(s.load_f(react_acc), s.float(0.01)))
    var r_slot = s.iadd(s.uint(UInt32(AAA_OIT_REACTIVE_BASE_WORDS)), pix_idx)
    var prev_r = s.load_float(b_oit, r_slot)
    s.store_float(b_oit, r_slot, s.nmax(prev_r, s.load_f(react_acc)))
    s.end_if()

    return s.finish()

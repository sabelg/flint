"""Area B (render state of the art): world-space radiance cache and path-traced GI resolve.

Three GPU pieces live here, all emitted through Flint's own SPIR-V builder:

1. **World radiance cache (WRC).** A camera-following, toroidally scrolled grid of irradiance
   probes (DDGI-style, Majercik et al. 2019 "Dynamic Diffuse Global Illumination with Ray-Traced
   Irradiance Fields"; probe scrolling as in Majercik et al. 2021 "Scaling Probe-Based Real-Time
   Dynamic Global Illumination for Production"). Each probe keeps order-2 spherical-harmonic
   radiance (Ramamoorthi & Hanrahan 2001 irradiance convolution) and a six-axis mean / mean-square
   hit-distance pair used for a Chebyshev visibility test (variance shadow maps, Donnelly & Lauritzen
   2006). One 64-thread workgroup updates one probe: one ray per thread, shared-memory reduction,
   hysteresis blend into a ping-pong bank. Probe rays that hit a surface evaluate sun + local lights
   at the hit and add `albedo * WRC(hit)` from the previous bank, so bounces accumulate over frames
   (infinite bounce through feedback). When the hit triangle has a surface-cache slot and the hybrid
   mode asks for it, the probe reads the surface cache instead (cheaper, already multi-bounce).
2. **GI resolve pass.** Runs after the deferred shade on a small G-buffer the shade writes into this
   module's region. Modes: probe-only GI (zero per-pixel rays), hybrid surface-cache rays with the
   WRC behind them (hits on triangles without a cache slot fall back to `direct + albedo * WRC`),
   and a real-time diffuse path tracer (N bounces, next-event estimation of the sun at each vertex,
   optional WRC termination, optional progressive accumulation for photo mode). Output goes to the
   demodulated GI channel so SVGF / TSR denoise it unchanged.
3. **Hooks** emitted into `aaa_gi_svgf.mojo` (deferred shade and surface-cache update). They are
   inert in mode 0, which is the default, so the legacy frame is bit-identical.

Storage: everything is a region at the end of `BIND_AAA_SURFCACHE` (word `SOTA_GI_BASE`), so no
descriptor-layout change is needed. Layout of the region is described by the `SOTA_GI_*` constants.
"""
from std.math import max, min, sqrt
from flint.core.math import Vec3
from flint.platform.os import env_now
from flint.platform.vulkan import put32, put_float
from flint.render.spirv import AaaSpirvModule, MODEL_GL_COMPUTE
from flint.render.aaa_visbuffer import (
    AAA_INSTANCE_WORDS,
    AAA_MAX_BINDLESS,
    AAA_PUSH_WORDS,
    BIND_AAA_BINDLESS_TEX,
    BIND_AAA_CLUSTERS,
    BIND_AAA_ENV_CUBE,
    BIND_AAA_GLOBALS,
    BIND_AAA_HDR_OUT,
    BIND_AAA_INDICES,
    BIND_AAA_INSTANCES,
    BIND_AAA_MATERIALS,
    BIND_AAA_SURFCACHE,
    BIND_AAA_TLAS,
    BIND_AAA_VERTICES,
    emit_mat4_mul_dir,
    emit_vec3_cross,
    emit_vec3_dot,
    emit_vec3_normalize,
)
from flint.render.aaa_bsdf import AAA_MATERIAL_WORDS
from flint.render.aaa_raytracing import (
    AAA_RT_MASK_OPAQUE,
    RAY_FLAG_NONE_KHR,
    _emit_rt_candidate_alpha_loop,
    emit_decode_vertex_color_rgb_spv,
    emit_froxel_local_lighting_at_hit,
    emit_rt_shadow_query,
)
from flint.render.sota_gi_core import (
    SOTA_CTRL_ACCUM,
    SOTA_CTRL_ACCUM_N,
    SOTA_CTRL_BASE_X,
    SOTA_CTRL_CAM_X,
    SOTA_CTRL_CLAMP,
    SOTA_CTRL_CONTACT_RADIUS,
    SOTA_CTRL_CONTACT_RAYS,
    SOTA_CTRL_CUR_BANK,
    SOTA_CTRL_DI_CUR,
    SOTA_CTRL_DI_MCAP,
    SOTA_CTRL_DI_PREV,
    SOTA_CTRL_DI_RADIUS,
    SOTA_CTRL_DI_SPATIAL,
    SOTA_CTRL_DIM_X,
    SOTA_CTRL_FLAGS,
    SOTA_CTRL_FRAME,
    SOTA_CTRL_GBUF,
    SOTA_CTRL_GI_CUR,
    SOTA_CTRL_GI_MCAP,
    SOTA_CTRL_GI_PREV,
    SOTA_CTRL_GI_RADIUS,
    SOTA_CTRL_GI_SPATIAL,
    SOTA_CTRL_HIST_CUR,
    SOTA_CTRL_HIST_PREV,
    SOTA_CTRL_LOWRES,
    SOTA_CTRL_MAX_HISTORY,
    SOTA_CTRL_MODE,
    SOTA_CTRL_PIXEL_RAYS,
    SOTA_CTRL_PREV_BANK,
    SOTA_CTRL_PROBE_COUNT,
    SOTA_CTRL_PT_BOUNCES,
    SOTA_CTRL_RENDER_H,
    SOTA_CTRL_RENDER_W,
    SOTA_CTRL_RESTIR_FLAGS,
    SOTA_CTRL_SPACING,
    SOTA_FLAG_ACCUMULATE,
    SOTA_FLAG_PROBE_USE_SURFCACHE,
    SOTA_FLAG_PT_WRC_TERMINATE,
    SOTA_FLAG_REFL_WRC,
    SOTA_FLAG_WRC_FEEDS_SURFCACHE,
    SOTA_FLAG_WRC_VALID,
    SOTA_GI_ACCUM_WORDS,
    SOTA_GI_BASE,
    SOTA_GI_CELL_BIAS,
    SOTA_GI_CTRL_WORDS,
    SOTA_GI_GBUF_WORDS,
    SOTA_GI_LOWRES_WORDS,
    SOTA_GI_MAX_PROBES,
    SOTA_GI_MODE_HYBRID,
    SOTA_GI_MODE_LEGACY,
    SOTA_GI_MODE_PATH,
    SOTA_GI_MODE_WRC,
    SOTA_GI_PROBE_RAYS,
    SOTA_GI_PROBE_WORDS,
    SOTA_GI_SURFCACHE_TRIS,
    SOTA_PROBE_BACKFACE,
    SOTA_PROBE_HIST,
    SOTA_PROBE_KEY,
    SOTA_PROBE_SH,
    SOTA_PROBE_VIS,
    SOTA_RESTIR_DI,
    SOTA_RESTIR_GI,
    SOTA_RESTIR_HIST_VALID,
    SOTA_RESTIR_HIST_WRITE,
    _PI,
    _ctrl,
    _ctrl_f,
    _emit_cosine_dir,
    _emit_hash,
    _emit_sh9_basis,
    emit_sota_gi_mode,
    emit_sota_wrc_irradiance,
    sota_gi_accum_offset,
    sota_gi_bank_offset,
    sota_gi_di_offset,
    sota_gi_gbuf_offset,
    sota_gi_gi_offset,
    sota_gi_hist_offset,
    sota_gi_lowres_offset,
    sota_gi_bayer_offset,
    sota_gi_region_words,
    sota_sh9_band_weight,
)
from flint.render.sota_gi_restir import (
    SotaGiReservoir,
    emit_sota_gi_stream_candidate,
    emit_sota_oct_pack,
    emit_sota_oct_unpack,
    emit_sota_hist_write,
    emit_sota_restir_gi_reuse,
)

# ---------------------------------------------------------------------------------------------
# Settings (the renderer owns one; every field is a cvar)
# ---------------------------------------------------------------------------------------------
struct SotaGiSettings(Copyable, Movable):
    """Area-B GI settings. `mode = SOTA_GI_MODE_LEGACY` (default) leaves the v5 frame unchanged.
    The other modes need `enable_rt_gi = True` and hardware ray queries; without ray queries the
    renderer silently stays on the legacy path."""
    var mode: Int
    var pt_bounces: Int  # path tracer: diffuse bounces after the primary hit (1..8)
    var pixel_rays: Int  # path tracer: paths per pixel per frame; hybrid: rays per pixel
    var pt_wrc_terminate: Bool  # end the last path vertex in the WRC (bias for fewer bounces) instead of truncating
    var accumulate: Bool  # progressive accumulation while the camera and scene stay still (photo mode)
    var firefly_clamp: Float32  # per-path luminance clamp in GI units (0 = off)
    var wrc_dims_x: Int
    var wrc_dims_y: Int
    var wrc_dims_z: Int
    var wrc_spacing: Float32  # metres between probes
    var wrc_center_fixed: Bool  # True: keep the grid centred on `wrc_center`; False: follow the camera
    var wrc_center: Vec3
    var wrc_max_history: Int  # hysteresis: blend weight is 1 / min(history + 1, max_history)
    var wrc_probe_use_surfcache: Bool  # hybrid mode: probe hits read the surface cache when a slot exists
    var contact_rays: Int  # WRC-only mode: short near-field rays per pixel in front of the probes (0 = off, default)
    var contact_radius: Float32  # metres; <= 0 means one probe spacing
    var refl_use_wrc: Bool  # RT reflection hits take their indirect diffuse from the WRC in WRC modes
    var restir_di_reuse: Bool  # temporal + spatial reuse for `enable_restir_di` (needs RT shadows)
    var restir_di_spatial: Int  # previous-frame neighbours per pixel (0..8; default 0: no gain on static views)
    var restir_di_radius: Float32  # pixels
    var restir_di_max_m: Float32  # temporal confidence cap (in candidates; 4 candidates per frame)
    var restir_gi: Bool  # ReSTIR GI in the path-traced mode (off while accumulating)
    var restir_gi_hybrid: Bool  # ReSTIR GI in the hybrid mode too (measured slightly worse there: off)
    var restir_gi_spatial: Int
    var restir_gi_radius: Float32
    var restir_gi_max_m: Float32
    var svgf_history_fix: Bool  # SVGF a-trous ping-pongs through its own scratch, not history bank A (default on)
    var svgf_jitter_tol: Bool  # round 4: SVGF reprojection depth test widened by the local depth slope (TSR jitter)
    var svgf_lowres_history: Float32  # round 4: SVGF history cap of the reduced-res path tracer (0 = v5's 24)
    var resolve_scale: Int  # round 3: GI resolve traces one pixel per `scale x scale` block (1, 2 or 4) + upsample
    var upsample_tent: Float32  # upsample tent radius in blocks (1 = bilinear between representative pixels)
    var resolve_jitter: Bool  # rotate the traced pixel of each block every frame (Bayer order) so history sees all
    # Runtime state (not cvars)
    var pipe_wrc: Int
    var pipe_resolve: Int
    var pipe_upsample: Int
    var cur_scale: Int  # effective resolve scale of the frame being recorded
    var cur_jx: Int
    var cur_jy: Int
    var prev_jx: Int  # representative sub-pixel of the previous frame (ReSTIR GI reservoirs live there)
    var prev_jy: Int
    var last_scale: Int
    var accum_frames: Int
    var wrc_frames: Int
    var last_cam_pos: Vec3
    var last_cam_fwd: Vec3
    var last_mode: Int
    var last_inst_count: Int
    var last_base: Tuple[Int, Int, Int]
    var last_sig: Int
    var restir_frames: Int
    var last_restir: UInt32
    var restir_active: UInt32

    def __init__(out self):
        self.mode = SOTA_GI_MODE_LEGACY
        self.pt_bounces = 3
        self.pixel_rays = 1
        self.pt_wrc_terminate = True
        self.accumulate = False
        self.firefly_clamp = Float32(0.0)
        self.wrc_dims_x = 16
        self.wrc_dims_y = 8
        self.wrc_dims_z = 16
        self.wrc_spacing = Float32(1.0)
        self.wrc_center_fixed = False
        self.wrc_center = Vec3(0.0, 0.0, 0.0)
        self.wrc_max_history = 32
        self.wrc_probe_use_surfcache = True
        self.contact_rays = 0
        self.contact_radius = Float32(0.0)
        self.refl_use_wrc = True
        self.restir_di_reuse = True
        self.restir_di_spatial = 0  # static Cornell, 128 lights: temporal 0.107 FLIP, +2 spatial (r 6) 0.115
        self.restir_di_radius = Float32(6.0)  # r 16 measured -4% dark and 0.140 FLIP
        self.restir_di_max_m = Float32(80.0)
        # Round 4: off by default. With the jitter-aware SVGF history (`svgf_jitter_tol`) ReSTIR GI's temporal
        # reuse correlates the samples SVGF averages: Cornell PT 1 spp 0.084 -> 0.066 FLIP (full res) and
        # 0.093 -> 0.075 (half res) without it, also right after camera motion (render-sota-b-r4.md).
        self.restir_gi = env_now("FLINT_SOTA_GI_RESTIR_GI") == "1"
        self.restir_gi_hybrid = False
        self.restir_gi_spatial = 2
        self.restir_gi_radius = Float32(12.0)
        self.restir_gi_max_m = Float32(30.0)
        self.svgf_history_fix = True
        self.svgf_jitter_tol = env_now("FLINT_SOTA_GI_SVGF_JITTER_TOL") != "0"
        self.svgf_lowres_history = Float32(48.0)
        var env_cap = env_now("FLINT_SOTA_GI_SVGF_LOWRES_CAP")
        if env_cap == "0" or env_cap == "24":
            self.svgf_lowres_history = Float32(0.0)
        elif env_cap == "32":
            self.svgf_lowres_history = Float32(32.0)
        elif env_cap == "64":
            self.svgf_lowres_history = Float32(64.0)
        self.resolve_scale = 2
        var env_scale = env_now("FLINT_SOTA_GI_RESOLVE_SCALE")
        if env_scale == "1":
            self.resolve_scale = 1
        elif env_scale == "4":
            self.resolve_scale = 4
        self.resolve_jitter = env_now("FLINT_SOTA_GI_RESOLVE_JITTER") != "0"
        self.upsample_tent = Float32(0.5)
        var env_tent = env_now("FLINT_SOTA_GI_UPS_TENT")
        if env_tent == "0.25":
            self.upsample_tent = Float32(0.25)
        elif env_tent == "1":
            self.upsample_tent = Float32(1.0)
        self.pipe_wrc = 0
        self.pipe_resolve = 0
        self.pipe_upsample = 0
        self.cur_scale = 1
        self.cur_jx = 0
        self.cur_jy = 0
        self.prev_jx = 0
        self.prev_jy = 0
        self.last_scale = 1
        self.accum_frames = 0
        self.wrc_frames = 0
        self.last_cam_pos = Vec3(1e30, 0.0, 0.0)
        self.last_cam_fwd = Vec3(0.0, 0.0, 0.0)
        self.last_mode = -1
        self.last_inst_count = -1
        self.last_base = (0, 0, 0)
        self.last_sig = -1
        self.restir_frames = 0
        self.last_restir = 0
        self.restir_active = 0

    def signature(self) -> Int:
        """Everything besides camera / scene that changes the converged image."""
        var sig = self.pt_bounces + 16 * self.pixel_rays + 4096 * (1 if self.pt_wrc_terminate else 0)
        sig = sig * 31 + self.wrc_dims_x + 64 * self.wrc_dims_y + 4096 * self.wrc_dims_z
        return sig * 31 + Int(self.wrc_spacing * Float32(1000.0)) + Int(self.firefly_clamp * Float32(1000.0)) * 7919

    def probe_count(self) -> Int:
        return self.wrc_dims_x * self.wrc_dims_y * self.wrc_dims_z

    def clamp_dims(mut self):
        self.wrc_dims_x = max(2, min(32, self.wrc_dims_x))
        self.wrc_dims_y = max(2, min(16, self.wrc_dims_y))
        self.wrc_dims_z = max(2, min(32, self.wrc_dims_z))
        self.wrc_spacing = max(Float32(0.05), self.wrc_spacing)
        self.pt_bounces = max(1, min(8, self.pt_bounces))
        self.pixel_rays = max(1, min(64, self.pixel_rays))
        self.wrc_max_history = max(1, min(1024, self.wrc_max_history))
        self.contact_rays = max(0, min(16, self.contact_rays))
        self.restir_di_spatial = max(0, min(8, self.restir_di_spatial))
        self.restir_gi_spatial = max(0, min(8, self.restir_gi_spatial))

    def uses_wrc(self, mode: Int) -> Bool:
        if mode == SOTA_GI_MODE_WRC or mode == SOTA_GI_MODE_HYBRID:
            return True
        return mode == SOTA_GI_MODE_PATH and self.pt_wrc_terminate

    def needs_surface_cache(self, mode: Int) -> Bool:
        return mode == SOTA_GI_MODE_LEGACY or mode == SOTA_GI_MODE_HYBRID

    def effective_resolve_scale(self, mode: Int) -> Int:
        """Resolve scale for `mode`: photo-mode accumulation and the probe-only mode without contact rays
        (no per-pixel rays, so nothing to save) stay at full resolution."""
        if self.accumulate or mode == SOTA_GI_MODE_LEGACY:
            return 1
        if mode == SOTA_GI_MODE_WRC and self.contact_rays == 0:
            return 1
        if self.resolve_scale >= 4:
            return 4
        return 2 if self.resolve_scale >= 2 else 1

    def svgf_history_cap(self, mode: Int, light_changed: Bool = False) -> Float32:
        """SVGF temporal history cap for this frame (0 = the v5 default of 24 frames). The reduced-resolution
        path tracer traces each pixel once per `scale^2` frames, so a longer history keeps its sample count
        up (measured: the half-res loss is sample count, see render-sota-b-r4.md). In the GI boost window after
        a light change the cap falls back to 24, which immediately puts every pixel back on v5's blend weight."""
        if light_changed or mode != SOTA_GI_MODE_PATH or self.cur_scale <= 1:
            return Float32(0.0)
        return self.svgf_lowres_history if self.svgf_lowres_history > Float32(1.0) else Float32(0.0)

    def resolve_grid(self, render_w: Int, render_h: Int) -> Tuple[Int, Int]:
        """Dispatch size (in pixels) of the resolve pass this frame."""
        var sc = self.cur_scale
        return ((render_w + sc - 1) // sc, (render_h + sc - 1) // sc)

    def reset_accumulation(mut self):
        self.accum_frames = 0

    def reset_cache(mut self):
        self.wrc_frames = 0
        self.accum_frames = 0

    def write_ctrl(
        mut self,
        cpu_address: Int,
        mode: Int,
        frame_index: Int,
        cam_pos: Vec3,
        cam_fwd: Vec3,
        render_w: Int,
        render_h: Int,
        max_pix: Int,
        inst_count: Int,
        restir_di: Bool = False,
        region_ready: Bool = False,
    ):
        """Packs the control block for this frame (call once per frame before recording).
        `restir_di`: the renderer runs ReSTIR DI this frame and wants reuse; `region_ready`: the
        area-B region (history and reservoir banks) is allocated."""
        self.clamp_dims()
        var base = SOTA_GI_BASE
        var c = self.wrc_center
        if not self.wrc_center_fixed:
            var half_z = Float32(self.wrc_dims_z) * self.wrc_spacing * Float32(0.5)
            c = cam_pos + cam_fwd * (half_z * Float32(0.6))
        var bx = Int((c.x / self.wrc_spacing) + Float32(SOTA_GI_CELL_BIAS) + Float32(0.5)) - self.wrc_dims_x // 2
        var by = Int((c.y / self.wrc_spacing) + Float32(SOTA_GI_CELL_BIAS) + Float32(0.5)) - self.wrc_dims_y // 2
        var bz = Int((c.z / self.wrc_spacing) + Float32(SOTA_GI_CELL_BIAS) + Float32(0.5)) - self.wrc_dims_z // 2

        # Progressive accumulation restarts whenever anything that changes the image changes
        var moved = (cam_pos - self.last_cam_pos).length() > Float32(1e-5) or (cam_fwd - self.last_cam_fwd).length() > Float32(1e-5)
        var sig = self.signature()
        if moved or mode != self.last_mode or inst_count != self.last_inst_count or not self.accumulate or sig != self.last_sig:
            self.accum_frames = 0
        self.last_sig = sig
        if mode != self.last_mode:
            self.wrc_frames = 0
            self.restir_frames = 0
        self.last_cam_pos = cam_pos
        self.last_cam_fwd = cam_fwd
        self.last_mode = mode
        self.last_inst_count = inst_count
        self.last_base = (bx, by, bz)

        var flags: UInt32 = 0
        if self.wrc_frames > 0 and self.uses_wrc(mode):
            flags |= SOTA_FLAG_WRC_VALID
        if self.refl_use_wrc:
            flags |= SOTA_FLAG_REFL_WRC
        if self.accumulate:
            flags |= SOTA_FLAG_ACCUMULATE
        if self.pt_wrc_terminate:
            flags |= SOTA_FLAG_PT_WRC_TERMINATE
        if mode == SOTA_GI_MODE_HYBRID:
            flags |= SOTA_FLAG_WRC_FEEDS_SURFCACHE
            if self.wrc_probe_use_surfcache:
                flags |= SOTA_FLAG_PROBE_USE_SURFCACHE
        var a = cpu_address + base * 4
        put32(a + SOTA_CTRL_MODE * 4, UInt32(mode))
        put32(a + SOTA_CTRL_FLAGS * 4, flags)
        put32(a + SOTA_CTRL_PT_BOUNCES * 4, UInt32(self.pt_bounces))
        put32(a + SOTA_CTRL_PIXEL_RAYS * 4, UInt32(self.pixel_rays))
        put32(a + SOTA_CTRL_ACCUM_N * 4, UInt32(self.accum_frames))
        put32(a + SOTA_CTRL_FRAME * 4, UInt32(frame_index & 0x3FFFFFFF))
        put32(a + SOTA_CTRL_MAX_HISTORY * 4, UInt32(self.wrc_max_history))
        put_float(a + SOTA_CTRL_CLAMP * 4, self.firefly_clamp)
        put32(a + (SOTA_CTRL_BASE_X + 0) * 4, UInt32(bx))
        put32(a + (SOTA_CTRL_BASE_X + 1) * 4, UInt32(by))
        put32(a + (SOTA_CTRL_BASE_X + 2) * 4, UInt32(bz))
        put_float(a + SOTA_CTRL_SPACING * 4, self.wrc_spacing)
        put32(a + (SOTA_CTRL_DIM_X + 0) * 4, UInt32(self.wrc_dims_x))
        put32(a + (SOTA_CTRL_DIM_X + 1) * 4, UInt32(self.wrc_dims_y))
        put32(a + (SOTA_CTRL_DIM_X + 2) * 4, UInt32(self.wrc_dims_z))
        put32(a + SOTA_CTRL_PROBE_COUNT * 4, UInt32(self.probe_count()))
        var prev_bank = self.wrc_frames & 1
        put32(a + SOTA_CTRL_PREV_BANK * 4, UInt32(sota_gi_bank_offset(prev_bank)))
        put32(a + SOTA_CTRL_CUR_BANK * 4, UInt32(sota_gi_bank_offset(1 - prev_bank)))
        put32(a + SOTA_CTRL_GBUF * 4, UInt32(sota_gi_gbuf_offset()))
        put32(a + SOTA_CTRL_ACCUM * 4, UInt32(sota_gi_accum_offset(max_pix)))
        put32(a + SOTA_CTRL_RENDER_W * 4, UInt32(render_w))
        put32(a + SOTA_CTRL_RENDER_H * 4, UInt32(render_h))

        # ReSTIR + contact words
        var rflags: UInt32 = 0
        if region_ready:
            if restir_di and self.restir_di_reuse:
                rflags |= SOTA_RESTIR_DI
            if not self.accumulate and (
                (mode == SOTA_GI_MODE_PATH and self.restir_gi) or (mode == SOTA_GI_MODE_HYBRID and self.restir_gi_hybrid)
            ):
                rflags |= SOTA_RESTIR_GI
        # Reduced-resolution resolve: pick this frame's representative sub-pixel of every block
        var scale = self.effective_resolve_scale(mode)
        var phase = frame_index if self.resolve_jitter else 0
        var j = sota_gi_bayer_offset(scale, phase)
        if scale == self.last_scale:
            self.prev_jx = self.cur_jx
            self.prev_jy = self.cur_jy
        else:
            self.prev_jx = j[0]
            self.prev_jy = j[1]
        self.cur_scale = scale
        self.cur_jx = j[0]
        self.cur_jy = j[1]
        put32(a + SOTA_CTRL_LOWRES * 4, UInt32(sota_gi_lowres_offset(max_pix)))
        if rflags != self.last_restir or scale != self.last_scale:
            self.restir_frames = 0
        self.last_scale = scale
        self.last_restir = rflags
        self.restir_active = rflags
        if rflags != 0:
            rflags |= SOTA_RESTIR_HIST_WRITE
            if self.restir_frames > 0:
                rflags |= SOTA_RESTIR_HIST_VALID
        var cur = frame_index & 1
        put32(a + SOTA_CTRL_RESTIR_FLAGS * 4, rflags)
        put32(a + SOTA_CTRL_HIST_PREV * 4, UInt32(sota_gi_hist_offset(max_pix, 1 - cur)))
        put32(a + SOTA_CTRL_HIST_CUR * 4, UInt32(sota_gi_hist_offset(max_pix, cur)))
        put32(a + SOTA_CTRL_DI_PREV * 4, UInt32(sota_gi_di_offset(max_pix, 1 - cur)))
        put32(a + SOTA_CTRL_DI_CUR * 4, UInt32(sota_gi_di_offset(max_pix, cur)))
        put32(a + SOTA_CTRL_GI_PREV * 4, UInt32(sota_gi_gi_offset(max_pix, 1 - cur)))
        put32(a + SOTA_CTRL_GI_CUR * 4, UInt32(sota_gi_gi_offset(max_pix, cur)))
        put32(a + SOTA_CTRL_DI_SPATIAL * 4, UInt32(self.restir_di_spatial))
        put_float(a + SOTA_CTRL_DI_RADIUS * 4, self.restir_di_radius)
        put_float(a + SOTA_CTRL_DI_MCAP * 4, max(Float32(1.0), self.restir_di_max_m))
        put32(a + SOTA_CTRL_GI_SPATIAL * 4, UInt32(self.restir_gi_spatial))
        put_float(a + SOTA_CTRL_GI_RADIUS * 4, self.restir_gi_radius)
        put_float(a + SOTA_CTRL_GI_MCAP * 4, max(Float32(1.0), self.restir_gi_max_m))
        put32(a + SOTA_CTRL_CONTACT_RAYS * 4, UInt32(self.contact_rays))
        put_float(a + SOTA_CTRL_CONTACT_RADIUS * 4, self.contact_radius if self.contact_radius > Float32(0.0) else self.wrc_spacing)
        put_float(a + (SOTA_CTRL_CAM_X + 0) * 4, cam_pos.x)
        put_float(a + (SOTA_CTRL_CAM_X + 1) * 4, cam_pos.y)
        put_float(a + (SOTA_CTRL_CAM_X + 2) * 4, cam_pos.z)

    def end_frame(mut self, mode: Int):
        """Advance the WRC bank / accumulation counters after a frame using `mode` was recorded."""
        if self.uses_wrc(mode):
            self.wrc_frames += 1
        if self.accumulate and mode != SOTA_GI_MODE_LEGACY:
            self.accum_frames += 1
        if self.restir_active != 0:
            self.restir_frames += 1


# ---------------------------------------------------------------------------------------------
# CPU twins (used by the tests and by anyone validating the GPU numbers)
# ---------------------------------------------------------------------------------------------
def sota_sh9_basis_cpu(d: Vec3) -> List[Float32]:
    var out = List[Float32](length=9, fill=0.0)
    out[0] = Float32(0.282095)
    out[1] = Float32(0.488603) * d.y
    out[2] = Float32(0.488603) * d.z
    out[3] = Float32(0.488603) * d.x
    out[4] = Float32(1.092548) * d.x * d.y
    out[5] = Float32(1.092548) * d.y * d.z
    out[6] = Float32(0.315392) * (Float32(3.0) * d.z * d.z - Float32(1.0))
    out[7] = Float32(1.092548) * d.x * d.z
    out[8] = Float32(0.546274) * (d.x * d.x - d.y * d.y)
    return out^


def sota_fib_dir_cpu(i: Int, n: Int, u: Float32, v: Float32) -> Vec3:
    """Jittered spherical-Fibonacci direction `i` of `n` (per-frame jitter `u, v` in [0, 1))."""
    from std.math import cos, sin
    var z = Float32(1.0) - Float32(2.0) * (Float32(i) + u) / Float32(n)
    var r = sqrt(max(Float32(0.0), Float32(1.0) - z * z))
    var ph = Float32(i) * Float32(0.6180339887) + v
    ph = ph - Float32(Int(ph))
    var phi = Float32(2.0) * _PI * ph
    return Vec3(r * cos(phi), r * sin(phi), z)


def sota_sh9_project_cpu(dirs: List[Vec3], radiance: List[Vec3]) -> List[Vec3]:
    """Monte-Carlo SH9 projection of uniformly distributed radiance samples."""
    var out = List[Vec3](length=9, fill=Vec3(0.0, 0.0, 0.0))
    var n = len(dirs)
    var w = Float32(4.0) * _PI / Float32(max(1, n))
    for i in range(n):
        var y = sota_sh9_basis_cpu(dirs[i])
        for k in range(9):
            out[k] = out[k] + radiance[i] * (y[k] * w)
    return out^


def sota_sh9_irradiance_over_pi_cpu(sh: List[Vec3], n: Vec3) -> Vec3:
    """`E(n) / pi` (the units of Flint's demodulated GI channel) from SH9 radiance."""
    var y = sota_sh9_basis_cpu(n)
    var acc = Vec3(0.0, 0.0, 0.0)
    for k in range(9):
        acc = acc + sh[k] * (y[k] * sota_sh9_band_weight(k))
    return Vec3(max(Float32(0.0), acc.x), max(Float32(0.0), acc.y), max(Float32(0.0), acc.z))


def sota_chebyshev_cpu(mean: Float32, mean2: Float32, dist: Float32) -> Float32:
    """One-sided Chebyshev visibility, cubed (DDGI)."""
    if dist <= mean:
        return Float32(1.0)
    var variance = max(abs(mean2 - mean * mean), Float32(1e-4))
    var d = dist - mean
    var c = variance / (variance + d * d)
    return c * c * c


def sota_wrc_slot_cpu(cell: Int, dim: Int) -> Int:
    """Toroidal storage slot of a (biased, non-negative) cell coordinate."""
    return cell % dim


struct SotaHit(Copyable, Movable):
    """SPIR-V ids describing a committed ray-query hit."""
    var hit: UInt32  # bool
    var t: UInt32
    var px: UInt32
    var py: UInt32
    var pz: UInt32
    var nx: UInt32  # faces the incoming ray
    var ny: UInt32
    var nz: UInt32
    var kd_r: UInt32  # diffuse albedo (1 - metallic folded in)
    var kd_g: UInt32
    var kd_b: UInt32
    var em_r: UInt32
    var em_g: UInt32
    var em_b: UInt32
    var backface: UInt32  # bool: hit the back of a single-sided triangle
    var sc_base: UInt32
    var prim: UInt32

    def __init__(out self):
        self.hit = 0
        self.t = 0
        self.px = 0
        self.py = 0
        self.pz = 0
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.kd_r = 0
        self.kd_g = 0
        self.kd_b = 0
        self.em_r = 0
        self.em_g = 0
        self.em_b = 0
        self.backface = 0
        self.sc_base = 0
        self.prim = 0


def emit_sota_trace_surface(
    mut s: AaaSpirvModule,
    tlas_var: UInt32,
    b_verts: UInt32,
    b_indices: UInt32,
    b_instances: UInt32,
    b_materials: UInt32,
    bindless_arr: Tuple[UInt32, UInt32],
    ox: UInt32,
    oy: UInt32,
    oz: UInt32,
    dx: UInt32,
    dy: UInt32,
    dz: UInt32,
    t_max: UInt32,
) -> SotaHit:
    """Closest-hit ray (alpha-tested, glass skipped) returning position, ray-facing normal,
    diffuse albedo and emission of the hit. Fields are only meaningful when `hit` is true;
    miss values are zeros so callers can use them unconditionally."""
    var rq = s.ray_query_var()
    s.ray_query_init(rq, tlas_var, s.uint(RAY_FLAG_NONE_KHR), s.uint(AAA_RT_MASK_OPAQUE), s.vec3(ox, oy, oz), s.float(0.001), s.vec3(dx, dy, dz), t_max)
    _emit_rt_candidate_alpha_loop(s, rq, b_verts, b_indices, b_instances, b_materials, bindless_arr)

    var r = SotaHit()
    var v_t = s.fvar(t_max)
    var v_px = s.fvar(s.float(0.0))
    var v_py = s.fvar(s.float(0.0))
    var v_pz = s.fvar(s.float(0.0))
    var v_nx = s.fvar(s.float(0.0))
    var v_ny = s.fvar(s.float(1.0))
    var v_nz = s.fvar(s.float(0.0))
    var v_kr = s.fvar(s.float(0.0))
    var v_kg = s.fvar(s.float(0.0))
    var v_kb = s.fvar(s.float(0.0))
    var v_er = s.fvar(s.float(0.0))
    var v_eg = s.fvar(s.float(0.0))
    var v_eb = s.fvar(s.float(0.0))
    var v_bf = s.uvar(s.uint(0))
    var v_sc = s.uvar(s.uint(0xFFFFFF))
    var v_pr = s.uvar(s.uint(0))

    var is_hit = s.ine(s.ray_query_committed_type(rq), s.uint(0))
    s.begin_if(is_hit)
    var t_hit = s.ray_query_committed_t(rq)
    var inst_id = s.ray_query_committed_instance_id(rq)
    var prim_id = s.ray_query_committed_prim_index(rq)
    var bary = s.ray_query_committed_barycentrics(rq)
    var b1 = s.extract_f(bary, 0)
    var b2 = s.extract_f(bary, 1)
    var b0 = s.fsub(s.fsub(s.float(1.0), b1), b2)
    var inst_base = s.imul(inst_id, s.uint(UInt32(AAA_INSTANCE_WORDS)))
    var first_idx = s.load_word(b_instances, s.iadd(inst_base, s.uint(38)))
    var w39 = s.load_word(b_instances, s.iadd(inst_base, s.uint(39)))
    var mat_id = s.bit_and(w39, s.uint(0xFFFF))
    var double_sided = s.ine(s.bit_and(s.shr(w39, s.uint(16)), s.uint(3)), s.uint(0))
    var idx_off = s.iadd(first_idx, s.imul(prim_id, s.uint(3)))
    var v0 = s.imul(s.load_word(b_indices, idx_off), s.uint(12))
    var v1 = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(1))), s.uint(12))
    var v2 = s.imul(s.load_word(b_indices, s.iadd(idx_off, s.uint(2))), s.uint(12))
    var lnx = s.fma(s.load_float(b_verts, s.iadd(v0, s.uint(3))), b0, s.fma(s.load_float(b_verts, s.iadd(v1, s.uint(3))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2, s.uint(3))), b2)))
    var lny = s.fma(s.load_float(b_verts, s.iadd(v0, s.uint(4))), b0, s.fma(s.load_float(b_verts, s.iadd(v1, s.uint(4))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2, s.uint(4))), b2)))
    var lnz = s.fma(s.load_float(b_verts, s.iadd(v0, s.uint(5))), b0, s.fma(s.load_float(b_verts, s.iadd(v1, s.uint(5))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2, s.uint(5))), b2)))
    var wn_raw = emit_mat4_mul_dir(s, b_instances, inst_base, lnx, lny, lnz)
    var wn = emit_vec3_normalize(s, wn_raw[0], wn_raw[1], wn_raw[2])
    var n_dot_d = emit_vec3_dot(s, wn[0], wn[1], wn[2], dx, dy, dz)
    var back = s.fgt(n_dot_d, s.float(0.0))
    var flip = s.select_f(back, s.float(-1.0), s.float(1.0))
    s.store(v_bf, s.select_u(s.logical_and(back, s.logical_not(double_sided)), s.uint(1), s.uint(0)))
    s.store(v_nx, s.fmul(wn[0], flip))
    s.store(v_ny, s.fmul(wn[1], flip))
    s.store(v_nz, s.fmul(wn[2], flip))
    s.store(v_t, t_hit)
    s.store(v_px, s.fma(dx, t_hit, ox))
    s.store(v_py, s.fma(dy, t_hit, oy))
    s.store(v_pz, s.fma(dz, t_hit, oz))
    s.store(v_sc, s.ray_query_committed_custom_index(rq))
    s.store(v_pr, prim_id)

    var m_base = s.imul(mat_id, s.uint(UInt32(AAA_MATERIAL_WORDS)))
    var met = s.load_float(b_materials, s.iadd(m_base, s.uint(5)))
    var kd_s = s.fsub(s.float(1.0), met)  # diffuse-only bounce: the hit's specular lobe is not traced
    var ar_v = s.fvar(s.load_float(b_materials, m_base))
    var ag_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(1))))
    var ab_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(2))))
    var er_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(8))))
    var eg_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(9))))
    var eb_v = s.fvar(s.load_float(b_materials, s.iadd(m_base, s.uint(10))))
    var no_furnace = s.ieq(s.uint(0), s.uint(1))
    var vc0 = emit_decode_vertex_color_rgb_spv(s, s.load_word(b_verts, s.iadd(v0, s.uint(11))), no_furnace)
    var vc1 = emit_decode_vertex_color_rgb_spv(s, s.load_word(b_verts, s.iadd(v1, s.uint(11))), no_furnace)
    var vc2 = emit_decode_vertex_color_rgb_spv(s, s.load_word(b_verts, s.iadd(v2, s.uint(11))), no_furnace)
    var vcr = s.fma(vc0[0], b0, s.fma(vc1[0], b1, s.fmul(vc2[0], b2)))
    var vcg = s.fma(vc0[1], b0, s.fma(vc1[1], b1, s.fmul(vc2[1], b2)))
    var vcb = s.fma(vc0[2], b0, s.fma(vc1[2], b1, s.fmul(vc2[2], b2)))
    var tex_pk = s.load_word(b_materials, s.iadd(m_base, s.uint(23)))
    var t_alb = s.bit_and(tex_pk, s.uint(0xFF))
    s.begin_if(s.ult(t_alb, s.uint(0xFF)))
    var hu = s.fma(s.load_float(b_verts, s.iadd(v0, s.uint(9))), b0, s.fma(s.load_float(b_verts, s.iadd(v1, s.uint(9))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2, s.uint(9))), b2)))
    var hv = s.fma(s.load_float(b_verts, s.iadd(v0, s.uint(10))), b0, s.fma(s.load_float(b_verts, s.iadd(v1, s.uint(10))), b1, s.fmul(s.load_float(b_verts, s.iadd(v2, s.uint(10))), b2)))
    var tx = s.sample_bindless_lod(bindless_arr, t_alb, s.vec2(hu, hv), s.float(3.0))
    s.store(ar_v, s.fmul(s.load_f(ar_v), s.extract_f(tx, 0)))
    s.store(ag_v, s.fmul(s.load_f(ag_v), s.extract_f(tx, 1)))
    s.store(ab_v, s.fmul(s.load_f(ab_v), s.extract_f(tx, 2)))
    s.store(er_v, s.fmul(s.load_f(er_v), s.extract_f(tx, 0)))
    s.store(eg_v, s.fmul(s.load_f(eg_v), s.extract_f(tx, 1)))
    s.store(eb_v, s.fmul(s.load_f(eb_v), s.extract_f(tx, 2)))
    s.end_if()
    s.store(v_kr, s.fmul(s.fmul(s.load_f(ar_v), vcr), kd_s))
    s.store(v_kg, s.fmul(s.fmul(s.load_f(ag_v), vcg), kd_s))
    s.store(v_kb, s.fmul(s.fmul(s.load_f(ab_v), vcb), kd_s))
    s.store(v_er, s.fmul(s.load_f(er_v), vcr))
    s.store(v_eg, s.fmul(s.load_f(eg_v), vcg))
    s.store(v_eb, s.fmul(s.load_f(eb_v), vcb))
    s.end_if()

    r.hit = is_hit
    r.t = s.load_f(v_t)
    r.px = s.load_f(v_px)
    r.py = s.load_f(v_py)
    r.pz = s.load_f(v_pz)
    r.nx = s.load_f(v_nx)
    r.ny = s.load_f(v_ny)
    r.nz = s.load_f(v_nz)
    r.kd_r = s.load_f(v_kr)
    r.kd_g = s.load_f(v_kg)
    r.kd_b = s.load_f(v_kb)
    r.em_r = s.load_f(v_er)
    r.em_g = s.load_f(v_eg)
    r.em_b = s.load_f(v_eb)
    r.backface = s.ine(s.load_u(v_bf), s.uint(0))
    r.sc_base = s.load_u(v_sc)
    r.prim = s.load_u(v_pr)
    return r^


def emit_sota_direct_at(
    mut s: AaaSpirvModule,
    tlas_var: UInt32,
    b_verts: UInt32,
    b_indices: UInt32,
    b_instances: UInt32,
    b_materials: UInt32,
    b_globals: UInt32,
    b_clusters: UInt32,
    bindless_arr: Tuple[UInt32, UInt32],
    hx: UInt32,
    hy: UInt32,
    hz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    """Direct irradiance / pi at a hit: sun (one RT shadow ray, NEE) + clustered local lights."""
    var r_flags = s.load_word(b_globals, s.uint(99))
    var sun_x = s.load_float(b_globals, s.uint(48))
    var sun_y = s.load_float(b_globals, s.uint(49))
    var sun_z = s.load_float(b_globals, s.uint(50))
    var ndl = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, sun_x, sun_y, sun_z))
    var vis = s.fvar(s.float(0.0))
    s.begin_if(s.fgt(ndl, s.float(0.0005)))
    var v = emit_rt_shadow_query(
        s, tlas_var, hx, hy, hz, nx, ny, nz, sun_x, sun_y, sun_z, s.float(500.0),
        b_verts=b_verts, b_indices=b_indices, b_instances=b_instances, b_materials=b_materials, bindless_arr=bindless_arr,
    )
    s.store(vis, v)
    s.end_if()
    var k = s.fmul(s.fmul(ndl, s.load_f(vis)), s.float(0.318309886))
    var d_r = s.fvar(s.fmul(s.load_float(b_globals, s.uint(52)), k))
    var d_g = s.fvar(s.fmul(s.load_float(b_globals, s.uint(53)), k))
    var d_b = s.fvar(s.fmul(s.load_float(b_globals, s.uint(54)), k))
    s.begin_if(s.ine(s.bit_and(r_flags, s.uint(32)), s.uint(0)))
    var loc = emit_froxel_local_lighting_at_hit(s, b_clusters, b_globals, hx, hy, hz, nx, ny, nz)
    s.store(d_r, s.fadd(s.load_f(d_r), loc[0]))
    s.store(d_g, s.fadd(s.load_f(d_g), loc[1]))
    s.store(d_b, s.fadd(s.load_f(d_b), loc[2]))
    s.end_if()
    return (s.load_f(d_r), s.load_f(d_g), s.load_f(d_b))


# ---------------------------------------------------------------------------------------------
# Pass 1: world radiance cache probe update (one 64-wide workgroup per probe)
# ---------------------------------------------------------------------------------------------
def sota_wrc_update_shader(enable_ray_query: Bool) -> List[UInt32]:
    """Probe update. Workgroup X = storage slot. Reads the previous bank, writes the current bank."""
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=SOTA_GI_PROBE_RAYS,
        group_height=1,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        enable_ray_query=enable_ray_query,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_sc = s.buffer(BIND_AAA_SURFCACHE)
    var b_clusters = s.buffer(BIND_AAA_CLUSTERS)
    var env_cube = s.sampled_image_2d(BIND_AAA_ENV_CUBE, is_cube=True)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var tlas_var: UInt32 = 0
    if enable_ray_query:
        tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var sh_l = s.workgroup_array_f32(SOTA_GI_PROBE_RAYS * 3)
    var sh_d = s.workgroup_array_f32(SOTA_GI_PROBE_RAYS * 3)
    var sh_t = s.workgroup_array_f32(SOTA_GI_PROBE_RAYS)
    var sh_bf = s.workgroup_array_f32(SOTA_GI_PROBE_RAYS)

    var slot = s.workgroup_axis(0)
    var lane = s.local_axis(0)
    var count = _ctrl(s, b_sc, SOTA_CTRL_PROBE_COUNT)
    s.return_if(s.uge(slot, count))

    var flags = _ctrl(s, b_sc, SOTA_CTRL_FLAGS)
    var prev_bank = _ctrl(s, b_sc, SOTA_CTRL_PREV_BANK)
    var cur_bank = _ctrl(s, b_sc, SOTA_CTRL_CUR_BANK)
    var frame = _ctrl(s, b_sc, SOTA_CTRL_FRAME)
    var spacing = _ctrl_f(s, b_sc, SOTA_CTRL_SPACING)
    var dim_x = _ctrl(s, b_sc, SOTA_CTRL_DIM_X)
    var dim_y = _ctrl(s, b_sc, SOTA_CTRL_DIM_X + 1)
    var dim_z = _ctrl(s, b_sc, SOTA_CTRL_DIM_X + 2)
    var base_x = _ctrl(s, b_sc, SOTA_CTRL_BASE_X)
    var base_y = _ctrl(s, b_sc, SOTA_CTRL_BASE_X + 1)
    var base_z = _ctrl(s, b_sc, SOTA_CTRL_BASE_X + 2)
    var wrc_valid = s.ine(s.bit_and(flags, s.uint(SOTA_FLAG_WRC_VALID)), s.uint(0))
    var probe_use_sc = s.ine(s.bit_and(flags, s.uint(SOTA_FLAG_PROBE_USE_SURFCACHE)), s.uint(0))

    # Slot -> the cell of the scrolled window that maps to it
    var sx = s.umod(slot, dim_x)
    var sy = s.umod(s.udiv(slot, dim_x), dim_y)
    var sz = s.udiv(slot, s.imul(dim_x, dim_y))
    var cx = s.iadd(base_x, s.umod(s.isub(s.iadd(sx, dim_x), s.umod(base_x, dim_x)), dim_x))
    var cy = s.iadd(base_y, s.umod(s.isub(s.iadd(sy, dim_y), s.umod(base_y, dim_y)), dim_y))
    var cz = s.iadd(base_z, s.umod(s.isub(s.iadd(sz, dim_z), s.umod(base_z, dim_z)), dim_z))
    var bias_f = s.float(Float32(SOTA_GI_CELL_BIAS))
    var ppx = s.fmul(s.fadd(s.fsub(s.float_from_uint(cx), bias_f), s.float(0.5)), spacing)
    var ppy = s.fmul(s.fadd(s.fsub(s.float_from_uint(cy), bias_f), s.float(0.5)), spacing)
    var ppz = s.fmul(s.fadd(s.fsub(s.float_from_uint(cz), bias_f), s.float(0.5)), spacing)
    var key = s.bit_or(
        s.bit_or(s.bit_and(cx, s.uint(0x3FF)), s.shl(s.bit_and(cy, s.uint(0x3FF)), s.uint(10))),
        s.bit_or(s.shl(s.bit_and(cz, s.uint(0x3FF)), s.uint(20)), s.uint(0x40000000)),
    )

    # Jittered spherical-Fibonacci ray of this lane
    var hj = _emit_hash(s, s.iadd(s.imul(frame, s.uint(2654435761)), s.uint(12345)))
    var hj2 = _emit_hash(s, hj[0])
    var fl = s.float_from_uint(lane)
    var dz = s.fsub(s.float(1.0), s.fdiv(s.fmul(s.float(2.0), s.fadd(fl, hj[1])), s.float(Float32(SOTA_GI_PROBE_RAYS))))
    var rr = s.fsqrt(s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fmul(dz, dz))))
    var phi = s.fmul(s.float(6.283185307), s.fract(s.fma(fl, s.float(0.6180339887), hj2[1])))
    var dx = s.fmul(rr, s.fcos(phi))
    var dy = s.fmul(rr, s.fsin(phi))

    var max_t = s.fmul(spacing, s.float(1.8))
    var l_r = s.fvar(s.float(0.0))
    var l_g = s.fvar(s.float(0.0))
    var l_b = s.fvar(s.float(0.0))
    var t_v = s.fvar(max_t)
    var bf_v = s.fvar(s.float(0.0))
    if enable_ray_query:
        var h = emit_sota_trace_surface(s, tlas_var, b_verts, b_indices, b_instances, b_materials, bindless_arr, ppx, ppy, ppz, dx, dy, dz, s.float(1000.0))
        s.begin_if(h.hit)
        s.begin_if(h.backface)
        # Inside geometry: no light, short distance (DDGI), counted for the backface ratio
        s.store(bf_v, s.float(1.0))
        s.store(t_v, s.fmul(h.t, s.float(0.2)))
        s.begin_else()
        s.store(t_v, s.nmin(h.t, max_t))
        var use_sc = s.logical_and(probe_use_sc, s.ine(h.sc_base, s.uint(0xFFFFFF)))
        s.begin_if(use_sc)
        var scw = s.imul(s.iadd(h.sc_base, h.prim), s.uint(4))
        s.store(l_r, s.load_float(b_sc, scw))
        s.store(l_g, s.load_float(b_sc, s.iadd(scw, s.uint(1))))
        s.store(l_b, s.load_float(b_sc, s.iadd(scw, s.uint(2))))
        s.begin_else()
        var hox = s.fma(h.nx, s.float(0.002), h.px)
        var hoy = s.fma(h.ny, s.float(0.002), h.py)
        var hoz = s.fma(h.nz, s.float(0.002), h.pz)
        var dl = emit_sota_direct_at(s, tlas_var, b_verts, b_indices, b_instances, b_materials, b_globals, b_clusters, bindless_arr, hox, hoy, hoz, h.nx, h.ny, h.nz)
        var ind_r = s.fvar(s.float(0.0))
        var ind_g = s.fvar(s.float(0.0))
        var ind_b = s.fvar(s.float(0.0))
        s.begin_if(wrc_valid)
        var wi = emit_sota_wrc_irradiance(s, b_sc, prev_bank, h.px, h.py, h.pz, h.nx, h.ny, h.nz)
        s.store(ind_r, wi[0])
        s.store(ind_g, wi[1])
        s.store(ind_b, wi[2])
        s.end_if()
        s.store(l_r, s.fma(h.kd_r, s.fadd(dl[0], s.load_f(ind_r)), h.em_r))
        s.store(l_g, s.fma(h.kd_g, s.fadd(dl[1], s.load_f(ind_g)), h.em_g))
        s.store(l_b, s.fma(h.kd_b, s.fadd(dl[2], s.load_f(ind_b)), h.em_b))
        s.end_if()
        s.end_if()
        s.begin_else()
        var ibl_int = s.load_float(b_globals, s.uint(55))
        var env = s.sample_cube_lod(env_cube, s.vec3(dx, dy, dz), s.float(1.0))
        s.store(l_r, s.fmul(s.extract_f(env, 0), ibl_int))
        s.store(l_g, s.fmul(s.extract_f(env, 1), ibl_int))
        s.store(l_b, s.fmul(s.extract_f(env, 2), ibl_int))
        s.end_if()

    var l3 = s.imul(lane, s.uint(3))
    s.store_workgroup_f32(sh_l[0], sh_l[1], l3, s.load_f(l_r))
    s.store_workgroup_f32(sh_l[0], sh_l[1], s.iadd(l3, s.uint(1)), s.load_f(l_g))
    s.store_workgroup_f32(sh_l[0], sh_l[1], s.iadd(l3, s.uint(2)), s.load_f(l_b))
    s.store_workgroup_f32(sh_d[0], sh_d[1], l3, dx)
    s.store_workgroup_f32(sh_d[0], sh_d[1], s.iadd(l3, s.uint(1)), dy)
    s.store_workgroup_f32(sh_d[0], sh_d[1], s.iadd(l3, s.uint(2)), dz)
    s.store_workgroup_f32(sh_t[0], sh_t[1], lane, s.load_f(t_v))
    s.store_workgroup_f32(sh_bf[0], sh_bf[1], lane, s.load_f(bf_v))
    s.workgroup_barrier()

    # Lanes 0..41 each reduce one output word over the 64 rays and blend it into the new bank
    var pb_prev = s.iadd(prev_bank, s.imul(slot, s.uint(UInt32(SOTA_GI_PROBE_WORDS))))
    var pb_cur = s.iadd(cur_bank, s.imul(slot, s.uint(UInt32(SOTA_GI_PROBE_WORDS))))
    var old_key = s.load_word(b_sc, s.iadd(pb_prev, s.uint(UInt32(SOTA_PROBE_KEY))))
    var old_hist = s.load_float(b_sc, s.iadd(pb_prev, s.uint(UInt32(SOTA_PROBE_HIST))))
    var same = s.logical_and(wrc_valid, s.ieq(old_key, key))
    var hist = s.select_f(same, old_hist, s.float(0.0))
    var max_hist = s.float_from_uint(_ctrl(s, b_sc, SOTA_CTRL_MAX_HISTORY))
    var alpha = s.fdiv(s.float(1.0), s.nmin(s.fadd(hist, s.float(1.0)), max_hist))

    s.begin_if(s.ult(lane, s.uint(27)))
    # SH coefficient k, channel c
    var k_u = s.udiv(lane, s.uint(3))
    var c_u = s.umod(lane, s.uint(3))
    var acc = s.fvar(s.float(0.0))
    var i_v = s.uvar(s.uint(0))
    s.begin_loop()
    var i_c = s.load_u(i_v)
    s.break_if(s.uge(i_c, s.uint(UInt32(SOTA_GI_PROBE_RAYS))))
    var i3 = s.imul(i_c, s.uint(3))
    var ddx = s.load_workgroup_f32(sh_d[0], sh_d[1], i3)
    var ddy = s.load_workgroup_f32(sh_d[0], sh_d[1], s.iadd(i3, s.uint(1)))
    var ddz = s.load_workgroup_f32(sh_d[0], sh_d[1], s.iadd(i3, s.uint(2)))
    var yb = _emit_sh9_basis(s, ddx, ddy, ddz)
    var y_sel = yb[8]
    for kk in range(7, -1, -1):
        y_sel = s.select_f(s.ieq(k_u, s.uint(UInt32(kk))), yb[kk], y_sel)
    var li = s.load_workgroup_f32(sh_l[0], sh_l[1], s.iadd(i3, c_u))
    s.store(acc, s.fma(li, y_sel, s.load_f(acc)))
    s.store(i_v, s.iadd(i_c, s.uint(1)))
    s.end_loop()
    var cur = s.fmul(s.load_f(acc), s.float(4.0 * _PI / Float32(SOTA_GI_PROBE_RAYS)))
    var old = s.load_float(b_sc, s.iadd(pb_prev, lane))
    var old_eff = s.select_f(same, old, cur)
    s.store_float(b_sc, s.iadd(pb_cur, lane), s.fma(s.fsub(cur, old_eff), alpha, old_eff))
    s.end_if()

    s.begin_if(s.logical_and(s.uge(lane, s.uint(27)), s.ult(lane, s.uint(39))))
    # Visibility moment: axis a, power m + 1
    var rel = s.isub(lane, s.uint(27))
    var a_u = s.udiv(rel, s.uint(2))
    var m_u = s.umod(rel, s.uint(2))
    var sgn = s.select_f(s.ieq(s.umod(a_u, s.uint(2)), s.uint(0)), s.float(1.0), s.float(-1.0))
    var comp = s.udiv(a_u, s.uint(2))
    var sw = s.fvar(s.float(0.0))
    var sm = s.fvar(s.float(0.0))
    var j_v = s.uvar(s.uint(0))
    s.begin_loop()
    var j_c = s.load_u(j_v)
    s.break_if(s.uge(j_c, s.uint(UInt32(SOTA_GI_PROBE_RAYS))))
    var dcomp = s.load_workgroup_f32(sh_d[0], sh_d[1], s.iadd(s.imul(j_c, s.uint(3)), comp))
    var cosw = s.nmax(s.float(0.0), s.fmul(dcomp, sgn))
    var w2 = s.fmul(cosw, cosw)
    var w = s.fmul(w2, w2)
    var tj = s.load_workgroup_f32(sh_t[0], sh_t[1], j_c)
    var tp = s.select_f(s.ieq(m_u, s.uint(0)), tj, s.fmul(tj, tj))
    s.store(sw, s.fadd(s.load_f(sw), w))
    s.store(sm, s.fma(w, tp, s.load_f(sm)))
    s.store(j_v, s.iadd(j_c, s.uint(1)))
    s.end_loop()
    var cur_m = s.fdiv(s.load_f(sm), s.nmax(s.load_f(sw), s.float(1e-6)))
    var old_m = s.load_float(b_sc, s.iadd(pb_prev, lane))
    var old_m_eff = s.select_f(same, old_m, cur_m)
    s.store_float(b_sc, s.iadd(pb_cur, lane), s.fma(s.fsub(cur_m, old_m_eff), alpha, old_m_eff))
    s.end_if()

    s.begin_if(s.ieq(lane, s.uint(UInt32(SOTA_PROBE_KEY))))
    s.store_word(b_sc, s.iadd(pb_cur, s.uint(UInt32(SOTA_PROBE_KEY))), key)
    s.store_float(b_sc, s.iadd(pb_cur, s.uint(UInt32(SOTA_PROBE_HIST))), s.fadd(hist, s.float(1.0)))
    var bsum = s.fvar(s.float(0.0))
    var q_v = s.uvar(s.uint(0))
    s.begin_loop()
    var q_c = s.load_u(q_v)
    s.break_if(s.uge(q_c, s.uint(UInt32(SOTA_GI_PROBE_RAYS))))
    s.store(bsum, s.fadd(s.load_f(bsum), s.load_workgroup_f32(sh_bf[0], sh_bf[1], q_c)))
    s.store(q_v, s.iadd(q_c, s.uint(1)))
    s.end_loop()
    var bf_cur = s.fmul(s.load_f(bsum), s.float(1.0 / Float32(SOTA_GI_PROBE_RAYS)))
    var bf_old = s.load_float(b_sc, s.iadd(pb_prev, s.uint(UInt32(SOTA_PROBE_BACKFACE))))
    var bf_eff = s.select_f(same, bf_old, bf_cur)
    s.store_float(b_sc, s.iadd(pb_cur, s.uint(UInt32(SOTA_PROBE_BACKFACE))), s.fma(s.fsub(bf_cur, bf_eff), alpha, bf_eff))
    s.end_if()

    return s.finish()


# ---------------------------------------------------------------------------------------------
# Pass 2: per-pixel GI resolve (probe GI / hybrid / path tracer)
# ---------------------------------------------------------------------------------------------
def sota_gi_resolve_shader(enable_ray_query: Bool) -> List[UInt32]:
    """Push[0] = render_w, Push[1] = render_h, Push[2] = 1 when SVGF demodulates GI."""
    var s = AaaSpirvModule(
        MODEL_GL_COMPUTE,
        group_width=8,
        group_height=8,
        push_words=AAA_PUSH_WORDS,
        enable_bindless=True,
        enable_ray_query=enable_ray_query,
    )
    var b_verts = s.buffer(BIND_AAA_VERTICES)
    var b_indices = s.buffer(BIND_AAA_INDICES)
    var b_instances = s.buffer(BIND_AAA_INSTANCES)
    var b_materials = s.buffer(BIND_AAA_MATERIALS)
    var b_globals = s.buffer(BIND_AAA_GLOBALS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_sc = s.buffer(BIND_AAA_SURFCACHE)
    var b_clusters = s.buffer(BIND_AAA_CLUSTERS)
    var env_cube = s.sampled_image_2d(BIND_AAA_ENV_CUBE, is_cube=True)
    var bindless_arr = s.bindless_sampled_array_2d(BIND_AAA_BINDLESS_TEX, AAA_MAX_BINDLESS)
    var tlas_var: UInt32 = 0
    if enable_ray_query:
        tlas_var = s.acceleration_structure(BIND_AAA_TLAS)

    var lx = s.invocation_axis(0)
    var ly = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var demod = s.ine(s.push_word(2), s.uint(0))
    # Round 3: reduced-resolution resolve. One thread per `sc x sc` block traces for one
    # representative pixel (sub-pixel `(jx, jy)`, rotated per frame) and writes a low-res record that
    # `sota_gi_upsample_shader` spreads back with depth / normal weights. `sc <= 1` is the full-res path.
    var sc = s.umax(s.push_word(3), s.uint(1))
    var lowres = s.uge(sc, s.uint(2))
    var jx = s.push_word(4)
    var jy = s.push_word(5)
    var lw = s.udiv(s.iadd(w_u, s.isub(sc, s.uint(1))), sc)
    var lh = s.udiv(s.iadd(h_u, s.isub(sc, s.uint(1))), sc)
    s.return_if(s.logical_or(s.uge(lx, lw), s.uge(ly, lh)))

    var mode = _ctrl(s, b_sc, SOTA_CTRL_MODE)
    var flags = _ctrl(s, b_sc, SOTA_CTRL_FLAGS)
    var frame = _ctrl(s, b_sc, SOTA_CTRL_FRAME)
    var gb_base = _ctrl(s, b_sc, SOTA_CTRL_GBUF)
    var stamp_want = s.iadd(frame, s.uint(1))
    var bx0 = s.imul(lx, sc)
    var by0 = s.imul(ly, sc)
    var px_v = s.uvar(s.umin(s.iadd(bx0, jx), s.isub(w_u, s.uint(1))))
    var py_v = s.uvar(s.umin(s.iadd(by0, jy), s.isub(h_u, s.uint(1))))
    var st0 = s.load_word(
        b_sc,
        s.iadd(s.iadd(gb_base, s.imul(s.iadd(s.imul(s.load_u(py_v), w_u), s.load_u(px_v)), s.uint(UInt32(SOTA_GI_GBUF_WORDS)))), s.uint(15)),
    )
    var found_v = s.uvar(s.select_u(s.ieq(st0, stamp_want), s.uint(1), s.uint(0)))
    # The rotated pixel is sky / not shaded: take the first shaded pixel of the block instead
    s.begin_if(s.logical_and(lowres, s.ieq(s.load_u(found_v), s.uint(0))))
    var k_s = s.uvar(s.uint(0))
    var n_s = s.imul(sc, sc)
    s.begin_loop()
    var ks = s.load_u(k_s)
    s.break_if(s.logical_or(s.uge(ks, n_s), s.ine(s.load_u(found_v), s.uint(0))))
    var cx = s.umin(s.iadd(bx0, s.umod(ks, sc)), s.isub(w_u, s.uint(1)))
    var cy = s.umin(s.iadd(by0, s.udiv(ks, sc)), s.isub(h_u, s.uint(1)))
    var st_c = s.load_word(b_sc, s.iadd(s.iadd(gb_base, s.imul(s.iadd(s.imul(cy, w_u), cx), s.uint(UInt32(SOTA_GI_GBUF_WORDS)))), s.uint(15)))
    s.begin_if(s.ieq(st_c, stamp_want))
    s.store(px_v, cx)
    s.store(py_v, cy)
    s.store(found_v, s.uint(1))
    s.end_if()
    s.store(k_s, s.iadd(ks, s.uint(1)))
    s.end_loop()
    s.end_if()
    # Round 4: 4-word f16 record (word 3 = 0 marks "no record")
    var lo = s.iadd(_ctrl(s, b_sc, SOTA_CTRL_LOWRES), s.imul(s.iadd(s.imul(ly, lw), lx), s.uint(UInt32(SOTA_GI_LOWRES_WORDS))))
    var found = s.ine(s.load_u(found_v), s.uint(0))
    s.begin_if(s.logical_and(lowres, s.logical_not(found)))
    s.store_word(b_sc, s.iadd(lo, s.uint(3)), s.uint(0))
    s.end_if()
    s.return_if(s.logical_not(found))
    s.return_if(s.ieq(mode, s.uint(0)))
    var px = s.load_u(px_v)
    var py = s.load_u(py_v)
    var pix = s.iadd(s.imul(py, w_u), px)
    var gb = s.iadd(gb_base, s.imul(pix, s.uint(UInt32(SOTA_GI_GBUF_WORDS))))

    var wx = s.load_float(b_sc, gb)
    var wy = s.load_float(b_sc, s.iadd(gb, s.uint(1)))
    var wz = s.load_float(b_sc, s.iadd(gb, s.uint(2)))
    var nx = s.load_float(b_sc, s.iadd(gb, s.uint(3)))
    var ny = s.load_float(b_sc, s.iadd(gb, s.uint(4)))
    var nz = s.load_float(b_sc, s.iadd(gb, s.uint(5)))
    var gnx = s.load_float(b_sc, s.iadd(gb, s.uint(6)))
    var gny = s.load_float(b_sc, s.iadd(gb, s.uint(7)))
    var gnz = s.load_float(b_sc, s.iadd(gb, s.uint(8)))

    var cur_bank = _ctrl(s, b_sc, SOTA_CTRL_CUR_BANK)
    var wrc_on = s.ine(s.bit_and(flags, s.uint(SOTA_FLAG_WRC_VALID)), s.uint(0))
    var terminate = s.logical_and(wrc_on, s.ine(s.bit_and(flags, s.uint(SOTA_FLAG_PT_WRC_TERMINATE)), s.uint(0)))
    var n_rays = _ctrl(s, b_sc, SOTA_CTRL_PIXEL_RAYS)
    var n_bounce = _ctrl(s, b_sc, SOTA_CTRL_PT_BOUNCES)
    var clamp_v = _ctrl_f(s, b_sc, SOTA_CTRL_CLAMP)
    var ibl_int = s.load_float(b_globals, s.uint(55))

    var gi_r = s.fvar(s.float(0.0))
    var gi_g = s.fvar(s.float(0.0))
    var gi_b = s.fvar(s.float(0.0))

    # Mode 1: probe-only GI. The WRC pass already ran this frame, so the current bank is fresh.
    s.begin_if(s.ieq(mode, s.uint(UInt32(SOTA_GI_MODE_WRC))))
    s.begin_if(wrc_on)
    var wi = emit_sota_wrc_irradiance(s, b_sc, cur_bank, wx, wy, wz, nx, ny, nz)
    s.store(gi_r, wi[0])
    s.store(gi_g, wi[1])
    s.store(gi_b, wi[2])
    if enable_ray_query:
        # Near-field contact rays: probes cannot resolve occluders closer than about one probe
        # spacing, so short rays that hit something use that surface (direct + albedo * WRC there)
        # and unoccluded rays keep the probe irradiance (near-field / far-field split).
        var n_c = _ctrl(s, b_sc, SOTA_CTRL_CONTACT_RAYS)
        s.begin_if(s.ine(n_c, s.uint(0)))
        var c_rad = _ctrl_f(s, b_sc, SOTA_CTRL_CONTACT_RADIUS)
        var inv_c = s.fdiv(s.float(1.0), s.float_from_uint(n_c))
        var ca_r = s.fvar(s.float(0.0))
        var ca_g = s.fvar(s.float(0.0))
        var ca_b = s.fvar(s.float(0.0))
        var c_seed = s.uvar(s.iadd(s.imul(pix, s.uint(7919)), s.imul(frame, s.uint(104729))))
        var c_o_x = s.fma(gnx, s.float(0.004), wx)
        var c_o_y = s.fma(gny, s.float(0.004), wy)
        var c_o_z = s.fma(gnz, s.float(0.004), wz)
        var ck_v = s.uvar(s.uint(0))
        s.begin_loop()
        var ck = s.load_u(ck_v)
        s.break_if(s.uge(ck, n_c))
        var ch1 = _emit_hash(s, s.load_u(c_seed))
        var ch2 = _emit_hash(s, ch1[0])
        s.store(c_seed, ch2[0])
        var cu1 = s.fmul(s.fadd(s.float_from_uint(ck), ch1[1]), inv_c)
        var cdir = _emit_cosine_dir(s, nx, ny, nz, cu1, ch2[1])
        var chh = emit_sota_trace_surface(
            s, tlas_var, b_verts, b_indices, b_instances, b_materials, bindless_arr,
            c_o_x, c_o_y, c_o_z, cdir[0], cdir[1], cdir[2], c_rad,
        )
        var cl_r = s.fvar(wi[0])
        var cl_g = s.fvar(wi[1])
        var cl_b = s.fvar(wi[2])
        s.begin_if(chh.hit)
        s.begin_if(chh.backface)
        s.store(cl_r, s.float(0.0))
        s.store(cl_g, s.float(0.0))
        s.store(cl_b, s.float(0.0))
        s.begin_else()
        var chx = s.fma(chh.nx, s.float(0.002), chh.px)
        var chy = s.fma(chh.ny, s.float(0.002), chh.py)
        var chz = s.fma(chh.nz, s.float(0.002), chh.pz)
        var cdl = emit_sota_direct_at(s, tlas_var, b_verts, b_indices, b_instances, b_materials, b_globals, b_clusters, bindless_arr, chx, chy, chz, chh.nx, chh.ny, chh.nz)
        var cwh = emit_sota_wrc_irradiance(s, b_sc, cur_bank, chh.px, chh.py, chh.pz, chh.nx, chh.ny, chh.nz)
        s.store(cl_r, s.fma(chh.kd_r, s.fadd(cdl[0], cwh[0]), chh.em_r))
        s.store(cl_g, s.fma(chh.kd_g, s.fadd(cdl[1], cwh[1]), chh.em_g))
        s.store(cl_b, s.fma(chh.kd_b, s.fadd(cdl[2], cwh[2]), chh.em_b))
        s.end_if()
        s.end_if()
        s.store(ca_r, s.fadd(s.load_f(ca_r), s.load_f(cl_r)))
        s.store(ca_g, s.fadd(s.load_f(ca_g), s.load_f(cl_g)))
        s.store(ca_b, s.fadd(s.load_f(ca_b), s.load_f(cl_b)))
        s.store(ck_v, s.iadd(ck, s.uint(1)))
        s.end_loop()
        s.store(gi_r, s.fmul(s.load_f(ca_r), inv_c))
        s.store(gi_g, s.fmul(s.load_f(ca_g), inv_c))
        s.store(gi_b, s.fmul(s.load_f(ca_b), inv_c))
        s.end_if()
    s.end_if()
    s.end_if()

    if enable_ray_query:
        s.begin_if(s.uge(mode, s.uint(UInt32(SOTA_GI_MODE_HYBRID))))
        var is_hybrid = s.ieq(mode, s.uint(UInt32(SOTA_GI_MODE_HYBRID)))
        var max_b = s.select_u(is_hybrid, s.uint(1), n_bounce)
        var inv_n = s.fdiv(s.float(1.0), s.float_from_uint(n_rays))
        var seed0 = s.iadd(s.imul(pix, s.uint(1973)), s.imul(frame, s.uint(9277)))
        var restir_on = s.ine(s.bit_and(_ctrl(s, b_sc, SOTA_CTRL_RESTIR_FLAGS), s.uint(SOTA_RESTIR_GI)), s.uint(0))
        var res = SotaGiReservoir(s)
        var r_seed = s.uvar(s.iadd(s.imul(pix, s.uint(48271)), s.imul(frame, s.uint(69621))))
        var k_v = s.uvar(s.uint(0))
        s.begin_loop()
        var k_c = s.load_u(k_v)
        s.break_if(s.uge(k_c, n_rays))
        # One path: start at the primary surface
        var o_x = s.fvar(s.fma(gnx, s.float(0.004), wx))
        var o_y = s.fvar(s.fma(gny, s.float(0.004), wy))
        var o_z = s.fvar(s.fma(gnz, s.float(0.004), wz))
        var s_nx = s.fvar(nx)
        var s_ny = s.fvar(ny)
        var s_nz = s.fvar(nz)
        var tp_r = s.fvar(s.float(1.0))
        var tp_g = s.fvar(s.float(1.0))
        var tp_b = s.fvar(s.float(1.0))
        var lp_r = s.fvar(s.float(0.0))
        var lp_g = s.fvar(s.float(0.0))
        var lp_b = s.fvar(s.float(0.0))
        var alive = s.uvar(s.uint(1))
        # ReSTIR GI candidate: the first vertex of this path (a far point for sky rays)
        var cs_x = s.fvar(s.float(0.0))
        var cs_y = s.fvar(s.float(0.0))
        var cs_z = s.fvar(s.float(0.0))
        var cn_x = s.fvar(s.float(0.0))
        var cn_y = s.fvar(s.float(1.0))
        var cn_z = s.fvar(s.float(0.0))
        var seed_v = s.uvar(s.iadd(seed0, s.imul(k_c, s.uint(26699))))
        var b_v = s.uvar(s.uint(0))
        s.begin_loop()
        var b_c = s.load_u(b_v)
        s.break_if(s.logical_or(s.uge(b_c, max_b), s.ieq(s.load_u(alive), s.uint(0))))
        var h1 = _emit_hash(s, s.load_u(seed_v))
        var h2 = _emit_hash(s, h1[0])
        s.store(seed_v, h2[0])
        # Stratify the first bounce over the per-pixel rays
        var u1 = s.select_f(s.ieq(b_c, s.uint(0)), s.fmul(s.fadd(s.float_from_uint(k_c), h1[1]), inv_n), h1[1])
        var dvec = _emit_cosine_dir(s, s.load_f(s_nx), s.load_f(s_ny), s.load_f(s_nz), u1, h2[1])
        var h = emit_sota_trace_surface(
            s, tlas_var, b_verts, b_indices, b_instances, b_materials, bindless_arr,
            s.load_f(o_x), s.load_f(o_y), s.load_f(o_z), dvec[0], dvec[1], dvec[2], s.float(1000.0),
        )
        s.begin_if(s.ieq(b_c, s.uint(0)))
        s.store(cs_x, s.select_f(h.hit, h.px, s.fma(dvec[0], s.float(10000.0), wx)))
        s.store(cs_y, s.select_f(h.hit, h.py, s.fma(dvec[1], s.float(10000.0), wy)))
        s.store(cs_z, s.select_f(h.hit, h.pz, s.fma(dvec[2], s.float(10000.0), wz)))
        s.store(cn_x, s.select_f(h.hit, h.nx, s.fneg(dvec[0])))
        s.store(cn_y, s.select_f(h.hit, h.ny, s.fneg(dvec[1])))
        s.store(cn_z, s.select_f(h.hit, h.nz, s.fneg(dvec[2])))
        s.end_if()
        s.begin_if(h.hit)
        s.begin_if(h.backface)
        s.store(alive, s.uint(0))
        s.begin_else()
        var use_sc = s.logical_and(is_hybrid, s.ine(h.sc_base, s.uint(0xFFFFFF)))
        s.begin_if(use_sc)
        # Hybrid: the surface cache holds the full outgoing radiance of the hit triangle
        var scw = s.imul(s.iadd(h.sc_base, h.prim), s.uint(4))
        s.store(lp_r, s.fma(s.load_f(tp_r), s.load_float(b_sc, scw), s.load_f(lp_r)))
        s.store(lp_g, s.fma(s.load_f(tp_g), s.load_float(b_sc, s.iadd(scw, s.uint(1))), s.load_f(lp_g)))
        s.store(lp_b, s.fma(s.load_f(tp_b), s.load_float(b_sc, s.iadd(scw, s.uint(2))), s.load_f(lp_b)))
        s.store(alive, s.uint(0))
        s.begin_else()
        var hox = s.fma(h.nx, s.float(0.002), h.px)
        var hoy = s.fma(h.ny, s.float(0.002), h.py)
        var hoz = s.fma(h.nz, s.float(0.002), h.pz)
        var dl = emit_sota_direct_at(s, tlas_var, b_verts, b_indices, b_instances, b_materials, b_globals, b_clusters, bindless_arr, hox, hoy, hoz, h.nx, h.ny, h.nz)
        s.store(lp_r, s.fma(s.load_f(tp_r), s.fma(h.kd_r, dl[0], h.em_r), s.load_f(lp_r)))
        s.store(lp_g, s.fma(s.load_f(tp_g), s.fma(h.kd_g, dl[1], h.em_g), s.load_f(lp_g)))
        s.store(lp_b, s.fma(s.load_f(tp_b), s.fma(h.kd_b, dl[2], h.em_b), s.load_f(lp_b)))
        s.store(tp_r, s.fmul(s.load_f(tp_r), h.kd_r))
        s.store(tp_g, s.fmul(s.load_f(tp_g), h.kd_g))
        s.store(tp_b, s.fmul(s.load_f(tp_b), h.kd_b))
        var last = s.uge(s.iadd(b_c, s.uint(1)), max_b)
        s.begin_if(s.logical_and(last, s.logical_or(terminate, is_hybrid)))
        # End the path in the world radiance cache (the remaining bounces, pre-integrated)
        s.begin_if(wrc_on)
        var wt = emit_sota_wrc_irradiance(s, b_sc, cur_bank, h.px, h.py, h.pz, h.nx, h.ny, h.nz)
        s.store(lp_r, s.fma(s.load_f(tp_r), wt[0], s.load_f(lp_r)))
        s.store(lp_g, s.fma(s.load_f(tp_g), wt[1], s.load_f(lp_g)))
        s.store(lp_b, s.fma(s.load_f(tp_b), wt[2], s.load_f(lp_b)))
        s.end_if()
        s.end_if()
        s.store(o_x, hox)
        s.store(o_y, hoy)
        s.store(o_z, hoz)
        s.store(s_nx, h.nx)
        s.store(s_ny, h.ny)
        s.store(s_nz, h.nz)
        s.end_if()
        s.end_if()
        s.begin_else()
        var env = s.sample_cube_lod(env_cube, s.vec3(dvec[0], dvec[1], dvec[2]), s.float(0.0))
        s.store(lp_r, s.fma(s.load_f(tp_r), s.fmul(s.extract_f(env, 0), ibl_int), s.load_f(lp_r)))
        s.store(lp_g, s.fma(s.load_f(tp_g), s.fmul(s.extract_f(env, 1), ibl_int), s.load_f(lp_g)))
        s.store(lp_b, s.fma(s.load_f(tp_b), s.fmul(s.extract_f(env, 2), ibl_int), s.load_f(lp_b)))
        s.store(alive, s.uint(0))
        s.end_if()
        s.store(b_v, s.iadd(b_c, s.uint(1)))
        s.end_loop()

        # Optional firefly clamp on the path's luminance
        var pr = s.load_f(lp_r)
        var pg = s.load_f(lp_g)
        var pbv = s.load_f(lp_b)
        var lum = s.fma(pr, s.float(0.2126), s.fma(pg, s.float(0.7152), s.fmul(pbv, s.float(0.0722))))
        var cl = s.select_f(
            s.logical_and(s.fgt(clamp_v, s.float(0.0)), s.fgt(lum, clamp_v)),
            s.fdiv(clamp_v, s.nmax(lum, s.float(1e-6))),
            s.float(1.0),
        )
        s.store(gi_r, s.fma(pr, s.fmul(cl, inv_n), s.load_f(gi_r)))
        s.store(gi_g, s.fma(pg, s.fmul(cl, inv_n), s.load_f(gi_g)))
        s.store(gi_b, s.fma(pbv, s.fmul(cl, inv_n), s.load_f(gi_b)))
        s.begin_if(restir_on)
        var rh = _emit_hash(s, s.load_u(r_seed))
        s.store(r_seed, rh[0])
        emit_sota_gi_stream_candidate(
            s, res, rh[1], wx, wy, wz, nx, ny, nz,
            s.load_f(cs_x), s.load_f(cs_y), s.load_f(cs_z), s.load_f(cn_x), s.load_f(cn_y), s.load_f(cn_z),
            s.fmul(pr, cl), s.fmul(pg, cl), s.fmul(pbv, cl),
        )
        s.end_if()
        s.store(k_v, s.iadd(k_c, s.uint(1)))
        s.end_loop()
        # ReSTIR GI: temporal + spatial reuse replaces the plain per-pixel average
        s.begin_if(restir_on)
        var rg = emit_sota_restir_gi_reuse(
            s, b_sc, b_hdr, res, px, py, w_u, h_u, wx, wy, wz, nx, ny, nz, gnx, gny, gnz,
            tlas_var, b_verts, b_indices, b_instances, b_materials, bindless_arr,
            snap=sc, snap_jx=s.push_word(6), snap_jy=s.push_word(7),
        )
        s.store(gi_r, rg[0])
        s.store(gi_g, rg[1])
        s.store(gi_b, rg[2])
        s.end_if()
        s.end_if()

    # Progressive accumulation (photo mode)
    s.begin_if(s.ine(s.bit_and(flags, s.uint(SOTA_FLAG_ACCUMULATE)), s.uint(0)))
    var acc_off = s.iadd(_ctrl(s, b_sc, SOTA_CTRL_ACCUM), s.imul(pix, s.uint(UInt32(SOTA_GI_ACCUM_WORDS))))
    var n_acc = s.float_from_uint(_ctrl(s, b_sc, SOTA_CTRL_ACCUM_N))
    var has_hist = s.fgt(n_acc, s.float(0.5))
    var a_r = s.select_f(has_hist, s.load_float(b_sc, acc_off), s.float(0.0))
    var a_g = s.select_f(has_hist, s.load_float(b_sc, s.iadd(acc_off, s.uint(1))), s.float(0.0))
    var a_b = s.select_f(has_hist, s.load_float(b_sc, s.iadd(acc_off, s.uint(2))), s.float(0.0))
    var inv_acc = s.fdiv(s.float(1.0), s.fadd(n_acc, s.float(1.0)))
    var m_r = s.fma(s.fsub(s.load_f(gi_r), a_r), inv_acc, a_r)
    var m_g = s.fma(s.fsub(s.load_f(gi_g), a_g), inv_acc, a_g)
    var m_b = s.fma(s.fsub(s.load_f(gi_b), a_b), inv_acc, a_b)
    s.store_float(b_sc, acc_off, m_r)
    s.store_float(b_sc, s.iadd(acc_off, s.uint(1)), m_g)
    s.store_float(b_sc, s.iadd(acc_off, s.uint(2)), m_b)
    s.store(gi_r, m_r)
    s.store(gi_g, m_g)
    s.store(gi_b, m_b)
    s.end_if()

    var g_r = s.load_f(gi_r)
    var g_g = s.load_f(gi_g)
    var g_b = s.load_f(gi_b)
    s.begin_if(lowres)
    # Position relative to this frame's camera: f16 then keeps ~5e-4 of the camera distance
    var rdx = s.fsub(wx, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X))
    var rdy = s.fsub(wy, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X + 1))
    var rdz = s.fsub(wz, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X + 2))
    s.store_word(b_sc, lo, s.pack_f16x2(g_r, g_g))
    s.store_word(b_sc, s.iadd(lo, s.uint(1)), s.pack_f16x2(g_b, rdx))
    s.store_word(b_sc, s.iadd(lo, s.uint(2)), s.pack_f16x2(rdy, rdz))
    s.store_word(b_sc, s.iadd(lo, s.uint(3)), emit_sota_oct_pack(s, nx, ny, nz))
    s.begin_else()
    _emit_sota_gi_store_hdr(s, b_sc, b_hdr, gb, pix, demod, g_r, g_g, g_b)
    s.end_if()
    return s.finish()


def _emit_sota_gi_store_hdr(
    mut s: AaaSpirvModule, b_sc: UInt32, b_hdr: UInt32, gb: UInt32, pix: UInt32, demod: UInt32, g_r: UInt32, g_g: UInt32, g_b: UInt32
):
    """Writes the GI of one pixel to `BIND_AAA_HDR_OUT`: demodulated for SVGF, else remodulated with
    the crisp (direct + specular) term and the diffuse weight from the G-buffer record at `gb`."""
    var ob = s.imul(pix, s.uint(8))
    # Round 4: the demodulated path (SVGF on, the usual case) no longer loads the 6 remodulation words
    s.begin_if(demod)
    s.store_float(b_hdr, ob, g_r)
    s.store_float(b_hdr, s.iadd(ob, s.uint(1)), g_g)
    s.store_float(b_hdr, s.iadd(ob, s.uint(2)), g_b)
    s.begin_else()
    var crisp_r = s.load_float(b_sc, s.iadd(gb, s.uint(9)))
    var crisp_g = s.load_float(b_sc, s.iadd(gb, s.uint(10)))
    var crisp_b = s.load_float(b_sc, s.iadd(gb, s.uint(11)))
    var dw_r = s.load_float(b_sc, s.iadd(gb, s.uint(12)))
    var dw_g = s.load_float(b_sc, s.iadd(gb, s.uint(13)))
    var dw_b = s.load_float(b_sc, s.iadd(gb, s.uint(14)))
    s.store_float(b_hdr, ob, s.fma(dw_r, g_r, crisp_r))
    s.store_float(b_hdr, s.iadd(ob, s.uint(1)), s.fma(dw_g, g_g, crisp_g))
    s.store_float(b_hdr, s.iadd(ob, s.uint(2)), s.fma(dw_b, g_b, crisp_b))
    s.end_if()


# ---------------------------------------------------------------------------------------------
# Pass 3 (round 3): depth / normal-aware upsample of the reduced-resolution resolve
# ---------------------------------------------------------------------------------------------
def sota_gi_upsample_shader() -> List[UInt32]:
    """Push[0..2] as the resolve; Push[3] = scale (2 or 4), Push[4], Push[5] = this frame's sub-pixel of
    the representative pixel in each block, Push[8] = tent radius in blocks (float, 0 = 1).

    Joint-bilateral upsample (Kopf et al. 2007 "Joint Bilateral Upsampling"; depth / normal weights
    as in the usual geometry-aware upsampling of half-res GI, e.g. the AMD GI-1.0 paper, Boisse et al.
    2023): every full-res pixel gathers the 3x3 low-res records around its block. Each weight is a
    tent around the block's representative pixel (plus a small floor so the whole 3x3 ring takes part)
    x a plane-distance weight relative to the camera distance x `max(0, n . n_s)^8`. Pixels with no
    geometrically similar record (thin features, a whole block disoccluded) fall back to the world
    radiance cache at their own position when it is valid, else to the plain tent average.

    An 8x8 workgroup first stages the `(8 / scale + 2)^2` low-res records it touches in shared memory
    (36 at half res), so each record is read from memory once per workgroup instead of 9x per pixel.
    Round 4: the records are 4-word f16 (`SOTA_GI_LOWRES_WORDS`, half the round-3 traffic) and are
    unpacked once while staging, so the 9 taps per pixel read plain floats (normal included)."""
    var s = AaaSpirvModule(MODEL_GL_COMPUTE, group_width=8, group_height=8, push_words=AAA_PUSH_WORDS)
    var b_hdr = s.buffer(BIND_AAA_HDR_OUT)
    var b_sc = s.buffer(BIND_AAA_SURFCACHE)
    comptime TILE_MAX = 36  # (8 / 2 + 2)^2
    comptime REC = 10  # r, g, b, x, y, z (relative to the camera), nx, ny, nz, valid (1 / 0)
    var lds = s.workgroup_array_f32(TILE_MAX * REC)
    var px = s.invocation_axis(0)
    var py = s.invocation_axis(1)
    var w_u = s.push_word(0)
    var h_u = s.push_word(1)
    var demod = s.ine(s.push_word(2), s.uint(0))
    var sc = s.umax(s.push_word(3), s.uint(2))
    var jx = s.push_word(4)
    var jy = s.push_word(5)
    var mode = _ctrl(s, b_sc, SOTA_CTRL_MODE)
    var flags = _ctrl(s, b_sc, SOTA_CTRL_FLAGS)
    var frame = _ctrl(s, b_sc, SOTA_CTRL_FRAME)
    var stamp_want = s.iadd(frame, s.uint(1))
    var lw = s.udiv(s.iadd(w_u, s.isub(sc, s.uint(1))), sc)
    var lh = s.udiv(s.iadd(h_u, s.isub(sc, s.uint(1))), sc)
    var lo_base = _ctrl(s, b_sc, SOTA_CTRL_LOWRES)

    # Stage the tile: blocks [wgb - 1, wgb + 8 / sc] in x and y, clamped to the low-res grid
    var tile_w = s.iadd(s.udiv(s.uint(8), sc), s.uint(2))
    var tile_n = s.imul(tile_w, tile_w)
    var wgbx = s.udiv(s.imul(s.workgroup_axis(0), s.uint(8)), sc)
    var wgby = s.udiv(s.imul(s.workgroup_axis(1), s.uint(8)), sc)
    var li = s.iadd(s.local_axis(0), s.imul(s.local_axis(1), s.uint(8)))
    s.begin_if(s.ult(li, tile_n))
    var tbx = s.umin(s.isub(s.umax(s.iadd(wgbx, s.umod(li, tile_w)), s.uint(1)), s.uint(1)), s.isub(lw, s.uint(1)))
    var tby = s.umin(s.isub(s.umax(s.iadd(wgby, s.udiv(li, tile_w)), s.uint(1)), s.uint(1)), s.isub(lh, s.uint(1)))
    var lo = s.iadd(lo_base, s.imul(s.iadd(s.imul(tby, lw), tbx), s.uint(UInt32(SOTA_GI_LOWRES_WORDS))))
    var w0 = s.unpack_f16x2(s.load_word(b_sc, lo))
    var w1 = s.unpack_f16x2(s.load_word(b_sc, s.iadd(lo, s.uint(1))))
    var w2 = s.unpack_f16x2(s.load_word(b_sc, s.iadd(lo, s.uint(2))))
    var nw = s.load_word(b_sc, s.iadd(lo, s.uint(3)))
    var sn = emit_sota_oct_unpack(s, nw)
    var valid_t = s.ine(nw, s.uint(0))
    var vals = [w0[0], w0[1], w1[0], w1[1], w2[0], w2[1], sn[0], sn[1], sn[2], s.select_f(valid_t, s.float(1.0), s.float(0.0))]
    var lb = s.imul(li, s.uint(UInt32(REC)))
    for k in range(REC):
        s.store_workgroup_f32(lds[0], lds[1], s.iadd(lb, s.uint(UInt32(k))), vals[k])
    s.end_if()
    s.workgroup_barrier()

    s.return_if(s.logical_or(s.uge(px, w_u), s.uge(py, h_u)))
    var pix = s.iadd(s.imul(py, w_u), px)
    var gb = s.iadd(_ctrl(s, b_sc, SOTA_CTRL_GBUF), s.imul(pix, s.uint(UInt32(SOTA_GI_GBUF_WORDS))))
    s.return_if(s.ine(s.load_word(b_sc, s.iadd(gb, s.uint(15))), stamp_want))
    s.return_if(s.ieq(mode, s.uint(0)))
    var wx = s.load_float(b_sc, gb)
    var wy = s.load_float(b_sc, s.iadd(gb, s.uint(1)))
    var wz = s.load_float(b_sc, s.iadd(gb, s.uint(2)))
    var nx = s.load_float(b_sc, s.iadd(gb, s.uint(3)))
    var ny = s.load_float(b_sc, s.iadd(gb, s.uint(4)))
    var nz = s.load_float(b_sc, s.iadd(gb, s.uint(5)))
    # Camera-relative position (the records store theirs relative to the same camera)
    var cdx = s.fsub(wx, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X))
    var cdy = s.fsub(wy, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X + 1))
    var cdz = s.fsub(wz, _ctrl_f(s, b_sc, SOTA_CTRL_CAM_X + 2))
    var cam_d = s.fsqrt(s.nmax(s.float(1e-8), emit_vec3_dot(s, cdx, cdy, cdz, cdx, cdy, cdz)))
    var inv_plane = s.fdiv(s.float(1.0), s.fmul(cam_d, s.float(0.01)))

    var tent_k = s.push_float(8)
    var tent_r = s.fmul(s.float_from_uint(sc), s.select_f(s.fgt(tent_k, s.float(0.0)), tent_k, s.float(1.0)))
    var inv_sc = s.fdiv(s.float(1.0), tent_r)
    var fpx = s.float_from_uint(px)
    var fpy = s.float_from_uint(py)
    var cbx = s.udiv(px, sc)
    var cby = s.udiv(py, sc)
    # Tile coordinate of block (cbx + ox - 1) is (cbx - wgb + ox)
    var tx0 = s.isub(cbx, wgbx)
    var ty0 = s.isub(cby, wgby)
    var acc_r = s.fvar(s.float(0.0))
    var acc_g = s.fvar(s.float(0.0))
    var acc_b = s.fvar(s.float(0.0))
    var acc_w = s.fvar(s.float(0.0))
    var tnt_r = s.fvar(s.float(0.0))
    var tnt_g = s.fvar(s.float(0.0))
    var tnt_b = s.fvar(s.float(0.0))
    var tnt_w = s.fvar(s.float(0.0))
    for oy in range(3):
        for ox in range(3):
            var tx = s.iadd(tx0, s.uint(UInt32(ox)))
            var ty = s.iadd(ty0, s.uint(UInt32(oy)))
            var tb = s.imul(s.iadd(s.imul(ty, tile_w), tx), s.uint(UInt32(REC)))
            var valid = s.fgt(s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(9))), s.float(0.5))
            s.begin_if(valid)
            var r = s.load_workgroup_f32(lds[0], lds[1], tb)
            var g = s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(1)))
            var b = s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(2)))
            var sx = s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(3)))
            var sy = s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(4)))
            var sz = s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(5)))
            var snx = s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(6)))
            var sny = s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(7)))
            var snz = s.load_workgroup_f32(lds[0], lds[1], s.iadd(tb, s.uint(8)))
            # Tent around the representative pixel of that (clamped) block
            var bxi = s.umin(s.isub(s.umax(s.iadd(cbx, s.uint(UInt32(ox))), s.uint(1)), s.uint(1)), s.isub(lw, s.uint(1)))
            var byi = s.umin(s.isub(s.umax(s.iadd(cby, s.uint(UInt32(oy))), s.uint(1)), s.uint(1)), s.isub(lh, s.uint(1)))
            var rpx = s.float_from_uint(s.iadd(s.imul(bxi, sc), jx))
            var rpy = s.float_from_uint(s.iadd(s.imul(byi, sc), jy))
            var tx_w = s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fmul(s.fabs(s.fsub(fpx, rpx)), inv_sc)))
            var ty_w = s.nmax(s.float(0.0), s.fsub(s.float(1.0), s.fmul(s.fabs(s.fsub(fpy, rpy)), inv_sc)))
            var tent = s.fadd(s.fmul(tx_w, ty_w), s.float(0.02))
            var plane = s.fmul(s.fabs(emit_vec3_dot(s, nx, ny, nz, s.fsub(sx, cdx), s.fsub(sy, cdy), s.fsub(sz, cdz))), inv_plane)
            var w_p = s.fdiv(s.float(1.0), s.fadd(s.float(1.0), s.fmul(plane, plane)))
            var nd = s.nmax(s.float(0.0), emit_vec3_dot(s, nx, ny, nz, snx, sny, snz))
            var nd2 = s.fmul(nd, nd)
            var nd4 = s.fmul(nd2, nd2)
            var w_g = s.fmul(tent, s.fmul(w_p, s.fmul(nd4, nd4)))
            s.store(acc_r, s.fma(w_g, r, s.load_f(acc_r)))
            s.store(acc_g, s.fma(w_g, g, s.load_f(acc_g)))
            s.store(acc_b, s.fma(w_g, b, s.load_f(acc_b)))
            s.store(acc_w, s.fadd(s.load_f(acc_w), w_g))
            s.store(tnt_r, s.fma(tent, r, s.load_f(tnt_r)))
            s.store(tnt_g, s.fma(tent, g, s.load_f(tnt_g)))
            s.store(tnt_b, s.fma(tent, b, s.load_f(tnt_b)))
            s.store(tnt_w, s.fadd(s.load_f(tnt_w), tent))
            s.end_if()
    var aw = s.load_f(acc_w)
    var ok = s.fgt(aw, s.float(1e-3))
    var inv_aw = s.fdiv(s.float(1.0), s.nmax(aw, s.float(1e-8)))
    var inv_tw = s.fdiv(s.float(1.0), s.nmax(s.load_f(tnt_w), s.float(1e-8)))
    var o_r = s.fvar(s.select_f(ok, s.fmul(s.load_f(acc_r), inv_aw), s.fmul(s.load_f(tnt_r), inv_tw)))
    var o_g = s.fvar(s.select_f(ok, s.fmul(s.load_f(acc_g), inv_aw), s.fmul(s.load_f(tnt_g), inv_tw)))
    var o_b = s.fvar(s.select_f(ok, s.fmul(s.load_f(acc_b), inv_aw), s.fmul(s.load_f(tnt_b), inv_tw)))
    var wrc_on = s.ine(s.bit_and(flags, s.uint(SOTA_FLAG_WRC_VALID)), s.uint(0))
    s.begin_if(s.logical_and(s.logical_not(ok), wrc_on))
    var wi = emit_sota_wrc_irradiance(s, b_sc, _ctrl(s, b_sc, SOTA_CTRL_CUR_BANK), wx, wy, wz, nx, ny, nz)
    s.store(o_r, wi[0])
    s.store(o_g, wi[1])
    s.store(o_b, wi[2])
    s.end_if()
    _emit_sota_gi_store_hdr(s, b_sc, b_hdr, gb, pix, demod, s.load_f(o_r), s.load_f(o_g), s.load_f(o_b))
    return s.finish()


# ---------------------------------------------------------------------------------------------
# Hooks emitted into shared shaders (`aaa_gi_svgf.mojo`)
# ---------------------------------------------------------------------------------------------
def emit_sota_gi_write_gbuffer(
    mut s: AaaSpirvModule,
    b_surfcache: UInt32,
    pix: UInt32,
    frame_idx: UInt32,
    wx: UInt32,
    wy: UInt32,
    wz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
    gnx: UInt32,
    gny: UInt32,
    gnz: UInt32,
    crisp_r: UInt32,
    crisp_g: UInt32,
    crisp_b: UInt32,
    dw_r: UInt32,
    dw_g: UInt32,
    dw_b: UInt32,
):
    """Deferred-shade hook: store this pixel's GI inputs when an area-B mode is active, and the
    surface history used by the ReSTIR passes when they run."""
    emit_sota_hist_write(s, b_surfcache, pix, wx, wy, wz, nx, ny, nz)
    var mode = emit_sota_gi_mode(s, b_surfcache)
    s.begin_if(s.ine(mode, s.uint(0)))
    var gb = s.iadd(_ctrl(s, b_surfcache, SOTA_CTRL_GBUF), s.imul(pix, s.uint(UInt32(SOTA_GI_GBUF_WORDS))))
    var vals = [wx, wy, wz, nx, ny, nz, gnx, gny, gnz, crisp_r, crisp_g, crisp_b, dw_r, dw_g, dw_b]
    for i in range(15):
        s.store_float(b_surfcache, s.iadd(gb, s.uint(UInt32(i))), vals[i])
    s.store_word(b_surfcache, s.iadd(gb, s.uint(15)), s.iadd(_ctrl(s, b_surfcache, SOTA_CTRL_FRAME), s.uint(1)))
    s.end_if()
    _ = frame_idx


def emit_sota_surfcache_wrc_bounce(
    mut s: AaaSpirvModule,
    b_surfcache: UInt32,
    cx: UInt32,
    cy: UInt32,
    cz: UInt32,
    nx: UInt32,
    ny: UInt32,
    nz: UInt32,
) -> Tuple[UInt32, UInt32, UInt32, UInt32]:
    """Surface-cache hook: in hybrid mode the triangle's indirect term is `WRC(center, n)` (previous
    bank). Returns `(use_wrc_bool, gi_r, gi_g, gi_b)`; gi is zero when `use_wrc` is false."""
    var flags = _ctrl(s, b_surfcache, SOTA_CTRL_FLAGS)
    var want = s.ine(s.bit_and(flags, s.uint(SOTA_FLAG_WRC_FEEDS_SURFCACHE)), s.uint(0))
    var valid = s.ine(s.bit_and(flags, s.uint(SOTA_FLAG_WRC_VALID)), s.uint(0))
    var g_r = s.fvar(s.float(0.0))
    var g_g = s.fvar(s.float(0.0))
    var g_b = s.fvar(s.float(0.0))
    s.begin_if(s.logical_and(want, valid))
    var wi = emit_sota_wrc_irradiance(s, b_surfcache, _ctrl(s, b_surfcache, SOTA_CTRL_PREV_BANK), cx, cy, cz, nx, ny, nz)
    s.store(g_r, wi[0])
    s.store(g_g, wi[1])
    s.store(g_b, wi[2])
    s.end_if()
    return (want, s.load_f(g_r), s.load_f(g_g), s.load_f(g_b))

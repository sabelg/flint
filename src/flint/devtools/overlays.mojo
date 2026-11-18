"""Game-agnostic Flint DevTools overlays for telemetry, memory, GPU inspector, and CVars."""
from flint.assets.meshlet import MESHLET_MAX_TRIS, MESHLET_MAX_VERTS
from flint.core.console import DevConsole
from flint.core.cvars import (
    ConsoleVars,
    CVAR_DEBUG_DRAW,
    CVAR_DEBUG_RANGE,
    CVAR_FX_SPEED_LINES,
    CVAR_PARTICLES_GPU,
    CVAR_SKIN_GPU,
    CVAR_TERRAIN_GPU_MESH,
    CVAR_TERRAIN_MESH_BUDGET,
    cvar_entry,
    show_value,
)
from flint.core.memory import (
    MEM_GPU_MESHES,
    MEM_GPU_SHADERS,
    MEM_GPU_TARGETS,
    MEM_GPU_TEXTURES,
    MEM_TAGS,
    W_CPU_LIVE,
    W_CPU_PEAK,
    W_VRAM_LIVE,
    mem_live,
    mem_tag_key,
    mem_total,
)
from flint.platform.input import InputState
from flint.platform.os import file_size, process_rss_bytes
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_vsm import VSM_PHYS_TILE_COUNT, VSM_VIRTUAL_RES
from flint.render.simd_canvas import SimdCanvas

comptime KEY_GRAVE: Int = 96
comptime KEY_MINUS: Int = 45
comptime KEY_EQUAL: Int = 61
comptime KEY_ENTER: Int = 257
comptime KEY_RIGHT: Int = 262
comptime KEY_LEFT: Int = 263
comptime KEY_DOWN: Int = 264
comptime KEY_UP: Int = 265
comptime KEY_F2: Int = 291
comptime KEY_F3: Int = 292
comptime KEY_F4: Int = 293
comptime KEY_F5: Int = 294


struct GpuInspectorStats(Copyable, Movable):
    var attached: Bool
    var device_name: String
    var render_w: Int
    var render_h: Int
    var output_w: Int
    var output_h: Int
    var frame_index: Int
    var total_vertices: Int
    var total_triangles: Int
    var total_meshlets: Int
    var culled_meshlet_draws: Int
    var submitted_instances: Int
    var vsm_enabled: Bool
    var vsm_cached_hits: Int
    var vsm_rendered_pages: Int
    var vsm_mapped_pages: Int
    var has_ray_query: Bool
    var has_bda: Bool
    var has_side_queue: Bool
    var blas_count: Int
    var tlas_active: Bool
    var rg_passes: Int
    var rg_aliased_kib: Int
    var rg_unaliased_kib: Int
    var rg_total_gpu_us: Int
    var surfcache_kib: Int
    var streamed_kib: Int
    var staging_kib: Int
    var geom_kib: Int
    var pso_cache_active: Bool
    var pipeline_count: Int
    var pso_cache_file_bytes: Int
    var validation_errors: Int
    var vk_error: String

    def __init__(out self):
        self.attached = False
        self.device_name = String()
        self.render_w = 0
        self.render_h = 0
        self.output_w = 0
        self.output_h = 0
        self.frame_index = 0
        self.total_vertices = 0
        self.total_triangles = 0
        self.total_meshlets = 0
        self.culled_meshlet_draws = 0
        self.submitted_instances = 0
        self.vsm_enabled = False
        self.vsm_cached_hits = 0
        self.vsm_rendered_pages = 0
        self.vsm_mapped_pages = 0
        self.has_ray_query = False
        self.has_bda = False
        self.has_side_queue = False
        self.blas_count = 0
        self.tlas_active = False
        self.rg_passes = 0
        self.rg_aliased_kib = 0
        self.rg_unaliased_kib = 0
        self.rg_total_gpu_us = 0
        self.surfcache_kib = 0
        self.streamed_kib = 0
        self.staging_kib = 0
        self.geom_kib = 0
        self.pso_cache_active = False
        self.pipeline_count = 0
        self.pso_cache_file_bytes = -1
        self.validation_errors = 0
        self.vk_error = String()


struct CVarSliderSpec(Copyable, ImplicitlyCopyable, Movable):
    var cvar_id: Int
    var min_val: Float32
    var max_val: Float32
    var step: Float32

    def __init__(out self, cvar_id: Int, min_val: Float32, max_val: Float32, step: Float32):
        self.cvar_id = cvar_id
        self.min_val = min_val
        self.max_val = max_val
        self.step = step


def _cvar_slider_specs() -> List[CVarSliderSpec]:
    var specs = List[CVarSliderSpec](capacity=7)
    specs.append(CVarSliderSpec(CVAR_DEBUG_DRAW, Float32(0.0), Float32(1.0), Float32(1.0)))
    specs.append(CVarSliderSpec(CVAR_DEBUG_RANGE, Float32(0.0), Float32(200.0), Float32(5.0)))
    specs.append(CVarSliderSpec(CVAR_TERRAIN_GPU_MESH, Float32(0.0), Float32(1.0), Float32(1.0)))
    specs.append(CVarSliderSpec(CVAR_TERRAIN_MESH_BUDGET, Float32(1.0), Float32(64.0), Float32(1.0)))
    specs.append(CVarSliderSpec(CVAR_PARTICLES_GPU, Float32(0.0), Float32(1.0), Float32(1.0)))
    specs.append(CVarSliderSpec(CVAR_SKIN_GPU, Float32(0.0), Float32(1.0), Float32(1.0)))
    specs.append(CVarSliderSpec(CVAR_FX_SPEED_LINES, Float32(0.0), Float32(1.0), Float32(1.0)))
    return specs^


struct FlintDevOverlays(Movable):
    var console: DevConsole
    var cvars: ConsoleVars
    var show_profiler: Bool
    var show_memory: Bool
    var show_gpu_inspector: Bool
    var show_cvars: Bool
    var cvar_cursor: Int
    var tick_us: Int
    var render_us: Int
    var fps: Int
    var active_backend_name: String
    var gpu_stats: GpuInspectorStats

    def __init__(out self):
        self.console = DevConsole()
        self.cvars = ConsoleVars()
        self.show_profiler = False
        self.show_memory = False
        self.show_gpu_inspector = False
        self.show_cvars = False
        self.cvar_cursor = 0
        self.tick_us = 0
        self.render_us = 0
        self.fps = 60
        self.active_backend_name = "SIMD-CPU"
        self.gpu_stats = GpuInspectorStats()
        self.console.say("FLINT ENGINE DEV CONSOLE READY - TYPE 'LIST' OR 'LOG'")

    def sync_aaa_renderer(mut self, ren: FlintAaaVulkanRenderer):
        self.gpu_stats.attached = True
        self.gpu_stats.device_name = ren.vk.device_name
        self.gpu_stats.render_w = ren.render_w
        self.gpu_stats.render_h = ren.render_h
        self.gpu_stats.output_w = ren.output_w
        self.gpu_stats.output_h = ren.output_h
        self.gpu_stats.frame_index = ren.frame_index
        self.gpu_stats.total_vertices = ren.geom.total_vertices()
        self.gpu_stats.total_triangles = ren.geom.total_triangles()
        self.gpu_stats.total_meshlets = ren.geom.total_meshlets()
        self.gpu_stats.culled_meshlet_draws = ren.last_culled_meshlet_draws
        self.gpu_stats.submitted_instances = len(ren.frame_instances)
        self.gpu_stats.vsm_enabled = ren.enable_vsm
        self.gpu_stats.vsm_cached_hits = ren.vsm.cached_hits_last_frame
        self.gpu_stats.vsm_rendered_pages = ren.vsm.newly_rendered_pages_last_frame
        var mapped = 0
        for i in range(len(ren.vsm.phys_to_virtual)):
            if ren.vsm.phys_to_virtual[i] >= 0:
                mapped += 1
        self.gpu_stats.vsm_mapped_pages = mapped
        self.gpu_stats.has_ray_query = ren.vk.has_ray_query
        self.gpu_stats.has_bda = ren.vk.has_bda
        self.gpu_stats.has_side_queue = ren.vk.has_side_queue()
        self.gpu_stats.blas_count = len(ren.mesh_blas)
        self.gpu_stats.tlas_active = ren.frame_tlas.handle != 0
        self.gpu_stats.rg_passes = len(ren.render_graph.passes)
        self.gpu_stats.rg_aliased_kib = ren.render_graph.aliased_peak_bytes // 1024
        self.gpu_stats.rg_unaliased_kib = ren.render_graph.unaliased_transient_bytes // 1024
        self.gpu_stats.rg_total_gpu_us = Int(ren.render_graph.total_gpu_ms * 1000.0)
        self.gpu_stats.surfcache_kib = ren.buf_surfcache.size // 1024
        self.gpu_stats.streamed_kib = ren.streamer.streamed_bytes_total // 1024
        self.gpu_stats.staging_kib = ren.buf_staging.size // 1024
        self.gpu_stats.geom_kib = (ren.buf_vertices.size + ren.buf_indices.size + ren.buf_meshlets.size) // 1024
        self.gpu_stats.pso_cache_active = ren.vk.pipeline_cache != 0
        self.gpu_stats.pipeline_count = len(ren.vk.pipelines)
        self.gpu_stats.pso_cache_file_bytes = file_size("build/vk_pso.cache")
        self.gpu_stats.validation_errors = ren.vk.validation_errors() if ren.vk.messages != 0 else 0
        self.gpu_stats.vk_error = ren.vk.error
        if ren.active:
            self.active_backend_name = "VULKAN-AAA-RT" if ren.vk.has_ray_query else "VULKAN-AAA"

    def move_cvar_cursor(mut self, delta: Int):
        var n = len(_cvar_slider_specs())
        if n <= 0:
            self.cvar_cursor = 0
            return
        self.cvar_cursor = (self.cvar_cursor + delta) % n
        if self.cvar_cursor < 0:
            self.cvar_cursor += n

    def adjust_cvar_slider(mut self, direction: Int):
        var specs = _cvar_slider_specs()
        if self.cvar_cursor < 0 or self.cvar_cursor >= len(specs):
            return
        var spec = specs[self.cvar_cursor]
        var entry = cvar_entry(spec.cvar_id)
        var cur = self.cvars.get(spec.cvar_id)
        if entry.flag:
            if direction == 0:
                self.cvars.set(spec.cvar_id, Float32(0.0) if cur != Float32(0.0) else Float32(1.0))
            elif direction > 0:
                self.cvars.set(spec.cvar_id, Float32(1.0))
            else:
                self.cvars.set(spec.cvar_id, Float32(0.0))
        else:
            if direction == 0:
                self.cvars.set(spec.cvar_id, entry.default)
            else:
                var next_v = cur + Float32(direction) * spec.step
                next_v = max(spec.min_val, min(spec.max_val, next_v))
                self.cvars.set(spec.cvar_id, next_v)

    def handle_input(mut self, mut input: InputState) -> Bool:
        """Processes `~`, `F2`-`F5`, CVar sliders, and console typing. Returns True while console consumes input."""
        if input.key_pressed(KEY_GRAVE):
            self.console.open = not self.console.open
            return self.console.open

        if self.console.open:
            for key_code in range(32, 310):
                if key_code != KEY_GRAVE and input.key_pressed(key_code):
                    self.console.key(key_code, self.cvars)
            return True

        if input.key_pressed(KEY_F2):
            self.show_profiler = not self.show_profiler
        if input.key_pressed(KEY_F3):
            self.show_memory = not self.show_memory
        if input.key_pressed(KEY_F4):
            self.show_gpu_inspector = not self.show_gpu_inspector
        if input.key_pressed(KEY_F5):
            self.show_cvars = not self.show_cvars
        if self.show_cvars:
            if input.key_pressed(KEY_UP):
                self.move_cvar_cursor(-1)
            if input.key_pressed(KEY_DOWN):
                self.move_cvar_cursor(1)
            if input.key_pressed(KEY_LEFT) or input.key_pressed(KEY_MINUS):
                self.adjust_cvar_slider(-1)
            if input.key_pressed(KEY_RIGHT) or input.key_pressed(KEY_EQUAL):
                self.adjust_cvar_slider(1)
            if input.key_pressed(KEY_ENTER):
                self.adjust_cvar_slider(0)
        return False

    def render(self, mut canvas: SimdCanvas):
        var top_off = 232 if self.console.open else 46
        if self.show_profiler:
            var px = 14
            var py = top_off
            canvas.draw_panel(px, py, 372, 118, 0x0B121C, 0x4AD295)
            canvas.draw_rect_dim(px + 2, py + 2, 368, 114, 0x0B121C)
            canvas.draw_text(px + 10, py + 10, "FLINT PROFILER [F2]", 0x4AD295, 2)
            canvas.draw_text(
                px + 10, py + 34,
                "BACKEND: " + self.active_backend_name + "  |  FPS: " + String(self.fps),
                0xF2F5F8, 1,
            )
            canvas.draw_text(
                px + 10, py + 52,
                "SIM TICK: " + String(self.tick_us) + " US  |  RENDER: " + String(self.render_us) + " US",
                0xFFD479, 1,
            )
            canvas.draw_text(
                px + 10, py + 70,
                "TRIANGLES: " + String(canvas.triangles_drawn) + "  |  SPRITES: " + String(canvas.sprites_drawn),
                0xB8C7D8, 1,
            )
            var bar_w = min(340, max(4, (self.render_us * 340) // 16667))
            canvas.draw_rect(px + 10, py + 92, 340, 10, 0x1E2A38)
            canvas.draw_rect(px + 10, py + 92, bar_w, 10, UInt32(0x4AD295) if bar_w < 260 else UInt32(0xF49E38))

        if self.show_memory:
            var mx = canvas.width - 364
            var my = top_off
            var live_kib = mem_total(W_CPU_LIVE) // 1024
            var peak_kib = mem_total(W_CPU_PEAK) // 1024
            var vram_kib = mem_total(W_VRAM_LIVE) // 1024
            var rss_mib = process_rss_bytes() // (1024 * 1024)
            canvas.draw_panel(mx, my, 350, 210, 0x0C1420, 0x58B4FF)
            canvas.draw_rect_dim(mx + 2, my + 2, 346, 206, 0x0C1420)
            canvas.draw_text(mx + 10, my + 10, "FLINT MEMORY BUDGETS [F3]", 0x58B4FF, 2)
            canvas.draw_text(
                mx + 10, my + 34,
                "CPU LIVE: " + String(live_kib) + " KIB  |  PEAK: " + String(peak_kib) + " KIB",
                0xF2F5F8, 1,
            )
            canvas.draw_text(
                mx + 10, my + 50,
                "VRAM LIVE: " + String(vram_kib) + " KIB  |  OS RSS: " + String(rss_mib) + " MIB",
                0xFFD479, 1,
            )
            var row_y = my + 72
            var shown = 0
            for tag in range(MEM_TAGS):
                var tag_bytes = mem_live(tag)
                if tag_bytes > 0 and shown < 8:
                    canvas.draw_text(
                        mx + 14, row_y,
                        mem_tag_key(tag).upper() + ": " + String(tag_bytes // 1024) + " KIB",
                        0xC8D6E5, 1,
                    )
                    row_y += 16
                    shown += 1

        if self.show_gpu_inspector:
            var gx = 14
            var gy = top_off + (124 if self.show_profiler else 0)
            canvas.draw_panel(gx, gy, 436, 186, 0x08101C, 0x82AAFF)
            canvas.draw_rect_dim(gx + 2, gy + 2, 432, 182, 0x08101C)
            canvas.draw_text(gx + 10, gy + 10, "GPU BUFFER INSPECTOR [F4]", 0x82AAFF, 2)
            if self.gpu_stats.attached:
                var bda_tag = String("BDA") if self.gpu_stats.has_bda else String("SW")
                var vsm_tag = String("ON") if self.gpu_stats.vsm_enabled else String("OFF")
                var rt_tag = String("VK_RAY_QUERY") if self.gpu_stats.has_ray_query else String("OFF")
                var tlas_tag = String("LIVE") if self.gpu_stats.tlas_active else String("NONE")
                var q_tag = String("SIDE_Q") if self.gpu_stats.has_side_queue else String("MAIN_Q")
                var pso_tag = String("ACTIVE") if self.gpu_stats.pso_cache_active else String("NONE")
                var disk_tag = String(self.gpu_stats.pso_cache_file_bytes // 1024) + "K" if self.gpu_stats.pso_cache_file_bytes >= 0 else String("NONE")
                var status_ok = self.gpu_stats.vk_error.byte_length() == 0 and self.gpu_stats.validation_errors == 0
                var status_str = String("OK") if status_ok else String("ERR: " + self.gpu_stats.vk_error)
                var status_col = UInt32(0x4AD295) if status_ok else UInt32(0xFF6B6B)
                var vram_kib = mem_total(W_VRAM_LIVE) // 1024
                canvas.draw_text(
                    gx + 10, gy + 34,
                    "VISBUFFER: " + String(self.gpu_stats.render_w) + "X" + String(self.gpu_stats.render_h) + " R32G32 -> " + String(self.gpu_stats.output_w) + "X" + String(self.gpu_stats.output_h) + " [" + bda_tag + "]",
                    0xC8D6E5, 1,
                )
                canvas.draw_text(
                    gx + 10, gy + 50,
                    "MESHLETS: " + String(self.gpu_stats.culled_meshlet_draws) + "/" + String(self.gpu_stats.total_meshlets) + " (" + String(MESHLET_MAX_TRIS) + "T/" + String(MESHLET_MAX_VERTS) + "V) INST:" + String(self.gpu_stats.submitted_instances),
                    0xC8D6E5, 1,
                )
                canvas.draw_text(
                    gx + 10, gy + 66,
                    "VSM " + String(VSM_VIRTUAL_RES) + ": " + vsm_tag + " HIT:" + String(self.gpu_stats.vsm_cached_hits) + " NEW:" + String(self.gpu_stats.vsm_rendered_pages) + " MAP:" + String(self.gpu_stats.vsm_mapped_pages) + "/" + String(VSM_PHYS_TILE_COUNT),
                    0xFFD479, 1,
                )
                canvas.draw_text(
                    gx + 10, gy + 82,
                    "RT/AS: " + rt_tag + " | BLAS:" + String(self.gpu_stats.blas_count) + " | TLAS:" + tlas_tag,
                    UInt32(0x4AD295) if self.gpu_stats.has_ray_query else UInt32(0xB8C7D8), 1,
                )
                canvas.draw_text(
                    gx + 10, gy + 98,
                    "RG: " + String(self.gpu_stats.rg_aliased_kib) + "K/" + String(self.gpu_stats.rg_unaliased_kib) + "K (" + String(self.gpu_stats.rg_passes) + "P " + String(self.gpu_stats.rg_total_gpu_us) + "US) SC:" + String(self.gpu_stats.surfcache_kib) + "K",
                    0xC8D6E5, 1,
                )
                canvas.draw_text(
                    gx + 10, gy + 114,
                    "STAGING: " + String(self.gpu_stats.staging_kib) + "K (" + q_tag + ") | GEOM:" + String(self.gpu_stats.geom_kib) + "K STRM:" + String(self.gpu_stats.streamed_kib) + "K",
                    0x58B4FF, 1,
                )
                canvas.draw_text(
                    gx + 10, gy + 130,
                    "PSO CACHE: " + pso_tag + " (" + String(self.gpu_stats.pipeline_count) + " PSOS) | DISK:" + disk_tag,
                    UInt32(0x4AD295) if self.gpu_stats.pso_cache_active else UInt32(0xFFD479), 1,
                )
                canvas.draw_text(
                    gx + 10, gy + 148,
                    "STATUS: " + status_str + " | F:" + String(self.gpu_stats.frame_index) + " VAL:" + String(self.gpu_stats.validation_errors) + " VRAM:" + String(vram_kib) + "K",
                    status_col, 1,
                )
            else:
                var pso_disk_bytes = file_size("build/vk_pso.cache")
                var pso_disk_str = String(pso_disk_bytes // 1024) + " KIB" if pso_disk_bytes >= 0 else String("NOT FOUND (build/vk_pso.cache)")
                var vram_kib = mem_total(W_VRAM_LIVE) // 1024
                var tgt_kib = mem_live(MEM_GPU_TARGETS) // 1024
                var tex_kib = mem_live(MEM_GPU_TEXTURES) // 1024
                var shd_kib = mem_live(MEM_GPU_SHADERS) // 1024
                var msh_kib = mem_live(MEM_GPU_MESHES) // 1024
                canvas.draw_text(gx + 10, gy + 34, "NO AAA RENDERER ATTACHED (BACKEND: " + self.active_backend_name + ")", 0xFFD479, 1)
                canvas.draw_text(gx + 10, gy + 50, "CANVAS: " + String(canvas.width) + "X" + String(canvas.height) + " | TRIS: " + String(canvas.triangles_drawn) + " | SPRITES: " + String(canvas.sprites_drawn), 0xC8D6E5, 1)
                canvas.draw_text(gx + 10, gy + 66, "MESHLET SPEC: " + String(MESHLET_MAX_TRIS) + " TRIS / " + String(MESHLET_MAX_VERTS) + " VERTS MAX", 0xC8D6E5, 1)
                canvas.draw_text(gx + 10, gy + 82, "VSM CONFIG: " + String(VSM_VIRTUAL_RES) + " VIRT / " + String(VSM_PHYS_TILE_COUNT) + " PHYS PAGES (IDLE)", 0xFFD479, 1)
                canvas.draw_text(gx + 10, gy + 98, "VRAM LIVE: " + String(vram_kib) + " KIB | TARGETS: " + String(tgt_kib) + " KIB", 0x58B4FF, 1)
                canvas.draw_text(gx + 10, gy + 114, "GPU MEM: TEX:" + String(tex_kib) + "K SHD:" + String(shd_kib) + "K MSH:" + String(msh_kib) + "K", 0x58B4FF, 1)
                canvas.draw_text(gx + 10, gy + 130, "PSO DISK CACHE: " + pso_disk_str, UInt32(0x4AD295) if pso_disk_bytes >= 0 else UInt32(0xB8C7D8), 1)
                canvas.draw_text(gx + 10, gy + 148, "STATUS: " + self.active_backend_name + " | RENDER: " + String(self.render_us) + " US | FPS: " + String(self.fps), 0x4AD295, 1)

        if self.show_cvars:
            var cx = canvas.width - 406
            var cy = top_off + (216 if self.show_memory else 0)
            canvas.draw_panel(cx, cy, 392, 216, 0x140E1E, 0xC792EA)
            canvas.draw_rect_dim(cx + 2, cy + 2, 388, 212, 0x140E1E)
            canvas.draw_text(cx + 10, cy + 10, "INTERACTIVE CVAR SLIDERS [F5]", 0xC792EA, 2)
            canvas.draw_text(cx + 10, cy + 28, "[UP/DN] SEL [LT/RT] ADJ [RET] TOGGLE", 0xB8C7D8, 1)
            var specs = _cvar_slider_specs()
            var item_y = cy + 44
            for idx in range(len(specs)):
                var spec = specs[idx]
                var entry = cvar_entry(spec.cvar_id)
                var val = self.cvars.get(spec.cvar_id)
                var is_sel = (idx == self.cvar_cursor)
                var prefix = String("> ") if is_sel else String("  ")
                var txt_col = UInt32(0xFFE082) if is_sel else UInt32(0xF2F5F8)
                canvas.draw_text(cx + 10, item_y, prefix + entry.name + " = " + show_value(val), txt_col, 1)
                canvas.draw_rect(cx + 10, item_y + 13, 220, 6, 0x2A1F3A)
                var span = max(Float32(1e-5), spec.max_val - spec.min_val)
                var frac = max(Float32(0.0), min(Float32(1.0), (val - spec.min_val) / span))
                var filled_w = Int(frac * Float32(220.0))
                if filled_w > 0:
                    canvas.draw_rect(cx + 10, item_y + 13, filled_w, 6, UInt32(0xFFE082) if is_sel else UInt32(0xC792EA))
                item_y += 23

        if self.console.open:
            var ch = 224
            canvas.draw_rect(0, 0, canvas.width, ch, 0x0A1018)
            canvas.draw_rect(0, 0, canvas.width, 2, 0xF49E38)
            canvas.draw_rect(0, ch - 2, canvas.width, 2, 0xF49E38)
            canvas.draw_text(14, 10, "FLINT DEV CONSOLE [~ TO CLOSE]", 0xF49E38, 2)
            var ly = 36
            for i in range(len(self.console.log)):
                canvas.draw_text(16, ly, self.console.log[i], 0xCFD9E4, 1)
                ly += 13
            canvas.draw_rect(12, ch - 30, canvas.width - 24, 22, 0x162232)
            canvas.draw_text(18, ch - 23, "> " + self.console.line + "_", 0xFFE082, 2)

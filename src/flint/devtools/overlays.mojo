"""Game-agnostic Flint DevTools overlays:
  - `~` (`KEY_GRAVE` = 96): Drop-down `DevConsole` (`flint.core.console`)
  - `F2` (`291`): Frame timing & zone telemetry HUD
  - `F3` (`292`): `MemCharge` live CPU & GPU VRAM subsystem budget inspector
"""
from flint.core.console import DevConsole
from flint.core.cvars import ConsoleVars
from flint.core.memory import (
    MEM_TAGS, W_CPU_LIVE, W_CPU_PEAK, W_VRAM_LIVE, mem_live, mem_tag_key, mem_total,
)
from flint.platform.input import InputState
from flint.platform.os import process_rss_bytes
from flint.render.simd_canvas import SimdCanvas

comptime KEY_GRAVE: Int = 96
comptime KEY_F2: Int = 291
comptime KEY_F3: Int = 292


struct FlintDevOverlays(Movable):
    var console: DevConsole
    var cvars: ConsoleVars
    var show_profiler: Bool
    var show_memory: Bool
    var tick_us: Int
    var render_us: Int
    var fps: Int
    var active_backend_name: String

    def __init__(out self):
        self.console = DevConsole()
        self.cvars = ConsoleVars()
        self.show_profiler = False
        self.show_memory = False
        self.tick_us = 0
        self.render_us = 0
        self.fps = 60
        self.active_backend_name = "SIMD-CPU"
        self.console.say("FLINT ENGINE DEV CONSOLE READY - TYPE 'LIST' OR 'LOG'")

    def handle_input(mut self, mut input: InputState) -> Bool:
        """Processes `~`, `F2`, `F3`, and console typing. Returns True while console consumes input."""
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
        return False

    def render(self, mut canvas: SimdCanvas):
        if self.show_profiler:
            var px = 14
            var py = 46
            canvas.draw_panel(px, py, 372, 118, 0x101824, 0x4AD295)
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
            var my = 46
            var live_kib = mem_total(W_CPU_LIVE) // 1024
            var peak_kib = mem_total(W_CPU_PEAK) // 1024
            var vram_kib = mem_total(W_VRAM_LIVE) // 1024
            var rss_mib = process_rss_bytes() // (1024 * 1024)
            canvas.draw_panel(mx, my, 350, 210, 0x121A26, 0x58B4FF)
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

        if self.console.open:
            var ch = 224
            canvas.draw_panel(0, 0, canvas.width, ch, 0x0A1018, 0xF49E38)
            canvas.draw_text(14, 10, "FLINT DEV CONSOLE [~ TO CLOSE]", 0xF49E38, 2)
            var ly = 36
            for i in range(len(self.console.log)):
                canvas.draw_text(16, ly, self.console.log[i], 0xCFD9E4, 1)
                ly += 13
            canvas.draw_rect(12, ch - 30, canvas.width - 24, 22, 0x162232)
            canvas.draw_text(18, ch - 23, "> " + self.console.line + "_", 0xFFE082, 2)

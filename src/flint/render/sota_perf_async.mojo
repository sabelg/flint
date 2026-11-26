"""Area F render SOTA round 2: frame pacing and CPU/GPU overlap (`flint.render.sota_perf_async`).

The v5 offscreen frame is synchronous: `render_frame()` records, submits and then waits on its own fence
(the readback of the LDR image, the meshlet counter and the GPU timestamps needs it). CPU time per frame is
therefore record + GPU time, and nothing the application does between frames overlaps the GPU.

`r.sota_perf.pipelined` (default off; env `FLINT_SOTA_PIPELINED=1`) turns on the renderer's existing
Phase-A deferred readback: frame N is submitted without a wait and retired at the start of frame N+1, so
the application's simulation + draw-list work for frame N+1 runs while the GPU renders frame N. Readback,
auto exposure and the GPU timings then lag by one frame (`flush_deferred_frame()` before a screenshot).
The retire still happens before the first host write of the next frame because several HOST_VISIBLE
buffers (surface-cache headers, OIT nodes, staging) are single-instance - ringing those is what would let
the record of N+1 overlap too.

`SotaFramePacing` collects wall-clock frame intervals and reports the median, p95, p99 and the p99/median
ratio (a 1.0 ratio is perfectly even pacing; > 1.5 shows hitches).

Async compute: `VulkanGpu` already exposes a side queue (compute-only family on NVIDIA). The renderer
records the whole frame into one primary command buffer and every buffer is `VK_SHARING_MODE_EXCLUSIVE`,
so moving a pass to the side queue needs a timeline-semaphore wait in the main submit plus queue-family
ownership transfers (or concurrent sharing). See render-sota-f-r2.md for the candidate passes and the
measured overlap of the side queue on GB10.

Sources: Vulkan 1.3 spec (timeline semaphores, queue family ownership transfer, 7.7.4); Wihlidal
"Optimizing the Graphics Pipeline with Compute" (GDC 2016); "Frame pacing" (Android Game SDK docs).
"""
from std.collections import List
from std.math import max, min
from flint.platform.os import env_now


def sota_pipelined_default() -> Bool:
    """cvar r.sota_perf.pipelined: off unless `FLINT_SOTA_PIPELINED=1`."""
    return env_now("FLINT_SOTA_PIPELINED") == "1"


def _sorted_copy(v: List[Float64]) -> List[Float64]:
    var s = v.copy()
    sort(s)
    return s^


def _pick(s: List[Float64], p: Float64) -> Float64:
    if len(s) == 0:
        return 0.0
    return s[min(len(s) - 1, Int(Float64(len(s) - 1) * p + 0.5))]


struct SotaFramePacing(Copyable, Movable):
    """Wall-clock frame intervals (ms) between consecutive `mark()` calls."""

    var intervals: List[Float64]
    var last_s: Float64
    var started: Bool

    def __init__(out self):
        self.intervals = List[Float64]()
        self.last_s = 0.0
        self.started = False

    def reset(mut self):
        self.intervals.clear()
        self.started = False

    def mark(mut self, now_s: Float64):
        if self.started:
            self.intervals.append((now_s - self.last_s) * 1000.0)
        self.last_s = now_s
        self.started = True

    def add_interval(mut self, ms: Float64):
        self.intervals.append(ms)

    def count(self) -> Int:
        return len(self.intervals)

    def median_ms(self) -> Float64:
        return _pick(_sorted_copy(self.intervals), 0.5)

    def p95_ms(self) -> Float64:
        return _pick(_sorted_copy(self.intervals), 0.95)

    def p99_ms(self) -> Float64:
        return _pick(_sorted_copy(self.intervals), 0.99)

    def hitch_ratio(self) -> Float64:
        """p99 / median; 1.0 = even pacing."""
        var m = self.median_ms()
        if m <= 0.0:
            return 0.0
        return self.p99_ms() / m

    def summary(self) -> String:
        return (
            "frames " + String(self.count()) + " median " + String(Int(self.median_ms() * 100.0 + 0.5) / 100.0)
            + " ms p95 " + String(Int(self.p95_ms() * 100.0 + 0.5) / 100.0)
            + " ms p99 " + String(Int(self.p99_ms() * 100.0 + 0.5) / 100.0)
            + " ms hitch " + String(Int(self.hitch_ratio() * 100.0 + 0.5) / 100.0)
        )


def sota_overlap_fraction(t_a_ms: Float64, t_b_ms: Float64, t_both_ms: Float64) -> Float64:
    """How much of the shorter of two workloads was hidden when both ran together:
    0 = fully serial (t_both = a + b), 1 = fully hidden (t_both = max(a, b))."""
    var lo = min(t_a_ms, t_b_ms)
    if lo <= 0.0:
        return 0.0
    var hidden = t_a_ms + t_b_ms - t_both_ms
    return max(0.0, min(1.0, hidden / lo))


# --- alpha-mask / outline route capacity (round 2) ---
# The v5 rings held a fixed 16384 draws; the Forest's foliage needs ~57k visible alpha-masked meshlets at
# 1080p, so ~70% of its leaves were silently dropped on both the CPU packer and Area E's GPU route.

comptime SOTA_ROUTE_DRAWS_LIMIT: Int = 4 * 1024 * 1024  # sanity bound on a counter read (garbage before frame 0)


def sota_route_draws_limit() -> Int:
    return SOTA_ROUTE_DRAWS_LIMIT


def sota_route_capacity_for(need: Int) -> Int:
    """New ring capacity for `need` visible draws: +25 % headroom, rounded up to 4096."""
    var c = need + need // 4
    return ((c + 4095) // 4096) * 4096

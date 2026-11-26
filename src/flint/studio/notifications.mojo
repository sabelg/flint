"""Non-Blocking Toast Notification Stack & Background Job Progress Tracker (`flint.studio.notifications`).

Pure-Mojo 1.2 foundation for Flint Studio feedback & long-running task visibility:
  - `ToastManager`: Stacked non-blocking toast lifecycle with 5 severities
    (`INFO`, `SUCCESS`, `WARN`, `ERROR`, `AGENT`), 5-second deduplication with repeat
    counters, own-job prioritization, typing-pause holding, and bounded history.
  - `JobProgressTracker`: Nielsen 0.1s / 1s / 10s response-time ladder (`QUIET` < 1s,
    compact `PILL` 1s..10s, `EXPANDED` card with cancel + clean log tail >= 10s) for
    NavMesh bakes, game-plugin `.so` builds, packaging, asset imports, and Agent runs.
"""
from std.collections import List
from std.math import isfinite
from flint.render.font import SdfFont
from flint.text.utf8 import decode_utf8, encode_utf8
from flint.ui.animator import SpringConfig, UI_ANIM_INSTANT, UiAnimator
from flint.ui.draw_list import UI_FLAG_BORDER, UI_FLAG_TOP_BEVEL, UiDrawList
from flint.ui.flexbox_dock import RectF
from flint.ui.icons import draw_studio_icon
from flint.ui.theme import (
    ICON_CHECK,
    ICON_CLOSE,
    ICON_SPARK,
    ICON_WARN,
    StudioTheme,
    lerp_rgba8,
    rgba8_scale_alpha,
    rgba8_with_alpha,
)

comptime TOAST_INFO: Int = 0
comptime TOAST_SUCCESS: Int = 1
comptime TOAST_WARN: Int = 2
comptime TOAST_ERROR: Int = 3
comptime TOAST_AGENT: Int = 4

comptime TOAST_DEFAULT_TTL_SEC: Float32 = 5.0
comptime TOAST_ERROR_TTL_SEC: Float32 = 10.0
comptime TOAST_DEDUP_WINDOW_SEC: Float32 = 5.0
comptime TOAST_TYPING_PAUSE_SEC: Float32 = 2.5
comptime TOAST_MAX_VISIBLE: Int = 3
comptime TOAST_HISTORY_MAX: Int = 100

comptime PROGRESS_QUIET_MS: Int = 1000
comptime PROGRESS_EXPANDED_MS: Int = 10000

comptime PROGRESS_STAGE_QUIET: Int = 0
comptime PROGRESS_STAGE_PILL: Int = 1
comptime PROGRESS_STAGE_EXPANDED: Int = 2

comptime JOB_KIND_GENERIC: Int = 0
comptime JOB_KIND_NAVMESH: Int = 1
comptime JOB_KIND_BUILD_SO: Int = 2
comptime JOB_KIND_PACKAGE: Int = 3
comptime JOB_KIND_ASSET_IMPORT: Int = 4
comptime JOB_KIND_AGENT: Int = 5

comptime JOB_RUNNING: Int = 0
comptime JOB_SUCCEEDED: Int = 1
comptime JOB_FAILED: Int = 2
comptime JOB_CANCELLED: Int = 3


def toast_severity_badge(severity: Int) -> String:
    if severity == TOAST_SUCCESS:
        return "OK"
    if severity == TOAST_WARN:
        return "!"
    if severity == TOAST_ERROR:
        return "X"
    if severity == TOAST_AGENT:
        return "AI"
    return "i"


@fieldwise_init
struct ToastItem(Copyable, Movable):
    """Single notification record managed by `ToastManager`."""
    var id: UInt64
    var severity: Int
    var title: String
    var message: String
    var action_label: String
    var action_entity_id: Int
    var created_sec: Float32
    var shown_at_sec: Float32
    var ttl_sec: Float32
    var remaining_sec: Float32
    var repeat_count: Int
    var visible: Bool
    var own_job: Bool
    var held: Bool

    def __init__(
        out self,
        id: UInt64 = 0,
        severity: Int = TOAST_INFO,
        title: String = "",
        message: String = "",
        created_sec: Float32 = 0.0,
        ttl_sec: Float32 = TOAST_DEFAULT_TTL_SEC,
        own_job: Bool = False,
        held: Bool = False,
        action_label: String = "",
        action_entity_id: Int = -1,
    ):
        self.id = id
        self.severity = severity
        self.title = title
        self.message = message
        self.action_label = action_label
        self.action_entity_id = action_entity_id
        self.created_sec = created_sec
        self.shown_at_sec = created_sec
        self.ttl_sec = max(Float32(0.1), ttl_sec)
        self.remaining_sec = self.ttl_sec
        self.repeat_count = 1
        self.visible = True
        self.own_job = own_job
        self.held = held

    def display_title(self) -> String:
        if self.repeat_count > 1:
            return self.title + " (x" + String(self.repeat_count) + ")"
        return self.title


struct ToastManager(Copyable, Movable):
    """Non-blocking toast stack + notification history center."""
    var items: List[ToastItem]
    var unread_count: Int
    var clock_sec: Float32
    var last_typing_sec: Float32
    var typing_active: Bool
    var next_id: UInt64
    var max_visible: Int
    var history_limit: Int

    def __init__(
        out self,
        max_visible: Int = TOAST_MAX_VISIBLE,
        history_limit: Int = TOAST_HISTORY_MAX,
    ):
        self.items = List[ToastItem]()
        self.unread_count = 0
        self.clock_sec = Float32(0.0)
        self.last_typing_sec = Float32(-100.0)
        self.typing_active = False
        self.next_id = UInt64(1)
        self.max_visible = max(1, max_visible)
        self.history_limit = max(1, history_limit)

    def __init__(out self, *, copy: Self):
        self.items = copy.items.copy()
        self.unread_count = copy.unread_count
        self.clock_sec = copy.clock_sec
        self.last_typing_sec = copy.last_typing_sec
        self.typing_active = copy.typing_active
        self.next_id = copy.next_id
        self.max_visible = copy.max_visible
        self.history_limit = copy.history_limit

    def note_user_typing(mut self):
        """Records active text/scrubber input so non-urgent background toasts wait for a pause."""
        self.typing_active = True
        self.last_typing_sec = self.clock_sec

    def push(
        mut self,
        severity: Int,
        title: String,
        message: String = "",
        own_job: Bool = False,
        ttl_sec: Float32 = -1.0,
        action_label: String = "",
        action_entity_id: Int = -1,
    ) -> UInt64:
        """Queues a toast, deduplicating identical recent notices and respecting typing-pause hold."""
        var sev = max(TOAST_INFO, min(TOAST_AGENT, severity))
        var resolved_ttl = (
            ttl_sec
            if (isfinite(ttl_sec) and ttl_sec > Float32(0.0))
            else (TOAST_ERROR_TTL_SEC if sev == TOAST_ERROR else TOAST_DEFAULT_TTL_SEC)
        )
        var should_hold = self.typing_active and (not own_job) and (sev != TOAST_ERROR)

        # Deduplicate against identical recent toast within the dedup window
        var i = len(self.items) - 1
        while i >= 0:
            ref existing = self.items[i]
            if (self.clock_sec - existing.created_sec) > TOAST_DEDUP_WINDOW_SEC:
                break
            if (
                existing.severity == sev
                and existing.title == title
                and existing.message == message
            ):
                self.items[i].created_sec = self.clock_sec
                self.items[i].shown_at_sec = self.clock_sec
                self.items[i].ttl_sec = resolved_ttl
                self.items[i].remaining_sec = resolved_ttl
                self.items[i].repeat_count += 1
                self.items[i].visible = True
                self.items[i].own_job = self.items[i].own_job or own_job
                self.items[i].held = should_hold and (not self.items[i].own_job)
                return self.items[i].id
            i -= 1

        var tid = self.next_id
        self.next_id += UInt64(1)
        var item = ToastItem(
            tid,
            sev,
            title,
            message,
            self.clock_sec,
            resolved_ttl,
            own_job,
            should_hold,
            action_label,
            action_entity_id,
        )
        self.items.append(item^)
        self.unread_count += 1
        while len(self.items) > self.history_limit:
            _ = self.items.pop(0)
        return tid

    def tick(mut self, dt: Float32):
        """Advances deterministic toast clock, releases held toasts on typing pause, expires TTLs."""
        if not isfinite(dt) or dt <= Float32(0.0):
            return
        self.clock_sec += dt

        if self.typing_active and (self.clock_sec - self.last_typing_sec) >= TOAST_TYPING_PAUSE_SEC:
            self.typing_active = False
            for i in range(len(self.items)):
                if self.items[i].held:
                    self.items[i].held = False
                    if self.items[i].visible:
                        self.items[i].shown_at_sec = self.clock_sec
                        self.items[i].remaining_sec = self.items[i].ttl_sec

        for i in range(len(self.items)):
            if self.items[i].visible and (not self.items[i].held):
                self.items[i].remaining_sec -= dt
                if self.items[i].remaining_sec <= Float32(0.0):
                    self.items[i].remaining_sec = Float32(0.0)
                    self.items[i].visible = False

    def visible_toasts(self) -> List[Int]:
        """Indices of up to `max_visible` active toasts: own jobs first, then others, newest first."""
        var out = List[Int](capacity=self.max_visible)
        for pass_own in range(2):
            var want_own = pass_own == 0
            var k = len(self.items) - 1
            while k >= 0 and len(out) < self.max_visible:
                ref it = self.items[k]
                if it.own_job == want_own and it.visible and (not it.held) and it.remaining_sec > Float32(0.0):
                    out.append(k)
                k -= 1
        return out^

    def held_count(self) -> Int:
        var cnt = 0
        for i in range(len(self.items)):
            if self.items[i].held and self.items[i].visible:
                cnt += 1
        return cnt

    def dismiss(mut self, toast_id: UInt64):
        for i in range(len(self.items)):
            if self.items[i].id == toast_id:
                self.items[i].visible = False
                self.items[i].held = False
                return

    def dismiss_all(mut self):
        for i in range(len(self.items)):
            self.items[i].visible = False
            self.items[i].held = False

    def mark_all_read(mut self):
        self.unread_count = 0

    def clear_history(mut self):
        self.items.clear()
        self.unread_count = 0


# ============================================================================
# Background Job Progress Tracker (Nielsen 0.1s / 1s / 10s Ladder)
# ============================================================================

@always_inline
def progress_stage(elapsed_ms: Int) -> Int:
    """Returns `PROGRESS_STAGE_QUIET` (<1s), `PROGRESS_STAGE_PILL` (1..10s), or `PROGRESS_STAGE_EXPANDED` (>=10s)."""
    if elapsed_ms < PROGRESS_QUIET_MS:
        return PROGRESS_STAGE_QUIET
    if elapsed_ms < PROGRESS_EXPANDED_MS:
        return PROGRESS_STAGE_PILL
    return PROGRESS_STAGE_EXPANDED


def format_elapsed_ms(elapsed_ms: Int) -> String:
    """Formats elapsed duration as `"3.2 s"` under 1 min, then `"1 min 05 s"`."""
    var ms = max(0, elapsed_ms)
    if ms < 60000:
        return String(ms // 1000) + "." + String((ms // 100) % 10) + " s"
    var s = ms // 1000
    var sec = s % 60
    return String(s // 60) + " min " + ("0" if sec < 10 else "") + String(sec) + " s"


def count_output_lines(data: List[UInt8]) -> Int:
    """Counts newline-delimited lines in raw process output (counting a non-empty trailing line)."""
    var n = 0
    for i in range(len(data)):
        if data[i] == 10:
            n += 1
    if len(data) > 0 and data[len(data) - 1] != 10:
        n += 1
    return n


def last_clean_output_line(data: List[UInt8]) -> String:
    """Extracts the latest non-empty line, handling `\\r` progress overwrites and stripping ANSI escapes."""
    var end = len(data)
    while end > 0:
        var start = end
        while start > 0 and data[start - 1] != 10:
            start -= 1
        var s = start
        for k in range(start, end):
            if data[k] == 13 and k + 1 < end:
                s = k + 1
        var out = List[UInt8]()
        var k = s
        while k < end:
            var b = data[k]
            if b == 27:
                k += 1
                if k < end and data[k] == 91:
                    k += 1
                    while k < end and (data[k] < 64 or data[k] > 126):
                        k += 1
                k += 1
                continue
            if b >= 32 or b == 9:
                out.append(b)
            k += 1
        var line = String(String(unsafe_from_utf8=out).strip())
        if line.byte_length() > 0:
            return line
        end = start - 1 if start > 0 else 0
    return ""


def job_kind_label(kind: Int) -> String:
    if kind == JOB_KIND_NAVMESH:
        return "NavMesh"
    if kind == JOB_KIND_BUILD_SO:
        return "Build"
    if kind == JOB_KIND_PACKAGE:
        return "Package"
    if kind == JOB_KIND_ASSET_IMPORT:
        return "Asset"
    if kind == JOB_KIND_AGENT:
        return "Agent"
    return "Task"


@fieldwise_init
struct ProgressJob(Copyable, Movable):
    """Single tracked background task displayed in Studio's progress pills."""
    var id: UInt64
    var kind: Int
    var title: String
    var detail: String
    var progress: Float32  # < 0.0 means indeterminate spinner; [0.0..1.0] determinate bar
    var elapsed_ms: Int
    var state: Int
    var cancellable: Bool
    var cancel_requested: Bool
    var always_show_pill: Bool

    def __init__(
        out self,
        id: UInt64 = 0,
        kind: Int = JOB_KIND_GENERIC,
        title: String = "",
        cancellable: Bool = True,
        always_show_pill: Bool = False,
    ):
        self.id = id
        self.kind = kind
        self.title = title
        self.detail = ""
        self.progress = Float32(-1.0)
        self.elapsed_ms = 0
        self.state = JOB_RUNNING
        self.cancellable = cancellable
        self.cancel_requested = False
        self.always_show_pill = always_show_pill

    def stage(self) -> Int:
        if self.state != JOB_RUNNING:
            return PROGRESS_STAGE_QUIET
        var st = progress_stage(self.elapsed_ms)
        if self.always_show_pill and st < PROGRESS_STAGE_PILL:
            return PROGRESS_STAGE_PILL
        return st

    def elapsed_label(self) -> String:
        return format_elapsed_ms(self.elapsed_ms)


struct JobProgressTracker(Copyable, Movable):
    """Non-blocking background job tracker for bakes, builds, packaging, and Agent turns."""
    var jobs: List[ProgressJob]
    var next_id: UInt64

    def __init__(out self):
        self.jobs = List[ProgressJob]()
        self.next_id = UInt64(1)

    def __init__(out self, *, copy: Self):
        self.jobs = copy.jobs.copy()
        self.next_id = copy.next_id

    def _find(self, job_id: UInt64) -> Int:
        for i in range(len(self.jobs)):
            if self.jobs[i].id == job_id:
                return i
        return -1

    def start_job(
        mut self,
        kind: Int,
        title: String,
        cancellable: Bool = True,
        always_show_pill: Bool = False,
    ) -> UInt64:
        var jid = self.next_id
        self.next_id += UInt64(1)
        self.jobs.append(ProgressJob(jid, kind, title, cancellable, always_show_pill))
        return jid

    def update_job(mut self, job_id: UInt64, progress: Float32 = -1.0, detail: String = ""):
        var idx = self._find(job_id)
        if idx < 0 or self.jobs[idx].state != JOB_RUNNING:
            return
        if isfinite(progress):
            if progress < Float32(0.0):
                self.jobs[idx].progress = Float32(-1.0)
            else:
                self.jobs[idx].progress = max(Float32(0.0), min(Float32(1.0), progress))
        if detail.byte_length() > 0:
            self.jobs[idx].detail = detail

    def feed_output_bytes(mut self, job_id: UInt64, data: List[UInt8]):
        """Updates a running job's detail string from raw stdout/stderr stream bytes."""
        var line = last_clean_output_line(data)
        if line.byte_length() > 0:
            self.update_job(job_id, progress=Float32(-1.0), detail=line)

    def request_cancel(mut self, job_id: UInt64) -> Bool:
        var idx = self._find(job_id)
        if idx < 0 or self.jobs[idx].state != JOB_RUNNING or (not self.jobs[idx].cancellable):
            return False
        self.jobs[idx].cancel_requested = True
        self.jobs[idx].detail = "Cancelling..."
        return True

    def is_cancel_requested(self, job_id: UInt64) -> Bool:
        var idx = self._find(job_id)
        if idx < 0:
            return False
        return self.jobs[idx].cancel_requested

    def complete_job(mut self, job_id: UInt64, succeeded: Bool, summary: String = "") -> Bool:
        var idx = self._find(job_id)
        if idx < 0:
            return False
        if self.jobs[idx].cancel_requested and (not succeeded):
            self.jobs[idx].state = JOB_CANCELLED
        else:
            self.jobs[idx].state = JOB_SUCCEEDED if succeeded else JOB_FAILED
        if succeeded:
            self.jobs[idx].progress = Float32(1.0)
        if summary.byte_length() > 0:
            self.jobs[idx].detail = summary
        return True

    def complete_job_with_toast(
        mut self,
        job_id: UInt64,
        succeeded: Bool,
        summary: String,
        mut toasts: ToastManager,
    ) -> Bool:
        """Completes `job_id` and emits a prioritized `own_job=True` toast."""
        var idx = self._find(job_id)
        if idx < 0:
            return False
        var title = self.jobs[idx].title
        var kind = self.jobs[idx].kind
        var cancelled = self.jobs[idx].cancel_requested and (not succeeded)
        _ = self.complete_job(job_id, succeeded, summary)
        var sev = TOAST_WARN if cancelled else (
            (TOAST_AGENT if kind == JOB_KIND_AGENT else TOAST_SUCCESS)
            if succeeded
            else TOAST_ERROR
        )
        var msg = summary if summary.byte_length() > 0 else (
            "Cancelled" if cancelled else ("Completed" if succeeded else "Failed")
        )
        _ = toasts.push(sev, title, msg, own_job=True)
        return True

    def tick(mut self, dt_ms: Int):
        if dt_ms <= 0:
            return
        for i in range(len(self.jobs)):
            if self.jobs[i].state == JOB_RUNNING:
                self.jobs[i].elapsed_ms += dt_ms

    def active_jobs(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.jobs)):
            if self.jobs[i].state == JOB_RUNNING:
                out.append(i)
        return out^

    def visible_pills(self) -> List[Int]:
        """Returns indices of running jobs past the 1s quiet threshold (or `always_show_pill`)."""
        var out = List[Int]()
        for i in range(len(self.jobs)):
            if self.jobs[i].stage() >= PROGRESS_STAGE_PILL:
                out.append(i)
        return out^

    def prune_finished(mut self, keep_last: Int = 16):
        var keep = max(0, keep_last)
        var finished_count = 0
        for i in range(len(self.jobs)):
            if self.jobs[i].state != JOB_RUNNING:
                finished_count += 1
        var to_drop = finished_count - keep
        if to_drop <= 0:
            return
        var next_jobs = List[ProgressJob](capacity=len(self.jobs) - to_drop)
        for i in range(len(self.jobs)):
            if self.jobs[i].state != JOB_RUNNING and to_drop > 0:
                to_drop -= 1
                continue
            next_jobs.append(self.jobs[i].copy())
        self.jobs = next_jobs^


# ============================================================================
# GPU SDF Rendering (`StudioTheme` + `add_glass_panel` + `add_progress_ring` + `UiAnimator`)
# ============================================================================


def _ntf_measure_text(font: SdfFont, text: String, size_px: Float32) -> Float32:
    if font.valid:
        return font.measure(text, size_px)
    return Float32(len(decode_utf8(text))) * size_px * Float32(0.56)


def _ntf_draw_text(
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    text: String,
    x: Float32,
    y: Float32,
    size_px: Float32,
    color: UInt32,
):
    if text.byte_length() == 0:
        return
    if font.valid:
        dl.add_text(font_slot, font, text, x, y + size_px * Float32(0.82), size_px, color)
    else:
        var n_cp = len(decode_utf8(text))
        if n_cp > 0:
            dl.add_rect(
                x,
                y + size_px * Float32(0.20),
                Float32(n_cp) * size_px * Float32(0.54),
                size_px * Float32(0.60),
                rgba8_scale_alpha(color, Float32(0.45)),
                Float32(2.0),
            )


def _ntf_truncate(font: SdfFont, text: String, size_px: Float32, max_w: Float32) -> String:
    if max_w <= Float32(6.0):
        return ""
    if _ntf_measure_text(font, text, size_px) <= max_w:
        return text
    var codes = decode_utf8(text)
    var ell = String("...")
    var ell_w = _ntf_measure_text(font, ell, size_px)
    var keep = len(codes)
    while keep > 0:
        var sub = encode_utf8(codes, 0, keep)
        if _ntf_measure_text(font, sub, size_px) + ell_w <= max_w:
            return sub + ell
        keep -= 1
    return ell


def toast_severity_color(theme: StudioTheme, severity: Int) -> UInt32:
    if severity == TOAST_SUCCESS:
        return theme.status_ok
    if severity == TOAST_WARN:
        return theme.status_warn
    if severity == TOAST_ERROR:
        return theme.status_danger
    if severity == TOAST_AGENT:
        return theme.agent_active
    return theme.accent_primary


@fieldwise_init
struct ToastDrawAction(Copyable, ImplicitlyCopyable, Movable):
    var dismissed_id: UInt64
    var clicked_action_id: UInt64
    var action_entity_id: Int
    var hovered: Bool

    def __init__(out self):
        self.dismissed_id = UInt64(0)
        self.clicked_action_id = UInt64(0)
        self.action_entity_id = -1
        self.hovered = False


def draw_toast_stack(
    mut manager: ToastManager,
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    theme: StudioTheme,
    mut anim: UiAnimator,
    vp: RectF,
    mouse_x: Float32,
    mouse_y: Float32,
    mouse_pressed: Bool,
    top_offset: Float32 = 48.0,
) -> ToastDrawAction:
    """Renders up to `max_visible` frosted-glass toasts anchored top-right inside `vp`."""
    var act = ToastDrawAction()
    var vis = manager.visible_toasts()
    if len(vis) == 0:
        return act

    var toast_w = min(Float32(316.0), max(Float32(180.0), vp.w - Float32(24.0)))
    var cur_y = vp.y + top_offset

    for vi in range(len(vis)):
        var idx = vis[vi]
        ref item = manager.items[idx]
        var wid = UInt64(0x544F4153540000) ^ item.id
        var slide = anim.drive_aux(wid, Float32(1.0), SpringConfig.drawer())
        if anim.mode == UI_ANIM_INSTANT:
            slide = Float32(1.0)
        var alpha = max(Float32(0.0), min(Float32(1.0), slide))
        var has_msg = item.message.byte_length() > 0
        var has_btn = item.action_label.byte_length() > 0
        var toast_h = Float32(54.0) if (has_msg or has_btn) else Float32(36.0)
        var tx = vp.right() - toast_w - Float32(12.0) + (Float32(1.0) - slide) * Float32(28.0)
        var tr = RectF(tx, cur_y, toast_w, toast_h)
        var hov = tr.contains(mouse_x, mouse_y)
        if hov:
            act.hovered = True

        var sev_col = toast_severity_color(theme, item.severity)
        dl.add_glass_panel(
            tr.x, tr.y, tr.w, tr.h,
            rgba8_scale_alpha(theme.l4_overlay, alpha),
            theme.radius_md, Float32(3.0), Float32(1.0),
            rgba8_scale_alpha(theme.border_strong if hov else theme.border_control, alpha),
            True, True,
        )
        # Left semantic accent stripe
        dl.add_rect(tr.x + Float32(1.0), tr.y + Float32(6.0), Float32(3.0), tr.h - Float32(12.0), rgba8_scale_alpha(sev_col, alpha), Float32(1.5))

        # Severity icon
        var ix = tr.x + Float32(12.0)
        var iy = tr.y + Float32(10.0)
        if item.severity == TOAST_SUCCESS:
            draw_studio_icon(dl, ICON_CHECK, ix, iy, Float32(14.0), rgba8_scale_alpha(sev_col, alpha), Float32(1.6))
        elif item.severity == TOAST_WARN:
            draw_studio_icon(dl, ICON_WARN, ix, iy, Float32(14.0), rgba8_scale_alpha(sev_col, alpha), Float32(1.5))
        elif item.severity == TOAST_ERROR:
            draw_studio_icon(dl, ICON_CLOSE, ix, iy, Float32(14.0), rgba8_scale_alpha(sev_col, alpha), Float32(1.6))
        elif item.severity == TOAST_AGENT:
            draw_studio_icon(dl, ICON_SPARK, ix, iy, Float32(14.0), rgba8_scale_alpha(sev_col, alpha), Float32(1.5))
        else:
            dl.add_circle(ix + Float32(7.0), iy + Float32(7.0), Float32(5.5), rgba8_scale_alpha(rgba8_with_alpha(sev_col, 50), alpha), Float32(1.4), rgba8_scale_alpha(sev_col, alpha))
            dl.add_circle(ix + Float32(7.0), iy + Float32(7.0), Float32(1.8), rgba8_scale_alpha(sev_col, alpha))

        # Close button
        var cls_r = RectF(tr.right() - Float32(24.0), tr.y + Float32(7.0), Float32(18.0), Float32(18.0))
        var cls_hov = cls_r.contains(mouse_x, mouse_y)
        if cls_hov:
            dl.add_rect(cls_r.x, cls_r.y, cls_r.w, cls_r.h, rgba8_scale_alpha(theme.l3_hover, alpha), Float32(4.0))
        draw_studio_icon(dl, ICON_CLOSE, cls_r.x + Float32(3.5), cls_r.y + Float32(3.5), Float32(11.0), rgba8_scale_alpha(theme.text_primary if cls_hov else theme.text_muted, alpha), Float32(1.4))
        if cls_hov and mouse_pressed:
            act.dismissed_id = item.id

        # Title + message
        var content_x = tr.x + Float32(32.0)
        var max_text_w = max(Float32(40.0), cls_r.x - content_x - Float32(6.0))
        var title_disp = _ntf_truncate(font, item.display_title(), Float32(12.5), max_text_w)
        _ntf_draw_text(dl, font_slot, font, title_disp, content_x, tr.y + Float32(9.0), Float32(12.5), rgba8_scale_alpha(theme.text_primary, alpha))

        if has_msg:
            var msg_max_w = max_text_w - (Float32(64.0) if has_btn else Float32(0.0))
            var msg_disp = _ntf_truncate(font, item.message, Float32(11.5), max(Float32(24.0), msg_max_w))
            _ntf_draw_text(dl, font_slot, font, msg_disp, content_x, tr.y + Float32(27.0), Float32(11.5), rgba8_scale_alpha(theme.text_secondary, alpha))

        if has_btn:
            var ab_w = min(Float32(68.0), _ntf_measure_text(font, item.action_label, Float32(11.0)) + Float32(12.0))
            var ab_r = RectF(tr.right() - ab_w - Float32(8.0), tr.bottom() - Float32(24.0), ab_w, Float32(18.0))
            var ab_hov = ab_r.contains(mouse_x, mouse_y)
            dl.add_rect(ab_r.x, ab_r.y, ab_r.w, ab_r.h, rgba8_scale_alpha(theme.l3_hover if ab_hov else theme.l2_card, alpha), Float32(4.0), Float32(1.0), rgba8_scale_alpha(sev_col, alpha * Float32(0.7)))
            _ntf_draw_text(dl, font_slot, font, item.action_label, ab_r.x + Float32(6.0), ab_r.y + Float32(3.5), Float32(10.5), rgba8_scale_alpha(theme.text_primary, alpha))
            if ab_hov and mouse_pressed:
                act.clicked_action_id = item.id
                act.action_entity_id = item.action_entity_id

        # Slim bottom TTL countdown bar
        var frac = max(Float32(0.0), min(Float32(1.0), item.remaining_sec / max(Float32(0.1), item.ttl_sec)))
        var bar_w = max(Float32(2.0), (tr.w - Float32(12.0)) * frac)
        dl.add_rect(tr.x + Float32(6.0), tr.bottom() - Float32(3.0), bar_w, Float32(2.0), rgba8_scale_alpha(sev_col, alpha * Float32(0.75)), Float32(1.0))

        cur_y += toast_h + Float32(8.0)

    if act.dismissed_id != 0:
        manager.dismiss(act.dismissed_id)
    return act


@fieldwise_init
struct JobProgressAction(Copyable, ImplicitlyCopyable, Movable):
    var cancel_job_id: UInt64
    var hovered: Bool

    def __init__(out self):
        self.cancel_job_id = UInt64(0)
        self.hovered = False


def draw_job_progress_pills(
    mut tracker: JobProgressTracker,
    mut dl: UiDrawList,
    font_slot: Int,
    font: SdfFont,
    theme: StudioTheme,
    mut anim: UiAnimator,
    vp: RectF,
    mouse_x: Float32,
    mouse_y: Float32,
    mouse_pressed: Bool,
    bottom_offset: Float32 = 122.0,
) -> JobProgressAction:
    """Renders non-blocking background job pills (Nielsen 1s compact pill / 10s expanded card)."""
    var act = JobProgressAction()
    var pills = tracker.visible_pills()
    if len(pills) == 0:
        return act

    var card_w = min(Float32(276.0), max(Float32(180.0), vp.w - Float32(24.0)))
    var bottom_y = vp.bottom() - bottom_offset

    for pi in range(len(pills)):
        var idx = pills[pi]
        ref job = tracker.jobs[idx]
        var stage = job.stage()
        var is_exp = stage >= PROGRESS_STAGE_EXPANDED
        var card_h = Float32(58.0) if is_exp else Float32(32.0)
        var wid = UInt64(0x4A4F4250524F4700) ^ job.id
        var reveal = anim.drive_aux(wid, Float32(1.0), SpringConfig.drawer())
        if anim.mode == UI_ANIM_INSTANT:
            reveal = Float32(1.0)
        var alpha = max(Float32(0.0), min(Float32(1.0), reveal))

        var cx = vp.right() - card_w - Float32(12.0)
        var cy = bottom_y - card_h + (Float32(1.0) - reveal) * Float32(10.0)
        var cr = RectF(cx, cy, card_w, card_h)
        var hov = cr.contains(mouse_x, mouse_y)
        if hov:
            act.hovered = True

        var accent = theme.agent_active if job.kind == JOB_KIND_AGENT else theme.accent_primary
        dl.add_glass_panel(
            cr.x, cr.y, cr.w, cr.h,
            rgba8_scale_alpha(theme.l4_overlay, alpha),
            theme.radius_md, Float32(3.0), Float32(1.0),
            rgba8_scale_alpha(theme.border_strong if hov else theme.border_control, alpha),
            True, True,
        )

        # Progress ring (determinate sweep or animated SDF arc spinner)
        var ring_x = cr.x + Float32(18.0)
        var ring_y = cr.y + Float32(16.0)
        if job.progress >= Float32(0.0):
            dl.add_progress_ring(
                ring_x, ring_y, Float32(7.5), Float32(2.2),
                job.progress,
                rgba8_scale_alpha(accent, alpha),
                rgba8_scale_alpha(theme.border_control, alpha),
                Float32(2.0),
            )
        else:
            var spin_rad = Float32(0.0) if anim.mode == UI_ANIM_INSTANT else (anim.time_sec * Float32(4.5))
            dl.add_arc(
                ring_x, ring_y, Float32(7.5), Float32(2.2),
                spin_rad, Float32(4.1),
                rgba8_scale_alpha(accent, alpha),
                rgba8_scale_alpha(theme.border_control, alpha),
                Float32(2.0),
            )

        # Right-side elapsed readout (+ Cancel chip when expanded & cancellable)
        var right_x = cr.right() - Float32(10.0)
        if is_exp and job.cancellable and not job.cancel_requested:
            var can_w = Float32(48.0)
            var can_r = RectF(right_x - can_w, cr.y + Float32(7.0), can_w, Float32(18.0))
            var can_hov = can_r.contains(mouse_x, mouse_y)
            dl.add_rect_styled(
                can_r.x, can_r.y, can_r.w, can_r.h,
                rgba8_scale_alpha(lerp_rgba8(theme.l2_card, theme.l3_hover, Float32(1.0) if can_hov else Float32(0.0)), alpha),
                rgba8_scale_alpha(theme.status_danger if can_hov else theme.border_control, alpha),
                Float32(4.0), Float32(1.0), Float32(1.0), UI_FLAG_BORDER | UI_FLAG_TOP_BEVEL,
            )
            _ntf_draw_text(dl, font_slot, font, "Cancel", can_r.x + Float32(6.0), can_r.y + Float32(3.5), Float32(10.5), rgba8_scale_alpha(theme.text_primary, alpha))
            if can_hov and mouse_pressed:
                act.cancel_job_id = job.id
            right_x = can_r.x - Float32(6.0)

        var el_str = job.elapsed_label()
        var el_w = _ntf_measure_text(font, el_str, Float32(11.0))
        _ntf_draw_text(dl, font_slot, font, el_str, right_x - el_w, cr.y + Float32(10.0), Float32(11.0), rgba8_scale_alpha(theme.text_muted, alpha))

        var title_x = cr.x + Float32(34.0)
        var title_max_w = max(Float32(40.0), right_x - el_w - title_x - Float32(8.0))
        var title_disp = _ntf_truncate(font, job.title, Float32(12.0), title_max_w)
        _ntf_draw_text(dl, font_slot, font, title_disp, title_x, cr.y + Float32(9.5), Float32(12.0), rgba8_scale_alpha(theme.text_primary, alpha))

        if is_exp:
            var det_txt = job.detail if job.detail.byte_length() > 0 else (job_kind_label(job.kind) + " in progress...")
            var det_disp = _ntf_truncate(font, det_txt, Float32(11.0), cr.w - Float32(22.0))
            _ntf_draw_text(dl, font_slot, font, det_disp, cr.x + Float32(11.0), cr.y + Float32(31.0), Float32(11.0), rgba8_scale_alpha(theme.text_secondary, alpha))

            # Bottom progress track
            var tr_x = cr.x + Float32(11.0)
            var tr_y = cr.bottom() - Float32(8.0)
            var tr_w = cr.w - Float32(22.0)
            dl.add_rect(tr_x, tr_y, tr_w, Float32(3.0), rgba8_scale_alpha(theme.l0_canvas, alpha), Float32(1.5))
            if job.progress >= Float32(0.0):
                dl.add_rect(tr_x, tr_y, max(Float32(3.0), tr_w * job.progress), Float32(3.0), rgba8_scale_alpha(accent, alpha), Float32(1.5))
            else:
                var phase = Float32(0.0) if anim.mode == UI_ANIM_INSTANT else (anim.time_sec * Float32(0.65) - Float32(Int(anim.time_sec * Float32(0.65))))
                var seg_w = tr_w * Float32(0.32)
                var seg_x = tr_x + phase * (tr_w - seg_w)
                dl.add_rect(seg_x, tr_y, seg_w, Float32(3.0), rgba8_scale_alpha(accent, alpha), Float32(1.5))

        bottom_y -= card_h + Float32(8.0)

    if act.cancel_job_id != 0:
        _ = tracker.request_cancel(act.cancel_job_id)
    return act

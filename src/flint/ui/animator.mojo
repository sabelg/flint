"""Retained Micro-Animation & Analytical Spring-Damper Engine (`flint.ui.animator`).

Provides zero-allocation-steady-state, NaN-safe, deterministic UI motion primitives:
  - `SpringConfig` & `UiSpring`: exact closed-form analytical damped harmonic oscillator
    (critically damped, underdamped, overdamped) preserving velocity on mid-flight retargeting.
  - `CubicBezier`: 4-iteration Newton-Raphson solver for timed transitions (toasts, fades).
  - `UiAnimState`, `UiTabIndicator`, `UiToastAnim`, `UiAgentTrail`: per-widget & agent motion records.
  - `UiAnimator`: sparse idle-evicted widget spring bank (`O(0)` cost per idle virtualized row)
    with `UI_ANIM_NORMAL`, `UI_ANIM_REDUCED`, and deterministic `UI_ANIM_INSTANT` modes.
"""
from std.collections import Dict, List
from std.math import abs, cos, exp, isfinite, max, min, sin, sqrt

comptime UI_ANIM_NORMAL: Int = 0
comptime UI_ANIM_REDUCED: Int = 1
comptime UI_ANIM_INSTANT: Int = 2

comptime TOAST_ANIM_ENTERING: Int = 0
comptime TOAST_ANIM_HOLDING: Int = 1
comptime TOAST_ANIM_EXITING: Int = 2
comptime TOAST_ANIM_DONE: Int = 3

comptime MAX_AGENT_TRAILS: Int = 64


@always_inline
def _safe_f32(v: Float32, fallback: Float32 = 0.0) -> Float32:
    return v if isfinite(v) else fallback


@always_inline
def lerp_rgba8(a: UInt32, b: UInt32, t_in: Float32) -> UInt32:
    """Linearly interpolates two packed `0xAARRGGBB` colors with clamped `t in [0, 1]`."""
    var t = max(Float32(0.0), min(Float32(1.0), _safe_f32(t_in, 0.0)))
    var ab = Float32(a & UInt32(0xFF))
    var ag = Float32((a >> 8) & UInt32(0xFF))
    var ar = Float32((a >> 16) & UInt32(0xFF))
    var aa = Float32((a >> 24) & UInt32(0xFF))
    var bb = Float32(b & UInt32(0xFF))
    var bg = Float32((b >> 8) & UInt32(0xFF))
    var br = Float32((b >> 16) & UInt32(0xFF))
    var ba = Float32((b >> 24) & UInt32(0xFF))
    var ob = UInt32(max(0, min(255, Int(ab + (bb - ab) * t + Float32(0.5)))))
    var og = UInt32(max(0, min(255, Int(ag + (bg - ag) * t + Float32(0.5)))))
    var or_c = UInt32(max(0, min(255, Int(ar + (br - ar) * t + Float32(0.5)))))
    var oa = UInt32(max(0, min(255, Int(aa + (ba - aa) * t + Float32(0.5)))))
    return ob | (og << 8) | (or_c << 16) | (oa << 24)


@fieldwise_init
struct SpringConfig(Copyable, ImplicitlyCopyable, Movable):
    """Damped harmonic oscillator tuning parameters."""
    var omega: Float32  # natural angular frequency (rad/s)
    var zeta: Float32   # damping ratio (1.0 = critically damped, <1.0 = underdamped)
    var eps: Float32    # settle tolerance

    @staticmethod
    def snappy() -> Self:
        """Crisp ~85ms critically damped spring for hover/press/buttons."""
        return Self(34.0, 1.0, 1.0e-3)

    @staticmethod
    def tab_pill() -> Self:
        """Slightly energetic spring for sliding active-tab indicators."""
        return Self(28.0, 0.92, 1.0e-3)

    @staticmethod
    def accordion() -> Self:
        """Smooth critically damped spring for accordions & chevron rotation."""
        return Self(24.0, 1.0, 1.0e-3)

    @staticmethod
    def focus_bloom() -> Self:
        """Fast critically damped spring for `:focus-visible` halo bloom."""
        return Self(30.0, 1.0, 1.0e-3)

    @staticmethod
    def scroll() -> Self:
        """Smooth pixel-space scroll inertia spring."""
        return Self(22.0, 1.0, 5.0e-2)

    @staticmethod
    def drawer() -> Self:
        """Gentle critically damped spring for drawers, toasts, and camera follow."""
        return Self(18.0, 1.0, 1.0e-3)

    @staticmethod
    def reduced() -> Self:
        """Reduced-motion fast critically damped crossfade preset (zero overshoot)."""
        return Self(64.0, 1.0, 1.0e-3)


@fieldwise_init
struct UiSpring(Copyable, ImplicitlyCopyable, Movable):
    """Unconditionally stable closed-form analytical 1D damped harmonic oscillator."""
    var value: Float32
    var velocity: Float32
    var target: Float32
    var omega: Float32
    var zeta: Float32
    var eps: Float32
    var initialized: Bool

    def __init__(out self):
        self.value = 0.0
        self.velocity = 0.0
        self.target = 0.0
        self.omega = 34.0
        self.zeta = 1.0
        self.eps = 1.0e-3
        self.initialized = False

    @staticmethod
    def create(initial: Float32 = 0.0, config: SpringConfig = SpringConfig.snappy()) -> Self:
        var v = _safe_f32(initial, 0.0)
        var w = max(Float32(0.1), _safe_f32(config.omega, 34.0))
        var z = max(Float32(0.05), _safe_f32(config.zeta, 1.0))
        var e = max(Float32(1.0e-5), _safe_f32(config.eps, 1.0e-3))
        return Self(v, 0.0, v, w, z, e, True)

    def snap(mut self, val: Float32):
        var v = _safe_f32(val, 0.0)
        self.value = v
        self.target = v
        self.velocity = 0.0
        self.initialized = True

    def set_config(mut self, config: SpringConfig):
        self.omega = max(Float32(0.1), _safe_f32(config.omega, 34.0))
        self.zeta = max(Float32(0.05), _safe_f32(config.zeta, 1.0))
        self.eps = max(Float32(1.0e-5), _safe_f32(config.eps, 1.0e-3))

    def set_target(mut self, target_val: Float32, snap_if_uninit: Bool = True):
        var t = _safe_f32(target_val, 0.0)
        if not self.initialized and snap_if_uninit:
            self.snap(t)
        else:
            self.target = t
            self.initialized = True

    def impulse(mut self, dv: Float32):
        if isfinite(dv):
            self.velocity = _safe_f32(self.velocity, 0.0) + dv
            self.initialized = True

    @always_inline
    def is_settled(self) -> Bool:
        if not (isfinite(self.value) and isfinite(self.velocity) and isfinite(self.target)):
            return True
        var dx = abs(self.value - self.target)
        var v_tol = self.eps * max(Float32(1.0), self.omega)
        return dx <= self.eps and abs(self.velocity) <= v_tol

    def step(mut self, dt_in: Float32, mode: Int = UI_ANIM_NORMAL) -> Float32:
        if not isfinite(self.target):
            self.target = 0.0
        if not (isfinite(self.value) and isfinite(self.velocity)):
            self.snap(self.target)
            return self.value
        if mode == UI_ANIM_INSTANT:
            self.snap(self.target)
            return self.value
        if not isfinite(dt_in) or dt_in <= Float32(0.0):
            return self.value
        if self.is_settled():
            self.value = self.target
            self.velocity = 0.0
            return self.value

        var dt = min(Float32(0.25), dt_in)
        var w = max(Float32(0.1), _safe_f32(self.omega, 34.0))
        var z = max(Float32(0.05), _safe_f32(self.zeta, 1.0))
        if mode == UI_ANIM_REDUCED:
            w = max(w * Float32(2.0), Float32(60.0))
            z = Float32(1.0)

        var x0 = self.value - self.target
        var v0 = self.velocity
        var x_new: Float32
        var v_new: Float32

        if abs(z - Float32(1.0)) < Float32(1.0e-3):
            # Critically damped: exact analytical closed form
            var e = exp(-w * dt)
            var c2 = v0 + w * x0
            x_new = (x0 + c2 * dt) * e
            v_new = (v0 - c2 * w * dt) * e
        elif z < Float32(1.0):
            # Underdamped: exact analytical damped sinusoid
            var wd = w * sqrt(max(Float32(1.0e-6), Float32(1.0) - z * z))
            var zw = z * w
            var e = exp(-zw * dt)
            var cs = cos(wd * dt)
            var sn = sin(wd * dt)
            var b = (v0 + zw * x0) / wd
            x_new = e * (x0 * cs + b * sn)
            v_new = e * ((b * wd - zw * x0) * cs - (x0 * wd + zw * b) * sn)
        else:
            # Overdamped: exact dual exponential
            var rt = sqrt(max(Float32(1.0e-6), z * z - Float32(1.0)))
            var r1 = -w * (z - rt)
            var r2 = -w * (z + rt)
            var denom = r2 - r1
            if abs(denom) < Float32(1.0e-6):
                var e = exp(-w * dt)
                x_new = x0 * e
                v_new = v0 * e
            else:
                var c2 = (v0 - r1 * x0) / denom
                var c1 = x0 - c2
                var e1 = exp(r1 * dt)
                var e2 = exp(r2 * dt)
                x_new = c1 * e1 + c2 * e2
                v_new = c1 * r1 * e1 + c2 * r2 * e2

        var nv = self.target + x_new
        if not (isfinite(nv) and isfinite(v_new)):
            self.snap(self.target)
            return self.value
        self.value = nv
        self.velocity = v_new
        if self.is_settled():
            self.value = self.target
            self.velocity = 0.0
        return self.value


@fieldwise_init
struct CubicBezier(Copyable, ImplicitlyCopyable, Movable):
    """Cubic Bézier timing curve `P0=(0,0), P1=(x1,y1), P2=(x2,y2), P3=(1,1)`."""
    var x1: Float32
    var y1: Float32
    var x2: Float32
    var y2: Float32

    @staticmethod
    def linear() -> Self:
        return Self(0.0, 0.0, 1.0, 1.0)

    @staticmethod
    def standard() -> Self:
        return Self(0.2, 0.0, 0.0, 1.0)

    @staticmethod
    def emphasized_decel() -> Self:
        return Self(0.05, 0.7, 0.1, 1.0)

    @staticmethod
    def emphasized_accel() -> Self:
        return Self(0.3, 0.0, 0.8, 0.15)

    @staticmethod
    @always_inline
    def _eval_axis(t: Float32, p1: Float32, p2: Float32) -> Float32:
        var u = Float32(1.0) - t
        return (Float32(3.0) * u * u * p1 + Float32(3.0) * u * t * p2 + t * t) * t

    @staticmethod
    @always_inline
    def _eval_deriv(t: Float32, p1: Float32, p2: Float32) -> Float32:
        var u = Float32(1.0) - t
        return Float32(3.0) * u * u * p1 + Float32(6.0) * u * t * (p2 - p1) + Float32(3.0) * t * t * (Float32(1.0) - p2)

    def sample(self, x_in: Float32) -> Float32:
        var x = max(Float32(0.0), min(Float32(1.0), _safe_f32(x_in, 0.0)))
        if x <= Float32(0.0):
            return 0.0
        if x >= Float32(1.0):
            return 1.0
        var lo = Float32(0.0)
        var hi = Float32(1.0)
        var t = x
        for _ in range(5):
            var bx = Self._eval_axis(t, self.x1, self.x2)
            var err = bx - x
            if abs(err) < Float32(1.0e-5):
                break
            if err < Float32(0.0):
                lo = t
            else:
                hi = t
            var dx = Self._eval_deriv(t, self.x1, self.x2)
            if abs(dx) > Float32(1.0e-5):
                var next_t = t - err / dx
                if next_t > lo and next_t < hi:
                    t = next_t
                    continue
            t = Float32(0.5) * (lo + hi)
        return Self._eval_axis(t, self.y1, self.y2)


@fieldwise_init
struct UiAnimState(Copyable, ImplicitlyCopyable, Movable):
    """Per-widget retained spring channels (evicted automatically once settled at rest)."""
    var hover: UiSpring
    var press: UiSpring
    var focus: UiSpring
    var expand: UiSpring
    var aux: UiSpring
    var scroll_y: UiSpring
    var last_epoch: UInt64

    def __init__(out self):
        self.hover = UiSpring.create(0.0, SpringConfig.snappy())
        self.press = UiSpring.create(0.0, SpringConfig.snappy())
        self.focus = UiSpring.create(0.0, SpringConfig.focus_bloom())
        self.expand = UiSpring.create(0.0, SpringConfig.accordion())
        self.aux = UiSpring.create(0.0, SpringConfig.snappy())
        self.scroll_y = UiSpring.create(0.0, SpringConfig.scroll())
        self.last_epoch = 0

    def step_all(mut self, dt: Float32, mode: Int) -> Int:
        """Steps all channels and returns the number of channels still actively moving."""
        var moving = 0
        if not self.hover.is_settled():
            _ = self.hover.step(dt, mode)
            if not self.hover.is_settled():
                moving += 1
        if not self.press.is_settled():
            _ = self.press.step(dt, mode)
            if not self.press.is_settled():
                moving += 1
        if not self.focus.is_settled():
            _ = self.focus.step(dt, mode)
            if not self.focus.is_settled():
                moving += 1
        if not self.expand.is_settled():
            _ = self.expand.step(dt, mode)
            if not self.expand.is_settled():
                moving += 1
        if not self.aux.is_settled():
            _ = self.aux.step(dt, mode)
            if not self.aux.is_settled():
                moving += 1
        if not self.scroll_y.is_settled():
            _ = self.scroll_y.step(dt, mode)
            if not self.scroll_y.is_settled():
                moving += 1
        return moving

    @always_inline
    def can_evict_at_rest(self) -> Bool:
        """Returns `True` when all springs are settled at their default zero rest pose."""
        return (
            self.hover.is_settled()
            and self.press.is_settled()
            and self.focus.is_settled()
            and self.expand.is_settled()
            and self.aux.is_settled()
            and self.scroll_y.is_settled()
            and abs(self.hover.value) <= self.hover.eps
            and abs(self.press.value) <= self.press.eps
            and abs(self.focus.value) <= self.focus.eps
            and abs(self.expand.value) <= self.expand.eps
            and abs(self.aux.value) <= self.aux.eps
            and abs(self.scroll_y.value) <= self.scroll_y.eps
        )


@fieldwise_init
struct UiTabIndicator(Copyable, ImplicitlyCopyable, Movable):
    """Sliding active-tab pill indicator state keyed by tab bar / dock node ID."""
    var x: UiSpring
    var w: UiSpring
    var last_epoch: UInt64

    def __init__(out self):
        self.x = UiSpring.create(0.0, SpringConfig.tab_pill())
        self.x.initialized = False
        self.w = UiSpring.create(0.0, SpringConfig.tab_pill())
        self.w.initialized = False
        self.last_epoch = 0

    def step(mut self, dt: Float32, mode: Int) -> Int:
        var moving = 0
        if not self.x.is_settled():
            _ = self.x.step(dt, mode)
            if not self.x.is_settled():
                moving += 1
        if not self.w.is_settled():
            _ = self.w.step(dt, mode)
            if not self.w.is_settled():
                moving += 1
        return moving


@fieldwise_init
struct UiToastAnim(Copyable, ImplicitlyCopyable, Movable):
    """Animated toast slide-in / hold / fade-out entry (supports hover-to-pause)."""
    var id: UInt64
    var slide: UiSpring
    var alpha: Float32
    var age_sec: Float32
    var ttl_sec: Float32
    var enter_sec: Float32
    var exit_sec: Float32
    var state: Int
    var paused: Bool

    @staticmethod
    def create(id: UInt64, ttl_sec: Float32 = 3.5) -> Self:
        var sp = UiSpring.create(0.0, SpringConfig.drawer())
        sp.set_target(1.0, snap_if_uninit=False)
        return Self(
            id,
            sp,
            0.0,
            0.0,
            max(Float32(0.25), _safe_f32(ttl_sec, 3.5)),
            0.22,
            0.16,
            TOAST_ANIM_ENTERING,
            False,
        )

    def step(mut self, dt_in: Float32, mode: Int):
        if mode == UI_ANIM_INSTANT:
            if self.state == TOAST_ANIM_ENTERING:
                self.slide.snap(1.0)
                self.alpha = 1.0
                self.state = TOAST_ANIM_HOLDING
            elif self.state == TOAST_ANIM_EXITING:
                self.slide.snap(0.0)
                self.alpha = 0.0
                self.state = TOAST_ANIM_DONE
            if (not self.paused) and isfinite(dt_in) and dt_in > Float32(0.0):
                self.age_sec += dt_in
                if self.state == TOAST_ANIM_HOLDING and self.age_sec >= self.ttl_sec:
                    self.slide.snap(0.0)
                    self.alpha = 0.0
                    self.state = TOAST_ANIM_DONE
            return

        if not isfinite(dt_in) or dt_in <= Float32(0.0):
            return
        var dt = min(Float32(0.25), dt_in)
        _ = self.slide.step(dt, mode)

        if self.state == TOAST_ANIM_ENTERING:
            self.age_sec += dt
            var u = min(Float32(1.0), self.age_sec / max(Float32(1.0e-3), self.enter_sec))
            self.alpha = CubicBezier.emphasized_decel().sample(u)
            if u >= Float32(1.0):
                self.alpha = 1.0
                self.state = TOAST_ANIM_HOLDING
                self.age_sec = 0.0
        elif self.state == TOAST_ANIM_HOLDING:
            self.alpha = 1.0
            if not self.paused:
                self.age_sec += dt
                if self.age_sec >= self.ttl_sec:
                    self.state = TOAST_ANIM_EXITING
                    self.age_sec = 0.0
                    self.slide.set_target(0.0)
        elif self.state == TOAST_ANIM_EXITING:
            self.age_sec += dt
            var u = min(Float32(1.0), self.age_sec / max(Float32(1.0e-3), self.exit_sec))
            self.alpha = Float32(1.0) - CubicBezier.emphasized_accel().sample(u)
            if u >= Float32(1.0):
                self.alpha = 0.0
                self.state = TOAST_ANIM_DONE


@fieldwise_init
struct UiAgentTrail(Copyable, ImplicitlyCopyable, Movable):
    """Fading viewport trail marker on a recently agent-modified entity."""
    var entity_id: UInt64
    var x: Float32
    var y: Float32
    var age_sec: Float32
    var max_age_sec: Float32
    var color_rgba8: UInt32

    @always_inline
    def alpha(self) -> Float32:
        if not isfinite(self.age_sec) or self.age_sec < Float32(0.0):
            return 1.0
        if self.age_sec >= self.max_age_sec:
            return 0.0
        return exp(-self.age_sec / Float32(1.2))


struct UiAnimator(Movable):
    """Sparse, idle-evicted UI spring & micro-animation bank keyed by widget ID.

    Idle virtualized rows (`tree_node_indexed`) incur `O(0)` allocations and zero
    hash lookups when no springs are active (`len(widgets) == 0`).
    """
    var mode: Int
    var time_sec: Float32
    var frame_epoch: UInt64
    var gc_stale_frames: UInt64
    var widgets: Dict[UInt64, UiAnimState]
    var tab_pills: Dict[UInt64, UiTabIndicator]
    var toasts: List[UiToastAnim]
    var agent_rings: Dict[UInt64, UiSpring]
    var agent_ring_epochs: Dict[UInt64, UInt64]
    var agent_trails: List[UiAgentTrail]
    var active_springs_count: Int

    def __init__(out self, mode: Int = UI_ANIM_NORMAL):
        self.mode = mode
        self.time_sec = 0.0
        self.frame_epoch = 0
        self.gc_stale_frames = 120
        self.widgets = Dict[UInt64, UiAnimState]()
        self.tab_pills = Dict[UInt64, UiTabIndicator]()
        self.toasts = List[UiToastAnim]()
        self.agent_rings = Dict[UInt64, UiSpring]()
        self.agent_ring_epochs = Dict[UInt64, UInt64]()
        self.agent_trails = List[UiAgentTrail]()
        self.active_springs_count = 0

    def set_mode(mut self, mode: Int):
        self.mode = mode
        if mode == UI_ANIM_INSTANT:
            self.snap_all()

    def snap_all(mut self):
        """Immediately snaps every tracked spring to its target (deterministic golden capture)."""
        var w_keys = List[UInt64](capacity=len(self.widgets))
        for item in self.widgets.items():
            w_keys.append(item.key)
        for k in w_keys:
            var found = self.widgets.get(k)
            if found:
                var st = found.value()
                _ = st.step_all(Float32(0.016), UI_ANIM_INSTANT)
                if st.can_evict_at_rest():
                    try:
                        _ = self.widgets.pop(k)
                    except:
                        pass
                else:
                    self.widgets[k] = st

        var t_keys = List[UInt64](capacity=len(self.tab_pills))
        for item in self.tab_pills.items():
            t_keys.append(item.key)
        for k in t_keys:
            var found = self.tab_pills.get(k)
            if found:
                var ti = found.value()
                _ = ti.step(Float32(0.016), UI_ANIM_INSTANT)
                self.tab_pills[k] = ti

        var a_keys = List[UInt64](capacity=len(self.agent_rings))
        for item in self.agent_rings.items():
            a_keys.append(item.key)
        for k in a_keys:
            var found = self.agent_rings.get(k)
            if found:
                var sp = found.value()
                _ = sp.step(Float32(0.016), UI_ANIM_INSTANT)
                self.agent_rings[k] = sp

        self.active_springs_count = 0

    def begin_frame(mut self, dt_in: Float32):
        """Advances all active springs, toasts, and trails; evicts converged idle widgets."""
        self.frame_epoch += 1
        var dt = _safe_f32(dt_in, Float32(1.0 / 60.0))
        if dt < Float32(0.0):
            dt = Float32(0.0)
        if self.mode != UI_ANIM_INSTANT:
            self.time_sec += min(Float32(0.25), dt)
        var moving = 0

        # 1. Step & immediately evict converged rest-state widgets (O(active_widgets))
        if len(self.widgets) > 0:
            var keys = List[UInt64](capacity=len(self.widgets))
            for item in self.widgets.items():
                keys.append(item.key)
            var cutoff = (
                self.frame_epoch - self.gc_stale_frames
                if self.frame_epoch > self.gc_stale_frames
                else UInt64(0)
            )
            for k in keys:
                var found = self.widgets.get(k)
                if not found:
                    continue
                var st = found.value()
                var m = st.step_all(dt, self.mode)
                moving += m
                if (m == 0 and st.can_evict_at_rest()) or (cutoff > 0 and st.last_epoch < cutoff):
                    try:
                        _ = self.widgets.pop(k)
                    except:
                        pass
                else:
                    self.widgets[k] = st

        # 2. Step sliding tab indicators + stale GC
        if len(self.tab_pills) > 0:
            var t_keys = List[UInt64](capacity=len(self.tab_pills))
            for item in self.tab_pills.items():
                t_keys.append(item.key)
            var cutoff = (
                self.frame_epoch - self.gc_stale_frames
                if self.frame_epoch > self.gc_stale_frames
                else UInt64(0)
            )
            for k in t_keys:
                var found = self.tab_pills.get(k)
                if not found:
                    continue
                var ti = found.value()
                var m = ti.step(dt, self.mode)
                moving += m
                if cutoff > 0 and ti.last_epoch < cutoff and m == 0:
                    try:
                        _ = self.tab_pills.pop(k)
                    except:
                        pass
                else:
                    self.tab_pills[k] = ti

        # 3. Step toast lifecycles & compact expired toasts
        if len(self.toasts) > 0:
            var kept = List[UiToastAnim](capacity=len(self.toasts))
            for i in range(len(self.toasts)):
                var ts = self.toasts[i]
                ts.step(dt, self.mode)
                if ts.state != TOAST_ANIM_DONE:
                    kept.append(ts)
            self.toasts = kept^

        # 4. Step agent progress/beam springs + stale GC
        if len(self.agent_rings) > 0:
            var a_keys = List[UInt64](capacity=len(self.agent_rings))
            for item in self.agent_rings.items():
                a_keys.append(item.key)
            var cutoff = (
                self.frame_epoch - self.gc_stale_frames
                if self.frame_epoch > self.gc_stale_frames
                else UInt64(0)
            )
            for k in a_keys:
                var found = self.agent_rings.get(k)
                if not found:
                    continue
                var sp = found.value()
                if not sp.is_settled():
                    _ = sp.step(dt, self.mode)
                    if not sp.is_settled():
                        moving += 1
                var ep_opt = self.agent_ring_epochs.get(k)
                var ep = ep_opt.value() if ep_opt else self.frame_epoch
                if cutoff > 0 and ep < cutoff and sp.is_settled():
                    try:
                        _ = self.agent_rings.pop(k)
                    except:
                        pass
                    try:
                        _ = self.agent_ring_epochs.pop(k)
                    except:
                        pass
                else:
                    self.agent_rings[k] = sp

        # 5. Age & prune viewport agent trails
        if len(self.agent_trails) > 0:
            var kept_trails = List[UiAgentTrail](capacity=len(self.agent_trails))
            var step_dt = min(Float32(0.25), dt)
            for i in range(len(self.agent_trails)):
                var tr = self.agent_trails[i]
                if self.mode == UI_ANIM_INSTANT:
                    tr.age_sec += step_dt
                else:
                    tr.age_sec += step_dt
                if tr.age_sec < tr.max_age_sec:
                    kept_trails.append(tr)
            self.agent_trails = kept_trails^

        self.active_springs_count = moving

    @always_inline
    def is_animating(self) -> Bool:
        """Returns `True` while any spring, toast, or viewport trail requires redraw."""
        return self.active_springs_count > 0 or len(self.toasts) > 0 or len(self.agent_trails) > 0

    def drive_widget(
        mut self, wid: UInt64, hovered: Bool, pressed: Bool, focused: Bool
    ) -> Tuple[Float32, Float32, Float32]:
        """Updates hover/press/focus targets and returns `(hover_t, press_t, focus_t)` in `[0, 1]`.

        Fast path: idle widgets pay `O(0)` allocations and zero dict insertions.
        """
        if not hovered and not pressed and not focused:
            if len(self.widgets) == 0:
                return (0.0, 0.0, 0.0)
            var existing = self.widgets.get(wid)
            if not existing:
                return (0.0, 0.0, 0.0)
            var st = existing.value()
            st.hover.set_target(0.0, snap_if_uninit=False)
            st.press.set_target(0.0, snap_if_uninit=False)
            st.focus.set_target(0.0, snap_if_uninit=False)
            if self.mode == UI_ANIM_INSTANT:
                st.hover.snap(0.0)
                st.press.snap(0.0)
                st.focus.snap(0.0)
            st.last_epoch = self.frame_epoch
            self.widgets[wid] = st
            return (st.hover.value, st.press.value, st.focus.value)

        var found = self.widgets.get(wid)
        var st = found.value() if found else UiAnimState()
        var th = Float32(1.0) if hovered else Float32(0.0)
        var tp = Float32(1.0) if pressed else Float32(0.0)
        var tf = Float32(1.0) if focused else Float32(0.0)
        st.hover.set_target(th, snap_if_uninit=False)
        st.press.set_target(tp, snap_if_uninit=False)
        st.focus.set_target(tf, snap_if_uninit=False)
        if self.mode == UI_ANIM_INSTANT:
            st.hover.snap(th)
            st.press.snap(tp)
            st.focus.snap(tf)
        st.last_epoch = self.frame_epoch
        self.widgets[wid] = st
        return (st.hover.value, st.press.value, st.focus.value)

    def drive_expand(mut self, wid: UInt64, expanded: Bool) -> Float32:
        """Smoothly animates accordion / tree-node expansion (`0.0` closed -> `1.0` open)."""
        var tgt = Float32(1.0) if expanded else Float32(0.0)
        var found = self.widgets.get(wid)
        if not found:
            if not expanded:
                return 0.0
            var st = UiAnimState()
            st.expand.snap(1.0)
            st.last_epoch = self.frame_epoch
            self.widgets[wid] = st
            return 1.0
        var st = found.value()
        st.expand.set_target(tgt, snap_if_uninit=False)
        if self.mode == UI_ANIM_INSTANT:
            st.expand.snap(tgt)
        st.last_epoch = self.frame_epoch
        self.widgets[wid] = st
        return st.expand.value

    def drive_aux(
        mut self, wid: UInt64, target: Float32, config: SpringConfig = SpringConfig.snappy()
    ) -> Float32:
        """General-purpose per-widget spring (toggle thumb, progress fill, badge pop-in)."""
        var tgt = _safe_f32(target, 0.0)
        var found = self.widgets.get(wid)
        if not found and abs(tgt) <= config.eps:
            return 0.0
        var st = found.value() if found else UiAnimState()
        st.aux.set_config(config)
        st.aux.set_target(tgt, snap_if_uninit=not found)
        if self.mode == UI_ANIM_INSTANT:
            st.aux.snap(tgt)
        elif not st.aux.is_settled():
            self.active_springs_count = max(self.active_springs_count, 1)
        st.last_epoch = self.frame_epoch
        self.widgets[wid] = st
        return st.aux.value

    def drive_reveal(
        mut self,
        wid: UInt64,
        target: Float32,
        initial: Float32 = 0.0,
        config: SpringConfig = SpringConfig.drawer(),
    ) -> Float32:
        """Animates enter/exit reveal transitions starting from `initial` (never snaps on frame 1 in live mode).

        In `UI_ANIM_INSTANT` mode, snaps immediately to `target` for deterministic golden captures.
        Hidden steady-state calls (`target == initial == 0.0`) allocate nothing (`O(0)`).
        """
        var tgt = _safe_f32(target, 0.0)
        var init_v = _safe_f32(initial, 0.0)
        var found = self.widgets.get(wid)
        if not found:
            if self.mode == UI_ANIM_INSTANT:
                if abs(tgt) <= config.eps:
                    return 0.0
                var st_i = UiAnimState()
                st_i.aux = UiSpring.create(tgt, config)
                st_i.last_epoch = self.frame_epoch
                self.widgets[wid] = st_i
                return tgt
            if abs(tgt - init_v) <= config.eps and abs(tgt) <= config.eps:
                return 0.0
            var st_new = UiAnimState()
            st_new.aux = UiSpring.create(init_v, config)
            st_new.aux.set_target(tgt, snap_if_uninit=False)
            if not st_new.aux.is_settled():
                self.active_springs_count = max(self.active_springs_count, 1)
            st_new.last_epoch = self.frame_epoch
            self.widgets[wid] = st_new
            return st_new.aux.value

        var st = found.value()
        st.aux.set_config(config)
        st.aux.set_target(tgt, snap_if_uninit=False)
        if self.mode == UI_ANIM_INSTANT:
            st.aux.snap(tgt)
        elif not st.aux.is_settled():
            self.active_springs_count = max(self.active_springs_count, 1)
        st.last_epoch = self.frame_epoch
        self.widgets[wid] = st
        return st.aux.value

    def drive_scroll(mut self, wid: UInt64, target_y: Float32, snap_now: Bool = False) -> Float32:
        """Smooth scroll inertia spring (`snap_now=True` for 1:1 scrollbar thumb dragging)."""
        var tgt = max(Float32(0.0), _safe_f32(target_y, 0.0))
        var found = self.widgets.get(wid)
        if not found and tgt <= Float32(0.05):
            return 0.0
        var st = found.value() if found else UiAnimState()
        if not found or snap_now or self.mode == UI_ANIM_INSTANT:
            st.scroll_y.snap(tgt)
        else:
            st.scroll_y.set_target(tgt, snap_if_uninit=True)
        st.last_epoch = self.frame_epoch
        self.widgets[wid] = st
        return st.scroll_y.value

    def drive_tab_pill(
        mut self, group_id: UInt64, target_x: Float32, target_w: Float32
    ) -> Tuple[Float32, Float32]:
        """Glides an active-tab indicator pill `(x, w)` smoothly across tab switches."""
        var tx = _safe_f32(target_x, 0.0)
        var tw = max(Float32(0.0), _safe_f32(target_w, 0.0))
        var found = self.tab_pills.get(group_id)
        var ti = found.value() if found else UiTabIndicator()
        if not found or self.mode == UI_ANIM_INSTANT:
            ti.x.snap(tx)
            ti.w.snap(tw)
        else:
            ti.x.set_target(tx, snap_if_uninit=True)
            ti.w.set_target(tw, snap_if_uninit=True)
        ti.last_epoch = self.frame_epoch
        self.tab_pills[group_id] = ti
        return (ti.x.value, ti.w.value)

    def push_toast(mut self, id: UInt64, ttl_sec: Float32 = 3.5):
        for i in range(len(self.toasts)):
            if self.toasts[i].id == id:
                self.toasts[i].age_sec = 0.0
                self.toasts[i].ttl_sec = max(Float32(0.25), _safe_f32(ttl_sec, 3.5))
                if self.toasts[i].state == TOAST_ANIM_EXITING:
                    self.toasts[i].state = TOAST_ANIM_ENTERING
                    self.toasts[i].slide.set_target(1.0)
                return
        var entry = UiToastAnim.create(id, ttl_sec)
        if self.mode == UI_ANIM_INSTANT:
            entry.slide.snap(1.0)
            entry.alpha = 1.0
            entry.state = TOAST_ANIM_HOLDING
        self.toasts.append(entry)

    def dismiss_toast(mut self, id: UInt64):
        for i in range(len(self.toasts)):
            if self.toasts[i].id == id and self.toasts[i].state < TOAST_ANIM_EXITING:
                self.toasts[i].state = TOAST_ANIM_EXITING
                self.toasts[i].paused = False
                self.toasts[i].age_sec = 0.0
                self.toasts[i].slide.set_target(0.0)
                if self.mode == UI_ANIM_INSTANT:
                    self.toasts[i].slide.snap(0.0)
                    self.toasts[i].alpha = 0.0
                    self.toasts[i].state = TOAST_ANIM_DONE
                return

    def sync_toast(
        mut self,
        id: UInt64,
        active: Bool,
        ttl_sec: Float32 = 5.0,
        hovered: Bool = False,
    ) -> Tuple[Float32, Float32]:
        """Synchronizes a toast's 3-phase `UiToastAnim` lifecycle and returns `(slide, alpha)` in `[0, 1]`.

        In `UI_ANIM_INSTANT` mode, active toasts snap cleanly to `(1.0, 1.0)` and inactive toasts to `(0.0, 0.0)`.
        """
        for i in range(len(self.toasts)):
            if self.toasts[i].id == id:
                if active:
                    self.toasts[i].paused = hovered
                    if self.toasts[i].state == TOAST_ANIM_EXITING:
                        self.toasts[i].state = TOAST_ANIM_ENTERING
                        self.toasts[i].age_sec = 0.0
                        self.toasts[i].slide.set_target(1.0)
                    if self.mode == UI_ANIM_INSTANT:
                        self.toasts[i].slide.snap(1.0)
                        self.toasts[i].alpha = 1.0
                        self.toasts[i].state = TOAST_ANIM_HOLDING
                else:
                    self.toasts[i].paused = False
                    if self.toasts[i].state < TOAST_ANIM_EXITING:
                        self.toasts[i].state = TOAST_ANIM_EXITING
                        self.toasts[i].age_sec = 0.0
                        self.toasts[i].slide.set_target(0.0)
                    if self.mode == UI_ANIM_INSTANT:
                        self.toasts[i].slide.snap(0.0)
                        self.toasts[i].alpha = 0.0
                        self.toasts[i].state = TOAST_ANIM_DONE
                var sl = max(Float32(0.0), min(Float32(1.0), self.toasts[i].slide.value))
                var al = max(Float32(0.0), min(Float32(1.0), self.toasts[i].alpha))
                return (sl, al)

        if not active:
            return (Float32(0.0), Float32(0.0))

        var entry = UiToastAnim.create(id, ttl_sec)
        entry.paused = hovered
        if self.mode == UI_ANIM_INSTANT:
            entry.slide.snap(1.0)
            entry.alpha = 1.0
            entry.state = TOAST_ANIM_HOLDING
        var out_sl = max(Float32(0.0), min(Float32(1.0), entry.slide.value))
        var out_al = max(Float32(0.0), min(Float32(1.0), entry.alpha))
        self.toasts.append(entry)
        return (out_sl, out_al)

    def toast_slide_alpha(self, id: UInt64) -> Tuple[Float32, Float32]:
        """Read-only query of a tracked toast's `(slide, alpha)` pair (`(0, 0)` if untracked)."""
        for i in range(len(self.toasts)):
            if self.toasts[i].id == id:
                var sl = max(Float32(0.0), min(Float32(1.0), self.toasts[i].slide.value))
                var al = max(Float32(0.0), min(Float32(1.0), self.toasts[i].alpha))
                return (sl, al)
        return (Float32(0.0), Float32(0.0))

    def drive_agent_ring(mut self, agent_id: UInt64, target_progress: Float32) -> Float32:
        """Smoothly drives an agent activity progress / beam-extend scalar in `[0, 1]`."""
        var tgt = max(Float32(0.0), min(Float32(1.0), _safe_f32(target_progress, 0.0)))
        var found = self.agent_rings.get(agent_id)
        var sp = found.value() if found else UiSpring.create(0.0, SpringConfig.drawer())
        sp.set_target(tgt, snap_if_uninit=False)
        if self.mode == UI_ANIM_INSTANT:
            sp.snap(tgt)
        self.agent_rings[agent_id] = sp
        self.agent_ring_epochs[agent_id] = self.frame_epoch
        return sp.value

    def push_agent_trail(
        mut self,
        entity_id: UInt64,
        x: Float32,
        y: Float32,
        color_rgba8: UInt32,
        max_age_sec: Float32 = 3.5,
    ):
        var sx = _safe_f32(x, 0.0)
        var sy = _safe_f32(y, 0.0)
        var ma = max(Float32(0.2), _safe_f32(max_age_sec, 3.5))
        for i in range(len(self.agent_trails)):
            if self.agent_trails[i].entity_id == entity_id:
                self.agent_trails[i].x = sx
                self.agent_trails[i].y = sy
                self.agent_trails[i].age_sec = 0.0
                self.agent_trails[i].max_age_sec = ma
                self.agent_trails[i].color_rgba8 = color_rgba8
                return
        if len(self.agent_trails) >= MAX_AGENT_TRAILS:
            # Evict oldest trail entry
            var oldest = 0
            for i in range(1, len(self.agent_trails)):
                if self.agent_trails[i].age_sec > self.agent_trails[oldest].age_sec:
                    oldest = i
            self.agent_trails[oldest] = UiAgentTrail(entity_id, sx, sy, 0.0, ma, color_rgba8)
            return
        self.agent_trails.append(UiAgentTrail(entity_id, sx, sy, 0.0, ma, color_rgba8))

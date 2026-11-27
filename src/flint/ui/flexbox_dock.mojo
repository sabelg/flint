"""Two-Pass CSS Flexbox Layout Solver & Recursive Multi-Tab DockSpace (`flint.ui.flexbox_dock`).

Implements:
  - `RectF`: 2D floating-point rectangle with containment, clipping, and overlap queries.
  - `FlexStyle`, `FlexNode`, `FlexTree`: Two-pass Flexbox layout engine (bottom-up intrinsic
    measurement + top-down CSS freeze-loop space distribution) supporting `FLEX_ROW`/`FLEX_COL`,
    `grow`, `shrink`, `basis`, `min/max`, `padding`, `gap`, `ALIGN_START/CENTER/END/STRETCH`,
    and `JUSTIFY_START/CENTER/END/SPACE_BETWEEN` with exact parent-space conservation.
  - `DockSpace`: Recursive binary-split + multi-tab workspace docking (`DOCK_SPLIT_H`,
    `DOCK_SPLIT_V`, `DOCK_TABS`, `DOCK_CENTRAL_VIEWPORT`) with zero-gap tiling, interactive
    splitter dragging, and `save_json()` / `load_json()` workspace persistence.
"""
from std.collections import List
from std.math import abs, isfinite
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote

# Flexbox Axes & Alignment
comptime FLEX_ROW: Int = 0
comptime FLEX_COL: Int = 1

comptime ALIGN_AUTO: Int = -1
comptime ALIGN_START: Int = 0
comptime ALIGN_CENTER: Int = 1
comptime ALIGN_END: Int = 2
comptime ALIGN_STRETCH: Int = 3

comptime JUSTIFY_START: Int = 0
comptime JUSTIFY_CENTER: Int = 1
comptime JUSTIFY_END: Int = 2
comptime JUSTIFY_SPACE_BETWEEN: Int = 3

# DockSpace Node Kinds
comptime DOCK_SPLIT_H: Int = 0          # Left / Right horizontal split
comptime DOCK_SPLIT_V: Int = 1          # Top / Bottom vertical split
comptime DOCK_TABS: Int = 2             # Multi-tab leaf panel
comptime DOCK_CENTRAL_VIEWPORT: Int = 3 # Central pass-through 3D viewport


@always_inline
def _fin_f32_ui(v: Float32, fallback: Float32 = 0.0) -> Float32:
    return v if isfinite(v) else fallback


struct RectF(Copyable, ImplicitlyCopyable, Movable):
    """Axis-aligned 2D UI rectangle `(x, y, w, h)`."""
    var x: Float32
    var y: Float32
    var w: Float32
    var h: Float32

    def __init__(out self):
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0

    def __init__(out self, x: Float32, y: Float32, w: Float32, h: Float32):
        self.x = x if isfinite(x) else Float32(0.0)
        self.y = y if isfinite(y) else Float32(0.0)
        self.w = w if isfinite(w) else Float32(0.0)
        self.h = h if isfinite(h) else Float32(0.0)

    @always_inline
    def right(self) -> Float32:
        return self.x + self.w

    @always_inline
    def bottom(self) -> Float32:
        return self.y + self.h

    @always_inline
    def center_x(self) -> Float32:
        return self.x + self.w * Float32(0.5)

    @always_inline
    def center_y(self) -> Float32:
        return self.y + self.h * Float32(0.5)

    @always_inline
    def area(self) -> Float32:
        return max(Float32(0.0), self.w) * max(Float32(0.0), self.h)

    @always_inline
    def contains(self, px: Float32, py: Float32) -> Bool:
        return px >= self.x and py >= self.y and px <= self.x + self.w and py <= self.y + self.h

    @always_inline
    def overlaps_interior(self, other: RectF, eps: Float32 = 1.0e-3) -> Bool:
        var ox = min(self.right(), other.right()) - max(self.x, other.x)
        var oy = min(self.bottom(), other.bottom()) - max(self.y, other.y)
        return ox > eps and oy > eps

    @always_inline
    def intersect(self, other: RectF) -> RectF:
        var nx0 = max(self.x, other.x)
        var ny0 = max(self.y, other.y)
        var nx1 = min(self.right(), other.right())
        var ny1 = min(self.bottom(), other.bottom())
        return RectF(nx0, ny0, max(Float32(0.0), nx1 - nx0), max(Float32(0.0), ny1 - ny0))

    @always_inline
    def inset(self, l: Float32, t: Float32, r: Float32, b: Float32) -> RectF:
        var sl = l if isfinite(l) else Float32(0.0)
        var st = t if isfinite(t) else Float32(0.0)
        var sr = r if isfinite(r) else Float32(0.0)
        var sb = b if isfinite(b) else Float32(0.0)
        var pl = min(max(Float32(0.0), sl), self.w)
        var pr = min(max(Float32(0.0), sr), max(Float32(0.0), self.w - pl))
        var pt = min(max(Float32(0.0), st), self.h)
        var pb = min(max(Float32(0.0), sb), max(Float32(0.0), self.h - pt))
        return RectF(self.x + pl, self.y + pt, max(Float32(0.0), self.w - pl - pr), max(Float32(0.0), self.h - pt - pb))


def _dock_json_int(doc: JsonDocument, node: Int) -> Tuple[Bool, Int]:
    if node < 0 or node >= len(doc.nodes) or doc.nodes[node].kind != 2:
        return (False, -1)
    var num = doc.nodes[node].number
    if not isfinite(num) or num < -9223372036854774784.0 or num > 9223372036854774784.0:
        return (False, -1)
    var iv = Int(num)
    if Float64(iv) != num:
        return (False, -1)
    return (True, iv)


@fieldwise_init
struct FlexStyle(Copyable, ImplicitlyCopyable, Movable):
    """Declarative Flexbox container & item style."""
    var direction: Int
    var align_items: Int
    var align_self: Int
    var justify_content: Int
    var basis: Float32     # < 0 means auto (use fixed or intrinsic size)
    var grow: Float32
    var shrink: Float32
    var min_w: Float32
    var min_h: Float32
    var max_w: Float32
    var max_h: Float32
    var pad_l: Float32
    var pad_t: Float32
    var pad_r: Float32
    var pad_b: Float32
    var gap: Float32
    var fixed_w: Float32   # <= 0 means auto
    var fixed_h: Float32   # <= 0 means auto

    def __init__(out self, direction: Int = FLEX_COL):
        self.direction = direction
        self.align_items = ALIGN_STRETCH
        self.align_self = ALIGN_AUTO
        self.justify_content = JUSTIFY_START
        self.basis = -1.0
        self.grow = 0.0
        self.shrink = 1.0
        self.min_w = 0.0
        self.min_h = 0.0
        self.max_w = 1.0e9
        self.max_h = 1.0e9
        self.pad_l = 0.0
        self.pad_t = 0.0
        self.pad_r = 0.0
        self.pad_b = 0.0
        self.gap = 0.0
        self.fixed_w = 0.0
        self.fixed_h = 0.0

    @staticmethod
    def row(gap: Float32 = 4.0, pad: Float32 = 0.0, align_items: Int = ALIGN_CENTER) -> Self:
        var s = Self(FLEX_ROW)
        s.gap = max(Float32(0.0), gap) if isfinite(gap) else Float32(0.0)
        var p = max(Float32(0.0), pad) if isfinite(pad) else Float32(0.0)
        s.pad_l = p; s.pad_t = p; s.pad_r = p; s.pad_b = p
        s.align_items = align_items
        return s^

    @staticmethod
    def col(gap: Float32 = 4.0, pad: Float32 = 0.0, align_items: Int = ALIGN_STRETCH) -> Self:
        var s = Self(FLEX_COL)
        s.gap = max(Float32(0.0), gap) if isfinite(gap) else Float32(0.0)
        var p = max(Float32(0.0), pad) if isfinite(pad) else Float32(0.0)
        s.pad_l = p; s.pad_t = p; s.pad_r = p; s.pad_b = p
        s.align_items = align_items
        return s^


def _sanitize_flex_style(st: FlexStyle) -> FlexStyle:
    var out = st
    out.basis = st.basis if isfinite(st.basis) else Float32(-1.0)
    out.grow = max(Float32(0.0), st.grow) if isfinite(st.grow) else Float32(0.0)
    out.shrink = max(Float32(0.0), st.shrink) if isfinite(st.shrink) else Float32(1.0)
    out.min_w = max(Float32(0.0), st.min_w) if isfinite(st.min_w) else Float32(0.0)
    out.min_h = max(Float32(0.0), st.min_h) if isfinite(st.min_h) else Float32(0.0)
    out.max_w = max(out.min_w, st.max_w if isfinite(st.max_w) else Float32(1.0e9))
    out.max_h = max(out.min_h, st.max_h if isfinite(st.max_h) else Float32(1.0e9))
    out.pad_l = max(Float32(0.0), st.pad_l) if isfinite(st.pad_l) else Float32(0.0)
    out.pad_t = max(Float32(0.0), st.pad_t) if isfinite(st.pad_t) else Float32(0.0)
    out.pad_r = max(Float32(0.0), st.pad_r) if isfinite(st.pad_r) else Float32(0.0)
    out.pad_b = max(Float32(0.0), st.pad_b) if isfinite(st.pad_b) else Float32(0.0)
    out.gap = max(Float32(0.0), st.gap) if isfinite(st.gap) else Float32(0.0)
    out.fixed_w = st.fixed_w if isfinite(st.fixed_w) else Float32(0.0)
    out.fixed_h = st.fixed_h if isfinite(st.fixed_h) else Float32(0.0)
    return out


struct FlexNode(Copyable, Movable):
    var parent: Int
    var style: FlexStyle
    var children: List[Int]
    var leaf_w: Float32
    var leaf_h: Float32
    var measured_w: Float32
    var measured_h: Float32
    var rect: RectF

    def __init__(out self, parent: Int, style: FlexStyle, leaf_w: Float32 = 0.0, leaf_h: Float32 = 0.0):
        self.parent = parent
        self.style = _sanitize_flex_style(style)
        self.children = List[Int]()
        self.leaf_w = max(Float32(0.0), leaf_w) if isfinite(leaf_w) else Float32(0.0)
        self.leaf_h = max(Float32(0.0), leaf_h) if isfinite(leaf_h) else Float32(0.0)
        self.measured_w = 0.0
        self.measured_h = 0.0
        self.rect = RectF()


struct FlexTree(Copyable, Movable):
    """Two-pass Flexbox layout tree."""
    var nodes: List[FlexNode]

    def __init__(out self):
        self.nodes = List[FlexNode]()

    def clear(mut self):
        self.nodes.clear()

    def add_node(mut self, parent: Int, style: FlexStyle, intrinsic_w: Float32 = 0.0, intrinsic_h: Float32 = 0.0) -> Int:
        var idx = len(self.nodes)
        self.nodes.append(FlexNode(parent, style, intrinsic_w, intrinsic_h))
        if parent >= 0 and parent < idx:
            self.nodes[parent].children.append(idx)
        return idx

    def _measure_single_node(mut self, idx: Int):
        var n_ch = len(self.nodes[idx].children)
        var st = _sanitize_flex_style(self.nodes[idx].style)
        var w = max(Float32(0.0), self.nodes[idx].leaf_w) if isfinite(self.nodes[idx].leaf_w) else Float32(0.0)
        var h = max(Float32(0.0), self.nodes[idx].leaf_h) if isfinite(self.nodes[idx].leaf_h) else Float32(0.0)

        if n_ch > 0:
            var gap_total = Float32(max(0, n_ch - 1)) * max(Float32(0.0), st.gap)
            if st.direction == FLEX_ROW:
                var sum_w = Float32(0.0)
                var max_h = Float32(0.0)
                for i in range(n_ch):
                    var c = self.nodes[idx].children[i]
                    sum_w += self.nodes[c].measured_w
                    if self.nodes[c].measured_h > max_h:
                        max_h = self.nodes[c].measured_h
                w = max(w, sum_w + gap_total + max(Float32(0.0), st.pad_l) + max(Float32(0.0), st.pad_r))
                h = max(h, max_h + max(Float32(0.0), st.pad_t) + max(Float32(0.0), st.pad_b))
            else:
                var max_w = Float32(0.0)
                var sum_h = Float32(0.0)
                for i in range(n_ch):
                    var c = self.nodes[idx].children[i]
                    if self.nodes[c].measured_w > max_w:
                        max_w = self.nodes[c].measured_w
                    sum_h += self.nodes[c].measured_h
                w = max(w, max_w + max(Float32(0.0), st.pad_l) + max(Float32(0.0), st.pad_r))
                h = max(h, sum_h + gap_total + max(Float32(0.0), st.pad_t) + max(Float32(0.0), st.pad_b))

        if st.fixed_w > Float32(0.0):
            w = st.fixed_w
        if st.fixed_h > Float32(0.0):
            h = st.fixed_h

        var min_w = max(Float32(0.0), st.min_w)
        var max_w = max(min_w, st.max_w)
        var min_h = max(Float32(0.0), st.min_h)
        var max_h = max(min_h, st.max_h)
        self.nodes[idx].measured_w = max(min_w, min(max_w, w))
        self.nodes[idx].measured_h = max(min_h, min(max_h, h))

    def _solve_single_node(mut self, idx: Int):
        var n_ch = len(self.nodes[idx].children)
        if n_ch == 0:
            return

        var st = _sanitize_flex_style(self.nodes[idx].style)
        var outer = self.nodes[idx].rect
        var inner = outer.inset(st.pad_l, st.pad_t, st.pad_r, st.pad_b)
        var is_row = st.direction == FLEX_ROW

        var main_avail = inner.w if is_row else inner.h
        var cross_avail = inner.h if is_row else inner.w

        var req_gap = max(Float32(0.0), st.gap)
        var eff_gap = req_gap
        var total_gap64 = Float64(max(0, n_ch - 1)) * Float64(eff_gap)
        if not isfinite(total_gap64) or total_gap64 > Float64(main_avail):
            eff_gap = main_avail / Float32(max(1, n_ch - 1)) if n_ch > 1 else Float32(0.0)
            total_gap64 = Float64(max(0, n_ch - 1)) * Float64(eff_gap)
        var total_gap = Float32(total_gap64) if isfinite(total_gap64) else Float32(0.0)
        var avail_items = max(Float32(0.0), main_avail - total_gap)
        var avail_items64 = Float64(avail_items)

        var base_m = List[Float32](length=n_ch, fill=0.0)
        var min_m = List[Float32](length=n_ch, fill=0.0)
        var max_m = List[Float32](length=n_ch, fill=0.0)
        var size_m = List[Float32](length=n_ch, fill=0.0)
        var grow_m = List[Float32](length=n_ch, fill=0.0)
        var shrink_m = List[Float32](length=n_ch, fill=1.0)
        var frozen = List[Bool](length=n_ch, fill=False)

        var sum_min64 = 0.0
        var sum_base64 = 0.0

        for i in range(n_ch):
            var c = self.nodes[idx].children[i]
            var cst = _sanitize_flex_style(self.nodes[c].style)
            var mn = max(Float32(0.0), cst.min_w if is_row else cst.min_h)
            var mx = max(mn, cst.max_w if is_row else cst.max_h)
            var pref: Float32
            if cst.basis >= Float32(0.0):
                pref = cst.basis
            elif is_row and cst.fixed_w > Float32(0.0):
                pref = cst.fixed_w
            elif (not is_row) and cst.fixed_h > Float32(0.0):
                pref = cst.fixed_h
            else:
                var mm = self.nodes[c].measured_w if is_row else self.nodes[c].measured_h
                pref = mm if isfinite(mm) else Float32(0.0)
            pref = max(mn, min(mx, pref))
            min_m[i] = mn
            max_m[i] = mx
            base_m[i] = pref
            size_m[i] = pref
            grow_m[i] = cst.grow
            shrink_m[i] = cst.shrink
            sum_min64 += Float64(mn)
            sum_base64 += Float64(pref)

        if sum_min64 > avail_items64 and sum_min64 > 1.0e-6:
            # Overconstrained container smaller than sum of child minimums: scale down proportionally to preserve bounds
            var scale64 = avail_items64 / sum_min64
            for i in range(n_ch):
                var scaled = Float64(min_m[i]) * scale64
                size_m[i] = Float32(scaled) if isfinite(scaled) else Float32(0.0)
        elif abs(avail_items64 - sum_base64) <= 1.0e-5:
            pass
        elif avail_items64 > sum_base64:
            # Positive free space: CSS Flexbox iterative freeze-loop over `grow`
            for i in range(n_ch):
                if grow_m[i] <= Float32(0.0):
                    frozen[i] = True

            for _ in range(n_ch + 1):
                var frozen_sum64 = 0.0
                var unfrozen_base_sum64 = 0.0
                var sum_grow64 = 0.0
                for i in range(n_ch):
                    if frozen[i]:
                        frozen_sum64 += Float64(size_m[i])
                    else:
                        unfrozen_base_sum64 += Float64(base_m[i])
                        sum_grow64 += Float64(max(Float32(0.0), grow_m[i]))

                if not isfinite(sum_grow64) or sum_grow64 <= 1.0e-7:
                    break

                var rem_free64 = max(0.0, avail_items64 - frozen_sum64 - unfrozen_base_sum64)
                var any_clamped = False
                for i in range(n_ch):
                    if frozen[i]:
                        continue
                    var g64 = Float64(max(Float32(0.0), grow_m[i]))
                    var target64 = Float64(base_m[i]) + rem_free64 * (g64 / sum_grow64)
                    var target = Float32(target64) if isfinite(target64) else max_m[i]
                    if target > max_m[i]:
                        size_m[i] = max_m[i]
                        frozen[i] = True
                        any_clamped = True
                    elif target < min_m[i]:
                        size_m[i] = min_m[i]
                        frozen[i] = True
                        any_clamped = True
                    else:
                        size_m[i] = target
                if not any_clamped:
                    break
        else:
            # Deficit space (avail_items < sum_base): CSS Flexbox iterative weighted shrink
            for i in range(n_ch):
                if shrink_m[i] <= Float32(0.0):
                    frozen[i] = True

            for _ in range(n_ch + 1):
                var frozen_sum64 = 0.0
                var unfrozen_base_sum64 = 0.0
                var sum_scaled_shrink64 = 0.0
                for i in range(n_ch):
                    if frozen[i]:
                        frozen_sum64 += Float64(size_m[i])
                    else:
                        unfrozen_base_sum64 += Float64(base_m[i])
                        var sh64 = Float64(max(Float32(0.0), shrink_m[i]))
                        sum_scaled_shrink64 += sh64 * max(1.0, Float64(base_m[i]))

                if not isfinite(sum_scaled_shrink64) or sum_scaled_shrink64 <= 1.0e-7:
                    break

                var rem_avail64 = max(0.0, avail_items64 - frozen_sum64)
                var any_clamped = False
                for i in range(n_ch):
                    if frozen[i]:
                        continue
                    var sh64 = Float64(max(Float32(0.0), shrink_m[i]))
                    var w_sh64 = sh64 * max(1.0, Float64(base_m[i]))
                    var frac64 = w_sh64 / sum_scaled_shrink64
                    var target64 = (Float64(base_m[i]) - unfrozen_base_sum64 * frac64) + rem_avail64 * frac64
                    var target = Float32(target64) if isfinite(target64) else min_m[i]
                    if target < min_m[i]:
                        size_m[i] = min_m[i]
                        frozen[i] = True
                        any_clamped = True
                    else:
                        size_m[i] = target
                if not any_clamped:
                    break

            # If unshrinkable (`shrink == 0`) children still exceed container, clamp excess down to min_m
            var cur_sum64 = 0.0
            for i in range(n_ch):
                cur_sum64 += Float64(size_m[i])
            if cur_sum64 > avail_items64 + 1.0e-4:
                var reducible64 = cur_sum64 - sum_min64
                var excess64 = cur_sum64 - avail_items64
                if reducible64 > 1.0e-6:
                    var frac64 = min(1.0, excess64 / reducible64)
                    for i in range(n_ch):
                        var v64 = Float64(size_m[i]) - (Float64(size_m[i]) - Float64(min_m[i])) * frac64
                        size_m[i] = Float32(v64) if isfinite(v64) else min_m[i]
                elif cur_sum64 > 1.0e-6:
                    var sc64 = avail_items64 / cur_sum64
                    for i in range(n_ch):
                        var v64 = Float64(size_m[i]) * sc64
                        size_m[i] = Float32(v64) if isfinite(v64) else Float32(0.0)

        # Main-axis placement (`justify_content`)
        var used_items64 = 0.0
        for i in range(n_ch):
            used_items64 += Float64(size_m[i])
        var leftover = Float32(max(0.0, avail_items64 - used_items64))
        var start_offset = Float32(0.0)
        var extra_gap = Float32(0.0)
        if st.justify_content == JUSTIFY_CENTER:
            start_offset = leftover * Float32(0.5)
        elif st.justify_content == JUSTIFY_END:
            start_offset = leftover
        elif st.justify_content == JUSTIFY_SPACE_BETWEEN and n_ch > 1:
            extra_gap = leftover / Float32(n_ch - 1)

        var cursor_m = (inner.x if is_row else inner.y) + start_offset
        var step_gap = eff_gap + extra_gap

        for i in range(n_ch):
            var c = self.nodes[idx].children[i]
            var cst = _sanitize_flex_style(self.nodes[c].style)
            var align = cst.align_self if cst.align_self != ALIGN_AUTO else st.align_items

            var mn_c = min(cross_avail, max(Float32(0.0), cst.min_h if is_row else cst.min_w))
            var mx_c = min(cross_avail, max(mn_c, cst.max_h if is_row else cst.max_w))
            var pref_c: Float32
            if is_row and cst.fixed_h > Float32(0.0):
                pref_c = cst.fixed_h
            elif (not is_row) and cst.fixed_w > Float32(0.0):
                pref_c = cst.fixed_w
            else:
                pref_c = self.nodes[c].measured_h if is_row else self.nodes[c].measured_w

            var size_c: Float32
            if align == ALIGN_STRETCH and ((is_row and cst.fixed_h <= Float32(0.0)) or ((not is_row) and cst.fixed_w <= Float32(0.0))):
                size_c = max(mn_c, min(mx_c, cross_avail))
            else:
                size_c = max(mn_c, min(mx_c, pref_c))

            var offset_c = Float32(0.0)
            if align == ALIGN_CENTER:
                offset_c = max(Float32(0.0), (cross_avail - size_c) * Float32(0.5))
            elif align == ALIGN_END:
                offset_c = max(Float32(0.0), cross_avail - size_c)

            var sm = max(Float32(0.0), size_m[i])
            var sc = max(Float32(0.0), size_c)
            if is_row:
                self.nodes[c].rect = RectF(cursor_m, inner.y + offset_c, sm, sc)
            else:
                self.nodes[c].rect = RectF(inner.x + offset_c, cursor_m, sc, sm)

            cursor_m += sm + step_gap

    def solve(mut self, root_rect: RectF, root_idx: Int = 0):
        """Solves the two-pass Flexbox layout iteratively starting at `root_idx` within `root_rect`."""
        if root_idx < 0 or root_idx >= len(self.nodes):
            return
        # Build iterative pre-order traversal of subtree rooted at `root_idx`
        var order = List[Int](capacity=len(self.nodes))
        var stack = List[Int]()
        stack.append(root_idx)
        while len(stack) > 0:
            var cur = stack.pop()
            order.append(cur)
            var n_ch = len(self.nodes[cur].children)
            var ci = n_ch - 1
            while ci >= 0:
                stack.append(self.nodes[cur].children[ci])
                ci -= 1

        # Pass 1: Bottom-up intrinsic measurement (reverse pre-order)
        var oi = len(order) - 1
        while oi >= 0:
            self._measure_single_node(order[oi])
            oi -= 1

        # Pass 2: Top-down CSS freeze-loop space distribution (forward pre-order)
        var rx = root_rect.x if isfinite(root_rect.x) else Float32(0.0)
        var ry = root_rect.y if isfinite(root_rect.y) else Float32(0.0)
        var rw = max(Float32(0.0), root_rect.w) if isfinite(root_rect.w) else Float32(0.0)
        var rh = max(Float32(0.0), root_rect.h) if isfinite(root_rect.h) else Float32(0.0)
        self.nodes[root_idx].rect = RectF(rx, ry, rw, rh)
        for i in range(len(order)):
            self._solve_single_node(order[i])


# ==============================================================================
# Recursive Multi-Tab DockSpace with Draggable Splitters & JSON Persistence
# ==============================================================================

struct DockNode(Copyable, Movable):
    var id: Int
    var kind: Int
    var ratio: Float32
    var child_a: Int
    var child_b: Int
    var tabs: List[String]
    var active_tab: Int
    var rect: RectF

    def __init__(out self, id: Int, kind: Int = DOCK_TABS):
        self.id = id
        self.kind = kind
        self.ratio = 0.5
        self.child_a = -1
        self.child_b = -1
        self.tabs = List[String]()
        self.active_tab = 0
        self.rect = RectF()


struct DockSpace(Copyable, Movable):
    """Recursive binary-split + multi-tab IDE/Engine `DockSpace`."""
    var nodes: List[DockNode]
    var root: Int
    var dragging_node: Int
    var hovered_splitter_node: Int
    var header_height: Float32
    var min_ratio: Float32
    var min_panel_px: Float32  # logical-px floor for either side of a split (on top of `min_ratio`)
    var dock_style_v2: Bool
    var collapsed_nodes: List[Bool]

    def __init__(out self):
        self.nodes = List[DockNode]()
        self.root = -1
        self.dragging_node = -1
        self.hovered_splitter_node = -1
        self.header_height = 24.0
        self.min_ratio = 0.08
        self.min_panel_px = 32.0
        self.dock_style_v2 = False
        self.collapsed_nodes = List[Bool]()

    def clear(mut self):
        self.nodes.clear()
        self.root = -1
        self.dragging_node = -1
        self.hovered_splitter_node = -1
        self.collapsed_nodes.clear()

    def set_dock_style_v2(mut self, enabled: Bool):
        """Enables or disables Obsidian v2 dock tabs, glass splitters, and collapse chevrons."""
        self.dock_style_v2 = enabled

    @always_inline
    def is_node_collapsed(self, node_id: Int) -> Bool:
        if node_id < 0 or node_id >= len(self.collapsed_nodes):
            return False
        return self.collapsed_nodes[node_id]

    def toggle_node_collapsed(mut self, node_id: Int) -> Bool:
        if node_id < 0 or node_id >= len(self.nodes):
            return False
        while len(self.collapsed_nodes) <= node_id:
            self.collapsed_nodes.append(False)
        self.collapsed_nodes[node_id] = not self.collapsed_nodes[node_id]
        return self.collapsed_nodes[node_id]

    def add_tabs_node(mut self, tabs: List[String], active_tab: Int = 0) -> Int:
        var id = len(self.nodes)
        var node = DockNode(id, DOCK_TABS)
        node.tabs = tabs.copy()
        node.active_tab = max(0, min(max(0, len(tabs) - 1), active_tab))
        self.nodes.append(node^)
        if self.root < 0:
            self.root = id
        return id

    def add_central_viewport(mut self, title: String = "Viewport") -> Int:
        var id = len(self.nodes)
        var node = DockNode(id, DOCK_CENTRAL_VIEWPORT)
        node.tabs.append(title)
        node.active_tab = 0
        self.nodes.append(node^)
        if self.root < 0:
            self.root = id
        return id

    def split_node(
        mut self,
        target_id: Int,
        split_kind: Int,
        ratio: Float32,
        new_tabs: List[String],
        place_new_first: Bool = False,
    ) -> Tuple[Int, Int]:
        """Splits an existing leaf node into `(existing, new_leaf)` (or `(new_leaf, existing)`)
        under `target_id` in-place so parent links remain valid. Returns `(child_a, child_b)`."""
        if target_id < 0 or target_id >= len(self.nodes):
            return (-1, -1)
        var moved_existing = DockNode(len(self.nodes), self.nodes[target_id].kind)
        moved_existing.ratio = self.nodes[target_id].ratio
        moved_existing.child_a = self.nodes[target_id].child_a
        moved_existing.child_b = self.nodes[target_id].child_b
        moved_existing.tabs = self.nodes[target_id].tabs.copy()
        moved_existing.active_tab = self.nodes[target_id].active_tab
        var existing_id = len(self.nodes)
        self.nodes.append(moved_existing^)

        var created_id = len(self.nodes)
        var created = DockNode(created_id, DOCK_TABS)
        created.tabs = new_tabs.copy()
        created.active_tab = 0
        self.nodes.append(created^)

        self.nodes[target_id].kind = split_kind
        self.nodes[target_id].ratio = self._clamp_ratio(ratio)
        self.nodes[target_id].tabs.clear()
        self.nodes[target_id].active_tab = 0
        if place_new_first:
            self.nodes[target_id].child_a = created_id
            self.nodes[target_id].child_b = existing_id
        else:
            self.nodes[target_id].child_a = existing_id
            self.nodes[target_id].child_b = created_id
        return (self.nodes[target_id].child_a, self.nodes[target_id].child_b)

    @staticmethod
    def create_studio_5panel() -> Self:
        """Creates a standard 5-panel studio layout:
        Left Outliner | (Top Toolbar / Central 3D Viewport / Bottom Console+Timeline) | Right Inspector."""
        var d = Self()
        var vp = d.add_central_viewport("SceneViewport")
        # Split Left Outliner (20%) | Center+Right (80%)
        var s1 = d.split_node(vp, DOCK_SPLIT_H, 0.20, [String("Outliner"), String("Hierarchy")], place_new_first=True)
        # Split Center (72% of remaining) | Right Inspector (28% of remaining)
        var s2 = d.split_node(s1[1], DOCK_SPLIT_H, 0.72, [String("Inspector"), String("AudioMixer")], place_new_first=False)
        # Split Top Toolbar (10%) / Center+Bottom (90%)
        var s3 = d.split_node(s2[0], DOCK_SPLIT_V, 0.10, [String("Toolbar")], place_new_first=True)
        # Split Central Viewport (70%) / Bottom Console & Assets (30%)
        _ = d.split_node(s3[1], DOCK_SPLIT_V, 0.70, [String("Console"), String("AssetBrowser"), String("CyclesLive")], place_new_first=False)
        return d^

    @always_inline
    def _clamp_ratio(self, ratio: Float32) -> Float32:
        var mr = min(Float32(0.49), max(Float32(0.0), self.min_ratio)) if isfinite(self.min_ratio) else Float32(0.08)
        if not isfinite(ratio):
            return Float32(0.5)
        return max(mr, min(Float32(1.0) - mr, ratio))

    @always_inline
    def _clamp_ratio_px(self, ratio: Float32, extent: Float32) -> Float32:
        """`_clamp_ratio` plus the `min_panel_px` floor for a split `extent` px long (a ratio alone
        lets a panel shrink to a sliver in a small window)."""
        var r = self._clamp_ratio(ratio)
        if not isfinite(extent) or extent <= Float32(0.0) or not isfinite(self.min_panel_px) or self.min_panel_px <= Float32(0.0):
            return r
        var mr = min(Float32(0.49), self.min_panel_px / extent)
        return max(mr, min(Float32(1.0) - mr, r))

    @always_inline
    def _is_collapsed_leaf(self, node_id: Int) -> Bool:
        return (
            self.dock_style_v2
            and node_id >= 0
            and node_id < len(self.nodes)
            and self.nodes[node_id].kind == DOCK_TABS
            and self.is_node_collapsed(node_id)
        )

    def split_has_collapsed_child(self, node_id: Int) -> Bool:
        if node_id < 0 or node_id >= len(self.nodes):
            return False
        return self._is_collapsed_leaf(self.nodes[node_id].child_a) or self._is_collapsed_leaf(self.nodes[node_id].child_b)

    def split_seam(self, node_id: Int) -> Float32:
        """Solved seam position of split `node_id` (x for `DOCK_SPLIT_H`, y for `DOCK_SPLIT_V`): where
        its second child starts, which differs from `ratio` when a side is collapsed or px-floored."""
        if node_id < 0 or node_id >= len(self.nodes):
            return Float32(0.0)
        var k = self.nodes[node_id].kind
        var r = self.nodes[node_id].rect
        var ratio = self.nodes[node_id].ratio
        var cb = self.nodes[node_id].child_b
        var has_b = cb >= 0 and cb < len(self.nodes)
        if k == DOCK_SPLIT_H:
            return self.nodes[cb].rect.x if has_b else r.x + r.w * ratio
        if k == DOCK_SPLIT_V:
            return self.nodes[cb].rect.y if has_b else r.y + r.h * ratio
        return Float32(0.0)

    def solve_layout(mut self, root_rect: RectF):
        if self.root < 0 or self.root >= len(self.nodes):
            return
        var safe_root = RectF(
            root_rect.x if isfinite(root_rect.x) else Float32(0.0),
            root_rect.y if isfinite(root_rect.y) else Float32(0.0),
            max(Float32(0.0), root_rect.w) if isfinite(root_rect.w) else Float32(0.0),
            max(Float32(0.0), root_rect.h) if isfinite(root_rect.h) else Float32(0.0),
        )
        var safe_hh = max(Float32(0.0), self.header_height) if isfinite(self.header_height) else Float32(24.0)
        var visited = List[Bool](length=len(self.nodes), fill=False)
        var stack_id = List[Int]()
        var stack_rect = List[RectF]()
        stack_id.append(self.root)
        stack_rect.append(safe_root)
        while len(stack_id) > 0:
            var node_id = stack_id.pop()
            var rect = stack_rect.pop()
            if node_id < 0 or node_id >= len(self.nodes) or visited[node_id]:
                continue
            visited[node_id] = True
            self.nodes[node_id].rect = rect
            var k = self.nodes[node_id].kind
            if k == DOCK_SPLIT_H:
                var r = self._clamp_ratio(self.nodes[node_id].ratio)
                self.nodes[node_id].ratio = r
                var ca = self.nodes[node_id].child_a
                var cb = self.nodes[node_id].child_b
                var wa = rect.w * self._clamp_ratio_px(r, rect.w)
                if not isfinite(wa):
                    wa = Float32(0.0)
                # A collapsed panel keeps only its header strip; its sibling takes the rest.
                var col_a = self._is_collapsed_leaf(ca)
                var col_b = self._is_collapsed_leaf(cb)
                if col_a and not col_b:
                    wa = min(safe_hh, rect.w)
                elif col_b and not col_a:
                    wa = max(Float32(0.0), rect.w - min(safe_hh, rect.w))
                var wb = max(Float32(0.0), rect.w - wa)
                var xb_f64 = Float64(rect.x) + Float64(wa)
                var xb = Float32(max(-3.4028234e38, min(3.4028234e38, xb_f64)))
                stack_id.append(cb); stack_rect.append(RectF(xb, rect.y, wb, rect.h))
                stack_id.append(ca); stack_rect.append(RectF(rect.x, rect.y, wa, rect.h))
            elif k == DOCK_SPLIT_V:
                var r = self._clamp_ratio(self.nodes[node_id].ratio)
                self.nodes[node_id].ratio = r
                var ha = rect.h * self._clamp_ratio_px(r, rect.h)
                if not isfinite(ha):
                    ha = Float32(0.0)
                var col_a = self._is_collapsed_leaf(self.nodes[node_id].child_a)
                var col_b = self._is_collapsed_leaf(self.nodes[node_id].child_b)
                if col_a and not col_b:
                    ha = min(safe_hh, rect.h)
                elif col_b and not col_a:
                    ha = max(Float32(0.0), rect.h - min(safe_hh, rect.h))
                var hb = max(Float32(0.0), rect.h - ha)
                var yb_f64 = Float64(rect.y) + Float64(ha)
                var yb = Float32(max(-3.4028234e38, min(3.4028234e38, yb_f64)))
                var ca = self.nodes[node_id].child_a
                var cb = self.nodes[node_id].child_b
                stack_id.append(cb); stack_rect.append(RectF(rect.x, yb, rect.w, hb))
                stack_id.append(ca); stack_rect.append(RectF(rect.x, rect.y, rect.w, ha))

    def leaf_rects(self) -> List[RectF]:
        var out = List[RectF]()
        if self.root < 0 or self.root >= len(self.nodes):
            return out^
        var visited = List[Bool](length=len(self.nodes), fill=False)
        var stack = List[Int]()
        stack.append(self.root)
        while len(stack) > 0:
            var node_id = stack.pop()
            if node_id < 0 or node_id >= len(self.nodes) or visited[node_id]:
                continue
            visited[node_id] = True
            var k = self.nodes[node_id].kind
            if k == DOCK_SPLIT_H or k == DOCK_SPLIT_V:
                stack.append(self.nodes[node_id].child_b)
                stack.append(self.nodes[node_id].child_a)
            else:
                out.append(self.nodes[node_id].rect)
        return out^

    def interact_splitters(mut self, mx: Float32, my: Float32, mouse_down: Bool, mouse_pressed: Bool, root_rect: RectF) -> Bool:
        """Handles mouse hover & dragging of DockSpace splitters; re-solves zero-gap layout when dragged."""
        if not mouse_down:
            self.dragging_node = -1

        self.hovered_splitter_node = -1
        if not isfinite(mx) or not isfinite(my):
            return False
        var hit_half = Float32(4.0)

        for i in range(len(self.nodes)):
            var k = self.nodes[i].kind
            var r = self.nodes[i].rect
            if (k == DOCK_SPLIT_H or k == DOCK_SPLIT_V) and self.split_has_collapsed_child(i):
                continue  # a collapsed side has no size to drag
            if k == DOCK_SPLIT_H:
                var seam_x = self.split_seam(i)
                if my >= r.y and my <= r.bottom() and abs(mx - seam_x) <= hit_half:
                    self.hovered_splitter_node = i
                    break
            elif k == DOCK_SPLIT_V:
                var seam_y = self.split_seam(i)
                if mx >= r.x and mx <= r.right() and abs(my - seam_y) <= hit_half:
                    self.hovered_splitter_node = i
                    break

        if mouse_pressed and self.hovered_splitter_node >= 0:
            self.dragging_node = self.hovered_splitter_node

        if self.dragging_node >= 0 and self.dragging_node < len(self.nodes) and mouse_down:
            var dn = self.dragging_node
            var r = self.nodes[dn].rect
            if self.nodes[dn].kind == DOCK_SPLIT_H and r.w > Float32(1.0):
                var nr = (mx - r.x) / r.w
                if not isfinite(nr):
                    return False
                self.nodes[dn].ratio = self._clamp_ratio_px(nr, r.w)
                self.solve_layout(root_rect)
                return True
            elif self.nodes[dn].kind == DOCK_SPLIT_V and r.h > Float32(1.0):
                var nr = (my - r.y) / r.h
                if not isfinite(nr):
                    return False
                self.nodes[dn].ratio = self._clamp_ratio_px(nr, r.h)
                self.solve_layout(root_rect)
                return True
        return False

    def find_panel(self, panel_name: String) -> Tuple[Int, Int, Bool, RectF]:
        """Returns `(node_id, tab_index, is_active_tab, content_rect)` for `panel_name`."""
        var safe_hh = max(Float32(0.0), self.header_height) if isfinite(self.header_height) else Float32(24.0)
        for i in range(len(self.nodes)):
            var k = self.nodes[i].kind
            if k == DOCK_TABS or k == DOCK_CENTRAL_VIEWPORT:
                for t in range(len(self.nodes[i].tabs)):
                    if self.nodes[i].tabs[t] == panel_name:
                        var full = self.nodes[i].rect
                        var fx = full.x if isfinite(full.x) else Float32(0.0)
                        var fy = full.y if isfinite(full.y) else Float32(0.0)
                        var fw = max(Float32(0.0), full.w) if isfinite(full.w) else Float32(0.0)
                        var fh = max(Float32(0.0), full.h) if isfinite(full.h) else Float32(0.0)
                        var hh = Float32(0.0) if k == DOCK_CENTRAL_VIEWPORT else min(safe_hh, fh)
                        if self.dock_style_v2 and k == DOCK_TABS and self.is_node_collapsed(i):
                            return (i, t, False, RectF(fx, fy + hh, fw, Float32(0.0)))
                        var content = RectF(fx, fy + hh, fw, max(Float32(0.0), fh - hh))
                        return (i, t, self.nodes[i].active_tab == t, content)
        return (-1, -1, False, RectF())

    def activate_tab(mut self, panel_name: String) -> Bool:
        for i in range(len(self.nodes)):
            var k = self.nodes[i].kind
            if k == DOCK_TABS or k == DOCK_CENTRAL_VIEWPORT:
                for t in range(len(self.nodes[i].tabs)):
                    if self.nodes[i].tabs[t] == panel_name:
                        self.nodes[i].active_tab = t
                        return True
        return False

    # ---- JSON Workspace Persistence ----

    @staticmethod
    def _fmt_ratio(v: Float32) -> String:
        # 6-decimal fixed-point representation for exact clean JSON round-trips
        if not isfinite(v):
            return "0.500000"
        var clamped = max(Float32(0.0), min(Float32(1.0), v))
        var scaled = Int(Float64(clamped) * 1000000.0 + 0.5)
        if scaled >= 1000000:
            return "1.0"
        var frac = scaled % 1000000
        var digits = String(frac)
        while digits.byte_length() < 6:
            digits = "0" + digits
        return "0." + digits

    def save_json(self) -> String:
        var safe_hh = Int(min(Float32(10000.0), max(Float32(0.0), self.header_height))) if isfinite(self.header_height) else 24
        var out = String('{"version": 1, "root": ') + String(self.root) + ', "header_height": ' + String(safe_hh) + ', "nodes": ['
        for i in range(len(self.nodes)):
            if i > 0:
                out += ", "
            ref n = self.nodes[i]
            out += '{"id": ' + String(n.id)
            out += ', "kind": ' + String(n.kind)
            out += ', "ratio": ' + Self._fmt_ratio(n.ratio)
            out += ', "child_a": ' + String(n.child_a)
            out += ', "child_b": ' + String(n.child_b)
            out += ', "active_tab": ' + String(n.active_tab)
            out += ', "tabs": ['
            for t in range(len(n.tabs)):
                if t > 0:
                    out += ", "
                out += json_quote(n.tabs[t])
            out += "]}"
        out += "]}"
        return out^

    def load_json(mut self, json_str: String) -> Bool:
        var bytes = List[UInt8]()
        for b in json_str.as_bytes():
            bytes.append(b)
        var doc = parse_json(bytes)
        if not doc.valid or doc.root < 0 or doc.root >= len(doc.nodes):
            return False
        if doc.nodes[doc.root].kind != 5:
            return False

        var ver_node = doc.object_get(doc.root, "version")
        if ver_node >= 0:
            var vr = _dock_json_int(doc, ver_node)
            if not vr[0] or vr[1] != 1:
                return False

        var root_node = doc.object_get(doc.root, "root")
        var parsed_root = -1
        if root_node >= 0:
            var rr = _dock_json_int(doc, root_node)
            if not rr[0]:
                return False
            parsed_root = rr[1]

        var nodes_arr = doc.object_get(doc.root, "nodes")
        if nodes_arr < 0 or doc.nodes[nodes_arr].kind != 4:
            return False

        var hh_node = doc.object_get(doc.root, "header_height")
        var parsed_nodes = List[DockNode]()
        var count = doc.array_size(nodes_arr)
        for i in range(count):
            var obj = doc.at(nodes_arr, i)
            if obj < 0 or doc.nodes[obj].kind != 5:
                return False
            var kind_node = doc.object_get(obj, "kind")
            var kr = _dock_json_int(doc, kind_node)
            if not kr[0]:
                return False
            var kind = kr[1]
            if kind < DOCK_SPLIT_H or kind > DOCK_CENTRAL_VIEWPORT:
                return False
            var id = doc.integer(doc.object_get(obj, "id"), i)
            var dn = DockNode(id, kind)
            dn.ratio = self._clamp_ratio(Float32(doc.number(doc.object_get(obj, "ratio"), 0.5)))
            var ca_node = doc.object_get(obj, "child_a")
            if ca_node >= 0:
                var car = _dock_json_int(doc, ca_node)
                if not car[0]:
                    return False
                dn.child_a = car[1]
            else:
                dn.child_a = -1
            var cb_node = doc.object_get(obj, "child_b")
            if cb_node >= 0:
                var cbr = _dock_json_int(doc, cb_node)
                if not cbr[0]:
                    return False
                dn.child_b = cbr[1]
            else:
                dn.child_b = -1
            var raw_active = doc.integer(doc.object_get(obj, "active_tab"), 0)
            var tabs_arr = doc.object_get(obj, "tabs")
            if tabs_arr >= 0:
                if doc.nodes[tabs_arr].kind != 4:
                    return False
                var tc = doc.array_size(tabs_arr)
                for ti in range(tc):
                    dn.tabs.append(doc.text(doc.at(tabs_arr, ti), ""))
            dn.active_tab = max(0, min(max(0, len(dn.tabs) - 1), raw_active))
            parsed_nodes.append(dn^)

        var root_idx = parsed_root if root_node >= 0 else (0 if len(parsed_nodes) > 0 else -1)
        var n_nodes = len(parsed_nodes)
        if n_nodes == 0:
            if root_idx != -1:
                return False
        else:
            if root_idx < 0 or root_idx >= n_nodes:
                return False
            for i in range(n_nodes):
                ref pn = parsed_nodes[i]
                if pn.child_a < -1 or pn.child_a >= n_nodes or pn.child_b < -1 or pn.child_b >= n_nodes:
                    return False
                if pn.kind == DOCK_SPLIT_H or pn.kind == DOCK_SPLIT_V:
                    if pn.child_a < 0 or pn.child_b < 0 or pn.child_a == pn.child_b:
                        return False
                else:
                    if pn.child_a != -1 or pn.child_b != -1:
                        return False
            var color = List[Int](length=n_nodes, fill=0)  # 0=unvisited, 1=visiting, 2=done
            var dfs_node = List[Int]()
            var dfs_exit = List[Bool]()
            for start in range(n_nodes):
                if color[start] != 0:
                    continue
                dfs_node.append(start)
                dfs_exit.append(False)
                while len(dfs_node) > 0:
                    var u = dfs_node.pop()
                    var exiting = dfs_exit.pop()
                    if exiting:
                        color[u] = 2
                        continue
                    if color[u] == 1:
                        return False
                    if color[u] == 2:
                        continue
                    color[u] = 1
                    dfs_node.append(u)
                    dfs_exit.append(True)
                    var ca = parsed_nodes[u].child_a
                    var cb = parsed_nodes[u].child_b
                    if ca >= 0:
                        if color[ca] == 1:
                            return False
                        if color[ca] == 0:
                            dfs_node.append(ca)
                            dfs_exit.append(False)
                    if cb >= 0:
                        if color[cb] == 1:
                            return False
                        if color[cb] == 0:
                            dfs_node.append(cb)
                            dfs_exit.append(False)

        self.clear()
        self.nodes = parsed_nodes^
        self.root = root_idx
        if hh_node >= 0:
            var hh = Float32(doc.number(hh_node, 24.0))
            self.header_height = max(Float32(0.0), hh) if isfinite(hh) else Float32(24.0)
        return True

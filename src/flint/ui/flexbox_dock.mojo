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
from std.math import abs
from flint.assets.json import parse_json
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


@fieldwise_init
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
        var pl = min(max(Float32(0.0), l), self.w)
        var pr = min(max(Float32(0.0), r), max(Float32(0.0), self.w - pl))
        var pt = min(max(Float32(0.0), t), self.h)
        var pb = min(max(Float32(0.0), b), max(Float32(0.0), self.h - pt))
        return RectF(self.x + pl, self.y + pt, max(Float32(0.0), self.w - pl - pr), max(Float32(0.0), self.h - pt - pb))


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
        s.gap = gap
        s.pad_l = pad; s.pad_t = pad; s.pad_r = pad; s.pad_b = pad
        s.align_items = align_items
        return s^

    @staticmethod
    def col(gap: Float32 = 4.0, pad: Float32 = 0.0, align_items: Int = ALIGN_STRETCH) -> Self:
        var s = Self(FLEX_COL)
        s.gap = gap
        s.pad_l = pad; s.pad_t = pad; s.pad_r = pad; s.pad_b = pad
        s.align_items = align_items
        return s^


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
        self.style = style
        self.children = List[Int]()
        self.leaf_w = max(Float32(0.0), leaf_w)
        self.leaf_h = max(Float32(0.0), leaf_h)
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
        var st = self.nodes[idx].style
        var w = self.nodes[idx].leaf_w
        var h = self.nodes[idx].leaf_h

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

        var st = self.nodes[idx].style
        var outer = self.nodes[idx].rect
        var inner = outer.inset(st.pad_l, st.pad_t, st.pad_r, st.pad_b)
        var is_row = st.direction == FLEX_ROW

        var main_avail = inner.w if is_row else inner.h
        var cross_avail = inner.h if is_row else inner.w

        var req_gap = max(Float32(0.0), st.gap)
        var eff_gap = req_gap
        var total_gap = Float32(max(0, n_ch - 1)) * eff_gap
        if total_gap > main_avail:
            eff_gap = main_avail / Float32(max(1, n_ch - 1)) if n_ch > 1 else Float32(0.0)
            total_gap = Float32(max(0, n_ch - 1)) * eff_gap
        var avail_items = max(Float32(0.0), main_avail - total_gap)

        var base_m = List[Float32](length=n_ch, fill=0.0)
        var min_m = List[Float32](length=n_ch, fill=0.0)
        var max_m = List[Float32](length=n_ch, fill=0.0)
        var size_m = List[Float32](length=n_ch, fill=0.0)
        var frozen = List[Bool](length=n_ch, fill=False)

        var sum_min = Float32(0.0)
        var sum_base = Float32(0.0)

        for i in range(n_ch):
            var c = self.nodes[idx].children[i]
            var cst = self.nodes[c].style
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
                pref = self.nodes[c].measured_w if is_row else self.nodes[c].measured_h
            pref = max(mn, min(mx, pref))
            min_m[i] = mn
            max_m[i] = mx
            base_m[i] = pref
            size_m[i] = pref
            sum_min += mn
            sum_base += pref

        if sum_min > avail_items and sum_min > Float32(1.0e-6):
            # Overconstrained container smaller than sum of child minimums: scale down proportionally to preserve bounds
            var scale = avail_items / sum_min
            for i in range(n_ch):
                size_m[i] = min_m[i] * scale
        elif abs(avail_items - sum_base) <= Float32(1.0e-5):
            pass
        elif avail_items > sum_base:
            # Positive free space: CSS Flexbox iterative freeze-loop over `grow`
            for i in range(n_ch):
                if self.nodes[self.nodes[idx].children[i]].style.grow <= Float32(0.0):
                    frozen[i] = True

            for _ in range(n_ch + 1):
                var frozen_sum = Float32(0.0)
                var unfrozen_base_sum = Float32(0.0)
                var sum_grow = Float32(0.0)
                for i in range(n_ch):
                    if frozen[i]:
                        frozen_sum += size_m[i]
                    else:
                        unfrozen_base_sum += base_m[i]
                        sum_grow += max(Float32(0.0), self.nodes[self.nodes[idx].children[i]].style.grow)

                if sum_grow <= Float32(1.0e-7):
                    break

                var rem_free = max(Float32(0.0), avail_items - frozen_sum - unfrozen_base_sum)
                var any_clamped = False
                for i in range(n_ch):
                    if frozen[i]:
                        continue
                    var g = max(Float32(0.0), self.nodes[self.nodes[idx].children[i]].style.grow)
                    var target = base_m[i] + rem_free * (g / sum_grow)
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
                if self.nodes[self.nodes[idx].children[i]].style.shrink <= Float32(0.0):
                    frozen[i] = True

            for _ in range(n_ch + 1):
                var frozen_sum = Float32(0.0)
                var unfrozen_base_sum = Float32(0.0)
                var sum_scaled_shrink = Float32(0.0)
                for i in range(n_ch):
                    if frozen[i]:
                        frozen_sum += size_m[i]
                    else:
                        unfrozen_base_sum += base_m[i]
                        var sh = max(Float32(0.0), self.nodes[self.nodes[idx].children[i]].style.shrink)
                        sum_scaled_shrink += sh * max(Float32(1.0), base_m[i])

                if sum_scaled_shrink <= Float32(1.0e-7):
                    break

                var deficit = max(Float32(0.0), (frozen_sum + unfrozen_base_sum) - avail_items)
                var any_clamped = False
                for i in range(n_ch):
                    if frozen[i]:
                        continue
                    var sh = max(Float32(0.0), self.nodes[self.nodes[idx].children[i]].style.shrink)
                    var w_sh = sh * max(Float32(1.0), base_m[i])
                    var target = base_m[i] - deficit * (w_sh / sum_scaled_shrink)
                    if target < min_m[i]:
                        size_m[i] = min_m[i]
                        frozen[i] = True
                        any_clamped = True
                    else:
                        size_m[i] = target
                if not any_clamped:
                    break

            # If unshrinkable (`shrink == 0`) children still exceed container, clamp excess down to min_m
            var cur_sum = Float32(0.0)
            for i in range(n_ch):
                cur_sum += size_m[i]
            if cur_sum > avail_items + Float32(1.0e-4):
                var reducible = cur_sum - sum_min
                var excess = cur_sum - avail_items
                if reducible > Float32(1.0e-6):
                    var frac = min(Float32(1.0), excess / reducible)
                    for i in range(n_ch):
                        size_m[i] = size_m[i] - (size_m[i] - min_m[i]) * frac
                elif cur_sum > Float32(1.0e-6):
                    var sc = avail_items / cur_sum
                    for i in range(n_ch):
                        size_m[i] *= sc

        # Main-axis placement (`justify_content`)
        var used_items = Float32(0.0)
        for i in range(n_ch):
            used_items += size_m[i]
        var leftover = max(Float32(0.0), avail_items - used_items)
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
            var cst = self.nodes[c].style
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
        self.nodes[root_idx].rect = RectF(
            root_rect.x,
            root_rect.y,
            max(Float32(0.0), root_rect.w),
            max(Float32(0.0), root_rect.h),
        )
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

    def __init__(out self):
        self.nodes = List[DockNode]()
        self.root = -1
        self.dragging_node = -1
        self.hovered_splitter_node = -1
        self.header_height = 24.0
        self.min_ratio = 0.08

    def clear(mut self):
        self.nodes.clear()
        self.root = -1
        self.dragging_node = -1
        self.hovered_splitter_node = -1

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
        self.nodes[target_id].ratio = max(self.min_ratio, min(Float32(1.0) - self.min_ratio, ratio))
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

    def solve_layout(mut self, root_rect: RectF):
        if self.root < 0 or self.root >= len(self.nodes):
            return
        var visited = List[Bool](length=len(self.nodes), fill=False)
        var stack_id = List[Int]()
        var stack_rect = List[RectF]()
        stack_id.append(self.root)
        stack_rect.append(root_rect)
        while len(stack_id) > 0:
            var node_id = stack_id.pop()
            var rect = stack_rect.pop()
            if node_id < 0 or node_id >= len(self.nodes) or visited[node_id]:
                continue
            visited[node_id] = True
            self.nodes[node_id].rect = rect
            var k = self.nodes[node_id].kind
            if k == DOCK_SPLIT_H:
                var r = max(self.min_ratio, min(Float32(1.0) - self.min_ratio, self.nodes[node_id].ratio))
                var wa = rect.w * r
                var wb = rect.w - wa
                var ca = self.nodes[node_id].child_a
                var cb = self.nodes[node_id].child_b
                stack_id.append(cb); stack_rect.append(RectF(rect.x + wa, rect.y, wb, rect.h))
                stack_id.append(ca); stack_rect.append(RectF(rect.x, rect.y, wa, rect.h))
            elif k == DOCK_SPLIT_V:
                var r = max(self.min_ratio, min(Float32(1.0) - self.min_ratio, self.nodes[node_id].ratio))
                var ha = rect.h * r
                var hb = rect.h - ha
                var ca = self.nodes[node_id].child_a
                var cb = self.nodes[node_id].child_b
                stack_id.append(cb); stack_rect.append(RectF(rect.x, rect.y + ha, rect.w, hb))
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
        var hit_half = Float32(4.0)

        for i in range(len(self.nodes)):
            var k = self.nodes[i].kind
            var r = self.nodes[i].rect
            if k == DOCK_SPLIT_H:
                var seam_x = r.x + r.w * self.nodes[i].ratio
                if my >= r.y and my <= r.bottom() and abs(mx - seam_x) <= hit_half:
                    self.hovered_splitter_node = i
                    break
            elif k == DOCK_SPLIT_V:
                var seam_y = r.y + r.h * self.nodes[i].ratio
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
                self.nodes[dn].ratio = max(self.min_ratio, min(Float32(1.0) - self.min_ratio, nr))
                self.solve_layout(root_rect)
                return True
            elif self.nodes[dn].kind == DOCK_SPLIT_V and r.h > Float32(1.0):
                var nr = (my - r.y) / r.h
                self.nodes[dn].ratio = max(self.min_ratio, min(Float32(1.0) - self.min_ratio, nr))
                self.solve_layout(root_rect)
                return True
        return False

    def find_panel(self, panel_name: String) -> Tuple[Int, Int, Bool, RectF]:
        """Returns `(node_id, tab_index, is_active_tab, content_rect)` for `panel_name`."""
        for i in range(len(self.nodes)):
            var k = self.nodes[i].kind
            if k == DOCK_TABS or k == DOCK_CENTRAL_VIEWPORT:
                for t in range(len(self.nodes[i].tabs)):
                    if self.nodes[i].tabs[t] == panel_name:
                        var full = self.nodes[i].rect
                        var hh = Float32(0.0) if k == DOCK_CENTRAL_VIEWPORT else min(self.header_height, full.h)
                        var content = RectF(full.x, full.y + hh, full.w, max(Float32(0.0), full.h - hh))
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
        var out = String('{"version": 1, "root": ') + String(self.root) + ', "header_height": ' + String(Int(self.header_height)) + ', "nodes": ['
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
        if not doc.valid or doc.root < 0:
            return False

        var root_node = doc.object_get(doc.root, "root")
        var nodes_arr = doc.object_get(doc.root, "nodes")
        if nodes_arr < 0:
            return False

        var hh_node = doc.object_get(doc.root, "header_height")
        var parsed_nodes = List[DockNode]()
        var count = doc.array_size(nodes_arr)
        for i in range(count):
            var obj = doc.at(nodes_arr, i)
            if obj < 0:
                return False
            var id = doc.integer(doc.object_get(obj, "id"), i)
            var kind = doc.integer(doc.object_get(obj, "kind"), DOCK_TABS)
            var dn = DockNode(id, kind)
            dn.ratio = Float32(doc.number(doc.object_get(obj, "ratio"), 0.5))
            dn.child_a = doc.integer(doc.object_get(obj, "child_a"), -1)
            dn.child_b = doc.integer(doc.object_get(obj, "child_b"), -1)
            dn.active_tab = max(0, doc.integer(doc.object_get(obj, "active_tab"), 0))
            var tabs_arr = doc.object_get(obj, "tabs")
            if tabs_arr >= 0:
                var tc = doc.array_size(tabs_arr)
                for ti in range(tc):
                    dn.tabs.append(doc.text(doc.at(tabs_arr, ti), ""))
            parsed_nodes.append(dn^)

        var root_idx = doc.integer(root_node, 0 if len(parsed_nodes) > 0 else -1)
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
            self.header_height = Float32(doc.number(hh_node, 24.0))
        return True

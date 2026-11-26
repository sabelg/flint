"""Flint Studio Workspace Preset Manager & LZ4 Agent Checkpoint Log (`flint.studio.studio_workspace`).

Provides two standalone Phase 3 studio infrastructure components (zero dependency on `studio_app.mojo`):
  1. `WorkspacePresetManager`: constructs zero-gap `DockSpace` v2 layouts for all 5 Studio
     workspace presets (`Edit`, `Agent Studio`, `Code + Scene`, `Lookdev`, `Play`), preserving
     canonical panel IDs (`"Hierarchy"`, `"Inspector"`, `"AssetBrowser"`, `"SceneViewport"` /
     `"Viewport"`, `"Agents"`, `"CodeDrawer"`, `"InputMap"`), per-preset user split persistence
     (`save_preset_json` / `reset_preset`), and one-call panel reveal (`ensure_panel_visible`).
  2. `AgentCheckpointLog`: LZ4-compressed `(tx_id, agent_id, turn_id, step_id)` world checkpoint
     store with O(1) lookup by `step_id`, `turn_id`, and `tx_id`, hash-verified 1-click Step &
     Turn rollback payload retrieval, and non-destructive Time-Travel live-tip snapshot storage.
"""
from std.collections import Dict, List
from std.math import max
from flint.core.lz4 import hash_list, lz4_pack, lz4_packed_raw_size, lz4_unpack
from flint.studio.viewport_hud import (
    WORKSPACE_AGENT_STUDIO,
    WORKSPACE_CODE_SCENE,
    WORKSPACE_COUNT,
    WORKSPACE_EDIT,
    WORKSPACE_LOOKDEV,
    WORKSPACE_PLAY,
)
from flint.ui.flexbox_dock import (
    DOCK_CENTRAL_VIEWPORT,
    DOCK_SPLIT_H,
    DOCK_SPLIT_V,
    DOCK_TABS,
    DockSpace,
    RectF,
)

# Canonical Studio Dock Panel IDs
comptime PANEL_SCENE_VIEWPORT: String = "SceneViewport"
comptime PANEL_VIEWPORT: String = "Viewport"
comptime PANEL_HIERARCHY: String = "Hierarchy"
comptime PANEL_OUTLINER: String = "Outliner"
comptime PANEL_INSPECTOR: String = "Inspector"
comptime PANEL_ASSET_BROWSER: String = "AssetBrowser"
comptime PANEL_AGENTS: String = "Agents"
comptime PANEL_CODE_DRAWER: String = "CodeDrawer"
comptime PANEL_INPUT_MAP: String = "InputMap"

comptime DEFAULT_CHECKPOINT_CAP: Int = 256


def workspace_preset_name(preset_id: Int) -> String:
    """Human-readable label for a `WORKSPACE_*` preset identifier."""
    if preset_id == WORKSPACE_AGENT_STUDIO:
        return "Agent Studio"
    if preset_id == WORKSPACE_CODE_SCENE:
        return "Code + Scene"
    if preset_id == WORKSPACE_LOOKDEV:
        return "Lookdev"
    if preset_id == WORKSPACE_PLAY:
        return "Play"
    return "Edit"


def _add_dual_viewport(mut d: DockSpace) -> Int:
    """Adds a central 3D viewport discoverable as both `'SceneViewport'` and `'Viewport'`."""
    var vp = d.add_central_viewport(PANEL_SCENE_VIEWPORT)
    if vp >= 0 and vp < len(d.nodes):
        d.nodes[vp].tabs.append(PANEL_VIEWPORT)
    return vp


def _str_list1(a: String) -> List[String]:
    var out = List[String](capacity=1)
    out.append(a)
    return out^


def _str_list2(a: String, b: String) -> List[String]:
    var out = List[String](capacity=2)
    out.append(a)
    out.append(b)
    return out^


def _str_list3(a: String, b: String, c: String) -> List[String]:
    var out = List[String](capacity=3)
    out.append(a)
    out.append(b)
    out.append(c)
    return out^


def _str_list5(a: String, b: String, c: String, d_s: String, e: String) -> List[String]:
    var out = List[String](capacity=5)
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d_s)
    out.append(e)
    return out^


def build_workspace_dockspace(preset_id: Int, dock_style_v2: Bool = True) -> DockSpace:
    """Builds a zero-gap `DockSpace` layout for the requested `WORKSPACE_*` preset.

    Every preset preserves canonical panel IDs (`Hierarchy`, `Inspector`, `AssetBrowser`,
    `SceneViewport`/`Viewport`, `Agents`, `CodeDrawer`) so layout switching never breaks
    panel queries or tests.
    """
    var d = DockSpace()
    d.header_height = Float32(26.0) if dock_style_v2 else Float32(22.0)
    d.min_ratio = Float32(0.06)
    d.set_dock_style_v2(dock_style_v2)

    var vp = _add_dual_viewport(d)

    if preset_id == WORKSPACE_AGENT_STUDIO:
        # Left Hierarchy (17%) | Center (Viewport 70% / CodeDrawer+Assets 30%) | Right Agents+Inspector (32%)
        var s_left = d.split_node(
            vp, DOCK_SPLIT_H, Float32(0.17), _str_list2(PANEL_HIERARCHY, PANEL_OUTLINER), place_new_first=True
        )
        var s_right = d.split_node(
            s_left[1], DOCK_SPLIT_H, Float32(0.61), _str_list2(PANEL_AGENTS, PANEL_INSPECTOR), place_new_first=False
        )
        _ = d.split_node(
            s_right[0],
            DOCK_SPLIT_V,
            Float32(0.70),
            _str_list3(PANEL_CODE_DRAWER, PANEL_ASSET_BROWSER, PANEL_INPUT_MAP),
            place_new_first=False,
        )
        return d^

    if preset_id == WORKSPACE_CODE_SCENE:
        # Left Hierarchy+Assets (16%) | Center-Left Live 3D Viewport (42%) | Center-Right CodeDrawer (36%) | Far-Right Inspector+Agents (22%)
        var s_left = d.split_node(
            vp, DOCK_SPLIT_H, Float32(0.16), _str_list2(PANEL_HIERARCHY, PANEL_ASSET_BROWSER), place_new_first=True
        )
        var s_code = d.split_node(
            s_left[1], DOCK_SPLIT_H, Float32(0.46), _str_list2(PANEL_CODE_DRAWER, PANEL_INPUT_MAP), place_new_first=False
        )
        _ = d.split_node(
            s_code[1], DOCK_SPLIT_H, Float32(0.62), _str_list2(PANEL_INSPECTOR, PANEL_AGENTS), place_new_first=False
        )
        return d^

    if preset_id == WORKSPACE_LOOKDEV:
        # Slim Left Hierarchy (14%) | Wide 3D Viewport (78% h) / Assets+Code (22% h) | Right Inspector+Agents (24%)
        var s_left = d.split_node(
            vp, DOCK_SPLIT_H, Float32(0.14), _str_list2(PANEL_HIERARCHY, PANEL_OUTLINER), place_new_first=True
        )
        var s_right = d.split_node(
            s_left[1], DOCK_SPLIT_H, Float32(0.72), _str_list2(PANEL_INSPECTOR, PANEL_AGENTS), place_new_first=False
        )
        _ = d.split_node(
            s_right[0],
            DOCK_SPLIT_V,
            Float32(0.78),
            _str_list3(PANEL_ASSET_BROWSER, PANEL_CODE_DRAWER, PANEL_INPUT_MAP),
            place_new_first=False,
        )
        return d^

    if preset_id == WORKSPACE_PLAY:
        # Maximized 3D Viewport (92% height) with a collapsed compact drawer tray at the bottom
        var s_bot = d.split_node(
            vp,
            DOCK_SPLIT_V,
            Float32(0.92),
            _str_list5(PANEL_ASSET_BROWSER, PANEL_HIERARCHY, PANEL_INSPECTOR, PANEL_AGENTS, PANEL_CODE_DRAWER),
            place_new_first=False,
        )
        if dock_style_v2 and s_bot[1] >= 0:
            _ = d.toggle_node_collapsed(s_bot[1])
        return d^

    # Default: WORKSPACE_EDIT
    # Left Hierarchy (18%) | Center (Viewport 74% / AssetBrowser+CodeDrawer+InputMap 26%) | Right Inspector+Agents (24%)
    var s_left = d.split_node(
        vp, DOCK_SPLIT_H, Float32(0.18), _str_list2(PANEL_HIERARCHY, PANEL_OUTLINER), place_new_first=True
    )
    var s_right = d.split_node(
        s_left[1], DOCK_SPLIT_H, Float32(0.70), _str_list2(PANEL_INSPECTOR, PANEL_AGENTS), place_new_first=False
    )
    _ = d.split_node(
        s_right[0],
        DOCK_SPLIT_V,
        Float32(0.74),
        _str_list3(PANEL_ASSET_BROWSER, PANEL_CODE_DRAWER, PANEL_INPUT_MAP),
        place_new_first=False,
    )
    return d^


struct WorkspacePresetManager(Copyable, Movable):
    """Manages active Studio workspace presets, saved per-preset JSON overrides & panel focus."""
    var active_preset: Int
    var prev_non_play_preset: Int
    var saved_overrides_json: List[String]
    var dock_style_v2: Bool

    def __init__(out self, initial_preset: Int = WORKSPACE_EDIT, dock_style_v2: Bool = True):
        self.active_preset = (
            initial_preset
            if (initial_preset >= 0 and initial_preset < WORKSPACE_COUNT)
            else WORKSPACE_EDIT
        )
        self.prev_non_play_preset = (
            self.active_preset if self.active_preset != WORKSPACE_PLAY else WORKSPACE_EDIT
        )
        self.saved_overrides_json = List[String](length=WORKSPACE_COUNT, fill=String(""))
        self.dock_style_v2 = dock_style_v2

    def save_current_layout(mut self, dock: DockSpace):
        """Caches the user's customized splitter ratios for the active preset."""
        if self.active_preset >= 0 and self.active_preset < len(self.saved_overrides_json):
            self.saved_overrides_json[self.active_preset] = dock.save_json()

    def reset_preset(mut self, preset_id: Int):
        """Clears any saved override for `preset_id` so the canonical layout is restored."""
        if preset_id >= 0 and preset_id < len(self.saved_overrides_json):
            self.saved_overrides_json[preset_id] = String("")

    def apply_preset(
        mut self,
        preset_id: Int,
        mut dock: DockSpace,
        root_rect: RectF,
        save_previous: Bool = True,
    ) -> Bool:
        """Switches `dock` to `preset_id`, restoring any cached user override and solving layout."""
        if preset_id < 0 or preset_id >= WORKSPACE_COUNT:
            return False
        if save_previous and preset_id != self.active_preset and len(dock.nodes) > 0:
            self.save_current_layout(dock)
        if self.active_preset != WORKSPACE_PLAY:
            self.prev_non_play_preset = self.active_preset

        var fresh = build_workspace_dockspace(preset_id, self.dock_style_v2)
        var override_json = self.saved_overrides_json[preset_id]
        if override_json.byte_length() > 0:
            var restored = DockSpace()
            restored.header_height = fresh.header_height
            restored.min_ratio = fresh.min_ratio
            restored.set_dock_style_v2(self.dock_style_v2)
            if restored.load_json(override_json) and restored.find_panel(PANEL_SCENE_VIEWPORT)[0] >= 0:
                fresh = restored^

        fresh.solve_layout(root_rect)
        dock = fresh^
        self.active_preset = preset_id
        return True

    def enter_play_workspace(mut self, mut dock: DockSpace, root_rect: RectF) -> Bool:
        """Switches to `WORKSPACE_PLAY` while remembering the prior authoring preset."""
        if self.active_preset != WORKSPACE_PLAY:
            self.prev_non_play_preset = self.active_preset
        return self.apply_preset(WORKSPACE_PLAY, dock, root_rect, save_previous=True)

    def exit_play_workspace(mut self, mut dock: DockSpace, root_rect: RectF) -> Bool:
        """Restores the preset that was active prior to entering `WORKSPACE_PLAY`."""
        var target = (
            self.prev_non_play_preset
            if (self.prev_non_play_preset >= 0 and self.prev_non_play_preset < WORKSPACE_COUNT and self.prev_non_play_preset != WORKSPACE_PLAY)
            else WORKSPACE_EDIT
        )
        return self.apply_preset(target, dock, root_rect, save_previous=False)

    @staticmethod
    def ensure_panel_visible(mut dock: DockSpace, panel_name: String) -> Bool:
        """Activates `panel_name`'s tab and uncollapses its host `DOCK_TABS` node if collapsed."""
        var info = dock.find_panel(panel_name)
        var nid = info[0]
        if nid < 0 or nid >= len(dock.nodes):
            return False
        if dock.nodes[nid].kind == DOCK_TABS:
            if dock.is_node_collapsed(nid):
                _ = dock.toggle_node_collapsed(nid)
            return dock.activate_tab(panel_name)
        return dock.nodes[nid].kind == DOCK_CENTRAL_VIEWPORT


# ============================================================================
# 2. `AgentCheckpointLog`: LZ4 Step/Turn Checkpoints & Time-Travel Snapshots
# ============================================================================

struct AgentCheckpointEntry(Copyable, Movable):
    """Single LZ4-compressed before/after world checkpoint linked to an agent step & transaction."""
    var tx_id: UInt64
    var agent_id: Int
    var turn_id: Int
    var step_id: Int
    var pre_world_lz4: List[UInt8]
    var post_world_lz4: List[UInt8]
    var pre_hash: UInt64
    var post_hash: UInt64
    var raw_pre_bytes: Int
    var raw_post_bytes: Int
    var label: String

    def __init__(
        out self,
        tx_id: UInt64 = 0,
        agent_id: Int = 0,
        turn_id: Int = 0,
        step_id: Int = 0,
        pre_hash: UInt64 = 0,
        post_hash: UInt64 = 0,
        label: String = "",
    ):
        self.tx_id = tx_id
        self.agent_id = agent_id
        self.turn_id = turn_id
        self.step_id = step_id
        self.pre_world_lz4 = List[UInt8]()
        self.post_world_lz4 = List[UInt8]()
        self.pre_hash = pre_hash
        self.post_hash = post_hash
        self.raw_pre_bytes = 0
        self.raw_post_bytes = 0
        self.label = label


struct AgentCheckpointLog(Movable):
    """O(1)-indexed LZ4 checkpoint store powering Step/Turn Rollback & non-destructive Time-Travel."""
    var entries: List[AgentCheckpointEntry]
    var step_to_idx: Dict[Int, Int]
    var tx_to_idx: Dict[UInt64, Int]
    var turn_first_idx: Dict[Int, Int]
    var turn_last_idx: Dict[Int, Int]
    var live_snapshot_lz4: List[UInt8]
    var live_snapshot_hash: UInt64
    var has_live_snapshot: Bool
    var max_entries: Int

    def __init__(out self, max_entries: Int = DEFAULT_CHECKPOINT_CAP):
        self.entries = List[AgentCheckpointEntry]()
        self.step_to_idx = Dict[Int, Int]()
        self.tx_to_idx = Dict[UInt64, Int]()
        self.turn_first_idx = Dict[Int, Int]()
        self.turn_last_idx = Dict[Int, Int]()
        self.live_snapshot_lz4 = List[UInt8]()
        self.live_snapshot_hash = UInt64(0)
        self.has_live_snapshot = False
        self.max_entries = max(4, max_entries)

    def clear(mut self):
        self.entries.clear()
        self.step_to_idx = Dict[Int, Int]()
        self.tx_to_idx = Dict[UInt64, Int]()
        self.turn_first_idx = Dict[Int, Int]()
        self.turn_last_idx = Dict[Int, Int]()
        self.live_snapshot_lz4.clear()
        self.live_snapshot_hash = UInt64(0)
        self.has_live_snapshot = False

    def count(self) -> Int:
        return len(self.entries)

    def _rebuild_indices(mut self):
        self.step_to_idx = Dict[Int, Int]()
        self.tx_to_idx = Dict[UInt64, Int]()
        self.turn_first_idx = Dict[Int, Int]()
        self.turn_last_idx = Dict[Int, Int]()
        for i in range(len(self.entries)):
            ref e = self.entries[i]
            self.step_to_idx[e.step_id] = i
            if e.tx_id != UInt64(0):
                self.tx_to_idx[e.tx_id] = i
            if e.turn_id not in self.turn_first_idx:
                self.turn_first_idx[e.turn_id] = i
            self.turn_last_idx[e.turn_id] = i

    def _append_entry(mut self, var entry: AgentCheckpointEntry) -> Int:
        if len(self.entries) >= self.max_entries:
            _ = self.entries.pop(0)
            self.entries.append(entry^)
            self._rebuild_indices()
            return len(self.entries) - 1
        var idx = len(self.entries)
        var sid = entry.step_id
        var tid = entry.turn_id
        var tx = entry.tx_id
        self.entries.append(entry^)
        self.step_to_idx[sid] = idx
        if tx != UInt64(0):
            self.tx_to_idx[tx] = idx
        if tid not in self.turn_first_idx:
            self.turn_first_idx[tid] = idx
        self.turn_last_idx[tid] = idx
        return idx

    def record_step_raw(
        mut self,
        tx_id: UInt64,
        agent_id: Int,
        turn_id: Int,
        step_id: Int,
        pre_world_raw: List[UInt8],
        post_world_raw: List[UInt8],
        label: String = "",
    ) -> Int:
        """LZ4-packs `pre_world_raw` & `post_world_raw`, records hashes, and indexes in O(1)."""
        var pre_h = hash_list(pre_world_raw)
        var post_h = hash_list(post_world_raw)
        var e = AgentCheckpointEntry(tx_id, agent_id, turn_id, step_id, pre_h, post_h, label)
        e.raw_pre_bytes = len(pre_world_raw)
        e.raw_post_bytes = len(post_world_raw)
        if len(pre_world_raw) > 0:
            e.pre_world_lz4 = lz4_pack(pre_world_raw)
        if len(post_world_raw) > 0:
            e.post_world_lz4 = lz4_pack(post_world_raw)
        return self._append_entry(e^)

    def record_step_packed(
        mut self,
        tx_id: UInt64,
        agent_id: Int,
        turn_id: Int,
        step_id: Int,
        pre_world_lz4: List[UInt8],
        post_world_lz4: List[UInt8],
        pre_hash: UInt64,
        post_hash: UInt64,
        label: String = "",
    ) -> Int:
        """Records already-LZ4-compressed checkpoint payloads without re-compressing."""
        var e = AgentCheckpointEntry(tx_id, agent_id, turn_id, step_id, pre_hash, post_hash, label)
        e.pre_world_lz4 = pre_world_lz4.copy()
        e.post_world_lz4 = post_world_lz4.copy()
        e.raw_pre_bytes = max(0, lz4_packed_raw_size(pre_world_lz4))
        e.raw_post_bytes = max(0, lz4_packed_raw_size(post_world_lz4))
        return self._append_entry(e^)

    def find_by_step(self, step_id: Int) -> Int:
        """O(1) lookup returning entry index for `step_id`, or `-1`."""
        return self.step_to_idx.get(step_id, -1)

    def find_by_tx(self, tx_id: UInt64) -> Int:
        """O(1) lookup returning entry index for `tx_id`, or `-1`."""
        return self.tx_to_idx.get(tx_id, -1)

    def find_turn_first(self, turn_id: Int) -> Int:
        """O(1) lookup returning earliest entry index recorded for `turn_id`, or `-1`."""
        return self.turn_first_idx.get(turn_id, -1)

    def find_turn_last(self, turn_id: Int) -> Int:
        """O(1) lookup returning latest entry index recorded for `turn_id`, or `-1`."""
        return self.turn_last_idx.get(turn_id, -1)

    def _unpack_verified(
        self, packed: List[UInt8], expected_hash: UInt64, expected_raw_len: Int
    ) -> Tuple[Bool, List[UInt8], UInt64]:
        if len(packed) == 0:
            if expected_raw_len == 0:
                var empty = List[UInt8]()
                return (hash_list(empty) == expected_hash or expected_hash == UInt64(0), empty^, expected_hash)
            return (False, List[UInt8](), UInt64(0))
        var raw = lz4_unpack(packed)
        if len(raw) == 0 and expected_raw_len > 0:
            return (False, List[UInt8](), UInt64(0))
        var h = hash_list(raw)
        if expected_hash != UInt64(0) and h != expected_hash:
            return (False, List[UInt8](), UInt64(0))
        return (True, raw^, h)

    def get_step_pre_bytes(self, step_id: Int) -> Tuple[Bool, List[UInt8], UInt64]:
        """Unpacks and hash-verifies the pre-step world bytes for 1-click Step Rollback."""
        var idx = self.find_by_step(step_id)
        if idx < 0 or idx >= len(self.entries):
            return (False, List[UInt8](), UInt64(0))
        ref e = self.entries[idx]
        return self._unpack_verified(e.pre_world_lz4, e.pre_hash, e.raw_pre_bytes)

    def get_step_post_bytes(self, step_id: Int) -> Tuple[Bool, List[UInt8], UInt64]:
        """Unpacks and hash-verifies the post-step world bytes for Time-Travel preview."""
        var idx = self.find_by_step(step_id)
        if idx < 0 or idx >= len(self.entries):
            return (False, List[UInt8](), UInt64(0))
        ref e = self.entries[idx]
        return self._unpack_verified(e.post_world_lz4, e.post_hash, e.raw_post_bytes)

    def get_turn_pre_bytes(self, turn_id: Int) -> Tuple[Bool, List[UInt8], UInt64]:
        """Unpacks and hash-verifies the pre-turn world bytes for 1-click Turn Rollback."""
        var idx = self.find_turn_first(turn_id)
        if idx < 0 or idx >= len(self.entries):
            return (False, List[UInt8](), UInt64(0))
        ref e = self.entries[idx]
        return self._unpack_verified(e.pre_world_lz4, e.pre_hash, e.raw_pre_bytes)

    def save_live_snapshot(mut self, live_world_raw: List[UInt8]) -> UInt64:
        """Compresses and stores the live scene tip before entering non-destructive Time-Travel."""
        self.live_snapshot_hash = hash_list(live_world_raw)
        if len(live_world_raw) > 0:
            self.live_snapshot_lz4 = lz4_pack(live_world_raw)
        else:
            self.live_snapshot_lz4.clear()
        self.has_live_snapshot = True
        return self.live_snapshot_hash

    def restore_live_snapshot(mut self, clear_after: Bool = True) -> Tuple[Bool, List[UInt8], UInt64]:
        """Unpacks and hash-verifies the saved live tip when exiting Time-Travel."""
        if not self.has_live_snapshot:
            return (False, List[UInt8](), UInt64(0))
        var raw_len = max(0, lz4_packed_raw_size(self.live_snapshot_lz4))
        var res = self._unpack_verified(self.live_snapshot_lz4, self.live_snapshot_hash, raw_len)
        if clear_after:
            self.live_snapshot_lz4.clear()
            self.live_snapshot_hash = UInt64(0)
            self.has_live_snapshot = False
        return res^

    def total_compressed_bytes(self) -> Int:
        var total = len(self.live_snapshot_lz4)
        for i in range(len(self.entries)):
            total += len(self.entries[i].pre_world_lz4) + len(self.entries[i].post_world_lz4)
        return total

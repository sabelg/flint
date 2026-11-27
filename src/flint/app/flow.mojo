"""Flint v5 Game-Flow Primitives (`flint.app.flow`).

Provides:
- `GameStateMachine`: explicit validated game-state transitions (`Menu -> Play -> Pause -> Victory`)
- `SaveSlotStore`: traversal-safe, durable JSON save-slot persistence on `flint.platform.os`
- `GameTimer` & `Tween`: NaN/non-positive-`dt` safe one-shot/repeating timers and clamped easings
- `GameFlow`: `Resource` combining `GameStateMachine`, `SaveSlotStore`, and atomic `Scene` transitions.

Saves default to a per-game folder (`player_save_directory_for(game_id)`), so a game that never
names a save directory still keeps progress. `save_versioned` / `load_versioned` wrap the payload
with a format version and timestamp so a later build can migrate old saves, and `save_world` /
`load_world` store a whole `World` snapshot. Scene transitions given an `AssetServer` read through
the mounted `.flpk` packs first (where `flint.sh package` puts scenes), then loose files.
"""
from std.collections import List
from std.math import floor, isfinite, max, min
from flint.app.engine_app import Resource
from flint.assets.json import parse_json
from flint.assets.json_write import json_quote, json_text
from flint.assets.server import AssetServer
from flint.ecs.world import World, comp_hash64
from flint.platform.os import (
    file_size,
    list_directory,
    make_dirs,
    player_save_directory_for,
    read_bytes,
    remove_file,
    unix_time_sec,
    write_file_durable,
)
from flint.scene.scene import ComponentRegistry, Scene

comptime SAVE_ENVELOPE_FORMAT: Int = 1
"""Version of the `save_versioned` wrapper itself (not the game's payload version)."""

comptime TWEEN_LINEAR: Int = 0
comptime TWEEN_EASE_IN: Int = 1
comptime TWEEN_EASE_OUT: Int = 2
comptime TWEEN_EASE_IN_OUT: Int = 3


def _ends_with_ascii(s: String, suffix: String) -> Bool:
    var sb = s.as_bytes()
    var suf = suffix.as_bytes()
    if len(sb) < len(suf):
        return False
    var off = len(sb) - len(suf)
    for i in range(len(suf)):
        if sb[off + i] != suf[i]:
            return False
    return True


def _strip_trailing_slashes(path: String) -> String:
    var b = path.as_bytes()
    var n = len(b)
    while n > 1 and b[n - 1] == 47:  # '/'
        n -= 1
    if n == len(b):
        return path
    var out = List[UInt8](capacity=n)
    for i in range(n):
        out.append(b[i])
    return String(unsafe_from_utf8=out)


struct GameStateMachine(Copyable, Movable):
    """Explicitly gated finite-state machine for game-flow state transitions."""
    var state: String
    var allowed_from: List[String]
    var allowed_to: List[String]

    def __init__(out self, initial_state: String = ""):
        self.state = initial_state
        self.allowed_from = List[String]()
        self.allowed_to = List[String]()

    def __init__(out self, *, copy: Self):
        self.state = copy.state
        self.allowed_from = copy.allowed_from.copy()
        self.allowed_to = copy.allowed_to.copy()

    def allow_transition(mut self, from_state: String, to_state: String):
        for i in range(len(self.allowed_from)):
            if self.allowed_from[i] == from_state and self.allowed_to[i] == to_state:
                return
        self.allowed_from.append(from_state)
        self.allowed_to.append(to_state)

    def can_transition(self, target_state: String) -> Bool:
        for i in range(len(self.allowed_from)):
            if self.allowed_from[i] == self.state and self.allowed_to[i] == target_state:
                return True
        return False

    def transition_to(mut self, target_state: String) -> Bool:
        if not self.can_transition(target_state):
            return False
        self.state = target_state
        return True

    @always_inline
    def current_state(self) -> String:
        return self.state

    def reset(mut self, initial_state: String):
        self.state = initial_state


struct SaveSlotStore(Copyable, Movable):
    """Directory-sandboxed, durable save-slot store rejecting path-traversal slot names."""
    var base_dir: String

    def __init__(out self, base_dir: String = ""):
        self.base_dir = _strip_trailing_slashes(base_dir) if base_dir.byte_length() > 0 else ""

    def __init__(out self, *, copy: Self):
        self.base_dir = copy.base_dir

    @staticmethod
    def _is_valid_slot_name(slot_name: String) -> Bool:
        var b = slot_name.as_bytes()
        var n = len(b)
        if n == 0:
            return False
        if b[0] == 46:  # '.' (blocks ".", "..", and hidden dotfiles)
            return False
        for i in range(n):
            var c = b[i]
            if c == 0 or c == 47 or c == 92 or c == 58:  # '\0', '/', '\\', ':'
                return False
            if c == 46 and i + 1 < n and b[i + 1] == 46:  # ".."
                return False
        return True

    def _slot_path(self, slot_name: String) -> String:
        if self.base_dir.byte_length() == 0 or not Self._is_valid_slot_name(slot_name):
            return ""
        return self.base_dir + "/" + slot_name + ".json"

    def has_slot(self, slot_name: String) -> Bool:
        var p = self._slot_path(slot_name)
        return p.byte_length() > 0 and file_size(p) >= 0

    def save_slot(self, slot_name: String, payload: String) -> Bool:
        var p = self._slot_path(slot_name)
        if p.byte_length() == 0:
            return False
        if not make_dirs(self.base_dir):
            return False
        var raw = List[UInt8](capacity=payload.byte_length())
        for b in payload.as_bytes():
            raw.append(b)
        return write_file_durable(p, raw)

    def load_slot(self, slot_name: String) -> String:
        var p = self._slot_path(slot_name)
        if p.byte_length() == 0 or file_size(p) <= 0:
            return ""
        var raw = read_bytes(p)
        if len(raw) == 0:
            return ""
        return String(unsafe_from_utf8=raw)

    def save_versioned(self, slot_name: String, payload: String, version: Int, label: String = "") -> Bool:
        """Writes `{"flint_save":1,"version":N,"saved_unix":T,"label":...,"payload":...}`. A payload that
        is valid JSON is embedded as JSON; anything else is stored as a JSON string."""
        var body = String(payload.strip())
        var raw = List[UInt8](capacity=body.byte_length())
        for b in body.as_bytes():
            raw.append(b)
        var embedded = json_quote(payload)
        if body.byte_length() > 0:
            var doc = parse_json(raw)
            if doc.valid and doc.root >= 0:
                embedded = json_text(doc, doc.root)
        var env = (
            '{"flint_save":' + String(SAVE_ENVELOPE_FORMAT)
            + ',"version":' + String(version)
            + ',"saved_unix":' + String(unix_time_sec())
            + ',"label":' + json_quote(label)
            + ',"payload":' + embedded + "}"
        )
        return self.save_slot(slot_name, env)

    def load_versioned(self, slot_name: String) -> SavedSlot:
        """Reads a slot written by `save_versioned`. A plain slot written by `save_slot` (no envelope)
        comes back with `version = 0` and the whole file as payload, so older saves still load."""
        var out = SavedSlot()
        var text = self.load_slot(slot_name)
        if text.byte_length() == 0:
            return out^
        out.found = True
        out.payload = text
        var raw = List[UInt8](capacity=text.byte_length())
        for b in text.as_bytes():
            raw.append(b)
        var doc = parse_json(raw)
        if not doc.valid or doc.root < 0 or doc.object_get(doc.root, "flint_save") < 0:
            return out^
        out.version = doc.integer(doc.object_get(doc.root, "version"), 0)
        out.saved_unix = doc.integer(doc.object_get(doc.root, "saved_unix"), 0)
        out.label = doc.text(doc.object_get(doc.root, "label"), "")
        var pn = doc.object_get(doc.root, "payload")
        if pn >= 0:
            out.payload = doc.text(pn, "") if doc.nodes[pn].kind == 3 else json_text(doc, pn)
        return out^

    def save_world(self, slot_name: String, world: World) -> Bool:
        """Stores a full `World` snapshot as `<slot>.world` next to the JSON slots."""
        var p = self._slot_path(slot_name)
        if p.byte_length() == 0 or not make_dirs(self.base_dir):
            return False
        try:
            var blob = world.snapshot_blob()
            return write_file_durable(self.base_dir + "/" + slot_name + ".world", blob)
        except:
            return False

    def load_world(self, slot_name: String, mut world: World) -> Bool:
        var p = self._slot_path(slot_name)
        if p.byte_length() == 0:
            return False
        var wp = self.base_dir + "/" + slot_name + ".world"
        if file_size(wp) <= 0:
            return False
        var blob = read_bytes(wp)
        try:
            world.restore_blob(Span(blob))
            return True
        except:
            return False

    def list_slots(self) -> List[String]:
        var out = List[String]()
        if self.base_dir.byte_length() == 0:
            return out^
        var entries = list_directory(self.base_dir)
        for i in range(len(entries)):
            var name = entries[i]
            if not _ends_with_ascii(name, ".json"):
                continue
            var nb = name.as_bytes()
            var stem_len = len(nb) - 5
            if stem_len <= 0:
                continue
            var stem_bytes = List[UInt8](capacity=stem_len)
            for k in range(stem_len):
                stem_bytes.append(nb[k])
            var stem = String(unsafe_from_utf8=stem_bytes)
            if Self._is_valid_slot_name(stem) and file_size(self.base_dir + "/" + name) >= 0:
                out.append(stem)
        return out^

    def delete_slot(self, slot_name: String) -> Bool:
        var p = self._slot_path(slot_name)
        if p.byte_length() == 0 or file_size(p) < 0:
            return False
        return remove_file(p)


struct SavedSlot(Copyable, Movable):
    """Result of `SaveSlotStore.load_versioned`."""
    var found: Bool
    var version: Int
    var saved_unix: Int
    var label: String
    var payload: String

    def __init__(out self):
        self.found = False
        self.version = 0
        self.saved_unix = 0
        self.label = String("")
        self.payload = String("")


struct GameTimer(Copyable, ImplicitlyCopyable, Movable):
    """One-shot or repeating gameplay timer guarding against NaN and non-positive `dt`."""
    var duration_sec: Float32
    var elapsed_sec: Float32
    var repeating: Bool
    var finished: Bool
    var fire_count: Int

    def __init__(out self, duration_sec: Float32, repeating: Bool = False):
        self.duration_sec = (
            max(Float32(0.0), duration_sec) if isfinite(duration_sec) else Float32(0.0)
        )
        self.elapsed_sec = Float32(0.0)
        self.repeating = repeating
        self.finished = False
        self.fire_count = 0

    def step(mut self, dt: Float32) -> Bool:
        if not isfinite(dt) or dt <= Float32(0.0):
            return False
        if not self.repeating and self.finished:
            return False
        if self.duration_sec <= Float32(0.0):
            self.fire_count += 1
            if not self.repeating:
                self.finished = True
            return True
        var dur64 = Float64(self.duration_sec)
        var total64 = Float64(self.elapsed_sec) + Float64(dt)
        if total64 >= dur64:
            self.fire_count += 1
            if self.repeating:
                var q = floor(total64 / dur64)
                var rem = total64 - q * dur64
                if not isfinite(rem) or rem < 0.0:
                    rem = 0.0
                var rem32 = Float32(rem)
                if not isfinite(rem32) or rem32 < Float32(0.0) or rem32 >= self.duration_sec:
                    rem32 = Float32(0.0)
                self.elapsed_sec = rem32
            else:
                self.elapsed_sec = self.duration_sec
                self.finished = True
            return True
        self.elapsed_sec = Float32(total64)
        return False

    def reset(mut self):
        self.elapsed_sec = Float32(0.0)
        self.finished = False
        self.fire_count = 0

    @always_inline
    def is_finished(self) -> Bool:
        return self.finished


struct Tween(Copyable, ImplicitlyCopyable, Movable):
    """Clamped scalar tween with linear/ease-in/ease-out/ease-in-out curves and NaN/`dt<=0` safety."""
    var start_val: Float32
    var end_val: Float32
    var duration_sec: Float32
    var elapsed_sec: Float32
    var finished: Bool
    var current_val: Float32
    var ease: Int

    def __init__(
        out self,
        start_val: Float32,
        end_val: Float32,
        duration_sec: Float32,
        ease: Int = TWEEN_LINEAR,
    ):
        self.start_val = start_val if isfinite(start_val) else Float32(0.0)
        self.end_val = end_val if isfinite(end_val) else Float32(0.0)
        self.duration_sec = (
            max(Float32(0.0), duration_sec) if isfinite(duration_sec) else Float32(0.0)
        )
        self.elapsed_sec = Float32(0.0)
        self.finished = False
        self.current_val = self.start_val
        self.ease = ease

    def step(mut self, dt: Float32) -> Float32:
        if not isfinite(dt) or dt <= Float32(0.0):
            return self.current_val
        if self.finished:
            return self.current_val
        if self.duration_sec <= Float32(0.0):
            self.elapsed_sec = Float32(0.0)
            self.finished = True
            self.current_val = self.end_val
            return self.current_val

        self.elapsed_sec = min(self.duration_sec, self.elapsed_sec + dt)
        var t = max(Float32(0.0), min(Float32(1.0), self.elapsed_sec / self.duration_sec))
        var eased = t
        if self.ease == TWEEN_EASE_IN:
            eased = t * t
        elif self.ease == TWEEN_EASE_OUT:
            var u = Float32(1.0) - t
            eased = Float32(1.0) - u * u
        elif self.ease == TWEEN_EASE_IN_OUT:
            eased = t * t * (Float32(3.0) - Float32(2.0) * t)

        var val = self.start_val + (self.end_val - self.start_val) * eased
        self.current_val = val if isfinite(val) else self.end_val
        if self.elapsed_sec >= self.duration_sec:
            self.finished = True
            self.current_val = self.end_val
        return self.current_val

    @always_inline
    def is_finished(self) -> Bool:
        return self.finished

    @always_inline
    def value(self) -> Float32:
        return self.current_val

    def reset(mut self):
        self.elapsed_sec = Float32(0.0)
        self.finished = False
        self.current_val = self.start_val


struct GameFlow(Resource, Copyable, Movable):
    """High-level game-flow `Resource` uniting `GameStateMachine`, `SaveSlotStore`, and `Scene` swaps."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.GameFlow")
    var fsm: GameStateMachine
    var saves: SaveSlotStore
    var current_scene: String

    def __init__(out self, save_dir: String = "", initial_state: String = "", game_id: String = ""):
        """`save_dir` empty means the per-game player folder for `game_id` (or `default`), so saves
        always land somewhere real instead of silently failing."""
        self.fsm = GameStateMachine(initial_state)
        var dir = save_dir if save_dir.byte_length() > 0 else player_save_directory_for(game_id)
        self.saves = SaveSlotStore(dir)
        self.current_scene = ""

    def __init__(out self, *, copy: Self):
        self.fsm = copy.fsm.copy()
        self.saves = copy.saves.copy()
        self.current_scene = copy.current_scene

    def transition_to_scene(
        mut self, mut world: World, reg: ComponentRegistry, scene_path: String
    ) -> Bool:
        if scene_path.byte_length() == 0 or file_size(scene_path) <= 0:
            return False
        var sc = Scene()
        try:
            if _ends_with_ascii(scene_path, ".flscnb"):
                _ = sc.load_flscnb_file(scene_path, world, reg, clear_existing=True)
            else:
                _ = sc.load_flscn_file(scene_path, world, reg, clear_existing=True)
            self.current_scene = scene_path
            return True
        except:
            return False

    def transition_to_scene(
        mut self, mut world: World, reg: ComponentRegistry, mut assets: AssetServer, scene_path: String
    ) -> Bool:
        """Loads `scene_path` through `assets`: mounted `.flpk` packs first, then loose files. This is
        the path a packaged build needs, because `flint.sh package` moves scenes into the pack."""
        if scene_path.byte_length() == 0:
            return False
        var data = List[UInt8]()
        var h = assets.load_raw(scene_path)
        var rp = assets.raws.get_ptr(h)
        if rp and len(rp.value()[unsafe_offset=0].bytes) > 0:
            data = rp.value()[unsafe_offset=0].bytes.copy()
        elif file_size(scene_path) > 0:
            data = read_bytes(scene_path)
        if len(data) == 0:
            return False
        var sc = Scene(scene_path)
        try:
            if _ends_with_ascii(scene_path, ".flscnb") or (len(data) > 0 and data[0] != 123 and data[0] != 32 and data[0] != 10):
                _ = sc.load_flscnb_bytes(world, reg, data, clear_existing=True)
            else:
                _ = sc.load_flscn_bytes(world, reg, data, clear_existing=True)
            self.current_scene = scene_path
            return True
        except:
            return False

    def save_slot(self, slot_idx: Int, payload: String) -> Bool:
        if slot_idx < 0:
            return False
        return self.saves.save_slot("slot_" + String(slot_idx), payload)

    def load_slot(self, slot_idx: Int) -> String:
        if slot_idx < 0:
            return ""
        return self.saves.load_slot("slot_" + String(slot_idx))

    def save_slot(self, slot_name: String, payload: String) -> Bool:
        return self.saves.save_slot(slot_name, payload)

    def load_slot(self, slot_name: String) -> String:
        return self.saves.load_slot(slot_name)

    def delete_slot(self, slot_name: String) -> Bool:
        return self.saves.delete_slot(slot_name)

    def list_slots(self) -> List[String]:
        return self.saves.list_slots()

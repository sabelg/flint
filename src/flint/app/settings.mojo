"""Player settings for Flint games (`flint.app.settings`).

`GameSettings` is a `Resource` holding what a settings menu edits: resolution, fullscreen, vsync,
render scale, quality preset, per-bus volumes, mouse sensitivity / invert, subtitles and language.
It persists as JSON in the game's own player folder (`player_save_directory_for(game_id)/
settings.json`), next to the key bindings (`bindings.json`, written from the live `InputMap`), so
rebinding and volume survive a restart.

- `GameSettings.load(game_id)` / `save()`: read / write the file; unknown or bad fields fall back to
  defaults, so an old or hand-edited file never stops the game from starting.
- `apply_audio(server)`: pushes the volumes to the mixer buses.
- `apply_window(window_res)`: matches fullscreen to the setting.
- `save_bindings(input_map)` / `load_bindings(input_map)`: persist `InputMap` rebinding.
"""
from std.math import isfinite, max, min
from flint.app.engine_app import Resource
from flint.app.window_plugin import WindowStateResource
from flint.assets.json import parse_json
from flint.assets.json_write import json_quote
from flint.audio.server import AudioServer, BUS_AMBIENCE, BUS_MASTER, BUS_MUSIC, BUS_SFX, BUS_UI, BUS_VOICE
from flint.ecs.world import comp_hash64
from flint.input.input_map import InputMap
from flint.platform.os import file_size, make_dirs, player_save_directory_for, read_bytes, write_file_durable

comptime SETTINGS_FORMAT: Int = 1
comptime QUALITY_LOW: Int = 0
comptime QUALITY_MEDIUM: Int = 1
comptime QUALITY_HIGH: Int = 2
comptime QUALITY_ULTRA: Int = 3


def _clamp01(v: Float64, fallback: Float32) -> Float32:
    if not isfinite(v):
        return fallback
    return Float32(max(0.0, min(1.0, v)))


def _bytes(text: String) -> List[UInt8]:
    var out = List[UInt8](capacity=text.byte_length())
    for b in text.as_bytes():
        out.append(b)
    return out^


struct GameSettings(Resource, Copyable, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.GameSettings")
    var game_id: String
    var width: Int
    var height: Int
    var fullscreen: Bool
    var vsync: Bool
    var render_scale: Float32
    """Internal render resolution as a fraction of the window (0.5..1.0)."""
    var quality: Int
    var master_volume: Float32
    var music_volume: Float32
    var sfx_volume: Float32
    var voice_volume: Float32
    var ambience_volume: Float32
    var ui_volume: Float32
    var mouse_sensitivity: Float32
    var invert_y: Bool
    var subtitles: Bool
    var caption_size: Int
    var language: String

    def __init__(out self, game_id: String = ""):
        self.game_id = game_id
        self.width = 1280
        self.height = 720
        self.fullscreen = False
        self.vsync = True
        self.render_scale = Float32(1.0)
        self.quality = QUALITY_HIGH
        self.master_volume = Float32(1.0)
        self.music_volume = Float32(0.8)
        self.sfx_volume = Float32(1.0)
        self.voice_volume = Float32(1.0)
        self.ambience_volume = Float32(0.8)
        self.ui_volume = Float32(0.9)
        self.mouse_sensitivity = Float32(1.0)
        self.invert_y = False
        self.subtitles = True
        self.caption_size = 22
        self.language = String("en")

    def folder(self) -> String:
        return player_save_directory_for(self.game_id)

    def path(self) -> String:
        return self.folder() + "/settings.json"

    def bindings_path(self) -> String:
        return self.folder() + "/bindings.json"

    def to_json(self) -> String:
        return (
            '{"format":' + String(SETTINGS_FORMAT)
            + ',"width":' + String(self.width)
            + ',"height":' + String(self.height)
            + ',"fullscreen":' + ("true" if self.fullscreen else "false")
            + ',"vsync":' + ("true" if self.vsync else "false")
            + ',"render_scale":' + String(self.render_scale)
            + ',"quality":' + String(self.quality)
            + ',"master_volume":' + String(self.master_volume)
            + ',"music_volume":' + String(self.music_volume)
            + ',"sfx_volume":' + String(self.sfx_volume)
            + ',"voice_volume":' + String(self.voice_volume)
            + ',"ambience_volume":' + String(self.ambience_volume)
            + ',"ui_volume":' + String(self.ui_volume)
            + ',"mouse_sensitivity":' + String(self.mouse_sensitivity)
            + ',"invert_y":' + ("true" if self.invert_y else "false")
            + ',"subtitles":' + ("true" if self.subtitles else "false")
            + ',"caption_size":' + String(self.caption_size)
            + ',"language":' + json_quote(self.language)
            + "}"
        )

    def apply_json(mut self, text: String) -> Bool:
        """Overlays the fields present and valid in `text`; everything else keeps its current value."""
        var doc = parse_json(_bytes(text))
        if not doc.valid or doc.root < 0:
            return False
        var r = doc.root
        var w = doc.integer(doc.object_get(r, "width"), self.width)
        var h = doc.integer(doc.object_get(r, "height"), self.height)
        if w >= 320 and w <= 16384 and h >= 200 and h <= 16384:
            self.width = w
            self.height = h
        self.fullscreen = doc.boolean(doc.object_get(r, "fullscreen"), self.fullscreen)
        self.vsync = doc.boolean(doc.object_get(r, "vsync"), self.vsync)
        var rs = doc.number(doc.object_get(r, "render_scale"), Float64(self.render_scale))
        if isfinite(rs):
            self.render_scale = Float32(max(0.5, min(1.0, rs)))
        var q = doc.integer(doc.object_get(r, "quality"), self.quality)
        if q >= QUALITY_LOW and q <= QUALITY_ULTRA:
            self.quality = q
        self.master_volume = _clamp01(doc.number(doc.object_get(r, "master_volume"), Float64(self.master_volume)), self.master_volume)
        self.music_volume = _clamp01(doc.number(doc.object_get(r, "music_volume"), Float64(self.music_volume)), self.music_volume)
        self.sfx_volume = _clamp01(doc.number(doc.object_get(r, "sfx_volume"), Float64(self.sfx_volume)), self.sfx_volume)
        self.voice_volume = _clamp01(doc.number(doc.object_get(r, "voice_volume"), Float64(self.voice_volume)), self.voice_volume)
        self.ambience_volume = _clamp01(doc.number(doc.object_get(r, "ambience_volume"), Float64(self.ambience_volume)), self.ambience_volume)
        self.ui_volume = _clamp01(doc.number(doc.object_get(r, "ui_volume"), Float64(self.ui_volume)), self.ui_volume)
        var ms = doc.number(doc.object_get(r, "mouse_sensitivity"), Float64(self.mouse_sensitivity))
        if isfinite(ms):
            self.mouse_sensitivity = Float32(max(0.05, min(10.0, ms)))
        self.invert_y = doc.boolean(doc.object_get(r, "invert_y"), self.invert_y)
        self.subtitles = doc.boolean(doc.object_get(r, "subtitles"), self.subtitles)
        self.caption_size = max(12, min(64, doc.integer(doc.object_get(r, "caption_size"), self.caption_size)))
        var lang = doc.text(doc.object_get(r, "language"), self.language)
        if lang.byte_length() > 0 and lang.byte_length() <= 16:
            self.language = lang
        return True

    @staticmethod
    def load(game_id: String) -> GameSettings:
        """Defaults overlaid with the saved file (if any)."""
        var s = GameSettings(game_id)
        var p = s.path()
        if file_size(p) > 0:
            var raw = read_bytes(p)
            _ = s.apply_json(String(unsafe_from_utf8=raw))
        return s^

    def save(self) -> Bool:
        if not make_dirs(self.folder()):
            return False
        var data = _bytes(self.to_json())
        return write_file_durable(self.path(), data)

    def bus_gain(self, bus_id: Int) -> Float32:
        if bus_id == BUS_MASTER:
            return self.master_volume
        if bus_id == BUS_MUSIC:
            return self.music_volume
        if bus_id == BUS_SFX:
            return self.sfx_volume
        if bus_id == BUS_VOICE:
            return self.voice_volume
        if bus_id == BUS_AMBIENCE:
            return self.ambience_volume
        if bus_id == BUS_UI:
            return self.ui_volume
        return Float32(1.0)

    def apply_audio(self, server: AudioServer) -> Int:
        """Sets every mixer bus gain from the volumes; returns how many buses accepted it."""
        var n = 0
        for bus in [BUS_MASTER, BUS_SFX, BUS_MUSIC, BUS_VOICE, BUS_AMBIENCE, BUS_UI]:
            if server.set_bus_gain(bus, self.bus_gain(bus)):
                n += 1
        return n

    def apply_window(self, mut win_res: WindowStateResource):
        if len(win_res.windows) == 0:
            return
        if win_res.windows[0].fullscreen != self.fullscreen:
            win_res.windows[0].toggle_fullscreen()

    def save_bindings(self, input_map: InputMap) -> Bool:
        if not make_dirs(self.folder()):
            return False
        var data = _bytes(input_map.save_json())
        return write_file_durable(self.bindings_path(), data)

    def load_bindings(self, mut input_map: InputMap) -> Bool:
        var p = self.bindings_path()
        if file_size(p) <= 0:
            return False
        var raw = read_bytes(p)
        return input_map.load_json(String(unsafe_from_utf8=raw))

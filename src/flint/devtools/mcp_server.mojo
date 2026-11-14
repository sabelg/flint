"""Game-agnostic Flint MCP & LiveLink introspection handler.

Allows AI agents and automated tools to query any running Flint game's build stamp,
tick counter, memory budgets, perceptual FLIP image diff, and screenshots with
zero game-specific coupling.
"""
from flint.assets.image import decode_png
from flint.assets.json import parse_json
from flint.assets.json_write import json_quote
from flint.core.build_info import build_stamp
from flint.core.memory import W_CPU_LIVE, W_VRAM_LIVE, mem_total
from flint.platform.os import read_bytes
from flint.render.flip import flip_error_map, flip_mean


struct FlintMcpInfo(Copyable, Movable):
    var app_name: String
    var tick: Int
    var backend: String
    var custom_state_json: String

    def __init__(
        out self,
        app_name: String = "flint-app",
        tick: Int = 0,
        backend: String = "SIMD-CPU",
        custom_state_json: String = "{}",
    ):
        self.app_name = app_name
        self.tick = tick
        self.backend = backend
        self.custom_state_json = custom_state_json


def handle_flint_mcp_query(method: String, args_json: String, info: FlintMcpInfo) -> String:
    """Pure JSON responder callable over LiveLink Unix sockets, stdio MCP, or unit tests."""
    if method == "flint_hello":
        return (
            '{"engine":"flint","version":"0.1.0","app":'
            + json_quote(info.app_name)
            + ',"build":'
            + json_quote(build_stamp())
            + ',"backend":'
            + json_quote(info.backend)
            + ',"tick":'
            + String(info.tick)
            + "}"
        )
    if method == "flint_memory":
        return (
            '{"cpu_live_bytes":'
            + String(mem_total(W_CPU_LIVE))
            + ',"vram_live_bytes":'
            + String(mem_total(W_VRAM_LIVE))
            + "}"
        )
    if method == "flint_state":
        return (
            '{"tick":'
            + String(info.tick)
            + ',"state":'
            + info.custom_state_json
            + "}"
        )
    if method == "flint_flip_diff":
        var raw = List[UInt8]()
        for b in args_json.as_bytes():
            raw.append(b)
        var doc = parse_json(raw)
        if not doc.valid:
            return '{"ok":false,"error":"invalid json"}'
        var path_a = doc.text(doc.object_get(doc.root, "a"))
        var path_b = doc.text(doc.object_get(doc.root, "b"))
        var img_a = decode_png(read_bytes(path_a), 4096)
        var img_b = decode_png(read_bytes(path_b), 4096)
        if not img_a.valid or not img_b.valid or img_a.width != img_b.width or img_a.height != img_b.height:
            return '{"ok":false,"error":"mismatched or unreadable pngs"}'
        var err = flip_mean(flip_error_map(img_a.pixels, img_b.pixels, img_a.width, img_a.height))
        return '{"ok":true,"flip_mean":' + String(err) + "}"
    return '{"ok":false,"error":' + json_quote("unknown method: " + method) + "}"

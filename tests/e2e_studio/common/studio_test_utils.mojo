"""Opaque-box Mojo 1.2 helpers for /home/user/flint/tests/e2e_studio/."""

from std.collections import List
from std.ffi import external_call
from std.math import sin
from flint.core.math import Vec3
from flint.assets.model import NativeModel, NativeMesh
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.glb_writer import write_native_glb
from flint.assets.image import write_png
from flint.render.image_diff import compare_with_png
from flint.platform.os import (
    env_now,
    make_dirs,
    parent_directory,
    read_bytes as os_read_bytes,
    write_bytes as os_write_bytes,
)


comptime STUDIO_GOLDEN_TOLERANCE: Int = 4
comptime STUDIO_GOLDEN_SPREAD_PER_MILLION: Int = 1000
comptime STUDIO_GOLDEN_FLIP_LIMIT: Float64 = 0.02


def _ensure_parent_dir(path: String):
    var parent = parent_directory(path)
    if parent.byte_length() > 0 and parent != ".":
        _ = make_dirs(parent)


def _is_safe_tag(tag: String) -> Bool:
    var b = tag.as_bytes()
    if len(b) == 0:
        return False
    for i in range(len(b)):
        var c = b[i]
        var ok = (
            (c >= 48 and c <= 57)     # 0-9
            or (c >= 65 and c <= 90)  # A-Z
            or (c >= 97 and c <= 122) # a-z
            or c == 95                # _
            or c == 45                # -
        )
        if not ok:
            return False
    return True


def _is_safe_tmp_path(path: String) -> Bool:
    var b = path.as_bytes()
    if len(b) == 0:
        return False
    for i in range(len(b)):
        var c = b[i]
        var ok = (
            (c >= 48 and c <= 57)     # 0-9
            or (c >= 65 and c <= 90)  # A-Z
            or (c >= 97 and c <= 122) # a-z
            or c == 95                # _
            or c == 45                # -
            or c == 46                # .
            or c == 47                # /
        )
        if not ok:
            return False
        if i + 1 < len(b) and c == 46 and b[i + 1] == 46:
            return False
    return True


def _resolve_tmp_base() -> String:
    var base = env_now("FLINT_STUDIO_E2E_TMP")
    while base.byte_length() > 1 and base.as_bytes()[base.byte_length() - 1] == 47:
        var trimmed = String(base[byte=0 : base.byte_length() - 1])
        base = trimmed^
    if base.byte_length() > 0 and _is_safe_tmp_path(base):
        return base
    return "/tmp"


def make_tmp_dir(tag: String) raises -> String:
    if not _is_safe_tag(tag):
        raise Error("Unsafe temp directory tag: " + tag)
    var pid = Int(external_call["getpid", Int32]())
    var base = _resolve_tmp_base()
    var path = base + "/flint_e2e_studio_" + String(pid) + "_" + tag
    if not _is_safe_tmp_path(path):
        raise Error("Unsafe temp directory path: " + path)
    var rm_cmd = "rm -rf '" + path + "'"
    _ = external_call["system", Int32](rm_cmd.as_c_string_span())
    if not make_dirs(path):
        raise Error("Failed to create temp directory: " + path)
    return path


def rm_rf_tmp(path: String):
    if env_now("FLINT_STUDIO_E2E_KEEP_TMP") == "1":
        return
    if not _is_safe_tmp_path(path):
        return
    var base = _resolve_tmp_base()
    var allowed_default = path.startswith("/tmp/flint_e2e_studio_")
    var allowed_env = path.startswith(base + "/flint_e2e_studio_")
    if not allowed_default and not allowed_env:
        return
    var cmd = "rm -rf '" + path + "'"
    _ = external_call["system", Int32](cmd.as_c_string_span())


def write_bytes(path: String, data: List[UInt8]) -> Bool:
    _ensure_parent_dir(path)
    var buf = data.copy()
    return os_write_bytes(path, buf)


def write_text(path: String, text: String) -> Bool:
    var buf = List[UInt8](capacity=text.byte_length())
    for b in text.as_bytes():
        buf.append(b)
    return write_bytes(path, buf)


def read_bytes(path: String) -> List[UInt8]:
    return os_read_bytes(path)


def read_text(path: String) -> String:
    var raw = os_read_bytes(path)
    if len(raw) == 0:
        return String()
    return String(unsafe_from_utf8=raw)


def write_minimal_glb(path: String) -> Bool:
    _ensure_parent_dir(path)
    var mb = MeshBuilder(
        base_color=0xFFE0A040,
        roughness=0.4,
        metallic=0.2,
    )
    mb.add_beveled_box(Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5), 0.05)
    var mesh = mb^.finish()
    var prims = List[NativeMesh]()
    prims.append(mesh^)
    var model = NativeModel()
    _ = append_node_meshes(model, "studio_test_box", prims)
    return write_native_glb(path, model)


def write_minimal_wav(
    path: String,
    duration_sec: Float32 = 0.05,
    sample_rate: Int = 48000,
    channels: Int = 2,
) -> Bool:
    var safe_rate = max(1, sample_rate)
    var safe_ch = 1 if channels <= 1 else 2
    var num_samples = max(1, Int(Float32(safe_rate) * duration_sec))
    var bits_per_sample = 16
    var block_align = safe_ch * (bits_per_sample // 8)
    var byte_rate = safe_rate * block_align
    var data_size = num_samples * block_align
    var total_size = 36 + data_size

    var out = List[UInt8](capacity=44 + data_size)
    # "RIFF"
    out.append(82); out.append(73); out.append(70); out.append(70)
    out.append(UInt8(total_size & 0xFF))
    out.append(UInt8((total_size >> 8) & 0xFF))
    out.append(UInt8((total_size >> 16) & 0xFF))
    out.append(UInt8((total_size >> 24) & 0xFF))
    # "WAVE"
    out.append(87); out.append(65); out.append(86); out.append(69)
    # "fmt "
    out.append(102); out.append(109); out.append(116); out.append(32)
    out.append(16); out.append(0); out.append(0); out.append(0)
    out.append(1); out.append(0)
    out.append(UInt8(safe_ch)); out.append(0)
    out.append(UInt8(safe_rate & 0xFF))
    out.append(UInt8((safe_rate >> 8) & 0xFF))
    out.append(UInt8((safe_rate >> 16) & 0xFF))
    out.append(UInt8((safe_rate >> 24) & 0xFF))
    out.append(UInt8(byte_rate & 0xFF))
    out.append(UInt8((byte_rate >> 8) & 0xFF))
    out.append(UInt8((byte_rate >> 16) & 0xFF))
    out.append(UInt8((byte_rate >> 24) & 0xFF))
    out.append(UInt8(block_align)); out.append(0)
    out.append(UInt8(bits_per_sample)); out.append(0)
    # "data"
    out.append(100); out.append(97); out.append(116); out.append(97)
    out.append(UInt8(data_size & 0xFF))
    out.append(UInt8((data_size >> 8) & 0xFF))
    out.append(UInt8((data_size >> 16) & 0xFF))
    out.append(UInt8((data_size >> 24) & 0xFF))

    var two_pi = 6.283185307179586
    var dt = Float32(two_pi * 440.0 / Float64(safe_rate))
    for i in range(num_samples):
        var s = sin(Float32(i) * dt) * 0.4
        var val16 = max(-32768, min(32767, Int(s * 32000.0)))
        var uval = UInt16(val16 & 0xFFFF)
        var lo = UInt8(uval & 0xFF)
        var hi = UInt8((uval >> 8) & 0xFF)
        for _ in range(safe_ch):
            out.append(lo)
            out.append(hi)
    return write_bytes(path, out)


def write_minimal_png(
    path: String,
    width: Int = 16,
    height: Int = 16,
    color_argb: UInt32 = 0xFF336699,
    stamped: Bool = False,
) -> Bool:
    _ensure_parent_dir(path)
    var w = max(1, width)
    var h = max(1, height)
    var px = List[UInt32](length=w * h, fill=color_argb)
    return write_png(path, px, w, h, stamped=stamped)


def verify_studio_golden(
    actual_pixels: List[UInt32],
    width: Int,
    height: Int,
    reference_png_path: String,
    tolerance: Int = STUDIO_GOLDEN_TOLERANCE,
    spread_per_million: Int = STUDIO_GOLDEN_SPREAD_PER_MILLION,
    flip_limit: Float64 = STUDIO_GOLDEN_FLIP_LIMIT,
) -> Tuple[Bool, String]:
    var diff = compare_with_png(actual_pixels, width, height, reference_png_path, tolerance)
    return (diff.passes(spread_per_million, flip_limit), diff.summary())


struct StudioSuiteRunner(Movable):
    var suite_name: String
    var passed: Int
    var failed: Int
    var total: Int

    def __init__(out self, suite_name: String = "e2e_studio"):
        self.suite_name = suite_name
        self.passed = 0
        self.failed = 0
        self.total = 0

    def pass_case(mut self, name: String):
        self.passed += 1
        self.total += 1
        print("[PASS] " + name)

    def fail_case(mut self, name: String, reason: String = "assertion failed"):
        self.failed += 1
        self.total += 1
        print("[FAIL] " + name + ": " + reason)

    def check(mut self, name: String, cond: Bool, reason: String = "assertion failed") -> Bool:
        if cond:
            self.pass_case(name)
        else:
            self.fail_case(name, reason)
        return cond

    def finish(self) raises:
        print("[SUMMARY] " + String(self.passed) + "/" + String(self.total) + " passed")
        if self.failed > 0 or self.total == 0:
            raise Error(self.suite_name + " failed")

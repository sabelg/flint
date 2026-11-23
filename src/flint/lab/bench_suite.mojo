"""Engine Lab Benchmark & Determinism Verification Suite (`flint.lab.bench_suite`).

Provides standardized p50/p99 latency percentile aggregation, dual-run deterministic
state-hash gating, golden-frame perceptual FLIP verification with automatic Magma
heatmap PNG export on failure, and `perf_history.mojo` CSV reporting.
"""
from std.collections import List
from std.math import isinf, isnan
from std.sys import argv
from flint.core.build_stamp import BUILD_COMMIT
from flint.core.perf_history import PerfTable, parse_table
from flint.assets.image import write_png
from flint.platform.os import (
    env_value,
    file_size,
    make_dirs,
    parent_directory,
    read_bytes,
    write_bytes,
)
from flint.render.flip import flip_error_map, flip_heat_image, flip_mean
from flint.render.image_diff import ImageDiff, compare_pixels, compare_with_png


def _cli_flag_value(flag_name: String) -> String:
    var args = argv()
    var eq_prefix = flag_name + "="
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == flag_name and i + 1 < len(args):
            return String(args[i + 1])
        if a.startswith(eq_prefix):
            return String(a[byte=eq_prefix.byte_length():a.byte_length()])
        i += 1
    return String()


def resolve_lab_csv_path(lab_slug: String) -> String:
    var cli = _cli_flag_value("--csv")
    if cli.byte_length() > 0:
        return cli
    var env_p = env_value("FLINT_PERF_CSV")
    if env_p.byte_length() > 0:
        return env_p
    return "build/labs/lab_" + lab_slug + "_perf.csv"


def resolve_lab_package(default_pkg: String = "v5-lab") -> String:
    var cli = _cli_flag_value("--package")
    if cli.byte_length() > 0:
        return cli
    var env_p = env_value("FLINT_PERF_PACKAGE")
    if env_p.byte_length() > 0:
        return env_p
    return default_pkg


def resolve_lab_date(default_date: String = "2026-11-20") -> String:
    var cli = _cli_flag_value("--date")
    if cli.byte_length() > 0:
        return cli
    var env_d = env_value("FLINT_PERF_DATE")
    if env_d.byte_length() > 0:
        return env_d
    return default_date


def _fmt_f64_3(val: Float64) -> String:
    if isnan(val):
        return "nan"
    if isinf(val):
        return "-inf" if val < 0.0 else "inf"
    var neg = val < 0.0
    var a = -val if neg else val
    if a >= 1.0e15:
        return ("-" if neg else "") + String(a)
    var scaled = Int(a * 1000.0 + 0.5)
    if scaled == 0:
        return "0.000"
    var whole = scaled // 1000
    var frac = scaled % 1000
    var frac_s = String(frac)
    if frac < 10:
        frac_s = "00" + frac_s
    elif frac < 100:
        frac_s = "0" + frac_s
    return ("-" if neg else "") + String(whole) + "." + frac_s


def _hex_u64(value: UInt64) -> String:
    comptime HEX = "0123456789abcdef"
    var bytes = List[UInt8](capacity=18)
    bytes.append(UInt8(48))  # '0'
    bytes.append(UInt8(120))  # 'x'
    var hex_bytes = HEX.as_bytes()
    for i in range(16):
        var shift = UInt64((15 - i) * 4)
        var nib = Int((value >> shift) & 0xF)
        bytes.append(hex_bytes[nib])
    return String(unsafe_from_utf8=bytes)


def compute_latency_percentiles(
    samples_ms: List[Float64],
) -> Tuple[Float64, Float64, Float64, Float64, Float64]:
    """Returns `(p50_ms, p99_ms, min_ms, max_ms, mean_ms)` sorted deterministically."""
    var n = len(samples_ms)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    var ordered = List[Float64](capacity=n)
    var sum_v = Float64(0.0)
    for i in range(n):
        var v = samples_ms[i]
        sum_v += v
        var pos = len(ordered)
        ordered.append(v)
        while pos > 0 and ordered[pos - 1] > v:
            ordered[pos] = ordered[pos - 1]
            pos -= 1
        ordered[pos] = v
    var p50_idx = (n - 1) * 50 // 100
    var p99_idx = (n - 1) * 99 // 100
    var p50 = ordered[p50_idx]
    var p99 = ordered[p99_idx]
    var min_v = ordered[0]
    var max_v = ordered[n - 1]
    var mean_v = sum_v / Float64(n)
    return (p50, p99, min_v, max_v, mean_v)


struct LabCaseResult(Copyable, Movable):
    """Result record for a single benchmark, determinism gate, or golden-frame check."""
    var name: String
    var passed: Bool
    var samples: Int
    var p50_ms: Float64
    var p99_ms: Float64
    var min_ms: Float64
    var max_ms: Float64
    var mean_ms: Float64
    var budget_ms: Float64
    var deterministic: Bool
    var hash_a: UInt64
    var hash_b: UInt64
    var flip_score: Float64
    var heatmap_path: String
    var detail: String

    def __init__(out self, name: String = ""):
        self.name = name
        self.passed = True
        self.samples = 0
        self.p50_ms = 0.0
        self.p99_ms = 0.0
        self.min_ms = 0.0
        self.max_ms = 0.0
        self.mean_ms = 0.0
        self.budget_ms = 0.0
        self.deterministic = True
        self.hash_a = 0
        self.hash_b = 0
        self.flip_score = 0.0
        self.heatmap_path = String()
        self.detail = String()

    def summary_line(self) -> String:
        var tag = "[PASS]" if self.passed else "[FAIL]"
        var line = "  " + tag + " " + self.name
        if self.samples > 0:
            line += (
                " | p50="
                + _fmt_f64_3(self.p50_ms)
                + "ms p99="
                + _fmt_f64_3(self.p99_ms)
                + "ms"
            )
            if self.budget_ms > 0.0:
                line += " (budget<=" + _fmt_f64_3(self.budget_ms) + "ms)"
        if self.hash_a != 0 or self.hash_b != 0:
            line += " | det=" + _hex_u64(self.hash_a)
            if not self.deterministic:
                line += " != " + _hex_u64(self.hash_b)
        if self.flip_score > 0.0 or self.heatmap_path.byte_length() > 0:
            line += " | flip=" + _fmt_f64_3(self.flip_score)
            if self.heatmap_path.byte_length() > 0:
                line += " heatmap=" + self.heatmap_path
        if self.detail.byte_length() > 0:
            line += " | " + self.detail
        return line


struct EngineBenchSuite(Movable):
    """Aggregates subsystem lab microbenchmarks, determinism gates, and FLIP golden diffs."""
    var suite_name: String
    var results: List[LabCaseResult]
    var passed_cases: Int
    var failed_cases: Int

    def __init__(out self, suite_name: String = "flint_lab"):
        self.suite_name = suite_name
        self.results = List[LabCaseResult]()
        self.passed_cases = 0
        self.failed_cases = 0

    def all_passed(self) -> Bool:
        return self.failed_cases == 0

    def _push_result(mut self, var item: LabCaseResult) -> Bool:
        var ok = item.passed
        if ok:
            self.passed_cases += 1
        else:
            self.failed_cases += 1
        self.results.append(item^)
        return ok

    def record_samples(
        mut self,
        name: String,
        samples_ms: List[Float64],
        budget_p99_ms: Float64 = 0.0,
        detail: String = "",
    ) -> Bool:
        var stats = compute_latency_percentiles(samples_ms)
        var r = LabCaseResult(name)
        r.samples = len(samples_ms)
        r.p50_ms = stats[0]
        r.p99_ms = stats[1]
        r.min_ms = stats[2]
        r.max_ms = stats[3]
        r.mean_ms = stats[4]
        r.budget_ms = budget_p99_ms
        r.detail = detail
        r.passed = len(samples_ms) > 0 and (
            budget_p99_ms <= 0.0 or r.p99_ms <= budget_p99_ms
        )
        return self._push_result(r^)

    def record_latency(
        mut self,
        name: String,
        samples_ms: List[Float64],
        budget_p99_ms: Float64 = 0.0,
        detail: String = "",
    ) -> Bool:
        return self.record_samples(name, samples_ms, budget_p99_ms, detail)

    def record_single(
        mut self,
        name: String,
        elapsed_ms: Float64,
        passed: Bool = True,
        budget_ms: Float64 = 0.0,
        detail: String = "",
    ) -> Bool:
        var r = LabCaseResult(name)
        r.samples = 1
        r.p50_ms = elapsed_ms
        r.p99_ms = elapsed_ms
        r.min_ms = elapsed_ms
        r.max_ms = elapsed_ms
        r.mean_ms = elapsed_ms
        r.budget_ms = budget_ms
        r.detail = detail
        r.passed = passed and (budget_ms <= 0.0 or elapsed_ms <= budget_ms)
        return self._push_result(r^)

    def record_determinism_gate(
        mut self,
        name: String,
        hash_a: UInt64,
        hash_b: UInt64,
        elapsed_ms: Float64 = 0.0,
        budget_ms: Float64 = 0.0,
        detail: String = "",
    ) -> Bool:
        var r = LabCaseResult(name)
        r.deterministic = hash_a == hash_b
        r.hash_a = hash_a
        r.hash_b = hash_b
        if elapsed_ms > 0.0:
            r.samples = 1
            r.p50_ms = elapsed_ms
            r.p99_ms = elapsed_ms
            r.min_ms = elapsed_ms
            r.max_ms = elapsed_ms
            r.mean_ms = elapsed_ms
        r.budget_ms = budget_ms
        r.detail = detail
        r.passed = r.deterministic and (
            budget_ms <= 0.0 or elapsed_ms <= budget_ms
        )
        return self._push_result(r^)

    def _emit_failure_heatmap(
        self,
        diff: ImageDiff,
        actual: List[UInt32],
        expected: List[UInt32],
        width: Int,
        height: Int,
        out_path: String,
    ) -> String:
        if width <= 0 or height <= 0:
            return String()
        var target = out_path
        if target.byte_length() == 0:
            target = "/tmp/flint_lab_" + self.suite_name + "_flip_heatmap.png"
        var err_map = diff.flip_map.copy()
        if len(err_map) != width * height and len(expected) >= width * height and len(actual) >= width * height:
            err_map = flip_error_map(expected, actual, width, height)
        if len(err_map) != width * height:
            err_map = List[Float32](length=width * height, fill=Float32(1.0))
        var heat_pixels = flip_heat_image(err_map)
        var parent = parent_directory(target)
        if parent.byte_length() > 0 and parent != ".":
            _ = make_dirs(parent)
        if write_png(target, heat_pixels, width, height, stamped=False):
            return target
        return String()

    def verify_golden_pixels(
        mut self,
        name: String,
        actual_pixels: List[UInt32],
        expected_pixels: List[UInt32],
        width: Int,
        height: Int,
        heatmap_out_path: String = "",
        tolerance: Int = 2,
        spread_per_million: Int = 500,
        flip_limit: Float64 = 0.05,
    ) -> Bool:
        var diff = compare_pixels(
            actual_pixels, expected_pixels, width, height, tolerance
        )
        var ok = diff.passes(spread_per_million, flip_limit)
        var r = LabCaseResult(name)
        r.passed = ok
        r.flip_score = diff.flip_mean if diff.same_size else 1.0
        r.detail = diff.summary()
        if not ok:
            var target = (
                heatmap_out_path
                if heatmap_out_path.byte_length() > 0
                else "/tmp/flint_lab_" + name + "_flip_heat.png"
            )
            r.heatmap_path = self._emit_failure_heatmap(
                diff, actual_pixels, expected_pixels, width, height, target
            )
        return self._push_result(r^)

    def verify_golden_frame(
        mut self,
        name: String,
        actual_pixels: List[UInt32],
        width: Int,
        height: Int,
        reference_png_path: String,
        heatmap_out_path: String = "",
        tolerance: Int = 2,
        spread_per_million: Int = 500,
        flip_limit: Float64 = 0.05,
        bootstrap_if_missing: Bool = False,
    ) -> Bool:
        if bootstrap_if_missing and file_size(reference_png_path) <= 0:
            var ref_dir = parent_directory(reference_png_path)
            if ref_dir.byte_length() > 0 and ref_dir != ".":
                _ = make_dirs(ref_dir)
            var copy_px = actual_pixels.copy()
            var wrote = write_png(
                reference_png_path, copy_px, width, height, stamped=False
            )
            var r_boot = LabCaseResult(name)
            r_boot.passed = wrote
            r_boot.detail = "bootstrapped golden PNG: " + reference_png_path
            return self._push_result(r_boot^)

        var diff = compare_with_png(
            actual_pixels, width, height, reference_png_path, tolerance
        )
        var ok = diff.passes(spread_per_million, flip_limit)
        var r = LabCaseResult(name)
        r.passed = ok
        r.flip_score = (
            diff.flip_mean
            if diff.same_size
            else (flip_mean(diff.flip_map) if len(diff.flip_map) > 0 else 1.0)
        )
        r.detail = diff.summary()
        if not ok:
            var target = (
                heatmap_out_path
                if heatmap_out_path.byte_length() > 0
                else reference_png_path + ".flip_heat.png"
            )
            r.heatmap_path = self._emit_failure_heatmap(
                diff, actual_pixels, List[UInt32](), width, height, target
            )
        return self._push_result(r^)

    def summary_report(self) -> String:
        var out = String(
            "============================================================\n"
        )
        out += (
            " FLINT LAB SUITE: "
            + self.suite_name
            + " ("
            + String(self.passed_cases)
            + "/"
            + String(len(self.results))
            + " passed)\n"
        )
        out += "============================================================\n"
        for i in range(len(self.results)):
            out += self.results[i].summary_line() + "\n"
        out += "------------------------------------------------------------\n"
        out += (
            " RESULT: "
            + ("PASS" if self.all_passed() else "FAIL")
            + " (passed="
            + String(self.passed_cases)
            + ", failed="
            + String(self.failed_cases)
            + ")\n"
        )
        out += "============================================================"
        return out

    def print_summary(self) -> Bool:
        print(self.summary_report())
        return self.all_passed()

    def to_perf_table(
        self,
        existing_csv: String = "",
        package_name: String = "v5-lab",
        date_str: String = "2026-11-18",
    ) -> PerfTable:
        var table = (
            parse_table(existing_csv)
            if existing_csv.byte_length() > 0
            else PerfTable()
        )
        for i in range(len(self.results)):
            if self.results[i].samples > 0:
                _ = table.ensure_column(self.results[i].name + "_p50_ms")
                _ = table.ensure_column(self.results[i].name + "_p99_ms")
        var row = List[String](length=len(table.header), fill=String())
        var c_date = table.column("date")
        var c_commit = table.column("commit")
        var c_pkg = table.column("package")
        var c_res = table.column("result")
        if c_date >= 0:
            row[c_date] = date_str
        if c_commit >= 0:
            row[c_commit] = BUILD_COMMIT
        if c_pkg >= 0:
            row[c_pkg] = package_name
        if c_res >= 0:
            row[c_res] = "pass" if self.all_passed() else "fail"
        for i in range(len(self.results)):
            if self.results[i].samples > 0:
                var col50 = table.column(self.results[i].name + "_p50_ms")
                var col99 = table.column(self.results[i].name + "_p99_ms")
                if col50 >= 0:
                    row[col50] = _fmt_f64_3(self.results[i].p50_ms)
                if col99 >= 0:
                    row[col99] = _fmt_f64_3(self.results[i].p99_ms)
        table.rows.append(row^)
        return table^

    def append_perf_history_csv(
        self,
        csv_path: String,
        package_name: String = "v5-lab",
        date_str: String = "2026-11-18",
    ) -> Bool:
        var existing = String()
        if file_size(csv_path) > 0:
            var raw = read_bytes(csv_path)
            if len(raw) > 0:
                existing = String(unsafe_from_utf8=raw)
        var table = self.to_perf_table(existing, package_name, date_str)
        var csv_text = table.text()
        var out_bytes = List[UInt8](capacity=csv_text.byte_length())
        for b in csv_text.as_bytes():
            out_bytes.append(b)
        var dir_p = parent_directory(csv_path)
        if dir_p.byte_length() > 0 and dir_p != ".":
            _ = make_dirs(dir_p)
        return write_bytes(csv_path, out_bytes)

    def finish_and_export(self, lab_slug: String) -> Bool:
        var passed = self.print_summary()
        var csv_path = resolve_lab_csv_path(lab_slug)
        var pkg = resolve_lab_package()
        var date_s = resolve_lab_date()
        var csv_ok = self.append_perf_history_csv(csv_path, pkg, date_s)
        print("[lab_" + lab_slug + "] perf_csv ->", csv_path)
        return passed and csv_ok

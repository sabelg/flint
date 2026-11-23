"""Flint Engine v5 Pure-Mojo Gate Verifier (`tools/gate_verify.mojo`).

Verifies:
  1. Golden-frame perceptual FLIP + pixel spread across all 5 `build/shots/*.png`
     (plus `flint_studio_smoke.png`) against unstamped `tests/golden/shots/*.png`.
  2. Subsystem lab p50/p99 regression & absolute budget gating via
     `flint.core.perf_history` against `docs/PERF_HISTORY.csv` and `docs/PERF_HISTORY.md`.
"""
from std.collections import List
from std.ffi import external_call
from std.math import isfinite
from std.sys import argv
from flint.assets.image import decode_png, write_png
from flint.core.build_stamp import BUILD_COMMIT
from flint.core.perf_history import (
    BASELINE_ROWS,
    PerfTable,
    accept_lines,
    append_candidate_row,
    drift,
    judge_lab_row,
    merge_lab_csvs,
    parse_table,
    perf_metrics,
    settle,
    value_at,
)
from flint.lab.bench_suite import (
    EngineBenchSuite,
    resolve_lab_date,
    resolve_lab_package,
)
from flint.platform.os import (
    env_value,
    file_size,
    make_dirs,
    read_bytes,
    remove_file,
    write_bytes,
)


def _read_text(path: String) -> String:
    if file_size(path) <= 0:
        return String()
    var raw = read_bytes(path)
    if len(raw) == 0:
        return String()
    return String(unsafe_from_utf8=raw)


def _write_text(path: String, text: String) -> Bool:
    var out_bytes = List[UInt8](capacity=text.byte_length())
    for b in text.as_bytes():
        out_bytes.append(b)
    return write_bytes(path, out_bytes)


def _fmt_3(val: Float64) -> String:
    if not isfinite(val):
        if val > 0.0:
            return String("inf")
        if val < 0.0:
            return String("-inf")
        return String("nan")
    var neg = val < 0.0
    var a = -val if neg else val
    var scaled = Int(min(a * 1000.0 + 0.5, Float64(1e15)))
    var whole = scaled // 1000
    var frac = scaled % 1000
    var frac_s = String(frac)
    if frac < 10:
        frac_s = "00" + frac_s
    elif frac < 100:
        frac_s = "0" + frac_s
    return ("-" if neg else "") + String(whole) + "." + frac_s


def _lab_binary_for_source(lab_bin_dir: String, source: String) -> String:
    if not source.startswith("lab_") or source.byte_length() <= 4:
        return String()
    var suffix = String(source[byte=4:source.byte_length()])
    return lab_bin_dir + "/flint-lab-" + suffix


def _rerun_lab_csv(lab_bin: String, source: String) -> Tuple[Int, String]:
    """Re-runs a lab binary once into a temporary CSV.
    Status: 1 = passed rerun, 0 = binary missing (skipped), -1 = rerun failed."""
    if lab_bin.byte_length() == 0 or file_size(lab_bin) <= 0:
        return (0, String())
    var pid = Int(external_call["getpid", Int32]())
    var tmp_csv = "/tmp/flint_gate_rerun_" + String(pid) + "_" + source + ".csv"
    var cmd = "\"" + lab_bin + "\" --csv \"" + tmp_csv + "\""
    var rc = Int(external_call["system", Int32](cmd.as_c_string_span()))
    var txt = _read_text(tmp_csv)
    _ = remove_file(tmp_csv)
    if rc != 0 or txt.byte_length() == 0:
        return (-1, String())
    return (1, txt)


def _bootstrap_unstamped_golden(src_png: String, dst_png: String) -> Bool:
    var raw = read_bytes(src_png)
    if len(raw) == 0:
        print("[GateVerify] ERROR: missing source shot:", src_png)
        return False
    var img = decode_png(raw, 16384)
    if img.width != 1280 or img.height != 720 or len(img.pixels) != 1280 * 720:
        print(
            "[GateVerify] ERROR: invalid shot dimensions for",
            src_png,
            img.width,
            "x",
            img.height,
        )
        return False
    var px = img.pixels.copy()
    if not write_png(dst_png, px, img.width, img.height, stamped=False):
        print("[GateVerify] ERROR: failed to write golden PNG:", dst_png)
        return False
    print("[GateVerify] Bootstrapped unstamped golden PNG ->", dst_png)
    return True


def _verify_shot(
    mut suite: EngineBenchSuite,
    name: String,
    actual_png: String,
    golden_png: String,
    tolerance: Int,
    spread_per_million: Int,
    flip_limit: Float64,
) -> Bool:
    var raw = read_bytes(actual_png)
    if len(raw) == 0:
        return suite.verify_golden_frame(
            name,
            List[UInt32](),
            0,
            0,
            golden_png,
            heatmap_out_path=actual_png + ".flip_heat.png",
            tolerance=tolerance,
            spread_per_million=spread_per_million,
            flip_limit=flip_limit,
            bootstrap_if_missing=False,
        )
    var img = decode_png(raw, 16384)
    return suite.verify_golden_frame(
        name,
        img.pixels,
        img.width,
        img.height,
        golden_png,
        heatmap_out_path=actual_png + ".flip_heat.png",
        tolerance=tolerance,
        spread_per_million=spread_per_million,
        flip_limit=flip_limit,
        bootstrap_if_missing=False,
    )


def main() raises:
    var bootstrap_mode = False
    var no_rerun = env_value("FLINT_GATE_NO_RERUN") == "1"
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--bootstrap-golden":
            bootstrap_mode = True
        elif a == "--no-rerun":
            no_rerun = True

    var shot_names: List[String] = [
        "flint_spark",
        "open_world_noon_vista",
        "open_world_golden_citadel",
        "open_world_night_ltc_bokeh",
        "flint_studio",
    ]

    if bootstrap_mode:
        _ = make_dirs("tests/golden/shots")
        for i in range(len(shot_names)):
            var sname = shot_names[i]
            if not _bootstrap_unstamped_golden(
                "build/shots/" + sname + ".png",
                "tests/golden/shots/" + sname + ".png",
            ):
                raise Error("failed bootstrapping golden PNG: " + sname)
        return

    print("============================================================")
    print(" FLINT GATE VERIFIER (Golden FLIP + perf_history p99)")
    print("============================================================")

    # 1. Golden-frame FLIP verification
    var flip_suite = EngineBenchSuite("flint_v5_golden_flip")
    _ = _verify_shot(
        flip_suite,
        "flint_spark",
        "build/shots/flint_spark.png",
        "tests/golden/shots/flint_spark.png",
        tolerance=1,
        spread_per_million=250,
        flip_limit=0.01,
    )
    _ = _verify_shot(
        flip_suite,
        "open_world_noon_vista",
        "build/shots/open_world_noon_vista.png",
        "tests/golden/shots/open_world_noon_vista.png",
        tolerance=4,
        spread_per_million=1000,
        flip_limit=0.02,
    )
    _ = _verify_shot(
        flip_suite,
        "open_world_golden_citadel",
        "build/shots/open_world_golden_citadel.png",
        "tests/golden/shots/open_world_golden_citadel.png",
        tolerance=4,
        spread_per_million=1000,
        flip_limit=0.02,
    )
    _ = _verify_shot(
        flip_suite,
        "open_world_night_ltc_bokeh",
        "build/shots/open_world_night_ltc_bokeh.png",
        "tests/golden/shots/open_world_night_ltc_bokeh.png",
        tolerance=4,
        spread_per_million=1000,
        flip_limit=0.02,
    )
    _ = _verify_shot(
        flip_suite,
        "flint_studio",
        "build/shots/flint_studio.png",
        "tests/golden/shots/flint_studio.png",
        tolerance=4,
        spread_per_million=1000,
        flip_limit=0.02,
    )
    _ = _verify_shot(
        flip_suite,
        "flint_studio_smoke",
        "build/shots/flint_studio_smoke.png",
        "tests/golden/shots/flint_studio.png",
        tolerance=4,
        spread_per_million=1000,
        flip_limit=0.02,
    )
    var flip_ok = flip_suite.print_summary()

    # 2. Merge 5 lab CSVs and judge against docs/PERF_HISTORY.csv + docs/PERF_HISTORY.md
    var lab_csv_paths: List[String] = [
        "build/labs/lab_physics_perf.csv",
        "build/labs/lab_anim_perf.csv",
        "build/labs/lab_render_perf.csv",
        "build/labs/lab_audio_ui_perf.csv",
        "build/labs/lab_net_perf.csv",
    ]
    var lab_texts = List[String](capacity=len(lab_csv_paths))
    for i in range(len(lab_csv_paths)):
        var txt = _read_text(lab_csv_paths[i])
        if txt.byte_length() == 0:
            raise Error("missing or empty lab CSV: " + lab_csv_paths[i])
        lab_texts.append(txt)

    var pkg = resolve_lab_package("v5-gate")
    var date_s = resolve_lab_date("2026-11-20")
    var candidate = merge_lab_csvs(lab_texts, pkg, date_s, BUILD_COMMIT)

    var res_col = candidate.column("result")
    var cand_result = (
        candidate.cell(0, res_col)
        if (len(candidate.rows) > 0 and res_col >= 0)
        else String("fail")
    )
    var csv_results_ok = cand_result == "pass"

    var hist_text = _read_text("docs/PERF_HISTORY.csv")
    if hist_text.byte_length() == 0:
        raise Error("missing docs/PERF_HISTORY.csv")
    var history = parse_table(hist_text)
    if len(history.rows) < BASELINE_ROWS:
        raise Error("docs/PERF_HISTORY.csv has fewer than BASELINE_ROWS baseline rows")

    var notes_text = _read_text("docs/PERF_HISTORY.md")
    var accepts = accept_lines(notes_text)
    var metrics = perf_metrics()
    var findings = judge_lab_row(metrics, candidate, 0, history, accepts)

    # Settle any noisy finding over its limit by re-running the affected lab binary once.
    var lab_bin_dir = env_value("FLINT_LAB_BIN_DIR")
    if lab_bin_dir.byte_length() == 0:
        lab_bin_dir = "build/labs"
    var rerun_sources = List[String]()
    var rerun_statuses = List[Int]()
    var rerun_tables = List[PerfTable]()
    var rerun_failed = False

    if csv_results_ok and not no_rerun:
        for i in range(len(findings)):
            if not findings[i].needs_again():
                continue
            var m_idx = -1
            for k in range(len(metrics)):
                if metrics[k].name == findings[i].metric:
                    m_idx = k
                    break
            if m_idx < 0:
                continue
            var src_name = metrics[m_idx].source
            var cache_idx = -1
            for c in range(len(rerun_sources)):
                if rerun_sources[c] == src_name:
                    cache_idx = c
                    break
            if cache_idx < 0:
                var lab_bin = _lab_binary_for_source(lab_bin_dir, src_name)
                print(
                    "[GateVerify] settle() re-running",
                    lab_bin,
                    "for",
                    findings[i].metric,
                    "...",
                )
                var rr = _rerun_lab_csv(lab_bin, src_name)
                var status = rr[0]
                var tbl = PerfTable()
                if status == 1:
                    tbl = parse_table(rr[1])
                    var rr_res_col = tbl.column("result")
                    if len(tbl.rows) == 0 or (
                        rr_res_col >= 0
                        and tbl.cell(len(tbl.rows) - 1, rr_res_col) != "pass"
                    ):
                        status = -1
                if status == 0:
                    print(
                        "[GateVerify] Note: lab binary unavailable for settle() rerun:",
                        lab_bin,
                    )
                elif status < 0:
                    print(
                        "[GateVerify] ERROR: settle() rerun failed for",
                        lab_bin,
                    )
                    rerun_failed = True
                cache_idx = len(rerun_sources)
                rerun_sources.append(src_name)
                rerun_statuses.append(status)
                rerun_tables.append(tbl^)

            if rerun_statuses[cache_idx] == 1:
                ref rtbl = rerun_tables[cache_idx]
                var r_row = len(rtbl.rows) - 1
                var r_col = rtbl.column(findings[i].metric)
                var parsed2 = value_at(rtbl, r_row, r_col)
                if parsed2[0]:
                    var settled = settle(
                        metrics[m_idx],
                        findings[i],
                        parsed2[1],
                        history,
                        accepts,
                        len(history.rows),
                    )
                    settled.accepted = findings[i].accepted
                    print(
                        "[GateVerify] settle()",
                        findings[i].metric,
                        "first=",
                        settled.first,
                        "second=",
                        settled.second,
                        "-> settled=",
                        settled.value,
                    )
                    findings[i] = settled^
                    var c_col = candidate.column(findings[i].metric)
                    if c_col >= 0 and len(candidate.rows) > 0:
                        candidate.rows[0][c_col] = _fmt_3(findings[i].value)

    var perf_suite = EngineBenchSuite("flint_v5_perf_regression_gate")
    var perf_ok = csv_results_ok and (not rerun_failed) and (len(findings) == len(metrics))
    if not csv_results_ok:
        print(
            "[PerfGate] ERROR: merged lab CSV result is '"
            + cand_result
            + "' (expected 'pass')"
        )
        _ = perf_suite.record_single(
            "lab_csv_determinism_result",
            0.0,
            passed=False,
            detail="merged_result=" + cand_result,
        )
    if len(findings) != len(metrics):
        print(
            "[PerfGate] ERROR: expected",
            len(metrics),
            "metrics in merged CSVs, found",
            len(findings),
        )

    var any_accepted_cost = False
    for i in range(len(findings)):
        ref f = findings[i]
        var m_budget = Float64(0.0)
        var m_idx = -1
        for k in range(len(metrics)):
            if metrics[k].name == f.metric:
                m_budget = metrics[k].budget
                m_idx = k
                break
        if f.worse and f.accepted:
            any_accepted_cost = True
        var case_ok = (not f.over_budget) and ((not f.worse) or f.accepted)
        if not case_ok:
            perf_ok = False
        var detail = (
            "base="
            + String(f.baseline)
            + " rows="
            + String(f.rows)
            + (" ACCEPTED" if f.accepted else "")
            + (" REMEASURED(" + _fmt_3(f.first) + "->" + _fmt_3(f.second) + ")" if f.remeasured else "")
        )
        _ = perf_suite.record_single(
            f.metric,
            f.value,
            passed=case_ok,
            budget_ms=m_budget,
            detail=detail,
        )
        if m_idx >= 0:
            var d = drift(
                history,
                accepts,
                metrics[m_idx],
                f.value,
                pkg,
                f.accepted,
                len(history.rows),
            )
            if d.warns():
                print("  [DRIFT WARN]", d.line())

    var suite_ok = perf_suite.print_summary()
    perf_ok = perf_ok and suite_ok

    if perf_ok and any_accepted_cost and len(candidate.rows) > 0 and res_col >= 0:
        candidate.rows[0][res_col] = "accepted"

    _ = make_dirs("build/labs")
    _ = _write_text("build/labs/perf_latest.csv", candidate.text())

    if env_value("FLINT_PERF_RECORD") == "1" and flip_ok and perf_ok:
        append_candidate_row(history, candidate, 0)
        if not _write_text("docs/PERF_HISTORY.csv", history.text()):
            raise Error("failed to append recorded row to docs/PERF_HISTORY.csv")
        print("[GateVerify] Appended recorded row to docs/PERF_HISTORY.csv")

    if not flip_ok or not perf_ok:
        raise Error("Flint gate verification failed")
    print("[GateVerify] PASS: All 6 golden FLIP checks and 33 p50/p99 perf gates passed.")

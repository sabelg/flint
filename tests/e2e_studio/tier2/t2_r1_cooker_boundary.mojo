# AREA: r1
"""Tier 2 Boundary & Negative Opaque API Tests: Area A — Cooker & FlpkV3 Boundaries (F04, F05, Bug #6)."""

from std.collections import List
from std.ffi import external_call
from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flpk_v3 import AssetUuid, FlpkV3MappedArchive
from flint.core.jobs import JobPool
from flint.platform.os import file_size, make_dirs, read_bytes
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    rm_rf_tmp,
    write_bytes,
    write_text,
)


def _bytes_equal_str(raw: List[UInt8], expected: String) -> Bool:
    var exp = expected.as_bytes()
    if len(raw) != len(exp):
        return False
    for i in range(len(raw)):
        if raw[i] != exp[i]:
            return False
    return True


def main() raises:
    var tmp = make_tmp_dir("t2_r1_api")
    var runner = StudioSuiteRunner("t2_r1_cooker_boundary")

    var pool = JobPool()
    _ = pool.start(2)

    # 1. Empty & nonexistent workspace -> 64-byte valid empty archive
    var empty_ws = tmp + "/empty_ws"
    _ = make_dirs(empty_ws)
    var empty_flpk = tmp + "/empty.flpk"
    var miss_flpk = tmp + "/missing.flpk"
    var rep_empty = cook_workspace(empty_ws, empty_flpk, pool, cache_dir=tmp + "/c_empty")
    var rep_miss = cook_workspace(tmp + "/does_not_exist", miss_flpk, pool, cache_dir=tmp + "/c_miss")
    var a_empty = FlpkV3MappedArchive.open(empty_flpk)
    var empty_ok = (
        rep_empty.ok
        and rep_empty.total_assets == 0
        and rep_empty.archive_bytes == 64
        and rep_miss.ok
        and rep_miss.total_assets == 0
        and rep_miss.archive_bytes == 64
        and a_empty.valid
        and a_empty.toc_count() == 0
    )
    a_empty.close()
    _ = runner.check("t2_api_cook_empty_and_nonexistent_workspace", empty_ok, "empty/nonexistent workspace failed")

    # 2. Symlink source dir & nested symlink loop guard
    var real_ws = tmp + "/real_ws"
    _ = write_text(real_ws + "/cfg.json", '{"loop_safe":true}\n')
    var sym_dir = tmp + "/sym_ws"
    var ln_cmd1 = "ln -s " + real_ws + " " + sym_dir
    _ = external_call["system", Int32](ln_cmd1.as_c_string_span())
    var ln_cmd2 = "ln -s " + real_ws + " " + real_ws + "/loop"
    _ = external_call["system", Int32](ln_cmd2.as_c_string_span())

    var rep_sym = cook_workspace(sym_dir, tmp + "/sym.flpk", pool, cache_dir=tmp + "/c_sym")
    var rep_loop = cook_workspace(real_ws, tmp + "/loop.flpk", pool, cache_dir=tmp + "/c_loop")
    var sym_ok = rep_sym.ok and rep_sym.total_assets == 0 and rep_loop.ok and rep_loop.total_assets == 1
    _ = runner.check(
        "t2_api_cook_symlink_dir_and_nested_symlink_loop",
        sym_ok,
        "sym_total=" + String(rep_sym.total_assets) + " loop_total=" + String(rep_loop.total_assets),
    )

    # 3. Unsupported extensions ignored & force_recook=True bypasses cache
    var unsup_ws = tmp + "/unsup_ws"
    _ = write_text(unsup_ws + "/notes.txt", "ignored\n")
    _ = write_text(unsup_ws + "/ghost.png.flmeta", "{}\n")
    var rep_unsup = cook_workspace(unsup_ws, tmp + "/unsup.flpk", pool, cache_dir=tmp + "/c_unsup")

    var frc_ws = tmp + "/frc_ws"
    _ = write_text(frc_ws + "/a.json", '{"frc":1}\n')
    var frc_cache = tmp + "/c_frc"
    var frc_flpk = tmp + "/frc.flpk"
    var rep_f1 = cook_workspace(frc_ws, frc_flpk, pool, cache_dir=frc_cache, force_recook=True)
    var rep_f2 = cook_workspace(frc_ws, frc_flpk, pool, cache_dir=frc_cache, force_recook=True)
    var frc_ok = (
        rep_unsup.ok
        and rep_unsup.total_assets == 0
        and rep_f1.ok
        and rep_f2.ok
        and rep_f2.cooked_count == 1
        and rep_f2.cache_hits == 0
    )
    _ = runner.check("t2_api_cook_unsupported_ignored_and_force_recook", frc_ok, "force_recook or unsupported check failed")

    # 4. Corrupt PNG reports failure cleanly
    var bad_ws = tmp + "/bad_ws"
    _ = write_text(bad_ws + "/good.json", '{"good":true}\n')
    var good_pack = cook_workspace(bad_ws, tmp + "/bad.flpk", pool, cache_dir=tmp + "/c_bad")
    var before_failure = read_bytes(tmp + "/bad.flpk")
    var bad_png = List[UInt8]()
    bad_png.append(0x89); bad_png.append(0x50); bad_png.append(0x4E); bad_png.append(0x47)
    bad_png.append(0x0D); bad_png.append(0x0A)
    _ = write_bytes(bad_ws + "/corrupt.png", bad_png)
    var rep_bad = cook_workspace(bad_ws, tmp + "/bad.flpk", pool, cache_dir=tmp + "/c_bad")
    var bad_ok = (
        good_pack.ok and len(before_failure) > 64 and not rep_bad.ok and rep_bad.failed_count == 1
        and read_bytes(tmp + "/bad.flpk") == before_failure
    )
    _ = runner.check(
        "t2_api_cook_corrupt_source_reports_failure",
        bad_ok,
        "ok=" + String(rep_bad.ok) + " failed_count=" + String(rep_bad.failed_count),
    )

    # 5. Same-size edit & page_align=False size comparison
    var pa_ws = tmp + "/pa_ws"
    var pa_cache = tmp + "/c_pa"
    var pa_aligned = tmp + "/pa_aligned.flpk"
    var pa_compact = tmp + "/pa_compact.flpk"
    _ = write_text(pa_ws + "/a.json", '{"k":"AAAA"}\n')
    _ = write_text(pa_ws + "/b.flmat", '{"roughness":0.5}\n')
    _ = write_text(pa_ws + "/c.flscn", '{"schema":2}\n')

    var rep_pa1 = cook_workspace(pa_ws, pa_aligned, pool, cache_dir=pa_cache, page_align=True)
    var rep_pa2 = cook_workspace(pa_ws, pa_compact, pool, cache_dir=pa_cache, page_align=False)
    var sz_aligned = file_size(pa_aligned)
    var sz_compact = file_size(pa_compact)

    var reference = tmp + "/timestamp-ref"
    _ = write_text(reference, "")
    var save_stamp = "touch -r " + pa_ws + "/a.json " + reference
    var restore_stamp = "touch -r " + reference + " " + pa_ws + "/a.json"
    var saved_stamp = external_call["system", Int32](save_stamp.as_c_string_span())
    _ = write_text(pa_ws + "/a.json", '{"k":"BBBB"}\n')
    var restored_stamp = external_call["system", Int32](restore_stamp.as_c_string_span())
    var rep_edit = cook_workspace(pa_ws, pa_compact, pool, cache_dir=pa_cache, page_align=False)
    var clean_pack = tmp + "/pa_clean.flpk"
    var rep_clean = cook_workspace(pa_ws, clean_pack, pool, cache_dir=tmp + "/fresh-cache", page_align=False)
    var arch_edit = FlpkV3MappedArchive.open(pa_compact)
    var rd_edit = arch_edit.read_by_path("a.json", 0)
    var edit_ok = (
        saved_stamp == 0 and restored_stamp == 0 and rep_clean.ok
        and read_bytes(pa_compact) == read_bytes(clean_pack)
        and rep_pa1.ok
        and rep_pa2.ok
        and sz_compact > 0
        and sz_aligned > sz_compact
        and rep_edit.ok
        and arch_edit.valid
        and rd_edit[0]
        and _bytes_equal_str(rd_edit[1], '{"k":"BBBB"}\n')
    )
    arch_edit.close()
    _ = runner.check(
        "t2_api_same_size_edit_and_page_align_false",
        edit_ok,
        "sz_aligned=" + String(sz_aligned) + " sz_compact=" + String(sz_compact),
    )

    var damaged = read_bytes(pa_compact)
    damaged[len(damaged) - 1] ^= 1
    _ = write_bytes(pa_compact, damaged)
    var repaired = cook_workspace(pa_ws, pa_compact, pool, cache_dir=pa_cache, page_align=False)
    _ = runner.check(
        "t2_api_cook_repairs_modified_output", repaired.ok and read_bytes(pa_compact) == read_bytes(clean_pack),
        "modified output was accepted as an unchanged pack",
    )

    # 6. FlpkV3MappedArchive negative boundaries
    var trunc_path = tmp + "/trunc.flpk"
    var trunc_bytes = List[UInt8](length=16, fill=0x46)
    _ = write_bytes(trunc_path, trunc_bytes)

    var a_missing = FlpkV3MappedArchive.open(tmp + "/nonexistent.flpk")
    var a_trunc = FlpkV3MappedArchive.open(trunc_path)
    var a_valid = FlpkV3MappedArchive.open(pa_compact)
    var neg_ok = (
        (not a_missing.valid)
        and (not a_trunc.valid)
        and a_valid.valid
        and a_valid.find_by_path("missing/file.json", 0) == -1
        and a_valid.find_by_uuid(AssetUuid(), 0) == -1
        and len(a_valid.read_entry_bytes(-1)) == 0
    )
    a_missing.close()
    a_trunc.close()
    a_valid.close()
    _ = runner.check("t2_api_flpk_archive_negative_boundaries", neg_ok, "archive negative boundary check failed")

    pool.stop()
    rm_rf_tmp(tmp)
    runner.finish()

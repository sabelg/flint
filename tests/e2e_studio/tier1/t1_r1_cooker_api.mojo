# AREA: r1
"""Tier 1 Opaque API Tests: Area A — R1 Cooker Extensions, Cache Keys, FlpkV3 & Logging (F01-F05, Bug #6)."""

from std.collections import List
from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flpk_v3 import (
    ASSET_KIND_RAW,
    AssetMeta,
    AssetUuid,
    FlpkV3MappedArchive,
    write_flmeta,
)
from flint.core.jobs import JobPool
from flint.core.log import LOG_CORE, flush_log, log_info, log_path, start_log
from flint.platform.os import make_dirs, remove_file
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    read_text,
    rm_rf_tmp,
    write_bytes,
    write_minimal_glb,
    write_minimal_png,
    write_minimal_wav,
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


def _populate_11_extensions(ws: String) -> Bool:
    var ok = True
    ok = ok and write_minimal_glb(ws + "/meshes/box.glb")
    ok = ok and write_minimal_png(ws + "/textures/albedo.png", 16, 16, 0xFF4488CC)
    ok = ok and write_minimal_wav(ws + "/audio/hit.wav", 0.02, 48000, 2)
    var qoa_bytes = List[UInt8]()
    for b in String("qoaf0001qoa_payload").as_bytes():
        qoa_bytes.append(b)
    ok = ok and write_bytes(ws + "/audio/music.qoa", qoa_bytes)
    ok = ok and write_text(ws + "/scenes/main.flscn", '{"schema":2,"entities":[]}\n')
    ok = ok and write_text(ws + "/scenes/bin.flscnb", "FLSCNB_V2_PAYLOAD")
    ok = ok and write_text(ws + "/prefabs/prop.flprefab", '{"name":"prop","entities":[]}\n')
    ok = ok and write_text(ws + "/materials/stone.flmat", '{"roughness":0.8}\n')
    ok = ok and write_text(ws + "/anims/idle.flanim", '{"clip":"idle","fps":30}\n')
    ok = ok and write_text(ws + "/rigs/hero.flrig", "FRIG_V1_PAYLOAD")
    ok = ok and write_text(ws + "/data/config.json", '{"version":1}\n')
    return ok


def main() raises:
    var tmp = make_tmp_dir("t1_r1_api")
    var runner = StudioSuiteRunner("t1_r1_cooker_api")

    var pool = JobPool()
    _ = pool.start(2)

    var ws = tmp + "/ws"
    var cache = tmp + "/cache"
    var flpk = tmp + "/pack.flpk"

    var prep_ok = _populate_11_extensions(ws)
    _ = runner.check("t1_api_prepare_11_extensions", prep_ok, "failed writing synthetic workspace")

    # 1. Cook all 11 R1 extensions & verify FlpkV3MappedArchive
    var rep = cook_workspace(ws, flpk, pool, cache_dir=cache)
    var cook_ok = rep.ok and rep.total_assets == 11 and rep.cooked_count == 11 and rep.cache_hits == 0 and rep.failed_count == 0
    _ = runner.check(
        "t1_api_cook_all_r1_extensions",
        cook_ok,
        "ok=" + String(rep.ok) + " total=" + String(rep.total_assets) + " cooked=" + String(rep.cooked_count) + " hits=" + String(rep.cache_hits) + " fail=" + String(rep.failed_count),
    )

    var arch = FlpkV3MappedArchive.open(flpk)
    var rels = List[String]()
    rels.append("meshes/box.glb")
    rels.append("textures/albedo.png")
    rels.append("audio/hit.wav")
    rels.append("audio/music.qoa")
    rels.append("scenes/main.flscn")
    rels.append("scenes/bin.flscnb")
    rels.append("prefabs/prop.flprefab")
    rels.append("materials/stone.flmat")
    rels.append("anims/idle.flanim")
    rels.append("rigs/hero.flrig")
    rels.append("data/config.json")

    var arch_ok = arch.valid
    if arch_ok:
        for i in range(len(rels)):
            var rel = rels[i]
            var idx = arch.find_by_path(rel, 0)
            var uidx = arch.find_by_uuid(AssetUuid.from_path(rel), 0)
            if idx < 0 or uidx != idx or not arch.verify_entry_crc64(idx):
                arch_ok = False
                break
        var raw_checks_ok = (
            arch.toc_entry(arch.find_by_path("scenes/main.flscn", 0)).asset_kind == ASSET_KIND_RAW
            and _bytes_equal_str(arch.read_by_path("scenes/main.flscn", 0)[1], '{"schema":2,"entities":[]}\n')
            and _bytes_equal_str(arch.read_by_path("materials/stone.flmat", 0)[1], '{"roughness":0.8}\n')
            and _bytes_equal_str(arch.read_by_path("data/config.json", 0)[1], '{"version":1}\n')
            and _bytes_equal_str(arch.read_by_path("audio/music.qoa", 0)[1], "qoaf0001qoa_payload")
        )
        arch_ok = arch_ok and raw_checks_ok
    arch.close()
    _ = runner.check("t1_api_flpk_v3_crc64_uuid_path_readback", arch_ok, "FlpkV3MappedArchive lookup/CRC64/payload mismatch")

    # 2. Incremental 100% cache hit
    var rep2 = cook_workspace(ws, flpk, pool, cache_dir=cache)
    var inc_ok = rep2.ok and rep2.total_assets == 11 and rep2.cache_hits == 11 and rep2.cooked_count == 0
    _ = runner.check(
        "t1_api_cook_incremental_cache_hit",
        inc_ok,
        "total=" + String(rep2.total_assets) + " hits=" + String(rep2.cache_hits) + " cooked=" + String(rep2.cooked_count),
    )

    # 3. Single-file edit recooks only changed file
    _ = write_text(ws + "/scenes/main.flscn", '{"schema":2,"entities":[{"name":"Updated"}]}\n')
    var rep3 = cook_workspace(ws, flpk, pool, cache_dir=cache)
    var mod_ok = rep3.ok and rep3.total_assets == 11 and rep3.cooked_count == 1 and rep3.cache_hits == 10
    var arch3 = FlpkV3MappedArchive.open(flpk)
    mod_ok = mod_ok and arch3.valid and _bytes_equal_str(
        arch3.read_by_path("scenes/main.flscn", 0)[1],
        '{"schema":2,"entities":[{"name":"Updated"}]}\n',
    )
    arch3.close()
    _ = runner.check(
        "t1_api_cook_modified_file_recooks_only_changed",
        mod_ok,
        "cooked=" + String(rep3.cooked_count) + " hits=" + String(rep3.cache_hits),
    )

    # 4. Rename/move identical file (without and with .flmeta) updates canonical rel_path
    var mv_ws = tmp + "/mv_ws"
    var mv_cache = tmp + "/mv_cache"
    var mv_flpk = tmp + "/mv.flpk"
    _ = make_dirs(mv_ws + "/data")
    _ = write_text(mv_ws + "/data/item_a.json", '{"id":42}\n')
    _ = write_text(mv_ws + "/data/tracked_a.flmat", '{"color":"red"}\n')
    var tracked_meta = AssetMeta()
    tracked_meta.uuid = AssetUuid.from_path("custom/shared_mat_uuid")
    tracked_meta.asset_kind = ASSET_KIND_RAW
    tracked_meta.source_path = "data/tracked_a.flmat"
    _ = write_flmeta(mv_ws + "/data/tracked_a.flmat.flmeta", tracked_meta)

    var mv_rep1 = cook_workspace(mv_ws, mv_flpk, pool, cache_dir=mv_cache)
    _ = remove_file(mv_ws + "/data/item_a.json")
    _ = remove_file(mv_ws + "/data/tracked_a.flmat")
    _ = remove_file(mv_ws + "/data/tracked_a.flmat.flmeta")
    _ = make_dirs(mv_ws + "/data/moved")
    _ = write_text(mv_ws + "/data/moved/item_b.json", '{"id":42}\n')
    _ = write_text(mv_ws + "/data/moved/tracked_b.flmat", '{"color":"red"}\n')
    tracked_meta.source_path = "data/moved/tracked_b.flmat"
    _ = write_flmeta(mv_ws + "/data/moved/tracked_b.flmat.flmeta", tracked_meta)

    var mv_rep2 = cook_workspace(mv_ws, mv_flpk, pool, cache_dir=mv_cache)
    var mv_arch = FlpkV3MappedArchive.open(mv_flpk)
    var mv_ok = (
        mv_rep1.ok
        and mv_rep2.ok
        and mv_arch.valid
        and mv_arch.find_by_path("data/moved/item_b.json", 0) >= 0
        and mv_arch.find_by_path("data/item_a.json", 0) == -1
        and mv_arch.find_by_path("data/moved/tracked_b.flmat", 0) >= 0
        and mv_arch.find_by_path("data/tracked_a.flmat", 0) == -1
    )
    mv_arch.close()
    _ = runner.check(
        "t1_api_cook_renamed_file_updates_rel_path",
        mv_ok,
        "stale rel_path remained or new rel_path missing after rename/move",
    )

    # 5. Canonical rel_path cache sharing across two different workspace dirs
    var ws_a = tmp + "/canon_a"
    var ws_b = tmp + "/canon_b"
    var canon_cache = tmp + "/canon_cache"
    _ = write_text(ws_a + "/sub/cfg.json", '{"shared":true}\n')
    _ = write_text(ws_b + "/sub/cfg.json", '{"shared":true}\n')
    var rep_ca = cook_workspace(ws_a, tmp + "/ca.flpk", pool, cache_dir=canon_cache)
    var rep_cb = cook_workspace(ws_b, tmp + "/cb.flpk", pool, cache_dir=canon_cache)
    var canon_ok = rep_ca.ok and rep_cb.ok and rep_cb.total_assets == 1 and rep_cb.cache_hits == 1 and rep_cb.cooked_count == 0
    _ = runner.check(
        "t1_api_canonical_relpath_cache_across_workspaces",
        canon_ok,
        "hits=" + String(rep_cb.cache_hits) + " cooked=" + String(rep_cb.cooked_count),
    )

    # 6. start_log uses explicit app_name
    var state_dir = tmp + "/log_state"
    var started = start_log("r1_custom_app", state_dir)
    log_info(LOG_CORE, "hello_r1_api")
    var flushed = flush_log()
    var lpath = log_path()
    var ltext = read_text(lpath)
    var log_ok = started and flushed >= 1 and lpath.endswith("/logs/r1_custom_app.log") and ltext.find("hello_r1_api") >= 0
    _ = runner.check("t1_api_start_log_uses_app_name", log_ok, "log_path=" + lpath)

    pool.stop()
    rm_rf_tmp(tmp)
    runner.finish()

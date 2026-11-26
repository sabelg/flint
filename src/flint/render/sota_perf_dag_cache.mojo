"""Sota:perf r3: the cluster-DAG `.flpk` cache on by default.

Area A's cache (`sota_geometry_dag_cache.mojo`) is keyed by mesh content, the DAG group size, the blob version and
the builder version, so a hit is always the DAG this build would make. This file picks where it lives and keeps it
bounded:

- `sota_dag_cache_default_dir()`: `$FLINT_DAG_CACHE_DIR`, else `$XDG_CACHE_HOME/flint/dag`, else
  `$HOME/.cache/flint/dag`; `FLINT_DAG_CACHE=0` turns it off (""). Read with getenv(3), so a test's setenv counts.
- `sota_dag_cache_prune(dir, budget_mb)`: once the `*.flpk` files exceed the budget (`FLINT_DAG_CACHE_MB`,
  default 4096), deletes the least recently written ones until 90 % of it is left.
"""
from std.collections import List
from std.ffi import external_call
from flint.platform.os import env_now, file_stamp, list_directory, make_dirs, remove_file

comptime SOTA_DAG_CACHE_DEFAULT_MB: Int = 4096


def sota_dag_cache_default_dir() -> String:
    """Default `sota_geom.dag_cache_dir` (cvar r.sota.geometry.dag_cache); "" when disabled."""
    if env_now("FLINT_DAG_CACHE") == "0":
        return ""
    var d = env_now("FLINT_DAG_CACHE_DIR")
    if d.byte_length() > 0:
        return d
    var xdg = env_now("XDG_CACHE_HOME")
    if xdg.byte_length() > 0:
        return xdg + "/flint/dag"
    var home = env_now("HOME")
    if home.byte_length() > 0:
        return home + "/.cache/flint/dag"
    return ""


def sota_dag_cache_budget_mb() -> Int:
    var s = env_now("FLINT_DAG_CACHE_MB")
    if s.byte_length() == 0:
        return SOTA_DAG_CACHE_DEFAULT_MB
    var v = 0
    for i in range(s.byte_length()):
        var c = Int(s.as_bytes()[i])
        if c < 48 or c > 57:
            return SOTA_DAG_CACHE_DEFAULT_MB
        v = v * 10 + (c - 48)
    return v


def sota_dag_cache_ensure_dir(dir: String) -> Bool:
    if dir.byte_length() == 0:
        return False
    return make_dirs(dir)


def _is_flpk(name: String) -> Bool:
    var n = name.byte_length()
    if n < 6:
        return False
    var b = name.as_bytes()
    return b[n - 5] == 46 and b[n - 4] == 102 and b[n - 3] == 108 and b[n - 2] == 112 and b[n - 1] == 107  # ".flpk"


def sota_dag_cache_prune(dir: String, budget_mb: Int) -> Int:
    """Deletes the oldest `*.flpk` entries while the cache is over `budget_mb`; returns how many it deleted."""
    if dir.byte_length() == 0 or budget_mb <= 0:
        return 0
    var names = list_directory(dir)
    var paths = List[String]()
    var sizes = List[Int]()
    var mtimes = List[Int]()
    var total = 0
    for i in range(len(names)):
        if not _is_flpk(names[i]):
            continue
        var p = dir + "/" + names[i]
        var st = file_stamp(p)
        if st[0] < 0:
            continue
        paths.append(p)
        sizes.append(st[0])
        mtimes.append(st[1])
        total += st[0]
    var budget = budget_mb * 1048576
    if total <= budget:
        return 0
    var keep = budget - budget // 10
    var removed = 0
    var gone = List[Bool](length=len(paths), fill=False)
    while total > keep:
        var oldest = -1
        for i in range(len(paths)):
            if not gone[i] and (oldest < 0 or mtimes[i] < mtimes[oldest]):
                oldest = i
        if oldest < 0:
            break
        gone[oldest] = True
        if remove_file(paths[oldest]):
            removed += 1
        total -= sizes[oldest]
    return removed


def sota_dag_cache_tmp_path(path: String) -> String:
    """A per-thread temporary name next to `path`; renamed over it once complete, so readers never see a torn
    entry and two writers of the same key (other processes, or upload-prepass jobs) do not interleave."""
    var pid = Int(external_call["getpid", Int32]())
    var tid = external_call["pthread_self", UInt64]()
    return path + ".tmp" + String(pid) + "_" + String(tid)

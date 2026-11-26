"""Flint `.flpk` v3 Single-File Asset Pack Bundler (`flint-pack`).

Usage:
  flint-pack <output.flpk> <asset1> [asset2 ...]
  flint-pack --cook <source_dir> <output.flpk> [--cache-dir <dir>]
  flint-pack --inspect <pack.flpk>
"""
from std.ffi import external_call
from std.sys import argv
from flint.assets.cooker_v2 import (
    DEFAULT_COOK_CACHE_DIR,
    _classify_asset_ext,
    cook_single_source_bytes,
    cook_workspace,
)
from flint.assets.flpk_v3 import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_FONT,
    ASSET_KIND_HDR,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    AssetUuid,
    CHUNK_COMP_NONE,
    FlpkBuildChunk,
    FlpkV3MappedArchive,
    SUB_ID_PRIMARY_OR_TAIL,
    load_or_init_flmeta,
    write_flpk_v3,
)
from flint.core.jobs import JobPool, default_worker_count
from flint.platform.os import (
    file_stamp,
    is_symlink,
    list_directory,
    make_dirs,
    parent_directory,
    read_bytes,
    remove_file,
)


comptime MAX_COOK_DEPTH: Int = 64


def _strip_trailing_slashes(path: String) -> String:
    var out = path
    while out.byte_length() > 1 and out.as_bytes()[out.byte_length() - 1] == 47:
        var trimmed = String(out[byte=0 : out.byte_length() - 1])
        out = trimmed^
    return out^


def _is_dir(path: String) -> Bool:
    var clean = _strip_trailing_slashes(path)
    if is_symlink(clean):
        return False
    var d = external_call["opendir", Int](clean.as_c_string_span())
    if d == 0:
        return False
    _ = external_call["closedir", Int32](Pointer[UInt8, MutAnyOrigin](unsafe_from_address=d))
    return True


def _is_cookable_ext(name: String) -> Bool:
    return _classify_asset_ext(name)[0]


def _has_cookable_sources(dir_path: String, depth: Int = 0) -> Bool:
    if depth >= MAX_COOK_DEPTH:
        return False
    var clean_dir = _strip_trailing_slashes(dir_path)
    if is_symlink(clean_dir):
        return False
    var entries = list_directory(clean_dir)
    for i in range(len(entries)):
        var name = entries[i]
        var full = (clean_dir + "/" + name) if clean_dir != "/" else ("/" + name)
        if is_symlink(full):
            continue
        if _is_dir(full):
            if _has_cookable_sources(full, depth + 1):
                return True
        elif _is_cookable_ext(name):
            return True
    return False


def _default_kind_for_path(path: String) -> UInt8:
    if path.endswith(".glb"):
        return ASSET_KIND_MESH
    if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
        return ASSET_KIND_TEXTURE
    if path.endswith(".hdr") or path.endswith(".exr"):
        return ASSET_KIND_HDR
    if path.endswith(".ttf") or path.endswith(".otf"):
        return ASSET_KIND_FONT
    if path.endswith(".wav"):
        return ASSET_KIND_AUDIO
    return ASSET_KIND_RAW


def _inspect_pack(pack_path: String) -> Int:
    var archive = FlpkV3MappedArchive.open(pack_path)
    if not archive.valid:
        print("[Flint Pack] ERROR: Invalid or unreadable .flpk v3 archive: " + pack_path)
        return 1
    var cnt = archive.toc_count()
    print("[Flint Pack] Archive " + pack_path + " (.flpk v3, entries=" + String(cnt) + ")")
    for i in range(cnt):
        var entry = archive.toc_entry(i)
        var p = archive.entry_path(i)
        print(
            "  [" + String(i) + "] " + p
            + " sub=" + String(Int(entry.sub_id))
            + " kind=" + String(Int(entry.asset_kind))
            + " bytes=" + String(Int(entry.compressed_bytes))
            + "/" + String(Int(entry.uncompressed_bytes))
        )
    archive.close()
    return 0


def _cook_dir(
    source_dir: String,
    output_pack: String,
    page_align: Bool,
    cache_dir: String = DEFAULT_COOK_CACHE_DIR,
) -> Int:
    var clean_src = _strip_trailing_slashes(source_dir)
    var clean_out = _strip_trailing_slashes(output_pack)
    var clean_cache = _strip_trailing_slashes(cache_dir if cache_dir.byte_length() > 0 else DEFAULT_COOK_CACHE_DIR)
    if is_symlink(clean_src):
        print("[Flint Pack] ERROR: Symlink source directory rejected: " + source_dir)
        return 1
    if is_symlink(clean_out):
        print("[Flint Pack] ERROR: Symlink output path rejected: " + output_pack)
        return 1
    if is_symlink(clean_cache):
        print("[Flint Pack] ERROR: Symlink cache directory rejected: " + cache_dir)
        return 1
    if _is_dir(clean_out):
        print("[Flint Pack] ERROR: Output path is a directory: " + output_pack)
        return 1
    if not _is_dir(clean_src):
        print("[Flint Pack] ERROR: Source directory missing or empty: " + source_dir)
        return 1
    var entries = list_directory(clean_src)
    if len(entries) == 0:
        print("[Flint Pack] ERROR: Source directory missing or empty: " + source_dir)
        return 1
    if not _has_cookable_sources(clean_src):
        print("[Flint Pack] ERROR: Failed to cook workspace into " + output_pack)
        return 1
    var existed_before = (not _is_dir(clean_out)) and file_stamp(clean_out)[0] >= 0
    var pool = JobPool()
    _ = pool.start(max(1, default_worker_count()))
    var rep = cook_workspace(clean_src, output_pack, pool, cache_dir=clean_cache, page_align=page_align)
    pool.stop()
    var total_bytes = file_stamp(output_pack)[0]
    if rep.ok and rep.total_assets > 0 and total_bytes > 0:
        print(
            "[Flint Pack] Packed "
            + String(rep.total_assets)
            + " asset(s) ("
            + String(rep.total_chunks)
            + " chunk(s), cooked="
            + String(rep.cooked_count)
            + ", cache_hits="
            + String(rep.cache_hits)
            + ") into "
            + output_pack
            + " ("
            + String(total_bytes)
            + " bytes)"
        )
        return 0
    else:
        if not existed_before:
            _ = remove_file(output_pack)
        print("[Flint Pack] ERROR: Failed to cook workspace into " + output_pack)
        return 1


def _run_pack() -> Int:
    var args = argv()
    if len(args) < 2:
        print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
        return 1

    var flag_raw = False
    var flag_page_align = True
    var flag_cook = False
    var flag_inspect = False
    var flag_cache_dir = String("")
    var positional = List[String]()

    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--raw":
            flag_raw = True
        elif a == "--no-page-align":
            flag_page_align = False
        elif a == "--cook":
            flag_cook = True
        elif a == "--inspect":
            flag_inspect = True
        elif a == "--cache-dir":
            if i + 1 >= len(args):
                print("[Flint Pack] ERROR: --cache-dir requires a directory argument")
                return 1
            i += 1
            flag_cache_dir = String(args[i])
            if flag_cache_dir.byte_length() == 0:
                print("[Flint Pack] ERROR: --cache-dir requires a non-empty directory")
                return 1
        elif a.startswith("--cache-dir="):
            flag_cache_dir = String(a[byte=12:])
            if flag_cache_dir.byte_length() == 0:
                print("[Flint Pack] ERROR: --cache-dir requires a non-empty directory")
                return 1
        elif a == "--help" or a == "-h":
            print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
            print("       flint-pack --cook <source_dir> <output.flpk> [--cache-dir <dir>]")
            print("       flint-pack --inspect <pack.flpk>")
            return 0
        elif a.startswith("-"):
            print("[Flint Pack] ERROR: Unknown option: " + a)
            print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
            return 1
        else:
            positional.append(a)
        i += 1

    if flag_inspect:
        if len(positional) < 1:
            print("Usage: flint-pack --inspect <pack.flpk>")
            return 1
        return _inspect_pack(positional[0])

    if flag_cook:
        if len(positional) < 2:
            print("Usage: flint-pack --cook <source_dir> <output.flpk> [--cache-dir <dir>]")
            return 1
        return _cook_dir(positional[0], positional[1], flag_page_align, flag_cache_dir)

    if len(positional) < 2:
        print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
        return 1

    var output_pack = positional[0]
    var clean_out = _strip_trailing_slashes(output_pack)
    if is_symlink(clean_out):
        print("[Flint Pack] ERROR: Symlink output path rejected: " + output_pack)
        return 1
    if _is_dir(clean_out):
        print("[Flint Pack] ERROR: Output path is a directory: " + output_pack)
        return 1
    var all_chunks = List[FlpkBuildChunk]()
    var packed_assets = 0

    for i in range(1, len(positional)):
        var p = positional[i]
        var clean_p = _strip_trailing_slashes(p)
        if is_symlink(clean_p):
            print("[Flint Pack] ERROR: Symlink input path rejected: " + p)
            return 1
        if _is_dir(clean_p):
            print("[Flint Pack] ERROR: Input path is a directory (use --cook): " + p)
            return 1
        var st = file_stamp(p)
        if st[0] <= 0:
            print("[Flint Pack] ERROR: Missing or empty input asset: " + p)
            return 1
        var src = read_bytes(p)
        if len(src) == 0:
            print("[Flint Pack] ERROR: Failed to read input asset: " + p)
            return 1
        if flag_raw:
            all_chunks.append(
                FlpkBuildChunk(
                    AssetUuid.from_path(p),
                    p,
                    SUB_ID_PRIMARY_OR_TAIL,
                    ASSET_KIND_RAW,
                    src^,
                    CHUNK_COMP_NONE,
                )
            )
            packed_assets += 1
        else:
            var kind = _default_kind_for_path(p)
            var meta = load_or_init_flmeta(p, p, kind, persist_if_missing=False)
            var bundle = cook_single_source_bytes(src, meta, p)
            if not bundle.valid or len(bundle.chunks) == 0:
                print("[Flint Pack] ERROR: Failed to cook asset '" + p + "': " + bundle.error)
                return 1
            for c in range(len(bundle.chunks)):
                all_chunks.append(bundle.chunks[c].copy())
            packed_assets += 1

    if packed_assets == 0 or len(all_chunks) == 0:
        print("[Flint Pack] ERROR: Failed to pack assets into " + output_pack)
        return 1

    var pdir = parent_directory(output_pack)
    if pdir.byte_length() > 0 and pdir != ".":
        _ = make_dirs(pdir)

    if not write_flpk_v3(output_pack, all_chunks, page_align=flag_page_align):
        print("[Flint Pack] ERROR: Failed to pack assets into " + output_pack)
        return 1

    var total_bytes = file_stamp(output_pack)[0]
    if total_bytes <= 0:
        print("[Flint Pack] ERROR: Failed to pack assets into " + output_pack)
        return 1

    var verify = FlpkV3MappedArchive.open(output_pack)
    if not verify.valid:
        verify.close()
        print("[Flint Pack] ERROR: Failed to verify packed archive: " + output_pack)
        return 1
    verify.close()

    print(
        "[Flint Pack] Packed "
        + String(packed_assets)
        + " asset(s) ("
        + String(len(all_chunks))
        + " chunk(s)) into "
        + output_pack
        + " ("
        + String(total_bytes)
        + " bytes)"
    )
    return 0


def main():
    var rc = _run_pack()
    if rc != 0:
        external_call["exit", NoneType](Int32(rc))

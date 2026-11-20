"""Flint `.flpk` v3 Single-File Asset Pack Bundler (`flint-pack`).

Usage:
  flint-pack <output.flpk> <asset1> [asset2 ...]
  flint-pack --cook <source_dir> <output.flpk>
  flint-pack --inspect <pack.flpk>
"""
from std.ffi import external_call
from std.sys import argv
from flint.assets.cooker_v2 import cook_single_source_bytes, cook_workspace
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
    list_directory,
    make_dirs,
    parent_directory,
    read_bytes,
    remove_file,
)


def _is_dir(path: String) -> Bool:
    var owned = path
    var d = external_call["opendir", Int](owned.as_c_string_span())
    if d == 0:
        return False
    _ = external_call["closedir", Int32](d)
    return True


def _is_cookable_ext(name: String) -> Bool:
    return (
        name.endswith(".glb")
        or name.endswith(".png")
        or name.endswith(".jpg")
        or name.endswith(".jpeg")
        or name.endswith(".hdr")
        or name.endswith(".exr")
        or name.endswith(".ttf")
        or name.endswith(".wav")
    )


def _has_cookable_sources(dir_path: String) -> Bool:
    var entries = list_directory(dir_path)
    for i in range(len(entries)):
        var name = entries[i]
        var full = (dir_path + "/" + name) if dir_path != "/" else ("/" + name)
        if _is_dir(full):
            if _has_cookable_sources(full):
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


def _inspect_pack(pack_path: String):
    var archive = FlpkV3MappedArchive.open(pack_path)
    if not archive.valid:
        print("[Flint Pack] ERROR: Invalid or unreadable .flpk v3 archive: " + pack_path)
        return
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


def _cook_dir(source_dir: String, output_pack: String, page_align: Bool):
    if not _is_dir(source_dir):
        print("[Flint Pack] ERROR: Source directory missing or empty: " + source_dir)
        return
    var entries = list_directory(source_dir)
    if len(entries) == 0:
        print("[Flint Pack] ERROR: Source directory missing or empty: " + source_dir)
        return
    if not _has_cookable_sources(source_dir):
        print("[Flint Pack] ERROR: Failed to cook workspace into " + output_pack)
        return
    var existed_before = (not _is_dir(output_pack)) and file_stamp(output_pack)[0] >= 0
    var pool = JobPool()
    _ = pool.start(max(1, default_worker_count()))
    var rep = cook_workspace(source_dir, output_pack, pool, page_align=page_align)
    pool.stop()
    var total_bytes = file_stamp(output_pack)[0]
    if rep.ok and rep.total_assets > 0 and total_bytes > 0:
        print(
            "[Flint Pack] Packed "
            + String(rep.total_assets)
            + " asset(s) ("
            + String(rep.total_chunks)
            + " chunk(s)) into "
            + output_pack
            + " ("
            + String(total_bytes)
            + " bytes)"
        )
    else:
        if not existed_before:
            _ = remove_file(output_pack)
        print("[Flint Pack] ERROR: Failed to cook workspace into " + output_pack)


def main():
    var args = argv()
    if len(args) < 2:
        print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
        return

    var flag_raw = False
    var flag_page_align = True
    var flag_cook = False
    var flag_inspect = False
    var positional = List[String]()

    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--raw":
            flag_raw = True
        elif a == "--no-page-align":
            flag_page_align = False
        elif a == "--cook":
            flag_cook = True
        elif a == "--inspect":
            flag_inspect = True
        elif a == "--help" or a == "-h":
            print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
            print("       flint-pack --cook <source_dir> <output.flpk>")
            print("       flint-pack --inspect <pack.flpk>")
            return
        elif a.startswith("-"):
            print("[Flint Pack] ERROR: Unknown option: " + a)
            print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
            return
        else:
            positional.append(a)

    if flag_inspect:
        if len(positional) < 1:
            print("Usage: flint-pack --inspect <pack.flpk>")
            return
        _inspect_pack(positional[0])
        return

    if flag_cook:
        if len(positional) < 2:
            print("Usage: flint-pack --cook <source_dir> <output.flpk>")
            return
        _cook_dir(positional[0], positional[1], flag_page_align)
        return

    if len(positional) < 2:
        print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
        return

    var output_pack = positional[0]
    if _is_dir(output_pack):
        print("[Flint Pack] ERROR: Output path is a directory: " + output_pack)
        return
    var all_chunks = List[FlpkBuildChunk]()
    var packed_assets = 0

    for i in range(1, len(positional)):
        var p = positional[i]
        if _is_dir(p):
            print("[Flint Pack] ERROR: Input path is a directory (use --cook): " + p)
            return
        var st = file_stamp(p)
        if st[0] <= 0:
            print("[Flint Pack] ERROR: Missing or empty input asset: " + p)
            return
        var src = read_bytes(p)
        if len(src) == 0:
            print("[Flint Pack] ERROR: Failed to read input asset: " + p)
            return
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
                return
            for c in range(len(bundle.chunks)):
                all_chunks.append(bundle.chunks[c].copy())
            packed_assets += 1

    if packed_assets == 0 or len(all_chunks) == 0:
        print("[Flint Pack] ERROR: Failed to pack assets into " + output_pack)
        return

    var pdir = parent_directory(output_pack)
    if pdir.byte_length() > 0 and pdir != ".":
        _ = make_dirs(pdir)

    if not write_flpk_v3(output_pack, all_chunks, page_align=flag_page_align):
        print("[Flint Pack] ERROR: Failed to pack assets into " + output_pack)
        return

    var total_bytes = file_stamp(output_pack)[0]
    if total_bytes <= 0:
        print("[Flint Pack] ERROR: Failed to pack assets into " + output_pack)
        return

    var verify = FlpkV3MappedArchive.open(output_pack)
    if not verify.valid:
        verify.close()
        print("[Flint Pack] ERROR: Failed to verify packed archive: " + output_pack)
        return
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

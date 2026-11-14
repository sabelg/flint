"""Flint VFS Single-File Asset Pack Bundler (`flint-pack`).

Usage:
  flint-pack <output.flpk> <asset1> [asset2 ...]
"""
from std.sys import argv
from flint.assets.vfs import write_flpk_pack
from flint.platform.os import file_stamp


def main():
    var args = argv()
    if len(args) < 3:
        print("Usage: flint-pack <output.flpk> <asset1> [asset2 ...]")
        return
    var output_pack = String(args[1])
    var paths = List[String]()
    for i in range(2, len(args)):
        paths.append(String(args[i]))
    var count = write_flpk_pack(output_pack, paths)
    var total_bytes = file_stamp(output_pack)[0]
    if count > 0 and total_bytes > 0:
        print(
            "[Flint Pack] Packed "
            + String(count)
            + " asset(s) into "
            + output_pack
            + " ("
            + String(total_bytes)
            + " bytes)"
        )
    else:
        print("[Flint Pack] ERROR: Failed to pack assets into " + output_pack)

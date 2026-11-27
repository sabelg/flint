#!/usr/bin/env python3
"""Resolve one Flint crash report against an archived release ELF and its split symbols."""
import argparse
import hashlib
import re
import subprocess
from pathlib import Path


def build_id(elf):
    notes = subprocess.check_output(["readelf", "-nW", elf], text=True, stderr=subprocess.DEVNULL)
    match = re.search(r"Build ID: ([0-9a-fA-F]+)", notes)
    return match[1].lower() if match else None


def executable_segments(elf):
    headers = subprocess.check_output(["readelf", "-lW", elf], text=True)
    segments = []
    for line in headers.splitlines():
        fields = line.split()
        if fields and fields[0] == "LOAD" and "E" in "".join(fields[6:-1]):
            offset, address, size = (int(fields[i], 16) for i in (1, 2, 4))
            segments.append((offset, address, size))
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Flint crash.txt from the failed process")
    parser.add_argument("elf", type=Path, help="installed player binary or exact staged gameplay.so")
    parser.add_argument("archive", type=Path, help="private directory supplied to package --symbols")
    parser.add_argument("--mapped-path", help="exact path in the report for a non-player module")
    args = parser.parse_args()
    report = args.report.read_text()
    with args.elf.open("rb") as binary:
        digest = hashlib.file_digest(binary, "sha256").hexdigest()
    symbols = args.archive / (digest + ".debug")
    if not symbols.is_file():
        parser.error("the private archive has no symbols for this ELF's SHA-256")
    identity = build_id(args.elf)
    if not identity or identity != build_id(symbols):
        parser.error("the ELF and its symbol file have different or missing GNU build IDs")
    if args.mapped_path:
        mapped = args.mapped_path
        # A guest's runtime mapping can be a generated hot-reload filename. If
        # the mapping is still on disk, verify it before reporting source lines.
        recorded_guest = re.search(r"\.build-id-([0-9a-f]+)$", mapped)
        observed = Path(mapped)
        if recorded_guest:
            if recorded_guest[1] != identity:
                parser.error("the crash report identifies a different gameplay library")
        elif observed.is_file():
            with observed.open("rb") as staged:
                if hashlib.file_digest(staged, "sha256").hexdigest() != digest:
                    parser.error("the mapped file's SHA-256 differs from the supplied ELF")
        else:
            parser.error("this older guest mapping needs its original staged ELF at --mapped-path to verify its identity")
    else:
        saved_id = re.search(r"^build_id: ([0-9a-fA-F]+)$", report, re.M)
        executable = re.search(r"^executable: (.+)$", report, re.M)
        if not executable or not saved_id or saved_id[1].lower() != identity:
            parser.error("the crash report identifies a different player ELF")
        mapped = executable[1]

    mappings = []
    for line in report.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[-1].removesuffix(" (deleted)") != mapped:
            continue
        if "x" in fields[1] and re.fullmatch(r"[0-9a-f]+-[0-9a-f]+", fields[0]):
            start, end = (int(x, 16) for x in fields[0].split("-"))
            mappings.append((start, end, int(fields[2], 16)))
    segments = executable_segments(args.elf)
    resolved = 0
    for frame, address in re.findall(r"^  (#\d+|pc) (0x[\da-fA-F]+)$", report, re.M):
        # The explicit fault PC points at the instruction. Unwind frames carry
        # return PCs; the preceding byte is in the calling instruction's range.
        pc = int(address, 16) - int(frame != "pc")
        for low, high, file_base in mappings:
            if not low <= pc < high:
                continue
            file_offset = pc - low + file_base
            for offset, virtual, size in segments:
                if offset <= file_offset < offset + size:
                    line = subprocess.check_output(
                        ["addr2line", "-f", "-i", "-p", "-e", symbols, hex(virtual + file_offset - offset)],
                        text=True,
                    ).strip()
                    print(frame, address, line)
                    resolved += int(any(re.search(r":\d+\b", entry) for entry in line.splitlines()))
                    break
    if not resolved:
        parser.exit(1, "No source line for this module appears in the captured frames.\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch selected 100STYLE BVH clips without downloading the whole 1.4 GB archive.

100STYLE (Mason, Starke, Komura 2022, https://doi.org/10.1145/3522618) is published on Zenodo
(record 8127870) under CC BY 4.0. This script reads the zip's central directory with HTTP range
requests and extracts only the styles you ask for, so a Neutral-only database costs a few MB.

usage: fetch_100style.py OUT_DIR [--styles Neutral,Old,...] [--list]

The clips land in OUT_DIR/<Style>/<Style>_<Motion>.bvh, next to a copy of the attribution in
OUT_DIR/ATTRIBUTION.txt. Keep both outside the Flint repository.
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/8127870/files/100STYLE.zip/content"

ATTRIBUTION = """100STYLE Dataset
Ian Mason, Sebastian Starke, Taku Komura.
"Real-Time Style Modelling of Human Locomotion via Feature-Wise Transformations and Local
Motion Phases", Proc. ACM Comput. Graph. Interact. Tech. 5(1), 2022. doi:10.1145/3522618
Data: https://zenodo.org/records/8127870 - licensed under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/).
Flint retargets these clips onto its stick biped and resamples them; that is a modification
of the original data.
"""


class RangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests (with a small read-ahead cache)."""

    def __init__(self, url, block=1 << 20):
        super().__init__()
        self.url = url
        self.pos = 0
        self.block = block
        self.cache_start = -1
        self.cache = b""
        req = urllib.request.Request(url, method="GET", headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as r:
            cr = r.headers.get("Content-Range")
        if not cr:
            raise RuntimeError("server does not support range requests")
        self.size = int(cr.split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def _fetch(self, start, end):
        req = urllib.request.Request(self.url, headers={"Range": "bytes=%d-%d" % (start, end - 1)})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if self.cache_start >= 0 and self.cache_start <= self.pos and self.pos + n <= self.cache_start + len(self.cache):
            off = self.pos - self.cache_start
            out = self.cache[off:off + n]
        elif n >= self.block:
            out = self._fetch(self.pos, self.pos + n)
        else:
            end = min(self.size, self.pos + max(n, self.block))
            self.cache = self._fetch(self.pos, end)
            self.cache_start = self.pos
            out = self.cache[:n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--styles", default="Neutral")
    ap.add_argument("--list", action="store_true", help="print the styles in the archive")
    args = ap.parse_args()

    zf = zipfile.ZipFile(RangeFile(URL))
    names = [n for n in zf.namelist()
             if n.lower().endswith(".bvh") and not os.path.basename(n).startswith("._") and "__MACOSX" not in n]
    if args.list:
        styles = sorted({n.split("/")[-2] for n in names if "/" in n})
        print("\n".join(styles))
        return
    want = [s.strip() for s in args.styles.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    got = 0
    for n in names:
        parts = n.split("/")
        if len(parts) < 2 or parts[-2] not in want:
            continue
        dst_dir = os.path.join(args.out_dir, parts[-2])
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(n))
        if not os.path.exists(dst):
            with zf.open(n) as src, open(dst, "wb") as f:
                f.write(src.read())
        got += 1
        print(dst)
    with open(os.path.join(args.out_dir, "ATTRIBUTION.txt"), "w") as f:
        f.write(ATTRIBUTION)
    if got == 0:
        sys.exit("no clips matched --styles %s (try --list)" % args.styles)


if __name__ == "__main__":
    main()

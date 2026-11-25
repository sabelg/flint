"""Text-to-image asset search with SigLIP 2 (`flint.ml.siglip.search`).

`AssetIndex` embeds every `.png` and `.glb` under a folder (recursively; directories starting
with `.` are skipped) and answers free-text queries in any language the Gemma tokenizer covers:

    var model = SiglipModel.load(model_dir)
    var index = AssetIndex.open(cache_path, model)      # empty if missing or for another model
    var report = index.update(model, assets_dir)        # embeds only new / changed files
    _ = index.save(cache_path)
    var hits = index.search(model, "rostige Schatztruhe", 10)

PNGs are decoded with `flint.assets.image`; a GLB is drawn with `render_glb_thumbnail` (written
as a PNG next to the cache when `thumbs_dir` is given). Images go through the exact SigLIP
preprocessing and are embedded in batches. A file is re-embedded when its size or modification
time changes; files that disappeared are dropped.

Ranking uses the model's sigmoid head: `logit = exp(scale) * cos(text, image) + bias` and
`prob = sigmoid(logit)`, the probability that the image matches the text on its own (not a
softmax over the folder). Queries are embedded once; scoring is a dot product per asset.

Cache file (`.flsg`, `flint.core.serial` blobs, tag `SGIX` version 1): model name, embedding
width, image size, logit scale / bias, then per asset its relative path, kind, size, mtime and
L2-normalised float32 embedding.
"""

from std.math import exp
from std.time import perf_counter_ns
from flint.assets.image import NativeImage, decode_png, write_png
from flint.assets.glb_native import load_native_glb
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.platform.os import read_bytes, write_bytes, file_stamp, list_directory, make_dirs
from flint.ml.siglip.model import SiglipModel, sigmoid
from flint.ml.siglip.image import siglip_preprocess, render_glb_thumbnail

comptime ASSET_PNG: Int = 0
comptime ASSET_GLB: Int = 1
comptime _CACHE_VERSION: Int = 1


struct AssetHit(Copyable, Movable):
    var path: String
    var kind: Int
    var logit: Float32
    var prob: Float32

    def __init__(out self, path: String, kind: Int, logit: Float32, prob: Float32):
        self.path = path
        self.kind = kind
        self.logit = logit
        self.prob = prob


struct IndexReport(Copyable, Movable):
    var scanned: Int
    var embedded: Int
    var reused: Int
    var removed: Int
    var failed: Int
    var seconds: Float64

    def __init__(out self):
        self.scanned = 0
        self.embedded = 0
        self.reused = 0
        self.removed = 0
        self.failed = 0
        self.seconds = 0.0


def _lower_ascii(s: String) -> String:
    var out = List[UInt8](capacity=s.byte_length())
    for c in s.as_bytes():
        out.append(c + 32 if c >= 65 and c <= 90 else c)
    return String(unsafe_from_utf8=out)


def asset_kind(name: String) -> Int:
    """`ASSET_PNG` / `ASSET_GLB` from the file name, -1 for anything else."""
    var l = _lower_ascii(name)
    if l.endswith(".png"):
        return ASSET_PNG
    if l.endswith(".glb"):
        return ASSET_GLB
    return -1


def scan_assets(root: String) -> List[String]:
    """Paths (relative to `root`, sorted) of the PNG / GLB files below it."""
    var out = List[String]()
    _scan(root, String(""), out, 0)
    return out^


def _scan(root: String, rel: String, mut out: List[String], depth: Int):
    if depth > 32:
        return
    var dir = root if rel.byte_length() == 0 else root + "/" + rel
    var names = list_directory(dir)
    for name in names:
        if name.startswith("."):
            continue
        var r = name if rel.byte_length() == 0 else rel + "/" + name
        if asset_kind(name) >= 0:
            out.append(r)
        elif len(list_directory(dir + "/" + name)) > 0:
            _scan(root, r, out, depth + 1)


def load_asset_image(path: String, kind: Int, thumbs_dir: String = "", rel: String = "") -> NativeImage:
    """The picture SigLIP sees for an asset: the PNG itself or a GLB thumbnail render."""
    if kind == ASSET_PNG:
        return decode_png(read_bytes(path), 1 << 14)
    var model = load_native_glb(path)
    var img = render_glb_thumbnail(model, 448)
    if thumbs_dir.byte_length() > 0 and rel.byte_length() > 0:
        var flat = rel.replace("/", "__")
        _ = make_dirs(thumbs_dir)
        var px = img.pixels.copy()
        _ = write_png(thumbs_dir + "/" + flat + ".png", px, img.width, img.height, False)
    return img^


struct AssetIndex(Movable):
    var model_name: String
    var dim: Int
    var image_size: Int
    var logit_scale: Float32
    var logit_bias: Float32
    var paths: List[String]
    var kinds: List[Int]
    var sizes: List[Int]
    var mtimes: List[Int]
    var emb: List[Float32]

    def __init__(out self, model_name: String, dim: Int, image_size: Int, logit_scale: Float32, logit_bias: Float32):
        self.model_name = model_name
        self.dim = dim
        self.image_size = image_size
        self.logit_scale = logit_scale
        self.logit_bias = logit_bias
        self.paths = List[String]()
        self.kinds = List[Int]()
        self.sizes = List[Int]()
        self.mtimes = List[Int]()
        self.emb = List[Float32]()

    def count(self) -> Int:
        return len(self.paths)

    def find(self, rel: String) -> Int:
        for i in range(len(self.paths)):
            if self.paths[i] == rel:
                return i
        return -1

    def add(mut self, rel: String, kind: Int, size: Int, mtime: Int, embedding: List[Float32]) raises:
        if len(embedding) != self.dim:
            raise Error("siglip search: embedding has " + String(len(embedding)) + " values, index uses " + String(self.dim))
        var at = self.find(rel)
        if at >= 0:
            self.kinds[at] = kind
            self.sizes[at] = size
            self.mtimes[at] = mtime
            for d in range(self.dim):
                self.emb[at * self.dim + d] = embedding[d]
            return
        self.paths.append(rel)
        self.kinds.append(kind)
        self.sizes.append(size)
        self.mtimes.append(mtime)
        self.emb.extend(embedding.copy())

    def remove_at(mut self, i: Int):
        var D = self.dim
        var last = len(self.paths) - 1
        if i != last:
            self.paths[i] = self.paths[last]
            self.kinds[i] = self.kinds[last]
            self.sizes[i] = self.sizes[last]
            self.mtimes[i] = self.mtimes[last]
            for d in range(D):
                self.emb[i * D + d] = self.emb[last * D + d]
        _ = self.paths.pop()
        _ = self.kinds.pop()
        _ = self.sizes.pop()
        _ = self.mtimes.pop()
        self.emb.resize(last * D, 0.0)

    # ------------------------------------------------------------------ cache file

    def serialize(self) -> List[UInt8]:
        var w = BlobWriter()
        w.begin(fourcc("SGIX"), _CACHE_VERSION)
        w.text(self.model_name)
        w.integer(self.dim)
        w.integer(self.image_size)
        w.f32(self.logit_scale)
        w.f32(self.logit_bias)
        w.integer(len(self.paths))
        for i in range(len(self.paths)):
            w.text(self.paths[i])
            w.integer(self.kinds[i])
            w.integer(self.sizes[i])
            w.integer(self.mtimes[i])
            w.write_bytes_raw(Int(self.emb.unsafe_ptr()) + i * self.dim * 4, self.dim * 4)
        w.end()
        return w.data.copy()

    @staticmethod
    def deserialize(var data: List[UInt8]) raises -> AssetIndex:
        var r = BlobReader(data^)
        _ = r.enter(fourcc("SGIX"), _CACHE_VERSION)
        var name = r.text()
        var dim = r.integer()
        var size = r.integer()
        var scale = r.f32()
        var bias = r.f32()
        var idx = AssetIndex(name, dim, size, scale, bias)
        var n = r.integer()
        if dim <= 0 or n < 0 or n * dim * 4 > r.remaining():
            raise Error("siglip search: corrupt cache header")
        idx.emb = List[Float32](length=n * dim, fill=0.0)
        for i in range(n):
            idx.paths.append(r.text())
            idx.kinds.append(r.integer())
            idx.sizes.append(r.integer())
            idx.mtimes.append(r.integer())
            if not r.read_bytes_raw(Int(idx.emb.unsafe_ptr()) + i * dim * 4, dim * 4):
                raise Error("siglip search: truncated cache")
        r.leave()
        if r.failed:
            raise Error("siglip search: corrupt cache")
        return idx^

    def save(self, path: String) -> Bool:
        var data = self.serialize()
        return write_bytes(path, data)

    @staticmethod
    def open(path: String, model: SiglipModel) -> AssetIndex:
        """The cache at `path` if it was written for this model, else an empty index."""
        var fresh = AssetIndex(model.name, model.dim, model.image_size(), model.logit_scale, model.logit_bias)
        var data = read_bytes(path)
        if len(data) == 0:
            return fresh^
        try:
            var idx = AssetIndex.deserialize(data^)
            if idx.model_name == model.name and idx.dim == model.dim and idx.image_size == model.image_size():
                return idx^
        except:
            pass
        return fresh^

    @staticmethod
    def open_cache(path: String) raises -> AssetIndex:
        """Just the cache (no model needed), e.g. for a text-only query tool."""
        var data = read_bytes(path)
        if len(data) == 0:
            raise Error("siglip search: cannot read " + path)
        return AssetIndex.deserialize(data^)

    # ------------------------------------------------------------------ indexing

    def update(mut self, mut model: SiglipModel, root: String, batch: Int = 16,
               thumbs_dir: String = "") raises -> IndexReport:
        """Brings the index up to date with the PNG / GLB files under `root`."""
        var t0 = perf_counter_ns()
        var rep = IndexReport()
        var files = scan_assets(root)
        rep.scanned = len(files)
        # drop vanished files
        var i = 0
        while i < len(self.paths):
            var keep = False
            for f in files:
                if f == self.paths[i]:
                    keep = True
                    break
            if keep:
                i += 1
            else:
                self.remove_at(i)
                rep.removed += 1
        var S = model.image_size()
        var pend_rel = List[String]()
        var pend_kind = List[Int]()
        var pend_size = List[Int]()
        var pend_mtime = List[Int]()
        var pixels = List[Float32]()
        for f in files:
            var full = root + "/" + f
            var st = file_stamp(full)
            var at = self.find(f)
            if at >= 0 and self.sizes[at] == st[0] and self.mtimes[at] == st[1]:
                rep.reused += 1
                continue
            var kind = asset_kind(f)
            var img = load_asset_image(full, kind, thumbs_dir, f)
            if img.width <= 0 or img.height <= 0 or len(img.pixels) < img.width * img.height:
                rep.failed += 1
                continue
            pixels.extend(siglip_preprocess(img, S))
            pend_rel.append(f)
            pend_kind.append(kind)
            pend_size.append(st[0])
            pend_mtime.append(st[1])
            if len(pend_rel) >= batch:
                self._flush(model, pend_rel, pend_kind, pend_size, pend_mtime, pixels)
                rep.embedded += batch
        if len(pend_rel) > 0:
            rep.embedded += len(pend_rel)
            self._flush(model, pend_rel, pend_kind, pend_size, pend_mtime, pixels)
        rep.seconds = Float64(perf_counter_ns() - t0) / 1e9
        return rep^

    def _flush(mut self, mut model: SiglipModel, mut rel: List[String], mut kind: List[Int], mut size: List[Int],
               mut mtime: List[Int], mut pixels: List[Float32]) raises:
        var n = len(rel)
        var e = model.embed_pixels(pixels, n)
        var D = model.dim
        for k in range(n):
            var row = List[Float32](length=D, fill=0.0)
            for d in range(D):
                row[d] = e[k * D + d]
            self.add(rel[k], kind[k], size[k], mtime[k], row)
        rel.clear()
        kind.clear()
        size.clear()
        mtime.clear()
        pixels.clear()

    # ------------------------------------------------------------------ queries

    def search_embedding(self, query: List[Float32], k: Int) -> List[AssetHit]:
        """Top-`k` assets for a normalised text embedding, best first."""
        var D = self.dim
        var n = len(self.paths)
        var s = exp(self.logit_scale)
        var logits = List[Float32](length=n, fill=0.0)
        for i in range(n):
            var acc: Float32 = 0
            for d in range(D):
                acc += query[d] * self.emb[i * D + d]
            logits[i] = acc * s + self.logit_bias
        # stable insertion into a top-k list
        var top = List[Int]()
        for i in range(n):
            var pos = len(top)
            while pos > 0 and logits[top[pos - 1]] < logits[i]:
                pos -= 1
            if pos < k:
                top.insert(pos, i)
                if len(top) > k:
                    _ = top.pop()
        var out = List[AssetHit]()
        for t in top:
            out.append(AssetHit(self.paths[t], self.kinds[t], logits[t], sigmoid(logits[t])))
        return out^

    def search(self, mut model: SiglipModel, query: String, k: Int = 10) raises -> List[AssetHit]:
        var texts = List[String]()
        texts.append(query)
        var q = model.embed_texts(texts)
        return self.search_embedding(q, k)

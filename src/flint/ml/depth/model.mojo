"""`DepthModel`: one loader and `predict` for every `flint.ml.depth` checkpoint.

`DepthModel.load(dir)` recognises the folder by its files:

  config.json with "DepthAnythingForDepthEstimation"   Depth Anything V2 (`transformers` layout)
  config.json with "model_name": "da3..."               Depth Anything 3 (any-view DualDPT with
                                                        camera decoder, or mono / metric DPT
                                                        with a sky head)
  model.pt (MoGe `model_config` pickle)                 MoGe-2 (points, normals, mask, scale)

Outputs (`DepthResult`):

  DA2           relative inverse depth (disparity) at the input size
  DA3 any-view  depth (shared relative scale across views) and confidence at the network size,
                camera extrinsics (world-to-camera, OpenCV) and pixel intrinsics per view
  DA3 mono      relative depth with sky pixels pushed to the 99th percentile of the rest
  DA3 metric    metres when a focal length in pixels is given (`focal * net / 300`), else the
                network's canonical depth at focal 300
  MoGe-2        metric depth, metric camera-space points, unit normals, validity mask and
                normalised intrinsics at the input size
"""

from std.math import sqrt, exp, tan, atan
from flint.platform.os import read_bytes, file_size
from flint.assets.json import parse_json, JsonDocument
from flint.ml.diffusion.nn import Runtime, Act, Linear
from flint.ml.depth.weights import Weights
from flint.ml.depth.layers import load_linear, relu_, take_tokens
from flint.ml.depth.vit import VitConfig, DinoVit, vit_size, VIT_HF, VIT_DINO, POS_SIZE, POS_OFFSET
from flint.ml.depth.dpt import DptHead, DptConfig, DptOutput, DPT_HF, DPT_DA3
from flint.ml.depth.moge import MogeModel, MogeRaw, MogeGeometry, moge_geometry, base_grid
from flint.ml.depth.image import (
    Rgb8, Planar, preprocess_da2, preprocess_da3, preprocess_moge, da3_resize, to_planar,
)
from flint.ml.depth.layers import resize_bilinear

comptime FAMILY_DA2: Int = 0
comptime FAMILY_DA3: Int = 1
comptime FAMILY_MOGE: Int = 2


def _file_exists(path: String) -> Bool:
    return file_size(path) > 0


struct DepthResult(Movable):
    var family: Int
    var views: Int
    var w: Int                       # map size
    var h: Int
    var depth: List[Float32]         # [views, h, w] (DA2: disparity; masked / sky-free MoGe: inf)
    var is_disparity: Bool
    var metric: Bool
    var conf: List[Float32]          # [views, h, w] (DA3 any-view)
    var sky: List[Float32]           # [views, h, w] (DA3 mono / metric: sky score, >= 0.3 is sky)
    var normal: List[Float32]        # [views, h, w, 3] (MoGe)
    var points: List[Float32]        # [views, h, w, 3] camera space (MoGe)
    var mask: List[Bool]             # [views, h, w] valid pixels
    var has_camera: Bool
    var extrinsics: List[Float64]    # [views, 3, 4] world-to-camera
    var intrinsics: List[Float64]    # [views, 3, 3] in pixels of the map
    var ref_view: Int
    var raw: List[Float32]           # [views, h, w, k] network output before activation (parity)
    var depth_head: List[Float32]    # DA3: depth before the sky handling / metric scaling
    var normal_raw: List[Float32]    # MoGe: normals before masking
    var prob: List[Float32]          # MoGe: mask probability
    var scale: Float64               # MoGe: metric scale

    def __init__(out self):
        self.family = 0
        self.views = 1
        self.w = 0
        self.h = 0
        self.depth = List[Float32]()
        self.is_disparity = False
        self.metric = False
        self.conf = List[Float32]()
        self.sky = List[Float32]()
        self.normal = List[Float32]()
        self.points = List[Float32]()
        self.mask = List[Bool]()
        self.has_camera = False
        self.extrinsics = List[Float64]()
        self.intrinsics = List[Float64]()
        self.ref_view = 0
        self.raw = List[Float32]()
        self.depth_head = List[Float32]()
        self.normal_raw = List[Float32]()
        self.prob = List[Float32]()
        self.scale = 1.0

    def fx(self, v: Int = 0) -> Float64:
        return self.intrinsics[9 * v] if self.has_camera else 0.0


struct DepthModel(Movable):
    var family: Int
    var name: String
    var rt: Runtime
    var w: Weights
    var vit: DinoVit
    var dpt: DptHead
    var moge: MogeModel
    var has_moge: Bool
    var any_view: Bool               # DA3 DualDPT (depth + confidence, camera decoder)
    var has_cam: Bool
    var cam: List[Linear]            # backbone.0, backbone.2, fc_t, fc_qvec, fc_fov.0
    var input_size: Int              # DA2 518 / DA3 504

    def __init__(out self, var rt: Runtime, var w: Weights, family: Int):
        self.family = family
        self.name = String()
        self.rt = rt^
        self.w = w^
        self.vit = DinoVit(VitConfig())
        self.dpt = DptHead(DptConfig())
        self.moge = MogeModel(DinoVit(VitConfig()))
        self.has_moge = False
        self.any_view = False
        self.has_cam = False
        self.cam = List[Linear]()
        self.input_size = 518

    @staticmethod
    def load(dir: String, use_gpu: Bool = True) raises -> DepthModel:
        var rt = Runtime(use_gpu)
        var cfg_path = dir + "/config.json"
        var cfg_bytes = read_bytes(cfg_path)
        if len(cfg_bytes) == 0:
            var pt = dir + "/model.pt"
            if not _file_exists(pt):
                raise Error("depth: " + dir + " has neither config.json nor model.pt")
            var w = Weights.open(pt)
            var m = DepthModel(rt^, w^, FAMILY_MOGE)
            m.name = "moge-2-" + m.w.cfg_str("encoder.backbone")
            m.moge = MogeModel.load(m.rt, m.w)
            m.has_moge = True
            return m^
        var doc = parse_json(cfg_bytes)
        if not doc.valid:
            raise Error("depth: cannot parse " + cfg_path)
        var model_name = doc.text(doc.object_get(doc.root, "model_name"))
        if model_name != "":
            return DepthModel._load_da3(rt^, dir, doc, model_name)
        var arch = doc.object_get(doc.root, "architectures")
        if doc.text(doc.at(arch, 0)) == "DepthAnythingForDepthEstimation":
            return DepthModel._load_da2(rt^, dir, doc)
        raise Error("depth: unrecognised model in " + dir)

    @staticmethod
    def _load_da2(var rt: Runtime, dir: String, doc: JsonDocument) raises -> DepthModel:
        var w = Weights.open(dir + "/model.safetensors")
        var m = DepthModel(rt^, w^, FAMILY_DA2)
        var bb = doc.object_get(doc.root, "backbone_config")
        var hidden = doc.integer(doc.object_get(bb, "hidden_size"), 384)
        var vc = VitConfig()
        vit_size("vits" if hidden == 384 else ("vitb" if hidden == 768 else "vitl"), vc)
        vc.heads = doc.integer(doc.object_get(bb, "num_attention_heads"), vc.heads)
        vc.style = VIT_HF
        vc.prefix = "backbone."
        vc.pos_mode = POS_SIZE
        vc.eps = 1e-6
        vc.final_eps = 1e-6
        var oi = doc.object_get(bb, "out_indices")
        for i in range(doc.array_size(oi)):
            vc.out_layers.append(doc.integer(doc.at(oi, i)) - 1)
        m.vit = DinoVit.load(m.rt, m.w, vc^)
        var dc = DptConfig()
        dc.style = DPT_HF
        dc.dim_in = hidden
        dc.features = doc.integer(doc.object_get(doc.root, "fusion_hidden_size"), 64)
        var nh = doc.object_get(doc.root, "neck_hidden_sizes")
        dc.out_channels = List[Int]()
        for i in range(doc.array_size(nh)):
            dc.out_channels.append(doc.integer(doc.at(nh, i)))
        dc.out_dim = 1
        m.dpt = DptHead.load(m.rt, m.w, dc^)
        m.name = "depth-anything-v2-" + String(hidden)
        m.input_size = 518
        return m^

    @staticmethod
    def _load_da3(var rt: Runtime, dir: String, doc: JsonDocument, model_name: String) raises -> DepthModel:
        var w = Weights.open(dir + "/model.safetensors")
        var m = DepthModel(rt^, w^, FAMILY_DA3)
        m.name = model_name
        var c = doc.object_get(doc.root, "config")
        var net = doc.object_get(c, "net")
        var vc = VitConfig()
        vit_size(doc.text(doc.object_get(net, "name")), vc)
        vc.style = VIT_DINO
        vc.prefix = "model.backbone.pretrained."
        vc.pos_mode = POS_OFFSET
        vc.eps = 1e-6
        vc.final_eps = 1e-5
        vc.qk_eps = 1e-5
        var ol = doc.object_get(net, "out_layers")
        for i in range(doc.array_size(ol)):
            vc.out_layers.append(doc.integer(doc.at(ol, i)))
        vc.alt_start = doc.integer(doc.object_get(net, "alt_start"), -1)
        vc.qknorm_start = doc.integer(doc.object_get(net, "qknorm_start"), -1)
        vc.rope_start = doc.integer(doc.object_get(net, "rope_start"), -1)
        vc.cat_token = doc.boolean(doc.object_get(net, "cat_token"), False)
        m.vit = DinoVit.load(m.rt, m.w, vc^)
        var head = doc.object_get(c, "head")
        var hname = doc.text(doc.object_get(doc.object_get(head, "__object__"), "name"))
        var dc = DptConfig()
        dc.style = DPT_DA3
        dc.prefix = "model.head."
        dc.dim_in = doc.integer(doc.object_get(head, "dim_in"), 1024)
        dc.features = doc.integer(doc.object_get(head, "features"), 256)
        var oc = doc.object_get(head, "out_channels")
        dc.out_channels = List[Int]()
        for i in range(doc.array_size(oc)):
            dc.out_channels.append(doc.integer(doc.at(oc, i)))
        dc.out_dim = doc.integer(doc.object_get(head, "output_dim"), 1)
        if hname == "DualDPT":
            m.any_view = True
            dc.in_norm = True
            dc.pos_embed = True
            dc.sky = False
        elif hname == "DPT":
            dc.sky = True
        else:
            raise Error("depth: unsupported Depth Anything 3 head " + hname + " (DPT / DualDPT)")
        m.dpt = DptHead.load(m.rt, m.w, dc^)
        if m.w.has("model.cam_dec.fc_t.weight"):
            m.has_cam = True
            m.cam.append(load_linear(m.rt, m.w, "model.cam_dec.backbone.0"))
            m.cam.append(load_linear(m.rt, m.w, "model.cam_dec.backbone.2"))
            m.cam.append(load_linear(m.rt, m.w, "model.cam_dec.fc_t"))
            m.cam.append(load_linear(m.rt, m.w, "model.cam_dec.fc_qvec"))
            m.cam.append(load_linear(m.rt, m.w, "model.cam_dec.fc_fov.0"))
        m.input_size = 504
        return m^

    def device(self) raises -> String:
        return self.rt.device_name()

    # ------------------------------------------------------------------ prediction

    def predict(mut self, img: Rgb8, focal_px: Float64 = 0.0) raises -> DepthResult:
        var imgs = List[Rgb8]()
        imgs.append(img.copy())
        return self.predict_views(imgs, focal_px)

    def predict_views(mut self, imgs: List[Rgb8], focal_px: Float64 = 0.0) raises -> DepthResult:
        """Depth for one image, or several views of one scene (Depth Anything 3 any-view models
        attend across views and return a camera per view). `focal_px`: the DA3 metric model's
        focal length in pixels of the original image (0: canonical output)."""
        var S = len(imgs)
        if S == 0:
            raise Error("depth: no images")
        if S > 1 and not (self.family == FAMILY_DA3 and self.any_view):
            raise Error("depth: only Depth Anything 3 any-view models take several views")
        if self.family == FAMILY_MOGE:
            var img = imgs[0].copy()
            var g = base_grid(self.moge.num_tokens(), img.h, img.w)
            var p = preprocess_moge(img, g[0], g[1], self.moge.mean, self.moge.std)
            return self.run_moge(p.data, g[0], g[1], img.h, img.w)
        if self.family == FAMILY_DA2:
            var img = imgs[0].copy()
            var p = preprocess_da2(img, self.input_size)
            return self.run_da(p.data, 1, p.h, p.w, img.h, img.w, 0.0)
        var all = List[Float32]()
        var H = 0
        var W = 0
        for s in range(S):
            var p = preprocess_da3_planar(imgs[s], self.input_size)
            if s == 0:
                H = p.h
                W = p.w
            elif p.h != H or p.w != W:
                raise Error("depth: all views must have the same size after resizing")
            all.extend(p.data.copy())
        var f = focal_px
        if f > 0:
            f = f * Float64(W) / Float64(imgs[0].w)
        return self.run_da(all, S, H, W, H, W, f)

    def run_da(mut self, pixels: List[Float32], S: Int, H: Int, W: Int, out_h: Int, out_w: Int,
               focal_net: Float64) raises -> DepthResult:
        """Depth Anything V2 / 3 on S normalised images [S, 3, H, W]; DA2 maps are resized
        (bilinear, align_corners) to out_h x out_w, DA3 maps stay at H x W."""
        var feats = self.vit.forward(self.rt, pixels, S, H, W)
        var P = self.vit.cfg.patch
        var N = (H // P) * (W // P)
        var pts = List[Act]()
        for i in range(len(feats)):
            pts.append(take_tokens(self.rt, feats[i], 1, N))
        while len(feats) > 0:
            var f = feats.pop()
            self.rt.free(f^)
        var o = self.dpt.forward(self.rt, pts, H, W)
        while len(pts) > 0:
            var t = pts.pop()
            self.rt.free(t^)
        var r = DepthResult()
        r.views = S
        r.ref_view = self.vit.ref_view
        r.raw = o.main.copy()
        var hw = o.h * o.w
        var od = self.dpt.cfg.out_dim
        if self.family == FAMILY_DA2:
            r.family = FAMILY_DA2
            r.is_disparity = True
            var d = List[Float32](length=hw, fill=0.0)
            for i in range(hw):
                d[i] = max(o.main[i * od], Float32(0))
            var a = self.rt.upload(1, o.h, o.w, 1, d)
            var b = resize_bilinear(self.rt, a, out_h, out_w, True)
            self.rt.free(a^)
            r.depth = self.rt.download(b)
            self.rt.free(b^)
            r.w = out_w
            r.h = out_h
            r.mask = List[Bool](length=out_w * out_h, fill=True)
            return r^
        r.family = FAMILY_DA3
        r.w = o.w
        r.h = o.h
        r.depth = List[Float32](length=S * hw, fill=0.0)
        for i in range(S * hw):
            r.depth[i] = exp(o.main[i * od])
        r.mask = List[Bool](length=S * hw, fill=True)
        if self.any_view and od >= 2:
            r.conf = List[Float32](length=S * hw, fill=0.0)
            for i in range(S * hw):
                r.conf[i] = exp(o.main[i * od + 1]) + 1
        r.depth_head = r.depth.copy()
        if len(o.sky) == S * hw:
            r.sky = List[Float32](length=S * hw, fill=0.0)
            for i in range(S * hw):
                r.sky[i] = max(o.sky[i], Float32(0))
            sky_to_far(r.depth, r.sky, r.mask)
        if self.has_cam:
            self._camera(r, S, o.h, o.w)
        if not self.any_view and focal_net > 0:
            for i in range(S * hw):
                r.depth[i] = Float32(Float64(r.depth[i]) * focal_net / 300.0)
            r.metric = True
        return r^

    def _camera(mut self, mut r: DepthResult, S: Int, H: Int, W: Int) raises:
        var C2 = len(self.vit.raw0) // S
        var x = self.rt.upload(S, 1, 1, C2, self.vit.raw0)
        for i in range(2):
            var y = self.rt.linear(x, self.cam[i])
            self.rt.free(x^)
            relu_(self.rt, y)
            x = y^
        var t = self.rt.linear(x, self.cam[2])
        var q = self.rt.linear(x, self.cam[3])
        var fv = self.rt.linear(x, self.cam[4])
        relu_(self.rt, fv)
        self.rt.free(x^)
        var ht = self.rt.download(t)
        var hq = self.rt.download(q)
        var hf = self.rt.download(fv)
        self.rt.free(t^)
        self.rt.free(q^)
        self.rt.free(fv^)
        r.has_camera = True
        r.extrinsics = List[Float64](length=S * 12, fill=0.0)
        r.intrinsics = List[Float64](length=S * 9, fill=0.0)
        for s in range(S):
            var R = quat_to_mat(Float64(hq[4 * s]), Float64(hq[4 * s + 1]), Float64(hq[4 * s + 2]), Float64(hq[4 * s + 3]))
            var T0 = Float64(ht[3 * s])
            var T1 = Float64(ht[3 * s + 1])
            var T2 = Float64(ht[3 * s + 2])
            # w2c = inverse([R | T]) = [R^T | -R^T T]
            for i in range(3):
                for j in range(3):
                    r.extrinsics[s * 12 + i * 4 + j] = R[j * 3 + i]
                r.extrinsics[s * 12 + i * 4 + 3] = -(R[0 * 3 + i] * T0 + R[1 * 3 + i] * T1 + R[2 * 3 + i] * T2)
            var fov_h = Float64(hf[2 * s])
            var fov_w = Float64(hf[2 * s + 1])
            var fy = (Float64(H) / 2.0) / max(tan(fov_h / 2.0), 1e-6)
            var fx = (Float64(W) / 2.0) / max(tan(fov_w / 2.0), 1e-6)
            r.intrinsics[s * 9 + 0] = fx
            r.intrinsics[s * 9 + 2] = Float64(W) / 2.0
            r.intrinsics[s * 9 + 4] = fy
            r.intrinsics[s * 9 + 5] = Float64(H) / 2.0
            r.intrinsics[s * 9 + 8] = 1.0

    def run_moge(mut self, pixels: List[Float32], bh: Int, bw: Int, H: Int, W: Int) raises -> DepthResult:
        var raw = self.moge.forward(self.rt, pixels, bh, bw, H, W)
        var g = moge_geometry(raw)
        var r = DepthResult()
        r.family = FAMILY_MOGE
        r.w = W
        r.h = H
        r.raw = raw.points.copy()
        r.normal_raw = raw.normal.copy()
        r.prob = raw.mask.copy()
        r.scale = raw.metric_scale
        r.depth = g.depth.copy()
        r.points = g.points.copy()
        r.normal = g.normal.copy()
        r.mask = g.mask.copy()
        r.metric = raw.has_scale
        r.has_camera = True
        r.extrinsics = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0]
        r.intrinsics = [g.fx * Float64(W), 0, 0.5 * Float64(W), 0, g.fy * Float64(H), 0.5 * Float64(H), 0, 0, 1.0]
        return r^


def preprocess_da3_planar(img: Rgb8, target: Int) -> Planar:
    return preprocess_da3(img, target)


def quat_to_mat(i: Float64, j: Float64, k: Float64, r: Float64) -> List[Float64]:
    """Depth Anything 3 `quat_to_mat` (x, y, z, w) -> row-major 3 x 3."""
    var two_s = 2.0 / (i * i + j * j + k * k + r * r)
    return [
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ]


def quantile(var v: List[Float32], q: Float64) -> Float32:
    """`torch.quantile` (linear interpolation)."""
    var n = len(v)
    if n == 0:
        return 0
    sort(v)
    var pos = q * Float64(n - 1)
    var lo = Int(pos)
    var hi = min(lo + 1, n - 1)
    var t = Float32(pos - Float64(lo))
    return v[lo] + (v[hi] - v[lo]) * t


def sky_to_far(mut depth: List[Float32], sky: List[Float32], mut mask: List[Bool]):
    """Depth Anything 3 mono sky handling: pixels with sky >= 0.3 take the 99th percentile of
    the other pixels' depth (all of them, where the reference samples 100 000 when there are
    more); skipped when either side has 10 pixels or fewer."""
    var n = len(depth)
    var keep = List[Float32]()
    for i in range(n):
        if sky[i] < 0.3:
            keep.append(depth[i])
    if len(keep) <= 10 or n - len(keep) <= 10:
        return
    var far = quantile(keep^, 0.99)
    for i in range(n):
        if sky[i] >= 0.3:
            depth[i] = far
            mask[i] = False

"""Generic latent-diffusion sampling for `flint.ml.diffusion`.

`Denoiser` owns a `UNet` and a `Scheduler` and turns a latent (host, NCHW, the layout the
diffusers schedulers and `torch.randn` use) into the next one:

- classifier-free guidance (uncond / cond batched in one UNet call, `guidance <= 1` runs one batch);
- an optional channel-concatenated condition (inpainting / sketch / depth style UNets);
- an optional timestep-condition vector (LCM guidance embedding);
- tiling: from step `tile_from` on every convolution wraps around (seamless textures).

`sample` runs the whole loop from seeded Gaussian noise. The conditioning tensors are plain host
lists so other pipelines (texture painting, LoRA-patched UNets) can build them however they like.
"""

from flint.ml.diffusion.nn import Runtime
from flint.ml.diffusion.unet import UNet
from flint.ml.diffusion.schedulers import Scheduler
from flint.ml.diffusion.rng import TorchGenerator


def nchw_to_nhwc(x: List[Float32], b: Int, c: Int, h: Int, w: Int) -> List[Float32]:
    var o = List[Float32](length=len(x), fill=0.0)
    var hw = h * w
    for bi in range(b):
        for ci in range(c):
            var src = (bi * c + ci) * hw
            for p in range(hw):
                o[(bi * hw + p) * c + ci] = x[src + p]
    return o^


def nhwc_to_nchw(x: List[Float32], b: Int, c: Int, h: Int, w: Int) -> List[Float32]:
    var o = List[Float32](length=len(x), fill=0.0)
    var hw = h * w
    for bi in range(b):
        for ci in range(c):
            var dst = (bi * c + ci) * hw
            for p in range(hw):
                o[dst + p] = x[(bi * hw + p) * c + ci]
    return o^


struct Denoiser(Movable):
    var unet: UNet
    var sched: Scheduler
    var tokens: Int
    var width: Int
    var ctx_cond: List[Float32]       # [tokens, width]
    var ctx_uncond: List[Float32]
    var guidance: Float32
    var concat_c: Int                 # channels appended after the latent (0 = none)
    var concat_cond: List[Float32]    # NHWC [h, w, concat_c]
    var concat_uncond: List[Float32]
    var tcond: List[Float32]          # timestep condition (empty = none)
    var tile_from: Int                # first step with wrap-around convolutions (-1 = never)

    def __init__(out self, var unet: UNet, var sched: Scheduler):
        self.unet = unet^
        self.sched = sched^
        self.tokens = 0
        self.width = 0
        self.ctx_cond = List[Float32]()
        self.ctx_uncond = List[Float32]()
        self.guidance = 1.0
        self.concat_c = 0
        self.concat_cond = List[Float32]()
        self.concat_uncond = List[Float32]()
        self.tcond = List[Float32]()
        self.tile_from = -1

    def set_context(mut self, var cond: List[Float32], var uncond: List[Float32], tokens: Int, width: Int,
                    guidance: Float32):
        """Cross-attention context [tokens, width]; `uncond` may be empty when guidance <= 1."""
        self.ctx_cond = cond^
        self.ctx_uncond = uncond^
        self.tokens = tokens
        self.width = width
        self.guidance = guidance

    def set_concat(mut self, var cond: List[Float32], var uncond: List[Float32], channels: Int):
        self.concat_cond = cond^
        self.concat_uncond = uncond^
        self.concat_c = channels

    def cfg(self) -> Bool:
        return self.guidance > 1.0 and len(self.ctx_uncond) > 0

    def predict(mut self, mut rt: Runtime, x: List[Float32], c: Int, h: Int, w: Int, i: Int) raises -> List[Float32]:
        """Guided model output (NCHW [1, c, h, w]) for latent `x` at step `i`."""
        var xs = self.sched.scale_model_input(x, i)
        var xh = nchw_to_nhwc(xs, 1, c, h, w)
        var B = 2 if self.cfg() else 1
        var C = c + self.concat_c
        var hw = h * w
        var inp = List[Float32](length=B * hw * C, fill=0.0)
        for b in range(B):
            var uncond = B == 2 and b == 0
            for p in range(hw):
                var o = (b * hw + p) * C
                for k in range(c):
                    inp[o + k] = xh[p * c + k]
                for k in range(self.concat_c):
                    inp[o + c + k] = self.concat_uncond[p * self.concat_c + k] if uncond else \
                        self.concat_cond[p * self.concat_c + k]
        var ctx = List[Float32]()
        if B == 2:
            ctx = self.ctx_uncond.copy()
        for v in self.ctx_cond:
            ctx.append(v)
        var xa = rt.upload(B, h, w, C, inp)
        var ca = rt.upload(B, self.tokens, 1, self.width, ctx)
        var tile = self.tile_from >= 0 and i >= self.tile_from
        var o = self.unet.forward(rt, xa, self.sched.timestep(i), ca, self.tcond, tile)
        rt.free(xa^)
        rt.free(ca^)
        var oh = rt.download(o)
        var co = o.c
        rt.free(o^)
        var eps: List[Float32]
        if B == 2:
            eps = List[Float32](length=hw * co, fill=0.0)
            for k in range(hw * co):
                var u = oh[k]
                eps[k] = u + self.guidance * (oh[hw * co + k] - u)
        else:
            eps = oh^
        return nhwc_to_nchw(eps, 1, co, h, w)

    def sample(mut self, mut rt: Runtime, mut gen: TorchGenerator, c: Int, h: Int, w: Int,
               steps: Int) raises -> List[Float32]:
        """Seeded txt2img loop; returns the final latent (NCHW [1, c, h, w], unscaled)."""
        self.sched.set_timesteps(steps)
        var x = gen.randn(c * h * w)
        var s = self.sched.init_noise_sigma
        if s != 1.0:
            for k in range(len(x)):
                x[k] *= s
        for i in range(self.sched.num_steps()):
            var eps = self.predict(rt, x, c, h, w, i)
            x = self.sched.step(eps, i, x, gen)
        return x^

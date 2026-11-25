"""Diffusion samplers (`flint.ml.diffusion.schedulers`): DDIM, Euler and LCM, as in diffusers.

`Scheduler` reproduces diffusers' `DDIMScheduler` (eta = 0), `EulerDiscreteScheduler`
(s_churn = 0) and `LCMScheduler` for epsilon / v-prediction models, including the float32
arithmetic order (betas from a float32 `linspace`, `alphas_cumprod` accumulated in double and
stored as float32 like ATen's CPU `cumprod`), so per-step latents match the Python reference to
float rounding. Usage:

    var s = Scheduler.from_config(json_bytes, SCHED_LCM)   # or Scheduler(SCHED_DDIM, ...)
    s.set_timesteps(4)
    var x = scale(gen.randn(n), s.init_noise_sigma)
    for i in range(s.num_steps()):
        var inp = s.scale_model_input(x, i)
        var eps = unet(inp, s.timestep(i), ...)
        x = s.step(eps, i, x, gen)          # LCM draws its re-noise from `gen`

`guidance_embedding(w, dim)` is the LCM w-conditioning vector and `timestep_embedding` the
sinusoidal UNet time projection.
"""

from std.ffi import external_call
from std.math import sqrt, log
from flint.assets.json import parse_json
from flint.ml.diffusion.rng import TorchGenerator

@always_inline
def _sinf(x: Float32) -> Float32:
    return external_call["sinf", Float32](x)


@always_inline
def _cosf(x: Float32) -> Float32:
    return external_call["cosf", Float32](x)


@always_inline
def _expf(x: Float32) -> Float32:
    return external_call["expf", Float32](x)


comptime SCHED_DDIM: Int = 0
comptime SCHED_EULER: Int = 1
comptime SCHED_LCM: Int = 2

comptime PRED_EPSILON: Int = 0
comptime PRED_V: Int = 1
comptime PRED_SAMPLE: Int = 2

comptime SPACING_LEADING: Int = 0
comptime SPACING_TRAILING: Int = 1
comptime SPACING_LINSPACE: Int = 2


def _linspace_f32(a: Float32, b: Float32, n: Int) -> List[Float32]:
    """ATen's CPU float32 linspace (front half from `a`, back half from `b`)."""
    var out = List[Float32](length=n, fill=0.0)
    if n == 1:
        out[0] = a
        return out^
    var step = (b - a) / Float32(n - 1)
    var half = n // 2
    for i in range(n):
        if i < half:
            out[i] = a + step * Float32(i)
        else:
            out[i] = b - step * Float32(n - i - 1)
    return out^


struct Scheduler(Copyable, Movable):
    var kind: Int
    var train_steps: Int
    var betas: List[Float32]
    var alphas_cumprod: List[Float32]
    var final_alpha_cumprod: Float32
    var steps_offset: Int
    var prediction: Int
    var spacing: Int
    var original_steps: Int       # LCM
    var timestep_scaling: Float64  # LCM
    var timesteps: List[Float64]
    var sigmas: List[Float32]      # Euler (len = steps + 1)
    var init_noise_sigma: Float32

    def __init__(out self, kind: Int, beta_start: Float64 = 0.00085, beta_end: Float64 = 0.012,
                 scaled_linear: Bool = True, train_steps: Int = 1000, set_alpha_to_one: Bool = False,
                 steps_offset: Int = 1, prediction: Int = PRED_EPSILON, spacing: Int = SPACING_LEADING):
        self.kind = kind
        self.train_steps = train_steps
        if scaled_linear:
            var lin = _linspace_f32(Float32(sqrt(beta_start)), Float32(sqrt(beta_end)), train_steps)
            self.betas = List[Float32](length=train_steps, fill=0.0)
            for i in range(train_steps):
                self.betas[i] = lin[i] * lin[i]
        else:
            self.betas = _linspace_f32(Float32(beta_start), Float32(beta_end), train_steps)
        self.alphas_cumprod = List[Float32](length=train_steps, fill=0.0)
        var acc: Float64 = 1.0
        for i in range(train_steps):
            acc *= Float64(Float32(1.0) - self.betas[i])
            self.alphas_cumprod[i] = Float32(acc)
        self.final_alpha_cumprod = Float32(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        self.steps_offset = steps_offset
        self.prediction = prediction
        self.spacing = spacing
        self.original_steps = 50
        self.timestep_scaling = 10.0
        self.timesteps = List[Float64]()
        self.sigmas = List[Float32]()
        self.init_noise_sigma = 1.0

    @staticmethod
    def from_config(data: List[UInt8], kind: Int) raises -> Scheduler:
        """From a diffusers `scheduler_config.json` (any scheduler class), run as `kind`."""
        var doc = parse_json(data)
        if not doc.valid:
            raise Error("scheduler: invalid scheduler_config.json")
        var r = doc.root
        var bs = doc.number(doc.object_get(r, "beta_start"), 0.00085)
        var be = doc.number(doc.object_get(r, "beta_end"), 0.012)
        var sched = doc.text(doc.object_get(r, "beta_schedule"), "scaled_linear")
        var n = doc.integer(doc.object_get(r, "num_train_timesteps"), 1000)
        var one = doc.boolean(doc.object_get(r, "set_alpha_to_one"), False)
        var off = doc.integer(doc.object_get(r, "steps_offset"), 0)
        var pt = doc.text(doc.object_get(r, "prediction_type"), "epsilon")
        var sp = doc.text(doc.object_get(r, "timestep_spacing"), "leading")
        if sched != "scaled_linear" and sched != "linear":
            raise Error("scheduler: unsupported beta_schedule " + sched)
        var pred = PRED_EPSILON
        if pt == "v_prediction":
            pred = PRED_V
        elif pt == "sample":
            pred = PRED_SAMPLE
        var spacing = SPACING_LEADING
        if sp == "trailing":
            spacing = SPACING_TRAILING
        elif sp == "linspace":
            spacing = SPACING_LINSPACE
        var s = Scheduler(kind, bs, be, sched == "scaled_linear", n, one, off, pred, spacing)
        var os = doc.object_get(r, "original_inference_steps")
        if os >= 0:
            s.original_steps = doc.integer(os, 50)
        var ts = doc.object_get(r, "timestep_scaling")
        if ts >= 0:
            s.timestep_scaling = doc.number(ts, 10.0)
        if doc.boolean(doc.object_get(r, "rescale_betas_zero_snr"), False):
            rescale_zero_terminal_snr(s, one)
        return s^

    # ------------------------------------------------------------------ schedule

    def set_timesteps(mut self, n: Int) raises:
        self.timesteps = List[Float64]()
        if n < 1:
            raise Error("scheduler: need at least one step")
        if self.kind == SCHED_LCM:
            var k = self.train_steps // self.original_steps
            var origin = List[Int]()
            for i in range(self.original_steps, 0, -1):
                origin.append(i * k - 1)
            if n > self.original_steps:
                raise Error("scheduler: LCM steps exceed original_inference_steps")
            for i in range(n):
                # floor(linspace(0, len, n, endpoint=False))
                var idx = Int(Float64(i) * Float64(len(origin)) / Float64(n))
                self.timesteps.append(Float64(origin[idx]))
            self.init_noise_sigma = 1.0
            return
        var ints = List[Int]()
        if self.spacing == SPACING_LEADING:
            var ratio = self.train_steps // n
            for i in range(n - 1, -1, -1):
                ints.append(i * ratio + self.steps_offset)
        elif self.spacing == SPACING_TRAILING:
            var ratio = Float64(self.train_steps) / Float64(n)
            var t = Float64(self.train_steps)
            for _ in range(n):
                ints.append(Int(_round_half_even(t)) - 1)
                t -= ratio
        else:
            for i in range(n - 1, -1, -1):
                var v = Float64(i) * Float64(self.train_steps - 1) / Float64(max(n - 1, 1))
                ints.append(Int(_round_half_even(v)))
        for t in ints:
            self.timesteps.append(Float64(t))
        if self.kind == SCHED_EULER:
            self.sigmas = List[Float32]()
            var mx: Float32 = 0.0
            for t in ints:
                var ac = self.alphas_cumprod[t]
                var s = sqrt((Float32(1.0) - ac) / ac)
                self.sigmas.append(s)
                mx = max(mx, s)
            self.sigmas.append(0.0)
            if self.spacing == SPACING_LEADING:
                self.init_noise_sigma = sqrt(mx * mx + Float32(1.0))
            else:
                self.init_noise_sigma = mx
        else:
            self.init_noise_sigma = 1.0

    def num_steps(self) -> Int:
        return len(self.timesteps)

    def timestep(self, i: Int) -> Float64:
        return self.timesteps[i]

    # ------------------------------------------------------------------ per step

    def scale_model_input(self, x: List[Float32], i: Int) -> List[Float32]:
        if self.kind != SCHED_EULER:
            return x.copy()
        var s = self.sigmas[i]
        var d = sqrt(s * s + Float32(1.0))
        var out = List[Float32](length=len(x), fill=0.0)
        for k in range(len(x)):
            out[k] = x[k] / d
        return out^

    def _x0(self, eps: List[Float32], x: List[Float32], a: Float32) -> List[Float32]:
        var out = List[Float32](length=len(x), fill=0.0)
        var sa = sqrt(a)
        var sb = sqrt(Float32(1.0) - a)
        for k in range(len(x)):
            if self.prediction == PRED_EPSILON:
                out[k] = (x[k] - sb * eps[k]) / sa
            elif self.prediction == PRED_V:
                out[k] = sa * x[k] - sb * eps[k]
            else:
                out[k] = eps[k]
        return out^

    def step(self, eps: List[Float32], i: Int, x: List[Float32], mut gen: TorchGenerator) raises -> List[Float32]:
        """x_{t-1} from the model output at step i (DDIM eta 0 / Euler / LCM)."""
        var n = len(x)
        var out = List[Float32](length=n, fill=0.0)
        var t = Int(self.timesteps[i])
        if self.kind == SCHED_DDIM:
            var prev = t - self.train_steps // len(self.timesteps)
            var a = self.alphas_cumprod[t]
            var ap = self.alphas_cumprod[prev] if prev >= 0 else self.final_alpha_cumprod
            var b = Float32(1.0) - a
            var sb = sqrt(b)
            var sa = sqrt(a)
            var sap = sqrt(ap)
            var dir = sqrt(Float32(1.0) - ap)
            for k in range(n):
                var x0: Float32
                var e: Float32
                if self.prediction == PRED_EPSILON:
                    x0 = (x[k] - sb * eps[k]) / sa
                    e = eps[k]
                elif self.prediction == PRED_V:
                    x0 = sa * x[k] - sb * eps[k]
                    e = sa * eps[k] + sb * x[k]
                else:
                    x0 = eps[k]
                    e = (x[k] - sa * x0) / sb
                out[k] = sap * x0 + dir * e
            return out^
        if self.kind == SCHED_EULER:
            var sigma = self.sigmas[i]
            var dt = self.sigmas[i + 1] - sigma
            # diffusers draws (unused, s_churn = 0) noise every step; keep the stream in sync
            _ = gen.randn(n)
            for k in range(n):
                var x0: Float32
                if self.prediction == PRED_EPSILON:
                    x0 = x[k] - sigma * eps[k]
                elif self.prediction == PRED_V:
                    var c = sigma * sigma + Float32(1.0)
                    x0 = eps[k] * (-sigma / sqrt(c)) + x[k] / c
                else:
                    x0 = eps[k]
                var der = (x[k] - x0) / sigma
                out[k] = x[k] + der * dt
            return out^
        # LCM
        var a = self.alphas_cumprod[t]
        var prev_t = Int(self.timesteps[i + 1]) if i + 1 < len(self.timesteps) else t
        var ap = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        var st = Float64(t) * self.timestep_scaling
        var c_skip = Float32(0.25 / (st * st + 0.25))
        var c_out = Float32(st / sqrt(st * st + 0.25))
        var x0 = self._x0(eps, x, a)
        var last = i == len(self.timesteps) - 1
        var noise = List[Float32]()
        if not last:
            noise = gen.randn(n)
        var sap = sqrt(ap)
        var sbp = sqrt(Float32(1.0) - ap)
        for k in range(n):
            var den = c_out * x0[k] + c_skip * x[k]
            if last:
                out[k] = den
            else:
                out[k] = sap * den + sbp * noise[k]
        return out^


def rescale_zero_terminal_snr(mut s: Scheduler, set_alpha_to_one: Bool = False):
    """diffusers `rescale_zero_terminal_snr` (Lin et al. 2023, used by Marigold v1.1 and other
    v-prediction fine-tunes): shift / scale sqrt(alphas_cumprod) so the last step has zero SNR,
    then rebuild betas and alphas_cumprod, all in float32 like the PyTorch reference."""
    var n = s.train_steps
    var sq = List[Float32](length=n, fill=0.0)
    for i in range(n):
        sq[i] = sqrt(s.alphas_cumprod[i])
    var s0 = sq[0]
    var sT = sq[n - 1]
    var k = s0 / (s0 - sT)
    var bar = List[Float32](length=n, fill=0.0)
    for i in range(n):
        var v = (sq[i] - sT) * k
        bar[i] = v * v
    for i in range(n):
        var a = bar[0] if i == 0 else bar[i] / bar[i - 1]
        s.betas[i] = Float32(1.0) - a
    var acc: Float64 = 1.0
    for i in range(n):
        acc *= Float64(Float32(1.0) - s.betas[i])
        s.alphas_cumprod[i] = Float32(acc)
    s.final_alpha_cumprod = Float32(1.0) if set_alpha_to_one else s.alphas_cumprod[0]


def _round_half_even(v: Float64) -> Float64:
    var f = Float64(Int(v)) if v >= 0 else -Float64(Int(-v))
    var r = v - f
    if r > 0.5 or (r == 0.5 and Int(f) % 2 != 0):
        return f + 1.0
    if r < -0.5 or (r == -0.5 and Int(f) % 2 != 0):
        return f - 1.0
    return f


def guidance_embedding(w: Float32, dim: Int) -> List[Float32]:
    """LCM guidance-scale embedding of `w` (= guidance_scale - 1): [sin, cos] of w*1000*freqs."""
    var half = dim // 2
    var out = List[Float32](length=dim, fill=0.0)
    var e = external_call["logf", Float32](Float32(10000.0)) / Float32(half - 1)
    var ww = w * Float32(1000.0)
    for i in range(half):
        var f = _expf(Float32(i) * -e)
        var a = ww * f
        out[i] = _sinf(a)
        out[half + i] = _cosf(a)
    return out^


def timestep_embedding(t: Float64, dim: Int, flip_sin_to_cos: Bool = True, shift: Float64 = 0.0,
                       max_period: Float64 = 10000.0) -> List[Float32]:
    """diffusers `get_timestep_embedding` for one timestep (float32 math as in PyTorch)."""
    var half = dim // 2
    var out = List[Float32](length=dim, fill=0.0)
    var lm = Float32(-log(max_period))
    for i in range(half):
        var ex = lm * Float32(i) / Float32(Float64(half) - shift)
        var a = Float32(t) * _expf(ex)
        var s = _sinf(a)
        var c = _cosf(a)
        if flip_sin_to_cos:
            out[i] = c
            out[half + i] = s
        else:
            out[i] = s
            out[half + i] = c
    return out^


def cfg_combine(uncond: List[Float32], cond: List[Float32], scale: Float32) -> List[Float32]:
    """Classifier-free guidance: uncond + scale * (cond - uncond)."""
    var out = List[Float32](length=len(cond), fill=0.0)
    for k in range(len(cond)):
        out[k] = uncond[k] + scale * (cond[k] - uncond[k])
    return out^

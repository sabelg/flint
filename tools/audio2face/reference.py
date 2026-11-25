"""Reference outputs for `flint.ml.audio2face` parity checks (`tests/test_flint_v5_audio2face.mojo`).

usage: python -I tools/audio2face/reference.py MODEL_DIR in.wav OUT_PREFIX [fps] [seconds]

  MODEL_DIR   a Hugging Face nvidia/Audio2Face-3D-v2.3-Mark / v2.3.1-Claire / v2.3.1-James folder
  in.wav      any rate / channel count (16-bit PCM or float); the first `seconds` are used when given
  fps         output frame rate (default 30)
  OUT_PREFIX  receives (all little-endian float32, row-major)
     OUT_PREFIX.wav16.f32   the 16 kHz mono input, resampled with the windowed-sinc filter that
                            `flint.ml.moonshine_audio.Resampler` implements
     OUT_PREFIX.net.f32     raw network output [frames, 301] (skin PCA | tongue PCA | jaw | eyes)
     OUT_PREFIX.skin.f32    ARKit skin weights [frames, 52] in bs_skin.npz `poseNames` order
     OUT_PREFIX.tongue.f32  tongue weights [frames, 16]
     OUT_PREFIX.jaw.f32     jaw transform [frames, 16] (column-major 4x4)
     OUT_PREFIX.eyes.f32    eye rotations [frames, 6] (right x, y, 0, left x, y, 0; degrees)
     OUT_PREFIX.json        sizes, pose names, timings

This is a numpy transcription of the NVIDIA Audio2Face-3D SDK (MIT,
https://github.com/NVIDIA/Audio2Face-3D-SDK) regression path, written from
`executor_regression.cpp`, `multitrack_animator*.{cpp,cu}`, `blendshape_solver*.cpp` and
`bvls.cpp`: audio windows, PCA reconstruction, the skin / tongue / teeth / eye animators and the
bounded least-squares blend-shape solve. The network runs in onnxruntime (CPU, fp32). Emotion
input is neutral (explicit emotions all 0, implicit emotion from `source_shot`/`source_frame`);
Audio2Emotion is not used. Needs numpy, onnxruntime, soundfile.
"""
import sys, os, math, json, time
import numpy as np
import onnxruntime as ort
import soundfile as sf


# ----------------------------------------------------------------------------- audio
def kaiser(x, beta):
    return np.i0(beta * np.sqrt(np.clip(1 - x * x, 0, 1))) / np.i0(beta)


def resample(x, fin, fout, zeros=16, beta=8.6, roll=0.945):
    """Windowed-sinc polyphase resampler (same as tools/moonshine/reference.py)."""
    if fin == fout:
        return x.astype(np.float32)
    g = math.gcd(fin, fout); L = fout // g; M = fin // g
    fc = min(1.0, fout / fin) * roll
    half = int(math.ceil(zeros / fc))
    taps = 2 * half
    n_out = (len(x) * L) // M
    table = np.zeros((L, taps), np.float64)
    for p in range(L):
        frac = p / L
        k = np.arange(-half + 1, half + 1)
        d = frac - k
        w = kaiser(d / half, beta)
        table[p] = fc * np.sinc(fc * d) * w
    y = np.zeros(n_out, np.float64)
    xp = np.concatenate([np.zeros(half), x.astype(np.float64), np.zeros(half + 1)])
    for n in range(n_out):
        base = (n * M) // L; p = (n * M) % L
        seg = xp[base + 1: base + 1 + taps]
        y[n] = np.dot(table[p], seg)
    return y.astype(np.float32)


def windows(audio16, frames, fps, buffer_len, buffer_ofs, strength):
    """WindowProgress + AudioAccumulator::Read: window i starts at floor(i*16000/fps) - ofs."""
    g = math.gcd(16000, fps); num = 16000 // g; den = fps // g
    out = np.zeros((frames, buffer_len), np.float32)
    n = len(audio16)
    for i in range(frames):
        start = (i * num) // den - buffer_ofs
        a = max(start, 0); b = min(start + buffer_len, n)
        if b > a:
            out[i, a - start: b - start] = audio16[a:b]
    return out * np.float32(strength)


def frame_count(n_samples, fps, buffer_ofs):
    """Closed-track GetNbAvailableWindows: every frame whose target sample lies inside the audio."""
    g = math.gcd(16000, fps); num = 16000 // g; den = fps // g
    k = 0
    while (k * num) // den < n_samples:
        k += 1
    return k


# ----------------------------------------------------------------------------- jaw
def rigid_xform(target, source):
    """ComputeJawTransformKernel: Kabsch fit source -> target, column-major 4x4."""
    tm = target.mean(0); sm = source.mean(0)
    H = (source - sm).T @ (target - tm)  # H[r, c] = sum s_r t_c
    U, S, Vt = np.linalg.svd(H.astype(np.float64))
    R = Vt.T @ U.T  # R maps source to target
    det = np.linalg.det(R)
    R[:, 2] /= det
    t = tm - R @ sm
    M = np.eye(4)
    M[:3, :3] = R; M[:3, 3] = t
    return M.T.reshape(16).astype(np.float32)  # column-major


# ----------------------------------------------------------------------------- BVLS
def _lstsq(A, b):
    return np.linalg.lstsq(A.astype(np.float64), b.astype(np.float64), rcond=None)[0].astype(np.float32)


def constrained_min(x, A, B, on_bound):
    free = [i for i in range(len(on_bound)) if on_bound[i] == 0]
    if not free:
        return
    if len(free) == len(on_bound):
        x[:] = _lstsq(A, B)
        return
    x0 = x.copy(); x0[free] = 0
    bf = B - A @ x0
    x[free] = _lstsq(A[:, free], bf)


def cost_grad(x, A, B):
    r = A @ x - B
    return np.float32(0.5) * np.dot(r, r), A.T @ r


def solve_bvls(A, B, l, u, tol):
    n = len(B)
    x = np.zeros(n, np.float32)
    on = np.zeros(n, np.int32)
    while True:  # initialFeasiblePoint
        constrained_min(x, A, B, on)
        found = 0
        for i in range(n):
            if on[i] != 0:
                continue
            if x[i] < l[i]:
                x[i] = l[i]; on[i] = -1; found += 1
            elif x[i] > u[i]:
                x[i] = u[i]; on[i] = 1; found += 1
        if found == 0:
            break

    def update_on():
        for i in range(n):
            on[i] = -1 if x[i] == l[i] else (1 if x[i] == u[i] else 0)

    def kkt(g):
        v = g * on.astype(np.float32)
        v[on == 0] = np.abs(g[on == 0])
        return v.max()

    cost, g = cost_grad(x, A, B)
    update_on()
    opt = kkt(g); prev = opt
    for _ in range(n):
        if opt < tol:
            break
        _, g2 = cost_grad(x, A, B)
        best = -1; bv = 0.0
        for i in range(n):
            if x[i] == l[i] and g2[i] < 0 and -g2[i] > bv:
                bv = -g2[i]; best = i
            elif x[i] == u[i] and g2[i] > 0 and g2[i] > bv:
                bv = g2[i]; best = i
        if best == -1:
            break
        on[best] = 0
        while True:
            s = x.copy()
            constrained_min(s, A, B, on)
            alpha = np.float32(1.0); idx = -1; bval = 0.0
            for i in range(n):
                if x[i] == s[i]:
                    continue
                if s[i] < l[i]:
                    at = (x[i] - l[i]) / (x[i] - s[i])
                    if at < alpha:
                        idx = i; bval = l[i]; alpha = at
                elif s[i] > u[i]:
                    at = (u[i] - x[i]) / (s[i] - x[i])
                    if at < alpha:
                        idx = i; bval = u[i]; alpha = at
            x += alpha * (s - x)
            if idx != -1:
                x[idx] = bval
            if alpha == 1.0:
                break
            update_on()
        cost_new, g = cost_grad(x, A, B)
        if cost_new > cost:
            break
        cost = cost_new
        opt = kkt(g)
        if abs(prev - opt) < 1e-6:
            break
        prev = opt
    return x


def pairs(active, pair_ids):
    act = [pair_ids[i] for i in range(len(active)) if active[i]]
    groups = {}
    for i, p in enumerate(act):
        groups.setdefault(p, []).append(i)
    return [tuple(v) for k, v in groups.items() if len(v) == 2]


class BsSolver:
    """BlendshapeSolverBase::Prepare + BlendshapeSolver::SolveAsync (CPU path)."""

    def __init__(self, npz, cfg_path):
        d = np.load(npz)
        names = [s.decode() for s in d["poseNames"]]
        assert names[0] == "neutral"
        self.names = names[1:]
        neutral = d["neutral"].reshape(-1).astype(np.float32)
        deltas = np.stack([d[nm].reshape(-1) for nm in self.names], 1).astype(np.float32)  # [P, K]
        mask = d["frontalMask"] if "frontalMask" in d.files else None
        c = json.load(open(cfg_path))["blendshape_params"]
        self.active = c["bsSolveActivePoses"]; self.mult = np.array(c["bsWeightMultipliers"], np.float32)
        self.offs = np.array(c["bsWeightOffsets"], np.float32)
        L1 = c["strengthL1regularization"]; L2 = c["strengthL2regularization"]
        T = c["strengthTemporalSmoothing"]; Sy = c["strengthSymmetry"]; bb = c.get("templateBBSize", 54.7)
        if mask is not None:
            idx = np.stack([3 * mask, 3 * mask + 1, 3 * mask + 2], 1).reshape(-1)
        else:
            idx = np.arange(len(neutral))
        self.idx = idx
        self.act_idx = [i for i in range(len(self.names)) if self.active[i]]
        D = deltas[idx][:, self.act_idx]
        self.D = D; self.neutral = neutral[idx]
        k = len(self.act_idx)
        nm = neutral.reshape(-1, 3)
        scale = np.float32((np.linalg.norm(nm.max(0) - nm.min(0)) / bb) ** 2)
        self.scale = scale
        sym = pairs(self.active, c["bsSolveSymmetryPoses"])
        Smat = np.zeros((len(sym), k), np.float32)
        for r, (a, b) in enumerate(sym):
            Smat[r, a] = 1; Smat[r, b] = -1
        self.cancel = pairs(self.active, c["bsSolveCancelPoses"])
        ATA = (D.astype(np.float64).T @ D.astype(np.float64)).astype(np.float32)
        self.A = (ATA + L1 * L1 * 0.25 * scale * np.ones((k, k), np.float32)
                  + L2 * 10 * scale * np.eye(k, dtype=np.float32)
                  + T * 100 * scale * np.eye(k, dtype=np.float32)
                  + Sy * 10 * scale * (Smat.T @ Smat)).astype(np.float32)
        self.T = T
        self.prev = np.zeros(k, np.float32)
        self.atb = []   # per-frame D^T (target - neutral), for dump_solver()
        self.raw = []   # per-frame active weights before multipliers / offsets

    def solve(self, pose):
        tgt = pose.reshape(-1)[self.idx] - self.neutral
        atb = (self.D.T @ tgt).astype(np.float32)
        self.atb.append(atb)
        b = (atb + np.float32(self.T) * self.scale * self.prev).astype(np.float32)
        k = len(b)
        l = np.zeros(k, np.float32); u = np.ones(k, np.float32)
        x = solve_bvls(self.A, b, l, u, 1e-10)
        if self.cancel:
            for i, j in self.cancel:
                u[j if x[i] >= x[j] else i] = 1e-10
            x = solve_bvls(self.A, b, l, u, 1e-10)
        self.prev = x.copy()
        self.raw.append(x.copy())
        w = np.zeros(len(self.names), np.float32)
        w[self.act_idx] = x
        return w * self.mult + self.offs

    def dump_solver(self, path):
        """Solver inputs in the layout of a C++ harness around the SDK's own bvls.cpp:
        i32 k, i32 frames, f32 temporal*scale, i32 ncancel, i32 pairs[2n], f32 A[k*k], f32 ATb[frames*k].
        The raw solutions go to path + '.x'."""
        k = self.A.shape[0]
        with open(path, "wb") as f:
            np.array([k, len(self.atb)], np.int32).tofile(f)
            np.array([np.float32(self.T) * self.scale], np.float32).tofile(f)
            np.array([len(self.cancel)], np.int32).tofile(f)
            np.array(self.cancel, np.int32).reshape(-1).tofile(f)
            self.A.T.astype(np.float32).tofile(f)
            np.stack(self.atb).astype(np.float32).tofile(f)
        np.stack(self.raw).astype(np.float32).tofile(path + ".x")


# ----------------------------------------------------------------------------- main
def main():
    model_dir, wav, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    secs = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    J = lambda f: json.load(open(os.path.join(model_dir, f)))
    ninfo = J("network_info.json"); cfg = J("model_config.json")["config"]
    ap = ninfo["audio_params"]; npar = ninfo["params"]
    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = data.mean(1)
    if secs > 0:
        x = x[: int(secs * sr)]
    a16 = resample(x, sr, 16000)
    a16.tofile(out + ".wav16.f32")
    frames = frame_count(len(a16), fps, ap["buffer_ofs"])
    win = windows(a16, frames, fps, ap["buffer_len"], ap["buffer_ofs"], cfg["input_strength"])

    emo = np.load(os.path.join(model_dir, "implicit_emo_db.npz"))
    names = [s.decode() for s in emo["emo_spec_names"]]
    si = names.index(cfg["source_shot"])
    implicit = emo["emo_db"][emo["emo_spec_start"][si] + cfg["source_frame"]]
    emotion = np.concatenate([implicit, np.zeros(len(npar["explicit_emotions"]), np.float32)]).astype(np.float32)

    sess = ort.InferenceSession(os.path.join(model_dir, "network.onnx"), providers=["CPUExecutionProvider"])
    t0 = time.time()
    net = sess.run(None, {"input": win[:, None, :],
                          "emotion": np.tile(emotion, (frames, 1))[:, None, :]})[0].reshape(frames, -1)
    t_net = time.time() - t0
    net.astype(np.float32).tofile(out + ".net.f32")

    md = np.load(os.path.join(model_dir, "model_data.npz"))
    ns, nt = npar["num_shapes_skin"], npar["num_shapes_tongue"]
    nj, ne = npar["result_jaw_size"], npar["result_eyes_size"]
    Ms = md["shapes_matrix_skin"].reshape(ns, -1); neutral_s = md["shapes_mean_skin"].reshape(-1)
    Mt = md["shapes_matrix_tongue"].reshape(nt, -1); neutral_t = md["shapes_mean_tongue"].reshape(-1)
    lip = md["lip_open_pose_delta"].reshape(-1); eye = md["eye_close_pose_delta"].reshape(-1)
    jaw0 = md["neutral_jaw"].astype(np.float32); sac = md["saccade_rot_matrix"].reshape(-1)

    dt = np.float32(1.0 / fps)
    alpha = lambda s: np.float32(1.0 - math.pow(0.5, dt / s)) if s > 0 else np.float32(-1)
    la, ua = alpha(cfg["lower_face_smoothing"]), alpha(cfg["upper_face_smoothing"])
    y = md["shapes_mean_skin"][:, 1]
    mn, mx = y.min(), y.max()
    mask = (1.0 / (1.0 + np.exp(-(cfg["face_mask_level"] - (y - mn) / (mx - mn)) / cfg["face_mask_softness"]))).astype(np.float32)
    maskv = np.repeat(mask, 3)

    skin_solver = BsSolver(os.path.join(model_dir, "bs_skin.npz"), os.path.join(model_dir, "bs_skin_config.json"))
    tongue_solver = BsSolver(os.path.join(model_dir, "bs_tongue.npz"), os.path.join(model_dir, "bs_tongue_config.json"))
    skin_w = np.zeros((frames, len(skin_solver.names)), np.float32)
    tongue_w = np.zeros((frames, len(tongue_solver.names)), np.float32)
    jaw_out = np.zeros((frames, 16), np.float32); eyes_out = np.zeros((frames, 6), np.float32)
    lower1 = lower2 = upper1 = upper2 = None
    live = np.float32(0.0); nsac = len(sac) // 2
    t0 = time.time()
    for f in range(frames):
        r = net[f]
        # skin
        d_in = (r[:ns] @ Ms).astype(np.float32)
        delta = (cfg["skin_strength"] * d_in + eye * (-cfg["eyelid_open_offset"] + 0.0 * cfg["blink_strength"])
                 + lip * cfg["lip_open_offset"]).astype(np.float32)
        if f == 0 or la <= 0:
            lower1 = delta.copy(); lower2 = delta.copy()
        else:
            lower1 = lower1 + (delta - lower1) * la; lower2 = lower2 + (lower1 - lower2) * la
        if f == 0 or ua <= 0:
            upper1 = delta.copy(); upper2 = delta.copy()
        else:
            upper1 = upper1 + (delta - upper1) * ua; upper2 = upper2 + (upper1 - upper2) * ua
        skin = (neutral_s + upper2 * cfg["upper_face_strength"] * (1 - maskv)
                + lower2 * cfg["lower_face_strength"] * maskv).astype(np.float32)
        # tongue
        tongue = (neutral_t + (r[ns:ns + nt] @ Mt) * cfg["tongue_strength"]).astype(np.float32).reshape(-1, 3)
        tongue[:, 1] += cfg["tongue_height_offset"]; tongue[:, 2] += cfg["tongue_depth_offset"]
        # jaw
        jd = r[ns + nt: ns + nt + nj].reshape(-1, 3)
        tgt = jaw0 + jd * cfg["lower_teeth_strength"] + np.array([0, cfg["lower_teeth_height_offset"], cfg["lower_teeth_depth_offset"]], np.float32)
        jaw_out[f] = rigid_xform(tgt.astype(np.float64), jaw0.astype(np.float64))
        # eyes
        ev = r[ns + nt + nj: ns + nt + nj + ne]
        offs = [cfg["right_eye_rot_x_offset"], cfg["right_eye_rot_y_offset"], cfg["left_eye_rot_x_offset"], cfg["left_eye_rot_y_offset"]]
        tt = math.fmod(cfg["eye_saccade_seed"] + live, nsac)
        if tt < 0: tt += nsac
        fi = int(tt)
        for e in range(2):
            for a in range(2):
                eyes_out[f, e * 3 + a] = np.float32(offs[e * 2 + a]) + np.float32(ev[e * 2 + a] * np.float32(cfg["eyeballs_strength"])) + np.float32(np.float32(cfg["saccade_strength"]) * sac[2 * fi + a])
        live = np.float32(math.fmod(live + dt * 30.0, nsac))
        skin_w[f] = skin_solver.solve(skin)
        tongue_w[f] = tongue_solver.solve(tongue)
    t_post = time.time() - t0
    for name, arr in [("skin", skin_w), ("tongue", tongue_w), ("jaw", jaw_out), ("eyes", eyes_out)]:
        arr.astype(np.float32).tofile(out + "." + name + ".f32")
    json.dump({"frames": frames, "fps": fps, "samples16": len(a16), "skin_names": skin_solver.names,
               "tongue_names": tongue_solver.names, "t_net": t_net, "t_post": t_post},
              open(out + ".json", "w"), indent=1)
    if os.environ.get("A2F_SOLVER_DUMP"):
        skin_solver.dump_solver(out + ".solve_skin.bin")
        tongue_solver.dump_solver(out + ".solve_tongue.bin")
    print(f"frames {frames}  net {t_net:.3f}s  post {t_post:.3f}s  audio {len(a16)/16000:.2f}s")
    top = np.argsort(-skin_w.max(0))[:8]
    print("max skin weights:", ", ".join(f"{skin_solver.names[i]}={skin_w[:, i].max():.3f}" for i in top))


if __name__ == "__main__":
    main()

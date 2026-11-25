#!/usr/bin/env python3
"""Offline training of the Learned Motion Matching networks for Flint.

Adapted from Daniel Holden's MIT reference (https://github.com/orangeduck/Motion-Matching,
`resources/train_decompressor.py`, `train_stepper.py`, `train_projector.py`; copyright (c) 2021
Daniel Holden, MIT licence, see licenses/README.md), following "Learned Motion Matching"
(Holden, Kanoun, Perepichka, Popa; SIGGRAPH 2020). PyTorch is used here only, at tool time;
Flint runs the result in pure Mojo (`flint.anim.lmm`).

usage: train.py DB_DIR [--out OUT_DIR] [--iters N] [--device cuda|cpu] [--seed 1234]

DB_DIR holds `database.bin` and `features.bin` written by `tools/learned_mm/build_database.mojo`.
Writes to OUT_DIR (default DB_DIR):

  decompressor.bin, stepper.bin, projector.bin   networks in the reference `nnet` layout
  latent.bin                                    per-frame latent codes from the compressor
  parity.bin                                    inputs and PyTorch outputs for the Mojo test
  quality.json                                  reconstruction / stepping / projection errors

The networks are derived from the training data (100STYLE, CC BY 4.0): keep them, like the
database, outside the repository and carry the 100STYLE attribution with them.
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- files

def load_database(path):
    with open(path, "rb") as f:
        def arr(dtype, shape):
            n = int(np.prod(shape))
            return np.frombuffer(f.read(n * np.dtype(dtype).itemsize), dtype=dtype, count=n).reshape(shape)
        nf, nb = struct.unpack("II", f.read(8)); pos = arr(np.float32, [nf, nb, 3])
        nf, nb = struct.unpack("II", f.read(8)); vel = arr(np.float32, [nf, nb, 3])
        nf, nb = struct.unpack("II", f.read(8)); rot = arr(np.float32, [nf, nb, 4])
        nf, nb = struct.unpack("II", f.read(8)); ang = arr(np.float32, [nf, nb, 3])
        nb = struct.unpack("I", f.read(4))[0]; parents = arr(np.int32, [nb])
        nr = struct.unpack("I", f.read(4))[0]; starts = arr(np.int32, [nr])
        nr = struct.unpack("I", f.read(4))[0]; stops = arr(np.int32, [nr])
        nf, nc = struct.unpack("II", f.read(8)); contacts = arr(np.uint8, [nf, nc])
    return dict(pos=pos, vel=vel, rot=rot, ang=ang, parents=parents, starts=starts, stops=stops, contacts=contacts)


def load_features(path):
    with open(path, "rb") as f:
        nf, nx = struct.unpack("II", f.read(8))
        X = np.frombuffer(f.read(nf * nx * 4), dtype=np.float32).reshape([nf, nx])
    return X


def save_network(path, layers, mean_in, std_in, mean_out, std_out):
    with torch.no_grad(), open(path, "wb") as f:
        for a in [mean_in, std_in, mean_out, std_out]:
            a = a.detach().cpu().numpy().astype(np.float32).ravel()
            f.write(struct.pack("I", a.shape[0]) + a.tobytes())
        f.write(struct.pack("I", len(layers)))
        for layer in layers:
            w = layer.weight.T.detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack("II", *w.shape) + np.ascontiguousarray(w).tobytes())
            b = layer.bias.detach().cpu().numpy().astype(np.float32)
            f.write(struct.pack("I", b.shape[0]) + b.tobytes())


# ----------------------------------------------------------------------------- quaternions (w, x, y, z)

def q_mul(a, b):
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    return np.concatenate([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def q_inv(q):
    return np.array([1, -1, -1, -1], dtype=np.float32) * q


def q_mul_vec(q, v):
    t = 2.0 * np.cross(q[..., 1:], v)
    return v + q[..., 0:1] * t + np.cross(q[..., 1:], t)


def q_inv_mul(a, b):
    return q_mul(q_inv(a), b)


def q_inv_mul_vec(q, v):
    return q_mul_vec(q_inv(q), v)


def q_to_xform(q):
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    x2, y2, z2 = x + x, y + y, z + z
    xx, yy, wx = x * x2, y * y2, w * x2
    xy, yz, wy = x * y2, y * z2, w * y2
    xz, zz, wz = x * z2, z * z2, w * z2
    return np.concatenate([
        np.concatenate([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)[..., None, :],
        np.concatenate([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)[..., None, :],
        np.concatenate([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)[..., None, :]], axis=-2)


def q_to_xform_xy(q):
    return q_to_xform(q)[..., :2]


def fk_vel(rot, pos, vel, ang, parents):
    gr, gp, gv, ga = [rot[:, 0]], [pos[:, 0]], [vel[:, 0]], [ang[:, 0]]
    for i in range(1, len(parents)):
        p = parents[i]
        lp = q_mul_vec(gr[p], pos[:, i])
        gp.append(lp + gp[p])
        gr.append(q_mul(gr[p], rot[:, i]))
        gv.append(q_mul_vec(gr[p], vel[:, i]) + np.cross(ga[p], lp) + gv[p])
        ga.append(q_mul_vec(gr[p], ang[:, i]) + ga[p])
    return np.stack(gr, 1), np.stack(gp, 1), np.stack(gv, 1), np.stack(ga, 1)


# ----------------------------------------------------------------------------- matrices (torch)

def _cross(a, b):
    return torch.cat([
        a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
        a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
        a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1]], dim=-1)


def t_from_xy(x):
    c2 = _cross(x[..., 0], x[..., 1])
    c2 = c2 / torch.sqrt(torch.sum(torch.square(c2), dim=-1))[..., None]
    c1 = _cross(c2, x[..., 0])
    c1 = c1 / torch.sqrt(torch.sum(torch.square(c1), dim=-1))[..., None]
    return torch.cat([x[..., 0][..., None], c1[..., None], c2[..., None]], dim=-1)


def t_mul_vec(x, v):
    return torch.matmul(x, v[..., None])[..., 0]


def t_fk_vel(lrot, lpos, lvel, lang, parents):
    gp, gr, gv, ga = [lpos[..., :1, :]], [lrot[..., :1, :, :]], [lvel[..., :1, :]], [lang[..., :1, :]]
    for i in range(1, len(parents)):
        p = parents[i]
        lp = t_mul_vec(gr[p], lpos[..., i:i + 1, :])
        gp.append(lp + gp[p])
        gr.append(torch.matmul(gr[p], lrot[..., i:i + 1, :, :]))
        gv.append(t_mul_vec(gr[p], lvel[..., i:i + 1, :]) + torch.cross(ga[p], lp, dim=-1) + gv[p])
        ga.append(t_mul_vec(gr[p], lang[..., i:i + 1, :]) + ga[p])
    return torch.cat(gr, dim=-3), torch.cat(gp, dim=-2), torch.cat(gv, dim=-2), torch.cat(ga, dim=-2)


# ----------------------------------------------------------------------------- networks

class Compressor(nn.Module):
    def __init__(self, n_in, n_out, h=512):
        super().__init__()
        self.linear0 = nn.Linear(n_in, h)
        self.linear1 = nn.Linear(h, h)
        self.linear2 = nn.Linear(h, h)
        self.linear3 = nn.Linear(h, n_out)

    def forward(self, x):
        x = F.elu(self.linear0(x))
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        return self.linear3(x)


class Decompressor(nn.Module):
    def __init__(self, n_in, n_out, h=512):
        super().__init__()
        self.linear0 = nn.Linear(n_in, h)
        self.linear1 = nn.Linear(h, n_out)

    def forward(self, x):
        return self.linear1(F.relu(self.linear0(x)))


class Stepper(nn.Module):
    def __init__(self, n, h=512):
        super().__init__()
        self.linear0 = nn.Linear(n, h)
        self.linear1 = nn.Linear(h, h)
        self.linear2 = nn.Linear(h, n)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(F.relu(self.linear0(x)))))


class Projector(nn.Module):
    def __init__(self, n_in, n_out, h=512):
        super().__init__()
        self.linear0 = nn.Linear(n_in, h)
        self.linear1 = nn.Linear(h, h)
        self.linear2 = nn.Linear(h, h)
        self.linear3 = nn.Linear(h, h)
        self.linear4 = nn.Linear(h, n_out)

    def forward(self, x):
        for l in [self.linear0, self.linear1, self.linear2, self.linear3]:
            x = F.relu(l(x))
        return self.linear4(x)


def _sched(opt, iters):
    # the reference decays by 0.99 every 1000 of 500k iterations; keep the same total decay
    return torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99 ** (500000.0 / max(iters, 1000)))


def _progress(name, i, n, loss, t0):
    if i % 500 == 0 or i == n - 1:
        sys.stdout.write("\r%s %7i/%i loss %.4f  %.0fs" % (name, i, n, loss, time.time() - t0))
        sys.stdout.flush()


# ----------------------------------------------------------------------------- training

def train_decompressor(db, X, args, dev, nlatent=32, dt=1.0 / 60.0):
    parents = db["parents"]
    Ypos, Yrot, Yvel, Yang = db["pos"], db["rot"], db["vel"], db["ang"]
    nframes, nbones = Ypos.shape[:2]
    nextra = db["contacts"].shape[1]
    Grot, Gpos, Gvel, Gang = fk_vel(Yrot, Ypos, Yvel, Yang, parents)
    Qrot = q_inv_mul(Grot[:, 0:1], Grot)
    Qpos = q_inv_mul_vec(Grot[:, 0:1], Gpos - Gpos[:, 0:1])
    Qvel = q_inv_mul_vec(Grot[:, 0:1], Gvel)
    Qang = q_inv_mul_vec(Grot[:, 0:1], Gang)
    Qxfm = q_to_xform(Qrot).astype(np.float32)
    Ytxy = q_to_xform_xy(Yrot).astype(np.float32)
    Qtxy = q_to_xform_xy(Qrot).astype(np.float32)
    Yrvel = q_inv_mul_vec(Yrot[:, 0], Yvel[:, 0])
    Yrang = q_inv_mul_vec(Yrot[:, 0], Yang[:, 0])
    Yextra = db["contacts"].astype(np.float32)

    def s(a):
        return np.float32(a.std())
    mean_out = np.hstack([Ypos[:, 1:].mean(0).ravel(), Ytxy[:, 1:].mean(0).ravel(), Yvel[:, 1:].mean(0).ravel(),
                          Yang[:, 1:].mean(0).ravel(), Yrvel.mean(0).ravel(), Yrang.mean(0).ravel(), Yextra.mean(0).ravel()])
    std_out = np.hstack([Ypos[:, 1:].std(0).ravel(), Ytxy[:, 1:].std(0).ravel(), Yvel[:, 1:].std(0).ravel(),
                         Yang[:, 1:].std(0).ravel(), Yrvel.std(0).ravel(), Yrang.std(0).ravel(), Yextra.std(0).ravel()])
    std_out = np.maximum(std_out, 1e-5)   # constant outputs (fixed bone offsets) stay finite
    comp_mean = np.hstack([Ypos[:, 1:].mean(0).ravel(), Ytxy[:, 1:].mean(0).ravel(), Yvel[:, 1:].mean(0).ravel(),
                           Yang[:, 1:].mean(0).ravel(), Qpos[:, 1:].mean(0).ravel(), Qtxy[:, 1:].mean(0).ravel(),
                           Qvel[:, 1:].mean(0).ravel(), Qang[:, 1:].mean(0).ravel(), Yrvel.mean(0).ravel(),
                           Yrang.mean(0).ravel(), Yextra.mean(0).ravel()])
    nb1 = nbones - 1
    comp_std = np.hstack([np.repeat(s(Ypos[:, 1:]), nb1 * 3), np.repeat(s(Ytxy[:, 1:]), nb1 * 6),
                          np.repeat(s(Yvel[:, 1:]), nb1 * 3), np.repeat(s(Yang[:, 1:]), nb1 * 3),
                          np.repeat(s(Qpos[:, 1:]), nb1 * 3), np.repeat(s(Qtxy[:, 1:]), nb1 * 6),
                          np.repeat(s(Qvel[:, 1:]), nb1 * 3), np.repeat(s(Qang[:, 1:]), nb1 * 3),
                          np.repeat(s(Yrvel), 3), np.repeat(s(Yrang), 3), np.repeat(s(Yextra), nextra)])
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    mean_out, std_out, comp_mean, comp_std = T(mean_out), T(std_out), T(comp_mean), T(comp_std)
    Ypos_t, Ytxy_t, Yvel_t, Yang_t = T(Ypos), T(Ytxy), T(Yvel), T(Yang)
    Qpos_t, Qxfm_t, Qtxy_t, Qvel_t, Qang_t = T(Qpos), T(Qxfm), T(Qtxy), T(Qvel), T(Qang)
    Yrvel_t, Yrang_t, Yextra_t, X_t = T(Yrvel), T(Yrang), T(Yextra), T(X)
    nfeatures = X.shape[1]

    comp = Compressor(len(comp_mean), nlatent).to(dev)
    dec = Decompressor(nfeatures + nlatent, len(mean_out)).to(dev)

    def comp_input(idx):
        return (torch.cat([
            Ypos_t[idx][..., 1:, :].reshape(list(idx.shape) + [-1]), Ytxy_t[idx][..., 1:, :, :].reshape(list(idx.shape) + [-1]),
            Yvel_t[idx][..., 1:, :].reshape(list(idx.shape) + [-1]), Yang_t[idx][..., 1:, :].reshape(list(idx.shape) + [-1]),
            Qpos_t[idx][..., 1:, :].reshape(list(idx.shape) + [-1]), Qtxy_t[idx][..., 1:, :, :].reshape(list(idx.shape) + [-1]),
            Qvel_t[idx][..., 1:, :].reshape(list(idx.shape) + [-1]), Qang_t[idx][..., 1:, :].reshape(list(idx.shape) + [-1]),
            Yrvel_t[idx], Yrang_t[idx], Yextra_t[idx]], dim=-1) - comp_mean) / comp_std

    window = 2
    valid = []
    for s0, s1 in zip(db["starts"], db["stops"]):
        valid.extend(range(s0, s1 - window + 1))
    valid = torch.as_tensor(np.array(valid), dtype=torch.long, device=dev)
    offs = torch.arange(window, device=dev)
    opt = torch.optim.AdamW(list(comp.parameters()) + list(dec.parameters()), lr=args.lr, amsgrad=True, weight_decay=0.001)
    sched = _sched(opt, args.iters)
    bs = args.batch
    t0 = time.time()
    rolling = None
    for i in range(args.iters):
        opt.zero_grad()
        batch = valid[torch.randint(0, len(valid), (bs,), device=dev)][:, None] + offs[None]
        Z = comp(comp_input(batch))
        Yt = dec(torch.cat([X_t[batch], Z], dim=-1)) * std_out + mean_out
        o = 0
        t_pos = Yt[..., o:o + nb1 * 3].reshape([bs, window, nb1, 3]); o += nb1 * 3
        t_txy = Yt[..., o:o + nb1 * 6].reshape([bs, window, nb1, 3, 2]); o += nb1 * 6
        t_vel = Yt[..., o:o + nb1 * 3].reshape([bs, window, nb1, 3]); o += nb1 * 3
        t_ang = Yt[..., o:o + nb1 * 3].reshape([bs, window, nb1, 3]); o += nb1 * 3
        t_rvel = Yt[..., o:o + 3]; o += 3
        t_rang = Yt[..., o:o + 3]; o += 3
        t_extra = Yt[..., o:o + nextra]
        g_pos, g_txy, g_vel, g_ang = Ypos_t[batch], Ytxy_t[batch], Yvel_t[batch], Yang_t[batch]
        t_pos = torch.cat([g_pos[:, :, 0:1], t_pos], dim=2)
        t_txy = torch.cat([g_txy[:, :, 0:1], t_txy], dim=2)
        t_vel = torch.cat([g_vel[:, :, 0:1], t_vel], dim=2)
        t_ang = torch.cat([g_ang[:, :, 0:1], t_ang], dim=2)
        t_xfm = t_from_xy(t_txy)
        G_xfm, G_pos, G_vel, G_ang = t_fk_vel(t_xfm, t_pos, t_vel, t_ang, parents)
        Rt = G_xfm[:, :, 0:1].transpose(-1, -2)
        q_xfm = torch.matmul(Rt, G_xfm)
        q_pos = t_mul_vec(Rt, G_pos - G_pos[:, :, 0:1])
        q_vel = t_mul_vec(Rt, G_vel)
        q_ang = t_mul_vec(Rt, G_ang)
        gq_pos, gq_xfm, gq_vel, gq_ang = Qpos_t[batch], Qxfm_t[batch], Qvel_t[batch], Qang_t[batch]
        d = lambda a: (a[:, 1:] - a[:, :-1]) / dt
        loss = (torch.mean(75.0 * torch.abs(g_pos - t_pos)) + torch.mean(10.0 * torch.abs(g_txy - t_txy))
                + torch.mean(10.0 * torch.abs(g_vel - t_vel)) + torch.mean(1.25 * torch.abs(g_ang - t_ang))
                + torch.mean(2.0 * torch.abs(Yrvel_t[batch] - t_rvel)) + torch.mean(2.0 * torch.abs(Yrang_t[batch] - t_rang))
                + torch.mean(2.0 * torch.abs(Yextra_t[batch] - t_extra))
                + torch.mean(15.0 * torch.abs(gq_pos - q_pos)) + torch.mean(5.0 * torch.abs(gq_xfm - q_xfm))
                + torch.mean(2.0 * torch.abs(gq_vel - q_vel)) + torch.mean(0.75 * torch.abs(gq_ang - q_ang))
                + torch.mean(10.0 * torch.abs(d(g_pos) - d(t_pos))) + torch.mean(1.75 * torch.abs(d(g_txy) - d(t_txy)))
                + torch.mean(2.0 * torch.abs(d(gq_pos) - d(q_pos))) + torch.mean(0.75 * torch.abs(d(gq_xfm) - d(q_xfm)))
                + torch.mean(0.1 * torch.abs(Z)) + torch.mean(0.1 * torch.square(Z)) + torch.mean(0.01 * torch.abs(d(Z))))
        loss.backward()
        opt.step()
        rolling = loss.item() if rolling is None else rolling * 0.99 + loss.item() * 0.01
        _progress("decompressor", i, args.iters, rolling, t0)
        if i % 1000 == 999:
            sched.step()
    print()
    with torch.no_grad():
        Zall = torch.cat([comp(comp_input(torch.arange(a, min(a + 8192, nframes), device=dev)))
                          for a in range(0, nframes, 8192)], dim=0)
    zero_in = torch.zeros(nfeatures + nlatent, device=dev)
    one_in = torch.ones(nfeatures + nlatent, device=dev)
    save_network(os.path.join(args.out, "decompressor.bin"), [dec.linear0, dec.linear1], zero_in, one_in, mean_out, std_out)
    Z = Zall.cpu().numpy().astype(np.float32)
    with open(os.path.join(args.out, "latent.bin"), "wb") as f:
        f.write(struct.pack("II", *Z.shape) + Z.tobytes())

    # quality: global joint positions (in the simulation frame) from ground-truth features + latent
    with torch.no_grad():
        err = []
        for a in range(0, nframes, 8192):
            idx = torch.arange(a, min(a + 8192, nframes), device=dev)
            Yt = dec(torch.cat([X_t[idx], Zall[idx]], dim=-1)) * std_out + mean_out
            n = len(idx)
            t_pos = torch.cat([Ypos_t[idx][:, 0:1], Yt[:, :nb1 * 3].reshape([n, nb1, 3])], dim=1)
            t_txy = torch.cat([Ytxy_t[idx][:, 0:1], Yt[:, nb1 * 3:nb1 * 9].reshape([n, nb1, 3, 2])], dim=1)
            z = torch.zeros_like(t_pos)
            _, G_pos, _, _ = t_fk_vel(t_from_xy(t_txy), t_pos, z, z, parents)
            g_xfm = t_from_xy(Ytxy_t[idx])
            _, GG_pos, _, _ = t_fk_vel(g_xfm, Ypos_t[idx], z, z, parents)
            err.append(torch.sqrt(torch.sum(torch.square(G_pos - GG_pos), dim=-1)).mean(dim=-1))
        err = torch.cat(err)
    pdec = dict(mean_joint_error_cm=float(err.mean() * 100), p95_joint_error_cm=float(torch.quantile(err[:: max(1, nframes // 50000)], 0.95) * 100))
    return Z, dec, mean_out, std_out, pdec


def train_stepper(db, X, Z, args, dev, dt=1.0 / 60.0):
    nfeatures, nlatent = X.shape[1], Z.shape[1]
    dX = np.concatenate([(X[s0 + 1:s1] - X[s0:s1 - 1]) / dt for s0, s1 in zip(db["starts"], db["stops"])])
    dZ = np.concatenate([(Z[s0 + 1:s1] - Z[s0:s1 - 1]) / dt for s0, s1 in zip(db["starts"], db["stops"])])
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    mean_in = T(np.hstack([X.mean(0), Z.mean(0)]))
    std_in = T(np.hstack([np.repeat(X.std(), nfeatures), np.repeat(Z.std(), nlatent)]))
    mean_out = T(np.hstack([dX.mean(0), dZ.mean(0)]))
    std_out = T(np.hstack([dX.std(0), dZ.std(0)]))
    std_out = torch.clamp(std_out, min=1e-5)
    Xt, Zt = T(X), T(Z)
    net = Stepper(nfeatures + nlatent).to(dev)
    window = 20
    valid = []
    for s0, s1 in zip(db["starts"], db["stops"]):
        valid.extend(range(s0, s1 - window + 1))
    valid = torch.as_tensor(np.array(valid), dtype=torch.long, device=dev)
    offs = torch.arange(window, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.001)
    sched = _sched(opt, args.iters)
    bs = args.batch
    t0 = time.time()
    rolling = None
    for i in range(args.iters):
        opt.zero_grad()
        batch = valid[torch.randint(0, len(valid), (bs,), device=dev)][:, None] + offs[None]
        Xg, Zg = Xt[batch], Zt[batch]
        xs, zs = [Xg[:, 0]], [Zg[:, 0]]
        for _ in range(1, window):
            o = net((torch.cat([xs[-1], zs[-1]], dim=-1) - mean_in) / std_in) * std_out + mean_out
            xs.append(xs[-1] + dt * o[:, :nfeatures])
            zs.append(zs[-1] + dt * o[:, nfeatures:])
        Xs, Zs = torch.stack(xs, 1), torch.stack(zs, 1)
        d = lambda a: (a[:, 1:] - a[:, :-1]) / dt
        loss = (torch.mean(2.0 * torch.abs(Xg - Xs)) + torch.mean(7.5 * torch.abs(Zg - Zs))
                + torch.mean(0.2 * torch.abs(d(Xg) - d(Xs))) + torch.mean(0.5 * torch.abs(d(Zg) - d(Zs))))
        loss.backward()
        opt.step()
        rolling = loss.item() if rolling is None else rolling * 0.99 + loss.item() * 0.01
        _progress("stepper", i, args.iters, rolling, t0)
        if i % 1000 == 999:
            sched.step()
    print()
    save_network(os.path.join(args.out, "stepper.bin"), [net.linear0, net.linear1, net.linear2], mean_in, std_in, mean_out, std_out)
    # quality: feature error after 20 free-running steps
    with torch.no_grad():
        idx = valid[torch.randint(0, len(valid), (4096,), device=dev, generator=torch.Generator(device=dev).manual_seed(7))]
        x, z = Xt[idx], Zt[idx]
        for _ in range(window - 1):
            o = net((torch.cat([x, z], dim=-1) - mean_in) / std_in) * std_out + mean_out
            x, z = x + dt * o[:, :nfeatures], z + dt * o[:, nfeatures:]
        e = torch.abs(x - Xt[idx + window - 1]).mean().item()
        e0 = torch.abs(Xt[idx] - Xt[idx + window - 1]).mean().item()
    return net, mean_in, std_in, mean_out, std_out, dict(feature_mae_after_20_steps=e, feature_mae_if_frozen=e0)


def train_projector(X, Z, args, dev):
    nfeatures, nlatent = X.shape[1], Z.shape[1]
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    Xt, Zt = T(X), T(Z)
    noise_std = T(X.std(0) + 1.0)
    mean_out = T(np.hstack([X.mean(0), Z.mean(0)]))
    std_out = T(np.hstack([X.std(0), Z.std(0)]))
    mean_in = T(X.mean(0))
    std_in = T(np.repeat(X.std(), nfeatures))
    net = Projector(nfeatures, nfeatures + nlatent).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=0.001)
    sched = _sched(opt, args.iters)
    bs = args.batch
    sq = torch.sum(Xt * Xt, dim=1)

    def nearest(q):
        d = torch.sum(q * q, dim=1, keepdim=True) - 2 * q @ Xt.T + sq[None]
        return torch.argmin(d, dim=1)
    t0 = time.time()
    rolling = None
    for i in range(args.iters):
        opt.zero_grad()
        samples = torch.randint(0, len(X), (bs,), device=dev)
        nsig = torch.rand((bs, 1), device=dev)
        Xh = Xt[samples] + noise_std * nsig * torch.randn((bs, nfeatures), device=dev)
        with torch.no_grad():
            nn_i = nearest(Xh)
        Xg, Zg = Xt[nn_i], Zt[nn_i]
        Dg = torch.sqrt(torch.sum(torch.square(Xh - Xg), dim=-1))
        o = net((Xh - mean_in) / std_in) * std_out + mean_out
        Xs, Zs = o[:, :nfeatures], o[:, nfeatures:]
        Ds = torch.sqrt(torch.sum(torch.square(Xh - Xs), dim=-1))
        loss = torch.mean(1.0 * torch.abs(Xg - Xs)) + torch.mean(5.0 * torch.abs(Zg - Zs)) + torch.mean(0.3 * torch.abs(Dg - Ds))
        loss.backward()
        opt.step()
        rolling = loss.item() if rolling is None else rolling * 0.99 + loss.item() * 0.01
        _progress("projector", i, args.iters, rolling, t0)
        if i % 1000 == 999:
            sched.step()
    print()
    save_network(os.path.join(args.out, "projector.bin"), [net.linear0, net.linear1, net.linear2, net.linear3, net.linear4],
                 mean_in, std_in, mean_out, std_out)
    with torch.no_grad():
        g = torch.Generator(device=dev).manual_seed(11)
        samples = torch.randint(0, len(X), (4096,), device=dev, generator=g)
        Xh = Xt[samples] + noise_std * 0.5 * torch.randn((4096, nfeatures), device=dev, generator=g)
        nn_i = nearest(Xh)
        o = net((Xh - mean_in) / std_in) * std_out + mean_out
        fe = torch.abs(o[:, :nfeatures] - Xt[nn_i]).mean().item()
        ze = torch.abs(o[:, nfeatures:] - Zt[nn_i]).mean().item()
        o2 = net((Xt[samples] - mean_in) / std_in) * std_out + mean_out
        fe0 = torch.abs(o2[:, :nfeatures] - Xt[samples]).mean().item()
    return net, mean_in, std_in, mean_out, std_out, dict(noisy_query_feature_mae=fe, noisy_query_latent_mae=ze, clean_query_feature_mae=fe0)


def write_parity(path, items):
    with open(path, "wb") as f:
        f.write(struct.pack("I", len(items)))
        for x, y in items:
            x = x.astype(np.float32); y = y.astype(np.float32)
            f.write(struct.pack("III", x.shape[0], x.shape[1], y.shape[1]) + x.tobytes() + y.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    args.out = args.out or args.db_dir
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    db = load_database(os.path.join(args.db_dir, "database.bin"))
    X = load_features(os.path.join(args.db_dir, "features.bin")).astype(np.float32)
    print("database: %d frames, %d bones, %d features, %d ranges, device %s" % (X.shape[0], db["pos"].shape[1], X.shape[1], len(db["starts"]), dev))

    Z, dec, dmean, dstd, qdec = train_decompressor(db, X, args, dev)
    print("decompressor:", qdec)
    step, s_mi, s_si, s_mo, s_so, qstep = train_stepper(db, X, Z, args, dev)
    print("stepper:", qstep)
    proj, p_mi, p_si, p_mo, p_so, qproj = train_projector(X, Z, args, dev)
    print("projector:", qproj)

    # parity: raw inputs (the Mojo input layer) and full outputs (denormalised)
    rng = np.random.RandomState(5)
    idx = rng.randint(0, X.shape[0], 16)
    xin = np.hstack([X[idx], Z[idx]]).astype(np.float32)
    with torch.no_grad():
        t = torch.as_tensor(xin, device=dev)
        y_dec = (dec(t) * dstd + dmean).cpu().numpy()
        y_step = (step((t - s_mi) / s_si) * s_so + s_mo).cpu().numpy()
        q = torch.as_tensor(X[idx] + rng.normal(size=X[idx].shape).astype(np.float32) * 0.3, device=dev)
        y_proj = (proj((q - p_mi) / p_si) * p_so + p_mo).cpu().numpy()
    write_parity(os.path.join(args.out, "parity.bin"), [(xin, y_dec), (xin, y_step), (q.cpu().numpy(), y_proj)])
    with open(os.path.join(args.out, "quality.json"), "w") as f:
        json.dump(dict(decompressor=qdec, stepper=qstep, projector=qproj, iters=args.iters, batch=args.batch,
                       frames=int(X.shape[0])), f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

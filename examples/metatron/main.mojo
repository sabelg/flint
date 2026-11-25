"""Metatron-v1 on a simulated Unitree G1 in Flint physics: stand, walk or turn under the policy.

usage: mojo run -I src examples/metatron/main.mojo DIR [--script idle|walk|turn|SPEED,HEADING]
                                                       [--seconds S] [--cpu]

  DIR        model.onnx, weights.bin and g1_holo_compat.xml from `bash tools/metatron/fetch.sh DIR`
  --script   idle: hold the calibrated rest pose; walk: 0.5 m/s forward after 1 s; turn: turn to
             90 degrees and walk 0.3 m/s along the new heading; "v,h": v m/s along heading h (rad)
  --seconds  simulated time (default 8)
  --cpu      run the network on the CPU even when a MAX GPU is present

Prints the pelvis position, height, tilt and torso heading twice a second and the timings.
"""
from std.sys import argv
from std.math import cos, sin, sqrt, acos
from std.time import perf_counter_ns
from flint.physics.mjcf import MjcfModel
from flint.ml.metatron import (MetatronPolicy, MetatronHost, MetatronTask, G1Sim, g1_rest_from_model,
                               mt_yaw_of, MT_PI)


def command(mut task: MetatronTask, script: String, t: Float32) raises:
    if script == "idle" or t < 1.0:
        task.set_idle()
    elif script == "walk":
        task.set_steering(0.0, 0.5, 0.0)
    elif script == "turn":
        var a = min(MT_PI / 2, (t - 1.0) * MT_PI / 2)
        task.set_steering(MT_PI / 2, 0.3 * cos(a), 0.3 * sin(a))
    else:
        var parts = script.split(",")
        var v = Float32(atof(parts[0]))
        var h = Float32(atof(parts[1])) if len(parts) > 1 else Float32(0)
        task.set_steering(h, v * cos(h), v * sin(h))


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: examples/metatron/main.mojo DIR [--script idle|walk|turn|v,h] [--seconds S] [--cpu]")
        return
    var dir = String(args[1])
    var script = String("walk")
    var seconds: Float64 = 8.0
    var gpu = True
    var i = 2
    while i < len(args):
        var a = String(args[i])
        if a == "--script" and i + 1 < len(args):
            script = String(args[i + 1]); i += 1
        elif a == "--seconds" and i + 1 < len(args):
            seconds = atof(String(args[i + 1])); i += 1
        elif a == "--cpu":
            gpu = False
        i += 1
    var model = MjcfModel.load(dir + "/g1_holo_compat.xml")
    var rest = g1_rest_from_model(model)
    var sim = G1Sim(model^)
    var policy = MetatronPolicy.load(dir + "/model.onnx", gpu)
    print("Metatron-v1:", "GPU" if policy.on_gpu else "CPU", "| G1:", len(sim.art.links), "links,", len(sim.model.actuators), "actuators,", sim.model.total_mass(), "kg")
    var host = MetatronHost(policy^, rest^)
    var steps = Int(seconds / 0.02 + 0.5)
    var q0 = sim.qpos()
    var t_net = 0.0
    var t_phys = 0.0
    var fell = False
    for k in range(steps):
        var t = Float32(k) * 0.02
        command(host.task, script, t)
        var rd = sim.reading()
        var b = sim.bodies()
        var bp = b[0].copy()
        var br = b[1].copy()
        var t0 = perf_counter_ns()
        var o = host.step(rd, bp, br)
        var t1 = perf_counter_ns()
        sim.apply(o)
        t_phys += Float64(perf_counter_ns() - t1) / 1e6
        t_net += Float64(t1 - t0) / 1e6
        var q = sim.qpos()
        if q[2] < 0.45 and not fell:
            fell = True
            print("fell at t =", t)
        if k % 25 == 24 or k == steps - 1:
            var tilt = acos(max(-1.0, min(1.0, 1 - 2 * (q[4] * q[4] + q[5] * q[5]))))
            print("t", Float64(t) + 0.02, "s  pelvis (", q[0], q[1], ") z", q[2], " tilt", tilt, " heading", mt_yaw_of(rd.anchor_rot))
    var q = sim.qpos()
    print("travelled", sqrt((q[0] - q0[0]) ** 2 + (q[1] - q0[1]) ** 2), "m;", "fell" if fell else "upright",
          "| network", t_net / Float64(steps), "ms/tick, physics", t_phys / Float64(steps), "ms/tick")

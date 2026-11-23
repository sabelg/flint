"""Inspect and edit a sampled motion using a shared 30 fps timeline."""
from std.math import sin, cos, isfinite
from flint.anim.choreography import MotionSequence, MotionPlayer, cue_kind, TRACK_FPS
from flint.anim.procedural import Placement
from flint.assets.model import NativeModel
from flint.core.math import Vec3
from flint.platform.audio import WavClip
from flint.render.simd_canvas import Camera3D, SimdCanvas

comptime TRANSITION = 1
comptime CONTACT = 2
comptime TIMING = 3


def effect_sound(kind: Int) -> WavClip:
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = 24000
    var n = 3600 if kind == 0 else 5200
    var noise = UInt32(1)
    var smooth = Float32(0)
    for i in range(n):
        noise = noise * 1664525 + 1013904223
        var white = Float32(noise >> 16) / 32768 - 1
        var t = Float32(i) / Float32(n)
        smooth = smooth * 0.65 + white * 0.35
        var wave = sin(t * (95 - 38 * t)) * 0.65 + smooth if kind == 0 else (white - smooth) * sin(
            t * 3.14159
        )
        clip.samples.append(Int16(wave * (1 - t) * (1 - t) * 12500))
    return clip^


def nearest_frame(seconds: Float32) -> Int:
    if not isfinite(seconds) or seconds <= Float32(0.0):
        return 0
    var scaled = Float64(seconds) * Float64(TRACK_FPS) + 0.5
    if scaled >= 1000000000.0:
        return 1000000000
    return max(0, Int(scaled))


def frame_joint(
    mut player: MotionPlayer, model: NativeModel, seq: MotionSequence, frame: Int, joint: Int
) -> Vec3:
    var actor = player.sample(model, seq, Placement(seq.ledge, yaw=seq.yaw), Float32(frame) / TRACK_FPS)
    if joint < 0 or joint >= len(player.markers):
        return actor.to_world(Vec3())
    return actor.to_world(player.bones.global_position(player.markers[joint]))


struct MotionTools(Movable):
    var tab: Int
    var quarter: Bool
    var loop_join: Bool
    var original: Bool
    var marker: Int
    var pinned: Bool
    var anchor: Vec3
    var pin_frame: Int
    var first_slip: Int
    var kind: Int
    var view: Camera3D
    var shown_joint: Vec3
    var drag_blend: Bool

    def __init__(out self):
        self.tab = 0
        self.quarter = True
        self.loop_join = True
        self.original = False
        self.marker = 0
        self.pinned = False
        self.anchor = Vec3()
        self.pin_frame = 0
        self.first_slip = -1
        self.kind = 0
        self.view = Camera3D()
        self.shown_joint = Vec3()
        self.drag_blend = False

    def pin(
        mut self,
        model: NativeModel,
        seq: MotionSequence,
        mut player: MotionPlayer,
        frame: Int,
        target: Vec3,
    ):
        if self.marker < 0 or self.marker >= len(player.markers):
            return
        self.anchor = target
        self.pin_frame = frame
        self.pinned = True
        self.first_slip = -1
        for i in range(frame, nearest_frame(player.duration(model, seq)) + 1):
            if frame_joint(player, model, seq, i, self.marker).distance_to(target) > 0.02:
                self.first_slip = i
                break

    def clicked_point(self, x: Int, y: Int) -> Vec3:
        # Pick on the screen-facing plane at the inspected joint's depth.
        var depth = (self.shown_joint - self.view.eye).dot(self.view.basis_f)
        return (
            self.view.eye
            + (
                self.view.basis_f
                + self.view.basis_r
                * ((Float32(x) - Float32(self.view.screen_w) * 0.5) / self.view.focal_px)
                + self.view.basis_u
                * ((Float32(self.view.screen_h) * 0.5 - Float32(y)) / self.view.focal_px)
            )
            * depth
        )

    def hud(
        self,
        mut c: SimdCanvas,
        model: NativeModel,
        seq: MotionSequence,
        player: MotionPlayer,
        frame: Int,
        bar: Int,
        status: String,
    ):
        var names = ["Motion", "Transition Preview", "Contact Inspector", "Timing Track"]
        for i in range(4):
            var x = i * c.width // 4
            c.draw_rect(x, 0, c.width // 4 - 1, 31, UInt32(0xFF365064 if i == self.tab else 0xFF1C2938))
            c.draw_text(x + 10, 8, names[i], 0xFFE2ECF3, 2)
        if self.tab == 0:
            return
        c.draw_rect(0, bar + 23, c.width, c.height - bar - 23, 0xFF15202E)
        if self.tab == TRANSITION:
            var ms = Int(
                seq.overlap
                * min(model.animations[player.first].duration, model.animations[player.second].duration)
                * 1000
            )
            c.draw_text(16, bar + 28, seq.first + " > " + seq.second, 0xFFE6EFF5, 2)
            c.draw_text(16, bar + 52, "Blend " + String(ms) + " ms", 0xFFFFC455, 2)
            c.draw_rect(215, bar + 53, 300, 8, 0xFF526473)
            c.draw_rect(215 + Int(294 * seq.overlap), bar + 48, 6, 18, 0xFF89D8CE)
            c.draw_text(530, bar + 52, "Original" if self.original else "Edited", 0xFFE6EFF5, 2)
            c.draw_text(660, bar + 52, "1/4x" if self.quarter else "1x", 0xFFE6EFF5, 2)
            c.draw_text(755, bar + 52, "Loop join" if self.loop_join else "Full move", 0xFFE6EFF5, 2)
            c.draw_text(
                16, bar + 78, "1/2 clips   A compare   Q speed   L loop   Space play", 0xFFBBD0DF, 2
            )
            c.draw_text(
                16, bar + 102, "Frame " + String(frame) + "   Arrows step   S save shown", 0xFFBBD0DF, 2
            )
        elif self.tab == CONTACT:
            var name = (
                seq.markers[self.marker]
                if (player.markers and self.marker >= 0 and self.marker < len(seq.markers))
                else String("No marked joints")
            )
            c.draw_text(16, bar + 30, name + "   F" + String(frame), 0xFFE6EFF5, 2)
            if self.pinned:
                c.draw_text(
                    16,
                    bar + 57,
                    "Pin F"
                    + String(self.pin_frame)
                    + "   Drift "
                    + String(Int(self.anchor.distance_to(self.shown_joint) * 1000))
                    + " mm   First >20 mm: "
                    + (String(self.first_slip) if self.first_slip >= 0 else "none"),
                    0xFFFFC455,
                    2,
                )
            c.draw_text(
                16, bar + 83, "J joint   P pin joint   L pin lip   Click scene to pin", 0xFFBBD0DF, 2
            )
            c.draw_text(16, bar + 105, "Arrows step   Drag timeline to inspect", 0xFFBBD0DF, 2)
        elif self.tab == TIMING:
            var last = max(1, Int(player.duration(model, seq) * TRACK_FPS + 0.0001))
            for k in range(4):
                var y = bar + 28 + k * 18
                c.draw_text(16, y, cue_kind(k), UInt32(0xFFFFC455 if k == self.kind else 0xFFBDCFDC), 2)
                c.draw_rect(
                    140, y + 2, c.width - 156, 11, UInt32(0xFF385365 if k == self.kind else 0xFF27394A)
                )
            for event in seq.events:
                if event.frame <= last:
                    var x = 140 + (c.width - 157) * event.frame // last
                    c.draw_rect(x - 2, bar + 29 + event.kind * 18, 6, 14, 0xFFFFC455)
            var head = 140 + (c.width - 157) * min(last, frame) // last
            c.draw_rect(head, bar + 26, 1, 73, 0xFFB7E9E0)
            c.draw_text(
                16,
                bar + 107,
                "F"
                + String(frame)
                + "   Click lane to add   1-4 type   Enter add   Del remove   [ / ] events   Space play  "
                " S save",
                0xFFBBD0DF,
                1,
            )
        c.draw_text(16, bar + 133, status, 0xFFFFC455, 1)


def shake_offset(seq: MotionSequence, seconds: Float32) -> Vec3:
    var offset = Vec3()
    for cue in seq.events:
        var age = seconds - Float32(cue.frame) / TRACK_FPS
        if cue.kind == 3 and age >= 0 and age < 0.25:
            var strength = 0.045 * (1 - age / 0.25)
            offset = offset + Vec3(cos(age * 88), sin(age * 112), 0) * strength
    return offset


def draw_dust(mut canvas: SimdCanvas, cam: Camera3D, at: Vec3, age: Float32):
    for i in range(10):
        var dir = Float32(i) * 2.4
        var radius = 0.06 + age * (0.6 + Float32(i % 3) * 0.15)
        var p = cam.project(
            at + Vec3(cos(dir) * radius, age * (0.8 + Float32(i % 4) * 0.2), sin(dir) * radius)
        )
        if p[3]:
            var size = max(1, Int(5 * (1 - age / 0.32)))
            canvas.draw_rect(Int(p[0]), Int(p[1]), size, size, 0xFFD7BB90)

"""The Long Way Home — Cozy Low-Cortisol 4-Chapter Storybook Adventure (Pure Mojo 1.2).

Inspired by painterly hand-crafted storybook journeys: a kid in a red beanie & ochre coat,
a loyal Shiba Inu companion, and a tranquil walk home across 4 atmospheric chapters:
  - Chapter 0: III  EMBERLEAF FALLS    (Autumn watermill, rotating wheel, leaf-pile Shiba leaps)
  - Chapter 1: IX   THE WATER CITY     (Venice golden-hour canals, arched bridge, gondola glide)
  - Chapter 2: X    STARFALL DUNES     (Twilight Milky Way desert, hand-lantern glow, kite caravan)
  - Chapter 3: XI   THE THOUSAND GATES (Kyoto vermilion Torii gates, koi pond, ringing temple bell)

Controls:
  - A / D or Left / Right : Walk peacefully along the path
  - Space / E             : Cozy interaction (leap into leaves, chime gondola, ring temple bell)
  - 1 / 2 / 3 / 4         : Jump directly to any chapter
  - M                     : Toggle procedural ambient & temple-bell audio
  - Esc                   : Quit

CLI:
  - --chapter <0..3>      : Start on a specific chapter
  - --shot <out.png>      : Render a single deterministic 1280x720 PNG and exit
"""
from std.collections import List
from std.math import abs, cos, exp, max, min, sin
from std.sys import argv
from flint.app import AppConfig, EngineApp
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.assets.image import write_png
from flint.audio.plugin import AudioPlugin
from flint.audio.server import AudioServer, BUS_AMBIENCE, BUS_SFX
from flint.input.plugin import InputPlugin, InputStateResource
from flint.platform.input import InputState
from flint.platform.os import env_flag, env_value, get_time, make_dirs, parent_directory
from flint.render.font import SdfFont
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    RenderWorld,
    ScalableRenderer,
)
from flint.ui.draw_list import (
    UI_FLAG_OUTER_GLOW,
    UI_FLAG_VERT_GRADIENT,
    UiDrawList,
)

comptime SCREEN_W: Int = 1280
comptime SCREEN_H: Int = 720
comptime CHAPTER_LEN: Float32 = 2200.0
comptime TWO_PI: Float32 = 6.2831853


@always_inline
def _rgba(r: Int, g: Int, b: Int, a: Int = 255) -> UInt32:
    var cr = max(0, min(255, r))
    var cg = max(0, min(255, g))
    var cb = max(0, min(255, b))
    var ca = max(0, min(255, a))
    return (UInt32(ca) << 24) | (UInt32(cr) << 16) | (UInt32(cg) << 8) | UInt32(cb)


@always_inline
def _clampf(v: Float32, lo: Float32, hi: Float32) -> Float32:
    return max(lo, min(hi, v))


@always_inline
def _hash01(n: Int) -> Float32:
    var x = UInt32(n * 747796405 + 2891336453)
    x = ((x >> ((x >> 28) + 4)) ^ x) * UInt32(277803737)
    x = (x >> 22) ^ x
    return Float32(x & UInt32(0xFFFF)) / Float32(65535.0)


@fieldwise_init
struct LeafParticle(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var vx: Float32
    var vy: Float32
    var life: Float32
    var max_life: Float32
    var color: UInt32
    var size: Float32


@fieldwise_init
struct RippleRing(Copyable, ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32
    var age: Float32
    var max_age: Float32
    var color: UInt32


struct StoryGame(Movable):
    var chapter: Int
    var player_x: Float32
    var player_vx: Float32
    var facing_right: Bool
    var walk_phase: Float32
    var dog_x: Float32
    var dog_y: Float32
    var dog_hop_t: Float32
    var cam_x: Float32
    var clock: Float32
    var bell_swing: Float32
    var bell_energy: Float32
    var sound_on: Bool
    var fade_alpha: Float32
    var fade_dir: Int
    var pending_chapter: Int
    var particles: List[LeafParticle]
    var ripples: List[RippleRing]
    var bell_sound_id: Int
    var chime_sound_id: Int
    var amb_voice_started: Bool

    def __init__(out self, start_chapter: Int = 3):
        self.chapter = max(0, min(3, start_chapter))
        self.player_x = Float32(420.0) if start_chapter == 3 else Float32(260.0)
        self.player_vx = 0.0
        self.facing_right = True
        self.walk_phase = 0.0
        self.dog_x = self.player_x + Float32(82.0)
        self.dog_y = 0.0
        self.dog_hop_t = 0.0
        self.cam_x = max(Float32(0.0), self.player_x - Float32(460.0))
        self.clock = Float32(12.4)
        self.bell_swing = 0.0
        self.bell_energy = 0.0
        self.sound_on = True
        self.fade_alpha = 0.0
        self.fade_dir = 0
        self.pending_chapter = self.chapter
        self.particles = List[LeafParticle](capacity=96)
        self.ripples = List[RippleRing](capacity=16)
        self.bell_sound_id = -1
        self.chime_sound_id = -1
        self.amb_voice_started = False

    def spawn_leaf_burst(mut self, wx: Float32, wy: Float32, count: Int):
        var cols = [
            _rgba(222, 68, 42, 245),
            _rgba(242, 128, 38, 245),
            _rgba(248, 186, 56, 245),
            _rgba(196, 48, 44, 240),
        ]
        for i in range(count):
            var a = _hash01(i * 31 + Int(self.clock * 100.0)) * Float32(3.14159)
            var spd = Float32(70.0) + _hash01(i * 17 + 7) * Float32(140.0)
            var p = LeafParticle(
                wx + (_hash01(i * 13) - Float32(0.5)) * Float32(36.0),
                wy - _hash01(i * 19) * Float32(16.0),
                cos(a) * spd,
                -abs(sin(a)) * spd - Float32(55.0),
                Float32(0.0),
                Float32(1.8) + _hash01(i * 23) * Float32(1.4),
                cols[i % 4],
                Float32(5.0) + _hash01(i * 29) * Float32(5.0),
            )
            self.particles.append(p)

    def spawn_bell_ripples(mut self, wx: Float32, wy: Float32):
        self.bell_energy = Float32(1.0)
        self.bell_swing = Float32(1.0)
        self.ripples.append(RippleRing(wx, wy, Float32(0.0), Float32(2.6), _rgba(255, 214, 112, 220)))
        self.ripples.append(RippleRing(wx, wy, Float32(-0.28), Float32(2.6), _rgba(255, 178, 86, 185)))
        self.ripples.append(RippleRing(wx, wy, Float32(-0.56), Float32(2.6), _rgba(255, 146, 78, 155)))

    def request_chapter(mut self, target: Int):
        var c = max(0, min(3, target))
        if c == self.chapter and self.fade_dir == 0:
            return
        self.pending_chapter = c
        self.fade_dir = 1

    def trigger_interact(mut self, mut audio: AudioServer):
        var ground_y = Float32(560.0)
        if self.chapter == 0:
            # Emberleaf Falls: Shiba leaps joyfully into autumn leaves!
            self.dog_hop_t = Float32(1.0)
            self.spawn_leaf_burst(self.dog_x, ground_y - Float32(12.0), 22)
            if self.sound_on and self.chime_sound_id >= 0:
                _ = audio.play_2d(self.chime_sound_id, volume=Float32(0.45), pitch=Float32(1.08), bus_id=BUS_SFX)
        elif self.chapter == 1:
            # Water City: gentle water sparkle ripples around the gondola
            self.ripples.append(RippleRing(self.player_x + Float32(20.0), ground_y + Float32(26.0), Float32(0.0), Float32(2.0), _rgba(255, 228, 162, 180)))
            self.dog_hop_t = Float32(0.75)
            if self.sound_on and self.chime_sound_id >= 0:
                _ = audio.play_2d(self.chime_sound_id, volume=Float32(0.40), pitch=Float32(0.95), bus_id=BUS_SFX)
        elif self.chapter == 2:
            # Starfall Dunes: warm lantern pulse & Shiba hop under the stars
            self.ripples.append(RippleRing(self.player_x + Float32(26.0), ground_y - Float32(58.0), Float32(0.0), Float32(2.2), _rgba(255, 206, 108, 210)))
            self.dog_hop_t = Float32(0.85)
            if self.sound_on and self.chime_sound_id >= 0:
                _ = audio.play_2d(self.chime_sound_id, volume=Float32(0.42), pitch=Float32(1.20), bus_id=BUS_SFX)
        else:
            # Thousand Gates (Kyoto): Ring the bronze temple bell (bonsho)!
            var bell_wx = Float32(438.0) if abs(self.player_x - Float32(438.0)) < Float32(560.0) else self.player_x + Float32(36.0)
            self.spawn_bell_ripples(bell_wx, ground_y - Float32(120.0))
            self.spawn_leaf_burst(bell_wx, ground_y - Float32(140.0), 12)
            if self.sound_on and self.bell_sound_id >= 0:
                _ = audio.play_2d(self.bell_sound_id, volume=Float32(0.68), pitch=Float32(1.0), bus_id=BUS_SFX)

    def update(mut self, dt: Float32, move_axis: Float32, interact_pressed: Bool, mut audio: AudioServer):
        self.clock += dt
        if self.fade_dir == 1:
            self.fade_alpha += dt * Float32(2.2)
            if self.fade_alpha >= Float32(1.0):
                self.fade_alpha = Float32(1.0)
                self.chapter = self.pending_chapter
                self.player_x = Float32(420.0) if self.chapter == 3 else Float32(180.0)
                self.dog_x = self.player_x + Float32(82.0)
                self.cam_x = max(Float32(0.0), self.player_x - Float32(420.0))
                self.particles.clear()
                self.ripples.clear()
                self.fade_dir = -1
        elif self.fade_dir == -1:
            self.fade_alpha -= dt * Float32(2.0)
            if self.fade_alpha <= Float32(0.0):
                self.fade_alpha = Float32(0.0)
                self.fade_dir = 0

        var max_speed = Float32(155.0) if self.chapter == 1 else Float32(185.0)
        var target_vx = move_axis * max_speed
        var blend = _clampf(dt * Float32(7.5), Float32(0.0), Float32(1.0))
        self.player_vx += (target_vx - self.player_vx) * blend
        self.player_x += self.player_vx * dt
        if abs(self.player_vx) > Float32(6.0):
            self.facing_right = self.player_vx > Float32(0.0)
            self.walk_phase += dt * abs(self.player_vx) * Float32(0.055)

        if self.player_x < Float32(70.0):
            self.player_x = Float32(70.0)
        if self.player_x > CHAPTER_LEN - Float32(90.0):
            # Gentle transition to the next chapter when reaching the far right!
            self.request_chapter((self.chapter + 1) % 4)
            self.player_x = CHAPTER_LEN - Float32(90.0)

        # Loyal Shiba companion trots slightly ahead or beside the Kid
        var dog_lead = Float32(84.0) if self.facing_right else Float32(-84.0)
        if self.chapter == 1:
            dog_lead = Float32(54.0)
        var dog_target = self.player_x + dog_lead
        self.dog_x += (dog_target - self.dog_x) * _clampf(dt * Float32(3.6), Float32(0.0), Float32(1.0))

        # Auto-hop in Chapter 0 when Shiba passes a raked leaf pile!
        if self.chapter == 0 and self.dog_hop_t <= Float32(0.0):
            if abs(self.dog_x - Float32(680.0)) < Float32(28.0) or abs(self.dog_x - Float32(1420.0)) < Float32(28.0):
                self.dog_hop_t = Float32(1.0)
                self.spawn_leaf_burst(self.dog_x, Float32(548.0), 16)
                if self.sound_on and self.chime_sound_id >= 0:
                    _ = audio.play_2d(self.chime_sound_id, volume=Float32(0.32), pitch=Float32(1.12), bus_id=BUS_SFX)

        if self.dog_hop_t > Float32(0.0):
            self.dog_hop_t = max(Float32(0.0), self.dog_hop_t - dt * Float32(1.8))
            var u = Float32(1.0) - self.dog_hop_t
            self.dog_y = Float32(4.0) * u * (Float32(1.0) - u) * Float32(52.0)
        else:
            var trot = abs(self.dog_x - dog_target)
            self.dog_y = abs(sin(self.clock * Float32(9.0))) * min(Float32(8.0), trot * Float32(0.18))

        if interact_pressed:
            self.trigger_interact(audio)

        # Bell oscillation decay
        if self.bell_energy > Float32(0.0):
            self.bell_energy = max(Float32(0.0), self.bell_energy - dt * Float32(0.38))
            self.bell_swing = sin(self.clock * Float32(7.0)) * self.bell_energy

        # Smooth camera follow
        var target_cam = _clampf(self.player_x - Float32(420.0), Float32(0.0), CHAPTER_LEN - Float32(SCREEN_W))
        self.cam_x += (target_cam - self.cam_x) * _clampf(dt * Float32(4.2), Float32(0.0), Float32(1.0))

        # Update leaf particles
        var i = len(self.particles) - 1
        while i >= 0:
            self.particles[i].life += dt
            self.particles[i].x += self.particles[i].vx * dt + sin(self.clock * Float32(3.2) + Float32(i)) * Float32(22.0) * dt
            self.particles[i].y += self.particles[i].vy * dt
            self.particles[i].vy += Float32(95.0) * dt
            self.particles[i].vx *= Float32(0.985)
            if self.particles[i].life >= self.particles[i].max_life:
                _ = self.particles.pop(i)
            i -= 1

        # Update sound ripples
        var ri = len(self.ripples) - 1
        while ri >= 0:
            self.ripples[ri].age += dt
            if self.ripples[ri].age >= self.ripples[ri].max_age:
                _ = self.ripples.pop(ri)
            ri -= 1


def _poly(mut dl: UiDrawList, pts: List[Float32], color: UInt32):
    var counts = List[Int](capacity=1)
    counts.append(len(pts) // 2)
    _ = dl.add_polygon(pts, counts, color)


def _draw_hills(
    mut dl: UiDrawList,
    cam_x: Float32,
    parallax: Float32,
    base_y: Float32,
    amp1: Float32,
    amp2: Float32,
    seed: Float32,
    color: UInt32,
):
    var pts = List[Float32](capacity=40)
    pts.append(-10.0)
    pts.append(Float32(SCREEN_H + 10))
    var steps = 16
    for i in range(steps + 1):
        var sx = Float32(i) * (Float32(SCREEN_W + 20) / Float32(steps)) - Float32(10.0)
        var wx = (sx + cam_x * parallax) * Float32(0.0042) + seed
        var hy = base_y - sin(wx) * amp1 - cos(wx * Float32(2.1) + seed) * amp2
        pts.append(sx)
        pts.append(hy)
    pts.append(Float32(SCREEN_W + 10))
    pts.append(Float32(SCREEN_H + 10))
    _poly(dl, pts, color)


def _draw_maple_tree(mut dl: UiDrawList, sx: Float32, ground_y: Float32, scale: Float32, t: Float32, crimson: Bool):
    var sway = sin(t * Float32(1.1) + sx * Float32(0.01)) * Float32(4.0) * scale
    # Organic curved trunk & limbs
    dl.add_line(sx, ground_y + Float32(4.0), sx + sway * Float32(0.3), ground_y - Float32(165.0) * scale, Float32(18.0) * scale, _rgba(66, 38, 30))
    dl.add_line(sx + Float32(2.0), ground_y - Float32(95.0) * scale, sx - Float32(55.0) * scale + sway, ground_y - Float32(155.0) * scale, Float32(9.0) * scale, _rgba(66, 38, 30))
    dl.add_line(sx + Float32(2.0), ground_y - Float32(110.0) * scale, sx + Float32(58.0) * scale + sway, ground_y - Float32(168.0) * scale, Float32(9.0) * scale, _rgba(66, 38, 30))
    # Layered painterly foliage puffs
    var c_dark = _rgba(148, 34, 32, 245) if crimson else _rgba(176, 56, 28, 245)
    var c_mid = _rgba(198, 56, 34, 240) if crimson else _rgba(224, 102, 34, 240)
    var c_lit = _rgba(236, 126, 46, 232) if crimson else _rgba(246, 174, 52, 232)
    dl.add_circle(sx - Float32(54.0) * scale + sway, ground_y - Float32(168.0) * scale, Float32(56.0) * scale, c_dark)
    dl.add_circle(sx + Float32(56.0) * scale + sway, ground_y - Float32(174.0) * scale, Float32(58.0) * scale, c_dark)
    dl.add_circle(sx + sway, ground_y - Float32(215.0) * scale, Float32(72.0) * scale, c_mid)
    dl.add_circle(sx - Float32(36.0) * scale + sway, ground_y - Float32(198.0) * scale, Float32(46.0) * scale, c_mid)
    dl.add_circle(sx + Float32(42.0) * scale + sway, ground_y - Float32(204.0) * scale, Float32(44.0) * scale, c_lit)
    dl.add_circle(sx + Float32(8.0) * scale + sway, ground_y - Float32(236.0) * scale, Float32(40.0) * scale, c_lit)


def _draw_kyoto_pine(mut dl: UiDrawList, sx: Float32, ground_y: Float32, scale: Float32, t: Float32):
    var sway = sin(t * Float32(0.8) + Float32(1.4)) * Float32(2.5) * scale
    # Sculpted curved Japanese pine trunk
    dl.add_quad_bezier(
        sx, ground_y + Float32(6.0),
        sx - Float32(26.0) * scale, ground_y - Float32(130.0) * scale,
        sx + Float32(14.0) * scale + sway, ground_y - Float32(275.0) * scale,
        Float32(19.0) * scale, _rgba(68, 46, 36),
    )
    # Horizontal boughs
    dl.add_line(sx - Float32(8.0) * scale, ground_y - Float32(145.0) * scale, sx - Float32(95.0) * scale + sway, ground_y - Float32(158.0) * scale, Float32(8.0) * scale, _rgba(68, 46, 36))
    dl.add_line(sx + Float32(2.0) * scale, ground_y - Float32(195.0) * scale, sx + Float32(96.0) * scale + sway, ground_y - Float32(206.0) * scale, Float32(8.0) * scale, _rgba(68, 46, 36))
    # Layered flat evergreen cloud pads
    var pads_x = [-92.0, 92.0, -34.0, 38.0, 6.0]
    var pads_y = [-164.0, -214.0, -252.0, -282.0, -314.0]
    var pads_w = [122.0, 128.0, 134.0, 114.0, 102.0]
    for i in range(5):
        var px = sx + Float32(pads_x[i]) * scale + sway
        var py = ground_y + Float32(pads_y[i]) * scale
        var pw = Float32(pads_w[i]) * scale
        var ph = Float32(34.0) * scale
        dl.add_rect(px - pw * Float32(0.5), py - ph * Float32(0.45), pw, ph, _rgba(36, 66, 44, 245), ph * Float32(0.5))
        dl.add_rect(px - pw * Float32(0.42), py - ph * Float32(0.62), pw * Float32(0.84), ph * Float32(0.66), _rgba(62, 104, 58, 235), ph * Float32(0.35))


def _draw_torii_gate(mut dl: UiDrawList, sx: Float32, ground_y: Float32, scale: Float32):
    var red = _rgba(206, 52, 34, 245)
    var red_dark = _rgba(154, 34, 24, 245)
    var cap = _rgba(34, 28, 36, 250)
    var h = Float32(175.0) * scale
    var w = Float32(126.0) * scale
    var pw = Float32(13.0) * scale
    # Stone plinths + pillars
    dl.add_rect(sx - w * Float32(0.44) - pw * Float32(0.6), ground_y - Float32(14.0) * scale, pw * Float32(1.2), Float32(18.0) * scale, cap, Float32(2.0))
    dl.add_rect(sx + w * Float32(0.44) - pw * Float32(0.6), ground_y - Float32(14.0) * scale, pw * Float32(1.2), Float32(18.0) * scale, cap, Float32(2.0))
    dl.add_rect(sx - w * Float32(0.42) - pw * Float32(0.5), ground_y - h, pw, h, red, Float32(2.0))
    dl.add_rect(sx + w * Float32(0.42) - pw * Float32(0.5), ground_y - h, pw, h, red, Float32(2.0))
    # Lower nuki tie-beam + central tablet
    dl.add_rect(sx - w * Float32(0.60), ground_y - h + Float32(34.0) * scale, w * Float32(1.20), Float32(12.0) * scale, red_dark, Float32(2.0))
    dl.add_rect(sx - Float32(10.0) * scale, ground_y - h + Float32(12.0) * scale, Float32(20.0) * scale, Float32(26.0) * scale, cap, Float32(2.0))
    # Upturned kasagi roof lintel
    var kasagi = List[Float32](capacity=12)
    kasagi.append(sx - w * Float32(0.76)); kasagi.append(ground_y - h - Float32(10.0) * scale)
    kasagi.append(sx - w * Float32(0.40)); kasagi.append(ground_y - h - Float32(2.0) * scale)
    kasagi.append(sx + w * Float32(0.40)); kasagi.append(ground_y - h - Float32(2.0) * scale)
    kasagi.append(sx + w * Float32(0.76)); kasagi.append(ground_y - h - Float32(10.0) * scale)
    kasagi.append(sx + w * Float32(0.70)); kasagi.append(ground_y - h + Float32(12.0) * scale)
    kasagi.append(sx - w * Float32(0.70)); kasagi.append(ground_y - h + Float32(12.0) * scale)
    _poly(dl, kasagi, red)
    dl.add_rect(sx - w * Float32(0.74), ground_y - h - Float32(11.0) * scale, w * Float32(1.48), Float32(8.0) * scale, cap, Float32(3.0))


def _draw_stone_lantern(mut dl: UiDrawList, sx: Float32, ground_y: Float32, scale: Float32, t: Float32):
    var st = _rgba(128, 122, 118, 245)
    var st_d = _rgba(88, 82, 82, 245)
    var pulse = Float32(0.88) + Float32(0.12) * sin(t * Float32(3.4) + sx * Float32(0.05))
    dl.add_rect(sx - Float32(14.0) * scale, ground_y - Float32(12.0) * scale, Float32(28.0) * scale, Float32(14.0) * scale, st_d, Float32(3.0))
    dl.add_rect(sx - Float32(7.0) * scale, ground_y - Float32(52.0) * scale, Float32(14.0) * scale, Float32(42.0) * scale, st, Float32(2.0))
    dl.add_glow_rect(
        sx - Float32(10.0) * scale, ground_y - Float32(72.0) * scale,
        Float32(20.0) * scale, Float32(20.0) * scale,
        _rgba(255, 224, 126, 250), _rgba(255, 174, 62, Int(Float32(175.0) * pulse)),
        Float32(4.0), Float32(28.0) * scale,
    )
    var cap = List[Float32](capacity=8)
    cap.append(sx - Float32(24.0) * scale); cap.append(ground_y - Float32(70.0) * scale)
    cap.append(sx); cap.append(ground_y - Float32(92.0) * scale)
    cap.append(sx + Float32(24.0) * scale); cap.append(ground_y - Float32(70.0) * scale)
    _poly(dl, cap, st_d)


def _draw_belfry_pavilion(mut dl: UiDrawList, sx: Float32, ground_y: Float32, swing: Float32, t: Float32):
    # Stone platform
    dl.add_rect(sx - Float32(138.0), ground_y - Float32(20.0), Float32(276.0), Float32(26.0), _rgba(126, 118, 114), Float32(4.0))
    # Back & front cedar pillars
    var wood = _rgba(78, 44, 30)
    var wood_d = _rgba(54, 30, 22)
    dl.add_rect(sx - Float32(106.0), ground_y - Float32(184.0), Float32(16.0), Float32(166.0), wood, Float32(2.0))
    dl.add_rect(sx + Float32(90.0), ground_y - Float32(184.0), Float32(16.0), Float32(166.0), wood, Float32(2.0))
    dl.add_rect(sx - Float32(118.0), ground_y - Float32(184.0), Float32(236.0), Float32(14.0), wood_d, Float32(2.0))

    # Hanging Bronze Temple Bell (Bonsho)
    var bx = sx - Float32(16.0) + swing * Float32(10.0)
    var by = ground_y - Float32(126.0)
    dl.add_line(sx - Float32(16.0), ground_y - Float32(174.0), bx, by - Float32(22.0), Float32(5.0), _rgba(48, 42, 40))
    dl.add_rect(bx - Float32(34.0), by - Float32(22.0), Float32(68.0), Float32(78.0), _rgba(76, 98, 92), Float32(24.0), Float32(2.0), _rgba(44, 58, 56))
    dl.add_rect(bx - Float32(39.0), by + Float32(42.0), Float32(78.0), Float32(16.0), _rgba(64, 84, 78), Float32(6.0))
    # Wooden striker log (Shumoku)
    var log_x = sx + Float32(46.0) - swing * Float32(24.0)
    var log_y = by + Float32(26.0)
    dl.add_line(sx + Float32(58.0), ground_y - Float32(174.0), log_x + Float32(20.0), log_y, Float32(2.5), _rgba(142, 118, 88))
    dl.add_rect(log_x - Float32(22.0), log_y - Float32(9.0), Float32(92.0), Float32(18.0), _rgba(138, 86, 48), Float32(7.0), Float32(1.5), _rgba(74, 42, 24))

    # Curved Storybook Slate Roof
    var roof = List[Float32](capacity=12)
    roof.append(sx - Float32(176.0)); roof.append(ground_y - Float32(176.0))
    roof.append(sx - Float32(96.0)); roof.append(ground_y - Float32(248.0))
    roof.append(sx + Float32(96.0)); roof.append(ground_y - Float32(248.0))
    roof.append(sx + Float32(176.0)); roof.append(ground_y - Float32(176.0))
    roof.append(sx + Float32(146.0)); roof.append(ground_y - Float32(166.0))
    roof.append(sx - Float32(146.0)); roof.append(ground_y - Float32(166.0))
    _poly(dl, roof, _rgba(54, 46, 58, 250))
    dl.add_rect(sx - Float32(104.0), ground_y - Float32(256.0), Float32(208.0), Float32(12.0), _rgba(38, 32, 44), Float32(5.0))

    # Two warm glowing paper lanterns under the eaves!
    var pulse = Float32(0.90) + Float32(0.10) * sin(t * Float32(2.6))
    for side in range(2):
        var lx = sx + (Float32(-72.0) if side == 0 else Float32(66.0))
        var ly = ground_y - Float32(158.0)
        dl.add_glow_rect(
            lx - Float32(12.0), ly, Float32(24.0), Float32(34.0),
            _rgba(255, 224, 128, 250), _rgba(255, 168, 56, Int(Float32(215.0) * pulse)),
            Float32(9.0), Float32(42.0),
        )
        dl.add_rect(lx - Float32(10.0), ly - Float32(3.0), Float32(20.0), Float32(5.0), _rgba(54, 30, 24), Float32(2.0))
        dl.add_rect(lx - Float32(10.0), ly + Float32(31.0), Float32(20.0), Float32(5.0), _rgba(54, 30, 24), Float32(2.0))


def _draw_kid(mut dl: UiDrawList, sx: Float32, ground_y: Float32, walk_phase: Float32, facing_right: Bool, carry_lantern: Bool, t: Float32):
    var dir = Float32(1.0) if facing_right else Float32(-1.0)
    var bob = abs(sin(walk_phase)) * Float32(4.0)
    var leg_sw = sin(walk_phase) * Float32(10.0)
    var y0 = ground_y - bob

    # Soft ground shadow
    dl.add_rect(sx - Float32(22.0), ground_y - Float32(4.0), Float32(44.0), Float32(10.0), _rgba(24, 18, 24, 95), Float32(5.0))
    # Dark trousers & boots
    dl.add_line(sx - Float32(5.0), y0 - Float32(34.0), sx - Float32(6.0) + leg_sw, ground_y - Float32(3.0), Float32(9.0), _rgba(46, 50, 68))
    dl.add_line(sx + Float32(5.0), y0 - Float32(34.0), sx + Float32(6.0) - leg_sw, ground_y - Float32(3.0), Float32(9.0), _rgba(38, 42, 58))
    # Cozy adventurer backpack (behind)
    dl.add_rect(sx - dir * Float32(22.0) - Float32(9.0), y0 - Float32(78.0), Float32(18.0), Float32(34.0), _rgba(116, 74, 46), Float32(7.0), Float32(1.5), _rgba(64, 38, 24))
    # Golden-ochre storybook jacket
    dl.add_rect(sx - Float32(15.0), y0 - Float32(82.0), Float32(30.0), Float32(48.0), _rgba(232, 166, 48), Float32(10.0), Float32(1.5), _rgba(128, 82, 24))
    # Red scarf
    dl.add_rect(sx - Float32(14.0), y0 - Float32(86.0), Float32(28.0), Float32(9.0), _rgba(202, 56, 46), Float32(4.0))
    # Head & face
    dl.add_circle(sx + dir * Float32(2.0), y0 - Float32(99.0), Float32(15.0), _rgba(246, 202, 164), Float32(1.5), _rgba(112, 74, 58))
    dl.add_circle(sx + dir * Float32(8.0), y0 - Float32(100.0), Float32(2.2), _rgba(38, 28, 28))
    # Iconic crimson red beanie + pom-pom!
    dl.add_rect(sx - Float32(17.0), y0 - Float32(117.0), Float32(34.0), Float32(16.0), _rgba(206, 44, 42), Float32(8.0), Float32(1.5), _rgba(112, 24, 24))
    dl.add_circle(sx - dir * Float32(5.0), y0 - Float32(120.0), Float32(6.0), _rgba(244, 224, 198))
    # Forward arm (+ optional warm glowing hand-lantern!)
    var hand_x = sx + dir * Float32(24.0)
    var hand_y = y0 - Float32(58.0)
    dl.add_line(sx + dir * Float32(6.0), y0 - Float32(72.0), hand_x, hand_y, Float32(8.0), _rgba(222, 154, 42))
    if carry_lantern:
        var pulse = Float32(0.88) + Float32(0.12) * sin(t * Float32(3.2))
        dl.add_line(hand_x, hand_y, hand_x + dir * Float32(4.0), hand_y + Float32(10.0), Float32(2.0), _rgba(96, 72, 48))
        dl.add_glow_rect(
            hand_x + dir * Float32(4.0) - Float32(7.0), hand_y + Float32(10.0),
            Float32(14.0), Float32(19.0),
            _rgba(255, 234, 146, 255), _rgba(255, 178, 58, Int(Float32(225.0) * pulse)),
            Float32(6.0), Float32(78.0),
        )


def _draw_shiba(mut dl: UiDrawList, sx: Float32, ground_y: Float32, hop_y: Float32, facing_right: Bool, t: Float32):
    var dir = Float32(1.0) if facing_right else Float32(-1.0)
    var y0 = ground_y - hop_y
    var wag = sin(t * Float32(11.0)) * Float32(5.0)
    # Shadow
    dl.add_rect(sx - Float32(18.0), ground_y - Float32(3.0), Float32(36.0), Float32(8.0), _rgba(24, 18, 24, 85), Float32(4.0))
    # Curled fluffy tail
    dl.add_arc(sx - dir * Float32(20.0), y0 - Float32(34.0) + wag * Float32(0.25), Float32(9.0), Float32(6.0), Float32(0.4), Float32(4.2), _rgba(232, 134, 52))
    # Legs
    dl.add_line(sx - Float32(11.0), y0 - Float32(18.0), sx - Float32(12.0), y0 - Float32(2.0), Float32(6.0), _rgba(246, 232, 206))
    dl.add_line(sx + Float32(11.0), y0 - Float32(18.0), sx + Float32(12.0), y0 - Float32(2.0), Float32(6.0), _rgba(246, 232, 206))
    # Orange-amber body + cream chest
    dl.add_rect(sx - Float32(19.0), y0 - Float32(34.0), Float32(38.0), Float32(20.0), _rgba(226, 126, 46), Float32(9.0), Float32(1.2), _rgba(128, 66, 24))
    dl.add_circle(sx + dir * Float32(12.0), y0 - Float32(24.0), Float32(9.5), _rgba(248, 236, 214))
    # Shiba head + cream cheeks + dark nose + perked triangular ears!
    var hx = sx + dir * Float32(17.0)
    var hy = y0 - Float32(40.0)
    dl.add_circle(hx, hy, Float32(12.5), _rgba(230, 132, 50), Float32(1.2), _rgba(128, 66, 24))
    dl.add_circle(hx + dir * Float32(6.0), hy + Float32(3.5), Float32(7.0), _rgba(250, 240, 222))
    dl.add_circle(hx + dir * Float32(12.5), hy + Float32(1.5), Float32(2.6), _rgba(36, 28, 26))
    dl.add_circle(hx + dir * Float32(5.0), hy - Float32(2.5), Float32(2.0), _rgba(32, 24, 24))
    # Triangular perked ears
    var ear1 = List[Float32](capacity=6)
    ear1.append(hx - Float32(8.0)); ear1.append(hy - Float32(8.0))
    ear1.append(hx - Float32(4.0)); ear1.append(hy - Float32(22.0))
    ear1.append(hx + Float32(2.0)); ear1.append(hy - Float32(10.0))
    _poly(dl, ear1, _rgba(220, 118, 42))
    var ear2 = List[Float32](capacity=6)
    ear2.append(hx + Float32(1.0)); ear2.append(hy - Float32(9.0))
    ear2.append(hx + Float32(7.0)); ear2.append(hy - Float32(21.0))
    ear2.append(hx + Float32(10.0)); ear2.append(hy - Float32(7.0))
    _poly(dl, ear2, _rgba(236, 144, 62))


def _render_chapter(mut dl: UiDrawList, game: StoryGame, font_slot: Int, font: SdfFont):
    var w = Float32(SCREEN_W)
    var h = Float32(SCREEN_H)
    var ground_y = Float32(560.0)
    var cx = game.cam_x
    var t = game.clock

    if game.chapter == 0:
        # III EMBERLEAF FALLS — Golden Autumn Sky & Mist
        dl.add_rect_styled(0.0, 0.0, w, h * Float32(0.62), _rgba(218, 128, 82), _rgba(248, 206, 134), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        dl.add_rect_styled(0.0, h * Float32(0.58), w, h * Float32(0.42), _rgba(248, 206, 134), _rgba(198, 136, 92), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        # Warm sunset sun
        dl.add_glow_rect(w * Float32(0.68) - Float32(46.0), Float32(118.0), Float32(92.0), Float32(92.0), _rgba(255, 240, 184), _rgba(255, 198, 96, 185), Float32(46.0), Float32(90.0))
        _draw_hills(dl, cx, Float32(0.18), Float32(420.0), Float32(48.0), Float32(24.0), Float32(1.2), _rgba(146, 86, 88, 210))
        _draw_hills(dl, cx, Float32(0.38), Float32(478.0), Float32(36.0), Float32(18.0), Float32(3.5), _rgba(116, 72, 58, 235))
        # Shimmering autumn lake strip
        dl.add_rect_styled(0.0, Float32(476.0), w, Float32(96.0), _rgba(136, 152, 168), _rgba(212, 166, 118), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        # Stone Watermill & Rotating Wooden Waterwheel!
        var mill_x = Float32(960.0) - cx * Float32(0.85)
        dl.add_rect(mill_x - Float32(72.0), ground_y - Float32(176.0), Float32(144.0), Float32(178.0), _rgba(136, 122, 112), Float32(6.0))
        var mroof = List[Float32](capacity=6)
        mroof.append(mill_x - Float32(94.0)); mroof.append(ground_y - Float32(172.0))
        mroof.append(mill_x); mroof.append(ground_y - Float32(246.0))
        mroof.append(mill_x + Float32(94.0)); mroof.append(ground_y - Float32(172.0))
        _poly(dl, mroof, _rgba(104, 52, 42))
        dl.add_glow_rect(mill_x - Float32(22.0), ground_y - Float32(126.0), Float32(30.0), Float32(38.0), _rgba(255, 224, 134), _rgba(255, 168, 56, 165), Float32(10.0), Float32(28.0))
        # Rotating wooden wheel
        var wx = mill_x + Float32(64.0)
        var wy = ground_y - Float32(54.0)
        dl.add_arc(wx, wy, Float32(52.0), Float32(8.0), Float32(0.0), TWO_PI, _rgba(114, 68, 38))
        for sp in range(6):
            var a = t * Float32(0.85) + Float32(sp) * Float32(1.0472)
            dl.add_line(wx - cos(a) * Float32(50.0), wy - sin(a) * Float32(50.0), wx + cos(a) * Float32(50.0), wy + sin(a) * Float32(50.0), Float32(5.0), _rgba(92, 54, 30))
        # Autumn Maple Trees along the path
        for ti in range(5):
            var tx = Float32(160 + ti * 460) - cx
            _draw_maple_tree(dl, tx, ground_y, Float32(0.95) + Float32(ti % 2) * Float32(0.18), t, (ti % 2) == 0)
        # Mossy autumn meadow & sunlit path
        dl.add_rect_styled(0.0, ground_y - Float32(10.0), w, h - ground_y + Float32(10.0), _rgba(96, 108, 54), _rgba(58, 66, 34), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        dl.add_rect(0.0, ground_y - Float32(4.0), w, Float32(36.0), _rgba(164, 128, 86, 230), 0.0)
        # Raked leaf piles & pumpkins!
        for pi in range(2):
            var lx = (Float32(680.0) if pi == 0 else Float32(1420.0)) - cx
            dl.add_circle(lx, ground_y + Float32(2.0), Float32(26.0), _rgba(218, 92, 32))
            dl.add_circle(lx - Float32(18.0), ground_y + Float32(6.0), Float32(19.0), _rgba(238, 146, 40))
            dl.add_circle(lx + Float32(20.0), ground_y + Float32(6.0), Float32(18.0), _rgba(192, 54, 30))
            dl.add_circle(lx - Float32(64.0), ground_y - Float32(6.0), Float32(15.0), _rgba(236, 114, 24), Float32(1.5), _rgba(136, 56, 16))
        _draw_kid(dl, game.player_x - cx, ground_y, game.walk_phase, game.facing_right, False, t)
        _draw_shiba(dl, game.dog_x - cx, ground_y, game.dog_y, game.facing_right, t)

    elif game.chapter == 1:
        # IX THE WATER CITY — Golden-Hour Venice Canals & Gondola Glide
        dl.add_rect_styled(0.0, 0.0, w, h * Float32(0.65), _rgba(146, 174, 206), _rgba(248, 204, 152), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        dl.add_glow_rect(w * Float32(0.24), Float32(110.0), Float32(84.0), Float32(84.0), _rgba(255, 242, 198), _rgba(255, 204, 112, 175), Float32(42.0), Float32(80.0))
        # Pastel terracotta, ochre & cream Venetian palazzos along the canal!
        var bcols = [_rgba(222, 156, 114), _rgba(238, 204, 146), _rgba(204, 116, 92), _rgba(232, 186, 134)]
        for bi in range(9):
            var bx = Float32(bi * 250) - cx * Float32(0.62)
            var bh = Float32(220 + (bi % 3) * 48)
            var bw = Float32(212.0)
            dl.add_rect(bx, Float32(470.0) - bh, bw, bh, bcols[bi % 4], Float32(4.0), Float32(1.5), _rgba(128, 86, 68))
            dl.add_rect(bx - Float32(6.0), Float32(470.0) - bh - Float32(12.0), bw + Float32(12.0), Float32(16.0), _rgba(154, 72, 54), Float32(4.0))
            # Arched windows with green Venetian shutters
            for wi in range(3):
                var wx = bx + Float32(34 + wi * 62)
                var wy = Float32(470.0) - bh + Float32(42.0)
                dl.add_rect(wx, wy, Float32(26.0), Float32(42.0), _rgba(62, 74, 88), Float32(12.0))
                dl.add_rect(wx - Float32(9.0), wy + Float32(8.0), Float32(8.0), Float32(34.0), _rgba(66, 116, 86), Float32(2.0))
                dl.add_rect(wx + Float32(27.0), wy + Float32(8.0), Float32(8.0), Float32(34.0), _rgba(66, 116, 86), Float32(2.0))
        # Arched Stone Footbridge
        var brx = Float32(1040.0) - cx * Float32(0.82)
        dl.add_arc(brx, Float32(512.0), Float32(138.0), Float32(30.0), Float32(3.14159), Float32(3.14159), _rgba(222, 206, 186))
        # Shimmering Blue-Gold Canal Water
        dl.add_rect_styled(0.0, Float32(470.0), w, h - Float32(470.0), _rgba(62, 114, 138), _rgba(36, 72, 98), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        for ri in range(18):
            var ry = Float32(486 + ri * 12)
            var rx = Float32((ri * 173) % 1280) + sin(t * Float32(1.8) + Float32(ri)) * Float32(24.0)
            dl.add_line(rx, ry, rx + Float32(72.0), ry, Float32(2.5), _rgba(255, 224, 152, 86))
        # Wooden mooring poles
        for mi in range(5):
            var mx = Float32(220 + mi * 460) - cx
            dl.add_rect(mx, Float32(436.0), Float32(12.0), Float32(105.0), _rgba(128, 72, 48), Float32(4.0))
        # Sculpted Gondola carrying the Kid & Shiba smoothly across the canal!
        var gx = game.player_x - cx
        var bob_g = sin(t * Float32(2.2)) * Float32(4.5)
        var gy = ground_y + Float32(16.0) + bob_g
        _draw_kid(dl, gx - Float32(12.0), gy - Float32(4.0), Float32(0.0), True, False, t)
        _draw_shiba(dl, gx + Float32(48.0), gy - Float32(2.0), game.dog_y * Float32(0.5), True, t)
        var hull = List[Float32](capacity=12)
        hull.append(gx - Float32(108.0)); hull.append(gy - Float32(26.0))
        hull.append(gx - Float32(76.0)); hull.append(gy + Float32(12.0))
        hull.append(gx + Float32(82.0)); hull.append(gy + Float32(12.0))
        hull.append(gx + Float32(118.0)); hull.append(gy - Float32(34.0))
        hull.append(gx + Float32(94.0)); hull.append(gy - Float32(6.0))
        hull.append(gx - Float32(82.0)); hull.append(gy - Float32(6.0))
        _poly(dl, hull, _rgba(36, 26, 28))
        dl.add_line(gx - Float32(78.0), gy - Float32(4.0), gx + Float32(88.0), gy - Float32(4.0), Float32(3.0), _rgba(226, 178, 86))

    elif game.chapter == 2:
        # X STARFALL DUNES — Indigo Milky Way Twilight & Lantern Glow
        dl.add_rect_styled(0.0, 0.0, w, h * Float32(0.70), _rgba(18, 22, 54), _rgba(66, 56, 108), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        # Twinkling Milky Way starfield!
        for si in range(56):
            var stx = _hash01(si * 73 + 11) * w
            var sty = _hash01(si * 41 + 5) * Float32(370.0)
            var tw = Float32(0.45) + Float32(0.55) * sin(t * Float32(2.4) + Float32(si) * Float32(1.3))
            var sr = Float32(1.4) + _hash01(si * 19) * Float32(2.2)
            dl.add_circle(stx, sty, sr, _rgba(246, 240, 214, Int(Float32(230.0) * tw)))
        # Crescent moon
        dl.add_glow_rect(w * Float32(0.76), Float32(86.0), Float32(44.0), Float32(44.0), _rgba(255, 244, 206), _rgba(196, 210, 255, 130), Float32(22.0), Float32(48.0))
        # Rolling Twilight Sand Dunes
        _draw_hills(dl, cx, Float32(0.20), Float32(436.0), Float32(52.0), Float32(22.0), Float32(0.6), _rgba(56, 48, 92))
        _draw_hills(dl, cx, Float32(0.42), Float32(496.0), Float32(42.0), Float32(18.0), Float32(2.4), _rgba(76, 60, 102))
        # Passing Camel Caravan & Bright Red-and-Gold Diamond Kite!
        var car_x = Float32(1060.0) - cx * Float32(0.65)
        for ci in range(2):
            var ccx = car_x + Float32(ci * 110)
            var ccy = Float32(468.0)
            dl.add_rect(ccx - Float32(26.0), ccy - Float32(34.0), Float32(52.0), Float32(24.0), _rgba(42, 34, 62), Float32(10.0))
            dl.add_circle(ccx, ccy - Float32(38.0), Float32(14.0), _rgba(42, 34, 62))
            dl.add_line(ccx - Float32(18.0), ccy - Float32(12.0), ccx - Float32(20.0), ccy + Float32(14.0), Float32(5.0), _rgba(42, 34, 62))
            dl.add_line(ccx + Float32(18.0), ccy - Float32(12.0), ccx + Float32(20.0), ccy + Float32(14.0), Float32(5.0), _rgba(42, 34, 62))
            dl.add_line(ccx + Float32(24.0), ccy - Float32(26.0), ccx + Float32(36.0), ccy - Float32(50.0), Float32(7.0), _rgba(42, 34, 62))
        var kx = car_x + Float32(138.0) + sin(t * Float32(1.8)) * Float32(18.0)
        var ky = Float32(226.0) + cos(t * Float32(2.2)) * Float32(12.0)
        dl.add_quad_bezier(car_x + Float32(6.0), Float32(422.0), car_x + Float32(78.0), Float32(340.0), kx, ky, Float32(1.6), _rgba(218, 206, 232, 150))
        var kite = List[Float32](capacity=8)
        kite.append(kx); kite.append(ky - Float32(24.0))
        kite.append(kx + Float32(18.0)); kite.append(ky)
        kite.append(kx); kite.append(ky + Float32(26.0))
        kite.append(kx - Float32(18.0)); kite.append(ky)
        _poly(dl, kite, _rgba(232, 62, 52))
        dl.add_circle(kx, ky, Float32(6.0), _rgba(255, 204, 54))
        # Foreground moonlit sand ridge
        dl.add_rect_styled(0.0, ground_y - Float32(12.0), w, h - ground_y + Float32(12.0), _rgba(92, 72, 108), _rgba(46, 36, 66), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        _draw_kid(dl, game.player_x - cx, ground_y, game.walk_phase, game.facing_right, True, t)
        _draw_shiba(dl, game.dog_x - cx, ground_y, game.dog_y, game.facing_right, t)

    else:
        # XI THE THOUSAND GATES — Kyoto Dusk Sunset, Torii Gates, Pine, Belfry & Koi Pond!
        dl.add_rect_styled(0.0, 0.0, w, h * Float32(0.45), _rgba(128, 92, 134), _rgba(234, 154, 122), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        dl.add_rect_styled(0.0, h * Float32(0.43), w, h * Float32(0.38), _rgba(234, 154, 122), _rgba(255, 214, 156), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        # Warm dusk sun behind the Kyoto hills
        dl.add_glow_rect(w * Float32(0.56) - Float32(54.0), Float32(246.0), Float32(108.0), Float32(108.0), _rgba(255, 242, 198), _rgba(255, 186, 98, 195), Float32(54.0), Float32(115.0))
        _draw_hills(dl, cx, Float32(0.18), Float32(436.0), Float32(44.0), Float32(22.0), Float32(2.1), _rgba(128, 96, 122, 220))
        _draw_hills(dl, cx, Float32(0.36), Float32(488.0), Float32(32.0), Float32(16.0), Float32(4.3), _rgba(88, 86, 76, 235))

        # Vermilion Shinto Torii Gates along the sunlit trail!
        for ti in range(4):
            var tx = Float32(980 + ti * 220) - cx * Float32(0.84)
            var tsc = Float32(1.0) - Float32(ti) * Float32(0.10)
            _draw_torii_gate(dl, tx, ground_y - Float32(18 + ti * 10), tsc)

        # Crimson Maples framing the left & right + Sculpted Kyoto Pine!
        _draw_maple_tree(dl, Float32(165.0) - cx, ground_y - Float32(6.0), Float32(1.12), t, True)
        _draw_kyoto_pine(dl, Float32(776.0) - cx, ground_y - Float32(8.0), Float32(1.05), t)
        _draw_maple_tree(dl, Float32(1240.0) - cx, ground_y - Float32(6.0), Float32(1.05), t, True)
        _draw_maple_tree(dl, Float32(1780.0) - cx, ground_y - Float32(6.0), Float32(1.10), t, False)

        # Lush Kyoto Meadow & Sunlit Stone Path
        dl.add_rect_styled(0.0, ground_y - Float32(12.0), w, h - ground_y + Float32(12.0), _rgba(78, 102, 56), _rgba(44, 58, 34), 0.0, 0.0, 1.0, UI_FLAG_VERT_GRADIENT)
        dl.add_rect(0.0, ground_y - Float32(6.0), w, Float32(36.0), _rgba(168, 142, 114, 225), 0.0)

        # Stone Garden Lanterns (Toro)
        _draw_stone_lantern(dl, Float32(654.0) - cx, ground_y - Float32(4.0), Float32(0.88), t)
        _draw_stone_lantern(dl, Float32(1460.0) - cx, ground_y - Float32(4.0), Float32(0.92), t)

        # Wooden Temple Belfry Pavilion & Hanging Bronze Bell (Bonsho)!
        _draw_belfry_pavilion(dl, Float32(438.0) - cx, ground_y - Float32(6.0), game.bell_swing, t)

        # Stone-Rimmed Lily-Pad Koi Pond (right foreground!)
        var px = Float32(1018.0) - cx
        var py = ground_y + Float32(6.0)
        dl.add_rect(px - Float32(114.0), py - Float32(28.0), Float32(228.0), Float32(56.0), _rgba(116, 112, 104), Float32(28.0))
        dl.add_rect(px - Float32(102.0), py - Float32(21.0), Float32(204.0), Float32(42.0), _rgba(34, 86, 78), Float32(21.0))
        # Lily pads + lotus + 3 gently swimming Golden Koi!
        dl.add_circle(px - Float32(52.0), py - Float32(4.0), Float32(15.0), _rgba(64, 134, 68))
        dl.add_circle(px + Float32(46.0), py + Float32(5.0), Float32(17.0), _rgba(58, 124, 64))
        dl.add_circle(px + Float32(46.0), py + Float32(2.0), Float32(6.0), _rgba(250, 162, 188))
        for ki in range(3):
            var ka = t * (Float32(0.75) + Float32(ki) * Float32(0.22)) + Float32(ki) * Float32(2.1)
            var kx = px + cos(ka) * Float32(58.0)
            var ky = py + sin(ka) * Float32(11.0)
            var kcol = _rgba(255, 128, 38) if ki == 0 else (_rgba(255, 206, 68) if ki == 1 else _rgba(248, 240, 224))
            dl.add_rect(kx - Float32(10.0), ky - Float32(3.5), Float32(20.0), Float32(7.0), kcol, Float32(3.5))

        # The Kid & Shiba Companion Dog!
        _draw_kid(dl, game.player_x - cx, ground_y, game.walk_phase, game.facing_right, False, t)
        _draw_shiba(dl, game.dog_x - cx, ground_y, game.dog_y, game.facing_right, t)

    # Drifting ambient leaves / fireflies across all chapters
    for fi in range(14):
        var fx = Float32((Int(t * Float32(26 + fi * 4) + Float32(fi * 197))) % (SCREEN_W + 80)) - Float32(40.0)
        var fy = Float32(110 + (fi * 37) % 390) + sin(t * Float32(1.5) + Float32(fi)) * Float32(22.0)
        if game.chapter == 2 or game.chapter == 3:
            dl.add_glow_rect(fx, fy, Float32(5.0), Float32(5.0), _rgba(255, 236, 134, 235), _rgba(255, 198, 72, 155), Float32(2.5), Float32(14.0))
        else:
            dl.add_circle(fx, fy, Float32(3.5), _rgba(244, 136, 44, 210))

    # Interactive burst particles (scattered autumn leaves!)
    for i in range(len(game.particles)):
        ref p = game.particles[i]
        var a_frac = _clampf(Float32(1.0) - p.life / p.max_life, Float32(0.0), Float32(1.0))
        var alpha = Int(a_frac * Float32(245.0))
        var col = (p.color & UInt32(0x00FFFFFF)) | (UInt32(alpha) << 24)
        dl.add_circle(p.x - cx, p.y, p.size, col)

    # Expanding Temple Bell / Chime Sound Ripples!
    for i in range(len(game.ripples)):
        ref r = game.ripples[i]
        if r.age > Float32(0.0):
            var u = _clampf(r.age / r.max_age, Float32(0.0), Float32(1.0))
            var rad = Float32(18.0) + u * Float32(220.0)
            var alpha = Int((Float32(1.0) - u) * (Float32(1.0) - u) * Float32(215.0))
            var col = (r.color & UInt32(0x00FFFFFF)) | (UInt32(alpha) << 24)
            dl.add_arc(r.x - cx, r.y, rad, Float32(3.2) * (Float32(1.0) - u * Float32(0.6)), Float32(0.0), TWO_PI, col)

    # Minimalist Diegetic Storybook HUD (matching Getting There!)
    var title = String("XI  THE THOUSAND GATES")
    if game.chapter == 0:
        title = "III  EMBERLEAF FALLS"
    elif game.chapter == 1:
        title = "IX  THE WATER CITY"
    elif game.chapter == 2:
        title = "X  STARFALL DUNES"

    # Top-left glowing lantern icon + Chapter Title + golden progress bar with paw indicator
    dl.add_glow_rect(Float32(24.0), Float32(22.0), Float32(14.0), Float32(20.0), _rgba(255, 224, 128), _rgba(255, 174, 62, 185), Float32(5.0), Float32(18.0))
    dl.add_text(font_slot, font, title, Float32(48.0), Float32(36.0), Float32(14.0), _rgba(250, 240, 218, 240), Float32(1.0), _rgba(24, 16, 22, 165))
    var bar_x = Float32(48.0)
    var bar_y = Float32(46.0)
    var bar_w = Float32(188.0)
    var prog = _clampf(game.player_x / CHAPTER_LEN, Float32(0.0), Float32(1.0))
    dl.add_rect(bar_x, bar_y, bar_w, Float32(3.5), _rgba(255, 240, 214, 72), Float32(2.0))
    dl.add_rect(bar_x, bar_y, max(Float32(6.0), bar_w * prog), Float32(3.5), _rgba(248, 198, 96, 235), Float32(2.0))
    dl.add_circle(bar_x + bar_w * prog, bar_y + Float32(1.8), Float32(4.5), _rgba(255, 226, 134, 250))

    # Top-right faint SOUND ON indicator
    var snd_lbl = String("SOUND ON") if game.sound_on else String("SOUND OFF")
    dl.add_text(font_slot, font, snd_lbl, w - Float32(102.0), Float32(34.0), Float32(11.0), _rgba(246, 236, 216, 170), Float32(1.0), _rgba(20, 14, 18, 120))

    # Whisper-soft bottom control hint
    dl.add_text(
        font_slot, font,
        "A / D  Walk    |    SPACE  Interact / Ring Bell    |    1-4  Chapters    |    M  Sound",
        Float32(24.0), h - Float32(16.0), Float32(11.0), _rgba(246, 236, 216, 155), Float32(1.0), _rgba(18, 12, 16, 135),
    )

    # Smooth chapter crossfade veil
    if game.fade_alpha > Float32(0.01):
        dl.add_rect(0.0, 0.0, w, h, _rgba(22, 16, 26, Int(game.fade_alpha * Float32(255.0))))


def _init_procedural_audio(mut audio: AudioServer, mut game: StoryGame):
    # 1. Warm pentatonic ambient pad (2 seconds loopable)
    var n_amb = 48000 * 2
    var amb = List[Int16](capacity=n_amb)
    for s in range(n_amb):
        var t = Float32(s) / Float32(48000.0)
        var w = (
            sin(TWO_PI * Float32(146.83) * t) * Float32(0.35)
            + sin(TWO_PI * Float32(220.00) * t) * Float32(0.25)
            + sin(TWO_PI * Float32(293.66) * t) * Float32(0.18)
            + sin(TWO_PI * Float32(369.99) * t) * Float32(0.12)
        )
        amb.append(Int16(Int(w * Float32(4200.0))))
    var amb_id = audio.register_pcm16_asset(amb, channels=1, sample_rate=48000)

    # 2. Resonant Kyoto Bronze Temple Bell (Bonsho) — 3 seconds warm decay!
    var n_bell = 48000 * 3
    var bell = List[Int16](capacity=n_bell)
    for s in range(n_bell):
        var t = Float32(s) / Float32(48000.0)
        var env = exp(-t * Float32(1.15)) * min(Float32(1.0), t * Float32(90.0))
        var wave = (
            sin(TWO_PI * Float32(146.83) * t) * Float32(0.50)
            + sin(TWO_PI * Float32(294.20) * t) * Float32(0.26) * exp(-t * Float32(0.6))
            + sin(TWO_PI * Float32(441.50) * t) * Float32(0.16) * exp(-t * Float32(1.2))
            + sin(TWO_PI * Float32(588.90) * t) * Float32(0.08) * exp(-t * Float32(1.9))
        )
        bell.append(Int16(Int(wave * env * Float32(18000.0))))
    game.bell_sound_id = audio.register_pcm16_asset(bell, channels=1, sample_rate=48000)

    # 3. Soft pentatonic wind-chime / joyful Shiba hop chime
    var n_chime = 48000
    var chime = List[Int16](capacity=n_chime)
    for s in range(n_chime):
        var t = Float32(s) / Float32(48000.0)
        var env = exp(-t * Float32(4.2)) * min(Float32(1.0), t * Float32(140.0))
        var wave = sin(TWO_PI * Float32(587.33) * t) * Float32(0.6) + sin(TWO_PI * Float32(880.0) * t) * Float32(0.35)
        chime.append(Int16(Int(wave * env * Float32(12000.0))))
    game.chime_sound_id = audio.register_pcm16_asset(chime, channels=1, sample_rate=48000)

    _ = audio.play_2d(amb_id, volume=Float32(0.28), looping=True, bus_id=BUS_AMBIENCE)
    game.amb_voice_started = True


def main():
    var start_chapter = 3
    var shot_path = String("")
    var args = argv()
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--chapter" and ai + 1 < len(args):
            try:
                start_chapter = Int(String(args[ai + 1]))
            except:
                start_chapter = 3
            ai += 1
        elif a == "--shot" and ai + 1 < len(args):
            shot_path = String(args[ai + 1])
            ai += 1
        ai += 1

    var is_headless = shot_path.byte_length() > 0 or env_flag("FLINT_HEADLESS")
    var cfg = AppConfig(title="The Long Way Home", headless=is_headless, fixed_hz=Float32(60.0))
    var app = EngineApp(cfg^)
    app.add_plugin(WindowPlugin(headless=is_headless, width=SCREEN_W, height=SCREEN_H, title="The Long Way Home"))
    app.add_plugin(InputPlugin(enable_evdev=not is_headless))
    app.add_plugin(AudioPlugin(headless=is_headless))
    app.startup()

    var tier = RENDER_TIER_0_HEADLESS_SIMD if is_headless else RENDER_TIER_2_AAA_VISBUFFER_RT
    var renderer = ScalableRenderer(tier, SCREEN_W, SCREEN_H)
    var font = SdfFont()
    if not font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 36, 5):
        if not font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36, 5):
            _ = font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
    var font_slot = renderer.register_ui_font(font)
    var world = RenderWorld()
    var dl = UiDrawList()
    var game = StoryGame(start_chapter)

    var aud_opt = app.resources.get_ptr_mut[AudioServer]()
    if aud_opt and not is_headless:
        _init_procedural_audio(aud_opt.value()[unsafe_offset=0], game)

    if is_headless:
        if start_chapter == 3:
            game.spawn_bell_ripples(Float32(438.0), Float32(440.0))
            game.ripples[0].age = Float32(0.55)
            game.ripples[1].age = Float32(0.28)
        dl.clear()
        _render_chapter(dl, game, font_slot, font)
        _ = renderer.render_world(world, dl, capture_readback=True, force_readback=True)
        var px = renderer.readback_pixels()
        var pdir = parent_directory(shot_path)
        if pdir.byte_length() > 0 and pdir != ".":
            _ = make_dirs(pdir)
        _ = write_png(shot_path, px, SCREEN_W, SCREEN_H)
        print("[LongWayHome] Wrote preview shot -> " + shot_path + " (prims=" + String(dl.count()) + ")")
        renderer.shutdown()
        app.shutdown()
        return

    var win_opt = app.resources.get_ptr_mut[WindowStateResource]()
    if not win_opt or not win_opt.value()[unsafe_offset=0].initialized or len(win_opt.value()[unsafe_offset=0].windows) == 0:
        print("[LongWayHome] No desktop display available; pass --shot <out.png> for headless capture.")
        renderer.shutdown()
        app.shutdown()
        return
    _ = renderer.attach_window_surface(win_opt.value()[unsafe_offset=0].windows[0])
    if renderer.uses_vulkan():
        ref gpu = renderer.gpu_slot[0]
        if gpu.vk.swapchain == 0:
            gpu.deferred_readback = True

    var out_w = SCREEN_W
    var out_h = SCREEN_H
    var frame_idx = 0
    var last_t = get_time()
    while True:
        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - last_t)))
        last_t = now
        app.step_one_frame(dt)

        var w_opt = app.resources.get_ptr_mut[WindowStateResource]()
        if not w_opt or not w_opt.value()[unsafe_offset=0].is_open():
            break
        if w_opt.value()[unsafe_offset=0].resized:
            out_w = max(64, w_opt.value()[unsafe_offset=0].width)
            out_h = max(64, w_opt.value()[unsafe_offset=0].height)
            _ = renderer.resize(out_w, out_h)

        var inp_opt = app.resources.get_ptr[InputStateResource]()
        var inp = inp_opt.value()[unsafe_offset=0].input.copy() if inp_opt else InputState()
        # Esc (256) quits
        if inp.key_pressed(256):
            break
        # Chapter hotkeys 1..4 (ASCII 49..52)
        if inp.key_pressed(49):
            game.request_chapter(0)
        elif inp.key_pressed(50):
            game.request_chapter(1)
        elif inp.key_pressed(51):
            game.request_chapter(2)
        elif inp.key_pressed(52):
            game.request_chapter(3)
        elif inp.key_pressed(77):  # M toggle sound
            game.sound_on = not game.sound_on

        var move_axis = Float32(0.0)
        if inp.key_down(65) or inp.key_down(263):  # A / Left
            move_axis -= Float32(1.0)
        if inp.key_down(68) or inp.key_down(262):  # D / Right
            move_axis += Float32(1.0)
        var interact = inp.key_pressed(32) or inp.key_pressed(69)  # Space / E

        var ao = app.resources.get_ptr_mut[AudioServer]()
        if ao:
            game.update(dt, move_axis, interact, ao.value()[unsafe_offset=0])

        dl.clear()
        _render_chapter(dl, game, font_slot, font)
        var need_rb = not renderer.surface.is_zero_copy_gpu()
        _ = renderer.render_world(world, dl, capture_readback=need_rb, force_readback=False)
        if need_rb:
            var frame_px = renderer.readback_pixels()
            ref win_res = w_opt.value()[unsafe_offset=0]
            if len(win_res.windows) > 0:
                win_res.windows[0].present(frame_px, renderer.width, renderer.height)
        if frame_idx == 0:
            print(
                "[LongWayHome] Window live ("
                + String(renderer.width) + "x" + String(renderer.height)
                + ", vulkan=" + String(renderer.uses_vulkan())
                + ", zero_copy=" + String(renderer.surface.is_zero_copy_gpu())
                + ", prims=" + String(dl.count()) + ")"
            )
        frame_idx += 1

    renderer.shutdown()
    app.shutdown()

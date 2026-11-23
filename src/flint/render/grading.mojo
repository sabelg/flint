"""Cinematic grading: a filmic tone curve, a colour grade per layer, and a
look director that blends between looks.

Everything here produces the engine's existing outputs: a 768-byte table
(one 256-entry curve per channel) and a saturation amount, which both the
CPU rasterizer and the GPU shader already apply. No shader changes.

Tone curve: Troy Sobotka's AgX sigmoid, as fitted by Benjamin Wrensch's
polynomial, applied per channel in log2 space. The engine's bytes are
already display-ready, so the curve is anchored: a -10 EV floor and 0.8
pre-exposure keep mid-grey where it was (0.46 -> 0.47) while adding a
filmic toe and a smooth shoulder. AgX's full highlight desaturation mixes
channels and needs a shader change; that is left for GPU work.

Grade: the ASC Color Decision List (slope, offset, power per channel), the
film industry's standard primary grade, which a per-channel table holds
exactly. Saturation stays the engine's separate cross-channel step.
"""
from std.math import isfinite, log2, pow
from flint.core.math import Vec3
from flint.core.springs import SpringFloat
from flint.render.lighting import LightingProfile, regional_profile

comptime AGX_MIN_EV: Float32 = -10.0
comptime AGX_MAX_EV: Float32 = 4.026069
comptime AGX_PRE_EXPOSURE: Float32 = 0.8
comptime LOOK_BLEND_SECONDS: Float32 = 1.2


def srgb_to_linear(value: Float32) -> Float32:
    if value <= 0.04045:
        return value / 12.92
    return pow((value + 0.055) / 1.055, Float32(2.4))


def agx_sigmoid(x: Float32) -> Float32:
    var x2 = x * x
    var x4 = x2 * x2
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232


def agx_raw(linear: Float32) -> Float32:
    if not isfinite(linear):
        return agx_sigmoid(Float32(0.0))
    var value = max(linear * AGX_PRE_EXPOSURE, Float32(1e-10))
    var x = (log2(value) - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV)
    return agx_sigmoid(max(Float32(0), min(Float32(1), x)))


def filmic_curve(display: Float32) -> Float32:
    """Maps a display value in [0, 1] through the anchored AgX curve,
    blended with a proportional shadow toe so weak primary channels are not
    crushed into 'Notorious Six' primary hue twists."""
    if not isfinite(display):
        return Float32(0.0)
    var d = max(Float32(0), min(Float32(1), display))
    var low = agx_raw(0)
    var high = agx_raw(1)
    var agx = max(Float32(0), min(Float32(1), (agx_raw(srgb_to_linear(d)) - low) / (high - low)))
    var w = Float32(0.62) + Float32(0.34) * d
    return max(Float32(0), min(Float32(1), agx * w + d * (Float32(1.0) - w)))


def cdl(value: Float32, slope: Float32, offset: Float32, power: Float32) -> Float32:
    if not (isfinite(value) and isfinite(slope) and isfinite(offset) and isfinite(power)):
        return Float32(0.0)
    var graded = max(Float32(0), value * slope + offset)
    return pow(graded, power)


def build_tone_table(profile: LightingProfile, mut table: List[UInt8]):
    """Exposure, then the filmic curve (blended by profile.filmic), then the
    CDL grade, per channel."""
    if len(table) != 768:
        table = List[UInt8](length=768, fill=0)
    var slope = Vec3(profile.cdl_slope.x * profile.grade_gain.x, profile.cdl_slope.y * profile.grade_gain.y, profile.cdl_slope.z * profile.grade_gain.z)
    for index in range(256):
        var exposed = min(Float32(1.6), Float32(index) / 255.0 * profile.exposure)
        # Above 1.0 the curve is extended linearly so exposure still rolls off.
        var base = filmic_curve(min(Float32(1), exposed))
        if exposed > 1:
            base = min(Float32(1), base + (exposed - 1) * 0.12)
        var shaped = exposed + (base - exposed) * profile.filmic
        if exposed > 1:
            shaped = base
        table[index] = to_byte(cdl(shaped, slope.x, profile.cdl_offset.x, profile.cdl_power.x))
        table[index + 256] = to_byte(cdl(shaped, slope.y, profile.cdl_offset.y, profile.cdl_power.y))
        table[index + 512] = to_byte(cdl(shaped, slope.z, profile.cdl_offset.z, profile.cdl_power.z))


def to_byte(value: Float32) -> UInt8:
    if not isfinite(value) or value <= Float32(0.0):
        return UInt8(0)
    if value >= Float32(1.0):
        return UInt8(255)
    return UInt8(max(0, min(255, Int(value * Float32(255.0) + Float32(0.5)))))


def mix3(a: Vec3, b: Vec3, t: Float32) -> Vec3:
    return a + (b - a) * t


def mix(a: Float32, b: Float32, t: Float32) -> Float32:
    return a + (b - a) * t


def blend_profiles(a: LightingProfile, b: LightingProfile, t: Float32) -> LightingProfile:
    var out = a
    out.key_direction = mix3(a.key_direction, b.key_direction, t).normalized()
    out.key_color = mix3(a.key_color, b.key_color, t)
    out.fill_direction = mix3(a.fill_direction, b.fill_direction, t).normalized()
    out.fill_color = mix3(a.fill_color, b.fill_color, t)
    out.hemisphere_sky = mix3(a.hemisphere_sky, b.hemisphere_sky, t)
    out.hemisphere_ground = mix3(a.hemisphere_ground, b.hemisphere_ground, t)
    out.rim_direction = mix3(a.rim_direction, b.rim_direction, t).normalized()
    out.rim_color = mix3(a.rim_color, b.rim_color, t)
    out.fog_color = mix3(a.fog_color, b.fog_color, t)
    out.fog_start = mix(a.fog_start, b.fog_start, t)
    out.fog_end = mix(a.fog_end, b.fog_end, t)
    out.height_fog = mix(a.height_fog, b.height_fog, t)
    out.exposure = mix(a.exposure, b.exposure, t)
    out.saturation = mix(a.saturation, b.saturation, t)
    out.grade_gain = mix3(a.grade_gain, b.grade_gain, t)
    out.cdl_slope = mix3(a.cdl_slope, b.cdl_slope, t)
    out.cdl_offset = mix3(a.cdl_offset, b.cdl_offset, t)
    out.cdl_power = mix3(a.cdl_power, b.cdl_power, t)
    out.filmic = mix(a.filmic, b.filmic, t)
    return out


def layer_look(layer_id: Int) -> LightingProfile:
    """The lighting and grade for each layer (see the Cinematic Lighting
    Research doc, 'Per-layer moods')."""
    if layer_id == 5:
        # Fulford trailhead: crisp high-altitude sun through golden aspens,
        # deep blue Colorado sky, cool clean shadows.
        var profile = LightingProfile()
        profile.key_direction = Vec3(-0.35, -0.85, 0.40).normalized()
        profile.key_color = Vec3(1.00, 0.93, 0.78)
        profile.fill_direction = Vec3(0.40, -0.30, -0.60).normalized()
        profile.fill_color = Vec3(0.10, 0.14, 0.22)
        profile.hemisphere_sky = Vec3(0.34, 0.44, 0.58)
        profile.hemisphere_ground = Vec3(0.20, 0.17, 0.10)
        profile.rim_direction = Vec3(0.30, -0.20, -0.93).normalized()
        profile.rim_color = Vec3(0.22, 0.18, 0.08)
        profile.fog_color = Vec3(118, 168, 222)
        profile.fog_start = 70
        profile.fog_end = 220
        profile.height_fog = 0.0
        profile.exposure = 1.10
        profile.saturation = 1.05
        profile.cdl_slope = Vec3(1.04, 1.02, 0.98)
        profile.cdl_offset = Vec3(-0.01, 0.0, 0.02)
        profile.cdl_power = Vec3(1.0, 1.0, 1.02)
        profile.filmic = 0.75
        return profile
    if layer_id == 6:
        # Glenwood: show-cave lighting. Warm red rock in the Barn, amber
        # calcite in King's Row, teal steam over the hot springs.
        var profile = LightingProfile()
        profile.key_direction = Vec3(0.30, -0.70, 0.65).normalized()
        profile.key_color = Vec3(0.86, 0.58, 0.40)
        profile.fill_direction = Vec3(-0.50, -0.20, -0.70).normalized()
        profile.fill_color = Vec3(0.10, 0.14, 0.18)
        profile.hemisphere_sky = Vec3(0.30, 0.22, 0.20)
        profile.hemisphere_ground = Vec3(0.16, 0.10, 0.08)
        profile.rim_direction = Vec3(0.20, -0.30, 0.93).normalized()
        profile.rim_color = Vec3(0.12, 0.24, 0.24)
        profile.fog_color = Vec3(34, 18, 16)
        profile.fog_start = 34
        profile.fog_end = 110
        profile.height_fog = 0.012
        profile.exposure = 1.14
        profile.saturation = 1.06
        profile.cdl_slope = Vec3(1.08, 1.0, 0.94)
        profile.cdl_offset = Vec3(0.0, -0.01, 0.01)
        profile.cdl_power = Vec3(1.0, 1.03, 1.06)
        profile.filmic = 0.8
        return profile
    if layer_id == 7:
        # Silver Queen Mine: lantern-yellow pools in near darkness.
        var profile = LightingProfile()
        profile.key_direction = Vec3(0.10, -0.80, 0.55).normalized()
        profile.key_color = Vec3(0.62, 0.50, 0.34)
        profile.fill_direction = Vec3(-0.40, -0.10, -0.80).normalized()
        profile.fill_color = Vec3(0.06, 0.07, 0.10)
        profile.hemisphere_sky = Vec3(0.18, 0.15, 0.12)
        profile.hemisphere_ground = Vec3(0.10, 0.08, 0.06)
        profile.rim_direction = Vec3(0.20, -0.30, 0.93).normalized()
        profile.rim_color = Vec3(0.14, 0.12, 0.08)
        profile.fog_color = Vec3(14, 11, 9)
        profile.fog_start = 26
        profile.fog_end = 85
        profile.height_fog = 0.010
        profile.exposure = 1.18
        profile.saturation = 1.0
        profile.cdl_slope = Vec3(1.08, 1.02, 0.92)
        profile.cdl_offset = Vec3(0.0, -0.005, -0.01)
        profile.cdl_power = Vec3(1.0, 1.02, 1.06)
        profile.filmic = 0.85
        return profile
    if layer_id == 105:
        # Inside Fulford: cold damp limestone lit by the lanterns and the ice; the
        # daylight from the culvert fades quickly to blue-black.
        var profile = LightingProfile()
        profile.key_direction = Vec3(-0.10, -0.55, 0.83).normalized()  # from the culvert
        profile.key_color = Vec3(0.42, 0.44, 0.48)
        profile.fill_direction = Vec3(0.50, -0.20, -0.70).normalized()
        profile.fill_color = Vec3(0.07, 0.11, 0.15)
        profile.hemisphere_sky = Vec3(0.17, 0.21, 0.26)
        profile.hemisphere_ground = Vec3(0.12, 0.10, 0.08)
        profile.rim_direction = Vec3(0.20, -0.30, 0.93).normalized()
        profile.rim_color = Vec3(0.10, 0.20, 0.24)
        profile.fog_color = Vec3(10, 16, 22)
        profile.fog_start = 30
        profile.fog_end = 95
        profile.height_fog = 0.010
        profile.exposure = 1.16
        profile.saturation = 0.95
        profile.cdl_slope = Vec3(1.02, 1.02, 1.06)
        profile.cdl_offset = Vec3(-0.01, 0.0, 0.025)
        profile.cdl_power = Vec3(1.04, 1.04, 1.02)
        profile.filmic = 0.85
        return profile
    if layer_id == 3:
        # Sunken Tidepool: high white-gold sun, bright cyan sky, warm sand
        # bounce, light haze over water, clean grade with teal shadows.
        var profile = LightingProfile()
        profile.key_direction = Vec3(-0.25, -0.92, -0.30).normalized()
        profile.key_color = Vec3(0.95, 0.90, 0.78)
        profile.fill_direction = Vec3(0.30, 0.55, 0.40).normalized()  # from below: sand bounce
        profile.fill_color = Vec3(0.16, 0.13, 0.08)
        profile.hemisphere_sky = Vec3(0.30, 0.42, 0.48)
        profile.hemisphere_ground = Vec3(0.18, 0.16, 0.12)
        profile.rim_direction = Vec3(0.30, -0.25, 0.92).normalized()
        profile.rim_color = Vec3(0.10, 0.24, 0.24)
        profile.fog_color = Vec3(120, 176, 190)
        profile.fog_start = 60
        profile.fog_end = 150
        profile.height_fog = 0.006
        profile.exposure = 1.12
        profile.saturation = 0.94
        profile.cdl_slope = Vec3(1.02, 1.05, 1.07)
        profile.cdl_offset = Vec3(-0.025, 0.012, 0.03)  # teal in the shadows
        profile.cdl_power = Vec3(1.0, 0.97, 0.96)
        profile.filmic = 0.7
        return profile
    if layer_id == 4:
        # Hollow Warrens: low amber sun raking across the mounds so every
        # furrow a burrower cuts throws a shadow; dusty ochre haze.
        var profile = regional_profile(0)
        profile.key_direction = Vec3(-0.70, -0.45, -0.25).normalized()
        profile.key_color = Vec3(0.98, 0.78, 0.52)
        profile.fill_color = Vec3(0.10, 0.10, 0.14)
        profile.hemisphere_sky = Vec3(0.30, 0.27, 0.24)
        profile.hemisphere_ground = Vec3(0.16, 0.11, 0.07)
        profile.rim_color = Vec3(0.20, 0.14, 0.08)
        profile.fog_color = Vec3(92, 70, 48)
        profile.fog_start = 40
        profile.fog_end = 120
        profile.height_fog = 0.012
        profile.exposure = 1.05
        profile.saturation = 0.92
        profile.cdl_slope = Vec3(1.08, 1.0, 0.88)
        profile.cdl_offset = Vec3(0.01, -0.005, -0.02)
        profile.cdl_power = Vec3(1.02, 1.05, 1.1)
        profile.filmic = 0.8
        return profile
    var profile = regional_profile(layer_id)
    if layer_id == 0:
        # Rooted Caverns: warm highlights over cool shadows; greens calmed.
        profile.cdl_slope = Vec3(1.08, 1.02, 0.92)
        profile.cdl_offset = Vec3(-0.012, 0.0, 0.025)
        profile.cdl_power = Vec3(1.06, 1.06, 1.04)
        profile.saturation = 0.94
        profile.filmic = 0.8
    elif layer_id == 1:
        # Crystal Foundry: violet shadows against orange light.
        profile.cdl_slope = Vec3(1.07, 0.96, 1.02)
        profile.cdl_offset = Vec3(0.0, -0.015, 0.04)
        profile.cdl_power = Vec3(1.05, 1.1, 1.0)
        profile.saturation = 1.06
        profile.filmic = 0.85
    else:
        # Resonant Deep: crushed blacks; only red and cyan survive.
        profile.cdl_slope = Vec3(1.08, 0.92, 1.0)
        profile.cdl_offset = Vec3(-0.025, -0.035, -0.012)
        profile.cdl_power = Vec3(1.12, 1.18, 1.08)
        profile.saturation = 0.92
        profile.filmic = 0.9
    return profile


def luminance(color: Vec3) -> Float32:
    return color.x * 0.2126 + color.y * 0.7152 + color.z * 0.0722


def drain_color(color: Vec3, amount: Float32, gain: Float32) -> Vec3:
    """A light's colour with the hum gone: most of its hue taken out, what
    is left pushed toward a cold blue-grey, a little dimmer."""
    var grey = luminance(color)
    var cold = Vec3(grey * 0.90, grey * 0.99, grey * 1.12)
    return mix3(color, cold, amount) * gain


def drain_amount(layer_id: Int) -> Float32:
    """How far each layer's look is drained. The daylight layers lose the
    most colour; the Deep is already crushed, so it loses less and is lifted."""
    if layer_id == 5 or layer_id == 3 or layer_id == 4:
        return 0.78
    if layer_id == 2:
        return 0.60
    if layer_id == 7 or layer_id == 105:
        return 0.66
    return 0.72


def drained_look(layer_id: Int) -> LightingProfile:
    """Package 3.24: the layer's look with its hum at 0, made from the
    layer's own: colder, paler, a little dimmer, the fog a blue-grey of the
    same weight (lifted where it was near black). Never dark: exposure stays
    within 6% of the layer's own, and the shadows are lifted, not crushed,
    so Maroon's back still reads. `LookDirector` blends from this to the
    layer's own look by the hum level."""
    var own = layer_look(layer_id)
    var amount = drain_amount(layer_id)
    var profile = own
    # The lantern-dark layers (the Silver Queen, inside Fulford, the Deep)
    # have no brightness to give: drained, they lose colour and warmth but
    # their light is held and the bounce lifted, so Maroon's back still
    # reads (tools/hum_shots.mojo measures it in every drained frame).
    var dark = layer_id == 7 or layer_id == 105 or layer_id == 2
    profile.key_color = drain_color(own.key_color, amount, Float32(1.0) if dark else Float32(0.90))
    profile.fill_color = drain_color(own.fill_color, amount, Float32(1.25) if dark else Float32(1.10))
    profile.hemisphere_sky = drain_color(own.hemisphere_sky, amount, Float32(1.22) if dark else Float32(0.98))
    profile.hemisphere_ground = drain_color(own.hemisphere_ground, amount, Float32(1.22) if dark else Float32(1.0))
    profile.rim_color = drain_color(own.rim_color, amount, 0.85)
    var fog = luminance(own.fog_color)
    var lifted = fog * 0.92 + 7
    profile.fog_color = mix3(own.fog_color, Vec3(lifted * 0.90, lifted * 1.0, lifted * 1.13), amount)
    profile.fog_start = own.fog_start * 0.88
    profile.fog_end = own.fog_end * 0.92
    profile.exposure = own.exposure * (Float32(1.04) if dark else Float32(0.95))
    profile.saturation = own.saturation * (1 - 0.48 * amount)
    # The grade: red pulled down, blue up, and a blue lift in the shadows.
    profile.cdl_slope = Vec3(own.cdl_slope.x * (1 - 0.09 * amount), own.cdl_slope.y * (1 - 0.01 * amount), own.cdl_slope.z * (1 + 0.07 * amount))
    profile.cdl_offset = Vec3(own.cdl_offset.x + 0.004 * amount, own.cdl_offset.y + 0.010 * amount, own.cdl_offset.z + 0.022 * amount)
    profile.cdl_power = mix3(own.cdl_power, Vec3(1.0, 1.0, 0.98), amount * 0.6)
    profile.filmic = own.filmic * (1 - 0.15 * amount)
    return profile


def hum_look(layer_id: Int, hum: Float32) -> LightingProfile:
    """The layer's look at a hum level: its own at 1 (exactly), drained at 0."""
    if hum >= 0.999:
        return layer_look(layer_id)
    return blend_profiles(drained_look(layer_id), layer_look(layer_id), max(Float32(0), hum))


struct LookDirector(Copyable, Movable):
    """Eases between layer looks and lays short colour moments on top:
    a transformation tint, the Deep's boss phases, and low-health desaturation."""
    var layer: Int
    var from_look: LightingProfile
    var to_look: LightingProfile
    var blend: Float32
    var moment: SpringFloat
    var moment_tint: Vec3
    var danger: SpringFloat
    var hum: Float32  # package 3.24: the layer's hum as shown; 1 is the layer's own look

    def __init__(out self):
        self.layer = -1
        self.from_look = LightingProfile()
        self.to_look = LightingProfile()
        self.blend = 1
        self.moment = SpringFloat(0)
        self.moment_tint = Vec3(1, 1, 1)
        self.danger = SpringFloat(0)
        self.hum = 1

    def update(
        mut self, layer: Int, dt: Float32, boss_phase: Int,
        moment_tint: Vec3, moment_strength: Float32, danger: Float32,
    ) -> LightingProfile:
        var target = hum_look(layer, self.hum)
        if layer == 2 and boss_phase > 1:
            # Each Deep Drill phase pushes the grade warmer and redder.
            var step = Float32(boss_phase - 1)
            target.cdl_slope = Vec3(target.cdl_slope.x + 0.03 * step, target.cdl_slope.y - 0.015 * step, target.cdl_slope.z - 0.01 * step)
            target.fog_color = target.fog_color + Vec3(6 * step, 0, -2 * step)
        if self.layer < 0:
            self.layer = layer
            self.from_look = target
            self.blend = 1
        elif layer != self.layer:
            # Start from wherever the current blend is, so rapid changes stay smooth.
            self.from_look = self.current()
            self.layer = layer
            self.blend = 0
        self.to_look = target
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt <= Float32(0.0)) else min(dt, Float32(10.0))
        self.blend = min(Float32(1), self.blend + safe_dt / LOOK_BLEND_SECONDS)
        if isfinite(moment_strength) and moment_strength > 0 and isfinite(moment_tint.x) and isfinite(moment_tint.y) and isfinite(moment_tint.z):
            self.moment_tint = moment_tint
        var safe_moment = moment_strength if isfinite(moment_strength) else Float32(0.0)
        var safe_danger = danger if isfinite(danger) else Float32(0.0)
        self.moment.update(safe_moment, 0.08, safe_dt)
        self.danger.update(safe_danger, 0.25, safe_dt)
        var look = self.current()
        var strength = max(Float32(0), min(Float32(1), self.moment.value))
        if strength > 0.001:
            # A moment tints the grade and lifts exposure briefly.
            look.cdl_slope = Vec3(
                look.cdl_slope.x * (1 + (self.moment_tint.x - 1) * 0.18 * strength),
                look.cdl_slope.y * (1 + (self.moment_tint.y - 1) * 0.18 * strength),
                look.cdl_slope.z * (1 + (self.moment_tint.z - 1) * 0.18 * strength),
            )
            look.exposure = look.exposure * (1 + 0.10 * strength)
        var fear = max(Float32(0), min(Float32(1), self.danger.value))
        if fear > 0.001:
            look.saturation = mix(look.saturation, 0.55, fear * 0.6)
        return look

    def current(self) -> LightingProfile:
        # Smoothstep keeps the start and end of a layer change gentle.
        var t = self.blend * self.blend * (3 - 2 * self.blend)
        return blend_profiles(self.from_look, self.to_look, t)

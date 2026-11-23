"""Data-driven cinematic lighting with fixed local-light and event budgets.

All illumination is evaluated at vertices. The rasterizer only interpolates the
resulting RGB light, fog, and emissive attributes; gameplay never reads them.
"""
from std.math import isfinite, sin, sqrt
from flint.core.math import Vec3

comptime MAX_REGISTERED_LIGHTS: Int = 256
comptime MAX_EVENT_LIGHTS: Int = 64
comptime MAX_SELECTED_LIGHTS: Int = 128


struct LightingProfile(Copyable, ImplicitlyCopyable, Movable):
    var key_direction: Vec3
    var key_color: Vec3
    var fill_direction: Vec3
    var fill_color: Vec3
    var hemisphere_sky: Vec3
    var hemisphere_ground: Vec3
    var rim_direction: Vec3
    var rim_color: Vec3
    var fog_color: Vec3
    var fog_start: Float32
    var fog_end: Float32
    var height_fog: Float32
    var exposure: Float32
    var saturation: Float32
    var grade_gain: Vec3
    var cdl_slope: Vec3    # ASC CDL primary grade, per channel
    var cdl_offset: Vec3
    var cdl_power: Vec3
    var filmic: Float32    # 0 = the old curve's linear mid-tones, 1 = full AgX shape

    def __init__(out self):
        self.key_direction = Vec3(-0.42, -0.82, -0.38).normalized()
        self.key_color = Vec3(0.72, 0.72, 0.72)
        self.fill_direction = Vec3(0.5, -0.2, 0.7).normalized()
        self.fill_color = Vec3(0.12, 0.12, 0.12)
        self.hemisphere_sky = Vec3(0.30, 0.30, 0.30)
        self.hemisphere_ground = Vec3(0.16, 0.16, 0.16)
        self.rim_direction = Vec3(0.2, -0.5, 0.8).normalized()
        self.rim_color = Vec3(0.12, 0.12, 0.12)
        self.fog_color = Vec3(14, 27, 36)
        self.fog_start = 45
        self.fog_end = 125
        self.height_fog = 0
        self.exposure = 1.0
        self.saturation = 1.0
        self.grade_gain = Vec3(1, 1, 1)
        self.cdl_slope = Vec3(1, 1, 1)
        self.cdl_offset = Vec3(0, 0, 0)
        self.cdl_power = Vec3(1, 1, 1)
        self.filmic = 0

    @staticmethod
    def _clamp_u8(v: Float32) -> UInt8:
        if not isfinite(v):
            return UInt8(0)
        return UInt8(Int(max(Float32(0.0), min(Float32(255.0), v))))

    def fog_rgb(self) -> Tuple[UInt8, UInt8, UInt8]:
        return (
            Self._clamp_u8(self.fog_color.x),
            Self._clamp_u8(self.fog_color.y),
            Self._clamp_u8(self.fog_color.z),
        )


def regional_profile(region: Int) -> LightingProfile:
    var profile = LightingProfile()
    if region == 0:
        # Warm canopy openings, cool readable fill, and turquoise resonance rims.
        profile.key_direction = Vec3(-0.48, -0.80, -0.35).normalized()
        profile.key_color = Vec3(0.82, 0.75, 0.61)
        profile.fill_direction = Vec3(0.60, -0.18, 0.72).normalized()
        profile.fill_color = Vec3(0.09, 0.16, 0.20)
        profile.hemisphere_sky = Vec3(0.27, 0.34, 0.37)
        profile.hemisphere_ground = Vec3(0.14, 0.12, 0.10)
        profile.rim_direction = Vec3(0.35, -0.35, 0.86).normalized()
        profile.rim_color = Vec3(0.08, 0.20, 0.18)
        profile.fog_color = Vec3(17, 31, 39)
        profile.fog_start = 48
        profile.fog_end = 132
        profile.height_fog = 0.010
        profile.exposure = 1.08
        profile.saturation = 1.045
        profile.grade_gain = Vec3(1.025, 1.015, 0.99)
    elif region == 1:
        # Amber foundry machinery against violet-crystal shadow separation.
        profile.key_direction = Vec3(0.40, -0.72, -0.56).normalized()
        profile.key_color = Vec3(0.84, 0.61, 0.39)
        profile.fill_direction = Vec3(-0.65, -0.12, 0.55).normalized()
        profile.fill_color = Vec3(0.16, 0.13, 0.27)
        profile.hemisphere_sky = Vec3(0.30, 0.27, 0.40)
        profile.hemisphere_ground = Vec3(0.18, 0.11, 0.10)
        profile.rim_direction = Vec3(-0.30, -0.30, 0.90).normalized()
        profile.rim_color = Vec3(0.20, 0.12, 0.28)
        profile.fog_color = Vec3(25, 20, 38)
        profile.fog_start = 42
        profile.fog_end = 116
        profile.height_fog = 0.014
        profile.exposure = 1.06
        profile.saturation = 1.055
        profile.grade_gain = Vec3(1.035, 0.99, 1.025)
    else:
        # Rose/violet depth with a warm mineral key and cyan support signals.
        profile.key_direction = Vec3(-0.30, -0.66, -0.69).normalized()
        profile.key_color = Vec3(0.76, 0.53, 0.39)
        profile.fill_direction = Vec3(0.62, -0.05, 0.58).normalized()
        profile.fill_color = Vec3(0.16, 0.10, 0.24)
        profile.hemisphere_sky = Vec3(0.29, 0.22, 0.37)
        profile.hemisphere_ground = Vec3(0.13, 0.09, 0.16)
        profile.rim_direction = Vec3(0.12, -0.24, 0.96).normalized()
        profile.rim_color = Vec3(0.09, 0.23, 0.25)
        profile.fog_color = Vec3(27, 13, 31)
        profile.fog_start = 38
        profile.fog_end = 108
        profile.height_fog = 0.018
        profile.exposure = 1.10
        profile.saturation = 1.04
        profile.grade_gain = Vec3(1.03, 0.975, 1.025)
    return profile


struct LocalLight(Copyable, ImplicitlyCopyable, Movable):
    var position: Vec3
    var seg_delta: Vec3  # zero = point light; non-zero = tube/capsule from position to position + seg_delta
    var color: Vec3
    var radius: Float32
    var intensity: Float32
    var priority: Int
    var stable_id: Int
    var active: Bool

    def __init__(out self):
        self.position = Vec3()
        self.seg_delta = Vec3()
        self.color = Vec3(1, 1, 1)
        self.radius = 1
        self.intensity = 0
        self.priority = 0
        self.stable_id = 0
        self.active = False


struct EventLight(Copyable, ImplicitlyCopyable, Movable):
    var light: LocalLight
    var remaining: Float32
    var duration: Float32

    def __init__(out self):
        self.light = LocalLight()
        self.remaining = 0
        self.duration = 0


struct EventLightPool(Copyable, Movable):
    var events: List[EventLight]
    var next_id: Int
    var emitted: Int
    var recycled: Int

    def __init__(out self):
        self.events = List[EventLight](length=MAX_EVENT_LIGHTS, fill=EventLight())
        self.next_id = 100000
        self.emitted = 0
        self.recycled = 0

    def renew(
        mut self,
        position: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        duration: Float32,
        stable_id: Int,
        priority: Int,
    ):
        """Updates and keeps alive an existing event light matching `stable_id`
        in-place (under its own `priority`) instead of allocating a new pool slot
        every frame."""
        for index in range(len(self.events)):
            if self.events[index].remaining > 0 and self.events[index].light.stable_id == stable_id:
                self.events[index].light.position = position
                self.events[index].light.color = color
                self.events[index].light.radius = max(Float32(0.1), radius)
                self.events[index].light.intensity = max(Float32(0), intensity)
                self.events[index].light.priority = priority
                self.events[index].remaining = max(Float32(0.001), duration)
                self.events[index].duration = self.events[index].remaining
                return
        self.emit(position, color, radius, intensity, duration, stable_id)
        for index in range(len(self.events)):
            if self.events[index].remaining > 0 and self.events[index].light.stable_id == stable_id:
                self.events[index].light.priority = priority

    def emit(mut self, position: Vec3, color: Vec3, radius: Float32, intensity: Float32, duration: Float32, stable_id: Int = -1):
        var slot = -1
        var oldest = 0
        var oldest_remaining = Float32(1000000)
        for index in range(len(self.events)):
            if self.events[index].remaining <= 0:
                slot = index
                break
            if self.events[index].remaining < oldest_remaining:
                oldest_remaining = self.events[index].remaining
                oldest = index
        if slot < 0:
            slot = oldest
            self.recycled += 1
        var event = EventLight()
        event.light.active = True
        event.light.position = position
        event.light.seg_delta = Vec3()
        event.light.color = color
        event.light.radius = max(Float32(0.1), radius)
        event.light.intensity = max(Float32(0), intensity)
        event.light.priority = 100
        event.light.stable_id = self.next_id if stable_id < 0 else stable_id
        event.remaining = max(Float32(0.001), duration)
        event.duration = event.remaining
        self.events[slot] = event
        self.next_id += 1
        self.emitted += 1

    def emit_tube(
        mut self,
        start_pos: Vec3,
        end_pos: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        duration: Float32,
        stable_id: Int = -1,
    ):
        var slot = -1
        var oldest = 0
        var oldest_remaining = Float32(1000000)
        for index in range(len(self.events)):
            if self.events[index].remaining <= 0:
                slot = index
                break
            if self.events[index].remaining < oldest_remaining:
                oldest_remaining = self.events[index].remaining
                oldest = index
        if slot < 0:
            slot = oldest
            self.recycled += 1
        var event = EventLight()
        event.light.active = True
        event.light.position = start_pos
        event.light.seg_delta = end_pos - start_pos
        event.light.color = color
        event.light.radius = max(Float32(0.1), radius)
        event.light.intensity = max(Float32(0), intensity)
        event.light.priority = 100
        event.light.stable_id = self.next_id if stable_id < 0 else stable_id
        event.remaining = max(Float32(0.001), duration)
        event.duration = event.remaining
        self.events[slot] = event
        self.next_id += 1
        self.emitted += 1

    def update(mut self, dt: Float32):
        for index in range(len(self.events)):
            if self.events[index].remaining > 0:
                self.events[index].remaining = max(Float32(0), self.events[index].remaining - max(Float32(0), dt))
                self.events[index].light.active = self.events[index].remaining > 0


struct LightingEnvironment(Copyable, Movable):
    var profile: LightingProfile
    var registered: List[LocalLight]
    var selected: List[LocalLight]
    var registered_count: Int
    var selected_count: Int
    var quality: Int
    var reduced_effects: Bool
    var camera: Vec3
    var events: EventLightPool
    var rejected_lights: Int
    var vertex_samples: Int
    var per_pixel: Bool
    var water_level: Float32  # -1000 = no water
    var time: Float32

    def __init__(out self):
        self.profile = LightingProfile()
        self.registered = List[LocalLight](length=MAX_REGISTERED_LIGHTS, fill=LocalLight())
        self.selected = List[LocalLight](length=MAX_SELECTED_LIGHTS, fill=LocalLight())
        self.registered_count = 0
        self.selected_count = 0
        self.quality = 1
        self.reduced_effects = False
        self.camera = Vec3()
        self.events = EventLightPool()
        self.rejected_lights = 0
        self.vertex_samples = 0
        self.per_pixel = False
        self.water_level = -1000
        self.time = 0

    def begin_frame(mut self, region: Int, camera: Vec3, quality: Int, reduced_effects: Bool):
        self.begin_frame_with(regional_profile(region), camera, quality, reduced_effects)

    def begin_frame_with(mut self, profile: LightingProfile, camera: Vec3, quality: Int, reduced_effects: Bool):
        self.profile = profile
        self.camera = camera
        self.quality = max(0, min(2, quality))
        self.reduced_effects = reduced_effects
        self.registered_count = 0
        self.selected_count = 0
        self.rejected_lights = 0
        self.vertex_samples = 0

    def light_budget(self) -> Int:
        if self.per_pixel:
            if self.quality <= 0:
                return 24
            if self.quality >= 2:
                return MAX_SELECTED_LIGHTS
            return 64
        if self.quality <= 0:
            return 3
        if self.quality >= 2:
            return 12
        return 8

    def register(mut self, position: Vec3, color: Vec3, radius: Float32, intensity: Float32, priority: Int, stable_id: Int):
        self.register_tube(position, position, color, radius, intensity, priority, stable_id)

    def register_tube(
        mut self,
        start_pos: Vec3,
        end_pos: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        priority: Int,
        stable_id: Int,
    ):
        var registration_limit = len(self.registered) if priority >= 100 else len(self.registered) - MAX_EVENT_LIGHTS
        if self.registered_count >= registration_limit or radius <= 0 or intensity <= 0:
            self.rejected_lights += 1
            return
        var light = LocalLight()
        light.active = True
        light.position = start_pos
        light.seg_delta = end_pos - start_pos
        light.color = color
        light.radius = radius
        light.intensity = intensity * (Float32(0.55) if self.reduced_effects and priority >= 50 and priority < 100 else Float32(1.0))
        light.priority = priority
        light.stable_id = stable_id
        self.registered[self.registered_count] = light
        self.registered_count += 1

    def finish_frame(mut self):
        # Active event lights join the same bounded candidate set as world lights.
        for index in range(len(self.events.events)):
            var event = self.events.events[index]
            if event.remaining > 0 and event.duration > 0:
                var light = event.light
                var life = event.remaining / event.duration
                light.intensity *= min(Float32(1), life * 3.0) * (Float32(0.45) if self.reduced_effects else Float32(1.0))
                self.register_tube(
                    light.position,
                    light.position + light.seg_delta,
                    light.color,
                    light.radius,
                    light.intensity,
                    light.priority,
                    light.stable_id,
                )
        var budget = self.light_budget()
        self.selected_count = 0
        var picked = List[Bool](length=self.registered_count, fill=False)
        # Selection sort over a fixed candidate array. Priority, then relative
        # camera influence, then stable numeric ID defines every tie.
        for _ in range(min(budget, self.registered_count)):
            var best = -1
            for candidate in range(self.registered_count):
                if picked[candidate]:
                    continue
                var cand_id = self.registered[candidate].stable_id
                if cand_id != 0:
                    var already_selected = False
                    for chosen in range(self.selected_count):
                        if self.selected[chosen].stable_id == cand_id:
                            already_selected = True
                            break
                    if already_selected:
                        picked[candidate] = True
                        continue
                if best < 0 or self.better(candidate, best):
                    best = candidate
            if best >= 0:
                picked[best] = True
                self.selected[self.selected_count] = self.registered[best]
                self.selected_count += 1
        self.rejected_lights += max(0, self.registered_count - self.selected_count)

    def better(self, candidate: Int, current: Int) -> Bool:
        var a = self.registered[candidate]
        var b = self.registered[current]
        if a.priority != b.priority:
            return a.priority > b.priority
        var a_mid = a.position + a.seg_delta * 0.5
        var b_mid = b.position + b.seg_delta * 0.5
        var a_influence = a.intensity * (a.radius + a.seg_delta.length() * 0.5) / max(Float32(0.25), a_mid.distance_to(self.camera))
        var b_influence = b.intensity * (b.radius + b.seg_delta.length() * 0.5) / max(Float32(0.25), b_mid.distance_to(self.camera))
        if abs(a_influence - b_influence) > 0.00001:
            return a_influence > b_influence
        return a.stable_id < b.stable_id

    def emit_event(mut self, position: Vec3, color: Vec3, radius: Float32, intensity: Float32, duration: Float32, stable_id: Int = -1):
        self.events.emit(position, color, radius, intensity, duration, stable_id)

    def emit_tube_event(
        mut self,
        start_pos: Vec3,
        end_pos: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        duration: Float32,
        stable_id: Int = -1,
    ):
        self.events.emit_tube(start_pos, end_pos, color, radius, intensity, duration, stable_id)

    def renew_event(
        mut self,
        position: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        duration: Float32,
        stable_id: Int,
        priority: Int,
    ):
        self.events.renew(position, color, radius, intensity, duration, stable_id, priority)

    def update_events(mut self, dt: Float32):
        self.events.update(dt)

    def sample(self, position: Vec3, normal: Vec3, ao: Float32 = 1.0) -> Vec3:
        """Read-only, so vertex threads can share it. Callers count vertex_samples."""
        var n = normal.normalized()
        var occlusion = max(Float32(0), min(Float32(1), ao))
        var ao_strength: Float32 = 0.38 if self.quality == 0 else (0.68 if self.quality == 1 else Float32(0.80))
        var ambient_ao = 1.0 - (1.0 - occlusion) * ao_strength
        var direct_ao = 0.78 + ambient_ao * 0.22
        var hemisphere = max(Float32(0), min(Float32(1), n.y * 0.5 + 0.5))
        var result = (self.profile.hemisphere_ground * (1.0 - hemisphere) + self.profile.hemisphere_sky * hemisphere) * ambient_ao
        var key = max(Float32(0), -dot_light(n, self.profile.key_direction)) * direct_ao
        var fill = max(Float32(0), -dot_light(n, self.profile.fill_direction))
        if self.per_pixel:
            # The shader adds the key and local lights, shadowed, for each pixel.
            key = 0
        result = result + self.profile.key_color * key + self.profile.fill_color * fill
        var view = (self.camera - position).normalized()
        var edge = 1.0 - abs(dot_light(n, view))
        var rim = edge * edge * (0.30 + max(Float32(0), -dot_light(n, self.profile.rim_direction)) * 0.70)
        result = result + self.profile.rim_color * rim * direct_ao
        for index in range(0 if self.per_pixel else self.selected_count):
            var light = self.selected[index]
            var offset = light.position - position
            var distance_squared = offset.length_squared()
            var radius_squared = light.radius * light.radius
            if distance_squared >= radius_squared:
                continue
            var attenuation = 1.0 - distance_squared / radius_squared
            attenuation *= attenuation
            var direction = offset.normalized()
            var lambert = 0.24 + max(Float32(0), dot_light(n, direction)) * 0.76
            result = result + light.color * (attenuation * lambert * light.intensity * ambient_ao)
        if position.y < self.water_level and n.y > 0.3:
            # Caustics: sunlight focused by the moving surface, brightest in
            # the shallows and on surfaces facing up (GPU Gems ch. 2 idea,
            # a procedural pattern instead of refracting real waves).
            var depth = self.water_level - position.y
            var strength = caustic_pattern(position.x, position.z, self.time) * n.y * 0.55 / (1.0 + depth * 0.35)
            result = result + self.profile.key_color * strength
        return Vec3(min(Float32(3), result.x), min(Float32(3), result.y), min(Float32(3), result.z))

    def ambient_occlusion(self, ao: Float32) -> Float32:
        var occlusion = max(Float32(0), min(Float32(1), ao))
        var ao_strength: Float32 = 0.38 if self.quality == 0 else (0.68 if self.quality == 1 else Float32(0.80))
        return 1.0 - (1.0 - occlusion) * ao_strength

    def fog_amount(self, position: Vec3, view_depth: Float32) -> Float32:
        var distance = max(Float32(0), (view_depth - self.profile.fog_start) / max(Float32(1), self.profile.fog_end - self.profile.fog_start))
        var low_height = max(Float32(0), 12.0 - position.y) * self.profile.height_fog
        return min(Float32(0.88), distance * 0.82 + low_height)


def dot_light(a: Vec3, b: Vec3) -> Float32:
    return a.x * b.x + a.y * b.y + a.z * b.z


def caustic_pattern(x: Float32, z: Float32, time: Float32) -> Float32:
    """Thin bright lines that drift and re-form, in [0, 1]."""
    var waves = (
        sin(x * 1.7 + time * 1.1) + sin(z * 1.3 - time * 0.9)
        + sin((x + z) * 0.8 + time * 1.4) + sin((x - z) * 1.1 - time * 0.7)
    )
    var line = max(Float32(0), 1.0 - abs(waves) * 0.5)
    var squared = line * line
    return squared * squared

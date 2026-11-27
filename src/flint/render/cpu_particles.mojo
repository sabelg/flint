"""Tier 0 software preview of authored generic world-space particle bursts."""
from std.math import isfinite, sqrt
from flint.core.math import Vec3
from flint.core.rng import DetRng
from flint.render.aaa_gpu_particles import AaaParticleEmitter, PARTICLE_GENERIC, PARTICLE_FLAG_ADDITIVE, PARTICLE_FLAG_LIT
from flint.render.simd_canvas import Camera3D, SimdCanvas
from flint.render.ui_cpu_raster import _blend_pixel, _pack_argb_f32, _unpack_argb_f32


@fieldwise_init
struct _Particle(Copyable, Movable):
    var position: Vec3
    var velocity: Vec3
    var age: Float32
    var lifetime: Float32
    var effect: AaaParticleEmitter

    def draw(self, mut canvas: SimdCanvas, camera: Camera3D, light: Float32):
        var projection = camera.project(self.position)
        if not projection[3]:
            return
        var t = self.age / self.lifetime
        ref e = self.effect
        var r = max(Float32(0.5), (e.size + (e.end_size - e.size)*t) * camera.focal_px * projection[2])
        var x = projection[0]
        var y = projection[1]
        if not isfinite(x) or not isfinite(y) or not isfinite(r) or x+r < 0 or y+r < 0 or x-r >= Float32(canvas.width) or y-r >= Float32(canvas.height):
            return
        var color = (e.color + (e.end_color - e.color)*t) * (light if e.flags & PARTICLE_FLAG_LIT else Float32(1))
        color = Vec3(sqrt(max(Float32(0), color.x)), sqrt(max(Float32(0), color.y)), sqrt(max(Float32(0), color.z)))
        var opacity = e.alpha + (e.end_alpha - e.alpha)*t
        for py in range(Int(max(Float32(0), y-r)), Int(min(Float32(canvas.height), y+r+1))):
            for px in range(Int(max(Float32(0), x-r)), Int(min(Float32(canvas.width), x+r+1))):
                var idx = py*canvas.width + px
                var dx = (Float32(px)+0.5-x)/r
                var dy = (Float32(py)+0.5-y)/r
                var alpha = (1-dx*dx-dy*dy)*opacity
                if alpha <= 0 or projection[2] < canvas.depth[idx]:
                    continue
                if e.flags & PARTICLE_FLAG_ADDITIVE:
                    var dst = _unpack_argb_f32(canvas.pixels[idx])
                    canvas.pixels[idx] = _pack_argb_f32(dst[0]+color.x*alpha, dst[1]+color.y*alpha, dst[2]+color.z*alpha)
                else:
                    _blend_pixel(canvas.pixels, idx, color.x, color.y, color.z, alpha)


struct CpuParticles(Movable):
    var particles: List[_Particle]
    var random: DetRng

    def __init__(out self):
        self.particles = List[_Particle]()
        self.random = DetRng()

    def _offset(mut self, radius: Float32) -> Vec3:
        var direction = Vec3(self.random.range_f32(-1, 1), self.random.range_f32(-1, 1), self.random.range_f32(-1, 1))
        return direction.normalized() * self.random.range_f32(0, radius)

    def render(mut self, mut canvas: SimdCanvas, camera: Camera3D, emitters: List[AaaParticleEmitter], dt: Float32, light: Float32):
        for e in emitters:
            if e.archetype != PARTICLE_GENERIC:
                continue
            # Keep software previews bounded independently of the GPU simulation budget.
            for _ in range(min(e.spawn_count, 8192-len(self.particles))):
                var pos = e.origin + self._offset(e.spawn_radius)
                var vel = e.velocity + self._offset(e.velocity_spread)
                var life = e.lifetime * self.random.range_f32(0.8, 1.2)
                self.particles.append(_Particle(pos, vel, 0, life, e.copy()))
        var step = max(Float32(0), min(Float32(0.1), dt))
        var alive = 0
        for i in range(len(self.particles)):
            ref p = self.particles[i]
            p.age += step
            if p.age >= p.lifetime:
                continue
            p.velocity.y -= 9.81*p.effect.gravity_scale*step
            p.velocity = p.velocity * max(Float32(0), 1-p.effect.drag*step)
            p.position = p.position + p.velocity*step
            if p.position.y < p.effect.bounce_ground_y:
                p.position.y = p.effect.bounce_ground_y
                p.velocity.y = abs(p.velocity.y)*0.3
            p.draw(canvas, camera, light)
            if i != alive:
                self.particles[alive] = p.copy()
            alive += 1
        while len(self.particles) > alive:
            _ = self.particles.pop()

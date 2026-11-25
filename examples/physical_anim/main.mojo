"""Physics-driven animation blending demo (`examples/physical_anim`).

A biped walks; it takes a punch to the hand, a shove to the chest, then a knockdown, gets up
and walks on. Writes one PNG per frame (30 fps): the animation as a grey ghost, the drawn
(blended) pose in colour, and a bar on top for the controller state (green idle, amber
blending, red fallen).

usage: physical_anim OUT_DIR [SECONDS]
"""
from std.sys import argv
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.core.detmath import det_sin, det_cos
from flint.core.ik import angle_axis
from flint.assets.model import NativeModel, NativeNode, NativeAnimation, NativeChannel
from flint.assets.image import write_png
from flint.anim.pose import Pose, SkeletonInstance
from flint.anim.procedural import Placement
from flint.anim.physical_anim import PhysicalAnimator, PA_IDLE, PA_BLEND, PA_FALLEN
from flint.physics.world import PhysicsWorld

comptime W: Int = 640
comptime H: Int = 360
comptime DT: Float32 = 1.0 / 60.0
comptime SPEED: Float32 = 1.1
comptime PX: Float32 = 150.0  # pixels per metre


def biped() -> NativeModel:
    var m = NativeModel()
    var names = [
        String("Pelvis"), String("Chest"), String("Head"),
        String("UpperArm.R"), String("Hand.R"),
        String("Thigh.L"), String("Shin.L"), String("Foot.L"),
        String("Thigh.R"), String("Shin.R"), String("Foot.R"),
    ]
    var parents = [-1, 0, 1, 1, 3, 0, 5, 6, 0, 8, 9]
    var offsets = [
        Vec3(0.0, 0.92, 0.0), Vec3(0.0, 0.42, 0.0), Vec3(0.0, 0.32, 0.0),
        Vec3(0.26, 0.24, 0.0), Vec3(0.0, -0.48, 0.0),
        Vec3(-0.14, 0.0, 0.0), Vec3(0.0, -0.44, 0.0), Vec3(0.0, -0.40, 0.0),
        Vec3(0.14, 0.0, 0.0), Vec3(0.0, -0.44, 0.0), Vec3(0.0, -0.40, 0.0),
    ]
    for i in range(len(names)):
        var nd = NativeNode()
        nd.name = names[i]
        nd.translation = offsets[i]
        nd.base_translation = offsets[i]
        nd.rotation = Quat()
        nd.base_rotation = Quat()
        nd.scale = Vec3(1, 1, 1)
        nd.base_scale = Vec3(1, 1, 1)
        m.nodes.append(nd^)
        var p = parents[i]
        if p >= 0:
            m.nodes[p].children.append(i)
        else:
            m.roots.append(i)
    # stand-in get-up clips: the pelvis rises back to standing height
    for cname in [String("GetupSupine"), String("GetupProne")]:
        var anim = NativeAnimation()
        anim.name = cname
        anim.duration = 0.8
        var ch = NativeChannel()
        ch.node = 0
        ch.path = 0
        ch.times = [Float32(0.0), Float32(0.8)]
        ch.values = [Float32(0.0), Float32(0.92), Float32(0.0), Float32(0.0), Float32(0.92), Float32(0.0)]
        anim.channels.append(ch^)
        m.animations.append(anim^)
    return m^


def walk(model: NativeModel, t: Float32) -> Pose:
    var p = Pose()
    p.rest(model)
    var ph = det_sin(Float32(6.2831853) * t)
    var x = Vec3(1, 0, 0)
    p.r[5] = angle_axis(Float32(0.45) * ph, x)
    p.r[8] = angle_axis(Float32(-0.45) * ph, x)
    p.r[6] = angle_axis(Float32(-0.25) * (ph * ph), x)
    p.r[9] = angle_axis(Float32(-0.25) * (ph * ph), x)
    p.r[3] = angle_axis(Float32(0.35) * ph, x)
    p.r[1] = angle_axis(Float32(0.06) * ph, Vec3(0, 1, 0))
    return p^


struct Canvas(Movable):
    var px: List[UInt32]
    var cam_z: Float32

    def __init__(out self):
        self.px = List[UInt32](length=W * H, fill=UInt32(0xFF1B1F27))
        self.cam_z = 0

    def clear(mut self):
        for i in range(W * H):
            var y = i // W
            self.px[i] = UInt32(0xFF1B1F27) if y < H - 60 else UInt32(0xFF2B3240)

    def project(self, p: Vec3) -> Tuple[Float32, Float32]:
        # camera looks across the walk, yawed 35 degrees so sideways motion shows
        var c = Float32(0.819)
        var s = Float32(0.574)
        var sx = Float32(W) * 0.5 + ((p.z - self.cam_z) * c - p.x * s) * PX
        var sy = Float32(H - 60) - (p.y + (p.z - self.cam_z) * s * 0.15 + p.x * c * 0.15) * PX
        return (sx, sy)

    def disc(mut self, cx: Float32, cy: Float32, r: Float32, col: UInt32):
        var ir = Int(r) + 1
        for dy in range(-ir, ir + 1):
            for dx in range(-ir, ir + 1):
                if Float32(dx * dx + dy * dy) <= r * r:
                    var x = Int(cx) + dx
                    var y = Int(cy) + dy
                    if x >= 0 and x < W and y >= 0 and y < H:
                        self.px[y * W + x] = col

    def line(mut self, a: Vec3, b: Vec3, r: Float32, col: UInt32):
        var pa = self.project(a)
        var pb = self.project(b)
        var dx = pb[0] - pa[0]
        var dy = pb[1] - pa[1]
        var n = Int(max(abs(dx), abs(dy))) + 1
        for k in range(n + 1):
            var t = Float32(k) / Float32(n)
            self.disc(pa[0] + dx * t, pa[1] + dy * t, r, col)

    def figure(mut self, skel: SkeletonInstance, model: NativeModel, place: Placement, r: Float32, col: UInt32, head: UInt32):
        for i in range(len(model.nodes)):
            for c in model.nodes[i].children:
                self.line(place.to_world(skel.global_position(i)), place.to_world(skel.global_position(c)), r, col)
        var hp = self.project(place.to_world(skel.global_position(2)))
        self.disc(hp[0], hp[1] - 10, 12, head)

    def bar(mut self, col: UInt32):
        for y in range(8):
            for x in range(W):
                self.px[y * W + x] = col


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: physical_anim OUT_DIR [SECONDS]")
        return
    var out = String(args[1])
    var seconds = Float32(9.0)
    if len(args) > 2:
        seconds = Float32(atof(String(args[2])))
    var model = biped()
    var world = PhysicsWorld()
    _ = world.add_box(Vec3(0.0, -0.5, 0.0), Vec3(200.0, 0.5, 200.0), mass=0.0)
    world.rebuild_broadphase()

    var pa = PhysicalAnimator()
    pa.rsm.supine_getup_clip = 0
    pa.rsm.prone_getup_clip = 1
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 0, -1, 15.0, 0.14, 0.30)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 1, 0, 12.0, 0.15, 0.32)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 3, 1, 3.5, 0.06, 0.48)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 5, 0, 7.0, 0.08, 0.44)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 6, 3, 4.0, 0.05, 0.40)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 8, 0, 7.0, 0.08, 0.44)
    _ = pa.rsm.ragdoll.add_limb_from_model(model, 9, 5, 4.0, 0.05, 0.40)

    var skel = SkeletonInstance(len(model.nodes))
    skel.reset_to_rest(model)
    var ghost = SkeletonInstance(len(model.nodes))
    ghost.reset_to_rest(model)
    var place = Placement()
    var cv = Canvas()
    var frames = Int(seconds / DT)
    var shot = 0
    var flash = 0
    var flash_at = Vec3()
    var anim_t: Float32 = 0
    for f in range(frames):
        var t = Float32(f) * DT
        if pa.state != PA_FALLEN:
            anim_t += DT
            place.position.z += SPEED * DT
        var anim = walk(model, anim_t)
        if f == 48:
            flash_at = place.to_world(skel.global_position(4))
            pa.hit(4, Vec3(-24.0, 0.0, 0.0), flash_at)
            flash = 8
        if f == 150:
            flash_at = place.to_world(skel.global_position(1))
            pa.hit(1, Vec3(0.0, 0.0, -75.0), flash_at)
            flash = 8
        if f == 252:
            flash_at = place.to_world(skel.global_position(1))
            pa.hit(1, Vec3(0.0, 10.0, -230.0), flash_at)
            flash = 8
        pa.update(world, skel, place, model, anim, DT)
        ghost.apply_pose(anim, model)
        if f % 2 == 0:
            cv.cam_z = cv.cam_z + (place.position.z - cv.cam_z) * 0.15
            cv.clear()
            cv.figure(ghost, model, place, 2, UInt32(0xFF59606E), UInt32(0xFF59606E))
            cv.figure(skel, model, place, 4, UInt32(0xFF4FC3F7), UInt32(0xFFFFD54F))
            if flash > 0:
                var fp = cv.project(flash_at)
                cv.disc(fp[0], fp[1], Float32(6 + 3 * flash), UInt32(0xFFFF5252))
            cv.bar(UInt32(0xFF66BB6A) if pa.state == PA_IDLE else (UInt32(0xFFFFB300) if pa.state == PA_BLEND else UInt32(0xFFE53935)))
            var name = String(shot)
            while name.byte_length() < 4:
                name = "0" + name
            _ = write_png(out + "/f" + name + ".png", cv.px, W, H, False)
            shot += 1
        if flash > 0 and f % 2 == 0:
            flash -= 1
        if f % 30 == 0:
            print("t", t, "state", pa.state, "pelvis err", pa.last_pelvis_error)
    print("frames", shot, "falls", pa.falls, "hits", pa.hits, "recoveries", pa.rsm.recovery_count)

"""Regression scenes for full-shape translational CCD and convex terrain contact."""
from std.collections import List
from std.math import abs, sqrt
from flint.core.ik import angle_axis
from flint.core.detmath import det_atan
from std.testing import assert_true
from flint.core.math import Quat, Vec3
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.narrowphase import ConvexHull
from flint.physics.shapes_v5 import CompoundChild, CompoundShape, HeightfieldCollider
from flint.physics.world import PhysicsWorld


def check_post(height: Float32, rotated: Bool, should_hit: Bool) raises:
    var world = PhysicsWorld(gravity=0.0)
    var post = world.add_box(Vec3(2, height, 0), Vec3(0.05, 0.1, 0.15), mass=0.0)
    var extent = Vec3(0.1, 1.5, 0.15)
    var orientation = Quat()
    if rotated:
        extent = Vec3(0.1, 0.15, 1.5)
        orientation = Quat(0.70710678, 0, 0, 0.70710678)
    var box = world.add_box(Vec3(), extent, mass=1, rot=orientation)
    world.bodies[box].enable_ccd = True
    world.set_velocity(box, Vec3(180, 0, 0))
    var cast = world._cast_ccd_body(box, post, Vec3(3, 0, 0), 1)
    assert_true(cast.hit == should_hit)
    if cast.hit:
        assert_true(cast.body_slot == post)
        assert_true(abs(cast.fraction * 3 - Float32(1.85)) < Float32(0.002))
        assert_true(cast.normal.x < Float32(-0.99))
    world.step(Float32(1.0 / 60.0))
    if should_hit:
        assert_true(world.bodies[box].pos.x < 2)
        assert_true(world.bodies[box].linear_vel.x < 0)
    else:
        assert_true(world.bodies[box].pos.x > 2.9)


def check_capsule_and_compound_movers() raises:
    for compound in [False, True]:
        for offset in [Float32(0.0), Float32(1.2)]:
            var world = PhysicsWorld(gravity=0)
            _ = world.add_sphere(Vec3(2, offset, 0), 0.1, mass=0)
            var mover: Int
            if compound:
                var shape = CompoundShape()
                shape.add_child(CompoundChild.sphere(Vec3(0, 1.2, 0), 0.1))
                shape.add_child(CompoundChild.sphere(Vec3(0, -1.2, 0), 0.1))
                mover = world.add_compound(Vec3(), shape, mass=1)
            else:
                mover = world.add_capsule(Vec3(), 0.1, 1.2, mass=1)
            world.bodies[mover].enable_ccd = True
            world.set_velocity(mover, Vec3(180, 0, 0))
            world.step(Float32(1.0 / 60.0))
            if compound and offset == 0:
                assert_true(world.bodies[mover].pos.x > 2.9)
            else:
                assert_true(world.bodies[mover].pos.x < 2)
                assert_true(world.bodies[mover].linear_vel.x <= 0)


def check_crossing_target() raises:
    var world = PhysicsWorld(gravity=0)
    var mover = world.add_sphere(Vec3(), 0.15, mass=1, restitution=1)
    var target = world.add_sphere(Vec3(1.5, 1.5, 0), 0.15, mass=1, restitution=1)
    world.bodies[mover].enable_ccd = True
    world.set_velocity(mover, Vec3(180, 0, 0))
    world.set_velocity(target, Vec3(0, -180, 0))
    world.step(Float32(1.0 / 60.0))
    assert_true(world.bodies[mover].linear_vel.y < -100)
    assert_true(world.bodies[target].linear_vel.x > 100)


def check_angular_sweep(translation: Float32, miss: Bool) raises:
    var world = PhysicsWorld(gravity=0)
    var step_dt = Float32(1.0 / 60.0)
    var extents = Vec3(0.1, 1.5, 0.1)
    var axis = Vec3(0, 0, -1)
    var post_pos = Vec3(0.85 + translation * Float32(0.45), 0.85, Float32(0.253) if miss else Float32(0))
    var post = world.add_box(post_pos, Vec3(0.07, 0.07, 0.15), mass=0)
    var bar = world.add_box(Vec3(), extents, mass=1)
    var delta = Vec3(translation, 0, 0)
    var omega = Float32(94.25)
    world.set_velocity(bar, delta * 60, axis * omega)
    world.bodies[bar].enable_ccd = True
    world.bodies[bar].angular_damping = 0
    world.bodies[bar].linear_damping = 0
    var steps = Float32(world.settings.substeps)
    var angle = 2 * steps * det_atan(omega * step_dt / (2 * steps))

    # Independent high-resolution discrete SAT reference, including the translated pivot.
    var reference = Float32(1)
    for i in range(2049):
        var fraction = Float32(i) / 2048
        world.bodies[bar].rot = angle_axis(angle * fraction, axis)
        world.bodies[bar].pos = delta * fraction
        if world._collide_pair(bar, post).has_collision:
            reference = fraction
            break
    world.bodies[bar].rot = Quat()
    world.bodies[bar].pos = Vec3()
    var hit = world._cast_angular_ccd(bar, post, delta, axis, angle, angle * extents.length(), 1)
    assert_true(hit.hit == (not miss))
    if miss:
        assert_true(reference == 1 and hit.fraction == 1)
    else:
        assert_true(reference > 0 and reference < 1)
        assert_true(hit.body_slot == post)
        assert_true(abs(hit.fraction - reference) < Float32(0.003))
    world.step(step_dt)
    var actual_tip = world.bodies[bar].rot.rotate(Vec3(0, 1.5, 0)) + world.bodies[bar].pos
    var expected_fraction = Float32(1) if miss else hit.fraction
    var expected_tip = angle_axis(angle * expected_fraction, axis).rotate(Vec3(0, 1.5, 0)) + delta * expected_fraction
    assert_true((actual_tip - expected_tip).length() < Float32(0.005))
    if miss:
        assert_true(abs(world.bodies[bar].angular_vel.z + omega) < Float32(0.1))
    else:
        # A tip strike changes both linear and angular momentum. Neither is erased on resume.
        assert_true(abs(world.bodies[bar].angular_vel.z + omega) > Float32(1))
        assert_true((world.bodies[bar].linear_vel - delta * 60).length() > Float32(1))
        var arm = hit.point - world.bodies[bar].pos
        var velocity_at_impact = world.bodies[bar].linear_vel + world.bodies[bar].angular_vel.cross(arm)
        assert_true(velocity_at_impact.dot(hit.normal) >= Float32(-0.05))


def hull_box(half: Vec3) -> ConvexHull:
    var points = List[Vec3]()
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                points.append(Vec3(Float32(x) * half.x, Float32(y) * half.y, Float32(z) * half.z))
    return ConvexHull(points)


def check_angular_shape_pairs() raises:
    for rounded_mover in [False, True]:
        for obstacle_type in range(3):
            var world = PhysicsWorld(gravity=0)
            var post_pos = Vec3(0.85, 0.85, 0)
            if obstacle_type == 0:
                _ = world.add_sphere(post_pos, 0.08, mass=0)
            elif obstacle_type == 1:
                _ = world.add_capsule(post_pos, 0.05, 0.10, mass=0)
            else:
                _ = world.add_convex_hull(post_pos, hull_box(Vec3(0.07, 0.07, 0.15)), mass=0)
            var bar: Int
            if rounded_mover:
                bar = world.add_capsule(Vec3(), 0.1, 1.4, mass=1)
            else:
                bar = world.add_convex_hull(Vec3(), hull_box(Vec3(0.1, 1.5, 0.1)), mass=1)
            world.bodies[bar].enable_ccd = True
            world.set_velocity(bar, Vec3(), Vec3(0, 0, -94.25))
            world.step(Float32(1.0 / 60.0))
            var tip = world.bodies[bar].rot.rotate(Vec3(0, 1.5, 0))
            assert_true(tip.x > 0.5 and tip.y > 0.8)
            assert_true(world.bodies[bar].linear_vel.length() > Float32(1))


def check_grazing_work_limit() raises:
    # The sphere is radially 2 mm clear of the turning corners at their closest.
    # The advancement budget may slow the first tick here. It must never bounce.
    var world = PhysicsWorld(gravity=0)
    var extent = Vec3(0.1, 1.5, 0.1)
    var radius = sqrt(Float32(1.5 * 1.5 + 0.1 * 0.1)) + Float32(0.052)
    var post = world.add_sphere(Vec3(0.70710678, 0.70710678, 0) * radius, 0.05, mass=0)
    var bar = world.add_box(Vec3(), extent, mass=1)
    world.bodies[bar].enable_ccd = True
    world.bodies[bar].angular_damping = 0
    var axis = Vec3(0, 0, -1)
    var omega = Float32(94.25)
    world.set_velocity(bar, Vec3(), axis * omega)
    var steps = Float32(world.settings.substeps)
    var angle = 2 * steps * det_atan(omega / (120 * steps))
    var result = world._cast_angular_ccd(bar, post, Vec3(), axis, angle, angle * extent.length(), 1)
    assert_true(not result.hit)
    for _ in range(3):
        world.step(Float32(1.0 / 60.0))
        assert_true(world.bodies[bar].linear_vel.length() < Float32(0.001))
        assert_true(abs(world.bodies[bar].angular_vel.z + omega) < Float32(0.1))
    # The arm made it through the closest approach and resumed free rotation.
    assert_true(world.bodies[bar].rot.rotate(Vec3(0, 1, 0)).y < Float32(0.2))


def add_surface(mut world: PhysicsWorld, heightfield: Bool) -> Int:
    # Equivalent finite quad, x in [0, 1], z in [-0.5, 0.5].
    if heightfield:
        var heights = List[Float32](length=4, fill=0)
        return world.add_heightfield(HeightfieldCollider.build(heights, 2, 2, Vec3(0, 0, -0.5), 1))
    var verts: List[Vec3] = [Vec3(0, 0, -0.5), Vec3(0, 0, 0.5), Vec3(1, 0, -0.5), Vec3(1, 0, 0.5)]
    var idxs: List[Int32] = [0, 1, 2, 2, 1, 3]
    return world.add_static_mesh(Bvh8MeshCollider.build(verts, idxs))


def check_fast_terrain(heightfield: Bool) raises:
    var world = PhysicsWorld(gravity=0)
    _ = add_surface(world, heightfield)
    # Centre ray is outside the finite terrain, but the translated box tip crosses it.
    var box = world.add_box(Vec3(-1.0, 2, 0), Vec3(1.2, 0.1, 0.15), mass=1)
    world.bodies[box].enable_ccd = True
    world.set_velocity(box, Vec3(0, -180, 0))
    world.step(Float32(1.0 / 60.0))
    assert_true(world.bodies[box].pos.y > 0)
    assert_true(world.bodies[box].linear_vel.y >= 0)


def first_wedge_contact(heightfield: Bool, compound: Bool) raises -> Float32:
    var world = PhysicsWorld(gravity=0)
    var terrain = add_surface(world, heightfield)
    # Bottom edge slopes from (-1,-1) to (+1,+1). On this finite patch
    # contact starts at x=0 with body y=0; the bounding box overlaps earlier.
    var vertices: List[Vec3] = [
        Vec3(-1, -1, -0.2), Vec3(-1, -1, 0.2),
        Vec3(-1, 1, -0.2), Vec3(-1, 1, 0.2),
        Vec3(1, 1, -0.2), Vec3(1, 1, 0.2),
    ]
    var hull = ConvexHull(vertices)
    var body: Int
    if compound:
        var shape = CompoundShape()
        shape.add_child(CompoundChild.convex(Vec3(), Quat(), hull))
        body = world.add_compound(Vec3(0, 0.5, 0), shape, mass=1)
    else:
        body = world.add_convex_hull(Vec3(0, 0.5, 0), hull, mass=1)
    assert_true(not world._collide_pair(terrain, body).has_collision)
    for i in range(12):
        var y = Float32(0.01) - Float32(i) * Float32(0.002)
        world.bodies[body].pos.y = y
        var manifold = world._collide_pair(terrain, body)
        if manifold.has_collision:
            assert_true(abs(y) <= Float32(0.0021))
            assert_true(manifold.p0.normal.y > Float32(0.5))
            return y
    raise Error("wedge failed to contact terrain")


def main() raises:
    check_post(0, False, True)
    check_post(1.2, False, True)
    check_post(1.2, True, True)
    check_post(1.8, False, False)
    check_capsule_and_compound_movers()
    check_crossing_target()
    check_angular_sweep(0, False)
    check_angular_sweep(0.8, False)
    check_angular_sweep(0, True)
    check_angular_sweep(0.8, True)
    check_angular_shape_pairs()
    check_grazing_work_limit()
    check_fast_terrain(False)
    check_fast_terrain(True)
    for compound in [False, True]:
        var mesh_y = first_wedge_contact(False, compound)
        var hf_y = first_wedge_contact(True, compound)
        assert_true(abs(mesh_y - hf_y) < Float32(0.0021))
    print("Full-shape CCD and convex terrain contact passed")

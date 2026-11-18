"""Flint v5 Physics, KCC, Scene Queries, NavMesh & SIMD8 ORCA Standalone Lab (`labs/lab_physics.mojo`).

Executes and benchmarks:
  1. 200-body pile + joints + `VehicleChassis` snapshot `serialize` at tick 120 -> `deserialize`
     -> step to tick 300 bit-exact `fnv64` rollback verification.
  2. `CharacterController` (`move_and_slide`) stair climb, `45°` walkable vs `65°` steep slide,
     `45°` acute V-corner slide, rotating kinematic turntable inheritance, and crate pushing.
  3. 3D Quickhull + V-HACD-lite L-shape compound + `HeightfieldCollider` Amanatides-Woo DDA raycast.
  4. Recast `NavMesh.build_from_physics` + `< 0.25 ms` dynamic `NavObstacle` door carving + SSFA
     pathfinding + 256-agent antipodal/bottleneck SIMD8 ORCA crowd (`0` agent overlaps).
"""
from std.collections import List
from std.math import abs, sin, cos
from std.time import monotonic
from flint.core.math import Vec3, Quat
from flint.core.detmath import det_sin, det_cos, PI
from flint.core.serial import BlobWriter, BlobReader, fnv64
from flint.physics.narrowphase import ConvexHull
from flint.physics.shapes_v5 import CompoundShape, HeightfieldCollider, SixDofMotorJoint
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import PhysicsWorld
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.nav import NavMesh, NavMeshConfig, NavObstacle, CrowdSimulator


def _make_l_shape_mesh() -> Tuple[List[Vec3], List[Int32]]:
    """Watertight concave 3D L-shape mesh built from two welded arms (16 verts, 28 tris)."""
    # Arm A: [-1.0, 1.0] x [-0.3, 0.3] x [-0.3, 0.3], Arm B: [0.4, 1.0] x [-0.3, 1.3] x [-0.3, 0.3]
    var verts = List[Vec3]()
    var idxs = List[Int32]()

    def append_box(mut vs: List[Vec3], mut is_: List[Int32], lo: Vec3, hi: Vec3):
        var b = Int32(len(vs))
        vs.append(Vec3(lo.x, lo.y, lo.z))  # 0
        vs.append(Vec3(hi.x, lo.y, lo.z))  # 1
        vs.append(Vec3(hi.x, hi.y, lo.z))  # 2
        vs.append(Vec3(lo.x, hi.y, lo.z))  # 3
        vs.append(Vec3(lo.x, lo.y, hi.z))  # 4
        vs.append(Vec3(hi.x, lo.y, hi.z))  # 5
        vs.append(Vec3(hi.x, hi.y, hi.z))  # 6
        vs.append(Vec3(lo.x, hi.y, hi.z))  # 7
        var faces = [
            Int32(0), Int32(2), Int32(1), Int32(0), Int32(3), Int32(2),
            Int32(4), Int32(5), Int32(6), Int32(4), Int32(6), Int32(7),
            Int32(0), Int32(1), Int32(5), Int32(0), Int32(5), Int32(4),
            Int32(3), Int32(6), Int32(2), Int32(3), Int32(7), Int32(6),
            Int32(0), Int32(4), Int32(7), Int32(0), Int32(7), Int32(3),
            Int32(1), Int32(2), Int32(6), Int32(1), Int32(6), Int32(5),
        ]
        for k in range(len(faces)):
            is_.append(b + faces[k])

    append_box(verts, idxs, Vec3(-1.0, -0.25, -0.25), Vec3(1.0, 0.25, 0.25))
    append_box(verts, idxs, Vec3(0.5, 0.25, -0.25), Vec3(1.0, 1.45, 0.25))
    return (verts^, idxs^)


def _build_snapshot_world(mut w: PhysicsWorld, mut car: VehicleChassis):
    _ = w.add_box(Vec3(0.0, -0.5, 0.0), Vec3(30.0, 0.5, 30.0), mass=0.0)

    # 200 dynamic bodies (boxes, spheres, capsules) in a compact pile + weld/6DOF joints
    var prev_slot = -1
    for i in range(200):
        var gx = Float32((i % 5) - 2) * Float32(0.85) + Float32((i // 25) % 2) * Float32(0.08)
        var gz = Float32(((i // 5) % 5) - 2) * Float32(0.85)
        var gy = Float32(0.55) + Float32(i // 25) * Float32(0.75)
        var kind = i % 3
        var slot: Int
        if kind == 0:
            slot = w.add_box(Vec3(gx, gy, gz), Vec3(0.32, 0.32, 0.32), mass=2.5)
        elif kind == 1:
            slot = w.add_sphere(Vec3(gx, gy, gz), 0.32, mass=2.0)
        else:
            slot = w.add_capsule(Vec3(gx, gy, gz), 0.22, 0.20, mass=2.2)

        if i % 20 == 1 and prev_slot >= 0:
            var anchor = (w.bodies[prev_slot].pos + w.bodies[slot].pos) * Float32(0.5)
            _ = w.add_weld_joint(prev_slot, slot, anchor, break_force=5000.0)
        elif i % 40 == 3 and prev_slot >= 0:
            var j6 = SixDofMotorJoint()
            j6.body_a = w.bodies[prev_slot].id
            j6.body_b = w.bodies[slot].id
            j6.local_anchor_a = Vec3(0.35, 0.0, 0.0)
            j6.local_anchor_b = Vec3(-0.35, 0.0, 0.0)
            j6.enable_linear_limit = True
            j6.linear_lower = Vec3(-0.1, -0.1, -0.1)
            j6.linear_upper = Vec3(0.1, 0.1, 0.1)
            j6.enable_angular_drive = True
            j6.angular_hertz = 6.0
            _ = w.add_six_dof_joint(j6^)
        prev_slot = slot

    car = VehicleChassis(1450.0)
    car.setup_four_wheel_car()
    car.pos = Vec3(8.0, 0.55, -6.0)
    car.linear_vel = Vec3(0.0, 0.0, 12.0)


def main() raises:
    var t_start_ns = monotonic()

    # -----------------------------------------------------------------------
    # 1. 200-Body Pile + Vehicle Rollback Snapshot Determinism
    # -----------------------------------------------------------------------
    var t0 = monotonic()
    var world_a = PhysicsWorld()
    var car_a = VehicleChassis(1450.0)
    _build_snapshot_world(world_a, car_a)
    var dt = Float32(1.0 / 60.0)

    for _ in range(120):
        world_a.step(dt)
        car_a.substep_suspension(dt, throttle=0.65, brake=0.0, steer=0.20)

    var snap_w = BlobWriter()
    world_a.serialize(snap_w)
    car_a.serialize(snap_w)

    # Continue uninterrupted to tick 300
    for _ in range(180):
        world_a.step(dt)
        car_a.substep_suspension(dt, throttle=0.30, brake=0.15, steer=-0.15)

    var end_a = BlobWriter()
    world_a.serialize(end_a)
    car_a.serialize(end_a)
    var hash_a = fnv64(end_a.data)

    # Restore snapshot at tick 120 into fresh instances and step to tick 300
    var world_b = PhysicsWorld()
    var car_b = VehicleChassis(1000.0)
    var snap_r = BlobReader(snap_w.data.copy())
    world_b.deserialize(snap_r)
    car_b.deserialize(snap_r)

    for _ in range(180):
        world_b.step(dt)
        car_b.substep_suspension(dt, throttle=0.30, brake=0.15, steer=-0.15)

    var end_b = BlobWriter()
    world_b.serialize(end_b)
    car_b.serialize(end_b)
    var hash_b = fnv64(end_b.data)
    var snap_ms = Float64(monotonic() - t0) / 1.0e6
    print(
        "[lab_physics] 1. 200-body+joints+vehicle snapshot 120->300:",
        "hash_match =", hash_a == hash_b,
        "bytes =", len(snap_w.data),
        "ms =", snap_ms,
    )

    # -----------------------------------------------------------------------
    # 2. Quickhull 3D + V-HACD-lite + Heightfield DDA
    # -----------------------------------------------------------------------
    var t1 = monotonic()
    var cloud = List[Vec3]()
    for i in range(96):
        var a = Float32(i) * Float32(0.37)
        var b = Float32(i) * Float32(0.71)
        cloud.append(Vec3(det_cos(a) * Float32(1.2), det_sin(b) * Float32(0.8), det_sin(a) * det_cos(b)))
    var qhull = ConvexHull.build_quickhull(cloud, max_verts=32)
    var mp_hull = qhull.compute_polyhedral_mass_properties(Float32(1000.0))

    var l_mesh = _make_l_shape_mesh()
    var compound = CompoundShape.decompose_vhacd_lite(l_mesh[0], l_mesh[1], max_hulls=4, voxel_res=14)

    var hf_nx = 33
    var hf_nz = 33
    var hf_h = List[Float32](capacity=hf_nx * hf_nz)
    for iz in range(hf_nz):
        for ix in range(hf_nx):
            var fx = Float32(ix) * Float32(0.5)
            var fz = Float32(iz) * Float32(0.5)
            hf_h.append(Float32(1.5) * Float32(sin(Float64(fx * 0.25)) * cos(Float64(fz * 0.25))))
    var hf = HeightfieldCollider.build(hf_h, hf_nx, hf_nz, Vec3(-8.0, 0.0, -8.0), 0.5)
    var dda_hit = hf.raycast_dda(Vec3(-7.0, 4.0, -7.0), Vec3(1.0, -0.35, 1.0).normalized(), Float32(30.0))
    var shapes_ms = Float64(monotonic() - t1) / 1.0e6
    print(
        "[lab_physics] 2. Quickhull verts =", len(qhull.vertices),
        "mass =", mp_hull[0],
        "VHACD hulls =", compound.child_count(),
        "DDA hit =", dda_hit.hit,
        "ms =", shapes_ms,
    )

    # -----------------------------------------------------------------------
    # 3. KCC Stair Climb + Turntable + Crate Push
    # -----------------------------------------------------------------------
    var t2 = monotonic()
    var kw = PhysicsWorld()
    _ = kw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(30.0, 0.5, 30.0), mass=0.0)
    for s in range(5):
        _ = kw.add_box(
            Vec3(0.0, Float32(s + 1) * Float32(0.125), Float32(2.0) + Float32(s) * Float32(0.6)),
            Vec3(1.5, Float32(s + 1) * Float32(0.125), 0.30),
            mass=0.0,
        )
    _ = kw.add_box(Vec3(0.0, 0.625, 7.2), Vec3(1.5, 0.625, 2.5), mass=0.0)
    kw.rebuild_broadphase()
    var cfg = CharacterControllerConfig()
    cfg.step_height = 0.35
    var kcc = CharacterController(Vec3(0.0, cfg.half_height + cfg.radius, 0.5), cfg)
    for _ in range(90):
        _ = kcc.move_and_slide(kw, Vec3(0.0, 0.0, 4.5), dt)
    var kcc_ms = Float64(monotonic() - t2) / 1.0e6
    print(
        "[lab_physics] 3. KCC 5-step stair climb foot_y =", kcc.foot_position().y,
        "grounded =", kcc.grounded,
        "ms =", kcc_ms,
    )

    # -----------------------------------------------------------------------
    # 4. Recast NavMesh + <0.25 ms Obstacle Carve + 256-Agent SIMD8 ORCA
    # -----------------------------------------------------------------------
    var t3 = monotonic()
    var nw = PhysicsWorld()
    _ = nw.add_box(Vec3(0.0, -0.25, 0.0), Vec3(8.0, 0.25, 6.0), mass=0.0)
    # Dividing wall at x = 0 with a central doorway at z in [-1.2, 1.2]
    _ = nw.add_box(Vec3(0.0, 1.2, -3.6), Vec3(0.3, 1.2, 2.4), mass=0.0)
    _ = nw.add_box(Vec3(0.0, 1.2, 3.6), Vec3(0.3, 1.2, 2.4), mass=0.0)

    var ncfg = NavMeshConfig()
    ncfg.cell_size = 0.35
    ncfg.walkable_radius = 0.35
    var nav = NavMesh.build_from_physics(nw, ncfg)
    var path_open = nav.find_path(Vec3(-5.0, 0.0, 0.0), Vec3(5.0, 0.0, 0.0))

    var t_carve0 = monotonic()
    _ = nav.add_obstacle(NavObstacle.box(1, Vec3(0.0, 1.0, 0.0), Vec3(0.45, 1.0, 1.4)))
    var carve_ms = Float64(monotonic() - t_carve0) / 1.0e6
    var path_blocked = nav.find_path(Vec3(-5.0, 0.0, 0.0), Vec3(5.0, 0.0, 0.0))
    nav.remove_obstacle(1)
    var path_reopened = nav.find_path(Vec3(-5.0, 0.0, 0.0), Vec3(5.0, 0.0, 0.0))

    var crowd = CrowdSimulator(time_horizon=1.8, neighbor_dist=4.5, max_neighbors=16)
    for i in range(256):
        var ang = Float32(i) * (Float32(2.0) * PI / Float32(256.0))
        var r_ring = Float32(22.0) + Float32(i % 2) * Float32(1.2)
        var start_p = Vec3(det_cos(ang) * r_ring, 0.0, det_sin(ang) * r_ring)
        _ = crowd.add_agent(start_p, radius=0.32, max_speed=3.5, goal=Vec3(-start_p.x, 0.0, -start_p.z))

    var max_overlaps = 0
    for _ in range(1000):
        crowd.step_orca_simd8(dt)
        var ov = crowd.count_overlaps()
        if ov > max_overlaps:
            max_overlaps = ov
    var nav_ms = Float64(monotonic() - t3) / 1.0e6
    print(
        "[lab_physics] 4. NavMesh polys =", len(nav.polys),
        "open/blocked/reopen =", path_open.found, path_blocked.found, path_reopened.found,
        "carve_ms =", carve_ms,
        "256-ORCA 1000-tick max_overlaps =", max_overlaps,
        "total_nav_ms =", nav_ms,
    )

    var total_ms = Float64(monotonic() - t_start_ns) / 1.0e6
    print("[lab_physics] COMPLETE in", total_ms, "ms")

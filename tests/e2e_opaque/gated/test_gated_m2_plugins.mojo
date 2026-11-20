"""Gated M2 Contract Verification Suite (`tests/e2e_opaque/gated/test_gated_m2_plugins.mojo`).

Standalone M2-gated contract tests (NOT imported by live suites; compiled and
executed in `--strict` mode once Milestone M2 lands).

Verified Milestone M2 Features (`PROJECT.md § Feature Inventory` & `§ Interface Contracts`):
- **F10**: `PhysicsWorld` two-way `VehicleChassis` coupling (vehicle hosted in
  `PhysicsWorld.bodies`, wheel casts hitting world geometry, two-way collision
  impulses with dynamic bodies).
- **F11**: `PhysicsWorld` two-way `Aero6DofState` coupling (aero bodies bound to
  `PhysicsWorld` slots, `detmath` `SIMD[DType.float32, 8]` evaluation, and
  two-way collision momentum transfer).
- **F12**: `PhysicsWorld` two-way `xpbd.mojo` cloth/chain coupling (`ParticleChain`/
  `ParticleMesh` + `RigidAttachment` stepped inside `PhysicsWorld.step()`).
- **F13**: `PhysicsWorld` two-way `ArticulatedRagdoll` coupling (ragdoll limbs
  colliding against `PhysicsWorld` bodies/`Bvh8MeshCollider` with two-way
  impulse exchange).
- **F14**: First-class `EnginePlugin` wrappers and ECS component schemas/systems
  (`WindowPlugin`, `InputPlugin`, `PhysicsPlugin`, `AnimPlugin`, `AudioPlugin`,
  `RenderPlugin`, `NetPlugin`) wired into `EngineApp` with portable
  `World.snapshot_blob()` / `restore_blob()` serialization.
"""
from std.collections import List
from std.math import abs
from std.testing import assert_true, assert_equal

from flint.core.math import Vec3, Quat
from flint.core.serial import float32_bits

from flint.app import EngineApp, AppConfig
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.input import InputMap
from flint.input.plugin import InputPlugin, InputStateResource, EvdevHubResource
from flint.physics.world import PhysicsWorld
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.vehicle import VehicleChassis
from flint.physics.aero6dof import Aero6DofConfig, Aero6DofControls, Aero6DofState
from flint.physics.xpbd import ParticleChain, ParticleMesh, RigidAttachment
from flint.physics.plugin import PhysicsPlugin, RigidBodyComponent
from flint.anim.pose import SkeletonInstance
from flint.anim.procedural import Placement
from flint.anim.ragdoll import ArticulatedRagdoll
from flint.anim.plugin import AnimPlugin, AnimAssetCache
from flint.assets.model import NativeModel, NativeNode
from flint.audio import AudioServer
from flint.audio.plugin import AudioPlugin, AudioEmitterComponent
from flint.render.render_world import RenderWorld, ScalableRenderer, RENDER_TIER_0_HEADLESS_SIMD
from flint.render.plugin import RenderPlugin, MeshRendererComponent
from flint.net.plugin import NetPlugin, NetConfigResource, NetworkIdentityComponent
from flint.scene import TransformPlugin, LocalTransform, GlobalTransform


def _make_ragdoll_biped() -> NativeModel:
    var m = NativeModel()
    var names = [String("Pelvis"), String("Chest"), String("Thigh.L"), String("Thigh.R")]
    var parents = [-1, 0, 0, 0]
    var offsets = [
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.42, 0.0),
        Vec3(-0.14, -0.35, 0.0),
        Vec3(0.14, -0.35, 0.0),
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
    return m^


def test_f14_seven_engine_plugins_and_schemas() raises:
    var app = EngineApp(AppConfig(title="GatedM2PluginsTest", headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(WindowPlugin(headless=True))
    app.add_plugin(InputPlugin())
    app.add_plugin(PhysicsPlugin())
    app.add_plugin(AnimPlugin())
    app.add_plugin(AudioPlugin(headless=True))
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=128, height=96))
    app.add_plugin(NetPlugin())

    app.startup()
    assert_true(app.resources.has[WindowStateResource]())
    assert_true(app.resources.has[InputMap]())
    assert_true(app.resources.has[InputStateResource]())
    assert_true(app.resources.has[EvdevHubResource]())
    assert_true(app.resources.has[PhysicsWorld]())
    assert_true(app.resources.has[AnimAssetCache]())
    assert_true(app.resources.has[AudioServer]())
    assert_true(app.resources.has[RenderWorld]())
    assert_true(app.resources.has[ScalableRenderer]())
    assert_true(app.resources.has[NetConfigResource]())

    var e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e, LocalTransform(Vec3(1.0, 2.0, 3.0), Quat(), Vec3(1, 1, 1)))
    _ = app.world.insert[GlobalTransform](e, GlobalTransform())
    _ = app.world.insert[RigidBodyComponent](e, RigidBodyComponent(mass=2.0, gravity_scale=0.0))
    _ = app.world.insert[MeshRendererComponent](e, MeshRendererComponent(mesh_id=0, material_id=0))
    _ = app.world.insert[AudioEmitterComponent](e, AudioEmitterComponent(asset_id=-1, volume=0.75))
    _ = app.world.insert[NetworkIdentityComponent](e, NetworkIdentityComponent(net_id=42))

    for _ in range(10):
        app.step_one_frame(Float32(1.0 / 60.0))

    var gt_ptr = app.world.get_ptr[GlobalTransform](e)
    assert_true(Bool(gt_ptr))
    assert_equal(float32_bits(gt_ptr.value()[unsafe_offset=0].translation.x), float32_bits(Float32(1.0)))
    assert_equal(float32_bits(gt_ptr.value()[unsafe_offset=0].translation.y), float32_bits(Float32(2.0)))
    assert_equal(float32_bits(gt_ptr.value()[unsafe_offset=0].translation.z), float32_bits(Float32(3.0)))

    var snap = app.world.snapshot_blob()
    assert_true(len(snap) > 0)

    # Mutate ECS state before restore_blob to prove snapshot restoration is genuine
    app.world.get_ptr_mut[LocalTransform](e).value()[unsafe_offset=0].translation = Vec3(99.0, 99.0, 99.0)
    app.world.get_ptr_mut[NetworkIdentityComponent](e).value()[unsafe_offset=0].net_id = 999
    app.world.get_ptr_mut[AudioEmitterComponent](e).value()[unsafe_offset=0].volume = Float32(0.1)

    app.world.restore_blob(snap)
    assert_true(app.world.has[LocalTransform](e))
    assert_true(app.world.has[RigidBodyComponent](e))
    assert_true(app.world.has[MeshRendererComponent](e))
    assert_true(app.world.has[AudioEmitterComponent](e))
    assert_true(app.world.has[NetworkIdentityComponent](e))
    ref lt_restored = app.world.get_ptr[LocalTransform](e).value()[unsafe_offset=0]
    assert_equal(float32_bits(lt_restored.translation.x), float32_bits(Float32(1.0)))
    assert_equal(float32_bits(lt_restored.translation.y), float32_bits(Float32(2.0)))
    assert_equal(float32_bits(lt_restored.translation.z), float32_bits(Float32(3.0)))
    assert_equal(app.world.get_ptr[NetworkIdentityComponent](e).value()[unsafe_offset=0].net_id, 42)
    assert_equal(
        float32_bits(app.world.get_ptr[AudioEmitterComponent](e).value()[unsafe_offset=0].volume),
        float32_bits(Float32(0.75)),
    )

    app.shutdown()


def test_f10_physics_world_two_way_vehicle_chassis() raises:
    var dt = Float32(1.0 / 60.0)

    # Unobstructed run
    var w_free = PhysicsWorld(gravity=9.81)
    _ = w_free.add_box(Vec3(0.0, -0.5, 0.0), Vec3(40.0, 0.5, 40.0), mass=0.0)
    var car_free = VehicleChassis(1450.0)
    car_free.setup_four_wheel_car()
    car_free.pos = Vec3(0.0, 0.55, 0.0)
    car_free.set_controls(throttle=1.0, brake=0.0, steer=0.0)
    var v_slot_free = w_free.add_vehicle(car_free^)
    w_free.rebuild_broadphase()
    for _ in range(120):
        w_free.step(dt)
    var free_z = w_free.vehicle_state(v_slot_free).pos.z

    # Obstructed run with dynamic crate in front of vehicle
    var w_hit = PhysicsWorld(gravity=9.81)
    _ = w_hit.add_box(Vec3(0.0, -0.5, 0.0), Vec3(40.0, 0.5, 40.0), mass=0.0)
    var crate_slot = w_hit.add_box(Vec3(0.0, 0.5, 4.0), Vec3(0.6, 0.5, 0.6), mass=500.0)
    var car_hit = VehicleChassis(1450.0)
    car_hit.setup_four_wheel_car()
    car_hit.pos = Vec3(0.0, 0.55, 0.0)
    car_hit.set_controls(throttle=1.0, brake=0.0, steer=0.0)
    var v_slot_hit = w_hit.add_vehicle(car_hit^)
    w_hit.rebuild_broadphase()
    for _ in range(120):
        w_hit.step(dt)

    var hit_z = w_hit.vehicle_state(v_slot_hit).pos.z
    assert_true(free_z > Float32(2.0))
    # Two-way coupling: crate moves forward AND vehicle decelerates relative to free run
    assert_true(w_hit.bodies[crate_slot].pos.z > Float32(4.2))
    assert_true(w_hit.bodies[crate_slot].linear_vel.length() > Float32(0.05))
    assert_true(hit_z < free_z - Float32(0.1))


def test_f11_physics_world_two_way_aero6dof() raises:
    var dt = Float32(1.0 / 60.0)

    # Unobstructed baseline flight
    var w_free = PhysicsWorld(gravity=9.81)
    var aero_free = Aero6DofState(Vec3(0.0, 500.0, 0.0), Vec3(0.0, 0.0, 60.0))
    var cfg_free = Aero6DofConfig()
    var ctrl_free = Aero6DofControls()
    ctrl_free.throttle = Float32(1.0)
    var aero_slot_free = w_free.add_aero_body(aero_free, cfg_free, ctrl_free)
    w_free.rebuild_broadphase()
    for _ in range(30):
        w_free.step(dt)
    var st_free = w_free.aero_state(aero_slot_free)

    # Obstructed flight colliding with dynamic target crate
    var w_hit = PhysicsWorld(gravity=9.81)
    var target_slot = w_hit.add_box(Vec3(0.0, 500.0, 12.0), Vec3(1.5, 1.5, 1.0), mass=250.0)

    var aero = Aero6DofState(Vec3(0.0, 500.0, 0.0), Vec3(0.0, 0.0, 60.0))
    var cfg = Aero6DofConfig()
    var ctrl = Aero6DofControls()
    ctrl.throttle = Float32(1.0)
    var aero_slot = w_hit.add_aero_body(aero, cfg, ctrl)
    w_hit.rebuild_broadphase()

    for _ in range(30):
        w_hit.step(dt)

    var st_hit = w_hit.aero_state(aero_slot)
    assert_true(st_hit.pos.x == st_hit.pos.x and st_hit.pos.y == st_hit.pos.y and st_hit.pos.z == st_hit.pos.z)
    # Two-way collision transfers momentum to target rigid body AND decelerates aero body
    assert_true(w_hit.bodies[target_slot].pos.z > Float32(12.2))
    assert_true(w_hit.bodies[target_slot].linear_vel.z > Float32(0.5))
    assert_true(st_hit.pos.z < st_free.pos.z - Float32(0.1))
    assert_true(st_hit.linear_vel.z < st_free.linear_vel.z - Float32(0.5))


def test_f12_physics_world_two_way_xpbd_cloth_chain() raises:
    var dt = Float32(1.0 / 60.0)
    var g = Float32(9.81)

    var w = PhysicsWorld(gravity=g)
    var weight_y0 = Float32(3.0)
    var weight_slot = w.add_box(Vec3(0.0, weight_y0, 0.0), Vec3(0.25, 0.25, 0.25), mass=4.0)

    var chain = ParticleChain.line(Vec3(0.0, 5.0, 0.0), Vec3(0.0, weight_y0, 0.0), 6, mass=0.5)
    chain.pin(0, Vec3(0.0, 5.0, 0.0))
    var att = RigidAttachment(5, weight_slot, Vec3(0.0, 0.0, 0.0), compliance=0.0)
    var c_slot = w.add_xpbd_chain(chain^, att)
    w.rebuild_broadphase()

    for _ in range(60):
        w.step(dt)

    var t = Float32(60.0) * dt
    var uncoupled_y = weight_y0 - Float32(0.5) * g * t * t
    var actual_y = w.bodies[weight_slot].pos.y
    # Weight pulls chain down, while chain tension arrests free-fall well above uncoupled parabola
    assert_true(actual_y < weight_y0)
    assert_true(actual_y > uncoupled_y + Float32(1.0))
    assert_true(w.xpbd_chain(c_slot).position[5].y < Float32(4.9))

    # ParticleMesh two-way coupling with RigidAttachment
    var wm = PhysicsWorld(gravity=g)
    var m_weight_y0 = Float32(3.0)
    var m_weight_slot = wm.add_box(Vec3(0.5, m_weight_y0, 0.0), Vec3(0.25, 0.25, 0.25), mass=4.0)
    var pmesh = ParticleMesh()
    var p0 = pmesh.add_particle(Vec3(0.0, 5.0, 0.0), 0.0, 0.05)
    var p1 = pmesh.add_particle(Vec3(1.0, 5.0, 0.0), 0.0, 0.05)
    var p2 = pmesh.add_particle(Vec3(0.0, m_weight_y0, 0.0), 0.5, 0.05)
    var p3 = pmesh.add_particle(Vec3(1.0, m_weight_y0, 0.0), 0.5, 0.05)
    pmesh.add_link(p0, p1, 0.0)
    pmesh.add_link(p0, p2, 0.0)
    pmesh.add_link(p1, p3, 0.0)
    pmesh.add_link(p2, p3, 0.0)
    pmesh.add_link(p0, p3, 0.0)
    pmesh.add_link(p1, p2, 0.0)
    var m_atts = List[RigidAttachment]()
    m_atts.append(RigidAttachment(p2, m_weight_slot, Vec3(-0.5, 0.0, 0.0), compliance=0.0))
    m_atts.append(RigidAttachment(p3, m_weight_slot, Vec3(0.5, 0.0, 0.0), compliance=0.0))
    var m_slot = wm.add_xpbd_mesh(pmesh^, m_atts)
    wm.rebuild_broadphase()

    for _ in range(60):
        wm.step(dt)

    var actual_mesh_body_y = wm.bodies[m_weight_slot].pos.y
    assert_true(actual_mesh_body_y < m_weight_y0)
    assert_true(actual_mesh_body_y > uncoupled_y + Float32(1.0))
    assert_true(wm.xpbd_mesh(m_slot).position[p2].y < Float32(4.9))


def test_f13_physics_world_two_way_articulated_ragdoll() raises:
    var dt = Float32(1.0 / 60.0)
    var w = PhysicsWorld(gravity=9.81)

    # Static BVH floor at y = 0
    var bvh_v: List[Vec3] = [
        Vec3(-10.0, 0.0, -10.0), Vec3(10.0, 0.0, -10.0),
        Vec3(10.0, 0.0, 10.0), Vec3(-10.0, 0.0, 10.0),
    ]
    var bvh_i: List[Int32] = [Int32(0), Int32(2), Int32(1), Int32(0), Int32(3), Int32(2)]
    _ = w.add_static_mesh(Bvh8MeshCollider.build(bvh_v, bvh_i))

    # Dynamic crate positioned to collide with ragdoll chest/pelvis
    var crate_slot = w.add_box(Vec3(0.0, 1.2, -1.0), Vec3(0.35, 0.35, 0.35), mass=12.0)
    w.set_velocity(crate_slot, Vec3(0.0, 0.0, 8.0))

    var biped = _make_ragdoll_biped()
    var skel = SkeletonInstance()
    skel.reset_to_rest(biped)
    var rag = ArticulatedRagdoll()
    _ = rag.add_limb_from_model(biped, 0, -1, 15.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(biped, 1, 0, 12.0, 0.15, 0.35)
    _ = rag.add_limb_from_model(biped, 2, 0, 6.0, 0.10, 0.35)
    _ = rag.add_limb_from_model(biped, 3, 0, 6.0, 0.10, 0.35)
    rag.handoff_from_skeleton(skel, Placement(), biped, dt)
    rag.go_limp()

    var r_slot = w.add_ragdoll(rag^)
    w.rebuild_broadphase()

    for _ in range(60):
        w.step(dt)

    # Ragdoll must not tunnel below BVH floor y=0, must displace forward, and crate must arrest against ragdoll
    var pelvis_pos = w.ragdoll_state(r_slot).pelvis_world_position()
    assert_true(pelvis_pos.y >= Float32(-0.1))
    assert_true(pelvis_pos.z > Float32(0.5))
    assert_true(w.bodies[crate_slot].pos.z < Float32(1.0))
    assert_true(w.bodies[crate_slot].linear_vel.z < Float32(1.0))


def main() raises:
    var passed = 0
    var total = 5

    test_f14_seven_engine_plugins_and_schemas()
    print("[PASS] test_f14_seven_engine_plugins_and_schemas")
    passed += 1

    test_f10_physics_world_two_way_vehicle_chassis()
    print("[PASS] test_f10_physics_world_two_way_vehicle_chassis")
    passed += 1

    test_f11_physics_world_two_way_aero6dof()
    print("[PASS] test_f11_physics_world_two_way_aero6dof")
    passed += 1

    test_f12_physics_world_two_way_xpbd_cloth_chain()
    print("[PASS] test_f12_physics_world_two_way_xpbd_cloth_chain")
    passed += 1

    test_f13_physics_world_two_way_articulated_ragdoll()
    print("[PASS] test_f13_physics_world_two_way_articulated_ragdoll")
    passed += 1

    print("[SUMMARY] " + String(passed) + "/" + String(total) + " passed")

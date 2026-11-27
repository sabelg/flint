"""Regression coverage for external collision geometry, crowd ownership and wheel contact materials."""
from std.testing import assert_equal, assert_true
from flint.app import AppConfig, EngineApp
from flint.core.math import Vec3
from flint.core.serial import BLOB_HEADER_BYTES, BlobReader, BlobWriter, fnv64
from flint.nav.plugin import NavAgentComponent, NavMeshComponent, NavPlugin, NavPluginStateResource
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.plugin import ColliderComponent, PhysicsPlugin, RigidBodyComponent
from flint.physics.shapes_v5 import HeightfieldCollider
from flint.physics.vehicle import VehicleChassis
from flint.physics.world import PhysicsWorld, WORLD_BLOB_TAG
from flint.scene import GlobalTransform, LocalTransform, TransformPlugin


def terrain(height: Float32) -> HeightfieldCollider:
    var samples = List[Float32](length=4, fill=height)
    return HeightfieldCollider.build(samples, 2, 2, Vec3(), 1)


def mesh(height: Float32) -> Bvh8MeshCollider:
    return Bvh8MeshCollider.build(
        [Vec3(0, height, 0), Vec3(0, height, 1), Vec3(1, height, 0)], [0, 1, 2]
    )


def snapshot_hash(world: PhysicsWorld) -> UInt64:
    var writer = BlobWriter()
    world.serialize(writer)
    return fnv64(writer.data)


def check_geometry_restore() raises:
    var source = PhysicsWorld()
    _ = source.add_static_mesh(mesh(1))
    _ = source.add_heightfield(terrain(2))
    _ = source.add_sphere(Vec3(0, 6, 0), 0.5, 1)
    source.step(Float32(1.0 / 60.0))
    var writer = BlobWriter()
    source.serialize(writer)

    # Same slot counts with different mesh or field data, and absent assets.
    for variant in range(3):
        var target = PhysicsWorld(gravity=2)
        if variant != 2:
            _ = target.add_static_mesh(mesh(Float32(3 if variant == 0 else 1)))
            _ = target.add_heightfield(terrain(Float32(4 if variant == 1 else 2)))
        var before = snapshot_hash(target)
        var reader = BlobReader(writer.data.copy())
        var rejected = False
        try:
            target.deserialize(reader)
        except:
            rejected = True
        assert_true(rejected)
        assert_equal(before, snapshot_hash(target))

    var restored = PhysicsWorld()
    restored.meshes.append(mesh(1))
    restored.heightfields.append(terrain(2))
    restored.heightfields.append(terrain(5))  # Additional unreferenced assets can stay resident.
    var reader = BlobReader(writer.data.copy())
    restored.deserialize(reader)
    assert_equal(snapshot_hash(restored), snapshot_hash(source))
    var hit = restored.raycast(Vec3(0.25, 4, 0.25), Vec3(0, -1, 0), 4)
    assert_true(hit.hit)
    assert_equal(hit.point.y, Float32(2))

    # Historical v3 writes the dynamic payload immediately after the 12-byte blob header.
    var legacy = BlobWriter()
    legacy.begin(WORLD_BLOB_TAG, 3)
    for i in range(BLOB_HEADER_BYTES + 24, len(writer.data)):
        legacy.u8(writer.data[i])
    legacy.end()
    var old_reader = BlobReader(legacy.data.copy())
    restored.deserialize(old_reader)
    assert_equal(snapshot_hash(restored), snapshot_hash(source))


def check_batched_queries() raises:
    var eager = PhysicsWorld()
    var batched = PhysicsWorld()
    batched.begin_body_batch()
    batched.begin_body_batch()
    for i in range(64):
        var center = Vec3(Float32(i * 2), 0, 0)
        var half = Vec3(0.5, 0.5, 0.5)
        _ = eager.add_box(center, half, 0)
        _ = batched.add_box(center, half, 0)
    batched.end_body_batch()
    eager.swap_remove_body(3)
    batched.swap_remove_body(3)
    batched.end_body_batch()
    assert_equal(snapshot_hash(eager), snapshot_hash(batched))
    for i in range(64):
        var origin = Vec3(Float32(i * 2), 3, 0)
        var a = eager.raycast(origin, Vec3(0, -1, 0), 5)
        var b = batched.raycast(origin, Vec3(0, -1, 0), 5)
        assert_equal(a.hit, b.hit)
        assert_equal(a.body_id, b.body_id)


def check_nav_lifecycle() raises:
    var app = EngineApp(AppConfig(headless=True, worker_threads=0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(PhysicsPlugin(gravity=0))
    app.add_plugin(NavPlugin(draw_debug=False))
    app.startup()
    var floor = app.world.spawn()
    _ = app.world.insert[LocalTransform](floor, LocalTransform(Vec3(0, -0.5, 0)))
    _ = app.world.insert[GlobalTransform](floor, GlobalTransform())
    _ = app.world.insert[ColliderComponent](floor, ColliderComponent.box(Vec3(10, 0.5, 10)))
    _ = app.world.insert[RigidBodyComponent](floor, RigidBodyComponent.static_body())
    var mesh_entity = app.world.spawn()
    _ = app.world.insert[NavMeshComponent](mesh_entity, NavMeshComponent(auto_bake=True))
    var entity = app.world.spawn()
    _ = app.world.insert[LocalTransform](entity, LocalTransform(Vec3(-4, 0, 0)))
    _ = app.world.insert[GlobalTransform](entity, GlobalTransform())
    _ = app.world.insert[NavAgentComponent](entity, NavAgentComponent(goal=Vec3(4, 0, 0), has_goal=True))
    var dt = Float32(1.0 / 60.0)
    for _ in range(15):
        app.step_one_frame(dt)
    ref state = app.resources.get_ptr[NavPluginStateResource]().value()[]
    var old_slot = app.world.get_ptr[NavAgentComponent](entity).value()[].agent_id
    assert_true(old_slot >= 0)
    app.world.get_ptr_mut[NavAgentComponent](entity).value()[].enabled = False
    app.step_one_frame(dt)
    assert_equal(state.crowd.count, 0)
    assert_true(not state.crowd.active[old_slot])
    assert_equal(app.world.get_ptr[NavAgentComponent](entity).value()[].bridge_idx, -1)
    app.world.get_ptr_mut[NavAgentComponent](entity).value()[].enabled = True
    app.step_one_frame(dt)
    assert_equal(app.world.get_ptr[NavAgentComponent](entity).value()[].agent_id, old_slot)
    _ = app.world.despawn(entity)
    for _ in range(100):
        var replacement = app.world.spawn()
        _ = app.world.insert[LocalTransform](replacement, LocalTransform())
        _ = app.world.insert[NavAgentComponent](replacement, NavAgentComponent())
        app.step_one_frame(dt)
        assert_equal(state.crowd.count, 1)
        assert_equal(len(state.bridge.agents), 1)
        assert_equal(len(state.entity_to_bridge), 1)
        _ = app.world.despawn(replacement)
    app.step_one_frame(dt)
    assert_equal(state.crowd.count, 0)
    assert_equal(len(state.entity_to_bridge), 0)
    app.shutdown()


def drive(surface: Float32) -> Float32:
    var world = PhysicsWorld()
    _ = world.add_box(Vec3(0, -0.5, 0), Vec3(100, 0.5, 100), 0, friction=surface)
    var car = VehicleChassis(1450)
    car.setup_four_wheel_car()
    car.pos = Vec3(0, 0.55, 0)
    car.set_controls(throttle=1, brake=0, steer=0)
    var vehicle = world.add_vehicle(car^)
    for _ in range(120):
        world.step(Float32(1.0 / 60.0))
    return world.vehicle_state(vehicle).linear_vel.z


def brake(left: Float32, right: Float32) -> Tuple[Float32, Float32]:
    var world = PhysicsWorld()
    _ = world.add_box(Vec3(-50, -0.5, 0), Vec3(50, 0.5, 100), 0, friction=left)
    _ = world.add_box(Vec3(50, -0.5, 0), Vec3(50, 0.5, 100), 0, friction=right)
    var car = VehicleChassis(1450)
    car.setup_four_wheel_car()
    car.pos = Vec3(0, 0.55, 0)
    car.linear_vel = Vec3(0, 0, 12)
    for i in range(len(car.wheels)):
        car.wheel_states[i].angular_velocity = 12 / car.wheels[i].radius
    car.set_controls(throttle=0, brake=1, steer=0)
    var vehicle = world.add_vehicle(car^)
    var yaw_at_half_second = Float32(0)
    for tick in range(1, 1201):
        world.step(Float32(1.0 / 60.0))
        ref state = world.vehicles[vehicle].chassis
        if tick == 30:
            yaw_at_half_second = state.angular_vel.y
        if state.linear_vel.length() < 0.5:
            return (state.pos.z, yaw_at_half_second)
    return (Float32(-1), yaw_at_half_second)


def main() raises:
    check_geometry_restore()
    check_batched_queries()
    check_nav_lifecycle()
    var ice = drive(0.01)
    var road = drive(1)
    assert_true(road > 1)
    assert_true(ice < road * 0.5)
    var dry_distance = brake(1, 1)[0]
    var slippery_distance = brake(0.2, 0.2)[0]
    assert_true(dry_distance > 0 and slippery_distance > 2 * dry_distance)
    var left_yaw = brake(1, 0.2)[1]
    var right_yaw = brake(0.2, 1)[1]
    assert_true(abs(left_yaw) > 0.1 and abs(left_yaw + right_yaw) < 0.001)
    print("[PASS] collision geometry identity, batched query parity, crowd lifetimes, wheel traction and split-grip braking")

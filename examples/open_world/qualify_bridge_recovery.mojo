"""CPU replay of 28 fall/recovery cycles using the bridge's normal update and checkpoint paths."""
from std.ffi import external_call
from std.sys import argv
from std.testing import assert_equal, assert_true
from std.time import monotonic
from bridge import BridgeRoute, load_bridge, update_bridge, valid_bridge_position
from flint.anim.plugin import AnimPlugin, AnimatorComponent, RagdollComponent
from flint.app import AppConfig, EngineApp, STAGE_PRE_UPDATE
from flint.assets.asset_plugin import AssetPlugin
from flint.core.math import Vec3
from flint.input.plugin import InputPlugin
from flint.physics.plugin import PhysicsPlugin
from flint.physics.world import PhysicsWorld
from flint.platform.os import make_dirs, remove_file
from flint.render.plugin import CameraComponent
from flint.scene import GlobalTransform, LocalTransform, TransformPlugin


struct TemporaryPlayerSaves(Movable):
    var directory: String

    def __init__(out self) raises:
        var parent = String("build/qualification")
        if not make_dirs(parent):
            raise Error("Cannot create bridge qualification directory")
        var directory = parent + "/bridge-" + String(external_call["getpid", Int32]()) + "-" + String(monotonic())
        if external_call["mkdir", Int32](directory.as_c_string_span(), UInt32(0o700)) != 0:
            raise Error("Cannot create private bridge qualification save directory")
        self.directory = directory
        var variable = String("FLINT_SAVE_DIR")
        if external_call["setenv", Int32](variable.as_c_string_span(), directory.as_c_string_span(), Int32(1)) != 0:
            raise Error("Cannot isolate bridge qualification saves")

    def __deinit__(deinit self):
        var subfolder = self.directory + "/sunspire"
        _ = remove_file(subfolder + "/bridge.json")
        _ = external_call["rmdir", Int32](subfolder.as_c_string_span())
        var owned = self.directory
        _ = external_call["rmdir", Int32](owned.as_c_string_span())


def main() raises:
    var args = argv()
    if len(args) > 2:
        raise Error("Usage: world --check-recovery [pack]")
    var pack = String(args[1]) if len(args) == 2 else String()
    var saves = TemporaryPlayerSaves()
    var app = EngineApp(AppConfig(headless=True, worker_threads=0, fixed_hz=60))
    app.add_plugin(TransformPlugin(snapshot_prev_in_fixed=True))
    app.add_plugin(InputPlugin(enable_evdev=False))
    app.add_plugin(AssetPlugin(loose_root="" if pack else "examples/character_course/assets"))
    app.add_plugin(PhysicsPlugin(gravity=9.81))
    app.add_plugin(AnimPlugin())
    var camera = app.world.spawn()
    _ = app.world.insert[LocalTransform](camera, LocalTransform())
    _ = app.world.insert[GlobalTransform](camera, GlobalTransform())
    _ = app.world.insert[CameraComponent](camera, CameraComponent())
    load_bridge(app, camera, pack, baseline=False, replay=True)
    _ = app.add_system(STAGE_PRE_UPDATE, "open_world.bridge", update_bridge, exclusive=True)
    app.startup()
    ref route = app.resources.get_ptr[BridgeRoute]().value()[]
    ref physics = app.resources.get_ptr[PhysicsWorld]().value()[]
    var middle = Vec3(0.3, 2.57, -3)
    var mid = route.recover_on_deck(physics, middle, -0.9)
    assert_true(mid[0] and (mid[1] - middle).length_squared() == 0)
    var unsupported = PhysicsWorld()
    assert_true(not route.recover_on_deck(unsupported, middle, -0.9)[0])
    var checked = 0
    var xpeak = Float32(0)
    var ymin = Float32(1000)
    var ymax = Float32(-1000)
    for frame in range(100800):
        app.step_one_frame(Float32(1.0 / 60.0))
        if frame == 2:
            assert_true(app.world.get_ptr[AnimatorComponent](route.actor).value()[].animator.configured())
        if not route.checkpoint_ready:
            continue
        assert_true(valid_bridge_position(route.position))
        xpeak = max(xpeak, abs(route.position.x))
        ymin = min(ymin, route.position.y)
        ymax = max(ymax, route.position.y)
        var recoveries = app.world.get_ptr[RagdollComponent](route.actor).value()[].state_machine.recovery_count
        if recoveries != checked:
            assert_equal(recoveries, checked + 1)
            assert_true(route.save_checkpoint())
            var restored = BridgeRoute(camera, replay=True)
            assert_true(restored.load_checkpoint())
            assert_true((restored.position - route.position).length_squared() < 1e-8)
            assert_equal(restored.interactions, route.interactions)
            checked = recoveries
    assert_equal(checked, 28)
    assert_equal(route.falls, checked)
    assert_true(route.interactions >= checked and route.pivots >= checked)
    var bad = route.copy()
    bad.position = Vec3(20, 1, 0)
    assert_true(not bad.save_checkpoint())
    var intact = BridgeRoute(camera, replay=True)
    assert_true(intact.load_checkpoint())
    print("[PASS] 100800 bridge frames, falls/recoveries/checkpoint round trips", checked, "pivots", route.pivots,
          "guide interactions", route.interactions, "max abs x", xpeak, "min/max y", ymin, ymax)
    app.shutdown()
    _ = saves^

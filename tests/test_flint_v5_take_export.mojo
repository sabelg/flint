"""An edited take retains its clip identity, markers and channel sampling in Studio GLB export."""
from std.ffi import external_call
from std.testing import assert_equal, assert_true
from flint.anim.take_optimize import optimize_bake
from flint.assets.glb_native import load_native_glb
from flint.assets.glb_take import append_take_glb
from flint.assets.manifest import ClipEvent
from flint.assets.model import INTERP_CUBICSPLINE, NativeAnimation, NativeChannel
from flint.platform.os import read_bytes, remove_file


def main() raises:
    var source = String("examples/character_course/assets/performer.glb")
    var output = "/tmp/flint-take-roundtrip-" + String(external_call["getpid", Int32]()) + ".glb"
    var model = load_native_glb(source)
    var clip = model.animations[1].copy()
    clip.name = 'Walk "edited"'
    clip.loop = False
    clip.events.append(ClipEvent("plant", 'left "sole"', 0.25))
    var scale = NativeChannel()
    scale.node = 0
    scale.path = 2
    scale.times = [Float32(0.125), 0.625]
    scale.interp_mode = INTERP_CUBICSPLINE
    scale.values = [Float32(0),0,0, 1,1,1, 0,0,0, 0,0,0, 1,1.2,1, 0,0,0]
    # Studio's synchronized bake optimizer must retain both cubic tangent vectors.
    var cubic = NativeAnimation()
    cubic.duration = 0.625
    cubic.channels.append(scale.copy())
    var cubic_index = len(model.animations)
    model.animations.append(cubic^)
    _ = optimize_bake(model, cubic_index, 0.001)
    assert_equal(model.animations[cubic_index].channels[0].values, scale.values)
    _ = model.animations.pop()
    clip.channels.append(scale^)
    clip.channels[0].step = True
    try:
        append_take_glb(source, output, clip)
        var edited = load_native_glb(output)
        assert_equal(len(edited.animations), len(model.animations) + 1)
        assert_equal(len(edited.skins), len(model.skins))
        ref got = edited.animations[len(model.animations)]
        assert_equal(got.name, clip.name)
        assert_equal(got.loop, clip.loop)
        assert_equal(got.ground_speed, clip.ground_speed)
        assert_equal(len(got.events), len(clip.events))
        for i in range(len(clip.events)):
            assert_equal(got.events[i].kind, clip.events[i].kind)
            assert_equal(got.events[i].arg, clip.events[i].arg)
            assert_equal(got.events[i].time, clip.events[i].time)
        assert_equal(len(got.channels), len(clip.channels))
        for i in range(len(clip.channels)):
            assert_equal(got.channels[i].node, clip.channels[i].node)
            assert_equal(got.channels[i].path, clip.channels[i].path)
            assert_equal(got.channels[i].times, clip.channels[i].times)
            assert_equal(got.channels[i].values, clip.channels[i].values)
        assert_true(got.channels[0].step)
        assert_equal(got.channels[len(got.channels)-1].interp_mode, INTERP_CUBICSPLINE)
    finally:
        _ = remove_file(output)
    _ = clip.channels[len(clip.channels)-1].values.pop()
    var rejected = False
    try:
        append_take_glb(source, output, clip)
    except:
        rejected = True
    assert_true(rejected and len(read_bytes(output)) == 0)
    print("Edited character take: original rig, named clip, markers and mixed track sampling passed")

"""Tier 5 Coverage & Regression Suite (`tests/test_flint_v5_tier5_coverage.mojo`).

Covers previously untested modules + Tier 5 adversarial regression boundaries:
  1. `flint.anim.choreography` (`MotionSequence`, `MotionPlayer`, overlap underflow 0.0/0.0 guard, save/load)
  2. `flint.anim.combat_timing` (`CombatTiming`, `active`, `can_chain`, `marker` valid & OOB `-1` sentinel, `move_marker`, save/load)
  3. `flint.studio.combat_editor` (`CombatTimeline` valid drag/nudge & zero-width / zero-end / zero-lane guards)
  4. `flint.studio.profile_compare` (`Capture` valid frames, stats, `at` valid/negative/OOB, incomplete JSON error)
  5. `flint.studio.traversal_room` (`TraversalRoom` parse/save/save_copy/`ground_point` & invalid-room rejection)
  6. `flint.studio.motion_editor` (`relative_to`, `next_clip`, `timing_click`, `FollowCamera`, `point`, `frame`)
  7. `flint.studio.ide_pty` (`IdePty` RAII move/drop, `stop(pid=0)`, `resize(-1)`)
  8. `flint.core.crash` (`crash_signals`, `write_signal_name`, `le_word`, `hex_digit`, `elf_build_id`, `executable_path`)
  9. `flint.core.log` (`level_name`, `level_letter`, `category_name`, `parse_level`, `find_category`, `log_enabled`)
 10. `flint.physics.{defaults,density_table,wind}` (`FxVec`, `WindField` gusts/vorticles/serialization, material densities)
 11. `flint.render.{dither,gpu_layout,lighting,occupancy_layout,particle_atlas,particle_layout}`
"""
from std.ffi import external_call
from std.math import abs, isinf, isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_false, assert_true
from flint.anim.choreography import MotionPlayer, MotionSequence, cue_kind
from flint.anim.combat_timing import CombatTiming
from flint.anim.flanim_ecs import load_flanim_bytes, save_flanim_bytes, FLANIM_FOURCC, FLANIM_VERSION
from flint.anim.graph import AnimGraphDef, Condition, StateDef, TransitionDef
from flint.anim.plugin import AnimatorComponent, SkeletonComponent
from flint.anim.pose import Inertializer, Pose, SkeletonInstance
from flint.anim.procedural import Placement, pose_position
from flint.anim.ragdoll import ArticulatedLimbDef, ArticulatedRagdoll, RAGDOLL_BLOB_TAG, RAGDOLL_BLOB_VERSION
from flint.assets.bc_compress import decode_bc5_normals, decode_bc6h_hdr, decode_bc7_rgba
from flint.assets.cooker_v2 import AssetMeta, AssetUuid, cook_native_model_bundle, cook_single_source_bytes
from flint.assets.flpk_v3 import (
    COOKED_BVH8_SLAB_MAGIC,
    decode_audio_pcm_slab,
    decode_bvh8_collider_slab,
    encode_audio_pcm_slab,
    encode_bvh8_collider_slab,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.hdr_image import (
    EXR_DWAA_COMPRESSION,
    EXR_MAGIC,
    EXR_ZIP_COMPRESSION,
    EXR_ZIPS_COMPRESSION,
    decode_exr_fp16,
    decode_radiance_hdr,
    encode_exr_fp16,
    encode_radiance_hdr,
)
from flint.assets.image import BitReader as ImageBitReader, decode_png, encode_png, inflate_zlib
from flint.assets.jpeg import decode_jpeg, encode_jpeg_baseline
from flint.assets.json import parse_json
from flint.assets.ktx2 import decode_ktx2, decompress_zstd
from flint.assets.meshlet import AaaGeometryPool
from flint.assets.model import (
    NativeAnimation,
    NativeChannel,
    NativeMesh,
    NativeModel,
    NativeNode,
)
from flint.assets.replay_stream import FlintReplayStream, REPLAY_STREAM_TAG, REPLAY_STREAM_VERSION
from flint.assets.server import AssetServer
from flint.assets.watcher import AssetWatcher
from flint.audio.qoa import QOA_MAGIC, QoaStreamCursor, decode_qoa, encode_qoa
from flint.audio.plugin import AudioEmitterComponent
from flint.audio.server import AudioServer, BUS_SFX
from flint.core.anchor import RawText
from flint.core.crash import (
    SIGABRT,
    SIGBUS,
    SIGFPE,
    SIGILL,
    SIGSEGV,
    SIGTRAP,
    crash_signals,
    elf_build_id,
    executable_path,
    hex_digit,
    le_word,
    write_signal_name,
)
from flint.core.jobs import JobPool, NO_JOB
from flint.core.log import (
    LOG_ASSETS,
    LOG_CORE,
    LOG_ERROR,
    LOG_FOLLOW,
    LOG_INFO,
    LOG_OFF,
    LOG_TRACE,
    LOG_WARN,
    category_name,
    find_category,
    level_letter,
    level_name,
    log_enabled,
    parse_level,
)
from flint.core.lz4 import lz4_compress, lz4_decompress, lz4_pack
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.cycles.pattern import Frac, K_ALT_ABS, Params, Pattern, bjorklund, chord_notes, mini, parse_number
from flint.cycles.script import eval_pattern, run_program
from flint.cycles.synth import CyclesEngine, render_loop_to_wav, render_to_wav
from flint.devtools.mcp_v2 import McpServerV2
from flint.ecs.world import World
from flint.ml.motionbricks import load_bvh
from flint.ml.onnx import OnnxModel
from flint.ml.protobuf import PbReader
from flint.ml.ops import (
    B_DIV,
    B_MOD,
    argmax,
    binary,
    clip,
    concat,
    conv1d,
    gather,
    gather_elements,
    gemm,
    layer_norm,
    matmul,
    range_op,
    reduce,
    resize_nearest,
    scatter_elements,
    slice,
    topk,
    transpose,
)
from flint.ml.tensor import DT_F32, DT_I64, Tensor, f32_from_bits, numel_of
from flint.nav.navmesh import NavMesh, NavMeshConfig, NavPoly, NavTile
from flint.nav.orca import CrowdSimulator
from flint.net.channel import NetChannel
from flint.net.crypto import AeadKey, aead_open_raw, aead_seal_raw
from flint.net.plugin import NetworkIdentityComponent
from flint.net.virtual_sim import VirtualNetSim
from flint.platform.input import InputState
from flint.platform.logind import dbus_message_length, dbus_method_call, dbus_parse
from flint.studio.command_menu import CommandMenu
from flint.platform.os import list_directory, make_dirs, read_bytes, remove_file, write_bytes
from flint.physics.bvh import Bvh8MeshCollider, Bvh8Node
from flint.physics.defaults import (
    BODY_MAX_SPEED,
    CONTACT_DAMPING_RATIO,
    CONTACT_HERTZ,
    JOINT_HERTZ,
    PHYSICS_SUBSTEPS,
    RESTITUTION_THRESHOLD,
    ROLLING_RESISTANCE,
    WATER_ANGULAR_DRAG,
)
from flint.physics.density_table import (
    DENSITY_MATERIALS,
    material_density_milli,
    material_rolling_factor,
    material_wind_drag,
)
from flint.physics.events import CachedContact, ContactCache
from flint.physics.islands import ConstraintGraph, IslandSet
from flint.physics.narrowphase import ConvexHull
from flint.physics.plugin import ClothMeshComponent, ColliderComponent, RigidBodyComponent
from flint.physics.shapes_v5 import COMPOUND_BLOB_TAG, CompoundChild, CompoundShape, SHAPES_V5_BLOB_VERSION
from flint.physics.vehicle import VEHICLE_BLOB_TAG, VEHICLE_BLOB_VERSION, VehicleChassis
from flint.physics.world import (
    BODY_SHAPE_COMPOUND,
    BODY_SHAPE_CONVEX_HULL,
    PhysicsWorld,
    WORLD_BLOB_TAG,
    WORLD_BLOB_VERSION,
)
from flint.physics.xpbd import ParticleChain, ParticleMesh, RigidAttachment, XPBD_CHAIN_TAG, XPBD_MESH_TAG, XPBD_BLOB_VERSION
from flint.physics.wind import (
    FALLOFF_LINEAR,
    FALLOFF_SMOOTH,
    FX_ONE,
    FxShape,
    FxVec,
    SHAPE_BOX,
    SHAPE_CAPSULE,
    SHAPE_SPHERE,
    Vorticle,
    WindField,
    WindSource,
    falloff_fx,
    fx_div,
    fx_mul,
    isqrt,
    make_vorticle,
    to_fx,
    vorticle_wind,
)
from flint.physics.forces import FieldVolume, ForceWorld, TARGET_WIND
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_render_graph import (
    AaaRenderGraph,
    RG_USAGE_COMPUTE_READ,
    RG_USAGE_COMPUTE_WRITE,
    RgResourceAccess,
)
from flint.render.aaa_vsm import VsmPageManager
from flint.render.dither import (
    DITHER_LEVELS,
    DITHER_OFF_INVERSE_Z,
    DITHER_OFF_OFFSET,
    bayer4,
    dither_view,
)
from flint.render.flip import flip_error_map, flip_mean
from flint.render.gpu_layout import (
    CONSTANT_BYTES,
    CONSTANT_CAMERA,
    CONSTANT_DITHER,
    CONSTANT_EFFECTS,
    DITHER_FLAG,
    RECORD_BYTES,
    RECORD_FLOATS,
    TILE_SIZE,
)
from flint.render.lighting import (
    EventLightPool,
    LightingProfile,
    MAX_EVENT_LIGHTS,
    MAX_REGISTERED_LIGHTS,
    MAX_SELECTED_LIGHTS,
    regional_profile,
)
from flint.render.occupancy_layout import (
    CHUNK_SIZE,
    OCC_HEADER_WORDS,
    OccupancyLayout,
    write_occupancy_header,
)
from flint.render.particle_atlas import (
    lattice as atlas_lattice,
    particle_atlas,
    to_byte as atlas_to_byte,
    value_noise as atlas_value_noise,
)
from flint.render.particle_layout import (
    ATLAS_BYTES,
    ATLAS_FRAME,
    ATLAS_FRAMES,
    EMITTER_WORDS,
    MAX_EMITTERS,
    PARTICLE_WORDS,
    STATE_HEADER_WORDS,
)
from flint.render.font import SdfFont
from flint.render.plugin import MeshRendererComponent
from flint.render.simd_canvas import Camera3D, SimdCanvas
from flint.render.spirv import (
    CAPABILITY_SHADER,
    OP_BRANCH,
    OP_BRANCH_CONDITIONAL,
    OP_CAPABILITY,
    OP_CONSTANT,
    OP_ENTRY_POINT,
    OP_FUNCTION,
    OP_FUNCTION_END,
    OP_LABEL,
    OP_MEMORY_MODEL,
    OP_RETURN,
    OP_TYPE_BOOL,
    OP_TYPE_FUNCTION,
    OP_TYPE_INT,
    OP_TYPE_VOID,
    SPIRV_MAGIC,
)
from flint.render.spirv_validate import validate_spirv_module
from flint.scene.scene import ComponentRegistry, SCENE_BIN_MAGIC, SCENE_SCHEMA_VERSION, Scene
from flint.scene.transform_hierarchy import LocalTransform, set_parent
from flint.studio.combat_editor import CombatTimeline
from flint.studio.ide_pty import IdePty
from flint.studio.ide_view import raster_ui
from flint.studio.motion_editor import FollowCamera, frame, next_clip, point, relative_to, timing_click
from flint.studio.motion_tools import MotionTools, CONTACT, TIMING, draw_dust, frame_joint, shake_offset
from flint.studio.profile_compare import Capture
from flint.studio.scene_menu import SceneMenu
from flint.studio.studio_app import FlintStudioState
from flint.studio.traversal_room import TraversalRoom
from flint.text.sdf_bake import (
    bake_glyph,
    bake_glyph_id,
    bake_glyph_msdf,
    bake_glyph_mtsdf,
    extract_edge_contours,
    flatten,
)
from flint.text.ttf import GlyphOutline, GsubGlyph, TrueTypeFont
from flint.ui.draw_list import GpuUiPrim, PRIM_GLYPH, PRIM_SHAPE, UiDrawList
from flint.ui.flexbox_dock import DockSpace, RectF


def _rm_dir_shallow(path: String):
    var entries = list_directory(path)
    for e in entries:
        _ = remove_file(path + "/" + e)
    var owned = path
    _ = external_call["rmdir", Int32](owned.as_c_string_span())


def _write_text(path: String, text: String) raises:
    var b = List[UInt8]()
    for byte in text.as_bytes():
        b.append(byte)
    assert_true(write_bytes(path, b), "write_bytes failed for " + path)


def _make_test_model(dur_a: Float32, dur_b: Float32) -> NativeModel:
    var m = NativeModel()
    var root = NativeNode()
    root.name = "hips"
    root.translation = Vec3(0.0, 1.0, 0.0)
    var hand = NativeNode()
    hand.name = "hand_r"
    hand.translation = Vec3(0.2, 0.5, 0.3)
    root.children.append(1)
    m.nodes.append(root^)
    m.nodes.append(hand^)

    var anim_a = NativeAnimation()
    anim_a.name = "jump"
    anim_a.duration = dur_a
    var ch_a = NativeChannel()
    ch_a.node = 0
    ch_a.path = 0
    ch_a.times = [Float32(0.0), dur_a]
    ch_a.values = [Float32(0.0), 1.0, 0.0, 0.0, 1.2, 0.0]
    anim_a.channels.append(ch_a^)
    m.animations.append(anim_a^)

    var anim_b = NativeAnimation()
    anim_b.name = "grab"
    anim_b.duration = dur_b
    var ch_b = NativeChannel()
    ch_b.node = 0
    ch_b.path = 0
    ch_b.times = [Float32(0.0), dur_b]
    ch_b.values = [Float32(0.0), 1.2, 0.0, 0.0, 1.4, 0.0]
    anim_b.channels.append(ch_b^)
    m.animations.append(anim_b^)
    return m^


def test_anim_choreography_and_combat_timing(tmp_dir: String) raises:
    # 1. MotionSequence parse, save/load, cue_crossed, and overlap underflow 0.0/0.0 guard
    var motion_path = tmp_dir + "/ledge_grab.motion.json"
    var motion_json = (
        '{\n'
        '  "version": 1,\n'
        '  "model": "hero.glb",\n'
        '  "first": "jump",\n'
        '  "second": "grab",\n'
        '  "contact": "hand_r",\n'
        '  "markers": ["hand_r"],\n'
        '  "support": "",\n'
        '  "overlap": 1e-15,\n'
        '  "scale": 1.0,\n'
        '  "start": [0.0, 0.0, -1.0],\n'
        '  "ledge": [0.0, 2.0, 0.0],\n'
        '  "yaw": 0.0,\n'
        '  "cameraStart": [2.0, 2.0, -3.0],\n'
        '  "cameraEnd": [2.0, 2.5, -1.0],\n'
        '  "cameraAim": [0.0, 1.5, 0.0],\n'
        '  "cue": 0.5,\n'
        '  "sound": ""\n'
        '}\n'
    )
    _write_text(motion_path, motion_json)
    var seq = MotionSequence(motion_path)
    assert_true(seq.cue_crossed(-1.0, 0.5))
    assert_false(seq.cue_crossed(0.5, 1.0))

    # [Med-11] Overlap underflow (1e-35 * 1e-15 == 0.0) must NOT produce NaN or teleport to wall at t=0
    var tiny_model = _make_test_model(Float32(1e-35), Float32(1e-35))
    var player_tiny = MotionPlayer(tiny_model, seq)
    var wall = Placement(Vec3(10.0, 2.0, 0.0), Vec3(1.0, 1.0, 1.0), 0.0)
    var actor0 = player_tiny.sample(tiny_model, seq, wall, Float32(0.0), entry=Vec3(0.0, 0.0, 0.0))
    assert_true(not isnan(actor0.position.x))
    assert_true(abs(actor0.position.x) < Float32(1.0e-4), "must start at entry, got " + String(actor0.position.x))

    # Normal sequence duration, can_catch, save & reload round-trip
    seq.overlap = Float32(0.25)
    var saved_motion = tmp_dir + "/saved.motion.json"
    seq.save(saved_motion)
    var seq2 = MotionSequence(saved_motion)
    assert_true(abs(seq2.overlap - Float32(0.25)) < Float32(1e-5))
    var model = _make_test_model(Float32(1.0), Float32(0.8))
    var player = MotionPlayer(model, seq2)
    assert_true(abs(player.duration(model, seq2) - Float32(1.6)) < Float32(1e-4))
    assert_true(player.can_catch(wall, wall.to_world(Vec3(0.0, 0.0, -0.1)), Vec3(0.0, -0.5, 1.0), Float32(0.5)))
    assert_false(player.can_catch(wall, wall.to_world(Vec3(0.0, 0.0, -0.1)), Vec3(0.0, 0.5, 1.0), Float32(0.5)))

    # [R2-6] MotionSequence.set_event invalid cue kind / negative frame & cue_kind OOB safety
    assert_equal(cue_kind(-1), "Unknown")
    assert_equal(cue_kind(99), "Unknown")
    var n_events_before = len(seq.events)
    seq.set_event(10, 99)
    seq.set_event(-1, 0)
    assert_equal(len(seq.events), n_events_before)
    seq.set_event(10, 1)
    assert_equal(len(seq.events), n_events_before + 1)
    seq.save(saved_motion)

    # 2. CombatTiming & [Low/Med-12] OOB marker(-1)/marker(99) -> -1 sentinel
    var combat_path = tmp_dir + "/slash.combat.json"
    _write_text(
        combat_path,
        '{"version":1,"fps":60,"end":30,"impact":8,"close":14,"chain":20,"reach":1.8}\n',
    )
    var ct = CombatTiming(combat_path)
    assert_true(ct.active(8.0) and ct.active(13.5) and not ct.active(14.0))
    assert_true(ct.can_chain(20.0) and ct.can_chain(29.0) and not ct.can_chain(30.0))
    assert_equal(ct.marker(0), 8)
    assert_equal(ct.marker(1), 14)
    assert_equal(ct.marker(2), 20)
    assert_equal(ct.marker(-1), -1)
    assert_equal(ct.marker(99), -1)
    ct.move_marker(0, 10)
    assert_equal(ct.impact, 10)
    ct.move_marker(0, 99)
    assert_equal(ct.impact, 13)  # clamped to close - 1
    var saved_combat = tmp_dir + "/saved.combat.json"
    ct.save(saved_combat)
    var ct2 = CombatTiming(saved_combat)
    assert_equal(ct2.impact, 13)

    # Bad order must raise
    var bad_combat = tmp_dir + "/bad.combat.json"
    _write_text(bad_combat, '{"version":1,"fps":60,"end":30,"impact":15,"close":10,"chain":20,"reach":1.8}\n')
    var caught_ct = False
    try:
        _ = CombatTiming(bad_combat)
    except:
        caught_ct = True
    assert_true(caught_ct)


def test_studio_editors_compare_and_room(tmp_dir: String) raises:
    # 1. CombatTimeline [MEDIUM-2] zero-span / zero-lane / valid drag & nudge
    var ct_path = tmp_dir + "/tl.combat.json"
    _write_text(ct_path, '{"version":1,"fps":60,"end":20,"impact":5,"close":10,"chain":15,"reach":2.0}\n')
    var ct = CombatTiming(ct_path)
    var zero_tl = CombatTimeline(10, 10, 50, 0)
    assert_equal(zero_tl.x(ct, 5.0), 10)
    var inp = InputState()
    inp.begin_frame()
    inp.move_mouse(12.0, 50.0)
    inp.set_mouse(0, True)
    assert_equal(zero_tl.input(ct, inp, 12, 50), -1)
    assert_false(zero_tl.changed)
    assert_equal(ct.impact, 5)
    inp.set_mouse(0, False)

    var valid_tl = CombatTimeline(100, 300, 40, 20)
    assert_equal(valid_tl.x(ct, 10.0), 200)
    inp.begin_frame()
    inp.move_mouse(160.0, 40.0)
    inp.set_mouse(0, True)
    var preview = valid_tl.input(ct, inp, 160, 40)
    assert_equal(preview, 6)
    assert_true(valid_tl.changed)
    assert_equal(ct.impact, 6)
    inp.set_mouse(0, False)

    # 2. profile_compare.Capture [MEDIUM-1] valid, missing traceEvents, OOB at(), incomplete error
    var cap_path = tmp_dir + "/trace.json"
    _write_text(
        cap_path,
        '{"traceEvents":[\n'
        '  {"name":"sim","ph":"X","ts":100,"dur":500},\n'
        '  {"name":"frame","ph":"X","ts":100,"dur":4000,"args":{"frame":0}},\n'
        '  {"name":"frame","ph":"X","ts":5000,"dur":6000,"args":{"frame":1}}\n'
        ']}\n',
    )
    var cap = Capture(cap_path)
    assert_equal(len(cap.starts), 2)
    assert_true(abs(cap.at("frame", 0) - 4.0) < 1e-6)
    assert_true(abs(cap.at("sim", 0) - 0.5) < 1e-6)
    assert_equal(cap.at("frame", -1), Float64(-1.0))
    assert_equal(cap.at("frame", 99), Float64(-1.0))
    assert_equal(cap.at("missing", 0), Float64(-1.0))
    var st = cap.stats("frame")
    assert_equal(st[2], 2)
    assert_true(abs(st[0] - 5.0) < 1e-6)

    var bad_cap_path = tmp_dir + "/bad_trace.json"
    _write_text(bad_cap_path, '{"traceEvents":[]}')
    var caught_cap = False
    try:
        _ = Capture(bad_cap_path)
    except:
        caught_cap = True
    assert_true(caught_cap)

    # 3. TraversalRoom parse, ground_point, save, save_copy, invalid rejection
    var room_path = tmp_dir + "/encounter.room.json"
    _write_text(
        room_path,
        '{"version":1,"motion":"ledge.motion.json","combat":"slash.combat.json",'
        '"lip":[1.0,2.5,0.0],"yaw":90.0,"landing":[0.0,0.0,1.0],"takeoff":[0.0,0.0,-2.0],"defender":[0.5,0.0,1.5]}\n',
    )
    var room = TraversalRoom(room_path)
    assert_true(abs(room.ground_point(Vec3(0.0, 0.3, 1.0)).y - Float32(0.3)) < Float32(1e-5))
    var room_save = tmp_dir + "/saved.room.json"
    room.save(room_save)
    var room2 = TraversalRoom(room_save)
    assert_equal(room2.motion, "ledge.motion.json")
    var copy_path = room2.save_copy(tmp_dir + "/copy_room")
    assert_true(copy_path.endswith("-2.room.json"))
    var room3 = TraversalRoom(copy_path)
    assert_equal(room3.combat, "slash.combat.json")

    # 4. motion_editor helpers (relative_to, next_clip, timing_click, FollowCamera, point, frame)
    assert_equal(relative_to("/a/b/p.json", "c.glb"), "/a/b/c.glb")
    assert_equal(relative_to("/a/b/p.json", "/abs.glb"), "/abs.glb")
    var model = _make_test_model(Float32(1.0), Float32(0.8))
    assert_equal(next_clip(model, 0), 1)
    assert_equal(len(timing_click().samples), 1200)
    var seq_path = tmp_dir + "/me.motion.json"
    _write_text(
        seq_path,
        '{"version":1,"model":"hero.glb","first":"jump","second":"grab","contact":"hand_r",'
        '"markers":["hand_r"],"support":"","overlap":0.25,"scale":1.0,"start":[0,0,-1],"ledge":[0,2,0],'
        '"yaw":0,"cameraStart":[2,2,-3],"cameraEnd":[2,2.5,-1],"cameraAim":[0,1.5,0],"cue":0.5,"sound":""}\n',
    )
    var seq = MotionSequence(seq_path)
    var player = MotionPlayer(model, seq)
    var cam = FollowCamera.view(Placement(Vec3(0.0, 0.0, 0.0)), Float32(0.5))
    var pt = point(SimdCanvas(960, 640), cam, Vec3(0.0, 1.0, 0.0))
    assert_true(pt[0] >= 0)
    var canvas = SimdCanvas(960, 640)
    frame[FollowCamera](canvas, model, seq, player, Float32(0.5), False, "TEST", True)
    var non_bg = 0
    for px in canvas.pixels:
        if px != canvas.pixels[0]:
            non_bg += 1
    assert_true(non_bg > 100)

    # 5. IdePty move & stop guards
    var p1 = IdePty()
    p1.fd = -1
    p1.pid = 0
    var p2 = p1^
    assert_equal(p2.fd, -1)
    p2.resize(80, 24)
    p2.stop()


def test_studio_command_menu() raises:
    var menu = CommandMenu()
    menu.add_group("File", 64)
    menu.item(1, "Unavailable", enabled=False)
    menu.item(2, "Open")
    menu.add_group("Rooms", 80)
    for i in range(15):
        menu.item(100+i, "Room "+String(i))
    menu.y = 552
    var inp = InputState()
    inp.set_key(299, True)
    assert_equal(menu.input(inp, 0, 0), 0)
    assert_equal(menu.row, 1)
    inp.begin_frame()
    inp.set_key(262, True)
    _ = menu.input(inp, 0, 0)
    assert_equal(menu.group, 1)
    assert_true(menu.popup_y()+menu.rows()*32 <= menu.y)
    for _ in range(14):
        inp.begin_frame()
        inp.set_key(264, False)
        inp.set_key(264, True)
        _ = menu.input(inp, 0, 0)
    assert_equal(menu.row, 14)
    assert_equal(menu.first, 5)
    inp.begin_frame()
    inp.set_key(257, True)
    assert_equal(menu.input(inp, 0, 0), 114)
    menu.open(0)
    inp.begin_frame()
    inp.set_mouse(0, True)
    assert_equal(menu.input(inp, 900, 10), 0)
    assert_equal(menu.group, -1)
    inp.begin_frame()
    inp.set_mouse(0, False)
    assert_equal(menu.input(inp, 900, 10), 0)
    inp.begin_frame()
    assert_equal(menu.input(inp, 900, 10), -1)

    # [R2-A1] CommandMenu.draw resize-after-scroll & empty groups guard
    var empty_menu = CommandMenu()
    assert_equal(empty_menu.rows(), 0)
    var ui_empty = UiDrawList()
    var font_empty = SdfFont()
    empty_menu.draw(ui_empty, font_empty, 0)

    var m_resize = CommandMenu()
    m_resize.add_group("Items", 80)
    for i in range(5):
        m_resize.item(200 + i, "Item " + String(i))
    m_resize.y = 32
    m_resize.height = 26
    m_resize.screen_h = 120  # rows() = 1
    m_resize.open(0)
    assert_equal(m_resize.rows(), 1)
    var r_inp = InputState()
    for _ in range(4):
        r_inp.begin_frame()
        r_inp.set_key(264, False)
        r_inp.set_key(264, True)
        _ = m_resize.input(r_inp, 0, 0)
    assert_equal(m_resize.first, 4)
    m_resize.screen_h = 720  # window grew -> rows() = 5, first + rows = 9 > 5
    assert_equal(m_resize.rows(), 5)
    var ui_resize = UiDrawList()
    m_resize.draw(ui_resize, font_empty, 0)
    assert_true(len(ui_resize.prims) > 0)


def _open_and_drop_scene_menu() raises -> Int32:
    var sm = SceneMenu()
    assert_true(sm.has_project)
    sm.open_library()
    var pid_drop = sm.library_pid
    assert_true(pid_drop > 0)
    assert_equal(external_call["kill", Int32](pid_drop, Int32(0)), Int32(0))
    # Double open_library must not spawn a second child
    sm.open_library()
    assert_equal(sm.library_pid, pid_drop)
    # Move preserves single-owner cleanup
    var sm_moved = sm^
    assert_equal(sm_moved.library_pid, pid_drop)
    _ = sm_moved^
    return pid_drop


def test_studio_motion_tools_and_scene_menu(tmp_dir: String) raises:
    # 1. [R2-A2] MotionTools.pin / frame_joint with empty player.markers & valid markers
    var empty_motion_path = tmp_dir + "/empty_markers.motion.json"
    _write_text(
        empty_motion_path,
        '{\n'
        '  "version": 1,\n'
        '  "model": "hero.glb",\n'
        '  "first": "jump",\n'
        '  "second": "grab",\n'
        '  "contact": "",\n'
        '  "markers": [],\n'
        '  "support": "",\n'
        '  "overlap": 0.25,\n'
        '  "scale": 1.0,\n'
        '  "start": [0.0, 0.0, -1.0],\n'
        '  "ledge": [0.0, 2.0, 0.0],\n'
        '  "yaw": 0.0,\n'
        '  "cameraStart": [2.0, 2.0, -3.0],\n'
        '  "cameraEnd": [2.0, 2.5, -1.0],\n'
        '  "cameraAim": [0.0, 1.5, 0.0],\n'
        '  "cue": 0.5,\n'
        '  "sound": ""\n'
        '}\n',
    )
    var seq_empty = MotionSequence(empty_motion_path)
    var model = _make_test_model(Float32(1.0), Float32(0.8))
    var player_empty = MotionPlayer(model, seq_empty)
    assert_equal(len(player_empty.markers), 0)
    var mt = MotionTools()
    mt.tab = CONTACT
    mt.pin(model, seq_empty, player_empty, 0, Vec3(0.0, 2.0, 0.0))
    assert_false(mt.pinned)
    _ = frame_joint(player_empty, model, seq_empty, 0, 0)
    _ = frame_joint(player_empty, model, seq_empty, 0, -1)
    var canvas = SimdCanvas(960, 640)
    mt.hud(canvas, model, seq_empty, player_empty, 0, 480, "OK")
    mt.tab = TIMING
    mt.hud(canvas, model, seq_empty, player_empty, 0, 480, "OK")
    seq_empty.set_event(3, 3)
    assert_true(shake_offset(seq_empty, Float32(3) / Float32(30) + Float32(0.02)).length() >= Float32(0.0))
    var cam = Camera3D()
    cam.eye = Vec3(0.0, 1.0, -5.0)
    cam.basis_f = Vec3(0.0, 0.0, 1.0)
    cam.basis_r = Vec3(1.0, 0.0, 0.0)
    cam.basis_u = Vec3(0.0, 1.0, 0.0)
    cam.focal_px = Float32(500.0)
    cam.screen_w = 960
    cam.screen_h = 640
    mt.view = cam
    mt.shown_joint = Vec3(0.0, 1.0, 0.0)
    _ = mt.clicked_point(480, 320)
    draw_dust(canvas, cam, Vec3(0.0, 1.0, 0.0), Float32(0.1))

    # 2. [R2-A3] SceneMenu.open_library child PID cleanup on drop, move & FlintStudioState.shutdown()
    var proj_dir = tmp_dir + "/proj"
    assert_true(make_dirs(proj_dir))
    _write_text(proj_dir + "/flint.toml", '[project]\nname = "t5"\n')
    var launcher_path = tmp_dir + "/launcher.sh"
    _write_text(launcher_path, "#!/bin/bash\nsleep 30\n")
    var lp_owned = launcher_path
    _ = external_call["chmod", Int32](lp_owned.as_c_string_span(), Int32(0o755))

    var k_proj = String("FLINT_STUDIO_PROJECT")
    var k_launch = String("FLINT_STUDIO_IDE_LAUNCHER")
    var v_proj = proj_dir
    var v_launch = launcher_path
    _ = external_call["setenv", Int32](k_proj.as_c_string_span(), v_proj.as_c_string_span(), Int32(1))
    _ = external_call["setenv", Int32](k_launch.as_c_string_span(), v_launch.as_c_string_span(), Int32(1))

    # 2a. SceneMenu drop kills & reaps child
    var pid_drop = _open_and_drop_scene_menu()
    assert_true(external_call["kill", Int32](pid_drop, Int32(0)) != Int32(0))

    # 2b. FlintStudioState.shutdown() closes menu child cleanly
    var studio = FlintStudioState(headless=True)
    assert_true(studio.menu.has_project)
    studio.menu.open_library()
    var pid_studio = studio.menu.library_pid
    assert_true(pid_studio > 0)
    assert_equal(external_call["kill", Int32](pid_studio, Int32(0)), Int32(0))
    studio.shutdown()
    assert_equal(studio.menu.library_pid, Int32(-1))
    assert_true(external_call["kill", Int32](pid_studio, Int32(0)) != Int32(0))

    _ = external_call["unsetenv", Int32](k_proj.as_c_string_span())
    _ = external_call["unsetenv", Int32](k_launch.as_c_string_span())
    _ = remove_file(proj_dir + "/flint.toml")
    _rm_dir_shallow(proj_dir)


def test_dockspace_codecs_watcher_and_qoa() raises:
    # 1. [R2-A4] DockSpace.load_json rejects cyclic split graphs without corrupting valid state
    var dock = DockSpace.create_studio_5panel()
    var prev_count = len(dock.nodes)
    assert_true(prev_count > 1)
    assert_false(
        dock.load_json(
            '{"nodes":[{"kind":1,"ratio":0.5,"child_a":1,"child_b":0,"tabs":[]},'
            '{"kind":1,"ratio":0.5,"child_a":0,"child_b":1,"tabs":[]}]}'
        )
    )
    assert_equal(len(dock.nodes), prev_count)
    assert_false(dock.load_json('{"root":0,"nodes":[{"kind":1,"ratio":0.5,"child_a":0,"child_b":0,"tabs":[]}]}'))
    dock.solve_layout(RectF(0.0, 0.0, 1280.0, 720.0))
    assert_true(len(dock.leaf_rects()) > 0)

    # 2. [R2-A5] decode_png & decode_jpeg maximum_size <= 0 must not overflow or hang
    var px: List[UInt32] = [
        0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF,
        0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF,
        0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF,
        0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF,
    ]
    var png_bytes = encode_png(px, 4, 4)
    var png0 = decode_png(png_bytes, maximum_size=0)
    assert_true(png0.valid and png0.width == 1 and png0.height == 1)
    var png_neg = decode_png(png_bytes, maximum_size=-1)
    assert_true(png_neg.valid and png_neg.width == 1 and png_neg.height == 1)

    var jpg_bytes = encode_jpeg_baseline(px, 4, 4, 85)
    var jpg0 = decode_jpeg(jpg_bytes, maximum_size=0)
    assert_true(jpg0.valid and jpg0.width == 1 and jpg0.height == 1)
    var jpg_neg = decode_jpeg(jpg_bytes, maximum_size=-1)
    assert_true(jpg_neg.valid and jpg_neg.width == 1 and jpg_neg.height == 1)

    # 3. [R2-A6] AssetWatcher._format_event_path with root_prefix == "/"
    var watcher = AssetWatcher()
    assert_equal(watcher._format_event_path("/textures", "/", "hero.png"), "textures/hero.png")
    assert_equal(watcher._format_event_path("/a/b", "/", "c.png"), "a/b/c.png")
    assert_equal(watcher._format_event_path("/", "/", "root.png"), "root.png")
    assert_equal(watcher._format_event_path("/textures", "/tex", "hero.png"), "/textures/hero.png")

    # 4. [R2-B8] QoaStreamCursor._load_frame_header rejects undersized fsize
    var pcm = List[Int16](length=20, fill=Int16(1000))
    var qoa_good = encode_qoa(pcm, 1, 44100)
    var cur_good = QoaStreamCursor()
    assert_true(cur_good.open(qoa_good))

    var qoa_bad_small = qoa_good.copy()
    qoa_bad_small[14] = UInt8(0)
    qoa_bad_small[15] = UInt8(1)  # fsize = 1 < 8 + 16 = 24
    var cur_bad1 = QoaStreamCursor()
    assert_false(cur_bad1.open(qoa_bad_small))

    var qoa_bad_noslice = qoa_good.copy()
    qoa_bad_noslice[14] = UInt8(0)
    qoa_bad_noslice[15] = UInt8(24)  # fsize = 24 (header only, excludes slice bytes)
    var cur_bad2 = QoaStreamCursor()
    assert_false(cur_bad2.open(qoa_bad_noslice))


def test_core_crash_and_log() raises:
    var sigs = crash_signals()
    assert_equal(len(sigs), 6)
    assert_equal(sigs[0], SIGSEGV)
    assert_equal(sigs[5], SIGABRT)
    var rt_buf = List[UInt8](length=128, fill=0)
    var rt = RawText(Int(rt_buf.unsafe_ptr()), len(rt_buf))
    write_signal_name(rt, SIGBUS)
    write_signal_name(rt, SIGFPE)
    write_signal_name(rt, SIGILL)
    write_signal_name(rt, SIGTRAP)
    assert_true(rt.length > 10)
    _ = len(rt_buf)
    assert_equal(hex_digit(0), "0")
    assert_equal(hex_digit(15), "f")
    var wbytes: List[UInt8] = [UInt8(0x78), UInt8(0x56), UInt8(0x34), UInt8(0x12)]
    assert_equal(le_word(wbytes, 0, 4), 0x12345678)
    var empty_elf: List[UInt8] = [UInt8(0), UInt8(1), UInt8(2)]
    assert_equal(elf_build_id(empty_elf), "")
    assert_true(executable_path().byte_length() > 0)

    assert_equal(level_name(LOG_INFO), "info")
    assert_equal(level_name(LOG_WARN), "warn")
    assert_equal(level_name(LOG_OFF), "off")
    assert_equal(level_letter(LOG_TRACE), 84)
    assert_equal(level_letter(LOG_ERROR), 69)
    assert_equal(category_name(LOG_CORE), "core")
    assert_equal(category_name(LOG_ASSETS), "assets")
    assert_equal(parse_level("warning"), LOG_WARN)
    assert_equal(parse_level("follow"), LOG_FOLLOW)
    assert_equal(parse_level("3"), LOG_WARN)
    assert_equal(parse_level("bogus"), -1)
    assert_equal(find_category("core"), LOG_CORE)
    assert_equal(find_category("nonexistent"), -1)
    assert_true(log_enabled(LOG_CORE, LOG_ERROR))


def test_physics_defaults_density_and_wind() raises:
    assert_equal(PHYSICS_SUBSTEPS, 4)
    assert_true(CONTACT_HERTZ > Float32(0.0) and CONTACT_DAMPING_RATIO > Float32(0.0))
    assert_true(JOINT_HERTZ > Float32(0.0) and RESTITUTION_THRESHOLD > Float32(0.0))
    assert_true(ROLLING_RESISTANCE > Float32(0.0) and BODY_MAX_SPEED > Float32(100.0))
    assert_true(WATER_ANGULAR_DRAG > Float32(0.0))

    assert_equal(DENSITY_MATERIALS, 25)
    assert_equal(material_density_milli(0), 0)
    assert_equal(material_density_milli(4), 2500)
    assert_equal(material_density_milli(999), 0)
    assert_equal(material_rolling_factor(4), Float32(1.0))
    assert_true(material_wind_drag(14) > material_wind_drag(4))

    assert_equal(to_fx(1.5), (3 * FX_ONE) // 2)
    assert_equal(fx_mul(2 * FX_ONE, 3 * FX_ONE), 6 * FX_ONE)
    assert_equal(fx_div(FX_ONE, 0), 0)
    assert_equal(isqrt(144), 12)
    assert_equal(falloff_fx(FALLOFF_LINEAR, FX_ONE // 2), FX_ONE // 2)
    assert_equal(falloff_fx(FALLOFF_SMOOTH, FX_ONE), 0)

    var wf = WindField()
    assert_true(wf.is_calm())
    var ws = WindSource()
    ws.id = 1
    ws.shape.shape = SHAPE_SPHERE
    ws.shape.falloff = FALLOFF_LINEAR
    ws.shape.center = FxVec.zero()
    ws.shape.radius = 10 * FX_ONE
    ws.base = FxVec.of(Vec3(4.0, 0.0, 0.0))
    ws.gust = FX_ONE // 4
    wf.sources.append(ws)
    var vort = make_vorticle(7, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 6.0, 2.0, 3.0, 5)
    wf.add_vorticle(vort)
    assert_false(wf.is_calm())
    var v_eval = wf.at_vec(Vec3(2.0, 0.0, 0.0), 10)
    assert_true(v_eval.length() > Float32(0.1))

    var bw = BlobWriter()
    vort.write(bw)
    var br = BlobReader(bw.data.copy())
    var vort2 = Vorticle.read(br)
    assert_equal(vort2.id, 7)
    for _ in range(6):
        wf.age()
    assert_equal(len(wf.vorticles), 0)
    var box_s = FxShape()
    box_s.shape = SHAPE_BOX
    box_s.half = FxVec.of(Vec3(2.0, 2.0, 2.0))
    assert_true(box_s.weight(FxVec.zero()) == FX_ONE)
    var cap_s = FxShape()
    cap_s.shape = SHAPE_CAPSULE
    cap_s.end = FxVec.of(Vec3(0.0, 4.0, 0.0))
    cap_s.radius = 2 * FX_ONE
    assert_true(cap_s.weight(FxVec.of(Vec3(0.0, 2.0, 0.0))) == FX_ONE)


def test_render_dither_lighting_layouts_and_particle_atlas() raises:
    # 1. dither
    assert_equal(DITHER_LEVELS, 16)
    assert_equal(bayer4(0, 0), 0)
    assert_true(0 <= bayer4(3, 2) < 16)
    var dv = dither_view(
        Vec3(0.0, 0.0, -5.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Float32(500.0),
        1280,
        720,
        Vec3(0.0, 0.0, 0.0),
        Float32(1.0),
        Float32(0.2),
        Float32(0.25),
    )
    assert_true(dv.active())
    assert_true(dv.offset < DITHER_OFF_OFFSET and dv.near_inverse_z < DITHER_OFF_INVERSE_Z)
    assert_false(dv.drops(640, 360, Float32(0.01)))

    # 2. gpu_layout & particle_layout
    assert_equal(TILE_SIZE, 32)
    assert_equal(RECORD_BYTES, RECORD_FLOATS * 4)
    assert_true(DITHER_FLAG != 0)
    comptime assert CONSTANT_DITHER < CONSTANT_EFFECTS
    comptime assert CONSTANT_EFFECTS < CONSTANT_CAMERA and CONSTANT_BYTES >= 16384
    assert_equal(PARTICLE_WORDS, 16)
    assert_equal(STATE_HEADER_WORDS, 64)
    assert_equal(EMITTER_WORDS, 32)
    assert_equal(MAX_EMITTERS, 256)
    assert_equal(ATLAS_BYTES, ATLAS_FRAMES * ATLAS_FRAME * ATLAS_FRAME)

    # 3. lighting
    assert_equal(MAX_REGISTERED_LIGHTS, 256)
    assert_equal(MAX_EVENT_LIGHTS, 64)
    assert_equal(MAX_SELECTED_LIGHTS, 128)
    var lp = LightingProfile()
    assert_equal(lp.fog_rgb()[0], UInt8(14))
    for r in range(3):
        var rp = regional_profile(r)
        assert_true(rp.exposure > Float32(1.0))
    var elp = EventLightPool()
    elp.renew(Vec3(1.0, 2.0, 3.0), Vec3(1.0, 0.5, 0.2), Float32(4.0), Float32(2.0), Float32(1.0), 42, 5)
    elp.emit_tube(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0), Vec3(0.2, 0.8, 1.0), Float32(3.0), Float32(1.5), Float32(0.5))
    assert_equal(elp.emitted, 2)
    elp.update(Float32(0.6))

    # 4. occupancy_layout
    assert_equal(CHUNK_SIZE, 32)
    var occ = OccupancyLayout(64, 64, 64)
    assert_equal(occ.chunks_x, 2)
    assert_equal(occ.bricks_x, 8)
    assert_true(occ.bytes() > OCC_HEADER_WORDS * 4)
    assert_true(occ.brick_index(1, 1, 1) > 0)
    assert_true(occ.chunk_index(1, 1, 1) > 0)
    var hdr_words = List[UInt32](length=OCC_HEADER_WORDS, fill=0)
    write_occupancy_header(Int(hdr_words.unsafe_ptr()), occ)
    assert_equal(Int(hdr_words[0]), occ.bricks_x)
    assert_equal(Int(hdr_words[10]), occ.total_words)

    # 5. particle_atlas
    assert_true(Float32(0.0) <= atlas_lattice(3, 5, 7) <= Float32(1.0))
    assert_true(Float32(0.0) <= atlas_value_noise(Float32(1.25), Float32(2.5), 11) <= Float32(1.0))
    assert_equal(atlas_to_byte(Float32(0.5)), UInt8(128))
    var atlas = particle_atlas()
    assert_equal(len(atlas), ATLAS_BYTES)
    var nonzero = 0
    for b in atlas:
        if b > 0:
            nonzero += 1
    assert_true(nonzero > ATLAS_BYTES // 10)


def _put_be16(mut buf: List[UInt8], val: Int):
    buf.append(UInt8((val >> 8) & 0xFF))
    buf.append(UInt8(val & 0xFF))


def _put_be32(mut buf: List[UInt8], val: Int):
    buf.append(UInt8((val >> 24) & 0xFF))
    buf.append(UInt8((val >> 16) & 0xFF))
    buf.append(UInt8((val >> 8) & 0xFF))
    buf.append(UInt8(val & 0xFF))


def _build_nonmonotonic_ttf() -> List[UInt8]:
    """Minimal SFNT where glyph 1 ('A') has non-monotonic endPtsOfContours=[10, 2]."""
    var head = List[UInt8](length=54, fill=0)
    head[18] = UInt8(0x03); head[19] = UInt8(0xE8)  # unitsPerEm = 1000
    head[50] = UInt8(0); head[51] = UInt8(0)        # indexToLocFormat = 0 (short)

    var hhea = List[UInt8](length=36, fill=0)
    hhea[4] = UInt8(0x03); hhea[5] = UInt8(0x20)    # ascender = 800
    hhea[6] = UInt8(0xFF); hhea[7] = UInt8(0x38)    # descender = -200
    hhea[34] = UInt8(0); hhea[35] = UInt8(2)        # numOfLongHorMetrics = 2

    var maxp = List[UInt8](length=6, fill=0)
    maxp[4] = UInt8(0); maxp[5] = UInt8(2)          # numGlyphs = 2

    var hmtx = List[UInt8](length=8, fill=0)
    hmtx[0] = UInt8(0x02); hmtx[1] = UInt8(0x00)    # glyph 0 adv = 512
    hmtx[4] = UInt8(0x02); hmtx[5] = UInt8(0x00)    # glyph 1 adv = 512

    # cmap format 4 mapping codepoint 65 ('A') -> glyph 1
    var cmap = List[UInt8]()
    _put_be16(cmap, 0)       # version
    _put_be16(cmap, 1)       # numTables
    _put_be16(cmap, 3)       # platformID = Windows
    _put_be16(cmap, 1)       # encodingID = Unicode BMP
    _put_be32(cmap, 12)      # subtable offset
    _put_be16(cmap, 4)       # format = 4
    _put_be16(cmap, 32)      # length
    _put_be16(cmap, 0)       # language
    _put_be16(cmap, 4)       # segCountX2 = 4 (2 segments)
    _put_be16(cmap, 4)       # searchRange
    _put_be16(cmap, 1)       # entrySelector
    _put_be16(cmap, 0)       # rangeShift
    _put_be16(cmap, 65)      # endCode[0]
    _put_be16(cmap, 0xFFFF)  # endCode[1]
    _put_be16(cmap, 0)       # reservedPad
    _put_be16(cmap, 65)      # startCode[0]
    _put_be16(cmap, 0xFFFF)  # startCode[1]
    _put_be16(cmap, 0xFFC0)  # idDelta[0] = -64 (65 + (-64) = 1)
    _put_be16(cmap, 1)       # idDelta[1]
    _put_be16(cmap, 0)       # idRangeOffset[0]
    _put_be16(cmap, 0)       # idRangeOffset[1]

    # glyf: glyph 0 empty, glyph 1 simple with numberOfContours=2, endPts=[10, 2], 3 pts
    var glyf = List[UInt8]()
    _put_be16(glyf, 2)       # numberOfContours = 2
    _put_be16(glyf, 0)       # xMin
    _put_be16(glyf, 0)       # yMin
    _put_be16(glyf, 500)     # xMax
    _put_be16(glyf, 500)     # yMax
    _put_be16(glyf, 10)      # endPtsOfContours[0] = 10
    _put_be16(glyf, 2)       # endPtsOfContours[1] = 2 (NON-MONOTONIC!)
    _put_be16(glyf, 0)       # instructionLength = 0
    for _ in range(3):
        glyf.append(UInt8(0x01))  # on-curve, dx=0, dy=0
    while len(glyf) % 2 != 0:
        glyf.append(UInt8(0))

    # loca (short format, 3 offsets for 2 glyphs): [0, 0, len(glyf)//2]
    var loca = List[UInt8]()
    _put_be16(loca, 0)
    _put_be16(loca, 0)
    _put_be16(loca, len(glyf) // 2)

    var tags = [String("cmap"), String("glyf"), String("head"), String("hhea"), String("hmtx"), String("loca"), String("maxp")]
    var blobs = [cmap^, glyf^, head^, hhea^, hmtx^, loca^, maxp^]

    var out = List[UInt8]()
    _put_be32(out, 0x00010000)
    _put_be16(out, len(tags))
    _put_be16(out, 64)
    _put_be16(out, 2)
    _put_be16(out, 48)
    var cur_off = 12 + len(tags) * 16
    for i in range(len(tags)):
        var tbytes = tags[i].as_bytes()
        for j in range(4):
            out.append(tbytes[j])
        _put_be32(out, 0)
        _put_be32(out, cur_off)
        _put_be32(out, len(blobs[i]))
        cur_off += (len(blobs[i]) + 3) & ~3
    for i in range(len(blobs)):
        for b in blobs[i]:
            out.append(b)
        while len(out) % 4 != 0:
            out.append(UInt8(0))
    return out^


def _test_job_inc(context: Int, chunk: Int):
    _ = chunk
    var ptr = Pointer[Int, origin=MutAnyOrigin](unsafe_from_address=context)
    ptr[unsafe_offset=0] += 1


def test_r3_partition_a_fixes() raises:
    # A1: TTF non-monotonic endPtsOfContours + sdf_bake flatten/extract_edge_contours guards
    var ttf_bytes = _build_nonmonotonic_ttf()
    var font = TrueTypeFont()
    assert_true(font.load_bytes(ttf_bytes^))
    assert_true(font.valid)
    assert_equal(font.glyph_id(65), 1)
    var bad_out = font.outline(1)
    assert_equal(len(bad_out.ends), 0)
    assert_equal(len(bad_out.xs), 0)
    var bg1 = bake_glyph(font, 65, 24, 3)
    var bg2 = bake_glyph_msdf(font, 65, 24, 3)
    assert_equal(bg1.w, 0)
    assert_equal(bg2.w, 0)

    var manual_out = GlyphOutline()
    manual_out.xs = [Float32(0.0), Float32(10.0), Float32(10.0)]
    manual_out.ys = [Float32(0.0), Float32(0.0), Float32(10.0)]
    manual_out.on = [True, True, True]
    manual_out.ends = [10, 2, -5]
    var segs = flatten(manual_out, Float32(1.0))
    var ec = extract_edge_contours(manual_out, Float32(1.0))
    assert_equal(len(segs.values), 0)
    assert_equal(len(ec), 0)

    # A2: FlintStudioState.deserialize bounded n_sel & n_links vs remaining blob bytes
    var studio = FlintStudioState(headless=True)
    var bw_sel = BlobWriter()
    bw_sel.begin(fourcc("STDV"), 1)
    bw_sel.integer(0)
    bw_sel.u32(UInt32(0x10000000))  # claims 268M selected IDs in tiny blob
    bw_sel.end()
    var br_sel = BlobReader(bw_sel.data.copy())
    var sel_raised = False
    try:
        studio.deserialize(br_sel)
    except:
        sel_raised = True
    assert_true(sel_raised)

    var bw_lnk = BlobWriter()
    bw_lnk.begin(fourcc("STDV"), 1)
    bw_lnk.integer(-1)
    bw_lnk.u32(UInt32(0))  # n_sel = 0
    bw_lnk.integer(-1)
    bw_lnk.integer(0)
    bw_lnk.vec3(Vec3(0.0, -9.81, 0.0))
    bw_lnk.f32(Float32(0.0))
    bw_lnk.boolean(False)
    bw_lnk.integer(-1)
    bw_lnk.f32(Float32(0.0))
    bw_lnk.f32(Float32(0.0))
    bw_lnk.f32(Float32(0.0))
    bw_lnk.f32(Float32(0.0))
    bw_lnk.boolean(False)
    bw_lnk.f32(Float32(0.0))
    bw_lnk.boolean(False)
    bw_lnk.boolean(False)
    bw_lnk.f32(Float32(0.0))
    bw_lnk.u32(UInt32(0))  # w_bytes len = 0
    bw_lnk.u32(UInt32(0))  # p_bytes len = 0
    bw_lnk.u32(UInt32(0x10000000))  # claims 268M runtime links
    bw_lnk.end()
    var br_lnk = BlobReader(bw_lnk.data.copy())
    var lnk_raised = False
    try:
        studio.deserialize(br_lnk)
    except:
        lnk_raised = True
    assert_true(lnk_raised)

    # A3: ide_view.raster_ui skips zero-extent PRIM_GLYPH (prevents 0.0/0.0 NaN UV bypass)
    var ui = UiDrawList()
    var bg_prim = GpuUiPrim()
    bg_prim.header = PRIM_SHAPE
    bg_prim.x0 = Float32(0.0)
    bg_prim.y0 = Float32(0.0)
    bg_prim.x1 = Float32(20.0)
    bg_prim.y1 = Float32(20.0)
    bg_prim.fill_rgba8 = 0x00302010  # ui_color -> 0xFF102030
    ui.prims.append(bg_prim)
    var z_glyph = GpuUiPrim()
    z_glyph.header = PRIM_GLYPH
    z_glyph.x0 = Float32(9.5)
    z_glyph.y0 = Float32(9.5)
    z_glyph.x1 = Float32(9.5)
    z_glyph.y1 = Float32(9.5)
    z_glyph.fill_rgba8 = 0xFFFFFFFF
    z_glyph.param0 = Float32(1.0)
    ui.prims.append(z_glyph)
    var pixels = List[UInt32]()
    var sdf_f = SdfFont()
    raster_ui(pixels, sdf_f, ui, 20, 20)
    assert_equal(len(pixels), 400)
    assert_equal(pixels[9 * 20 + 9], UInt32(0xFF102030))


def test_r3_partition_b_fixes(tmp_dir: String) raises:
    # B1: JobPool null-address safety before start() and after stop()
    var jp = JobPool()
    assert_false(jp.allocated())
    assert_false(jp.ready())
    assert_true(jp.stopping())
    assert_equal(jp.jobs_completed(), 0)
    assert_equal(jp.jobs_helped(), 0)
    var j_unalloc = jp.create(_test_job_inc, 0)
    assert_equal(j_unalloc, NO_JOB)
    jp.depend(j_unalloc, j_unalloc)
    jp.submit(j_unalloc)
    jp.wait(j_unalloc)
    assert_true(jp.finished(j_unalloc))
    assert_false(jp.help())
    assert_equal(jp.claim(j_unalloc), -1)
    assert_false(jp.work_on(j_unalloc, False))
    jp.release(j_unalloc)
    jp.complete(j_unalloc)
    jp.lock(0)
    jp.unlock(0)
    jp.rouse()
    jp.sleep()
    jp.futex_wake(1)
    var counter = 0
    jp.run(_test_job_inc, Int(Pointer(to=counter)), 3)
    assert_equal(counter, 3)
    var c_id = jp.chain(_test_job_inc, Int(Pointer(to=counter)), 2)
    assert_equal(c_id, NO_JOB)
    assert_equal(counter, 5)
    assert_true(jp.start(1))
    jp.stop()
    assert_false(jp.allocated())
    assert_true(jp.stopping())
    assert_equal(jp.jobs_completed(), 0)
    assert_equal(jp.jobs_helped(), 0)
    assert_equal(jp.create(_test_job_inc, 0), NO_JOB)

    # B2: lz4_decompress negative & zero raw_size
    var src_bytes = List[UInt8]()
    for i in range(16):
        src_bytes.append(UInt8(65 + i))
    var blk = lz4_compress(src_bytes)
    assert_equal(len(lz4_decompress(blk, -1)), 0)
    assert_equal(len(lz4_decompress(blk, -5)), 0)
    assert_equal(len(lz4_decompress(blk, 0)), 0)
    assert_equal(len(lz4_decompress(blk, 16)), 16)

    # B3: Pattern K_ALT / K_RANDCAT / K_ALT_ABS / K_SEQ empty child lists & negative cycles
    var pat = Pattern()
    var r_slow_empty = pat.slowcat(List[Int]())
    var r_rand_empty = pat.randcat(List[Int]())
    var r_alt_abs_empty = pat.with_kids(K_ALT_ABS, List[Int](), List[Frac]())
    var r_seq_empty = pat.seq(List[Int]())
    assert_equal(len(pat.query_node(r_slow_empty, Frac(0, 1), Frac(2, 1))), 0)
    assert_equal(len(pat.query_node(r_rand_empty, Frac(0, 1), Frac(2, 1))), 0)
    assert_equal(len(pat.query_node(r_alt_abs_empty, Frac(0, 1), Frac(2, 1))), 0)
    assert_equal(len(pat.query_node(r_seq_empty, Frac(0, 1), Frac(2, 1))), 0)

    var p_bd = Params()
    p_bd.set_sound("bd")
    var p_sd = Params()
    p_sd.set_sound("sd")
    var kids2 = List[Int]()
    kids2.append(pat.atom(p_bd^))
    kids2.append(pat.atom(p_sd^))
    var r_slow_2 = pat.slowcat(kids2)
    var r_rand_2 = pat.randcat(kids2)
    assert_true(len(pat.query_node(r_slow_2, Frac(-2, 1), Frac(1, 1))) > 0)
    assert_true(len(pat.query_node(r_rand_2, Frac(-2, 1), Frac(1, 1))) > 0)

    # B4: CyclesEngine render / render_i16 / render_to_wav / set_cps / render_loop_to_wav
    var eng = CyclesEngine(48000)
    assert_equal(len(eng.render(0)), 0)
    assert_equal(len(eng.render(-1)), 0)
    assert_equal(len(eng.render_i16(0)), 0)
    assert_equal(len(eng.render_i16(-5)), 0)
    eng.set_cps(1.0e-8)
    assert_true(eng.cps_frac.num > 0)
    eng.set_cps(-2.0)
    assert_true(eng.cps_frac.num > 0)
    eng.set_cps(1.0e12)
    assert_true(eng.cps_frac.num > 0)
    assert_equal(render_to_wav("d1 $ s \"bd\"", -1.0, tmp_dir + "/neg.wav", 48000), 0)
    assert_equal(render_to_wav("d1 $ s \"bd\"", 0.0, tmp_dir + "/zero.wav", 48000), 0)
    var loop_f = render_loop_to_wav("setcps 1e-8\nd1 $ s \"bd\"", 1, tmp_dir + "/tiny_cps.wav", 48000)
    assert_true(loop_f > 0)
    assert_equal(render_loop_to_wav("d1 $ s \"bd\"", 0, tmp_dir + "/zero_cyc.wav", 48000), 0)

    # B5: AaaRenderGraph.compile out-of-range resource_id returns False cleanly
    var rg = AaaRenderGraph()
    _ = rg.add_resource("buf0", 1024)
    var acc_oob = List[RgResourceAccess]()
    acc_oob.append(RgResourceAccess(99, RG_USAGE_COMPUTE_WRITE))
    _ = rg.add_pass("bad_pass", acc_oob)
    assert_false(rg.compile())

    var rg_neg = AaaRenderGraph()
    _ = rg_neg.add_resource("buf0", 1024)
    var acc_neg = List[RgResourceAccess]()
    acc_neg.append(RgResourceAccess(-1, RG_USAGE_COMPUTE_WRITE))
    _ = rg_neg.add_pass("neg_pass", acc_neg)
    assert_false(rg_neg.compile())

    # B6: VsmPageManager.process_requests rejects short table_words cleanly
    var vsm = VsmPageManager()
    var short_tbl = List[UInt32](length=10, fill=UInt32(1))
    assert_equal(len(vsm.process_requests(short_tbl)), 0)

    # B7: PhysicsWorld / CompoundShape / NavMesh deserialize count & remaining-bytes validation
    var bw_phys = BlobWriter()
    bw_phys.begin(fourcc("PHYW"), 2)
    bw_phys.f32(Float32(-9.81))
    bw_phys.boolean(False); bw_phys.f32(Float32(0.0)); bw_phys.boolean(False); bw_phys.f32(Float32(1000.0))
    bw_phys.boolean(True); bw_phys.integer(30); bw_phys.f32(Float32(0.0)); bw_phys.integer(0)
    bw_phys.integer(1); bw_phys.integer(1); bw_phys.integer(1)
    bw_phys.integer(4); bw_phys.f32(Float32(30.0)); bw_phys.f32(Float32(1.0)); bw_phys.f32(Float32(5.0))
    bw_phys.f32(Float32(1.0)); bw_phys.f32(Float32(100.0)); bw_phys.boolean(True); bw_phys.boolean(False)
    bw_phys.f32(Float32(0.0)); bw_phys.boolean(False)
    bw_phys.integer(1)          # n_hulls = 1
    bw_phys.integer(1 << 30)    # nv = 1<<30 (malformed!)
    bw_phys.end()
    var pw_bad = PhysicsWorld()
    var br_phys = BlobReader(bw_phys.data.copy())
    var phys_raised = False
    try:
        pw_bad.deserialize(br_phys)
    except:
        phys_raised = True
    assert_true(phys_raised)

    var bw_cmp = BlobWriter()
    bw_cmp.begin(fourcc("CMPD"), 1)
    bw_cmp.integer(1)           # n_ch = 1
    bw_cmp.integer(0); bw_cmp.vec3(Vec3()); bw_cmp.quat(Quat())
    bw_cmp.f32(Float32(1.0)); bw_cmp.f32(Float32(1.0)); bw_cmp.vec3(Vec3()); bw_cmp.f32(Float32(1.0))
    bw_cmp.integer(1 << 30)     # nv = 1<<30
    bw_cmp.end()
    var br_cmp = BlobReader(bw_cmp.data.copy())
    var cmp_raised = False
    try:
        _ = CompoundShape.deserialize(br_cmp)
    except:
        cmp_raised = True
    assert_true(cmp_raised)

    var bw_nav_neg = BlobWriter()
    bw_nav_neg.begin(fourcc("NAVM"), 1)
    for _ in range(8):
        bw_nav_neg.f32(Float32(0.5))
    for _ in range(5):
        bw_nav_neg.integer(1)
    bw_nav_neg.vec3(Vec3()); bw_nav_neg.f32(Float32(0.5)); bw_nav_neg.f32(Float32(0.5))
    bw_nav_neg.integer(-1)      # n_cols = -1
    bw_nav_neg.end()
    var nm_bad = NavMesh()
    var br_nav_neg = BlobReader(bw_nav_neg.data.copy())
    var nav_neg_raised = False
    try:
        nm_bad.deserialize(br_nav_neg)
    except:
        nav_neg_raised = True
    assert_true(nav_neg_raised)

    var bw_nav_big = BlobWriter()
    bw_nav_big.begin(fourcc("NAVM"), 1)
    for _ in range(8):
        bw_nav_big.f32(Float32(0.5))
    for _ in range(5):
        bw_nav_big.integer(1)
    bw_nav_big.vec3(Vec3()); bw_nav_big.f32(Float32(0.5)); bw_nav_big.f32(Float32(0.5))
    bw_nav_big.integer(1 << 30) # n_cols = 1<<30
    bw_nav_big.end()
    var br_nav_big = BlobReader(bw_nav_big.data.copy())
    var nav_big_raised = False
    try:
        nm_bad.deserialize(br_nav_big)
    except:
        nav_big_raised = True
    assert_true(nav_big_raised)


def _put_le32(mut buf: List[UInt8], val: Int):
    buf.append(UInt8(val & 0xFF))
    buf.append(UInt8((val >> 8) & 0xFF))
    buf.append(UInt8((val >> 16) & 0xFF))
    buf.append(UInt8((val >> 24) & 0xFF))


def _put_le64(mut buf: List[UInt8], val: UInt64):
    for b in range(8):
        buf.append(UInt8((val >> UInt64(b * 8)) & 0xFF))


def _build_dwa_exr_with_header_counts(
    unknown_uncomp_sz: UInt64,
    unknown_comp_sz: UInt64,
    ac_comp_sz: UInt64,
    dc_comp_sz: UInt64,
    rle_comp_sz: UInt64,
    rle_uncomp_sz: UInt64,
    rle_raw_sz: UInt64,
    total_ac_cnt: UInt64,
    total_dc_cnt: UInt64,
    ac_compression: UInt64,
) -> List[UInt8]:
    var out = List[UInt8]()
    _put_le32(out, EXR_MAGIC)
    _put_le32(out, 2)
    # channels: R, G, B (HALF)
    for b in String("channels").as_bytes():
        out.append(b)
    out.append(0)
    for b in String("chlist").as_bytes():
        out.append(b)
    out.append(0)
    _put_le32(out, 3 * 18 + 1)
    for ch in ["B", "G", "R"]:
        out.append(UInt8(ord(ch)))
        out.append(0)
        _put_le32(out, 1)  # HALF
        out.append(0); out.append(0); out.append(0); out.append(0)
        _put_le32(out, 1); _put_le32(out, 1)
    out.append(0)
    # compression = DWAA
    for b in String("compression").as_bytes():
        out.append(b)
    out.append(0)
    for b in String("compression").as_bytes():
        out.append(b)
    out.append(0)
    _put_le32(out, 1)
    out.append(UInt8(EXR_DWAA_COMPRESSION))
    # dataWindow = 0..7, 0..7
    for b in String("dataWindow").as_bytes():
        out.append(b)
    out.append(0)
    for b in String("box2i").as_bytes():
        out.append(b)
    out.append(0)
    _put_le32(out, 16)
    _put_le32(out, 0); _put_le32(out, 0); _put_le32(out, 7); _put_le32(out, 7)
    out.append(0)  # end of header

    var tbl_off = len(out)
    _put_le64(out, UInt64(tbl_off + 8))

    var chunk = List[UInt8]()
    _put_le64(chunk, 2)  # version
    _put_le64(chunk, unknown_uncomp_sz)
    _put_le64(chunk, unknown_comp_sz)
    _put_le64(chunk, ac_comp_sz)
    _put_le64(chunk, dc_comp_sz)
    _put_le64(chunk, rle_comp_sz)
    _put_le64(chunk, rle_uncomp_sz)
    _put_le64(chunk, rle_raw_sz)
    _put_le64(chunk, total_ac_cnt)
    _put_le64(chunk, total_dc_cnt)
    _put_le64(chunk, ac_compression)
    # Minimal 2-byte empty rule table
    chunk.append(UInt8(2)); chunk.append(UInt8(0))
    var payload_bytes = Int(unknown_comp_sz + ac_comp_sz + dc_comp_sz + rle_comp_sz)
    for _ in range(payload_bytes):
        chunk.append(UInt8(0))

    _put_le32(out, 0)
    _put_le32(out, len(chunk))
    for b in chunk:
        out.append(b)
    return out^


def _build_valid_or_extreme_ttf(units_per_em: Int, dx2: Int, dy2: Int) -> List[UInt8]:
    var head = List[UInt8](length=54, fill=0)
    head[18] = UInt8((units_per_em >> 8) & 0xFF)
    head[19] = UInt8(units_per_em & 0xFF)
    head[50] = UInt8(0); head[51] = UInt8(0)

    var hhea = List[UInt8](length=36, fill=0)
    hhea[4] = UInt8(0x03); hhea[5] = UInt8(0x20)
    hhea[6] = UInt8(0xFF); hhea[7] = UInt8(0x38)
    hhea[34] = UInt8(0); hhea[35] = UInt8(2)

    var maxp = List[UInt8](length=6, fill=0)
    maxp[4] = UInt8(0); maxp[5] = UInt8(2)

    var hmtx = List[UInt8](length=8, fill=0)
    hmtx[0] = UInt8(0x02); hmtx[1] = UInt8(0x00)
    hmtx[4] = UInt8(0x02); hmtx[5] = UInt8(0x00)

    var cmap = List[UInt8]()
    _put_be16(cmap, 0); _put_be16(cmap, 1)
    _put_be16(cmap, 3); _put_be16(cmap, 1); _put_be32(cmap, 12)
    _put_be16(cmap, 4); _put_be16(cmap, 32); _put_be16(cmap, 0)
    _put_be16(cmap, 4); _put_be16(cmap, 4); _put_be16(cmap, 1); _put_be16(cmap, 0)
    _put_be16(cmap, 65); _put_be16(cmap, 0xFFFF); _put_be16(cmap, 0)
    _put_be16(cmap, 65); _put_be16(cmap, 0xFFFF)
    _put_be16(cmap, 0xFFC0); _put_be16(cmap, 1)
    _put_be16(cmap, 0); _put_be16(cmap, 0)

    # Simple 3-point triangle glyph 1: (0,0) -> (dx2, 0) -> (dx2, dy2)
    var glyf = List[UInt8]()
    _put_be16(glyf, 1)  # numberOfContours = 1
    _put_be16(glyf, 0); _put_be16(glyf, 0)
    _put_be16(glyf, 500); _put_be16(glyf, 500)
    _put_be16(glyf, 2)  # endPtsOfContours[0] = 2
    _put_be16(glyf, 0)  # instructionLength = 0
    # flags: 3 points, on-curve=1, 16-bit signed dx/dy (bit4=0, bit1=0, bit5=0, bit2=0 -> 0x01)
    glyf.append(UInt8(0x01)); glyf.append(UInt8(0x01)); glyf.append(UInt8(0x01))
    # dx: 0, dx2, 0
    _put_be16(glyf, 0); _put_be16(glyf, dx2); _put_be16(glyf, 0)
    # dy: 0, 0, dy2
    _put_be16(glyf, 0); _put_be16(glyf, 0); _put_be16(glyf, dy2)
    while len(glyf) % 2 != 0:
        glyf.append(UInt8(0))

    var loca = List[UInt8]()
    _put_be16(loca, 0); _put_be16(loca, 0); _put_be16(loca, len(glyf) // 2)

    var tags = [String("cmap"), String("glyf"), String("head"), String("hhea"), String("hmtx"), String("loca"), String("maxp")]
    var blobs = [cmap^, glyf^, head^, hhea^, hmtx^, loca^, maxp^]
    var out = List[UInt8]()
    _put_be32(out, 0x00010000)
    _put_be16(out, len(tags)); _put_be16(out, 64); _put_be16(out, 2); _put_be16(out, 48)
    var cur_off = 12 + len(tags) * 16
    for i in range(len(tags)):
        var tbytes = tags[i].as_bytes()
        for j in range(4):
            out.append(tbytes[j])
        _put_be32(out, 0); _put_be32(out, cur_off); _put_be32(out, len(blobs[i]))
        cur_off += (len(blobs[i]) + 3) & ~3
    for i in range(len(blobs)):
        for b in blobs[i]:
            out.append(b)
        while len(out) % 4 != 0:
            out.append(UInt8(0))
    return out^


def test_r4_partition_a_fixes() raises:
    # A4a: inflate_zlib must clamp initial pre-allocation capacity on untrusted expected_bytes
    var inf_huge = inflate_zlib(List[UInt8](), 0, expected_bytes=0x4000000000000000)
    assert_equal(len(inf_huge), 0)

    # A4b: OpenEXR DWAA untrusted uncompressed counts & Int64 * 2 overflow must reject cleanly
    var exr_bad_unk = _build_dwa_exr_with_header_counts(
        UInt64(0x4000000000000000), UInt64(8), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
        UInt64(0), UInt64(0), UInt64(1),
    )
    assert_false(decode_exr_fp16(exr_bad_unk).valid)

    var exr_bad_ac_deflate = _build_dwa_exr_with_header_counts(
        UInt64(0), UInt64(0), UInt64(8), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
        UInt64(0x2000000000000000), UInt64(0), UInt64(1),
    )
    assert_false(decode_exr_fp16(exr_bad_ac_deflate).valid)

    var exr_bad_ac_ovf = _build_dwa_exr_with_header_counts(
        UInt64(0), UInt64(0), UInt64(8), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
        UInt64(0x4000000000000000), UInt64(0), UInt64(1),
    )
    assert_false(decode_exr_fp16(exr_bad_ac_ovf).valid)

    var exr_bad_ac_huf = _build_dwa_exr_with_header_counts(
        UInt64(0), UInt64(0), UInt64(24), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
        UInt64(0x2000000000000000), UInt64(0), UInt64(0),
    )
    assert_false(decode_exr_fp16(exr_bad_ac_huf).valid)

    var exr_bad_dc_rle = _build_dwa_exr_with_header_counts(
        UInt64(0), UInt64(0), UInt64(0), UInt64(8), UInt64(8), UInt64(0x4000000000000000), UInt64(0x4000000000000000),
        UInt64(0), UInt64(0x2000000000000000), UInt64(1),
    )
    assert_false(decode_exr_fp16(exr_bad_dc_rle).valid)

    # A4c: Valid EXR round-trip continues passing
    var hdr_px: List[Vec3] = [
        Vec3(1.0, 0.5, 0.25), Vec3(0.0, 2.0, 0.5),
        Vec3(0.25, 0.25, 4.0), Vec3(0.5, 0.5, 0.5),
    ]
    var exr_ok_bytes = encode_exr_fp16(hdr_px, 2, 2)
    var exr_ok = decode_exr_fp16(exr_ok_bytes)
    assert_true(exr_ok.valid)
    assert_equal(exr_ok.width, 2)
    assert_equal(exr_ok.height, 2)
    assert_true(abs(exr_ok.pixels[0].x - Float32(1.0)) < Float32(1.0e-3))

    # A5: sdf_bake extreme Int16 bounds with units_per_em=16 must return empty BakedGlyph cleanly (no OOM/hang)
    var extreme_ttf = _build_valid_or_extreme_ttf(16, 32767, 32767)
    var font_ext = TrueTypeFont()
    assert_true(font_ext.load_bytes(extreme_ttf^))
    var bg_sdf_ext = bake_glyph_id(font_ext, 1, 64, 4)
    assert_equal(bg_sdf_ext.w, 0)
    assert_equal(bg_sdf_ext.h, 0)
    assert_equal(len(bg_sdf_ext.pixels), 0)
    var bg_msdf_ext = bake_glyph_msdf(font_ext, 65, 1024, 4)
    assert_equal(bg_msdf_ext.w, 0)
    assert_equal(bg_msdf_ext.h, 0)
    assert_equal(len(bg_msdf_ext.pixels), 0)
    var bg_mtsdf_ext = bake_glyph_mtsdf(font_ext, 65, 1024, 4)
    assert_equal(bg_mtsdf_ext.w, 0)
    assert_equal(bg_mtsdf_ext.h, 0)
    assert_equal(len(bg_mtsdf_ext.pixels), 0)

    # Valid normal glyph baking must still succeed
    var ok_ttf = _build_valid_or_extreme_ttf(1000, 600, 700)
    var font_ok = TrueTypeFont()
    assert_true(font_ok.load_bytes(ok_ttf^))
    var bg_ok_1 = bake_glyph(font_ok, 65, 32, 4)
    var bg_ok_3 = bake_glyph_msdf(font_ok, 65, 32, 4)
    var bg_ok_4 = bake_glyph_mtsdf(font_ok, 65, 32, 4)
    assert_true(bg_ok_1.w > 0 and bg_ok_1.h > 0 and len(bg_ok_1.pixels) == bg_ok_1.w * bg_ok_1.h)
    assert_true(bg_ok_3.w > 0 and bg_ok_3.h > 0 and len(bg_ok_3.pixels) == bg_ok_3.w * bg_ok_3.h * 3)
    assert_true(bg_ok_4.w > 0 and bg_ok_4.h > 0 and len(bg_ok_4.pixels) == bg_ok_4.w * bg_ok_4.h * 4)


def test_r4_partition_b_fixes() raises:
    # B1a: ParticleChain.deserialize remaining-bytes & n_r vs n_p validation + valid round-trip
    var bw_xc_big = BlobWriter()
    bw_xc_big.begin(XPBD_CHAIN_TAG, XPBD_BLOB_VERSION)
    bw_xc_big.integer(1 << 20)
    bw_xc_big.end()
    var ch_bad = ParticleChain()
    var r_xc_big = BlobReader(bw_xc_big.data.copy())
    var xc_big_raised = False
    try:
        ch_bad.deserialize(r_xc_big)
    except:
        xc_big_raised = True
    assert_true(xc_big_raised)

    var bw_xc_nr = BlobWriter()
    bw_xc_nr.begin(XPBD_CHAIN_TAG, XPBD_BLOB_VERSION)
    bw_xc_nr.integer(2)  # n_p = 2
    for _ in range(2):
        bw_xc_nr.vec3(Vec3()); bw_xc_nr.vec3(Vec3()); bw_xc_nr.vec3(Vec3()); bw_xc_nr.f32(Float32(1.0))
    bw_xc_nr.integer(5)  # n_r = 5 > n_p - 1!
    for _ in range(5):
        bw_xc_nr.f32(Float32(0.5))
    bw_xc_nr.integer(0)
    bw_xc_nr.f32(Float32(0.0)); bw_xc_nr.f32(Float32(0.4)); bw_xc_nr.f32(Float32(1.5)); bw_xc_nr.f32(Float32(-1e9))
    bw_xc_nr.end()
    var r_xc_nr = BlobReader(bw_xc_nr.data.copy())
    var xc_nr_raised = False
    try:
        ch_bad.deserialize(r_xc_nr)
    except:
        xc_nr_raised = True
    assert_true(xc_nr_raised)

    var ch_ok = ParticleChain.line(Vec3(0.0, 2.0, 0.0), Vec3(1.0, 2.0, 0.0), 4, 0.1)
    ch_ok.pin(0, Vec3(0.0, 2.0, 0.0))
    var bw_xc_ok = BlobWriter()
    ch_ok.serialize(bw_xc_ok)
    var ch_ok2 = ParticleChain()
    var r_xc_ok = BlobReader(bw_xc_ok.data.copy())
    ch_ok2.deserialize(r_xc_ok)
    assert_equal(ch_ok2.count(), 4)
    assert_equal(len(ch_ok2.rest), 3)

    # B1b: ParticleMesh.deserialize remaining-bytes & link endpoint OOB validation + valid round-trip
    var bw_xm_big = BlobWriter()
    bw_xm_big.begin(XPBD_MESH_TAG, XPBD_BLOB_VERSION)
    bw_xm_big.integer(0)
    bw_xm_big.integer(1 << 21)
    bw_xm_big.end()
    var pm_bad = ParticleMesh()
    var r_xm_big = BlobReader(bw_xm_big.data.copy())
    var xm_big_raised = False
    try:
        pm_bad.deserialize(r_xm_big)
    except:
        xm_big_raised = True
    assert_true(xm_big_raised)

    var bw_xm_oob = BlobWriter()
    bw_xm_oob.begin(XPBD_MESH_TAG, XPBD_BLOB_VERSION)
    bw_xm_oob.integer(2)
    for _ in range(2):
        bw_xm_oob.vec3(Vec3()); bw_xm_oob.vec3(Vec3()); bw_xm_oob.vec3(Vec3())
        bw_xm_oob.f32(Float32(1.0)); bw_xm_oob.f32(Float32(0.05)); bw_xm_oob.f32(Float32(-1e9))
        bw_xm_oob.vec3(Vec3()); bw_xm_oob.f32(Float32(0.0)); bw_xm_oob.vec3(Vec3())
    bw_xm_oob.integer(1)  # n_l = 1
    bw_xm_oob.integer(99); bw_xm_oob.integer(1)  # OOB link_a = 99
    bw_xm_oob.f32(Float32(1.0)); bw_xm_oob.f32(Float32(0.0)); bw_xm_oob.integer(0)
    bw_xm_oob.integer(0)
    bw_xm_oob.f32(Float32(0.4)); bw_xm_oob.f32(Float32(0.0)); bw_xm_oob.f32(Float32(0.35)); bw_xm_oob.f32(Float32(1.0))
    bw_xm_oob.integer(0); bw_xm_oob.integer(0)
    bw_xm_oob.end()
    var r_xm_oob = BlobReader(bw_xm_oob.data.copy())
    var xm_oob_raised = False
    try:
        pm_bad.deserialize(r_xm_oob)
    except:
        xm_oob_raised = True
    assert_true(xm_oob_raised)

    var pm_ok = ParticleMesh()
    var p0 = pm_ok.add_particle(Vec3(0.0, 1.0, 0.0), 0.5)
    var p1 = pm_ok.add_particle(Vec3(1.0, 1.0, 0.0), 0.5)
    pm_ok.add_link(p0, p1)
    var bw_xm_ok = BlobWriter()
    pm_ok.serialize(bw_xm_ok)
    var pm_ok2 = ParticleMesh()
    var r_xm_ok = BlobReader(bw_xm_ok.data.copy())
    pm_ok2.deserialize(r_xm_ok)
    assert_equal(pm_ok2.count(), 2)
    assert_equal(pm_ok2.links(), 1)
    assert_true(pm_ok2.link_error() < Float32(1.0e-5))

    # B2: flanim_ecs.deserialize_flanim & _read_conditions bounded u32 counts + valid round-trip
    var bw_fl_params = BlobWriter()
    bw_fl_params.begin(FLANIM_FOURCC, FLANIM_VERSION)
    bw_fl_params.text("bad")
    bw_fl_params.f32(Float32(0.2))
    bw_fl_params.u32(UInt32(0xFFFFFFFF))  # n_params = 4B
    bw_fl_params.end()
    var fl_params_raised = False
    try:
        _ = load_flanim_bytes(bw_fl_params.data)
    except:
        fl_params_raised = True
    assert_true(fl_params_raised)

    var bw_fl_conds = BlobWriter()
    bw_fl_conds.begin(FLANIM_FOURCC, FLANIM_VERSION)
    bw_fl_conds.text("bad_cond")
    bw_fl_conds.f32(Float32(0.2))
    bw_fl_conds.u32(0); bw_fl_conds.u32(0); bw_fl_conds.u32(0); bw_fl_conds.u32(0)
    bw_fl_conds.u32(1)  # n_states = 1
    bw_fl_conds.text("idle")
    bw_fl_conds.integer(0); bw_fl_conds.integer(-1); bw_fl_conds.integer(-1); bw_fl_conds.f32(Float32(1.0))
    bw_fl_conds.integer(-1); bw_fl_conds.integer(-1); bw_fl_conds.text(""); bw_fl_conds.boolean(False)
    bw_fl_conds.integer(0); bw_fl_conds.integer(0); bw_fl_conds.boolean(False)
    bw_fl_conds.u32(UInt32(0xFFFFFFFF))  # _read_conditions n = 4B
    bw_fl_conds.end()
    var fl_conds_raised = False
    try:
        _ = load_flanim_bytes(bw_fl_conds.data)
    except:
        fl_conds_raised = True
    assert_true(fl_conds_raised)

    var g_ok = AnimGraphDef()
    g_ok.name = "hero_loco"
    g_ok.params.append("speed")
    var st_ok = StateDef()
    st_ok.name = "run"
    st_ok.clip = 0
    st_ok.when.append(Condition(0, 0, Float32(1.0)))
    g_ok.states.append(st_ok^)
    g_ok.transitions.append(TransitionDef(0, 0, Float32(0.15)))
    var fl_bytes = save_flanim_bytes(g_ok)
    var g_ok2 = load_flanim_bytes(fl_bytes)
    assert_equal(g_ok2.name, "hero_loco")
    assert_equal(len(g_ok2.states), 1)
    assert_equal(len(g_ok2.states[0].when), 1)

    # B3: Scene.load_flscnb_bytes count & num_comps bounds + valid round-trip
    var reg = ComponentRegistry.with_builtins()
    var w_scn = World()
    var sc_bad = Scene("bad")

    var bw_sc_cnt = BlobWriter()
    bw_sc_cnt.begin(SCENE_BIN_MAGIC, SCENE_SCHEMA_VERSION)
    bw_sc_cnt.text("bad")
    bw_sc_cnt.integer(1 << 30)
    bw_sc_cnt.end()
    var sc_cnt_raised = False
    try:
        _ = sc_bad.load_flscnb_bytes(w_scn, reg, lz4_pack(bw_sc_cnt.data))
    except:
        sc_cnt_raised = True
    assert_true(sc_cnt_raised)

    var bw_sc_nc = BlobWriter()
    bw_sc_nc.begin(SCENE_BIN_MAGIC, SCENE_SCHEMA_VERSION)
    bw_sc_nc.text("bad_nc")
    bw_sc_nc.integer(1)
    bw_sc_nc.integer(-1)
    bw_sc_nc.integer(1 << 20)
    bw_sc_nc.end()
    var sc_nc_raised = False
    try:
        _ = sc_bad.load_flscnb_bytes(w_scn, reg, lz4_pack(bw_sc_nc.data))
    except:
        sc_nc_raised = True
    assert_true(sc_nc_raised)

    var w_src = World()
    reg.ensure_registered_in_world(w_src)
    var e_parent = w_src.spawn_empty()
    var e_child = w_src.spawn_empty()
    assert_true(w_src.insert(e_parent, LocalTransform(Vec3(1.0, 2.0, 3.0))))
    assert_true(w_src.insert(e_child, LocalTransform(Vec3(0.5, 0.0, 0.0))))
    assert_true(set_parent(w_src, e_child, e_parent))
    var sc_ok = Scene("valid_bin")
    var sc_packed = sc_ok.save_flscnb_bytes(w_src, reg)
    var w_dst = World()
    var sc_ok2 = Scene("empty")
    var spawned_ents = sc_ok2.load_flscnb_bytes(w_dst, reg, sc_packed)
    assert_equal(len(spawned_ents), 2)
    assert_equal(sc_ok2.name, "valid_bin")

    # B4: ArticulatedRagdoll.deserialize n_states == n_limbs & parent_limb bounds + valid round-trip
    var bw_rag_sm = BlobWriter()
    bw_rag_sm.begin(RAGDOLL_BLOB_TAG, RAGDOLL_BLOB_VERSION)
    bw_rag_sm.integer(1)  # n_limbs = 1
    bw_rag_sm.integer(-1); bw_rag_sm.integer(-1)
    bw_rag_sm.f32(Float32(10.0)); bw_rag_sm.f32(Float32(0.15)); bw_rag_sm.f32(Float32(0.4))
    bw_rag_sm.vec3(Vec3(0.0, 1.0, 0.0)); bw_rag_sm.vec3(Vec3()); bw_rag_sm.vec3(Vec3())
    bw_rag_sm.quat(Quat()); bw_rag_sm.boolean(False); bw_rag_sm.vec3(Vec3(1.0, 0.0, 0.0))
    for _ in range(8):
        bw_rag_sm.f32(Float32(0.0))
    bw_rag_sm.integer(0)  # n_states = 0 != n_limbs
    bw_rag_sm.integer(0); bw_rag_sm.integer(0); bw_rag_sm.integer(0); bw_rag_sm.integer(0)
    bw_rag_sm.quat(Quat()); bw_rag_sm.vec3(Vec3()); bw_rag_sm.boolean(True); bw_rag_sm.vec3(Vec3(0.0, 1.0, 0.0))
    for _ in range(5):
        bw_rag_sm.f32(Float32(1.0))
    bw_rag_sm.integer(4)
    for _ in range(5):
        bw_rag_sm.f32(Float32(1.0))
    bw_rag_sm.quat(Quat()); bw_rag_sm.boolean(False); bw_rag_sm.boolean(False)
    bw_rag_sm.f32(Float32(0.0)); bw_rag_sm.integer(0)
    bw_rag_sm.end()
    var rag_bad = ArticulatedRagdoll()
    var r_rag_sm = BlobReader(bw_rag_sm.data.copy())
    var rag_sm_raised = False
    try:
        rag_bad.deserialize(r_rag_sm)
    except:
        rag_sm_raised = True
    assert_true(rag_sm_raised)

    var rag_ok = ArticulatedRagdoll()
    var ld0 = ArticulatedLimbDef(bone_node=-1, parent_limb=-1, mass=12.0, radius=0.15, length=0.35)
    var ld1 = ArticulatedLimbDef(bone_node=-1, parent_limb=0, mass=8.0, radius=0.12, length=0.35)
    _ = rag_ok.add_limb(ld0)
    _ = rag_ok.add_limb(ld1)
    var bw_rag_ok = BlobWriter()
    rag_ok.serialize(bw_rag_ok)
    var rag_ok2 = ArticulatedRagdoll()
    var r_rag_ok = BlobReader(bw_rag_ok.data.copy())
    rag_ok2.deserialize(r_rag_ok)
    assert_equal(rag_ok2.limb_count(), 2)
    _ = rag_ok2.pelvis_world_position()

    # B5: ContactCache, IslandSet, ConstraintGraph deserialize remaining-bytes & failed checks + round-trips
    var bw_cc_bad = BlobWriter()
    bw_cc_bad.integer(1)
    bw_cc_bad.integer(1 << 20)
    var cc_bad = ContactCache()
    var r_cc_bad = BlobReader(bw_cc_bad.data.copy())
    var cc_raised = False
    try:
        cc_bad.deserialize(r_cc_bad)
    except:
        cc_raised = True
    assert_true(cc_raised)

    var cc_ok = ContactCache()
    cc_ok.entries.append(CachedContact(1, 2, 0, 7, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0), Float32(2.5)))
    cc_ok.touching_a.append(1)
    cc_ok.touching_b.append(2)
    var bw_cc_ok = BlobWriter()
    cc_ok.serialize(bw_cc_ok)
    var cc_ok2 = ContactCache()
    var r_cc_ok = BlobReader(bw_cc_ok.data.copy())
    cc_ok2.deserialize(r_cc_ok)
    assert_equal(len(cc_ok2.entries), 1)
    assert_equal(len(cc_ok2.touching_a), 1)

    var bw_is_bad = BlobWriter()
    bw_is_bad.integer(0)
    bw_is_bad.integer(1 << 20)
    var is_bad = IslandSet()
    var r_is_bad = BlobReader(bw_is_bad.data.copy())
    var is_raised = False
    try:
        is_bad.deserialize(r_is_bad)
    except:
        is_raised = True
    assert_true(is_raised)

    var is_ok = IslandSet()
    is_ok.label.append(10)
    is_ok.link_a.append(10)
    is_ok.link_b.append(20)
    is_ok.pending.append(10)
    var bw_is_ok = BlobWriter()
    is_ok.serialize(bw_is_ok)
    var is_ok2 = IslandSet()
    var r_is_ok = BlobReader(bw_is_ok.data.copy())
    is_ok2.deserialize(r_is_ok)
    assert_equal(len(is_ok2.label), 1)
    assert_equal(len(is_ok2.link_a), 1)
    assert_equal(len(is_ok2.pending), 1)

    var bw_cg_bad = BlobWriter()
    bw_cg_bad.integer(1 << 20)
    var cg_bad = ConstraintGraph()
    var r_cg_bad = BlobReader(bw_cg_bad.data.copy())
    var cg_raised = False
    try:
        cg_bad.deserialize(r_cg_bad)
    except:
        cg_raised = True
    assert_true(cg_raised)

    var cg_ok = ConstraintGraph()
    cg_ok.keys.append(42)
    cg_ok.colours.append(3)
    var bw_cg_ok = BlobWriter()
    cg_ok.serialize(bw_cg_ok)
    var cg_ok2 = ConstraintGraph()
    var r_cg_ok = BlobReader(bw_cg_ok.data.copy())
    cg_ok2.deserialize(r_cg_ok)
    assert_equal(len(cg_ok2.keys), 1)
    assert_equal(cg_ok2.colours[0], 3)


def _str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def test_r5_partition_a_fixes(tmp_dir: String) raises:
    # A1: decode_radiance_hdr Int64 overflow & pre-EOF allocation checks + valid round-trip
    var hdr_ovf = _str_to_bytes("#?\n\n-Y 4294967296 +X 4294967296\n")
    assert_false(decode_radiance_hdr(hdr_ovf).valid)
    var hdr_trunc = _str_to_bytes("#?\n\n-Y 10000 +X 3355\n")
    assert_false(decode_radiance_hdr(hdr_trunc).valid)

    var hdr_px: List[Vec3] = [
        Vec3(1.0, 0.25, 0.5), Vec3(0.5, 1.0, 0.25),
        Vec3(0.25, 0.5, 1.0), Vec3(0.75, 0.75, 0.75),
    ]
    var hdr_ok_bytes = encode_radiance_hdr(hdr_px, 2, 2)
    var hdr_ok = decode_radiance_hdr(hdr_ok_bytes)
    assert_true(hdr_ok.valid)
    assert_equal(hdr_ok.width, 2)
    assert_equal(hdr_ok.height, 2)

    # A2: decode_qoa truncated frame check before allocation + valid round-trip
    var qoa_trunc = List[UInt8]()
    _put_be32(qoa_trunc, Int(QOA_MAGIC))
    _put_be32(qoa_trunc, 50_000_000)
    qoa_trunc.append(UInt8(8))
    qoa_trunc.append(UInt8(0)); qoa_trunc.append(UInt8(0xBB)); qoa_trunc.append(UInt8(0x80))  # 48000 Hz
    qoa_trunc.append(UInt8(0)); qoa_trunc.append(UInt8(10)); qoa_trunc.append(UInt8(0)); qoa_trunc.append(UInt8(16))
    assert_false(decode_qoa(qoa_trunc).valid)

    var qoa_pcm = List[Int16](length=64, fill=Int16(1234))
    var qoa_ok_bytes = encode_qoa(qoa_pcm, 1, 48000)
    var qoa_ok = decode_qoa(qoa_ok_bytes)
    assert_true(qoa_ok.valid)
    assert_equal(qoa_ok.samples_per_channel, 64)

    # A3: decode_bvh8_collider_slab index bounds + valid round-trip
    var bvh_verts: List[Vec3] = [Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)]
    var bvh_tris: List[Int32] = [Int32(0), Int32(1), Int32(2)]
    var bvh_valid = Bvh8MeshCollider.build(bvh_verts, bvh_tris)
    var bvh_valid_bytes = encode_bvh8_collider_slab(bvh_valid)
    var bvh_dec_ok = decode_bvh8_collider_slab(bvh_valid_bytes)
    assert_true(bvh_dec_ok[0])
    var rh_ok = bvh_dec_ok[1].raycast(Vec3(0.2, 0.2, 1.0), Vec3(0.0, 0.0, -1.0), 10.0)
    assert_true(rh_ok.hit)

    var bvh_mut_oob = bvh_valid_bytes.copy()
    # Mutate lane 0 of node 0 (starts at 16 + 1 * 52 = 68; child_indices[0] is at 68 + 24 = 92; child_counts[0] is at 68 + 28 = 96)
    bvh_mut_oob[92] = UInt8(0x3F); bvh_mut_oob[93] = UInt8(0x42); bvh_mut_oob[94] = UInt8(0x0F); bvh_mut_oob[95] = UInt8(0x00)  # 999999
    bvh_mut_oob[96] = UInt8(0); bvh_mut_oob[97] = UInt8(0); bvh_mut_oob[98] = UInt8(0); bvh_mut_oob[99] = UInt8(0)  # c_cnt = 0 (internal)
    assert_false(decode_bvh8_collider_slab(bvh_mut_oob)[0])

    var bvh_mut_root = bvh_valid_bytes.copy()
    bvh_mut_root[12] = UInt8(5); bvh_mut_root[13] = UInt8(0); bvh_mut_root[14] = UInt8(0); bvh_mut_root[15] = UInt8(0)  # root_node = 5 >= n_nodes(1)
    assert_false(decode_bvh8_collider_slab(bvh_mut_root)[0])

    var bvh_mut_leaf = bvh_valid_bytes.copy()
    bvh_mut_leaf[96] = UInt8(99)  # leaf count 99 > n_tri(1)
    assert_false(decode_bvh8_collider_slab(bvh_mut_leaf)[0])

    # A4: decompress_zstd output ceiling inside block/sequence loops + valid round-trip
    # RLE block claiming 1024 bytes when expected_size=16 (no FCS)
    var zstd_bomb_nofcs: List[UInt8] = [
        UInt8(0x28), UInt8(0xB5), UInt8(0x2F), UInt8(0xFD),
        UInt8(0x20),  # single_seg=1, fcs_flag=0 -> 1-byte FCS? Wait: fcs_flag=0 single_seg=0 -> 0 bytes FCS
    ]
    zstd_bomb_nofcs[4] = UInt8(0x00)  # single_seg=0, fcs_flag=0
    zstd_bomb_nofcs.append(UInt8(0x00))  # window descriptor
    var bh_rle = 1 | (1 << 1) | (1024 << 3)  # last=1, btype=1 (RLE), bsize=1024
    zstd_bomb_nofcs.append(UInt8(bh_rle & 0xFF))
    zstd_bomb_nofcs.append(UInt8((bh_rle >> 8) & 0xFF))
    zstd_bomb_nofcs.append(UInt8((bh_rle >> 16) & 0xFF))
    zstd_bomb_nofcs.append(UInt8(0x41))
    assert_false(decompress_zstd(zstd_bomb_nofcs, 0, len(zstd_bomb_nofcs), 16)[0])

    # RLE block claiming 1024 bytes with FCS=16
    var zstd_bomb_fcs: List[UInt8] = [
        UInt8(0x28), UInt8(0xB5), UInt8(0x2F), UInt8(0xFD),
        UInt8(0x20),  # single_seg=1, fcs_flag=0 -> 1-byte FCS
        UInt8(16),    # FCS = 16
        UInt8(bh_rle & 0xFF), UInt8((bh_rle >> 8) & 0xFF), UInt8((bh_rle >> 16) & 0xFF),
        UInt8(0x41),
    ]
    assert_false(decompress_zstd(zstd_bomb_fcs, 0, len(zstd_bomb_fcs), 16)[0])

    # Valid raw zstd frame ("TEST")
    var bh_raw = 1 | (0 << 1) | (4 << 3)
    var zstd_ok: List[UInt8] = [
        UInt8(0x28), UInt8(0xB5), UInt8(0x2F), UInt8(0xFD),
        UInt8(0x20), UInt8(4),
        UInt8(bh_raw & 0xFF), UInt8((bh_raw >> 8) & 0xFF), UInt8((bh_raw >> 16) & 0xFF),
        UInt8(84), UInt8(69), UInt8(83), UInt8(84),
    ]
    var zres_ok = decompress_zstd(zstd_ok, 0, len(zstd_ok), 4)
    assert_true(zres_ok[0])
    assert_equal(len(zres_ok[1]), 4)
    assert_equal(zres_ok[1][0], UInt8(84))

    # A5: inflate_zlib max_out enforcement & decode_png Int64 width*height overflow + valid round-trip
    # Stored zlib stream of 64 bytes when expected_bytes=16
    var zlib_bomb: List[UInt8] = [UInt8(0x78), UInt8(0x01), UInt8(0x01), UInt8(64), UInt8(0), UInt8(0xBF), UInt8(0xFF)]
    for _ in range(64):
        zlib_bomb.append(UInt8(0x42))
    assert_equal(len(inflate_zlib(zlib_bomb, 0, 16)), 0)

    var png_px: List[UInt32] = [UInt32(0xFF112233), UInt32(0xFF445566), UInt32(0xFF778899), UInt32(0xFFAABBCC)]
    var png_ok_bytes = encode_png(png_px, 2, 2)
    var png_ok = decode_png(png_ok_bytes)
    assert_true(png_ok.valid)
    assert_equal(png_ok.width, 2)
    var png_ovf = png_ok_bytes.copy()
    # Mutate IHDR width & height (bytes 16..23) to 0xC0000000 so width * height wraps negative in Int64
    png_ovf[16] = UInt8(0xC0); png_ovf[17] = UInt8(0); png_ovf[18] = UInt8(0); png_ovf[19] = UInt8(0)
    png_ovf[20] = UInt8(0xC0); png_ovf[21] = UInt8(0); png_ovf[22] = UInt8(0); png_ovf[23] = UInt8(0)
    assert_false(decode_png(png_ovf).valid)

    # A6: decode_exr_fp16 offset table validation before pixel buffer allocation + valid round-trip
    var exr_ok_bytes = encode_exr_fp16(hdr_px, 2, 2)
    var exr_ok = decode_exr_fp16(exr_ok_bytes)
    assert_true(exr_ok.valid)
    # Build minimal truncated 8192x4096 EXR header with 0 chunk offset table bytes
    var exr_trunc = List[UInt8]()
    _put_le32(exr_trunc, EXR_MAGIC)
    _put_le32(exr_trunc, 2)
    for b in String("channels").as_bytes(): exr_trunc.append(b)
    exr_trunc.append(0)
    for b in String("chlist").as_bytes(): exr_trunc.append(b)
    exr_trunc.append(0)
    _put_le32(exr_trunc, 19)
    exr_trunc.append(UInt8(ord("R"))); exr_trunc.append(0)
    _put_le32(exr_trunc, 1); _put_le32(exr_trunc, 0); _put_le32(exr_trunc, 1); _put_le32(exr_trunc, 1)
    exr_trunc.append(0)
    for b in String("dataWindow").as_bytes(): exr_trunc.append(b)
    exr_trunc.append(0)
    for b in String("box2i").as_bytes(): exr_trunc.append(b)
    exr_trunc.append(0)
    _put_le32(exr_trunc, 16)
    _put_le32(exr_trunc, 0); _put_le32(exr_trunc, 0); _put_le32(exr_trunc, 8191); _put_le32(exr_trunc, 4095)
    exr_trunc.append(0)  # end of header, 0 bytes of offset table
    assert_false(decode_exr_fp16(exr_trunc).valid)

    # A7: McpServerV2 render.capture_viewport updates cap_w/cap_h on clamped software preview (>2048)
    var studio = FlintStudioState(headless=True)
    var assets = AssetServer(enable_watcher=False)
    var pool = JobPool()
    var mcp = McpServerV2(workspace_root=tmp_dir)
    _ = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"t5"}}}',
        studio, assets, pool,
    )
    var r_cap_w = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"cap_wide.png","width":2500,"height":1000}}}',
        studio, assets, pool,
    )
    var d_cap_w = parse_json(_str_to_bytes(r_cap_w))
    assert_true(d_cap_w.valid)
    var sc_w = d_cap_w.object_get(d_cap_w.object_get(d_cap_w.root, "result"), "structuredContent")
    assert_true(d_cap_w.boolean(d_cap_w.object_get(sc_w, "ok")))
    assert_equal(d_cap_w.integer(d_cap_w.object_get(sc_w, "width")), 2048)
    assert_equal(d_cap_w.integer(d_cap_w.object_get(sc_w, "height")), 1000)
    var cap_img_w = decode_png(read_bytes(tmp_dir + "/cap_wide.png"), 4096)
    assert_true(cap_img_w.valid)
    assert_equal(cap_img_w.width, 2048)
    assert_equal(cap_img_w.height, 1000)

    var r_cap_h = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"render.capture_viewport","arguments":{"output_path":"cap_tall.png","width":160,"height":2500}}}',
        studio, assets, pool,
    )
    var d_cap_h = parse_json(_str_to_bytes(r_cap_h))
    assert_true(d_cap_h.valid)
    var sc_h = d_cap_h.object_get(d_cap_h.object_get(d_cap_h.root, "result"), "structuredContent")
    assert_true(d_cap_h.boolean(d_cap_h.object_get(sc_h, "ok")))
    assert_equal(d_cap_h.integer(d_cap_h.object_get(sc_h, "width")), 160)
    assert_equal(d_cap_h.integer(d_cap_h.object_get(sc_h, "height")), 2048)


def test_r5_partition_b_fixes(tmp_dir: String) raises:
    # B8: pose_position short/empty parents guard + ArticulatedRagdoll.begin_recovery safety
    var p4 = Pose(4)
    p4.t[1] = Vec3(1.0, 2.0, 3.0)
    var pos_empty_par = pose_position(p4, List[Int](), 1)
    assert_true(abs(pos_empty_par.y - Float32(2.0)) < Float32(1.0e-5))

    var rag_unbound = ArticulatedRagdoll()
    _ = rag_unbound.add_limb(ArticulatedLimbDef(bone_node=0, parent_limb=-1, mass=12.0, radius=0.15, length=0.35))
    var bw_rag_unb = BlobWriter()
    rag_unbound.serialize(bw_rag_unb)
    var rag_unb2 = ArticulatedRagdoll()
    var br_rag_unb = BlobReader(bw_rag_unb.data.copy())
    rag_unb2.deserialize(br_rag_unb)
    var inert_u = Inertializer()
    var rp_u = Vec3()
    var ry_u = Float32(0.0)
    rag_unb2.begin_recovery(rp_u, ry_u, inert_u, p4)
    assert_true(inert_u.has_pose)

    # B9: VehicleChassis.deserialize n_gears >= 1 & n_states == n_wheels + valid round-trip
    var veh_ok = VehicleChassis(1450.0)
    veh_ok.setup_four_wheel_car()
    var bw_veh_ok = BlobWriter()
    veh_ok.serialize(bw_veh_ok)
    var veh_ok2 = VehicleChassis(900.0)
    var br_veh_ok = BlobReader(bw_veh_ok.data.copy())
    veh_ok2.deserialize(br_veh_ok)
    assert_equal(len(veh_ok2.wheels), 4)
    assert_equal(len(veh_ok2.wheel_states), 4)
    veh_ok2.substep_suspension(Float32(1.0 / 60.0))

    var bw_v_0g = BlobWriter()
    bw_v_0g.begin(VEHICLE_BLOB_TAG, VEHICLE_BLOB_VERSION)
    bw_v_0g.f32(1450.0); bw_v_0g.f32(1.0 / 1450.0)
    bw_v_0g.vec3(Vec3(1, 1, 1)); bw_v_0g.vec3(Vec3(1, 1, 1))
    bw_v_0g.vec3(Vec3()); bw_v_0g.quat(Quat()); bw_v_0g.vec3(Vec3()); bw_v_0g.vec3(Vec3())
    for _ in range(8):
        bw_v_0g.f32(Float32(0.0))
    bw_v_0g.integer(0)  # current_gear
    bw_v_0g.integer(0)  # n_gears = 0 (invalid!)
    bw_v_0g.end()
    var veh_bad_0g = VehicleChassis(1000.0)
    var br_v_0g = BlobReader(bw_v_0g.data.copy())
    var v_0g_raised = False
    try:
        veh_bad_0g.deserialize(br_v_0g)
    except:
        v_0g_raised = True
    assert_true(v_0g_raised)

    var bw_v_mm = BlobWriter()
    bw_v_mm.begin(VEHICLE_BLOB_TAG, VEHICLE_BLOB_VERSION)
    bw_v_mm.f32(1450.0); bw_v_mm.f32(1.0 / 1450.0)
    bw_v_mm.vec3(Vec3(1, 1, 1)); bw_v_mm.vec3(Vec3(1, 1, 1))
    bw_v_mm.vec3(Vec3()); bw_v_mm.quat(Quat()); bw_v_mm.vec3(Vec3()); bw_v_mm.vec3(Vec3())
    for _ in range(8):
        bw_v_mm.f32(Float32(0.0))
    bw_v_mm.integer(0)
    bw_v_mm.integer(1); bw_v_mm.f32(Float32(3.0))  # n_gears = 1
    for _ in range(20):  # final_drive..tp
        bw_v_mm.f32(Float32(1.0))
    bw_v_mm.integer(1)  # n_wheels = 1
    bw_v_mm.vec3(Vec3())
    for _ in range(7):
        bw_v_mm.f32(Float32(0.3))
    bw_v_mm.boolean(True); bw_v_mm.boolean(False); bw_v_mm.f32(Float32(1.0))
    bw_v_mm.integer(0)  # n_states = 0 != n_wheels(1)
    bw_v_mm.end()
    var veh_bad_mm = VehicleChassis(1000.0)
    var br_v_mm = BlobReader(bw_v_mm.data.copy())
    var v_mm_raised = False
    try:
        veh_bad_mm.deserialize(br_v_mm)
    except:
        v_mm_raised = True
    assert_true(v_mm_raised)

    # B10: CompoundShape & PhysicsWorld deserialize hull vertex index validation + valid round-trip
    var bw_cmp_idx = BlobWriter()
    bw_cmp_idx.begin(COMPOUND_BLOB_TAG, SHAPES_V5_BLOB_VERSION)
    bw_cmp_idx.integer(1)
    bw_cmp_idx.integer(0); bw_cmp_idx.vec3(Vec3()); bw_cmp_idx.quat(Quat())
    bw_cmp_idx.f32(Float32(1.0)); bw_cmp_idx.f32(Float32(1.0)); bw_cmp_idx.vec3(Vec3()); bw_cmp_idx.f32(Float32(1.0))
    bw_cmp_idx.integer(4)
    bw_cmp_idx.vec3(Vec3(0, 0, 0)); bw_cmp_idx.vec3(Vec3(1, 0, 0))
    bw_cmp_idx.vec3(Vec3(0, 1, 0)); bw_cmp_idx.vec3(Vec3(0, 0, 1))
    bw_cmp_idx.integer(3)
    bw_cmp_idx.integer(0); bw_cmp_idx.integer(1); bw_cmp_idx.integer(999)  # OOB index 999 >= nv(4)
    bw_cmp_idx.end()
    var br_cmp_idx = BlobReader(bw_cmp_idx.data.copy())
    var cmp_idx_raised = False
    try:
        _ = CompoundShape.deserialize(br_cmp_idx)
    except:
        cmp_idx_raised = True
    assert_true(cmp_idx_raised)

    var bw_pw_idx = BlobWriter()
    bw_pw_idx.begin(WORLD_BLOB_TAG, WORLD_BLOB_VERSION)
    bw_pw_idx.f32(Float32(-9.81))
    bw_pw_idx.boolean(False); bw_pw_idx.f32(Float32(0.0)); bw_pw_idx.boolean(False); bw_pw_idx.f32(Float32(1000.0))
    bw_pw_idx.boolean(True); bw_pw_idx.integer(30); bw_pw_idx.f32(Float32(0.0)); bw_pw_idx.integer(0)
    bw_pw_idx.integer(1); bw_pw_idx.integer(1); bw_pw_idx.integer(1)
    bw_pw_idx.integer(4); bw_pw_idx.f32(Float32(30.0)); bw_pw_idx.f32(Float32(1.0)); bw_pw_idx.f32(Float32(5.0))
    bw_pw_idx.f32(Float32(1.0)); bw_pw_idx.f32(Float32(100.0)); bw_pw_idx.boolean(True); bw_pw_idx.boolean(False)
    bw_pw_idx.f32(Float32(0.0)); bw_pw_idx.boolean(False)
    bw_pw_idx.integer(1)  # n_hulls = 1
    bw_pw_idx.integer(4)
    bw_pw_idx.vec3(Vec3(0, 0, 0)); bw_pw_idx.vec3(Vec3(1, 0, 0))
    bw_pw_idx.vec3(Vec3(0, 1, 0)); bw_pw_idx.vec3(Vec3(0, 0, 1))
    bw_pw_idx.integer(3)
    bw_pw_idx.integer(0); bw_pw_idx.integer(-1); bw_pw_idx.integer(2)  # negative index -1
    bw_pw_idx.end()
    var pw_idx = PhysicsWorld()
    var br_pw_idx = BlobReader(bw_pw_idx.data.copy())
    var pw_idx_raised = False
    try:
        pw_idx.deserialize(br_pw_idx)
    except:
        pw_idx_raised = True
    assert_true(pw_idx_raised)

    var hull_ok = ConvexHull.build_quickhull([Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)])
    var cmp_ok = CompoundShape()
    cmp_ok.add_child(CompoundChild.convex(Vec3(), Quat(), hull_ok))
    var bw_cmp_ok = BlobWriter()
    cmp_ok.serialize(bw_cmp_ok)
    var br_cmp_ok = BlobReader(bw_cmp_ok.data.copy())
    var cmp_ok2 = CompoundShape.deserialize(br_cmp_ok)
    var mp_ok = cmp_ok2.compute_mass_properties(Float32(1000.0))
    assert_true(mp_ok[0] > Float32(0.0))

    # B11: CrowdSimulator._query_wall_candidates span cap & large-wall tracking
    var crowd = CrowdSimulator()
    crowd.add_wall_segment(Vec3(0.0, 0.0, 0.0), Vec3(10.0, 0.0, 0.0))
    _ = crowd.add_agent(Vec3(0.0, 0.0, 0.0), radius=0.35, max_speed=3.5, goal=Vec3(1.0e7, 0.0, 1.0e7))
    crowd.step_orca_simd8(Float32(1.0 / 60.0))
    var ap = crowd.agent_position(0)
    assert_true(ap.x > Float32(0.0) or ap.z > Float32(0.0))

    var crowd_lw = CrowdSimulator()
    crowd_lw.add_wall_segment(Vec3(-500.0, 0.0, 5.0), Vec3(500.0, 0.0, 5.0))
    _ = crowd_lw.add_agent(Vec3(0.0, 0.0, 0.0), radius=0.35, max_speed=4.0, goal=Vec3(0.0, 0.0, 10.0))
    for _ in range(30):
        crowd_lw.step_orca_simd8(Float32(1.0 / 30.0))
    assert_true(crowd_lw.agent_position(0).z < Float32(4.7))

    # B12: ml.ops.transpose 0-D float tensor + 2-D transpose
    var t0 = Tensor.floats(List[Int](), [Float32(42.0)])
    var t0_tr = transpose(t0, List[Int]())
    assert_equal(t0_tr.rank(), 0)
    assert_true(abs(t0_tr.f[0] - Float32(42.0)) < Float32(1.0e-6))
    var t2 = Tensor.floats([2, 3], [Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0), Float32(5.0), Float32(6.0)])
    var t2_tr = transpose(t2, [1, 0])
    assert_equal(t2_tr.shape[0], 3)
    assert_equal(t2_tr.shape[1], 2)
    assert_true(abs(t2_tr.f[1] - Float32(4.0)) < Float32(1.0e-6))

    # B13: load_bvh overflow-safe frame check & numel_of overflow/negative detection
    var bvh_ovf_path = tmp_dir + "/ovf.bvh"
    _write_text(
        bvh_ovf_path,
        "HIERARCHY\nROOT Hips\n{\n  OFFSET 0 0 0\n  CHANNELS 3 Xposition Yposition Zposition\n  End Site\n  {\n    OFFSET 0 1 0\n  }\n}\nMOTION\nFrames: 6148914691236517206\nFrame Time: 0.033333\n0 0 0\n",
    )
    var bvh_raised = False
    try:
        _ = load_bvh(bvh_ovf_path)
    except:
        bvh_raised = True
    assert_true(bvh_raised)

    var bvh_ok_path = tmp_dir + "/ok.bvh"
    _write_text(
        bvh_ok_path,
        "HIERARCHY\nROOT Hips\n{\n  OFFSET 0 0 0\n  CHANNELS 3 Xposition Yposition Zposition\n  End Site\n  {\n    OFFSET 0 1 0\n  }\n}\nMOTION\nFrames: 1\nFrame Time: 0.033333\n1 2 3\n",
    )
    var bvh_ok = load_bvh(bvh_ok_path)
    assert_equal(len(bvh_ok.frames), 1)

    assert_equal(numel_of([1 << 32, 1 << 32]), -1)
    assert_equal(numel_of([2, -3]), -1)
    assert_equal(numel_of([2, 3, 4]), 24)

    # B14: flip_error_map negative (-w, -h) guard + valid 4x4 FLIP diff
    var black4 = List[UInt32](length=16, fill=UInt32(0xFF000000))
    var white4 = List[UInt32](length=16, fill=UInt32(0xFFFFFFFF))
    assert_equal(len(flip_error_map(white4, black4, -4, -4)), 0)
    var fmap_ok = flip_error_map(white4, black4, 4, 4)
    assert_equal(len(fmap_ok), 16)
    assert_true(flip_mean(fmap_ok) > 0.5)


def _build_test_glb(n_tris: Int, bad_index: Int) -> List[UInt8]:
    """Builds a minimal 3-vertex GLB with `n_tris` triangles (putting `bad_index` in triangle 0 if >= 0)."""
    var bin_buf = List[UInt8]()
    # 3 float32 positions: (0,0,0), (1,0,0), (0,1,0) = 36 bytes
    var pos_vals: List[Float32] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for pv in pos_vals:
        _put_le32(bin_buf, Int(bitcast[DType.uint32](pv)))
    var idx_off = len(bin_buf)
    var idx_cnt = n_tris * 3
    for ti in range(n_tris):
        var i0 = bad_index if (ti == 0 and bad_index >= 0) else 0
        _put_le32(bin_buf, i0)
        _put_le32(bin_buf, 1)
        _put_le32(bin_buf, 2)
    var idx_bytes = idx_cnt * 4
    while len(bin_buf) % 4 != 0:
        bin_buf.append(UInt8(0))

    var json_str = (
        '{"asset":{"version":"2.0"},'
        '"buffers":[{"byteLength":' + String(len(bin_buf)) + '}],'
        '"bufferViews":['
        '{"buffer":0,"byteOffset":0,"byteLength":36},'
        '{"buffer":0,"byteOffset":' + String(idx_off) + ',"byteLength":' + String(idx_bytes) + '}'
        '],'
        '"accessors":['
        '{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},'
        '{"bufferView":1,"componentType":5125,"count":' + String(idx_cnt) + ',"type":"SCALAR"}'
        '],'
        '"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}]}'
    )
    var json_bytes = _str_to_bytes(json_str)
    while len(json_bytes) % 4 != 0:
        json_bytes.append(UInt8(0x20))

    var total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    var out = List[UInt8](capacity=total_len)
    _put_le32(out, 0x46546C67)  # "glTF"
    _put_le32(out, 2)
    _put_le32(out, total_len)
    _put_le32(out, len(json_bytes))
    _put_le32(out, 0x4E4F534A)  # "JSON"
    for b in json_bytes:
        out.append(b)
    _put_le32(out, len(bin_buf))
    _put_le32(out, 0x004E4942)  # "BIN\0"
    for b in bin_buf:
        out.append(b)
    return out^


def _spv_inst(op: UInt32, wc: Int) -> UInt32:
    return (UInt32(wc) << 16) | op


def test_r6_partition_a_fixes() raises:
    # 1. Cyclic BVH8 slab rejection in decode_bvh8_collider_slab & bounded traversal in Bvh8MeshCollider
    var cyc_col = Bvh8MeshCollider()
    cyc_col.root_node = 0
    var n0 = Bvh8Node()
    n0.min_x[0] = Float32(-1.0); n0.min_y[0] = Float32(-1.0); n0.min_z[0] = Float32(-1.0)
    n0.max_x[0] = Float32(1.0); n0.max_y[0] = Float32(1.0); n0.max_z[0] = Float32(1.0)
    n0.child_indices[0] = Int32(1)
    n0.child_counts[0] = Int32(0)
    var n1 = Bvh8Node()
    n1.min_x[0] = Float32(-1.0); n1.min_y[0] = Float32(-1.0); n1.min_z[0] = Float32(-1.0)
    n1.max_x[0] = Float32(1.0); n1.max_y[0] = Float32(1.0); n1.max_z[0] = Float32(1.0)
    n1.child_indices[0] = Int32(1)  # self-loop 1 -> 1
    n1.child_counts[0] = Int32(0)
    cyc_col.nodes.append(n0)
    cyc_col.nodes.append(n1)

    var cyc_bytes = encode_bvh8_collider_slab(cyc_col)
    assert_false(decode_bvh8_collider_slab(cyc_bytes)[0])

    # Mutual 2-cycle 1 -> 2 -> 1
    var mut_col = Bvh8MeshCollider()
    mut_col.root_node = 0
    var ma = Bvh8Node()
    ma.child_indices[0] = Int32(1); ma.child_counts[0] = Int32(0)
    var mb = Bvh8Node()
    mb.child_indices[0] = Int32(2); mb.child_counts[0] = Int32(0)
    var mc = Bvh8Node()
    mc.child_indices[0] = Int32(1); mc.child_counts[0] = Int32(0)
    mut_col.nodes.append(ma); mut_col.nodes.append(mb); mut_col.nodes.append(mc)
    assert_false(decode_bvh8_collider_slab(encode_bvh8_collider_slab(mut_col))[0])

    # Manual cyclic Bvh8MeshCollider traversal must terminate cleanly without hanging
    var rh_cyc = cyc_col.raycast(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 20.0)
    assert_false(rh_cyc.hit)
    var sw_cyc = cyc_col.sweep_sphere(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0), 0.5, 20.0)
    assert_false(sw_cyc.hit)
    var q_tris = List[Int]()
    cyc_col._query_aabb_triangles(Vec3(-2.0, -2.0, -2.0), Vec3(2.0, 2.0, 2.0), q_tris)
    assert_equal(len(q_tris), 0)

    # Valid BVH8 round-trip still works
    var ok_verts: List[Vec3] = [Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0)]
    var ok_tris: List[Int32] = [Int32(0), Int32(1), Int32(2)]
    var bvh_ok = Bvh8MeshCollider.build(ok_verts, ok_tris)
    var dec_bvh = decode_bvh8_collider_slab(encode_bvh8_collider_slab(bvh_ok))
    assert_true(dec_bvh[0])
    assert_true(dec_bvh[1].raycast(Vec3(0.2, 0.2, 1.0), Vec3(0.0, 0.0, -1.0), 5.0).hit)

    # 2. Out-of-range GLB vertex index rejection (small <=124 tris & large >124 tris) + direct cook/meshlet/bvh guards
    var meta = AssetMeta()
    meta.uuid = AssetUuid(1, 2)
    var bad_glb_big = _build_test_glb(130, 999999)
    assert_false(parse_native_glb(bad_glb_big).loaded())
    assert_false(cook_single_source_bytes(bad_glb_big, meta, "bad_big.glb").valid)

    var bad_glb_small = _build_test_glb(10, 999999)
    assert_false(parse_native_glb(bad_glb_small).loaded())
    assert_false(cook_single_source_bytes(bad_glb_small, meta, "bad_small.glb").valid)

    var oob_mesh = NativeMesh()
    oob_mesh.positions = [Float32(0.0), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    for _ in range(130):
        oob_mesh.indices.append(999999)
        oob_mesh.indices.append(1)
        oob_mesh.indices.append(2)
    var oob_mdl = NativeModel()
    oob_mdl.meshes.append(oob_mesh.copy())
    assert_false(cook_native_model_bundle(oob_mdl, meta, "oob.glb").valid)
    var pool = AaaGeometryPool()
    _ = pool.append_mesh(oob_mesh, 0, build_lods=False)
    assert_equal(pool.total_triangles(), 0)
    var bvh_oob = Bvh8MeshCollider.build(ok_verts, [Int32(999999), Int32(1), Int32(2)])
    assert_equal(len(bvh_oob.triangles), 0)

    var ok_glb = _build_test_glb(130, -1)
    var ok_mdl = parse_native_glb(ok_glb)
    assert_true(ok_mdl.loaded())
    var ok_bundle = cook_single_source_bytes(ok_glb, meta, "ok.glb")
    assert_true(ok_bundle.valid)

    # 3. Truncated 8000x5000 SOF2/SOF0 JPEG rejection without multi-hundred-MB allocation + valid round-trip
    var sof2_trunc: List[UInt8] = [
        UInt8(0xFF), UInt8(0xD8),
        UInt8(0xFF), UInt8(0xC2), UInt8(0x00), UInt8(17),
        UInt8(8),
        UInt8(0x13), UInt8(0x88),  # height = 5000
        UInt8(0x1F), UInt8(0x40),  # width = 8000
        UInt8(3),
        UInt8(1), UInt8(0x44), UInt8(0),
        UInt8(2), UInt8(0x44), UInt8(1),
        UInt8(3), UInt8(0x44), UInt8(1),
    ]
    assert_false(decode_jpeg(sof2_trunc).valid)

    # Even with a fake SOS header attached, total bytes < blocks/8 must reject before allocating
    var sof2_fake_sos = sof2_trunc.copy()
    sof2_fake_sos.append(UInt8(0xFF)); sof2_fake_sos.append(UInt8(0xDA))
    sof2_fake_sos.append(UInt8(0x00)); sof2_fake_sos.append(UInt8(8))
    sof2_fake_sos.append(UInt8(1)); sof2_fake_sos.append(UInt8(1)); sof2_fake_sos.append(UInt8(0x00))
    sof2_fake_sos.append(UInt8(1)); sof2_fake_sos.append(UInt8(63)); sof2_fake_sos.append(UInt8(0x00))
    sof2_fake_sos.append(UInt8(0x00))
    assert_false(decode_jpeg(sof2_fake_sos).valid)

    var jpg_px: List[UInt32] = [
        0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF,
        0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF,
        0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF,
        0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFFFF,
    ]
    var jpg_ok = decode_jpeg(encode_jpeg_baseline(jpg_px, 4, 4, 90))
    assert_true(jpg_ok.valid and jpg_ok.width == 4 and jpg_ok.height == 4)

    # 4. FlintReplayStream.from_reader pre-bounds check before reserve() + valid V3 round-trip
    var bw_rep = BlobWriter()
    bw_rep.begin(REPLAY_STREAM_TAG, REPLAY_STREAM_VERSION)
    bw_rep.text("adv")
    bw_rep.u64(0)
    bw_rep.u32(UInt32(100_000_000))  # init_len claims 100M bytes
    bw_rep.end()
    var rep_raised = False
    try:
        _ = FlintReplayStream.unpack_bytes(lz4_pack(bw_rep.data))
    except:
        rep_raised = True
    assert_true(rep_raised)

    var bw_rep2 = BlobWriter()
    bw_rep2.begin(REPLAY_STREAM_TAG, REPLAY_STREAM_VERSION)
    bw_rep2.text("adv2")
    bw_rep2.u64(0)
    bw_rep2.u32(UInt32(0))  # init_len = 0
    bw_rep2.u32(UInt32(50_000_000))  # count = 50M
    bw_rep2.end()
    var rep2_raised = False
    try:
        _ = FlintReplayStream.unpack_bytes(lz4_pack(bw_rep2.data))
    except:
        rep2_raised = True
    assert_true(rep2_raised)

    var strm_ok = FlintReplayStream()
    var w_init = BlobWriter()
    w_init.begin(fourcc("TEST"), 1)
    w_init.integer(77)
    w_init.end()
    strm_ok.begin_recording("demo", w_init)
    strm_ok.record_tick(Float32(1.0 / 60.0), [UInt8(1), UInt8(2)])
    strm_ok.add_camera_knot(0, Vec3(0, 2, -5), Vec3(0, 0, 0))
    var packed_strm = strm_ok.pack_bytes()
    var strm_dec = FlintReplayStream.unpack_bytes(packed_strm)
    assert_equal(strm_dec.app_name, "demo")
    assert_equal(strm_dec.tick_count(), 1)
    assert_equal(strm_dec.camera_knot_count(), 1)


def test_r6_partition_b_fixes() raises:
    # 5 & 6. Tensor constructors & ml.ops overflow guards (transpose, concat, gather) + valid round-trips
    var t_bad_f = Tensor.floats([1 << 32, 1 << 32], List[Float32]())
    assert_equal(t_bad_f.numel(), 0)
    var t_bad_short = Tensor.floats([2, 3], [Float32(1.0)])
    assert_equal(t_bad_short.numel(), 0)
    var t_bad_i = Tensor.ints(DT_I64, [1 << 32, 1 << 32], List[Int]())
    assert_equal(t_bad_i.numel(), 0)
    var t_bad_z = Tensor.zeros([1 << 32, 1 << 32])
    assert_equal(t_bad_z.numel(), 0)
    var dummy_f = Float32(1.0)
    var dummy_ptr = Int(Pointer(to=dummy_f))
    var t_bad_b = Tensor.borrowed([1 << 32, 1 << 32], dummy_ptr)
    assert_equal(t_bad_b.numel(), 0)
    assert_equal(Tensor.borrowed([4], 0).numel(), 0)

    # Manually crafted overflowing Tensor (bypassing constructor) must raise Error in transpose
    var t_ovf = Tensor()
    t_ovf.shape = [1 << 32, 1 << 32]
    t_ovf.ext = dummy_ptr
    var tr_raised = False
    try:
        _ = transpose(t_ovf, [1, 0])
    except:
        tr_raised = True
    assert_true(tr_raised)

    # Valid positive-numel inputs whose concat sum overflows Int64 must raise Error cleanly
    var big_a = Tensor.borrowed([1 << 62, 1], dummy_ptr)
    var big_b = Tensor.borrowed([1 << 62, 1], dummy_ptr)
    assert_true(big_a.numel() > 0)
    var concat_ovf_raised = False
    try:
        var xs = List[Tensor]()
        xs.append(big_a^)
        xs.append(big_b^)
        _ = concat(xs, 0)
    except:
        concat_ovf_raised = True
    assert_true(concat_ovf_raised)

    var gather_data = Tensor.borrowed([1 << 32, 1 << 30], dummy_ptr)
    var gather_idx = Tensor.ints(DT_I64, [1 << 32], List[Int](length=0, fill=0))
    _ = gather_idx
    var gather_data_small = Tensor.floats([2, 2], [Float32(1), 2, 3, 4])
    var gather_idx_ovf = Tensor()
    gather_idx_ovf.dtype = DT_I64
    gather_idx_ovf.shape = [1 << 62]
    var gather_raised = False
    try:
        _ = gather(gather_data, gather_idx_ovf, 0)
    except:
        gather_raised = True
    assert_true(gather_raised)
    var gather_raised2 = False
    try:
        _ = gather(gather_data_small, gather_idx_ovf, 0)
    except:
        gather_raised2 = True
    assert_true(gather_raised2)

    # Valid concat & gather round-trip
    var ca = Tensor.floats([1, 2], [Float32(1.0), 2.0])
    var cb = Tensor.floats([1, 2], [Float32(3.0), 4.0])
    var c_list = List[Tensor]()
    c_list.append(ca^); c_list.append(cb^)
    var cc = concat(c_list, 0)
    assert_equal(cc.shape[0], 2)
    assert_equal(cc.f[3], Float32(4.0))
    var g_out = gather(cc, Tensor.ints(DT_I64, [1], [1]), 0)
    assert_equal(g_out.f[0], Float32(3.0))

    # 7 & 10. spirv_validate wc < 2 (OpTypeVoid / OpLabel) & O(num_b) dominator scaling (30k blocks)
    var spv_bad_void: List[UInt32] = [
        SPIRV_MAGIC, 0x00010300, 0, 8, 0,
        _spv_inst(OP_TYPE_VOID, 1),
    ]
    assert_false(validate_spirv_module(spv_bad_void).valid)

    var spv_bad_label: List[UInt32] = [
        SPIRV_MAGIC, 0x00010300, 0, 8, 0,
        _spv_inst(OP_CAPABILITY, 2), CAPABILITY_SHADER,
        _spv_inst(OP_MEMORY_MODEL, 3), 0, 1,
        _spv_inst(OP_TYPE_VOID, 2), 1,
        _spv_inst(OP_TYPE_FUNCTION, 3), 2, 1,
        _spv_inst(OP_FUNCTION, 5), 1, 3, 0, 2,
        _spv_inst(OP_LABEL, 1),
        _spv_inst(OP_RETURN, 1),
        _spv_inst(OP_FUNCTION_END, 1),
    ]
    assert_false(validate_spirv_module(spv_bad_label).valid)

    # Valid minimal SPIR-V module + 30,000-block chain (must stay O(num_b) memory and pass cleanly)
    var spv_ok: List[UInt32] = [
        SPIRV_MAGIC, 0x00010300, 0, 16, 0,
        _spv_inst(OP_CAPABILITY, 2), CAPABILITY_SHADER,
        _spv_inst(OP_MEMORY_MODEL, 3), 0, 1,
        _spv_inst(OP_ENTRY_POINT, 4), 5, 3, 0x00006E69,  # "in\0\0"
        _spv_inst(OP_TYPE_VOID, 2), 1,
        _spv_inst(OP_TYPE_FUNCTION, 3), 2, 1,
        _spv_inst(OP_FUNCTION, 5), 1, 3, 0, 2,
        _spv_inst(OP_LABEL, 2), 4,
        _spv_inst(OP_RETURN, 1),
        _spv_inst(OP_FUNCTION_END, 1),
    ]
    assert_true(validate_spirv_module(spv_ok).valid)

    var nb = 30000
    var spv_big = List[UInt32](capacity=30 + nb * 4)
    spv_big.append(SPIRV_MAGIC); spv_big.append(0x00010300); spv_big.append(0); spv_big.append(UInt32(10 + nb)); spv_big.append(0)
    spv_big.append(_spv_inst(OP_CAPABILITY, 2)); spv_big.append(CAPABILITY_SHADER)
    spv_big.append(_spv_inst(OP_MEMORY_MODEL, 3)); spv_big.append(0); spv_big.append(1)
    spv_big.append(_spv_inst(OP_ENTRY_POINT, 4)); spv_big.append(5); spv_big.append(3); spv_big.append(0x00006E69)
    spv_big.append(_spv_inst(OP_TYPE_VOID, 2)); spv_big.append(1)
    spv_big.append(_spv_inst(OP_TYPE_FUNCTION, 3)); spv_big.append(2); spv_big.append(1)
    spv_big.append(_spv_inst(OP_FUNCTION, 5)); spv_big.append(1); spv_big.append(3); spv_big.append(0); spv_big.append(2)
    for bi in range(nb):
        var lbl = UInt32(4 + bi)
        spv_big.append(_spv_inst(OP_LABEL, 2)); spv_big.append(lbl)
        if bi + 1 < nb:
            spv_big.append(_spv_inst(OP_BRANCH, 2)); spv_big.append(UInt32(4 + bi + 1))
        else:
            spv_big.append(_spv_inst(OP_RETURN, 1))
    spv_big.append(_spv_inst(OP_FUNCTION_END, 1))
    var rep_big = validate_spirv_module(spv_big)
    assert_true(rep_big.valid)
    assert_equal(rep_big.block_count, nb)

    # 8. NavMesh.deserialize cross-index validation + valid round-trip
    var nm_src = NavMesh()
    nm_src.tiles_x = 1; nm_src.tiles_z = 1; nm_src.tile_size = 16
    var t0_nav = NavTile(0, 0, 0, 1, 0, 1)
    t0_nav.poly_ids.append(0); t0_nav.poly_ids.append(1)
    nm_src.tiles.append(t0_nav^)
    var p_a = NavPoly()
    p_a.verts = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 0, 1), Vec3(0, 0, 1)]
    p_a.center = Vec3(0.5, 0.0, 0.5); p_a.plane_n = Vec3(0, 1, 0)
    p_a.neighbors.append(1); p_a.portal_left.append(Vec3(1, 0, 0)); p_a.portal_right.append(Vec3(1, 0, 1))
    var p_b = NavPoly()
    p_b.verts = [Vec3(1, 0, 0), Vec3(2, 0, 0), Vec3(2, 0, 1), Vec3(1, 0, 1)]
    p_b.center = Vec3(1.5, 0.0, 0.5); p_b.plane_n = Vec3(0, 1, 0)
    p_b.neighbors.append(0); p_b.portal_left.append(Vec3(1, 0, 1)); p_b.portal_right.append(Vec3(1, 0, 0))
    nm_src.polys.append(p_a^); nm_src.polys.append(p_b^)
    nm_src.poly_owner_tile = [0, 0]

    var bw_nm_ok = BlobWriter()
    nm_src.serialize(bw_nm_ok)
    var nm_dec = NavMesh()
    var br_nm_ok = BlobReader(bw_nm_ok.data.copy())
    nm_dec.deserialize(br_nm_ok)
    assert_equal(len(nm_dec.polys), 2)

    # Now craft a NavMesh with OOB neighbor 9999 -> must raise Error
    nm_src.polys[0].neighbors[0] = 9999
    var bw_nm_bad = BlobWriter()
    nm_src.serialize(bw_nm_bad)
    var nm_bad = NavMesh()
    var br_nm_bad = BlobReader(bw_nm_bad.data.copy())
    var nm_raised = False
    try:
        nm_bad.deserialize(br_nm_bad)
    except:
        nm_raised = True
    assert_true(nm_raised)

    # 9. PhysicsWorld.deserialize RigidBody.shape_index validation + valid hull/compound round-trip
    var pw_ok = PhysicsWorld()
    var h_ok = ConvexHull.build_quickhull([Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1)])
    var c_ok = CompoundShape()
    c_ok.add_child(CompoundChild.convex(Vec3(), Quat(), h_ok))
    _ = pw_ok.add_convex_hull(Vec3(0, 1, 0), h_ok, Float32(1.0))
    _ = pw_ok.add_compound(Vec3(2, 1, 0), c_ok, Float32(1.0))
    var bw_pw_ok = BlobWriter()
    pw_ok.serialize(bw_pw_ok)
    var pw_dec = PhysicsWorld()
    var br_pw_ok = BlobReader(bw_pw_ok.data.copy())
    pw_dec.deserialize(br_pw_ok)
    assert_equal(len(pw_dec.bodies), 2)
    assert_equal(pw_dec.bodies[0].shape, BODY_SHAPE_CONVEX_HULL)
    assert_equal(pw_dec.bodies[1].shape, BODY_SHAPE_COMPOUND)
    pw_dec.step(Float32(1.0 / 60.0))

    # OOB convex hull shape_index = 9999
    pw_ok.bodies[0].shape_index = 9999
    var bw_pw_bad_h = BlobWriter()
    pw_ok.serialize(bw_pw_bad_h)
    var br_pw_bad_h = BlobReader(bw_pw_bad_h.data.copy())
    var pw_h_raised = False
    try:
        var pw_tmp = PhysicsWorld()
        pw_tmp.deserialize(br_pw_bad_h)
    except:
        pw_h_raised = True
    assert_true(pw_h_raised)

    # OOB compound shape_index = 9999
    pw_ok.bodies[0].shape_index = 0
    pw_ok.bodies[1].shape_index = 9999
    var bw_pw_bad_c = BlobWriter()
    pw_ok.serialize(bw_pw_bad_c)
    var br_pw_bad_c = BlobReader(bw_pw_bad_c.data.copy())
    var pw_c_raised = False
    try:
        var pw_tmp2 = PhysicsWorld()
        pw_tmp2.deserialize(br_pw_bad_c)
    except:
        pw_c_raised = True
    assert_true(pw_c_raised)

    # 11. flint.cycles unbounded range & CyclesEngine(sample_rate<=0)
    var cyc_range_raised = False
    try:
        _ = mini("0..1000000000")
    except:
        cyc_range_raised = True
    assert_true(cyc_range_raised)
    var ok_pat = mini("0..3")
    assert_equal(len(ok_pat.onsets(Frac.of(0), Frac.of(1))), 4)

    var ce0 = CyclesEngine(sample_rate=0)
    var ce_neg = CyclesEngine(sample_rate=-44100)
    assert_equal(len(ce0.render(64)), 128)
    assert_equal(len(ce_neg.render(64)), 128)


def test_r6b_remediation_fixes() raises:
    # 1. NavMesh.deserialize must enforce n_own == n_polys + valid round-trip with carve_obstacles()
    var wn = PhysicsWorld(gravity=0.0)
    _ = wn.add_box(Vec3(0.0, -0.25, 0.0), Vec3(6.0, 0.25, 6.0), mass=0.0)
    wn.rebuild_broadphase()
    var ncfg = NavMeshConfig()
    ncfg.cell_size = 0.5
    ncfg.tile_size = 8
    var nav_ok = NavMesh.build_from_physics(wn, ncfg)
    assert_true(nav_ok.poly_count() > 0)
    assert_equal(len(nav_ok.poly_owner_tile), len(nav_ok.polys))

    var bw_nav_ok = BlobWriter()
    nav_ok.serialize(bw_nav_ok)
    var nav_dec = NavMesh()
    var br_nav_ok = BlobReader(bw_nav_ok.data.copy())
    nav_dec.deserialize(br_nav_ok)
    assert_equal(len(nav_dec.poly_owner_tile), len(nav_dec.polys))
    nav_dec.carve_obstacles()
    assert_true(nav_dec.poly_count() > 0)
    assert_equal(len(nav_dec.poly_owner_tile), len(nav_dec.polys))

    # Mismatched n_own = 0 when n_polys > 0 must raise Error
    var nav_empty_own = nav_ok.copy()
    nav_empty_own.poly_owner_tile.clear()
    var bw_nav_0 = BlobWriter()
    nav_empty_own.serialize(bw_nav_0)
    var br_nav_0 = BlobReader(bw_nav_0.data.copy())
    var nav0_raised = False
    try:
        var tmp = NavMesh()
        tmp.deserialize(br_nav_0)
    except:
        nav0_raised = True
    assert_true(nav0_raised)

    # Mismatched n_own > n_polys must also raise Error
    var nav_extra_own = nav_ok^
    nav_extra_own.poly_owner_tile.append(0)
    var bw_nav_extra = BlobWriter()
    nav_extra_own.serialize(bw_nav_extra)
    var br_nav_extra = BlobReader(bw_nav_extra.data.copy())
    var nav_extra_raised = False
    try:
        var tmp = NavMesh()
        tmp.deserialize(br_nav_extra)
    except:
        nav_extra_raised = True
    assert_true(nav_extra_raised)

    # 2. ml.ops.{topk, matmul, resize_nearest, range_op, clip, layer_norm} tensor validation
    var dummy_f = Float32(1.0)
    var dummy_ptr = Int(Pointer(to=dummy_f))
    var t_ovf = Tensor()
    t_ovf.shape = [1 << 32, 1 << 32]
    t_ovf.ext = dummy_ptr
    var t_bad = Tensor()
    t_bad.shape = [2, 2]
    var t_ok = Tensor.floats([2, 2], [Float32(1.0), 2.0, 3.0, 4.0])

    # topk rejects t_ovf and invalid-storage t_bad; valid topk succeeds
    var topk_ovf_raised = False
    try:
        _ = topk(t_ovf, 1, 0, True)
    except:
        topk_ovf_raised = True
    assert_true(topk_ovf_raised)

    var topk_bad_raised = False
    try:
        _ = topk(t_bad, 1, 0, True)
    except:
        topk_bad_raised = True
    assert_true(topk_bad_raised)

    var topk_ok = topk(t_ok, 1, 1, True)
    assert_equal(topk_ok[0].fget(0), Float32(2.0))
    assert_equal(topk_ok[0].fget(1), Float32(4.0))

    # matmul rejects invalid-storage, overflowing, and integer tensors; valid matmul succeeds
    var pool = JobPool()
    var mm_bad_raised = False
    try:
        _ = matmul(pool, t_bad, t_ok)
    except:
        mm_bad_raised = True
    assert_true(mm_bad_raised)

    var mm_ovf_raised = False
    try:
        _ = matmul(pool, t_ovf, t_ok)
    except:
        mm_ovf_raised = True
    assert_true(mm_ovf_raised)

    var t_int = Tensor.ints(DT_I64, [2, 2], [1, 2, 3, 4])
    var mm_int_raised = False
    try:
        _ = matmul(pool, t_int, t_ok)
    except:
        mm_int_raised = True
    assert_true(mm_int_raised)

    var mm_ok = matmul(pool, t_ok, t_ok)
    assert_equal(mm_ok.fget(0), Float32(7.0))
    assert_equal(mm_ok.fget(3), Float32(22.0))

    # resize_nearest rejects empty input -> non-empty output, allows empty -> empty and valid resize
    var t_empty = Tensor.floats([0, 3], List[Float32]())
    var rn_empty_raised = False
    try:
        _ = resize_nearest(t_empty, [2, 3])
    except:
        rn_empty_raised = True
    assert_true(rn_empty_raised)

    var rn_empty_ok = resize_nearest(t_empty, [0, 3])
    assert_equal(rn_empty_ok.numel(), 0)
    var rn_ok = resize_nearest(Tensor.floats([1, 2], [Float32(5.0), 9.0]), [2, 4])
    assert_equal(rn_ok.numel(), 8)
    assert_equal(rn_ok.fget(0), Float32(5.0))
    assert_equal(rn_ok.fget(7), Float32(9.0))

    # range_op, clip, layer_norm reject invalid-storage tensors
    var t_bad_scalar = Tensor()
    t_bad_scalar.shape = [1]
    var range_bad_raised = False
    try:
        _ = range_op(t_bad_scalar, Tensor.floats([1], [4.0]), Tensor.floats([1], [1.0]))
    except:
        range_bad_raised = True
    assert_true(range_bad_raised)

    var clip_bad_raised = False
    try:
        _ = clip(t_bad, 0.0, 1.0)
    except:
        clip_bad_raised = True
    assert_true(clip_bad_raised)

    var ln_bad_raised = False
    try:
        _ = layer_norm(t_ok, Tensor.floats([2], [1.0, 1.0]), t_bad, True, Float32(1e-5))
    except:
        ln_bad_raised = True
    assert_true(ln_bad_raised)


def test_r6c_remediation_fixes() raises:
    var pool = JobPool()
    var empty_f = Tensor()

    # 1. ml.ops.gemm: rank == 2, inner dim k match, and bias C broadcast compatibility
    var g_a = Tensor.floats([2, 3], [Float32(1), 2, 3, 4, 5, 6])
    var g_b_bad_k = Tensor.floats([1, 4], [Float32(1), 2, 3, 4])
    var gemm_k_raised = False
    try:
        _ = gemm(pool, g_a, g_b_bad_k, empty_f, False, 1.0, 1.0, False, False)
    except:
        gemm_k_raised = True
    assert_true(gemm_k_raised)

    var gemm_tb_raised = False
    try:
        _ = gemm(pool, g_a, Tensor.floats([4, 2], [Float32(1), 2, 3, 4, 5, 6, 7, 8]), empty_f, False, 1.0, 1.0, False, True)
    except:
        gemm_tb_raised = True
    assert_true(gemm_tb_raised)

    var gemm_big_raised = False
    try:
        _ = gemm(pool, Tensor.zeros([1, 10_000_000]), Tensor.zeros([1, 1]), empty_f, False, 1.0, 1.0, False, False)
    except:
        gemm_big_raised = True
    assert_true(gemm_big_raised)

    var gemm_rank3_raised = False
    try:
        _ = gemm(pool, Tensor.zeros([1, 2, 3]), Tensor.zeros([3, 2]), empty_f, False, 1.0, 1.0, False, False)
    except:
        gemm_rank3_raised = True
    assert_true(gemm_rank3_raised)

    var g_b_ok = Tensor.floats([3, 2], [Float32(1), 0, 0, 1, 1, 1])
    var gemm_c_bad_raised = False
    try:
        _ = gemm(pool, g_a, g_b_ok, Tensor.floats([3], [1.0, 2.0, 3.0]), True, 1.0, 1.0, False, False)
    except:
        gemm_c_bad_raised = True
    assert_true(gemm_c_bad_raised)

    var gemm_c3d_raised = False
    try:
        _ = gemm(pool, g_a, g_b_ok, Tensor.zeros([1, 2, 2]), True, 1.0, 1.0, False, False)
    except:
        gemm_c3d_raised = True
    assert_true(gemm_c3d_raised)

    # Column bias [2, 1] when m == n == 2 must broadcast across columns, not act as row bias
    var g_col = gemm(pool, g_a, g_b_ok, Tensor.floats([2, 1], [10.0, 20.0]), True, 1.0, 1.0, False, False)
    assert_equal(g_col.fget(0), Float32(14.0))
    assert_equal(g_col.fget(1), Float32(15.0))
    assert_equal(g_col.fget(2), Float32(30.0))
    assert_equal(g_col.fget(3), Float32(31.0))

    var g_row = gemm(pool, g_a, g_b_ok, Tensor.floats([1, 2], [10.0, 20.0]), True, 1.0, 1.0, False, False)
    assert_equal(g_row.fget(0), Float32(14.0))
    assert_equal(g_row.fget(1), Float32(25.0))

    # 2. ml.ops.conv1d: w.shape[1] == cin, dil >= 1, pad >= 0, kw >= 1, 1-D bias len == cout
    var c_x = Tensor.floats([1, 4, 5], List[Float32](length=20, fill=1.0))
    var c_w_bad = Tensor.floats([2, 1, 3], List[Float32](length=6, fill=1.0))
    var conv_cin_raised = False
    try:
        _ = conv1d(pool, c_x, c_w_bad, empty_f, False, 0, 0, 1, 1, 1)
    except:
        conv_cin_raised = True
    assert_true(conv_cin_raised)

    var c_w_ok = Tensor.floats([2, 4, 3], List[Float32](length=24, fill=1.0))
    var conv_b_raised = False
    try:
        _ = conv1d(pool, c_x, c_w_ok, Tensor.floats([1], [5.0]), True, 0, 0, 1, 1, 1)
    except:
        conv_b_raised = True
    assert_true(conv_b_raised)

    var conv_dil_raised = False
    try:
        _ = conv1d(pool, c_x, c_w_ok, empty_f, False, 0, 0, 0, 1, 1)
    except:
        conv_dil_raised = True
    assert_true(conv_dil_raised)

    var conv_pad_raised = False
    try:
        _ = conv1d(pool, c_x, c_w_ok, empty_f, False, -1, 0, 1, 1, 1)
    except:
        conv_pad_raised = True
    assert_true(conv_pad_raised)

    var conv_kw_raised = False
    try:
        _ = conv1d(pool, c_x, Tensor.floats([2, 4, 0], List[Float32]()), empty_f, False, 0, 0, 1, 1, 1)
    except:
        conv_kw_raised = True
    assert_true(conv_kw_raised)

    var c_ok = conv1d(pool, c_x, c_w_ok, Tensor.floats([2], [1.0, 2.0]), True, 0, 0, 1, 1, 1)
    assert_equal(c_ok.shape[0], 1)
    assert_equal(c_ok.shape[1], 2)
    assert_equal(c_ok.shape[2], 3)
    assert_equal(c_ok.fget(0), Float32(13.0))
    assert_equal(c_ok.fget(3), Float32(14.0))

    # 3. ml.ops.slice & ml.ops.reduce: axis bounds, metadata lengths, duplicate axes, negative step
    var sl_x = Tensor.floats([2, 3], [Float32(1), 2, 3, 4, 5, 6])
    var sl_oob_raised = False
    try:
        _ = slice(sl_x, [0], [1], [99], [1])
    except:
        sl_oob_raised = True
    assert_true(sl_oob_raised)

    var sl_len_raised = False
    try:
        _ = slice(sl_x, [0, 0], [1], List[Int](), List[Int]())
    except:
        sl_len_raised = True
    assert_true(sl_len_raised)

    var sl_dup_raised = False
    try:
        _ = slice(sl_x, [0, 0], [1, 1], [0, -2], [1, 1])
    except:
        sl_dup_raised = True
    assert_true(sl_dup_raised)

    var sl_ok = slice(sl_x, [2], [0], [1], [-1])
    assert_equal(sl_ok.shape[0], 2)
    assert_equal(sl_ok.shape[1], 2)
    assert_equal(sl_ok.fget(0), Float32(3.0))
    assert_equal(sl_ok.fget(1), Float32(2.0))

    var red_oob_raised = False
    try:
        _ = reduce(sl_x, [99], False, False)
    except:
        red_oob_raised = True
    assert_true(red_oob_raised)

    var red_ok = reduce(sl_x, [1], False, False)
    assert_equal(red_ok.fget(0), Float32(6.0))
    assert_equal(red_ok.fget(1), Float32(15.0))

    # 4. ml.ops.argmax: empty reduction axis
    var am_empty_raised = False
    try:
        _ = argmax(Tensor.floats([2, 0], List[Float32]()), 1, False)
    except:
        am_empty_raised = True
    assert_true(am_empty_raised)

    var am_ok = argmax(sl_x, 1, False)
    assert_equal(am_ok.i[0], 2)
    assert_equal(am_ok.i[1], 2)

    # 5. ml.ops.gather_elements & scatter_elements: off-axis shape bounds & updates shape match
    var ge_data = Tensor.floats([2, 2], [Float32(10), 20, 30, 40])
    var ge_bad_idx = Tensor.ints(DT_I64, [3, 2], [0, 1, 0, 1, 0, 1])
    var ge_oob_raised = False
    try:
        _ = gather_elements(ge_data, ge_bad_idx, 1)
    except:
        ge_oob_raised = True
    assert_true(ge_oob_raised)

    var se_oob_raised = False
    try:
        _ = scatter_elements(ge_data, ge_bad_idx, Tensor.zeros([3, 2]), 1)
    except:
        se_oob_raised = True
    assert_true(se_oob_raised)

    var se_upd_raised = False
    try:
        _ = scatter_elements(ge_data, Tensor.ints(DT_I64, [2, 1], [1, 0]), Tensor.floats([2], [99.0, 88.0]), 1)
    except:
        se_upd_raised = True
    assert_true(se_upd_raised)

    var ge_ok = gather_elements(ge_data, Tensor.ints(DT_I64, [2, 1], [1, 0]), 1)
    assert_equal(ge_ok.fget(0), Float32(20.0))
    assert_equal(ge_ok.fget(1), Float32(30.0))
    var se_ok = scatter_elements(ge_data, Tensor.ints(DT_I64, [2, 1], [1, 0]), Tensor.floats([2, 1], [99.0, 88.0]), 1)
    assert_equal(se_ok.fget(1), Float32(99.0))
    assert_equal(se_ok.fget(2), Float32(88.0))

    # 6. ml.ops.range_op: non-finite & extreme allocation cap
    var r_nan = Tensor.floats([1], [f32_from_bits(0x7FC00000)])
    var r_inf = Tensor.floats([1], [f32_from_bits(0x7F800000)])
    var rng_nan_raised = False
    try:
        _ = range_op(Tensor.floats([1], [0.0]), r_nan, Tensor.floats([1], [1.0]))
    except:
        rng_nan_raised = True
    assert_true(rng_nan_raised)

    var rng_inf_raised = False
    try:
        _ = range_op(Tensor.floats([1], [0.0]), r_inf, Tensor.floats([1], [1.0]))
    except:
        rng_inf_raised = True
    assert_true(rng_inf_raised)

    var rng_cap_raised = False
    try:
        _ = range_op(Tensor.floats([1], [0.0]), Tensor.floats([1], [1e12]), Tensor.floats([1], [1.0]))
    except:
        rng_cap_raised = True
    assert_true(rng_cap_raised)

    var rng_ok = range_op(Tensor.floats([1], [1.0]), Tensor.floats([1], [4.0]), Tensor.floats([1], [1.0]))
    assert_equal(rng_ok.numel(), 3)
    assert_equal(rng_ok.fget(2), Float32(3.0))

    # 7. ml.ops.binary integer div/mod by zero
    var b_div0_raised = False
    try:
        _ = binary(B_DIV, Tensor.ints(DT_I64, [2], [4, 5]), Tensor.ints(DT_I64, [2], [2, 0]))
    except:
        b_div0_raised = True
    assert_true(b_div0_raised)

    var b_mod0_raised = False
    try:
        _ = binary(B_MOD, Tensor.ints(DT_I64, [2], [4, 5]), Tensor.ints(DT_I64, [2], [2, 0]))
    except:
        b_mod0_raised = True
    assert_true(b_mod0_raised)


def _r7_put_u16_le(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))


def _r7_put_u32_le(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


def _r7_put_u64_le(mut out: List[UInt8], v: UInt64):
    for i in range(8):
        out.append(UInt8((v >> UInt64(i * 8)) & 0xFF))


def _r7_put_u16_be(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _r7_put_u32_be(mut out: List[UInt8], v: Int):
    out.append(UInt8((v >> 24) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8(v & 0xFF))


def _r7_pb_varint(mut out: List[UInt8], var v: Int):
    while True:
        var b = UInt8(v & 0x7F)
        v >>= 7
        if v > 0:
            out.append(b | UInt8(0x80))
        else:
            out.append(b)
            break


def _r7_pb_bytes(mut out: List[UInt8], field: Int, payload: List[UInt8]):
    _r7_pb_varint(out, (field << 3) | 2)
    _r7_pb_varint(out, len(payload))
    for b in payload:
        out.append(b)


def _r7_pb_str(mut out: List[UInt8], field: Int, s: String):
    var bs = List[UInt8]()
    for b in s.as_bytes():
        bs.append(b)
    _r7_pb_bytes(out, field, bs)


def test_r7_partition_ab_fixes(tmp_dir: String) raises:
    # 1. KTX2 BasisLZ 16Kx16K level_count=1 DoS & invalid Huffman pre-alloc rejection
    var k_blz = List[UInt8]()
    var k_magic: List[UInt8] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A]
    for b in k_magic:
        k_blz.append(b)
    _r7_put_u32_le(k_blz, 0)       # vkFormat = 0
    _r7_put_u32_le(k_blz, 1)       # typeSize
    _r7_put_u32_le(k_blz, 16384)   # pixelWidth
    _r7_put_u32_le(k_blz, 16384)   # pixelHeight
    _r7_put_u32_le(k_blz, 0)       # pixelDepth
    _r7_put_u32_le(k_blz, 0)       # layerCount
    _r7_put_u32_le(k_blz, 1)       # faceCount
    _r7_put_u32_le(k_blz, 1)       # levelCount = 1
    _r7_put_u32_le(k_blz, 1)       # supercompressionScheme = KTX2_SC_BASIS_LZ
    _r7_put_u32_le(k_blz, 0); _r7_put_u32_le(k_blz, 0)  # dfd
    _r7_put_u32_le(k_blz, 0); _r7_put_u32_le(k_blz, 0)  # kvd
    _r7_put_u64_le(k_blz, 104)     # sgdByteOffset
    _r7_put_u64_le(k_blz, 41)      # sgdByteLength
    # Level 0 index (24 bytes at 80..103)
    _r7_put_u64_le(k_blz, 145)     # byteOffset
    _r7_put_u64_le(k_blz, 1)       # byteLength
    _r7_put_u64_le(k_blz, 0)       # uncompressedByteLength
    # SGD at 104..144 (20 + 20 + 1 = 41 bytes)
    _r7_put_u16_le(k_blz, 1); _r7_put_u16_le(k_blz, 1)  # 1 ep, 1 sel
    _r7_put_u32_le(k_blz, 0); _r7_put_u32_le(k_blz, 0); _r7_put_u32_le(k_blz, 1); _r7_put_u32_le(k_blz, 0)
    _r7_put_u32_le(k_blz, 0); _r7_put_u32_le(k_blz, 0); _r7_put_u32_le(k_blz, 1); _r7_put_u32_le(k_blz, 0); _r7_put_u32_le(k_blz, 0)
    k_blz.append(UInt8(0xFF))      # invalid Huffman table byte
    k_blz.append(UInt8(0x00))      # 1-byte mip slice at 145
    assert_false(decode_ktx2(k_blz, 4096).valid, "16Kx16K level_count=1 BasisLZ must be rejected")

    # 4096x4096 BasisLZ with invalid Huffman table must reject before pixel allocation
    k_blz[20] = UInt8(0x00); k_blz[21] = UInt8(0x10); k_blz[22] = UInt8(0x00); k_blz[23] = UInt8(0x00)
    k_blz[24] = UInt8(0x00); k_blz[25] = UInt8(0x10); k_blz[26] = UInt8(0x00); k_blz[27] = UInt8(0x00)
    assert_false(decode_ktx2(k_blz, 4096).valid, "4Kx4K invalid BasisLZ Huffman must be rejected cleanly")

    # 2. bc_compress: truncated / empty / zero-dim BC7, BC5, BC6H buffers return empty cleanly
    var short_bc = List[UInt8](length=15, fill=0)
    assert_equal(len(decode_bc7_rgba(List[UInt8](), 4096, 4096)), 0)
    assert_equal(len(decode_bc7_rgba(short_bc, 4, 4)), 0)
    assert_equal(len(decode_bc5_normals(List[UInt8](), 64, 64)), 0)
    assert_equal(len(decode_bc5_normals(short_bc, 4, 4)), 0)
    assert_equal(len(decode_bc6h_hdr(List[UInt8](), 64, 64)), 0)
    assert_equal(len(decode_bc6h_hdr(short_bc, 4, 4)), 0)

    # 3. flpk_v3: decode_audio_pcm_slab rejects ch==0, ch>8, sr==0, cnt==0, cnt%ch!=0
    var good_aud = encode_audio_pcm_slab(2, 48000, [Int16(10), Int16(-10), Int16(20), Int16(-20)])
    assert_true(decode_audio_pcm_slab(good_aud)[0])
    var aud_ch0 = good_aud.copy()
    aud_ch0[4] = UInt8(0)
    assert_false(decode_audio_pcm_slab(aud_ch0)[0])
    var aud_ch9 = good_aud.copy()
    aud_ch9[4] = UInt8(9)
    assert_false(decode_audio_pcm_slab(aud_ch9)[0])
    var aud_sr0 = good_aud.copy()
    aud_sr0[8] = UInt8(0); aud_sr0[9] = UInt8(0); aud_sr0[10] = UInt8(0); aud_sr0[11] = UInt8(0)
    assert_false(decode_audio_pcm_slab(aud_sr0)[0])
    var aud_cnt0 = good_aud.copy()
    aud_cnt0[12] = UInt8(0)
    assert_false(decode_audio_pcm_slab(aud_cnt0)[0])
    var aud_mod = encode_audio_pcm_slab(2, 48000, [Int16(10), Int16(-10), Int16(20)])
    assert_false(decode_audio_pcm_slab(aud_mod)[0])

    # 4. ttf.read_axes: bounds-check fvar axisCount against fvar table length
    var sfnt = List[UInt8]()
    _r7_put_u32_be(sfnt, 0x00010000)
    _r7_put_u16_be(sfnt, 1)  # 1 table: fvar
    _r7_put_u16_be(sfnt, 16); _r7_put_u16_be(sfnt, 0); _r7_put_u16_be(sfnt, 0)
    _r7_put_u32_be(sfnt, 0x66766172)  # "fvar"
    _r7_put_u32_be(sfnt, 0)           # checksum
    _r7_put_u32_be(sfnt, 28)          # offset = 12 + 16 = 28
    _r7_put_u32_be(sfnt, 16)          # length = 16
    # fvar header (16 bytes): major=1, minor=0, axesOffset=16, reserved=2, axisCount=65535, axisSize=20
    _r7_put_u16_be(sfnt, 1); _r7_put_u16_be(sfnt, 0)
    _r7_put_u16_be(sfnt, 16); _r7_put_u16_be(sfnt, 2)
    _r7_put_u16_be(sfnt, 65535); _r7_put_u16_be(sfnt, 20)
    _r7_put_u16_be(sfnt, 0); _r7_put_u16_be(sfnt, 0)
    var f_ttf = TrueTypeFont()
    f_ttf.data = sfnt^
    f_ttf.tables[0x66766172] = 28
    f_ttf.lengths[0x66766172] = 16
    f_ttf.read_axes()
    assert_equal(len(f_ttf.axes), 0, "Truncated fvar with axisCount=65535 must yield 0 axes")

    # 5 & 6 & 7. protobuf signed overflow, onnx _parse_tensor overflow, ConstantOfShape cap, GPU _gpu_product
    var pb_ovf: List[UInt8] = [
        UInt8(0x3A),
        UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0x80), UInt8(0xC0), UInt8(0x7F),
    ]
    var pr_ovf = PbReader(Int(pb_ovf.unsafe_ptr()), Int(pb_ovf.unsafe_ptr()) + len(pb_ovf))
    pr_ovf.key()
    var pb_raised = False
    try:
        _ = pr_ovf.length_delimited()
    except:
        pb_raised = True
    _ = len(pb_ovf)
    assert_true(pb_raised)

    var t_ovf = List[UInt8]()
    _r7_pb_varint(t_ovf, (1 << 3) | 0); _r7_pb_varint(t_ovf, 4611686018427387904)
    _r7_pb_varint(t_ovf, (1 << 3) | 0); _r7_pb_varint(t_ovf, 1)
    _r7_pb_varint(t_ovf, (2 << 3) | 0); _r7_pb_varint(t_ovf, DT_F32)
    _r7_pb_str(t_ovf, 8, "w")
    _r7_pb_bytes(t_ovf, 9, List[UInt8]())
    var g_ovf = List[UInt8](); _r7_pb_bytes(g_ovf, 5, t_ovf)
    var m_ovf = List[UInt8](); _r7_pb_bytes(m_ovf, 7, g_ovf)
    var p_ovf = tmp_dir + "/r7_tovf.onnx"
    assert_true(write_bytes(p_ovf, m_ovf))
    var tovf_raised = False
    try:
        _ = OnnxModel.load(p_ovf, 1)
    except:
        tovf_raised = True
    assert_true(tovf_raised)

    var vi_in = List[UInt8](); _r7_pb_str(vi_in, 1, "x")
    var vi_out = List[UInt8](); _r7_pb_str(vi_out, 1, "y")
    var n_cos = List[UInt8](); _r7_pb_str(n_cos, 1, "x"); _r7_pb_str(n_cos, 2, "y"); _r7_pb_str(n_cos, 4, "ConstantOfShape")
    var g_cos = List[UInt8](); _r7_pb_bytes(g_cos, 1, n_cos); _r7_pb_bytes(g_cos, 11, vi_in); _r7_pb_bytes(g_cos, 12, vi_out)
    var m_cos = List[UInt8](); _r7_pb_bytes(m_cos, 7, g_cos)
    var p_cos = tmp_dir + "/r7_cos.onnx"
    assert_true(write_bytes(p_cos, m_cos))
    var cos_mod = OnnxModel.load(p_cos, 1)
    cos_mod.set_input("x", Tensor.ints(DT_I64, [1], [100_000_000_000]))
    var cos_raised = False
    try:
        cos_mod.run()
    except:
        cos_raised = True
    assert_true(cos_raised)

    # 8. cycles: parse_number exponent cap, mini !N / <@N> / (k,N) caps, bjorklund cap, script run/ply/chunk caps
    var p_huge = parse_number("1e9000000000000000000")
    assert_true(p_huge[1] and p_huge[0] > 1.0e300)
    var p_tiny = parse_number("1e-9000000000000000000")
    assert_true(p_tiny[1] and p_tiny[0] == 0.0)
    var p_zero = parse_number("0e999999999999999999")
    assert_true(p_zero[1] and p_zero[0] == 0.0)

    var mini_cases: List[String] = [
        String("bd!1000000000"),
        String("<bd@1000000000>"),
        String("bd(1,1000000000)"),
        String("bd*1000000000"),
        String("{bd}%1000000000"),
    ]
    for mc in mini_cases:
        var raised = False
        try:
            _ = mini(mc)
        except:
            raised = True
        assert_true(raised, "Adversarial mini pattern must fail cleanly: " + mc)

    assert_equal(len(bjorklund(1, 1000000000)), 4096)

    var prog_big = run_program("d1 $ n (run 1000000000)\nd2 $ ply 1000000000 $ s \"bd\"\nd3 $ chunk 1000000000 (# speed 2) $ s \"bd\"")
    assert_equal(len(prog_big.errors), 3)
    assert_equal(len(prog_big.tracks), 0)


def _r8_build_sfnt(tags: List[Int], tables: List[List[UInt8]], version: Int = 0x00010000) -> List[UInt8]:
    var out = List[UInt8]()
    var n = len(tags)
    _r7_put_u32_be(out, version)
    _r7_put_u16_be(out, n)
    _r7_put_u16_be(out, 16)
    _r7_put_u16_be(out, 0)
    _r7_put_u16_be(out, 0)
    var cur_off = 12 + n * 16
    for i in range(n):
        _r7_put_u32_be(out, tags[i])
        _r7_put_u32_be(out, 0)
        _r7_put_u32_be(out, cur_off)
        _r7_put_u32_be(out, len(tables[i]))
        cur_off += len(tables[i])
    for i in range(n):
        for b in tables[i]:
            out.append(b)
    return out^


def _r8_cff_index(items: List[List[UInt8]]) -> List[UInt8]:
    var out = List[UInt8]()
    var n = len(items)
    _r7_put_u16_be(out, n)
    if n == 0:
        return out^
    out.append(UInt8(4))  # offSize = 4
    var off = 1
    _r7_put_u32_be(out, off)
    for i in range(n):
        off += len(items[i])
        _r7_put_u32_be(out, off)
    for i in range(n):
        for b in items[i]:
            out.append(b)
    return out^


def _r8_cff_i32(mut out: List[UInt8], v: Int):
    out.append(UInt8(29))
    _r7_put_u32_be(out, v)


def test_r8_partition_ab_fixes() raises:
    # 1. TTF GDEF v1.3 item_delta / region_scalar: axisCount mismatch & bounds safety + valid delta
    var head_t = List[UInt8](length=54, fill=0)
    head_t[18] = UInt8(0x03); head_t[19] = UInt8(0xE8)  # unitsPerEm = 1000
    var hhea_t = List[UInt8](length=36, fill=0)
    hhea_t[35] = UInt8(1)  # metric_count = 1
    var maxp_t = List[UInt8](length=6, fill=0)
    maxp_t[5] = UInt8(2)   # glyph_count = 2
    var hmtx_t = List[UInt8](length=8, fill=0)
    var loca_t = List[UInt8](length=6, fill=0)
    var glyf_t = List[UInt8](length=4, fill=0)
    var cmap_t = List[UInt8]()
    _r7_put_u16_be(cmap_t, 0); _r7_put_u16_be(cmap_t, 1)
    _r7_put_u16_be(cmap_t, 3); _r7_put_u16_be(cmap_t, 1); _r7_put_u32_be(cmap_t, 12)
    # format 4 minimal subtable at +12
    _r7_put_u16_be(cmap_t, 4); _r7_put_u16_be(cmap_t, 16); _r7_put_u16_be(cmap_t, 0)
    _r7_put_u16_be(cmap_t, 2); _r7_put_u16_be(cmap_t, 2); _r7_put_u16_be(cmap_t, 0); _r7_put_u16_be(cmap_t, 0)
    _r7_put_u16_be(cmap_t, 0xFFFF); _r7_put_u16_be(cmap_t, 0); _r7_put_u16_be(cmap_t, 0xFFFF)
    _r7_put_u16_be(cmap_t, 1); _r7_put_u16_be(cmap_t, 0)
    # fvar: 1 axis 'wght' [100, 400, 900]
    var fvar_t = List[UInt8]()
    _r7_put_u16_be(fvar_t, 1); _r7_put_u16_be(fvar_t, 0)
    _r7_put_u16_be(fvar_t, 16); _r7_put_u16_be(fvar_t, 2)
    _r7_put_u16_be(fvar_t, 1); _r7_put_u16_be(fvar_t, 20)
    _r7_put_u16_be(fvar_t, 0); _r7_put_u16_be(fvar_t, 0)
    _r7_put_u32_be(fvar_t, 0x77676874)  # 'wght'
    _r7_put_u32_be(fvar_t, 100 << 16); _r7_put_u32_be(fvar_t, 400 << 16); _r7_put_u32_be(fvar_t, 900 << 16)
    _r7_put_u16_be(fvar_t, 0); _r7_put_u16_be(fvar_t, 0)
    # GDEF v1.3 with ItemVariationStore: first with axisCount=0 (must not SIGTRAP)
    var gdef_bad = List[UInt8](length=18, fill=0)
    gdef_bad[1] = UInt8(1); gdef_bad[3] = UInt8(3)  # version 1.3
    gdef_bad[17] = UInt8(18)  # store offset = 18
    # Store at +18: format=1, regionListOffset=12, dataCount=1, dataOffset[0]=20
    _r7_put_u16_be(gdef_bad, 1); _r7_put_u32_be(gdef_bad, 12)
    _r7_put_u16_be(gdef_bad, 1); _r7_put_u32_be(gdef_bad, 20)
    # RegionList at store+12: axisCount=0, regionCount=1
    _r7_put_u16_be(gdef_bad, 0); _r7_put_u16_be(gdef_bad, 1)
    _r7_put_u16_be(gdef_bad, 0); _r7_put_u16_be(gdef_bad, 0)
    # ItemVarData at store+20: itemCount=1, wordDeltaCount=1, regionIndexCount=1, regionIdx=0, delta=50
    _r7_put_u16_be(gdef_bad, 1); _r7_put_u16_be(gdef_bad, 1)
    _r7_put_u16_be(gdef_bad, 1); _r7_put_u16_be(gdef_bad, 0)
    _r7_put_u16_be(gdef_bad, 50)
    # GPOS with 'kern' lookup (format 1, xAdvance=10 + VariationIndex(0,0))
    var gpos_t = List[UInt8]()
    _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 0)
    _r7_put_u16_be(gpos_t, 10); _r7_put_u16_be(gpos_t, 12); _r7_put_u16_be(gpos_t, 36)
    _r7_put_u16_be(gpos_t, 0)  # empty ScriptList at +10
    # FeatureList at +12: count=1, 'kern' at +8 (offset 20)
    _r7_put_u16_be(gpos_t, 1); _r7_put_u32_be(gpos_t, 0x6B65726E); _r7_put_u16_be(gpos_t, 8)
    _r7_put_u16_be(gpos_t, 0); _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 0)  # Feature: lookup 0
    # LookupList at +26? Wait: 12 + 2 + 6 + 6 = 26 -> adjust offset to 26 -> let's pad to 36:
    while len(gpos_t) < 36:
        gpos_t.append(UInt8(0))
    # LookupList at +36: count=1, lookup[0] at +4 (40)
    _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 4)
    # Lookup at +40: type=2 (PairPos), flag=0, subCount=1, sub[0]=8 (48)
    _r7_put_u16_be(gpos_t, 2); _r7_put_u16_be(gpos_t, 0); _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 8)
    # PairPos fmt 1 at +48: fmt=1, covOff=12 (60), valFmt1=0x0044 (xAdv + xAdvDevice), valFmt2=0, pairSetCount=1, pairSetOff[0]=18 (66)
    _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 12); _r7_put_u16_be(gpos_t, 0x0044)
    _r7_put_u16_be(gpos_t, 0); _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 18)
    # Coverage fmt 1 at +60: fmt=1, count=1, gid=0 (6 bytes)
    _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 0)
    # PairSet at +66: pairCount=1, secondGlyph=1, xAdvance=10, devOff=26 (sub+26 = 48+26 = 74)
    _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 1); _r7_put_u16_be(gpos_t, 10); _r7_put_u16_be(gpos_t, 26)
    # VariationIndex table at +74: outer=0, inner=0, format=0x8000
    _r7_put_u16_be(gpos_t, 0); _r7_put_u16_be(gpos_t, 0); _r7_put_u16_be(gpos_t, 0x8000)

    var tags_gdef: List[Int] = [
        0x68656164, 0x68686561, 0x6D617870, 0x686D7478,
        0x6C6F6361, 0x676C7966, 0x636D6170, 0x66766172,
        0x47444546, 0x47504F53,
    ]
    var tbls_gdef: List[List[UInt8]] = [
        head_t.copy(), hhea_t.copy(), maxp_t.copy(), hmtx_t.copy(),
        loca_t.copy(), glyf_t.copy(), cmap_t.copy(), fvar_t.copy(),
        gdef_bad.copy(), gpos_t.copy(),
    ]
    var font_bad_gdef = TrueTypeFont()
    assert_true(font_bad_gdef.load_bytes(_r8_build_sfnt(tags_gdef, tbls_gdef)))
    assert_true(font_bad_gdef.set_variation("wght", Float32(900.0)))
    assert_equal(font_bad_gdef.kerning(0, 1), Float32(10.0), "mismatched GDEF axisCount must return base kern without SIGTRAP")
    assert_equal(font_bad_gdef.region_scalar(List[Float32](), List[Float32](), List[Float32](), True), Float32(0.0))

    # Positive GDEF v1.3 check: axisCount=1, region(start=0, peak=1, end=1) -> delta=50 -> kern = 60.0
    var gdef_ok = List[UInt8](length=18, fill=0)
    gdef_ok[1] = UInt8(1); gdef_ok[3] = UInt8(3)
    gdef_ok[17] = UInt8(18)
    _r7_put_u16_be(gdef_ok, 1); _r7_put_u32_be(gdef_ok, 12)
    _r7_put_u16_be(gdef_ok, 1); _r7_put_u32_be(gdef_ok, 22)
    _r7_put_u16_be(gdef_ok, 1); _r7_put_u16_be(gdef_ok, 1)  # axisCount=1, regionCount=1
    _r7_put_u16_be(gdef_ok, 0); _r7_put_u16_be(gdef_ok, 16384); _r7_put_u16_be(gdef_ok, 16384)  # start=0, peak=1, end=1
    _r7_put_u16_be(gdef_ok, 1); _r7_put_u16_be(gdef_ok, 1); _r7_put_u16_be(gdef_ok, 1); _r7_put_u16_be(gdef_ok, 0)
    _r7_put_u16_be(gdef_ok, 50)
    tbls_gdef[8] = gdef_ok^
    var font_ok_gdef = TrueTypeFont()
    assert_true(font_ok_gdef.load_bytes(_r8_build_sfnt(tags_gdef, tbls_gdef)))
    assert_true(font_ok_gdef.set_variation("wght", Float32(900.0)))
    assert_equal(font_ok_gdef.kerning(0, 1), Float32(60.0))

    # 2. TTF GSUB MultipleSubst: cascading 3x100 explosion must stay bounded & fast; valid 1->3 works
    var gsub_t = List[UInt8]()
    # Header: ver 1.0, scriptList=10, featureList=38, lookupList=60
    _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 0)
    _r7_put_u16_be(gsub_t, 10); _r7_put_u16_be(gsub_t, 38); _r7_put_u16_be(gsub_t, 60)
    # ScriptList at +10: count=1, 'latn' off=8 (18)
    _r7_put_u16_be(gsub_t, 1); _r7_put_u32_be(gsub_t, 0x6C61746E); _r7_put_u16_be(gsub_t, 8)
    # Script at +18: defaultLangSys=4 (22), langCount=0
    _r7_put_u16_be(gsub_t, 4); _r7_put_u16_be(gsub_t, 0)
    # LangSys at +22 (16 bytes to pad to +38): lookupOrder=0, req=0xFFFF, featCount=1, featIdx[0]=0
    _r7_put_u16_be(gsub_t, 0); _r7_put_u16_be(gsub_t, 0xFFFF); _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 0)
    while len(gsub_t) < 38:
        gsub_t.append(UInt8(0))
    # FeatureList at +38: count=1, 'liga' off=8 (46)
    _r7_put_u16_be(gsub_t, 1); _r7_put_u32_be(gsub_t, 0x6C696761); _r7_put_u16_be(gsub_t, 8)
    # Feature at +46: params=0, lookupCount=3, lookups=[0, 1, 2] (10 bytes -> +56)
    _r7_put_u16_be(gsub_t, 0); _r7_put_u16_be(gsub_t, 3)
    _r7_put_u16_be(gsub_t, 0); _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 2)
    while len(gsub_t) < 60:
        gsub_t.append(UInt8(0))
    # LookupList at +60: count=3, offsets=[8, 16, 24] -> lookups at 68, 76, 84
    _r7_put_u16_be(gsub_t, 3); _r7_put_u16_be(gsub_t, 8); _r7_put_u16_be(gsub_t, 16); _r7_put_u16_be(gsub_t, 24)
    # Lookup 0 at +68, Lookup 1 at +76, Lookup 2 at +84 (each 8 bytes pointing to same subtable at +92)
    _r7_put_u16_be(gsub_t, 2); _r7_put_u16_be(gsub_t, 0); _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 24)  # 68+24=92
    _r7_put_u16_be(gsub_t, 2); _r7_put_u16_be(gsub_t, 0); _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 16)  # 76+16=92
    _r7_put_u16_be(gsub_t, 2); _r7_put_u16_be(gsub_t, 0); _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 8)   # 84+8=92
    # MultipleSubst fmt 1 at +92: fmt=1, covOff=8 (100), seqCount=1, seqOff[0]=14 (106)
    _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 8); _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 14)
    # Coverage fmt 1 at +100: fmt=1, count=1, gid=1
    _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 1); _r7_put_u16_be(gsub_t, 1)
    # Sequence at +106: glyphCount=100, 100 * gid=1
    _r7_put_u16_be(gsub_t, 100)
    for _ in range(100):
        _r7_put_u16_be(gsub_t, 1)

    var tags_gsub: List[Int] = [
        0x68656164, 0x68686561, 0x6D617870, 0x686D7478,
        0x6C6F6361, 0x676C7966, 0x636D6170, 0x47535542,
    ]
    var tbls_gsub: List[List[UInt8]] = [
        head_t.copy(), hhea_t.copy(), maxp_t.copy(), hmtx_t.copy(),
        loca_t.copy(), glyf_t.copy(), cmap_t.copy(), gsub_t.copy(),
    ]
    var font_gsub = TrueTypeFont()
    assert_true(font_gsub.load_bytes(_r8_build_sfnt(tags_gsub, tbls_gsub)))
    var gs_bomb: List[GsubGlyph] = [GsubGlyph(1)]
    font_gsub.apply_gsub(gs_bomb, "latn")
    assert_true(len(gs_bomb) >= 100 and len(gs_bomb) <= 4096, "MultipleSubst cascade must expand validly and hit cap")

    # 3. TTF CFF FDArray x LocalSubr deduplication + valid charstring execution
    var cff = List[UInt8]()
    cff.append(1); cff.append(0); cff.append(4); cff.append(4)  # CFF header (4 bytes)
    var name_idx = _r8_cff_index([[UInt8(65)]])
    for b in name_idx:
        cff.append(b)
    # Reserve space for Top DICT INDEX (predict fixed-size 5-byte integers for offsets)
    var top_start = len(cff)
    var dummy_top = List[UInt8](length=30, fill=0)
    var top_idx_ph = _r8_cff_index([dummy_top^])
    for b in top_idx_ph:
        cff.append(b)
    var str_idx = _r8_cff_index(List[List[UInt8]]())
    for b in str_idx:
        cff.append(b)
    var gsubr_idx = _r8_cff_index(List[List[UInt8]]())
    for b in gsubr_idx:
        cff.append(b)
    var cs_off = len(cff)
    var cs_idx = _r8_cff_index([[UInt8(14)]])  # glyph 0 = endchar
    for b in cs_idx:
        cff.append(b)
    var fdsel_off = len(cff)
    cff.append(UInt8(0)); cff.append(UInt8(0))  # format 0: glyph 0 -> FD 0
    var priv_off = len(cff)
    # Private DICT (6 bytes): Subrs rel_off = 6 (op 19)
    var priv_dict = List[UInt8]()
    _r8_cff_i32(priv_dict, 6)
    priv_dict.append(UInt8(19))
    for b in priv_dict:
        cff.append(b)
    # Shared LocalSubrs INDEX at priv_off + 6: 200 empty subrs
    var lsub_items = List[List[UInt8]]()
    for _ in range(200):
        lsub_items.append([UInt8(11)])  # return
    var lsub_idx = _r8_cff_index(lsub_items)
    for b in lsub_idx:
        cff.append(b)
    var fdarr_off = len(cff)
    # 200 FDs all sharing the same Private DICT & LocalSubrs
    var fd_dict = List[UInt8]()
    _r8_cff_i32(fd_dict, len(priv_dict))
    _r8_cff_i32(fd_dict, priv_off)
    fd_dict.append(UInt8(18))  # Private
    var fd_items = List[List[UInt8]]()
    for _ in range(200):
        fd_items.append(fd_dict.copy())
    var fdarr_idx = _r8_cff_index(fd_items)
    for b in fdarr_idx:
        cff.append(b)
    # Patch exact Top DICT of length 30
    var top_dict = List[UInt8]()
    _r8_cff_i32(top_dict, cs_off); top_dict.append(UInt8(17))
    _r8_cff_i32(top_dict, fdarr_off); top_dict.append(UInt8(12)); top_dict.append(UInt8(36))
    _r8_cff_i32(top_dict, fdsel_off); top_dict.append(UInt8(12)); top_dict.append(UInt8(37))
    while len(top_dict) < 30:
        top_dict.append(UInt8(139))  # push 0 (harmless padding)
    var real_top_idx = _r8_cff_index([top_dict^])
    for i in range(len(real_top_idx)):
        cff[top_start + i] = real_top_idx[i]

    var cff_sfnt = _r8_build_sfnt([0x43464620], [cff^], 0x4F54544F)
    var font_cff = TrueTypeFont()
    assert_true(font_cff.load_bytes(cff_sfnt^))
    assert_equal(len(font_cff.cff_fd_privates), 200)
    assert_equal(len(font_cff.cff_local_subr_pools), 1, "shared LocalSubrs INDEX must be deduplicated into 1 pool")
    _ = font_cff.cff_outline(0)

    # 4. logind.dbus_parse header field bounds safety
    var db_bad = List[UInt8](length=32, fill=0)
    db_bad[0] = UInt8(108); db_bad[1] = UInt8(3); db_bad[2] = UInt8(0); db_bad[3] = UInt8(1)
    db_bad[12] = UInt8(16)  # fields_end = 32
    db_bad[16] = UInt8(4)   # ERROR_NAME
    db_bad[17] = UInt8(1); db_bad[18] = UInt8(ord("s")); db_bad[19] = UInt8(0)
    db_bad[20] = UInt8(0x0F); db_bad[21] = UInt8(0x27)  # text_length = 9999 (OOB!)
    assert_equal(dbus_message_length(db_bad), 32)
    var db_parsed = dbus_parse(db_bad, 32)
    assert_equal(db_parsed.error_name, "")
    var db_overlen = dbus_parse(db_bad, 1000)
    assert_equal(db_overlen.kind, 3)
    var db_ok = dbus_method_call(11, "org.test", "/obj", "org.iface", "Ping", "u", [1, 0, 0, 0])
    var db_ok_parsed = dbus_parse(db_ok, len(db_ok))
    assert_equal(db_ok_parsed.serial, 11)
    assert_equal(len(db_ok_parsed.body), 4)

    # 5. AudioServer pitch sanitization (+Inf / NaN / extreme) & O(1) looping wrap
    var srv_r8 = AudioServer(48000, 256)
    var clip_pcm = List[Int16](length=256, fill=Int16(1000))
    var aid_r8 = srv_r8.register_pcm16_asset(clip_pcm, 1, 48000)
    var inf_f32 = f32_from_bits(0x7F800000)
    var nan_f32 = f32_from_bits(0x7FC00000)
    var h_inf = srv_r8.play_2d(aid_r8, volume=0.5, pitch=inf_f32, looping=True)
    _ = srv_r8.play_2d(aid_r8, volume=0.5, pitch=nan_f32, looping=True)
    _ = srv_r8.play_2d(aid_r8, volume=0.5, pitch=Float32(1.0e12), looping=True)
    # Also push >64 virtualized looping emitters with huge pitch to hit _virtualize_voices
    for _ in range(70):
        _ = srv_r8.play_2d(aid_r8, volume=0.1, pitch=Float32(1.0e9), looping=True)
    _ = srv_r8.update_emitter(h_inf, Vec3(0.0, 0.0, 0.0), pitch=inf_f32)
    var out_lr = List[Float32](length=512, fill=0.0)
    srv_r8.render_offline_quantum(out_lr, 256)
    for smp in out_lr:
        assert_false(isnan(smp) or isinf(smp))

    # 6. NetChannel max_mtu_bytes=1400 + AEAD encryption (1300-byte packet, no UAF)
    var sim_mtu = VirtualNetSim(seed=0x1400, one_way_latency_ms=5.0, jitter_ms=0.0, loss_rate=0.0, mtu_bytes=1400)
    var ep0 = sim_mtu.create_ipv4_peer(0, 9940)
    var ep1 = sim_mtu.create_ipv4_peer(1, 9940)
    var ch_tx = NetChannel(ep1.local_endpoint(), 1400)
    var ch_rx = NetChannel(ep0.local_endpoint(), 1400)
    var key_r8 = AeadKey(0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888)
    ch_tx.enable_encryption(key_r8)
    ch_rx.enable_encryption(key_r8)
    var pay1300 = List[UInt8](capacity=1300)
    for i in range(1300):
        pay1300.append(UInt8((i * 17 + 3) & 0xFF))
    var sent1300 = ch_tx.send_unreliable_sequenced(pay1300, ep0, sim_mtu.now_ms())
    assert_true(sent1300[0], "1300-byte encrypted unreliable_sequenced must send on MTU=1400 channel")
    sim_mtu.advance_ms(20.0)
    ch_rx.poll(ep1, sim_mtu.now_ms())
    _ = sim_mtu.now_ms()
    var rx1300 = ch_rx.pop_unreliable_message()
    assert_true(rx1300[0], "1300-byte encrypted packet must authenticate and deliver cleanly")
    assert_equal(len(rx1300[1]), 1300)
    assert_equal(ch_rx.crypto_auth_failures, 0)
    for i in range(1300):
        assert_equal(rx1300[1][i], pay1300[i])
    # Also verify aead_seal_raw when aad_ptr aliases out_ct_tag across a resizing append
    var alias_buf = List[UInt8](capacity=16)
    for i in range(16):
        alias_buf.append(UInt8(i + 1))
    var pt_big = List[UInt8](length=128, fill=UInt8(0x5A))
    aead_seal_raw(key_r8, 0, 1, 2, Int(alias_buf.unsafe_ptr()), 16, Int(pt_big.unsafe_ptr()), 128, alias_buf)
    _ = len(pt_big)
    var dec_pt = List[UInt8]()
    var aad_check = List[UInt8](capacity=16)
    for i in range(16):
        aad_check.append(UInt8(i + 1))
    var ok_alias = aead_open_raw(
        key_r8, 0, 1, 2,
        Int(aad_check.unsafe_ptr()), 16,
        Int(alias_buf.unsafe_ptr()) + 16, 128,
        Int(alias_buf.unsafe_ptr()) + 16 + 128,
        dec_pt,
    )
    _ = len(aad_check)
    _ = len(alias_buf)
    assert_true(ok_alias)
    assert_equal(len(dec_pt), 128)

    # 7. cycles.pattern chord_notes cap
    assert_false(chord_notes("c'maj'1000000000")[1])
    var ch_valid = chord_notes("c'maj'4")
    assert_true(ch_valid[1] and len(ch_valid[0]) == 4)
    var p_chord_big = mini("c'maj'1000000000")
    assert_true(len(p_chord_big.query(Frac.of(0), Frac.of(1))) <= 4096)

    # 8. cycles.pattern chained/nested fast bound
    var p_fast_chain = mini("a*4096*4096")
    var ev_chain = p_fast_chain.query(Frac.of(0), Frac.of(1))
    assert_equal(len(ev_chain), 4096)
    var p_fast_seq = mini("[a b]*4096*4096")
    var ev_seq = p_fast_seq.query(Frac.of(0), Frac.of(1))
    assert_true(len(ev_seq) > 0 and len(ev_seq) <= 16384)
    var pat_manual = Pattern()
    var atom_id = pat_manual.atom(Params())
    pat_manual.root = pat_manual.fast(Frac.of(4096), pat_manual.fast(Frac.of(4096), atom_id))
    assert_equal(len(pat_manual.query(Frac.of(0), Frac.of(1))), 4096)
    assert_equal(len(mini("a*2*3").query(Frac.of(0), Frac.of(1))), 6)

    # 9. cycles.pattern Frac.make128 overflow rescaling
    var f_a = Frac.make(1, 1 << 60)
    var f_b = Frac.make(1, (1 << 60) + 1)
    var f_sum = f_a + f_b
    assert_true(f_sum.den > 0 and f_sum.num > 0)
    var f_val = f_sum.to_float()
    assert_true(f_val > 1.0e-18 and f_val < 2.5e-18, "1/2^60 + 1/(2^60+1) must stay ~1.73e-18, not corrupt to 2.0")
    var f_huge = Frac.make128(Int128(1) << 100, Int128(3))
    assert_true(f_huge.num > 0 and f_huge.den == 1)

    # 10. SimdCanvas.blit_sprite_simd null/zero guard
    var sc = SimdCanvas(64, 64)
    sc.blit_sprite_simd(0, 16, 16, 0, 0, Float32(0.5))
    sc.blit_sprite_simd(0, 0, 16, 0, 0, Float32(0.5))
    assert_equal(sc.sprites_drawn, 0)

    # 11. AaaRenderGraph.compile UInt32 truncation & zero-size guard
    var rg_huge = AaaRenderGraph()
    var r_huge = rg_huge.add_resource("huge", (Int(1) << 44) - 4096, True, False, 8192)
    _ = rg_huge.add_pass("p0", [RgResourceAccess(r_huge, RG_USAGE_COMPUTE_WRITE)])
    assert_false(rg_huge.compile(65536), "UInt32-overflowing resource size must fail compile()")

    var rg_zero = AaaRenderGraph()
    var r_zero = rg_zero.add_resource("z", 1024, True, False, 256)
    rg_zero.resources[r_zero].size_bytes = 0
    _ = rg_zero.add_pass("p0", [RgResourceAccess(r_zero, RG_USAGE_COMPUTE_WRITE)])
    assert_false(rg_zero.compile(65536))

    var rg_ok = AaaRenderGraph()
    var r_ok = rg_ok.add_resource("ok", 4096, True, False, 256)
    _ = rg_ok.add_pass("p0", [RgResourceAccess(r_ok, RG_USAGE_COMPUTE_WRITE)])
    _ = rg_ok.add_pass("p1", [RgResourceAccess(r_ok, RG_USAGE_COMPUTE_READ)])
    assert_true(rg_ok.compile(65536))


def _r9_build_cff_otf(gsubrs: List[List[UInt8]], charstrings: List[List[UInt8]]) -> List[UInt8]:
    var cff = List[UInt8]()
    cff.append(1); cff.append(0); cff.append(4); cff.append(4)
    var name_idx = _r8_cff_index([[UInt8(65)]])
    for b in name_idx:
        cff.append(b)
    var top_start = len(cff)
    var dummy_top = List[UInt8](length=12, fill=0)
    var top_idx_ph = _r8_cff_index([dummy_top^])
    for b in top_idx_ph:
        cff.append(b)
    var str_idx = _r8_cff_index(List[List[UInt8]]())
    for b in str_idx:
        cff.append(b)
    var gsubr_idx = _r8_cff_index(gsubrs)
    for b in gsubr_idx:
        cff.append(b)
    var cs_off = len(cff)
    var cs_idx = _r8_cff_index(charstrings)
    for b in cs_idx:
        cff.append(b)
    var priv_off = len(cff)
    cff.append(UInt8(139)); cff.append(UInt8(20))  # defaultWidthX = 0 (2 bytes)
    var top_dict = List[UInt8]()
    _r8_cff_i32(top_dict, cs_off); top_dict.append(UInt8(17))  # CharStrings (6 bytes)
    _r8_cff_i32(top_dict, 2); _r8_cff_i32(top_dict, priv_off); top_dict.append(UInt8(18))  # Private (11 bytes)
    # Trim or pad to exact 12 bytes: use Private at len=2, off=priv_off only if fits, so pad dummy_top=18
    while len(top_dict) < 12:
        top_dict.append(UInt8(139))
    var cff2 = List[UInt8]()
    cff2.append(1); cff2.append(0); cff2.append(4); cff2.append(4)
    var name_idx2 = _r8_cff_index([[UInt8(65)]])
    for b in name_idx2:
        cff2.append(b)
    var top_start2 = len(cff2)
    var dummy_top2 = List[UInt8](length=18, fill=0)
    var top_ph2 = _r8_cff_index([dummy_top2^])
    for b in top_ph2:
        cff2.append(b)
    var str_idx2 = _r8_cff_index(List[List[UInt8]]())
    for b in str_idx2:
        cff2.append(b)
    var gsubr_idx2 = _r8_cff_index(gsubrs)
    for b in gsubr_idx2:
        cff2.append(b)
    var cs_off2 = len(cff2)
    var cs_idx2 = _r8_cff_index(charstrings)
    for b in cs_idx2:
        cff2.append(b)
    var priv_off2 = len(cff2)
    cff2.append(UInt8(139)); cff2.append(UInt8(20))
    var top_dict2 = List[UInt8]()
    _r8_cff_i32(top_dict2, cs_off2); top_dict2.append(UInt8(17))
    _r8_cff_i32(top_dict2, 2); _r8_cff_i32(top_dict2, priv_off2); top_dict2.append(UInt8(18))
    top_dict2.append(UInt8(139))  # 18 bytes total
    var real_top2 = _r8_cff_index([top_dict2^])
    for i in range(len(real_top2)):
        cff2[top_start2 + i] = real_top2[i]
    _ = top_start
    _ = cs_off
    _ = priv_off
    return _r8_build_sfnt([0x43464620], [cff2^], 0x4F54544F)


def _r9_build_1x1_exr(compression: Int, payload: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    _put_le32(out, EXR_MAGIC)
    _put_le32(out, 2)
    # channels: R (HALF)
    for b in String("channels").as_bytes(): out.append(b)
    out.append(0)
    for b in String("chlist").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 19)
    out.append(UInt8(ord("R"))); out.append(0)
    _put_le32(out, 1)  # HALF
    out.append(0); out.append(0); out.append(0); out.append(0)
    _put_le32(out, 1); _put_le32(out, 1)
    out.append(0)
    # compression
    for b in String("compression").as_bytes(): out.append(b)
    out.append(0)
    for b in String("compression").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 1)
    out.append(UInt8(compression))
    # dataWindow 0..0, 0..0
    for b in String("dataWindow").as_bytes(): out.append(b)
    out.append(0)
    for b in String("box2i").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 16)
    _put_le32(out, 0); _put_le32(out, 0); _put_le32(out, 0); _put_le32(out, 0)
    out.append(0)  # end of header
    var chunk_off = len(out) + 8
    _put_le64(out, UInt64(chunk_off))
    _put_le32(out, 0)  # y = 0
    _put_le32(out, len(payload))
    for b in payload:
        out.append(b)
    return out^


def test_r9_partition_ab_fixes() raises:
    # 1. TTF CFF Type2 exponential callgsubr bomb (DoS guard) + valid CFF outline
    var bomb_gsubrs = List[List[UInt8]]()
    for k in range(15):
        var gs = List[UInt8]()
        var push_byte = UInt8(33 + k)  # bias(16)=107 -> (33+k-139)+107 = k+1
        for _ in range(8):
            gs.append(push_byte)
            gs.append(UInt8(29))  # callgsubr
        gs.append(UInt8(11))  # return
        bomb_gsubrs.append(gs^)
    bomb_gsubrs.append([UInt8(11)])  # gsubr[15] = return
    var bomb_cs: List[UInt8] = [UInt8(32), UInt8(29), UInt8(14)]  # push -107 -> gsubr[0], callgsubr, endchar
    var bomb_otf = _r9_build_cff_otf(bomb_gsubrs, [bomb_cs^])
    var font_bomb = TrueTypeFont()
    assert_true(font_bomb.load_bytes(bomb_otf^))
    var out_bomb = font_bomb.cff_outline(0)
    assert_equal(len(out_bomb.xs), 0)

    # Valid CFF outline invoking gsubr[0] to draw a triangle
    var ok_gsubr = List[UInt8]()
    ok_gsubr.append(UInt8(239)); ok_gsubr.append(UInt8(139)); ok_gsubr.append(UInt8(21))  # 100 0 rmoveto
    ok_gsubr.append(UInt8(239)); ok_gsubr.append(UInt8(139)); ok_gsubr.append(UInt8(5))   # 100 0 rlineto
    ok_gsubr.append(UInt8(139)); ok_gsubr.append(UInt8(239)); ok_gsubr.append(UInt8(5))   # 0 100 rlineto
    ok_gsubr.append(UInt8(11))  # return
    var ok_cs: List[UInt8] = [UInt8(32), UInt8(29), UInt8(14)]  # bias(1)=107 -> 32-139+107 = 0
    var ok_otf = _r9_build_cff_otf([ok_gsubr^], [ok_cs^])
    var font_ok_cff = TrueTypeFont()
    assert_true(font_ok_cff.load_bytes(ok_otf^))
    var out_ok_cff = font_ok_cff.cff_outline(0)
    assert_equal(len(out_ok_cff.xs), 3)
    assert_equal(out_ok_cff.xs[1], Float32(200.0))
    assert_equal(out_ok_cff.ys[2], Float32(100.0))

    # 2. OpenEXR ZIPS / ZIP packed_sz > expected_raw must inflate_zlib + undo predictor
    var z_one: List[UInt8] = [
        UInt8(0x78), UInt8(0x01), UInt8(0x01), UInt8(0x02), UInt8(0x00),
        UInt8(0xFD), UInt8(0xFF), UInt8(0x00), UInt8(0xBC),
        UInt8(0x00), UInt8(0xBF), UInt8(0x00), UInt8(0xBD),
    ]
    var exr_zips = _r9_build_1x1_exr(EXR_ZIPS_COMPRESSION, z_one)
    var img_zips = decode_exr_fp16(exr_zips)
    assert_true(img_zips.valid, "1x1 EXR_ZIPS with packed_sz=13 > expected_raw=2 must decode")
    assert_true(abs(img_zips.pixels[0].x - Float32(1.0)) < Float32(1.0e-5))

    var exr_zip = _r9_build_1x1_exr(EXR_ZIP_COMPRESSION, z_one)
    var img_zip = decode_exr_fp16(exr_zip)
    assert_true(img_zip.valid, "1x1 EXR_ZIP with packed_sz=13 > expected_raw=2 must decode")
    assert_true(abs(img_zip.pixels[0].x - Float32(1.0)) < Float32(1.0e-5))

    var z_bad = List[UInt8](length=13, fill=UInt8(0xFF))
    z_bad[0] = UInt8(0x78); z_bad[1] = UInt8(0x01); z_bad[2] = UInt8(0x07)  # invalid deflate block type 3
    var img_bad_zip = decode_exr_fp16(_r9_build_1x1_exr(EXR_ZIPS_COMPRESSION, z_bad))
    assert_false(img_bad_zip.valid, "corrupt zlib chunk with packed_sz > expected_raw must fail")

    # 3. cycles.pattern.Frac same-denominator __add__ / __sub__ / __truediv__ / __neg__
    var f_big = Frac.make(1 << 62, 3)
    var f_big_neg = Frac.make(-(1 << 62), 3)
    var f_add_same = f_big + f_big
    var f_sub_same = f_big - f_big_neg
    assert_true(f_add_same.num > 0 and f_add_same.den > 0)
    assert_true(f_add_same == f_sub_same)
    var exp_big = Float64(1 << 62) * 2.0 / 3.0
    assert_true(abs(f_add_same.to_float() - exp_big) / exp_big < 1.0e-9)
    assert_true(-f_big == f_big_neg)
    assert_true(f_big / Frac.of(0) == Frac(0, 1))
    assert_true(Frac.make(1, 3) + Frac.make(1, 3) == Frac.make(2, 3))
    assert_true(Frac.make(5, 7) - Frac.make(2, 7) == Frac.make(3, 7))

    # 4. Component read_bin remaining-byte + reader.failed hardening + valid round-trips
    var bw_col = BlobWriter()
    var col_ref = ColliderComponent.convex_hull([Vec3(1.0, 2.0, 3.0), Vec3(4.0, 5.0, 6.0)])
    col_ref.write_bin(bw_col)
    var br_col_ok = BlobReader(bw_col.data.copy())
    var col_rt = ColliderComponent.read_bin(br_col_ok, 1)
    assert_equal(len(col_rt.points), 2)
    assert_equal(col_rt.points[1].x, Float32(4.0))
    assert_equal(col_rt.points[1].y, Float32(5.0))
    assert_equal(col_rt.points[1].z, Float32(6.0))

    var bw_col_bad = BlobWriter()
    bw_col_bad.integer(0); bw_col_bad.f32(0.5); bw_col_bad.f32(0.5)
    bw_col_bad.vec3(Vec3()); bw_col_bad.vec3(Vec3()); bw_col_bad.quat(Quat())
    bw_col_bad.boolean(False)
    bw_col_bad.u32(1); bw_col_bad.u32(1); bw_col_bad.integer(0)
    bw_col_bad.integer(1 << 20)  # truncated points count
    var col_raised = False
    try:
        var br_col_bad = BlobReader(bw_col_bad.data.copy())
        _ = ColliderComponent.read_bin(br_col_bad, 1)
    except:
        col_raised = True
    assert_true(col_raised, "truncated ColliderComponent blob must raise Error")

    var bw_cloth = BlobWriter()
    var cloth_ref = ClothMeshComponent()
    cloth_ref.attachments.append(RigidAttachment(0, 1, Vec3(0.0, 1.0, 0.0)))
    cloth_ref.write_bin(bw_cloth)
    var br_cloth_ok = BlobReader(bw_cloth.data.copy())
    var cloth_rt = ClothMeshComponent.read_bin(br_cloth_ok, 1)
    assert_equal(len(cloth_rt.attachments), 1)

    var bw_cloth_bad = BlobWriter()
    ParticleMesh().serialize(bw_cloth_bad)
    bw_cloth_bad.integer(1 << 16)  # truncated attachment count
    var cloth_raised = False
    try:
        var br_cloth_bad = BlobReader(bw_cloth_bad.data.copy())
        _ = ClothMeshComponent.read_bin(br_cloth_bad, 1)
    except:
        cloth_raised = True
    assert_true(cloth_raised, "truncated ClothMeshComponent blob must raise Error")

    var bw_skel = BlobWriter()
    var skel_ref = SkeletonComponent(42, SkeletonInstance(3), False)
    skel_ref.skeleton.pose.t[2] = Vec3(7.0, 8.0, 9.0)
    skel_ref.write_bin(bw_skel)
    var br_skel_ok = BlobReader(bw_skel.data.copy())
    var skel_rt = SkeletonComponent.read_bin(br_skel_ok, 1)
    assert_equal(skel_rt.model_id, UInt64(42))
    assert_equal(skel_rt.skeleton.pose.count(), 3)
    assert_equal(skel_rt.skeleton.pose.t[2].x, Float32(7.0))
    assert_equal(skel_rt.skeleton.pose.t[2].y, Float32(8.0))
    assert_equal(skel_rt.skeleton.pose.t[2].z, Float32(9.0))

    var bw_skel_bad = BlobWriter()
    bw_skel_bad.u64(1); bw_skel_bad.boolean(True); bw_skel_bad.integer(1 << 16)
    var skel_raised = False
    try:
        var br_skel_bad = BlobReader(bw_skel_bad.data.copy())
        _ = SkeletonComponent.read_bin(br_skel_bad, 1)
    except:
        skel_raised = True
    assert_true(skel_raised, "truncated SkeletonComponent blob must raise Error")

    var bw_anim = BlobWriter()
    var anim_ref = AnimatorComponent(77, True, True, 5)
    anim_ref.animator.params.append(Float32(0.25))
    anim_ref.animator.params.append(Float32(0.75))
    anim_ref.foot_plant.add_leg(1, 2, 3, Float32(0.1), Vec3(0.0, 0.0, 1.0))
    anim_ref.foot_in_stance.append(True)
    anim_ref.write_bin(bw_anim)
    var br_anim_ok = BlobReader(bw_anim.data.copy())
    var anim_rt = AnimatorComponent.read_bin(br_anim_ok, 1)
    assert_equal(anim_rt.graph_id, UInt64(77))
    assert_equal(len(anim_rt.animator.params), 2)
    assert_equal(len(anim_rt.foot_plant.legs), 1)
    assert_equal(len(anim_rt.foot_in_stance), 1)

    var bw_anim_bad = BlobWriter()
    bw_anim_bad.u64(77); bw_anim_bad.boolean(False); bw_anim_bad.boolean(False); bw_anim_bad.boolean(False)
    bw_anim_bad.integer(-1); bw_anim_bad.integer(0); bw_anim_bad.integer(0)
    for _ in range(5):
        bw_anim_bad.f32(0.0)
    bw_anim_bad.integer(4096)  # truncated param count
    var anim_raised = False
    try:
        var br_anim_bad = BlobReader(bw_anim_bad.data.copy())
        _ = AnimatorComponent.read_bin(br_anim_bad, 1)
    except:
        anim_raised = True
    assert_true(anim_raised, "truncated AnimatorComponent blob must raise Error")

    # Sibling fixed-size read_bin empty-blob raises
    var sib_raised = 0
    try:
        var r_empty = BlobReader(List[UInt8]())
        _ = RigidBodyComponent.read_bin(r_empty, 1)
    except:
        sib_raised += 1
    try:
        var r_empty = BlobReader(List[UInt8]())
        _ = LocalTransform.read_bin(r_empty, 1)
    except:
        sib_raised += 1
    try:
        var r_empty = BlobReader(List[UInt8]())
        _ = AudioEmitterComponent.read_bin(r_empty, 1)
    except:
        sib_raised += 1
    try:
        var r_empty = BlobReader(List[UInt8]())
        _ = MeshRendererComponent.read_bin(r_empty, 1)
    except:
        sib_raised += 1
    try:
        var r_empty = BlobReader(List[UInt8]())
        _ = NetworkIdentityComponent.read_bin(r_empty, 1)
    except:
        sib_raised += 1
    assert_equal(sib_raised, 5)

    # 5. aead_open_raw in-place aliasing (offset > 0, offset == 0, and tampered tag)
    var key_r9 = AeadKey(0xCAFE, 0xBABE, 0xDEAD, 0xBEEF, 0x1234, 0x5678, 0x9ABC, 0xDEF0)
    var pt_r9 = List[UInt8](capacity=32)
    for i in range(32):
        pt_r9.append(UInt8(65 + (i % 26)))
    var alias_open = List[UInt8](capacity=16)
    for i in range(16):
        alias_open.append(UInt8(0x10 + i))
    aead_seal_raw(
        key_r9, 1, 2, 3,
        Int(alias_open.unsafe_ptr()), 16,
        Int(pt_r9.unsafe_ptr()), 32,
        alias_open,
    )
    _ = len(pt_r9)
    # Offset > 0 in-place decryption (aad at 0..16, ct at 16..48, tag at 48..64, out_pt = alias_open)
    var ok_off_alias = aead_open_raw(
        key_r9, 1, 2, 3,
        Int(alias_open.unsafe_ptr()), 16,
        Int(alias_open.unsafe_ptr()) + 16, 32,
        Int(alias_open.unsafe_ptr()) + 48,
        alias_open,
    )
    assert_true(ok_off_alias)
    assert_equal(len(alias_open), 32)
    for i in range(32):
        assert_equal(alias_open[i], pt_r9[i])

    # Offset == 0 in-place decryption (ct at 0..32, tag at 32..48, out_pt = zero_alias)
    var zero_alias = List[UInt8]()
    aead_seal_raw(key_r9, 4, 5, 6, 0, 0, Int(pt_r9.unsafe_ptr()), 32, zero_alias)
    var ok_zero_alias = aead_open_raw(
        key_r9, 4, 5, 6, 0, 0,
        Int(zero_alias.unsafe_ptr()), 32,
        Int(zero_alias.unsafe_ptr()) + 32,
        zero_alias,
    )
    assert_true(ok_zero_alias)
    assert_equal(len(zero_alias), 32)
    for i in range(32):
        assert_equal(zero_alias[i], pt_r9[i])

    # Tampered tag in-place must return False
    var bad_alias = List[UInt8]()
    aead_seal_raw(key_r9, 7, 8, 9, 0, 0, Int(pt_r9.unsafe_ptr()), 32, bad_alias)
    bad_alias[40] ^= UInt8(1)
    assert_false(
        aead_open_raw(
            key_r9, 7, 8, 9, 0, 0,
            Int(bad_alias.unsafe_ptr()), 32,
            Int(bad_alias.unsafe_ptr()) + 32,
            bad_alias,
        )
    )

    # 6. AaaRenderGraph.compile huge-alignment unaliased_transient_bytes overflow guard
    var rg_align_ovf = AaaRenderGraph()
    var ra = rg_align_ovf.add_resource("ra", 1024, True, False, Int(1) << 62)
    var rb = rg_align_ovf.add_resource("rb", 1024, True, False, Int(1) << 62)
    _ = rg_align_ovf.add_pass("p0", [RgResourceAccess(ra, RG_USAGE_COMPUTE_WRITE), RgResourceAccess(rb, RG_USAGE_COMPUTE_WRITE)])
    assert_false(rg_align_ovf.compile(1 << 20))
    assert_equal(rg_align_ovf.unaliased_transient_bytes, 0)


def _r10_build_2x2_zips_exr(packed_sz0: Int, payload0: List[UInt8], payload1: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    _put_le32(out, EXR_MAGIC)
    _put_le32(out, 2)
    for b in String("channels").as_bytes(): out.append(b)
    out.append(0)
    for b in String("chlist").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 19)
    out.append(UInt8(ord("R"))); out.append(0)
    _put_le32(out, 1)  # HALF
    out.append(0); out.append(0); out.append(0); out.append(0)
    _put_le32(out, 1); _put_le32(out, 1)
    out.append(0)
    for b in String("compression").as_bytes(): out.append(b)
    out.append(0)
    for b in String("compression").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 1)
    out.append(UInt8(EXR_ZIPS_COMPRESSION))
    for b in String("dataWindow").as_bytes(): out.append(b)
    out.append(0)
    for b in String("box2i").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 16)
    _put_le32(out, 0); _put_le32(out, 0); _put_le32(out, 1); _put_le32(out, 1)  # 2x2 (expected_raw=4 per line)
    out.append(0)
    var off0 = len(out) + 16
    var off1 = off0 + 8 + len(payload0)
    _put_le64(out, UInt64(off0))
    _put_le64(out, UInt64(off1))
    _put_le32(out, 0); _put_le32(out, packed_sz0)
    for b in payload0:
        out.append(b)
    _put_le32(out, 1); _put_le32(out, len(payload1))
    for b in payload1:
        out.append(b)
    return out^


def _r10_build_ktx2_1x1(vk_format: Int, supercomp: Int, payload: List[UInt8], uncomp_len: UInt64) -> List[UInt8]:
    var out = List[UInt8]()
    for b in [
        UInt8(0xAB), UInt8(0x4B), UInt8(0x54), UInt8(0x58),
        UInt8(0x20), UInt8(0x32), UInt8(0x30), UInt8(0xBB),
        UInt8(0x0D), UInt8(0x0A), UInt8(0x1A), UInt8(0x0A),
    ]:
        out.append(b)
    _put_le32(out, vk_format)
    _put_le32(out, 1)  # typeSize
    _put_le32(out, 1)  # width
    _put_le32(out, 1)  # height
    _put_le32(out, 0)  # depth
    _put_le32(out, 0)  # layerCount
    _put_le32(out, 1)  # faceCount
    _put_le32(out, 1)  # levelCount
    _put_le32(out, supercomp)
    _put_le32(out, 0); _put_le32(out, 0)  # dfd
    _put_le32(out, 0); _put_le32(out, 0)  # kvd
    _put_le64(out, 0); _put_le64(out, 0)  # sgd
    _put_le64(out, 104)
    _put_le64(out, UInt64(len(payload)))
    _put_le64(out, uncomp_len)
    for b in payload:
        out.append(b)
    return out^


def test_r10_partition_ab_fixes() raises:
    # 1. inflate_zlib max_in_bytes window bound + multi-chunk EXR ZIPS packed_sz=0/2
    var z_one: List[UInt8] = [
        UInt8(0x78), UInt8(0x01), UInt8(0x01), UInt8(0x02), UInt8(0x00),
        UInt8(0xFD), UInt8(0xFF), UInt8(0x00), UInt8(0xBC),
        UInt8(0x00), UInt8(0xBF), UInt8(0x00), UInt8(0xBD),
    ]
    var z_combined = z_one.copy()
    for b in z_one:
        z_combined.append(b)
    assert_equal(len(inflate_zlib(z_combined, 0, 2, 2)), 0, "inflate_zlib must fail when max_in_bytes=2")
    assert_equal(len(inflate_zlib(z_combined, 0, 2, 13)), 2, "inflate_zlib with exact max_in_bytes=13 must succeed")

    # 2x2 HALF R -> expected_raw = 4 per scanline; predictor+reorder of [0x00, 0x3C, 0x00, 0x3C] (two 1.0f16)
    # reordered bytes: [0x00, 0x00, 0x3C, 0x3C]; delta-encoded: [0x00, 0x80, 0xBC, 0x80]
    var z_two: List[UInt8] = [
        UInt8(0x78), UInt8(0x01), UInt8(0x01), UInt8(0x04), UInt8(0x00),
        UInt8(0xFB), UInt8(0xFF), UInt8(0x00), UInt8(0x80), UInt8(0xBC), UInt8(0x80),
        UInt8(0x03), UInt8(0x7C), UInt8(0x01), UInt8(0xBD),
    ]
    var exr_2_ok = _r10_build_2x2_zips_exr(len(z_two), z_two, z_two)
    var img_2_ok = decode_exr_fp16(exr_2_ok)
    assert_true(img_2_ok.valid and img_2_ok.width == 2 and img_2_ok.height == 2)
    assert_true(abs(img_2_ok.pixels[0].x - Float32(1.0)) < Float32(1.0e-5))
    assert_true(abs(img_2_ok.pixels[3].x - Float32(1.0)) < Float32(1.0e-5))

    var exr_2_bad0 = _r10_build_2x2_zips_exr(0, z_two, z_two)
    assert_false(decode_exr_fp16(exr_2_bad0).valid, "EXR ZIPS chunk0 packed_sz=0 must fail")
    var exr_2_bad2 = _r10_build_2x2_zips_exr(2, z_two, z_two)
    assert_false(decode_exr_fp16(exr_2_bad2).valid, "EXR ZIPS chunk0 packed_sz=2 (< expected_raw=4) must fail")

    # 2. EXR DWA zero-uncompressed-size DoS prevention
    var dwa_unk0 = _build_dwa_exr_with_header_counts(
        UInt64(0), UInt64(32), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
        UInt64(0), UInt64(0), UInt64(1),
    )
    assert_false(decode_exr_fp16(dwa_unk0).valid, "DWA unknown_comp_sz>0 with unknown_uncomp_sz=0 must fail")
    var dwa_rle0 = _build_dwa_exr_with_header_counts(
        UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(32), UInt64(0), UInt64(0),
        UInt64(0), UInt64(0), UInt64(1),
    )
    assert_false(decode_exr_fp16(dwa_rle0).valid, "DWA rle_comp_sz>0 with rle_uncomp_sz=0 must fail")
    var dwa_ac0 = _build_dwa_exr_with_header_counts(
        UInt64(0), UInt64(0), UInt64(32), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
        UInt64(0), UInt64(0), UInt64(1),
    )
    assert_false(decode_exr_fp16(dwa_ac0).valid, "DWA ac_comp_sz>0 with total_ac_cnt=0 must fail")

    # 3. KTX2 mip_len window + uncomp_len=0 Zlib/Zstd bomb rejection + valid round-trip
    var ktx_z_ok_pay: List[UInt8] = [
        UInt8(0x78), UInt8(0x01), UInt8(0x01), UInt8(0x04), UInt8(0x00),
        UInt8(0xFB), UInt8(0xFF), UInt8(0x11), UInt8(0x22), UInt8(0x33), UInt8(0xFF),
        UInt8(0x02), UInt8(0x6D), UInt8(0x01), UInt8(0x66),
    ]
    var ktx_z_ok = decode_ktx2(_r10_build_ktx2_1x1(37, 3, ktx_z_ok_pay, UInt64(4)))
    assert_true(ktx_z_ok.valid and ktx_z_ok.pixels[0] == UInt32(0xFF112233))

    # Truncated mip_len=2 followed by full valid zlib bytes
    var ktx_trunc = _r10_build_ktx2_1x1(37, 3, ktx_z_ok_pay, UInt64(4))
    ktx_trunc[88] = UInt8(2)  # byteLength = 2
    assert_false(decode_ktx2(ktx_trunc).valid, "KTX2 Zlib with mip_len=2 must reject")

    # Zlib uncomp_len=0
    assert_false(decode_ktx2(_r10_build_ktx2_1x1(37, 3, ktx_z_ok_pay, UInt64(0))).valid, "KTX2 Zlib uncomp_len=0 must reject")

    # Zstd valid 1x1 RGBA8 + uncomp_len=0 8KB bomb + inflated uncomp_len mismatch
    var zstd_ok_1x1: List[UInt8] = [
        UInt8(0x28), UInt8(0xB5), UInt8(0x2F), UInt8(0xFD),
        UInt8(0x20), UInt8(0x04),
        UInt8(0x21), UInt8(0x00), UInt8(0x00),
        UInt8(0x11), UInt8(0x22), UInt8(0x33), UInt8(0xFF),
    ]
    var ktx_zstd_ok = decode_ktx2(_r10_build_ktx2_1x1(37, 2, zstd_ok_1x1, UInt64(4)))
    assert_true(ktx_zstd_ok.valid and ktx_zstd_ok.pixels[0] == UInt32(0xFF112233))
    assert_false(decode_ktx2(_r10_build_ktx2_1x1(37, 2, zstd_ok_1x1, UInt64(0))).valid, "KTX2 Zstd uncomp_len=0 must reject")
    assert_false(decode_ktx2(_r10_build_ktx2_1x1(37, 2, zstd_ok_1x1, UInt64(8))).valid, "KTX2 Zstd uncomp_len>expected must reject")

    # 8KB multi-RLE-block Zstd bomb (uncomp_len=0, no FCS)
    var zstd_bomb = List[UInt8]()
    zstd_bomb.append(UInt8(0x28)); zstd_bomb.append(UInt8(0xB5)); zstd_bomb.append(UInt8(0x2F)); zstd_bomb.append(UInt8(0xFD))
    zstd_bomb.append(UInt8(0x00)); zstd_bomb.append(UInt8(0x00))  # no single_seg, no FCS, window=0
    for bi in range(64):
        var is_last = 1 if bi == 63 else 0
        var bh = is_last | (1 << 1) | (131071 << 3)  # RLE block of 131071 bytes
        zstd_bomb.append(UInt8(bh & 0xFF))
        zstd_bomb.append(UInt8((bh >> 8) & 0xFF))
        zstd_bomb.append(UInt8((bh >> 16) & 0xFF))
        zstd_bomb.append(UInt8(0x41))
    assert_false(decode_ktx2(_r10_build_ktx2_1x1(37, 2, zstd_bomb, UInt64(0))).valid, "KTX2 Zstd 8MB bomb with uncomp_len=0 must reject")

    # 4. SPIR-V validator OOB rtype >= bound (no SIGTRAP)
    var spv_bad: List[UInt32] = [
        SPIRV_MAGIC, UInt32(0x00010300), UInt32(0), UInt32(16), UInt32(0),
        (UInt32(2) << 16) | OP_CAPABILITY, CAPABILITY_SHADER,
        (UInt32(3) << 16) | OP_MEMORY_MODEL, UInt32(0), UInt32(1),
        (UInt32(4) << 16) | OP_ENTRY_POINT, UInt32(5), UInt32(4), UInt32(0x6E69616D),  # "main"
        (UInt32(2) << 16) | OP_TYPE_VOID, UInt32(1),
        (UInt32(2) << 16) | OP_TYPE_BOOL, UInt32(2),
        (UInt32(3) << 16) | OP_TYPE_FUNCTION, UInt32(3), UInt32(1),
        (UInt32(4) << 16) | OP_CONSTANT, UInt32(0x7FFFFFFF), UInt32(5), UInt32(1),  # OOB rtype!
        (UInt32(5) << 16) | OP_FUNCTION, UInt32(1), UInt32(4), UInt32(0), UInt32(3),
        (UInt32(2) << 16) | OP_LABEL, UInt32(6),
        (UInt32(4) << 16) | OP_BRANCH_CONDITIONAL, UInt32(5), UInt32(7), UInt32(7),
        (UInt32(2) << 16) | OP_LABEL, UInt32(7),
        (UInt32(1) << 16) | OP_RETURN,
        (UInt32(1) << 16) | OP_FUNCTION_END,
    ]
    var spv_rep = validate_spirv_module(spv_bad)
    assert_false(spv_rep.valid, "SPIR-V with rtype>=bound must fail validation without SIGTRAP")

    # 5. FlintAaaVulkanRenderer inactive readback methods must not SIGSEGV
    var aaa = FlintAaaVulkanRenderer(64, 64)
    aaa.shutdown()
    assert_false(aaa.active)
    var px_off = aaa.readback_pixels()
    assert_equal(len(px_off), 64 * 64)
    assert_equal(px_off[0], UInt32(0))
    var dst_off = List[UInt32]()
    aaa.readback_pixels_into(dst_off)
    assert_equal(len(dst_off), 64 * 64)
    var exp_off = aaa.get_adapted_exposure()
    assert_false(isnan(exp_off[0]) or isinf(exp_off[0]))
    assert_false(isnan(exp_off[1]) or isinf(exp_off[1]))

    # 6. cycles parser recursion depth limits (20,000 nested brackets & parens)
    var p_ok = mini("[[a b] c]")
    assert_true(p_ok.root >= 0)
    var deep_mini = List[UInt8](capacity=40001)
    for _ in range(20000):
        deep_mini.append(UInt8(ord("[")))
    deep_mini.append(UInt8(ord("a")))
    for _ in range(20000):
        deep_mini.append(UInt8(ord("]")))
    var mini_raised = False
    try:
        _ = mini(String(unsafe_from_utf8=deep_mini))
    except:
        mini_raised = True
    assert_true(mini_raised, "20000-deep nested [...] in mini must raise cleanly")

    var prog_ok = eval_pattern("((rev \"a b\"))")
    assert_equal(len(prog_ok.tracks), 1)
    var deep_parens = List[UInt8](capacity=40001)
    for _ in range(20000):
        deep_parens.append(UInt8(ord("(")))
    deep_parens.append(UInt8(ord("1")))
    for _ in range(20000):
        deep_parens.append(UInt8(ord(")")))
    var script_raised = False
    try:
        _ = eval_pattern(String(unsafe_from_utf8=deep_parens))
    except:
        script_raised = True
    assert_true(script_raised, "20000-deep nested (...) in eval_pattern must raise cleanly")

    # 7. ForceWorld / FieldVolume / Vorticle read hardening + valid round-trip
    var fw_ok = ForceWorld()
    fw_ok.add_volume(FieldVolume.box(1, TARGET_WIND, Vec3(-5, -5, -5), Vec3(5, 5, 5)))
    _ = fw_ok.add_vorticle(Vec3(0, 0, 0), Vec3(0, 1, 0), Float32(4.0), Float32(2.0), Float32(1.0), 60)
    var bw_fw = BlobWriter()
    fw_ok.write(bw_fw)
    var fw_rt = ForceWorld()
    var br_fw_ok = BlobReader(bw_fw.data.copy())
    fw_rt.read(br_fw_ok)
    assert_equal(len(fw_rt.volumes), 1)
    assert_equal(len(fw_rt.wind.vorticles), 1)
    assert_equal(fw_rt.wind.vorticles[0].life, 60)
    _ = fw_rt.wind_at(Vec3(1, 0, 0))

    var bw_fw_trunc = BlobWriter()
    bw_fw_trunc.integer(0)  # 0 volumes
    bw_fw_trunc.integer(1)  # next_vorticle_id
    bw_fw_trunc.integer(1)  # 1 vorticle, truncated before fields
    bw_fw_trunc.integer(1)
    var fw_trunc_raised = False
    try:
        var br_fw_trunc = BlobReader(bw_fw_trunc.data.copy())
        var fw_bad = ForceWorld()
        fw_bad.read(br_fw_trunc)
    except:
        fw_trunc_raised = True
    assert_true(fw_trunc_raised, "truncated ForceWorld blob must raise Error")

    var bw_v0 = BlobWriter()
    bw_v0.integer(1)
    FxVec.zero().write(bw_v0); FxVec(0, FX_ONE, 0).write(bw_v0)
    bw_v0.u64(UInt64(FX_ONE * 4)); bw_v0.u64(0); bw_v0.u64(0)
    bw_v0.integer(0)  # life = 0!
    bw_v0.integer(0)
    var v0_raised = False
    try:
        var br_v0 = BlobReader(bw_v0.data.copy())
        _ = Vorticle.read(br_v0)
    except:
        v0_raised = True
    assert_true(v0_raised, "Vorticle with life=0 must raise Error")
    var v_zero_life = Vorticle()
    v_zero_life.life = 0
    var vw_zero = vorticle_wind(v_zero_life, FxVec(FX_ONE, 0, 0))
    assert_equal(vw_zero.x, Int64(0))

    # 8. cycles crush=200 finite PCM & flip ppd=0 / subnormal underflow & BitReader start=-1
    var br_buf: List[UInt8] = [UInt8(0xAA), UInt8(0xBB)]
    var br_neg = ImageBitReader(br_buf, -1)
    assert_equal(br_neg.position, 0)
    assert_equal(br_neg.length, 0)
    assert_equal(br_neg.read_bit(), 0)
    assert_false(br_neg.valid)

    var eng_crush = CyclesEngine(48000)
    var errs_crush = eng_crush.load("d1 $ s \"sine\" # crush 200")
    assert_equal(len(errs_crush), 0)
    var pcm_crush = eng_crush.render(64)
    assert_equal(len(pcm_crush), 128)
    for s in pcm_crush:
        assert_false(isnan(s) or isinf(s), "crush=200 must produce finite PCM")

    var flat_img = List[UInt32](length=4, fill=UInt32(0xFF808080))
    var white_img = List[UInt32](length=4, fill=UInt32(0xFFFFFFFF))
    var black_img = List[UInt32](length=4, fill=UInt32(0xFF000000))
    assert_true(flip_mean(flip_error_map(flat_img, flat_img, 2, 2, Float32(0.0))) == 0.0)
    assert_true(flip_mean(flip_error_map(flat_img, flat_img, 2, 2, Float32(-1.0))) == 0.0)
    assert_true(flip_mean(flip_error_map(flat_img, flat_img, 2, 2, Float32(1.0e-25))) == 0.0)
    assert_true(flip_mean(flip_error_map(flat_img, flat_img, 2, 2, Float32(1.0e-38))) == 0.0)
    assert_true(flip_mean(flip_error_map(flat_img, flat_img, 2, 2, Float32(1.0e-45))) == 0.0)
    var diff_ppd0 = flip_mean(flip_error_map(black_img, white_img, 2, 2, Float32(0.0)))
    assert_true(diff_ppd0 > 0.5 and diff_ppd0 <= 1.0)
    var diff_ppd_tiny = flip_mean(flip_error_map(black_img, white_img, 2, 2, Float32(1.0e-25)))
    assert_true(diff_ppd_tiny > 0.5 and diff_ppd_tiny <= 1.0)
    var diff_ppd_sub = flip_mean(flip_error_map(black_img, white_img, 2, 2, Float32(1.0e-38)))
    assert_true(diff_ppd_sub > 0.5 and diff_ppd_sub <= 1.0)

    # 9. AaaRenderGraph.compile single-resource failure resets unaliased_transient_bytes
    var rg_single = AaaRenderGraph()
    var r_big = rg_single.add_resource("big", 1024, True, False, Int(1) << 62)
    _ = rg_single.add_pass("p0", [RgResourceAccess(r_big, RG_USAGE_COMPUTE_WRITE)])
    assert_false(rg_single.compile(1 << 20))
    assert_equal(rg_single.unaliased_transient_bytes, 0)
    assert_equal(rg_single.aliased_peak_bytes, 0)
    assert_equal(rg_single.aliasing_metrics().saved_bytes, 0)
    assert_true(rg_single.aliasing_savings_ratio() == Float32(0.0))


def main() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp_dir = "/tmp/flint_t5_cov_" + String(pid)
    _rm_dir_shallow(tmp_dir)
    assert_true(make_dirs(tmp_dir))

    test_anim_choreography_and_combat_timing(tmp_dir)
    print("[PASS] (1) flint.anim.{choreography,combat_timing} + overlap underflow & marker OOB")
    test_studio_editors_compare_and_room(tmp_dir)
    test_studio_command_menu()
    test_studio_motion_tools_and_scene_menu(tmp_dir)
    print("[PASS] (2) flint.studio.{combat_editor,command_menu,motion_tools,scene_menu,profile_compare,traversal_room,motion_editor,ide_pty}")
    test_dockspace_codecs_watcher_and_qoa()
    print("[PASS] (2b) DockSpace cycle rejection, PNG/JPEG max_size<=0, AssetWatcher '/' & QoaStreamCursor fsize")
    test_core_crash_and_log()
    print("[PASS] (3) flint.core.{crash,log}")
    test_physics_defaults_density_and_wind()
    print("[PASS] (4) flint.physics.{defaults,density_table,wind}")
    test_render_dither_lighting_layouts_and_particle_atlas()
    print("[PASS] (5) flint.render.{dither,gpu_layout,lighting,occupancy_layout,particle_atlas,particle_layout}")
    test_r3_partition_a_fixes()
    print("[PASS] (6) Round 3 Partition A: TTF/SDF non-monotonic ends, Studio STDV bounds, ide_view zero-extent glyph")
    test_r3_partition_b_fixes(tmp_dir)
    print("[PASS] (7) Round 3 Partition B: JobPool null guards, LZ4 raw<=0, Cycles empty/cps, RG/VSM OOB, Physics/Nav deser")
    test_r4_partition_a_fixes()
    print("[PASS] (8) Round 4 Partition A: inflate_zlib & DWA EXR bounds + sdf_bake extreme bounds")
    test_r4_partition_b_fixes()
    print("[PASS] (9) Round 4 Partition B: xpbd, flanim_ecs, scene flscnb, ragdoll, events & islands deser bounds")
    test_r5_partition_a_fixes(tmp_dir)
    print("[PASS] (10) Round 5 Partition A: hdr/exr/qoa alloc bounds, bvh8 slab idx, zstd/zlib caps, png overflow, mcp capture clamp")
    test_r5_partition_b_fixes(tmp_dir)
    print("[PASS] (11) Round 5 Partition B: pose_position/ragdoll, vehicle/compound/world deser, orca span cap, ml 0D/overflow, flip neg")
    test_r6_partition_a_fixes()
    print("[PASS] (12) Round 6 Partition A: cyclic BVH8 slab/traversal, GLB OOB index, JPEG SOF2 alloc, replay_stream reserve")
    test_r6_partition_b_fixes()
    print("[PASS] (13) Round 6 Partition B: Tensor/ops overflow, SPIR-V wc<2 & O(N) idom, NavMesh/PhysicsWorld deser, Cycles range/sr<=0")
    test_r6b_remediation_fixes()
    print("[PASS] (13b) Round 6b Remediation: NavMesh n_own==n_polys + carve_obstacles, ml.ops.{topk,matmul,resize_nearest,range_op,clip,layer_norm}")
    test_r6c_remediation_fixes()
    print("[PASS] (13c) Round 6c Remediation: ml.ops.{gemm,conv1d,slice,reduce,argmax,gather_elements,scatter_elements,range_op,binary}")
    test_r7_partition_ab_fixes(tmp_dir)
    print("[PASS] (14) Round 7 Partition A+B: ktx2 BasisLZ pre-alloc, BCn OOB, flpk audio slab, ttf fvar, ml gpu/protobuf/onnx, cycles caps")
    test_r8_partition_ab_fixes()
    print("[PASS] (15) Round 8 Partition A+B: ttf gdef/gsub/cff, dbus_parse, audio pitch/wrap, net mtu=1400 AEAD, cycles chord/fast/frac, simd_canvas blit, rg u32")
    test_r9_partition_ab_fixes()
    print("[PASS] (16) Round 9 Partition A+B: ttf type2 subr budget, exr zip/zips > raw, frac same-den 128b, read_bin guards, aead_open_raw alias, rg align overflow")
    test_r10_partition_ab_fixes()
    print("[PASS] (17) Round 10 Partition A+B: inflate_zlib window, exr dwa uncomp=0, ktx2 uncomp/bomb, spirv rtype OOB, aaa inactive readback, cycles depth/crush, forces read, flip ppd=0, rg single fail")

    _rm_dir_shallow(tmp_dir)
    print("=== ALL TIER 5 COVERAGE & REGRESSION TESTS PASSED ===")

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
from std.collections import Dict
from std.ffi import external_call
from std.math import abs, isfinite, isinf, isnan
from std.memory import bitcast
from std.testing import assert_equal, assert_false, assert_true
from std.time import perf_counter_ns
from flint.ai.behavior_tree import (
    BehaviorTree,
    Blackboard,
    BtExecutionState,
    UtilityAiScorer,
    UtilityConsideration,
    UtilityOption,
)
from flint.anim.animator import Animator
from flint.anim.choreography import MotionPlayer, MotionSequence, cue_kind
from flint.anim.combat_timing import CombatTiming
from flint.anim.flanim_ecs import (
    FLANIM_FOURCC,
    FLANIM_VERSION,
    compute_volume_preserving_smear_scale,
    load_flanim_bytes,
    save_flanim_bytes,
)
from flint.anim.graph import (
    AnimGraphDef,
    BlendSample2DDef,
    BlendSpace2DDef,
    Condition,
    LayerDef,
    StateDef,
    TransitionDef,
)
from flint.anim.motion_matching import MotionDatabase, MotionFeature
from flint.anim.plugin import AnimatorComponent, SkeletonComponent
from flint.anim.pose import Inertializer, Pose, SkeletonInstance, blend_into, sample_pose
from flint.app.engine_app import AppConfig, EngineApp, FixedTime, FrameTime
from flint.anim.procedural import (
    Breathing,
    FootPlantSolver,
    HeadStabilizer,
    Lean,
    LookAt,
    MotionWarp,
    OrientationWarp,
    Placement,
    Recoil,
    pose_position,
)
from flint.anim.ragdoll import ArticulatedLimbDef, ArticulatedRagdoll, RAGDOLL_BLOB_TAG, RAGDOLL_BLOB_VERSION
from flint.anim.sync import ClipInfo, wrap_time
from flint.anim.take_constraints import clean_stances, follow_path, pin_frame, pin_joint, warp_contact
from flint.anim.take_optimize import optimize_bake
from flint.assets.bc_compress import (
    decode_bc5_normals,
    decode_bc6h_hdr,
    decode_bc7_rgba,
    downsample_argb_2x,
    downsample_normals_2x,
    f32_to_uf16_bits,
)
from flint.assets.cooker_v2 import AssetMeta, AssetUuid, cook_native_model_bundle, cook_single_source_bytes
from flint.assets.flpk_v3 import (
    COOKED_BVH8_SLAB_MAGIC,
    decode_audio_pcm_slab,
    decode_bvh8_collider_slab,
    encode_audio_pcm_slab,
    encode_bvh8_collider_slab,
    parse_flmeta_bytes,
)
from flint.assets.glb import AnimationPlayback
from flint.assets.glb_native import parse_native_glb
from flint.assets.glb_take import append_take_glb
from flint.assets.glb_writer import encode_native_glb
from flint.assets.hdr_image import (
    EXR_DWAA_COMPRESSION,
    EXR_MAGIC,
    EXR_NO_COMPRESSION,
    EXR_PIXEL_FLOAT,
    EXR_ZIP_COMPRESSION,
    EXR_ZIPS_COMPRESSION,
    HdrImage,
    _dwa_to_linear,
    decode_exr_fp16,
    decode_radiance_hdr,
    encode_exr_fp16,
    encode_radiance_hdr,
)
from flint.assets.image import BitReader as ImageBitReader, decode_png, encode_png, inflate_zlib
from flint.assets.jpeg import decode_jpeg, encode_jpeg_baseline
from flint.assets.json import parse_json
from flint.assets.ktx2 import (
    VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    decode_ktx2,
    decompress_zstd,
)
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.meshlet import AaaGeometryPool, simplify_mesh_qem
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
from flint.audio.acoustics import AcousticMaterialTable, probe_room_sabine_rt60
from flint.audio.qoa import QOA_MAGIC, QoaStreamCursor, decode_qoa, encode_qoa
from flint.audio.plugin import AudioEmitterComponent
from flint.audio.server import AudioServer, BUS_SFX, KWeightedLufsMeter, TruePeakLimiter, compute_woodworth_hrtf
from flint.camera.rig import CAM_MODE_FREE_FLY, CameraRig, _det_perlin_1d
from flint.core.anchor import RawText
from flint.core.console import DevConsole
from flint.core.cvars import CVAR_DEBUG_RANGE, ConsoleVars, parse_value, show_value
from flint.core.ecs import ArchetypeSlots, FLINT_ECS_VERSION, ROW_LIVE
from flint.core.perf_history import (
    AcceptLine,
    PerfMetric,
    PerfTable,
    judge_from,
    parse_number as perf_parse_number,
    signed_percent,
)
from flint.core.offset_allocator import OffsetAllocation, OffsetAllocator
from flint.core.profiler import ProfileTimeline, percentile_of, report_ms
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
from flint.core.ik import (
    aim_rotation,
    clamp_cone_direction,
    fabrik,
    safe_normalized,
    solve_four_bar_linkage,
    solve_two_bone_pole,
    two_bone_ik,
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
from flint.core.math import Quat, Vec2, Vec3
from flint.core.springs import (
    SpringAngle,
    SpringFloat,
    SpringVec3,
    decay_spring,
    decay_spring_scalar,
    fast_negexp,
    wrap_degrees,
)
from flint.core.replay import ByteReader as ReplayByteReader, is_dev_record_at, read_dev_record, word_at
from flint.core.serial import BlobReader, BlobWriter, fourcc
from flint.cycles.pattern import Frac, K_ALT_ABS, Params, Pattern, bjorklund, chord_notes, mini, parse_number
from flint.cycles.script import eval_pattern, run_program
from flint.cycles.synth import Clip, CyclesEngine, render_loop_to_wav, render_to_wav
from flint.devtools.mcp_v2 import McpServerV2, _fmt_vec3_json, _render_studio_software_preview
from flint.ecs.world import World
from flint.input.evdev import EvdevGamepad, pack_ff_rumble_effect
from flint.input.input_map import ACTION_AXIS2D, AxisCurve, BIND_MOUSE_DELTA_2D, InputBinding, InputContext, InputMap
from flint.lab.bench_suite import _fmt_f64_3
from flint.ml.kimodo_bvh import write_kimodo_bvh
from flint.ml.motionbricks import MbClip, MbQuat, MbVec3, MbXform, load_bvh, quat_slerp, soma_joint_names, world_pose
from flint.ml.take_edit import edit_take, find_cycle
from flint.ml.onnx import OnnxModel
from flint.ml.protobuf import PbReader
from flint.ml.ops import (
    B_DIV,
    B_MOD,
    argmax,
    binary,
    cast,
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
    softmax,
    topk,
    transpose,
)
from flint.ml.tensor import DT_F32, DT_I32, DT_I64, Tensor, f32_from_bits, numel_of
from flint.nav.navmesh import NavMesh, NavMeshConfig, NavPoly, NavTile
from flint.nav.orca import CrowdSimulator
from flint.net.bitpack import BitReader as NetBitReader, BitWriter as NetBitWriter, pack_quat_smallest_three
from flint.net.channel import NetChannel
from flint.net.crypto import AeadKey, StatelessCookieAuth, aead_open_raw, aead_seal_raw
from flint.net.plugin import NetworkIdentityComponent
from flint.net.replicator import SnapshotClientReceiver, SnapshotReplicator
from flint.net.socket_udp import DatagramView, NetEndpoint
from flint.net.virtual_sim import VirtualNetSim
from flint.platform.audio import AUDIO_CMD_SET_BUS, AudioCommand, NativeAudio, WavClip, clip_sample
from flint.platform.dsp import (
    CytomicSVF,
    ItdDelayLine,
    ModalResonator,
    StereoChorus,
    StereoReverbFDN,
    TimpaniDrum,
    compute_spatial_3d,
    equal_power_gains,
    poly_blep_saw,
    soft_clip_tanh,
)
from flint.platform.input import InputState, PAD_AXIS_LEFT_X, PAD_AXIS_LEFT_Y, PAD_DEADZONE, PadState, identity_map
from flint.platform.logind import dbus_message_length, dbus_method_call, dbus_parse
from flint.platform.macos_audio import MacOsAudio
from flint.platform.win32_audio import Win32Audio
from flint.studio.command_menu import CommandMenu
from flint.platform.os import LinuxGamepad, list_directory, make_dirs, read_bytes, remove_file, write_bytes
from flint.physics.aero6dof import (
    Aero6DofConfig,
    Aero6DofControls,
    Aero6DofState,
    step_aero6dof,
    step_zem_apn_simd8,
)
from flint.physics.bvh import Bvh8MeshCollider, Bvh8Node, MeshTriangle
from flint.physics.character import CharacterController, CharacterControllerConfig
from flint.physics.solver_bodies import F8, SolverBodies, V8
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
from flint.physics.shapes_v5 import COMPOUND_BLOB_TAG, CompoundChild, CompoundShape, HeightfieldCollider, SHAPES_V5_BLOB_VERSION
from flint.physics.tires import PacejkaTire
from flint.physics.vehicle import VEHICLE_BLOB_TAG, VEHICLE_BLOB_VERSION, VehicleChassis
from flint.physics.water import (
    buoyant_acceleration,
    eval_beaufort_wave,
    eval_box_buoyancy_drag,
    eval_sphere_buoyancy_drag,
    submerged_fraction,
)
from flint.procedural.foliage import build_manicured_hedge_and_planter
from flint.procedural.loft_bevel import extrude_profile_beveled, sweep_profile_rmf_banked
from flint.procedural.subd import SubdCage
from flint.physics.world import (
    BODY_MOTION_STATIC,
    BODY_SHAPE_COMPOUND,
    BODY_SHAPE_CONVEX_HULL,
    PhysicsWorld,
    WORLD_BLOB_TAG,
)
from flint.physics.xpbd import ChainCollider, ParticleChain, ParticleMesh, RigidAttachment, XPBD_CHAIN_TAG, XPBD_MESH_TAG, XPBD_BLOB_VERSION
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
from flint.render.aaa_npr_vfx import eval_value_noise_periodic_cpu
from flint.render.aaa_post_tsr import eval_display_grade_cpu
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_render_graph import (
    AaaRenderGraph,
    RG_USAGE_COMPUTE_READ,
    RG_USAGE_COMPUTE_WRITE,
    RgResourceAccess,
)
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import RestirReservoir
from flint.render.aaa_sky_atmosphere import BrunetonAtmosphereParams, BrunetonSkyLuts, _bilerp_lut
from flint.render.aaa_vsm import VsmPageManager, compute_vsm_cascade_page_origins, get_vsm_cascade_index
from flint.render.debug_draw import DEBUG_RED, DebugDraw, DebugView
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
from flint.render.font import Glyph, SdfFont
from flint.render.fx_atlas import clamp01
from flint.render.grading import LookDirector, filmic_curve, to_byte as grading_to_byte
from flint.render.image_diff import compare_pixels
from flint.render.material_graph import MaterialAsset, bake_vat_crowd_clip_rgba16f, sample_vat_vertex_cpu
from flint.render.plugin import MeshRendererComponent
from flint.render.render_world import DebugLineBatch, RENDER_TIER_0_HEADLESS_SIMD, RenderWorld, ScalableRenderer
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
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    HierarchyDepth,
    LocalTransform,
    Parent,
    TransformPropagationCache,
    interpolate_global_transform,
    propagate_transforms_parallel_cached,
    register_transform_components,
    set_parent,
)
from flint.studio.combat_editor import CombatTimeline
from flint.studio.gizmo import (
    GIZMO_AXIS_NONE,
    GIZMO_AXIS_X,
    GIZMO_MODE_SCALE,
    GIZMO_MODE_TRANSLATE,
    GIZMO_SCALE_UNIFORM,
    ViewportRay,
    _emit_thick_line_ui,
    gizmo_rotation_delta_rad,
    gizmo_scale_drag,
    pick_gizmo_handle,
    ray_axis_closest_param,
    ray_plane_hit,
    snap_scalar,
)
from flint.studio.ide_pty import IdePty
from flint.studio.ide_view import raster_ui, toolbar_label
from flint.studio.motion_editor import FollowCamera, frame, next_clip, point, relative_to, timing_click
from flint.studio.motion_tools import MotionTools, CONTACT, TIMING, draw_dust, frame_joint, nearest_frame, shake_offset
from flint.studio.profile_compare import Capture
from flint.studio.scene_menu import SceneMenu
from flint.studio.studio_app import FlintStudioState, _fmt_f32_2dp
from flint.studio.traversal_room import TraversalRoom
from flint.text.line_break import item_widths, run_width, wrap_shaped
from flint.text.rich_text import RichSpan, SPAN_BUTTON, SPAN_TEXT, layout_rich, layout_rich_shaped
from flint.text.shaper import FontFallbackChain, TextShaper
from flint.ui.button_glyphs import ACTION_JUMP, SHAPE_STICK, pad_label, pad_shape
from flint.ui.context import UiContext, format_float2
from flint.text.sdf_bake import (
    EdgeSegment,
    Segments,
    bake_font,
    bake_glyph,
    bake_glyph_id,
    bake_glyph_msdf,
    bake_glyph_mtsdf,
    extract_edge_contours,
    fixed64,
    flatten,
    signed_distance,
)
from flint.text.ttf import GlyphOutline, GsubGlyph, TrueTypeFont
from flint.ui.draw_list import GpuUiPrim, PRIM_GLYPH, PRIM_SHAPE, UiDrawList
from flint.ui.flexbox_dock import DOCK_SPLIT_V, DockSpace, FLEX_ROW, FlexStyle, FlexTree, RectF
from flint.ui.layout import canvas_ceil, canvas_floor, ui_frame
from flint.ui.nine_slice import NineSlice, slice_map


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
    bw_pw_idx.begin(WORLD_BLOB_TAG, 3)  # v3 fixture with an invalid convex-hull index
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


def test_r11_partition_ab_fixes(tmp_dir: String) raises:
    # 1. TTF GSUB exponential hang (Type 5 / Type 6 OOB scnt & recursive budget) + valid GSUB
    var head_t = List[UInt8](length=54, fill=0)
    head_t[19] = UInt8(100)  # unitsPerEm = 100
    var hhea_t = List[UInt8](length=36, fill=0)
    hhea_t[35] = UInt8(2)
    var maxp_t = List[UInt8](length=6, fill=0)
    maxp_t[5] = UInt8(4)
    var hmtx_t = List[UInt8](length=8, fill=0)
    var loca_t = List[UInt8](length=10, fill=0)
    var glyf_t = List[UInt8](length=4, fill=0)
    var cmap_t = List[UInt8]()
    _r7_put_u16_be(cmap_t, 0); _r7_put_u16_be(cmap_t, 1)
    _r7_put_u16_be(cmap_t, 3); _r7_put_u16_be(cmap_t, 1); _r7_put_u32_be(cmap_t, 12)
    _r7_put_u16_be(cmap_t, 4); _r7_put_u16_be(cmap_t, 16); _r7_put_u16_be(cmap_t, 0)
    _r7_put_u16_be(cmap_t, 2); _r7_put_u16_be(cmap_t, 2); _r7_put_u16_be(cmap_t, 0); _r7_put_u16_be(cmap_t, 0)
    _r7_put_u16_be(cmap_t, 0xFFFF); _r7_put_u16_be(cmap_t, 0); _r7_put_u16_be(cmap_t, 0xFFFF)
    _r7_put_u16_be(cmap_t, 1); _r7_put_u16_be(cmap_t, 0)

    # 1a. Malicious GSUB placed last: Lookup 0 is ContextSubst fmt=3 (gcnt=1, scnt=65535) near EOF -> must finish fast
    var gsub_dos = List[UInt8]()
    _r7_put_u16_be(gsub_dos, 1); _r7_put_u16_be(gsub_dos, 0)
    _r7_put_u16_be(gsub_dos, 10); _r7_put_u16_be(gsub_dos, 34); _r7_put_u16_be(gsub_dos, 52)
    # ScriptList at +10: count=1, 'DFLT' off=8 (18)
    _r7_put_u16_be(gsub_dos, 1); _r7_put_u32_be(gsub_dos, 0x44464C54); _r7_put_u16_be(gsub_dos, 8)
    _r7_put_u16_be(gsub_dos, 4); _r7_put_u16_be(gsub_dos, 0)  # Script at +18 -> LangSys at +22
    _r7_put_u16_be(gsub_dos, 0); _r7_put_u16_be(gsub_dos, 0xFFFF); _r7_put_u16_be(gsub_dos, 1); _r7_put_u16_be(gsub_dos, 0)
    while len(gsub_dos) < 34:
        gsub_dos.append(UInt8(0))
    # FeatureList at +34: count=1, 'liga' off=8 (42)
    _r7_put_u16_be(gsub_dos, 1); _r7_put_u32_be(gsub_dos, 0x6C696761); _r7_put_u16_be(gsub_dos, 8)
    _r7_put_u16_be(gsub_dos, 0); _r7_put_u16_be(gsub_dos, 1); _r7_put_u16_be(gsub_dos, 0)
    while len(gsub_dos) < 52:
        gsub_dos.append(UInt8(0))
    # LookupList at +52: count=1, off[0]=4 (+56)
    _r7_put_u16_be(gsub_dos, 1); _r7_put_u16_be(gsub_dos, 4)
    # Lookup 0 at +56: ltype=5, lflag=0, subCount=1, subOff[0]=8 (+64)
    _r7_put_u16_be(gsub_dos, 5); _r7_put_u16_be(gsub_dos, 0); _r7_put_u16_be(gsub_dos, 1); _r7_put_u16_be(gsub_dos, 8)
    # ContextSubst fmt=3 at +64: fmt=3, gcnt=1, scnt=65535, covOff[0]=8 (+72)
    _r7_put_u16_be(gsub_dos, 3); _r7_put_u16_be(gsub_dos, 1); _r7_put_u16_be(gsub_dos, 65535); _r7_put_u16_be(gsub_dos, 8)
    # Coverage fmt=1 at +72: fmt=1, count=1, gid=1
    _r7_put_u16_be(gsub_dos, 1); _r7_put_u16_be(gsub_dos, 1); _r7_put_u16_be(gsub_dos, 1)

    var tags_dos: List[Int] = [
        0x636D6170, 0x676C7966, 0x68656164, 0x68686561,
        0x686D7478, 0x6C6F6361, 0x6D617870, 0x47535542,
    ]
    var tbls_dos: List[List[UInt8]] = [
        cmap_t.copy(), glyf_t.copy(), head_t.copy(), hhea_t.copy(),
        hmtx_t.copy(), loca_t.copy(), maxp_t.copy(), gsub_dos^,
    ]
    var f_dos = TrueTypeFont()
    assert_true(f_dos.load_bytes(_r8_build_sfnt(tags_dos, tbls_dos)))
    var g_dos: List[GsubGlyph] = [GsubGlyph(1)]
    f_dos.apply_gsub(g_dos, "DFLT")
    assert_equal(len(g_dos), 1)

    # 1b. In-bounds recursive Type 5 -> Type 5 cycle (scnt=4 pointing back to Lookup 0) + valid Type 5 -> Type 1
    var gsub_cyc = List[UInt8]()
    _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 0)
    _r7_put_u16_be(gsub_cyc, 10); _r7_put_u16_be(gsub_cyc, 34); _r7_put_u16_be(gsub_cyc, 52)
    _r7_put_u16_be(gsub_cyc, 1); _r7_put_u32_be(gsub_cyc, 0x44464C54); _r7_put_u16_be(gsub_cyc, 8)
    _r7_put_u16_be(gsub_cyc, 4); _r7_put_u16_be(gsub_cyc, 0)
    _r7_put_u16_be(gsub_cyc, 0); _r7_put_u16_be(gsub_cyc, 0xFFFF); _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 0)
    while len(gsub_cyc) < 34:
        gsub_cyc.append(UInt8(0))
    _r7_put_u16_be(gsub_cyc, 1); _r7_put_u32_be(gsub_cyc, 0x6C696761); _r7_put_u16_be(gsub_cyc, 8)
    _r7_put_u16_be(gsub_cyc, 0); _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 0)
    while len(gsub_cyc) < 52:
        gsub_cyc.append(UInt8(0))
    # LookupList at +52: count=2, off[0]=6 (+58), off[1]=36 (+88)
    _r7_put_u16_be(gsub_cyc, 2); _r7_put_u16_be(gsub_cyc, 6); _r7_put_u16_be(gsub_cyc, 36)
    # Lookup 0 at +58 (8 bytes): ltype=5, lflag=0, subCount=1, subOff=8 (+66)
    _r7_put_u16_be(gsub_cyc, 5); _r7_put_u16_be(gsub_cyc, 0); _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 8)
    # ContextSubst fmt=3 at +66 (22 bytes -> +88): fmt=3, gcnt=1, scnt=3, covOff=20 (+86? wait: 66+38=104), seq=[0->1, 0->0, 0->0]
    _r7_put_u16_be(gsub_cyc, 3); _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 3); _r7_put_u16_be(gsub_cyc, 40)  # cov at 66+40=106
    _r7_put_u16_be(gsub_cyc, 0); _r7_put_u16_be(gsub_cyc, 1)
    _r7_put_u16_be(gsub_cyc, 0); _r7_put_u16_be(gsub_cyc, 0)
    _r7_put_u16_be(gsub_cyc, 0); _r7_put_u16_be(gsub_cyc, 0)
    while len(gsub_cyc) < 88:
        gsub_cyc.append(UInt8(0))
    # Lookup 1 at +88 (8 bytes -> +96): SingleSubst fmt=1 (+1)
    _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 0); _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 8)
    # SingleSubst fmt=1 at +96 (6 bytes -> +102): fmt=1, covOff=10 (+106), deltaGlyphId=1
    _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 10); _r7_put_u16_be(gsub_cyc, 1)
    while len(gsub_cyc) < 106:
        gsub_cyc.append(UInt8(0))
    # Coverage fmt=1 at +106: gid=1
    _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 1); _r7_put_u16_be(gsub_cyc, 1)
    tbls_dos[7] = gsub_cyc^
    var f_cyc = TrueTypeFont()
    assert_true(f_cyc.load_bytes(_r8_build_sfnt(tags_dos, tbls_dos)))
    var g_cyc: List[GsubGlyph] = [GsubGlyph(1)]
    f_cyc.apply_gsub(g_cyc, "DFLT")
    assert_equal(g_cyc[0].glyph_id, 2, "Valid Type 5 -> Type 1 sub must still apply before recursive cycle terminates")

    # 2. TTF CFF charstring abort cleanup (op_count > 65536 and subr_call_count > 4096 must clear partial contours)
    var cs_70k = List[UInt8]()
    cs_70k.append(UInt8(139)); cs_70k.append(UInt8(139)); cs_70k.append(UInt8(21))  # 0 0 rmoveto
    for _ in range(25000):
        cs_70k.append(UInt8(140)); cs_70k.append(UInt8(139)); cs_70k.append(UInt8(5))  # 1 0 rlineto (3 ops * 25000 = 75000 > 65536)
    cs_70k.append(UInt8(14))
    var otf_70k = _r9_build_cff_otf(List[List[UInt8]](), [cs_70k^])
    var f_70k = TrueTypeFont()
    assert_true(f_70k.load_bytes(otf_70k^))
    var out_70k = f_70k.cff_outline(0)
    assert_equal(len(out_70k.xs), 0)
    assert_equal(len(out_70k.ys), 0)
    assert_equal(len(out_70k.on), 0)
    assert_equal(len(out_70k.ends), 0)

    var gs_line = List[UInt8]()
    gs_line.append(UInt8(140)); gs_line.append(UInt8(139)); gs_line.append(UInt8(5))  # 1 0 rlineto
    gs_line.append(UInt8(11))  # return
    var cs_4100 = List[UInt8]()
    cs_4100.append(UInt8(139)); cs_4100.append(UInt8(139)); cs_4100.append(UInt8(21))  # 0 0 rmoveto
    for _ in range(4100):
        cs_4100.append(UInt8(32)); cs_4100.append(UInt8(29))  # push 0 (with bias 107 -> 32-139=-107), callgsubr
    cs_4100.append(UInt8(14))
    var otf_4100 = _r9_build_cff_otf([gs_line^], [cs_4100^])
    var f_4100 = TrueTypeFont()
    assert_true(f_4100.load_bytes(otf_4100^))
    var out_4100 = f_4100.cff_outline(0)
    assert_equal(len(out_4100.xs), 0)
    assert_equal(len(out_4100.ends), 0)

    # 3. ktx2.decompress_zstd overflow-safe bounds
    var zstd_ok: List[UInt8] = [
        UInt8(0x28), UInt8(0xB5), UInt8(0x2F), UInt8(0xFD),
        UInt8(0x20), UInt8(0x01), UInt8(0x09), UInt8(0x00), UInt8(0x00), UInt8(0x5A),
    ]
    assert_false(decompress_zstd(zstd_ok, 1, 0x7FFFFFFFFFFFFFFF, 1)[0])
    assert_false(decompress_zstd(zstd_ok, 0x7FFFFFFFFFFFFFF0, 32, 1)[0])
    var z_good = decompress_zstd(zstd_ok, 0, len(zstd_ok), 1)
    assert_true(z_good[0])
    assert_equal(len(z_good[1]), 1)
    assert_equal(z_good[1][0], UInt8(0x5A))

    # 4. motionbricks.MbClip.sample 0-frame / NaN guard & load_bvh NaN/1e30/1e-30 rejection
    var mb_empty = MbClip()
    mb_empty.skeleton.names = ["Hips"]
    mb_empty.skeleton.parents = [-1]
    var ref_xf = MbXform.identity()
    ref_xf.p = MbVec3(0.0, 1.0, 0.0)
    mb_empty.skeleton.reference = [ref_xf]
    var p_empty = mb_empty.sample(Float32(0.0))
    assert_equal(len(p_empty), 1)
    assert_true(abs(p_empty[0].p.y - Float32(1.0)) < Float32(1e-5))
    mb_empty.sample_rate = Float32(0.0)
    assert_equal(mb_empty.duration(), Float32(0.0))

    var bvh_bad_fts: List[String] = [String("nan"), String("1e30"), String("1e-30")]
    for fi in range(len(bvh_bad_fts)):
        var bp = tmp_dir + "/bad_ft_" + String(fi) + ".bvh"
        var btxt = (
            "HIERARCHY\nROOT Hips\n{\n  OFFSET 0 0 0\n  CHANNELS 3 Xposition Yposition Zposition\n"
            "  End Site\n  {\n    OFFSET 0 1 0\n  }\n}\n"
            "MOTION\nFrames: 2\nFrame Time: " + bvh_bad_fts[fi] + "\n0 0 0\n0 0 0\n"
        )
        var bbytes = List[UInt8]()
        for b in btxt.as_bytes():
            bbytes.append(b)
        assert_true(write_bytes(bp, bbytes))
        var bvh_raised = False
        try:
            _ = load_bvh(bp)
        except:
            bvh_raised = True
        _ = remove_file(bp)
        assert_true(bvh_raised, "load_bvh must reject Frame Time: " + bvh_bad_fts[fi])

    # 5. take_optimize.optimize_bake empty anim / <2 keys / NaN tolerance & valid reduction
    var nm = NativeModel()
    var n0 = NativeNode()
    n0.name = "Root"
    nm.nodes.append(n0^)
    nm.parents.append(-1)
    nm.topo_order.append(0)
    var anim_0ch = NativeAnimation()
    nm.animations.append(anim_0ch^)
    var opt_0ch_raised = False
    try:
        _ = optimize_bake(nm, 0, Float32(0.01))
    except:
        opt_0ch_raised = True
    assert_true(opt_0ch_raised, "optimize_bake with 0 channels must raise Error")

    var ch0 = NativeChannel()
    ch0.node = 0
    ch0.path = 0
    var anim_0k = NativeAnimation()
    anim_0k.channels.append(ch0.copy())
    nm.animations.append(anim_0k^)
    var opt_0k_raised = False
    try:
        _ = optimize_bake(nm, 1, Float32(0.01))
    except:
        opt_0k_raised = True
    assert_true(opt_0k_raised, "optimize_bake with 0 keys must raise Error")

    var ch1 = NativeChannel()
    ch1.node = 0; ch1.path = 0
    ch1.times = [Float32(0.0)]
    ch1.values = [Float32(0.0), 0.0, 0.0]
    var anim_1k = NativeAnimation()
    anim_1k.channels.append(ch1^)
    nm.animations.append(anim_1k^)
    var opt_1k_raised = False
    try:
        _ = optimize_bake(nm, 2, Float32(0.01))
    except:
        opt_1k_raised = True
    assert_true(opt_1k_raised, "optimize_bake with 1 key must raise Error")

    var ch_spike = NativeChannel()
    ch_spike.node = 0; ch_spike.path = 0
    ch_spike.times = [Float32(0.0), Float32(0.5), Float32(1.0)]
    ch_spike.values = [Float32(0.0), 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    var anim_spike = NativeAnimation()
    anim_spike.duration = Float32(1.0)
    anim_spike.channels.append(ch_spike^)
    nm.animations.append(anim_spike^)
    var opt_nan_raised = False
    try:
        _ = optimize_bake(nm, 3, f32_from_bits(0x7FC00000))
    except:
        opt_nan_raised = True
    assert_true(opt_nan_raised, "optimize_bake with NaN tolerance must raise Error")
    assert_equal(len(nm.animations[3].channels[0].times), 3, "failed NaN optimize_bake must not drop keys")

    var peak_spike = optimize_bake(nm, 3, Float32(0.01))
    assert_true(peak_spike <= Float32(0.01))
    assert_equal(len(nm.animations[3].channels[0].times), 3, "spike key must be retained within tolerance")

    # 6. take_constraints clean_stances, pin_joint, pin_frame missing-joint & shallow-rig guards
    var c_1j = MbClip()
    c_1j.skeleton.names = ["Hips"]
    c_1j.skeleton.parents = [-1]
    c_1j.skeleton.reference = [MbXform.identity()]
    c_1j.frames = [[MbXform.identity()], [MbXform.identity()], [MbXform.identity()]]
    assert_equal(clean_stances(c_1j, Float32(0.1), Float32(0.05), 2), 0)
    pin_frame(c_1j, 0, 0, Vec3(0.0, 0.0, 0.0))
    var pin_miss_raised = False
    try:
        _ = pin_joint(c_1j, "LeftFoot", 0, 1, Vec3(0.0, 0.0, 0.0), 0, Float32(1.0))
    except:
        pin_miss_raised = True
    assert_true(pin_miss_raised, "pin_joint on missing LeftFoot must raise Error")

    var c_shallow = MbClip()
    c_shallow.skeleton.names = ["Hips", "LeftFoot"]
    c_shallow.skeleton.parents = [-1, 0]
    c_shallow.skeleton.reference = [MbXform.identity(), MbXform.identity()]
    c_shallow.frames = [[MbXform.identity(), MbXform.identity()], [MbXform.identity(), MbXform.identity()]]
    assert_equal(clean_stances(c_shallow, Float32(0.1), Float32(0.05), 2), 0)
    pin_frame(c_shallow, 0, 1, Vec3(0.0, 0.0, 0.0))
    var pin_shallow_raised = False
    try:
        _ = pin_joint(c_shallow, "LeftFoot", 0, 1, Vec3(0.0, 0.0, 0.0), 0, Float32(1.0))
    except:
        pin_shallow_raised = True
    assert_true(pin_shallow_raised, "pin_joint on shallow 2-bone chain must raise Error")

    var c_0f = MbClip()
    c_0f.skeleton.names = ["Hips"]
    c_0f.skeleton.parents = [-1]
    c_0f.skeleton.reference = [MbXform.identity()]
    assert_equal(clean_stances(c_0f, Float32(0.1), Float32(0.05), 2), 0)
    pin_frame(c_0f, 0, 0, Vec3(0.0, 0.0, 0.0))

    # 7. DebugDraw.paint null target & non-positive stride/dimensions safety + valid paint
    var dd = DebugDraw()
    dd.set_enabled(True, Float32(0.016))
    dd.line(Vec3(0.0, 0.0, 5.0), Vec3(1.0, 0.0, 5.0), DEBUG_RED)
    dd.text(Vec3(0.0, 0.0, 5.0), "hi", DEBUG_RED)
    var dv_ok = DebugView(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), Float32(64.0), 64, 64)
    dd.paint(dv_ok, 0, 64, UInt32(0), False)
    assert_equal(dd.lines_painted, 0)
    var fb = List[UInt32](length=64 * 64, fill=UInt32(0))
    var fb_ptr = Int(fb.unsafe_ptr())
    dd.paint(dv_ok, fb_ptr, 0, UInt32(0), False)
    assert_equal(dd.lines_painted, 0)
    var dv_w0 = DebugView(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), Float32(64.0), 0, 64)
    dd.paint(dv_w0, fb_ptr, 64, UInt32(0), False)
    assert_equal(dd.lines_painted, 0)
    dd.paint(dv_ok, fb_ptr, 64, UInt32(0xFF000000), True)
    assert_true(dd.lines_painted > 0)
    assert_true(len(dd.written) > 0)
    _ = len(fb)


def test_r11b_remediation_fixes() raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)
    var ninf_f32 = f32_from_bits(0xFF800000)

    # 1. optimize_bake rejects NaN times, non-monotonic times, and NaN values without mutating keys
    var nm = NativeModel()
    var n0 = NativeNode()
    n0.name = "Root"
    nm.nodes.append(n0^)
    nm.parents.append(-1)
    nm.topo_order.append(0)

    var ch_nan_t = NativeChannel()
    ch_nan_t.node = 0; ch_nan_t.path = 0
    ch_nan_t.times = [Float32(0.0), nan_f32, Float32(1.0)]
    ch_nan_t.values = [Float32(0.0), 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    var anim_nan_t = NativeAnimation()
    anim_nan_t.duration = Float32(1.0)
    anim_nan_t.channels.append(ch_nan_t^)
    nm.animations.append(anim_nan_t^)
    var raised_nan_t = False
    try:
        _ = optimize_bake(nm, 0, Float32(0.01))
    except:
        raised_nan_t = True
    assert_true(raised_nan_t, "optimize_bake must reject NaN keyframe times")
    assert_equal(len(nm.animations[0].channels[0].times), 3)

    var ch_nonmono = NativeChannel()
    ch_nonmono.node = 0; ch_nonmono.path = 0
    ch_nonmono.times = [Float32(1.0), Float32(0.5), Float32(0.0)]
    ch_nonmono.values = [Float32(0.0), 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    var anim_nonmono = NativeAnimation()
    anim_nonmono.duration = Float32(1.0)
    anim_nonmono.channels.append(ch_nonmono^)
    nm.animations.append(anim_nonmono^)
    var raised_nonmono = False
    try:
        _ = optimize_bake(nm, 1, Float32(0.01))
    except:
        raised_nonmono = True
    assert_true(raised_nonmono, "optimize_bake must reject decreasing keyframe times")
    assert_equal(len(nm.animations[1].channels[0].times), 3)

    var ch_nan_v = NativeChannel()
    ch_nan_v.node = 0; ch_nan_v.path = 0
    ch_nan_v.times = [Float32(0.0), Float32(0.5), Float32(1.0)]
    ch_nan_v.values = [Float32(0.0), 0.0, 0.0, 0.0, nan_f32, 0.0, 0.0, 0.0, 0.0]
    var anim_nan_v = NativeAnimation()
    anim_nan_v.duration = Float32(1.0)
    anim_nan_v.channels.append(ch_nan_v^)
    nm.animations.append(anim_nan_v^)
    var raised_nan_v = False
    try:
        _ = optimize_bake(nm, 2, Float32(0.01))
    except:
        raised_nan_v = True
    assert_true(raised_nan_v, "optimize_bake must reject NaN channel values")
    assert_equal(len(nm.animations[2].channels[0].times), 3)

    # 2. warp_contact rejects limit=NaN without mutating root
    var c_warp = MbClip()
    c_warp.sample_rate = Float32(30.0)
    c_warp.skeleton.names = ["Hips", "LeftLeg", "LeftShin", "LeftFoot"]
    c_warp.skeleton.parents = [-1, 0, 1, 2]
    c_warp.skeleton.reference = [MbXform.identity(), MbXform.identity(), MbXform.identity(), MbXform.identity()]
    c_warp.frames = [c_warp.skeleton.reference.copy(), c_warp.skeleton.reference.copy()]
    var warp_nan_raised = False
    try:
        warp_contact(c_warp, "LeftFoot", 0, 1, Vec3(100.0, 0.0, 0.0), nan_f32)
    except:
        warp_nan_raised = True
    assert_true(warp_nan_raised, "warp_contact must reject NaN limit")
    assert_equal(c_warp.frames[1][0].p.x, Float32(0.0))

    # 3. pin_joint, pin_frame, clean_stances, and world_pose survive malformed parents=[-1, 999, 1]
    var c_oob = MbClip()
    c_oob.sample_rate = Float32(30.0)
    c_oob.skeleton.names = ["Hips", "LeftShin", "LeftFoot"]
    c_oob.skeleton.parents = [-1, 999, 1]
    c_oob.skeleton.reference = [MbXform.identity(), MbXform.identity(), MbXform.identity()]
    c_oob.frames = [c_oob.skeleton.reference.copy(), c_oob.skeleton.reference.copy()]
    var wp_oob = world_pose(c_oob.skeleton, c_oob.frames[0], MbXform.identity())
    assert_equal(len(wp_oob), 3)
    pin_frame(c_oob, 0, 2, Vec3(0.0, 0.0, 0.0))
    assert_equal(clean_stances(c_oob, Float32(0.1), Float32(0.05), 2), 0)
    var pin_oob_raised = False
    try:
        _ = pin_joint(c_oob, "LeftFoot", 0, 1, Vec3(0.0, 0.0, 0.0), 0, Float32(1.0))
    except:
        pin_oob_raised = True
    assert_true(pin_oob_raised, "pin_joint must reject OOB parent chain without SIGTRAP")

    # 4. MbClip.sample(+Inf / 1e38 -> last, -Inf / NaN -> frame 0)
    var c_samp = MbClip()
    c_samp.sample_rate = Float32(30.0)
    c_samp.skeleton.names = ["Hips"]
    c_samp.skeleton.parents = [-1]
    c_samp.skeleton.reference = [MbXform.identity()]
    var f0 = MbXform.identity(); f0.p = MbVec3(1.0, 0.0, 0.0)
    var f1 = MbXform.identity(); f1.p = MbVec3(2.0, 0.0, 0.0)
    c_samp.frames = [[f0], [f1]]
    assert_true(abs(c_samp.sample(pinf_f32)[0].p.x - Float32(2.0)) < Float32(1e-5))
    assert_true(abs(c_samp.sample(Float32(1e38))[0].p.x - Float32(2.0)) < Float32(1e-5))
    assert_true(abs(c_samp.sample(ninf_f32)[0].p.x - Float32(1.0)) < Float32(1e-5))
    assert_true(abs(c_samp.sample(nan_f32)[0].p.x - Float32(1.0)) < Float32(1e-5))

    # 5. DebugDraw.paint clears self.target and self.row_pixels on invalid target/dimensions
    var dd = DebugDraw()
    dd.set_enabled(True, Float32(0.016))
    dd.line(Vec3(0.0, 0.0, 5.0), Vec3(1.0, 0.0, 5.0), DEBUG_RED)
    var dv_ok = DebugView(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), Float32(64.0), 64, 64)
    var fb = List[UInt32](length=64 * 64, fill=UInt32(0))
    var fb_ptr = Int(fb.unsafe_ptr())
    dd.paint(dv_ok, fb_ptr, 64, UInt32(0), False)
    assert_equal(dd.target, fb_ptr)
    assert_equal(dd.row_pixels, 64)
    var dv_w0 = DebugView(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), Float32(64.0), 0, 64)
    dd.paint(dv_w0, fb_ptr, 64, UInt32(0), False)
    assert_equal(dd.target, 0)
    assert_equal(dd.row_pixels, 0)
    dd.paint(dv_ok, fb_ptr, 64, UInt32(0), False)
    dd.paint(dv_ok, 0, 64, UInt32(0), False)
    assert_equal(dd.target, 0)
    assert_equal(dd.row_pixels, 0)
    _ = len(fb)


def _r12_minimal_ttf_tables(num_glyphs: Int, glyf: List[UInt8], loca: List[UInt8]) -> Tuple[List[Int], List[List[UInt8]]]:
    var head = List[UInt8](length=54, fill=0)
    head[19] = UInt8(100)  # unitsPerEm = 100
    var hhea = List[UInt8](length=36, fill=0)
    hhea[35] = UInt8(num_glyphs)
    var maxp = List[UInt8](length=6, fill=0)
    maxp[4] = UInt8((num_glyphs >> 8) & 0xFF)
    maxp[5] = UInt8(num_glyphs & 0xFF)
    var hmtx = List[UInt8](length=num_glyphs * 4, fill=0)
    for g in range(num_glyphs):
        hmtx[g * 4] = UInt8(0x01)
    var cmap = List[UInt8]()
    _r7_put_u16_be(cmap, 0); _r7_put_u16_be(cmap, 1)
    _r7_put_u16_be(cmap, 3); _r7_put_u16_be(cmap, 1); _r7_put_u32_be(cmap, 12)
    _r7_put_u16_be(cmap, 4); _r7_put_u16_be(cmap, 16); _r7_put_u16_be(cmap, 0)
    _r7_put_u16_be(cmap, 2); _r7_put_u16_be(cmap, 2); _r7_put_u16_be(cmap, 0); _r7_put_u16_be(cmap, 0)
    _r7_put_u16_be(cmap, 0xFFFF); _r7_put_u16_be(cmap, 0); _r7_put_u16_be(cmap, 0xFFFF)
    _r7_put_u16_be(cmap, 1); _r7_put_u16_be(cmap, 0)
    var tags: List[Int] = [
        0x636D6170, 0x676C7966, 0x68656164, 0x68686561,
        0x686D7478, 0x6C6F6361, 0x6D617870,
    ]
    var tbls: List[List[UInt8]] = [cmap^, glyf.copy(), head^, hhea^, hmtx^, loca.copy(), maxp^]
    return (tags^, tbls^)


def _r12_make_soma_clip(frames_count: Int, sample_rate: Float32) -> MbClip:
    var clip = MbClip()
    clip.sample_rate = sample_rate
    clip.skeleton.names = soma_joint_names()
    for j in range(30):
        clip.skeleton.parents.append(-1 if j == 0 else 0)
        clip.skeleton.reference.append(MbXform.identity())
    for f in range(frames_count):
        var pose = clip.skeleton.reference.copy()
        pose[0].p = MbVec3(Float32(f) * Float32(0.01), Float32(1.0), Float32(0.0))
        clip.frames.append(pose^)
    return clip^


def test_r12_partition_ab_fixes(tmp_dir: String) raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)

    # 1. TTF GSUB budget bypass hang (Type 5 fmt 1 fan-out with scnt=0) + valid Type 1 GSUB
    var loca_empty = List[UInt8](length=10, fill=0)
    var glyf_empty = List[UInt8](length=4, fill=0)
    var base1 = _r12_minimal_ttf_tables(4, glyf_empty, loca_empty)
    var tags_gsub = base1[0].copy()
    var tbls_gsub = base1[1].copy()
    tags_gsub.append(0x47535542)  # GSUB

    # Build Type 5 fmt 1 GSUB: 100 lookups -> 1 shared Lookup table (50 subtables -> 1 RuleSet with 50 rules, all scnt=0)
    var gsub_hang = List[UInt8]()
    _r7_put_u16_be(gsub_hang, 1); _r7_put_u16_be(gsub_hang, 0)
    _r7_put_u16_be(gsub_hang, 10); _r7_put_u16_be(gsub_hang, 34); _r7_put_u16_be(gsub_hang, 250)
    _r7_put_u16_be(gsub_hang, 1); _r7_put_u32_be(gsub_hang, 0x44464C54); _r7_put_u16_be(gsub_hang, 8)
    _r7_put_u16_be(gsub_hang, 4); _r7_put_u16_be(gsub_hang, 0)
    _r7_put_u16_be(gsub_hang, 0); _r7_put_u16_be(gsub_hang, 0xFFFF); _r7_put_u16_be(gsub_hang, 1); _r7_put_u16_be(gsub_hang, 0)
    while len(gsub_hang) < 34:
        gsub_hang.append(UInt8(0))
    # FeatureList at +34: 1 feature 'liga' referencing 100 lookups (0..99)
    _r7_put_u16_be(gsub_hang, 1); _r7_put_u32_be(gsub_hang, 0x6C696761); _r7_put_u16_be(gsub_hang, 8)
    _r7_put_u16_be(gsub_hang, 0); _r7_put_u16_be(gsub_hang, 100)
    for li in range(100):
        _r7_put_u16_be(gsub_hang, li)
    while len(gsub_hang) < 250:
        gsub_hang.append(UInt8(0))
    # LookupList at +250: 100 lookups all pointing to +250 + 2 + 200 = +452
    _r7_put_u16_be(gsub_hang, 100)
    for _ in range(100):
        _r7_put_u16_be(gsub_hang, 2 + 200)
    # Lookup at +452: ltype=5, lflag=0, subCount=50, all pointing to subtable at +452 + 6 + 100 = +558
    _r7_put_u16_be(gsub_hang, 5); _r7_put_u16_be(gsub_hang, 0); _r7_put_u16_be(gsub_hang, 50)
    for _ in range(50):
        _r7_put_u16_be(gsub_hang, 6 + 100)
    # Subtable Type 5 fmt 1 at +558: fmt=1, covOff=8 (+566), rsetCnt=1, rsetOff[0]=14 (+572)
    _r7_put_u16_be(gsub_hang, 1); _r7_put_u16_be(gsub_hang, 8); _r7_put_u16_be(gsub_hang, 1); _r7_put_u16_be(gsub_hang, 14)
    # Coverage fmt=1 at +566: gid=1
    _r7_put_u16_be(gsub_hang, 1); _r7_put_u16_be(gsub_hang, 1); _r7_put_u16_be(gsub_hang, 1)
    # RuleSet at +572: rcnt=50, all pointing to non-matching rule at +572 + 2 + 100 = +674
    _r7_put_u16_be(gsub_hang, 50)
    for _ in range(50):
        _r7_put_u16_be(gsub_hang, 2 + 100)
    # Rule at +674: gcnt=2 (expects glyph 99 next -> never matches), scnt=0
    _r7_put_u16_be(gsub_hang, 2); _r7_put_u16_be(gsub_hang, 0); _r7_put_u16_be(gsub_hang, 99)
    tbls_gsub.append(gsub_hang^)

    var f_gsub_hang = TrueTypeFont()
    assert_true(f_gsub_hang.load_bytes(_r8_build_sfnt(tags_gsub, tbls_gsub)))
    var glyphs_hang = List[GsubGlyph]()
    for _ in range(32):
        glyphs_hang.append(GsubGlyph(1))
    f_gsub_hang.apply_gsub(glyphs_hang, "DFLT")
    assert_equal(len(glyphs_hang), 32)

    # Valid GSUB Type 5 fmt 1 -> Type 1 fmt 1 substitution
    var gsub_ok5 = List[UInt8]()
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 0)
    _r7_put_u16_be(gsub_ok5, 10); _r7_put_u16_be(gsub_ok5, 34); _r7_put_u16_be(gsub_ok5, 52)
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u32_be(gsub_ok5, 0x44464C54); _r7_put_u16_be(gsub_ok5, 8)
    _r7_put_u16_be(gsub_ok5, 4); _r7_put_u16_be(gsub_ok5, 0)
    _r7_put_u16_be(gsub_ok5, 0); _r7_put_u16_be(gsub_ok5, 0xFFFF); _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 0)
    while len(gsub_ok5) < 34:
        gsub_ok5.append(UInt8(0))
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u32_be(gsub_ok5, 0x6C696761); _r7_put_u16_be(gsub_ok5, 8)
    _r7_put_u16_be(gsub_ok5, 0); _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 0)
    while len(gsub_ok5) < 52:
        gsub_ok5.append(UInt8(0))
    # LookupList at +52: count=2, off[0]=6 (+58), off[1]=34 (+86)
    _r7_put_u16_be(gsub_ok5, 2); _r7_put_u16_be(gsub_ok5, 6); _r7_put_u16_be(gsub_ok5, 34)
    # Lookup 0 at +58: ltype=5, lflag=0, subCount=1, subOff=8 (+66)
    _r7_put_u16_be(gsub_ok5, 5); _r7_put_u16_be(gsub_ok5, 0); _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 8)
    # Type 5 fmt 1 at +66: fmt=1, covOff=34 (+100), rsetCnt=1, rsetOff[0]=8 (+74)
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 34); _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 8)
    # RuleSet at +74: rcnt=1, ruleOff[0]=4 (+78)
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 4)
    # Rule at +78: gcnt=1, scnt=1, seq=[0, 1] (+78..+86)
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 0); _r7_put_u16_be(gsub_ok5, 1)
    # Lookup 1 at +86: ltype=1, lflag=0, subCount=1, subOff=8 (+94)
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 0); _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 8)
    # SingleSubst fmt 1 at +94: fmt=1, covOff=6 (+100), delta=1
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 6); _r7_put_u16_be(gsub_ok5, 1)
    # Coverage at +100: fmt=1, count=1, gid=1
    _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 1); _r7_put_u16_be(gsub_ok5, 1)
    tbls_gsub[len(tbls_gsub) - 1] = gsub_ok5^
    var f_gsub_ok5 = TrueTypeFont()
    assert_true(f_gsub_ok5.load_bytes(_r8_build_sfnt(tags_gsub, tbls_gsub)))
    var glyphs_ok5: List[GsubGlyph] = [GsubGlyph(1)]
    f_gsub_ok5.apply_gsub(glyphs_ok5, "DFLT")
    assert_equal(glyphs_ok5[0].glyph_id, 2)

    # 2. TTF outline_at composite exponential blowup (16-component self-referential glyph 0) + valid composite
    var glyf_comp_loop = List[UInt8]()
    _r7_put_u16_be(glyf_comp_loop, 0xFFFF)  # numberOfContours = -1 (composite)
    _r7_put_u16_be(glyf_comp_loop, 0); _r7_put_u16_be(glyf_comp_loop, 0)
    _r7_put_u16_be(glyf_comp_loop, 100); _r7_put_u16_be(glyf_comp_loop, 100)
    for ci in range(16):
        var flags = 0x0023 if ci < 15 else 0x0003  # ARGS_ARE_XY_VALUES | MORE_COMPONENTS (except last)
        _r7_put_u16_be(glyf_comp_loop, flags)
        _r7_put_u16_be(glyf_comp_loop, 0)  # self-ref glyph 0
        _r7_put_u16_be(glyf_comp_loop, 0); _r7_put_u16_be(glyf_comp_loop, 0)
    var loca_comp_loop = List[UInt8]()
    _r7_put_u16_be(loca_comp_loop, 0)
    _r7_put_u16_be(loca_comp_loop, len(glyf_comp_loop) // 2)
    _r7_put_u16_be(loca_comp_loop, len(glyf_comp_loop) // 2)
    var comp_loop_base = _r12_minimal_ttf_tables(2, glyf_comp_loop, loca_comp_loop)
    var f_comp_loop = TrueTypeFont()
    assert_true(f_comp_loop.load_bytes(_r8_build_sfnt(comp_loop_base[0], comp_loop_base[1])))
    var out_comp_loop = f_comp_loop.outline(0)
    assert_equal(len(out_comp_loop.xs), 0, "16^8 self-referential composite glyph must abort cleanly")
    assert_equal(len(out_comp_loop.ends), 0)

    # Valid composite glyph 0 referencing 2 copies of simple triangle glyph 1
    var glyf_comp_ok = List[UInt8]()
    _r7_put_u16_be(glyf_comp_ok, 0xFFFF)
    _r7_put_u16_be(glyf_comp_ok, 0); _r7_put_u16_be(glyf_comp_ok, 0)
    _r7_put_u16_be(glyf_comp_ok, 200); _r7_put_u16_be(glyf_comp_ok, 100)
    _r7_put_u16_be(glyf_comp_ok, 0x0023); _r7_put_u16_be(glyf_comp_ok, 1); _r7_put_u16_be(glyf_comp_ok, 0); _r7_put_u16_be(glyf_comp_ok, 0)
    _r7_put_u16_be(glyf_comp_ok, 0x0003); _r7_put_u16_be(glyf_comp_ok, 1); _r7_put_u16_be(glyf_comp_ok, 100); _r7_put_u16_be(glyf_comp_ok, 0)
    var g1_off = len(glyf_comp_ok)
    _r7_put_u16_be(glyf_comp_ok, 1)
    _r7_put_u16_be(glyf_comp_ok, 0); _r7_put_u16_be(glyf_comp_ok, 0); _r7_put_u16_be(glyf_comp_ok, 50); _r7_put_u16_be(glyf_comp_ok, 50)
    _r7_put_u16_be(glyf_comp_ok, 2); _r7_put_u16_be(glyf_comp_ok, 0)
    glyf_comp_ok.append(UInt8(0x01)); glyf_comp_ok.append(UInt8(0x01)); glyf_comp_ok.append(UInt8(0x01))
    _r7_put_u16_be(glyf_comp_ok, 0); _r7_put_u16_be(glyf_comp_ok, 50); _r7_put_u16_be(glyf_comp_ok, 0)
    _r7_put_u16_be(glyf_comp_ok, 0); _r7_put_u16_be(glyf_comp_ok, 0); _r7_put_u16_be(glyf_comp_ok, 50)
    while len(glyf_comp_ok) % 2 != 0:
        glyf_comp_ok.append(UInt8(0))
    var loca_comp_ok = List[UInt8]()
    _r7_put_u16_be(loca_comp_ok, 0); _r7_put_u16_be(loca_comp_ok, g1_off // 2); _r7_put_u16_be(loca_comp_ok, len(glyf_comp_ok) // 2)
    var comp_ok_base = _r12_minimal_ttf_tables(2, glyf_comp_ok, loca_comp_ok)
    var f_comp_ok = TrueTypeFont()
    assert_true(f_comp_ok.load_bytes(_r8_build_sfnt(comp_ok_base[0], comp_ok_base[1])))
    var out_comp_ok = f_comp_ok.outline(0)
    assert_equal(len(out_comp_ok.xs), 6)

    # 3. TTF GPOS find_kern_lookups deduplication & kerning subtable count clamp + valid pair kern
    var gpos_dos = List[UInt8]()
    _r7_put_u16_be(gpos_dos, 1); _r7_put_u16_be(gpos_dos, 0)
    _r7_put_u16_be(gpos_dos, 10); _r7_put_u16_be(gpos_dos, 12); _r7_put_u16_be(gpos_dos, 4030)
    _r7_put_u16_be(gpos_dos, 0)  # ScriptList empty at +10
    # FeatureList at +12: 1 feature 'kern' referencing 2000 lookups
    _r7_put_u16_be(gpos_dos, 1); _r7_put_u32_be(gpos_dos, 0x6B65726E); _r7_put_u16_be(gpos_dos, 8)
    _r7_put_u16_be(gpos_dos, 0); _r7_put_u16_be(gpos_dos, 2000)
    for li in range(2000):
        _r7_put_u16_be(gpos_dos, li)
    while len(gpos_dos) < 4030:
        gpos_dos.append(UInt8(0))
    # LookupList at +4030: 2000 lookups all pointing to 2 + 4000 = 4002 (+8032)
    _r7_put_u16_be(gpos_dos, 2000)
    for _ in range(2000):
        _r7_put_u16_be(gpos_dos, 4002)
    # Lookup at EOF (+8032): kind=2, flag=0, subCount=65535
    _r7_put_u16_be(gpos_dos, 2); _r7_put_u16_be(gpos_dos, 0); _r7_put_u16_be(gpos_dos, 65535)
    var tags_gpos = base1[0].copy()
    var tbls_gpos = base1[1].copy()
    tags_gpos.append(0x47504F53)  # GPOS
    tbls_gpos.append(gpos_dos^)
    var f_gpos_dos = TrueTypeFont()
    assert_true(f_gpos_dos.load_bytes(_r8_build_sfnt(tags_gpos, tbls_gpos)))
    assert_equal(len(f_gpos_dos.kern_lookups), 1, "Duplicate GPOS kern lookup offsets must be deduplicated")
    assert_equal(f_gpos_dos.kerning(1, 2), Float32(0.0))

    # Valid GPOS pair adjustment (left=1, right=2 -> -40)
    var gpos_ok = List[UInt8]()
    _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 0)
    _r7_put_u16_be(gpos_ok, 10); _r7_put_u16_be(gpos_ok, 12); _r7_put_u16_be(gpos_ok, 26)
    _r7_put_u16_be(gpos_ok, 0)  # +10 ScriptList
    _r7_put_u16_be(gpos_ok, 1); _r7_put_u32_be(gpos_ok, 0x6B65726E); _r7_put_u16_be(gpos_ok, 8)  # +12 FeatureList
    _r7_put_u16_be(gpos_ok, 0); _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 0)  # +20 Feature
    _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 4)  # LookupList at +26 -> Lookup at +30
    _r7_put_u16_be(gpos_ok, 2); _r7_put_u16_be(gpos_ok, 0); _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 8)  # Lookup at +30 -> Subtable at +38
    # PairPos fmt=1 at +38: fmt=1, covOff=12 (+50), valFmt1=4 (XAdvance), valFmt2=0, pairSetCnt=1, pairSetOff[0]=18 (+56)
    _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 12); _r7_put_u16_be(gpos_ok, 4); _r7_put_u16_be(gpos_ok, 0)
    _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 18)
    _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 1)  # Coverage at +50: gid=1
    _r7_put_u16_be(gpos_ok, 1); _r7_put_u16_be(gpos_ok, 2); _r7_put_u16_be(gpos_ok, 0xFFD8)  # PairSet at +56: right=2, xAdvance=-40
    tbls_gpos[len(tbls_gpos) - 1] = gpos_ok^
    var f_gpos_ok = TrueTypeFont()
    assert_true(f_gpos_ok.load_bytes(_r8_build_sfnt(tags_gpos, tbls_gpos)))
    assert_equal(f_gpos_ok.kerning(1, 2), Float32(-40.0))

    # 4. rich_text bake_px <= 0 division-by-zero NaN guard
    var font_zero_bake = SdfFont()
    font_zero_bake.bake_px = Float32(0.0)
    var rt_spans: List[RichSpan] = [RichSpan(SPAN_TEXT, "hello", -1, 0, 100, 0)]
    var lay_zero = layout_rich(rt_spans, font_zero_bake, font_zero_bake, Float32(24.0), 0, Float32(0.0), 0, False, False)
    assert_equal(lay_zero.lines, 1)
    assert_false(isnan(lay_zero.width) or isinf(lay_zero.width))
    assert_false(isnan(lay_zero.height) or isinf(lay_zero.height))
    assert_equal(len(lay_zero.runs), 1)
    assert_false(isnan(lay_zero.runs[0].width) or isinf(lay_zero.runs[0].width))
    assert_false(isnan(lay_zero.runs[0].baseline) or isinf(lay_zero.runs[0].baseline))
    var lay_shaped_zero = layout_rich_shaped(rt_spans, font_zero_bake, font_zero_bake, Float32(24.0), 0, Float32(0.0), 0, False, False)
    assert_equal(lay_shaped_zero.lines, 1)
    assert_false(isnan(lay_shaped_zero.width) or isinf(lay_shaped_zero.width))
    assert_false(isnan(lay_shaped_zero.height) or isinf(lay_shaped_zero.height))

    var font_ok_bake = SdfFont()
    font_ok_bake.bake_px = Float32(48.0)
    font_ok_bake.line_height = Float32(56.0)
    font_ok_bake.cap_height = Float32(36.0)
    var lay_ok = layout_rich(rt_spans, font_ok_bake, font_ok_bake, Float32(24.0), 0, Float32(0.0), 0, False, False)
    assert_true(lay_ok.height > Float32(0.0) and isfinite(lay_ok.height))

    # 5. DebugDraw row_pixels < view.width OOB, plot bounds, and NaN endpoint rejection
    var dd = DebugDraw()
    dd.set_enabled(True, Float32(0.016))
    dd.line(Vec3(0.0, -2.4, 5.0), Vec3(2.4, -2.4, 5.0), DEBUG_RED)
    var dv_64 = DebugView(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), Float32(64.0), 64, 64)
    var canary_buf = List[UInt32](length=256 + 512, fill=UInt32(0xDEADBEEF))
    dd.paint(dv_64, Int(canary_buf.unsafe_ptr()), 4, UInt32(0), False)
    assert_equal(dd.lines_painted, 0)
    assert_equal(dd.target, 0)
    assert_equal(dd.view_w, 0)
    for idx in range(256, 256 + 512):
        assert_equal(canary_buf[idx], UInt32(0xDEADBEEF))

    var dv_8 = DebugView(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0), Float32(8.0), 8, 8)
    var buf_8 = List[UInt32](length=64 + 128, fill=UInt32(0xCAFEBABE))
    dd.paint(dv_8, Int(buf_8.unsafe_ptr()), 8, UInt32(0), False)
    dd.plot(0, 10, UInt32(0x11223344))
    dd.plot(10, 0, UInt32(0x11223344))
    dd.plot(-1, 0, UInt32(0x11223344))
    for idx in range(64, 64 + 128):
        assert_equal(buf_8[idx], UInt32(0xCAFEBABE))

    var fb_nan = List[UInt32](length=64 * 64, fill=UInt32(0))
    dd.shapes.clear()
    dd.line(Vec3(nan_f32, 0.0, 5.0), Vec3(0.0, 0.0, 5.0), DEBUG_RED)
    dd.text(Vec3(nan_f32, 0.0, 5.0), "nan", DEBUG_RED)
    dd.paint(dv_64, Int(fb_nan.unsafe_ptr()), 64, UInt32(0xFF000000), False)
    assert_equal(dd.lines_painted, 0)
    assert_equal(fb_nan[0], UInt32(0))

    dd.line(Vec3(0.0, 0.0, 5.0), Vec3(1.0, 0.0, 5.0), DEBUG_RED)
    dd.paint(dv_64, Int(fb_nan.unsafe_ptr()), 64, UInt32(0xFF000000), False)
    assert_equal(dd.lines_painted, 1)
    _ = len(canary_buf); _ = len(buf_8); _ = len(fb_nan)

    # 6. take_constraints follow_path empty frame & NaN points + pin_joint NaN/Inf goal + pin_frame NaN
    var c_path_empty = _r12_make_soma_clip(3, Float32(30.0))
    c_path_empty.frames[1] = List[MbXform]()
    var fp_empty_raised = False
    try:
        follow_path(c_path_empty, [0, 2], [Vec3(0, 0, 0), Vec3(1, 0, 0)])
    except:
        fp_empty_raised = True
    assert_true(fp_empty_raised, "follow_path with empty frame must raise Error")

    var c_path_nan = _r12_make_soma_clip(3, Float32(30.0))
    var fp_nan_raised = False
    try:
        follow_path(c_path_nan, [0, 2], [Vec3(nan_f32, 0, 0), Vec3(1, 0, 0)])
    except:
        fp_nan_raised = True
    assert_true(fp_nan_raised, "follow_path with NaN point must raise Error")
    assert_false(isnan(c_path_nan.frames[0][0].p.x))

    var c_pin = _r12_make_soma_clip(3, Float32(30.0))
    c_pin.skeleton.parents = [-1]
    for j in range(1, 30):
        c_pin.skeleton.parents.append(j - 1)
    var pin_nan_raised = False
    try:
        _ = pin_joint(c_pin, "LeftFoot", 0, 2, Vec3(nan_f32, 0.0, 0.0), 0, Float32(1.0))
    except:
        pin_nan_raised = True
    assert_true(pin_nan_raised, "pin_joint with NaN goal must raise Error")
    var pin_inf_raised = False
    try:
        _ = pin_joint(c_pin, "LeftFoot", 0, 2, Vec3(pinf_f32, 0.0, 0.0), 0, Float32(1.0))
    except:
        pin_inf_raised = True
    assert_true(pin_inf_raised, "pin_joint with +Inf goal must raise Error")
    pin_frame(c_pin, 0, c_pin.skeleton.joint_index("LeftFoot"), Vec3(nan_f32, 0.0, 0.0))
    assert_false(isnan(c_pin.frames[0][0].q.w))

    var c_path_ok = _r12_make_soma_clip(3, Float32(30.0))
    follow_path(c_path_ok, [0, 2], [Vec3(0.0, 0.0, 0.0), Vec3(2.0, 0.0, 1.0)])
    assert_true(abs(c_path_ok.frames[2][0].p.x - Float32(2.0)) < Float32(1e-4))

    # 7. take_edit edit_take & find_cycle empty frame & NaN/<=0 sample_rate
    var c_edit_empty = _r12_make_soma_clip(3, Float32(30.0))
    c_edit_empty.frames[1] = List[MbXform]()
    var edit_empty_raised = False
    try:
        _ = edit_take(c_edit_empty, 0, 2, False, True, False)
    except:
        edit_empty_raised = True
    assert_true(edit_empty_raised, "edit_take with empty frame must raise Error")

    var c_cyc_empty = _r12_make_soma_clip(20, Float32(30.0))
    c_cyc_empty.frames[5] = List[MbXform]()
    var cyc_empty_raised = False
    try:
        _ = find_cycle(c_cyc_empty, 0, 19)
    except:
        cyc_empty_raised = True
    assert_true(cyc_empty_raised, "find_cycle with empty frame must raise Error")

    var c_cyc_nan_sr = _r12_make_soma_clip(20, nan_f32)
    var cyc_nan_raised = False
    try:
        _ = find_cycle(c_cyc_nan_sr, 0, 19)
    except:
        cyc_nan_raised = True
    assert_true(cyc_nan_raised, "find_cycle with NaN sample_rate must raise Error")

    var c_cyc_ok = _r12_make_soma_clip(34, Float32(24.0))
    var cyc_res = find_cycle(c_cyc_ok, 0, 33)
    assert_true(cyc_res[1] >= cyc_res[0])
    var edited_ok = edit_take(c_cyc_ok, 0, 5, True, True, False)
    assert_equal(edited_ok.num_frames(), 6)

    # 8. kimodo_bvh write_kimodo_bvh short frame & <=0 / NaN sample_rate + valid round-trip
    var bvh_path = tmp_dir + "/r12_bvh_test.bvh"
    var c_bvh_short = _r12_make_soma_clip(2, Float32(30.0))
    c_bvh_short.frames[1].resize(5, MbXform.identity())
    var bvh_short_raised = False
    try:
        write_kimodo_bvh(c_bvh_short, bvh_path)
    except:
        bvh_short_raised = True
    assert_true(bvh_short_raised, "write_kimodo_bvh with short frame must raise Error")

    var c_bvh_sr0 = _r12_make_soma_clip(2, Float32(0.0))
    var bvh_sr0_raised = False
    try:
        write_kimodo_bvh(c_bvh_sr0, bvh_path)
    except:
        bvh_sr0_raised = True
    assert_true(bvh_sr0_raised, "write_kimodo_bvh with sample_rate=0 must raise Error")

    var c_bvh_ok = _r12_make_soma_clip(3, Float32(30.0))
    write_kimodo_bvh(c_bvh_ok, bvh_path)
    var bvh_reloaded = load_bvh(bvh_path)
    _ = remove_file(bvh_path)
    assert_equal(bvh_reloaded.num_frames(), 3)
    assert_equal(bvh_reloaded.sample_rate, Float32(30.0))


def test_r13_partition_ab_fixes(tmp_dir: String) raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)
    var nan_f64 = Float64(nan_f32)
    var pinf_f64 = Float64(pinf_f32)

    # A1: TTF GSUB IGNORE_MARKS + _next_non_ignored / _prev_non_ignored budget exhaustion (Type 5 fmt 1, Type 4, Type 6 fmt 1)
    var loca_empty = List[UInt8](length=14, fill=0)
    var glyf_empty = List[UInt8](length=4, fill=0)
    var base_a1 = _r12_minimal_ttf_tables(6, glyf_empty, loca_empty)

    # Build GDEF with Mark GlyphClassDef: glyph 2 -> class 3 (Mark)
    var gdef = List[UInt8]()
    _r7_put_u16_be(gdef, 1); _r7_put_u16_be(gdef, 0)  # version 1.0
    _r7_put_u16_be(gdef, 12)  # glyphClassDefOffset = 12
    _r7_put_u16_be(gdef, 0); _r7_put_u16_be(gdef, 0); _r7_put_u16_be(gdef, 0)
    # ClassDef fmt 1 at +12: startGlyphID=2, glyphCount=1, classValue=3
    _r7_put_u16_be(gdef, 1); _r7_put_u16_be(gdef, 2); _r7_put_u16_be(gdef, 1); _r7_put_u16_be(gdef, 3)

    # Type 5 fmt 1 GSUB with LOOKUP_FLAG_IGNORE_MARKS (0x0008), 100 lookups -> 1 Lookup (50 subtables -> 50 rules, gcnt=2)
    var gsub_t5 = List[UInt8]()
    _r7_put_u16_be(gsub_t5, 1); _r7_put_u16_be(gsub_t5, 0)
    _r7_put_u16_be(gsub_t5, 10); _r7_put_u16_be(gsub_t5, 34); _r7_put_u16_be(gsub_t5, 250)
    _r7_put_u16_be(gsub_t5, 1); _r7_put_u32_be(gsub_t5, 0x44464C54); _r7_put_u16_be(gsub_t5, 8)
    _r7_put_u16_be(gsub_t5, 4); _r7_put_u16_be(gsub_t5, 0)
    _r7_put_u16_be(gsub_t5, 0); _r7_put_u16_be(gsub_t5, 0xFFFF); _r7_put_u16_be(gsub_t5, 1); _r7_put_u16_be(gsub_t5, 0)
    while len(gsub_t5) < 34:
        gsub_t5.append(UInt8(0))
    _r7_put_u16_be(gsub_t5, 1); _r7_put_u32_be(gsub_t5, 0x6C696761); _r7_put_u16_be(gsub_t5, 8)
    _r7_put_u16_be(gsub_t5, 0); _r7_put_u16_be(gsub_t5, 100)
    for li in range(100):
        _r7_put_u16_be(gsub_t5, li)
    while len(gsub_t5) < 250:
        gsub_t5.append(UInt8(0))
    _r7_put_u16_be(gsub_t5, 100)
    for _ in range(100):
        _r7_put_u16_be(gsub_t5, 2 + 200)
    _r7_put_u16_be(gsub_t5, 5); _r7_put_u16_be(gsub_t5, 0x0008); _r7_put_u16_be(gsub_t5, 50)
    for _ in range(50):
        _r7_put_u16_be(gsub_t5, 6 + 100)
    _r7_put_u16_be(gsub_t5, 1); _r7_put_u16_be(gsub_t5, 8); _r7_put_u16_be(gsub_t5, 1); _r7_put_u16_be(gsub_t5, 14)
    _r7_put_u16_be(gsub_t5, 1); _r7_put_u16_be(gsub_t5, 1); _r7_put_u16_be(gsub_t5, 1)  # Coverage gid=1
    _r7_put_u16_be(gsub_t5, 50)
    for _ in range(50):
        _r7_put_u16_be(gsub_t5, 2 + 100)
    _r7_put_u16_be(gsub_t5, 2); _r7_put_u16_be(gsub_t5, 0); _r7_put_u16_be(gsub_t5, 99)

    var tags_a1 = base_a1[0].copy()
    var tbls_a1 = base_a1[1].copy()
    tags_a1.append(0x47444546)  # GDEF
    tbls_a1.append(gdef.copy())
    tags_a1.append(0x47535542)  # GSUB
    tbls_a1.append(gsub_t5^)
    var f_a1 = TrueTypeFont()
    assert_true(f_a1.load_bytes(_r8_build_sfnt(tags_a1, tbls_a1)))

    var glyphs_marks = List[GsubGlyph](capacity=30001)
    glyphs_marks.append(GsubGlyph(1))
    for _ in range(30000):
        glyphs_marks.append(GsubGlyph(2))  # Mark glyph
    var t0_ns = perf_counter_ns()
    f_a1.apply_gsub(glyphs_marks, "DFLT")
    var dt_ms = Float64(perf_counter_ns() - t0_ns) / 1_000_000.0
    assert_equal(len(glyphs_marks), 30001)
    assert_true(dt_ms < 250.0, "GSUB IGNORE_MARKS mark-skipping DoS must exhaust budget quickly")

    # Positive control: GSUB Type 4 ligature with IGNORE_MARKS across 1 mark still ligatures [1, 2, 3] -> [4, 2]
    var gsub_lig = List[UInt8]()
    _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 0)
    _r7_put_u16_be(gsub_lig, 10); _r7_put_u16_be(gsub_lig, 34); _r7_put_u16_be(gsub_lig, 52)
    _r7_put_u16_be(gsub_lig, 1); _r7_put_u32_be(gsub_lig, 0x44464C54); _r7_put_u16_be(gsub_lig, 8)
    _r7_put_u16_be(gsub_lig, 4); _r7_put_u16_be(gsub_lig, 0)
    _r7_put_u16_be(gsub_lig, 0); _r7_put_u16_be(gsub_lig, 0xFFFF); _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 0)
    while len(gsub_lig) < 34:
        gsub_lig.append(UInt8(0))
    _r7_put_u16_be(gsub_lig, 1); _r7_put_u32_be(gsub_lig, 0x6C696761); _r7_put_u16_be(gsub_lig, 8)
    _r7_put_u16_be(gsub_lig, 0); _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 0)
    while len(gsub_lig) < 52:
        gsub_lig.append(UInt8(0))
    _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 4)  # LookupList -> Lookup at +56
    _r7_put_u16_be(gsub_lig, 4); _r7_put_u16_be(gsub_lig, 0x0008); _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 8)  # Type 4 at +64
    _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 8); _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 14)  # Subtable at +64
    _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 1)  # Coverage at +72: gid=1
    _r7_put_u16_be(gsub_lig, 1); _r7_put_u16_be(gsub_lig, 4)  # LigatureSet at +78 -> Ligature at +82
    _r7_put_u16_be(gsub_lig, 4); _r7_put_u16_be(gsub_lig, 2); _r7_put_u16_be(gsub_lig, 3)  # ligGlyph=4, compCount=2, comp=[3]
    tbls_a1[len(tbls_a1) - 1] = gsub_lig^
    var f_lig = TrueTypeFont()
    assert_true(f_lig.load_bytes(_r8_build_sfnt(tags_a1, tbls_a1)))
    var g_lig: List[GsubGlyph] = [GsubGlyph(1), GsubGlyph(2), GsubGlyph(3)]
    f_lig.apply_gsub(g_lig, "DFLT")
    assert_equal(len(g_lig), 2)
    assert_equal(g_lig[0].glyph_id, 4)
    assert_equal(g_lig[1].glyph_id, 2)

    # A2: rich_text overflow & SdfFont.measure / line_break zero-bake & Inf/NaN guards
    var f_tiny_bake = SdfFont()
    f_tiny_bake.valid = True
    f_tiny_bake.bake_px = Float32(1e-10)
    f_tiny_bake.line_height = Float32(12.0)
    f_tiny_bake.cap_height = Float32(10.0)
    var spans_ovf: List[RichSpan] = [RichSpan(SPAN_TEXT, "A", -1, 0, 100, 0)]
    var lay_ovf = layout_rich(spans_ovf, f_tiny_bake, f_tiny_bake, Float32(3e37), 0, Float32(0.0), 1, False, False)
    assert_true(isfinite(lay_ovf.width) and isfinite(lay_ovf.height))
    assert_equal(len(lay_ovf.runs), 1)
    assert_true(isfinite(lay_ovf.runs[0].x) and isfinite(lay_ovf.runs[0].width))
    var lay_sh_ovf = layout_rich_shaped(spans_ovf, f_tiny_bake, f_tiny_bake, Float32(3e37), 0, Float32(0.0), 1, False, False)
    assert_true(isfinite(lay_sh_ovf.width) and isfinite(lay_sh_ovf.height))

    # R13b: SPAN_BUTTON with large px + scale=400 & center/right alignment must stay finite
    var spans_btn_ovf: List[RichSpan] = [RichSpan(SPAN_BUTTON, "", -1, 0, 400, ACTION_JUMP)]
    for al in range(3):
        var lay_btn = layout_rich(spans_btn_ovf, f_tiny_bake, f_tiny_bake, Float32(3e38), 0, Float32(0.0), al, False, False)
        assert_true(isfinite(lay_btn.width) and isfinite(lay_btn.height))
        assert_equal(len(lay_btn.runs), 1)
        assert_true(isfinite(lay_btn.runs[0].x) and isfinite(lay_btn.runs[0].width) and isfinite(lay_btn.runs[0].px))
        var lay_btn_sh = layout_rich_shaped(spans_btn_ovf, f_tiny_bake, f_tiny_bake, Float32(3e38), 0, Float32(0.0), al, False, False)
        assert_true(isfinite(lay_btn_sh.width) and isfinite(lay_btn_sh.height))
        assert_equal(len(lay_btn_sh.runs), 1)
        assert_true(isfinite(lay_btn_sh.runs[0].x) and isfinite(lay_btn_sh.runs[0].width) and isfinite(lay_btn_sh.runs[0].px))
    var lay_btn_ok = layout_rich([RichSpan(SPAN_BUTTON, "", -1, 0, 100, ACTION_JUMP)], f_tiny_bake, f_tiny_bake, Float32(16.0), 0, Float32(0.0), 1, True, False)
    assert_equal(len(lay_btn_ok.runs), 1)
    assert_true(isfinite(lay_btn_ok.runs[0].width) and lay_btn_ok.runs[0].width > Float32(0.0))

    var f_zbake = SdfFont()
    f_zbake.valid = True
    f_zbake.bake_px = Float32(0.0)
    assert_equal(f_zbake.measure("abc", Float32(24.0)), Float32(0.0))
    assert_equal(run_width(f_zbake, [65, 66], 0, 2, Float32(24.0)), Float32(0.0))
    var adv_z = List[Float32]()
    var kern_z = List[Float32]()
    item_widths(f_zbake, [65, 66], Float32(24.0), adv_z, kern_z)
    assert_equal(len(adv_z), 2)
    assert_equal(adv_z[0], Float32(0.0))
    assert_equal(kern_z[1], Float32(0.0))
    f_zbake.bake_px = Float32(1e-10)
    assert_equal(f_zbake.measure("abc", Float32(3e37)), Float32(0.0))
    assert_equal(f_zbake.measure("abc", nan_f32), Float32(0.0))
    assert_equal(run_width(f_zbake, [65, 66], 0, 2, pinf_f32), Float32(0.0))

    # A3: perf_history NaN/Inf rejection & PerfTable.cell bounds
    assert_false(perf_parse_number("nan")[0])
    assert_false(perf_parse_number("inf")[0])
    assert_false(perf_parse_number("-inf")[0])
    assert_false(perf_parse_number("1e999")[0])
    var ok_pn = perf_parse_number("12.5")
    assert_true(ok_pn[0] and ok_pn[1] == Float64(12.5))

    var ptbl = PerfTable()
    ptbl.header = ["date", "commit", "package", "result", "m1"]
    ptbl.rows.append(["2026-11-20", "abc", "v5.0", "pass", "1.0"])
    assert_equal(ptbl.cell(-1, 0), "")
    assert_equal(ptbl.cell(99, 0), "")
    assert_equal(ptbl.cell(0, -1), "")
    assert_equal(ptbl.cell(0, 99), "")
    assert_equal(ptbl.cell(0, 4), "1.0")

    var pmetric = PerfMetric("m1", "lab_physics", "m1=", 0.1, 10.0)
    var f_nan = judge_from(pmetric, nan_f64, ptbl, List[AcceptLine](), len(ptbl.rows))
    assert_true(f_nan.worse and f_nan.over_budget and f_nan.needs_again())
    var f_inf = judge_from(pmetric, pinf_f64, ptbl, List[AcceptLine](), len(ptbl.rows))
    assert_true(f_inf.worse and f_inf.over_budget and f_inf.needs_again())
    var f_good = judge_from(pmetric, Float64(1.0), ptbl, List[AcceptLine](), len(ptbl.rows))
    assert_false(f_good.worse or f_good.over_budget or f_good.needs_again())
    assert_equal(signed_percent(nan_f64), "nan%")
    assert_equal(signed_percent(pinf_f64), "+inf%")
    assert_equal(signed_percent(Float64(0.123)), "+12.3%")

    # A4: ArchetypeSlots.read_blob rejects oversized cap / truncated rows without mutating existing state
    var slots_a4 = ArchetypeSlots(7, 16)
    var h_a4 = slots_a4.spawn()
    assert_true(h_a4.valid())
    var bw_trunc = BlobWriter()
    bw_trunc.integer(FLINT_ECS_VERSION)
    bw_trunc.integer(99)
    bw_trunc.integer(1_000_000)  # cap
    bw_trunc.integer(500)        # used > available bytes (0)
    bw_trunc.integer(0)
    bw_trunc.integer(500)
    var br_trunc = BlobReader(bw_trunc.data.copy())
    assert_false(slots_a4.read_blob(br_trunc))
    assert_equal(slots_a4.archetype, 7)
    assert_equal(slots_a4.capacity, 16)
    assert_equal(slots_a4.used, 1)
    assert_true(slots_a4.resolve(h_a4) >= 0)

    var bw_huge = BlobWriter()
    bw_huge.integer(FLINT_ECS_VERSION)
    bw_huge.integer(99)
    bw_huge.integer(2_000_000_000)
    bw_huge.integer(0)
    bw_huge.integer(0)
    bw_huge.integer(0)
    var br_huge = BlobReader(bw_huge.data.copy())
    assert_false(slots_a4.read_blob(br_huge))
    assert_equal(slots_a4.capacity, 16)

    var bw_ok = BlobWriter()
    slots_a4.write_blob(bw_ok)
    var slots_restored = ArchetypeSlots(1, 8)
    var br_ok = BlobReader(bw_ok.data.copy())
    assert_true(slots_restored.read_blob(br_ok))
    assert_equal(slots_restored.archetype, 7)
    assert_equal(slots_restored.capacity, 16)
    assert_equal(slots_restored.used, 1)
    assert_true(slots_restored.resolve(h_a4) >= 0)

    # A5: TruePeakLimiter & KWeightedLufsMeter NaN/Inf sanitization + glb_take non-finite duration rejection
    var limiter = TruePeakLimiter()
    var lim0 = limiter.process(nan_f32, Float32(0.25))
    assert_true(isfinite(lim0[0]) and isfinite(lim0[1]))
    var lim1 = limiter.process(pinf_f32, Float32(0.25))
    assert_true(isfinite(lim1[0]) and isfinite(lim1[1]))
    for _ in range(2, 128):
        var lim_s = limiter.process(Float32(0.25), Float32(0.25))
        assert_true(isfinite(lim_s[0]) and isfinite(lim_s[1]), "TruePeakLimiter must never emit NaN/Inf from delay line")

    var lufs = KWeightedLufsMeter(48000)
    lufs.process_frame(nan_f32, Float32(0.25))
    lufs.process_frame(pinf_f32, Float32(0.25))
    for _ in range(2000):
        lufs.process_frame(Float32(0.2), Float32(0.2))
    assert_true(isfinite(lufs.momentary_lufs()))
    assert_true(isfinite(lufs.integrated_lufs()))

    # Minimal valid GLB for append_take_glb testing
    var min_json = _str_to_bytes(
        '{"asset":{"version":"2.0"},"buffers":[{"byteLength":0}],"bufferViews":[],"accessors":[],"nodes":[{"name":"Hips"}]}'
    )
    while len(min_json) % 4 != 0:
        min_json.append(UInt8(0x20))
    var min_glb = List[UInt8]()
    _put_le32(min_glb, 0x46546C67)
    _put_le32(min_glb, 2)
    _put_le32(min_glb, 12 + 8 + len(min_json) + 8)
    _put_le32(min_glb, len(min_json))
    _put_le32(min_glb, 0x4E4F534A)
    for b in min_json:
        min_glb.append(b)
    _put_le32(min_glb, 0)
    _put_le32(min_glb, 0x004E4942)
    var in_glb_path = tmp_dir + "/r13_in.glb"
    var out_glb_bad = tmp_dir + "/r13_out_bad.glb"
    var out_glb_ok = tmp_dir + "/r13_out_ok.glb"
    assert_true(write_bytes(in_glb_path, min_glb))

    var bad_anim = NativeAnimation()
    bad_anim.duration = nan_f32
    var ch_a5 = NativeChannel()
    ch_a5.node = 0
    ch_a5.path = 0
    ch_a5.times = [Float32(0.0), Float32(0.5)]
    ch_a5.values = [Float32(0.0), 0.0, 0.0, Float32(1.0), 0.0, 0.0]
    bad_anim.channels.append(ch_a5.copy())
    var glb_nan_raised = False
    try:
        append_take_glb(in_glb_path, out_glb_bad, bad_anim)
    except:
        glb_nan_raised = True
    assert_true(glb_nan_raised, "append_take_glb must reject non-finite animation duration")

    var ok_anim = NativeAnimation()
    ok_anim.duration = Float32(0.5)
    ok_anim.channels.append(ch_a5^)
    _ = remove_file(out_glb_ok)
    append_take_glb(in_glb_path, out_glb_ok, ok_anim)
    var reloaded_take = parse_native_glb(read_bytes(out_glb_ok))
    assert_true(reloaded_take.loaded())
    assert_equal(len(reloaded_take.animations), 1)
    _ = remove_file(in_glb_path)
    _ = remove_file(out_glb_bad)
    _ = remove_file(out_glb_ok)

    # B4: CompoundShape.decompose_vhacd_lite OOB & negative triangle index guard + valid tet control
    var tet_verts: List[Vec3] = [
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    ]
    var oob_tris: List[Int32] = [Int32(0), Int32(1), Int32(99), Int32(-1), Int32(0), Int32(2)]
    var cs_oob = CompoundShape.decompose_vhacd_lite(tet_verts, oob_tris)
    assert_equal(len(cs_oob.children), 0)

    var ok_tris: List[Int32] = [
        Int32(0), Int32(2), Int32(1),
        Int32(0), Int32(1), Int32(3),
        Int32(1), Int32(2), Int32(3),
        Int32(2), Int32(0), Int32(3),
    ]
    var cs_ok = CompoundShape.decompose_vhacd_lite(tet_verts, ok_tris)
    assert_true(len(cs_ok.children) >= 1)

    # R13b: CompoundShape.decompose_vhacd_lite vertex finiteness checked before n_tris == 0 quickhull fallback
    var nan_tet_verts: List[Vec3] = [
        Vec3(nan_f32, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
    ]
    var cs_nan_empty_tris = CompoundShape.decompose_vhacd_lite(nan_tet_verts, List[Int32]())
    assert_equal(len(cs_nan_empty_tris.children), 0)
    var cs_ok_empty_tris = CompoundShape.decompose_vhacd_lite(tet_verts, List[Int32]())
    assert_equal(len(cs_ok_empty_tris.children), 1)


def test_r14_partition_ab_fixes(tmp_dir: String) raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)
    var ninf_f32 = f32_from_bits(0xFF800000)
    var nan_f64 = Float64(nan_f32)
    var pinf_f64 = Float64(pinf_f32)
    var ninf_f64 = Float64(ninf_f32)

    # A1: cvars.show_value LLVM fptosi UB guard, parse_value NaN/Inf rejection & DevConsole.submit
    assert_equal(show_value(Float32(42.0)), "42")
    assert_equal(show_value(Float32(3.5)), "3.5")
    assert_equal(show_value(nan_f32), "nan")
    assert_equal(show_value(pinf_f32), "inf")
    assert_equal(show_value(ninf_f32), "-inf")
    var sv_big = show_value(Float32(1.0e30))
    assert_true(sv_big.byte_length() > 0)
    assert_true(parse_value("nan") is None)
    assert_true(parse_value("inf") is None)
    assert_true(parse_value("-inf") is None)
    assert_true(parse_value("1e999") is None)
    var pv_ok = parse_value("80.0")
    assert_true(pv_ok is not None and pv_ok.value() == Float32(80.0))

    var cv = ConsoleVars()
    var dc = DevConsole()
    var range_before = cv.get(CVAR_DEBUG_RANGE)
    dc.line = "debug.range nan"
    dc.submit(cv)
    assert_equal(cv.get(CVAR_DEBUG_RANGE), range_before)
    dc.line = "debug.range inf"
    dc.submit(cv)
    assert_equal(cv.get(CVAR_DEBUG_RANGE), range_before)
    dc.line = "debug.range 120"
    dc.submit(cv)
    assert_equal(cv.get(CVAR_DEBUG_RANGE), Float32(120.0))

    # A2: rich_text layout_rich & layout_rich_shaped finite-sum Inf overflow (horizontal & multi-line)
    var f_big = SdfFont()
    f_big.valid = True
    f_big.bake_px = Float32(1.0)
    f_big.line_height = Float32(1.0)
    f_big.cap_height = Float32(0.8)
    var g_a = Glyph()
    g_a.advance = Float32(1.0)
    f_big.glyphs.append(g_a)
    f_big.lookup = List[Int](length=128, fill=-1)
    f_big.lookup[65] = 0
    f_big.lookup[10] = 0
    var s200 = String("A") * 200
    var lay200 = layout_rich([RichSpan(SPAN_TEXT, s200, -1, 0, 100, 0)], f_big, f_big, Float32(3.0e36), 0, Float32(0.0), 0, False, False)
    assert_true(isfinite(lay200.width) and lay200.width > Float32(0.0) and isfinite(lay200.height))
    assert_equal(len(lay200.runs), 1)
    assert_true(isfinite(lay200.runs[0].x) and isfinite(lay200.runs[0].width) and isfinite(lay200.runs[0].baseline))
    var lay200_sh = layout_rich_shaped([RichSpan(SPAN_TEXT, s200, -1, 0, 100, 0)], f_big, f_big, Float32(3.0e36), 0, Float32(0.0), 0, False, False)
    assert_true(isfinite(lay200_sh.width) and lay200_sh.width > Float32(0.0) and isfinite(lay200_sh.height))
    for r in lay200_sh.runs:
        assert_true(isfinite(r.x) and isfinite(r.width) and isfinite(r.baseline))

    var f_tall = f_big.copy()
    f_tall.line_height = Float32(1.0e37)
    var ml_text = String("A\n") * 50 + "A"
    var lay_ml = layout_rich([RichSpan(SPAN_TEXT, ml_text, -1, 0, 100, 0)], f_tall, f_tall, Float32(1.0), 0, Float32(0.0), 0, False, False)
    assert_true(isfinite(lay_ml.height) and lay_ml.height > Float32(0.0))
    for r in lay_ml.runs:
        assert_true(isfinite(r.baseline))
    var lay_ml_sh = layout_rich_shaped([RichSpan(SPAN_TEXT, ml_text, -1, 0, 100, 0)], f_tall, f_tall, Float32(1.0), 0, Float32(0.0), 0, False, False)
    assert_true(isfinite(lay_ml_sh.height) and lay_ml_sh.height > Float32(0.0))
    for r in lay_ml_sh.runs:
        assert_true(isfinite(r.baseline))

    var lay_norm = layout_rich([RichSpan(SPAN_TEXT, "AA\nAA", -1, 0, 100, 0)], f_big, f_big, Float32(16.0), 0, Float32(0.0), 0, False, False)
    assert_equal(lay_norm.lines, 2)
    assert_true(lay_norm.width > Float32(0.0) and lay_norm.height > Float32(0.0))

    # A3: DockSpace._fmt_ratio NaN, load_json validation (header_height, ratio, active_tab), solve_layout NaN
    assert_equal(DockSpace._fmt_ratio(nan_f32), "0.500000")
    assert_equal(DockSpace._fmt_ratio(pinf_f32), "0.500000")
    assert_equal(DockSpace._fmt_ratio(Float32(2.0)), "1.0")
    assert_equal(DockSpace._fmt_ratio(Float32(0.25)), "0.250000")
    var ds_bad = DockSpace()
    assert_true(
        ds_bad.load_json(
            '{"root":0,"header_height":1e999,"nodes":['
            '{"id":0,"kind":0,"ratio":1e999,"child_a":1,"child_b":2,"active_tab":0,"tabs":[]},'
            '{"id":1,"kind":2,"ratio":0.5,"child_a":-1,"child_b":-1,"active_tab":999,"tabs":["A"]},'
            '{"id":2,"kind":2,"ratio":0.5,"child_a":-1,"child_b":-1,"active_tab":0,"tabs":["B"]}'
            ']}'
        )
    )
    assert_true(isfinite(ds_bad.header_height) and ds_bad.header_height == Float32(24.0))
    assert_true(isfinite(ds_bad.nodes[0].ratio) and ds_bad.nodes[0].ratio >= ds_bad.min_ratio and ds_bad.nodes[0].ratio <= Float32(1.0) - ds_bad.min_ratio)
    assert_equal(ds_bad.nodes[1].active_tab, 0)
    ds_bad.nodes[0].ratio = nan_f32
    ds_bad.solve_layout(RectF(0.0, 0.0, 1280.0, 720.0))
    for lr in ds_bad.leaf_rects():
        assert_true(isfinite(lr.x) and isfinite(lr.y) and isfinite(lr.w) and isfinite(lr.h))
    ds_bad.solve_layout(RectF(Float32(3.0e38), Float32(0.0), Float32(3.0e38), Float32(100.0)))
    for lr in ds_bad.leaf_rects():
        assert_true(isfinite(lr.x) and isfinite(lr.y) and isfinite(lr.w) and isfinite(lr.h))
    ds_bad.nodes[0].kind = DOCK_SPLIT_V
    ds_bad.solve_layout(RectF(Float32(0.0), Float32(3.0e38), Float32(100.0), Float32(3.0e38)))
    for lr in ds_bad.leaf_rects():
        assert_true(isfinite(lr.x) and isfinite(lr.y) and isfinite(lr.w) and isfinite(lr.h))

    var ds_ok = DockSpace.create_studio_5panel()
    var ds_json = ds_ok.save_json()
    var ds_rt = DockSpace()
    assert_true(ds_rt.load_json(ds_json))
    assert_equal(len(ds_rt.nodes), len(ds_ok.nodes))

    # A4: profiler.report_ms negative, NaN/Inf & huge magnitudes
    assert_equal(report_ms(Float32(-1.5)), "-1.500")
    assert_equal(report_ms(Float32(-0.0)), "0.000")
    assert_equal(report_ms(Float32(0.005)), "0.005")
    assert_equal(report_ms(Float32(1.234)), "1.234")
    assert_equal(report_ms(nan_f32), "nan")
    assert_equal(report_ms(pinf_f32), "inf")
    assert_equal(report_ms(ninf_f32), "-inf")
    var rm_big = report_ms(Float32(1.0e30))
    assert_true(rm_big.byte_length() > 0 and rm_big != "9223372036854775.807")

    # A-Notes: AxisCurve, TruePeakLimiter.__init__, JsonDocument.integer
    var curve = AxisCurve()
    assert_equal(curve.eval_1d(nan_f32), Float32(0.0))
    assert_equal(curve.eval_1d(pinf_f32), Float32(0.0))
    var e2_bad = curve.eval_2d(nan_f32, pinf_f32)
    assert_equal(e2_bad[0], Float32(0.0))
    assert_equal(e2_bad[1], Float32(0.0))
    var e2_huge = curve.eval_2d(Float32(1.0e30), Float32(1.0e30))
    assert_true(isfinite(e2_huge[0]) and isfinite(e2_huge[1]))
    assert_true(curve.eval_1d(Float32(0.5)) > Float32(0.0))

    var lim_nan = TruePeakLimiter(nan_f32, nan_f32, Float32(0.0))
    assert_true(isfinite(lim_nan.ceiling) and lim_nan.ceiling > Float32(0.0) and lim_nan.ceiling <= Float32(1.0))
    assert_true(isfinite(lim_nan.release_coeff))
    for _ in range(100):
        var out_l = lim_nan.process(Float32(2.0), Float32(-2.0))
        assert_true(isfinite(out_l[0]) and isfinite(out_l[1]))

    var jbytes = List[UInt8]()
    for b in String('{"big":1e999,"neg":-1e999,"ok":42}').as_bytes():
        jbytes.append(b)
    var jdoc = parse_json(jbytes)
    assert_true(jdoc.valid)
    assert_equal(jdoc.integer(jdoc.object_get(jdoc.root, "big"), -1), -1)
    assert_equal(jdoc.integer(jdoc.object_get(jdoc.root, "neg"), -7), -7)
    assert_equal(jdoc.integer(jdoc.object_get(jdoc.root, "ok"), -1), 42)

    # B1: HeightfieldCollider.build overflow, NaN heights, extreme span & valid control
    var hf_small: List[Float32] = [Float32(0.0), 1.0, 2.0, 3.0]
    var hf_ovf = HeightfieldCollider.build(hf_small, 1 << 32, 1 << 32)
    assert_equal(hf_ovf.nx, 0)
    assert_equal(len(hf_ovf.heights_u16), 0)

    var hf_nan_in: List[Float32] = [nan_f32, 1.0, 2.0, 3.0]
    var hf_nan = HeightfieldCollider.build(hf_nan_in, 2, 2)
    assert_equal(hf_nan.nx, 0)
    assert_equal(len(hf_nan.heights_u16), 0)

    var hf_big_in: List[Float32] = [Float32(-3.0e38), Float32(3.0e38), Float32(0.0), Float32(1.0)]
    var hf_big = HeightfieldCollider.build(hf_big_in, 2, 2)
    assert_equal(hf_big.nx, 2)
    assert_true(isfinite(hf_big.y_scale))
    assert_true(isfinite(hf_big.dequant_y(0)))
    assert_true(isfinite(hf_big.dequant_y(65535)))
    assert_true(isfinite(hf_big.height_at_grid(0, 0)) and isfinite(hf_big.height_at_grid(1, 0)))
    var big_s0 = hf_big.sample_height_and_normal(0.5, 0.5)
    assert_true(isfinite(big_s0[0]) and isfinite(big_s0[1].x) and isfinite(big_s0[1].y) and isfinite(big_s0[1].z))
    var big_s1 = hf_big.sample_height_and_normal(0.8, 0.8)
    assert_true(isfinite(big_s1[0]) and isfinite(big_s1[1].x) and isfinite(big_s1[1].y) and isfinite(big_s1[1].z))
    var big_tris = hf_big.cell_triangles(0, 0)
    assert_true(
        isfinite(big_tris[0].normal.x)
        and isfinite(big_tris[0].normal.y)
        and isfinite(big_tris[0].normal.z)
        and isfinite(big_tris[1].normal.x)
        and isfinite(big_tris[1].normal.y)
        and isfinite(big_tris[1].normal.z)
    )

    var hf_ok = HeightfieldCollider.build(hf_small, 2, 2)
    assert_equal(hf_ok.nx, 2)
    assert_equal(len(hf_ok.heights_u16), 4)
    assert_true(isfinite(hf_ok.sample_height_and_normal(0.5, 0.5)[0]))

    # B2: cycles.synth render_to_wav & render_loop_to_wav overflow / OOM / NaN guards + valid render
    var wav_path = tmp_dir + "/r14_synth.wav"
    assert_equal(render_to_wav("d1 $ s \"bd\"", Float64(1.0e18), wav_path, 48000), 0)
    assert_equal(render_to_wav("d1 $ s \"bd\"", Float64(1.0e11), wav_path, 48000), 0)
    assert_equal(render_to_wav("d1 $ s \"bd\"", nan_f64, wav_path, 48000), 0)
    assert_equal(render_to_wav("d1 $ s \"bd\"", pinf_f64, wav_path, 48000), 0)
    assert_equal(render_loop_to_wav("d1 $ s \"bd\"", 1_000_000_000, wav_path, 48000), 0)
    var ev_ok = render_to_wav("d1 $ s \"bd\"", 0.05, wav_path, 48000)
    assert_true(ev_ok > 0)
    var loop_ok = render_loop_to_wav("d1 $ s \"bd\"", 1, wav_path, 48000)
    assert_true(loop_ok > 0)
    _ = remove_file(wav_path)

    # B3: procedural.foliage.build_manicured_hedge_and_planter huge length / NaN clamp
    var hedge_huge = build_manicured_hedge_and_planter(Float32(1.0e12))
    assert_true(hedge_huge[1].vertex_count() > 0 and hedge_huge[1].vertex_count() < 50000)
    var hedge_nan = build_manicured_hedge_and_planter(nan_f32)
    assert_true(hedge_nan[1].vertex_count() > 0)

    # B4 & B-Notes: ORCA NaN goal, compare_pixels neg/overflow, _fmt_f64_3 NaN/Inf/huge, Vehicle substep dt=1e8
    var crowd = CrowdSimulator()
    var a0 = crowd.add_agent(Vec3(0.0, 0.0, 0.0), 0.35, 3.0, Vec3(nan_f32, 0.0, 1.0))
    var a1 = crowd.add_agent(Vec3(1.0, 0.0, 0.0), 0.35, 3.0, Vec3(2.0, 0.0, 0.0))
    crowd.add_wall_segment(Vec3(-2.0, 0.0, 2.0), Vec3(2.0, 0.0, 2.0))
    crowd.step_orca_simd8(nan_f32)
    crowd.step_orca_simd8(Float32(0.016))
    var p0 = crowd.agent_position(a0)
    var p1 = crowd.agent_position(a1)
    assert_true(isfinite(p0.x) and isfinite(p0.z) and isfinite(crowd.vx[a0]))
    assert_true(isfinite(p1.x) and isfinite(p1.z))

    var px_2x2: List[UInt32] = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]
    assert_false(compare_pixels(px_2x2, px_2x2, -2, -2, 0).same_size)
    assert_false(compare_pixels(px_2x2, px_2x2, 1 << 32, 1 << 32, 0).same_size)
    var diff_ok = compare_pixels(px_2x2, px_2x2, 2, 2, 0)
    assert_true(diff_ok.same_size and diff_ok.differing == 0)

    assert_equal(_fmt_f64_3(nan_f64), "nan")
    assert_equal(_fmt_f64_3(pinf_f64), "inf")
    assert_equal(_fmt_f64_3(ninf_f64), "-inf")
    assert_equal(_fmt_f64_3(-1.5), "-1.500")
    var fmt_huge = _fmt_f64_3(Float64(1.0e20))
    assert_true(fmt_huge.byte_length() > 0 and fmt_huge != "9223372036854775.807")

    var vc = VehicleChassis()
    vc.setup_four_wheel_car()
    var vt0 = perf_counter_ns()
    vc.substep_suspension(nan_f32)
    vc.substep_suspension(Float32(1.0e8))
    var vt_ms = Float64(perf_counter_ns() - vt0) / 1_000_000.0
    assert_true(vt_ms < 250.0, "VehicleChassis.substep_suspension must cap extreme dt substeps")
    assert_true(isfinite(vc.pos.x) and isfinite(vc.pos.y) and isfinite(vc.pos.z))


def test_r15_partition_ab_fixes(tmp_dir: String) raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)
    var ninf_f32 = f32_from_bits(0xFF800000)
    var nan_f64 = Float64(nan_f32)
    var pinf_f64 = Float64(pinf_f32)

    # 1. profile_compare.Capture: negative frame id rejection & gpu_ms args bounds
    var bad_trace = tmp_dir + "/r15_bad_trace.json"
    _write_text(bad_trace, '[{"ph":"X","name":"frame","ts":0,"dur":1000,"args":{"frame":-1}},{"ph":"C","name":"gpu_ms"}]')
    var cap_bad_raised = False
    try:
        _ = Capture(bad_trace)
    except:
        cap_bad_raised = True
    assert_true(cap_bad_raised, "Capture must reject negative frame id in Pass 1")
    _ = remove_file(bad_trace)

    var ok_trace = tmp_dir + "/r15_ok_trace.json"
    _write_text(
        ok_trace,
        '[{"ph":"X","name":"frame","ts":0,"dur":2000,"args":{"frame":0}},'
        '{"ph":"C","name":"gpu_ms"},'
        '{"ph":"C","name":"gpu_ms","args":[1]},'
        '{"ph":"C","name":"gpu_ms","args":{"frame":0,"pass_a":1.25}}]',
    )
    var cap_ok = Capture(ok_trace)
    assert_equal(cap_ok.at("pass_a", 0), Float64(1.25))
    _ = remove_file(ok_trace)

    # 2. format_float2, _fmt_f32_2dp & glb_native float-to-UInt8 color clamping
    assert_equal(format_float2(nan_f32), "nan")
    assert_equal(format_float2(pinf_f32), "inf")
    assert_equal(format_float2(ninf_f32), "-inf")
    assert_equal(format_float2(Float32(12.34)), "12.34")
    assert_equal(format_float2(Float32(-0.5)), "-0.50")
    var ff_big = format_float2(Float32(1.0e30))
    assert_true(ff_big.byte_length() > 0 and ff_big != "92233720368547758.07")
    var ff_nbig = format_float2(Float32(-1.0e30))
    assert_true(ff_nbig.startswith("-") and ff_nbig != "-92233720368547758.07")

    assert_equal(_fmt_f32_2dp(nan_f32), "nan")
    assert_equal(_fmt_f32_2dp(pinf_f32), "inf")
    assert_equal(_fmt_f32_2dp(ninf_f32), "-inf")
    assert_equal(_fmt_f32_2dp(Float32(-0.5)), "-0.50")
    var sf_big = _fmt_f32_2dp(Float32(1.0e30))
    assert_true(sf_big.byte_length() > 0 and sf_big != "92233720368547758.07")
    var sf_nbig = _fmt_f32_2dp(Float32(-1.0e30))
    assert_true(sf_nbig.startswith("-") and sf_nbig != "-92233720368547758.07")

    var glb_col_json = _str_to_bytes(
        '{"asset":{"version":"2.0"},"buffers":[{"byteLength":84}],'
        '"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":48}],'
        '"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},'
        '{"bufferView":1,"componentType":5126,"count":3,"type":"VEC4"}],'
        '"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1e400,-1e400,0.5,1.0]},"alphaMode":"MASK","alphaCutoff":0.25}],'
        '"meshes":[{"primitives":[{"attributes":{"POSITION":0,"COLOR_0":1},"material":0}]}],'
        '"nodes":[{"mesh":0}]}'
    )
    while len(glb_col_json) % 4 != 0:
        glb_col_json.append(UInt8(0x20))
    var glb_col = List[UInt8]()
    _put_le32(glb_col, 0x46546C67)
    _put_le32(glb_col, 2)
    _put_le32(glb_col, 12 + 8 + len(glb_col_json) + 8 + 84)
    _put_le32(glb_col, len(glb_col_json))
    _put_le32(glb_col, 0x4E4F534A)
    for b in glb_col_json:
        glb_col.append(b)
    _put_le32(glb_col, 84)
    _put_le32(glb_col, 0x004E4942)
    for _ in range(9):
        _put_le32(glb_col, 0)  # 3 zero Vec3 positions
    for _ in range(3):
        _put_le32(glb_col, 0x7FC00000)  # NaN
        _put_le32(glb_col, 0x7F800000)  # +Inf
        _put_le32(glb_col, 0xFF800000)  # -Inf
        _put_le32(glb_col, 0x3F000000)  # 0.5
    var parsed_col = parse_native_glb(glb_col)
    assert_true(parsed_col.loaded())
    assert_equal(len(parsed_col.meshes), 1)
    assert_equal(parsed_col.meshes[0].colors[0], UInt8(0))
    assert_equal(parsed_col.meshes[0].colors[1], UInt8(255))
    assert_equal(parsed_col.meshes[0].colors[2], UInt8(0))
    assert_equal(parsed_col.meshes[0].colors[3], UInt8(127))
    assert_equal(parsed_col.meshes[0].base_color, UInt32(0xFFFF00BC))
    assert_equal(parsed_col.meshes[0].metallic, Float32(1))
    assert_equal(parsed_col.meshes[0].roughness, Float32(1))
    assert_equal(parsed_col.meshes[0].alpha_cutoff, Float32(0.25))

    # 3. mcp_v2 non-finite argument rejection & valid JSON emission
    assert_equal(_fmt_vec3_json(Vec3(nan_f32, pinf_f32, ninf_f32)), "[0.0,0.0,0.0]")
    var mcp = McpServerV2()
    var studio_mcp = FlintStudioState()
    var assets_mcp = AssetServer()
    var pool_mcp = JobPool()
    var resp_bad_spawn = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ecs.spawn_entity",'
        '"arguments":{"name":"Bad","position":[1e400,-1e400,1e30],"mass":1e400}}}',
        studio_mcp, assets_mcp, pool_mcp,
    )
    assert_true(parse_json(_str_to_bytes(resp_bad_spawn)).valid)
    assert_true(resp_bad_spawn.find('"isError":true') >= 0)

    var resp_ok_spawn = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ecs.spawn_entity",'
        '"arguments":{"name":"GoodBox","position":[1.0,2.0,3.0],"mass":5.0}}}',
        studio_mcp, assets_mcp, pool_mcp,
    )
    assert_true(resp_ok_spawn.find('"isError":false') >= 0)
    # Even if an entity holds non-finite fields directly, ecs.query_entities emits valid JSON
    if len(studio_mcp.entities) > 0:
        studio_mcp.entities[0].translation = Vec3(pinf_f32, ninf_f32, nan_f32)
        studio_mcp.entities[0].mass = pinf_f32
    var resp_query = mcp.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ecs.query_entities","arguments":{}}}',
        studio_mcp, assets_mcp, pool_mcp,
    )
    assert_true(parse_json(_str_to_bytes(resp_query)).valid, "ecs.query_entities must emit valid JSON")
    assert_true(resp_query.find("inf") < 0 and resp_query.find("nan") < 0)

    # 4. platform/audio.clip_sample negative channel, non-finite position & empty clip
    var wclip = WavClip()
    wclip.valid = True
    wclip.channels = 2
    wclip.sample_rate = 48000
    wclip.samples = [Int16(16384), Int16(-16384), Int16(8192), Int16(-8192)]
    assert_true(isfinite(clip_sample(wclip, 0.0, -1, False)))
    assert_true(abs(clip_sample(wclip, 0.0, -1, False) - Float32(0.5)) < Float32(1e-4))
    assert_true(abs(clip_sample(wclip, 0.0, 99, False) - Float32(-0.5)) < Float32(1e-4))
    assert_equal(clip_sample(wclip, nan_f64, 0, False), Float32(0.0))
    assert_equal(clip_sample(wclip, pinf_f64, 0, True), Float32(0.0))
    assert_true(isfinite(clip_sample(wclip, Float64(1.0e30), 0, True)))
    assert_true(isfinite(clip_sample(wclip, Float64(-1.0e30), 0, False)))
    assert_true(abs(clip_sample(wclip, 0.5, 1, False) - Float32(-0.375)) < Float32(1e-4))
    var empty_clip = WavClip()
    assert_equal(clip_sample(empty_clip, 0.0, -1, False), Float32(0.0))

    # 5. PhysicsWorld._step_vehicles_pre_solve huge dt / NaN dt substep cap
    var pw = PhysicsWorld(gravity=-9.81)
    _ = pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), mass=0.0)
    var car = VehicleChassis()
    car.setup_four_wheel_car()
    car.pos = Vec3(0.0, 1.0, 0.0)
    var vslot = pw.add_vehicle(car^)
    var pwt0 = perf_counter_ns()
    pw.step(nan_f32)
    pw.step(Float32(1.0e8))
    var pwt_ms = Float64(perf_counter_ns() - pwt0) / 1_000_000.0
    assert_true(pwt_ms < 500.0, "PhysicsWorld.step with VehicleChassis must cap dt=1e8 substeps")
    pw.step(Float32(1.0 / 60.0))
    var vst = pw.vehicle_state(vslot)
    assert_true(isfinite(vst.pos.x) and isfinite(vst.pos.y) and isfinite(vst.pos.z))

    # 6. Animator.advance & update_layers O(1) wrap on extreme dt / NaN
    var anim_model = NativeModel()
    var rnode = NativeNode()
    rnode.name = "Root"
    anim_model.nodes.append(rnode^)
    var nclip = NativeAnimation()
    nclip.name = "walk"
    nclip.duration = Float32(1.0)
    var nch = NativeChannel()
    nch.node = 0
    nch.path = 0
    nch.times = [Float32(0.0), Float32(1.0)]
    nch.values = [Float32(0.0), 0.0, 0.0, Float32(1.0), 0.0, 0.0]
    nclip.channels.append(nch^)
    anim_model.animations.append(nclip^)

    var ag = AnimGraphDef()
    var st0 = StateDef()
    st0.name = "walk"
    st0.clip = 0
    ag.states.append(st0^)
    var lay0 = LayerDef()
    lay0.name = "upper"
    lay0.clip = 0
    lay0.loop = True
    lay0.weight = Float32(1.0)
    lay0.fade = Float32(0.01)
    ag.layers.append(lay0^)

    var cinfo = ClipInfo("walk", Float32(1.0), True, Float32(1.0))
    var anim = Animator()
    anim.configure(ag, [cinfo^])
    var at0 = perf_counter_ns()
    anim.update(anim_model, 1, Float32(1.0e8))
    anim.update(anim_model, 1, Float32(-1.0e8))
    anim.update(anim_model, 1, nan_f32)
    anim.update(anim_model, 1, pinf_f32)
    var at_ms = Float64(perf_counter_ns() - at0) / 1_000_000.0
    assert_true(at_ms < 250.0, "Animator.update must wrap in O(1) for extreme dt")
    assert_true(isfinite(anim.clip_time) and anim.clip_time >= Float32(0.0) and anim.clip_time < Float32(1.0))
    assert_true(isfinite(anim.layers[0].time) and anim.layers[0].time >= Float32(0.0) and anim.layers[0].time < Float32(1.0))
    anim.reset()
    anim.update(anim_model, 1, Float32(0.25))
    anim.update(anim_model, 1, Float32(0.25))
    assert_true(abs(anim.clip_time - Float32(0.25)) < Float32(0.30))

    # 7. ScalableRenderer._render_tier0_simd PRIM_LINE viewport clipping on 1e12 / NaN
    var ren = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, 128, 64)
    var rw = RenderWorld()
    var ui = UiDrawList()
    ui.add_line(0.0, 0.0, Float32(1.0e12), 0.0, 1.0, 0xFFFFFFFF)
    ui.add_line(nan_f32, 0.0, 10.0, 10.0, 1.0, 0xFFFFFFFF)
    ui.add_line(Float32(-1.0e12), 32.0, Float32(1.0e12), 32.0, 1.0, 0xFF00FF00)
    ui.add_line(0.0, 10.0, 20.0, 10.0, 1.0, 0xFFFF0000)
    var rt0 = perf_counter_ns()
    assert_true(ren.render_world(rw, ui))
    var rt_ms = Float64(perf_counter_ns() - rt0) / 1_000_000.0
    assert_true(rt_ms < 250.0, "ScalableRenderer PRIM_LINE must clip extreme coords to viewport")
    assert_equal(ren.cpu_canvas.pixels[32 * ren.width + 64], UInt32(0xFF00FF00))
    assert_equal(ren.cpu_canvas.pixels[10 * ren.width + 10], UInt32(0xFFFF0000))

    # 8. MotionDatabase.bake_clip_features sample_fps cap & finiteness guard
    var mdb = MotionDatabase()
    mdb.bake_clip_features(anim_model, 0, 0, 0, 0, Float32(1.0e8))
    assert_true(len(mdb.frames) > 0 and len(mdb.frames) <= 16384)
    var n_before_nan = len(mdb.frames)
    mdb.bake_clip_features(anim_model, 0, 0, 0, 0, nan_f32)
    assert_equal(len(mdb.frames), n_before_nan)
    var mdb_ok = MotionDatabase()
    mdb_ok.bake_clip_features(anim_model, 0, 0, 0, 0, Float32(30.0))
    assert_equal(len(mdb_ok.frames), 30)

    # 9. bake_vat_crowd_clip_rgba16f cap & sample_vat_vertex_cpu finiteness
    var vat_model = anim_model.copy()
    var vmesh = NativeMesh()
    vmesh.positions = [Float32(0.0), 0.0, 0.0, Float32(1.0), 0.0, 0.0, Float32(0.0), 1.0, 0.0]
    vmesh.normals = [Float32(0.0), 0.0, 1.0, Float32(0.0), 0.0, 1.0, Float32(0.0), 0.0, 1.0]
    vmesh.indices = [0, 1, 2]
    vat_model.meshes.append(vmesh^)
    var vat_huge = bake_vat_crowd_clip_rgba16f(vat_model, 0, 0, Float32(1.0e9))
    assert_true(vat_huge.desc.frame_count > 0 and vat_huge.desc.frame_count <= 4096)
    var vat_nan = bake_vat_crowd_clip_rgba16f(vat_model, 0, 0, nan_f32)
    assert_equal(vat_nan.desc.vertex_count, 0)
    var vat_ok = bake_vat_crowd_clip_rgba16f(vat_model, 0, 0, Float32(30.0))
    var sv_nan = sample_vat_vertex_cpu(vat_ok, 0, nan_f32, pinf_f32)
    assert_true(isfinite(sv_nan[0].x) and isfinite(sv_nan[1].y) and isfinite(sv_nan[2].x))
    var sv_huge = sample_vat_vertex_cpu(vat_ok, 0, Float32(1.0e30), Float32(-1.0e30))
    assert_true(isfinite(sv_huge[0].x) and isfinite(sv_huge[1].y) and isfinite(sv_huge[2].x))
    var sv_ok = sample_vat_vertex_cpu(vat_ok, 0, Float32(0.25), Float32(0.2))
    assert_true(isfinite(sv_ok[0].x) and isfinite(sv_ok[1].z))

    # 10. MeshTriangle Float64 normal/closest_point/ray_intersect & _quant_vert
    var big_tri = MeshTriangle(Vec3(0.0, 0.0, 0.0), Vec3(1.0e20, 0.0, 0.0), Vec3(0.0, 0.0, 1.0e20))
    assert_true(isfinite(big_tri.normal.x) and isfinite(big_tri.normal.y) and isfinite(big_tri.normal.z))
    assert_true(abs(big_tri.normal.y) > Float32(0.99))
    var big_bvh = Bvh8MeshCollider.build(
        [Vec3(0.0, 0.0, 0.0), Vec3(1.0e20, 0.0, 0.0), Vec3(0.0, 0.0, 1.0e20)],
        [Int32(0), Int32(1), Int32(2)],
    )
    var big_ray = big_bvh.raycast(Vec3(100.0, 10.0, 100.0), Vec3(0.0, -1.0, 0.0), Float32(100.0))
    assert_true(big_ray.hit and abs(big_ray.t - Float32(10.0)) < Float32(1.0e-3))
    var big_sph = big_bvh.collide_sphere(Vec3(100.0, 0.5, 100.0), Float32(1.0))
    assert_true(big_sph.has_collision)

    # 11. HeightfieldCollider.bounds overflow & raycast_dda / _query_aabb_triangles extreme coords
    var hf_3x3 = HeightfieldCollider.build(
        [Float32(0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        3, 3, Vec3(), Float32(2.0e38),
    )
    var hf_b = hf_3x3.bounds()
    assert_true(isfinite(hf_b[1].x) and isfinite(hf_b[1].z))
    var hf_unit = HeightfieldCollider.build([Float32(0.0), 0.0, 0.0, 0.0], 2, 2)
    var hf_rh_ext = hf_unit.raycast_dda(Vec3(1.0e20, 10.0, 1.0e20), Vec3(0.0, -1.0, 0.0))
    assert_false(hf_rh_ext.hit)
    var hf_rh_nan = hf_unit.raycast_dda(Vec3(nan_f32, 10.0, 0.0), Vec3(0.0, -1.0, 0.0))
    assert_false(hf_rh_nan.hit)
    var hf_sw_ext = hf_unit.sweep_sphere(Vec3(1.0e20, 1.0, 1.0e20), Vec3(-1.0e20, 0.0, -1.0e20), Float32(1.0), Float32(1.0))
    assert_true(isfinite(hf_sw_ext.toi))
    var hf_rh_ok = hf_unit.raycast_dda(Vec3(0.5, 5.0, 0.5), Vec3(0.0, -1.0, 0.0))
    assert_true(hf_rh_ok.hit and abs(hf_rh_ok.t - Float32(5.0)) < Float32(1.0e-3))

    # 12. anim/sync.wrap_time extreme/non-finite inputs
    var wt_huge = wrap_time(Float32(1.0e20), Float32(1.0))
    assert_true(isfinite(wt_huge) and wt_huge >= Float32(0.0) and wt_huge < Float32(1.0))
    var wt_nhuge = wrap_time(Float32(-1.0e20), Float32(1.0))
    assert_true(isfinite(wt_nhuge) and wt_nhuge >= Float32(0.0) and wt_nhuge < Float32(1.0))
    var wt_eps = wrap_time(Float32(-1.0e-12), Float32(1.0))
    assert_true(wt_eps >= Float32(0.0) and wt_eps < Float32(1.0))
    assert_equal(wrap_time(nan_f32, Float32(1.0)), Float32(0.0))
    assert_equal(wrap_time(pinf_f32, Float32(1.0)), Float32(0.0))
    assert_equal(wrap_time(Float32(0.5), Float32(0.0)), Float32(0.0))
    assert_equal(wrap_time(Float32(1.25), Float32(1.0)), Float32(0.25))

    # 13. compute_vsm_cascade_page_origins & eval_value_noise_periodic_cpu fptosi/NaN guards
    var vsm_ext = compute_vsm_cascade_page_origins(Vec3(0.0, -1.0, 0.0), Vec3(1.0e20, 0.0, 0.0))
    assert_equal(len(vsm_ext), 4)
    assert_true(vsm_ext[0][0] >= -1_000_000_000_000_000 and vsm_ext[0][0] <= 1_000_000_000_000_000)
    var vsm_nan = compute_vsm_cascade_page_origins(Vec3(0.0, -1.0, 0.0), Vec3(nan_f32, 0.0, 0.0))
    assert_equal(vsm_nan[0][0], 0)
    assert_equal(vsm_nan[0][1], 0)
    var vsm_ok = compute_vsm_cascade_page_origins(Vec3(0.0, -1.0, 0.0), Vec3(0.0, 0.0, 0.0))
    assert_equal(vsm_ok[0][0], 0)
    assert_equal(vsm_ok[0][1], 0)

    var vn_inf = eval_value_noise_periodic_cpu(Float32(1.0e38) * Float32(10.0), Float32(0.0))
    assert_true(isfinite(vn_inf) and vn_inf >= Float32(0.0) and vn_inf <= Float32(1.0))
    var vn_nan = eval_value_noise_periodic_cpu(nan_f32, pinf_f32)
    assert_true(isfinite(vn_nan) and vn_nan >= Float32(0.0) and vn_nan <= Float32(1.0))
    var vn_big = eval_value_noise_periodic_cpu(Float32(1.0e38), Float32(0.0))
    assert_true(isfinite(vn_big) and vn_big >= Float32(0.0) and vn_big <= Float32(1.0))
    var vn_ok = eval_value_noise_periodic_cpu(Float32(12.34), Float32(56.78))
    assert_true(isfinite(vn_ok) and vn_ok >= Float32(0.0) and vn_ok <= Float32(1.0))


def _r16_put_le16(mut buf: List[UInt8], val: Int):
    buf.append(UInt8(val & 0xFF))
    buf.append(UInt8((val >> 8) & 0xFF))


def test_r16_partition_ab_fixes() raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)
    var ninf_f32 = f32_from_bits(0xFF800000)
    var nan_f64 = Float64(nan_f32)

    # 1. hdr_image._float3_to_rgbe / encode_radiance_hdr + sample_equirect non-finite guards
    var hdr_inf = encode_radiance_hdr([Vec3(pinf_f32, 0.0, 0.0)], 1, 1)
    var dec_inf = decode_radiance_hdr(hdr_inf)
    assert_true(dec_inf.valid and len(dec_inf.pixels) == 1)
    assert_true(isfinite(dec_inf.pixels[0].x) and dec_inf.pixels[0].x > Float32(1.0e37))
    assert_true(isfinite(dec_inf.pixels[0].y) and dec_inf.pixels[0].y < dec_inf.pixels[0].x * Float32(0.01))

    var hdr_mix = encode_radiance_hdr([Vec3(nan_f32, 1.0, ninf_f32)], 1, 1)
    var dec_mix = decode_radiance_hdr(hdr_mix)
    assert_true(dec_mix.valid and len(dec_mix.pixels) == 1)
    assert_true(isfinite(dec_mix.pixels[0].x) and dec_mix.pixels[0].x < Float32(0.01))
    assert_true(abs(dec_mix.pixels[0].y - Float32(1.0)) < Float32(0.05))
    assert_true(isfinite(dec_mix.pixels[0].z) and dec_mix.pixels[0].z < Float32(0.01))
    var eq_nan = dec_mix.sample_equirect(Vec3(nan_f32, 0.0, 1.0))
    assert_true(isfinite(eq_nan.x) and isfinite(eq_nan.y) and isfinite(eq_nan.z))

    # 2. glb_native.parse_native_glb emissiveFactor / emissiveStrength huge/non-finite sanitization
    var glb_em_json = _str_to_bytes(
        '{"asset":{"version":"2.0"},"buffers":[{"byteLength":36}],'
        '"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],'
        '"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],'
        '"materials":[{"emissiveFactor":[1e500,0,0],"extensions":{"KHR_materials_emissive_strength":{"emissiveStrength":1e500}}}],'
        '"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],'
        '"nodes":[{"mesh":0}]}'
    )
    while len(glb_em_json) % 4 != 0:
        glb_em_json.append(UInt8(0x20))
    var glb_em = List[UInt8]()
    _put_le32(glb_em, 0x46546C67)
    _put_le32(glb_em, 2)
    _put_le32(glb_em, 12 + 8 + len(glb_em_json) + 8 + 36)
    _put_le32(glb_em, len(glb_em_json))
    _put_le32(glb_em, 0x4E4F534A)
    for b in glb_em_json:
        glb_em.append(b)
    _put_le32(glb_em, 36)
    _put_le32(glb_em, 0x004E4942)
    for _ in range(9):
        _put_le32(glb_em, 0)
    var parsed_em = parse_native_glb(glb_em)
    assert_true(parsed_em.loaded() and len(parsed_em.meshes) == 1)
    assert_true(isfinite(parsed_em.meshes[0].emissive_color.x))
    assert_true(isfinite(parsed_em.meshes[0].emissive_color.y))
    assert_true(isfinite(parsed_em.meshes[0].emissive_color.z))
    assert_true(isfinite(parsed_em.meshes[0].emissive_strength) and parsed_em.meshes[0].emissive_strength > Float32(0.0))
    assert_true(abs(parsed_em.meshes[0].emissive_color.x - Float32(1.0)) < Float32(1e-5))

    # 3. ktx2.decode_ktx2 signed SFLOAT16 & SFLOAT32 NaN/Inf/negative clamping
    var f16_neg_pay = List[UInt8]()
    _r16_put_le16(f16_neg_pay, 0xBC00)  # -1.0
    _r16_put_le16(f16_neg_pay, 0x0000)  # 0.0
    _r16_put_le16(f16_neg_pay, 0x0000)  # 0.0
    _r16_put_le16(f16_neg_pay, 0x3C00)  # 1.0
    var k16_neg = decode_ktx2(_r10_build_ktx2_1x1(VK_FORMAT_R16G16B16A16_SFLOAT, 0, f16_neg_pay, UInt64(8)))
    assert_true(k16_neg.valid and len(k16_neg.pixels) == 1)
    assert_equal(k16_neg.pixels[0], UInt32(0xFF000000))

    var f16_nan_pay = List[UInt8]()
    _r16_put_le16(f16_nan_pay, 0x7E00)  # NaN
    _r16_put_le16(f16_nan_pay, 0x3800)  # 0.5
    _r16_put_le16(f16_nan_pay, 0x3800)  # 0.5
    _r16_put_le16(f16_nan_pay, 0x3C00)  # 1.0
    var k16_nan = decode_ktx2(_r10_build_ktx2_1x1(VK_FORMAT_R16G16B16A16_SFLOAT, 0, f16_nan_pay, UInt64(8)))
    assert_true(k16_nan.valid and len(k16_nan.pixels) == 1)
    assert_equal(k16_nan.pixels[0], UInt32(0xFF008080))

    var f16_inf_pay = List[UInt8]()
    _r16_put_le16(f16_inf_pay, 0x7C00)  # +Inf
    _r16_put_le16(f16_inf_pay, 0xFC00)  # -Inf
    _r16_put_le16(f16_inf_pay, 0x0200)  # +subnormal
    _r16_put_le16(f16_inf_pay, 0x8200)  # -subnormal
    var k16_inf = decode_ktx2(_r10_build_ktx2_1x1(VK_FORMAT_R16G16B16A16_SFLOAT, 0, f16_inf_pay, UInt64(8)))
    assert_true(k16_inf.valid and len(k16_inf.pixels) == 1)
    assert_equal(k16_inf.pixels[0], UInt32(0x00FF0000))

    var f32_pay = List[UInt8]()
    _put_le32(f32_pay, 0x7FC00000)  # NaN
    _put_le32(f32_pay, Int(0xFF800000))  # -Inf
    _put_le32(f32_pay, 0x7F800000)  # +Inf
    _put_le32(f32_pay, 0x3F000000)  # 0.5
    var k32 = decode_ktx2(_r10_build_ktx2_1x1(VK_FORMAT_R32G32B32A32_SFLOAT, 0, f32_pay, UInt64(16)))
    assert_true(k32.valid and len(k32.pixels) == 1)
    assert_equal(k32.pixels[0], UInt32(0x800000FF))

    var f32_zero_pay = List[UInt8]()
    _put_le32(f32_zero_pay, 0x7FC00000)
    _put_le32(f32_zero_pay, Int(0xFF800000))
    _put_le32(f32_zero_pay, 0x7FC00000)
    _put_le32(f32_zero_pay, 0x7FC00000)
    var k32_z = decode_ktx2(_r10_build_ktx2_1x1(VK_FORMAT_R32G32B32A32_SFLOAT, 0, f32_zero_pay, UInt64(16)))
    assert_true(k32_z.valid and k32_z.pixels[0] == UInt32(0x00000000))

    # 4. UiContext.scroll_visible_range subnormal/non-finite stride guards
    var ui_ctx = UiContext()
    var inp = InputState()
    ui_ctx.begin_frame(inp, 800.0, 600.0, 1.0 / 60.0)
    ui_ctx.begin_scroll_view("r16_sv", 300.0, 200.0, 2000.0)
    var r_sub = ui_ctx.scroll_visible_range(100, Float32(1.0e-39))
    assert_equal(r_sub[0], 0)
    assert_equal(r_sub[1], 100)
    var r_nan = ui_ctx.scroll_visible_range(100, nan_f32)
    assert_equal(r_nan[0], 0)
    assert_equal(r_nan[1], 100)
    var r_inf = ui_ctx.scroll_visible_range(100, pinf_f32)
    assert_equal(r_inf[0], 0)
    assert_equal(r_inf[1], 100)
    var r_ok = ui_ctx.scroll_visible_range(100, Float32(20.0))
    assert_true(r_ok[0] == 0 and r_ok[1] == 13)
    ui_ctx.end_scroll_view()
    ui_ctx.end_frame()

    # 5. NativeAnimation.frame_count non-finite / huge duration guard
    var na = NativeAnimation()
    na.duration = Float32(1.0e38)
    assert_true(na.frame_count() >= 1)
    na.duration = pinf_f32
    assert_equal(na.frame_count(), 1)
    na.duration = nan_f32
    assert_equal(na.frame_count(), 1)
    na.duration = Float32(-5.0)
    assert_equal(na.frame_count(), 1)
    na.duration = Float32(2.0)
    assert_equal(na.frame_count(), 61)

    # 6. mcp_v2._render_studio_software_preview direct non-finite base_color guard
    var mcp_r16 = McpServerV2()
    var studio_r16 = FlintStudioState()
    var assets_r16 = AssetServer()
    var pool_r16 = JobPool()
    _ = mcp_r16.handle_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ecs.spawn_entity",'
        '"arguments":{"name":"PreviewBadCol","position":[0.0,0.0,0.0]}}}',
        studio_r16, assets_r16, pool_r16,
    )
    if len(studio_r16.entities) > 0:
        studio_r16.entities[0].base_color = Vec3(nan_f32, pinf_f32, ninf_f32)
    var prev_c = _render_studio_software_preview(studio_r16, 64, 64)
    assert_equal(len(prev_c.pixels), 64 * 64)

    # 7. (B1) SimdCanvas.draw_line_3d & ScalableRenderer._render_tier0_simd debug_lines viewport clipping
    var canvas = SimdCanvas(128, 128)
    var cam = Camera3D(128, 128)
    var t_b1 = perf_counter_ns()
    canvas.draw_line_3d(cam, Vec3(0.0, 0.0, 0.0), Vec3(1.0e12, 0.0, 0.0), UInt32(0xFFFFFFFF))
    canvas.draw_line_3d(cam, Vec3(nan_f32, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), UInt32(0xFFFFFFFF))
    var rw_b1 = RenderWorld()
    rw_b1.debug_lines.add_line(Vec3(0.0, 0.0, 0.0), Vec3(1.0e12, 0.0, 0.0), UInt32(0xFF00FF00), thickness_px=nan_f32)
    rw_b1.debug_lines.add_line(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), UInt32(0xFF00FF00), thickness_px=Float32(1.0e30))
    rw_b1.debug_lines.add_line(Vec3(0.0, 0.0, 0.0), Vec3(0.0, -1.0, 0.0), UInt32(0xFF00FF00), thickness_px=Float32(-1.0e30))
    var ren_b1 = ScalableRenderer(RENDER_TIER_0_HEADLESS_SIMD, 128, 128)
    var ui_b1 = UiDrawList()
    ui_b1.add_line(Float32(-1.0e37), Float32(10.0), Float32(3.0e37), Float32(10.0), Float32(2.0), UInt32(0xFFFFFFFF))
    assert_true(ren_b1.render_world(rw_b1, ui_b1))
    var b1_ms = Float64(perf_counter_ns() - t_b1) / 1_000_000.0
    assert_true(b1_ms < 250.0, "draw_line_3d must clip huge endpoints to viewport in O(viewport)")
    var drew_any = False
    for px in canvas.pixels:
        if px == UInt32(0xFFFFFFFF):
            drew_any = True
            break
    assert_true(drew_any, "clipped draw_line_3d must still draw visible segment pixels")

    # 8. (B2) PhysicsWorld XPBD & ArticulatedRagdoll substeps/iterations cap (creation + deser + step)
    var pw_b2 = PhysicsWorld(gravity=-9.81)
    var ch_b2 = ParticleChain.line(Vec3(0.0, 3.0, 0.0), Vec3(1.2, 3.0, 0.0), 4, Float32(0.1))
    var cslot = pw_b2.add_xpbd_chain(ch_b2^, substeps=1_000_000_000, iterations=1_000_000_000)
    assert_equal(pw_b2.xpbd_chains[cslot].substeps, 64)
    assert_equal(pw_b2.xpbd_chains[cslot].iterations, 64)
    var pm_b2 = ParticleMesh()
    _ = pm_b2.add_particle(Vec3(0.0, 2.0, 0.0), Float32(1.0))
    _ = pm_b2.add_particle(Vec3(0.5, 2.0, 0.0), Float32(1.0))
    pm_b2.add_link(0, 1, Float32(0.0))
    var mslot = pw_b2.add_xpbd_mesh(pm_b2^, substeps=1_000_000_000, iterations=1_000_000_000)
    assert_equal(pw_b2.xpbd_meshes[mslot].substeps, 64)
    assert_equal(pw_b2.xpbd_meshes[mslot].iterations, 64)
    # Tamper serialized blob so xpbd chain has uncapped substeps/iterations on wire
    pw_b2.xpbd_chains[cslot].substeps = 0x11223344
    var bw_b2 = BlobWriter()
    pw_b2.serialize(bw_b2)
    var pw_b2_deser = PhysicsWorld()
    var r_b2 = BlobReader(bw_b2.data.copy())
    pw_b2_deser.deserialize(r_b2)
    assert_equal(pw_b2_deser.xpbd_chains[0].substeps, 64)
    var t_b2 = perf_counter_ns()
    pw_b2_deser.step(Float32(1.0 / 60.0))
    var rag_b2 = ArticulatedRagdoll()
    _ = rag_b2.add_limb(ArticulatedLimbDef(bone_node=-1, parent_limb=-1, mass=10.0, radius=0.15, length=0.35))
    rag_b2.step(Float32(0.016), substeps=1_000_000_000)
    var b2_ms = Float64(perf_counter_ns() - t_b2) / 1_000_000.0
    assert_true(b2_ms < 250.0, "XPBD & ArticulatedRagdoll must cap substeps")
    assert_true(isfinite(rag_b2.states[0].com_pos.y))

    # 9. (B3) CameraRig.update & _det_perlin_1d huge/NaN dt guards
    assert_true(isfinite(_det_perlin_1d(123, Float32(1.0e38))))
    assert_true(isfinite(_det_perlin_1d(123, nan_f32)))
    assert_true(isfinite(_det_perlin_1d(123, pinf_f32)))
    var pw_cam = PhysicsWorld(gravity=-9.81)
    var rig1 = CameraRig()
    rig1.trauma = Float32(1.0)
    rig1.update(pw_cam, Vec3(), Float32(1.0e38))
    assert_true(isfinite(rig1.world_pos.x) and isfinite(rig1.world_pos.y) and isfinite(rig1.world_pos.z))
    assert_true(isfinite(rig1.world_rot.x) and isfinite(rig1.world_rot.w))
    var rig2 = CameraRig()
    rig2.trauma = Float32(1.0)
    rig2.update(pw_cam, Vec3(), nan_f32)
    assert_true(isfinite(rig2.world_pos.x) and isfinite(rig2.world_pos.y) and isfinite(rig2.world_pos.z))
    assert_true(isfinite(rig2.world_rot.x) and isfinite(rig2.world_rot.w))

    # 10. (B4) PhysicsWorld.step huge gravity + huge dt NaN guard
    var pw_g = PhysicsWorld(gravity=Float32(1.0e30))
    var sp_b = pw_g.add_sphere(Vec3(0.0, 10.0, 0.0), Float32(0.5), Float32(1.0))
    pw_g.step(Float32(1.0e12))
    assert_true(sp_b >= 0 and sp_b < len(pw_g.bodies))
    ref bs = pw_g.bodies[sp_b]
    assert_true(isfinite(bs.pos.x) and isfinite(bs.pos.y) and isfinite(bs.pos.z))
    assert_true(isfinite(bs.linear_vel.x) and isfinite(bs.linear_vel.y) and isfinite(bs.linear_vel.z))
    assert_true(isfinite(bs.angular_vel.x) and isfinite(bs.angular_vel.y) and isfinite(bs.angular_vel.z))

    # 11. (B5) Breathing.apply, Lean.apply + sibling procedural animators, particle_atlas, cycles.Pattern.scale
    var br_model = _make_test_model(1.0, 1.0)
    var br = Breathing()
    br.apply(br_model, 0, -1, Float32(0.5), Float32(0.25), Float32(0.5), Float32(2.0), Float32(1.0e20))
    br.apply(br_model, 0, -1, Float32(0.5), Float32(0.25), Float32(0.5), Float32(2.0), nan_f32)
    assert_true(isfinite(br.phase) and isfinite(br.applied))

    # Round 16b regression: Lean.apply dt=1e38 + non-finite inputs + sibling SpringFloat callers
    var sp_huge = SpringFloat(1.0)
    sp_huge.update(Float32(0.0), Float32(0.12), Float32(1.0e38))
    assert_true(isfinite(sp_huge.value) and isfinite(sp_huge.velocity))
    assert_equal(sp_huge.value, Float32(0.0))

    var lean = Lean()
    lean.roll.value = Float32(1.0)
    lean.pitch.value = Float32(-1.0)
    lean.apply(br_model, 0, Float32(10.0), Float32(5.0), Float32(1.0), Float32(1.0), Float32(30.0), Float32(1.0e38))
    assert_true(isfinite(lean.roll.value) and isfinite(lean.roll.velocity))
    assert_true(isfinite(lean.pitch.value) and isfinite(lean.pitch.velocity))
    lean.apply(br_model, 0, nan_f32, pinf_f32, nan_f32, pinf_f32, nan_f32, Float32(1.0e38))
    lean.apply(br_model, 0, Float32(1.0e38), Float32(-1.0e38), Float32(1.0), Float32(1.0), Float32(30.0), nan_f32)
    lean.apply(br_model, 0, Float32(15.0), Float32(4.0), Float32(1.0), Float32(1.0), Float32(30.0), pinf_f32)
    assert_true(isfinite(lean.roll.value) and isfinite(lean.pitch.value))
    assert_true(isfinite(lean.last_yaw) and isfinite(lean.last_speed))

    var look_r16b = LookAt()
    look_r16b.weight.value = Float32(1.0)
    look_r16b.apply(br_model, 0, -1, Vec3(Float32(1.0), Float32(0.0), Float32(2.0)), True, Float32(60.0), Float32(45.0), Float32(0.3), Float32(0.1), Float32(1.0e38))
    look_r16b.apply(br_model, 0, -1, Vec3(nan_f32, pinf_f32, Float32(0.0)), True, nan_f32, nan_f32, nan_f32, nan_f32, Float32(1.0e38))
    assert_true(isfinite(look_r16b.weight.value) and isfinite(look_r16b.yaw.value) and isfinite(look_r16b.pitch.value))

    var stab_r16b = HeadStabilizer()
    stab_r16b.apply(br_model, 0, Float32(0.5), Float32(0.1), Float32(0.016))
    stab_r16b.apply(br_model, 0, nan_f32, nan_f32, Float32(1.0e38))
    assert_true(isfinite(stab_r16b.smoothed.w) and isfinite(stab_r16b.smoothed.x))

    var recoil_r16b = Recoil()
    recoil_r16b.kick(Vec3(0.0, 0.0, -1.0), Float32(3.0))
    _ = recoil_r16b.step(Float32(120.0), Float32(14.0), Float32(1.0e38))
    _ = recoil_r16b.step(nan_f32, nan_f32, Float32(1.0e38))
    assert_true(isfinite(recoil_r16b.offset.x) and isfinite(recoil_r16b.velocity.x))

    var ow_r16b = OrientationWarp()
    ow_r16b.angle.value = Float32(15.0)
    ow_r16b.apply(br_model, 0, -1, Float32(0.0), Float32(45.0), True, Float32(40.0), Float32(0.5), Float32(1.0e38))
    ow_r16b.apply(br_model, 0, -1, nan_f32, pinf_f32, True, nan_f32, nan_f32, Float32(1.0e38))
    assert_true(isfinite(ow_r16b.angle.value) and isfinite(ow_r16b.applied))

    var mw_r16b = MotionWarp()
    mw_r16b.begin(0, Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Float32(0.5), Float32(2.0))
    mw_r16b.apply(br_model, 0, Float32(0.25), Float32(1.0e38), Float32(0.1))
    assert_true(isfinite(mw_r16b.shown.value.x) and isfinite(mw_r16b.shown.velocity.x))

    var fp_r16b = FootPlantSolver(0)
    var skel_r16b = SkeletonInstance(len(br_model.nodes))
    fp_r16b.add_leg(0, 0, 0)
    fp_r16b.pelvis_offset_y.value = Float32(-0.2)
    fp_r16b.apply_to_skeleton(
        skel_r16b, br_model, Placement(),
        [False], [nan_f32], [Vec3(nan_f32, 1.0, 0.0)],
        Float32(1.0e38),
    )
    assert_true(isfinite(fp_r16b.pelvis_offset_y.value) and isfinite(fp_r16b.legs[0].stance_blend.value))

    var vn_nan2 = atlas_value_noise(nan_f32, Float32(0.0), 1)
    assert_true(isfinite(vn_nan2) and vn_nan2 >= Float32(0.0) and vn_nan2 <= Float32(1.0))
    var vn_huge2 = atlas_value_noise(Float32(1.0e30), Float32(-1.0e30), 1)
    assert_true(isfinite(vn_huge2) and vn_huge2 >= Float32(0.0) and vn_huge2 <= Float32(1.0))
    assert_equal(atlas_to_byte(nan_f32), UInt8(0))

    var pat_sc = Pattern()
    var pn_huge = pat_sc.pure_number(Float64(1.0e30))
    pat_sc.root = pat_sc.scale(0, pn_huge)
    var ev_huge = pat_sc.query(Frac(0, 1), Frac(1, 1))
    assert_equal(len(ev_huge), 1)
    assert_true(isfinite(ev_huge[0].value.num) and abs(ev_huge[0].value.num) < Float64(1.0e15))
    var pat_nan = Pattern()
    pat_nan.root = pat_nan.scale(0, pat_nan.pure_number(nan_f64))
    var ev_nan = pat_nan.query(Frac(0, 1), Frac(1, 1))
    assert_equal(len(ev_nan), 1)
    assert_equal(ev_nan[0].value.num, Float64(0.0))


def _r17_build_2x1_float_exr(
    r0: UInt32, g0: UInt32, b0: UInt32,
    r1: UInt32, g1: UInt32, b1: UInt32,
) -> List[UInt8]:
    var out = List[UInt8]()
    _put_le32(out, EXR_MAGIC)
    _put_le32(out, 2)
    # channels: B, G, R (FLOAT = 2)
    for b in String("channels").as_bytes(): out.append(b)
    out.append(0)
    for b in String("chlist").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 3 * 18 + 1)
    for cname in ["B", "G", "R"]:
        out.append(cname.as_bytes()[0]); out.append(0)
        _put_le32(out, EXR_PIXEL_FLOAT)
        out.append(1); out.append(0); out.append(0); out.append(0)
        _put_le32(out, 1); _put_le32(out, 1)
    out.append(0)
    # compression = NONE
    for b in String("compression").as_bytes(): out.append(b)
    out.append(0)
    for b in String("compression").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 1)
    out.append(UInt8(EXR_NO_COMPRESSION))
    # dataWindow 0..1, 0..0 (2x1)
    for b in String("dataWindow").as_bytes(): out.append(b)
    out.append(0)
    for b in String("box2i").as_bytes(): out.append(b)
    out.append(0)
    _put_le32(out, 16)
    _put_le32(out, 0); _put_le32(out, 0); _put_le32(out, 1); _put_le32(out, 0)
    out.append(0)  # end of header
    var chunk_off = len(out) + 8
    _put_le64(out, UInt64(chunk_off))
    var raw_bytes = 2 * 3 * 4
    _put_le32(out, 0)  # y = 0
    _put_le32(out, raw_bytes)
    _put_le32(out, Int(b0)); _put_le32(out, Int(b1))
    _put_le32(out, Int(g0)); _put_le32(out, Int(g1))
    _put_le32(out, Int(r0)); _put_le32(out, Int(r1))
    return out^


def test_r17_partition_ab_fixes(tmp_dir: String) raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)
    var ninf_f32 = f32_from_bits(0xFF800000)

    # 1. (A1) ProfileTimeline register_zone cap/escape + >1 MiB buffer flush bounds safety
    var huge_name = ("A" * 4000) + "\"\\\n\t" + ("B" * 4192)
    var pt = ProfileTimeline()
    var zid1 = pt.register_zone(huge_name)
    assert_equal(pt.register_zone(huge_name), zid1)
    assert_true(pt.zone_label(zid1).byte_length() <= 96)
    var trace_path = tmp_dir + "/r17_trace.json"
    assert_true(pt.start(trace_path, overlay=False))
    var huge_post = "\"quoted\"\n\r\t" + ("Z" * 8192)
    var zid2 = pt.register_zone(huge_post)
    assert_true(pt.zone_label(zid2).byte_length() <= 96)
    for _ in range(8):
        pt.begin_frame()
        for _ in range(750):
            pt.handle.record(zid1, UInt64(100), UInt64(200))
            pt.handle.record(zid2, UInt64(250), UInt64(350))
        pt.end_frame()
    pt.finish()
    var trace_bytes = read_bytes(trace_path)
    assert_true(len(trace_bytes) > 1_048_576, "Trace must cross 1 MiB FILE_BUFFER_BYTES without overflow")
    var trace_doc = parse_json(trace_bytes)
    assert_true(trace_doc.valid, "ProfileTimeline trace JSON must remain valid after escaping & truncation")
    assert_true(trace_doc.array_size(trace_doc.root) > 12_000)
    _ = remove_file(trace_path)

    # 2. (A2) OpenEXR +Inf / NaN / DWA overflow sanitization -> finite sample_equirect
    var exr_inf_bytes = _r17_build_2x1_float_exr(
        UInt32(0x7FC00000), UInt32(0x7F800000), UInt32(0xFF800000),
        UInt32(0x3F800000), UInt32(0x00000000), UInt32(0x3F000000),
    )
    var exr_inf = decode_exr_fp16(exr_inf_bytes)
    assert_true(exr_inf.valid and len(exr_inf.pixels) == 2)
    assert_true(isfinite(exr_inf.pixels[0].x) and exr_inf.pixels[0].x == Float32(0.0))
    assert_true(isfinite(exr_inf.pixels[0].y) and exr_inf.pixels[0].y > Float32(0.0))
    assert_true(isfinite(exr_inf.pixels[0].z) and exr_inf.pixels[0].z == Float32(0.0))
    var s_eq0 = exr_inf.sample_equirect(Vec3(0.0, 0.0, 1.0))
    var s_eq1 = exr_inf.sample_equirect(Vec3(1.0, 0.0, 0.0))
    assert_true(isfinite(s_eq0.x) and isfinite(s_eq0.y) and isfinite(s_eq0.z))
    assert_true(isfinite(s_eq1.x) and isfinite(s_eq1.y) and isfinite(s_eq1.z))
    assert_true(isfinite(_dwa_to_linear(Float32(100.0))))
    assert_true(isfinite(_dwa_to_linear(pinf_f32)))
    assert_true(isfinite(_dwa_to_linear(ninf_f32)))
    assert_equal(_dwa_to_linear(nan_f32), Float32(0.0))
    var raw_hdr = HdrImage()
    raw_hdr.valid = True; raw_hdr.width = 2; raw_hdr.height = 1
    raw_hdr.pixels = [Vec3(pinf_f32, nan_f32, ninf_f32), Vec3(1.0, 2.0, 3.0)]
    var s_raw = raw_hdr.sample_equirect(Vec3(0.0, 0.0, 1.0))
    assert_true(isfinite(s_raw.x) and isfinite(s_raw.y) and isfinite(s_raw.z))

    # 3. (B1) springs.mojo Float64 intermediate + isfinite post-guard (2.5e38)
    var sf_25 = SpringFloat(Float32(2.5e38))
    sf_25.update(Float32(0.0), Float32(2.0), Float32(0.5))
    assert_true(isfinite(sf_25.value) and isfinite(sf_25.velocity))
    assert_true(sf_25.value < Float32(2.5e38))
    var sv_25 = SpringVec3(Vec3(Float32(2.5e38), Float32(-2.5e38), Float32(0.0)))
    sv_25.update(Vec3(), Float32(2.0), Float32(0.5))
    assert_true(isfinite(sv_25.value.x) and isfinite(sv_25.value.y) and isfinite(sv_25.value.z))
    assert_true(isfinite(sv_25.velocity.x) and isfinite(sv_25.velocity.y))
    var sa_25 = SpringAngle(Float32(2.5e38))
    sa_25.update(Float32(0.0), Float32(2.0), Float32(0.5))
    assert_true(isfinite(sa_25.value) and isfinite(sa_25.velocity))
    var dec_off = Vec3(Float32(2.5e38), Float32(0.0), Float32(-2.5e38))
    var dec_vel = Vec3(Float32(2.5e38), Float32(0.0), Float32(-2.5e38))
    decay_spring(dec_off, dec_vel, Float32(2.0), Float32(0.5))
    assert_true(isfinite(dec_off.x) and isfinite(dec_off.z) and isfinite(dec_vel.x))
    var ds_off = Float32(2.5e38)
    var ds_vel = Float32(2.5e38)
    decay_spring_scalar(ds_off, ds_vel, Float32(2.0), Float32(0.5))
    assert_true(isfinite(ds_off) and isfinite(ds_vel))
    assert_equal(wrap_degrees(nan_f32), Float32(0.0))
    assert_equal(wrap_degrees(pinf_f32), Float32(0.0))
    assert_equal(fast_negexp(Float32(-2.5e38)), Float32(1.0))
    assert_equal(fast_negexp(Float32(-3.4e38)), Float32(1.0))
    assert_equal(fast_negexp(Float32(0.0)), Float32(1.0))
    assert_equal(fast_negexp(Float32(1.0e12)), Float32(0.0))
    assert_equal(fast_negexp(Float32(1.0e38)), Float32(0.0))
    assert_equal(fast_negexp(nan_f32), Float32(1.0))
    assert_equal(fast_negexp(pinf_f32), Float32(0.0))
    assert_equal(fast_negexp(ninf_f32), Float32(1.0))
    assert_true(fast_negexp(Float32(1.0)) > Float32(0.35) and fast_negexp(Float32(1.0)) < Float32(0.38))

    # 4. (B2) aero6dof.mojo step_aero6dof NaN/1e38 dt + substeps=1e9 hang guard + ZEM-APN
    var acfg = Aero6DofConfig()
    var actrl = Aero6DofControls()
    var ast_nan = Aero6DofState()
    step_aero6dof(ast_nan, acfg, actrl, nan_f32)
    assert_true(isfinite(ast_nan.pos.y) and isfinite(ast_nan.linear_vel.y))
    var ast_bad = Aero6DofState()
    ast_bad.pos = Vec3(nan_f32, pinf_f32, ninf_f32)
    ast_bad.linear_vel = Vec3(pinf_f32, nan_f32, Float32(1.0e30))
    ast_bad.angular_vel = Vec3(ninf_f32, pinf_f32, nan_f32)
    ast_bad.rot = Quat(nan_f32, 0.0, 0.0, pinf_f32)
    ast_bad.morph_u = nan_f32
    ast_bad.fuel_mass = ninf_f32
    ast_bad.mach = pinf_f32
    ast_bad.g_load = nan_f32
    step_aero6dof(ast_bad, acfg, actrl, nan_f32)
    assert_true(isfinite(ast_bad.pos.x) and isfinite(ast_bad.pos.y) and isfinite(ast_bad.pos.z))
    assert_true(isfinite(ast_bad.linear_vel.x) and isfinite(ast_bad.linear_vel.y) and isfinite(ast_bad.linear_vel.z))
    assert_true(isfinite(ast_bad.angular_vel.x) and isfinite(ast_bad.angular_vel.y) and isfinite(ast_bad.angular_vel.z))
    assert_true(isfinite(ast_bad.rot.w) and isfinite(ast_bad.mach) and isfinite(ast_bad.g_load))
    var ast_vel30 = Aero6DofState(Vec3(0.0, 500.0, 0.0), Vec3(Float32(1.0e30), Float32(-1.0e30), Float32(1.0e30)))
    step_aero6dof(ast_vel30, acfg, actrl, Float32(0.016))
    assert_true(isfinite(ast_vel30.pos.x) and isfinite(ast_vel30.linear_vel.x) and isfinite(ast_vel30.mach) and isfinite(ast_vel30.g_load))
    var ast_huge = Aero6DofState()
    step_aero6dof(ast_huge, acfg, actrl, Float32(1.0e38))
    assert_true(isfinite(ast_huge.pos.y) and isfinite(ast_huge.linear_vel.y) and isfinite(ast_huge.rot.w))
    var ast_sub = Aero6DofState()
    var t_aero = perf_counter_ns()
    step_aero6dof(ast_sub, acfg, actrl, Float32(0.016), substeps=1_000_000_000)
    var aero_ms = Float64(perf_counter_ns() - t_aero) / 1_000_000.0
    assert_true(aero_ms < 250.0, "step_aero6dof must cap substeps")
    assert_true(isfinite(ast_sub.pos.y) and isfinite(ast_sub.linear_vel.y))
    var m_pos = V8(F8(0.0), F8(0.0), F8(0.0))
    var m_vel = V8(F8(0.0), F8(0.0), F8(100.0))
    var m_ph = F8(0.0)
    var m_r1 = step_zem_apn_simd8(m_pos, m_vel, m_ph, V8(F8(0.0), F8(0.0), F8(500.0)), V8.zero(), V8.zero(), nan_f32)
    var m_r2 = step_zem_apn_simd8(m_pos, m_vel, m_ph, V8(F8(0.0), F8(0.0), F8(500.0)), V8.zero(), V8.zero(), Float32(1.0e38))
    assert_true(isfinite(m_r1[0]) and isfinite(m_r2[0]) and isfinite(m_pos.z[0]))
    var m_pos_ext = V8.zero()
    var m_vel_ext = V8(F8(0.0), F8(0.0), F8(100.0))
    var m_ph_ext = F8(0.0)
    var m_r30 = step_zem_apn_simd8(
        m_pos_ext,
        m_vel_ext,
        m_ph_ext,
        V8(F8(1.0e30), F8(-1.0e30), F8(1.0e30)),
        V8.zero(),
        V8.zero(),
        Float32(1.0e38),
    )
    var m_r30_nan = step_zem_apn_simd8(
        m_pos_ext,
        m_vel_ext,
        m_ph_ext,
        V8(F8(1.0e30), F8(-1.0e30), F8(1.0e30)),
        V8.zero(),
        V8.zero(),
        nan_f32,
    )
    comptime for lane in range(8):
        assert_true(isfinite(m_r30[lane]) and m_r30[lane] > Float32(1.0e29))
        assert_true(isfinite(m_r30_nan[lane]) and m_r30_nan[lane] > Float32(1.0e29))
        assert_true(isfinite(m_pos_ext.x[lane]) and isfinite(m_pos_ext.y[lane]) and isfinite(m_pos_ext.z[lane]))
        assert_true(isfinite(m_vel_ext.x[lane]) and isfinite(m_vel_ext.y[lane]) and isfinite(m_vel_ext.z[lane]))
    var m_pos_nan = V8.zero()
    var m_vel_nan = V8(F8(1.0e30), F8(nan_f32), F8(pinf_f32))
    var m_ph_nan = F8(nan_f32)
    var m_r_nan = step_zem_apn_simd8(
        m_pos_nan,
        m_vel_nan,
        m_ph_nan,
        V8(F8(nan_f32), F8(pinf_f32), F8(ninf_f32)),
        V8(F8(nan_f32), F8(0.0), F8(1.0e30)),
        V8(F8(pinf_f32), F8(nan_f32), F8(1.0e30)),
        Float32(1.0e38),
    )
    comptime for lane in range(8):
        assert_true(isfinite(m_r_nan[lane]))
        assert_true(isfinite(m_pos_nan.x[lane]) and isfinite(m_pos_nan.y[lane]) and isfinite(m_pos_nan.z[lane]))
        assert_true(isfinite(m_vel_nan.x[lane]) and isfinite(m_vel_nan.y[lane]) and isfinite(m_vel_nan.z[lane]))

    # 5. (B3) orca.mojo CrowdSimulator.step_orca_simd8 dt=1e38 + extreme 1e30 agent/wall coords
    var crowd = CrowdSimulator()
    _ = crowd.add_agent(Vec3(0.0, 0.0, 0.0), Float32(0.35), Float32(5.0), Vec3(10.0, 0.0, 10.0))
    _ = crowd.add_agent(Vec3(Float32(1.0e30), 0.0, Float32(-1.0e30)), Float32(0.35), Float32(5.0))
    crowd.add_wall_segment(Vec3(Float32(1.0e30), 0.0, 0.0), Vec3(Float32(1.0e30), 0.0, 10.0))
    crowd.step_orca_simd8(Float32(1.0e38))
    assert_true(isfinite(crowd.px[0]) and isfinite(crowd.pz[0]))
    assert_true(isfinite(crowd.px[1]) and isfinite(crowd.pz[1]))

    # 6. (B4) navmesh.mojo build_from_triangle_lists 1e30 & NaN triangles
    var nav_huge = NavMesh.build_from_triangle_lists(
        [Vec3(0.0, 0.0, 0.0)],
        [Vec3(Float32(1.0e30), 0.0, 0.0)],
        [Vec3(0.0, 0.0, 10.0)],
    )
    assert_equal(nav_huge.chf.width, 1024)
    var nav_nan = NavMesh.build_from_triangle_lists(
        [Vec3(nan_f32, 0.0, 0.0)],
        [Vec3(1.0, 0.0, 0.0)],
        [Vec3(0.0, 0.0, 1.0)],
    )
    assert_equal(nav_nan.poly_count(), 0)
    var nav_mix = NavMesh.build_from_triangle_lists(
        [Vec3(nan_f32, 0.0, 0.0), Vec3(-4.0, 0.0, -4.0), Vec3(-4.0, 0.0, -4.0)],
        [Vec3(1.0, 0.0, 0.0), Vec3(-4.0, 0.0, 4.0), Vec3(4.0, 0.0, 4.0)],
        [Vec3(0.0, 0.0, 1.0), Vec3(4.0, 0.0, 4.0), Vec3(4.0, 0.0, -4.0)],
    )
    assert_true(nav_mix.poly_count() > 0)
    assert_equal(nav_mix.find_nearest_poly(Vec3(nan_f32, 0.0, 0.0))[0], -1)

    # 7. (B5) ml/ops.mojo B_MOD IEEE fmod + cast saturation
    var mod_res = binary(
        B_MOD,
        Tensor.floats([1], [Float32(1.0e10)]),
        Tensor.floats([1], [Float32(1.0e-10)]),
    )
    assert_true(isfinite(mod_res.fget(0)) and mod_res.fget(0) >= Float32(0.0) and mod_res.fget(0) < Float32(1.0e-10))
    var mod_edge = binary(
        B_MOD,
        Tensor.floats([3], [Float32(5.5), Float32(-5.5), Float32(1.0)]),
        Tensor.floats([3], [Float32(2.0), Float32(2.0), Float32(0.0)]),
    )
    assert_true(abs(mod_edge.fget(0) - Float32(1.5)) < Float32(1e-5))
    assert_true(abs(mod_edge.fget(1) + Float32(1.5)) < Float32(1e-5))
    assert_true(isnan(mod_edge.fget(2)))

    var cast_src = Tensor.floats([6], [Float32(1.0e30), Float32(-1.0e30), nan_f32, pinf_f32, ninf_f32, Float32(42.0)])
    var cast_i32 = cast(cast_src, DT_I32)
    assert_equal(cast_i32.i[0], 2147483647)
    assert_equal(cast_i32.i[1], -2147483648)
    assert_equal(cast_i32.i[2], 0)
    assert_equal(cast_i32.i[3], 2147483647)
    assert_equal(cast_i32.i[4], -2147483648)
    assert_equal(cast_i32.i[5], 42)
    var cast_i64 = cast(cast_src, DT_I64)
    assert_equal(cast_i64.i[0], 0x7FFFFFFFFFFFFFFF)
    assert_equal(cast_i64.i[1], Int(-0x7FFFFFFFFFFFFFFF - 1))
    assert_equal(cast_i64.i[2], 0)

    # 8. (B6) render/grading.mojo to_byte NaN/Inf + LookDirector non-finite dt
    assert_equal(grading_to_byte(nan_f32), UInt8(0))
    assert_equal(grading_to_byte(ninf_f32), UInt8(0))
    assert_equal(grading_to_byte(pinf_f32), UInt8(0))
    assert_equal(grading_to_byte(Float32(1.5)), UInt8(255))
    assert_equal(grading_to_byte(Float32(0.5)), UInt8(128))
    var ld = LookDirector()
    var lp = ld.update(0, nan_f32, 1, Vec3(1.0, 1.0, 1.0), nan_f32, nan_f32)
    assert_true(isfinite(lp.exposure) and isfinite(lp.saturation))


def test_r18_partition_ab_fixes(tmp_dir: String) raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)
    var ninf_f32 = f32_from_bits(0xFF800000)

    # 1. (A1) flpk_v3.mojo parse_flmeta_bytes / AssetMeta.to_json / settings_hash
    var bad_flmeta = _str_to_bytes('{"uuid":"11111111-2222-3333-4444-555555555555","toksvig_scale":1e40}')
    assert_false(parse_flmeta_bytes(bad_flmeta)[0])
    var ok_flmeta = _str_to_bytes('{"uuid":"11111111-2222-3333-4444-555555555555","toksvig_scale":1.5}')
    var ok_meta_res = parse_flmeta_bytes(ok_flmeta)
    assert_true(ok_meta_res[0])
    assert_true(abs(ok_meta_res[1].toksvig_scale - Float32(1.5)) < Float32(1.0e-5))
    var dirty_meta = AssetMeta()
    dirty_meta.uuid = ok_meta_res[1].uuid
    dirty_meta.toksvig_scale = pinf_f32
    _ = dirty_meta.settings_hash()
    var dirty_json = dirty_meta.to_json()
    var rt_meta = parse_flmeta_bytes(_str_to_bytes(dirty_json))
    assert_true(rt_meta[0])
    assert_true(isfinite(rt_meta[1].toksvig_scale))

    # 2. (A2) scene.mojo & studio_app.mojo transactional scene loading
    var st = FlintStudioState(headless=True)
    var e1 = st.spawn_entity("Hero", Vec3(1.0, 2.0, 3.0))
    var e2 = st.spawn_entity("Sidekick", Vec3(4.0, 5.0, 6.0))
    assert_equal(st.app.world.entity_count(), 2)
    var bad_flscn_path = tmp_dir + "/bad_r18.flscn"
    _write_text(
        bad_flscn_path,
        '{"format":"flscn","version":1,"name":"Broken","entities":[{"slot":0,"parent":99,"components":{"LocalTransform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]}}}]}',
    )
    assert_false(st.load_scene_file(bad_flscn_path))
    assert_equal(st.app.world.entity_count(), 2)
    assert_true(st.app.world.is_alive(st.resolve_entity(e1)))
    assert_true(st.app.world.is_alive(st.resolve_entity(e2)))
    assert_equal(st.scene.name, "FlintStudioScene")

    # Also verify out-of-range parent slot & parent cycle & non-finite transform fail transactionally
    var reg_s = ComponentRegistry.with_builtins()
    var w_tx = World()
    reg_s.ensure_registered_in_world(w_tx)
    var ew = w_tx.spawn_empty()
    assert_true(w_tx.insert(ew, LocalTransform(Vec3(7.0, 8.0, 9.0))))
    var sc_tx = Scene("LiveScene")
    var bad_parent_json = _str_to_bytes(
        '{"format":"flscn","version":1,"name":"BadPar","entities":[{"slot":0,"parent":99,"components":{}}]}'
    )
    var raised_par = False
    try:
        _ = sc_tx.load_flscn_bytes(w_tx, reg_s, bad_parent_json)
    except:
        raised_par = True
    assert_true(raised_par)
    assert_equal(w_tx.entity_count(), 1)
    assert_equal(sc_tx.name, "LiveScene")

    var bad_nan_json = _str_to_bytes(
        '{"format":"flscn","version":1,"name":"BadNan","entities":[{"slot":0,"parent":-1,"components":{"LocalTransform":{"translation":[1e40,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]}}}]}'
    )
    var raised_nan_scn = False
    try:
        _ = sc_tx.load_flscn_bytes(w_tx, reg_s, bad_nan_json)
    except:
        raised_nan_scn = True
    assert_true(raised_nan_scn)
    assert_equal(w_tx.entity_count(), 1)
    assert_equal(sc_tx.name, "LiveScene")

    # 3. (A3) ttf.mojo & sdf_bake.mojo CFF Type 2 overflow -> NaN guard
    # Type 2: push 30000 (28, 0x75, 0x30), multiply 10 times to overflow +Inf, then 0 * +Inf = NaN -> rmoveto
    var cs_ovf = List[UInt8]()
    cs_ovf.append(UInt8(28)); cs_ovf.append(UInt8(0x75)); cs_ovf.append(UInt8(0x30))
    for _ in range(10):
        cs_ovf.append(UInt8(28)); cs_ovf.append(UInt8(0x75)); cs_ovf.append(UInt8(0x30))
        cs_ovf.append(UInt8(12)); cs_ovf.append(UInt8(24))  # mul
    cs_ovf.append(UInt8(139))  # 0
    cs_ovf.append(UInt8(12)); cs_ovf.append(UInt8(24))  # 0 * +Inf = NaN
    cs_ovf.append(UInt8(139))  # 0
    cs_ovf.append(UInt8(21))   # rmoveto
    cs_ovf.append(UInt8(14))   # endchar
    var ovf_otf = _r9_build_cff_otf(List[List[UInt8]](), [cs_ovf^])
    var font_ovf = TrueTypeFont()
    assert_true(font_ovf.load_bytes(ovf_otf^))
    var out_ovf = font_ovf.cff_outline(0)
    assert_equal(len(out_ovf.xs), 0)
    var baked_ovf = bake_glyph_id(font_ovf, 0, 32, 4)
    assert_equal(baked_ovf.w, 0)
    assert_equal(baked_ovf.h, 0)

    # 4. (A4) ui/context.mojo slider_float NaN & extreme ±2e38 bounds
    var uictx = UiContext()
    var inp_ui = InputState()
    inp_ui.move_mouse(Float32(40.0), Float32(12.0))
    inp_ui.set_mouse(0, True)
    var slider_v = nan_f32
    uictx.begin_frame(inp_ui, Float32(1280.0), Float32(720.0))
    _ = uictx.slider_float("sl_nan", "S", slider_v, Float32(0.0), Float32(10.0))
    uictx.end_frame()
    assert_true(isfinite(slider_v))

    var ext_v = Float32(0.0)
    var inp_ext = InputState()
    inp_ext.move_mouse(Float32(60.0), Float32(12.0))
    inp_ext.set_mouse(0, True)
    var uictx2 = UiContext()
    uictx2.begin_frame(inp_ext, Float32(1280.0), Float32(720.0))
    var ext_chg = uictx2.slider_float("sl_ext", "E", ext_v, Float32(-2.0e38), Float32(2.0e38))
    uictx2.end_frame()
    assert_true(ext_chg and isfinite(ext_v))

    # 5. (A5) scene/transform_hierarchy.mojo _f32_json & read_json validation
    var bad_lt = LocalTransform(Vec3(pinf_f32, nan_f32, Float32(1.0)))
    var lt_json = bad_lt.write_json()
    var lt_doc = parse_json(_str_to_bytes(lt_json))
    assert_true(lt_doc.valid)
    var lt_rt = LocalTransform.read_json(lt_doc, lt_doc.root)
    assert_true(isfinite(lt_rt.translation.x) and isfinite(lt_rt.translation.y))
    var lt_ovf_doc = parse_json(_str_to_bytes('{"translation":[1e40,0,0],"rotation":[0,0,0,1],"scale":[1,1,1]}'))
    assert_true(lt_ovf_doc.valid)
    var lt_ovf_raised = False
    try:
        _ = LocalTransform.read_json(lt_ovf_doc, lt_ovf_doc.root)
    except:
        lt_ovf_raised = True
    assert_true(lt_ovf_raised)

    # 6. (A6) input/input_map.mojo _fmt_f & save_json
    var im = InputMap()
    _ = im.register_action("move", 1)
    var im_ctx = InputContext("default")
    im_ctx.bind_key("move", 65, scale_x=Float32(1.0e40), scale_y=nan_f32)
    im.add_context(im_ctx^)
    im.push_context("default")
    var im_json = im.save_json()
    var im_doc = parse_json(_str_to_bytes(im_json))
    assert_true(im_doc.valid)
    var im2 = InputMap()
    assert_true(im2.load_json(im_json))

    # 7. (A7) ui/layout.mojo ui_frame scale<=0 & canvas_floor/ceil NaN/Inf
    var uf = ui_frame(1280, 720, 0, 0, Float32(0.0))
    assert_true(uf.width() > 0 and uf.height() > 0 and isfinite(uf.scale) and uf.scale > Float32(0.0))
    assert_equal(canvas_floor(nan_f32), 0)
    assert_equal(canvas_ceil(pinf_f32), 0)

    # 8. (A8) assets/meshlet.mojo simplify_mesh_qem NaN vertex guard
    var bad_mesh = NativeMesh()
    for _ in range(9):
        bad_mesh.positions.append(nan_f32)
        bad_mesh.positions.append(Float32(0.0))
        bad_mesh.positions.append(Float32(0.0))
        bad_mesh.normals.append(Float32(0.0))
        bad_mesh.normals.append(Float32(1.0))
        bad_mesh.normals.append(Float32(0.0))
        bad_mesh.texcoords.append(Float32(0.0))
        bad_mesh.texcoords.append(Float32(0.0))
    for ti in range(9):
        bad_mesh.indices.append(ti % 9)
        bad_mesh.indices.append((ti + 1) % 9)
        bad_mesh.indices.append((ti + 2) % 9)
    var simp_bad = simplify_mesh_qem(bad_mesh, 2)
    assert_equal(simp_bad[0].vertex_count(), 0)

    # 9. (A9) studio/gizmo.mojo _emit_thick_line_ui NaN guard
    var giz_dl = UiDrawList()
    _emit_thick_line_ui(giz_dl, nan_f32, Float32(0.0), Float32(10.0), Float32(10.0), Float32(3.0), 0xFFFFFFFF)
    assert_equal(len(giz_dl.prims), 0)

    # 10. (B1-B2) physics/character.mojo move_and_slide dt=NaN & dt=1e38
    var pw_kcc = PhysicsWorld()
    var kcc_r18 = CharacterController()
    var kcc_p1 = kcc_r18.move_and_slide(pw_kcc, Vec3(1.0, 0.0, 0.0), nan_f32)
    assert_true(isfinite(kcc_p1.x) and isfinite(kcc_p1.y) and isfinite(kcc_p1.z))
    var kcc_p2 = kcc_r18.move_and_slide(pw_kcc, Vec3(1.0, 0.0, 0.0), Float32(1.0e38))
    assert_true(isfinite(kcc_p2.x) and isfinite(kcc_p2.y) and isfinite(kcc_p2.z))
    assert_true(isfinite(kcc_r18.velocity.x) and isfinite(kcc_r18.vertical_vel))

    # 11. (B3) physics/vehicle.mojo step_world_substep / substep_suspension NaN, 1e30, 1e35 & 3.4e38 drag
    var veh_r18 = VehicleChassis(1450.0)
    veh_r18.setup_four_wheel_car()
    veh_r18.linear_vel = Vec3(Float32(1.0e30), 0.0, 0.0)
    var w_forces = List[Vec3]()
    veh_r18.step_world_substep(nan_f32, nan_f32, nan_f32, nan_f32, Float32(9.81), List[Vec3](), w_forces)
    assert_true(isfinite(veh_r18.linear_vel.x) and isfinite(veh_r18.pos.x))
    veh_r18.step_world_substep(Float32(1.0 / 60.0), Float32(0.5), Float32(0.0), nan_f32, Float32(9.81), List[Vec3](), w_forces)
    assert_true(isfinite(veh_r18.linear_vel.x) and isfinite(veh_r18.pos.x))
    veh_r18.linear_vel = Vec3(Float32(1.0e35), 0.0, 0.0)
    veh_r18.step_world_substep(Float32(1.0 / 60.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(9.81), List[Vec3](), w_forces)
    assert_true(isfinite(veh_r18.linear_vel.x) and isfinite(veh_r18.linear_vel.y) and isfinite(veh_r18.pos.x))
    veh_r18.linear_vel = Vec3(Float32(3.4e38), 0.0, 0.0)
    veh_r18.angular_vel = Vec3(Float32(3.4e38), Float32(1.0e38), 0.0)
    veh_r18.step_world_substep(Float32(1.0 / 60.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(9.81), List[Vec3](), w_forces, apply_gyro=True)
    assert_true(
        isfinite(veh_r18.linear_vel.x) and isfinite(veh_r18.linear_vel.y) and isfinite(veh_r18.linear_vel.z)
        and isfinite(veh_r18.pos.x) and isfinite(veh_r18.pos.y) and isfinite(veh_r18.pos.z)
        and isfinite(veh_r18.angular_vel.x) and isfinite(veh_r18.rot.x) and isfinite(veh_r18.rot.w)
    )
    veh_r18.substep_suspension(Float32(1.0 / 60.0), nan_f32, nan_f32, nan_f32)
    assert_true(isfinite(veh_r18.linear_vel.x) and isfinite(veh_r18.pos.x))

    # 12. (B4) physics/xpbd.mojo step_substep_coupled h=1e38
    var xp_atts = List[RigidAttachment]()
    var xp_bodies = SolverBodies()
    var xp_chain = ParticleChain.line(Vec3(0.0, 2.0, 0.0), Vec3(1.0, 2.0, 0.0), 4, 0.1)
    xp_chain.pin(0, Vec3(0.0, 2.0, 0.0))
    xp_chain.step_substep_coupled(Float32(1.0e38), Float32(0.0), Float32(9.81), List[Vec3](), List[ChainCollider](), xp_atts, xp_bodies)
    for pi in range(xp_chain.count()):
        assert_true(isfinite(xp_chain.position[pi].x) and isfinite(xp_chain.position[pi].y) and isfinite(xp_chain.position[pi].z))
    var xp_mesh = ParticleMesh()
    _ = xp_mesh.add_particle(Vec3(0.0, 2.0, 0.0), 0.5)
    _ = xp_mesh.add_particle(Vec3(1.0, 2.0, 0.0), 0.5)
    xp_mesh.add_link(0, 1)
    xp_mesh.step_substep_coupled(Float32(1.0e38), Float32(0.0), Float32(9.81), List[Vec3](), List[ChainCollider](), xp_atts, xp_bodies)
    for pi in range(xp_mesh.count()):
        assert_true(isfinite(xp_mesh.position[pi].x) and isfinite(xp_mesh.position[pi].y) and isfinite(xp_mesh.position[pi].z))

    # 13. (B5) physics/water.mojo eval_sphere_buoyancy_drag & eval_box_buoyancy_drag 1e30 drag
    var bw_sph = eval_sphere_buoyancy_drag(Vec3(0.0, -0.5, 0.0), Float32(0.5), Vec3(Float32(1.0e30), 0.0, 0.0), Vec3(), Float32(0.0))
    assert_true(isfinite(bw_sph.force.x) and isfinite(bw_sph.force.y) and isfinite(bw_sph.force.z))
    var bw_box = eval_box_buoyancy_drag(Vec3(0.0, -0.5, 0.0), Quat(), Vec3(0.5, 0.5, 0.5), Vec3(Float32(1.0e30), 0.0, 0.0), Vec3(), Float32(0.0))
    assert_true(isfinite(bw_box.force.x) and isfinite(bw_box.force.y) and isfinite(bw_box.force.z))

    # 14. (B6) physics/tires.mojo PacejkaTire.calc_forces_bristle dt=NaN & 1e30
    var tire = PacejkaTire()
    var bx = Float32(0.0)
    var by = Float32(0.0)
    var tf_nan = tire.calc_forces_bristle(Float32(1.0e30), Float32(0.0), Float32(0.0), Float32(1.0e30), nan_f32, bx, by)
    assert_true(isfinite(tf_nan[0]) and isfinite(tf_nan[1]) and isfinite(tf_nan[2]))
    var tf_ext = tire.calc_forces_bristle(Float32(1.0e30), Float32(0.0), Float32(0.0), Float32(1.0e30), Float32(0.016), bx, by)
    assert_true(isfinite(tf_ext[0]) and isfinite(tf_ext[1]) and isfinite(tf_ext[2]))

    # 15. (B7) anim/motion_matching.mojo NaN & 1e30 feature normalization
    var mdb = MotionDatabase()
    var mf_bad = MotionFeature()
    mf_bad.traj_pos_200 = Vec3(nan_f32, Float32(1.0e30), Float32(0.0))
    mdb.add_frame(0, 0, Float32(0.0), mf_bad)
    mdb.add_frame(0, 1, Float32(0.1), MotionFeature())
    mdb.finalize_and_normalize()
    for d in range(len(mdb.means)):
        assert_true(isfinite(mdb.means[d]) and isfinite(mdb.stds[d]) and isfinite(mdb.scales[d]))

    # 16. (B8) render/render_world.mojo DebugLineBatch.emit_to_ui_draw_list NaN camera/coords
    var dlb = DebugLineBatch()
    dlb.add_line(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 0xFFFFFFFF)
    var dlb_ui = UiDrawList()
    var dlb_emitted = dlb.emit_to_ui_draw_list(dlb_ui, Vec3(nan_f32, 0.0, 5.0), Vec3(0.0, 0.0, 0.0), Float32(60.0), Float32(0.1), 1280, 720)
    assert_equal(dlb_emitted, 0)
    assert_equal(len(dlb_ui.prims), 0)

    # 17. (B9) render/lighting.mojo LightingProfile.fog_rgb NaN/1e38/-1e38
    var lprof = LightingProfile()
    assert_equal(Int(lprof.fog_rgb()[0]), 14)
    lprof.fog_color = Vec3(nan_f32, Float32(1.0e38), Float32(-1.0e38))
    var frgb = lprof.fog_rgb()
    assert_equal(Int(frgb[0]), 0)
    assert_equal(Int(frgb[1]), 255)
    assert_equal(Int(frgb[2]), 0)

    # 18. (B10) procedural/loft_bevel.mojo extrude_profile_beveled & sweep_profile_rmf_banked NaN/extreme guards
    var sq_poly: List[Vec2] = [Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(1.0, 1.0), Vec2(0.0, 1.0)]
    var loft_nan = extrude_profile_beveled(sq_poly, nan_f32)
    assert_equal(loft_nan.vertex_count(), 0)
    var loft_ok = extrude_profile_beveled(sq_poly, Float32(1.0))
    assert_true(loft_ok.vertex_count() > 0)
    var sweep_path: List[Vec3] = [Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 2.0)]
    var sweep_bad_prof: List[Vec2] = [Vec2(nan_f32, 0.0), Vec2(1.0, 0.0), Vec2(1.0, 1.0)]
    var sweep_nan_p = sweep_profile_rmf_banked(sweep_path, sweep_bad_prof, List[Float32]())
    assert_equal(sweep_nan_p.vertex_count(), 0)
    var sweep_bad_path: List[Vec3] = [Vec3(0.0, 0.0, 0.0), Vec3(0.0, nan_f32, 2.0)]
    var sweep_nan_path = sweep_profile_rmf_banked(sweep_bad_path, sq_poly, List[Float32]())
    assert_equal(sweep_nan_path.vertex_count(), 0)
    var sweep_ok = sweep_profile_rmf_banked(sweep_path, sq_poly, [nan_f32, Float32(0.1)])
    assert_true(sweep_ok.vertex_count() > 0)
    for k in range(len(sweep_ok.positions)):
        assert_true(isfinite(sweep_ok.positions[k]))

    # 19. (B11) ml/ops.mojo softmax(+Inf) & layer_norm(eps < 0 / NaN / 3.4e38)
    var sm_inf = softmax(Tensor.floats([2], [pinf_f32, Float32(1.0)]), 0)
    assert_true(abs(sm_inf.fget(0) - Float32(1.0)) < Float32(1.0e-5))
    assert_true(abs(sm_inf.fget(1) - Float32(0.0)) < Float32(1.0e-5))
    var sm_two_inf = softmax(Tensor.floats([3], [pinf_f32, pinf_f32, Float32(1.0)]), 0)
    assert_true(abs(sm_two_inf.fget(0) - Float32(0.5)) < Float32(1.0e-5))
    assert_true(abs(sm_two_inf.fget(1) - Float32(0.5)) < Float32(1.0e-5))
    assert_true(abs(sm_two_inf.fget(2) - Float32(0.0)) < Float32(1.0e-5))
    var sm_all_ninf = softmax(Tensor.floats([2], [ninf_f32, ninf_f32]), 0)
    assert_true(abs(sm_all_ninf.fget(0) - Float32(0.5)) < Float32(1.0e-5))

    var ln_neg_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [1.0, 2.0]), Tensor.floats([2], [1.0, 1.0]), Tensor(), False, Float32(-1.0))
    except:
        ln_neg_raised = True
    assert_true(ln_neg_raised)

    var ln_nan_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [1.0, 2.0]), Tensor.floats([2], [1.0, 1.0]), Tensor(), False, nan_f32)
    except:
        ln_nan_raised = True
    assert_true(ln_nan_raised)

    var ln_in_nan_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [nan_f32, 1.0]), Tensor.floats([2], [1.0, 1.0]), Tensor(), False, Float32(1.0e-5))
    except:
        ln_in_nan_raised = True
    assert_true(ln_in_nan_raised)

    var ln_inf_raised = False
    try:
        _ = layer_norm(Tensor.floats([2], [pinf_f32, 1.0]), Tensor.floats([2], [1.0, 1.0]), Tensor(), False, Float32(1.0e-5))
    except:
        ln_inf_raised = True
    assert_true(ln_inf_raised)

    var ln_ext = layer_norm(
        Tensor.floats([2], [Float32(3.4e38), Float32(3.4e38)]),
        Tensor.floats([2], [Float32(1.0), Float32(1.0)]),
        Tensor.floats([2], [Float32(0.0), Float32(0.0)]),
        True,
        Float32(1.0e-5),
    )
    assert_true(isfinite(ln_ext.fget(0)) and isfinite(ln_ext.fget(1)))
    assert_true(abs(ln_ext.fget(0)) < Float32(1.0e-5) and abs(ln_ext.fget(1)) < Float32(1.0e-5))

    # 20. (B-Note) anim/animator.mojo negative dt=-5.0 must not rewind state_time negative
    var anim_m = _make_test_model(1.0, 1.0)
    var ag = AnimGraphDef()
    var st_a = StateDef()
    st_a.name = "walk"
    st_a.clip = 0
    ag.states.append(st_a^)
    var anim_r18 = Animator()
    anim_r18.configure(ag, [ClipInfo("walk", Float32(1.0), True, Float32(1.0))])
    anim_r18.update(anim_m, 1, Float32(0.25))
    anim_r18.update(anim_m, 1, Float32(-5.0))
    assert_true(anim_r18.state_time >= Float32(0.0))
    assert_true(anim_r18.clip_time >= Float32(0.0))


def test_r19_partition_ab_fixes() raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var pinf_f32 = f32_from_bits(0x7F800000)
    var ninf_f32 = f32_from_bits(0xFF800000)

    # 1. UiContext.drag_float3 writes back sanitized finite values even when idle (not dragged)
    var uictx = UiContext()
    var inp_ui = InputState()
    uictx.begin_frame(inp_ui, Float32(1280.0), Float32(720.0))
    var dv = Vec3(nan_f32, pinf_f32, ninf_f32)
    var drag_changed = uictx.drag_float3("drag_r19", "Pos", dv)
    uictx.end_frame()
    assert_false(drag_changed)
    assert_true(isfinite(dv.x) and isfinite(dv.y) and isfinite(dv.z))

    # 2. UiDrawList.push_clip clamps to [0, 65535] without mod-65536 wraparound & handles inverted bounds
    var dl = UiDrawList()
    dl.push_clip(65536, 65537, 70000, 70001)
    var c_hi = dl.current_clip()
    assert_equal(Int(c_hi[0]), 65535)
    assert_equal(Int(c_hi[1]), 65535)
    assert_equal(Int(c_hi[2]), 65535)
    assert_equal(Int(c_hi[3]), 65535)
    dl.pop_clip()
    dl.push_clip(200, 300, 100, 150)
    var c_inv = dl.current_clip()
    assert_true(c_inv[0] <= c_inv[2] and c_inv[1] <= c_inv[3])
    dl.pop_clip()

    # 3. DockSpace.interact_splitters ignores NaN/Inf mouse coords instead of snapping ratio to 0.92
    var ds = DockSpace()
    assert_true(
        ds.load_json(
            '{"root":0,"header_height":24,"nodes":['
            '{"id":0,"kind":0,"ratio":0.5,"child_a":1,"child_b":2,"active_tab":0,"tabs":[]},'
            '{"id":1,"kind":2,"ratio":0.5,"child_a":-1,"child_b":-1,"active_tab":0,"tabs":["L"]},'
            '{"id":2,"kind":2,"ratio":0.5,"child_a":-1,"child_b":-1,"active_tab":0,"tabs":["R"]}'
            ']}'
        )
    )
    var sp_id = 0
    var root_rect = RectF(0.0, 0.0, 1000.0, 600.0)
    ds.solve_layout(root_rect)
    assert_true(ds.interact_splitters(Float32(500.0), Float32(300.0), True, True, root_rect))
    assert_false(ds.interact_splitters(nan_f32, Float32(300.0), True, False, root_rect))
    assert_true(abs(ds.nodes[sp_id].ratio - Float32(0.5)) < Float32(1.0e-5))
    assert_false(ds.interact_splitters(pinf_f32, Float32(300.0), True, False, root_rect))
    assert_true(abs(ds.nodes[sp_id].ratio - Float32(0.5)) < Float32(1.0e-5))

    # 4. evdev.pack_ff_rumble_effect guards NaN/Inf motor levels -> 0 (not 65535)
    var ff_nan = pack_ff_rumble_effect(-1, nan_f32, pinf_f32, 250, 0)
    var strong_nan = Int(ff_nan[16]) | (Int(ff_nan[17]) << 8)
    var weak_inf = Int(ff_nan[18]) | (Int(ff_nan[19]) << 8)
    assert_equal(strong_nan, 0)
    assert_equal(weak_inf, 0)
    var ff_one = pack_ff_rumble_effect(-1, Float32(1.0), Float32(0.5), 250, 0)
    assert_equal(Int(ff_one[16]) | (Int(ff_one[17]) << 8), 65535)

    # 5. MacOsAudio / Win32Audio / dsp NaN/Inf guards
    var mac_aud = MacOsAudio()
    mac_aud.trigger(0, nan_f32, nan_f32)
    mac_aud.set_ambient_drone(nan_f32)
    mac_aud.mix_into_slot(0, nan_f32, 64)
    assert_true(mac_aud.voices[0].freq_hz < 300.0)
    assert_true(mac_aud.voices[0].gain == Float32(0.0))
    assert_true(mac_aud.drone_level == Float32(0.0))

    var win_aud = Win32Audio()
    win_aud.trigger(0, nan_f32, nan_f32)
    win_aud.set_ambient_drone(nan_f32)
    win_aud.mix_into_slot(0, nan_f32, 64)
    assert_true(win_aud.voices[0].freq_hz < 300.0)
    assert_true(win_aud.voices[0].gain == Float32(0.0))
    assert_true(win_aud.drone_level == Float32(0.0))

    assert_true(soft_clip_tanh(nan_f32) == Float32(0.0))
    var res = ModalResonator(440.0, 0.15, 1.0, 48000.0)
    res.configure(nan_f32, nan_f32, nan_f32, nan_f32)
    assert_true(isfinite(res.step(nan_f32)))
    var chorus = StereoChorus()
    var ch_out = chorus.process(Float32(0.5), Float32(-0.5), nan_f32)
    assert_true(isfinite(ch_out[0]) and isfinite(ch_out[1]))

    # 6. glb_native read_vec3/read_quat 1e40 overflow guard & glb_writer NaN JSON guard
    var gltf_json = String(
        '{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"translation":[1e40,0,0],"rotation":[1e40,0,0,1]}]}'
    )
    var jb = List[UInt8]()
    for b in gltf_json.as_bytes():
        jb.append(b)
    while len(jb) % 4 != 0:
        jb.append(UInt8(0x20))
    var glb_ovf = List[UInt8]()
    var tot_glb = 12 + 8 + len(jb) + 8
    glb_ovf.append(UInt8(0x67)); glb_ovf.append(UInt8(0x6C)); glb_ovf.append(UInt8(0x54)); glb_ovf.append(UInt8(0x46))
    glb_ovf.append(UInt8(2)); glb_ovf.append(UInt8(0)); glb_ovf.append(UInt8(0)); glb_ovf.append(UInt8(0))
    glb_ovf.append(UInt8(tot_glb & 255)); glb_ovf.append(UInt8((tot_glb >> 8) & 255)); glb_ovf.append(UInt8((tot_glb >> 16) & 255)); glb_ovf.append(UInt8((tot_glb >> 24) & 255))
    glb_ovf.append(UInt8(len(jb) & 255)); glb_ovf.append(UInt8((len(jb) >> 8) & 255)); glb_ovf.append(UInt8((len(jb) >> 16) & 255)); glb_ovf.append(UInt8((len(jb) >> 24) & 255))
    glb_ovf.append(UInt8(0x4A)); glb_ovf.append(UInt8(0x53)); glb_ovf.append(UInt8(0x4F)); glb_ovf.append(UInt8(0x4E))
    for b in jb:
        glb_ovf.append(b)
    glb_ovf.append(UInt8(0)); glb_ovf.append(UInt8(0)); glb_ovf.append(UInt8(0)); glb_ovf.append(UInt8(0))
    glb_ovf.append(UInt8(0x42)); glb_ovf.append(UInt8(0x49)); glb_ovf.append(UInt8(0x4E)); glb_ovf.append(UInt8(0))
    var mdl_ovf = parse_native_glb(glb_ovf)
    assert_equal(len(mdl_ovf.nodes), 1)
    assert_true(isfinite(mdl_ovf.nodes[0].translation.x))
    assert_true(isfinite(mdl_ovf.nodes[0].rotation.x) and isfinite(mdl_ovf.nodes[0].rotation.w))

    var nm_nan = NativeModel()
    var mesh_nan = NativeMesh()
    mesh_nan.positions = [nan_f32, Float32(0.0), Float32(0.0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(1.0), Float32(0.0)]
    mesh_nan.normals = [Float32(0.0), Float32(0.0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(1.0)]
    mesh_nan.texcoords = [Float32(0.0), Float32(0.0), Float32(1.0), Float32(0.0), Float32(0.0), Float32(1.0)]
    mesh_nan.indices = [0, 1, 2]
    nm_nan.meshes.append(mesh_nan^)
    var glb_nan_bytes = encode_native_glb(nm_nan)
    var json_len_w = Int(glb_nan_bytes[12]) | (Int(glb_nan_bytes[13]) << 8) | (Int(glb_nan_bytes[14]) << 16) | (Int(glb_nan_bytes[15]) << 24)
    var json_slice = List[UInt8]()
    for i in range(20, 20 + json_len_w):
        json_slice.append(glb_nan_bytes[i])
    var json_doc_w = parse_json(json_slice)
    assert_true(json_doc_w.valid, "encode_native_glb must emit valid JSON even on NaN positions")

    # 7. AnimationPlayback.update & advance NaN/huge dt guards
    var apb = AnimationPlayback()
    var f_nan = apb.update(0, nan_f32, 10, True)
    assert_true(f_nan >= 0 and f_nan < 10)
    var f_huge = apb.update(0, Float32(1.0e38), 10, True)
    assert_true(f_huge >= 0 and f_huge < 10)
    var t_huge = apb.advance(0, Float32(1.0e38), Float32(1.0), True)
    assert_true(isfinite(t_huge) and t_huge >= Float32(0.0) and t_huge < Float32(1.0))

    # 8. motion_tools.nearest_frame & ide_view.raster_ui NaN/huge guards
    assert_equal(nearest_frame(nan_f32), 0)
    assert_equal(nearest_frame(Float32(-5.0)), 0)
    assert_equal(nearest_frame(Float32(1.0e38)), 1000000000)

    var ui_r19 = UiDrawList()
    var bad_prim = GpuUiPrim()
    bad_prim.header = PRIM_SHAPE
    bad_prim.x0 = nan_f32
    bad_prim.y0 = Float32(0.0)
    bad_prim.x1 = Float32(8.0)
    bad_prim.y1 = Float32(8.0)
    bad_prim.fill_rgba8 = 0x112233FF
    ui_r19.prims.append(bad_prim)
    var px_ui = List[UInt32]()
    var fnt_ui = SdfFont()
    raster_ui(px_ui, fnt_ui, ui_r19, 8, 8)
    assert_equal(len(px_ui), 64)
    assert_equal(px_ui[0], UInt32(0))

    # 9. encode_png / encode_exr_fp16 / encode_radiance_hdr reject non-positive dimensions
    var px_img = List[UInt32](length=4, fill=UInt32(0xFFFFFFFF))
    assert_equal(len(encode_png(px_img, 0, 0)), 0)
    assert_equal(len(encode_png(px_img, -1, 1)), 0)
    var px_hdr = List[Vec3](length=4, fill=Vec3(1.0, 1.0, 1.0))
    assert_equal(len(encode_exr_fp16(px_hdr, 0, 0)), 0)
    assert_equal(len(encode_exr_fp16(px_hdr, -1, 1)), 0)
    assert_equal(len(encode_radiance_hdr(px_hdr, 0, 0)), 0)

    # 10. ArchetypeSlots.read_blob rejects inconsistent pending/live_count vs row states
    var bw_ecs1 = BlobWriter()
    bw_ecs1.integer(FLINT_ECS_VERSION)
    bw_ecs1.integer(0)  # arch
    bw_ecs1.integer(8)  # cap
    bw_ecs1.integer(1)  # used
    bw_ecs1.integer(0)  # pending = 0 (invalid for ROW_SPAWNING)
    bw_ecs1.integer(0)  # live_count
    bw_ecs1.u8(UInt8(1))  # ROW_SPAWNING
    bw_ecs1.u32(UInt32(1))
    var slots_bad1 = ArchetypeSlots(0, 8)
    var r_ecs1 = BlobReader(bw_ecs1.data.copy())
    assert_false(slots_bad1.read_blob(r_ecs1))

    var bw_ecs2 = BlobWriter()
    bw_ecs2.integer(FLINT_ECS_VERSION)
    bw_ecs2.integer(0)
    bw_ecs2.integer(8)
    bw_ecs2.integer(1)   # used
    bw_ecs2.integer(0)   # pending
    bw_ecs2.integer(99)  # live_count > used
    bw_ecs2.u8(ROW_LIVE)
    bw_ecs2.u32(UInt32(1))
    var slots_bad2 = ArchetypeSlots(0, 8)
    var r_ecs2 = BlobReader(bw_ecs2.data.copy())
    assert_false(slots_bad2.read_blob(r_ecs2))

    # 11. replay_stream.evaluate_camera_spline(NaN) returns (False, ...) safely
    var rs = FlintReplayStream()
    rs.add_camera_knot(0, Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0))
    rs.add_camera_knot(10, Vec3(10.0, 1.0, 0.0), Vec3(0.0, 0.0, 0.0))
    var sp_nan = rs.evaluate_camera_spline(nan_f32)
    assert_false(sp_nan[0])
    assert_true(isfinite(sp_nan[1].eye.x) and isfinite(sp_nan[1].fov_y))

    # 12. vehicle.mojo Float64 -> Float32 narrowing clamp in _probe_flat_ground / _probe_bvh_ground
    var floor_bvh = Bvh8MeshCollider.build(
        [
            Vec3(-50.0, 0.0, -50.0), Vec3(50.0, 0.0, -50.0), Vec3(50.0, 0.0, 50.0),
            Vec3(-50.0, 0.0, -50.0), Vec3(50.0, 0.0, 50.0), Vec3(-50.0, 0.0, 50.0),
        ],
        [Int32(0), Int32(1), Int32(2), Int32(3), Int32(4), Int32(5)],
    )
    var vc1 = VehicleChassis()
    vc1.setup_four_wheel_car()
    vc1.pos = Vec3(0.0, 0.4, 0.0)
    vc1.linear_vel = Vec3(0.0, Float32(3.4e38), 0.0)
    vc1.angular_vel = Vec3(Float32(-3.4e38), 0.0, 0.0)
    vc1.substep_suspension(Float32(1.0 / 240.0))
    assert_true(isfinite(vc1.wheel_states[0].compression_rate))

    var vc2 = VehicleChassis()
    vc2.setup_four_wheel_car()
    vc2.pos = Vec3(0.0, 0.4, 0.0)
    vc2.linear_vel = Vec3(0.0, Float32(3.4e38), 0.0)
    vc2.angular_vel = Vec3(Float32(-3.4e38), 0.0, 0.0)
    vc2.substep_suspension_bvh(Float32(1.0 / 240.0), floor_bvh)
    assert_true(isfinite(vc2.wheel_states[0].compression_rate))

    # 13. AaaRenderGraph.compile non-256-multiple alignment slack prevents heap overlap
    var rg_al = AaaRenderGraph()
    var r_a = rg_al.add_resource("r0", 512, True, False, 256)
    var r_b = rg_al.add_resource("r1", 512, True, False, 511)
    var r_c = rg_al.add_resource("r2", 512, True, False, 256)
    var r_d = rg_al.add_resource("r3", 512, True, False, 513)
    var r_e = rg_al.add_resource("r4", 512, True, False, 768)
    _ = rg_al.add_pass("p0", [
        RgResourceAccess(r_a, RG_USAGE_COMPUTE_WRITE),
        RgResourceAccess(r_b, RG_USAGE_COMPUTE_WRITE),
        RgResourceAccess(r_c, RG_USAGE_COMPUTE_WRITE),
        RgResourceAccess(r_d, RG_USAGE_COMPUTE_WRITE),
        RgResourceAccess(r_e, RG_USAGE_COMPUTE_WRITE),
    ])
    _ = rg_al.add_pass("p1", [
        RgResourceAccess(r_a, RG_USAGE_COMPUTE_READ),
        RgResourceAccess(r_b, RG_USAGE_COMPUTE_READ),
        RgResourceAccess(r_c, RG_USAGE_COMPUTE_READ),
        RgResourceAccess(r_d, RG_USAGE_COMPUTE_READ),
        RgResourceAccess(r_e, RG_USAGE_COMPUTE_READ),
    ])
    assert_true(rg_al.compile(1024 * 1024))
    for i in range(5):
        var si = rg_al.resources[i].aliased_offset_bytes
        var ei = si + rg_al.resources[i].size_bytes
        assert_equal(si % rg_al.resources[i].alignment_bytes, 0)
        for j in range(i + 1, 5):
            var sj = rg_al.resources[j].aliased_offset_bytes
            var ej = sj + rg_al.resources[j].size_bytes
            assert_true(ei <= sj or ej <= si, "Simultaneously live transient resources must never overlap")

    # 14. audio/acoustics probe_room_sabine_rt60 max_probe_dist=1e20 & NaN finite volume guard
    var mats = AcousticMaterialTable()
    var rp_big = probe_room_sabine_rt60(floor_bvh, mats, Vec3(0.0, 1.0, 0.0), Float32(1.0e20))
    assert_true(isfinite(rp_big.estimated_volume_m3) and isfinite(rp_big.sabine_rt60_sec) and isfinite(rp_big.fdn_feedback))
    var rp_nan = probe_room_sabine_rt60(floor_bvh, mats, Vec3(0.0, 1.0, 0.0), nan_f32)
    assert_true(isfinite(rp_nan.estimated_volume_m3) and isfinite(rp_nan.sabine_rt60_sec))

    # 15. ArticulatedRagdoll.apply_impulse_at_bone & step NaN/1e38 guards
    var rag_r19 = ArticulatedRagdoll()
    _ = rag_r19.add_limb(ArticulatedLimbDef(bone_node=0, parent_limb=-1, mass=10.0, radius=0.15, length=0.35))
    rag_r19.apply_impulse_at_bone(0, Vec3(Float32(1.0e38), Float32(-1.0e38), nan_f32), Vec3(1.0, 2.0, 3.0))
    rag_r19.step(Float32(0.016))
    assert_true(isfinite(rag_r19.states[0].com_pos.x) and isfinite(rag_r19.states[0].com_pos.y) and isfinite(rag_r19.states[0].com_pos.z))
    assert_true(isfinite(rag_r19.states[0].rotation.x) and isfinite(rag_r19.states[0].rotation.y) and isfinite(rag_r19.states[0].rotation.z) and isfinite(rag_r19.states[0].rotation.w))


def test_r20_partition_ab_fixes() raises:
    var nan_f32 = f32_from_bits(0x7FC00000)
    var inf_f32 = f32_from_bits(0x7F800000)
    var ninf_f32 = f32_from_bits(0xFF800000)
    var nan_f64 = Float64(nan_f32)
    var inf_f64 = Float64(inf_f32)

    # 1. NineSlice empty / zero-dim / OOB bounds
    var ns_empty = NineSlice(0, 0, 0, 0, 0, 0)
    assert_equal(ns_empty.at(0, 0), UInt32(0))
    assert_equal(ns_empty.at(-3, 5), UInt32(0))
    ns_empty.set(0, 0, UInt32(0xFFFFFFFF))
    var ns_neg = NineSlice(-4, -2, 1, 1, 1, 1)
    assert_equal(ns_neg.at(0, 0), UInt32(0))
    var ns_ok = NineSlice(4, 4, 1, 1, 1, 1)
    ns_ok.set(3, 2, UInt32(0x11223344))
    assert_equal(ns_ok.at(10, 2), UInt32(0x11223344))
    var sm_bad = slice_map(4, 0, 1, 1, nan_f32)
    assert_equal(len(sm_bad), 4)

    # 2. NativeModel.sample_clip OOB index & non-finite time
    var mdl_empty = NativeModel()
    mdl_empty.sample_clip(-1, Float32(0.0))
    mdl_empty.sample_clip(0, nan_f32)
    assert_equal(mdl_empty.active_animation, -1)
    assert_equal(mdl_empty.active_time, Float32(0.0))
    var mdl_one = _make_test_model(Float32(1.0), Float32(2.0))
    mdl_one.sample_clip(0, nan_f32)
    assert_true(isfinite(mdl_one.active_time))
    assert_true(isfinite(mdl_one.nodes[0].translation.x) and isfinite(mdl_one.nodes[0].translation.y))
    mdl_one.animate(0, inf_f32, inf_f32, nan_f32)
    assert_true(isfinite(mdl_one.nodes[0].translation.y))

    # 3. NativeAudio process_commands, set_bus_volumes, play_voice, mix_stereo NaN/Inf guards
    var aud = NativeAudio()
    var clip = WavClip()
    clip.valid = True
    clip.channels = 1
    clip.sample_rate = 48000
    clip.samples = List[Int16](length=64, fill=Int16(8000))
    aud.clips.append(clip^)
    _ = aud.send_command(AudioCommand(AUDIO_CMD_SET_BUS, bus=0, volume=nan_f32))
    aud.process_commands()
    assert_true(isfinite(aud.bus_volume[0]))
    aud.set_bus_volumes(nan_f32, inf_f32, ninf_f32)
    assert_true(aud.bus_volume[0] == Float32(0.0) and aud.bus_volume[1] == Float32(0.0) and aud.bus_volume[2] == Float32(0.0))
    aud.set_bus_volumes(Float32(1.0), Float32(1.0), Float32(1.0))
    aud.play_voice(0, pitch=nan_f32, volume=nan_f32, priority=1, bank=-1, limit=0, pan=nan_f32, lpf_coeff=nan_f32, reverb_send=nan_f32, elevation=nan_f32)
    assert_true(isfinite(aud.voices[0].pitch) and aud.voices[0].pitch == Float32(1.0))
    assert_true(isfinite(aud.voices[0].volume) and aud.voices[0].volume == Float32(0.0))
    assert_true(isfinite(aud.voices[0].pan) and aud.voices[0].pan == Float32(0.0))
    var pcm_clean = aud.mix_stereo(4)
    assert_equal(len(pcm_clean), 8)
    for s in pcm_clean:
        assert_equal(s, Int16(0))

    # 4. encode_jpeg_baseline 16-bit SOF0 dimension limits (reject > 65535 & overflow)
    var px_big = List[UInt32](length=65536, fill=UInt32(0xFF808080))
    assert_equal(len(encode_jpeg_baseline(px_big, 65536, 1, 75)), 0)
    assert_equal(len(encode_jpeg_baseline(px_big, 1, 65536, 75)), 0)
    assert_equal(len(encode_jpeg_baseline(px_big, 65535, 65535, 75)), 0)
    var px_ok = List[UInt32](length=64, fill=UInt32(0xFF808080))
    var jpg_ok = encode_jpeg_baseline(px_ok, 8, 8, 75)
    assert_true(len(jpg_ok) > 10)
    var jpg_dec = decode_jpeg(jpg_ok)
    assert_true(jpg_dec.valid and jpg_dec.width == 8 and jpg_dec.height == 8)

    # 5. dsp.mojo StereoReverbFDN, equal_power_gains, CytomicSVF, poly_blep_saw
    var fdn = StereoReverbFDN(nan_f32, nan_f32)
    assert_true(isfinite(fdn.feedback) and isfinite(fdn.damp))
    var w_nan = fdn.process_send(nan_f32, inf_f32)
    assert_true(isfinite(w_nan[0]) and isfinite(w_nan[1]))
    var w_after = fdn.process_send(Float32(0.25), Float32(0.25))
    assert_true(isfinite(w_after[0]) and isfinite(w_after[1]))
    var ep_nan = equal_power_gains(nan_f32)
    assert_true(abs(ep_nan[0] - Float32(0.70710677)) < Float32(1.0e-3))
    assert_true(abs(ep_nan[1] - Float32(0.70710677)) < Float32(1.0e-3))
    var svf = CytomicSVF(nan_f32, nan_f32, nan_f32)
    var svf_y = svf.process(nan_f32)
    assert_true(isfinite(svf_y) and isfinite(svf.lp) and isfinite(svf.hp))
    assert_true(isfinite(poly_blep_saw(nan_f32, Float32(0.01))))
    assert_true(isfinite(poly_blep_saw(Float32(0.5), nan_f32)))

    # 6. core/ik.mojo safe_normalized & fabrik non-finite guards
    var fb = Vec3(0.0, 1.0, 0.0)
    var sn_nan = safe_normalized(Vec3(nan_f32, 1.0, 0.0), fb)
    assert_true(sn_nan.x == fb.x and sn_nan.y == fb.y and sn_nan.z == fb.z)
    var sn_ovf = safe_normalized(Vec3(Float32(1.0e30), Float32(1.0e30), Float32(1.0e30)), fb)
    assert_true(sn_ovf.x == fb.x and sn_ovf.y == fb.y and sn_ovf.z == fb.z)
    var pts: List[Vec3] = [Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0)]
    assert_false(fabrik(pts, Vec3(nan_f32, 0.0, 0.0)))
    assert_true(pts[1].x == Float32(1.0) and pts[2].x == Float32(2.0))
    var scratch = List[Float32]()
    assert_false(fabrik(pts, Vec3(inf_f32, 0.0, 0.0), scratch))
    assert_true(pts[1].x == Float32(1.0) and pts[2].x == Float32(2.0))

    # 7. InputMap.update +Inf dt & NaN mouse_delta
    var imap = InputMap()
    _ = imap.register_action("look", ACTION_AXIS2D)
    var ctx = InputContext("game", 0)
    ctx.bindings.append(InputBinding("look", BIND_MOUSE_DELTA_2D, 0))
    imap.add_context(ctx^)
    imap.push_context("game")
    var inp_bad = InputState()
    inp_bad.delta_x = nan_f32
    inp_bad.delta_y = inf_f32
    imap.update(inp_bad, inf_f32)
    assert_true(isfinite(imap.elapsed_time) and imap.elapsed_time == 0.0)
    var look_v = imap.axis_2d("look")
    assert_true(isfinite(look_v[0]) and isfinite(look_v[1]))
    assert_equal(look_v[0], Float32(0.0))
    assert_equal(look_v[1], Float32(0.0))

    # 8. UiContext.begin_frame & curve_editor non-finite guards
    var uictx = UiContext()
    uictx.begin_frame(InputState(), screen_w=inf_f32, screen_h=Float32(-10.0), dt=inf_f32)
    assert_true(isfinite(uictx.screen_w) and uictx.screen_w >= Float32(1.0))
    assert_true(isfinite(uictx.screen_h) and uictx.screen_h == Float32(1.0))
    assert_true(isfinite(uictx.dt) and isfinite(uictx.ui_time_sec))
    var cx_ctrl = nan_f32
    var cy_ctrl = inf_f32
    _ = uictx.curve_editor("r20_curve", cx_ctrl, cy_ctrl)
    assert_true(isfinite(cx_ctrl) and isfinite(cy_ctrl))
    assert_true(cx_ctrl >= Float32(0.0) and cx_ctrl <= Float32(1.0))
    assert_true(cy_ctrl >= Float32(0.0) and cy_ctrl <= Float32(1.0))
    for i in range(len(uictx.draw_list.prims)):
        ref pr = uictx.draw_list.prims[i]
        assert_true(isfinite(pr.x0) and isfinite(pr.y0) and isfinite(pr.x1) and isfinite(pr.y1))

    # 9. MeshBuilder.add_ellipsoid(0, 0) & bc_compress.f32_to_uf16_bits(NaN/-Inf/+Inf)
    var mb = MeshBuilder()
    mb.add_ellipsoid(Vec3(), Vec3(1.0, 1.0, 1.0), rings=0, segments=0)
    assert_equal(mb.mesh.vertex_count(), 0)
    mb.add_ellipsoid(Vec3(), Vec3(nan_f32, 1.0, 1.0), rings=2, segments=4)
    assert_true(mb.mesh.vertex_count() > 0)
    for p in mb.mesh.positions:
        assert_true(isfinite(p))
    assert_equal(f32_to_uf16_bits(nan_f32), 0)
    assert_equal(f32_to_uf16_bits(ninf_f32), 0)
    assert_equal(f32_to_uf16_bits(inf_f32), 0x7BFF)

    # 10. studio/gizmo.mojo ray_axis_closest_param & snap_scalar
    var r_hit = ray_axis_closest_param(
        ViewportRay(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0)),
        Vec3(nan_f32, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
    )
    assert_false(r_hit[0])
    assert_true(isfinite(r_hit[1]) and isfinite(r_hit[2]))
    var sn_huge = snap_scalar(Float32(1.0e38), Float32(1.0e-5))
    assert_true(isfinite(sn_huge))
    assert_equal(snap_scalar(nan_f32, Float32(0.5)), Float32(0.0))

    # 11. AaaRenderGraph non-power-of-2 alignment < 256 (e.g. 192 -> lcm(256, 192) = 768)
    var rg_192 = AaaRenderGraph()
    var rg_r0 = rg_192.add_resource("r0", 256, True, False, 256)
    var rg_r1 = rg_192.add_resource("r1", 256, True, False, 192)
    assert_equal(rg_192.resources[rg_r1].alignment_bytes, 192)
    _ = rg_192.add_pass("p0", [RgResourceAccess(rg_r0, RG_USAGE_COMPUTE_WRITE), RgResourceAccess(rg_r1, RG_USAGE_COMPUTE_WRITE)])
    _ = rg_192.add_pass("p1", [RgResourceAccess(rg_r0, RG_USAGE_COMPUTE_READ), RgResourceAccess(rg_r1, RG_USAGE_COMPUTE_READ)])
    assert_true(rg_192.compile(1024 * 1024))
    assert_equal(rg_192.resources[rg_r1].aliased_offset_bytes % 192, 0)
    assert_equal(rg_192.resources[rg_r1].aliased_offset_bytes % 256, 0)

    # 12. CameraRig CAM_MODE_FREE_FLY +Inf fly_speed * 0.0 dt
    var pw_cam = PhysicsWorld()
    var rig_fly = CameraRig(CAM_MODE_FREE_FLY)
    rig_fly.update(pw_cam, Vec3(), Float32(0.0), fly_move_local=Vec3(1.0, 0.0, 0.0), fly_speed=inf_f32)
    assert_true(isfinite(rig_fly.world_pos.x) and isfinite(rig_fly.world_pos.y) and isfinite(rig_fly.world_pos.z))
    assert_true(isfinite(rig_fly.fly_pos.x) and isfinite(rig_fly.fly_pos.y) and isfinite(rig_fly.fly_pos.z))

    # 13. physics/wind.mojo to_fx saturation & NaN/Inf guard
    assert_equal(to_fx(nan_f32), Int64(0))
    assert_equal(to_fx(inf_f32), Int64(0))
    assert_equal(to_fx(Float32(1.0e20)), Int64.MAX)
    assert_equal(to_fx(Float32(-1.0e20)), Int64.MIN)
    assert_equal(to_fx(Float32(1.5)), Int64(98304))

    # 14. ml/motionbricks.mojo quat_slerp NaN/Inf guard
    var q_nan_t = quat_slerp(MbQuat.identity(), MbQuat(0.0, 1.0, 0.0, 0.0), nan_f32)
    assert_true(isfinite(q_nan_t.x) and isfinite(q_nan_t.y) and isfinite(q_nan_t.z) and isfinite(q_nan_t.w))
    var q_inf_t = quat_slerp(MbQuat.identity(), MbQuat(0.0, 1.0, 0.0, 0.0), inf_f32)
    assert_true(isfinite(q_inf_t.x) and isfinite(q_inf_t.y) and isfinite(q_inf_t.z) and isfinite(q_inf_t.w))

    # 15. PhysicsWorld.raycast NaN max_dist / origin / dir guard
    var ray_nan = pw_cam.raycast(Vec3(), Vec3(0.0, 0.0, -1.0), nan_f32)
    assert_false(ray_nan.hit)
    assert_equal(ray_nan.distance, Float32(0.0))
    var ray_all_nan = pw_cam.raycast_all(Vec3(), Vec3(0.0, 0.0, -1.0), nan_f32)
    assert_equal(len(ray_all_nan), 0)

    # 16. ai/behavior_tree.mojo UtilityAiScorer & BehaviorTree.tick NaN/Inf guards
    assert_equal(UtilityAiScorer.compensate_dave_mark(nan_f32, 2), Float32(0.0))
    assert_equal(UtilityAiScorer.compensate_geometric_mean(nan_f32, 2), Float32(0.0))
    assert_equal(UtilityAiScorer.compensate_dave_mark(inf_f32, 2), Float32(0.0))
    assert_equal(UtilityAiScorer.compensate_geometric_mean(inf_f32, 2), Float32(0.0))
    var bt = BehaviorTree()
    _ = bt.add_action(1)
    var bb = Blackboard()
    var bt_st = BtExecutionState()
    _ = bt.tick(bb, bt_st, inf_f32)
    assert_true(isfinite(bt_st.time_s) and bt_st.time_s == Float32(0.0))
    _ = bt.tick(bb, bt_st, nan_f32)
    assert_true(isfinite(bt_st.time_s) and bt_st.time_s == Float32(0.0))

    # 17. net/crypto.mojo StatelessCookieAuth.issue_cookie / verify_cookie now_ms guard
    var auth = StatelessCookieAuth()
    var ep = NetEndpoint()
    var c_inf = auth.issue_cookie(ep, inf_f64)
    var c_nan = auth.issue_cookie(ep, nan_f64)
    var c_big = auth.issue_cookie(ep, Float64(1.0e30))
    var c_zero = auth.issue_cookie(ep, Float64(0.0))
    assert_true(c_inf[0] == c_zero[0] and c_nan[0] == c_zero[0])
    assert_true(c_big[0] != c_zero[0] or c_big[1] != c_zero[1])
    assert_true(auth.verify_cookie(ep, Int(c_big.unsafe_ptr()), Float64(1.0e30)))



def test_r21_partition_ab_fixes() raises:
    var nan_f32 = bitcast[DType.float32](UInt32(0x7FC00000))
    var inf_f32 = bitcast[DType.float32](UInt32(0x7F800000))
    var ninf_f32 = bitcast[DType.float32](UInt32(0xFF800000))

    # 1. NativeModel global_rotation / global_position / solve_two_bone OOB & NaN guards
    var empty_m = NativeModel()
    var gr0 = empty_m.global_rotation(0)
    assert_true(isfinite(gr0.x) and isfinite(gr0.y) and isfinite(gr0.z) and isfinite(gr0.w))
    var gp0 = empty_m.global_position(0)
    assert_true(gp0.x == Float32(0.0) and gp0.y == Float32(0.0) and gp0.z == Float32(0.0))
    empty_m.solve_two_bone(0, 0, 0, Vec3(1.0, 2.0, 3.0), Vec3(0.0, 0.0, 1.0))
    empty_m.solve_two_bone(-1, 5, 9, Vec3(1.0, 2.0, 3.0), Vec3(0.0, 0.0, 1.0))
    var arm_m = NativeModel()
    var n0 = NativeNode()
    n0.translation = Vec3(0.0, 0.0, 0.0)
    n0.children.append(1)
    var n1 = NativeNode()
    n1.translation = Vec3(0.0, 1.0, 0.0)
    n1.children.append(2)
    var n2 = NativeNode()
    n2.translation = Vec3(0.0, 1.0, 0.0)
    arm_m.nodes.append(n0^)
    arm_m.nodes.append(n1^)
    arm_m.nodes.append(n2^)
    arm_m.roots.append(0)
    arm_m.solve_two_bone(0, 1, 2, Vec3(nan_f32, 1.0, 0.0), Vec3(0.0, 0.0, 1.0))
    assert_true(isfinite(arm_m.nodes[0].rotation.w) and isfinite(arm_m.nodes[1].rotation.w))

    # 2. PadState.apply_event / mapped_button / set_control negative & OOB guards
    var pad = PadState()
    pad.apply_event(2, -1, 100)
    pad.apply_event(2, 99, 100)
    pad.apply_event(1, -5, 1)
    pad.apply_event(1, 99, 1)
    assert_equal(pad.mapped_button(-1), -1)
    assert_equal(pad.mapped_button(-5), -1)
    pad.apply_event(2, 0, 16384)
    assert_true(pad.axes[0] > Float32(0.2))

    # 3. MeshBuilder.add_lathe_z mismatched station lengths & non-finite guards
    var mb_lathe = MeshBuilder()
    var zs: List[Float32] = [Float32(0.0), Float32(1.0)]
    var rxs: List[Float32] = [Float32(0.5)]
    var rys: List[Float32] = [Float32(0.5)]
    var yos: List[Float32] = [Float32(0.0)]
    mb_lathe.add_lathe_z(zs, rxs, rys, yos)
    assert_equal(mb_lathe.mesh.vertex_count(), 0)
    var rxs_nan: List[Float32] = [nan_f32, Float32(0.5)]
    var rys_ok: List[Float32] = [Float32(0.5), Float32(0.5)]
    var yos_ok: List[Float32] = [Float32(0.0), Float32(0.0)]
    mb_lathe.add_lathe_z(zs, rxs_nan, rys_ok, yos_ok)
    assert_equal(mb_lathe.mesh.vertex_count(), 0)
    var rxs_ok: List[Float32] = [Float32(0.5), Float32(0.5)]
    mb_lathe.add_lathe_z(zs, rxs_ok, rys_ok, yos_ok, segments=6)
    assert_true(mb_lathe.mesh.vertex_count() > 0)

    # 4. pad_label / pad_shape / toolbar_label negative & OOB index guards
    assert_equal(pad_label(-1, False), "")
    assert_equal(pad_label(-99, True), "")
    assert_equal(pad_shape(-1), SHAPE_STICK)
    assert_equal(pad_label(0, False), "A")
    assert_equal(toolbar_label(-1), "")
    assert_equal(toolbar_label(99), "")
    assert_equal(toolbar_label(0), "Open")

    # 5. nine_slice.slice_map huge scale Int overflow guard
    var sm10 = slice_map(8, 4, 1, 1, Float32(10.0))
    var sm38 = slice_map(8, 4, 1, 1, Float32(1.0e38))
    assert_equal(len(sm38), 8)
    for idx in range(8):
        assert_equal(sm38[idx], sm10[idx])
    assert_equal(sm38[0], 0)
    assert_equal(sm38[3], 0)
    assert_equal(sm38[4], 3)
    assert_equal(sm38[7], 3)

    # 6. dsp.mojo TimpaniDrum, ItdDelayLine, compute_spatial_3d NaN/Inf guards
    var timp = TimpaniDrum()
    _ = timp.process(nan_f32)
    assert_true(isfinite(timp.mallet_lp))
    var timp_out = timp.process(Float32(1.0))
    assert_true(isfinite(timp_out) and timp_out != Float32(0.0))

    var itd = ItdDelayLine()
    itd.push(nan_f32)
    itd.push(inf_f32)
    var itd0 = itd.read_delayed(Float32(0.0))
    var itd_nan = itd.read_delayed(nan_f32)
    assert_true(isfinite(itd0) and itd0 == Float32(0.0))
    assert_true(isfinite(itd_nan))

    var sp_nan = compute_spatial_3d(
        Vec3(nan_f32, 0.0, 0.0),
        Vec3(nan_f32, 0.0, 0.0),
        Vec3(nan_f32, 0.0, 0.0),
        Vec3(0.0, 0.0, 10.0),
        Vec3(),
        nan_f32,
        nan_f32,
        nan_f32,
        Vec3(nan_f32, 0.0, 1.0),
    )
    assert_true(
        isfinite(sp_nan.gain)
        and isfinite(sp_nan.pan)
        and isfinite(sp_nan.doppler_pitch)
        and isfinite(sp_nan.lpf_coeff)
        and isfinite(sp_nan.reverb_send)
    )
    assert_equal(sp_nan.gain, Float32(0.0))
    assert_equal(sp_nan.pan, Float32(0.0))
    var sp_r_nan = compute_spatial_3d(
        Vec3(0.0, 0.0, 0.0),
        Vec3(),
        Vec3(nan_f32, 0.0, 0.0),
        Vec3(10.0, 0.0, 0.0),
        Vec3(),
    )
    assert_equal(sp_r_nan.pan, Float32(0.0))

    # 7. core/ik.mojo clamp_cone_direction & solve_four_bar_linkage NaN/Inf guards
    var cone_nan = clamp_cone_direction(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), nan_f32)
    assert_true(isfinite(cone_nan.x) and isfinite(cone_nan.y) and isfinite(cone_nan.z))
    var fb_nan = solve_four_bar_linkage(
        Vec3(0.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0), 1.0, 1.5, 1.5, nan_f32, Vec3(0.0, 0.0, 1.0)
    )
    assert_true(isfinite(fb_nan[0].x) and isfinite(fb_nan[1].x) and isfinite(fb_nan[2]) and isfinite(fb_nan[3]))

    # 8. gizmo ray_plane_hit, gizmo_scale_drag & FlintStudioState configure_entity_npr_cel/stylized_vfx
    var rph = ray_plane_hit(ViewportRay(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0)), Vec3(), Vec3(nan_f32, 0.0, 1.0))
    assert_false(rph[0])
    assert_true(isfinite(rph[1].x) and isfinite(rph[1].y) and isfinite(rph[1].z) and isfinite(rph[2]))
    var gs_inf = gizmo_scale_drag(
        ViewportRay(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0)),
        ViewportRay(Vec3(0.5, 0.0, -5.0), Vec3(0.0, 0.0, 1.0)),
        Vec3(),
        GIZMO_AXIS_X,
        Float32(1.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(1.0, 1.0, 1.0),
        snap_step=inf_f32,
    )
    assert_true(isfinite(gs_inf.x) and isfinite(gs_inf.y) and isfinite(gs_inf.z))
    var gs_uni = gizmo_scale_drag(
        ViewportRay(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0)),
        ViewportRay(Vec3(0.5, 0.0, -5.0), Vec3(0.0, 0.0, 1.0)),
        Vec3(),
        GIZMO_SCALE_UNIFORM,
        Float32(1.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(1.0, 1.0, 1.0),
        snap_step=nan_f32,
    )
    assert_true(isfinite(gs_uni.x) and isfinite(gs_uni.y) and isfinite(gs_uni.z))

    var studio = FlintStudioState(headless=True)
    var eid = studio.spawn_entity("r21_cel")
    assert_true(
        studio.configure_entity_npr_cel(
            eid,
            cel_shadow_tint=Vec3(nan_f32, 0.3, 0.4),
            cel_outline_width_px=nan_f32,
            smear_stretch=nan_f32,
        )
    )
    assert_equal(studio.entities[eid].cel_outline_width_px, Float32(2.0))
    assert_equal(studio.entities[eid].smear_stretch, Float32(1.0))
    assert_true(isfinite(studio.entities[eid].cel_shadow_tint.x))
    assert_true(
        studio.configure_entity_stylized_vfx(
            eid,
            vfx_erosion=nan_f32,
            vfx_scroll_speed=inf_f32,
            smear_stretch=nan_f32,
        )
    )
    assert_equal(studio.entities[eid].vfx_erosion, Float32(0.35))
    assert_equal(studio.entities[eid].vfx_scroll_speed, Float32(3.2))
    assert_equal(studio.entities[eid].smear_stretch, Float32(1.0))

    # 9. replay_stream.evaluate_camera_spline NaN knot guard
    var rs = FlintReplayStream()
    rs.add_camera_knot(0, Vec3(nan_f32, 0.0, 0.0), Vec3(0.0, nan_f32, 0.0))
    rs.add_camera_knot(10, Vec3(10.0, 2.0, -5.0), Vec3(0.0, 1.0, 0.0))
    var sp_cam = rs.evaluate_camera_spline(Float32(5.0))
    assert_true(sp_cam[0])
    assert_true(
        isfinite(sp_cam[1].eye.x)
        and isfinite(sp_cam[1].eye.y)
        and isfinite(sp_cam[1].eye.z)
        and isfinite(sp_cam[1].target.x)
        and isfinite(sp_cam[1].target.y)
        and isfinite(sp_cam[1].target.z)
    )
    var sp_cam0 = rs.evaluate_camera_spline(Float32(0.0))
    assert_true(sp_cam0[0] and isfinite(sp_cam0[1].eye.x) and isfinite(sp_cam0[1].target.y))

    # 10. aaa_vsm.VsmPageManager.invalidate_region NaN & +-Inf guards
    var vsm = VsmPageManager()
    vsm.virtual_valid[0] = True
    vsm.invalidate_region(nan_f32, nan_f32, nan_f32, nan_f32)
    assert_true(vsm.virtual_valid[0])
    vsm.invalidate_region(ninf_f32, ninf_f32, inf_f32, inf_f32)
    assert_false(vsm.virtual_valid[0])
    assert_false(vsm.virtual_valid[16383])
    assert_equal(get_vsm_cascade_index(nan_f32), 0)

    # 11. ai/behavior_tree.UtilityConsideration.evaluate NaN Blackboard value guard
    var bb_u = Blackboard()
    var k_hp = bb_u.def_key("hp")
    bb_u.set_float(k_hp, nan_f32)
    var uc = UtilityConsideration(k_hp, Float32(0.0), Float32(100.0))
    assert_equal(uc.evaluate(bb_u), Float32(0.0))
    var uc_inv = UtilityConsideration(k_hp, Float32(0.0), Float32(100.0), invert=True)
    assert_equal(uc_inv.evaluate(bb_u), Float32(0.0))
    var k_ok = bb_u.def_key("ok")
    bb_u.set_float(k_ok, Float32(50.0))
    var scorer = UtilityAiScorer()
    var opt_nan = UtilityOption(1)
    opt_nan.add_consideration(uc)
    var opt_ok = UtilityOption(2)
    opt_ok.add_consideration(UtilityConsideration(k_ok, Float32(0.0), Float32(100.0)))
    _ = scorer.add_option(opt_nan^)
    _ = scorer.add_option(opt_ok^)
    var best = scorer.select_best(bb_u)
    assert_equal(best[0], 2)
    assert_true(abs(best[1] - Float32(0.5)) < Float32(1.0e-4))

    # 12. physics/water.mojo submerged_fraction, buoyant_acceleration, eval_beaufort_wave NaN/Inf guards
    assert_equal(submerged_fraction(nan_f32, Float32(1.0), Float32(0.0)), Float32(0.0))
    assert_equal(submerged_fraction(Float32(0.0), nan_f32, Float32(1.0)), Float32(0.0))
    assert_equal(submerged_fraction(Float32(-1.0e38), Float32(1.0), Float32(1.0e38)), Float32(1.0))
    assert_equal(buoyant_acceleration(nan_f32, Float32(1.0), Float32(9.81)), Float32(0.0))
    var bw_nan = eval_beaufort_wave(nan_f32, Float32(0.0), Float32(0.0))
    assert_true(
        isfinite(bw_nan[0])
        and isfinite(bw_nan[1].x) and isfinite(bw_nan[1].y) and isfinite(bw_nan[1].z)
        and isfinite(bw_nan[2].x) and isfinite(bw_nan[2].y) and isfinite(bw_nan[2].z)
    )
    var bw_inf = eval_beaufort_wave(inf_f32, ninf_f32, inf_f32)
    assert_true(isfinite(bw_inf[0]) and isfinite(bw_inf[1].y) and isfinite(bw_inf[2].x))

    # 13. net/replicator.SnapshotReplicator._quant_pos, _quant_vel & set_entity NaN/Inf guards
    var rep = SnapshotReplicator()
    assert_equal(rep._quant_pos(nan_f32), rep._quant_pos(Float32(0.0)))
    assert_equal(rep._quant_vel(nan_f32), rep._quant_vel(Float32(0.0)))
    assert_equal(rep._quant_pos(inf_f32), rep._quant_pos(Float32(0.0)))
    assert_equal(rep._quant_vel(ninf_f32), rep._quant_vel(Float32(0.0)))
    rep.set_client_view(0, Vec3())
    rep.set_entity(0, Vec3(1.0, 2.0, 3.0), Quat.identity(), Vec3(nan_f32, 0.0, 0.0))
    rep.prepare_tick()
    var pkt = rep.build_client_packet(0)
    var recv = SnapshotClientReceiver()
    var dec = recv.apply_packet(pkt[1])
    assert_true(dec[0])
    assert_true(abs(recv.entities[0].vel.x) < Float32(0.1))


def test_r22_partition_ab_fixes() raises:
    var nan_f32 = bitcast[DType.float32](UInt32(0x7FC00000))
    var inf_f32 = bitcast[DType.float32](UInt32(0x7F800000))
    var nan_f64 = bitcast[DType.float64](UInt64(0x7FF8000000000000))
    var inf_f64 = bitcast[DType.float64](UInt64(0x7FF0000000000000))

    # --- Partition A (9 defects) ---
    # A1: gizmo_rotation_delta_rad non-finite ring_world_radius & ring_axis
    var r_start = ViewportRay(Vec3(0.0, 0.0, -5.0), Vec3(0.0, 0.0, 1.0))
    var r_cur = ViewportRay(Vec3(0.5, 0.0, -5.0), Vec3(0.0, 0.0, 1.0))
    assert_equal(
        gizmo_rotation_delta_rad(r_start, r_cur, Vec3(), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), nan_f32),
        Float32(0.0),
    )
    assert_equal(
        gizmo_rotation_delta_rad(r_start, r_cur, Vec3(), Vec3(nan_f32, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), 1.0),
        Float32(0.0),
    )

    # A2: core/ik.mojo two_bone_ik & solve_two_bone_pole Quat finite validation
    var ik_res = two_bone_ik(
        Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0), Vec3(0.5, 1.5, 0),
        Quat(nan_f32, 0, 0, 1), Quat(), Vec3(0, 0, 1),
    )
    assert_true(isfinite(ik_res.upper.x) and isfinite(ik_res.upper.w))
    assert_true(isfinite(ik_res.middle.x) and isfinite(ik_res.middle.w))
    var pole_res = solve_two_bone_pole(
        Vec3(0, 0, 0), Vec3(0, 1, 0), Vec3(0, 2, 0), Vec3(0.5, 1.5, 0), Vec3(0, 1, 1),
        Quat(nan_f32, 0, 0, 1), Quat(),
    )
    assert_true(isfinite(pole_res[0].x) and isfinite(pole_res[0].w))
    assert_true(isfinite(pole_res[1].x) and isfinite(pole_res[1].w))

    # A3: EngineApp / AppConfig / FixedTime NaN dt/hz/max_frame_dt guards
    var cfg_nan = AppConfig(fixed_hz=nan_f32, default_frame_dt=nan_f32, max_frame_dt=nan_f32)
    assert_true(isfinite(cfg_nan.fixed_hz) and cfg_nan.fixed_hz > Float32(0.0))
    assert_true(isfinite(cfg_nan.default_frame_dt) and cfg_nan.default_frame_dt > Float32(0.0))
    assert_true(isfinite(cfg_nan.max_frame_dt) and cfg_nan.max_frame_dt > Float32(0.0))
    var app_nan = EngineApp(AppConfig(worker_threads=0))
    app_nan.step_one_frame(nan_f32)
    var ft_res = app_nan.resources.get_ptr[FrameTime]().value()[unsafe_offset=0]
    assert_equal(ft_res.delta_seconds, Float32(0.0))
    assert_true(isfinite(ft_res.elapsed_seconds))
    var fx_nan = FixedTime(hz=nan_f32)
    assert_true(isfinite(fx_nan.dt))
    assert_equal(fx_nan.advance_frame(nan_f32), 0)
    assert_true(isfinite(fx_nan.overstep_alpha))
    app_nan.shutdown()

    # A4: interpolate_global_transform NaN alpha -> 0.0 (prev)
    var gt_prev = GlobalTransform.from_local(LocalTransform(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    var gt_curr = GlobalTransform.from_local(LocalTransform(Vec3(10.0, 0.0, 0.0), Quat(), Vec3(2.0, 2.0, 2.0)))
    var gt_interp = interpolate_global_transform(gt_prev, gt_curr, nan_f32)
    assert_equal(gt_interp.translation.x, Float32(0.0))

    # A5: propagate_transforms_parallel_cached bounded HierarchyDepth bucket allocation
    var th_world = World()
    register_transform_components(th_world)
    var th_pool = JobPool()
    var th_cache = TransformPropagationCache()
    var e_par = th_world.spawn_empty()
    var e_ch = th_world.spawn_empty()
    _ = th_world.insert[LocalTransform](e_par, LocalTransform(Vec3(1.0, 2.0, 3.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _ = th_world.insert[LocalTransform](e_ch, LocalTransform(Vec3(4.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    _ = th_world.insert[Parent](e_ch, Parent(e_par))
    _ = th_world.insert[HierarchyDepth](e_ch, HierarchyDepth(10_000_000))
    propagate_transforms_parallel_cached(th_world, th_pool, th_cache)
    assert_true(len(th_cache.buckets) <= 8)
    assert_equal(th_world.get[GlobalTransform](e_ch).translation.x, Float32(5.0))
    th_pool.stop()

    # A6: UiContext begin_scroll_view & begin_modal non-finite guards
    var ui_r22 = UiContext()
    ui_r22.begin_frame(InputState(), screen_w=1280.0, screen_h=720.0, dt=Float32(1.0 / 60.0))
    ui_r22.begin_scroll_view("sv_nan", nan_f32, inf_f32, total_content_h=inf_f32)
    ui_r22.end_scroll_view()
    var modal_open = True
    assert_true(ui_r22.begin_modal("m_nan", "Bad", nan_f32, inf_f32, modal_open))
    ui_r22.end_modal()
    ui_r22.end_frame()
    for i in range(len(ui_r22.draw_list.prims)):
        ref p = ui_r22.draw_list.prims[i]
        assert_true(isfinite(p.x0) and isfinite(p.y0) and isfinite(p.x1) and isfinite(p.y1))

    # A7: FlintStudioState spawn_entity, _apply_view_to_world, apply_gizmo_drag_selected NaN guards
    var st_r22 = FlintStudioState(headless=True)
    var sp_eid = st_r22.spawn_entity("nan_mat", roughness=nan_f32, metallic=nan_f32)
    assert_true(abs(st_r22.entities[sp_eid].roughness - Float32(0.38)) < Float32(1.0e-5))
    assert_true(abs(st_r22.entities[sp_eid].metallic - Float32(0.10)) < Float32(1.0e-5))
    var bad_v = st_r22.entities[sp_eid].copy()
    bad_v.roughness = nan_f32
    bad_v.metallic = nan_f32
    assert_true(st_r22._apply_view_to_world(sp_eid, bad_v))
    st_r22._sync_ecs_and_physics()
    assert_true(abs(st_r22.entities[sp_eid].roughness - Float32(0.38)) < Float32(1.0e-5))
    assert_true(abs(st_r22.entities[sp_eid].metallic - Float32(0.10)) < Float32(1.0e-5))
    var snaps = st_r22.capture_selected_transforms()
    assert_true(st_r22.apply_gizmo_drag_selected(snaps, Vec3(), Vec3(), Quat(), Vec3(nan_f32, 2.0, nan_f32), GIZMO_MODE_SCALE))
    st_r22._sync_ecs_and_physics()
    assert_true(isfinite(st_r22.entities[sp_eid].scale.x) and st_r22.entities[sp_eid].scale.x == Float32(1.0))

    # A8: EvdevGamepad.sync_into_input_state NaN deadzone guard
    var ev_pad = EvdevGamepad()
    ev_pad.connected = True
    ev_pad.axes[PAD_AXIS_LEFT_X] = Float32(0.5)
    ev_pad.axes[PAD_AXIS_LEFT_Y] = Float32(0.0)
    var inp_dz = InputState()
    inp_dz.deadzone_left = nan_f32
    inp_dz.deadzone_right = nan_f32
    ev_pad.sync_into_input_state(inp_dz, 0)
    assert_true(inp_dz.pads[0].axes[PAD_AXIS_LEFT_X] > Float32(0.3))

    # A9: LinuxGamepad.__init__ NaN deadzone & curve_power guard
    var lg_nan = LinuxGamepad(deadzone=nan_f32, curve_power=nan_f32)
    assert_true(abs(lg_nan.deadzone - Float32(0.12)) < Float32(1.0e-5))
    assert_true(abs(lg_nan.curve_power - Float32(1.55)) < Float32(1.0e-5))

    # --- Partition B (8 defect groups) ---
    # B1: QoaStreamCursor.open cache/valid reset on failed re-open
    var qoa_pcm = List[Int16](length=64, fill=Int16(966))
    var qoa_good = encode_qoa(qoa_pcm, 1, 48000)
    var qoa_cur = QoaStreamCursor()
    assert_true(qoa_cur.open(qoa_good))
    var ql = Int16(0)
    var qr = Int16(0)
    assert_true(qoa_cur.read_next_frame_i16(qoa_good, ql, qr))
    var qoa_bad = List[UInt8](length=16, fill=UInt8(0))
    qoa_bad[0] = qoa_good[0]; qoa_bad[1] = qoa_good[1]; qoa_bad[2] = qoa_good[2]; qoa_bad[3] = qoa_good[3]
    qoa_bad[4] = qoa_good[4]; qoa_bad[5] = qoa_good[5]; qoa_bad[6] = qoa_good[6]; qoa_bad[7] = qoa_good[7]
    qoa_bad[8] = qoa_good[8]; qoa_bad[9] = qoa_good[9]; qoa_bad[10] = qoa_good[10]; qoa_bad[11] = qoa_good[11]
    assert_false(qoa_cur.open(qoa_bad))
    assert_false(qoa_cur.valid)
    assert_false(qoa_cur.read_next_frame_i16(qoa_bad, ql, qr))

    # B2: VirtualNetSim advance_ms & submit_datagram Int64 overflow saturation + NaN rates
    var vsim = VirtualNetSim(
        one_way_latency_ms=10.0, jitter_ms=0.0, loss_rate=nan_f64, dup_rate=nan_f64
    )
    assert_equal(vsim._storage[0].loss_rate, 0.0)
    assert_equal(vsim._storage[0].dup_rate, 0.0)
    var vsim_ep0 = vsim.create_ipv4_peer(0)
    var vsim_ep1 = vsim.create_ipv4_peer(1)
    vsim.advance_ms(9.0e15)
    vsim.advance_ms(9.0e15)
    assert_true(vsim._storage[0].now_us > 0 and vsim.now_ms() > 0.0)
    var vsim_msg = List[UInt8](length=4, fill=UInt8(0xAB))
    assert_true(vsim_ep0.send_to(vsim_ep1.local_endpoint(), vsim_msg))
    vsim._storage[0].pump_due()
    var vsim_rx = List[DatagramView]()
    assert_equal(vsim_ep1.recv_batch(vsim_rx), 0)
    assert_equal(vsim.packets_delivered(), 0)
    assert_equal(vsim.in_flight_count(), 1)

    var vsim_jit = VirtualNetSim(
        one_way_latency_ms=9.0e15, jitter_ms=9.0e15, loss_rate=0.0, dup_rate=1.0
    )
    var jit_ep0 = vsim_jit.create_ipv4_peer(0)
    var jit_ep1 = vsim_jit.create_ipv4_peer(1)
    vsim_jit.advance_ms(9.0e15)
    assert_true(jit_ep0.send_to(jit_ep1.local_endpoint(), vsim_msg))
    assert_equal(vsim_jit.packets_delivered(), 0)
    assert_equal(vsim_jit.in_flight_count(), 2)

    # B3: bitpack.mojo NaN Quat -> identity in 32-bit & 48-bit packers
    var bw_bp = NetBitWriter()
    bw_bp.write_quat_smallest_three(Quat(nan_f32, 0.0, 0.0, 1.0))
    bw_bp.write_quat_smallest_three_48(Quat(nan_f32, 0.0, 0.0, 1.0))
    var br_bp = NetBitReader(bw_bp.finish())
    var dq32 = br_bp.read_quat_smallest_three()
    var dq48 = br_bp.read_quat_smallest_three_48()
    assert_true(dq32.w > Float32(0.99) and isfinite(dq32.x))
    assert_true(dq48.w > Float32(0.99) and isfinite(dq48.x))
    _ = pack_quat_smallest_three(Quat(0.0, 0.0, 0.0, 0.0))

    # B4: anim/pose.mojo sample_pose & blend_into NaN time/weight guards
    var pose_m = _make_test_model(10.0, 1.0)
    var ch_t = NativeChannel()
    ch_t.node = 0
    ch_t.path = 0
    ch_t.times = [Float32(0.0), Float32(10.0)]
    ch_t.values = [Float32(0.0), 0.0, 0.0, Float32(10.0), 0.0, 0.0]
    pose_m.animations[0].channels.append(ch_t^)
    var p_samp = Pose(len(pose_m.nodes))
    sample_pose(pose_m, 0, nan_f32, p_samp)
    assert_equal(p_samp.t[0].x, Float32(0.0))
    var p_base = Pose(len(pose_m.nodes))
    p_base.t[0] = Vec3(1.0, 2.0, 3.0)
    var p_other = Pose(len(pose_m.nodes))
    p_other.t[0] = Vec3(5.0, 6.0, 7.0)
    blend_into(p_base, p_other, nan_f32)
    assert_equal(p_base.t[0].x, Float32(1.0))

    # B5: flanim_ecs.compute_volume_preserving_smear_scale & BlendSpace2DDef.weights NaN guards
    var sm_nan = compute_volume_preserving_smear_scale(nan_f32, Float32(0.5), Float32(3.2))
    assert_equal(sm_nan.x, Float32(1.0))
    assert_equal(sm_nan.y, Float32(1.0))
    assert_equal(sm_nan.z, Float32(1.0))
    var bs2d = BlendSpace2DDef()
    bs2d.samples.append(BlendSample2DDef(0, 0.0, 0.0))
    bs2d.samples.append(BlendSample2DDef(1, 1.0, 0.0))
    bs2d.samples.append(BlendSample2DDef(2, 0.0, 1.0))
    var w_nan = bs2d.weights(nan_f32, Float32(0.0))
    assert_equal(len(w_nan), 3)
    assert_equal(w_nan[0], Float32(1.0))
    assert_equal(w_nan[1], Float32(0.0))
    assert_equal(w_nan[2], Float32(0.0))

    # B6: cycles/script.mojo outside 0 raises instead of 1/0 = +Inf blowup
    var out0_raised = False
    try:
        _ = eval_pattern('outside 0 rev (s "bd")')
    except:
        out0_raised = True
    assert_true(out0_raised)

    # B7: procedural/subd.mojo add_vert & set_edge_sharpness NaN sharpness -> 0.0
    var scage = SubdCage()
    var v0 = scage.add_vert(Vec3(0, 0, 0), nan_f32)
    var v1 = scage.add_vert(Vec3(1, 0, 0), Float32(2.0))
    assert_equal(scage.vert_sharpness[v0], Float32(0.0))
    scage.set_edge_sharpness(v0, v1, nan_f32)
    assert_equal(scage.get_edge_sharpness(v0, v1), Float32(0.0))

    # B8: render/{grading, fx_atlas, aaa_post_tsr, aaa_sky_atmosphere} NaN -> 0.0
    assert_equal(filmic_curve(nan_f32), Float32(0.0))
    assert_equal(clamp01(nan_f32), Float32(0.0))
    var dg_nan = eval_display_grade_cpu(Vec3(1.0, 1.0, 1.0), nan_f32, nan_f32)
    assert_equal(dg_nan.x, Float32(0.0))
    assert_equal(dg_nan.y, Float32(0.0))
    assert_equal(dg_nan.z, Float32(0.0))
    var sky_luts = BrunetonSkyLuts(BrunetonAtmosphereParams())
    var st_nan = sky_luts.sample_transmittance(nan_f32, nan_f32)
    assert_equal(st_nan.x, Float32(0.0))
    assert_equal(st_nan.y, Float32(0.0))
    assert_equal(st_nan.z, Float32(0.0))


def test_r23_partition_ab_fixes(tmp_dir: String) raises:
    var nan_f32 = bitcast[DType.float32](UInt32(0x7FC00000))
    var inf_f32 = bitcast[DType.float32](UInt32(0x7F800000))
    var ninf_f32 = bitcast[DType.float32](UInt32(0xFF800000))
    var nan_f64 = bitcast[DType.float64](UInt64(0x7FF8000000000000))

    # --- Partition A (5 defects + mcp_v2 viewport cap) ---
    # A1: core/replay.mojo negative offset OOB guards
    var rep_buf = List[UInt8](length=500, fill=UInt8(0))
    assert_false(is_dev_record_at(rep_buf, -400))
    assert_false(is_dev_record_at(rep_buf, -1))
    assert_equal(word_at(rep_buf, -30), UInt64(0))
    assert_equal(word_at(rep_buf, 496), UInt64(0))
    var r_neg = ReplayByteReader(rep_buf.copy())
    r_neg.offset = -10
    assert_equal(r_neg.u8(), UInt8(0))
    assert_true(r_neg.failed)
    var r_dev = ReplayByteReader(rep_buf.copy())
    r_dev.offset = -400
    var dev_rec = read_dev_record(r_dev)
    assert_true(r_dev.failed)
    assert_equal(len(dev_rec.payload), 0)

    # A2: ui/flexbox_dock.mojo RectF, FlexStyle, leaf_w/h, header_height, min_ratio/_clamp_ratio
    var ft = FlexTree()
    var r_st = FlexStyle(FLEX_ROW)
    r_st.grow = inf_f32
    r_st.max_w = inf_f32
    var root_id = ft.add_node(-1, FlexStyle(FLEX_ROW))
    var ch_id = ft.add_node(root_id, r_st, intrinsic_w=nan_f32, intrinsic_h=Float32(50.0))
    ft.solve(RectF(nan_f32, Float32(0.0), inf_f32, Float32(100.0)), root_id)
    assert_true(isfinite(ft.nodes[ch_id].rect.x) and isfinite(ft.nodes[ch_id].rect.w))
    assert_true(ft.nodes[ch_id].rect.w < Float32(1.0e6))

    var dock = DockSpace()
    var vp = dock.add_central_viewport("VP")
    var sp = dock.split_node(vp, DOCK_SPLIT_V, Float32(0.5), [String("Side")])
    dock.header_height = nan_f32
    dock.min_ratio = nan_f32
    dock.nodes[vp].ratio = Float32(2.0)
    dock.solve_layout(RectF(0.0, 0.0, 800.0, 600.0))
    assert_true(dock.nodes[vp].ratio >= Float32(0.08) and dock.nodes[vp].ratio <= Float32(0.92))
    assert_true(dock.nodes[sp[0]].rect.h > Float32(0.0) and dock.nodes[sp[1]].rect.h > Float32(0.0))
    var fp = dock.find_panel("Side")
    assert_true(fp[0] >= 0 and fp[3].h > Float32(0.0))
    dock.min_ratio = Float32(0.9)
    assert_true(dock._clamp_ratio(Float32(0.0)) <= Float32(0.49))

    # A3: ui/context.mojo non-finite w/h/FlexStyle guards
    var ui_a3 = UiContext()
    ui_a3.begin_frame(InputState(), screen_w=1280.0, screen_h=720.0, dt=Float32(1.0 / 60.0))
    var bad_box_st = FlexStyle(FLEX_ROW)
    bad_box_st.fixed_w = inf_f32
    bad_box_st.fixed_h = nan_f32
    ui_a3.begin_box("box_nan", bad_box_st)
    _ = ui_a3.button("btn_inf", "OK", w=inf_f32, h=nan_f32)
    var s_val = Float32(0.5)
    _ = ui_a3.slider_float("sl_inf", "S", s_val, Float32(0.0), Float32(1.0), w=inf_f32)
    var d_val = Vec3(1.0, 2.0, 3.0)
    _ = ui_a3.drag_float3("dr_inf", "D", d_val, w=nan_f32)
    var ti_val = String("abc")
    _ = ui_a3.text_input("ti_inf", ti_val, w=inf_f32)
    ui_a3.end_box()
    ui_a3.end_frame()
    assert_true(len(ui_a3.draw_list.prims) > 0)
    for i in range(len(ui_a3.draw_list.prims)):
        ref p = ui_a3.draw_list.prims[i]
        assert_true(isfinite(p.x0) and isfinite(p.y0) and isfinite(p.x1) and isfinite(p.y1))

    # A4: text/shaper.mojo & text/line_break.mojo non-finite & negative px guards
    var chain = FontFallbackChain()
    var cp_map = Dict[Int, Int]()
    cp_map[65] = 1
    var advs = List[Float32]()
    advs.append(Float32(12.0))
    chain.set_primary(TrueTypeFont())
    chain.set_baked_primary(inf_f32, cp_map^, advs^, Dict[Int, Float32]())
    assert_true(isfinite(chain.baked_bake_px) and chain.baked_bake_px == Float32(1.0))
    assert_equal(chain.scale_for(0, nan_f32), Float32(0.0))
    assert_equal(chain.scale_for(0, inf_f32), Float32(0.0))
    assert_equal(chain.scale_for(0, Float32(-12.0)), Float32(0.0))
    var shaper = TextShaper(chain.copy())
    var runs_nan = shaper.shape_runs("A", nan_f32)
    assert_equal(len(runs_nan), 0)
    var runs_inf = shaper.shape_runs("A", inf_f32)
    assert_equal(len(runs_inf), 0)
    var runs_neg = shaper.shape_runs("A", Float32(-16.0))
    assert_equal(len(runs_neg), 0)
    var wlines = wrap_shaped(chain, "A", Float32(-16.0), Float32(100.0))
    for wl in wlines:
        assert_true(isfinite(wl.width) and wl.width >= Float32(0.0))

    # A5: text/sdf_bake.mojo NaN outline & EdgeSegment.signed_distance_param guards
    var segs = Segments()
    segs.add(nan_f32, Float32(0.0), Float32(10.0), Float32(0.0))
    assert_equal(segs.count(), 0)
    var bad_go = GlyphOutline()
    bad_go.xs = [nan_f32, Float32(100.0), Float32(100.0)]
    bad_go.ys = [Float32(0.0), Float32(0.0), Float32(100.0)]
    bad_go.on = [True, True, True]
    bad_go.ends = [2]
    var flat_nan = flatten(bad_go, Float32(1.0))
    assert_equal(flat_nan.count(), 0)
    var flat_inf_scale = flatten(bad_go, inf_f32)
    assert_equal(flat_inf_scale.count(), 0)
    assert_equal(len(extract_edge_contours(bad_go, Float32(1.0))), 0)
    var edge_nan = EdgeSegment.line(nan_f32, Float32(0.0), Float32(10.0), Float32(0.0))
    var esd = edge_nan.signed_distance_param(Float32(5.0), Float32(5.0))
    assert_true(isfinite(esd[0]) and isfinite(esd[1]) and isfinite(esd[2]))
    assert_true(isfinite(signed_distance(flat_nan, nan_f32, Float32(0.0))))

    # Extra A6: mcp_v2 viewport capture upper bound
    var mcp_st = FlintStudioState(headless=True)
    var mcp_srv = McpServerV2(tmp_dir)
    var mcp_assets = AssetServer()
    var mcp_pool = JobPool()
    var mcp_req_bytes = List[UInt8]()
    for b in String('{"width":999999999,"height":999999999,"output_path":"cap_r23.png"}').as_bytes():
        mcp_req_bytes.append(b)
    var mcp_req = parse_json(mcp_req_bytes)
    var mcp_res = mcp_srv._dispatch_tool_call(
        "render.capture_viewport", mcp_req, mcp_req.root, mcp_st, mcp_assets, mcp_pool, False, 0
    )
    assert_false(mcp_res[0])
    mcp_pool.stop()

    # --- Partition B (6 defects) ---
    # B1: cycles/synth.mojo NaN clamps & 1e30 fptosi UB
    var cyc = CyclesEngine(48000)
    var cyc_errs = cyc.load(
        'd1 $ s "sine" # delaytime ((1e200*1e200)/(1e200*1e200)) # delayfb ((1e200*1e200)/(1e200*1e200)) # size ((1e200*1e200)/(1e200*1e200)) # resonance ((1e200*1e200)/(1e200*1e200)) # n "1e30" # coarse "1e30"'
    )
    assert_equal(len(cyc_errs), 0)
    var cyc_buf = cyc.render(64)
    assert_equal(len(cyc_buf), 128)
    assert_true(cyc.delay_time < Float32(1.98) and isfinite(cyc.delay_time))
    assert_true(cyc.delay_fb < Float32(0.94) and isfinite(cyc.delay_fb))
    assert_true(cyc.reverb.feedback < Float32(0.92) and isfinite(cyc.reverb.feedback))
    for s in cyc_buf:
        assert_true(isfinite(s))

    # B2: render/aaa_sky_atmosphere.mojo _bilerp_lut huge finite wrap_u
    var sky_lut = List[Vec3](length=16, fill=Vec3(0.25, 0.5, 0.75))
    var bl_huge = _bilerp_lut(sky_lut, 4, 4, Float32(1.0e20), Float32(0.5), wrap_u=True)
    assert_true(isfinite(bl_huge.x) and isfinite(bl_huge.y) and isfinite(bl_huge.z))
    assert_true(abs(bl_huge.x - Float32(0.25)) < Float32(1.0e-5))

    # B3: audio/server.mojo compute_woodworth_hrtf NaN inputs
    var hrtf_nan = compute_woodworth_hrtf(
        Vec3(),
        Vec3(),
        Vec3(0.0, 0.0, -1.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(nan_f32, 0.0, -5.0),
        Vec3(nan_f32, 0.0, 0.0),
        occlusion_gain=nan_f32,
        occlusion_cutoff_hz=nan_f32,
        sample_rate=nan_f32,
    )
    assert_true(
        isfinite(hrtf_nan.gain_l)
        and isfinite(hrtf_nan.gain_r)
        and isfinite(hrtf_nan.delay_l_samples)
        and isfinite(hrtf_nan.delay_r_samples)
        and isfinite(hrtf_nan.cutoff_l_hz)
        and isfinite(hrtf_nan.cutoff_r_hz)
        and isfinite(hrtf_nan.doppler_ratio)
        and isfinite(hrtf_nan.reverb_send)
    )

    # B4: render/aaa_bsdf.mojo & render/material_graph.mojo NaN material defaults
    var aaa_m = AaaMaterial(Vec3(nan_f32, 0.2, inf_f32), roughness=nan_f32, metallic=nan_f32)
    assert_equal(aaa_m.roughness, Float32(0.5))
    assert_equal(aaa_m.metallic, Float32(0.0))
    assert_true(isfinite(aaa_m.base_color.x) and isfinite(aaa_m.base_color.z))
    aaa_m.specular_factor = nan_f32
    aaa_m.clearcoat = nan_f32
    aaa_m.clearcoat_roughness = nan_f32
    aaa_m.base_color = Vec3(nan_f32, nan_f32, nan_f32)
    aaa_m.emissive = Vec3(nan_f32, inf_f32, ninf_f32)
    var aaa_words = List[UInt32]()
    aaa_m.append_words(aaa_words)
    assert_equal(len(aaa_words), 24)
    for wi in range(23):
        assert_true(isfinite(bitcast[DType.float32](aaa_words[wi])))
    var mat_g = MaterialAsset(Vec3(nan_f32, 0.4, 0.5), roughness=nan_f32, metallic=nan_f32)
    assert_equal(mat_g.roughness, Float32(0.5))
    assert_equal(mat_g.metallic, Float32(0.0))
    assert_true(isfinite(mat_g.base_color.x))

    # B5: net/channel.mojo _update_rtt NaN guard
    var vsim_ch = VirtualNetSim()
    var ep_ch = vsim_ch.create_ipv4_peer(0)
    var ch = NetChannel(ep_ch.local_endpoint())
    ch._update_rtt(nan_f64)
    assert_false(ch.rtt_initialized)
    assert_true(isfinite(ch.srtt_ms) and isfinite(ch.jitter_ms) and isfinite(ch.rto_ms))
    ch.poll(ep_ch, nan_f64)
    assert_true(isfinite(ch.srtt_ms) and isfinite(ch.rto_ms))
    _ = vsim_ch.in_flight_count()

    # B6: physics/world.mojo add_sphere / add_capsule / add_box NaN mass -> static
    var pw_nan = PhysicsWorld()
    var s_slot = pw_nan.add_sphere(Vec3(0.0, 1.0, 0.0), nan_f32, nan_f32)
    var c_slot = pw_nan.add_capsule(Vec3(2.0, 1.0, 0.0), nan_f32, nan_f32, nan_f32)
    var b_slot = pw_nan.add_box(Vec3(4.0, 1.0, 0.0), Vec3(nan_f32, 1.0, 1.0), nan_f32)
    for slot_i in [s_slot, c_slot, b_slot]:
        assert_equal(pw_nan.bodies[slot_i].motion_mode, BODY_MOTION_STATIC)
        assert_equal(pw_nan.bodies[slot_i].mass, Float32(0.0))
        assert_equal(pw_nan.bodies[slot_i].inv_mass, Float32(0.0))
        assert_true(isfinite(pw_nan.bodies[slot_i].radius))


def test_r24_partition_ab_fixes(tmp_dir: String) raises:
    _ = tmp_dir
    var nan_f32 = bitcast[DType.float32](UInt32(0x7FC00000))
    var inf_f32 = bitcast[DType.float32](UInt32(0x7F800000))
    var ninf_f32 = bitcast[DType.float32](UInt32(0xFF800000))
    var nan_f64 = bitcast[DType.float64](UInt64(0x7FF8000000000000))

    # --- Partition A ---
    # A1: bake_font negative/huge atlas_width + fixed64 NaN/Inf/huge float
    var bf_neg = bake_font(TrueTypeFont(), List[Int](), 32, 4, -1, 1)
    assert_equal(bf_neg.atlas_width, 0)
    assert_equal(bf_neg.atlas_height, 0)
    assert_equal(len(bf_neg.atlas), 0)
    var bf_huge = bake_font(TrueTypeFont(), List[Int](), 32, 4, 1 << 30, 1)
    assert_equal(len(bf_huge.atlas), 0)
    assert_equal(fixed64(nan_f32), 0)
    assert_equal(fixed64(inf_f32), 0)
    assert_equal(fixed64(Float32(1.0e30)), 2147483647)
    assert_equal(fixed64(Float32(-1.0e30)), -2147483648)
    assert_equal(fixed64(Float32(2.5)), 160)

    # A2: EdgeSegment.signed_distance_param 1e20 coordinates (line, quad, cubic)
    var line_1e20 = EdgeSegment.line(0.0, 0.0, Float32(1.0e20), Float32(1.0e20)).signed_distance_param(0.0, Float32(1.0e20))
    assert_true(isfinite(line_1e20[0]) and abs(line_1e20[0] - Float32(7.071068e19)) < Float32(1.0e15))
    assert_true(isfinite(line_1e20[1]) and isfinite(line_1e20[2]))
    var quad_1e20 = EdgeSegment.quad(0.0, 0.0, Float32(1.0e20), Float32(1.0e20), Float32(2.0e20), 0.0).signed_distance_param(0.0, Float32(1.0e20))
    assert_true(isfinite(quad_1e20[0]) and abs(quad_1e20[0]) > Float32(1.0e18))
    assert_true(isfinite(quad_1e20[1]) and isfinite(quad_1e20[2]))
    var cub_1e20 = EdgeSegment.cubic(0.0, 0.0, Float32(1.0e20), Float32(1.0e20), Float32(2.0e20), Float32(1.0e20), Float32(3.0e20), 0.0).signed_distance_param(0.0, Float32(1.0e20))
    assert_true(isfinite(cub_1e20[0]) and abs(cub_1e20[0]) > Float32(1.0e18))
    assert_true(isfinite(cub_1e20[1]) and isfinite(cub_1e20[2]))

    # A3: FlexTree shrink overflow + direct style.pad_l=NaN mutation
    var ft_sh = FlexTree()
    var r_sh = ft_sh.add_node(-1, FlexStyle(FLEX_ROW))
    var st_sh = FlexStyle()
    st_sh.fixed_w = Float32(1.0e20)
    st_sh.shrink = Float32(1.0e20)
    st_sh.max_w = Float32(1.0e38)
    var c0_sh = ft_sh.add_node(r_sh, st_sh)
    var c1_sh = ft_sh.add_node(r_sh, st_sh)
    ft_sh.solve(RectF(0.0, 0.0, 100.0, 50.0), r_sh)
    assert_true(isfinite(ft_sh.nodes[c0_sh].rect.w) and abs(ft_sh.nodes[c0_sh].rect.w - Float32(50.0)) < Float32(1.0e-2))
    assert_true(isfinite(ft_sh.nodes[c1_sh].rect.w) and abs(ft_sh.nodes[c1_sh].rect.w - Float32(50.0)) < Float32(1.0e-2))

    var ft_pad = FlexTree()
    var r_pad = ft_pad.add_node(-1, FlexStyle.row(pad=10.0))
    var c_pad_st = FlexStyle()
    c_pad_st.fixed_w = Float32(20.0)
    var c_pad = ft_pad.add_node(r_pad, c_pad_st)
    ft_pad.nodes[r_pad].style.pad_l = nan_f32
    ft_pad.solve(RectF(0.0, 0.0, 100.0, 50.0), r_pad)
    assert_true(isfinite(ft_pad.nodes[c_pad].rect.x) and abs(ft_pad.nodes[c_pad].rect.w - Float32(20.0)) < Float32(1.0e-2))

    # A4: UiContext slider_float / end_scroll_view NaN mouse + begin_dock_panel + pick_gizmo_handle
    var ui_sl = UiContext()
    var inp_p = InputState()
    inp_p.move_mouse(20.0, 10.0)
    inp_p.set_mouse(0, True)
    ui_sl.begin_frame(inp_p, screen_w=800.0, screen_h=600.0, dt=Float32(1.0 / 60.0))
    var sv = Float32(0.25)
    _ = ui_sl.slider_float("sl_r24", "S", sv, Float32(0.0), Float32(1.0), w=200.0)
    ui_sl.end_frame()
    var sv_before = sv

    var inp_nan = InputState()
    inp_nan.last_mouse_x = nan_f32
    inp_nan.last_mouse_y = nan_f32
    inp_nan.has_mouse_position = True
    inp_nan.mouse[0] = True
    ui_sl.begin_frame(inp_nan, screen_w=800.0, screen_h=600.0, dt=Float32(1.0 / 60.0))
    _ = ui_sl.slider_float("sl_r24", "S", sv, Float32(0.0), Float32(1.0), w=200.0)
    ui_sl.begin_scroll_view("sc_r24", Float32(180.0), Float32(60.0), total_content_h=Float32(300.0))
    ui_sl.active_id = ui_sl.scroll_stack[len(ui_sl.scroll_stack) - 1].id_hash
    ui_sl.end_scroll_view()
    var dock = DockSpace()
    var dp = dock.add_tabs_node(["Inspector"])
    dock.root = dp
    dock.solve_layout(RectF(0.0, 0.0, 400.0, 300.0))
    dock.nodes[dp].rect = RectF(0.0, 0.0, Float32(1.0e30), Float32(1.0e30))
    assert_true(ui_sl.begin_dock_panel(dock, "Inspector"))
    ui_sl.end_dock_panel()
    ui_sl.end_frame()
    assert_equal(sv, sv_before)
    for pr in ui_sl.draw_list.prims:
        assert_true(isfinite(pr.x0) and isfinite(pr.y0) and isfinite(pr.x1) and isfinite(pr.y1))

    var gh_nan = pick_gizmo_handle(
        nan_f32, nan_f32, Vec3(0.0, 0.0, 5.0), GIZMO_MODE_TRANSLATE,
        0.0, 0.0, 800.0, Float32(1.0e30),
        Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 1.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), 60.0,
    )
    assert_equal(gh_nan, GIZMO_AXIS_NONE)

    # A5: profiler.percentile_of out-of-range & NaN fraction
    var pvals: List[Float32] = [Float32(1.0), Float32(5.0), Float32(9.0)]
    assert_equal(percentile_of(pvals, -2.0), Float32(1.0))
    assert_equal(percentile_of(pvals, 5.0), Float32(9.0))
    assert_equal(percentile_of(pvals, nan_f64), Float32(1.0))
    var empty_p = List[Float32]()
    assert_equal(percentile_of(empty_p, 0.5), Float32(0.0))

    # A6: BlobReader negative offset guards
    var br_neg = BlobReader([UInt8(1), UInt8(2)])
    br_neg.offset = -1
    assert_equal(br_neg.peek_tag(), UInt32(0))
    assert_equal(br_neg.u8(), UInt8(0))
    assert_true(br_neg.failed)
    br_neg.failed = False
    assert_equal(br_neg.text(), String())
    assert_true(br_neg.failed)
    br_neg.failed = False
    assert_false(br_neg.read_bytes_raw(0, 1))
    assert_true(br_neg.failed)

    # A7: OffsetAllocator.free & allocation_size OOB metadata guard
    var oa = OffsetAllocator(1024, 64)
    var bad_alloc = OffsetAllocation(0, UInt32(999999))
    assert_equal(oa.allocation_size(bad_alloc), UInt32(0))
    oa.free(bad_alloc)
    assert_equal(oa.allocations, 0)

    # A8: downsample_argb_2x & downsample_normals_2x empty/zero guards
    var ds_a = downsample_argb_2x(List[UInt32](), 0, 0)
    assert_equal(len(ds_a[0]), 0)
    assert_equal(ds_a[1], 0)
    assert_equal(ds_a[2], 0)
    var ds_a_short = downsample_argb_2x([UInt32(1)], 4, 4)
    assert_equal(len(ds_a_short[0]), 0)
    var ds_n = downsample_normals_2x(List[Vec3](), 0, 0)
    assert_equal(len(ds_n[0]), 0)
    assert_equal(ds_n[1], 0)
    assert_equal(ds_n[2], 0)

    # A9: identity_map negative & zero count
    assert_equal(len(identity_map(-1)), 0)
    assert_equal(len(identity_map(0)), 0)
    assert_equal(len(identity_map(3)), 3)

    # --- Partition B ---
    # B1: Clip.at channels<=0, ch<0, pos=NaN
    var clip_samples: List[Int16] = [Int16(1000), Int16(2000), Int16(3000), Int16(4000)]
    var clip_zero = Clip(0, 44100, clip_samples.copy())
    assert_equal(clip_zero.frames(), 0)
    assert_equal(clip_zero.at(0.25, 0), Float32(0.0))
    var clip_st = Clip(2, 44100, clip_samples.copy())
    var s_neg_ch = clip_st.at(0.25, -1)
    assert_true(isfinite(s_neg_ch) and abs(s_neg_ch - clip_st.at(0.25, 0)) < Float32(1.0e-5))
    assert_equal(clip_st.at(nan_f64, 0), Float32(0.0))

    # B2: ParticleMesh.place OOB guard
    var pm = ParticleMesh()
    _ = pm.add_particle(Vec3(0.0, 0.0, 0.0), Float32(1.0))
    pm.place(-1, Vec3(1.0, 2.0, 3.0))
    pm.place(5, Vec3(1.0, 2.0, 3.0))
    assert_equal(pm.position[0].x, Float32(0.0))
    pm.place(0, Vec3(1.0, 2.0, 3.0))
    assert_equal(pm.position[0].x, Float32(1.0))

    # B3: CrowdSimulator.agent_position & agent_velocity OOB guard
    var cs = CrowdSimulator()
    _ = cs.add_agent(Vec3(3.0, 0.0, 4.0))
    var p_oob_lo = cs.agent_position(-1)
    var p_oob_hi = cs.agent_position(9)
    var v_oob_lo = cs.agent_velocity(-1)
    var v_oob_hi = cs.agent_velocity(9)
    assert_equal(p_oob_lo.x, Float32(0.0))
    assert_equal(p_oob_hi.z, Float32(0.0))
    assert_equal(v_oob_lo.x, Float32(0.0))
    assert_equal(v_oob_hi.z, Float32(0.0))
    assert_equal(cs.agent_position(0).x, Float32(3.0))

    # B4: RestirReservoir NaN/Inf guards
    var r1 = RestirReservoir()
    assert_false(r1.update(UInt32(7), Float32(1.0), nan_f32, Float32(0.5)))
    assert_false(r1.update(UInt32(7), nan_f32, Float32(1.0), Float32(0.5)))
    assert_false(r1.update(UInt32(7), Float32(1.0), inf_f32, Float32(0.5)))
    assert_equal(r1.M_count, UInt32(0))
    assert_equal(r1.weight_sum, Float32(0.0))
    assert_equal(r1.unbiased_weight(), Float32(0.0))
    var r_bad = RestirReservoir(UInt32(1), nan_f32, UInt32(2), nan_f32)
    assert_equal(r_bad.unbiased_weight(), Float32(0.0))
    var r2 = RestirReservoir()
    assert_false(r2.merge(r_bad, Float32(1.0), Float32(0.5)))
    assert_true(isfinite(r2.weight_sum))

    # B5: CharacterController.move_and_slide NaN/Inf cfg.gravity, cfg.up, cfg.max_slope_deg
    var kcc_pw = PhysicsWorld()
    _ = kcc_pw.add_box(Vec3(0.0, -0.5, 0.0), Vec3(20.0, 0.5, 20.0), Float32(0.0))
    var kcc_cfg = CharacterControllerConfig()
    kcc_cfg.gravity = nan_f32
    kcc_cfg.up = Vec3(nan_f32, Float32(1.0), ninf_f32)
    kcc_cfg.max_slope_deg = nan_f32
    var kcc = CharacterController(Vec3(0.0, 2.0, 0.0), kcc_cfg)
    var kcc_pos = kcc.move_and_slide(kcc_pw, Vec3(1.0, 0.0, 0.0), Float32(1.0 / 60.0))
    assert_true(isfinite(kcc_pos.x) and isfinite(kcc_pos.y) and isfinite(kcc_pos.z))
    assert_true(isfinite(kcc.vertical_vel) and isfinite(kcc.velocity.x))


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
    test_r11_partition_ab_fixes(tmp_dir)
    print("[PASS] (18) Round 11 Partition A+B: ttf GSUB hang/bounds & CFF abort cleanup, ktx2 zstd overflow, motionbricks sample/BVH, take_optimize, take_constraints, DebugDraw.paint")
    test_r11b_remediation_fixes()
    print("[PASS] (18b) Round 11b Remediation: optimize_bake NaN/non-monotonic, warp_contact NaN limit, pin_joint/world_pose OOB parents, MbClip.sample +Inf/1e38, DebugDraw stale ptr clear")
    test_r12_partition_ab_fixes(tmp_dir)
    print("[PASS] (19) Round 12 Partition A+B: ttf GSUB budget / outline_at blowup / GPOS kern loop, rich_text bake_px<=0, DebugDraw stride/plot/NaN, take_constraints/take_edit/kimodo_bvh")
    test_r13_partition_ab_fixes(tmp_dir)
    print("[PASS] (20) Round 13 Partition A+B: ttf IGNORE_MARKS budget, rich_text/font/line_break scale/Inf/0-bake, perf_history NaN/cell, ArchetypeSlots read_blob, audio NaN, glb_take NaN, vhacd OOB")
    test_r14_partition_ab_fixes(tmp_dir)
    print("[PASS] (21) Round 14 Partition A+B: cvars fptosi/NaN, rich_text Inf sums, DockSpace JSON/NaN, report_ms neg/Inf, Heightfield ovf/NaN, synth wav caps, foliage clamp, orca/diff/fmt/vehicle")
    test_r15_partition_ab_fixes(tmp_dir)
    print("[PASS] (22) Round 15 Partition A+B: profile_compare neg-id, format_float2/_fmt_f32_2dp/glb u8, mcp_v2 JSON, clip_sample, vehicle/animator/line/mm/vat/bvh/hf/wrap_time/vsm/noise")
    test_r16_partition_ab_fixes()
    print("[PASS] (23) Round 16 Partition A+B: hdr_image inf, glb emissive, ktx2 sfloat16/32, scroll_visible_range, frame_count, mcp base_color, draw_line_3d, xpbd/ragdoll substeps, camera_rig, physics step, breathing/value_noise/k_scale")
    test_r17_partition_ab_fixes(tmp_dir)
    print("[PASS] (24) Round 17 Partition A+B: profiler >1MiB/escape, exr +inf/equirect, springs 2.5e38, aero6dof substeps/dt, orca 1e38/coords, navmesh 1e30/nan, ops B_MOD/cast, grading to_byte")
    test_r18_partition_ab_fixes(tmp_dir)
    print("[PASS] (25) Round 18 Partition A+B: flpk_v3 flmeta, scene transactional load, ttf/sdf_bake CFF overflow, ui slider/layout, transform_hierarchy, input_map, meshlet/gizmo, physics kcc/vehicle/xpbd/water/tires, motion_matching, render_world/lighting, loft_bevel, ml softmax/layer_norm, animator neg dt")
    test_r19_partition_ab_fixes()
    print("[PASS] (26) Round 19 Partition A+B: ui drag_float3/push_clip/dockspace, evdev rumble, macos/win32/dsp audio, glb_native/writer/playback, motion_tools/ide_view, png/exr/hdr dims, ecs slots, replay spline, vehicle f64->f32, rg non-256 align, acoustics 1e20, ragdoll impulse/step")
    test_r20_partition_ab_fixes()
    print("[PASS] (27) Round 20 Partition A+B: nine_slice, model sample_clip, audio NaN/PCM, jpeg 65536, dsp fdn/pan/svf/blep, ik, input_map, ui begin_frame/curve_editor, mesh_builder/uf16, gizmo, rg 192 align, camera free_fly, wind to_fx, motionbricks slerp, raycast nan, behavior_tree, crypto cookie")
    test_r21_partition_ab_fixes()
    print("[PASS] (28) Round 21 Partition A+B: model OOB/IK, pad OOB, lathe mismatch, button/toolbar OOB, slice_map 1e38, dsp timpani/ITD/spatial, ik cone/4bar, gizmo/studio cel NaN, replay spline, VSM invalidate_region, utility eval NaN, water submerged/wave, replicator quant NaN")
    test_r22_partition_ab_fixes()
    print("[PASS] (29) Round 22 Partition A+B: gizmo_rot, ik quat, engine_app dt/hz, transform_hierarchy alpha/depth, ui scroll/modal, studio nan, evdev/os dz, qoa cursor reset, virtual_sim i64/nan, bitpack quat, pose/smear/bs2d, cycles outside 0, subd sharp, render NaN clamps")
    test_r23_partition_ab_fixes(tmp_dir)
    print("[PASS] (30) Round 23 Partition A+B: replay neg-offset, flexbox_dock NaN/Inf/ratio, ui/context NaN/Inf, text/shaper & line_break px, sdf_bake NaN/Inf, mcp_v2 cap, cycles synth NaN/1e30, sky _bilerp_lut 1e20, audio hrtf NaN, aaa_bsdf/material_graph NaN, net/channel RTT NaN, physics NaN mass")
    test_r24_partition_ab_fixes(tmp_dir)
    print("[PASS] (31) Round 24 Partition A+B: sdf_bake bake_font/fixed64/1e20 EdgeSegment, flexbox_dock shrink-ovf/pad_l, ui slider/scroll/dock_panel/gizmo NaN, profiler percentile_of, serial BlobReader neg-offset, offset_allocator OOB, bc_compress downsample 0, input identity_map, cycles Clip.at, xpbd place, orca OOB, restir NaN, kcc cfg NaN")

    _rm_dir_shallow(tmp_dir)
    print("=== ALL TIER 5 COVERAGE & REGRESSION TESTS PASSED ===")

"""Flint Studio Visual Scene Editor State, Reversible Undo/Redo, 1-Click LZ4 PIE & Procedural Forge (`studio_app.mojo`).

Implements:
  1. Unified Engine v5 Spine (`EngineApp` + `WindowPlugin` + `TransformPlugin` + `InputPlugin` +
     `PhysicsPlugin` + `RenderPlugin` + `flint.ecs.World` + `flint.scene.Scene` + `PhysicsWorld`):
     Authoritative ECS components (`LocalTransform`, `GlobalTransform`, `PrevGlobalTransform`,
     `Parent`, `Children`, `HierarchyDepth`, `MeshRendererComponent`, `MaterialOverrideComponent`,
     `RigidBodyComponent`, `ColliderComponent`, `StudioEntityMetaComponent`, `StudioStyleComponent`)
     with a derived `StudioEntityView` projection for MCP/Inspector reflection.
  2. Multi-Select & Cycle-Safe Hierarchy Reparenting (`selected_entity_ids`, Ctrl/Shift selection,
     `reparent_entity`, `reparent_selected_to`, centroid gizmos, bulk Inspector/duplicate/delete).
  3. Reversible Coalesced `EditorTransaction` Undo/Redo Stack (`World.snapshot_blob`/`restore_blob`)
     with bit-exact `fnv64` `state_hash()` invertibility.
  4. **1-Click Play-In-Editor (`PieSession`)** backed by `Scene` + `World.snapshot_blob`/`restore_blob`
     + `PhysicsWorld.serialize`/`deserialize` + LZ4 `checkpoint.mojo`.
  5. `.flscn` / `.flscnb` scene persistence, `PhysicsWorld.raycast_all` scene raycasting,
     Procedural Forge bake-to-`.glb`, and docked AAA Studio Editor UI (`emit_studio_ui_draw_list`)
     with collapsible indented hierarchy tree & drag-and-drop reparenting.
"""
from flint.studio.source_editor import EmbeddedIde
from flint.studio import scene_menu
from std.collections import Dict
from std.math import abs, asin, cos, isfinite, isinf, isnan, max, min, sin, sqrt, tan
from std.memory import bitcast
from flint.ai import AiPlugin, BehaviorTreeComponent, register_ai_codecs
from flint.ai.behavior_tree import BtExecutionState
from flint.app.hot_reload import (
    HotReloadHost,
    POD_FIELD_BOOL,
    POD_FIELD_COLOR_RGB,
    POD_FIELD_F32,
    POD_FIELD_INT,
    POD_FIELD_QUAT,
    POD_FIELD_STRING,
    POD_FIELD_VEC3,
    PodComponentSchema,
    PodSchemaRegistryResource,
)
from flint.anim.flanim_ecs import (
    BoneSocketAttachment,
    SteppedAnimClock,
    compute_volume_preserving_smear_scale,
    load_flanim_bytes,
    load_flanim_file,
)
from flint.anim.flrig import FrigRigDef, load_flrig_bytes, load_flrig_file
from flint.anim.plugin import (
    AnimAssetCache,
    AnimPlugin,
    AnimatorComponent,
    BoneSocketComponent,
    RigComponent,
    SkeletonComponent,
    register_anim_codecs,
)
from flint.anim.pose import SkeletonInstance, sample_pose
from flint.anim.procedural import Placement
from flint.anim.sync import wrap_time
from flint.assets.asset_plugin import (
    AssetPlugin,
    PieResolverEpochSnapshot,
    SceneAssetResolver,
    _compute_skeleton_topo_order,
)
from flint.assets.asset_ref import AssetRef, normalize_asset_rel_path
from flint.assets.flmat import (
    FlmatAsset,
    MAT_SHADING_NPR_CEL,
    MAT_SHADING_PBR,
    MAT_SHADING_STYLIZED_VFX,
    load_flmat_bytes,
    load_flmat_file,
    save_flmat_file,
)
from flint.assets.server import AssetServer
from flint.audio.plugin import (
    AudioEmitterComponent,
    AudioPlugin,
    AudioSyncStateResource,
    register_audio_codecs,
)
from flint.audio.server import AudioServer, SoundHandle
from flint.app.engine_app import (
    AppConfig,
    EngineApp,
    FixedTime,
    FrameTime,
    Resource,
    Resources,
    STAGE_FIRST,
    STAGE_FIXED_UPDATE,
    STAGE_LAST,
    STAGE_POST_UPDATE,
    STAGE_PRE_UPDATE,
    STAGE_UPDATE,
    WorldTickSyncResource,
)
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.core.log import state_directory
from flint.assets.checkpoint import pack_checkpoint_lz4, unpack_checkpoint_lz4
from flint.core.lz4 import lz4_pack, lz4_packed_raw_size, lz4_unpack
from flint.assets.flpk_v3 import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    ASSET_KIND_TEXTURE,
    FlpkV3MappedArchive,
    UntypedHandle,
    decode_cooked_rig_slab,
    load_or_init_flmeta,
)
from flint.assets.glb_native import parse_native_glb
from flint.assets.image import decode_png
from flint.assets.jpeg import decode_image_auto
from flint.assets.json import JsonDocument, parse_json
from flint.assets.json_write import json_quote
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh, NativeModel
from flint.audio.qoa import decode_qoa
from flint.camera.rig import CAM_MODE_FREE_FLY, CAM_MODE_ORBIT_FOLLOW, CameraRig
from flint.core.detmath import PI, det_atan2, set_sim_float_mode
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.platform.audio import decode_wav
from flint.core.serial import BlobReader, BlobWriter, fnv64, fourcc
from flint.core.transform import Mat4, trs
from flint.ecs.world import (
    Entity,
    NULL_ENTITY,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.input.input_map import (
    ACTION_AXIS2D,
    ACTION_BOOL,
    BIND_KEY,
    BIND_MOUSE_BUTTON,
    BIND_PAD_BUTTON,
    InputBinding,
    InputContext,
    InputMap,
    MOD_ALT,
    MOD_CTRL,
    MOD_NONE,
    MOD_SHIFT,
    _keycode_display_name,
)
from flint.input.plugin import InputPlugin
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.filter import CollisionFilter
from flint.physics.narrowphase import ConvexHull
from flint.physics.plugin import (
    ColliderComponent,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    RigidBodyComponent,
    TriggerEventsResource,
    register_physics_codecs,
)
from flint.physics.world import (
    BODY_MOTION_DYNAMIC,
    BODY_MOTION_KINEMATIC,
    BODY_MOTION_STATIC,
    BODY_SHAPE_BOX,
    BODY_SHAPE_CAPSULE,
    BODY_SHAPE_CONVEX_HULL,
    BODY_SHAPE_SPHERE,
    PhysicsWorld,
)
from flint.nav.navmesh import NavMesh
from flint.nav.plugin import NavPlugin, NavPluginStateResource, register_nav_codecs
from flint.platform.input import InputState
from flint.platform.os import (
    env_value,
    file_size,
    file_stamp,
    remove_file,
    is_directory,
    is_symlink,
    list_directory,
    make_dirs,
    parent_directory,
    read_bytes,
    real_path,
    run_process_redirected,
    unix_time_sec,
    write_bytes,
    write_file_durable,
)
from flint.procedural.arch_kit import (
    build_colonnade_arcade_bay,
    build_street_bollard_and_grate,
)
from flint.procedural.foliage import (
    build_botanical_oak,
    build_botanical_pine,
)
from flint.procedural.subd import SubdCage, subdivide_to_mesh
from flint.assets.glb_writer import encode_native_glb, write_native_glb
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_npr_cel import AaaCelParams
from flint.render.aaa_npr_vfx import AaaImpactFrameParams, AaaShockwaveRing, AaaStylizedVfxParams
from flint.render.aaa_post_tsr import AAA_DISPLAY_AGX_REC709
from flint.render.aaa_skinning_rt import DualQuat
from flint.render.aaa_sky_atmosphere import BrunetonAtmosphereParams
from flint.render.font import SdfFont
from flint.render.material_graph import (
    MAT_DOMAIN_CLEARCOAT,
    MAT_DOMAIN_STANDARD_PBR,
    MAT_FLAG_CLEARCOAT,
    MaterialAsset,
)
from flint.render.plugin import (
    CameraComponent,
    EnvironmentComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    PointLightComponent,
    RenderPlugin,
    RenderPluginStateResource,
    SpotLightComponent,
    build_overridden_material,
    register_render_codecs,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderCameraPacket,
    RenderWorld,
    ScalableRenderer,
    make_rh_look_at,
    make_rh_reverse_z_perspective,
)
from flint.scene.scene import (
    ComponentRegistry,
    Prefab,
    PrefabInstanceComponent,
    Scene,
    UnknownComponentsComponent,
    duplicate_entities,
    load_flprefab_bytes,
    load_flprefab_file,
)
from flint.scene.transform_hierarchy import (
    Children,
    GlobalTransform,
    HierarchyDepth,
    LocalTransform,
    Parent,
    PrevGlobalTransform,
    SerializableComponent,
    TransformPlugin,
    TransformPropagationCache,
    clear_parent,
    propagate_transforms_parallel_cached,
    set_parent,
    snapshot_prev_global_transforms,
)
from flint.studio.agent_bridge import BridgedMcpEvent, StudioAgentRuntime
from flint.studio.agent_bus import (
    AGENT_STATUS_BUILDING,
    AGENT_STATUS_EDITING_SCENE,
    AGENT_STATUS_ERROR,
    AGENT_STATUS_IDLE,
    AGENT_STATUS_PLANNING,
    AGENT_STATUS_TESTING_PIE,
    AGENT_STATUS_THINKING,
    AGENT_STATUS_WAITING_APPROVAL,
    AGENT_STATUS_WRITING_CODE,
    APPROVAL_AUTO,
    APPROVAL_EDITS,
    APPROVAL_SRC_MCP_TOOL,
    GHOST_TRANSFORM,
    APPROVAL_STEP,
    ROLE_EXTERNAL_MCP,
    ROLE_GAMEPLAY_CODER,
    ROLE_LOOKDEV_ARTIST,
    ROLE_QA_PLAYTESTER,
    ROLE_SCENE_BUILDER,
    AgentTelemetryBus,
)
from flint.studio.agent_panel import (
    AgentMissionControlPanel,
    AgentPanelAction,
    PANEL_ACT_APPROVE,
    PANEL_ACT_APPROVE_TURN,
    PANEL_ACT_CANCEL_TURN,
    PANEL_ACT_FOCUS_ENTITY,
    PANEL_ACT_OPEN_FILE_DIFF,
    PANEL_ACT_REJECT,
    PANEL_ACT_REVERT_STEP,
    PANEL_ACT_REVERT_TURN,
    PANEL_ACT_RUN_PROMPT,
    PANEL_ACT_SELECT_OR_SPAWN_ROLE,
    PANEL_ACT_SET_POLICY,
    PANEL_ACT_TIME_TRAVEL_EXIT,
    PANEL_ACT_TIME_TRAVEL_SCRUB,
    PANEL_ACT_TOGGLE_FOLLOW_CAMERA,
    PANEL_ACT_TOGGLE_THEATRE,
)
from flint.studio.agent_theatre_3d import (
    HIT_KIND_APPROVE,
    HIT_KIND_CALLOUT_FOCUS,
    HIT_KIND_PRESENCE,
    HIT_KIND_REJECT,
    TheatrePillHit,
    emit_agent_viewport_3d_overlays,
    emit_agent_viewport_hud_overlays_with_font,
)
from flint.studio.code_drawer import (
    BUILD_STATUS_BUILDING,
    BUILD_STATUS_ERROR,
    BUILD_STATUS_OK,
    CODE_ACT_ACCEPT_ALL_DIFF,
    CODE_ACT_ACCEPT_HUNK,
    CODE_ACT_ASK_AGENT_FIX,
    CODE_ACT_BUILD_HOT_RELOAD,
    CODE_ACT_CLOSE_TAB,
    CODE_ACT_JUMP_DIAGNOSTIC,
    CODE_ACT_OPEN_EXTERNAL_IDE,
    CODE_ACT_REJECT_ALL_DIFF,
    CODE_ACT_REJECT_HUNK,
    CODE_MODE_DIFF_UNIFIED,
    CodeDrawer,
)
from flint.studio.viewport_transform import ViewportTransform
from flint.studio.gizmo import (
    GIZMO_AXIS_NONE,
    GIZMO_AXIS_VIEW,
    GIZMO_AXIS_X,
    GIZMO_AXIS_Y,
    GIZMO_AXIS_Z,
    GIZMO_MODE_ROTATE,
    GIZMO_MODE_SCALE,
    GIZMO_MODE_SELECT,
    GIZMO_MODE_TRANSLATE,
    GIZMO_PLANE_XY,
    GIZMO_PLANE_XZ,
    GIZMO_PLANE_YZ,
    GIZMO_SCALE_UNIFORM,
    GIZMO_TARGET_PX,
    GIZMO_SPACE_LOCAL,
    GIZMO_SPACE_WORLD,
    ViewportRay,
    emit_gizmo_draw_list,
    gizmo_rotation_delta_rad,
    gizmo_scale_drag,
    gizmo_uniform_scale_screen,
    gizmo_world_scale,
    pick_gizmo_handle,
    project_world_to_viewport,
    ray_axis_closest_param,
    ray_plane_hit,
    snap_angle_deg,
    snap_constrained_delta,
    snap_scalar,
)
from flint.studio.notifications import (
    JobProgressTracker,
    TOAST_AGENT,
    TOAST_ERROR,
    TOAST_INFO,
    TOAST_SUCCESS,
    TOAST_WARN,
    ToastManager,
    draw_job_progress_pills,
    draw_toast_stack,
)
from flint.studio.palette import (
    KeyChordHud,
    PALETTE_AGENTS,
    PALETTE_ALL,
    PALETTE_ITEM_AGENT,
    PALETTE_ITEM_ASSET,
    PALETTE_ITEM_COMMAND,
    PALETTE_ITEM_ENTITY,
    PALETTE_ITEM_SPAWN,
    PALETTE_SPAWN,
    PaletteActionResult,
    StudioCommandPalette,
    draw_command_palette,
    draw_key_chord_hud,
)
from flint.studio.palette_registry import (
    PAL_CMD_AGENT_APPROVE_STEP,
    PAL_CMD_AGENT_EXIT_TIME_TRAVEL,
    PAL_CMD_AGENT_GAMEPLAY_CODER,
    PAL_CMD_AGENT_LOOKDEV_ARTIST,
    PAL_CMD_AGENT_PAUSE_RESUME,
    PAL_CMD_AGENT_QA_PLAYTESTER,
    PAL_CMD_AGENT_REJECT_STEP,
    PAL_CMD_AGENT_REVERT_LAST_STEP,
    PAL_CMD_AGENT_SCENE_BUILDER,
    PAL_CMD_AGENT_TOGGLE_FOLLOW_CAM,
    PAL_CMD_AGENT_TOGGLE_PANEL,
    PAL_CMD_AGENT_TOGGLE_THEATRE_3D,
    PAL_CMD_AGENT_WATCH_DEMO,
    PAL_CMD_BAKE_NAVMESH,
    PAL_CMD_BUILD_HOT_RELOAD,
    PAL_CMD_CLEAR_SELECTION,
    PAL_CMD_COPY,
    PAL_CMD_DELETE,
    PAL_CMD_DUPLICATE,
    PAL_CMD_FRAME_SELECTION,
    PAL_CMD_LOAD_SCENE,
    PAL_CMD_NEW_SCENE,
    PAL_CMD_OPEN_PALETTE,
    PAL_CMD_OPEN_QUICK_SPAWN,
    PAL_CMD_PACKAGE_PROFILES,
    PAL_CMD_PACKAGE_PROJECT,
    PAL_CMD_PARENT_TO_PRIMARY,
    PAL_CMD_PASTE,
    PAL_CMD_PIE_EJECT_CAMERA,
    PAL_CMD_PIE_PAUSE,
    PAL_CMD_PIE_PLAY,
    PAL_CMD_PIE_STEP,
    PAL_CMD_PIE_STOP,
    PAL_CMD_REDO,
    PAL_CMD_RELOAD_SCENE,
    PAL_CMD_RECOVER_AUTOSAVE,
    PAL_CMD_RENAME,
    PAL_CMD_RESCAN_ASSETS,
    PAL_CMD_RESET_CAMERA,
    PAL_CMD_RESET_TRANSFORM,
    PAL_CMD_SAVE_SCENE,
    PAL_CMD_SELECT_ALL,
    PAL_CMD_SHOW_DIAGNOSTICS,
    PAL_CMD_SNAP_VIEW_FRONT,
    PAL_CMD_SNAP_VIEW_RIGHT,
    PAL_CMD_SNAP_VIEW_TOP,
    PAL_CMD_SPAWN_AUDIO,
    PAL_CMD_SPAWN_CAMERA,
    PAL_CMD_SPAWN_CUBE,
    PAL_CMD_SPAWN_EMPTY,
    PAL_CMD_SPAWN_PHYSICS_BOX,
    PAL_CMD_SPAWN_PLANE,
    PAL_CMD_SPAWN_POINT_LIGHT,
    PAL_CMD_SPAWN_PREFAB,
    PAL_CMD_SPAWN_SPHERE,
    PAL_CMD_SPAWN_SPOT_LIGHT,
    PAL_CMD_SPAWN_SUN_LIGHT,
    PAL_CMD_SPAWN_TRIGGER,
    PAL_CMD_THEME_CYCLE,
    PAL_CMD_THEME_MIDNIGHT,
    PAL_CMD_THEME_OBSIDIAN,
    PAL_CMD_THEME_OKABE_ITO,
    PAL_CMD_THEME_PAPER_LIGHT,
    PAL_CMD_TOGGLE_CODE_DRAWER,
    PAL_CMD_TOGGLE_COLLIDER_OVERLAY,
    PAL_CMD_TOGGLE_GIZMO_SPACE,
    PAL_CMD_TOGGLE_GRID_OVERLAY,
    PAL_CMD_TOGGLE_NAVMESH_OVERLAY,
    PAL_CMD_TOGGLE_ORTHO_CAMERA,
    PAL_CMD_TOGGLE_PERF_SPARKLINE,
    PAL_CMD_TOGGLE_SNAP,
    PAL_CMD_TOOL_MOVE,
    PAL_CMD_TOOL_ROTATE,
    PAL_CMD_TOOL_SCALE,
    PAL_CMD_TOOL_SELECT,
    PAL_CMD_UNDO,
    PAL_CMD_UNPARENT,
    PAL_CMD_WORKSPACE_AGENT_STUDIO,
    PAL_CMD_WORKSPACE_CODE_SCENE,
    PAL_CMD_WORKSPACE_EDIT,
    PAL_CMD_WORKSPACE_LOOKDEV,
    PAL_CMD_WORKSPACE_PLAY,
    begin_studio_chord,
    populate_studio_palette,
)
from flint.studio.studio_panels import (
    ASSET_FILTER_ALL,
    AssetBrowserPanelState,
    ENTITY_KIND_AUDIO,
    ENTITY_KIND_CAMERA,
    ENTITY_KIND_COLLIDER,
    ENTITY_KIND_EMPTY,
    ENTITY_KIND_FOLIAGE,
    ENTITY_KIND_LIGHT_POINT,
    ENTITY_KIND_LIGHT_SPOT,
    ENTITY_KIND_MESH,
    ENTITY_KIND_PREFAB,
    InspectorPanelState,
    OutlinerPanelState,
    OutlinerRowView,
    asset_filter_accent,
    classify_asset_filter,
    draw_outliner_panel_v2,
)
from flint.studio.studio_workspace import (
    AgentCheckpointLog,
    PANEL_AGENTS,
    PANEL_ASSET_BROWSER,
    PANEL_CODE_DRAWER,
    PANEL_HIERARCHY,
    PANEL_INPUT_MAP,
    PANEL_INSPECTOR,
    PANEL_SCENE_VIEWPORT,
    WorkspacePresetManager,
    build_workspace_dockspace,
)
from flint.studio.viewport_hud import (
    HDR_ACT_AGENT_TOGGLE,
    HDR_ACT_BUILD,
    HDR_ACT_COMMAND_PALETTE,
    HDR_ACT_MENU,
    HDR_ACT_NONE,
    HDR_ACT_SAVE,
    HDR_ACT_THEME_CYCLE,
    HDR_ACT_WIN_CLOSE,
    HDR_ACT_WIN_DRAG_MOVE,
    HDR_ACT_WIN_MAXIMIZE,
    HDR_ACT_WIN_MINIMIZE,
    HDR_ACT_WORKSPACE,
    STUDIO_HEADER_BAR_H,
    STUDIO_STATUS_BAR_H,
    StudioHeaderBar,
    VHUD_ACT_CAM_MODE,
    VHUD_ACT_NONE,
    VHUD_ACT_PIE_EJECT,
    VHUD_ACT_PIE_PAUSE,
    VHUD_ACT_PIE_PLAY,
    VHUD_ACT_PIE_STEP,
    VHUD_ACT_PIE_STOP,
    VHUD_ACT_SNAP_VIEW,
    VHUD_ACT_TOGGLE_COLLIDERS,
    VHUD_ACT_TOGGLE_GRID,
    VHUD_ACT_TOGGLE_NAVMESH,
    VHUD_ACT_TOGGLE_ORTHO,
    VHUD_ACT_TOGGLE_PERF,
    VHUD_ACT_TOGGLE_SNAP,
    VHUD_ACT_TOGGLE_SPACE,
    VHUD_ACT_TOOL_MOVE,
    VHUD_ACT_TOOL_ROTATE,
    VHUD_ACT_TOOL_SCALE,
    VHUD_ACT_TOOL_SELECT,
    VHUD_PIE_PAUSED,
    VHUD_PIE_PLAYING,
    VHUD_PIE_STOPPED,
    VHUD_TOOL_MOVE,
    VHUD_TOOL_ROTATE,
    VHUD_TOOL_SCALE,
    VHUD_TOOL_SELECT,
    ViewportHud,
    WORKSPACE_AGENT_STUDIO,
    WORKSPACE_CODE_SCENE,
    WORKSPACE_COUNT,
    WORKSPACE_EDIT,
    WORKSPACE_LOOKDEV,
    WORKSPACE_PLAY,
)
from flint.ui.animator import UI_ANIM_INSTANT, UI_ANIM_NORMAL, UI_ANIM_REDUCED
from flint.ui.context import (
    utf8_drop_last_char,
    UI_KEY_BACKSPACE,
    UI_KEY_DOWN,
    UI_KEY_ENTER,
    UI_KEY_ESCAPE,
    UI_KEY_TAB,
    UI_KEY_UP,
    UiContext,
    hash_ui_id,
    pack_rgba8,
)
from flint.ui.draw_list import PRIM_GLYPH, UiDrawList
from flint.ui.hud import hud_bar
from flint.ui.theme import (
    AGENT_STATE_BUILDING,
    AGENT_STATE_EDITING_SCENE,
    AGENT_STATE_ERROR,
    AGENT_STATE_IDLE,
    AGENT_STATE_PLANNING,
    AGENT_STATE_TESTING_PIE,
    AGENT_STATE_THINKING,
    AGENT_STATE_WAITING_APPROVAL,
    AGENT_STATE_WRITING_CODE,
    StudioTheme,
    THEME_FLINT_OBSIDIAN,
    THEME_MIDNIGHT_STUDIO,
    THEME_OKABE_ITO,
    THEME_PAPER_LIGHT,
    rgba8_with_alpha,
)
from flint.ui.flexbox_dock import (
    DOCK_SPLIT_H,
    DOCK_SPLIT_V,
    DOCK_TABS,
    DockSpace,
    FlexStyle,
    RectF,
    RectF as _RectF,
)

comptime PIVOT_BOUNDS_CENTER: Int = 0
comptime PIVOT_ACTIVE: Int = 1
comptime GIZMO_DRAG_SEQ_TAG: UInt64 = 0x4000000000000000
comptime STUDIO_CONSOLE_CAP: Int = 300
comptime STUDIO_RECENT_SCENES_CAP: Int = 10
"""High bit marking gizmo-drag coalesce keys, so they never collide with UI-widget coalesce keys."""
comptime COLLIDER_NONE: Int = 0
comptime COLLIDER_SPHERE: Int = 1
comptime COLLIDER_BOX: Int = 2
comptime COLLIDER_CAPSULE: Int = 3
comptime COLLIDER_STATIC_MESH: Int = 4

comptime SHADING_PBR: Int = 0
comptime SHADING_NPR_CEL: Int = 1
comptime SHADING_STYLIZED_VFX: Int = 2

comptime PIE_STOPPED: Int = 0
comptime PIE_PLAYING: Int = 1
comptime PIE_PAUSED: Int = 2

comptime TX_SPAWN_ENTITY: Int = 1
comptime TX_DESPAWN_ENTITY: Int = 2
comptime TX_PATCH_ENTITY: Int = 3
comptime TX_REPARENT_ENTITY: Int = 4
comptime TX_BATCH_EDIT: Int = 5

comptime FIELD_BOOL: Int = 1
comptime FIELD_INT: Int = 2
comptime FIELD_F32: Int = 3
comptime FIELD_STRING: Int = 4
comptime FIELD_VEC3: Int = 5
comptime FIELD_QUAT: Int = 6
comptime FIELD_COLOR_RGB: Int = 7

comptime STUDIO_CHECKPOINT_FOURCC: UInt32 = fourcc("STDV")
comptime MAX_UNDO_STACK: Int = 512
comptime MAX_UNDO_BLOB_BYTES: Int = 64 * 1024 * 1024


@always_inline
def _f32_s(v: Float32) -> String:
    return String(Float64(v))


def _vec3_json(v: Vec3) -> String:
    return "[" + _f32_s(v.x) + ", " + _f32_s(v.y) + ", " + _f32_s(v.z) + "]"


def _read_vec3_json(doc: JsonDocument, node: Int, fallback: Vec3 = Vec3()) -> Vec3:
    if doc.array_size(node) < 3:
        return fallback
    return Vec3(
        Float32(doc.number(doc.at(node, 0), Float64(fallback.x))),
        Float32(doc.number(doc.at(node, 1), Float64(fallback.y))),
        Float32(doc.number(doc.at(node, 2), Float64(fallback.z))),
    )


# ============================================================================
# 1. `FieldVisitor` & `trait Inspectable` Reflection
# ============================================================================

struct InspectedField(Copyable, Movable):
    var name: String
    var kind: Int
    var b_val: Bool
    var i_val: Int
    var f_val: Float32
    var s_val: String
    var v3_val: Vec3
    var q_val: Quat

    def __init__(out self, name: String = "", kind: Int = FIELD_F32):
        self.name = name
        self.kind = kind
        self.b_val = False
        self.i_val = 0
        self.f_val = 0.0
        self.s_val = String()
        self.v3_val = Vec3()
        self.q_val = Quat()


struct FieldVisitor(Movable):
    """Bidirectional property visitor: collects fields for the Studio Inspector UI or mutates a named field."""
    var fields: List[InspectedField]
    var mutate_target: String
    var mutate_field: InspectedField
    var mutated: Bool

    def __init__(out self):
        self.fields = List[InspectedField]()
        self.mutate_target = String()
        self.mutate_field = InspectedField()
        self.mutated = False

    @staticmethod
    def for_patch(field_name: String, new_val: InspectedField) -> Self:
        var v = Self()
        v.mutate_target = field_name
        v.mutate_field = new_val.copy()
        return v^

    def visit_bool(mut self, name: String, mut value: Bool):
        if self.mutate_target.byte_length() > 0:
            if self.mutate_target == name and self.mutate_field.kind == FIELD_BOOL:
                value = self.mutate_field.b_val
                self.mutated = True
            return
        var f = InspectedField(name, FIELD_BOOL)
        f.b_val = value
        self.fields.append(f^)

    def visit_int(mut self, name: String, mut value: Int):
        if self.mutate_target.byte_length() > 0:
            if self.mutate_target == name and self.mutate_field.kind == FIELD_INT:
                value = self.mutate_field.i_val
                self.mutated = True
            return
        var f = InspectedField(name, FIELD_INT)
        f.i_val = value
        self.fields.append(f^)

    def visit_f32(mut self, name: String, mut value: Float32):
        if self.mutate_target.byte_length() > 0:
            if self.mutate_target == name and self.mutate_field.kind == FIELD_F32:
                value = self.mutate_field.f_val
                self.mutated = True
            return
        var f = InspectedField(name, FIELD_F32)
        f.f_val = value
        self.fields.append(f^)

    def visit_string(mut self, name: String, mut value: String):
        if self.mutate_target.byte_length() > 0:
            if self.mutate_target == name and self.mutate_field.kind == FIELD_STRING:
                value = self.mutate_field.s_val
                self.mutated = True
            return
        var f = InspectedField(name, FIELD_STRING)
        f.s_val = value
        self.fields.append(f^)

    def visit_vec3(mut self, name: String, mut value: Vec3, is_color: Bool = False):
        var k = FIELD_COLOR_RGB if is_color else FIELD_VEC3
        if self.mutate_target.byte_length() > 0:
            if self.mutate_target == name and (self.mutate_field.kind == FIELD_VEC3 or self.mutate_field.kind == FIELD_COLOR_RGB):
                value = self.mutate_field.v3_val
                self.mutated = True
            return
        var f = InspectedField(name, k)
        f.v3_val = value
        self.fields.append(f^)

    def visit_quat(mut self, name: String, mut value: Quat):
        if self.mutate_target.byte_length() > 0:
            if self.mutate_target == name and self.mutate_field.kind == FIELD_QUAT:
                value = self.mutate_field.q_val.normalized()
                self.mutated = True
            return
        var f = InspectedField(name, FIELD_QUAT)
        f.q_val = value
        self.fields.append(f^)


trait Inspectable:
    def inspect_fields(mut self, mut visitor: FieldVisitor):
        ...


# ============================================================================
# 2. Studio ECS Components (`SerializableComponent`) & Derived Compat View
# ============================================================================

struct StudioEntityMetaComponent(SerializableComponent):
    """Authoritative Studio editor metadata stored in `flint.ecs.World`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.studio.StudioEntityMetaComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "StudioEntityMetaComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("SMET")
    comptime SCHEMA_VERSION: Int = 2

    var name: String
    var locked: Bool
    var enabled: Bool
    var mesh_handle_packed: UInt64
    var material_handle_packed: UInt64
    var collider_kind: Int
    var collider_half_extents: Vec3
    var restitution: Float32
    var friction: Float32
    var bvh_slot: Int

    def __init__(
        out self,
        name: String = "Entity",
        locked: Bool = False,
        mesh_handle_packed: UInt64 = 0,
        material_handle_packed: UInt64 = 0,
        collider_kind: Int = COLLIDER_BOX,
        collider_half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        restitution: Float32 = 0.32,
        friction: Float32 = 0.55,
        bvh_slot: Int = -1,
        enabled: Bool = True,
    ):
        self.name = name
        self.locked = locked
        self.enabled = enabled
        self.mesh_handle_packed = mesh_handle_packed
        self.material_handle_packed = material_handle_packed
        self.collider_kind = collider_kind
        self.collider_half_extents = collider_half_extents
        self.restitution = restitution
        self.friction = friction
        self.bvh_slot = bvh_slot

    def write_bin(self, mut out: BlobWriter):
        out.text(self.name)
        out.boolean(self.locked)
        out.u64(self.mesh_handle_packed)
        out.u64(self.material_handle_packed)
        out.integer(self.collider_kind)
        out.vec3(self.collider_half_extents)
        out.f32(self.restitution)
        out.f32(self.friction)
        out.integer(self.bvh_slot)
        out.boolean(self.enabled)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var nm = reader.text()
        var lk = reader.boolean()
        var mh = reader.u64()
        var mat_h = reader.u64()
        var ck = reader.integer()
        var che = reader.vec3()
        var rest = reader.f32()
        var fric = reader.f32()
        var bvh = reader.integer()
        var en = reader.boolean() if version >= 2 else True
        if reader.failed:
            raise Error("Truncated StudioEntityMetaComponent")
        return Self(nm, lk, mh, mat_h, ck, che, rest, fric, bvh, en)

    def write_json(self) -> String:
        return (
            "{\"name\": " + json_quote(self.name)
            + ", \"locked\": " + ("true" if self.locked else "false")
            + ", \"mesh_handle_packed\": " + String(self.mesh_handle_packed)
            + ", \"material_handle_packed\": " + String(self.material_handle_packed)
            + ", \"collider_kind\": " + String(self.collider_kind)
            + ", \"collider_half_extents\": " + _vec3_json(self.collider_half_extents)
            + ", \"restitution\": " + _f32_s(self.restitution)
            + ", \"friction\": " + _f32_s(self.friction)
            + ", \"bvh_slot\": " + String(self.bvh_slot)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var nm = doc.text(doc.object_get(node, "name"), "Entity")
        var lk = doc.boolean(doc.object_get(node, "locked"), False)
        var mh = doc.u64(doc.object_get(node, "mesh_handle_packed"), 0)
        var mat_h = doc.u64(doc.object_get(node, "material_handle_packed"), 0)
        var ck = doc.integer(doc.object_get(node, "collider_kind"), COLLIDER_BOX)
        var che = _read_vec3_json(doc, doc.object_get(node, "collider_half_extents"), Vec3(0.5, 0.5, 0.5))
        var rest = Float32(doc.number(doc.object_get(node, "restitution"), 0.32))
        var fric = Float32(doc.number(doc.object_get(node, "friction"), 0.55))
        var bvh = doc.integer(doc.object_get(node, "bvh_slot"), -1)
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        return Self(nm, lk, mh, mat_h, ck, che, rest, fric, bvh, en)


struct StudioStyleComponent(SerializableComponent, ImplicitlyCopyable):
    """Authoritative Studio NPR Cel / Stylized VFX / Clearcoat / Stepped-Anim parameters in `World`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.studio.StudioStyleComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "StudioStyleComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("SSTY")
    comptime SCHEMA_VERSION: Int = 1

    var shading_mode: Int
    var cel_shadow_tint: Vec3
    var cel_outline_width_px: Float32
    var cel_outline_color: Vec3
    var cel_rim_color: Vec3
    var cel_hair_halo: Float32
    var vfx_erosion: Float32
    var vfx_scroll_speed: Float32
    var vfx_core_color: Vec3
    var vfx_edge_color: Vec3
    var anim_step_hz: Float32
    var smear_stretch: Float32
    var clearcoat: Float32
    var clearcoat_roughness: Float32

    def __init__(out self):
        self.shading_mode = SHADING_PBR
        self.cel_shadow_tint = Vec3(0.55, 0.38, 0.44)
        self.cel_outline_width_px = Float32(2.0)
        self.cel_outline_color = Vec3(0.015, 0.012, 0.018)
        self.cel_rim_color = Vec3(0.45, 0.52, 0.68)
        self.cel_hair_halo = Float32(0.0)
        self.vfx_erosion = Float32(0.35)
        self.vfx_scroll_speed = Float32(3.2)
        self.vfx_core_color = Vec3(8.5, 6.2, 2.2)
        self.vfx_edge_color = Vec3(4.2, 0.30, 0.04)
        self.anim_step_hz = Float32(0.0)
        self.smear_stretch = Float32(1.0)
        self.clearcoat = Float32(0.0)
        self.clearcoat_roughness = Float32(0.05)

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.shading_mode)
        out.vec3(self.cel_shadow_tint)
        out.f32(self.cel_outline_width_px)
        out.vec3(self.cel_outline_color)
        out.vec3(self.cel_rim_color)
        out.f32(self.cel_hair_halo)
        out.f32(self.vfx_erosion)
        out.f32(self.vfx_scroll_speed)
        out.vec3(self.vfx_core_color)
        out.vec3(self.vfx_edge_color)
        out.f32(self.anim_step_hz)
        out.f32(self.smear_stretch)
        out.f32(self.clearcoat)
        out.f32(self.clearcoat_roughness)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var s = Self()
        s.shading_mode = reader.integer()
        s.cel_shadow_tint = reader.vec3()
        s.cel_outline_width_px = reader.f32()
        s.cel_outline_color = reader.vec3()
        s.cel_rim_color = reader.vec3()
        s.cel_hair_halo = reader.f32()
        s.vfx_erosion = reader.f32()
        s.vfx_scroll_speed = reader.f32()
        s.vfx_core_color = reader.vec3()
        s.vfx_edge_color = reader.vec3()
        s.anim_step_hz = reader.f32()
        s.smear_stretch = reader.f32()
        s.clearcoat = reader.f32()
        s.clearcoat_roughness = reader.f32()
        if reader.failed:
            raise Error("Truncated StudioStyleComponent")
        return s^

    def write_json(self) -> String:
        return (
            "{\"shading_mode\": " + String(self.shading_mode)
            + ", \"cel_shadow_tint\": " + _vec3_json(self.cel_shadow_tint)
            + ", \"cel_outline_width_px\": " + _f32_s(self.cel_outline_width_px)
            + ", \"cel_outline_color\": " + _vec3_json(self.cel_outline_color)
            + ", \"cel_rim_color\": " + _vec3_json(self.cel_rim_color)
            + ", \"cel_hair_halo\": " + _f32_s(self.cel_hair_halo)
            + ", \"vfx_erosion\": " + _f32_s(self.vfx_erosion)
            + ", \"vfx_scroll_speed\": " + _f32_s(self.vfx_scroll_speed)
            + ", \"vfx_core_color\": " + _vec3_json(self.vfx_core_color)
            + ", \"vfx_edge_color\": " + _vec3_json(self.vfx_edge_color)
            + ", \"anim_step_hz\": " + _f32_s(self.anim_step_hz)
            + ", \"smear_stretch\": " + _f32_s(self.smear_stretch)
            + ", \"clearcoat\": " + _f32_s(self.clearcoat)
            + ", \"clearcoat_roughness\": " + _f32_s(self.clearcoat_roughness)
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var s = Self()
        s.shading_mode = doc.integer(doc.object_get(node, "shading_mode"), SHADING_PBR)
        s.cel_shadow_tint = _read_vec3_json(doc, doc.object_get(node, "cel_shadow_tint"), s.cel_shadow_tint)
        s.cel_outline_width_px = Float32(doc.number(doc.object_get(node, "cel_outline_width_px"), Float64(s.cel_outline_width_px)))
        s.cel_outline_color = _read_vec3_json(doc, doc.object_get(node, "cel_outline_color"), s.cel_outline_color)
        s.cel_rim_color = _read_vec3_json(doc, doc.object_get(node, "cel_rim_color"), s.cel_rim_color)
        s.cel_hair_halo = Float32(doc.number(doc.object_get(node, "cel_hair_halo"), Float64(s.cel_hair_halo)))
        s.vfx_erosion = Float32(doc.number(doc.object_get(node, "vfx_erosion"), Float64(s.vfx_erosion)))
        s.vfx_scroll_speed = Float32(doc.number(doc.object_get(node, "vfx_scroll_speed"), Float64(s.vfx_scroll_speed)))
        s.vfx_core_color = _read_vec3_json(doc, doc.object_get(node, "vfx_core_color"), s.vfx_core_color)
        s.vfx_edge_color = _read_vec3_json(doc, doc.object_get(node, "vfx_edge_color"), s.vfx_edge_color)
        s.anim_step_hz = Float32(doc.number(doc.object_get(node, "anim_step_hz"), Float64(s.anim_step_hz)))
        s.smear_stretch = Float32(doc.number(doc.object_get(node, "smear_stretch"), Float64(s.smear_stretch)))
        s.clearcoat = Float32(doc.number(doc.object_get(node, "clearcoat"), Float64(s.clearcoat)))
        s.clearcoat_roughness = Float32(doc.number(doc.object_get(node, "clearcoat_roughness"), Float64(s.clearcoat_roughness)))
        return s


struct StudioEditorIdComponent(SerializableComponent, ImplicitlyCopyable):
    """Stable editor identity for an entity (`editor_id`), saved with the scene and with undo blobs.

    ECS slot indices are reused after a despawn, so a slot remembered by a deferred action (an agent
    approval, an MCP client, a pending inspector edit) can silently point at a different entity later.
    `editor_id` is never reused within a scene: references hold it (see `EntityRef`) and resolve it
    back to the current slot through `FlintStudioState.slot_for_editor_id`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.studio.StudioEditorIdComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "StudioEditorIdComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("SEID")
    comptime SCHEMA_VERSION: Int = 1

    var editor_id: UInt64

    def __init__(out self, editor_id: UInt64 = 0):
        self.editor_id = editor_id

    def write_bin(self, mut out: BlobWriter):
        out.u64(self.editor_id)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var v = reader.u64()
        if reader.failed:
            raise Error("Truncated StudioEditorIdComponent")
        return Self(v)

    def write_json(self) -> String:
        return "{\"editor_id\": " + String(self.editor_id) + "}"

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        return Self(doc.u64(doc.object_get(node, "editor_id"), 0))


@fieldwise_init
struct EntityRef(Copyable, ImplicitlyCopyable, Movable):
    """A reference to an entity that survives slot reuse: the stable `editor_id` plus the slot and
    generation it had when the reference was taken (a fast path, re-checked on resolve)."""
    var editor_id: UInt64
    var slot: Int
    var generation: UInt32

    @staticmethod
    def null() -> Self:
        return Self(UInt64(0), -1, UInt32(0))

    @always_inline
    def is_null(self) -> Bool:
        return self.editor_id == 0 and self.slot < 0


struct StudioEntityView(Copyable, Movable, Inspectable):
    """Derived non-authoritative ECS projection view for Inspector / MCP / compat access."""
    var entity_id: Int
    var alive: Bool
    var name: String
    var parent_id: Int
    var translation: Vec3
    var rotation: Quat
    var scale: Vec3
    var linear_velocity: Vec3
    var angular_velocity: Vec3
    var mesh_handle: UntypedHandle
    var mesh_id: Int
    var material_handle: UntypedHandle
    var material_id: Int
    var base_color: Vec3
    var roughness: Float32
    var metallic: Float32
    var emissive: Vec3
    var opacity: Float32
    var alpha_cutoff: Float32
    var clearcoat: Float32
    var clearcoat_roughness: Float32
    var sheen: Float32
    var subsurface: Float32
    var collider_kind: Int
    var collider_half_extents: Vec3
    var is_trigger: Bool
    var mass: Float32
    var restitution: Float32
    var friction: Float32
    var visible: Bool
    var enabled: Bool
    var locked: Bool
    var shading_mode: Int
    var cel_shadow_tint: Vec3
    var cel_outline_width_px: Float32
    var cel_outline_color: Vec3
    var cel_rim_color: Vec3
    var cel_hair_halo: Float32
    var vfx_erosion: Float32
    var vfx_scroll_speed: Float32
    var vfx_core_color: Vec3
    var vfx_edge_color: Vec3
    var anim_step_hz: Float32
    var smear_stretch: Float32
    var material_override_enabled: Bool
    var has_mesh_renderer: Bool
    var mesh_ref: AssetRef
    var material_ref: AssetRef
    var override_material_ref: AssetRef
    var has_rig: Bool
    var rig_ref: AssetRef
    var rig_bound: Bool
    var rig_joint_count: Int
    var has_skeleton: Bool
    var skeleton_node_count: Int
    var skeleton_model_id: UInt64
    var has_animator: Bool
    var graph_ref: AssetRef
    var clip_index: Int
    var clip_name: String
    var anim_playing: Bool
    var anim_loop: Bool
    var anim_speed: Float32
    var has_bone_socket: Bool
    var socket_name: String
    var socket_bone_node: Int
    var has_audio: Bool
    var audio_ref: AssetRef
    var audio_playing: Bool
    var has_prefab: Bool
    var prefab_ref: AssetRef
    var prefab_name: String

    def __init__(out self, entity_id: Int = 0, name: String = "Entity", alive: Bool = True):
        self.entity_id = entity_id
        self.alive = alive
        self.name = name
        self.parent_id = -1
        self.translation = Vec3(0.0, 0.0, 0.0)
        self.rotation = Quat(0.0, 0.0, 0.0, 1.0)
        self.scale = Vec3(1.0, 1.0, 1.0)
        self.linear_velocity = Vec3(0.0, 0.0, 0.0)
        self.angular_velocity = Vec3(0.0, 0.0, 0.0)
        self.mesh_handle = UntypedHandle()
        self.mesh_id = 0
        self.material_handle = UntypedHandle()
        self.material_id = 0
        self.base_color = Vec3(0.82, 0.84, 0.88)
        self.roughness = Float32(0.38)
        self.metallic = Float32(0.10)
        self.emissive = Vec3(0.0, 0.0, 0.0)
        self.opacity = Float32(1.0)
        self.alpha_cutoff = Float32(0.0)
        self.clearcoat = Float32(0.0)
        self.clearcoat_roughness = Float32(0.08)
        self.sheen = Float32(0.0)
        self.subsurface = Float32(0.0)
        self.collider_kind = COLLIDER_BOX
        self.collider_half_extents = Vec3(0.5, 0.5, 0.5)
        self.is_trigger = False
        self.mass = Float32(0.0)
        self.restitution = Float32(0.32)
        self.friction = Float32(0.55)
        self.visible = True
        self.enabled = True
        self.locked = False
        self.shading_mode = SHADING_PBR
        self.cel_shadow_tint = Vec3(0.55, 0.38, 0.44)
        self.cel_outline_width_px = Float32(2.0)
        self.cel_outline_color = Vec3(0.015, 0.012, 0.018)
        self.cel_rim_color = Vec3(0.45, 0.52, 0.68)
        self.cel_hair_halo = Float32(0.0)
        self.vfx_erosion = Float32(0.35)
        self.vfx_scroll_speed = Float32(3.2)
        self.vfx_core_color = Vec3(8.5, 6.2, 2.2)
        self.vfx_edge_color = Vec3(4.2, 0.30, 0.04)
        self.anim_step_hz = Float32(0.0)
        self.smear_stretch = Float32(1.0)
        self.material_override_enabled = True
        self.has_mesh_renderer = False
        self.mesh_ref = AssetRef()
        self.material_ref = AssetRef()
        self.override_material_ref = AssetRef()
        self.has_rig = False
        self.rig_ref = AssetRef()
        self.rig_bound = False
        self.rig_joint_count = 0
        self.has_skeleton = False
        self.skeleton_node_count = 0
        self.skeleton_model_id = 0
        self.has_animator = False
        self.graph_ref = AssetRef()
        self.clip_index = -1
        self.clip_name = ""
        self.anim_playing = True
        self.anim_loop = True
        self.anim_speed = Float32(1.0)
        self.has_bone_socket = False
        self.socket_name = ""
        self.socket_bone_node = -1
        self.has_audio = False
        self.audio_ref = AssetRef()
        self.audio_playing = False
        self.has_prefab = False
        self.prefab_ref = AssetRef()
        self.prefab_name = ""

    def local_matrix(self) -> Mat4:
        return trs(self.translation, self.rotation, self.scale)

    def inspect_fields(mut self, mut visitor: FieldVisitor):
        visitor.visit_string("name", self.name)
        visitor.visit_int("parent_id", self.parent_id)
        visitor.visit_vec3("translation", self.translation)
        visitor.visit_quat("rotation", self.rotation)
        visitor.visit_vec3("scale", self.scale)
        visitor.visit_vec3("base_color", self.base_color, is_color=True)
        visitor.visit_f32("roughness", self.roughness)
        visitor.visit_f32("metallic", self.metallic)
        visitor.visit_vec3("emissive", self.emissive, is_color=True)
        visitor.visit_f32("opacity", self.opacity)
        visitor.visit_f32("alpha_cutoff", self.alpha_cutoff)
        visitor.visit_f32("clearcoat", self.clearcoat)
        visitor.visit_f32("clearcoat_roughness", self.clearcoat_roughness)
        visitor.visit_f32("sheen", self.sheen)
        visitor.visit_f32("subsurface", self.subsurface)
        visitor.visit_int("collider_kind", self.collider_kind)
        visitor.visit_vec3("collider_half_extents", self.collider_half_extents)
        visitor.visit_bool("is_trigger", self.is_trigger)
        visitor.visit_f32("mass", self.mass)
        visitor.visit_f32("restitution", self.restitution)
        visitor.visit_f32("friction", self.friction)
        visitor.visit_bool("visible", self.visible)
        visitor.visit_bool("enabled", self.enabled)
        visitor.visit_bool("locked", self.locked)
        visitor.visit_int("shading_mode", self.shading_mode)
        visitor.visit_vec3("cel_shadow_tint", self.cel_shadow_tint, is_color=True)
        visitor.visit_f32("cel_outline_width_px", self.cel_outline_width_px)
        visitor.visit_vec3("cel_outline_color", self.cel_outline_color, is_color=True)
        visitor.visit_vec3("cel_rim_color", self.cel_rim_color, is_color=True)
        visitor.visit_f32("cel_hair_halo", self.cel_hair_halo)
        visitor.visit_f32("vfx_erosion", self.vfx_erosion)
        visitor.visit_f32("vfx_scroll_speed", self.vfx_scroll_speed)
        visitor.visit_vec3("vfx_core_color", self.vfx_core_color, is_color=True)
        visitor.visit_vec3("vfx_edge_color", self.vfx_edge_color, is_color=True)
        visitor.visit_f32("anim_step_hz", self.anim_step_hz)
        visitor.visit_f32("smear_stretch", self.smear_stretch)

    def inspect(mut self, mut v: FieldVisitor):
        self.inspect_fields(v)


@fieldwise_init
struct HierarchyRow(Copyable, ImplicitlyCopyable, Movable):
    var entity_id: Int
    var parent_id: Int
    var depth: Int
    var has_children: Bool
    var expanded: Bool


@fieldwise_init
struct SelectedTransformSnap(Copyable, ImplicitlyCopyable, Movable):
    var entity_id: Int
    var local_pos: Vec3
    var local_rot: Quat
    var local_scl: Vec3
    var world_pos: Vec3
    var world_rot: Quat
    var world_scl: Vec3
    var parent_id: Int


# ============================================================================
# 3. Reversible Coalesced `EditorTransaction` & `PieSession`
# ============================================================================

def _copy_str_u64_dict(src: Dict[String, UInt64]) -> Dict[String, UInt64]:
    var dst = Dict[String, UInt64]()
    for item in src.items():
        dst[item.key] = item.value
    return dst^


def _copy_u64_style_epoch_dict(
    src: Dict[UInt64, Tuple[String, UInt64]]
) -> Dict[UInt64, Tuple[String, UInt64]]:
    var dst = Dict[UInt64, Tuple[String, UInt64]]()
    for item in src.items():
        dst[item.key] = item.value
    return dst^


@always_inline
def _studio_mat_ref_key(aref: AssetRef) -> String:
    if aref.rel_path.byte_length() > 0:
        var norm = normalize_asset_rel_path(aref.rel_path)
        return norm[1] if norm[0] else aref.rel_path
    if not aref.uuid.is_nil():
        return String(aref.uuid.hi) + ":" + String(aref.uuid.lo)
    return String()


def _apply_flmat_to_studio_style(mut st: StudioStyleComponent, flmat: FlmatAsset):
    st.shading_mode = flmat.shading_mode
    st.clearcoat = flmat.clearcoat
    st.clearcoat_roughness = flmat.clearcoat_roughness
    st.cel_shadow_tint = flmat.cel_params.sss_shadow_tint
    st.cel_outline_width_px = flmat.cel_params.outline_width_px
    st.cel_outline_color = flmat.cel_params.outline_color
    st.cel_rim_color = flmat.cel_params.rim_color
    st.cel_hair_halo = flmat.cel_params.hair_halo_strength
    st.vfx_erosion = flmat.vfx_params.erosion
    st.vfx_scroll_speed = flmat.vfx_params.uv_scroll_v
    st.vfx_core_color = flmat.vfx_params.core_color
    st.vfx_edge_color = flmat.vfx_params.edge_color


struct StudioFlmatStyleSyncResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.studio.StudioFlmatStyleSyncResource")

    var synced_entity_epochs: Dict[UInt64, Tuple[String, UInt64]]
    var known_internal_mat_epochs: Dict[String, UInt64]
    var external_mat_reload_epochs: Dict[String, UInt64]

    def __init__(out self):
        self.synced_entity_epochs = Dict[UInt64, Tuple[String, UInt64]]()
        self.known_internal_mat_epochs = Dict[String, UInt64]()
        self.external_mat_reload_epochs = Dict[String, UInt64]()


def _sync_studio_flmat_styles(mut world: World, mut resources: Resources):
    var srv_opt = resources.get_ptr_mut[AssetServer]()
    var res_opt = resources.get_ptr[SceneAssetResolver]()
    var sync_opt = resources.get_ptr_mut[StudioFlmatStyleSyncResource]()
    if (not srv_opt) or (not res_opt) or (not sync_opt):
        return
    ref srv = srv_opt.value()[unsafe_offset=0]
    ref res = res_opt.value()[unsafe_offset=0]
    ref sync = sync_opt.value()[unsafe_offset=0]

    for item in res.mat_epochs.items():
        var k = item.key
        var ep = item.value
        if ep == 0:
            continue
        var ki = sync.known_internal_mat_epochs.get(k)
        if not ki:
            if ep == UInt64(1):
                sync.known_internal_mat_epochs[k] = ep
            else:
                sync.external_mat_reload_epochs[k] = ep
        elif ki.value() != ep:
            sync.external_mat_reload_epochs[k] = ep

    if world.component_id[StudioStyleComponent]() < 0:
        return
    var has_mo = world.component_id[MaterialOverrideComponent]() >= 0
    var has_mr = world.component_id[MeshRendererComponent]() >= 0
    if (not has_mo) and (not has_mr):
        return
    var st_ents = world.query_entities(world.bit_for[StudioStyleComponent]())
    for i in range(len(st_ents)):
        var e = st_ents[i]
        var ebits = e.to_bits()
        var mat_ref = AssetRef()
        if has_mo:
            var mo_p = world.get_ptr[MaterialOverrideComponent](e)
            if mo_p and not mo_p.value()[].material_ref.is_empty():
                mat_ref = mo_p.value()[].material_ref.copy()
        if mat_ref.is_empty() and has_mr:
            var mr_p = world.get_ptr[MeshRendererComponent](e)
            if mr_p and not mr_p.value()[].material_ref.is_empty():
                mat_ref = mr_p.value()[].material_ref.copy()
        if mat_ref.is_empty():
            _ = sync.synced_entity_epochs.pop(ebits, (String(), UInt64(0)))
            continue
        var key = _studio_mat_ref_key(mat_ref)
        var ep_opt = res.mat_epochs.get(key)
        if (not ep_opt) or ep_opt.value() == 0:
            continue
        var cur_ep = ep_opt.value()
        var prev_opt = sync.synced_entity_epochs.get(ebits)
        var needs_sync = (
            (not prev_opt)
            or prev_opt.value()[0] != key
            or prev_opt.value()[1] != cur_ep
        )
        if not needs_sync:
            continue
        var rh = srv.resolve_raw(mat_ref)
        if not srv.raws.is_loaded(rh):
            continue
        var rptr = srv.raws.get_ptr(rh)
        if not rptr:
            continue
        try:
            var flmat = load_flmat_bytes(rptr.value()[].bytes)
            var st_p = world.get_ptr_mut[StudioStyleComponent](e)
            if st_p:
                _apply_flmat_to_studio_style(st_p.value()[unsafe_offset=0], flmat)
            sync.synced_entity_epochs[ebits] = (key, cur_ep)
        except:
            pass


def _sys_studio_sync_flmat_styles(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    pool_ptr: Pointer[JobPool, MutAnyOrigin],
    cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    _ = pool_ptr
    _ = cmds_ptr
    _ = ctx
    _sync_studio_flmat_styles(world_ptr[], res_ptr[])


struct EditorTransaction(Copyable, Movable):
    var op: Int
    var entity_id: Int
    var affected_ids: List[Int]
    var before_world_blob: List[UInt8]
    var after_world_blob: List[UInt8]
    var prev_selected_id: Int
    var next_selected_id: Int
    var prev_selected_ids: List[Int]
    var next_selected_ids: List[Int]
    var prev_anchor_id: Int
    var next_anchor_id: Int
    var coalesce_key: UInt64
    var label: String
    var external_mat_reload_epochs: Dict[String, UInt64]
    var tx_id: UInt64
    var agent_id: Int
    var step_id: UInt64
    var changes_scene_path: Bool
    """A scene load: undo / redo also switch `scene_file_path`, so a save after undoing a load writes
    the scene the user is looking at to *its* file, never over the file that was loaded."""
    var prev_scene_path: String
    var next_scene_path: String
    var prev_saved_tx_id: UInt64
    """The save point before a load, so undoing the load knows whether the old file is clean again."""
    var load_leaves_clean: Bool
    """A plain load (not an autosave recovery) leaves the scene matching its file, so redo is clean."""

    def __init__(out self, op: Int = TX_PATCH_ENTITY, entity_id: Int = 0, label: String = "Edit"):
        self.op = op
        self.entity_id = entity_id
        self.affected_ids = List[Int]()
        self.before_world_blob = List[UInt8]()
        self.after_world_blob = List[UInt8]()
        self.prev_selected_id = -1
        self.next_selected_id = -1
        self.prev_selected_ids = List[Int]()
        self.next_selected_ids = List[Int]()
        self.prev_anchor_id = -1
        self.next_anchor_id = -1
        self.coalesce_key = 0
        self.label = label
        self.external_mat_reload_epochs = Dict[String, UInt64]()
        self.tx_id = UInt64(0)
        self.agent_id = 0
        self.step_id = UInt64(0)
        self.changes_scene_path = False
        self.prev_scene_path = String()
        self.next_scene_path = String()
        self.prev_saved_tx_id = UInt64(0xFFFFFFFFFFFFFFFF)
        self.load_leaves_clean = False

    def __init__(out self, *, copy: Self):
        self.op = copy.op
        self.entity_id = copy.entity_id
        self.affected_ids = copy.affected_ids.copy()
        self.before_world_blob = copy.before_world_blob.copy()
        self.after_world_blob = copy.after_world_blob.copy()
        self.prev_selected_id = copy.prev_selected_id
        self.next_selected_id = copy.next_selected_id
        self.prev_selected_ids = copy.prev_selected_ids.copy()
        self.next_selected_ids = copy.next_selected_ids.copy()
        self.prev_anchor_id = copy.prev_anchor_id
        self.next_anchor_id = copy.next_anchor_id
        self.coalesce_key = copy.coalesce_key
        self.label = copy.label
        self.external_mat_reload_epochs = _copy_str_u64_dict(copy.external_mat_reload_epochs)
        self.tx_id = copy.tx_id
        self.agent_id = copy.agent_id
        self.step_id = copy.step_id
        self.changes_scene_path = copy.changes_scene_path
        self.prev_scene_path = copy.prev_scene_path
        self.next_scene_path = copy.next_scene_path
        self.prev_saved_tx_id = copy.prev_saved_tx_id
        self.load_leaves_clean = copy.load_leaves_clean

    def approx_bytes(self) -> Int:
        return len(self.before_world_blob) + len(self.after_world_blob) + 128


struct PieSession(Copyable, Movable):
    """1-Click Play-In-Editor session backed by `World.snapshot_blob`, `PhysicsWorld.serialize`, and LZ4."""
    var mode: Int
    var checkpoint_lz4: List[UInt8]
    var pre_play_hash: UInt64
    var ticks_simulated: Int
    var last_restore_verified: Bool

    def __init__(out self):
        self.mode = PIE_STOPPED
        self.checkpoint_lz4 = List[UInt8]()
        self.pre_play_hash = 0
        self.ticks_simulated = 0
        self.last_restore_verified = False


@fieldwise_init
struct StudioRaycastHit(Copyable, ImplicitlyCopyable, Movable):
    var hit: Bool
    var entity_id: Int
    var distance: Float32
    var point: Vec3
    var normal: Vec3

    def __init__(out self):
        self.hit = False
        self.entity_id = -1
        self.distance = Float32(1.0e30)
        self.point = Vec3()
        self.normal = Vec3(0.0, 1.0, 0.0)


def _push_unique_free_slot(mut pool: List[Int], slot_id: Int):
    if slot_id < 0:
        return
    for i in range(len(pool)):
        if pool[i] == slot_id:
            return
    pool.append(slot_id)


struct StudioGpuMaterials(Movable):
    """Cached GPU PBR/NPR-Cel/VFX material slot mappings per Studio entity slot."""
    var grid_mesh_id: Int
    var orb_mesh_id: Int
    var torus_mesh_id: Int
    var colonnade_mesh_id: Int
    var foliage_trunk_id: Int
    var foliage_canopy_id: Int
    var mat_trunk_id: Int
    var mat_foliage_id: Int
    var entity_pbr_mat_ids: List[Int]
    var entity_cel_mat_ids: List[Int]
    var entity_vfx_mat_ids: List[Int]
    var entity_generations: List[UInt32]
    var entity_style_hashes: List[UInt64]
    var free_pbr_mat_ids: List[Int]
    var free_cel_mat_ids: List[Int]
    var free_vfx_mat_ids: List[Int]
    var uploaded: Bool

    def __init__(out self):
        self.grid_mesh_id = 0
        self.orb_mesh_id = 0
        self.torus_mesh_id = 0
        self.colonnade_mesh_id = 0
        self.foliage_trunk_id = 0
        self.foliage_canopy_id = 0
        self.mat_trunk_id = 0
        self.mat_foliage_id = 0
        self.entity_pbr_mat_ids = List[Int]()
        self.entity_cel_mat_ids = List[Int]()
        self.entity_vfx_mat_ids = List[Int]()
        self.entity_generations = List[UInt32]()
        self.entity_style_hashes = List[UInt64]()
        self.free_pbr_mat_ids = List[Int]()
        self.free_cel_mat_ids = List[Int]()
        self.free_vfx_mat_ids = List[Int]()
        self.uploaded = False

    def release_pbr_slot(mut self, index: Int):
        if index >= 0 and index < len(self.entity_pbr_mat_ids):
            _push_unique_free_slot(self.free_pbr_mat_ids, self.entity_pbr_mat_ids[index])
            self.entity_pbr_mat_ids[index] = -1

    def release_cel_slot(mut self, index: Int):
        if index >= 0 and index < len(self.entity_cel_mat_ids):
            _push_unique_free_slot(self.free_cel_mat_ids, self.entity_cel_mat_ids[index])
            self.entity_cel_mat_ids[index] = -1

    def release_vfx_slot(mut self, index: Int):
        if index >= 0 and index < len(self.entity_vfx_mat_ids):
            _push_unique_free_slot(self.free_vfx_mat_ids, self.entity_vfx_mat_ids[index])
            self.entity_vfx_mat_ids[index] = -1

    def clear_entity_slots(mut self):
        for i in range(len(self.entity_pbr_mat_ids) - 1, -1, -1):
            _push_unique_free_slot(self.free_pbr_mat_ids, self.entity_pbr_mat_ids[i])
        for i in range(len(self.entity_cel_mat_ids) - 1, -1, -1):
            _push_unique_free_slot(self.free_cel_mat_ids, self.entity_cel_mat_ids[i])
        for i in range(len(self.entity_vfx_mat_ids) - 1, -1, -1):
            _push_unique_free_slot(self.free_vfx_mat_ids, self.entity_vfx_mat_ids[i])
        self.entity_pbr_mat_ids.clear()
        self.entity_cel_mat_ids.clear()
        self.entity_vfx_mat_ids.clear()
        self.entity_generations.clear()
        self.entity_style_hashes.clear()

    def invalidate_entity(mut self, index: Int):
        if index < 0:
            return
        self.release_pbr_slot(index)
        self.release_cel_slot(index)
        self.release_vfx_slot(index)
        if index < len(self.entity_generations):
            self.entity_generations[index] = 0
        if index < len(self.entity_style_hashes):
            self.entity_style_hashes[index] = 0


def _studio_mat_hash(
    st: StudioStyleComponent,
    mo: MaterialOverrideComponent,
    base_mat_id: Int,
    renderer: ScalableRenderer,
) -> UInt64:
    var w = BlobWriter()
    w.integer(st.shading_mode)
    w.vec3(st.cel_shadow_tint)
    w.f32(st.cel_outline_width_px)
    w.vec3(st.cel_outline_color)
    w.vec3(st.cel_rim_color)
    w.f32(st.cel_hair_halo)
    w.f32(st.vfx_erosion)
    w.f32(st.vfx_scroll_speed)
    w.vec3(st.vfx_core_color)
    w.vec3(st.vfx_edge_color)
    w.f32(st.clearcoat)
    w.f32(st.clearcoat_roughness)
    w.boolean(mo.enabled)
    w.integer(base_mat_id)
    if base_mat_id >= 0 and base_mat_id < len(renderer.materials):
        ref bm = renderer.materials[base_mat_id]
        w.integer(Int(bm.albedo_slot))
        w.integer(Int(bm.normal_slot))
        w.integer(Int(bm.orm_slot))
        w.integer(Int(bm.emissive_slot))
        w.u32(bm.domain)
        w.u32(bm.flags)
        w.vec3(bm.base_color)
        w.f32(bm.roughness)
        w.f32(bm.metallic)
        w.vec3(bm.emissive)
    w.vec3(mo.base_color)
    w.f32(mo.roughness)
    w.f32(mo.metallic)
    w.vec3(mo.emissive)
    w.f32(mo.opacity)
    w.f32(mo.alpha_cutoff)
    w.f32(mo.clearcoat)
    w.f32(mo.clearcoat_roughness)
    w.f32(mo.sheen)
    w.f32(mo.subsurface)
    var h = fnv64(w.data)
    return h if h != 0 else UInt64(1)


def _build_studio_material_asset(
    renderer: ScalableRenderer,
    base_mat_id: Int,
    mo: MaterialOverrideComponent,
    eff_ov: MaterialOverrideComponent,
    st: StudioStyleComponent,
) -> MaterialAsset:
    if mo.enabled:
        return build_overridden_material(renderer, base_mat_id, eff_ov)
    var ma = (
        renderer.materials[base_mat_id].copy()
        if (base_mat_id >= 0 and base_mat_id < len(renderer.materials))
        else MaterialAsset()
    )
    if isfinite(st.clearcoat) and st.clearcoat > Float32(0.01):
        ma.clearcoat = max(Float32(0.0), min(Float32(1.0), st.clearcoat))
        if isfinite(st.clearcoat_roughness):
            ma.clearcoat_roughness = max(
                Float32(0.04), min(Float32(1.0), st.clearcoat_roughness)
            )
        ma.flags |= MAT_FLAG_CLEARCOAT
        if ma.domain == MAT_DOMAIN_STANDARD_PBR:
            ma.domain = MAT_DOMAIN_CLEARCOAT
    return ma^


# ============================================================================
# 4. `FlintStudioState`: EngineApp + World + Scene + PhysicsWorld + Multi-Select
# ============================================================================

def _studio_layout_path() -> String:
    return state_directory("flint") + "/studio_layout.json"


def _create_studio_dockspace() -> DockSpace:
    """Builds the default Studio v2 Edit `DockSpace` layout (`Hierarchy` left,
    `SceneViewport` center, `Inspector`/`Agents`/`CameraRig` right,
    `AssetBrowser`/`CodeDrawer`/`InputMap`/`MCP v2` bottom)."""
    return build_workspace_dockspace(WORKSPACE_EDIT, True)


def _ensure_dock_panel_tab(
    mut d: DockSpace,
    panel_name: String,
    preferred_sibling: String = "AssetBrowser",
):
    if d.find_panel(panel_name)[0] >= 0:
        return
    var sib = d.find_panel(preferred_sibling)
    if sib[0] >= 0 and sib[0] < len(d.nodes) and d.nodes[sib[0]].kind == DOCK_TABS:
        d.nodes[sib[0]].tabs.append(panel_name)
        return
    for ni in range(len(d.nodes)):
        if d.nodes[ni].kind == DOCK_TABS:
            d.nodes[ni].tabs.append(panel_name)
            return


def _load_studio_dockspace() -> DockSpace:
    var d = _create_studio_dockspace()
    var path = _studio_layout_path()
    if file_size(path) <= 0:
        return d^
    var raw = read_bytes(path)
    if len(raw) == 0:
        return d^
    var loaded = _create_studio_dockspace()
    var json_text = String(unsafe_from_utf8=raw)
    var ok = loaded.load_json(json_text)
    if (
        ok
        and len(loaded.nodes) >= 5
        and loaded.find_panel("SceneViewport")[0] >= 0
        and loaded.find_panel("Hierarchy")[0] >= 0
        and loaded.find_panel("Inspector")[0] >= 0
        and loaded.find_panel("AssetBrowser")[0] >= 0
        and loaded.find_panel("Agents")[0] >= 0
        and loaded.find_panel("Toolbar")[0] < 0
    ):
        _ensure_dock_panel_tab(loaded, "CodeDrawer", "AssetBrowser")
        _ensure_dock_panel_tab(loaded, "InputMap", "AssetBrowser")
        _ensure_dock_panel_tab(loaded, "MCP v2", "AssetBrowser")
        return loaded^
    return d^


def _save_studio_dockspace(d: DockSpace):
    if len(d.nodes) < 5:
        return
    var path = _studio_layout_path()
    var pdir = parent_directory(path)
    if pdir.byte_length() > 0:
        _ = make_dirs(pdir)
    var json_text = d.save_json()
    var sb = json_text.as_bytes()
    var out = List[UInt8](capacity=len(sb))
    for i in range(len(sb)):
        out.append(sb[i])
    _ = write_file_durable(path, out)


def _resolve_flint_launcher_path() -> String:
    var env_l = env_value("FLINT_STUDIO_IDE_LAUNCHER")
    if env_l.byte_length() > 0 and file_size(env_l) > 0:
        return env_l
    var flint_root = env_value("FLINT_ROOT")
    while flint_root.byte_length() > 1 and flint_root.endswith("/"):
        var trimmed = String(flint_root[byte=0 : flint_root.byte_length() - 1])
        flint_root = trimmed^
    if flint_root.byte_length() > 0:
        var cand = (flint_root + "/tools/flint.sh") if flint_root != "/" else String("/tools/flint.sh")
        if file_size(cand) > 0:
            return cand
    var exe_rp = real_path("/proc/self/exe")
    if exe_rp[0] and exe_rp[1].byte_length() > 0:
        var dir = parent_directory(exe_rp[1])
        for _ in range(6):
            if dir.byte_length() == 0:
                break
            var cand = (dir + "/tools/flint.sh") if dir != "/" else String("/tools/flint.sh")
            if file_size(cand) > 0:
                return cand
            if dir == "/" or dir == ".":
                break
            dir = parent_directory(dir)
    var cwd_cand = real_path("tools/flint.sh")
    if cwd_cand[0] and cwd_cand[1].byte_length() > 0 and file_size(cwd_cand[1]) > 0:
        return cwd_cand[1]
    var cwd_rp = real_path(".")
    if cwd_rp[0] and cwd_rp[1].byte_length() > 0:
        var dir = cwd_rp[1]
        for _ in range(6):
            if dir.byte_length() == 0:
                break
            var cand = (dir + "/tools/flint.sh") if dir != "/" else String("/tools/flint.sh")
            if file_size(cand) > 0:
                return cand
            if dir == "/" or dir == ".":
                break
            dir = parent_directory(dir)
    return String("")


def _resolve_default_scene_path(project_root: String) -> String:
    var root = project_root
    while root.byte_length() > 1 and root.endswith("/"):
        var trimmed = String(root[byte=0 : root.byte_length() - 1])
        root = trimmed
    if root.byte_length() > 0:
        var assets_main = root + "/assets/main.flscn"
        if file_size(assets_main) > 0:
            return assets_main
        var scenes_main = root + "/scenes/main.flscn"
        if file_size(scenes_main) > 0:
            return scenes_main
        return root + "/scenes/studio_scene.flscn"
    if file_size("assets/main.flscn") > 0:
        return "assets/main.flscn"
    return "scenes/studio_scene.flscn"


def _create_studio_input_map() -> InputMap:
    """Declarative `InputMap` for `Flint Studio`: split `StudioEditor` (prio 10) and `StudioPlay` (prio 20)."""
    var imap = InputMap()
    _ = imap.register_action("gizmo_translate", ACTION_BOOL)
    _ = imap.register_action("gizmo_rotate", ACTION_BOOL)
    _ = imap.register_action("gizmo_scale", ACTION_BOOL)
    _ = imap.register_action("gizmo_toggle_space", ACTION_BOOL)
    _ = imap.register_action("frame_selection", ACTION_BOOL)
    _ = imap.register_action("editor_undo", ACTION_BOOL)
    _ = imap.register_action("editor_redo", ACTION_BOOL)
    _ = imap.register_action("source_editor", ACTION_BOOL)
    _ = imap.register_action("editor_save", ACTION_BOOL)
    _ = imap.register_action("editor_load", ACTION_BOOL)
    _ = imap.register_action("editor_new", ACTION_BOOL)
    _ = imap.register_action("editor_duplicate", ACTION_BOOL)
    _ = imap.register_action("editor_copy", ACTION_BOOL)
    _ = imap.register_action("editor_paste", ACTION_BOOL)
    _ = imap.register_action("editor_delete", ACTION_BOOL)
    _ = imap.register_action("pie_toggle", ACTION_BOOL)
    _ = imap.register_action("pie_stop", ACTION_BOOL)
    _ = imap.register_action("pie_pause", ACTION_BOOL)
    _ = imap.register_action("pie_step", ACTION_BOOL)
    _ = imap.register_action("pie_release_cursor", ACTION_BOOL)
    _ = imap.register_action("pie_toggle_camera", ACTION_BOOL)
    _ = imap.register_action("editor_quit", ACTION_BOOL)
    _ = imap.register_action("editor_rename", ACTION_BOOL)
    _ = imap.register_action("editor_select_all", ACTION_BOOL)
    _ = imap.register_action("tool_select", ACTION_BOOL)
    _ = imap.register_action("toggle_snap", ACTION_BOOL)
    _ = imap.register_action("build_reload", ACTION_BOOL)
    _ = imap.register_action("editor_save_as", ACTION_BOOL)
    _ = imap.register_action("editor_open", ACTION_BOOL)
    _ = imap.register_action("editor_help", ACTION_BOOL)
    _ = imap.register_action("cam_orbit", ACTION_AXIS2D)

    var ctx = InputContext("StudioEditor", priority=10)
    ctx.bind_key("gizmo_translate", 87)                  # W
    ctx.bind_key("gizmo_rotate", 69)                     # E
    ctx.bind_key("gizmo_scale", 82)                      # R
    ctx.bind_key("gizmo_toggle_space", 88)               # X (Local/World)
    ctx.bind_key("frame_selection", 70)                  # F (Frame selection)
    ctx.bind_key("editor_undo", 90, modifiers=MOD_CTRL)  # Ctrl+Z
    ctx.bind_key("editor_redo", 89, modifiers=MOD_CTRL)  # Ctrl+Y
    ctx.bind_key("editor_redo", 90, modifiers=MOD_CTRL | MOD_SHIFT)  # Ctrl+Shift+Z
    ctx.bind_key("source_editor", 297)                   # F8
    ctx.bind_key("editor_save", 83, modifiers=MOD_CTRL)  # Ctrl+S
    ctx.bind_key("editor_load", 79, modifiers=MOD_CTRL)  # Ctrl+O
    ctx.bind_key("editor_new", 78, modifiers=MOD_CTRL)   # Ctrl+N
    ctx.bind_key("editor_duplicate", 68, modifiers=MOD_CTRL)  # Ctrl+D
    ctx.bind_key("editor_copy", 67, modifiers=MOD_CTRL)  # Ctrl+C
    ctx.bind_key("editor_paste", 86, modifiers=MOD_CTRL) # Ctrl+V
    ctx.bind_key("editor_delete", 261)                   # Delete
    ctx.bind_key("pie_toggle", 294)                      # F5
    ctx.bind_key("pie_pause", 295)                       # F6
    ctx.bind_key("pie_step", 296)                        # F7
    ctx.bind_key("editor_quit", 81, modifiers=MOD_CTRL)  # Ctrl+Q (bare Esc never quits!)
    ctx.bind_key("editor_rename", 291)                   # F2
    ctx.bind_key("editor_select_all", 65, modifiers=MOD_CTRL)  # Ctrl+A
    ctx.bind_key("tool_select", 81)                      # Q (select tool, no gizmo)
    ctx.bind_key("toggle_snap", 71)                      # G (grid snap)
    ctx.bind_key("build_reload", 66, modifiers=MOD_CTRL) # Ctrl+B (reload gameplay .so)
    ctx.bind_key("editor_save_as", 83, modifiers=MOD_CTRL | MOD_SHIFT)  # Ctrl+Shift+S
    ctx.bind_key("editor_open", 79, modifiers=MOD_CTRL | MOD_SHIFT)     # Ctrl+Shift+O
    ctx.bind_key("editor_help", 290)                     # F1
    ctx.bind_key("cam_orbit", 263, scale_x=-1.0, scale_y=0.0, consume=False)  # Left
    ctx.bind_key("cam_orbit", 262, scale_x=1.0, scale_y=0.0, consume=False)   # Right
    ctx.bind_key("cam_orbit", 265, scale_x=0.0, scale_y=-1.0, consume=False)  # Up (look up)
    ctx.bind_key("cam_orbit", 264, scale_x=0.0, scale_y=1.0, consume=False)   # Down (look down)
    imap.add_context(ctx^)

    var play_ctx = InputContext("StudioPlay", priority=20, blocks_lower_all=True)
    play_ctx.bind_key("pie_stop", 294, modifiers=MOD_SHIFT)  # Shift+F5
    play_ctx.bind_key("pie_toggle", 294)                     # F5
    play_ctx.bind_key("pie_pause", 295)                      # F6
    play_ctx.bind_key("pie_step", 296)                       # F7
    play_ctx.bind_key("pie_toggle_camera", 290, modifiers=MOD_SHIFT)   # Shift+F1 toggles PIE camera possession
    play_ctx.bind_key("pie_release_cursor", 256)             # Esc releases pointer lock in PIE
    imap.add_context(play_ctx^)

    imap.push_context("StudioEditor")
    return imap^


def _write_blob_bytes(mut out: BlobWriter, data: List[UInt8]):
    out.u32(UInt32(len(data)))
    for i in range(len(data)):
        out.u8(data[i])


def _read_blob_bytes(mut r: BlobReader) -> List[UInt8]:
    var n = Int(r.u32())
    if n < 0 or r.failed:
        return List[UInt8]()
    var rem = r.limit() - r.offset
    if n > rem:
        r.failed = True
        return List[UInt8]()
    var buf = List[UInt8](capacity=n)
    for _ in range(n):
        buf.append(r.u8())
    return buf^


def _write_filter_canon(mut out: BlobWriter, f: CollisionFilter):
    out.u32(f.category)
    out.u32(f.mask)
    out.integer(Int(f.group))


def _studio_pod_read_u8(base_ptr: Int, offset: Int) -> UInt8:
    return Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=base_ptr + offset)[unsafe_offset=0]


def _studio_pod_write_u8(base_ptr: Int, offset: Int, val: UInt8):
    Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base_ptr + offset)[unsafe_offset=0] = val


def _studio_pod_read_u32(base_ptr: Int, offset: Int) -> UInt32:
    var p = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=base_ptr + offset)
    return (
        UInt32(p[unsafe_offset=0])
        | (UInt32(p[unsafe_offset=1]) << 8)
        | (UInt32(p[unsafe_offset=2]) << 16)
        | (UInt32(p[unsafe_offset=3]) << 24)
    )


def _studio_pod_write_u32(base_ptr: Int, offset: Int, val: UInt32):
    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=base_ptr + offset)
    p[unsafe_offset=0] = UInt8(val & 0xFF)
    p[unsafe_offset=1] = UInt8((val >> 8) & 0xFF)
    p[unsafe_offset=2] = UInt8((val >> 16) & 0xFF)
    p[unsafe_offset=3] = UInt8((val >> 24) & 0xFF)


def _studio_pod_read_i32(base_ptr: Int, offset: Int) -> Int32:
    return bitcast[DType.int32](_studio_pod_read_u32(base_ptr, offset))


def _studio_pod_write_i32(base_ptr: Int, offset: Int, val: Int32):
    _studio_pod_write_u32(base_ptr, offset, bitcast[DType.uint32](val))


def _studio_pod_read_u64(base_ptr: Int, offset: Int) -> UInt64:
    var lo = UInt64(_studio_pod_read_u32(base_ptr, offset))
    var hi = UInt64(_studio_pod_read_u32(base_ptr, offset + 4))
    return lo | (hi << 32)


def _studio_pod_write_u64(base_ptr: Int, offset: Int, val: UInt64):
    _studio_pod_write_u32(base_ptr, offset, UInt32(val & 0xFFFFFFFF))
    _studio_pod_write_u32(base_ptr, offset + 4, UInt32((val >> 32) & 0xFFFFFFFF))


def _studio_pod_read_i64(base_ptr: Int, offset: Int) -> Int64:
    return bitcast[DType.int64](_studio_pod_read_u64(base_ptr, offset))


def _studio_pod_write_i64(base_ptr: Int, offset: Int, val: Int64):
    _studio_pod_write_u64(base_ptr, offset, bitcast[DType.uint64](val))


def _studio_pod_read_f32(base_ptr: Int, offset: Int) -> Float32:
    var v = bitcast[DType.float32](_studio_pod_read_u32(base_ptr, offset))
    return v if isfinite(v) else Float32(0.0)


def _studio_pod_write_f32(base_ptr: Int, offset: Int, val: Float32):
    var safe = val if isfinite(val) else Float32(0.0)
    _studio_pod_write_u32(base_ptr, offset, bitcast[DType.uint32](safe))


def _parse_signed_int64(raw: String) -> Tuple[Bool, Int64]:
    var sb = raw.as_bytes()
    var lo = 0
    var hi = len(sb)
    while lo < hi and (sb[lo] == 32 or sb[lo] == 9 or sb[lo] == 10 or sb[lo] == 13):
        lo += 1
    while hi > lo and (sb[hi - 1] == 32 or sb[hi - 1] == 9 or sb[hi - 1] == 10 or sb[hi - 1] == 13):
        hi -= 1
    if lo >= hi:
        return (False, Int64(0))
    var neg = False
    if sb[lo] == 45:
        neg = True
        lo += 1
    elif sb[lo] == 43:
        lo += 1
    if lo >= hi:
        return (False, Int64(0))
    var limit = UInt64(9223372036854775808) if neg else UInt64(9223372036854775807)
    var acc = UInt64(0)
    while lo < hi:
        var b = sb[lo]
        if b < 48 or b > 57:
            return (False, Int64(0))
        var d = UInt64(b - 48)
        if acc > (limit - d) // UInt64(10):
            return (False, Int64(0))
        acc = acc * UInt64(10) + d
        lo += 1
    if neg:
        if acc == UInt64(9223372036854775808):
            return (True, Int64(-9223372036854775807) - Int64(1))
        return (True, -Int64(acc))
    return (True, Int64(acc))


def _is_internal_structural_component(name: String) -> Bool:
    return (
        name == "GlobalTransform"
        or name == "PrevGlobalTransform"
        or name == "Parent"
        or name == "Children"
        or name == "HierarchyDepth"
        or name == "StudioEntityMetaComponent"
        or name == "StudioEditorIdComponent"
        or name == "UnknownComponentsComponent"
    )


def _studio_format_input_binding(b: InputBinding) -> String:
    if b.source_kind == BIND_KEY:
        var prefix = String()
        if (b.modifiers & MOD_CTRL) != 0:
            prefix += "Ctrl+"
        if (b.modifiers & MOD_SHIFT) != 0:
            prefix += "Shift+"
        if (b.modifiers & MOD_ALT) != 0:
            prefix += "Alt+"
        return prefix + _keycode_display_name(b.code)
    if b.source_kind == BIND_MOUSE_BUTTON:
        return "Mouse" + String(b.code)
    if b.source_kind == BIND_PAD_BUTTON:
        return "Pad" + String(b.code)
    return "Code" + String(b.code)


comptime MAX_PIE_TRIGGER_EVENT_BUFFER: Int = 4096


@fieldwise_init
struct BufferedTriggerEvent(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var trigger_id: Int
    var body_id: Int
    var trigger_entity_id: Int
    var entity_id: Int
    var point: Vec3


struct PieTriggerEventBuffer(Movable):
    var ring: List[BufferedTriggerEvent]
    var head: Int
    var count: Int
    var base_seq: Int
    var total_seq: Int

    def __init__(out self):
        self.ring = List[BufferedTriggerEvent]()
        self.head = 0
        self.count = 0
        self.base_seq = 0
        self.total_seq = 0

    def push(mut self, ev: BufferedTriggerEvent):
        if len(self.ring) < MAX_PIE_TRIGGER_EVENT_BUFFER:
            self.ring.append(ev)
            self.count = len(self.ring)
            self.total_seq += 1
        else:
            self.ring[self.head] = ev
            self.head = (self.head + 1) % MAX_PIE_TRIGGER_EVENT_BUFFER
            self.base_seq += 1
            self.total_seq += 1

    def read_since(self, mut cursor: Int) -> List[BufferedTriggerEvent]:
        var out = List[BufferedTriggerEvent]()
        var start = max(cursor, self.base_seq)
        if start < self.total_seq and self.count > 0:
            var first_off = start - self.base_seq
            var n = len(self.ring)
            for off in range(first_off, self.count):
                out.append(self.ring[(self.head + off) % n])
        cursor = self.total_seq
        return out^

    def reset(mut self):
        self.ring.clear()
        self.head = 0
        self.count = 0
        self.base_seq = self.total_seq


@always_inline
def _ui_coalesce_key(seq: UInt64, widget_tag: String) -> UInt64:
    return hash_ui_id(seq, widget_tag)


@fieldwise_init
struct StudioToast(Copyable, ImplicitlyCopyable, Movable):
    var message: String
    var severity: Int  # 0=info, 1=warn, 2=error
    var ttl_sec: Float32


struct FlintStudioState(Movable):
    """Unified Engine v5 Studio State backed by `EngineApp`, `World`, `Scene`, and `PhysicsWorld`."""
    var app: EngineApp
    var scene: Scene
    var registry: ComponentRegistry
    var entities: List[StudioEntityView]  # Derived compat projection synced from `app.world`
    var static_bvh_colliders: List[Bvh8MeshCollider]
    var gpu_materials: StudioGpuMaterials

    var selected_entity_id: Int
    var selected_entity_ids: List[Int]
    var selection_anchor_id: Int
    var hierarchy_expanded: Dict[Int, Bool]

    var gizmo_mode: Int
    var gizmo_local_space: Bool
    var active_gizmo_handle: Int
    var viewport_xf: ViewportTransform
    """This frame's viewport transform in physical pixels (set by the host each frame); gizmo picks,
    drags, marquee selection and framing all project through it."""
    var pivot_mode: Int
    """`PIVOT_BOUNDS_CENTER` (centre of the selection's bounds) or `PIVOT_ACTIVE` (the primary entity)."""
    var scale_snap: Float32
    """Scale-factor step used when snapping is on (0.1 = 10%)."""
    var gz_dragging: Bool
    var gz_handle: Int
    var gz_mode: Int
    var gz_seq: UInt64
    var gz_seq_counter: UInt64
    var gz_start_ray: ViewportRay
    var gz_prev_ray: ViewportRay
    var gz_start_pivot: Vec3
    var gz_rot: Quat
    var gz_snaps: List[SelectedTransformSnap]
    var gz_angle: Float32
    """Rotation accumulated frame by frame, so one drag can turn past +-180 degrees."""
    var gz_start_mx: Float32
    var gz_start_my: Float32
    var marquee_active: Bool
    var viewport_hovered: Bool
    """Set by the host each frame: the mouse is over the scene panel (wheel / middle / Alt navigation)."""
    var marquee_x0: Float32
    var marquee_y0: Float32
    var marquee_x1: Float32
    var marquee_y1: Float32
    var snap_enabled: Bool
    var grid_snap_m: Float32
    var angle_snap_deg: Float32
    var sim_tick: Int
    var gravity: Vec3
    var ground_plane_y: Float32
    var has_ground_plane: Bool
    var _ground_body_id: Int

    var undo_stack: List[EditorTransaction]
    var redo_stack: List[EditorTransaction]
    var pie: PieSession
    var pie_camera_ejected: Bool
    var _pie_frame_updated: Bool
    var _pie_trigger_buffer: PieTriggerEventBuffer
    var _pie_trigger_drain_cursor: Int
    var _rmb_fly_active: Bool
    var input_rebind_active: Bool
    var input_rebind_context: String
    var input_rebind_action: String
    var hot_reload_host: HotReloadHost
    var gameplay_so_path: String
    var _synced_pod_schema_epoch: UInt64
    var _insp_add_comp_idx: Int
    var _int_edit_wid: UInt64
    var _int_edit_entity_id: Int
    var _int_edit_buf: String

    var stepped_clock: SteppedAnimClock
    var preview_impact_blend: Float32
    var preview_impact_invert: Bool
    var preview_shockwave_active: Bool
    var preview_shockwave_radius: Float32
    var viewport_anim_preview: Bool
    var preview_anim_time: Float32
    var _last_preview_anim_time: Float32
    var _last_hot_reload_count: Int
    var _pie_entry_hot_reloads: Int
    var _pie_entry_epochs: PieResolverEpochSnapshot
    var _pie_entry_style_epochs: Dict[UInt64, Tuple[String, UInt64]]
    var import_src_input: String
    var import_dst_subdir: String

    var dock: DockSpace
    var ui_ctx: UiContext
    var camera_rig: CameraRig
    var _cam_dummy_world: PhysicsWorld
    var input_map: InputMap
    var cam_target: Vec3
    var ui_edit_seq: UInt64
    var _edit_focus_id: UInt64
    var menu: scene_menu.SceneMenu
    var source_editor: EmbeddedIde
    var scene_file_path: String
    var project_root: String
    var asset_browser_dir: String
    var asset_browser_pack: String
    var asset_browser_entries: List[String]
    var asset_browser_scanned: Bool

    var dirty: Bool
    var quit_requested: Bool
    var confirm_modal_open: Bool
    var confirm_modal_action: Int  # 0=none, 1=quit, 2=new_scene, 3=load_scene (`pending_load_path`)
    var pending_load_path: String
    var file_modal_open: Bool
    var file_modal_save_as: Bool
    """True for File -> Save As, False for File -> Open."""
    var file_modal_path: String
    var recent_scenes: List[String]
    var console_open: Bool
    var console_log: List[String]
    """Every toast message (newest last, capped) so a missed toast can be read again."""
    var help_open: Bool
    var welcome_open: Bool
    var rename_focus_requested: Bool
    var next_editor_id: UInt64
    var autosave_interval_sec: Float32
    """Seconds of dirty editing between autosaves (0 disables)."""
    var _autosave_accum_sec: Float32
    var last_autosave_path: String
    var editor_id_slots: Dict[UInt64, Int]
    var _saved_tx_id: UInt64
    """Top undo entry when the scene was last saved / loaded (`_refresh_dirty_from_history`)."""
    var mcp_replay_client: List[Int]
    var mcp_replay_tool: List[String]
    var mcp_replay_args: List[String]
    var mcp_replay_step: List[UInt64]
    var mcp_replay_editor_id: List[UInt64]
    """Gated MCP calls the user approved, queued for the host to re-run exactly as the agent sent them
    (connection, tool, arguments, the agent step they complete, and the entity the user saw)."""
    var mcp_client_agents: Dict[Int, UInt32]
    """MCP connection key -> its own agent session in the Agents panel."""
    var flmat_pending_path: String
    """Disk path of a .flmat whose Inspector text fields are being edited; saved once on blur, not per keystroke."""
    var flmat_pending: FlmatAsset
    var flmat_pending_eid: Int
    var flmat_pending_ref: String
    var flmat_pending_editor_id: UInt64
    """Stable id of the entity the pending material edit is for (the slot may be reused)."""
    var flmat_pending_stamp: Tuple[Int, Int]
    """(size, mtime) of the .flmat when the edit began: a file changed on disk is never overwritten."""
    var flmat_pending_touch_sec: Float32
    """Last time the Inspector showed the pending edit; once it is gone from view the edit is saved."""
    var toasts: List[StudioToast]
    var clipboard_editor_ids: List[UInt64]
    """Copied entities by stable `editor_id` (slots are reused after delete / undo / load)."""
    var _time_travel_view_blob: List[UInt8]
    """The checkpoint time travel is showing; an edit attempted meanwhile is rolled back to it."""
    var gz_saved_redo: List[EditorTransaction]
    """Redo history parked while a gizmo drag runs; a cancelled (or motionless) drag gives it back."""
    var fail_world_capture_for_test: Bool
    """Test hook: `_capture_world_blob` fails, as a world that cannot be serialized would."""
    var show_navmesh_overlay: Bool
    var show_collider_overlay: Bool

    var cam_pos: Vec3
    var cam_fwd: Vec3
    var cam_right: Vec3
    var cam_up: Vec3
    var fov_y_deg: Float32
    var near_z: Float32

    # ---- Studio v2 Redesign State (Theme, Workspaces, Floating HUD, Palette, Agents, Code) ----
    var studio_theme: StudioTheme
    var studio_theme_idx: Int
    var ui_time_sec: Float32
    var anim_mode: Int
    var ui_scale: Float32
    var _next_tx_id: UInt64
    var _active_tx_agent_id: Int
    var _active_tx_step_id: UInt64
    var workspace_mgr: WorkspacePresetManager
    var header_bar: StudioHeaderBar
    var viewport_hud: ViewportHud
    var command_palette: StudioCommandPalette
    var key_chord_hud: KeyChordHud
    var toast_mgr: ToastManager
    var job_tracker: JobProgressTracker
    var outliner_state: OutlinerPanelState
    var inspector_state: InspectorPanelState
    var asset_browser_state: AssetBrowserPanelState
    var agent_bus: AgentTelemetryBus
    var agent_panel: AgentMissionControlPanel
    var agent_runtime: StudioAgentRuntime
    var agent_checkpoints: AgentCheckpointLog
    var _last_theatre_hits: List[TheatrePillHit]
    var code_drawer: CodeDrawer
    var show_grid_overlay: Bool
    var cam_ortho: Bool
    var gallery_play_hud_active: Bool
    var _last_screen_w: Float32
    var _last_screen_h: Float32

    def __init__(
        out self,
        headless: Bool = True,
        render_tier: UInt32 = RENDER_TIER_0_HEADLESS_SIMD,
        width: Int = 1280,
        height: Int = 720,
        worker_threads: Int = 0,
        project_root: String = "",
    ):
        self.app = EngineApp(
            AppConfig(
                title="FLINT STUDIO v5 — AAA VISUAL EDITOR",
                headless=headless,
                worker_threads=worker_threads,
                fixed_hz=60.0,
            )
        )
        self.app.add_plugin(WindowPlugin(headless=headless, width=width, height=height, title="FLINT STUDIO v5 — AAA VISUAL EDITOR"))
        self.app.add_plugin(TransformPlugin())
        self.app.add_plugin(InputPlugin())
        self.app.add_plugin(PhysicsPlugin(gravity=9.81))
        self.app.add_plugin(RenderPlugin(tier=render_tier, width=width, height=height, tsr_scale=1.0))
        self.app.add_plugin(AnimPlugin())
        self.app.add_plugin(AudioPlugin(headless=headless))
        self.app.add_plugin(NavPlugin(draw_debug=True))
        self.app.add_plugin(AiPlugin())
        self.app.add_plugin(
            AssetPlugin(
                loose_root=project_root,
                enable_watcher=(not headless) and project_root.byte_length() > 0,
                enable_resolver=True,
            )
        )
        self.app.resources.insert(StudioFlmatStyleSyncResource())
        self.app.resources.insert(PodSchemaRegistryResource())

        self.registry = ComponentRegistry.with_builtins()
        register_physics_codecs(self.registry)
        register_render_codecs(self.registry)
        register_anim_codecs(self.registry)
        register_audio_codecs(self.registry)
        register_nav_codecs(self.registry)
        register_ai_codecs(self.registry)
        self.registry.register[StudioEntityMetaComponent]()
        self.registry.register[StudioStyleComponent]()
        self.registry.register[StudioEditorIdComponent]()
        self.registry.ensure_registered_in_world(self.app.world)
        self.app.startup()
        var asset_pre_sid = -1
        for si in range(len(self.app.schedule.stages[STAGE_PRE_UPDATE].systems)):
            if self.app.schedule.stages[STAGE_PRE_UPDATE].systems[si].name == "asset_tick_resolve_pre":
                asset_pre_sid = self.app.schedule.stages[STAGE_PRE_UPDATE].systems[si].id
                break
        _ = self.app.add_system(
            STAGE_PRE_UPDATE,
            "studio_sync_flmat_styles",
            _sys_studio_sync_flmat_styles,
            after_system_id=asset_pre_sid,
        )
        self.app.schedule.compile_all()

        self.scene = Scene("FlintStudioScene")
        self.entities = List[StudioEntityView]()
        self.static_bvh_colliders = List[Bvh8MeshCollider]()
        self.gpu_materials = StudioGpuMaterials()

        self.selected_entity_id = -1
        self.selected_entity_ids = List[Int]()
        self.selection_anchor_id = -1
        self.hierarchy_expanded = Dict[Int, Bool]()

        self.gizmo_mode = GIZMO_MODE_TRANSLATE
        self.gizmo_local_space = False
        self.active_gizmo_handle = GIZMO_AXIS_NONE
        self.viewport_xf = ViewportTransform(
            RectF(0.0, 0.0, Float32(max(1, width)), Float32(max(1, height))),
            Float32(max(1, width)),
            Float32(max(1, height)),
            Float32(1.0),
        )
        self.pivot_mode = PIVOT_BOUNDS_CENTER
        self.scale_snap = Float32(0.1)
        self.gz_dragging = False
        self.gz_handle = GIZMO_AXIS_NONE
        self.gz_mode = GIZMO_MODE_TRANSLATE
        self.gz_seq = UInt64(0)
        self.gz_seq_counter = UInt64(0)
        self.gz_start_ray = ViewportRay()
        self.gz_prev_ray = ViewportRay()
        self.gz_start_pivot = Vec3()
        self.gz_rot = Quat.identity()
        self.gz_snaps = List[SelectedTransformSnap]()
        self.gz_angle = Float32(0.0)
        self.gz_start_mx = Float32(0.0)
        self.gz_start_my = Float32(0.0)
        self.marquee_active = False
        self.viewport_hovered = False
        self.marquee_x0 = Float32(0.0)
        self.marquee_y0 = Float32(0.0)
        self.marquee_x1 = Float32(0.0)
        self.marquee_y1 = Float32(0.0)
        self.snap_enabled = True
        self.grid_snap_m = Float32(0.25)
        self.angle_snap_deg = Float32(15.0)
        self.sim_tick = 0
        self.gravity = Vec3(0.0, -9.81, 0.0)
        self.ground_plane_y = Float32(0.0)
        self.has_ground_plane = True
        self._ground_body_id = -1

        self.undo_stack = List[EditorTransaction]()
        self.redo_stack = List[EditorTransaction]()
        self.pie = PieSession()
        self.pie_camera_ejected = False
        self._pie_frame_updated = False
        self._pie_trigger_buffer = PieTriggerEventBuffer()
        self._pie_trigger_drain_cursor = 0
        self._rmb_fly_active = False
        self.input_rebind_active = False
        self.input_rebind_context = ""
        self.input_rebind_action = ""
        self.hot_reload_host = HotReloadHost()
        self.gameplay_so_path = String()
        self._synced_pod_schema_epoch = UInt64(0)
        self._insp_add_comp_idx = 0
        self._int_edit_wid = UInt64(0)
        self._int_edit_entity_id = -1
        self._int_edit_buf = String()

        self.stepped_clock = SteppedAnimClock()
        self.preview_impact_blend = Float32(0.0)
        self.preview_impact_invert = False
        self.preview_shockwave_active = False
        self.preview_shockwave_radius = Float32(0.22)
        self.viewport_anim_preview = True
        self.preview_anim_time = Float32(0.0)
        self._last_preview_anim_time = Float32(0.0)
        self._last_hot_reload_count = 0
        self._pie_entry_hot_reloads = 0
        self._pie_entry_epochs = PieResolverEpochSnapshot()
        self._pie_entry_style_epochs = Dict[UInt64, Tuple[String, UInt64]]()
        self.import_src_input = ""
        self.import_dst_subdir = "assets"

        self.dock = _load_studio_dockspace()
        self.ui_ctx = UiContext(preset_hud=False)
        self.ui_ctx.theme.font_size = Float32(13.0)
        self.ui_ctx.theme.row_height = Float32(22.0)
        self.studio_theme = StudioTheme.flint_obsidian()
        self.studio_theme_idx = THEME_FLINT_OBSIDIAN
        self.ui_time_sec = Float32(0.0)
        self.anim_mode = UI_ANIM_NORMAL
        self.ui_scale = Float32(1.0)
        self._next_tx_id = UInt64(1)
        self._active_tx_agent_id = 0
        self._active_tx_step_id = UInt64(0)
        self.ui_ctx.dock_style_v2 = True
        self.ui_ctx.set_studio_theme(self.studio_theme.copy())
        self.workspace_mgr = WorkspacePresetManager()
        self.header_bar = StudioHeaderBar()
        self.viewport_hud = ViewportHud()
        self.command_palette = StudioCommandPalette()
        self.key_chord_hud = KeyChordHud()
        self.toast_mgr = ToastManager()
        self.job_tracker = JobProgressTracker()
        self.outliner_state = OutlinerPanelState()
        self.inspector_state = InspectorPanelState()
        self.asset_browser_state = AssetBrowserPanelState()
        self.agent_bus = AgentTelemetryBus()
        self.agent_panel = AgentMissionControlPanel()
        self.agent_runtime = StudioAgentRuntime()
        self.agent_runtime.workspace_root = project_root
        self.agent_checkpoints = AgentCheckpointLog()
        self._last_theatre_hits = List[TheatrePillHit]()
        self.code_drawer = CodeDrawer()
        self.show_grid_overlay = True
        self.cam_ortho = False
        self.gallery_play_hud_active = False
        self._last_screen_w = Float32(max(320, width))
        self._last_screen_h = Float32(max(240, height))
        self.camera_rig = CameraRig(CAM_MODE_ORBIT_FOLLOW)
        # Editor views must reach straight top-down / bottom-up (Top view used to stop at 85 degrees).
        self.camera_rig.min_pitch_deg = Float32(-89.5)
        self.camera_rig.max_pitch_deg = Float32(89.5)
        self.camera_rig.fov_deg = Float32(52.0)
        self.camera_rig.look_offset = Vec3(0.0, 0.0, 0.0)
        self._cam_dummy_world = PhysicsWorld()
        self.input_map = _create_studio_input_map()
        self.cam_target = Vec3(0.0, 0.9, 1.0)
        self.ui_edit_seq = UInt64(50000)
        self._edit_focus_id = UInt64(0)
        self.source_editor = EmbeddedIde()
        self.menu = scene_menu.SceneMenu(project_root)
        self.project_root = project_root
        self.scene_file_path = _resolve_default_scene_path(project_root)
        self.asset_browser_dir = project_root if project_root.byte_length() > 0 else ""
        self.asset_browser_pack = ""
        self.asset_browser_entries = List[String]()
        self.asset_browser_scanned = False
        self.dirty = False
        self.quit_requested = False
        self.confirm_modal_open = False
        self.confirm_modal_action = 0
        self.pending_load_path = String("")
        self.file_modal_open = False
        self.file_modal_save_as = False
        self.file_modal_path = String("")
        self.recent_scenes = List[String]()
        self.console_open = False
        self.console_log = List[String]()
        self.help_open = False
        self.welcome_open = False
        self.rename_focus_requested = False
        self._saved_tx_id = UInt64(0)
        self.next_editor_id = UInt64(1)
        self.autosave_interval_sec = Float32(120.0)
        self._autosave_accum_sec = Float32(0.0)
        self.last_autosave_path = String("")
        self.editor_id_slots = Dict[UInt64, Int]()
        self.mcp_replay_client = List[Int]()
        self.mcp_replay_tool = List[String]()
        self.mcp_replay_args = List[String]()
        self.mcp_replay_step = List[UInt64]()
        self.mcp_replay_editor_id = List[UInt64]()
        self.mcp_client_agents = Dict[Int, UInt32]()
        self.flmat_pending_path = String("")
        self.flmat_pending = FlmatAsset()
        self.flmat_pending_eid = -1
        self.flmat_pending_ref = String("")
        self.flmat_pending_editor_id = UInt64(0)
        self.flmat_pending_stamp = (-1, 0)
        self.flmat_pending_touch_sec = Float32(0.0)
        self.toasts = List[StudioToast]()
        self.clipboard_editor_ids = List[UInt64]()
        self._time_travel_view_blob = List[UInt8]()
        self.gz_saved_redo = List[EditorTransaction]()
        self.fail_world_capture_for_test = False
        self.show_navmesh_overlay = True
        self.show_collider_overlay = False

        self.cam_pos = Vec3(-6.5, 4.2, -9.5)
        self.cam_fwd = Vec3(0.52, -0.26, 0.81).normalized()
        self.cam_right = Vec3(0.0, 1.0, 0.0).cross(self.cam_fwd).normalized()
        self.cam_up = self.cam_fwd.cross(self.cam_right).normalized()
        self.fov_y_deg = Float32(52.0)
        self.near_z = Float32(0.15)
        self.look_at(Vec3(-6.5, 4.2, -9.5), Vec3(0.0, 0.9, 1.0))
        var rps_init = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_init:
            rps_init.value()[unsafe_offset=0].editor_camera_override = True
            rps_init.value()[unsafe_offset=0].editor_camera = RenderCameraPacket(
                self.cam_pos,
                self.cam_pos + self.cam_fwd * Float32(10.0),
                Vec3(0.0, 1.0, 0.0),
                self.fov_y_deg,
                self.near_z,
                Float32(1000.0),
                Float32(1.08),
            )
            rps_init.value()[unsafe_offset=0].has_editor_camera = True
        var fix_init = self.app.resources.get_ptr_mut[FixedTime]()
        if fix_init:
            fix_init.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)

    def shutdown(mut self):
        if self.hot_reload_host.is_loaded():
            _ = self.hot_reload_host.unload(self.app.world, self.app.resources, is_reload=False)
        _save_studio_dockspace(self.dock)
        self.menu.close()
        self.source_editor.close()
        self.app.shutdown()

    # ---- Entity Resolution & Sync Helpers ----

    @always_inline
    def resolve_entity(self, entity_id: Int) -> Entity:
        if entity_id < 0 or entity_id >= len(self.app.world.entities):
            return NULL_ENTITY
        if not self.app.world.entities[entity_id].alive:
            return NULL_ENTITY
        return Entity(UInt32(entity_id), self.app.world.entities[entity_id].generation)

    @always_inline
    def is_alive_id(self, entity_id: Int) -> Bool:
        return self.resolve_entity(entity_id).valid()

    def entity_ids(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.app.world.entities)):
            if self.app.world.entities[i].alive:
                out.append(i)
        return out^

    def parent_of(self, entity_id: Int) -> Int:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return -1
        var p_opt = self.app.world.get_ptr[Parent](e)
        if not p_opt:
            return -1
        var pe = p_opt.value()[unsafe_offset=0].entity
        if not self.app.world.is_alive(pe):
            return -1
        return Int(pe.index)

    def children_of(self, entity_id: Int) -> List[Int]:
        var out = List[Int]()
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return out^
        var ch_opt = self.app.world.get_ptr[Children](e)
        if not ch_opt:
            return out^
        ref ch = ch_opt.value()[unsafe_offset=0]
        for i in range(ch.len()):
            var ce = ch.at(i)
            if self.app.world.is_alive(ce):
                out.append(Int(ce.index))
        # Sort ascending by entity_id for deterministic tree ordering
        for i in range(1, len(out)):
            var key = out[i]
            var j = i - 1
            while j >= 0 and out[j] > key:
                out[j + 1] = out[j]
                j -= 1
            out[j + 1] = key
        return out^

    def _ensure_ground_plane(mut self):
        var phys_opt = self.app.resources.get_ptr_mut[PhysicsWorld]()
        if not phys_opt:
            return
        ref physics = phys_opt.value()[unsafe_offset=0]
        physics.gravity = -self.gravity.y
        if self.has_ground_plane:
            var floor_center = Vec3(0.0, self.ground_plane_y - Float32(1.0), 0.0)
            var slot = physics.find_slot_by_id_any(self._ground_body_id) if self._ground_body_id >= 0 else -1
            if slot < 0:
                slot = physics.add_box(
                    floor_center,
                    Vec3(500.0, 1.0, 500.0),
                    mass=0.0,
                    restitution=0.32,
                    friction=0.55,
                )
                self._ground_body_id = physics.body_id(slot)
            else:
                physics.bodies[slot].pos = floor_center
                physics.bodies[slot].active = True
        elif self._ground_body_id >= 0:
            var slot = physics.find_slot_by_id_any(self._ground_body_id)
            if slot >= 0:
                physics.swap_remove_body(slot)
            self._ground_body_id = -1

    def _propagate_transforms_immediate(mut self):
        var cache_opt = self.app.resources.get_ptr_mut[TransformPropagationCache]()
        if cache_opt:
            cache_opt.value()[unsafe_offset=0].invalidate()
            propagate_transforms_parallel_cached(
                self.app.world,
                self.app.pool,
                cache_opt.value()[unsafe_offset=0],
                copy_to_prev=False,
                allow_insert_missing=True,
            )

    def _sync_physics_pre_only(mut self):
        self._ensure_ground_plane()
        self.app.schedule.run_stage(
            STAGE_PRE_UPDATE,
            self.app.world,
            self.app.resources,
            self.app.pool,
            self.app.commands,
        )
        self.app.world.flush_commands(self.app.commands)

    def _reset_physics_world_and_sync(mut self):
        var phys_opt = self.app.resources.get_ptr_mut[PhysicsWorld]()
        if phys_opt:
            phys_opt.value()[unsafe_offset=0] = PhysicsWorld(gravity=-self.gravity.y)
            phys_opt.value()[unsafe_offset=0].gravity = -self.gravity.y
        var sync_opt = self.app.resources.get_ptr_mut[PhysicsSyncStateResource]()
        if sync_opt:
            sync_opt.value()[unsafe_offset=0] = PhysicsSyncStateResource()
        self._ground_body_id = -1
        self._sync_ecs_and_physics()

    def _ensure_editor_ids(mut self):
        """Gives every live entity a unique `editor_id` and rebuilds the id -> slot map. Entities from
        older scenes get fresh ids; a duplicated id (copied components) is reassigned on the later slot."""
        var seen = Dict[UInt64, Int]()
        var max_id = UInt64(0)
        var missing = List[Int]()
        for i in range(len(self.app.world.entities)):
            if not self.app.world.entities[i].alive:
                continue
            var e = Entity(UInt32(i), self.app.world.entities[i].generation)
            var id_opt = self.app.world.get_ptr[StudioEditorIdComponent](e)
            if not id_opt:
                missing.append(i)
                continue
            var eid_v = id_opt.value()[unsafe_offset=0].editor_id
            if eid_v == 0 or eid_v in seen:
                missing.append(i)
                continue
            seen[eid_v] = i
            if eid_v > max_id:
                max_id = eid_v
        if self.next_editor_id <= max_id:
            self.next_editor_id = max_id + 1
        for mi in range(len(missing)):
            var slot = missing[mi]
            var e = Entity(UInt32(slot), self.app.world.entities[slot].generation)
            var nid = self.next_editor_id
            self.next_editor_id += 1
            _ = self.app.world.insert[StudioEditorIdComponent](e, StudioEditorIdComponent(nid))
            seen[nid] = slot
        self.editor_id_slots = seen^

    def editor_id_of(self, entity_id: Int) -> UInt64:
        """Stable `editor_id` of a live slot, or 0."""
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return UInt64(0)
        var id_opt = self.app.world.get_ptr[StudioEditorIdComponent](e)
        if not id_opt:
            return UInt64(0)
        return id_opt.value()[unsafe_offset=0].editor_id

    def slot_for_editor_id(self, editor_id: UInt64) -> Int:
        """Current slot of the entity with this `editor_id`, or -1 when it no longer exists."""
        if editor_id == 0:
            return -1
        var slot = self.editor_id_slots.get(editor_id, -1)
        if slot >= 0 and self.editor_id_of(slot) == editor_id:
            return slot
        # Map is stale (restore/load since the last sync): scan once.
        for i in range(len(self.app.world.entities)):
            if self.app.world.entities[i].alive and self.editor_id_of(i) == editor_id:
                return i
        return -1

    def entity_ref(self, entity_id: Int) -> EntityRef:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return EntityRef.null()
        return EntityRef(self.editor_id_of(entity_id), entity_id, e.generation)

    def resolve_ref(self, r: EntityRef) -> Int:
        """Slot the reference points at now, or -1. A slot that was reused by another entity never
        resolves: the generation (and `editor_id`, when the ref has one) must still match."""
        if r.editor_id != 0:
            return self.slot_for_editor_id(r.editor_id)
        var e = self.resolve_entity(r.slot)
        if e.valid() and e.generation == r.generation:
            return r.slot
        return -1

    def _sync_compat_view(mut self):
        self._ensure_editor_ids()
        var cap = len(self.app.world.entities)
        while len(self.entities) < cap:
            self.entities.append(StudioEntityView(len(self.entities), "Entity", alive=False))
        while len(self.entities) > cap:
            _ = self.entities.pop()

        for i in range(cap):
            if not self.app.world.entities[i].alive:
                self.entities[i].entity_id = i
                self.entities[i].alive = False
                self.entities[i].parent_id = -1
                continue

            var e = Entity(UInt32(i), self.app.world.entities[i].generation)
            ref v = self.entities[i]
            v.entity_id = i
            v.alive = True
            v.parent_id = self.parent_of(i)

            var lt_opt = self.app.world.get_ptr[LocalTransform](e)
            if lt_opt:
                ref lt = lt_opt.value()[unsafe_offset=0]
                v.translation = lt.translation
                v.rotation = lt.rotation
                v.scale = lt.scale
            else:
                v.translation = Vec3()
                v.rotation = Quat()
                v.scale = Vec3(1.0, 1.0, 1.0)

            var mr_opt = self.app.world.get_ptr[MeshRendererComponent](e)
            if mr_opt:
                ref mr = mr_opt.value()[unsafe_offset=0]
                v.has_mesh_renderer = True
                v.mesh_id = mr.mesh_id
                v.material_id = mr.material_id
                v.visible = mr.visible
                v.mesh_ref = mr.mesh_ref.copy()
                v.material_ref = mr.material_ref.copy()
            else:
                v.has_mesh_renderer = False
                v.mesh_id = -1
                v.material_id = 0
                v.visible = True
                v.mesh_ref = AssetRef()
                v.material_ref = AssetRef()

            var mo_opt = self.app.world.get_ptr[MaterialOverrideComponent](e)
            if mo_opt:
                ref mo = mo_opt.value()[unsafe_offset=0]
                if mo.enabled and mo.material_id >= 0:
                    v.material_id = mo.material_id
                v.material_override_enabled = mo.enabled
                v.override_material_ref = mo.material_ref.copy()
                v.base_color = mo.base_color
                v.roughness = mo.roughness
                v.metallic = mo.metallic
                v.emissive = mo.emissive
                v.opacity = mo.opacity
                v.alpha_cutoff = mo.alpha_cutoff
                v.clearcoat = mo.clearcoat
                v.clearcoat_roughness = mo.clearcoat_roughness
                v.sheen = mo.sheen
                v.subsurface = mo.subsurface
            else:
                v.material_override_enabled = True
                v.override_material_ref = AssetRef()
                v.opacity = Float32(1.0)
                v.alpha_cutoff = Float32(0.0)
                v.clearcoat = Float32(0.0)
                v.clearcoat_roughness = Float32(0.08)
                v.sheen = Float32(0.0)
                v.subsurface = Float32(0.0)

            var meta_opt = self.app.world.get_ptr[StudioEntityMetaComponent](e)
            if meta_opt:
                ref meta = meta_opt.value()[unsafe_offset=0]
                v.name = meta.name
                v.locked = meta.locked
                v.enabled = meta.enabled
                if not mr_opt:
                    v.visible = meta.enabled
                v.mesh_handle = UntypedHandle(meta.mesh_handle_packed)
                v.material_handle = UntypedHandle(meta.material_handle_packed)
                v.collider_kind = meta.collider_kind
                v.collider_half_extents = meta.collider_half_extents
                v.restitution = meta.restitution
                v.friction = meta.friction
            else:
                v.name = "Entity_" + String(i)
                v.locked = False
                v.enabled = True
                v.collider_kind = COLLIDER_NONE

            var col_opt = self.app.world.get_ptr[ColliderComponent](e)
            if col_opt:
                v.collider_half_extents = col_opt.value()[unsafe_offset=0].half_extents
                v.is_trigger = col_opt.value()[unsafe_offset=0].is_trigger
            else:
                v.is_trigger = False

            var rb_opt = self.app.world.get_ptr[RigidBodyComponent](e)
            if rb_opt:
                ref rb = rb_opt.value()[unsafe_offset=0]
                v.linear_velocity = rb.linear_vel
                v.angular_velocity = rb.angular_vel
                v.mass = rb.mass if rb.motion_mode == BODY_MOTION_DYNAMIC else Float32(0.0)
                v.restitution = rb.restitution
                v.friction = rb.friction
            else:
                v.linear_velocity = Vec3()
                v.angular_velocity = Vec3()
                v.mass = Float32(0.0)

            var st_opt = self.app.world.get_ptr[StudioStyleComponent](e)
            if st_opt:
                ref st = st_opt.value()[unsafe_offset=0]
                v.shading_mode = st.shading_mode
                v.cel_shadow_tint = st.cel_shadow_tint
                v.cel_outline_width_px = st.cel_outline_width_px
                v.cel_outline_color = st.cel_outline_color
                v.cel_rim_color = st.cel_rim_color
                v.cel_hair_halo = st.cel_hair_halo
                v.vfx_erosion = st.vfx_erosion
                v.vfx_scroll_speed = st.vfx_scroll_speed
                v.vfx_core_color = st.vfx_core_color
                v.vfx_edge_color = st.vfx_edge_color
                v.anim_step_hz = st.anim_step_hz
                v.smear_stretch = st.smear_stretch

            var rig_opt = self.app.world.get_ptr[RigComponent](e)
            if rig_opt:
                ref rc = rig_opt.value()[unsafe_offset=0]
                v.has_rig = True
                v.rig_ref = rc.rig_ref.copy()
                v.rig_bound = rc.bound
                v.rig_joint_count = rc.joint_count
            else:
                v.has_rig = False
                v.rig_ref = AssetRef()
                v.rig_bound = False
                v.rig_joint_count = 0

            var sk_opt = self.app.world.get_ptr[SkeletonComponent](e)
            if sk_opt:
                ref sk = sk_opt.value()[unsafe_offset=0]
                v.has_skeleton = True
                v.skeleton_node_count = sk.skeleton.node_count
                v.skeleton_model_id = sk.model_id
                if v.rig_ref.is_empty() and not sk.rig_ref.is_empty():
                    v.rig_ref = sk.rig_ref.copy()
            else:
                v.has_skeleton = False
                v.skeleton_node_count = 0
                v.skeleton_model_id = 0

            var ac_opt = self.app.world.get_ptr[AnimatorComponent](e)
            if ac_opt:
                ref ac = ac_opt.value()[unsafe_offset=0]
                v.has_animator = True
                v.graph_ref = ac.graph_ref.copy()
                v.clip_index = ac.clip_index
                v.clip_name = ac.clip_name
                v.anim_playing = ac.playing
                v.anim_loop = ac.loop
                v.anim_speed = ac.speed
            else:
                v.has_animator = False
                v.graph_ref = AssetRef()
                v.clip_index = -1
                v.clip_name = ""
                v.anim_playing = True
                v.anim_loop = True
                v.anim_speed = Float32(1.0)

            var bs_opt = self.app.world.get_ptr[BoneSocketComponent](e)
            if bs_opt:
                ref bs = bs_opt.value()[unsafe_offset=0]
                v.has_bone_socket = True
                v.socket_name = bs.attachment.name
                v.socket_bone_node = bs.attachment.bone_node
            else:
                v.has_bone_socket = False
                v.socket_name = ""
                v.socket_bone_node = -1

            var au_opt = self.app.world.get_ptr[AudioEmitterComponent](e)
            if au_opt:
                ref au = au_opt.value()[unsafe_offset=0]
                v.has_audio = True
                v.audio_ref = au.audio_ref.copy()
                v.audio_playing = au.playing
            else:
                v.has_audio = False
                v.audio_ref = AssetRef()
                v.audio_playing = False

            var pf_opt = self.app.world.get_ptr[PrefabInstanceComponent](e)
            if pf_opt:
                ref pf = pf_opt.value()[unsafe_offset=0]
                v.has_prefab = True
                v.prefab_ref = pf.prefab_ref.copy()
                v.prefab_name = pf.prefab_name
            else:
                v.has_prefab = False
                v.prefab_ref = AssetRef()
                v.prefab_name = ""

    def _sync_ecs_and_physics(mut self):
        self._propagate_transforms_immediate()
        self._sync_physics_pre_only()
        self._sync_compat_view()

    def _sync_ecs_physics_and_render(mut self):
        self._sync_ecs_and_physics()

    def _capture_world_blob(self) -> List[UInt8]:
        """Whole-world snapshot, or an empty list when it cannot be taken: callers must then refuse the
        edit / load rather than record an undo entry that can never be restored."""
        if self.fail_world_capture_for_test:
            return List[UInt8]()
        var w = BlobWriter()
        try:
            self.app.world.snapshot_blob(w)
        except:
            return List[UInt8]()
        var out = List[UInt8]()
        swap(out, w.data)
        return out^

    def sync_flmat_studio_styles(mut self):
        _sync_studio_flmat_styles(self.app.world, self.app.resources)
        self._sync_compat_view()

    def _restore_world_blob(
        mut self,
        data: List[UInt8],
        tx_ext_epochs: Dict[String, UInt64] = Dict[String, UInt64](),
    ) -> Bool:
        if len(data) == 0:
            return False
        try:
            # Undo entries are stored LZ4-packed (`_pack_undo_blob`); checkpoints and captures are raw.
            var r = BlobReader(lz4_unpack(data) if lz4_packed_raw_size(data) >= 0 else data.copy())
            self.app.world.restore_blob(r)
            if r.failed:
                return False
        except:
            return False
        self.registry.ensure_registered_in_world(self.app.world)
        self.gpu_materials.clear_entity_slots()

        var res_opt = self.app.resources.get_ptr_mut[SceneAssetResolver]()
        var sync_opt = self.app.resources.get_ptr_mut[StudioFlmatStyleSyncResource]()
        if res_opt and sync_opt:
            ref res = res_opt.value()[unsafe_offset=0]
            ref sync = sync_opt.value()[unsafe_offset=0]
            var stale_keys = Dict[String, Bool]()
            for item in sync.external_mat_reload_epochs.items():
                var k = item.key
                var cur_ext = item.value
                var tx_ext_opt = tx_ext_epochs.get(k)
                var tx_ext = tx_ext_opt.value() if tx_ext_opt else UInt64(0)
                if cur_ext != tx_ext:
                    stale_keys[k] = True
                    res.mat_epochs[k] = UInt64(0)
            var next_synced = Dict[UInt64, Tuple[String, UInt64]]()
            var has_mo = self.app.world.component_id[MaterialOverrideComponent]() >= 0
            var has_mr = self.app.world.component_id[MeshRendererComponent]() >= 0
            var has_st = self.app.world.component_id[StudioStyleComponent]() >= 0
            for i in range(len(self.app.world.entities)):
                if not self.app.world.entities[i].alive:
                    continue
                var e = Entity(UInt32(i), self.app.world.entities[i].generation)
                var mkey = String()
                if has_mo:
                    var mo_p = self.app.world.get_ptr[MaterialOverrideComponent](e)
                    if mo_p and not mo_p.value()[].material_ref.is_empty():
                        mkey = _studio_mat_ref_key(mo_p.value()[].material_ref)
                if mkey.byte_length() == 0 and has_mr:
                    var mr_p = self.app.world.get_ptr[MeshRendererComponent](e)
                    if mr_p and not mr_p.value()[].material_ref.is_empty():
                        mkey = _studio_mat_ref_key(mr_p.value()[].material_ref)
                if mkey.byte_length() > 0 and has_st and self.app.world.has[StudioStyleComponent](e):
                    var is_stale = stale_keys.get(mkey)
                    if not (is_stale and is_stale.value()):
                        var ep_opt = res.mat_epochs.get(mkey)
                        if ep_opt and ep_opt.value() > 0:
                            next_synced[e.to_bits()] = (mkey, ep_opt.value())
            sync.synced_entity_epochs = next_synced^

        self._reset_physics_world_and_sync()
        return True

    # ---- Camera ----

    def _sync_from_camera_rig(mut self):
        self.cam_pos = self.camera_rig.world_pos
        self.cam_fwd = self.camera_rig.forward()
        self.cam_right = self.camera_rig.right()
        self.cam_up = self.camera_rig.up()
        self.fov_y_deg = self.camera_rig.fov_deg

    def look_at(mut self, eye: Vec3, target: Vec3):
        self.cam_target = target
        var d = target - eye
        var dist = max(Float32(0.35), d.length())
        var fwd = (d * (Float32(1.0) / dist)) if dist > Float32(1.0e-5) else Vec3(0.0, 0.0, 1.0)
        var horiz = sqrt(max(Float32(1.0e-8), fwd.x * fwd.x + fwd.z * fwd.z))
        var rad_to_deg = Float32(180.0) / PI
        self.camera_rig.yaw_deg = det_atan2(fwd.x, fwd.z) * rad_to_deg
        self.camera_rig.pitch_deg = det_atan2(-fwd.y, horiz) * rad_to_deg
        self.camera_rig.fov_deg = self.fov_y_deg
        self.camera_rig.look_offset = Vec3(0.0, 0.0, 0.0)
        self.camera_rig.pivot_spring.value = target
        self.camera_rig.pivot_spring.velocity = Vec3()
        self.camera_rig.pivot_initialized = True
        self.camera_rig.spring_arm.target_length = dist
        self.camera_rig.spring_arm.current_length.value = dist
        self.camera_rig.spring_arm.current_length.velocity = Float32(0.0)
        self.camera_rig.fly_pos = eye
        self.camera_rig.update(self._cam_dummy_world, target, Float32(0.0))
        self._sync_from_camera_rig()

    def update_camera(
        mut self,
        dt: Float32,
        orbit_delta_yaw_deg: Float32 = 0.0,
        orbit_delta_pitch_deg: Float32 = 0.0,
        fly_move_local: Vec3 = Vec3(),
        fly_speed: Float32 = 8.0,
    ):
        var user_cam_moved = (
            orbit_delta_yaw_deg != Float32(0.0)
            or orbit_delta_pitch_deg != Float32(0.0)
            or fly_move_local.length_squared() > Float32(1.0e-6)
        )
        if user_cam_moved:
            self.agent_bus.notify_user_camera_input()
        if orbit_delta_yaw_deg != Float32(0.0) or orbit_delta_pitch_deg != Float32(0.0):
            self.camera_rig.rotate_orbit(orbit_delta_yaw_deg, orbit_delta_pitch_deg)
        self.camera_rig.update(self._cam_dummy_world, self.cam_target, dt, fly_move_local=fly_move_local, fly_speed=fly_speed)
        self._sync_from_camera_rig()

    def set_ui_scale(mut self, scale: Float32):
        if isfinite(scale):
            self.ui_scale = max(Float32(0.75), min(Float32(3.0), scale))

    @always_inline
    def effective_ui_scale(self) -> Float32:
        return max(Float32(0.75), min(Float32(3.0), self.ui_scale))

    def viewport_rect(mut self, screen_w: Int, screen_h: Int) -> RectF:
        var s = self.effective_ui_scale()
        var lw = Float32(screen_w) / s
        var lh = Float32(screen_h) / s
        self.dock.solve_layout(
            RectF(
                0.0,
                STUDIO_HEADER_BAR_H,
                lw,
                max(Float32(64.0), lh - STUDIO_HEADER_BAR_H - STUDIO_STATUS_BAR_H),
            )
        )
        var vp = self.dock.find_panel("SceneViewport")
        if vp[0] >= 0:
            var r = vp[3]
            return RectF(r.x * s, r.y * s, r.w * s, r.h * s)
        return RectF(
            Float32(236.0),
            STUDIO_HEADER_BAR_H + Float32(14.0),
            Float32(max(160, screen_w - 524)),
            Float32(max(120, screen_h - 184)),
        )

    # ---- Multi-Select & Hierarchy Tree Queries ----

    def _normalize_selection_from_primary(mut self):
        if len(self.selected_entity_ids) == 0 and self.selected_entity_id >= 0 and self.is_alive_id(self.selected_entity_id):
            self.selected_entity_ids.append(self.selected_entity_id)
            if self.selection_anchor_id < 0:
                self.selection_anchor_id = self.selected_entity_id

    def _prune_selection(mut self):
        self._normalize_selection_from_primary()
        var kept = List[Int]()
        for i in range(len(self.selected_entity_ids)):
            var eid = self.selected_entity_ids[i]
            if not self.is_alive_id(eid):
                continue
            var dup = False
            for j in range(len(kept)):
                if kept[j] == eid:
                    dup = True
                    break
            if not dup:
                kept.append(eid)
        self.selected_entity_ids = kept^
        if not self.is_alive_id(self.selected_entity_id):
            if len(self.selected_entity_ids) > 0:
                self.selected_entity_id = self.selected_entity_ids[len(self.selected_entity_ids) - 1]
            else:
                self.selected_entity_id = -1
        if not self.is_alive_id(self.selection_anchor_id):
            self.selection_anchor_id = self.selected_entity_id

    def _restore_selection(mut self, ids: List[Int], primary_id: Int, anchor_id: Int = -1):
        self.selected_entity_ids = ids.copy()
        self.selected_entity_id = primary_id
        self.selection_anchor_id = anchor_id if anchor_id >= 0 else primary_id
        self._prune_selection()

    def is_selected(self, entity_id: Int) -> Bool:
        if entity_id < 0:
            return False
        for i in range(len(self.selected_entity_ids)):
            if self.selected_entity_ids[i] == entity_id:
                return True
        return len(self.selected_entity_ids) == 0 and self.selected_entity_id == entity_id

    def selected_count(self) -> Int:
        if len(self.selected_entity_ids) > 0:
            return len(self.selected_entity_ids)
        if self.selected_entity_id >= 0 and self.is_alive_id(self.selected_entity_id):
            return 1
        return 0

    def selected_entities(self) -> List[Entity]:
        var out = List[Entity]()
        if len(self.selected_entity_ids) > 0:
            for i in range(len(self.selected_entity_ids)):
                var e = self.resolve_entity(self.selected_entity_ids[i])
                if e.valid():
                    out.append(e)
        elif self.selected_entity_id >= 0:
            var e = self.resolve_entity(self.selected_entity_id)
            if e.valid():
                out.append(e)
        return out^

    def clear_selection(mut self):
        self.selected_entity_ids.clear()
        self.selected_entity_id = -1
        self.selection_anchor_id = -1

    def is_hierarchy_expanded(self, entity_id: Int) -> Bool:
        var found = self.hierarchy_expanded.get(entity_id)
        if found:
            return found.value()
        return True

    def set_hierarchy_expanded(mut self, entity_id: Int, expanded: Bool):
        self.hierarchy_expanded[entity_id] = expanded

    def toggle_hierarchy_expanded(mut self, entity_id: Int) -> Bool:
        var next_v = not self.is_hierarchy_expanded(entity_id)
        self.hierarchy_expanded[entity_id] = next_v
        return next_v

    def _collect_hierarchy_dfs(
        self,
        entity_id: Int,
        parent_id: Int,
        depth: Int,
        mut out: List[HierarchyRow],
        mut visited: List[Bool],
        ignore_collapse: Bool = False,
    ):
        if entity_id < 0 or entity_id >= len(visited) or visited[entity_id] or depth >= 64:
            return
        visited[entity_id] = True
        var ch = self.children_of(entity_id)
        var has_ch = len(ch) > 0
        var exp = self.is_hierarchy_expanded(entity_id)
        out.append(HierarchyRow(entity_id, parent_id, depth, has_ch, exp))
        if has_ch and (exp or ignore_collapse):
            for i in range(len(ch)):
                self._collect_hierarchy_dfs(ch[i], entity_id, depth + 1, out, visited, ignore_collapse)

    def hierarchy_key_nav(mut self, up: Bool, down: Bool, left: Bool, right: Bool) -> Bool:
        """Arrow keys in the Hierarchy: Up/Down move the selection through the visible rows,
        Left collapses (or jumps to the parent), Right expands (or steps into the first child)."""
        if not (up or down or left or right):
            return False
        var rows = self.hierarchy_visible_rows()
        if len(rows) == 0:
            return False
        var cur = -1
        for i in range(len(rows)):
            if rows[i].entity_id == self.selected_entity_id:
                cur = i
                break
        if cur < 0:
            self.select_entity(rows[0].entity_id)
            return True
        var row = rows[cur]
        if up and cur > 0:
            self.select_entity(rows[cur - 1].entity_id)
        elif down and cur + 1 < len(rows):
            self.select_entity(rows[cur + 1].entity_id)
        elif left:
            if row.has_children and row.expanded:
                self.hierarchy_expanded[row.entity_id] = False
            elif row.parent_id >= 0:
                self.select_entity(row.parent_id)
        elif right and row.has_children:
            if not row.expanded:
                self.hierarchy_expanded[row.entity_id] = True
            elif cur + 1 < len(rows):
                self.select_entity(rows[cur + 1].entity_id)
        return True

    def hierarchy_visible_rows(self, ignore_collapse: Bool = False) -> List[HierarchyRow]:
        var out = List[HierarchyRow]()
        var n = len(self.app.world.entities)
        if n == 0:
            return out^
        var visited = List[Bool](length=n, fill=False)
        for i in range(n):
            if self.app.world.entities[i].alive and self.parent_of(i) < 0:
                self._collect_hierarchy_dfs(i, -1, 0, out, visited, ignore_collapse)
        # Fallback for any orphan whose parent died externally (never emit children of collapsed live parents)
        for i in range(n):
            if self.app.world.entities[i].alive and not visited[i] and self.parent_of(i) < 0:
                self._collect_hierarchy_dfs(i, -1, 0, out, visited, ignore_collapse)
        return out^

    def hierarchy_rows_for_query(self, query: String) -> List[HierarchyRow]:
        """Outliner rows for a search: every entity whose name matches, including children of collapsed
        parents (a search that skips collapsed branches hides exactly what the user is looking for)."""
        var q = Self._ascii_lower(String(query.strip()))
        if q.byte_length() == 0:
            return self.hierarchy_visible_rows()
        var all_rows = self.hierarchy_visible_rows(ignore_collapse=True)
        var out = List[HierarchyRow]()
        for i in range(len(all_rows)):
            var eid = all_rows[i].entity_id
            if eid < len(self.entities) and Self._ascii_lower(self.entities[eid].name).find(q) >= 0:
                out.append(all_rows[i])
        return out^

    def set_entity_locked(mut self, entity_id: Int, locked: Bool) -> Bool:
        """Locks an entity against viewport picking, marquee and gizmo moves. Stored on the entity's
        metadata component, so it is saved with the scene and follows the entity (never a reused slot)."""
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return False
        var m = self.app.world.get_ptr_mut[StudioEntityMetaComponent](e)
        if not m:
            return False
        m.value()[unsafe_offset=0].locked = locked
        if entity_id < len(self.entities):
            self.entities[entity_id].locked = locked
        self.mark_dirty(True)
        return True

    def is_entity_locked(self, entity_id: Int) -> Bool:
        return entity_id >= 0 and entity_id < len(self.entities) and self.is_alive_id(entity_id) and self.entities[entity_id].locked

    def hierarchy_display_order(self) -> List[Int]:
        var rows = self.hierarchy_visible_rows()
        var order = List[Int](capacity=len(rows))
        for i in range(len(rows)):
            order.append(rows[i].entity_id)
        return order^

    def select_entity(
        mut self,
        entity_id: Int,
        multi_select: Bool = False,
        range_select: Bool = False,
        visible_order: List[Int] = List[Int](),
    ):
        self._normalize_selection_from_primary()
        if not self.is_alive_id(entity_id):
            if (not multi_select) and (not range_select):
                self.clear_selection()
            return

        if range_select:
            var order = visible_order.copy() if len(visible_order) > 0 else self.hierarchy_display_order()
            var anchor = self.selection_anchor_id if self.is_alive_id(self.selection_anchor_id) else self.selected_entity_id
            if not self.is_alive_id(anchor):
                anchor = entity_id
            var ia = -1
            var ib = -1
            for i in range(len(order)):
                if order[i] == anchor:
                    ia = i
                if order[i] == entity_id:
                    ib = i
            if ia >= 0 and ib >= 0:
                var lo = min(ia, ib)
                var hi = max(ia, ib)
                if not multi_select:
                    self.selected_entity_ids.clear()
                    self.selected_entity_id = -1
                for k in range(lo, hi + 1):
                    var cand = order[k]
                    if self.is_alive_id(cand) and not self.is_selected(cand):
                        self.selected_entity_ids.append(cand)
                self.selected_entity_id = entity_id
                self.selection_anchor_id = anchor
                return

        if multi_select:
            var found_idx = -1
            for i in range(len(self.selected_entity_ids)):
                if self.selected_entity_ids[i] == entity_id:
                    found_idx = i
                    break
            if found_idx >= 0:
                var next_list = List[Int](capacity=max(0, len(self.selected_entity_ids) - 1))
                for i in range(len(self.selected_entity_ids)):
                    if i != found_idx:
                        next_list.append(self.selected_entity_ids[i])
                self.selected_entity_ids = next_list^
                if self.selected_entity_id == entity_id:
                    self.selected_entity_id = (
                        self.selected_entity_ids[len(self.selected_entity_ids) - 1]
                        if len(self.selected_entity_ids) > 0
                        else -1
                    )
                if self.selection_anchor_id == entity_id:
                    self.selection_anchor_id = self.selected_entity_id
            else:
                self.selected_entity_ids.append(entity_id)
                self.selected_entity_id = entity_id
                self.selection_anchor_id = entity_id
            return

        self.selected_entity_ids.clear()
        self.selected_entity_ids.append(entity_id)
        self.selected_entity_id = entity_id
        self.selection_anchor_id = entity_id

    def _top_level_selected_ids(self) -> List[Int]:
        """Filters selection to exclude any entity whose ancestor is also selected (avoids double-transform)."""
        var ids = List[Int]()
        if len(self.selected_entity_ids) > 0:
            for i in range(len(self.selected_entity_ids)):
                if self.is_alive_id(self.selected_entity_ids[i]):
                    ids.append(self.selected_entity_ids[i])
        elif self.is_alive_id(self.selected_entity_id):
            ids.append(self.selected_entity_id)

        var top = List[Int](capacity=len(ids))
        for i in range(len(ids)):
            var eid = ids[i]
            var has_sel_anc = False
            var cur = self.parent_of(eid)
            var hops = 0
            while cur >= 0 and hops < 64:
                var found_anc = False
                for j in range(len(ids)):
                    if ids[j] == cur:
                        found_anc = True
                        break
                if found_anc:
                    has_sel_anc = True
                    break
                cur = self.parent_of(cur)
                hops += 1
            if not has_sel_anc:
                top.append(eid)
        return top^

    def selection_gizmo_pivot(self) -> Tuple[Bool, Vec3]:
        """Gizmo pivot: the centre of the selection's bounding box (not the mean of positions, which a
        cluster of objects on one side pulls off-centre), or the primary entity with `PIVOT_ACTIVE`."""
        var ents = self.selected_entities()
        if len(ents) == 0:
            return (False, Vec3())
        if self.pivot_mode == PIVOT_ACTIVE and self.is_alive_id(self.selected_entity_id):
            var pe = self.resolve_entity(self.selected_entity_id)
            var pgt = self.app.world.get_ptr[GlobalTransform](pe)
            if pgt:
                return (True, pgt.value()[unsafe_offset=0].translation)
        var lo = Vec3(Float32(1.0e30), Float32(1.0e30), Float32(1.0e30))
        var hi = Vec3(Float32(-1.0e30), Float32(-1.0e30), Float32(-1.0e30))
        var cnt = 0
        for i in range(len(ents)):
            var p = Vec3()
            var gt_opt = self.app.world.get_ptr[GlobalTransform](ents[i])
            if gt_opt:
                p = gt_opt.value()[unsafe_offset=0].translation
            else:
                var lt_opt = self.app.world.get_ptr[LocalTransform](ents[i])
                if not lt_opt:
                    continue
                p = lt_opt.value()[unsafe_offset=0].translation
            lo = Vec3(min(lo.x, p.x), min(lo.y, p.y), min(lo.z, p.z))
            hi = Vec3(max(hi.x, p.x), max(hi.y, p.y), max(hi.z, p.z))
            cnt += 1
        if cnt == 0:
            return (False, Vec3())
        return (True, (lo + hi) * Float32(0.5))

    def toggle_pivot_mode(mut self):
        self.pivot_mode = PIVOT_ACTIVE if self.pivot_mode == PIVOT_BOUNDS_CENTER else PIVOT_BOUNDS_CENTER
        self.push_toast("Pivot: Active Entity" if self.pivot_mode == PIVOT_ACTIVE else "Pivot: Bounds Centre", 0, 1.8)

    # ---- Editor UX Helpers (Dirty, Toasts, Space, Clipboard, NavBake, Spawn) ----

    def _top_tx_id(self) -> UInt64:
        return self.undo_stack[len(self.undo_stack) - 1].tx_id if len(self.undo_stack) > 0 else UInt64(0)

    def _refresh_dirty_from_history(mut self):
        """After undo / redo the scene is clean exactly when history is back at the last save or load
        (undoing your edits back to the saved state no longer leaves a bogus unsaved-changes mark)."""
        var clean = self._top_tx_id() == self._saved_tx_id
        var keep_saved = self._saved_tx_id
        self.mark_dirty(not clean)
        self._saved_tx_id = keep_saved

    def mark_dirty(mut self, dirty_val: Bool = True):
        self.dirty = dirty_val
        if not dirty_val:
            self._saved_tx_id = self._top_tx_id()
        var base_title = "FLINT STUDIO v5 — AAA VISUAL EDITOR"
        var full_title = base_title + " *" if dirty_val else base_title
        self.app.config.title = full_title
        var win_res = self.app.resources.get_ptr_mut[WindowStateResource]()
        if win_res:
            win_res.value()[unsafe_offset=0].title = full_title
            for wi in range(len(win_res.value()[unsafe_offset=0].windows)):
                win_res.value()[unsafe_offset=0].windows[wi].set_title(full_title)

    def push_toast(mut self, message: String, severity: Int = 0, ttl_sec: Float32 = 3.0):
        var tag = String("[error] ") if severity == 2 else (String("[warn] ") if severity == 1 else String(""))
        self.console_log.append(tag + message)
        while len(self.console_log) > STUDIO_CONSOLE_CAP:
            _ = self.console_log.pop(0)
        self.toasts.append(StudioToast(message, severity, ttl_sec))
        while len(self.toasts) > 5:
            _ = self.toasts.pop(0)
        var sev_v2 = TOAST_ERROR if severity == 2 else (TOAST_WARN if severity == 1 else TOAST_INFO)
        _ = self.toast_mgr.push(sev_v2, message, "", ttl_sec=ttl_sec)

    def tick_toasts(mut self, dt: Float32):
        var i = len(self.toasts) - 1
        while i >= 0:
            self.toasts[i].ttl_sec -= dt
            if self.toasts[i].ttl_sec <= Float32(0.0):
                _ = self.toasts.pop(i)
            i -= 1
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else dt
        self.toast_mgr.tick(safe_dt)
        self.key_chord_hud.tick(safe_dt)

    # ---- Studio v2 Theme, Animation Mode, Workspaces, Command Palette & Agent Studio ----

    def set_studio_theme(mut self, theme_id: Int):
        var tid = theme_id % 4
        if tid < 0:
            tid += 4
        self.studio_theme_idx = tid
        self.studio_theme = StudioTheme.by_id(tid)
        self.ui_ctx.set_studio_theme(self.studio_theme.copy())

    def set_anim_mode(mut self, mode: Int):
        self.anim_mode = mode
        self.ui_ctx.set_anim_mode(mode)
        self.agent_panel.anim_mode = mode
        if mode == UI_ANIM_INSTANT:
            self.ui_time_sec = Float32(0.0)

    def apply_workspace_preset(
        mut self,
        preset_id: Int,
        screen_w: Float32 = 0.0,
        screen_h: Float32 = 0.0,
    ) -> Bool:
        var sw = screen_w if screen_w > Float32(100.0) else self._last_screen_w
        var sh = screen_h if screen_h > Float32(100.0) else self._last_screen_h
        var dock_r = RectF(Float32(0.0), STUDIO_HEADER_BAR_H, sw, max(Float32(64.0), sh - STUDIO_HEADER_BAR_H - STUDIO_STATUS_BAR_H))
        return self.workspace_mgr.apply_preset(preset_id, self.dock, dock_r)

    def wants_keyboard(self) -> Bool:
        return (
            self.confirm_modal_open
            or self.ui_ctx.modal_depth > 0
            or self.ui_ctx.had_modal_prev_frame
            or self.command_palette.is_open
            or self.agent_panel.wants_keyboard()
            or self.code_drawer.has_keyboard_focus
            or self.ui_ctx.wants_keyboard()
            or self.menu.bar.group >= 0
        )

    def wants_mouse(self) -> Bool:
        return (
            self.confirm_modal_open
            or self.ui_ctx.modal_depth > 0
            or self.ui_ctx.had_modal_prev_frame
            or self.command_palette.is_open
            or self.ui_ctx.wants_mouse()
            or self.menu.bar.group >= 0
        )

    def wants_mouse_at(self, mx: Float32, my: Float32) -> Bool:
        if self.wants_mouse():
            return True
        var s = self.effective_ui_scale()
        var lmx = mx / s
        var lmy = my / s
        if self.header_bar.contains_point(lmx, lmy):
            return True
        if self.viewport_hud.contains_point(lmx, lmy):
            return True
        for i in range(len(self._last_theatre_hits)):
            if self._last_theatre_hits[i].contains(lmx, lmy):
                return True
        return False

    def _classify_entity_kind(self, entity_id: Int) -> Int:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return ENTITY_KIND_EMPTY
        if self.app.world.has[SpotLightComponent](e):
            return ENTITY_KIND_LIGHT_SPOT
        if self.app.world.has[PointLightComponent](e):
            return ENTITY_KIND_LIGHT_POINT
        if self.app.world.has[CameraComponent](e):
            return ENTITY_KIND_CAMERA
        if self.app.world.has[AudioEmitterComponent](e):
            return ENTITY_KIND_AUDIO
        if self.app.world.has[PrefabInstanceComponent](e):
            return ENTITY_KIND_PREFAB
        var low = FlintStudioState._ascii_lower(
            self.entities[entity_id].name if entity_id < len(self.entities) else String("")
        )
        if low.find("foliage") >= 0 or low.find("canopy") >= 0 or low.find("tree") >= 0:
            return ENTITY_KIND_FOLIAGE
        if self.app.world.has[MeshRendererComponent](e):
            return ENTITY_KIND_MESH
        if self.app.world.has[ColliderComponent](e):
            return ENTITY_KIND_COLLIDER
        return ENTITY_KIND_EMPTY

    def _aggregate_agent_status(self) -> Int:
        if self.agent_bus.pending_approval_count() > 0:
            return AGENT_STATE_WAITING_APPROVAL
        var best = AGENT_STATE_IDLE
        for i in range(len(self.agent_bus.sessions)):
            var st = self.agent_bus.sessions[i].status
            if st == AGENT_STATUS_WAITING_APPROVAL:
                return AGENT_STATE_WAITING_APPROVAL
            if st == AGENT_STATUS_ERROR:
                best = AGENT_STATE_ERROR
            elif st == AGENT_STATUS_EDITING_SCENE:
                if best == AGENT_STATE_IDLE or best == AGENT_STATE_THINKING:
                    best = AGENT_STATE_EDITING_SCENE
            elif st == AGENT_STATUS_WRITING_CODE or st == AGENT_STATUS_BUILDING:
                if best == AGENT_STATE_IDLE or best == AGENT_STATE_THINKING:
                    best = AGENT_STATE_WRITING_CODE
            elif st == AGENT_STATUS_TESTING_PIE:
                if best == AGENT_STATE_IDLE or best == AGENT_STATE_THINKING:
                    best = AGENT_STATE_TESTING_PIE
            elif st == AGENT_STATUS_THINKING or st == AGENT_STATUS_PLANNING:
                if best == AGENT_STATE_IDLE:
                    best = AGENT_STATE_THINKING
        return best

    def _hero_entity_id_and_pos(self) -> Tuple[Int, Vec3, Vec3]:
        var eid = self.selected_entity_id
        if not self.is_alive_id(eid):
            for i in range(len(self.entities)):
                if self.entities[i].alive and self.entities[i].name == "SubD_MaterialOrb":
                    eid = i
                    break
        if not self.is_alive_id(eid):
            for i in range(len(self.entities)):
                if self.entities[i].alive:
                    eid = i
                    break
        if eid >= 0 and self.is_alive_id(eid):
            var p = self.entities[eid].translation
            return (eid, p, p + Vec3(Float32(-1.35), Float32(0.85), Float32(-1.15)))
        return (0, Vec3(0.0, 0.92, 0.0), Vec3(-1.35, 1.77, -1.15))

    def refresh_command_palette(mut self):
        var ent_names = List[String]()
        var ent_ids = List[Int]()
        for i in range(len(self.entities)):
            if self.entities[i].alive:
                ent_names.append(self.entities[i].name)
                ent_ids.append(i)
        var has_proj = self.menu.has_project or (
            self.project_root.byte_length() > 0 and file_size(self.project_root + "/flint.toml") >= 0
        )
        populate_studio_palette(
            self.command_palette,
            ent_names,
            ent_ids,
            self.asset_browser_entries,
            pie_active=self.pie.mode != PIE_STOPPED,
            can_undo=len(self.undo_stack) > 0,
            can_redo=len(self.redo_stack) > 0,
            has_selection=self.selected_count() > 0,
            has_project=has_proj,
            gizmo_local=self.gizmo_local_space,
            snap_enabled=self.snap_enabled,
            show_navmesh=self.show_navmesh_overlay,
            show_colliders=self.show_collider_overlay,
            follow_agent_cam=self.agent_bus.follow_agent_camera,
            show_agent_overlays=self.agent_bus.theatre_overlays_enabled,
        )

    def open_command_palette(mut self, mode: Int = PALETTE_ALL, initial_query: String = ""):
        self.refresh_command_palette()
        # `open()` defaults would reset the PIE gate set by `refresh_command_palette` and pin the
        # frecency clock to 0 (no decay), so pass both explicitly.
        self.command_palette.open(
            mode, initial_query, pie_active=self.pie.mode != PIE_STOPPED, now_sec=unix_time_sec()
        )

    def _snap_camera_direction(mut self, view_fwd: Vec3):
        var fwd = view_fwd.normalized()
        var dist = max(Float32(2.0), self.camera_rig.spring_arm.target_length)
        var tgt = self.cam_target
        var eye = tgt - fwd * dist
        self.look_at(eye, tgt)

    def execute_palette_action(mut self, res: PaletteActionResult) -> Bool:
        if not res.executed:
            return False
        if res.kind == PALETTE_ITEM_ENTITY:
            if self.is_alive_id(res.action_id):
                self.select_entity(res.action_id)
                if res.secondary:
                    self.frame_selection()
                return True
            return False
        if res.kind == PALETTE_ITEM_ASSET:
            if res.action_id >= 0 and res.action_id < len(self.asset_browser_entries):
                var entry_name = self.asset_browser_entries[res.action_id]
                var e_low = FlintStudioState._ascii_lower(entry_name)
                if e_low.endswith(".mojo") or e_low.endswith(".toml") or e_low.endswith(".json"):
                    var disk_p = self._resolve_asset_disk_path(entry_name)
                    _ = self.code_drawer.open_file(disk_p)
                    _ = WorkspacePresetManager.ensure_panel_visible(self.dock, PANEL_CODE_DRAWER)
                    return True
                var tgt = self.selected_entity_id if (e_low.endswith(".flmat") or e_low.endswith(".flrig") or e_low.endswith(".flanim")) else -1
                return self.activate_asset_entry(entry_name, target_entity_id=tgt, from_drop=False) >= 0
            return False
        return self.execute_palette_command(res.action_id)

    def execute_palette_command(mut self, cmd_id: Int, entity_id: Int = -1) -> Bool:
        var in_play = self.pie.mode != PIE_STOPPED

        if cmd_id == PAL_CMD_OPEN_PALETTE:
            self.open_command_palette(PALETTE_ALL)
            return True
        if cmd_id == PAL_CMD_OPEN_QUICK_SPAWN:
            self.open_command_palette(PALETTE_SPAWN)
            return True
        if cmd_id == PAL_CMD_SAVE_SCENE and not in_play:
            var ok = self.save_scene_file()
            self.push_toast("Scene saved" if ok else "Save failed", 0 if ok else 2, 2.5)
            return ok
        if cmd_id == PAL_CMD_LOAD_SCENE and not in_play:
            return self.request_load_scene()
        if cmd_id == PAL_CMD_NEW_SCENE and not in_play:
            return self.new_scene(force=False)
        if cmd_id == PAL_CMD_RELOAD_SCENE and not in_play:
            return self.request_load_scene()
        if cmd_id == PAL_CMD_RECOVER_AUTOSAVE and not in_play:
            return self.recover_autosave()
        if cmd_id == PAL_CMD_UNDO and not in_play:
            return self.undo()
        if cmd_id == PAL_CMD_REDO and not in_play:
            return self.redo()
        if cmd_id == PAL_CMD_DUPLICATE and not in_play:
            if len(self.selected_entity_ids) > 1:
                return len(self.duplicate_selected(record_undo=True)) > 0
            if self.selected_entity_id >= 0:
                return self.duplicate_entity(self.selected_entity_id, record_undo=True) >= 0
            return False
        if cmd_id == PAL_CMD_DELETE and not in_play:
            if len(self.selected_entity_ids) > 1:
                return self.despawn_selected(record_undo=True) > 0
            if self.selected_entity_id >= 0:
                return self.despawn_entity(self.selected_entity_id, record_undo=True)
            return False
        if cmd_id == PAL_CMD_RENAME:
            self.begin_rename_selected()
            return True
        if cmd_id == PAL_CMD_COPY:
            return self.copy_selected() > 0
        if cmd_id == PAL_CMD_PASTE and not in_play:
            return len(self.paste_clipboard(record_undo=True)) > 0
        if cmd_id == PAL_CMD_SELECT_ALL:
            self.select_all_entities()
            return True
        if cmd_id == PAL_CMD_CLEAR_SELECTION:
            self.selected_entity_ids.clear()
            self.selected_entity_id = -1
            self.selection_anchor_id = -1
            return True
        if cmd_id == PAL_CMD_UNPARENT and not in_play:
            return self.reparent_selected_to(-1, preserve_world_transform=True, record_undo=True) > 0
        if cmd_id == PAL_CMD_PARENT_TO_PRIMARY and not in_play:
            if self.selected_count() >= 2 and self.selected_entity_id >= 0:
                return self.reparent_selected_to(self.selected_entity_id, preserve_world_transform=True, record_undo=True) > 0
            return False

        if cmd_id == PAL_CMD_TOOL_SELECT:
            self.active_gizmo_handle = GIZMO_AXIS_NONE
            return True
        if cmd_id == PAL_CMD_TOOL_MOVE:
            self.gizmo_mode = GIZMO_MODE_TRANSLATE
            return True
        if cmd_id == PAL_CMD_TOOL_ROTATE:
            self.gizmo_mode = GIZMO_MODE_ROTATE
            return True
        if cmd_id == PAL_CMD_TOOL_SCALE:
            self.gizmo_mode = GIZMO_MODE_SCALE
            return True
        if cmd_id == PAL_CMD_TOGGLE_GIZMO_SPACE:
            self.toggle_gizmo_space()
            return True
        if cmd_id == PAL_CMD_TOGGLE_SNAP:
            self.snap_enabled = not self.snap_enabled
            return True
        if cmd_id == PAL_CMD_FRAME_SELECTION:
            self.frame_selection()
            return True
        if cmd_id == PAL_CMD_RESET_TRANSFORM and not in_play:
            var eid = entity_id if self.is_alive_id(entity_id) else self.selected_entity_id
            if self.is_alive_id(eid):
                var cur = self.entities[eid].copy()
                cur.translation = Vec3(0.0, 0.0, 0.0)
                cur.rotation = Quat.identity()
                cur.scale = Vec3(1.0, 1.0, 1.0)
                return self.commit_node_edit(eid, cur^, "Reset Transform")
            return False

        if cmd_id == PAL_CMD_PIE_PLAY:
            return self.pie_play()
        if cmd_id == PAL_CMD_PIE_PAUSE:
            if self.pie.mode == PIE_PAUSED:
                return self.pie_play()
            return self.pie_pause()
        if cmd_id == PAL_CMD_PIE_STEP:
            return self.pie_step(1, Float32(1.0 / 60.0)) > 0
        if cmd_id == PAL_CMD_PIE_STOP:
            try:
                return self.pie_stop()
            except:
                self._abort_pie_session()
                return True
        if cmd_id == PAL_CMD_PIE_EJECT_CAMERA:
            self.pie_camera_ejected = not self.pie_camera_ejected
            var rps_e = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
            if rps_e:
                rps_e.value()[unsafe_offset=0].editor_camera_override = self.pie_camera_ejected
            return True

        if cmd_id == PAL_CMD_WORKSPACE_EDIT:
            return self.apply_workspace_preset(WORKSPACE_EDIT)
        if cmd_id == PAL_CMD_WORKSPACE_AGENT_STUDIO:
            return self.apply_workspace_preset(WORKSPACE_AGENT_STUDIO)
        if cmd_id == PAL_CMD_WORKSPACE_CODE_SCENE:
            return self.apply_workspace_preset(WORKSPACE_CODE_SCENE)
        if cmd_id == PAL_CMD_WORKSPACE_LOOKDEV:
            return self.apply_workspace_preset(WORKSPACE_LOOKDEV)
        if cmd_id == PAL_CMD_WORKSPACE_PLAY:
            return self.apply_workspace_preset(WORKSPACE_PLAY)

        if cmd_id == PAL_CMD_THEME_OBSIDIAN:
            self.set_studio_theme(THEME_FLINT_OBSIDIAN)
            return True
        if cmd_id == PAL_CMD_THEME_MIDNIGHT:
            self.set_studio_theme(THEME_MIDNIGHT_STUDIO)
            return True
        if cmd_id == PAL_CMD_THEME_OKABE_ITO:
            self.set_studio_theme(THEME_OKABE_ITO)
            return True
        if cmd_id == PAL_CMD_THEME_PAPER_LIGHT:
            self.set_studio_theme(THEME_PAPER_LIGHT)
            return True
        if cmd_id == PAL_CMD_THEME_CYCLE:
            self.set_studio_theme((self.studio_theme_idx + 1) % 4)
            return True

        if cmd_id == PAL_CMD_TOGGLE_CODE_DRAWER:
            _ = WorkspacePresetManager.ensure_panel_visible(self.dock, PANEL_CODE_DRAWER)
            return True
        if cmd_id == PAL_CMD_BUILD_HOT_RELOAD:
            var ok = self.reload_gameplay_plugin()
            self.code_drawer.set_build_status(
                BUILD_STATUS_OK if ok else BUILD_STATUS_ERROR,
                self.code_drawer.build_generation + 1,
                0 if ok else 1,
                0,
            )
            self.push_toast(
                "Reloaded gameplay .so" if ok else ("Reload .so failed: " + self.hot_reload_host.last_error),
                0 if ok else 2,
                2.8,
            )
            return ok
        if cmd_id == PAL_CMD_SHOW_DIAGNOSTICS:
            self.code_drawer.problems_open = True
            _ = WorkspacePresetManager.ensure_panel_visible(self.dock, PANEL_CODE_DRAWER)
            return True
        if cmd_id == PAL_CMD_BAKE_NAVMESH and not in_play:
            var pc = self.bake_navmesh()
            self.push_toast("NavMesh baked (" + String(pc) + " polys)", 0, 2.5)
            return pc > 0
        if cmd_id == PAL_CMD_PACKAGE_PROJECT and not in_play:
            var pr = self.package_project()
            self.push_toast(pr[1], 0 if pr[0] else 2, 3.2)
            return pr[0]
        if cmd_id == PAL_CMD_PACKAGE_PROFILES and not in_play:
            var pr = self.package_project()
            self.push_toast(pr[1], 0 if pr[0] else 2, 3.2)
            return pr[0]
        if cmd_id == PAL_CMD_TOGGLE_NAVMESH_OVERLAY:
            self.show_navmesh_overlay = not self.show_navmesh_overlay
            var nav_res = self.app.resources.get_ptr_mut[NavPluginStateResource]()
            if nav_res:
                nav_res.value()[unsafe_offset=0].draw_debug = self.show_navmesh_overlay
            return True
        if cmd_id == PAL_CMD_TOGGLE_COLLIDER_OVERLAY:
            self.show_collider_overlay = not self.show_collider_overlay
            return True
        if cmd_id == PAL_CMD_TOGGLE_GRID_OVERLAY:
            self.show_grid_overlay = not self.show_grid_overlay
            return True
        if cmd_id == PAL_CMD_TOGGLE_ORTHO_CAMERA:
            self.toggle_ortho_camera()
            return True
        if cmd_id == PAL_CMD_TOGGLE_PERF_SPARKLINE:
            self.viewport_hud.perf_expanded = not self.viewport_hud.perf_expanded
            return True
        if cmd_id == PAL_CMD_RESET_CAMERA:
            self.look_at(Vec3(-5.6, 3.75, -7.4), Vec3(0.15, 0.95, 1.1))
            return True
        if cmd_id == PAL_CMD_SNAP_VIEW_FRONT:
            self._snap_camera_direction(Vec3(0.0, 0.0, 1.0))
            return True
        if cmd_id == PAL_CMD_SNAP_VIEW_RIGHT:
            self._snap_camera_direction(Vec3(-1.0, 0.0, 0.0))
            return True
        if cmd_id == PAL_CMD_SNAP_VIEW_TOP:
            self._snap_camera_direction(Vec3(0.0, -0.999, 0.01))
            return True
        if cmd_id == PAL_CMD_RESCAN_ASSETS:
            _ = self.scan_asset_browser()
            return True

        if not in_play:
            if cmd_id == PAL_CMD_SPAWN_EMPTY:
                return self.spawn_card_entity(0) >= 0
            if cmd_id == PAL_CMD_SPAWN_CUBE or cmd_id == PAL_CMD_SPAWN_SPHERE or cmd_id == PAL_CMD_SPAWN_PLANE:
                return self.spawn_card_entity(1) >= 0
            if cmd_id == PAL_CMD_SPAWN_POINT_LIGHT or cmd_id == PAL_CMD_SPAWN_SUN_LIGHT:
                return self.spawn_card_entity(2) >= 0
            if cmd_id == PAL_CMD_SPAWN_SPOT_LIGHT:
                return self.spawn_card_entity(3) >= 0
            if cmd_id == PAL_CMD_SPAWN_CAMERA:
                return self.spawn_card_entity(4) >= 0
            if cmd_id == PAL_CMD_SPAWN_AUDIO:
                return self.spawn_card_entity(5) >= 0
            if cmd_id == PAL_CMD_SPAWN_PHYSICS_BOX:
                return self.spawn_entity("PhysBox_" + String(self.live_entity_count() + 1), position=Vec3(0.0, 2.0, 0.0), collider_kind=COLLIDER_BOX, mass=5.0, mesh_id=1) >= 0
            if cmd_id == PAL_CMD_SPAWN_TRIGGER:
                var tid = self.spawn_entity("Trigger_" + String(self.live_entity_count() + 1), position=Vec3(0.0, 1.0, 0.0), collider_kind=COLLIDER_BOX, mass=0.0, mesh_id=-1)
                if tid >= 0:
                    var te = self.resolve_entity(tid)
                    var c_opt = self.app.world.get_ptr_mut[ColliderComponent](te)
                    if c_opt:
                        c_opt.value()[unsafe_offset=0].is_trigger = True
                    self._sync_compat_view()
                return tid >= 0
            if cmd_id == PAL_CMD_SPAWN_PREFAB:
                return self.spawn_card_entity(0) >= 0

        if cmd_id == PAL_CMD_AGENT_SCENE_BUILDER:
            _ = self.apply_workspace_preset(WORKSPACE_AGENT_STUDIO)
            var a = AgentPanelAction(PANEL_ACT_SELECT_OR_SPAWN_ROLE)
            a.role = ROLE_SCENE_BUILDER
            self.apply_agent_panel_action(a)
            return True
        if cmd_id == PAL_CMD_AGENT_GAMEPLAY_CODER:
            _ = self.apply_workspace_preset(WORKSPACE_AGENT_STUDIO)
            var a = AgentPanelAction(PANEL_ACT_SELECT_OR_SPAWN_ROLE)
            a.role = ROLE_GAMEPLAY_CODER
            self.apply_agent_panel_action(a)
            return True
        if cmd_id == PAL_CMD_AGENT_LOOKDEV_ARTIST:
            _ = self.apply_workspace_preset(WORKSPACE_AGENT_STUDIO)
            var a = AgentPanelAction(PANEL_ACT_SELECT_OR_SPAWN_ROLE)
            a.role = ROLE_LOOKDEV_ARTIST
            self.apply_agent_panel_action(a)
            return True
        if cmd_id == PAL_CMD_AGENT_QA_PLAYTESTER:
            _ = self.apply_workspace_preset(WORKSPACE_AGENT_STUDIO)
            var a = AgentPanelAction(PANEL_ACT_SELECT_OR_SPAWN_ROLE)
            a.role = ROLE_QA_PLAYTESTER
            self.apply_agent_panel_action(a)
            return True
        if cmd_id == PAL_CMD_AGENT_TOGGLE_PANEL:
            if self.workspace_mgr.active_preset == WORKSPACE_AGENT_STUDIO:
                _ = self.apply_workspace_preset(WORKSPACE_EDIT)
            else:
                _ = WorkspacePresetManager.ensure_panel_visible(self.dock, PANEL_AGENTS)
            return True
        if cmd_id == PAL_CMD_AGENT_TOGGLE_FOLLOW_CAM:
            self.agent_bus.follow_agent_camera = not self.agent_bus.follow_agent_camera
            if self.agent_bus.follow_agent_camera:
                self.agent_bus.resume_follow_camera()
            return True
        if cmd_id == PAL_CMD_AGENT_TOGGLE_THEATRE_3D:
            self.agent_bus.theatre_overlays_enabled = not self.agent_bus.theatre_overlays_enabled
            return True
        if cmd_id == PAL_CMD_AGENT_WATCH_DEMO:
            _ = self.apply_workspace_preset(WORKSPACE_AGENT_STUDIO)
            var hero = self._hero_entity_id_and_pos()
            self.agent_runtime.start_scripted_demo(
                self.agent_bus,
                self.code_drawer,
                self.toast_mgr,
                hero[0],
                hero[1],
                hero[2],
            )
            return True
        if cmd_id == PAL_CMD_AGENT_APPROVE_STEP:
            if len(self.agent_bus.pending_approvals) > 0:
                var a = AgentPanelAction(PANEL_ACT_APPROVE)
                a.approval_id = self.agent_bus.pending_approvals[0].approval_id
                self.apply_agent_panel_action(a)
                return True
            return False
        if cmd_id == PAL_CMD_AGENT_REJECT_STEP:
            if len(self.agent_bus.pending_approvals) > 0:
                var a = AgentPanelAction(PANEL_ACT_REJECT)
                a.approval_id = self.agent_bus.pending_approvals[0].approval_id
                self.apply_agent_panel_action(a)
                return True
            return False
        if cmd_id == PAL_CMD_AGENT_PAUSE_RESUME:
            var si = self.agent_bus.find_session_index(self.agent_bus.selected_agent_id)
            if si >= 0:
                self.agent_bus.set_session_paused(self.agent_bus.selected_agent_id, not self.agent_bus.sessions[si].paused)
                return True
            return False
        if cmd_id == PAL_CMD_AGENT_REVERT_LAST_STEP:
            var si = self.agent_bus.find_session_index(self.agent_bus.selected_agent_id)
            if si >= 0 and len(self.agent_bus.sessions[si].steps) > 0:
                var last_st = self.agent_bus.sessions[si].steps[len(self.agent_bus.sessions[si].steps) - 1].step_id
                var a = AgentPanelAction(PANEL_ACT_REVERT_STEP)
                a.agent_id = self.agent_bus.selected_agent_id
                a.step_id = last_st
                self.apply_agent_panel_action(a)
                return True
            return False
        if cmd_id == PAL_CMD_AGENT_EXIT_TIME_TRAVEL:
            var a = AgentPanelAction(PANEL_ACT_TIME_TRAVEL_EXIT)
            self.apply_agent_panel_action(a)
            return True

        return False

    def record_agent_step_checkpoint(
        mut self,
        tx_id: UInt64,
        agent_id: Int,
        turn_id: Int,
        step_id: Int,
        pre_world_raw: List[UInt8],
        post_world_raw: List[UInt8],
        label: String = "",
    ) -> Int:
        return self.agent_checkpoints.record_step_raw(
            tx_id,
            agent_id,
            turn_id,
            step_id,
            pre_world_raw,
            post_world_raw,
            label,
        )

    def apply_agent_panel_action(mut self, act: AgentPanelAction):
        if act.kind == PANEL_ACT_SELECT_OR_SPAWN_ROLE:
            var found_id = UInt32(0)
            for i in range(len(self.agent_bus.sessions)):
                if self.agent_bus.sessions[i].role == act.role:
                    found_id = self.agent_bus.sessions[i].agent_id
                    break
            if found_id == 0:
                var pol = APPROVAL_STEP if act.role == ROLE_SCENE_BUILDER else APPROVAL_EDITS
                found_id = self.agent_bus.create_session(act.role, "", pol)
            self.agent_bus.selected_agent_id = found_id

        elif act.kind == PANEL_ACT_SET_POLICY:
            self.agent_bus.set_session_policy(act.agent_id, act.policy)

        elif act.kind == PANEL_ACT_RUN_PROMPT:
            var aid = act.agent_id
            if self.agent_bus.find_session_index(aid) < 0:
                aid = self.agent_bus.create_session(act.role, "", act.policy)
            self.agent_bus.selected_agent_id = aid
            var pre_raw = self._capture_world_blob()
            var pre_hash = self.state_hash()
            var turn_id = self.agent_bus.begin_turn(aid, act.text, pre_raw, pre_hash, Float64(self.ui_time_sec))
            if self.agent_runtime.acp.active():
                _ = self.agent_runtime.send_prompt(
                    self.agent_bus, aid, act.text, pre_raw, pre_hash, Float64(self.ui_time_sec)
                )
            else:
                var hero = self._hero_entity_id_and_pos()
                var st_id = self.agent_bus.begin_mcp_tool_step(
                    aid,
                    "agent.prompt",
                    act.text,
                    "{}",
                    entity_id=hero[0],
                    world_pos=hero[1],
                    has_world_pos=True,
                    pre_world_lz4=pre_raw,
                    pre_hash=pre_hash,
                    now_sec=Float64(self.ui_time_sec),
                )
                _ = self.record_agent_step_checkpoint(
                    UInt64(0), Int(aid), Int(turn_id), Int(st_id), pre_raw, pre_raw, act.text
                )
                _ = self.toast_mgr.push(TOAST_AGENT, "Agent Task Queued", act.text)

        elif act.kind == PANEL_ACT_CANCEL_TURN:
            self.agent_runtime.cancel_active_turn(self.agent_bus, act.agent_id, Float64(self.ui_time_sec))

        elif act.kind == PANEL_ACT_TOGGLE_FOLLOW_CAMERA:
            self.agent_bus.follow_agent_camera = not self.agent_bus.follow_agent_camera
            if self.agent_bus.follow_agent_camera:
                self.agent_bus.resume_follow_camera()

        elif act.kind == PANEL_ACT_TOGGLE_THEATRE:
            self.agent_bus.theatre_overlays_enabled = not self.agent_bus.theatre_overlays_enabled

        elif act.kind == PANEL_ACT_APPROVE or act.kind == PANEL_ACT_APPROVE_TURN or act.kind == PANEL_ACT_REJECT:
            var approved = act.kind != PANEL_ACT_REJECT
            var allow_turn = act.kind == PANEL_ACT_APPROVE_TURN
            var res = self.agent_bus.resolve_approval(act.approval_id, approved, allow_turn)
            if res[0]:
                var pa = res[1].copy()
                if pa.source == APPROVAL_SRC_MCP_TOOL:
                    # Approving runs the agent's exact call (same tool, same arguments, same client),
                    # re-targeted through the stable editor id at the entity the user saw. Nothing is
                    # granted by tool name, so the approval can never wave through a different call.
                    var target_eid = pa.ghost.editor_id if pa.ghost.editor_id != 0 else pa.target_editor_id
                    if approved and target_eid != 0 and self.slot_for_editor_id(target_eid) < 0:
                        self.push_toast("Approval target no longer exists", 1, 3.0)
                        _ = self.agent_bus.complete_mcp_tool_step(
                            pa.agent_id, pa.step_id, False, '{"ok":false,"error":"approval target no longer exists"}', Float32(0.0)
                        )
                    elif approved and pa.tool_name.byte_length() > 0:
                        self.mcp_replay_client.append(pa.mcp_client_key)
                        self.mcp_replay_tool.append(pa.tool_name)
                        self.mcp_replay_args.append(pa.raw_payload_json)
                        self.mcp_replay_step.append(pa.step_id)
                        self.mcp_replay_editor_id.append(target_eid)
                    elif approved and pa.ghost.kind == GHOST_TRANSFORM:
                        # A preview-only approval (no tool call to replay): apply the previewed move.
                        var ghost_slot = self.slot_for_editor_id(target_eid) if target_eid != 0 else pa.ghost.entity_id
                        if ghost_slot >= 0 and self.is_alive_id(ghost_slot):
                            var cur = self.entities[ghost_slot].copy()
                            var pre_raw = self._capture_world_blob()
                            cur.translation = pa.ghost.target_pos
                            var prev_origin = self._active_tx_agent_id
                            self._active_tx_agent_id = Int(pa.agent_id)
                            var committed = self.commit_node_edit(ghost_slot, cur^, "Agent: " + pa.title)
                            self._active_tx_agent_id = prev_origin
                            if committed:
                                var post_raw = self._capture_world_blob()
                                var tx_id = self.undo_stack[len(self.undo_stack) - 1].tx_id if len(self.undo_stack) > 0 else UInt64(0)
                                _ = self.record_agent_step_checkpoint(
                                    tx_id, Int(pa.agent_id), 1, Int(pa.step_id), pre_raw, post_raw, pa.title
                                )
                                _ = self.agent_bus.complete_mcp_tool_step(
                                    pa.agent_id,
                                    pa.step_id,
                                    True,
                                    '{"ok":true,"approved":true}',
                                    Float32(1.0),
                                    tx_id,
                                    entity_id=ghost_slot,
                                    world_pos=pa.ghost.target_pos,
                                    has_world_pos=True,
                                    post_world_lz4=post_raw,
                                    post_hash=self.state_hash(),
                                )
                    _ = self.toast_mgr.push(
                        TOAST_SUCCESS if approved else TOAST_INFO,
                        ("Approved: " if approved else "Rejected: ") + pa.title,
                        "",
                    )
                else:
                    _ = self.agent_runtime.answer_acp_approval(pa, approved, allow_turn)

        elif act.kind == PANEL_ACT_FOCUS_ENTITY:
            if self.is_alive_id(act.entity_id):
                self.select_entity(act.entity_id)
                self.frame_selection()

        elif act.kind == PANEL_ACT_OPEN_FILE_DIFF:
            if act.text.byte_length() > 0:
                if self.code_drawer.find_tab(act.text) >= 0:
                    self.code_drawer.active_tab = self.code_drawer.find_tab(act.text)
                    self.code_drawer.view_mode = CODE_MODE_DIFF_UNIFIED
                else:
                    var disk_p = self._resolve_asset_disk_path(act.text)
                    _ = self.code_drawer.open_file(disk_p)
                _ = WorkspacePresetManager.ensure_panel_visible(self.dock, PANEL_CODE_DRAWER)

        elif act.kind == PANEL_ACT_TIME_TRAVEL_SCRUB:
            _ = self.time_travel_scrub(act.agent_id, act.step_index)

        elif act.kind == PANEL_ACT_TIME_TRAVEL_EXIT:
            self.exit_time_travel()

        elif act.kind == PANEL_ACT_REVERT_STEP or act.kind == PANEL_ACT_REVERT_TURN:
            var plan = (
                self.agent_bus.plan_revert_step(act.agent_id, act.step_id)
                if act.kind == PANEL_ACT_REVERT_STEP
                else self.agent_bus.plan_revert_turn(act.agent_id, act.turn_id)
            )
            var later_user_edits = self.user_edits_after_agent_steps(act.agent_id, plan.reverted_step_ids) if plan.valid else 0
            if plan.valid and later_user_edits > 0:
                # A revert restores a whole-world checkpoint; doing it now would silently throw away
                # edits the user made after the agent acted. Refuse and say why instead.
                _ = self.toast_mgr.push(
                    TOAST_WARN,
                    "Revert blocked",
                    "It would discard " + String(later_user_edits) + " of your later edits. Undo those first.",
                )
            elif plan.valid:
                # The revert is one undoable edit of the live world (never of a time-travel view).
                self.exit_time_travel()
                var rv_before = self._capture_world_blob()
                var rv_prev_ids = self.selected_entity_ids.copy()
                var rv_prev_primary = self.selected_entity_id
                var rv_prev_anchor = self.selection_anchor_id
                var restored_world = False
                if act.kind == PANEL_ACT_REVERT_STEP:
                    var cp = self.agent_checkpoints.get_step_pre_bytes(Int(act.step_id))
                    if cp[0] and len(cp[1]) > 0:
                        restored_world = self._restore_world_blob(cp[1])
                else:
                    var cp_t = self.agent_checkpoints.get_turn_pre_bytes(Int(act.turn_id))
                    if cp_t[0] and len(cp_t[1]) > 0:
                        restored_world = self._restore_world_blob(cp_t[1])
                if not restored_world and len(plan.restore_world_lz4) > 0:
                    restored_world = self._restore_world_blob(plan.restore_world_lz4)
                if restored_world:
                    # Tagged with the agent, so the revert never counts as a later *user* edit that
                    # would block reverting the rest of the agent's work.
                    var keep_agent = self._active_tx_agent_id
                    self._active_tx_agent_id = Int(act.agent_id)
                    self._record_world_tx(
                        TX_BATCH_EDIT, -1, List[Int](), rv_before^, rv_prev_ids, rv_prev_primary, rv_prev_anchor,
                        "Revert Agent Step" if act.kind == PANEL_ACT_REVERT_STEP else "Revert Agent Turn",
                    )
                    self._active_tx_agent_id = keep_agent
                    self.mark_dirty(True)
                for fi in range(len(plan.file_paths)):
                    if plan.file_existed[fi]:
                        var sb = plan.file_old_texts[fi].as_bytes()
                        var fbytes = List[UInt8](capacity=len(sb))
                        for bi in range(len(sb)):
                            fbytes.append(sb[bi])
                        _ = write_bytes(plan.file_paths[fi], fbytes)
                self.agent_bus.commit_revert_plan(plan)
                _ = self.toast_mgr.push(
                    TOAST_SUCCESS,
                    "Reverted Agent Step" if act.kind == PANEL_ACT_REVERT_STEP else "Reverted Agent Turn",
                    "",
                )

    def time_travel_scrub(mut self, agent_id: UInt32, step_index: Int) -> Bool:
        """Shows an agent step's checkpoint. Scrubbing is read-only: while it is active saves,
        autosaves, undo / redo and edits are refused (`_time_travel_blocks_edit`), and the live world
        comes back on `exit_time_travel`."""
        var first = not self.agent_bus.time_travel_active
        if first:
            _ = self.commit_pending_flmat_edit()  # a half-typed material edit belongs to the live scene
        var live_raw = self._capture_world_blob() if first else List[UInt8]()
        if first and len(live_raw) == 0:
            self.push_toast("Time travel unavailable: could not snapshot the scene", 2, 3.5)
            return False
        var live_hash = self.state_hash()
        if first:
            _ = self.agent_checkpoints.save_live_snapshot(live_raw)
        var tt = self.agent_bus.enter_time_travel(agent_id, step_index, live_raw, live_hash)
        var si = self.agent_bus.find_session_index(agent_id)
        var shown = False
        if si >= 0 and step_index >= 0 and step_index < len(self.agent_bus.sessions[si].steps):
            var sid = Int(self.agent_bus.sessions[si].steps[step_index].step_id)
            var cp = self.agent_checkpoints.get_step_post_bytes(sid)
            if cp[0] and len(cp[1]) > 0:
                shown = self._restore_world_blob(cp[1])
            elif tt[0] and len(tt[1]) > 0:
                shown = self._restore_world_blob(tt[1])
        if shown:
            self._time_travel_view_blob = self._capture_world_blob()
        return shown

    def exit_time_travel(mut self):
        """Leaves time travel and puts the live world back (no-op when not time travelling)."""
        if not self.agent_bus.time_travel_active:
            return
        var ex = self.agent_bus.exit_time_travel()
        var live_cp = self.agent_checkpoints.restore_live_snapshot(clear_after=True)
        if live_cp[0] and len(live_cp[1]) > 0:
            _ = self._restore_world_blob(live_cp[1])
        elif ex[0] and len(ex[1]) > 0:
            _ = self._restore_world_blob(ex[1])
        self._time_travel_view_blob = List[UInt8]()

    def _time_travel_blocks_edit(mut self) -> Bool:
        """True (with a toast) while time travel is showing an agent checkpoint."""
        if not self.agent_bus.time_travel_active:
            return False
        self.push_toast("Time travel is read-only: exit it to edit", 1, 2.5)
        return True

    def ingest_mcp_events(mut self, var events: List[BridgedMcpEvent]) -> Int:
        for i in range(len(events)):
            if not events[i].has_world_pos and len(events[i].affected_entities) > 0:
                var eid = events[i].affected_entities[0]
                if self.is_alive_id(eid):
                    events[i].world_pos = self.entities[eid].translation
                    events[i].has_world_pos = True
                    events[i].half_extents = self.entities[eid].collider_half_extents
        var n = self.agent_runtime.drain_mcp_events(
            self.agent_bus, events, self.toast_mgr, Float64(self.ui_time_sec)
        )
        self._stamp_approval_editor_ids()
        return n

    def user_edits_after_agent_steps(self, agent_id: UInt32, step_ids: List[UInt64]) -> Int:
        """How many of the user's own undo entries (origin 0) come after the earliest transaction of
        these agent steps. Those edits would be lost by a whole-world checkpoint revert."""
        var si = self.agent_bus.find_session_index(agent_id)
        if si < 0:
            return 0
        var first_tx = UInt64(0)
        var agent_txs = List[UInt64]()
        for k in range(len(self.agent_bus.sessions[si].steps)):
            ref st = self.agent_bus.sessions[si].steps[k]
            if st.tx_id == UInt64(0):
                continue
            agent_txs.append(st.tx_id)  # the agent's own steps are never "user edits"
            if st.step_id not in step_ids:
                continue
            if first_tx == UInt64(0) or st.tx_id < first_tx:
                first_tx = st.tx_id
        if first_tx == UInt64(0):
            # Ordinary MCP / ACP steps do not record their transaction. Fall back to the undo history's
            # origin tags: the agent's earliest entry still on the stack bounds where its work starts.
            # This can only over-count (block a revert that was safe), never miss a user edit.
            for i in range(len(self.undo_stack)):
                ref tx = self.undo_stack[i]
                if tx.agent_id == Int(agent_id) and (first_tx == UInt64(0) or tx.tx_id < first_tx):
                    first_tx = tx.tx_id
        if first_tx == UInt64(0):
            return 0
        var n = 0
        for i in range(len(self.undo_stack)):
            ref tx = self.undo_stack[i]
            if tx.tx_id > first_tx and tx.agent_id == 0 and tx.tx_id not in agent_txs:
                n += 1
        return n

    def begin_edit_interaction(mut self):
        """Called by a panel when one of its fields changed. Starts a new undo step only when a new
        interaction starts: a mouse drag is one interaction (the seq also moves on press/release), and
        so is a typing session in one focused text field, so typing a name is one undo step, not one
        per keystroke."""
        if self.ui_ctx.mouse_down_l:
            return
        var f = self.ui_ctx.focused_id
        if f != 0 and f == self._edit_focus_id:
            return
        self._edit_focus_id = f
        self.ui_edit_seq += 1

    def agent_for_mcp_client(mut self, client_key: Int, client_name: String, from_acp_agent: Bool = False) -> UInt32:
        """The agent session that owns an MCP connection. A connection adopts the Studio-launched ACP
        agent only when the host proved it comes from that agent's process tree (`from_acp_agent`, via
        `SO_PEERCRED`); any other connection gets its own session, so approvals, gate policy and theatre
        attribution never leak between agents. A new session starts with the approval policy of the
        agent selected in the Agents panel (Edits when none is), never with no gate at all."""
        var known = self.mcp_client_agents.get(client_key, UInt32(0))
        if known != 0 and self.agent_bus.find_session_index(known) >= 0:
            return known
        var aid = UInt32(0)
        var acp = self.agent_runtime.acp_agent_id
        if from_acp_agent and acp != 0 and self.agent_bus.find_session_index(acp) >= 0:
            var claimed = False
            for item in self.mcp_client_agents.items():
                if item.value == acp and item.key != client_key:
                    claimed = True
            if not claimed:
                aid = acp
        if aid == 0:
            var nm = client_name if client_name.byte_length() > 0 and client_name != "unknown-mcp-client" else (
                "MCP client " + String(client_key)
            )
            var policy = APPROVAL_EDITS
            var sel = self.agent_bus.find_session_index(self.agent_bus.selected_agent_id)
            if sel >= 0:
                policy = self.agent_bus.sessions[sel].approval_policy
            aid = self.agent_bus.create_session(ROLE_EXTERNAL_MCP, nm, policy)
        self.mcp_client_agents[client_key] = aid
        return aid

    def release_mcp_client(mut self, client_key: Int):
        """A connection closed: its pending approvals are rejected (there is nobody to answer), its key
        stops claiming an agent (so Studio's ACP agent is adopted again when it reconnects), and its
        session is removed from the Agents panel if it never did anything."""
        var aid = self.mcp_client_agents.get(client_key, UInt32(0))
        try:
            if client_key in self.mcp_client_agents:
                _ = self.mcp_client_agents.pop(client_key)
        except:
            pass
        var i = len(self.agent_bus.pending_approvals) - 1
        while i >= 0:
            ref pa = self.agent_bus.pending_approvals[i]
            if pa.source == APPROVAL_SRC_MCP_TOOL and pa.mcp_client_key == client_key and not pa.resolved:
                _ = self.agent_bus.resolve_approval(pa.approval_id, False, False, "Agent disconnected")
            i -= 1
        if aid == 0 or aid == self.agent_runtime.acp_agent_id:
            return
        var si = self.agent_bus.find_session_index(aid)
        if si >= 0 and len(self.agent_bus.sessions[si].steps) == 0:
            _ = self.agent_bus.sessions.pop(si)
            if self.agent_bus.selected_agent_id == aid:
                self.agent_bus.selected_agent_id = self.agent_bus.sessions[0].agent_id if len(self.agent_bus.sessions) > 0 else UInt32(0)

    def reject_all_mcp_approvals(mut self, reason: String):
        """Rejects every queued MCP approval (used when the scene they were made against is replaced)."""
        var i = len(self.agent_bus.pending_approvals) - 1
        while i >= 0:
            ref pa = self.agent_bus.pending_approvals[i]
            if pa.source == APPROVAL_SRC_MCP_TOOL and not pa.resolved:
                _ = self.agent_bus.resolve_approval(pa.approval_id, False, False, reason)
            i -= 1
        self.mcp_replay_client.clear()
        self.mcp_replay_tool.clear()
        self.mcp_replay_args.clear()
        self.mcp_replay_step.clear()
        self.mcp_replay_editor_id.clear()

    def _stamp_approval_editor_ids(mut self):
        """Pins each newly queued approval ghost to its entity's stable `editor_id`, so approving it
        later cannot act on a different entity that reused the slot in the meantime."""
        for i in range(len(self.agent_bus.pending_approvals)):
            ref pa = self.agent_bus.pending_approvals[i]
            if pa.ghost.editor_id == 0 and pa.ghost.entity_id >= 0:
                pa.ghost.editor_id = self.editor_id_of(pa.ghost.entity_id)
            if pa.target_editor_id == 0 and pa.source == APPROVAL_SRC_MCP_TOOL and pa.raw_payload_json.byte_length() > 0:
                # Non-ghost calls aimed at an entity are pinned too (an explicit editor_id is kept as is).
                var raw = List[UInt8](capacity=pa.raw_payload_json.byte_length())
                for b in pa.raw_payload_json.as_bytes():
                    raw.append(b)
                var d = parse_json(raw)
                if d.valid and d.object_get(d.root, "editor_id") < 0:
                    var e = d.integer(d.object_get(d.root, "entity_id"), -1)
                    if e >= 0 and self.is_alive_id(e):
                        pa.target_editor_id = self.editor_id_of(e)

    def update_follow_agent_camera(mut self, dt: Float32):
        if not self.agent_bus.should_follow_camera():
            return
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(0.1))
        var alpha = Float32(1.0) if self.anim_mode == UI_ANIM_INSTANT else min(Float32(1.0), safe_dt * Float32(6.0))
        self.cam_target = self.cam_target + (self.agent_bus.focus_target_pos - self.cam_target) * alpha
        self.camera_rig.update(self._cam_dummy_world, self.cam_target, safe_dt)
        self._sync_from_camera_rig()

    def poll_agent_runtime(mut self, dt: Float32):
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(0.1))
        self.agent_runtime.poll_runtime(
            self.agent_bus,
            self.code_drawer,
            self.toast_mgr,
            Float64(self.ui_time_sec),
            safe_dt,
        )
        self.update_follow_agent_camera(safe_dt)

    def configure_gallery_shot(mut self, shot_idx: Int, width: Int = 1280, height: Int = 720):
        """Deterministically configures Studio state for any of the 6 `--gallery` showcase shots."""
        self.set_anim_mode(UI_ANIM_INSTANT)
        self.set_studio_theme(THEME_FLINT_OBSIDIAN)
        self.command_palette.close()
        self.key_chord_hud.cancel()
        self.confirm_modal_open = False
        self.gallery_play_hud_active = False
        self.menu.bar.group = -1
        if self.pie.mode != PIE_STOPPED:
            try:
                _ = self.pie_stop()
            except:
                self._abort_pie_session()
        self.workspace_mgr = WorkspacePresetManager()
        self.header_bar = StudioHeaderBar()
        self.viewport_hud = ViewportHud()
        self.toast_mgr = ToastManager()
        self.toasts.clear()
        self.job_tracker = JobProgressTracker()
        self.outliner_state = OutlinerPanelState()
        self.inspector_state = InspectorPanelState()
        self.asset_browser_state = AssetBrowserPanelState()
        self.code_drawer = CodeDrawer()
        self.agent_panel = AgentMissionControlPanel()
        self.agent_panel.anim_mode = UI_ANIM_INSTANT
        self.agent_bus = AgentTelemetryBus()
        self.agent_bus.follow_agent_camera = False
        self.agent_bus.theatre_overlays_enabled = True
        self.ui_ctx.animator.time_sec = Float32(0.0)
        self.ui_ctx.animator.snap_all()
        self.ui_time_sec = Float32(0.0)
        self._last_screen_w = Float32(max(320, width))
        self._last_screen_h = Float32(max(240, height))

        var sw = Float32(max(320, width))
        var sh = Float32(max(240, height))

        if shot_idx == 0:
            # 01_edit_default: Obsidian Edit workspace with hero SubD_MaterialOrb selected
            _ = self.apply_workspace_preset(WORKSPACE_EDIT, sw, sh)
        elif shot_idx == 1:
            # 02_command_palette: Frosted-glass Ctrl+K Command Palette open filtering "agent"
            _ = self.apply_workspace_preset(WORKSPACE_EDIT, sw, sh)
            self.open_command_palette(PALETTE_ALL, "agent")
        elif shot_idx == 2:
            # 03_agent_theatre: Agent Studio workspace with seeded multi-step Scene Builder + 3D Theatre
            _ = self.apply_workspace_preset(WORKSPACE_AGENT_STUDIO, sw, sh)
            var hero = self._hero_entity_id_and_pos()
            self.agent_runtime.seed_demo_agent_showcase(
                self.agent_bus, self.code_drawer, hero[0], hero[1], hero[2]
            )
            self.agent_bus.follow_agent_camera = False
            self.ui_time_sec = Float32(1.45)
            _ = WorkspacePresetManager.ensure_panel_visible(self.dock, PANEL_AGENTS)
        elif shot_idx == 3:
            # 04_code_scene_split: Code + Scene split workspace with Mojo syntax diff & Problems diagnostics
            _ = self.apply_workspace_preset(WORKSPACE_CODE_SCENE, sw, sh)
            var hero = self._hero_entity_id_and_pos()
            self.agent_runtime.seed_demo_agent_showcase(
                self.agent_bus, self.code_drawer, hero[0], hero[1], hero[2]
            )
            self.agent_bus.follow_agent_camera = False
            self.agent_bus.theatre_overlays_enabled = False
            self.code_drawer.view_mode = CODE_MODE_DIFF_UNIFIED
            self.code_drawer.problems_open = True
            _ = WorkspacePresetManager.ensure_panel_visible(self.dock, PANEL_CODE_DRAWER)
        elif shot_idx == 4:
            # 05_pie_running: Live PIE running with emerald viewport border & active Pause/Step/Stop pill
            _ = self.apply_workspace_preset(WORKSPACE_EDIT, sw, sh)
            _ = self.pie_play()
            self.pie_camera_ejected = False
            self.toast_mgr = ToastManager()
            self.toasts.clear()
        elif shot_idx == 5:
            # 06_play_hud: Maximized Play workspace with active PIE & gameplay HUD inside viewport
            _ = self.apply_workspace_preset(WORKSPACE_PLAY, sw, sh)
            _ = self.pie_play()
            self.pie_camera_ejected = False
            self.toast_mgr = ToastManager()
            self.toasts.clear()
            self.gallery_play_hud_active = True

    def release_pointer_lock(mut self):
        var win_res = self.app.resources.get_ptr_mut[WindowStateResource]()
        if win_res:
            for wi in range(len(win_res.value()[unsafe_offset=0].windows)):
                win_res.value()[unsafe_offset=0].windows[wi].set_pointer_lock(False)

    def selection_gizmo_rotation(self) -> Quat:
        """Gizmo axes for the current mode. Scale always uses the primary entity's own axes: a world-axis
        scale of a rotated object is a shear the transform cannot hold, so it would scale the wrong way."""
        if not self.gizmo_local_space and self.gizmo_mode != GIZMO_MODE_SCALE:
            return Quat.identity()
        return self._primary_world_rotation()

    def _primary_world_rotation(self) -> Quat:
        var eid = self.selected_entity_id
        if not self.is_alive_id(eid) and len(self.selected_entity_ids) > 0:
            for i in range(len(self.selected_entity_ids)):
                if self.is_alive_id(self.selected_entity_ids[i]):
                    eid = self.selected_entity_ids[i]
                    break
        var e = self.resolve_entity(eid)
        if not e.valid():
            return Quat.identity()
        var q = Quat.identity()
        var gt_opt = self.app.world.get_ptr[GlobalTransform](e)
        if gt_opt:
            q = gt_opt.value()[unsafe_offset=0].rotation
        else:
            var lt_opt = self.app.world.get_ptr[LocalTransform](e)
            if lt_opt:
                q = lt_opt.value()[unsafe_offset=0].rotation
        var lsq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
        if isfinite(lsq) and lsq > Float32(1.0e-8):
            return q.normalized()
        return Quat.identity()

    def toggle_gizmo_space(mut self):
        self.gizmo_local_space = not self.gizmo_local_space
        self.push_toast("Gizmo: Local Space" if self.gizmo_local_space else "Gizmo: World Space", 0, 1.8)

    def frame_selection(mut self):
        """Frames the selection's bounds so it fills the viewport panel and sits at the panel's centre.
        The renderer's projection is centred on the whole window, so the camera is offset to put the
        selection on the ray through the panel centre, at the distance that fits its bounding sphere."""
        var b = self.selection_bounds()
        if not b[0]:
            return
        var center = (b[1] + b[2]) * Float32(0.5)
        var radius = max(Float32(0.3), (b[2] - b[1]).length() * Float32(0.5))
        var fwd = self.cam_target - self.cam_pos
        var flen = fwd.length()
        if not isfinite(flen) or flen < Float32(1.0e-4):
            fwd = Vec3(Float32(0.6), Float32(-0.37), Float32(0.71))
        fwd = fwd.normalized()
        var right = fwd.cross(Vec3(0.0, 1.0, 0.0))
        if right.length_squared() < Float32(1.0e-6):
            right = Vec3(1.0, 0.0, 0.0)
        right = right.normalized()
        var up = right.cross(fwd).normalized()
        var vx = self.viewport_xf
        var tan_y = tan(self.fov_y_deg * Float32(3.14159265 / 360.0))
        var tan_x = tan_y * (vx.target_w / max(Float32(1.0), vx.target_h))
        # Fit the sphere inside the panel's share of the frustum (the panel may be much smaller than
        # the window the projection covers).
        var pan_tan_y = tan_y * (vx.panel.h / max(Float32(1.0), vx.target_h))
        var pan_tan_x = tan_x * (vx.panel.w / max(Float32(1.0), vx.target_w))
        var fit_tan = max(Float32(0.02), min(pan_tan_x, pan_tan_y))
        var dist = max(Float32(1.0), min(Float32(500.0), radius * Float32(1.15) / fit_tan + radius))
        var ndc_x = ((vx.panel.x + vx.panel.w * Float32(0.5)) / max(Float32(1.0), vx.target_w)) * Float32(2.0) - Float32(1.0)
        var ndc_y = Float32(1.0) - ((vx.panel.y + vx.panel.h * Float32(0.5)) / max(Float32(1.0), vx.target_h)) * Float32(2.0)
        var ray_dir = (fwd + right * (ndc_x * tan_x) + up * (ndc_y * tan_y)).normalized()
        var eye = center - ray_dir * dist
        self.look_at(eye, eye + fwd * dist)
        var rps_opt = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_opt:
            ref rps = rps_opt.value()[unsafe_offset=0]
            rps.editor_camera_override = True
            rps.editor_camera = RenderCameraPacket(
                self.cam_pos,
                self.cam_target,
                Vec3(0.0, 1.0, 0.0),
                self.fov_y_deg,
                self.near_z,
                Float32(1000.0),
                Float32(1.08),
            )
            rps.has_editor_camera = True

    def copy_selected(mut self) -> Int:
        """Copies the selection by stable `editor_id`, so a paste after a delete / undo / reload never
        duplicates whatever entity reused the old slot."""
        self._normalize_selection_from_primary()
        self._ensure_editor_ids()
        self.clipboard_editor_ids.clear()
        for i in range(len(self.selected_entity_ids)):
            if self.is_alive_id(self.selected_entity_ids[i]):
                var eid = self.editor_id_of(self.selected_entity_ids[i])
                if eid != 0:
                    self.clipboard_editor_ids.append(eid)
        if len(self.clipboard_editor_ids) == 0 and self.is_alive_id(self.selected_entity_id):
            var eid0 = self.editor_id_of(self.selected_entity_id)
            if eid0 != 0:
                self.clipboard_editor_ids.append(eid0)
        if len(self.clipboard_editor_ids) > 0:
            self.push_toast("Copied " + String(len(self.clipboard_editor_ids)) + " entity(s)", 0, 1.8)
        return len(self.clipboard_editor_ids)

    def clipboard_slots(self) -> List[Int]:
        """Current slots of the copied entities that still exist."""
        var out = List[Int]()
        for i in range(len(self.clipboard_editor_ids)):
            var slot = self.slot_for_editor_id(self.clipboard_editor_ids[i])
            if slot >= 0 and self.is_alive_id(slot):
                out.append(slot)
        return out^

    def paste_clipboard(mut self, record_undo: Bool = True) -> List[Int]:
        if self._time_travel_blocks_edit():
            return List[Int]()
        var srcs = self._subtree_roots(self.clipboard_slots())
        var created = List[Int]()
        if len(srcs) == 0:
            return created^

        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        for i in range(len(srcs)):
            var nid = self._duplicate_entity_raw(srcs[i])
            if nid >= 0:
                created.append(nid)
        if len(created) == 0:
            return created^

        self.selected_entity_ids = created.copy()
        self.selected_entity_id = created[len(created) - 1]
        self.selection_anchor_id = created[0]
        self._sync_ecs_and_physics()

        if record_undo:
            var aff = created.copy()
            self._record_world_tx(
                TX_BATCH_EDIT,
                self.selected_entity_id,
                aff^,
                before_blob^,
                prev_ids,
                prev_primary,
                prev_anchor,
                "Paste " + String(len(created)) + " Entities",
            )
        self.push_toast("Pasted " + String(len(created)) + " entity(s)", 0, 1.8)
        return created^

    def _clear_render_and_resolver_scene_caches(mut self):
        var rps_opt = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_opt:
            ref rps = rps_opt.value()[unsafe_offset=0]
            for item in rps.override_slots.items():
                if item.value.slot_id >= 0:
                    rps.free_override_slots.append(item.value.slot_id)
            rps.override_slots.clear()
            rps.studio_override_mat_ids.clear()
            rps.entity_smear_scales.clear()
            rps.skinned_entries.clear()
            rps.skinned_model_ids.clear()
            rps.skinned_invalidated.clear()
            rps.preview_bone_palettes.clear()
            rps.preview_prev_bone_palettes.clear()
            rps.preview_world_transforms.clear()
        var res_opt = self.app.resources.get_ptr_mut[SceneAssetResolver]()
        if res_opt:
            res_opt.value()[unsafe_offset=0].skel_entity_rig_keys.clear()
            res_opt.value()[unsafe_offset=0].mesh_default_mat_ids.clear()
        var sync_style_opt = self.app.resources.get_ptr_mut[StudioFlmatStyleSyncResource]()
        if sync_style_opt:
            sync_style_opt.value()[unsafe_offset=0].synced_entity_epochs.clear()
            sync_style_opt.value()[unsafe_offset=0].known_internal_mat_epochs.clear()
            sync_style_opt.value()[unsafe_offset=0].external_mat_reload_epochs.clear()

    def commit_pending_flmat_edit(mut self) -> Bool:
        """Saves a buffered .flmat text edit (one disk write and one undo step for the whole edit)."""
        if self.flmat_pending_path.byte_length() == 0:
            return False
        var path = self.flmat_pending_path
        var ref_path = self.flmat_pending_ref
        var eid = (
            self.slot_for_editor_id(self.flmat_pending_editor_id) if self.flmat_pending_editor_id != 0 else self.flmat_pending_eid
        )
        var asset = self.flmat_pending.copy()
        var stamp = self.flmat_pending_stamp
        self.flmat_pending_path = String("")
        self.flmat_pending_ref = String("")
        self.flmat_pending_eid = -1
        self.flmat_pending_editor_id = UInt64(0)
        self.flmat_pending_stamp = (-1, 0)
        var now_stamp = file_stamp(path)
        if stamp[0] >= 0 and (now_stamp[0] != stamp[0] or now_stamp[1] != stamp[1]):
            self.push_toast("Material changed on disk; your unsaved edit to it was not written", 1, 4.0)
            return False
        _ = save_flmat_file(path, asset)
        if eid >= 0 and self.is_alive_id(eid):
            _ = self._assign_flmat_to_entity(eid, ref_path)
        return True

    def flush_stale_flmat_edit(mut self):
        """Saves a pending material edit the Inspector no longer shows (deselected, collapsed, closed)."""
        if self.flmat_pending_path.byte_length() > 0 and self.ui_time_sec - self.flmat_pending_touch_sec > Float32(0.3):
            _ = self.commit_pending_flmat_edit()

    def request_load_scene(mut self, path: String = "", force: Bool = False) -> Bool:
        """Interactive scene load: asks Save / Discard / Cancel first when the current scene is dirty,
        because `load_scene_file` replaces the world (the load itself is one undo step)."""
        if not force and self.dirty:
            self.pending_load_path = path
            self.confirm_modal_open = True
            self.confirm_modal_action = 3
            return False
        return self.load_scene_file(path)

    def new_scene(mut self, force: Bool = False) -> Bool:
        """Empties the world into an *untitled* scene (the first save asks where, so it can never
        overwrite the level that was open). New is one undoable step, like a load: undo brings the old
        scene, its file path and its saved / unsaved state back."""
        _ = self.commit_pending_flmat_edit()
        if not force and self.dirty:
            self.confirm_modal_open = True
            self.confirm_modal_action = 2
            return False
        self.exit_time_travel()
        if self.pie.mode != PIE_STOPPED:
            self._abort_pie_session()
        var before_blob = self._capture_world_blob()
        if len(before_blob) == 0:
            self.push_toast("New scene refused: the current scene could not be snapshotted for undo", 2, 4.0)
            return False
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id
        var prev_scene_path = self.scene_file_path
        var prev_saved = self._saved_tx_id
        if self.dirty:
            self.discard_autosave()  # the user chose Discard for this scene's unsaved work
        for i in range(len(self.app.world.entities)):
            if self.app.world.entities[i].alive:
                var e = Entity(UInt32(i), self.app.world.entities[i].generation)
                _ = self.app.world.despawn(e)
        self.clipboard_editor_ids.clear()
        self.clear_selection()
        self.reject_all_mcp_approvals("Scene changed before approval")
        self.gpu_materials.clear_entity_slots()
        self._pie_frame_updated = False
        self._int_edit_wid = UInt64(0)
        self._int_edit_entity_id = -1
        self._int_edit_buf = ""
        self.preview_anim_time = Float32(0.0)
        self._last_preview_anim_time = Float32(0.0)
        self._clear_render_and_resolver_scene_caches()
        self._reset_physics_world_and_sync()
        self.scene_file_path = String("")
        self.discard_autosave()  # a stale untitled autosave must not be offered for this new scene
        var top_before = self._top_tx_id()
        self._record_world_tx(
            TX_BATCH_EDIT, -1, List[Int](), before_blob^, prev_ids, prev_primary, prev_anchor, "New Scene"
        )
        if self._top_tx_id() != top_before:
            ref top = self.undo_stack[len(self.undo_stack) - 1]
            top.changes_scene_path = True
            top.prev_scene_path = prev_scene_path
            top.next_scene_path = String("")
            top.prev_saved_tx_id = prev_saved
            top.load_leaves_clean = True
        self.mark_dirty(False)
        self.push_toast("New scene created", 0, 2.5)
        return True

    def request_quit(mut self) -> Bool:
        """Returns True if quit was deferred by an unsaved-changes modal; False if clean to quit immediately."""
        _ = self.commit_pending_flmat_edit()  # a half-typed material path is still the user's edit
        if self.dirty:
            self.confirm_modal_open = True
            self.confirm_modal_action = 1
            return True
        self.quit_requested = True
        return False

    def bake_navmesh(mut self) -> Int:
        var nav_pw = PhysicsWorld(gravity=Float32(0.0))
        var added = 0
        for i in range(len(self.app.world.entities)):
            if not self.app.world.entities[i].alive:
                continue
            var e = Entity(UInt32(i), self.app.world.entities[i].generation)
            var meta_opt = self.app.world.get_ptr[StudioEntityMetaComponent](e)
            if meta_opt and not meta_opt.value()[unsafe_offset=0].enabled:
                continue
            var col_opt = self.app.world.get_ptr[ColliderComponent](e)
            if not col_opt:
                continue
            ref col = col_opt.value()[unsafe_offset=0]
            if col.is_trigger:
                continue
            var rb_opt = self.app.world.get_ptr[RigidBodyComponent](e)
            if rb_opt and rb_opt.value()[unsafe_offset=0].motion_mode == BODY_MOTION_DYNAMIC:
                continue
            var pos = Vec3()
            var rot = Quat.identity()
            var gt_opt = self.app.world.get_ptr[GlobalTransform](e)
            if gt_opt:
                pos = gt_opt.value()[unsafe_offset=0].translation
                rot = gt_opt.value()[unsafe_offset=0].rotation
            else:
                var lt_opt = self.app.world.get_ptr[LocalTransform](e)
                if lt_opt:
                    pos = lt_opt.value()[unsafe_offset=0].translation
                    rot = lt_opt.value()[unsafe_offset=0].rotation
            var wpos = pos + rot.rotate(col.local_offset)
            var wrot = (rot * col.local_rot).normalized()
            if col.shape == BODY_SHAPE_SPHERE:
                _ = nav_pw.add_sphere(wpos, max(Float32(0.05), col.radius), Float32(0.0), wrot)
            elif col.shape == BODY_SHAPE_CAPSULE:
                _ = nav_pw.add_capsule(wpos, max(Float32(0.05), col.radius), max(Float32(0.05), col.half_height), Float32(0.0), wrot)
            elif col.shape == BODY_SHAPE_CONVEX_HULL and len(col.points) >= 4:
                var hull = ConvexHull.build_quickhull(col.points)
                _ = nav_pw.add_convex_hull(wpos, hull, Float32(0.0), wrot)
            else:
                var he = Vec3(
                    max(Float32(0.05), col.half_extents.x),
                    max(Float32(0.05), col.half_extents.y),
                    max(Float32(0.05), col.half_extents.z),
                )
                _ = nav_pw.add_box(wpos, he, Float32(0.0), wrot)
            added += 1
        if added == 0:
            _ = nav_pw.add_box(Vec3(0.0, -0.25, 0.0), Vec3(12.0, 0.25, 12.0), Float32(0.0))

        var poly_count = 0
        var nav_res = self.app.resources.get_ptr_mut[NavPluginStateResource]()
        if nav_res:
            poly_count = nav_res.value()[unsafe_offset=0].bake_from_physics(nav_pw)
            nav_res.value()[unsafe_offset=0].draw_debug = self.show_navmesh_overlay
        if poly_count > 0:
            self.push_toast("NavMesh baked (" + String(poly_count) + " polys)", 0, 2.5)
        else:
            self.push_toast("NavMesh bake produced 0 polys", 1, 3.0)
        return poly_count

    def spawn_card_entity(mut self, card_idx: Int) -> Int:
        var before_blob = self._capture_world_blob()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        var label: String
        if card_idx == 1:
            label = "Cube"
        elif card_idx == 2:
            label = "PointLight"
        elif card_idx == 3:
            label = "SpotLight"
        elif card_idx == 4:
            label = "Camera"
        elif card_idx == 5:
            label = "AudioEmitter"
        else:
            label = "Empty"

        var e = self.app.world.spawn_empty()
        var eid = Int(e.index)
        var pos = Vec3(0.0, 1.0, 0.0) if card_idx != 1 else Vec3(0.0, 0.5, 0.0)
        var lt = LocalTransform(pos, Quat.identity(), Vec3(1.0, 1.0, 1.0))
        var gt = GlobalTransform.from_local(lt)
        _ = self.app.world.insert[LocalTransform](e, lt)
        _ = self.app.world.insert[GlobalTransform](e, gt.copy())
        _ = self.app.world.insert[PrevGlobalTransform](e, PrevGlobalTransform(gt^))
        _ = self.app.world.insert[StudioEntityMetaComponent](
            e,
            StudioEntityMetaComponent(
                name=label + "_" + String(eid),
                collider_kind=COLLIDER_BOX if card_idx == 1 else COLLIDER_NONE,
                collider_half_extents=Vec3(0.5, 0.5, 0.5),
            ),
        )
        _ = self.app.world.insert[StudioStyleComponent](e, StudioStyleComponent())
        if card_idx == 1:
            _ = self.app.world.insert[MeshRendererComponent](e, MeshRendererComponent(0, -1, True, True))
            _ = self.app.world.insert[MaterialOverrideComponent](
                e, MaterialOverrideComponent(-1, Vec3(0.82, 0.84, 0.88), Float32(0.38), Float32(0.10), Vec3(), True)
            )
            self._apply_collider_and_rb(e, COLLIDER_BOX, Vec3(0.5, 0.5, 0.5), Float32(0.0), Float32(0.32), Float32(0.55), Vec3(), Vec3())
        elif card_idx == 2:
            _ = self.app.world.insert[PointLightComponent](
                e, PointLightComponent(radiance=Vec3(12.0, 11.4, 10.56), radius=Float32(14.0), enabled=True)
            )
        elif card_idx == 3:
            _ = self.app.world.insert[SpotLightComponent](
                e,
                SpotLightComponent(
                    radiance=Vec3(16.0, 15.36, 14.4),
                    range=Float32(25.0),
                    inner_deg=Float32(25.0),
                    outer_deg=Float32(42.0),
                    direction=Vec3(0.0, -1.0, 0.0),
                    enabled=True,
                ),
            )
        elif card_idx == 4:
            _ = self.app.world.insert[CameraComponent](
                e, CameraComponent(fov_y_deg=Float32(60.0), near_z=Float32(0.1), far_z=Float32(500.0), active=False)
            )
        elif card_idx == 5:
            _ = self.app.world.insert[AudioEmitterComponent](e, AudioEmitterComponent())

        self.select_entity(eid)
        self._sync_ecs_and_physics()
        var aff = List[Int]()
        aff.append(eid)
        self._record_world_tx(
            TX_SPAWN_ENTITY, eid, aff^, before_blob^, prev_ids, prev_primary, prev_anchor, "Spawn " + label
        )
        self.push_toast("Spawned " + label, 0, 2.0)
        return eid

    # ---- Editor Input ----

    def handle_command_menu(mut self, inp: InputState, width: Int, height: Int) -> Bool:
        if self.ui_ctx.modal_depth > 0 or self.ui_ctx.had_modal_prev_frame or self.command_palette.is_open:
            self.menu.bar.group = -1
            return False
        self.menu.poll()
        # The menu bar is drawn into the scaled `ui_ctx` list, so lay it out and hit-test it in logical pixels.
        var ui_s = self.effective_ui_scale()
        var lw = Float32(width) / ui_s
        var lh = Float32(height) / ui_s
        self.dock.solve_layout(
            RectF(0.0, STUDIO_HEADER_BAR_H, lw, max(Float32(64.0), lh - STUDIO_HEADER_BAR_H - STUDIO_STATUS_BAR_H))
        )
        var panel = self.dock.find_panel("Toolbar")
        var has_toolbar = panel[0] >= 0 and panel[2]
        ref bar = self.menu.bar
        if has_toolbar:
            bar.x = Int(panel[3].x) + 4
            bar.y = Int(panel[3].y)
            bar.height = min(30, max(1, Int(panel[3].h)))
        else:
            bar.x = 4
            bar.y = 0
            bar.height = Int(STUDIO_HEADER_BAR_H)
        bar.screen_w = Int(lw)
        bar.screen_h = Int(lh)
        var in_play = self.pie.mode != PIE_STOPPED
        bar.groups[0].items[0].enabled = not in_play
        bar.groups[0].items[1].enabled = not in_play
        bar.groups[0].items[2].enabled = not in_play
        bar.groups[1].items[0].enabled = (not in_play) and Bool(self.undo_stack)
        bar.groups[1].items[1].enabled = (not in_play) and Bool(self.redo_stack)
        bar.groups[1].items[2].enabled = (not in_play) and self.selected_count() > 0
        bar.groups[1].items[3].enabled = (not in_play) and self.selected_count() > 0
        bar.groups[2].items[3].title = "Disable grid snapping" if self.snap_enabled else "Enable grid snapping"
        bar.groups[2].items[4].title = "Gizmo: Local Space" if self.gizmo_local_space else "Gizmo: World Space"
        bar.groups[2].items[5].enabled = self.selected_count() > 0
        bar.groups[3].items[0].title = "Resume simulation" if self.pie.mode==PIE_PAUSED else "Play scene"
        bar.groups[3].items[0].enabled = self.pie.mode!=PIE_PLAYING
        bar.groups[3].items[1].enabled = self.pie.mode==PIE_PLAYING
        bar.groups[3].items[2].enabled = self.pie.mode!=PIE_STOPPED
        bar.groups[3].items[3].enabled = self.pie.mode!=PIE_PLAYING
        if len(bar.groups[3].items) > 4:
            bar.groups[3].items[4].enabled = (
                self.gameplay_so_path.byte_length() > 0
                or self.hot_reload_host.is_loaded()
                or self.project_root.byte_length() > 0
            )
        bar.groups[4].items[0].enabled = self.menu.has_project and self.menu.has_launcher and self.menu.library_pid <= 0
        if len(bar.groups) >= 7:
            bar.groups[6].items[1].title = "Hide NavMesh Overlay" if self.show_navmesh_overlay else "Show NavMesh Overlay"
            bar.groups[6].items[2].title = "Hide Collider Overlay" if self.show_collider_overlay else "Show Collider Overlay"
        if len(bar.groups) >= 8:
            var can_pkg = (not in_play) and (
                self.menu.has_project
                or (self.project_root.byte_length() > 0 and file_size(self.project_root + "/flint.toml") >= 0)
            )
            bar.groups[7].items[0].enabled = can_pkg
            bar.groups[7].items[1].enabled = can_pkg
        if not has_toolbar and bar.group < 0:
            return False
        var action = bar.input(inp, Int(inp.last_mouse_x / ui_s), Int(inp.last_mouse_y / ui_s))
        if action==scene_menu.SAVE and not in_play:
            var ok = self.save_scene_file()
            self.menu.message = "Scene saved" if ok else "Could not save scene"
        elif action==scene_menu.LOAD and not in_play:
            var ok = self.request_load_scene()
            self.menu.message = "Scene loaded" if ok else ("" if self.confirm_modal_open else "Could not load scene")
        elif action==scene_menu.NEW_SCENE and not in_play:
            _ = self.new_scene(force=False)
        elif action==scene_menu.OPEN_SCENE and not in_play:
            self.open_file_modal(False)
        elif action==scene_menu.SAVE_AS and not in_play:
            self.open_file_modal(True)
        elif action==scene_menu.RECOVER_AUTOSAVE and not in_play:
            if self.autosave_available():
                _ = self.recover_autosave()
            else:
                self.push_toast("No autosave newer than the scene file", 0, 2.5)
        elif action==scene_menu.SELECT_ALL and not in_play:
            self.select_all_entities()
        elif action==scene_menu.RENAME and not in_play:
            self.begin_rename_selected()
        elif action==scene_menu.PIVOT_MODE:
            self.toggle_pivot_mode()
        elif action==scene_menu.WINDOW_RESET_LAYOUT:
            self.reset_layout()
        elif action==scene_menu.WINDOW_CONSOLE:
            self.console_open = True
        elif action==scene_menu.HELP_SHORTCUTS:
            self.help_open = True
        elif action==scene_menu.HELP_PALETTE:
            self.open_command_palette(PALETTE_ALL)
        elif action==scene_menu.UNDO and not in_play:
            _ = self.undo()
        elif action==scene_menu.REDO and not in_play:
            _ = self.redo()
        elif action==scene_menu.DUPLICATE and not in_play and self.selected_count()>0:
            if len(self.selected_entity_ids) > 1:
                _ = self.duplicate_selected(record_undo=True)
            elif self.selected_entity_id >= 0:
                _ = self.duplicate_entity(self.selected_entity_id, record_undo=True)
        elif action==scene_menu.DELETE_SEL and not in_play and self.selected_count()>0:
            if len(self.selected_entity_ids) > 1:
                _ = self.despawn_selected(record_undo=True)
            elif self.selected_entity_id >= 0:
                _ = self.despawn_entity(self.selected_entity_id, record_undo=True)
        elif action==scene_menu.MOVE:
            self.gizmo_mode = GIZMO_MODE_TRANSLATE
        elif action==scene_menu.ROTATE:
            self.gizmo_mode = GIZMO_MODE_ROTATE
        elif action==scene_menu.SCALE:
            self.gizmo_mode = GIZMO_MODE_SCALE
        elif action==scene_menu.SNAP:
            self.snap_enabled = not self.snap_enabled
        elif action==scene_menu.GIZMO_SPACE:
            self.toggle_gizmo_space()
        elif action==scene_menu.FRAME_SEL:
            self.frame_selection()
        elif action==scene_menu.PLAY:
            _ = self.pie_play()
        elif action==scene_menu.PAUSE:
            _ = self.pie_pause()
        elif action==scene_menu.STOP:
            try:
                _ = self.pie_stop()
            except err:
                self.menu.message = String(err)
        elif action==scene_menu.STEP:
            _ = self.pie_step(1, Float32(1.0 / 60.0))
        elif action==scene_menu.RELOAD_SO:
            var ok = self.reload_gameplay_plugin()
            self.menu.message = "Reloaded gameplay .so" if ok else ("Reload .so failed: " + self.hot_reload_host.last_error)
        elif action==scene_menu.SPAWN_EMPTY:
            _ = self.spawn_card_entity(0)
        elif action==scene_menu.SPAWN_CUBE:
            _ = self.spawn_card_entity(1)
        elif action==scene_menu.SPAWN_POINT_LIGHT:
            _ = self.spawn_card_entity(2)
        elif action==scene_menu.SPAWN_SPOT_LIGHT:
            _ = self.spawn_card_entity(3)
        elif action==scene_menu.SPAWN_CAMERA:
            _ = self.spawn_card_entity(4)
        elif action==scene_menu.SPAWN_AUDIO:
            _ = self.spawn_card_entity(5)
        elif action==scene_menu.BAKE_NAVMESH:
            var pc = self.bake_navmesh()
            self.menu.message = "NavMesh baked (" + String(pc) + " polys)"
        elif action==scene_menu.TOGGLE_NAV_OVERLAY:
            self.show_navmesh_overlay = not self.show_navmesh_overlay
            var nav_res = self.app.resources.get_ptr_mut[NavPluginStateResource]()
            if nav_res:
                nav_res.value()[unsafe_offset=0].draw_debug = self.show_navmesh_overlay
        elif action==scene_menu.TOGGLE_COL_OVERLAY:
            self.show_collider_overlay = not self.show_collider_overlay
        elif action==scene_menu.SOURCE:
            self.source_editor.requested = True
        elif action==scene_menu.MOTION_TAKES:
            self.menu.open_takes()
        elif action==scene_menu.KIMODO:
            self.menu.open_kimodo()
        elif action==scene_menu.LIBRARY:
            self.menu.open_library()
        elif action == scene_menu.PACKAGE and not in_play:
            var pkg_res = self.package_project()
            if pkg_res[0]:
                self.menu.message = "Packaged -> " + pkg_res[1]
                self.push_toast("Packaged -> " + pkg_res[1], 0, 3.0)
            else:
                self.menu.message = pkg_res[1]
                self.push_toast(pkg_res[1], 2, 4.0)
        elif action == scene_menu.PACKAGE_PROFILES and not in_play:
            var launcher = _resolve_flint_launcher_path()
            var proj = self.project_root if self.project_root.byte_length() > 0 else env_value("FLINT_STUDIO_PROJECT")
            if launcher.byte_length() == 0 or proj.byte_length() == 0:
                var msg = String("Could not locate tools/flint.sh or project")
                self.menu.message = msg
                self.push_toast(msg, 2, 3.5)
            else:
                _ = make_dirs(proj + "/build")
                var args = List[String]()
                args.append("/bin/bash")
                args.append(launcher)
                args.append("studio")
                args.append("--build-profiles")
                var rc = run_process_redirected(args, cwd=proj, log_path=proj + "/build/package.log")
                var msg = String("Profile package complete") if rc == Int32(0) else String("Profile package failed (see build/package.log)")
                self.menu.message = msg
                self.push_toast(msg, 0 if rc == Int32(0) else 2, 3.5)
        return action>=0

    def process_editor_input(mut self, input: InputState, dt: Float32) -> Bool:
        """Evaluates `InputMap` editor hotkeys (`W/E/R/X/F`, `Ctrl+Z`, `Ctrl+Shift+Z`/`Ctrl+Y`, `F5-F7`, `Ctrl+Q` quit)
        and `CameraRig` orbit/fly while respecting `UiContext` keyboard/mouse capture barriers and `StudioPlay`."""
        self.tick_toasts(dt)
        self.input_map.update(input, dt)
        var modal_open = self.confirm_modal_open or self.ui_ctx.modal_depth > 0 or self.ui_ctx.had_modal_prev_frame
        var ctrl_down = input.key_down(341) or input.key_down(345)
        var shift_down = input.key_down(340) or input.key_down(344)
        var alt_down = input.key_down(342) or input.key_down(346)

        # Global Studio v2 Command Palette toggle (`Ctrl+K`) & Quick Spawn (`Shift+A`)
        if not modal_open and not self.input_rebind_active:
            if ctrl_down and input.key_pressed(75):  # Ctrl+K
                if self.command_palette.is_open:
                    self.command_palette.close()
                else:
                    self.open_command_palette(PALETTE_ALL)
                return True
            if self.command_palette.is_open:
                if input.key_pressed(UI_KEY_BACKSPACE) and self.command_palette.raw_query.byte_length() > 0:
                    var nq = utf8_drop_last_char(self.command_palette.raw_query)
                    self.command_palette.set_query(nq)
                elif input.text_input_utf8.byte_length() > 0:
                    var q_acc = self.command_palette.raw_query
                    for b in input.text_input_utf8.as_bytes():
                        if b >= 32 and b < 127:
                            var ch_b = List[UInt8](capacity=1)
                            ch_b.append(b)
                            q_acc += String(unsafe_from_utf8=ch_b)
                    self.command_palette.set_query(q_acc)
                var pal_act = self.command_palette.handle_key(
                    key_up=input.key_pressed(UI_KEY_UP),
                    key_down=input.key_pressed(UI_KEY_DOWN),
                    key_tab=input.key_pressed(UI_KEY_TAB),
                    key_shift=shift_down,
                    key_enter=input.key_pressed(UI_KEY_ENTER),
                    key_esc=input.key_pressed(UI_KEY_ESCAPE),
                )
                if pal_act.executed:
                    _ = self.execute_palette_action(pal_act)
                return True

        var kbd_captured = (
            modal_open
            or self.command_palette.is_open
            or self.agent_panel.wants_keyboard()
            or self.code_drawer.has_keyboard_focus
            or self.ui_ctx.wants_keyboard()
            or self.menu.bar.group >= 0
        )
        var _sc = self.effective_ui_scale()
        var _lmx = input.last_mouse_x / _sc
        var _lmy = input.last_mouse_y / _sc
        var mouse_captured = (
            modal_open
            or self.command_palette.is_open
            or self.ui_ctx.wants_mouse()
            or self.header_bar.contains_point(_lmx, _lmy)
            or self.viewport_hud.contains_point(_lmx, _lmy)
            or self.menu.bar.group >= 0
        )
        var in_play = self.pie.mode != PIE_STOPPED

        if not kbd_captured:
            if alt_down and input.key_pressed(65):  # Alt+A: Toggle Agent Mission Control
                _ = self.execute_palette_command(PAL_CMD_AGENT_TOGGLE_PANEL)
                _ = begin_studio_chord(
                    self.key_chord_hud,
                    "Alt+A",
                    gizmo_local=self.gizmo_local_space,
                    snap_enabled=self.snap_enabled,
                    follow_agent_cam=self.agent_bus.follow_agent_camera,
                    show_agent_overlays=self.agent_bus.theatre_overlays_enabled,
                )
            elif shift_down and not ctrl_down and not alt_down and not in_play and input.key_pressed(65):  # Shift+A: Quick Spawn
                self.open_command_palette(PALETTE_SPAWN)
                return True
            elif ctrl_down and shift_down and not alt_down and input.key_pressed(61):  # Ctrl + '+' (Shift+= on US)
                self.set_ui_scale(self.ui_scale + Float32(0.25))
                self.push_toast("UI Scale: " + String(Int(self.ui_scale * Float32(100.0) + Float32(0.5))) + "%", 0, 1.8)
            elif ctrl_down and not shift_down and not alt_down:
                if input.key_pressed(61) or input.key_pressed(334):  # Ctrl + = / + : UI Zoom In
                    self.set_ui_scale(self.ui_scale + Float32(0.25))
                    self.push_toast("UI Scale: " + String(Int(self.ui_scale * Float32(100.0) + Float32(0.5))) + "%", 0, 1.8)
                elif input.key_pressed(45) or input.key_pressed(333):  # Ctrl + - : UI Zoom Out
                    self.set_ui_scale(self.ui_scale - Float32(0.25))
                    self.push_toast("UI Scale: " + String(Int(self.ui_scale * Float32(100.0) + Float32(0.5))) + "%", 0, 1.8)
                elif input.key_pressed(48):  # Ctrl + 0 : Reset UI Scale 200%
                    self.set_ui_scale(Float32(2.0))
                    self.push_toast("UI Scale: 200%", 0, 1.8)
                elif input.key_pressed(49):
                    _ = self.apply_workspace_preset(WORKSPACE_EDIT)
                elif input.key_pressed(50):
                    _ = self.apply_workspace_preset(WORKSPACE_AGENT_STUDIO)
                elif input.key_pressed(51):
                    _ = self.apply_workspace_preset(WORKSPACE_CODE_SCENE)
                elif input.key_pressed(52):
                    _ = self.apply_workspace_preset(WORKSPACE_LOOKDEV)
                elif input.key_pressed(53):
                    _ = self.apply_workspace_preset(WORKSPACE_PLAY)
            elif input.key_pressed(UI_KEY_ESCAPE) and self.key_chord_hud.active:
                self.key_chord_hud.cancel()

        if not kbd_captured:
            if in_play:
                if self.input_map.just_pressed("pie_release_cursor"):
                    self.release_pointer_lock()
                if self.pie.mode == PIE_PLAYING and (
                    self.input_map.just_pressed("pie_toggle_camera")
                    or (self.input_map.is_down("pie_toggle_camera") and input.key_pressed(290))
                ):
                    self.pie_camera_ejected = not self.pie_camera_ejected
                    if self.pie_camera_ejected:
                        self.release_pointer_lock()
                    var rps_tog = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
                    if rps_tog:
                        rps_tog.value()[unsafe_offset=0].editor_camera_override = self.pie_camera_ejected
                    self.push_toast(
                        "PIE Camera: Editor (Shift+F1)" if self.pie_camera_ejected else "PIE Camera: Game (Shift+F1)",
                        0,
                        1.8,
                    )
                if self.input_map.just_pressed("pie_stop"):
                    try:
                        _ = self.pie_stop()
                    except:
                        self._abort_pie_session()
                elif self.input_map.just_pressed("pie_toggle"):
                    try:
                        _ = self.pie_stop()
                    except:
                        self._abort_pie_session()
                if self.input_map.just_pressed("pie_pause"):
                    if self.pie.mode == PIE_PLAYING:
                        _ = self.pie_pause()
                    elif self.pie.mode == PIE_PAUSED:
                        _ = self.pie_play()
                if self.input_map.just_pressed("pie_step"):
                    _ = self.pie_step(1, Float32(1.0 / 60.0))
            else:
                if self.input_map.just_pressed("editor_quit"):
                    return self.request_quit()
                if self.input_map.just_pressed("source_editor"):
                    self.source_editor.requested = True
                if self.input_map.just_pressed("editor_undo"):
                    _ = self.undo()
                if self.input_map.just_pressed("editor_redo"):
                    _ = self.redo()
                if self.input_map.just_pressed("editor_save"):
                    _ = self.save_scene_file()
                if self.input_map.just_pressed("editor_load"):
                    _ = self.request_load_scene()
                if self.input_map.just_pressed("editor_new"):
                    _ = self.new_scene(force=False)
                if self.input_map.just_pressed("editor_save_as"):
                    self.open_file_modal(True)
                if self.input_map.just_pressed("editor_open"):
                    self.open_file_modal(False)
                if self.input_map.just_pressed("editor_help"):
                    self.help_open = True
                if self.input_map.just_pressed("editor_rename"):
                    self.begin_rename_selected()
                if self.input_map.just_pressed("editor_select_all"):
                    self.select_all_entities()
                if self.input_map.just_pressed("build_reload"):
                    var ok_so = self.reload_gameplay_plugin()
                    if ok_so:
                        self.push_toast("Reloaded gameplay .so", 0, 2.0)
                    else:
                        self.push_toast("Reload failed: " + self.hot_reload_host.last_error, 2, 4.0)
                if self.input_map.just_pressed("editor_copy") and self.selected_count() > 0:
                    _ = self.copy_selected()
                if self.input_map.just_pressed("editor_paste") and len(self.clipboard_editor_ids) > 0:
                    _ = self.paste_clipboard(record_undo=True)
                if self.input_map.just_pressed("editor_duplicate") and self.selected_count() > 0:
                    if len(self.selected_entity_ids) > 1:
                        _ = self.duplicate_selected(record_undo=True)
                    elif self.selected_entity_id >= 0:
                        _ = self.duplicate_entity(self.selected_entity_id, record_undo=True)
                if self.input_map.just_pressed("editor_delete") and self.selected_count() > 0:
                    if len(self.selected_entity_ids) > 1:
                        _ = self.despawn_selected(record_undo=True)
                    elif self.selected_entity_id >= 0:
                        _ = self.despawn_entity(self.selected_entity_id, record_undo=True)
                if self.input_map.just_pressed("frame_selection"):
                    self.frame_selection()
                if self.input_map.just_pressed("gizmo_toggle_space"):
                    self.toggle_gizmo_space()
                if self.input_map.just_pressed("pie_toggle"):
                    _ = self.pie_play()
                if self.input_map.just_pressed("pie_pause"):
                    _ = self.pie_pause()
                if self.input_map.just_pressed("pie_step"):
                    _ = self.pie_step(1, Float32(1.0 / 60.0))

        # Arrow keys drive the Hierarchy while the pointer is over it (they orbit the camera elsewhere).
        var hier_hover = False
        if not kbd_captured and not in_play:
            var h_info = self.dock.find_panel("Hierarchy")
            if h_info[0] >= 0 and h_info[2] and h_info[3].contains(_lmx, _lmy):
                hier_hover = True
                _ = self.hierarchy_key_nav(
                    input.key_pressed(UI_KEY_UP),
                    input.key_pressed(UI_KEY_DOWN),
                    input.key_pressed(263),
                    input.key_pressed(262),
                )
        var editor_cam_controllable = (self.pie.mode != PIE_PLAYING) or self.pie_camera_ejected
        if (
            not editor_cam_controllable
            or self.ui_ctx.modal_depth > 0
            or self.ui_ctx.had_modal_prev_frame
            or self.menu.bar.group >= 0
            or not input.mouse_down(1)
        ):
            self._rmb_fly_active = False
        elif not self._rmb_fly_active and input.mouse_pressed(1) and not mouse_captured:
            self._rmb_fly_active = True
        var rmb_fly = self._rmb_fly_active
        if not rmb_fly and not kbd_captured and not in_play:
            if self.input_map.just_pressed("gizmo_translate"):
                self.gizmo_mode = GIZMO_MODE_TRANSLATE
            if self.input_map.just_pressed("gizmo_rotate"):
                self.gizmo_mode = GIZMO_MODE_ROTATE
            if self.input_map.just_pressed("gizmo_scale"):
                self.gizmo_mode = GIZMO_MODE_SCALE
            if self.input_map.just_pressed("tool_select"):
                self.gizmo_mode = GIZMO_MODE_SELECT
            if self.input_map.just_pressed("toggle_snap"):
                self.snap_enabled = not self.snap_enabled
                self.push_toast("Grid snap on" if self.snap_enabled else "Grid snap off", 0, 1.2)

        if editor_cam_controllable:
            var orb = self.input_map.axis_2d("cam_orbit") if not (kbd_captured or hier_hover) else (Float32(0.0), Float32(0.0))
            var dyaw = orb[0] * Float32(80.0) * dt
            var dpitch = orb[1] * Float32(65.0) * dt
            var fly_move = Vec3()
            if rmb_fly:
                self.camera_rig.mode = CAM_MODE_FREE_FLY
                self.camera_rig.fly_pos = self.cam_pos
                dyaw += input.mouse_delta_x() * Float32(0.18)
                dpitch += input.mouse_delta_y() * Float32(0.18)
                if not kbd_captured:
                    if input.key_down(87): fly_move.z += Float32(1.0)
                    if input.key_down(83): fly_move.z -= Float32(1.0)
                    if input.key_down(68): fly_move.x += Float32(1.0)
                    if input.key_down(65): fly_move.x -= Float32(1.0)
                    if input.key_down(69): fly_move.y += Float32(1.0)
                    if input.key_down(81): fly_move.y -= Float32(1.0)
                self.update_camera(dt, dyaw, dpitch, fly_move_local=fly_move, fly_speed=Float32(8.5))
                self.cam_target = self.cam_pos + self.cam_fwd * self.camera_rig.spring_arm.target_length
                self.camera_rig.pivot_spring.value = self.cam_target
            else:
                self.camera_rig.mode = CAM_MODE_ORBIT_FOLLOW
                # Mouse navigation over the viewport: wheel dolly, middle-drag pan, Alt+left orbit,
                # Alt+right dolly (the DCC conventions; right-drag stays fly mode).
                if self.viewport_hovered and not mouse_captured:
                    var arm = max(Float32(0.35), self.camera_rig.spring_arm.target_length)
                    var wheel = input.mouse_wheel()
                    var dolly = Float32(1.0)
                    if wheel != Float32(0.0) and isfinite(wheel):
                        dolly = Float32(0.88) if wheel > Float32(0.0) else Float32(1.0 / 0.88)
                        var steps = min(Float32(10.0), abs(wheel))
                        var k = Float32(1.0)
                        for _ in range(Int(steps + Float32(0.5))):
                            k *= dolly
                        dolly = k
                    if alt_down and input.mouse_down(1):
                        dolly *= max(Float32(0.5), min(Float32(2.0), Float32(1.0) + input.mouse_delta_y() * Float32(0.01)))
                    if dolly != Float32(1.0):
                        var new_arm = max(Float32(0.35), min(Float32(800.0), arm * dolly))
                        self.camera_rig.spring_arm.target_length = new_arm
                        self.camera_rig.spring_arm.current_length.value = new_arm
                        self.agent_bus.notify_user_camera_input()
                    if input.mouse_down(2):
                        var k_pan = arm * Float32(0.0016) / max(Float32(0.5), self.effective_ui_scale())
                        var pan = (self.cam_right * (-input.mouse_delta_x()) + self.cam_up * input.mouse_delta_y()) * k_pan
                        if isfinite(pan.x) and isfinite(pan.y) and isfinite(pan.z):
                            self.cam_target = self.cam_target + pan
                            self.camera_rig.pivot_spring.value = self.cam_target
                            self.camera_rig.pivot_spring.velocity = Vec3()
                            self.agent_bus.notify_user_camera_input()
                    if alt_down and input.mouse_down(0):
                        dyaw += input.mouse_delta_x() * Float32(0.25)
                        dpitch += input.mouse_delta_y() * Float32(0.25)
                self.update_camera(dt, dyaw, dpitch)
        return True

    # ---- Authoritative Binary Serialization, `.flscn`/`.flscnb` Persistence & `fnv64` Hash ----

    def _collect_entity_runtime_links(self) -> List[Tuple[Int, Int, Int]]:
        var links = List[Tuple[Int, Int, Int]]()
        for i in range(len(self.app.world.entities)):
            if not self.app.world.entities[i].alive:
                continue
            var e = Entity(UInt32(i), self.app.world.entities[i].generation)
            var bid = -1
            var tid = -1
            var rb_opt = self.app.world.get_ptr[RigidBodyComponent](e)
            if rb_opt:
                bid = rb_opt.value()[unsafe_offset=0].body_id
            var col_opt = self.app.world.get_ptr[ColliderComponent](e)
            if col_opt:
                tid = col_opt.value()[unsafe_offset=0].trigger_id
            if bid >= 0 or tid >= 0:
                links.append((i, bid, tid))
        return links^

    def _restore_entity_runtime_links(mut self, links: List[Tuple[Int, Int, Int]]):
        var sync_opt = self.app.resources.get_ptr_mut[PhysicsSyncStateResource]()
        if sync_opt:
            sync_opt.value()[unsafe_offset=0] = PhysicsSyncStateResource()
        for li in range(len(links)):
            var eid = links[li][0]
            var bid = links[li][1]
            var tid = links[li][2]
            var e = self.resolve_entity(eid)
            if not e.valid():
                continue
            if bid >= 0:
                var rb_opt = self.app.world.get_ptr_mut[RigidBodyComponent](e)
                if rb_opt:
                    rb_opt.value()[unsafe_offset=0].body_id = bid
                    if sync_opt:
                        var lt_opt = self.app.world.get_ptr[LocalTransform](e)
                        var p = lt_opt.value()[unsafe_offset=0].translation if lt_opt else Vec3()
                        var r = lt_opt.value()[unsafe_offset=0].rotation if lt_opt else Quat()
                        sync_opt.value()[unsafe_offset=0].record_body(e, bid, p, r)
                        sync_opt.value()[unsafe_offset=0].record_body_collider(e, self.app.world.get_ptr[ColliderComponent](e))
            if tid >= 0:
                var col_opt = self.app.world.get_ptr_mut[ColliderComponent](e)
                if col_opt:
                    col_opt.value()[unsafe_offset=0].trigger_id = tid
                    if sync_opt:
                        sync_opt.value()[unsafe_offset=0].record_trigger(e, tid)

    def serialize(self, mut out: BlobWriter):
        out.begin(STUDIO_CHECKPOINT_FOURCC, 1)
        out.integer(self.selected_entity_id)
        out.u32(UInt32(len(self.selected_entity_ids)))
        for i in range(len(self.selected_entity_ids)):
            out.integer(self.selected_entity_ids[i])
        out.integer(self.selection_anchor_id)
        out.integer(self.sim_tick)
        out.vec3(self.gravity)
        out.f32(self.ground_plane_y)
        out.boolean(self.has_ground_plane)
        out.integer(self._ground_body_id)
        out.f32(self.stepped_clock.raw_time_sec)
        out.f32(self.stepped_clock.stepped_time_sec)
        out.f32(self.stepped_clock.step_hz)
        out.f32(self.stepped_clock.hitstop_remaining_sec)
        out.boolean(self.stepped_clock.pose_changed_this_tick)
        out.f32(self.preview_impact_blend)
        out.boolean(self.preview_impact_invert)
        out.boolean(self.preview_shockwave_active)
        out.f32(self.preview_shockwave_radius)

        var wb = BlobWriter()
        try:
            self.app.world.snapshot_blob(wb)
        except:
            pass
        _write_blob_bytes(out, wb.data)

        var pb = BlobWriter()
        var phys_opt = self.app.resources.get_ptr[PhysicsWorld]()
        if phys_opt:
            phys_opt.value()[unsafe_offset=0].serialize(pb)
        _write_blob_bytes(out, pb.data)

        var links = self._collect_entity_runtime_links()
        out.u32(UInt32(len(links)))
        for i in range(len(links)):
            out.integer(links[i][0])
            out.integer(links[i][1])
            out.integer(links[i][2])
        out.end()

    def deserialize(mut self, mut input: BlobReader) raises:
        # Read and validate the whole checkpoint into locals first; editor state is only assigned once
        # the blob is known to be complete (a truncated blob used to leave selection/clock/gravity from
        # the checkpoint paired with the old world).
        _ = input.enter(STUDIO_CHECKPOINT_FOURCC, 1)
        var sel_primary = input.integer()
        var n_sel = Int(input.u32())
        var rem_sel = max(0, input.limit() - input.offset)
        if input.failed or n_sel < 0 or n_sel > rem_sel // 8:
            input.failed = True
            input.leave()
            raise Error("Failed to read Studio checkpoint blob")
        var sel_ids = List[Int](capacity=n_sel)
        for _ in range(n_sel):
            sel_ids.append(input.integer())
        var sel_anchor = input.integer()
        var r_sim_tick = input.integer()
        var r_gravity = input.vec3()
        var r_ground_y = input.f32()
        var r_has_ground = input.boolean()
        var r_ground_body = input.integer()
        var r_raw_time = input.f32()
        var r_stepped_time = input.f32()
        var r_step_hz = input.f32()
        var r_hitstop = input.f32()
        var r_pose_changed = input.boolean()
        var r_impact_blend = input.f32()
        var r_impact_invert = input.boolean()
        var r_shock_active = input.boolean()
        var r_shock_radius = input.f32()

        var w_bytes = _read_blob_bytes(input)
        var p_bytes = _read_blob_bytes(input)
        var n_links = Int(input.u32())
        var rem_links = max(0, input.limit() - input.offset)
        if input.failed or n_links < 0 or n_links > rem_links // 24:
            input.failed = True
            input.leave()
            raise Error("Failed to read Studio checkpoint blob")
        var links = List[Tuple[Int, Int, Int]](capacity=n_links)
        for _ in range(n_links):
            var eid = input.integer()
            var bid = input.integer()
            var tid = input.integer()
            links.append((eid, bid, tid))
        input.leave()
        if input.failed:
            raise Error("Failed to read Studio checkpoint blob")

        self.selected_entity_id = sel_primary
        self.selected_entity_ids = sel_ids^
        self.selection_anchor_id = sel_anchor
        self.sim_tick = r_sim_tick
        self.gravity = r_gravity
        self.ground_plane_y = r_ground_y
        self.has_ground_plane = r_has_ground
        self._ground_body_id = r_ground_body
        self.stepped_clock.raw_time_sec = r_raw_time
        self.stepped_clock.stepped_time_sec = r_stepped_time
        self.stepped_clock.step_hz = r_step_hz
        self.stepped_clock.hitstop_remaining_sec = r_hitstop
        self.stepped_clock.pose_changed_this_tick = r_pose_changed
        self.preview_impact_blend = r_impact_blend
        self.preview_impact_invert = r_impact_invert
        self.preview_shockwave_active = r_shock_active
        self.preview_shockwave_radius = r_shock_radius

        if len(w_bytes) > 0:
            var wr = BlobReader(w_bytes^)
            self.app.world.restore_blob(wr)
            self.registry.ensure_registered_in_world(self.app.world)

        if len(p_bytes) > 0:
            var phys_opt = self.app.resources.get_ptr_mut[PhysicsWorld]()
            if phys_opt:
                var pr = BlobReader(p_bytes^)
                phys_opt.value()[unsafe_offset=0].deserialize(pr)
                phys_opt.value()[unsafe_offset=0].gravity = -self.gravity.y
            self._restore_entity_runtime_links(links)
            self._propagate_transforms_immediate()
            self._sync_compat_view()
        else:
            self._reset_physics_world_and_sync()
        self._prune_selection()

    def state_hash(self) -> UInt64:
        """Computes bit-exact `fnv64` digest over canonical authored ECS/Scene + Studio sim state
        (strictly invariant to ephemeral `PhysicsWorld` runtime IDs)."""
        var w = BlobWriter()
        w.begin(fourcc("SHSH"), 1)
        w.integer(self.sim_tick)
        w.vec3(self.gravity)
        w.f32(self.ground_plane_y)
        w.boolean(self.has_ground_plane)
        w.f32(self.stepped_clock.raw_time_sec)
        w.f32(self.stepped_clock.stepped_time_sec)
        w.f32(self.stepped_clock.step_hz)
        w.f32(self.stepped_clock.hitstop_remaining_sec)
        w.boolean(self.stepped_clock.pose_changed_this_tick)
        w.f32(self.preview_impact_blend)
        w.boolean(self.preview_impact_invert)
        w.boolean(self.preview_shockwave_active)
        w.f32(self.preview_shockwave_radius)

        # Canonical sorted entities (excluding ephemeral runtime body_id/trigger_id/runtime_index)
        var ents = List[Entity]()
        var ent_to_slot = List[Int](length=max(1, len(self.app.world.entities)), fill=-1)
        for idx in range(len(self.app.world.entities)):
            if self.app.world.entities[idx].alive:
                ent_to_slot[idx] = len(ents)
                ents.append(Entity(UInt32(idx), self.app.world.entities[idx].generation))

        w.integer(len(ents))
        var sorted_codecs = self.registry.sorted_entry_indices()
        var rb_hash = RigidBodyComponent.COMPONENT_ID
        var col_hash = ColliderComponent.COMPONENT_ID
        var mr_hash = MeshRendererComponent.COMPONENT_ID
        var mo_hash = MaterialOverrideComponent.COMPONENT_ID
        var sk_hash = SkeletonComponent.COMPONENT_ID
        var ae_hash = AudioEmitterComponent.COMPONENT_ID
        var prev_gt_hash = PrevGlobalTransform.COMPONENT_ID
        var gt_hash = GlobalTransform.COMPONENT_ID
        # Editor identity is metadata, not scene state: ids are assigned lazily (load, restore, sync),
        # and that must not change the hash PIE and replay checks compare.
        var edid_hash = StudioEditorIdComponent.COMPONENT_ID

        for slot in range(len(ents)):
            var e = ents[slot]
            var p_slot = -1
            var p_opt = self.app.world.get_ptr[Parent](e)
            if p_opt:
                var pe = p_opt.value()[unsafe_offset=0].entity
                if self.app.world.is_alive(pe) and Int(pe.index) < len(ent_to_slot):
                    p_slot = ent_to_slot[Int(pe.index)]
            w.integer(p_slot)

            var has_local_tf = self.app.world.has[LocalTransform](e)
            for ci in range(len(sorted_codecs)):
                var cidx = sorted_codecs[ci]
                var chash = self.registry.entries[cidx].component_hash
                if chash == prev_gt_hash or chash == edid_hash or (has_local_tf and chash == gt_hash):
                    continue
                if not self.registry.has_on_entity(cidx, self.app.world, e):
                    continue
                w.u64(chash)
                if chash == rb_hash:
                    var rb_opt = self.app.world.get_ptr[RigidBodyComponent](e)
                    if rb_opt:
                        ref rb = rb_opt.value()[unsafe_offset=0]
                        w.u8(rb.motion_mode)
                        w.f32(rb.mass)
                        w.vec3(rb.linear_vel)
                        w.vec3(rb.angular_vel)
                        w.f32(rb.friction)
                        w.f32(rb.restitution)
                        w.f32(rb.rolling_resistance)
                        w.f32(rb.linear_damping)
                        w.f32(rb.angular_damping)
                        w.f32(rb.gravity_scale)
                        _write_filter_canon(w, rb.filter)
                        w.boolean(rb.can_sleep)
                        w.boolean(rb.enable_ccd)
                        w.integer(rb.material)
                        w.boolean(rb.active)
                elif chash == col_hash:
                    var c_opt = self.app.world.get_ptr[ColliderComponent](e)
                    if c_opt:
                        ref c = c_opt.value()[unsafe_offset=0]
                        w.integer(c.shape)
                        w.f32(c.radius)
                        w.f32(c.half_height)
                        w.vec3(c.half_extents)
                        w.vec3(c.local_offset)
                        w.quat(c.local_rot)
                        w.boolean(c.is_trigger)
                        _write_filter_canon(w, c.filter)
                        w.integer(len(c.points))
                        for pi in range(len(c.points)):
                            w.vec3(c.points[pi])
                elif chash == mr_hash:
                    var mr_opt = self.app.world.get_ptr[MeshRendererComponent](e)
                    if mr_opt:
                        ref mr = mr_opt.value()[unsafe_offset=0]
                        w.begin(MeshRendererComponent.SCHEMA_FOURCC, MeshRendererComponent.SCHEMA_VERSION)
                        w.integer(mr.mesh_id if mr.mesh_ref.is_empty() else -1)
                        w.integer(
                            mr.material_id
                            if (mr.material_ref.is_empty() and mr.mesh_ref.is_empty())
                            else -1
                        )
                        w.boolean(mr.cast_shadows)
                        w.boolean(mr.visible)
                        mr.mesh_ref.write_bin(w)
                        mr.material_ref.write_bin(w)
                        w.end()
                elif chash == mo_hash:
                    var mo_opt = self.app.world.get_ptr[MaterialOverrideComponent](e)
                    if mo_opt:
                        ref mo = mo_opt.value()[unsafe_offset=0]
                        w.begin(MaterialOverrideComponent.SCHEMA_FOURCC, MaterialOverrideComponent.SCHEMA_VERSION)
                        w.integer(mo.material_id if mo.material_ref.is_empty() else -1)
                        w.vec3(mo.base_color)
                        w.f32(mo.roughness)
                        w.f32(mo.metallic)
                        w.vec3(mo.emissive)
                        w.boolean(mo.enabled)
                        w.f32(mo.opacity)
                        w.f32(mo.alpha_cutoff)
                        w.f32(mo.clearcoat)
                        w.f32(mo.clearcoat_roughness)
                        w.f32(mo.sheen)
                        w.f32(mo.subsurface)
                        mo.material_ref.write_bin(w)
                        w.end()
                elif chash == sk_hash:
                    var sk_opt = self.app.world.get_ptr[SkeletonComponent](e)
                    if sk_opt:
                        ref sk = sk_opt.value()[unsafe_offset=0]
                        w.begin(SkeletonComponent.SCHEMA_FOURCC, SkeletonComponent.SCHEMA_VERSION)
                        w.u64(sk.model_id)
                        w.boolean(False)  # dirty_topology is ephemeral runtime state
                        var n_pose = sk.skeleton.pose.count()
                        w.integer(n_pose)
                        for pi in range(n_pose):
                            w.vec3(sk.skeleton.pose.t[pi])
                            w.quat(sk.skeleton.pose.r[pi])
                            w.vec3(sk.skeleton.pose.s[pi])
                        sk.rig_ref.write_bin(w)
                        var par_n = min(n_pose, len(sk.skeleton.parents))
                        for pi in range(n_pose):
                            w.integer(sk.skeleton.parents[pi] if pi < par_n else -1)
                        w.end()
                elif chash == ae_hash:
                    var ae_opt = self.app.world.get_ptr[AudioEmitterComponent](e)
                    if ae_opt:
                        ref ae = ae_opt.value()[unsafe_offset=0]
                        w.begin(AudioEmitterComponent.SCHEMA_FOURCC, AudioEmitterComponent.SCHEMA_VERSION)
                        w.integer(ae.asset_id if ae.audio_ref.is_empty() else -1)
                        w.text(ae.asset_path)
                        w.f32(ae.volume)
                        w.f32(ae.pitch)
                        w.integer(ae.bus_id)
                        w.boolean(ae.looping)
                        w.boolean(ae.spatial)
                        w.integer(ae.priority)
                        w.f32(ae.ref_dist)
                        w.f32(ae.max_dist)
                        w.boolean(ae.play_on_spawn)
                        w.boolean(ae.playing)
                        w.boolean(ae.auto_play_pending)
                        w.boolean(ae.enable_occlusion)
                        w.f32(ae.occlusion_gain)
                        w.f32(ae.cutoff_hz)
                        w.f32(ae.obstruction)
                        w.vec3(ae.prev_pos)
                        w.boolean(ae.has_prev_pos)
                        ae.audio_ref.write_bin(w)
                        w.end()
                else:
                    _ = self.registry.write_entity_bin_slots(cidx, self.app.world, e, ent_to_slot, w)
            var unk_opt = self.app.world.get_ptr[UnknownComponentsComponent](e)
            if unk_opt:
                w.u64(UnknownComponentsComponent.COMPONENT_ID)
                unk_opt.value()[unsafe_offset=0].write_bin(w)
        w.end()
        return fnv64(w.data)

    def _resolve_scene_io_path(self, path: String) -> String:
        var target = path if path.byte_length() > 0 else self.scene_file_path
        if (
            path.byte_length() > 0
            and not path.startswith("/")
            and self.project_root.byte_length() > 0
            and not path.startswith(self.project_root)
        ):
            target = self.project_root + "/" + path
        return target

    def recent_scenes_path(self) -> String:
        var root = self.project_root if self.project_root.byte_length() > 0 else String(".")
        return root + "/.flint/recent_scenes.txt"

    def load_recent_scenes(mut self):
        """Reads `<project>/.flint/recent_scenes.txt` (one path per line, newest first)."""
        self.recent_scenes.clear()
        var raw = read_bytes(self.recent_scenes_path())
        var line = List[UInt8]()
        for i in range(len(raw) + 1):
            if i == len(raw) or raw[i] == 10:
                if len(line) > 0 and len(self.recent_scenes) < STUDIO_RECENT_SCENES_CAP:
                    self.recent_scenes.append(String(unsafe_from_utf8=line))
                line.clear()
            elif raw[i] != 13:
                line.append(raw[i])

    def note_recent_scene(mut self, path: String):
        """Moves `path` to the front of the recent-scenes list and writes the list back."""
        if path.byte_length() == 0:
            return
        var out_list = List[String]()
        out_list.append(path)
        for i in range(len(self.recent_scenes)):
            if self.recent_scenes[i] != path and len(out_list) < STUDIO_RECENT_SCENES_CAP:
                out_list.append(self.recent_scenes[i])
        self.recent_scenes = out_list^
        if self.project_root.byte_length() == 0:
            return  # no project: keep the list in memory, never write into whatever the cwd is
        var buf = List[UInt8]()
        for i in range(len(self.recent_scenes)):
            for b in self.recent_scenes[i].as_bytes():
                buf.append(b)
            buf.append(10)
        var dst = self.recent_scenes_path()
        _ = make_dirs(parent_directory(dst))
        _ = write_file_durable(dst, buf)

    def open_file_modal(mut self, save_as: Bool):
        self.file_modal_open = True
        self.file_modal_save_as = save_as
        self.file_modal_path = self.scene_file_path
        self.load_recent_scenes()

    def submit_file_modal(mut self) -> Bool:
        """Runs the Open / Save As the file dialog was opened for; keeps the dialog open on failure."""
        var path = self.file_modal_path
        if path.byte_length() == 0:
            self.push_toast("Enter a scene path", 1, 2.5)
            return False
        if self.file_modal_save_as:
            if not (path.endswith(".flscn") or path.endswith(".flscnb") or path.endswith(".flscene")):
                path += ".flscnb"
            var ok = self.save_scene_file(path)
            if ok:
                self.file_modal_open = False
                self.push_toast("Saved " + self.scene_file_path, 0, 2.5)
            return ok
        self.file_modal_open = False
        var ok = self.request_load_scene(path)
        if not ok and not self.confirm_modal_open:
            self.file_modal_open = True  # the load failed: let the user fix the path
        return ok

    def reset_layout(mut self):
        self.workspace_mgr.reset_preset(WORKSPACE_EDIT)
        _ = self.apply_workspace_preset(WORKSPACE_EDIT)
        self.push_toast("Layout reset", 0, 1.8)

    def select_all_entities(mut self):
        self.selected_entity_ids.clear()
        for i in range(len(self.entities)):
            if self.entities[i].alive and not self.is_entity_locked(i):
                self.selected_entity_ids.append(i)
        if len(self.selected_entity_ids) > 0:
            self.selected_entity_id = self.selected_entity_ids[0]
            self.selection_anchor_id = self.selected_entity_id

    def welcome_marker_path(self) -> String:
        var root = self.project_root if self.project_root.byte_length() > 0 else String(".")
        return root + "/.flint/welcome_seen"

    def maybe_show_welcome(mut self):
        """First run in a project shows the Welcome dialog once (interactive sessions only)."""
        if file_size(self.welcome_marker_path()) < 0:
            self.welcome_open = True

    def dismiss_welcome(mut self):
        self.welcome_open = False
        var dst = self.welcome_marker_path()
        _ = make_dirs(parent_directory(dst))
        var one = List[UInt8]()
        one.append(49)
        _ = write_bytes(dst, one)

    def begin_rename_selected(mut self):
        if self.selected_entity_id < 0:
            return
        _ = WorkspacePresetManager.ensure_panel_visible(self.dock, PANEL_INSPECTOR)
        self.rename_focus_requested = True

    def autosave_source_path(self) -> String:
        """The full path the autosave belongs to: the resolved scene path, or "" while untitled."""
        if self.scene_file_path.byte_length() == 0:
            return String("")
        return self._resolve_scene_io_path(self.scene_file_path)

    def autosave_path(self) -> String:
        """`<project>/.flint/autosave/<scene>-<hash>.autosave.flscnb`: next to the project, never the
        user's file. The hash is of the full scene path, so `a/level.flscn` and `b/level.flscn` never
        share (and overwrite) one autosave."""
        var root = self.project_root if self.project_root.byte_length() > 0 else String(".")
        var name = String("untitled")
        if self.scene_file_path.byte_length() > 0:
            var b = self.scene_file_path.as_bytes()
            var start = 0
            var stop = len(b)
            for i in range(len(b)):
                if b[i] == 47:  # '/'
                    start = i + 1
            for i in range(len(b) - 1, start - 1, -1):
                if b[i] == 46:  # '.'
                    stop = i
                    break
            if stop > start:
                name = String(unsafe_from_utf8=b[start:stop])
        # Never stack suffixes (`level.autosave.autosave`) if a path ever points at an autosave.
        while name.endswith(".autosave"):
            var trimmed = String(name[byte=0 : name.byte_length() - 9])
            name = trimmed^
        if name.byte_length() == 0:
            name = String("untitled")
        var src = self.autosave_source_path()
        var key = List[UInt8](capacity=src.byte_length())
        for c in src.as_bytes():
            key.append(c)
        var h = fnv64(key)
        var digits = String("0123456789abcdef").as_bytes()
        var hex = List[UInt8](capacity=16)
        for k in range(16):
            hex.append(digits[Int((h >> UInt64(60 - 4 * k)) & UInt64(15))])
        return root + "/.flint/autosave/" + name + "-" + String(unsafe_from_utf8=hex) + ".autosave.flscnb"

    @staticmethod
    def _wrap_autosave(source: String, payload: List[UInt8]) -> List[UInt8]:
        """Autosave file: `FLAS`, u32 LE length + the scene path it belongs to, then the `.flscnb` bytes."""
        var sb = source.as_bytes()
        var out = List[UInt8](capacity=8 + len(sb) + len(payload))
        for c in String("FLAS").as_bytes():
            out.append(c)
        var n = UInt32(len(sb))
        for k in range(4):
            out.append(UInt8((n >> UInt32(8 * k)) & UInt32(255)))
        for c in sb:
            out.append(c)
        for k in range(len(payload)):
            out.append(payload[k])
        return out^

    @staticmethod
    def _unwrap_autosave(raw: List[UInt8]) -> Tuple[Bool, String, List[UInt8]]:
        """(valid, source scene path, `.flscnb` bytes) of an autosave file."""
        if len(raw) < 8 or raw[0] != 70 or raw[1] != 76 or raw[2] != 65 or raw[3] != 83:  # "FLAS"
            return (False, String(""), List[UInt8]())
        var n = Int(raw[4]) | (Int(raw[5]) << 8) | (Int(raw[6]) << 16) | (Int(raw[7]) << 24)
        if n < 0 or 8 + n > len(raw):
            return (False, String(""), List[UInt8]())
        var src = String(unsafe_from_utf8=raw[8 : 8 + n])
        var payload = List[UInt8](capacity=len(raw) - 8 - n)
        for k in range(8 + n, len(raw)):
            payload.append(raw[k])
        return (True, src^, payload^)

    def autosave_available(self) -> Bool:
        """True when an autosave of *this* scene path, newer than the scene file, exists (offered for
        recovery at startup and after a load)."""
        var p = self.autosave_path()
        var a = file_stamp(p)
        if a[0] <= 0:
            return False
        var un = Self._unwrap_autosave(read_bytes(p))
        if not un[0] or un[1] != self.autosave_source_path():
            return False
        if self.scene_file_path.byte_length() == 0:
            return True
        var f = file_stamp(self.autosave_source_path())
        return f[0] < 0 or a[1] > f[1]

    def discard_autosave(mut self):
        var p = self.autosave_path()
        if file_size(p) > 0:
            _ = remove_file(p)

    def write_autosave(mut self) -> Bool:
        """Writes the current world to `autosave_path()` without touching `scene_file_path` or the dirty
        flag, so a crash loses at most one autosave interval of work. The write is durable (tmp + fsync
        + rename) and the file records which scene path it belongs to."""
        var target = self.autosave_path()
        var parent = parent_directory(target)
        if parent.byte_length() > 0 and parent != ".":
            _ = make_dirs(parent)
        var payload = self.scene.save_flscnb_bytes(self.app.world, self.registry)
        if len(payload) == 0:
            return False
        var wrapped = Self._wrap_autosave(self.autosave_source_path(), payload)
        if not write_file_durable(target, wrapped):
            return False
        self.last_autosave_path = target
        return True

    def tick_autosave(mut self, dt: Float32) -> Bool:
        """Call once per frame. Autosaves after `autosave_interval_sec` of dirty, stopped (not Play) time.
        Play pauses the timer (the world is the Play world then); it does not reset it."""
        self.flush_stale_flmat_edit()
        if self.autosave_interval_sec <= Float32(0.0) or not self.dirty:
            self._autosave_accum_sec = Float32(0.0)
            return False
        if self.pie.mode != PIE_STOPPED or self.agent_bus.time_travel_active:
            return False  # Play / time travel show a world that is not the user's scene
        if isfinite(dt) and dt > Float32(0.0):
            self._autosave_accum_sec += min(dt, Float32(1.0))
        if self._autosave_accum_sec < self.autosave_interval_sec:
            return False
        self._autosave_accum_sec = Float32(0.0)
        return self.write_autosave()

    def recover_autosave(mut self) -> Bool:
        """Brings back the autosave as one undoable step. The scene keeps its own file path and stays
        dirty, so the recovered work is saved to the real scene file (and quitting still prompts)."""
        var p = self.autosave_path()
        if file_size(p) <= 0:
            self.push_toast("No autosave found", 1, 3.0)
            return False
        return self.load_scene_file(p, recovery=True)

    def _untitled_save_suggestion(self) -> String:
        """A Save As default for an untitled scene that names no existing file."""
        var root = self.project_root if self.project_root.byte_length() > 0 else String(".")
        var base = root + "/scenes/untitled"
        var cand = base + ".flscnb"
        var n = 2
        while file_size(cand) >= 0 and n < 1000:
            cand = base + "-" + String(n) + ".flscnb"
            n += 1
        return cand

    def save_scene_file(mut self, path: String = "") -> Bool:
        """Saves the Studio scene via `Scene.save_flscnb_file` (`.flscnb`/`.flscene`) or `Scene.save_flscn_file` (`.flscn`)."""
        _ = self.commit_pending_flmat_edit()
        if self.agent_bus.time_travel_active:
            # The world on screen is an agent checkpoint being scrubbed, not the user's scene.
            self.push_toast("Exit time travel before saving", 1, 3.0)
            return False
        if path.byte_length() == 0 and self.scene_file_path.byte_length() == 0:
            # Untitled (File > New): the first save asks where, never overwriting another level.
            self.open_file_modal(True)
            self.file_modal_path = self._untitled_save_suggestion()
            self.push_toast("Choose where to save the new scene", 0, 3.0)
            return False
        var target = self._resolve_scene_io_path(path)
        if target.byte_length() == 0:
            self.push_toast("Save failed: empty path", 2, 3.0)
            return False
        var parent = parent_directory(target)
        if parent.byte_length() > 0 and parent != ".":
            _ = make_dirs(parent)
        var ok: Bool
        if target.endswith(".flscn"):
            try:
                ok = self.scene.save_flscn_file(target, self.app.world, self.registry)
            except:
                ok = False
        else:
            ok = self.scene.save_flscnb_file(target, self.app.world, self.registry)
        if not ok:
            self.push_toast("Failed to save scene", 2, 3.5)
            return False  # the old file and any autosave are left untouched
        if self.scene_file_path.byte_length() == 0:
            self.discard_autosave()  # an untitled scene's autosave is superseded by its first real file
        self.scene_file_path = target
        self.note_recent_scene(target)
        self.mark_dirty(False)
        self.discard_autosave()  # the real file is now newer than any autosave
        self.push_toast("Saved " + target, 0, 2.5)
        return True

    @staticmethod
    def _is_valid_profile_name(name: String) -> Bool:
        if name.byte_length() == 0:
            return False
        for b in name.as_bytes():
            var ok = (
                (b >= 97 and b <= 122)
                or (b >= 65 and b <= 90)
                or (b >= 48 and b <= 57)
                or b == 95
                or b == 45
            )
            if not ok:
                return False
        return True

    @staticmethod
    def _package_log_tail(log_path: String) -> String:
        var raw = read_bytes(log_path)
        if len(raw) == 0:
            return String("")
        var start = max(0, len(raw) - 512)
        var out = List[UInt8]()
        for i in range(start, len(raw)):
            var b = raw[i]
            if b == 10 or b == 13:
                if len(out) > 0 and out[len(out) - 1] != 32:
                    out.append(32)
            elif b >= 32 and b < 127:
                out.append(b)
        return String(unsafe_from_utf8=out)

    def package_project(
        mut self,
        output_dir: String = "",
        profile: String = "",
        loose_assets: Bool = False,
        target: String = "",
    ) -> Tuple[Bool, String]:
        """Packages the open project into a standalone bundle via `tools/flint.sh package`."""
        var proj = self.project_root if self.project_root.byte_length() > 0 else env_value("FLINT_STUDIO_PROJECT")
        while proj.byte_length() > 1 and proj.endswith("/"):
            var trimmed = String(proj[byte=0 : proj.byte_length() - 1])
            proj = trimmed^
        if proj.byte_length() == 0 or not is_directory(proj, follow_symlinks=True):
            return (False, "no valid project_root is open")
        if target.byte_length() == 0 and file_size(proj + "/flint.toml") < 0:
            return (False, "project flint.toml not found in " + proj)
        if profile.byte_length() > 0 and target.byte_length() > 0:
            return (False, "cannot specify both profile and target")
        if profile.byte_length() > 0 and not Self._is_valid_profile_name(profile):
            return (False, "invalid profile name: " + profile)
        if target.byte_length() > 0:
            if target.startswith("-") or target.find("..") >= 0:
                return (False, "invalid target: " + target)
            for b in target.as_bytes():
                if b < 32 or b == 92 or b == 127:
                    return (False, "invalid target bytes")

        var launcher = _resolve_flint_launcher_path()
        if launcher.byte_length() == 0:
            return (False, "could not locate tools/flint.sh launcher")

        var abs_out: String
        if output_dir.byte_length() > 0:
            if output_dir.startswith("/"):
                abs_out = output_dir
            else:
                abs_out = (proj + "/" + output_dir) if proj != "/" else ("/" + output_dir)
        elif profile.byte_length() > 0:
            abs_out = proj + "/build/packages/" + profile
        else:
            abs_out = proj + "/dist/package"

        if is_symlink(abs_out) or (file_size(abs_out) >= 0 and not is_directory(abs_out, follow_symlinks=False)):
            return (False, "output_dir exists and is not a directory: " + abs_out)

        var build_dir = proj + "/build"
        _ = make_dirs(build_dir)
        var log_path = build_dir + "/package.log"

        var args = List[String]()
        args.append("/bin/bash")
        args.append(launcher)
        args.append("package")
        args.append("--out")
        args.append(abs_out)
        if profile.byte_length() > 0:
            args.append("--profile")
            args.append(profile)
        if loose_assets:
            args.append("--loose-assets")
        if target.byte_length() > 0:
            args.append("--target")
            args.append(target)

        var rc = run_process_redirected(args, cwd=proj, log_path=log_path)
        if rc != Int32(0):
            var tail = Self._package_log_tail(log_path)
            var msg = "package failed (exit " + String(Int(rc)) + ")"
            if tail.byte_length() > 0:
                msg += ": " + tail
            return (False, msg)
        if file_size(abs_out + "/run.sh") <= 0:
            return (False, "package finished without executable run.sh at " + abs_out)
        return (True, abs_out)

    def _has_default_showcase_entities(self) -> Bool:
        for i in range(len(self.app.world.entities)):
            if not self.app.world.entities[i].alive:
                continue
            var e = Entity(UInt32(i), self.app.world.entities[i].generation)
            var mr_opt = self.app.world.get_ptr[MeshRendererComponent](e)
            if not mr_opt or not mr_opt.value()[unsafe_offset=0].mesh_ref.is_empty():
                continue
            var meta_opt = self.app.world.get_ptr[StudioEntityMetaComponent](e)
            if not meta_opt:
                continue
            var nm = meta_opt.value()[unsafe_offset=0].name
            if (
                nm == "Studio_ReferenceGrid"
                or nm == "SubD_MaterialOrb"
                or nm == "SubD_CatmullTorus"
                or nm == "ArchKit_Colonnade"
                or nm == "Botanical_FoliageSample"
                or nm == "Botanical_FoliageCanopy"
            ):
                return True
        return False

    @staticmethod
    def _raw_starts_with_json_object(raw: List[UInt8]) -> Bool:
        for i in range(len(raw)):
            var b = raw[i]
            if b == UInt8(32) or b == UInt8(9) or b == UInt8(10) or b == UInt8(13):
                continue
            return b == UInt8(123)  # '{'
        return False

    def load_scene_file(mut self, path: String = "", recovery: Bool = False) -> Bool:
        """Loads a `.flscn`, `.flscnb`, or `.flscene` file into `app.world` and syncs transforms/physics.
        `recovery` (autosave restore) keeps the current scene path and leaves the scene dirty."""
        _ = self.commit_pending_flmat_edit()  # before the entity it belongs to is replaced
        var target = self._resolve_scene_io_path(path)
        if target.byte_length() == 0:
            self.push_toast("Load failed: empty path", 2, 3.0)
            return False
        var raw = read_bytes(target)
        if len(raw) == 0:
            self.push_toast("Load failed: " + target, 2, 3.5)
            return False
        if recovery:
            # An autosave carries the scene path it was taken from: never restore another scene's work.
            var un = Self._unwrap_autosave(raw)
            if not un[0] or un[1] != self.autosave_source_path():
                self.push_toast("Autosave does not belong to this scene", 2, 3.5)
                return False
            raw = un[2].copy()
        # Time travel shows an agent checkpoint; the load (and its undo entry) starts from the live scene.
        self.exit_time_travel()
        var is_json = target.endswith(".flscn") or Self._raw_starts_with_json_object(raw)
        var pie_world_unknown = False
        if self.pie.mode != PIE_STOPPED:
            # A file that does not parse must leave Play running untouched: test-parse it into a
            # scratch world before stopping anything.
            var probe = World()
            self.registry.ensure_registered_in_world(probe)
            try:
                if is_json:
                    _ = self.scene.load_flscn_bytes(probe, self.registry, raw, clear_existing=True)
                else:
                    _ = self.scene.load_flscnb_bytes(probe, self.registry, raw, clear_existing=True)
            except:
                self.push_toast("Failed to parse " + target, 2, 3.5)
                return False
            # Back to the edit world first, so the load (and its undo entry) starts from the scene the
            # user was editing, not from a mid-simulation Play world.
            var restored = True
            try:
                _ = self.pie_stop()
            except:
                # Either a restore-hash mismatch (the edit world is back) or the checkpoint failed to
                # deserialize (mode is still Play, the world may be the Play world).
                restored = self.pie.mode == PIE_STOPPED
            if self.pie.mode != PIE_STOPPED:
                self._abort_pie_session()
            pie_world_unknown = not restored
        # Scene load is an undoable command: the world before the load is kept as its undo state,
        # instead of the old behaviour of wiping the user's whole undo/redo history.
        var before_blob = self._capture_world_blob()
        if len(before_blob) == 0 and not pie_world_unknown:
            # Without a snapshot a failed parse could not put the scene back, and the load could not be undone.
            self.push_toast("Load refused: the current scene could not be snapshotted", 2, 4.0)
            return False
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id
        try:
            if is_json:
                _ = self.scene.load_flscn_bytes(self.app.world, self.registry, raw, clear_existing=True)
            else:
                _ = self.scene.load_flscnb_bytes(self.app.world, self.registry, raw, clear_existing=True)
        except:
            # `clear_existing` may have emptied the world before the parse failed: put the scene back.
            if len(before_blob) > 0:
                _ = self._restore_world_blob(before_blob)
            self._sync_ecs_and_physics()
            self.push_toast("Failed to parse " + target, 2, 3.5)
            return False
        self.clear_selection()
        self.clipboard_editor_ids.clear()  # copied ids belonged to the old scene
        if len(self.app.world.entities) > 0 and self.app.world.entities[0].alive:
            self.select_entity(0)
        var prev_scene_path = self.scene_file_path
        if not recovery:
            self.scene_file_path = target
            self.note_recent_scene(target)
        # Approvals were queued against the old scene; its editor ids mean other entities now.
        self.reject_all_mcp_approvals("Scene changed before approval")
        self.gpu_materials.clear_entity_slots()
        self._pie_frame_updated = False
        self._int_edit_wid = UInt64(0)
        self._int_edit_entity_id = -1
        self._int_edit_buf = ""
        self.preview_anim_time = Float32(0.0)
        self._last_preview_anim_time = Float32(0.0)
        self._clear_render_and_resolver_scene_caches()
        for ei in range(len(self.app.world.entities)):
            if not self.app.world.entities[ei].alive:
                continue
            var e = Entity(UInt32(ei), self.app.world.entities[ei].generation)
            var mr_opt = self.app.world.get_ptr_mut[MeshRendererComponent](e)
            if mr_opt:
                ref mr = mr_opt.value()[unsafe_offset=0]
                if (not mr.mesh_ref.is_empty()) or (not mr.material_ref.is_empty()):
                    mr.material_id = -1
        if self._has_default_showcase_entities():
            upload_default_studio_gpu_assets(self)
        self._reset_physics_world_and_sync()
        if pie_world_unknown:
            # The pre-load world is a broken Play restore, not the user's scene: do not offer it as undo.
            self.push_toast("Play could not be restored; this load cannot be undone", 1, 4.0)
        elif len(before_blob) > 0:
            var top_before = self._top_tx_id()
            var prev_saved = self._saved_tx_id
            self._record_world_tx(
                TX_BATCH_EDIT, -1, List[Int](), before_blob^, prev_ids, prev_primary, prev_anchor,
                ("Recover autosave" if recovery else "Load " + target),
            )
            if self._top_tx_id() != top_before:
                ref top = self.undo_stack[len(self.undo_stack) - 1]
                top.changes_scene_path = True
                top.prev_scene_path = prev_scene_path
                top.next_scene_path = self.scene_file_path
                top.prev_saved_tx_id = prev_saved
                top.load_leaves_clean = not recovery
        if recovery:
            self.mark_dirty(True)
            self.push_toast("Recovered autosave. Save to keep it in " + self.scene_file_path, 0, 4.0)
        else:
            self.mark_dirty(False)
            self.push_toast("Loaded " + target, 0, 2.5)
            if self.autosave_available():
                self.push_toast("This scene has a newer autosave: File > Recover Autosave brings it back", 1, 6.0)
        return True

    @staticmethod
    def _is_asset_browser_ext(name: String) -> Bool:
        return (
            name.endswith(".flmat")
            or name.endswith(".flprefab")
            or name.endswith(".flscn")
            or name.endswith(".flscnb")
            or name.endswith(".flscene")
            or name.endswith(".glb")
            or name.endswith(".png")
            or name.endswith(".jpg")
            or name.endswith(".jpeg")
            or name.endswith(".hdr")
            or name.endswith(".exr")
            or name.endswith(".wav")
            or name.endswith(".qoa")
            or name.endswith(".flanim")
            or name.endswith(".flrig")
            or name.endswith(".json")
            or name.endswith(".flpk")
            or name.endswith(".ttf")
            or name.endswith(".otf")
        )

    def scan_asset_browser(mut self, dir_path: String = "", pack_path: String = "") -> Int:
        """Recursively scans a workspace directory (symlink-safe) and/or `.flpk` v3 archive for the Asset Browser."""
        self.asset_browser_entries.clear()
        var target_dir = dir_path if dir_path.byte_length() > 0 else self.asset_browser_dir
        if target_dir.byte_length() == 0 and self.project_root.byte_length() > 0:
            target_dir = self.project_root
        var target_pack = pack_path if pack_path.byte_length() > 0 else self.asset_browser_pack

        if target_pack.byte_length() > 0:
            var arch = FlpkV3MappedArchive.open(target_pack)
            if arch.valid:
                for i in range(arch.entry_count):
                    var rp = arch.entry_path(i)
                    if rp.byte_length() == 0:
                        continue
                    var dup = False
                    for e in self.asset_browser_entries:
                        if e == rp:
                            dup = True
                            break
                    if not dup:
                        self.asset_browser_entries.append(rp)
                arch.close()

        if target_dir.byte_length() > 0 and is_directory(target_dir, follow_symlinks=True):
            var dir_stack = List[String]()
            var rel_stack = List[String]()
            var depth_stack = List[Int]()
            dir_stack.append(target_dir)
            rel_stack.append("")
            depth_stack.append(0)
            while len(dir_stack) > 0:
                var cur_dir = dir_stack.pop()
                var cur_rel = rel_stack.pop()
                var cur_depth = depth_stack.pop()
                var names = list_directory(cur_dir)
                for name in names:
                    if (
                        name == "."
                        or name == ".."
                        or name == ".git"
                        or name == "build"
                        or name == ".pixi"
                        or name == "__pycache__"
                        or name == ".agents"
                    ):
                        continue
                    var full = cur_dir + "/" + name
                    if is_symlink(full):
                        continue
                    var rel = name if cur_rel.byte_length() == 0 else cur_rel + "/" + name
                    if is_directory(full, follow_symlinks=False):
                        if cur_depth < 16:
                            dir_stack.append(full)
                            rel_stack.append(rel)
                            depth_stack.append(cur_depth + 1)
                    elif Self._is_asset_browser_ext(Self._ascii_lower(name)):
                        var dup = False
                        for e in self.asset_browser_entries:
                            if e == rel:
                                dup = True
                                break
                        if not dup:
                            self.asset_browser_entries.append(rel)

        # Deterministic lexicographic order
        var n_ent = len(self.asset_browser_entries)
        for i in range(1, n_ent):
            var key = self.asset_browser_entries[i]
            var j = i - 1
            while j >= 0 and self.asset_browser_entries[j] > key:
                self.asset_browser_entries[j + 1] = self.asset_browser_entries[j]
                j -= 1
            self.asset_browser_entries[j + 1] = key

        self.asset_browser_scanned = True
        return len(self.asset_browser_entries)

    @staticmethod
    def _ascii_lower(s: String) -> String:
        var b = s.as_bytes()
        var out = List[UInt8](capacity=len(b))
        for i in range(len(b)):
            var c = b[i]
            out.append(c + 32 if (c >= 65 and c <= 90) else c)
        return String(unsafe_from_utf8=out)

    @staticmethod
    def _hash_str_u64(s: String) -> UInt64:
        var h = UInt64(14695981039346656037)
        var b = s.as_bytes()
        for i in range(len(b)):
            h = (h ^ UInt64(b[i])) * UInt64(1099511628211)
        return h if h != 0 else UInt64(1)

    @staticmethod
    def _basename(path: String) -> String:
        var b = path.as_bytes()
        var last_slash = -1
        for i in range(len(b)):
            if b[i] == 47 or b[i] == 92:
                last_slash = i
        return String(path[byte=last_slash + 1 : len(b)])

    def _resolve_asset_disk_path(self, rel_path: String) -> String:
        var norm = normalize_asset_rel_path(rel_path)
        var clean = norm[1] if norm[0] else rel_path
        if self.project_root.byte_length() > 0:
            var cand_proj = self.project_root + "/" + clean
            if file_size(cand_proj) > 0:
                return cand_proj
        if self.asset_browser_dir.byte_length() > 0:
            var cand_ab = self.asset_browser_dir + "/" + clean
            if file_size(cand_ab) > 0:
                return cand_ab
        if self.project_root.byte_length() > 0:
            return self.project_root + "/" + clean
        if self.asset_browser_dir.byte_length() > 0:
            return self.asset_browser_dir + "/" + clean
        return clean

    def _project_rel_for_entry(self, entry_rel: String) -> String:
        var norm = normalize_asset_rel_path(entry_rel)
        var clean = norm[1] if norm[0] else entry_rel
        if self.asset_browser_dir.byte_length() > 0 and self.project_root.byte_length() > 0:
            if self.asset_browser_dir != self.project_root:
                var prefix = self.project_root + "/"
                var joined = self.asset_browser_dir + "/" + clean
                if joined.startswith(prefix):
                    var sub = String(joined[byte=prefix.byte_length() : joined.byte_length()])
                    var snorm = normalize_asset_rel_path(sub)
                    if snorm[0]:
                        return snorm[1]
        return clean

    @staticmethod
    def _asset_kind_for_ext(lower_path: String) -> UInt8:
        if lower_path.endswith(".glb"):
            return ASSET_KIND_MESH
        if lower_path.endswith(".png") or lower_path.endswith(".jpg") or lower_path.endswith(".jpeg"):
            return ASSET_KIND_TEXTURE
        if lower_path.endswith(".wav") or lower_path.endswith(".qoa"):
            return ASSET_KIND_AUDIO
        return ASSET_KIND_RAW

    @staticmethod
    def _validate_asset_payload(lower_path: String, raw: List[UInt8]) -> Tuple[Bool, String]:
        if len(raw) == 0:
            return (False, "empty asset file")
        if lower_path.endswith(".glb"):
            if len(raw) < 12 or raw[0] != 0x67 or raw[1] != 0x6C or raw[2] != 0x54 or raw[3] != 0x46:
                return (False, "invalid GLB header")
            var mdl = parse_native_glb(raw)
            if not mdl.loaded() or (len(mdl.meshes) == 0 and len(mdl.animations) == 0):
                return (False, "corrupt or empty GLB payload")
            return (True, "")
        if lower_path.endswith(".png"):
            var img = decode_png(raw, 4096)
            if not img.valid or img.width <= 0 or img.height <= 0:
                return (False, "corrupt PNG payload")
            return (True, "")
        if lower_path.endswith(".jpg") or lower_path.endswith(".jpeg"):
            var img = decode_image_auto(raw, 4096)
            if not img.valid or img.width <= 0 or img.height <= 0:
                return (False, "corrupt image payload")
            return (True, "")
        if lower_path.endswith(".wav"):
            var wav = decode_wav(raw)
            if not wav.valid:
                return (False, "corrupt WAV payload")
            return (True, "")
        if lower_path.endswith(".qoa"):
            var qoa = decode_qoa(raw)
            if not qoa.valid:
                return (False, "corrupt QOA payload")
            return (True, "")
        if lower_path.endswith(".flmat"):
            try:
                _ = load_flmat_bytes(raw)
                return (True, "")
            except err:
                return (False, "invalid .flmat: " + String(err))
        if lower_path.endswith(".flprefab"):
            try:
                _ = load_flprefab_bytes(raw)
                return (True, "")
            except err:
                return (False, "invalid .flprefab: " + String(err))
        if lower_path.endswith(".flrig"):
            try:
                _ = load_flrig_bytes(raw)
                return (True, "")
            except err:
                return (False, "invalid .flrig: " + String(err))
        if lower_path.endswith(".flanim"):
            try:
                _ = load_flanim_bytes(raw)
                return (True, "")
            except err:
                return (False, "invalid .flanim: " + String(err))
        if lower_path.endswith(".flpk"):
            if len(raw) < 4 or raw[0] != 0x46 or raw[1] != 0x4C or raw[2] != 0x50 or raw[3] != 0x4B:
                return (False, "invalid .flpk header")
            return (True, "")
        if lower_path.endswith(".flscn") or lower_path.endswith(".flscnb") or lower_path.endswith(".flscene") or lower_path.endswith(".json") or lower_path.endswith(".hdr") or lower_path.endswith(".exr") or lower_path.endswith(".ttf") or lower_path.endswith(".otf"):
            return (True, "")
        return (False, "unsupported asset extension")

    def import_asset_to_project(
        mut self, src_path: String, dest_subdir: String = "assets"
    ) -> Tuple[Bool, AssetRef]:
        """Validates `src_path`, copies into `<project_root>/<dest_subdir>/<filename>`, refreshes AssetServer & browser."""
        var root = self.project_root if self.project_root.byte_length() > 0 else self.asset_browser_dir
        if root.byte_length() == 0 or src_path.byte_length() == 0 or is_symlink(src_path):
            self.push_toast("Import failed: invalid project or source path", 2, 3.0)
            return (False, AssetRef())
        var base = Self._basename(src_path)
        var lower_base = Self._ascii_lower(base)
        if base.byte_length() == 0 or not Self._is_asset_browser_ext(lower_base):
            self.push_toast("Import failed: unsupported asset type", 2, 3.0)
            return (False, AssetRef())
        var raw = read_bytes(src_path)
        var val = Self._validate_asset_payload(lower_base, raw)
        if not val[0]:
            self.push_toast("Import rejected: " + val[1], 2, 3.5)
            return (False, AssetRef())
        var sub_norm = normalize_asset_rel_path(dest_subdir)
        if not sub_norm[0]:
            self.push_toast("Import failed: invalid destination subdir", 2, 3.0)
            return (False, AssetRef())
        var rel_dst = (sub_norm[1] + "/" + base) if sub_norm[1].byte_length() > 0 else base
        var norm_dst = normalize_asset_rel_path(rel_dst)
        if not norm_dst[0] or norm_dst[1].byte_length() == 0:
            return (False, AssetRef())
        var clean_rel = norm_dst[1]
        var abs_dst = root + "/" + clean_rel
        var pdir = parent_directory(abs_dst)
        if pdir.byte_length() > 0 and pdir != ".":
            _ = make_dirs(pdir)
        if not write_bytes(abs_dst, raw):
            self.push_toast("Import failed: could not write " + clean_rel, 2, 3.5)
            return (False, AssetRef())
        var kind = Self._asset_kind_for_ext(lower_base)
        var srv_opt = self.app.resources.get_ptr_mut[AssetServer]()
        if srv_opt:
            if srv_opt.value()[].loose_root.byte_length() == 0:
                srv_opt.value()[].loose_root = root
            srv_opt.value()[].index_loose_directory()
        self.asset_browser_scanned = False
        _ = self.scan_asset_browser()
        var aref = AssetRef.from_path(clean_rel, kind)
        self.push_toast("Imported " + clean_rel, 0, 2.5)
        return (True, aref^)

    def _compute_viewport_drop_pos(self, px: Float32, py: Float32, vp: RectF) -> Vec3:
        if vp.w > Float32(1.0) and vp.h > Float32(1.0) and vp.contains(px, py):
            var vxf = ViewportTransform.from_logical(vp, self._last_screen_w, self._last_screen_h, Float32(1.0))
            var ray = vxf.ray_logical(px, py, self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov_y_deg)
            var hit = self.raycast_scene(ray.origin, ray.dir, Float32(250.0))
            if hit.hit:
                return hit.point
            var gp = ray_plane_hit(ray, Vec3(0.0, self.ground_plane_y, 0.0), Vec3(0.0, 1.0, 0.0))
            if gp[0]:
                return gp[1]
        return self.cam_target

    def _assign_flmat_to_entity(mut self, entity_id: Int, rel_mat: String) -> Bool:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return False
        var disk_path = self._resolve_asset_disk_path(rel_mat)
        var raw = read_bytes(disk_path)
        if len(raw) == 0:
            self.push_toast("Material missing: " + rel_mat, 2, 3.0)
            return False
        var flmat: FlmatAsset
        try:
            flmat = load_flmat_bytes(raw)
        except err:
            self.push_toast("Invalid .flmat: " + String(err), 2, 3.5)
            return False
        var before_blob = self._capture_world_blob()
        var mref = AssetRef.from_path(rel_mat, ASSET_KIND_RAW)
        var mr_opt = self.app.world.get_ptr_mut[MeshRendererComponent](e)
        if mr_opt:
            mr_opt.value()[].material_ref = mref.copy()
        var mo = MaterialOverrideComponent(
            -1,
            flmat.base_color,
            flmat.roughness,
            flmat.metallic,
            flmat.emissive,
            enabled=True,
            opacity=flmat.opacity,
            alpha_cutoff=flmat.alpha_cutoff,
            clearcoat=flmat.clearcoat,
            clearcoat_roughness=flmat.clearcoat_roughness,
            sheen=flmat.sheen,
            subsurface=flmat.subsurface,
            material_ref=mref.copy(),
        )
        _ = self.app.world.insert[MaterialOverrideComponent](e, mo^)
        var st = self.app.world.get[StudioStyleComponent](e) if self.app.world.has[StudioStyleComponent](e) else StudioStyleComponent()
        _apply_flmat_to_studio_style(st, flmat)
        _ = self.app.world.insert[StudioStyleComponent](e, st)
        var srv_opt = self.app.resources.get_ptr_mut[AssetServer]()
        if srv_opt:
            ref srv = srv_opt.value()[unsafe_offset=0]
            if srv.loose_root.byte_length() == 0 and self.project_root.byte_length() > 0:
                srv.loose_root = self.project_root
            _ = srv.reload_raw_from_disk(rel_mat)
            var rh = srv.resolve_raw(mref)
            if srv.raws.is_loaded(rh):
                var cur_ep = srv.raws.reload_epoch(rh)
                var mkey = _studio_mat_ref_key(mref)
                var sync_opt = self.app.resources.get_ptr_mut[StudioFlmatStyleSyncResource]()
                if sync_opt:
                    ref sync = sync_opt.value()[unsafe_offset=0]
                    sync.known_internal_mat_epochs[mkey] = cur_ep
                    sync.synced_entity_epochs[e.to_bits()] = (mkey, cur_ep)
        _ = self.commit_ecs_edit(entity_id, "Assign Material " + Self._basename(rel_mat), before_blob^)
        self.push_toast("Assigned " + rel_mat, 0, 2.0)
        return True

    def activate_asset_entry(
        mut self,
        entry_rel: String,
        target_entity_id: Int = -1,
        drop_pos: Vec3 = Vec3(0.0, 0.0, 0.0),
        from_drop: Bool = False,
    ) -> Int:
        """Handles Asset Browser click or Viewport/Inspector drop for `.flscn`, `.glb`, `.flprefab`, `.flmat`, `.png`/`.jpg`, `.wav`/`.qoa`, `.flrig`, `.flanim`."""
        var rel_path = self._project_rel_for_entry(entry_rel)
        var lower = Self._ascii_lower(rel_path)
        var full_path = self._resolve_asset_disk_path(entry_rel)
        var base_name = Self._basename(rel_path)

        if lower.endswith(".flscn") or lower.endswith(".flscnb") or lower.endswith(".flscene"):
            if not from_drop:
                _ = self.request_load_scene(full_path)
            return -1

        if lower.endswith(".glb"):
            var raw = read_bytes(full_path)
            if len(raw) < 12 or raw[0] != 0x67 or raw[1] != 0x6C or raw[2] != 0x54 or raw[3] != 0x46:
                self.push_toast("Cannot spawn invalid GLB: " + base_name, 2, 3.0)
                return -1
            var mdl = parse_native_glb(raw)
            if not mdl.loaded() or len(mdl.meshes) == 0:
                self.push_toast("Empty or corrupt GLB: " + base_name, 2, 3.0)
                return -1
            var has_skin = len(mdl.skins) > 0
            var up_mesh_id = 0
            var up_mat_id = -1
            var rend_opt = self.app.resources.get_ptr_mut[ScalableRenderer]()
            if rend_opt:
                var up = rend_opt.value()[unsafe_offset=0].upload_native_model(
                    mdl, build_lods=True, upload_skinned=has_skin
                )
                if len(up.mesh_entries) > 0:
                    up_mesh_id = up.mesh_entries[0].mesh_id
                if len(up.material_ids) > 0:
                    up_mat_id = up.material_ids[0]
            var mref = AssetRef.from_path(rel_path, ASSET_KIND_MESH)
            var before_blob = self._capture_world_blob()
            var prev_ids = self.selected_entity_ids.copy()
            var prev_primary = self.selected_entity_id
            var prev_anchor = self.selection_anchor_id

            var e = self.app.world.spawn_empty()
            var eid = Int(e.index)
            var lt = LocalTransform(drop_pos, Quat.identity(), Vec3(1.0, 1.0, 1.0))
            var gt = GlobalTransform.from_local(lt)
            _ = self.app.world.insert[LocalTransform](e, lt)
            _ = self.app.world.insert[GlobalTransform](e, gt.copy())
            _ = self.app.world.insert[PrevGlobalTransform](e, PrevGlobalTransform(gt^))
            _ = self.app.world.insert[MeshRendererComponent](
                e, MeshRendererComponent(up_mesh_id, up_mat_id, True, True, mref.copy(), AssetRef())
            )
            _ = self.app.world.insert[MaterialOverrideComponent](
                e, MaterialOverrideComponent(-1, Vec3(0.82, 0.84, 0.88), Float32(0.38), Float32(0.10), Vec3(), enabled=False)
            )
            _ = self.app.world.insert[StudioEntityMetaComponent](
                e, StudioEntityMetaComponent(name=base_name, collider_kind=COLLIDER_BOX)
            )
            _ = self.app.world.insert[StudioStyleComponent](e, StudioStyleComponent())
            self._apply_collider_and_rb(e, COLLIDER_BOX, Vec3(0.5, 0.5, 0.5), Float32(0.0), Float32(0.32), Float32(0.55), Vec3(), Vec3())

            if has_skin or len(mdl.animations) > 0:
                var mid_u64 = Self._hash_str_u64(rel_path)
                var skel = SkeletonInstance(len(mdl.nodes))
                skel.ensure_topology(mdl)
                for ni in range(len(mdl.nodes)):
                    skel.pose.t[ni] = mdl.nodes[ni].translation
                    skel.pose.r[ni] = mdl.nodes[ni].rotation
                    skel.pose.s[ni] = mdl.nodes[ni].scale
                var sib_rig_ref = AssetRef()
                if rel_path.byte_length() > 4:
                    var cand_rig = String(rel_path[byte=0 : rel_path.byte_length() - 4]) + ".flrig"
                    if file_size(self._resolve_asset_disk_path(cand_rig)) > 0:
                        sib_rig_ref = AssetRef.from_path(cand_rig, ASSET_KIND_RAW)
                _ = self.app.world.insert[SkeletonComponent](
                    e, SkeletonComponent(mid_u64, skel^, True, sib_rig_ref.copy())
                )
                if not sib_rig_ref.is_empty():
                    _ = self.app.world.insert[RigComponent](e, RigComponent(sib_rig_ref.copy()))
                var clip0 = mdl.animations[0].name if len(mdl.animations) > 0 else String()
                _ = self.app.world.insert[AnimatorComponent](
                    e, AnimatorComponent(0, clip_index=0 if len(mdl.animations) > 0 else -1, clip_name=clip0, playing=True, loop=True)
                )
                var anim_cache_opt = self.app.resources.get_ptr_mut[AnimAssetCache]()
                if anim_cache_opt:
                    anim_cache_opt.value()[unsafe_offset=0].register_model(mid_u64, mdl^)

            self.select_entity(eid)
            self._sync_ecs_and_physics()
            var aff = List[Int]()
            aff.append(eid)
            self._record_world_tx(
                TX_SPAWN_ENTITY, eid, aff^, before_blob^, prev_ids, prev_primary, prev_anchor, "Spawn GLB " + base_name
            )
            self.push_toast("Spawned " + base_name, 0, 2.0)
            return eid

        if lower.endswith(".flprefab"):
            var pf: Prefab
            try:
                pf = load_flprefab_file(full_path)
            except err:
                self.push_toast("Invalid .flprefab: " + String(err), 2, 3.0)
                return -1
            if len(pf.nodes) == 0:
                self.push_toast("Empty .flprefab: " + base_name, 2, 3.0)
                return -1
            var before_blob = self._capture_world_blob()
            var prev_ids = self.selected_entity_ids.copy()
            var prev_primary = self.selected_entity_id
            var prev_anchor = self.selection_anchor_id
            var spawned: List[Entity]
            try:
                spawned = pf.instantiate(self.app.world, self.registry)
            except err:
                self.push_toast("Prefab spawn failed: " + String(err), 2, 3.0)
                return -1
            if len(spawned) == 0:
                return -1
            var pf_ref = AssetRef.from_path(rel_path, ASSET_KIND_RAW)
            var aff = List[Int]()
            for si in range(len(spawned)):
                var se = spawned[si]
                aff.append(Int(se.index))
                var nname = pf.nodes[si].name if si < len(pf.nodes) else (pf.name + "_" + String(si))
                if not self.app.world.has[StudioEntityMetaComponent](se):
                    _ = self.app.world.insert[StudioEntityMetaComponent](se, StudioEntityMetaComponent(name=nname))
                if not self.app.world.has[StudioStyleComponent](se):
                    _ = self.app.world.insert[StudioStyleComponent](se, StudioStyleComponent())
                if si < len(pf.nodes) and pf.nodes[si].parent_slot < 0:
                    var lt_opt = self.app.world.get_ptr_mut[LocalTransform](se)
                    if lt_opt:
                        lt_opt.value()[unsafe_offset=0].translation = lt_opt.value()[unsafe_offset=0].translation + drop_pos
                    else:
                        _ = self.app.world.insert[LocalTransform](se, LocalTransform(drop_pos, Quat.identity(), Vec3(1.0, 1.0, 1.0)))
                var lt_cur = self.app.world.get[LocalTransform](se) if self.app.world.has[LocalTransform](se) else LocalTransform()
                var gt_cur = GlobalTransform.from_local(lt_cur)
                if not self.app.world.has[GlobalTransform](se):
                    _ = self.app.world.insert[GlobalTransform](se, gt_cur.copy())
                if not self.app.world.has[PrevGlobalTransform](se):
                    _ = self.app.world.insert[PrevGlobalTransform](se, PrevGlobalTransform(gt_cur^))
                if si == 0 and not self.app.world.has[PrefabInstanceComponent](se):
                    _ = self.app.world.insert[PrefabInstanceComponent](se, PrefabInstanceComponent(pf.name, pf_ref.copy(), 0))
            var root_eid = Int(spawned[0].index)
            self.select_entity(root_eid)
            self._sync_ecs_and_physics()
            self._record_world_tx(
                TX_SPAWN_ENTITY, root_eid, aff^, before_blob^, prev_ids, prev_primary, prev_anchor, "Spawn Prefab " + pf.name
            )
            self.push_toast("Spawned Prefab " + pf.name, 0, 2.0)
            return root_eid

        if lower.endswith(".flmat"):
            if target_entity_id >= 0 and self.is_alive_id(target_entity_id):
                if self._assign_flmat_to_entity(target_entity_id, rel_path):
                    return target_entity_id
                return -1
            var raw = read_bytes(full_path)
            var flmat: FlmatAsset
            try:
                flmat = load_flmat_bytes(raw)
            except err:
                self.push_toast("Invalid .flmat: " + String(err), 2, 3.0)
                return -1
            var eid = self.spawn_entity(
                base_name,
                position=drop_pos,
                collider_kind=COLLIDER_SPHERE,
                base_color=flmat.base_color,
                roughness=flmat.roughness,
                metallic=flmat.metallic,
                mesh_id=1,
                record_undo=True,
            )
            if eid >= 0:
                _ = self._assign_flmat_to_entity(eid, rel_path)
            return eid

        if lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"):
            var raw = read_bytes(full_path)
            var img = decode_image_auto(raw, 4096)
            if not img.valid:
                self.push_toast("Invalid image: " + base_name, 2, 3.0)
                return -1
            var tex_ref = AssetRef.from_path(rel_path, ASSET_KIND_TEXTURE)
            if target_entity_id >= 0 and self.is_alive_id(target_entity_id):
                var te = self.resolve_entity(target_entity_id)
                var mo_p = self.app.world.get_ptr[MaterialOverrideComponent](te)
                var mr_p = self.app.world.get_ptr[MeshRendererComponent](te)
                var mat_rel = String()
                if mo_p and not mo_p.value()[].material_ref.is_empty():
                    mat_rel = mo_p.value()[].material_ref.rel_path
                elif mr_p and not mr_p.value()[].material_ref.is_empty():
                    mat_rel = mr_p.value()[].material_ref.rel_path
                if mat_rel.byte_length() > 0:
                    var mat_disk = self._resolve_asset_disk_path(mat_rel)
                    try:
                        var flm = load_flmat_file(mat_disk)
                        flm.albedo_tex = tex_ref.copy()
                        _ = save_flmat_file(mat_disk, flm)
                        self.push_toast("Bound " + base_name + " -> " + mat_rel, 0, 2.2)
                        return target_entity_id
                    except:
                        pass
            self.push_toast("Texture ready: " + rel_path + " (drop on .flmat albedo slot)", 0, 2.2)
            return -1

        if lower.endswith(".wav") or lower.endswith(".qoa"):
            var raw = read_bytes(full_path)
            var audio_ok = decode_wav(raw).valid if lower.endswith(".wav") else decode_qoa(raw).valid
            if not audio_ok:
                self.push_toast("Invalid audio file: " + base_name, 2, 3.0)
                return -1
            var aref = AssetRef.from_path(rel_path, ASSET_KIND_AUDIO)
            if target_entity_id >= 0 and self.is_alive_id(target_entity_id):
                var te = self.resolve_entity(target_entity_id)
                if self.app.world.has[AudioEmitterComponent](te):
                    var before_blob = self._capture_world_blob()
                    var au_p = self.app.world.get_ptr_mut[AudioEmitterComponent](te)
                    if au_p:
                        au_p.value()[].audio_ref = aref.copy()
                        au_p.value()[].asset_path = rel_path
                        au_p.value()[].asset_id = -1
                        au_p.value()[].playing = True
                        au_p.value()[].auto_play_pending = True
                    _ = self.commit_ecs_edit(target_entity_id, "Assign Audio " + base_name, before_blob^)
                    return target_entity_id
            var top_before_spawn = self._top_tx_id()
            var eid = self.spawn_card_entity(5)
            var e = self.resolve_entity(eid)
            if e.valid():
                var lt_p = self.app.world.get_ptr_mut[LocalTransform](e)
                if lt_p:
                    lt_p.value()[].translation = drop_pos
                var au_p = self.app.world.get_ptr_mut[AudioEmitterComponent](e)
                if au_p:
                    au_p.value()[].audio_ref = aref.copy()
                    au_p.value()[].asset_path = rel_path
                    au_p.value()[].playing = True
                    au_p.value()[].auto_play_pending = True
                self._sync_ecs_and_physics()
                # Fold the emitter setup into the spawn's own undo entry, and only if the spawn made one
                # (during Play it does not, and the top entry is then an unrelated user edit).
                if self._top_tx_id() != top_before_spawn and len(self.undo_stack) > 0:
                    self.undo_stack[len(self.undo_stack) - 1].after_world_blob = Self._pack_undo_blob(self._capture_world_blob())
            return eid

        if lower.endswith(".flrig"):
            var raw = read_bytes(full_path)
            try:
                var slab = decode_cooked_rig_slab(raw)
                var rig_def = load_flrig_bytes(slab[1]) if (slab[0] and len(slab[1]) > 0) else load_flrig_bytes(raw)
                var nj = len(rig_def.joints)
                var si = SkeletonInstance(nj)
                for ji in range(nj):
                    si.parents[ji] = rig_def.joints[ji].parent_index
                    si.pose.t[ji] = rig_def.joints[ji].translation
                    si.pose.r[ji] = rig_def.joints[ji].rotation
                    si.pose.s[ji] = rig_def.joints[ji].scale
                _compute_skeleton_topo_order(si)
                var rref = AssetRef.from_path(rel_path, ASSET_KIND_RAW)
                var tid = target_entity_id if (target_entity_id >= 0 and self.is_alive_id(target_entity_id)) else self.selected_entity_id
                if tid >= 0 and self.is_alive_id(tid):
                    var te = self.resolve_entity(tid)
                    var before_blob = self._capture_world_blob()
                    _ = self.app.world.insert[RigComponent](te, RigComponent(rref.copy(), True, nj))
                    if not self.app.world.has[SkeletonComponent](te):
                        _ = self.app.world.insert[SkeletonComponent](te, SkeletonComponent(0, si^, False, rref.copy()))
                    else:
                        var sk_p = self.app.world.get_ptr_mut[SkeletonComponent](te)
                        if sk_p:
                            sk_p.value()[].rig_ref = rref.copy()
                            sk_p.value()[].skeleton = si^
                            sk_p.value()[].dirty_topology = False
                    _ = self.commit_ecs_edit(tid, "Assign Rig " + base_name, before_blob^)
                    self.push_toast("Assigned rig " + base_name, 0, 2.0)
                    return tid
                var before_blob = self._capture_world_blob()
                var prev_ids = self.selected_entity_ids.copy()
                var prev_primary = self.selected_entity_id
                var prev_anchor = self.selection_anchor_id
                var e = self.app.world.spawn_empty()
                var eid = Int(e.index)
                var lt = LocalTransform(drop_pos, Quat.identity(), Vec3(1.0, 1.0, 1.0))
                var gt = GlobalTransform.from_local(lt)
                _ = self.app.world.insert[LocalTransform](e, lt)
                _ = self.app.world.insert[GlobalTransform](e, gt.copy())
                _ = self.app.world.insert[PrevGlobalTransform](e, PrevGlobalTransform(gt^))
                _ = self.app.world.insert[StudioEntityMetaComponent](e, StudioEntityMetaComponent(name=base_name))
                _ = self.app.world.insert[StudioStyleComponent](e, StudioStyleComponent())
                _ = self.app.world.insert[RigComponent](e, RigComponent(rref.copy(), True, nj))
                _ = self.app.world.insert[SkeletonComponent](e, SkeletonComponent(0, si^, False, rref.copy()))
                self.select_entity(eid)
                self._sync_ecs_and_physics()
                var aff = List[Int]()
                aff.append(eid)
                self._record_world_tx(
                    TX_SPAWN_ENTITY, eid, aff^, before_blob^, prev_ids, prev_primary, prev_anchor, "Spawn Rig " + base_name
                )
                self.push_toast("Spawned rig " + base_name, 0, 2.0)
                return eid
            except err:
                self.push_toast("Invalid .flrig: " + String(err), 2, 3.0)
                return -1

        if lower.endswith(".flanim"):
            var raw = read_bytes(full_path)
            try:
                _ = load_flanim_bytes(raw)
                var tid = target_entity_id if (target_entity_id >= 0 and self.is_alive_id(target_entity_id)) else self.selected_entity_id
                if tid >= 0 and self.is_alive_id(tid):
                    var te = self.resolve_entity(tid)
                    var before_blob = self._capture_world_blob()
                    var gref = AssetRef.from_path(rel_path, ASSET_KIND_RAW)
                    var ac_p = self.app.world.get_ptr_mut[AnimatorComponent](te)
                    if ac_p:
                        ac_p.value()[].graph_ref = gref.copy()
                        ac_p.value()[].graph_id = 0
                        ac_p.value()[].bound = False
                    else:
                        _ = self.app.world.insert[AnimatorComponent](te, AnimatorComponent(0, graph_ref=gref.copy()))
                    _ = self.commit_ecs_edit(tid, "Assign AnimGraph " + base_name, before_blob^)
                    self.push_toast("Assigned graph " + base_name, 0, 2.0)
                    return tid
                self.push_toast("Select or drop onto an entity to assign graph: " + base_name, 1, 2.5)
            except err:
                self.push_toast("Invalid .flanim: " + String(err), 2, 3.0)
                return -1
            return -1

        return -1

    def live_entity_count(self) -> Int:
        return self.app.world.entity_count()

    def world_matrix(self, entity_id: Int) -> Mat4:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return Mat4()
        var gt_opt = self.app.world.get_ptr[GlobalTransform](e)
        if gt_opt:
            return gt_opt.value()[unsafe_offset=0].matrix.copy()
        var lt_opt = self.app.world.get_ptr[LocalTransform](e)
        if lt_opt:
            return lt_opt.value()[unsafe_offset=0].to_mat4()
        return Mat4()

    # ---- Reversible Coalesced Transaction Stack ----

    @staticmethod
    def _pack_undo_blob(var raw: List[UInt8]) -> List[UInt8]:
        """Undo keeps two world blobs per step; LZ4-packing them (with an integrity hash) cuts undo memory
        several-fold. Already-packed and tiny blobs are kept as they are."""
        if len(raw) < 1024 or lz4_packed_raw_size(raw) >= 0:
            return raw^
        var packed = lz4_pack(raw)
        if len(packed) >= len(raw):
            return raw^
        return packed^

    def _push_undo(mut self, var tx: EditorTransaction):
        tx.before_world_blob = Self._pack_undo_blob(tx.before_world_blob^)
        tx.after_world_blob = Self._pack_undo_blob(tx.after_world_blob^)
        if tx.tx_id == UInt64(0):
            tx.tx_id = self._next_tx_id
            self._next_tx_id += UInt64(1)
        if tx.agent_id == 0 and self._active_tx_agent_id != 0:
            tx.agent_id = self._active_tx_agent_id
        if tx.step_id == UInt64(0) and self._active_tx_step_id != UInt64(0):
            tx.step_id = self._active_tx_step_id
        self.undo_stack.append(tx^)
        self._enforce_undo_budget()

    def undo_history_bytes(self) -> Int:
        """Approximate bytes held by undo *and* redo history."""
        var total = 0
        for i in range(len(self.undo_stack)):
            total += self.undo_stack[i].approx_bytes()
        for i in range(len(self.redo_stack)):
            total += self.redo_stack[i].approx_bytes()
        return total

    @staticmethod
    def _drop_front(mut stack: List[EditorTransaction], n: Int):
        """Removes the `n` oldest entries in place: entries are moved down (swapped), never deep-copied."""
        if n <= 0:
            return
        var count = len(stack)
        for i in range(n, count):
            stack.swap_elements(i - n, i)
        for _ in range(min(n, count)):
            _ = stack.pop()

    def _enforce_undo_budget(mut self):
        """Keeps history within `MAX_UNDO_STACK` entries and `MAX_UNDO_BLOB_BYTES` (undo + redo), evicting
        the oldest undo entries first, then the furthest redo entries. The newest undo entry is always
        kept. When the saved state's entry is evicted, the save point moves to wherever it is still
        reachable (the bottom of the stack), or becomes unreachable so undo never fakes a clean scene."""
        var total = self.undo_history_bytes()
        var n = len(self.undo_stack)
        var drop = 0
        while n - drop > 1 and (n - drop > MAX_UNDO_STACK or total > MAX_UNDO_BLOB_BYTES):
            total -= self.undo_stack[drop].approx_bytes()
            drop += 1
        if drop > 0:
            var last_dropped = self.undo_stack[drop - 1].tx_id
            var saved_dropped = self._saved_tx_id == UInt64(0)
            for k in range(drop):
                if self.undo_stack[k].tx_id == self._saved_tx_id:
                    saved_dropped = True
            if self._saved_tx_id == last_dropped:
                self._saved_tx_id = UInt64(0)  # the empty-undo state *is* the saved one now
            elif saved_dropped:
                self._saved_tx_id = UInt64(0xFFFFFFFFFFFFFFFF)
            Self._drop_front(self.undo_stack, drop)
        var rdrop = 0
        while rdrop < len(self.redo_stack) and total > MAX_UNDO_BLOB_BYTES:
            total -= self.redo_stack[rdrop].approx_bytes()
            rdrop += 1
        Self._drop_front(self.redo_stack, rdrop)

    def _record_world_tx(
        mut self,
        op: Int,
        primary_id: Int,
        var affected: List[Int],
        var before_blob: List[UInt8],
        prev_ids: List[Int],
        prev_primary: Int,
        prev_anchor: Int,
        label: String,
        coalesce_key: UInt64 = 0,
    ):
        self._ensure_editor_ids()  # new entities get their id before the after-blob is captured
        if self.pie.mode != PIE_STOPPED:
            # Play-In-Editor edits are thrown away by `pie_stop` (it restores the pre-play world), so
            # they must not enter undo/redo: an entry here would replay a mid-simulation world after Stop.
            return
        if self.agent_bus.time_travel_active:
            # Time travel is read-only: roll the attempted edit back to the checkpoint being shown.
            if len(before_blob) > 0:
                _ = self._restore_world_blob(before_blob)
            elif len(self._time_travel_view_blob) > 0:
                var view = self._time_travel_view_blob.copy()
                _ = self._restore_world_blob(view)
            self.push_toast("Time travel is read-only: exit it to edit", 1, 2.5)
            return
        var after_blob = self._capture_world_blob()
        var sync_tx_opt = self.app.resources.get_ptr[StudioFlmatStyleSyncResource]()
        var ext_snap = (
            _copy_str_u64_dict(sync_tx_opt.value()[unsafe_offset=0].external_mat_reload_epochs)
            if sync_tx_opt
            else Dict[String, UInt64]()
        )
        if coalesce_key != 0 and len(self.undo_stack) > 0 and len(self.redo_stack) == 0:
            ref top = self.undo_stack[len(self.undo_stack) - 1]
            if top.op == op and top.entity_id == primary_id and top.coalesce_key == coalesce_key and len(after_blob) > 0:
                top.after_world_blob = Self._pack_undo_blob(after_blob^)
                top.next_selected_id = self.selected_entity_id
                top.next_selected_ids = self.selected_entity_ids.copy()
                top.next_anchor_id = self.selection_anchor_id
                top.external_mat_reload_epochs = ext_snap^
                if top.tx_id == self._saved_tx_id:
                    self._saved_tx_id = UInt64(0xFFFFFFFFFFFFFFFF)  # the saved entry now holds newer edits
                self._enforce_undo_budget()  # the replaced after-blob can be larger than the old one
                self.mark_dirty(True)
                return

        if len(before_blob) == 0 or len(after_blob) == 0:
            # Never record an entry that cannot be restored: undo would fail on it forever.
            if len(before_blob) > 0:
                _ = self._restore_world_blob(before_blob)  # refuse the edit: put the scene back
                self.push_toast(label + " refused: the scene could not be snapshotted for undo", 2, 4.0)
            else:
                self.push_toast(label + " cannot be undone: the scene could not be snapshotted", 2, 4.0)
                self.mark_dirty(True)
            return

        var tx = EditorTransaction(op, primary_id, label)
        tx.affected_ids = affected^
        tx.before_world_blob = before_blob^
        tx.after_world_blob = after_blob^
        tx.prev_selected_id = prev_primary
        tx.next_selected_id = self.selected_entity_id
        tx.prev_selected_ids = prev_ids.copy()
        tx.next_selected_ids = self.selected_entity_ids.copy()
        tx.prev_anchor_id = prev_anchor
        tx.next_anchor_id = self.selection_anchor_id
        tx.coalesce_key = coalesce_key
        tx.external_mat_reload_epochs = ext_snap^
        self._push_undo(tx^)
        self.redo_stack.clear()
        self.mark_dirty(True)

    def _apply_collider_and_rb(
        mut self,
        e: Entity,
        collider_kind: Int,
        collider_half_extents: Vec3,
        mass: Float32,
        restitution: Float32,
        friction: Float32,
        lin_vel: Vec3,
        ang_vel: Vec3,
        update_trigger: Bool = False,
        is_trigger: Bool = False,
    ):
        var he = Vec3(
            max(Float32(0.01), collider_half_extents.x),
            max(Float32(0.01), collider_half_extents.y),
            max(Float32(0.01), collider_half_extents.z),
        )
        var m = max(Float32(0.0), mass)

        if collider_kind != COLLIDER_NONE:
            var existing_col = self.app.world.get_ptr[ColliderComponent](e)
            var target_shape = BODY_SHAPE_BOX
            if collider_kind == COLLIDER_SPHERE:
                target_shape = BODY_SHAPE_SPHERE
            elif collider_kind == COLLIDER_CAPSULE:
                target_shape = BODY_SHAPE_CAPSULE
            elif (
                collider_kind == COLLIDER_STATIC_MESH
                and existing_col
                and existing_col.value()[unsafe_offset=0].shape == BODY_SHAPE_CONVEX_HULL
            ):
                target_shape = BODY_SHAPE_CONVEX_HULL

            var col: ColliderComponent
            if existing_col:
                col = existing_col.value()[unsafe_offset=0].copy()
                if col.shape != target_shape:
                    col.shape = target_shape
                    if target_shape != BODY_SHAPE_CONVEX_HULL:
                        col.points.clear()
                col.half_extents = he
                if target_shape == BODY_SHAPE_SPHERE:
                    col.radius = he.x
                elif target_shape == BODY_SHAPE_CAPSULE:
                    col.radius = he.x
                    col.half_height = he.y
                else:
                    col.radius = he.length()
            else:
                if target_shape == BODY_SHAPE_SPHERE:
                    col = ColliderComponent.sphere(he.x)
                    col.half_extents = he
                elif target_shape == BODY_SHAPE_CAPSULE:
                    col = ColliderComponent.capsule(he.x, he.y)
                    col.half_extents = he
                else:
                    col = ColliderComponent.box(he)
            if update_trigger:
                col.is_trigger = is_trigger
            _ = self.app.world.insert[ColliderComponent](e, col^)
        else:
            if self.app.world.has[ColliderComponent](e):
                _ = self.app.world.remove[ColliderComponent](e)

        if collider_kind != COLLIDER_NONE or m > Float32(0.0):
            var existing_rb = self.app.world.get_ptr[RigidBodyComponent](e)
            var rb: RigidBodyComponent
            if existing_rb:
                rb = existing_rb.value()[unsafe_offset=0].copy()
                if m <= Float32(0.0):
                    if rb.motion_mode != BODY_MOTION_KINEMATIC:
                        rb.motion_mode = BODY_MOTION_STATIC
                    rb.mass = Float32(0.0)
                else:
                    rb.motion_mode = BODY_MOTION_DYNAMIC
                    rb.mass = m
                rb.restitution = restitution
                rb.friction = friction
                rb.linear_vel = lin_vel
                rb.angular_vel = ang_vel
                if collider_kind == COLLIDER_NONE:
                    rb.filter = CollisionFilter(0, 0, 0)
            else:
                rb = RigidBodyComponent.static_body() if m <= Float32(0.0) else RigidBodyComponent.dynamic(m)
                rb.restitution = restitution
                rb.friction = friction
                rb.linear_vel = lin_vel
                rb.angular_vel = ang_vel
                rb.linear_damping = Float32(0.04)
                if collider_kind == COLLIDER_NONE:
                    rb.filter = CollisionFilter(0, 0, 0)
            _ = self.app.world.insert[RigidBodyComponent](e, rb)
        else:
            if self.app.world.has[RigidBodyComponent](e):
                _ = self.app.world.remove[RigidBodyComponent](e)

    def spawn_entity(
        mut self,
        name: String,
        position: Vec3 = Vec3(),
        rotation: Quat = Quat(0.0, 0.0, 0.0, 1.0),
        scale: Vec3 = Vec3(1.0, 1.0, 1.0),
        collider_kind: Int = COLLIDER_BOX,
        collider_half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        mass: Float32 = 0.0,
        base_color: Vec3 = Vec3(0.82, 0.84, 0.88),
        roughness: Float32 = 0.38,
        metallic: Float32 = 0.10,
        mesh_id: Int = 0,
        parent_id: Int = -1,
        record_undo: Bool = True,
    ) -> Int:
        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        var e = self.app.world.spawn_empty()
        var eid = Int(e.index)
        var safe_pos = Vec3(
            position.x if isfinite(position.x) else Float32(0.0),
            position.y if isfinite(position.y) else Float32(0.0),
            position.z if isfinite(position.z) else Float32(0.0),
        )
        var q = Quat.identity()
        if isfinite(rotation.x) and isfinite(rotation.y) and isfinite(rotation.z) and isfinite(rotation.w):
            var qn = rotation.normalized()
            if isfinite(qn.x) and isfinite(qn.y) and isfinite(qn.z) and isfinite(qn.w):
                q = qn
        var safe_scl = Vec3(
            scale.x if isfinite(scale.x) else Float32(1.0),
            scale.y if isfinite(scale.y) else Float32(1.0),
            scale.z if isfinite(scale.z) else Float32(1.0),
        )
        var lt = LocalTransform(safe_pos, q, safe_scl)
        var gt = GlobalTransform.from_local(lt)
        _ = self.app.world.insert[LocalTransform](e, lt)
        _ = self.app.world.insert[GlobalTransform](e, gt.copy())
        _ = self.app.world.insert[PrevGlobalTransform](e, PrevGlobalTransform(gt^))
        _ = self.app.world.insert[MeshRendererComponent](e, MeshRendererComponent(mesh_id, -1, True, True))
        var safe_r = roughness if isfinite(roughness) else Float32(0.38)
        var safe_m = metallic if isfinite(metallic) else Float32(0.10)
        var clamp_r = max(Float32(0.04), min(Float32(1.0), safe_r))
        var clamp_m = max(Float32(0.0), min(Float32(1.0), safe_m))
        var safe_col = Vec3(
            base_color.x if isfinite(base_color.x) else Float32(0.82),
            base_color.y if isfinite(base_color.y) else Float32(0.84),
            base_color.z if isfinite(base_color.z) else Float32(0.88),
        )
        _ = self.app.world.insert[MaterialOverrideComponent](
            e, MaterialOverrideComponent(-1, safe_col, clamp_r, clamp_m, Vec3(), True)
        )
        _ = self.app.world.insert[StudioEntityMetaComponent](
            e,
            StudioEntityMetaComponent(
                name=name,
                collider_kind=collider_kind,
                collider_half_extents=collider_half_extents,
            ),
        )
        _ = self.app.world.insert[StudioStyleComponent](e, StudioStyleComponent())
        self._apply_collider_and_rb(
            e, collider_kind, collider_half_extents, mass, Float32(0.32), Float32(0.55), Vec3(), Vec3()
        )

        if parent_id >= 0:
            var pe = self.resolve_entity(parent_id)
            if pe.valid():
                _ = set_parent(self.app.world, e, pe)

        self.select_entity(eid)
        self._sync_ecs_and_physics()

        if record_undo:
            var aff = List[Int]()
            aff.append(eid)
            self._record_world_tx(
                TX_SPAWN_ENTITY, eid, aff^, before_blob^, prev_ids, prev_primary, prev_anchor, "Spawn " + name
            )
        return eid

    def _subtree_slots(self, root_id: Int) -> List[Int]:
        """`root_id` and all its live descendants, parents before children."""
        var out = List[Int]()
        if not self.is_alive_id(root_id):
            return out^
        out.append(root_id)
        var k = 0
        while k < len(out) and len(out) <= len(self.app.world.entities):
            var ch = self.children_of(out[k])
            for ci in range(len(ch)):
                if self.is_alive_id(ch[ci]) and ch[ci] not in out:
                    out.append(ch[ci])
            k += 1
        return out^

    def _subtree_roots(self, ids: List[Int]) -> List[Int]:
        """The entries of `ids` none of whose ancestors are also in `ids` (each duplicates its own subtree,
        so a selected child of a selected parent is copied once, with its parent)."""
        var out = List[Int]()
        for i in range(len(ids)):
            var id = ids[i]
            if not self.is_alive_id(id) or id in out:
                continue
            var covered = False
            var p = self.parent_of(id)
            var guard = 0
            while p >= 0 and guard <= len(self.app.world.entities):
                if p in ids:
                    covered = True
                    break
                p = self.parent_of(p)
                guard += 1
            if not covered:
                out.append(id)
        return out^

    def _duplicate_entity_raw(mut self, src_id: Int) -> Int:
        """Duplicates `src_id` and its whole subtree with every registered component (the registry's
        binary codecs, so new component types need no change here) plus unknown components. Parent
        links and entity references inside the subtree point at the copies; runtime handles (physics
        bodies, trigger ids, BVH slots, playing sounds, editor ids) are reset. Returns the copy's slot."""
        var slots = self._subtree_slots(src_id)
        if len(slots) == 0:
            return -1
        var srcs = List[Entity](capacity=len(slots))
        for i in range(len(slots)):
            srcs.append(self.resolve_entity(slots[i]))
        var copies = duplicate_entities(self.app.world, self.registry, srcs)
        if len(copies) != len(srcs):
            self.push_toast("Duplicate failed: a component could not be copied", 2, 3.5)
            return -1
        for k in range(len(copies)):
            var ce = copies[k]
            _ = self.app.world.remove[StudioEditorIdComponent](ce)  # a fresh id below, never the source's
            var meta_opt = self.app.world.get_ptr_mut[StudioEntityMetaComponent](ce)
            if meta_opt:
                meta_opt.value()[unsafe_offset=0].bvh_slot = -1
                if k == 0:
                    meta_opt.value()[unsafe_offset=0].name = meta_opt.value()[unsafe_offset=0].name + "_Copy"
            var col_opt = self.app.world.get_ptr_mut[ColliderComponent](ce)
            if col_opt:
                col_opt.value()[unsafe_offset=0].trigger_id = -1
            var rb_opt = self.app.world.get_ptr_mut[RigidBodyComponent](ce)
            if rb_opt:
                rb_opt.value()[unsafe_offset=0].body_id = -1
            var ae_opt = self.app.world.get_ptr_mut[AudioEmitterComponent](ce)
            if ae_opt:
                ae_opt.value()[unsafe_offset=0].handle = SoundHandle()
        var root = copies[0]
        var lt_opt = self.app.world.get_ptr_mut[LocalTransform](root)
        if lt_opt:
            lt_opt.value()[unsafe_offset=0].translation = (
                lt_opt.value()[unsafe_offset=0].translation + Vec3(Float32(0.5), Float32(0.0), Float32(0.0))
            )
        else:
            var lt = LocalTransform()
            lt.translation = Vec3(Float32(0.5), Float32(0.0), Float32(0.0))
            _ = self.app.world.insert[LocalTransform](root, lt)
        self._ensure_editor_ids()
        return Int(root.index)

    def duplicate_entity(mut self, entity_id: Int, record_undo: Bool = True) -> Int:
        """Duplicates an existing live entity as a single Undo/Redo transaction."""
        if not self.is_alive_id(entity_id):
            return -1
        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        var new_id = self._duplicate_entity_raw(entity_id)
        if new_id < 0:
            return -1
        self.select_entity(new_id)
        self._sync_ecs_and_physics()

        if record_undo:
            var aff = List[Int]()
            aff.append(new_id)
            var nm = self.entities[new_id].name if new_id < len(self.entities) else "Entity"
            self._record_world_tx(
                TX_SPAWN_ENTITY, new_id, aff^, before_blob^, prev_ids, prev_primary, prev_anchor, "Duplicate " + nm
            )
        return new_id

    def duplicate_selected(mut self, record_undo: Bool = True) -> List[Int]:
        """Duplicates all currently selected entities in a single atomic transaction."""
        self._normalize_selection_from_primary()
        if self._time_travel_blocks_edit():
            return List[Int]()
        var srcs = self._subtree_roots(self.selected_entity_ids)
        var created = List[Int]()
        if len(srcs) == 0:
            return created^

        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        for i in range(len(srcs)):
            var nid = self._duplicate_entity_raw(srcs[i])
            if nid >= 0:
                created.append(nid)

        if len(created) == 0:
            return created^

        self.selected_entity_ids = created.copy()
        self.selected_entity_id = created[len(created) - 1]
        self.selection_anchor_id = created[0]
        self._sync_ecs_and_physics()

        if record_undo:
            var aff = created.copy()
            self._record_world_tx(
                TX_BATCH_EDIT,
                self.selected_entity_id,
                aff^,
                before_blob^,
                prev_ids,
                prev_primary,
                prev_anchor,
                "Duplicate " + String(len(created)) + " Entities",
            )
        return created^

    def _despawn_entity_raw(mut self, entity_id: Int) -> Bool:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return False
        # Detach live children and parent BEFORE World.despawn(e) so no dangling Children/Parent links remain.
        # Children move up to the deleted entity's parent and keep their world transform (they used to
        # snap to their parent-relative LocalTransform interpreted as world space).
        var grand_parent = self.parent_of(entity_id)
        var ch = self.children_of(entity_id)
        for ci in range(len(ch)):
            var ce = self.resolve_entity(ch[ci])
            if ce.valid():
                if not self._reparent_entity_raw(ch[ci], grand_parent, True):
                    _ = clear_parent(self.app.world, ce)
        _ = clear_parent(self.app.world, e)
        _ = self.app.world.despawn(e)
        self.gpu_materials.invalidate_entity(entity_id)
        return True

    def despawn_entity(mut self, entity_id: Int, record_undo: Bool = True) -> Bool:
        if not self.is_alive_id(entity_id):
            return False
        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id
        var nm = self.entities[entity_id].name if entity_id < len(self.entities) else "Entity"

        if not self._despawn_entity_raw(entity_id):
            return False

        self._prune_selection()
        if self.selected_entity_id < 0:
            for i in range(len(self.app.world.entities)):
                if self.app.world.entities[i].alive:
                    self.select_entity(i)
                    break

        self._sync_ecs_and_physics()
        if record_undo:
            var aff = List[Int]()
            aff.append(entity_id)
            self._record_world_tx(
                TX_DESPAWN_ENTITY, entity_id, aff^, before_blob^, prev_ids, prev_primary, prev_anchor, "Despawn " + nm
            )
        return True

    def despawn_selected(mut self, record_undo: Bool = True) -> Int:
        self._normalize_selection_from_primary()
        var targets = List[Int]()
        for i in range(len(self.selected_entity_ids)):
            if self.is_alive_id(self.selected_entity_ids[i]):
                targets.append(self.selected_entity_ids[i])
        if len(targets) == 0:
            return 0

        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        var count = 0
        for i in range(len(targets)):
            if self._despawn_entity_raw(targets[i]):
                count += 1

        self.clear_selection()
        self._sync_ecs_and_physics()
        if record_undo and count > 0:
            self._record_world_tx(
                TX_BATCH_EDIT,
                targets[0],
                targets^,
                before_blob^,
                prev_ids,
                prev_primary,
                prev_anchor,
                "Delete " + String(count) + " Entities",
            )
        return count

    def _is_ancestor(self, candidate_anc_id: Int, node_id: Int) -> Bool:
        var cur = node_id
        var hops = 0
        while cur >= 0 and hops < 64:
            if cur == candidate_anc_id:
                return True
            cur = self.parent_of(cur)
            hops += 1
        return False

    def _reparent_entity_raw(
        mut self,
        child_id: Int,
        new_parent_id: Int,
        preserve_world_transform: Bool,
    ) -> Bool:
        var child_e = self.resolve_entity(child_id)
        if not child_e.valid() or child_id == new_parent_id:
            return False
        if new_parent_id >= 0:
            if not self.is_alive_id(new_parent_id):
                return False
            if self._is_ancestor(child_id, new_parent_id):
                return False

        var pre_w_pos = Vec3()
        var pre_w_rot = Quat()
        var pre_w_scl = Vec3(1.0, 1.0, 1.0)
        if preserve_world_transform:
            var gt_opt = self.app.world.get_ptr[GlobalTransform](child_e)
            if gt_opt:
                ref gt = gt_opt.value()[unsafe_offset=0]
                pre_w_pos = gt.translation
                pre_w_rot = gt.rotation
                pre_w_scl = gt.scale
            else:
                var lt_opt = self.app.world.get_ptr[LocalTransform](child_e)
                if lt_opt:
                    ref lt = lt_opt.value()[unsafe_offset=0]
                    pre_w_pos = lt.translation
                    pre_w_rot = lt.rotation
                    pre_w_scl = lt.scale

        var ok: Bool
        if new_parent_id < 0:
            var had_parent = self.parent_of(child_id) >= 0
            ok = clear_parent(self.app.world, child_e) if had_parent else True
        else:
            var parent_e = self.resolve_entity(new_parent_id)
            ok = set_parent(self.app.world, child_e, parent_e)
        if not ok:
            return False

        if preserve_world_transform:
            var new_lt: LocalTransform
            if new_parent_id < 0:
                new_lt = LocalTransform(pre_w_pos, pre_w_rot.normalized(), pre_w_scl)
            else:
                var parent_e = self.resolve_entity(new_parent_id)
                var pgt_opt = self.app.world.get_ptr[GlobalTransform](parent_e)
                if pgt_opt:
                    ref pgt = pgt_opt.value()[unsafe_offset=0]
                    var inv_q = pgt.rotation.conjugate()
                    var rel_p = inv_q.rotate(pre_w_pos - pgt.translation)
                    var sx = pgt.scale.x if abs(pgt.scale.x) > Float32(1.0e-6) else Float32(1.0)
                    var sy = pgt.scale.y if abs(pgt.scale.y) > Float32(1.0e-6) else Float32(1.0)
                    var sz = pgt.scale.z if abs(pgt.scale.z) > Float32(1.0e-6) else Float32(1.0)
                    new_lt = LocalTransform(
                        Vec3(rel_p.x / sx, rel_p.y / sy, rel_p.z / sz),
                        (inv_q * pre_w_rot).normalized(),
                        Vec3(pre_w_scl.x / sx, pre_w_scl.y / sy, pre_w_scl.z / sz),
                    )
                else:
                    new_lt = LocalTransform(pre_w_pos, pre_w_rot.normalized(), pre_w_scl)
            _ = self.app.world.insert[LocalTransform](child_e, new_lt)
        return True

    def reparent_entity(
        mut self,
        child_id: Int,
        new_parent_id: Int,
        preserve_world_transform: Bool = False,
        record_undo: Bool = True,
    ) -> Bool:
        if not self.is_alive_id(child_id) or child_id == new_parent_id:
            return False
        if new_parent_id >= 0:
            if not self.is_alive_id(new_parent_id) or self._is_ancestor(child_id, new_parent_id):
                return False
        if self.parent_of(child_id) == new_parent_id:
            return True

        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        if not self._reparent_entity_raw(child_id, new_parent_id, preserve_world_transform):
            return False

        self._sync_ecs_and_physics()
        if record_undo:
            var aff = List[Int]()
            aff.append(child_id)
            if new_parent_id >= 0:
                aff.append(new_parent_id)
            self._record_world_tx(
                TX_REPARENT_ENTITY,
                child_id,
                aff^,
                before_blob^,
                prev_ids,
                prev_primary,
                prev_anchor,
                "Reparent #" + String(child_id),
            )
        return True

    def reparent_selected_to(
        mut self,
        target_parent_id: Int,
        preserve_world_transform: Bool = False,
        record_undo: Bool = True,
    ) -> Int:
        self._normalize_selection_from_primary()
        if target_parent_id >= 0 and not self.is_alive_id(target_parent_id):
            return 0

        var eligible = List[Int]()
        for i in range(len(self.selected_entity_ids)):
            var eid = self.selected_entity_ids[i]
            if not self.is_alive_id(eid) or eid == target_parent_id:
                continue
            if self.parent_of(eid) == target_parent_id:
                continue
            if target_parent_id >= 0 and self._is_ancestor(eid, target_parent_id):
                continue
            eligible.append(eid)

        if len(eligible) == 0:
            return 0

        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        var moved = 0
        for i in range(len(eligible)):
            if self._reparent_entity_raw(eligible[i], target_parent_id, preserve_world_transform):
                moved += 1

        if moved > 0:
            self._sync_ecs_and_physics()
            if record_undo:
                self._record_world_tx(
                    TX_REPARENT_ENTITY,
                    eligible[0],
                    eligible^,
                    before_blob^,
                    prev_ids,
                    prev_primary,
                    prev_anchor,
                    "Reparent Selected (" + String(moved) + ")",
                )
        return moved

    def _apply_view_to_world(mut self, entity_id: Int, updated: StudioEntityView) -> Bool:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return False

        var cur_par = self.parent_of(entity_id)
        if updated.parent_id != cur_par:
            if updated.parent_id < 0:
                _ = clear_parent(self.app.world, e)
            else:
                var pe = self.resolve_entity(updated.parent_id)
                if not pe.valid() or not set_parent(self.app.world, e, pe):
                    return False

        var safe_tr = Vec3(
            updated.translation.x if isfinite(updated.translation.x) else Float32(0.0),
            updated.translation.y if isfinite(updated.translation.y) else Float32(0.0),
            updated.translation.z if isfinite(updated.translation.z) else Float32(0.0),
        )
        var safe_rot = Quat.identity()
        if (
            isfinite(updated.rotation.x)
            and isfinite(updated.rotation.y)
            and isfinite(updated.rotation.z)
            and isfinite(updated.rotation.w)
        ):
            var rn = updated.rotation.normalized()
            if isfinite(rn.x) and isfinite(rn.y) and isfinite(rn.z) and isfinite(rn.w):
                safe_rot = rn
        var safe_scl = Vec3(
            updated.scale.x if isfinite(updated.scale.x) else Float32(1.0),
            updated.scale.y if isfinite(updated.scale.y) else Float32(1.0),
            updated.scale.z if isfinite(updated.scale.z) else Float32(1.0),
        )
        var prev_vis = self.entities[entity_id].visible if (entity_id >= 0 and entity_id < len(self.entities)) else True
        var prev_en = self.entities[entity_id].enabled if (entity_id >= 0 and entity_id < len(self.entities)) else True
        var prev_trig = self.entities[entity_id].is_trigger if (entity_id >= 0 and entity_id < len(self.entities)) else False
        var eff_en = updated.enabled
        if updated.enabled != prev_en:
            eff_en = updated.enabled
        elif updated.visible != prev_vis:
            eff_en = updated.visible

        _ = self.app.world.insert[LocalTransform](
            e, LocalTransform(safe_tr, safe_rot, safe_scl)
        )
        if self.app.world.has[MeshRendererComponent](e) or updated.mesh_id >= 0:
            var new_mr = MeshRendererComponent(updated.mesh_id, updated.material_id, True, eff_en)
            var old_mr = self.app.world.get_ptr[MeshRendererComponent](e)
            if old_mr:
                new_mr.mesh_ref = old_mr.value()[].mesh_ref.copy()
                new_mr.material_ref = old_mr.value()[].material_ref.copy()
            _ = self.app.world.insert[MeshRendererComponent](e, new_mr^)
        var safe_r = updated.roughness if isfinite(updated.roughness) else Float32(0.38)
        var safe_m = updated.metallic if isfinite(updated.metallic) else Float32(0.10)
        var clamp_r = max(Float32(0.04), min(Float32(1.0), safe_r))
        var clamp_m = max(Float32(0.0), min(Float32(1.0), safe_m))
        var safe_col = Vec3(
            updated.base_color.x if isfinite(updated.base_color.x) else Float32(0.82),
            updated.base_color.y if isfinite(updated.base_color.y) else Float32(0.84),
            updated.base_color.z if isfinite(updated.base_color.z) else Float32(0.88),
        )
        var safe_em = Vec3(
            updated.emissive.x if isfinite(updated.emissive.x) else Float32(0.0),
            updated.emissive.y if isfinite(updated.emissive.y) else Float32(0.0),
            updated.emissive.z if isfinite(updated.emissive.z) else Float32(0.0),
        )
        var safe_op = max(Float32(0.0), min(Float32(1.0), updated.opacity)) if isfinite(updated.opacity) else Float32(1.0)
        var safe_ac = max(Float32(0.0), min(Float32(1.0), updated.alpha_cutoff)) if isfinite(updated.alpha_cutoff) else Float32(0.0)
        var safe_cc = max(Float32(0.0), min(Float32(1.0), updated.clearcoat)) if isfinite(updated.clearcoat) else Float32(0.0)
        var safe_ccr = max(Float32(0.04), min(Float32(1.0), updated.clearcoat_roughness)) if isfinite(updated.clearcoat_roughness) else Float32(0.08)
        var safe_sh = max(Float32(0.0), min(Float32(1.0), updated.sheen)) if isfinite(updated.sheen) else Float32(0.0)
        var safe_sss = max(Float32(0.0), min(Float32(1.0), updated.subsurface)) if isfinite(updated.subsurface) else Float32(0.0)
        var new_mo = MaterialOverrideComponent(
            updated.material_id,
            safe_col,
            clamp_r,
            clamp_m,
            safe_em,
            enabled=updated.material_override_enabled,
            opacity=safe_op,
            alpha_cutoff=safe_ac,
            clearcoat=safe_cc,
            clearcoat_roughness=safe_ccr,
            sheen=safe_sh,
            subsurface=safe_sss,
        )
        var old_mo = self.app.world.get_ptr[MaterialOverrideComponent](e)
        if old_mo:
            new_mo.material_ref = old_mo.value()[].material_ref.copy()
        _ = self.app.world.insert[MaterialOverrideComponent](e, new_mo^)

        var bvh_s = -1
        var old_meta = self.app.world.get_ptr[StudioEntityMetaComponent](e)
        if old_meta:
            bvh_s = old_meta.value()[unsafe_offset=0].bvh_slot
        _ = self.app.world.insert[StudioEntityMetaComponent](
            e,
            StudioEntityMetaComponent(
                updated.name,
                updated.locked,
                updated.mesh_handle.packed,
                updated.material_handle.packed,
                updated.collider_kind,
                updated.collider_half_extents,
                updated.restitution,
                updated.friction,
                bvh_s,
                eff_en,
            ),
        )

        var st = StudioStyleComponent()
        var old_st = self.app.world.get_ptr[StudioStyleComponent](e)
        if old_st:
            st = old_st.value()[unsafe_offset=0]
        st.clearcoat = safe_cc
        st.clearcoat_roughness = safe_ccr
        st.shading_mode = updated.shading_mode
        st.cel_shadow_tint = updated.cel_shadow_tint
        st.cel_outline_width_px = updated.cel_outline_width_px
        st.cel_outline_color = updated.cel_outline_color
        st.cel_rim_color = updated.cel_rim_color
        st.cel_hair_halo = updated.cel_hair_halo
        st.vfx_erosion = updated.vfx_erosion
        st.vfx_scroll_speed = updated.vfx_scroll_speed
        st.vfx_core_color = updated.vfx_core_color
        st.vfx_edge_color = updated.vfx_edge_color
        st.anim_step_hz = updated.anim_step_hz
        st.smear_stretch = updated.smear_stretch
        _ = self.app.world.insert[StudioStyleComponent](e, st)

        self._apply_collider_and_rb(
            e,
            updated.collider_kind,
            updated.collider_half_extents,
            updated.mass,
            updated.restitution,
            updated.friction,
            updated.linear_velocity,
            updated.angular_velocity,
            update_trigger=(updated.is_trigger != prev_trig),
            is_trigger=updated.is_trigger,
        )
        return True

    def commit_node_edit(
        mut self,
        entity_id: Int,
        var updated_view: StudioEntityView,
        label: String = "Edit Entity",
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        if not self.is_alive_id(entity_id):
            return False

        var can_coalesce = (
            coalesce_key != 0
            and len(self.undo_stack) > 0
            and len(self.redo_stack) == 0
            and self.undo_stack[len(self.undo_stack) - 1].op == TX_PATCH_ENTITY
            and self.undo_stack[len(self.undo_stack) - 1].entity_id == entity_id
            and self.undo_stack[len(self.undo_stack) - 1].coalesce_key == coalesce_key
        )
        var before_blob = List[UInt8]() if can_coalesce else self._capture_world_blob()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        if not self._apply_view_to_world(entity_id, updated_view):
            return False

        self._sync_ecs_and_physics()
        var aff = List[Int]()
        aff.append(entity_id)
        self._record_world_tx(
            TX_PATCH_ENTITY,
            entity_id,
            aff^,
            before_blob^,
            prev_ids,
            prev_primary,
            prev_anchor,
            label,
            coalesce_key=coalesce_key,
        )
        return True

    def commit_ecs_edit(
        mut self,
        entity_id: Int,
        label: String,
        var before_blob: List[UInt8],
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        """Commits direct ECS component edits on `entity_id` with full Undo/Redo and physics/compat sync."""
        if not self.is_alive_id(entity_id):
            return False
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id
        self._sync_ecs_and_physics()
        var aff = List[Int]()
        aff.append(entity_id)
        self._record_world_tx(
            TX_PATCH_ENTITY,
            entity_id,
            aff^,
            before_blob^,
            prev_ids,
            prev_primary,
            prev_anchor,
            label,
            coalesce_key=coalesce_key,
        )
        return True

    def register_pod_schema(mut self, schema: PodComponentSchema) -> Bool:
        if not self.registry.register_pod_schema(schema):
            return False
        var tid = self.app.world.register_pod_component(
            schema.component_hash, schema.byte_size, schema.alignment, schema.storage_mode
        )
        if tid < 0:
            return False
        if not self.app.resources.has[PodSchemaRegistryResource]():
            self.app.resources.insert(PodSchemaRegistryResource())
        var ps_opt = self.app.resources.get_ptr_mut[PodSchemaRegistryResource]()
        if ps_opt:
            _ = ps_opt.value()[unsafe_offset=0].register_schema(schema)
            self._synced_pod_schema_epoch = ps_opt.value()[unsafe_offset=0].epoch
        return True

    def add_component_by_name(mut self, entity_id: Int, comp_name: String, record_undo: Bool = True) -> Bool:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return False
        var cidx = self.registry.find_by_name(comp_name)
        if cidx < 0:
            return False
        if self.registry.has_on_entity(cidx, self.app.world, e):
            return False
        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        if not self.registry.insert_default_on_entity(self.app.world, e, comp_name):
            return False
        if comp_name == "ColliderComponent":
            var meta_opt = self.app.world.get_ptr_mut[StudioEntityMetaComponent](e)
            if meta_opt and meta_opt.value()[unsafe_offset=0].collider_kind == COLLIDER_NONE:
                meta_opt.value()[unsafe_offset=0].collider_kind = COLLIDER_BOX
        if record_undo:
            return self.commit_ecs_edit(entity_id, "Add " + comp_name, before_blob^)
        self._sync_ecs_and_physics()
        return True

    def remove_component_by_name(mut self, entity_id: Int, comp_name: String, record_undo: Bool = True) -> Bool:
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return False
        var cidx = self.registry.find_by_name(comp_name)
        if cidx < 0:
            return False
        if not self.registry.has_on_entity(cidx, self.app.world, e):
            return False
        var before_blob = self._capture_world_blob() if record_undo else List[UInt8]()
        if not self.registry.remove_from_entity(self.app.world, e, comp_name):
            return False
        if comp_name == "ColliderComponent":
            var meta_opt = self.app.world.get_ptr_mut[StudioEntityMetaComponent](e)
            if meta_opt:
                meta_opt.value()[unsafe_offset=0].collider_kind = COLLIDER_NONE
        if record_undo:
            return self.commit_ecs_edit(entity_id, "Remove " + comp_name, before_blob^)
        self._sync_ecs_and_physics()
        return True

    def open_input_map_panel(mut self):
        _ensure_dock_panel_tab(self.dock, "InputMap", "AssetBrowser")
        _ = self.dock.activate_tab("InputMap")

    def cancel_input_rebind(mut self):
        self.input_rebind_active = False
        self.input_rebind_context = ""
        self.input_rebind_action = ""
        self.ui_ctx.set_keyboard_capture(False)

    def begin_input_rebind(mut self, context_name: String, action_name: String) -> Bool:
        var ci = self.input_map._context_index(context_name)
        if ci < 0 or action_name.byte_length() == 0:
            return False
        var known = self.input_map._action_index(action_name) >= 0
        if not known:
            for bi in range(len(self.input_map.contexts[ci].bindings)):
                if self.input_map.contexts[ci].bindings[bi].action_name == action_name:
                    known = True
                    break
        if not known:
            return False
        self.input_rebind_active = True
        self.input_rebind_context = context_name
        self.input_rebind_action = action_name
        self.ui_ctx.set_keyboard_capture(True)
        return True

    def _resolve_input_map_path(self, path: String = "") -> String:
        if path.byte_length() > 0:
            return path
        if self.project_root.byte_length() > 0:
            var root = self.project_root
            while root.byte_length() > 1 and root.endswith("/"):
                var trimmed = String(root[byte=0 : root.byte_length() - 1])
                root = trimmed^
            return root + "/config/input_map.json"
        return state_directory("flint") + "/input_map.json"

    def save_input_map_file(mut self, path: String = "") -> Bool:
        var target = self._resolve_input_map_path(path)
        var pdir = parent_directory(target)
        if pdir.byte_length() > 0:
            _ = make_dirs(pdir)
        var json_text = self.input_map.save_json()
        var sb = json_text.as_bytes()
        var out = List[UInt8](capacity=len(sb))
        for i in range(len(sb)):
            out.append(sb[i])
        var ok = write_file_durable(target, out)
        if ok:
            self.push_toast("Saved InputMap", 0, 2.0)
        else:
            self.push_toast("Failed saving InputMap", 2, 2.5)
        return ok

    def load_input_map_file(mut self, path: String = "") -> Bool:
        var target = self._resolve_input_map_path(path)
        if file_size(target) <= 0:
            self.push_toast("InputMap file not found", 1, 2.0)
            return False
        var raw = read_bytes(target)
        if len(raw) == 0:
            self.push_toast("InputMap file empty", 2, 2.0)
            return False
        var json_text = String(unsafe_from_utf8=raw)
        var ok = self.input_map.load_json(json_text)
        if ok:
            self.cancel_input_rebind()
            self.push_toast("Loaded InputMap", 0, 2.0)
        else:
            self.push_toast("Invalid InputMap JSON", 2, 2.5)
        return ok

    def set_entity_transform(
        mut self,
        entity_id: Int,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        if not self.is_alive_id(entity_id):
            return False
        var v = self.entities[entity_id].copy()
        v.translation = position
        v.rotation = rotation.normalized()
        v.scale = scale
        return self.commit_node_edit(entity_id, v^, "Transform", coalesce_key)

    def patch_entity_field(
        mut self,
        entity_id: Int,
        field_name: String,
        new_val: InspectedField,
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        if not self.is_alive_id(entity_id):
            return False
        var v = self.entities[entity_id].copy()
        var vis = FieldVisitor.for_patch(field_name, new_val)
        v.inspect_fields(vis)
        if not vis.mutated:
            return False
        if field_name == "parent_id":
            if v.parent_id >= 0 and (not self.is_alive_id(v.parent_id) or v.parent_id == entity_id or self._is_ancestor(entity_id, v.parent_id)):
                return False
        return self.commit_node_edit(entity_id, v^, "Patch " + field_name, coalesce_key)

    # ---- Viewport gizmo drag controller (host feeds mouse in physical pixels) ----

    def gizmo_pick(self, mx: Float32, my: Float32) -> Int:
        """The gizmo handle under the mouse, using this frame's `viewport_xf`."""
        var piv = self.selection_gizmo_pivot()
        if not piv[0] or self.pie.mode != PIE_STOPPED or self.gizmo_mode == GIZMO_MODE_SELECT:
            return GIZMO_AXIS_NONE
        var vx = self.viewport_xf
        var s = vx.ui_scale
        return pick_gizmo_handle(
            mx, my, piv[1], self.gizmo_mode, Float32(0.0), Float32(0.0), vx.proj_phys_w(), vx.proj_phys_h(),
            self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov_y_deg,
            target_px=GIZMO_TARGET_PX * s, gizmo_rot=self.selection_gizmo_rotation(), px_scale=s,
        )

    def gizmo_try_begin(mut self, mx: Float32, my: Float32) -> Bool:
        """Starts a gizmo drag if the press hit a handle. The drag is one undo step (coalesced) and can be
        cancelled with `gizmo_drag_cancel`, which puts every selected object back where it started."""
        var h = self.gizmo_pick(mx, my)
        if h == GIZMO_AXIS_NONE:
            return False
        var piv = self.selection_gizmo_pivot()
        self.gz_dragging = True
        self.gz_handle = h
        self.gz_mode = self.gizmo_mode
        self.gz_seq_counter += 1
        self.gz_seq = GIZMO_DRAG_SEQ_TAG | self.gz_seq_counter
        self.gz_start_ray = self.viewport_xf.ray_phys(mx, my, self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov_y_deg)
        self.gz_prev_ray = self.gz_start_ray
        self.gz_start_pivot = piv[1]
        self.gz_rot = self.selection_gizmo_rotation()
        self.gz_snaps = self.capture_selected_transforms()
        self.gz_angle = Float32(0.0)
        self.gz_start_mx = mx
        self.gz_start_my = my
        self.active_gizmo_handle = h
        # The drag's first update would clear redo; park it so a cancelled drag can give it back.
        self.gz_saved_redo.clear()
        swap(self.gz_saved_redo, self.redo_stack)
        return True

    def gizmo_drag_update(mut self, mx: Float32, my: Float32) -> Bool:
        if not self.gz_dragging or len(self.gz_snaps) == 0:
            return False
        if self.gizmo_mode != self.gz_mode or self.pie.mode != PIE_STOPPED:
            # Switching W/E/R (or entering Play) mid-drag abandons the drag instead of mixing modes.
            self.gizmo_drag_cancel()
            return False
        var snaps = self.gz_snaps.copy()
        var vx = self.viewport_xf
        var cur = vx.ray_phys(mx, my, self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov_y_deg)
        var piv = self.gz_start_pivot
        var rot = self.gz_rot
        var h = self.gz_handle
        var wscale = gizmo_world_scale(piv, self.cam_pos, self.cam_fwd, self.fov_y_deg, Int(vx.proj_phys_h()), GIZMO_TARGET_PX * vx.ui_scale)
        if self.gz_mode == GIZMO_MODE_TRANSLATE:
            var delta_p = Vec3()
            if h == GIZMO_AXIS_X or h == GIZMO_AXIS_Y or h == GIZMO_AXIS_Z:
                var ax = rot.rotate(Vec3(1, 0, 0) if h == GIZMO_AXIS_X else (Vec3(0, 1, 0) if h == GIZMO_AXIS_Y else Vec3(0, 0, 1)))
                var t0 = ray_axis_closest_param(self.gz_start_ray, piv, ax)
                var t1 = ray_axis_closest_param(cur, piv, ax)
                if t0[0] and t1[0]:
                    delta_p = ax * (t1[1] - t0[1])
            elif h == GIZMO_PLANE_XY or h == GIZMO_PLANE_XZ or h == GIZMO_PLANE_YZ:
                var pn = rot.rotate(Vec3(0, 0, 1) if h == GIZMO_PLANE_XY else (Vec3(0, 1, 0) if h == GIZMO_PLANE_XZ else Vec3(1, 0, 0)))
                var p0 = ray_plane_hit(self.gz_start_ray, piv, pn)
                var p1 = ray_plane_hit(cur, piv, pn)
                if p0[0] and p1[0]:
                    delta_p = p1[1] - p0[1]
            if self.snap_enabled:
                delta_p = snap_constrained_delta(piv, delta_p, h, self.grid_snap_m, rot)
            return self.apply_gizmo_drag_selected(
                snaps, piv, delta_p, Quat.identity(), Vec3(1.0, 1.0, 1.0), GIZMO_MODE_TRANSLATE,
                coalesce_key=self.gz_seq, gizmo_rot=rot,
            )
        if self.gz_mode == GIZMO_MODE_ROTATE:
            var rax: Vec3
            if h == GIZMO_AXIS_VIEW:
                rax = self.cam_fwd * Float32(-1.0)
            else:
                rax = rot.rotate(Vec3(1, 0, 0) if h == GIZMO_AXIS_X else (Vec3(0, 1, 0) if h == GIZMO_AXIS_Y else Vec3(0, 0, 1)))
            # Accumulate frame-to-frame deltas: a single start-to-current angle wraps at +-180 degrees.
            self.gz_angle += gizmo_rotation_delta_rad(self.gz_prev_ray, cur, piv, rax, self.cam_fwd, wscale)
            self.gz_prev_ray = cur
            var drad = self.gz_angle
            if self.snap_enabled:
                drad = snap_angle_deg(drad, self.angle_snap_deg)
            var hs = sin(drad * Float32(0.5))
            var dq = Quat(rax.x * hs, rax.y * hs, rax.z * hs, cos(drad * Float32(0.5)))
            return self.apply_gizmo_drag_selected(
                snaps, piv, Vec3(), dq, Vec3(1.0, 1.0, 1.0), GIZMO_MODE_ROTATE,
                coalesce_key=self.gz_seq, gizmo_rot=rot,
            )
        # Scale
        var sf: Vec3
        if h == GIZMO_SCALE_UNIFORM:
            var c = project_world_to_viewport(
                piv, Float32(0.0), Float32(0.0), vx.proj_phys_w(), vx.proj_phys_h(),
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov_y_deg,
            )
            var f = gizmo_uniform_scale_screen(self.gz_start_mx, self.gz_start_my, mx, my, c[0], c[1]) if c[3] else Float32(1.0)
            sf = Vec3(f, f, f)
        else:
            sf = gizmo_scale_drag(
                self.gz_start_ray, cur, piv, h, wscale, self.cam_fwd, self.cam_right, Vec3(1.0, 1.0, 1.0), gizmo_rot=rot
            )
        if self.snap_enabled and self.scale_snap > Float32(0.0):
            sf = Vec3(
                max(self.scale_snap, snap_scalar(sf.x, self.scale_snap)),
                max(self.scale_snap, snap_scalar(sf.y, self.scale_snap)),
                max(self.scale_snap, snap_scalar(sf.z, self.scale_snap)),
            )
        return self.apply_gizmo_drag_selected(
            snaps, piv, Vec3(), Quat.identity(), sf, GIZMO_MODE_SCALE, coalesce_key=self.gz_seq, gizmo_rot=rot
        )

    def gizmo_drag_cancel(mut self):
        """Esc / right-click / mode switch during a drag: everything goes back to where it was and the
        drag leaves no undo entry behind."""
        if not self.gz_dragging:
            return
        var restored = False
        var had_tx = False
        if len(self.undo_stack) > 0 and len(self.redo_stack) == 0:
            ref top = self.undo_stack[len(self.undo_stack) - 1]
            if top.coalesce_key == self.gz_seq and top.op == TX_BATCH_EDIT:
                had_tx = True
                var tx = self.undo_stack.pop()
                restored = self._restore_world_blob(tx.before_world_blob, tx.external_mat_reload_epochs)
                if not restored:
                    self.undo_stack.append(tx^)
                else:
                    self._restore_selection(tx.prev_selected_ids, tx.prev_selected_id, tx.prev_anchor_id)
                    self._refresh_dirty_from_history()
        if not restored and len(self.gz_snaps) > 0:
            # Nothing coalesced (no motion yet, or Play swallowed the entry): re-apply the start pose.
            var start_snaps = self.gz_snaps.copy()
            var start_piv = self.gz_start_pivot
            _ = self.apply_gizmo_drag_selected(
                start_snaps, start_piv, Vec3(), Quat.identity(), Vec3(1.0, 1.0, 1.0), GIZMO_MODE_TRANSLATE,
                coalesce_key=self.gz_seq,
            )
            if (
                not had_tx
                and len(self.undo_stack) > 0
                and self.undo_stack[len(self.undo_stack) - 1].coalesce_key == self.gz_seq
            ):
                _ = self.undo_stack.pop()  # the start pose re-applied is no edit: leave no entry
                self._refresh_dirty_from_history()
        if restored or not had_tx:
            # The drag left no entry behind: the redo history it parked is valid again.
            swap(self.gz_saved_redo, self.redo_stack)
        self.gz_saved_redo.clear()
        self._end_gizmo_drag_state()

    def gizmo_drag_end(mut self):
        if self.gz_dragging and len(self.redo_stack) == 0:
            var moved = (
                len(self.undo_stack) > 0
                and self.undo_stack[len(self.undo_stack) - 1].coalesce_key == self.gz_seq
            )
            if not moved:
                swap(self.gz_saved_redo, self.redo_stack)  # a click without motion keeps redo
        self.gz_saved_redo.clear()
        self._end_gizmo_drag_state()

    def _end_gizmo_drag_state(mut self):
        self.gz_dragging = False
        self.gz_handle = GIZMO_AXIS_NONE
        self.gz_snaps.clear()
        self.active_gizmo_handle = GIZMO_AXIS_NONE

    # ---- Marquee selection, nudging and framing ----

    def entities_in_screen_rect(self, x0: Float32, y0: Float32, x1: Float32, y1: Float32) -> List[Int]:
        """Entities whose origin projects inside the rect (physical pixels) and inside the viewport panel."""
        var out = List[Int]()
        var vx = self.viewport_xf
        var pnl = vx.panel_phys()
        var lx = max(min(x0, x1), pnl.x)
        var hx = min(max(x0, x1), pnl.x + pnl.w)
        var ly = max(min(y0, y1), pnl.y)
        var hy = min(max(y0, y1), pnl.y + pnl.h)
        if hx <= lx or hy <= ly:
            return out^
        for i in range(len(self.entities)):
            if not self.is_alive_id(i) or self.entities[i].locked:
                continue
            var p = self.entities[i].translation
            var gt = self.app.world.get_ptr[GlobalTransform](self.resolve_entity(i))
            if gt:
                p = gt.value()[unsafe_offset=0].translation
            var sp = project_world_to_viewport(
                p, Float32(0.0), Float32(0.0), vx.proj_phys_w(), vx.proj_phys_h(),
                self.cam_pos, self.cam_fwd, self.cam_right, self.cam_up, self.fov_y_deg,
            )
            if sp[3] and sp[0] >= lx and sp[0] <= hx and sp[1] >= ly and sp[1] <= hy:
                out.append(i)
        return out^

    def marquee_select(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32, additive: Bool) -> Int:
        var hits = self.entities_in_screen_rect(x0, y0, x1, y1)
        if not additive:
            self.clear_selection()
        for k in range(len(hits)):
            self.select_entity(hits[k], multi_select=True)
        return len(hits)

    def nudge_selection(mut self, dir_x: Int, dir_y: Int, dir_z: Int) -> Bool:
        """Arrow-key nudge by one grid step. X/Z follow the camera's heading (snapped to the nearest
        world axis) so "left" on screen moves left; PageUp/PageDown move along world Y."""
        var snaps = self.capture_selected_transforms()
        if len(snaps) == 0:
            return False
        var step = self.grid_snap_m if self.grid_snap_m > Float32(1.0e-4) else Float32(0.25)
        var right = Vec3(self.cam_right.x, 0.0, self.cam_right.z)
        var fwd = Vec3(self.cam_fwd.x, 0.0, self.cam_fwd.z)
        var one = Float32(1.0)
        var zero = Float32(0.0)
        var r_ax = Vec3(one if right.x >= zero else -one, zero, zero) if abs(right.x) >= abs(right.z) else Vec3(zero, zero, one if right.z >= zero else -one)
        var f_ax = Vec3(one if fwd.x >= zero else -one, zero, zero) if abs(fwd.x) > abs(fwd.z) else Vec3(zero, zero, one if fwd.z >= zero else -one)
        var d = (r_ax * Float32(dir_x) + Vec3(0.0, Float32(dir_y), 0.0) + f_ax * Float32(dir_z)) * step
        var piv = self.selection_gizmo_pivot()
        return self.apply_gizmo_drag_selected(snaps, piv[1], d, Quat.identity(), Vec3(1.0, 1.0, 1.0), GIZMO_MODE_TRANSLATE)

    def selection_bounds(self) -> Tuple[Bool, Vec3, Vec3]:
        """World AABB of the selection from each entity's position and scaled collider extents."""
        var lo = Vec3(Float32(1.0e30), Float32(1.0e30), Float32(1.0e30))
        var hi = Vec3(Float32(-1.0e30), Float32(-1.0e30), Float32(-1.0e30))
        var any = False
        var ids = self.selected_entity_ids.copy()
        if len(ids) == 0 and self.is_alive_id(self.selected_entity_id):
            ids.append(self.selected_entity_id)
        for k in range(len(ids)):
            var i = ids[k]
            if not self.is_alive_id(i):
                continue
            var p = self.entities[i].translation
            var gt = self.app.world.get_ptr[GlobalTransform](self.resolve_entity(i))
            if gt:
                p = gt.value()[unsafe_offset=0].translation
            var he = self.entities[i].collider_half_extents
            var sc = self.entities[i].scale
            var ext = Vec3(abs(he.x * sc.x), abs(he.y * sc.y), abs(he.z * sc.z))
            var r = max(Float32(0.25), max(ext.x, max(ext.y, ext.z)))
            lo = Vec3(min(lo.x, p.x - r), min(lo.y, p.y - r), min(lo.z, p.z - r))
            hi = Vec3(max(hi.x, p.x + r), max(hi.y, p.y + r), max(hi.z, p.z + r))
            any = True
        return (any, lo, hi)

    def toggle_ortho_camera(mut self):
        """The renderer has no orthographic path yet: say so instead of flipping a flag that changes
        nothing on screen."""
        self.cam_ortho = False
        self.push_toast("Orthographic view is not supported by the renderer yet", 1, 3.0)

    def capture_selected_transforms(self) -> List[SelectedTransformSnap]:
        var top_ids = self._top_level_selected_ids()
        var snaps = List[SelectedTransformSnap](capacity=len(top_ids))
        for i in range(len(top_ids)):
            var eid = top_ids[i]
            var e = self.resolve_entity(eid)
            if not e.valid() or self.is_entity_locked(eid):
                continue  # locked entities never move with the gizmo or nudges
            var lt = self.app.world.get[LocalTransform](e) if self.app.world.has[LocalTransform](e) else LocalTransform()
            var gt = self.app.world.get[GlobalTransform](e) if self.app.world.has[GlobalTransform](e) else GlobalTransform.from_local(lt)
            snaps.append(
                SelectedTransformSnap(
                    eid,
                    lt.translation,
                    lt.rotation,
                    lt.scale,
                    gt.translation,
                    gt.rotation,
                    gt.scale,
                    self.parent_of(eid),
                )
            )
        return snaps^

    def apply_gizmo_drag_selected(
        mut self,
        snaps: List[SelectedTransformSnap],
        pivot: Vec3,
        world_delta_pos: Vec3,
        world_dq_rot: Quat,
        scale_factor: Vec3,
        mode: Int,
        coalesce_key: UInt64 = 0,
        gizmo_rot: Quat = Quat.identity(),
    ) -> Bool:
        if len(snaps) == 0:
            return False
        var g_rot = gizmo_rot.normalized() if gizmo_rot.length_squared() > Float32(1.0e-8) else Quat.identity()
        var g_inv = g_rot.conjugate()

        var primary_id = self.selected_entity_id if self.selected_entity_id >= 0 else snaps[0].entity_id
        var can_coalesce = (
            coalesce_key != 0
            and len(self.undo_stack) > 0
            and len(self.redo_stack) == 0
            and self.undo_stack[len(self.undo_stack) - 1].op == TX_BATCH_EDIT
            and self.undo_stack[len(self.undo_stack) - 1].entity_id == primary_id
            and self.undo_stack[len(self.undo_stack) - 1].coalesce_key == coalesce_key
        )
        var before_blob = List[UInt8]() if can_coalesce else self._capture_world_blob()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        var aff = List[Int](capacity=len(snaps))
        for i in range(len(snaps)):
            ref sn = snaps[i]
            var e = self.resolve_entity(sn.entity_id)
            if not e.valid():
                continue
            aff.append(sn.entity_id)

            var target_w_pos = sn.world_pos
            var target_w_rot = sn.world_rot
            var target_local_scl = sn.local_scl
            if mode == GIZMO_MODE_TRANSLATE:
                if isfinite(world_delta_pos.x) and isfinite(world_delta_pos.y) and isfinite(world_delta_pos.z):
                    target_w_pos = sn.world_pos + world_delta_pos
            elif mode == GIZMO_MODE_ROTATE:
                if (
                    isfinite(pivot.x) and isfinite(pivot.y) and isfinite(pivot.z)
                    and isfinite(world_dq_rot.x) and isfinite(world_dq_rot.y)
                    and isfinite(world_dq_rot.z) and isfinite(world_dq_rot.w)
                ):
                    var rel = sn.world_pos - pivot
                    target_w_pos = pivot + world_dq_rot.rotate(rel)
                    target_w_rot = world_dq_rot.multiply(sn.world_rot).normalized()
            elif mode == GIZMO_MODE_SCALE:
                var sx_f = scale_factor.x if isfinite(scale_factor.x) else Float32(1.0)
                var sy_f = scale_factor.y if isfinite(scale_factor.y) else Float32(1.0)
                var sz_f = scale_factor.z if isfinite(scale_factor.z) else Float32(1.0)
                var lx = sn.local_scl.x * sx_f
                var ly = sn.local_scl.y * sy_f
                var lz = sn.local_scl.z * sz_f
                target_local_scl = Vec3(
                    max(Float32(0.01), lx) if isfinite(lx) else max(Float32(0.01), sn.local_scl.x),
                    max(Float32(0.01), ly) if isfinite(ly) else max(Float32(0.01), sn.local_scl.y),
                    max(Float32(0.01), lz) if isfinite(lz) else max(Float32(0.01), sn.local_scl.z),
                )
                if len(snaps) > 1 and isfinite(pivot.x) and isfinite(pivot.y) and isfinite(pivot.z):
                    # A multi-selection scales as a group: positions spread from the pivot along the
                    # gizmo axes, instead of every object growing in place.
                    var rel_g = g_inv.rotate(sn.world_pos - pivot)
                    var moved = pivot + g_rot.rotate(Vec3(rel_g.x * sx_f, rel_g.y * sy_f, rel_g.z * sz_f))
                    if isfinite(moved.x) and isfinite(moved.y) and isfinite(moved.z):
                        target_w_pos = moved

            var new_lt = LocalTransform(target_w_pos, target_w_rot, target_local_scl)
            if sn.parent_id >= 0:
                var pe = self.resolve_entity(sn.parent_id)
                if pe.valid():
                    var pgt_opt = self.app.world.get_ptr[GlobalTransform](pe)
                    if pgt_opt:
                        ref pgt = pgt_opt.value()[unsafe_offset=0]
                        var inv_q = pgt.rotation.conjugate()
                        var lp = inv_q.rotate(target_w_pos - pgt.translation)
                        var sx = pgt.scale.x if abs(pgt.scale.x) > Float32(1.0e-6) else Float32(1.0)
                        var sy = pgt.scale.y if abs(pgt.scale.y) > Float32(1.0e-6) else Float32(1.0)
                        var sz = pgt.scale.z if abs(pgt.scale.z) > Float32(1.0e-6) else Float32(1.0)
                        new_lt = LocalTransform(
                            Vec3(lp.x / sx, lp.y / sy, lp.z / sz),
                            (inv_q * target_w_rot).normalized(),
                            target_local_scl,
                        )
            _ = self.app.world.insert[LocalTransform](e, new_lt)

        if len(aff) == 0:
            return False

        self._sync_ecs_and_physics()
        self._record_world_tx(
            TX_BATCH_EDIT,
            primary_id,
            aff^,
            before_blob^,
            prev_ids,
            prev_primary,
            prev_anchor,
            "Transform Selected",
            coalesce_key=coalesce_key,
        )
        return True

    def bulk_translate_selected(mut self, delta: Vec3, coalesce_key: UInt64 = 0) -> Bool:
        var snaps = self.capture_selected_transforms()
        if len(snaps) == 0:
            return False
        var piv = self.selection_gizmo_pivot()
        return self.apply_gizmo_drag_selected(
            snaps,
            piv[1],
            delta,
            Quat(0.0, 0.0, 0.0, 1.0),
            Vec3(1.0, 1.0, 1.0),
            GIZMO_MODE_TRANSLATE,
            coalesce_key=coalesce_key,
        )

    def bulk_patch_field(
        mut self,
        field_name: String,
        new_val: InspectedField,
        coalesce_key: UInt64 = 0,
    ) -> Int:
        self._normalize_selection_from_primary()
        var targets = List[Int]()
        for i in range(len(self.selected_entity_ids)):
            if self.is_alive_id(self.selected_entity_ids[i]):
                targets.append(self.selected_entity_ids[i])
        if len(targets) == 0:
            return 0

        var primary_id = self.selected_entity_id if self.selected_entity_id >= 0 else targets[0]
        var can_coalesce = (
            coalesce_key != 0
            and len(self.undo_stack) > 0
            and len(self.redo_stack) == 0
            and self.undo_stack[len(self.undo_stack) - 1].op == TX_BATCH_EDIT
            and self.undo_stack[len(self.undo_stack) - 1].entity_id == primary_id
            and self.undo_stack[len(self.undo_stack) - 1].coalesce_key == coalesce_key
        )
        var before_blob = List[UInt8]() if can_coalesce else self._capture_world_blob()
        var prev_ids = self.selected_entity_ids.copy()
        var prev_primary = self.selected_entity_id
        var prev_anchor = self.selection_anchor_id

        var patched = 0
        for i in range(len(targets)):
            var eid = targets[i]
            var v = self.entities[eid].copy()
            var vis = FieldVisitor.for_patch(field_name, new_val)
            v.inspect_fields(vis)
            if vis.mutated and self._apply_view_to_world(eid, v):
                patched += 1

        if patched > 0:
            self._sync_ecs_and_physics()
            self._record_world_tx(
                TX_BATCH_EDIT,
                primary_id,
                targets^,
                before_blob^,
                prev_ids,
                prev_primary,
                prev_anchor,
                "Bulk Patch " + field_name,
                coalesce_key=coalesce_key,
            )
        return patched

    def configure_entity_npr_cel(
        mut self,
        entity_id: Int,
        cel_shadow_tint: Vec3 = Vec3(0.55, 0.38, 0.44),
        cel_outline_width_px: Float32 = 2.0,
        cel_outline_color: Vec3 = Vec3(0.015, 0.012, 0.018),
        cel_rim_color: Vec3 = Vec3(0.45, 0.52, 0.68),
        cel_hair_halo: Float32 = 0.0,
        anim_step_hz: Float32 = 0.0,
        smear_stretch: Float32 = 1.0,
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        if not self.is_alive_id(entity_id):
            return False
        var v = self.entities[entity_id].copy()
        v.shading_mode = SHADING_NPR_CEL
        v.cel_shadow_tint = Vec3(
            cel_shadow_tint.x if isfinite(cel_shadow_tint.x) else Float32(0.55),
            cel_shadow_tint.y if isfinite(cel_shadow_tint.y) else Float32(0.38),
            cel_shadow_tint.z if isfinite(cel_shadow_tint.z) else Float32(0.44),
        )
        var ow = cel_outline_width_px if isfinite(cel_outline_width_px) else Float32(2.0)
        v.cel_outline_width_px = max(Float32(0.0), min(Float32(24.0), ow))
        v.cel_outline_color = Vec3(
            cel_outline_color.x if isfinite(cel_outline_color.x) else Float32(0.015),
            cel_outline_color.y if isfinite(cel_outline_color.y) else Float32(0.012),
            cel_outline_color.z if isfinite(cel_outline_color.z) else Float32(0.018),
        )
        v.cel_rim_color = Vec3(
            cel_rim_color.x if isfinite(cel_rim_color.x) else Float32(0.45),
            cel_rim_color.y if isfinite(cel_rim_color.y) else Float32(0.52),
            cel_rim_color.z if isfinite(cel_rim_color.z) else Float32(0.68),
        )
        var hh = cel_hair_halo if isfinite(cel_hair_halo) else Float32(0.0)
        var hz = anim_step_hz if isfinite(anim_step_hz) else Float32(0.0)
        var sm = smear_stretch if isfinite(smear_stretch) else Float32(1.0)
        v.cel_hair_halo = max(Float32(0.0), hh)
        v.anim_step_hz = max(Float32(0.0), hz)
        v.smear_stretch = max(Float32(0.55), min(Float32(3.2), sm))
        return self.commit_node_edit(entity_id, v^, "Configure NPR Cel", coalesce_key)

    def configure_entity_stylized_vfx(
        mut self,
        entity_id: Int,
        vfx_erosion: Float32 = 0.35,
        vfx_scroll_speed: Float32 = 3.2,
        vfx_core_color: Vec3 = Vec3(8.5, 6.2, 2.2),
        vfx_edge_color: Vec3 = Vec3(4.2, 0.30, 0.04),
        anim_step_hz: Float32 = 0.0,
        smear_stretch: Float32 = 1.0,
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        if not self.is_alive_id(entity_id):
            return False
        var v = self.entities[entity_id].copy()
        v.shading_mode = SHADING_STYLIZED_VFX
        var er = vfx_erosion if isfinite(vfx_erosion) else Float32(0.35)
        v.vfx_erosion = max(Float32(0.0), min(Float32(1.0), er))
        v.vfx_scroll_speed = vfx_scroll_speed if isfinite(vfx_scroll_speed) else Float32(3.2)
        v.vfx_core_color = Vec3(
            vfx_core_color.x if isfinite(vfx_core_color.x) else Float32(8.5),
            vfx_core_color.y if isfinite(vfx_core_color.y) else Float32(6.2),
            vfx_core_color.z if isfinite(vfx_core_color.z) else Float32(2.2),
        )
        v.vfx_edge_color = Vec3(
            vfx_edge_color.x if isfinite(vfx_edge_color.x) else Float32(4.2),
            vfx_edge_color.y if isfinite(vfx_edge_color.y) else Float32(0.30),
            vfx_edge_color.z if isfinite(vfx_edge_color.z) else Float32(0.04),
        )
        var hz = anim_step_hz if isfinite(anim_step_hz) else Float32(0.0)
        var sm = smear_stretch if isfinite(smear_stretch) else Float32(1.0)
        v.anim_step_hz = max(Float32(0.0), hz)
        v.smear_stretch = max(Float32(0.55), min(Float32(3.2), sm))
        return self.commit_node_edit(entity_id, v^, "Configure Stylized VFX", coalesce_key)

    def attach_static_bvh_collider(mut self, entity_id: Int, var bvh: Bvh8MeshCollider):
        var e = self.resolve_entity(entity_id)
        if not e.valid():
            return
        var slot = len(self.static_bvh_colliders)
        self.static_bvh_colliders.append(bvh^)
        var meta_opt = self.app.world.get_ptr_mut[StudioEntityMetaComponent](e)
        if meta_opt:
            meta_opt.value()[unsafe_offset=0].collider_kind = COLLIDER_STATIC_MESH
            meta_opt.value()[unsafe_offset=0].bvh_slot = slot
        self._sync_compat_view()

    def can_undo(self) -> Bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> Bool:
        return len(self.redo_stack) > 0

    def undo(mut self) -> Bool:
        if self.pie.mode != PIE_STOPPED or len(self.undo_stack) == 0:
            return False
        if self._time_travel_blocks_edit():
            return False
        self._int_edit_wid = UInt64(0)
        self._int_edit_entity_id = -1
        self._int_edit_buf = ""
        var tx = self.undo_stack.pop()
        if not self._restore_world_blob(tx.before_world_blob, tx.external_mat_reload_epochs):
            self.undo_stack.append(tx^)
            return False
        self._restore_selection(tx.prev_selected_ids, tx.prev_selected_id, tx.prev_anchor_id)
        if tx.changes_scene_path:
            self.scene_file_path = tx.prev_scene_path
            self._saved_tx_id = tx.prev_saved_tx_id  # clean again only if it was clean before the load
        self.redo_stack.append(tx^)
        self._refresh_dirty_from_history()
        return True

    def redo(mut self) -> Bool:
        if self.pie.mode != PIE_STOPPED or len(self.redo_stack) == 0:
            return False
        if self._time_travel_blocks_edit():
            return False
        self._int_edit_wid = UInt64(0)
        self._int_edit_entity_id = -1
        self._int_edit_buf = ""
        var tx = self.redo_stack.pop()
        if not self._restore_world_blob(tx.after_world_blob, tx.external_mat_reload_epochs):
            self.redo_stack.append(tx^)
            return False
        self._restore_selection(tx.next_selected_ids, tx.next_selected_id, tx.next_anchor_id)
        var redo_clean = tx.changes_scene_path and tx.load_leaves_clean
        if tx.changes_scene_path:
            self.scene_file_path = tx.next_scene_path
        # Redo keeps the entry's original origin: whoever triggers the redo, it is still the edit of
        # whoever made it (an agent's MCP redo must not relabel the user's work as the agent's).
        var keep_agent = self._active_tx_agent_id
        var keep_step = self._active_tx_step_id
        self._active_tx_agent_id = 0
        self._active_tx_step_id = UInt64(0)
        self._push_undo(tx^)
        self._active_tx_agent_id = keep_agent
        self._active_tx_step_id = keep_step
        if redo_clean:
            self._saved_tx_id = self._top_tx_id()  # re-applying a load matches the loaded file again
        self._refresh_dirty_from_history()
        return True

    # ---- Deterministic `PhysicsWorld` Step, Editor Frame Render & Analytical/Physics Raycast ----

    def _advance_frame_time(mut self, dt: Float32, clamp_max: Bool = True) -> Float32:
        var safe_dt = dt if isfinite(dt) else Float32(0.0)
        var eff_dt = max(Float32(0.0), safe_dt)
        if clamp_max:
            var safe_max = (
                self.app.config.max_frame_dt
                if isfinite(self.app.config.max_frame_dt)
                else Float32(0.1)
            )
            eff_dt = min(safe_max, eff_dt)
        var ft_opt = self.app.resources.get_ptr_mut[FrameTime]()
        if ft_opt:
            var ft_ptr = ft_opt.value()
            ft_ptr[unsafe_offset=0].delta_seconds = eff_dt
            ft_ptr[unsafe_offset=0].elapsed_seconds += Float64(eff_dt)
            ft_ptr[unsafe_offset=0].frame_index += 1
        return eff_dt

    def _capture_pie_trigger_events(mut self):
        var tev_opt = self.app.resources.get_ptr_mut[TriggerEventsResource]()
        if not tev_opt:
            return
        ref tev = tev_opt.value()[unsafe_offset=0]
        if self.pie.mode == PIE_STOPPED:
            self._pie_trigger_drain_cursor = tev.events.total_sent
            return
        var evs = tev.events.read_since(self._pie_trigger_drain_cursor)
        self._pie_trigger_drain_cursor = tev.events.total_sent
        var sync_opt = self.app.resources.get_ptr[PhysicsSyncStateResource]()
        for i in range(len(evs)):
            ref ev = evs[i]
            var trig_eid = -1
            var body_eid = -1
            if sync_opt:
                ref sync = sync_opt.value()[unsafe_offset=0]
                var te = sync.lookup_trigger_entity(ev.trigger_id)
                if self.app.world.is_alive(te):
                    trig_eid = Int(te.index)
                var be = sync.lookup_entity(ev.body_id)
                if self.app.world.is_alive(be):
                    body_eid = Int(be.index)
            self._pie_trigger_buffer.push(
                BufferedTriggerEvent(ev.kind, ev.trigger_id, ev.body_id, trig_eid, body_eid, ev.point)
            )

    def _reset_pie_trigger_buffer(mut self):
        self._pie_trigger_buffer.reset()
        var tev_opt = self.app.resources.get_ptr[TriggerEventsResource]()
        self._pie_trigger_drain_cursor = tev_opt.value()[unsafe_offset=0].events.total_sent if tev_opt else 0

    def _invoke_event_swap_hooks(mut self):
        self._capture_pie_trigger_events()
        var res_ptr = Pointer[EngineApp, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=self.app)))
        var r_ptr = Pointer(to=res_ptr[unsafe_offset=0].resources)
        for i in range(len(self.app.event_swap_hooks)):
            var hook = self.app.event_swap_hooks[i]
            hook(r_ptr)

    def step_simulation(
        mut self,
        ticks: Int = 1,
        dt: Float32 = Float32(1.0 / 60.0),
        step_gameplay_plugin: Bool = False,
    ) -> Int:
        """Advances `ticks` deterministic simulation steps (`FrameTime`, `STAGE_PRE_UPDATE`,
        `STAGE_FIXED_UPDATE`, `STAGE_UPDATE`, `STAGE_POST_UPDATE`, and `event_swap_hooks`)."""
        if ticks <= 0:
            return 0
        set_sim_float_mode()
        self._ensure_ground_plane()
        var safe_dt = max(Float32(0.0), dt) if isfinite(dt) else Float32(0.0)

        for _ in range(ticks):
            _ = self._advance_frame_time(safe_dt, clamp_max=False)
            self.app._observe_tick_sync()

            self.app.schedule.run_stage(
                STAGE_PRE_UPDATE,
                self.app.world,
                self.app.resources,
                self.app.pool,
                self.app.commands,
            )
            self.app.world.flush_commands(self.app.commands)
            self.app._observe_tick_sync()

            var fix_opt = self.app.resources.get_ptr_mut[FixedTime]()
            var saved_dt = Float32(1.0 / 60.0)
            var saved_dt_f64 = Float64(1.0 / 60.0)
            if fix_opt:
                saved_dt = fix_opt.value()[unsafe_offset=0].dt
                saved_dt_f64 = fix_opt.value()[unsafe_offset=0].dt_f64
                fix_opt.value()[unsafe_offset=0].dt = safe_dt
                fix_opt.value()[unsafe_offset=0].dt_f64 = Float64(safe_dt)
                fix_opt.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)

            _ = self.app.world.advance_tick()
            self.app.schedule.run_stage(
                STAGE_FIXED_UPDATE,
                self.app.world,
                self.app.resources,
                self.app.pool,
                self.app.commands,
            )
            self.app.world.flush_commands(self.app.commands)
            self.app._observe_tick_sync()
            self.sim_tick += 1

            var fix_post = self.app.resources.get_ptr_mut[FixedTime]()
            if fix_post:
                fix_post.value()[unsafe_offset=0].tick_count += 1
                fix_post.value()[unsafe_offset=0].dt = saved_dt
                fix_post.value()[unsafe_offset=0].dt_f64 = saved_dt_f64
                fix_post.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)

            self.app.schedule.run_stage(
                STAGE_UPDATE,
                self.app.world,
                self.app.resources,
                self.app.pool,
                self.app.commands,
            )
            self.app.world.flush_commands(self.app.commands)
            self.app._observe_tick_sync()

            if (
                self.pie.mode == PIE_PLAYING
                or (step_gameplay_plugin and self.pie.mode == PIE_PAUSED)
            ) and self.hot_reload_host.is_loaded():
                _ = self.hot_reload_host.step(self.app.world, self.app.resources, safe_dt)
                self._sync_registered_pod_schemas()
                self.app.world.flush_commands(self.app.commands)
                self.app._observe_tick_sync()

            self.app.schedule.run_stage(
                STAGE_POST_UPDATE,
                self.app.world,
                self.app.resources,
                self.app.pool,
                self.app.commands,
            )
            self.app.world.flush_commands(self.app.commands)

            var sync_rst = self.app.resources.get_ptr_mut[WorldTickSyncResource]()
            if sync_rst:
                sync_rst.value()[unsafe_offset=0].reset()

            self._invoke_event_swap_hooks()
            _ = self.app.world.advance_tick()

        self._sync_compat_view()
        return ticks

    def step_pie_frame(mut self, dt: Float32) -> Int:
        """Advances one interactive PIE frame using `FixedTime.advance_frame(dt)` (`STAGE_PRE_UPDATE`
        -> `due_steps` x `STAGE_FIXED_UPDATE` -> `STAGE_UPDATE` -> `STAGE_POST_UPDATE`), deferring
        `STAGE_LAST` + `event_swap_hooks` to the subsequent `render_editor_frame` call."""
        if self.pie.mode != PIE_PLAYING:
            return 0
        set_sim_float_mode()
        self._ensure_ground_plane()
        if self.selected_entity_id >= 0 and self.is_alive_id(self.selected_entity_id):
            var st_opt = self.app.world.get_ptr[StudioStyleComponent](self.resolve_entity(self.selected_entity_id))
            if st_opt and st_opt.value()[unsafe_offset=0].anim_step_hz > Float32(0.0):
                self.stepped_clock.step_hz = st_opt.value()[unsafe_offset=0].anim_step_hz

        var clamped_dt = self._advance_frame_time(dt, clamp_max=True)
        self.app._observe_tick_sync()

        self.app.schedule.run_stage(
            STAGE_PRE_UPDATE,
            self.app.world,
            self.app.resources,
            self.app.pool,
            self.app.commands,
        )
        self.app.world.flush_commands(self.app.commands)
        self.app._observe_tick_sync()

        var due_steps = 0
        var fixed_dt = Float32(1.0 / 60.0)
        var fix_opt = self.app.resources.get_ptr_mut[FixedTime]()
        if fix_opt:
            due_steps = fix_opt.value()[unsafe_offset=0].advance_frame(clamped_dt)
            fixed_dt = fix_opt.value()[unsafe_offset=0].dt

        for _ in range(due_steps):
            _ = self.app.world.advance_tick()
            self.app.schedule.run_stage(
                STAGE_FIXED_UPDATE,
                self.app.world,
                self.app.resources,
                self.app.pool,
                self.app.commands,
            )
            self.app.world.flush_commands(self.app.commands)
            self.app._observe_tick_sync()
            var fp2 = self.app.resources.get_ptr_mut[FixedTime]()
            if fp2:
                fp2.value()[unsafe_offset=0].tick_count += 1
            self.sim_tick += 1
            _ = self.stepped_clock.advance(fixed_dt)
        self.pie.ticks_simulated += due_steps

        self.app.schedule.run_stage(
            STAGE_UPDATE,
            self.app.world,
            self.app.resources,
            self.app.pool,
            self.app.commands,
        )
        self.app.world.flush_commands(self.app.commands)
        self.app._observe_tick_sync()

        if self.pie.mode == PIE_PLAYING and self.hot_reload_host.is_loaded():
            _ = self.hot_reload_host.step(self.app.world, self.app.resources, clamped_dt)
            self._sync_registered_pod_schemas()
            self.app.world.flush_commands(self.app.commands)
            self.app._observe_tick_sync()

        self.app.schedule.run_stage(
            STAGE_POST_UPDATE,
            self.app.world,
            self.app.resources,
            self.app.pool,
            self.app.commands,
        )
        self.app.world.flush_commands(self.app.commands)

        var sync_rst = self.app.resources.get_ptr_mut[WorldTickSyncResource]()
        if sync_rst:
            sync_rst.value()[unsafe_offset=0].reset()

        self._pie_frame_updated = True
        self._sync_compat_view()
        return due_steps

    def step_physics_deterministic(mut self, ticks: Int = 1, dt: Float32 = Float32(1.0 / 60.0)) -> Int:
        """Forwarding alias to `step_simulation(ticks, dt)` (`PhysicsPlugin` + `PhysicsWorld`)."""
        return self.step_simulation(ticks, dt)

    def _sync_studio_pre_render(
        mut self,
        screen_w: Int,
        screen_h: Int,
        gpu_ms: Float64 = 0.0,
        mounted_packs: Int = 0,
        mcp_clients: Int = 0,
        input: InputState = InputState(),
        dt: Float32 = 1.0 / 60.0,
    ):
        if self.pie.mode != PIE_PLAYING:
            snapshot_prev_global_transforms(self.app.world, self.app.pool)
            var fix_edit = self.app.resources.get_ptr_mut[FixedTime]()
            if fix_edit:
                fix_edit.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)

        var srv_hr = self.app.resources.get_ptr[AssetServer]()
        if srv_hr and srv_hr.value()[unsafe_offset=0].hot_reloads_applied != self._last_hot_reload_count:
            self._last_hot_reload_count = srv_hr.value()[unsafe_offset=0].hot_reloads_applied
            self.asset_browser_scanned = False

        # 1. Sync camera (isolate Edit/Paused mode or Shift+F1 ejected PIE from CameraComponent hijack)
        var use_editor_cam = (self.pie.mode != PIE_PLAYING) or self.pie_camera_ejected
        var ed_cam = RenderCameraPacket(
            self.cam_pos,
            self.cam_pos + self.cam_fwd * Float32(10.0),
            Vec3(0.0, 1.0, 0.0),
            self.fov_y_deg,
            self.near_z,
            Float32(1000.0),
            Float32(1.08),
        )
        var rps_cam_opt = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_cam_opt:
            rps_cam_opt.value()[unsafe_offset=0].editor_camera_override = use_editor_cam
            rps_cam_opt.value()[unsafe_offset=0].editor_camera = ed_cam
            rps_cam_opt.value()[unsafe_offset=0].has_editor_camera = True
            rps_cam_opt.value()[unsafe_offset=0].studio_override_mat_ids.clear()
            rps_cam_opt.value()[unsafe_offset=0].entity_smear_scales.clear()
            if self.pie.mode != PIE_STOPPED or not self.viewport_anim_preview:
                rps_cam_opt.value()[unsafe_offset=0].preview_bone_palettes.clear()
                rps_cam_opt.value()[unsafe_offset=0].preview_prev_bone_palettes.clear()

        var has_cam_ent = False
        if not use_editor_cam:
            var cam_mask = self.app.world.bit_for[CameraComponent]()
            var cam_ents = self.app.world.query_entities(cam_mask)
            for i in range(len(cam_ents)):
                var c_opt = self.app.world.get_ptr[CameraComponent](cam_ents[i])
                if c_opt and c_opt.value()[unsafe_offset=0].active:
                    has_cam_ent = True
                    break
        if use_editor_cam or not has_cam_ent:
            var rw_opt = self.app.resources.get_ptr_mut[RenderWorld]()
            if rw_opt:
                rw_opt.value()[unsafe_offset=0].camera = ed_cam

        # 2. Sync GPU NPR Cel / VFX materials & post-FX onto ScalableRenderer + gpu_slot[0]
        var rend_opt = self.app.resources.get_ptr_mut[ScalableRenderer]()
        var font_slot = -1
        var font_copy = SdfFont()
        if rend_opt:
            ref renderer = rend_opt.value()[unsafe_offset=0]
            var n_ents = len(self.app.world.entities)
            while len(self.gpu_materials.entity_pbr_mat_ids) < n_ents:
                self.gpu_materials.entity_pbr_mat_ids.append(-1)
                self.gpu_materials.entity_cel_mat_ids.append(-1)
                self.gpu_materials.entity_vfx_mat_ids.append(-1)
                self.gpu_materials.entity_generations.append(0)
                self.gpu_materials.entity_style_hashes.append(0)

            for i in range(n_ents):
                if not self.app.world.entities[i].alive:
                    self.gpu_materials.invalidate_entity(i)
                else:
                    var gen = self.app.world.entities[i].generation
                    if self.gpu_materials.entity_generations[i] != gen:
                        self.gpu_materials.invalidate_entity(i)
                        self.gpu_materials.entity_generations[i] = gen

            if renderer.uses_vulkan():
                ref gpu = renderer.gpu_slot[0]
                font_slot = gpu.ui_font_atlas_slot
                # `emit_studio_ui_draw_list` only reads this font until `ui_ctx.font` is set, so copy the
                # multi-MB SdfFont (atlas, glyph/kerning tables, TTF source) once instead of every frame.
                if not self.ui_ctx.font.valid:
                    font_copy = gpu.ui_font.copy()

                if self.preview_shockwave_active:
                    gpu.submit_shockwave(
                        AaaShockwaveRing(
                            center_u=Float32(0.5),
                            center_v=Float32(0.5),
                            radius_uv=self.preview_shockwave_radius,
                        )
                    )
                if self.preview_impact_blend > Float32(0.0):
                    gpu.set_impact_frame(
                        AaaImpactFrameParams(
                            blend=self.preview_impact_blend,
                            invert_polarity=self.preview_impact_invert,
                        )
                    )
                else:
                    gpu.clear_impact_frame()

                for i in range(n_ents):
                    if not self.app.world.entities[i].alive:
                        continue
                    var e = Entity(UInt32(i), self.app.world.entities[i].generation)
                    var mr_opt = self.app.world.get_ptr[MeshRendererComponent](e)
                    var mo_opt = self.app.world.get_ptr[MaterialOverrideComponent](e)
                    var st_opt = self.app.world.get_ptr[StudioStyleComponent](e)
                    if not mr_opt or not mo_opt or not st_opt:
                        self.gpu_materials.release_pbr_slot(i)
                        self.gpu_materials.release_cel_slot(i)
                        self.gpu_materials.release_vfx_slot(i)
                        self.gpu_materials.entity_style_hashes[i] = 0
                        continue
                    ref mr = mr_opt.value()[unsafe_offset=0]
                    ref mo = mo_opt.value()[unsafe_offset=0]
                    ref st = st_opt.value()[unsafe_offset=0]

                    var base_mat_id = (
                        mo.material_id
                        if (mo.enabled and mo.material_id >= 0)
                        else mr.material_id
                    )
                    if base_mat_id < 0 and mr.mesh_id >= 0 and mr.mesh_id < len(renderer.cpu_mesh_materials):
                        base_mat_id = renderer.cpu_mesh_materials[mr.mesh_id]

                    var eff_ov = mo.copy()
                    if st.clearcoat > mo.clearcoat:
                        eff_ov.clearcoat = st.clearcoat
                        eff_ov.clearcoat_roughness = st.clearcoat_roughness

                    var style_hash = _studio_mat_hash(st, mo, base_mat_id, renderer)
                    var dirty = self.gpu_materials.entity_style_hashes[i] != style_hash

                    if st.shading_mode == SHADING_NPR_CEL:
                        self.gpu_materials.release_pbr_slot(i)
                        self.gpu_materials.release_vfx_slot(i)
                        var ma = _build_studio_material_asset(
                            renderer, base_mat_id, mo, eff_ov, st
                        )
                        var mat = ma.to_aaa_material()
                        var cel = AaaCelParams(
                            sss_shadow_tint=st.cel_shadow_tint,
                            rim_color=st.cel_rim_color,
                            hair_halo_strength=st.cel_hair_halo,
                            outline_width_px=st.cel_outline_width_px,
                            outline_color=st.cel_outline_color,
                        )
                        if self.gpu_materials.entity_cel_mat_ids[i] < 0:
                            var recycled_cel = False
                            while len(self.gpu_materials.free_cel_mat_ids) > 0:
                                var cand = self.gpu_materials.free_cel_mat_ids.pop()
                                if cand >= 0 and cand < len(gpu.materials) and gpu.update_cel_material(cand, cel):
                                    self.gpu_materials.entity_cel_mat_ids[i] = cand
                                    gpu.queue_material_update(cand, mat)
                                    recycled_cel = True
                                    dirty = False
                                    break
                            if not recycled_cel:
                                self.gpu_materials.entity_cel_mat_ids[i] = gpu.add_cel_material(mat, cel)
                                dirty = False
                        elif dirty:
                            gpu.queue_material_update(self.gpu_materials.entity_cel_mat_ids[i], mat)
                            _ = gpu.update_cel_material(self.gpu_materials.entity_cel_mat_ids[i], cel)
                        self.gpu_materials.entity_style_hashes[i] = style_hash
                        if rps_cam_opt and self.gpu_materials.entity_cel_mat_ids[i] >= 0:
                            rps_cam_opt.value()[unsafe_offset=0].studio_override_mat_ids[e.to_bits()] = (
                                self.gpu_materials.entity_cel_mat_ids[i]
                            )
                    elif st.shading_mode == SHADING_STYLIZED_VFX:
                        self.gpu_materials.release_pbr_slot(i)
                        self.gpu_materials.release_cel_slot(i)
                        var ma = _build_studio_material_asset(
                            renderer, base_mat_id, mo, eff_ov, st
                        )
                        var vfx_mat = ma.to_aaa_material()
                        if vfx_mat.alpha_cutoff <= Float32(0.001):
                            vfx_mat.alpha_cutoff = Float32(0.48)
                        var vfx = AaaStylizedVfxParams.anime_flame(
                            core_color=st.vfx_core_color,
                            edge_color=st.vfx_edge_color,
                            erosion=st.vfx_erosion,
                            scroll_speed=st.vfx_scroll_speed,
                        )
                        if self.gpu_materials.entity_vfx_mat_ids[i] < 0:
                            var recycled_vfx = False
                            while len(self.gpu_materials.free_vfx_mat_ids) > 0:
                                var cand = self.gpu_materials.free_vfx_mat_ids.pop()
                                if cand >= 0 and cand < len(gpu.materials) and gpu.update_vfx_material(cand, vfx):
                                    self.gpu_materials.entity_vfx_mat_ids[i] = cand
                                    gpu.queue_material_update(cand, vfx_mat)
                                    recycled_vfx = True
                                    dirty = False
                                    break
                            if not recycled_vfx:
                                self.gpu_materials.entity_vfx_mat_ids[i] = gpu.add_vfx_material(vfx_mat, vfx)
                                dirty = False
                        elif dirty:
                            gpu.queue_material_update(self.gpu_materials.entity_vfx_mat_ids[i], vfx_mat)
                            _ = gpu.update_vfx_material(self.gpu_materials.entity_vfx_mat_ids[i], vfx)
                        self.gpu_materials.entity_style_hashes[i] = style_hash
                        if rps_cam_opt and self.gpu_materials.entity_vfx_mat_ids[i] >= 0:
                            rps_cam_opt.value()[unsafe_offset=0].studio_override_mat_ids[e.to_bits()] = (
                                self.gpu_materials.entity_vfx_mat_ids[i]
                            )
                    else:
                        self.gpu_materials.release_cel_slot(i)
                        self.gpu_materials.release_vfx_slot(i)
                        if (not mo.enabled) and st.clearcoat <= Float32(0.01):
                            self.gpu_materials.release_pbr_slot(i)
                            self.gpu_materials.entity_style_hashes[i] = 0
                            continue
                        var ma = _build_studio_material_asset(
                            renderer, base_mat_id, mo, eff_ov, st
                        )
                        var just_alloc = False
                        var recycled_pbr = False
                        if self.gpu_materials.entity_pbr_mat_ids[i] < 0:
                            while len(self.gpu_materials.free_pbr_mat_ids) > 0:
                                var cand = self.gpu_materials.free_pbr_mat_ids.pop()
                                if cand >= 0 and cand < len(renderer.materials):
                                    self.gpu_materials.entity_pbr_mat_ids[i] = cand
                                    recycled_pbr = True
                                    break
                            if self.gpu_materials.entity_pbr_mat_ids[i] < 0:
                                # Allocate an isolated slot rather than reusing `mr.material_id` in-place
                                # so shared base materials (e.g. `mat0` on non-Studio entities/meshes) are never clobbered.
                                self.gpu_materials.entity_pbr_mat_ids[i] = renderer.register_material(ma)
                                just_alloc = True
                        var pbr_id = self.gpu_materials.entity_pbr_mat_ids[i]
                        if pbr_id >= 0 and (recycled_pbr or (dirty and not just_alloc)):
                            renderer.update_material_asset_deferred(pbr_id, ma)
                        self.gpu_materials.entity_style_hashes[i] = style_hash
                        if rps_cam_opt and pbr_id >= 0:
                            rps_cam_opt.value()[unsafe_offset=0].studio_override_mat_ids[e.to_bits()] = pbr_id

        # Populate non-destructive smear scales for both Tier-0 and Vulkan extraction
        if rps_cam_opt:
            ref rps_sm = rps_cam_opt.value()[unsafe_offset=0]
            var sm_mask = self.app.world.bit_for[StudioStyleComponent]().union(
                self.app.world.bit_for[MeshRendererComponent]()
            )
            var sm_ents = self.app.world.query_entities(sm_mask)
            for si in range(len(sm_ents)):
                var se = sm_ents[si]
                var st_opt = self.app.world.get_ptr[StudioStyleComponent](se)
                if st_opt and st_opt.value()[unsafe_offset=0].smear_stretch != Float32(1.0):
                    rps_sm.entity_smear_scales[se.to_bits()] = compute_volume_preserving_smear_scale(
                        st_opt.value()[unsafe_offset=0].smear_stretch
                    )

        # Edit-mode viewport idle animation preview (strictly read-only to ECS components & state_hash)
        if self.pie.mode == PIE_STOPPED and self.viewport_anim_preview and rps_cam_opt:
            var prev_t = self.preview_anim_time
            var step_dt = max(Float32(0.0), dt)
            self._last_preview_anim_time = prev_t
            self.preview_anim_time = prev_t + step_dt
            var anim_cache_opt = self.app.resources.get_ptr[AnimAssetCache]()
            if anim_cache_opt and self.app.world.component_id[SkeletonComponent]() >= 0:
                ref anim_cache = anim_cache_opt.value()[unsafe_offset=0]
                ref rps_anim = rps_cam_opt.value()[unsafe_offset=0]
                var sk_ents = self.app.world.query_entities(self.app.world.bit_for[SkeletonComponent]())
                var has_sock_comp = self.app.world.component_id[BoneSocketComponent]() >= 0
                var sock_ents = (
                    self.app.world.query_entities(self.app.world.bit_for[BoneSocketComponent]())
                    if has_sock_comp
                    else List[Entity]()
                )
                for si in range(len(sk_ents)):
                    var se = sk_ents[si]
                    var sk_opt = self.app.world.get_ptr[SkeletonComponent](se)
                    if not sk_opt:
                        continue
                    ref sk = sk_opt.value()[unsafe_offset=0]
                    if sk.model_id == 0:
                        continue
                    var m_idx = anim_cache.find_model(sk.model_id)
                    if m_idx < 0:
                        continue
                    ref model = anim_cache.models[m_idx]
                    if len(model.skins) == 0 or len(model.meshes) == 0:
                        continue
                    ref skin = model.skins[0]
                    var nj = len(skin.joints)
                    if nj == 0:
                        continue

                    var clip_idx = 0 if len(model.animations) > 0 else -1
                    var anim_spd = Float32(1.0)
                    var anim_loop = True
                    var anim_playing = True
                    var ac_opt = self.app.world.get_ptr[AnimatorComponent](se)
                    if ac_opt:
                        ref ac = ac_opt.value()[unsafe_offset=0]
                        anim_playing = ac.playing
                        anim_loop = ac.loop
                        anim_spd = ac.speed
                        if ac.clip_index >= 0 and ac.clip_index < len(model.animations):
                            clip_idx = ac.clip_index
                        elif ac.clip_name.byte_length() > 0 and len(model.animations) > 0:
                            var tlow = Self._ascii_lower(ac.clip_name)
                            for ci in range(len(model.animations)):
                                if model.animations[ci].name == ac.clip_name or Self._ascii_lower(model.animations[ci].name) == tlow:
                                    clip_idx = ci
                                    break
                        elif ac.graph_id != 0:
                            var g_idx = anim_cache.find_graph(ac.graph_id)
                            if g_idx >= 0 and len(anim_cache.graphs[g_idx].states) > 0 and len(model.animations) > 0:
                                var st_idx = max(0, min(len(anim_cache.graphs[g_idx].states) - 1, ac.animator.state if ac.animator.state >= 0 else 0))
                                var g_clip = anim_cache.graphs[g_idx].states[st_idx].clip
                                if g_clip >= 0 and g_clip < len(model.animations):
                                    clip_idx = g_clip

                    var tmp_skel = sk.skeleton.copy()
                    tmp_skel.ensure_topology(model)
                    var tmp_pose = tmp_skel.pose.copy()
                    if clip_idx >= 0 and clip_idx < len(model.animations):
                        var dur = model.animations[clip_idx].duration
                        var t_prev = (prev_t * anim_spd) if anim_playing else Float32(0.0)
                        var t_cur = (self.preview_anim_time * anim_spd) if anim_playing else Float32(0.0)
                        if ac_opt and not anim_playing:
                            t_prev = ac_opt.value()[unsafe_offset=0].time
                            t_cur = t_prev
                        if anim_loop and dur > Float32(1.0e-5):
                            t_prev = wrap_time(t_prev, dur)
                            t_cur = wrap_time(t_cur, dur)
                        elif dur > Float32(0.0):
                            t_prev = max(Float32(0.0), min(dur, t_prev))
                            t_cur = max(Float32(0.0), min(dur, t_cur))
                        sample_pose(model, clip_idx, t_prev, tmp_pose)
                        tmp_skel.apply_pose(tmp_pose, model)
                        tmp_skel.update_matrices(model)
                        tmp_skel.snapshot_prev_matrices()
                        sample_pose(model, clip_idx, t_cur, tmp_pose)
                        tmp_skel.apply_pose(tmp_pose, model)
                        tmp_skel.update_matrices(model)
                    else:
                        tmp_skel.update_matrices(model)
                        tmp_skel.snapshot_prev_matrices()

                    var cur_bones = List[DualQuat](capacity=nj)
                    var prev_bones = List[DualQuat](capacity=nj)
                    var n_wm = len(tmp_skel.world_matrices)
                    var has_prev_wm = len(tmp_skel.prev_world_matrices) == n_wm
                    for j in range(nj):
                        var jn = skin.joints[j]
                        var wm = tmp_skel.world_matrices[jn].copy() if (jn >= 0 and jn < n_wm) else Mat4()
                        var pm = (
                            tmp_skel.prev_world_matrices[jn].copy()
                            if (has_prev_wm and jn >= 0 and jn < n_wm)
                            else wm.copy()
                        )
                        var ibm = Mat4()
                        if (j + 1) * 16 <= len(skin.inverse_bind):
                            for k in range(16):
                                ibm.values[k] = skin.inverse_bind[j * 16 + k]
                        cur_bones.append(DualQuat.from_mat4(wm.multiply(ibm)))
                        prev_bones.append(DualQuat.from_mat4(pm.multiply(ibm)))
                    rps_anim.preview_bone_palettes[se.to_bits()] = cur_bones^
                    rps_anim.preview_prev_bone_palettes[se.to_bits()] = prev_bones^

                    var sk_place = Placement()
                    var lt_sk = self.app.world.get_ptr[LocalTransform](se)
                    var gt_sk = self.app.world.get_ptr[GlobalTransform](se)
                    if gt_sk:
                        ref g = gt_sk.value()[unsafe_offset=0]
                        var yrad = det_atan2(
                            Float32(2.0) * (g.rotation.w * g.rotation.y + g.rotation.x * g.rotation.z),
                            Float32(1.0) - Float32(2.0) * (g.rotation.y * g.rotation.y + g.rotation.z * g.rotation.z),
                        )
                        sk_place = Placement(g.translation, g.scale, yrad * (Float32(180.0) / PI))
                    elif lt_sk:
                        ref l = lt_sk.value()[unsafe_offset=0]
                        var yrad = det_atan2(
                            Float32(2.0) * (l.rotation.w * l.rotation.y + l.rotation.x * l.rotation.z),
                            Float32(1.0) - Float32(2.0) * (l.rotation.y * l.rotation.y + l.rotation.z * l.rotation.z),
                        )
                        sk_place = Placement(l.translation, l.scale, yrad * (Float32(180.0) / PI))

                    for bsi in range(len(sock_ents)):
                        var sock_e = sock_ents[bsi]
                        var bs_opt = self.app.world.get_ptr[BoneSocketComponent](sock_e)
                        if bs_opt and bs_opt.value()[unsafe_offset=0].skeleton_entity == se:
                            var tmp_att = bs_opt.value()[unsafe_offset=0].attachment.copy()
                            tmp_att.update(tmp_skel, sk_place)
                            rps_anim.preview_world_transforms[sock_e.to_bits()] = tmp_att.world_matrix.copy()

        # 3. Emit Studio DockSpace + Hierarchy + Inspector + Gizmos into RenderPluginStateResource.ui_draw_list
        var ui_copy = UiDrawList()
        if self.anim_mode != UI_ANIM_INSTANT:
            var safe_dt_t = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(0.1))
            self.ui_time_sec += safe_dt_t
        self.poll_agent_runtime(dt)
        var rend_time_opt = self.app.resources.get_ptr_mut[ScalableRenderer]()
        if rend_time_opt:
            rend_time_opt.value()[unsafe_offset=0].ui_time_sec = self.ui_time_sec

        emit_studio_ui_draw_list(
            ui_copy,
            self,
            font_slot,
            font_copy,
            screen_w,
            screen_h,
            gpu_ms=gpu_ms,
            mounted_packs=mounted_packs,
            mcp_clients=mcp_clients,
            input=input,
            dt=dt,
        )
        var nav_res = self.app.resources.get_ptr_mut[NavPluginStateResource]()
        if nav_res:
            nav_res.value()[unsafe_offset=0].draw_debug = self.show_navmesh_overlay

        if self.agent_bus.has_active_visuals():
            var rw_ag = self.app.resources.get_ptr_mut[RenderWorld]()
            if rw_ag:
                _ = emit_agent_viewport_3d_overlays(
                    self.agent_bus,
                    rw_ag.value()[unsafe_offset=0].overlay_lines,
                    self.ui_time_sec,
                )

        if self.show_collider_overlay:
            var rw_col = self.app.resources.get_ptr_mut[RenderWorld]()
            if rw_col:
                ref ol = rw_col.value()[unsafe_offset=0].overlay_lines
                var col_line_color = UInt32(0xFF40D8FF)
                var col_mask = self.app.world.bit_for[ColliderComponent]()
                var col_ents = self.app.world.query_entities(col_mask)
                for ci in range(len(col_ents)):
                    var ce = col_ents[ci]
                    var meta_c = self.app.world.get_ptr[StudioEntityMetaComponent](ce)
                    if meta_c and not meta_c.value()[unsafe_offset=0].enabled:
                        continue
                    var c_opt = self.app.world.get_ptr[ColliderComponent](ce)
                    if not c_opt:
                        continue
                    ref col = c_opt.value()[unsafe_offset=0]
                    var wpos = Vec3()
                    var wrot = Quat.identity()
                    var gt_c = self.app.world.get_ptr[GlobalTransform](ce)
                    if gt_c:
                        wpos = gt_c.value()[unsafe_offset=0].translation
                        wrot = gt_c.value()[unsafe_offset=0].rotation
                    else:
                        var lt_c = self.app.world.get_ptr[LocalTransform](ce)
                        if lt_c:
                            wpos = lt_c.value()[unsafe_offset=0].translation
                            wrot = lt_c.value()[unsafe_offset=0].rotation
                    var center = wpos + wrot.rotate(col.local_offset)
                    var rot = (wrot * col.local_rot).normalized()
                    var he = col.half_extents
                    if col.shape == BODY_SHAPE_SPHERE:
                        var r = max(Float32(0.05), col.radius)
                        he = Vec3(r, r, r)
                    elif col.shape == BODY_SHAPE_CAPSULE:
                        var r = max(Float32(0.05), col.radius)
                        he = Vec3(r, max(r, col.half_height + r), r)
                    var c0 = center + rot.rotate(Vec3(-he.x, -he.y, -he.z))
                    var c1 = center + rot.rotate(Vec3( he.x, -he.y, -he.z))
                    var c2 = center + rot.rotate(Vec3( he.x, -he.y,  he.z))
                    var c3 = center + rot.rotate(Vec3(-he.x, -he.y,  he.z))
                    var c4 = center + rot.rotate(Vec3(-he.x,  he.y, -he.z))
                    var c5 = center + rot.rotate(Vec3( he.x,  he.y, -he.z))
                    var c6 = center + rot.rotate(Vec3( he.x,  he.y,  he.z))
                    var c7 = center + rot.rotate(Vec3(-he.x,  he.y,  he.z))
                    ol.add_line(c0, c1, col_line_color, Float32(1.5))
                    ol.add_line(c1, c2, col_line_color, Float32(1.5))
                    ol.add_line(c2, c3, col_line_color, Float32(1.5))
                    ol.add_line(c3, c0, col_line_color, Float32(1.5))
                    ol.add_line(c4, c5, col_line_color, Float32(1.5))
                    ol.add_line(c5, c6, col_line_color, Float32(1.5))
                    ol.add_line(c6, c7, col_line_color, Float32(1.5))
                    ol.add_line(c7, c4, col_line_color, Float32(1.5))
                    ol.add_line(c0, c4, col_line_color, Float32(1.5))
                    ol.add_line(c1, c5, col_line_color, Float32(1.5))
                    ol.add_line(c2, c6, col_line_color, Float32(1.5))
                    ol.add_line(c3, c7, col_line_color, Float32(1.5))

        var has_point_light = False
        var pl_mask = self.app.world.bit_for[PointLightComponent]()
        var pl_ents = self.app.world.query_entities(pl_mask)
        for i in range(len(pl_ents)):
            var pl_opt = self.app.world.get_ptr[PointLightComponent](pl_ents[i])
            if pl_opt and pl_opt.value()[unsafe_offset=0].enabled:
                has_point_light = True
                break
        if not has_point_light:
            var sl_mask = self.app.world.bit_for[SpotLightComponent]()
            var sl_ents = self.app.world.query_entities(sl_mask)
            for i in range(len(sl_ents)):
                var sl_opt = self.app.world.get_ptr[SpotLightComponent](sl_ents[i])
                if sl_opt and sl_opt.value()[unsafe_offset=0].enabled:
                    has_point_light = True
                    break

        var rps_opt = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_opt:
            ref rps = rps_opt.value()[unsafe_offset=0]
            rps.ui_draw_list = ui_copy^
            rps.extra_lights.clear()
            if not has_point_light:
                rps.extra_lights.append(
                    AaaClusterLight.point(Vec3(-2.8, 3.8, -2.5), Vec3(18.0, 14.0, 10.0), Float32(14.0))
                )
                rps.extra_lights.append(
                    AaaClusterLight.point(Vec3(4.8, 4.2, 0.5), Vec3(8.0, 14.0, 24.0), Float32(14.0))
                )

    def render_editor_frame(
        mut self,
        screen_w: Int,
        screen_h: Int,
        gpu_ms: Float64 = 0.0,
        mounted_packs: Int = 0,
        mcp_clients: Int = 0,
        input: InputState = InputState(),
        dt: Float32 = 1.0 / 60.0,
    ):
        """Runs Edit/Paused pre/post stages (`STAGE_PRE_UPDATE`, `STAGE_POST_UPDATE` — never `STAGE_UPDATE`)
        or completes an interactive PIE frame after `step_pie_frame`, then renders via `STAGE_LAST`."""
        self._ensure_ground_plane()
        var already_updated = self._pie_frame_updated
        self._pie_frame_updated = False
        if not already_updated:
            _ = self._advance_frame_time(dt, clamp_max=True)
            self.app._observe_tick_sync()
            self.app.schedule.run_stage(
                STAGE_PRE_UPDATE,
                self.app.world,
                self.app.resources,
                self.app.pool,
                self.app.commands,
            )
            self.app.world.flush_commands(self.app.commands)
            self.app._observe_tick_sync()
            self.app.schedule.run_stage(
                STAGE_POST_UPDATE,
                self.app.world,
                self.app.resources,
                self.app.pool,
                self.app.commands,
            )
            self.app.world.flush_commands(self.app.commands)
            var sync_rst = self.app.resources.get_ptr_mut[WorldTickSyncResource]()
            if sync_rst:
                sync_rst.value()[unsafe_offset=0].reset()

        self._sync_studio_pre_render(screen_w, screen_h, gpu_ms, mounted_packs, mcp_clients, input, dt)

        self.app.schedule.run_stage(STAGE_LAST, self.app.world, self.app.resources, self.app.pool, self.app.commands)
        self.app.world.flush_commands(self.app.commands)

        self._invoke_event_swap_hooks()
        _ = self.app.world.advance_tick()
        self._sync_compat_view()

    # ---- 1-Click Play-In-Editor (`PieSession` + `Scene` + `PhysicsWorld` + LZ4) ----

    def pie_play(mut self) -> Bool:
        self._pie_frame_updated = False
        self.preview_anim_time = Float32(0.0)
        self._last_preview_anim_time = Float32(0.0)
        var fix_rst = self.app.resources.get_ptr_mut[FixedTime]()
        if fix_rst:
            fix_rst.value()[unsafe_offset=0].accumulator = 0.0
            fix_rst.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)
            fix_rst.value()[unsafe_offset=0].clamped_last_frame = False
        if self.pie.mode == PIE_PLAYING:
            return True
        if self.pie.mode == PIE_PAUSED:
            self.pie.mode = PIE_PLAYING
            var rps_res = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
            if rps_res:
                rps_res.value()[unsafe_offset=0].editor_camera_override = self.pie_camera_ejected
            return True
        self.pie_camera_ejected = False
        self._reset_pie_trigger_buffer()
        self._ensure_editor_ids()  # the Play checkpoint carries every id, so Stop restores the same ones
        self.pie.pre_play_hash = self.state_hash()
        var srv_entry = self.app.resources.get_ptr[AssetServer]()
        self._pie_entry_hot_reloads = (
            srv_entry.value()[unsafe_offset=0].hot_reloads_applied if srv_entry else 0
        )
        var res_entry = self.app.resources.get_ptr[SceneAssetResolver]()
        self._pie_entry_epochs = (
            res_entry.value()[unsafe_offset=0].capture_pie_epoch_snapshot()
            if res_entry
            else PieResolverEpochSnapshot()
        )
        var style_entry = self.app.resources.get_ptr[StudioFlmatStyleSyncResource]()
        self._pie_entry_style_epochs = (
            _copy_u64_style_epoch_dict(style_entry.value()[unsafe_offset=0].synced_entity_epochs)
            if style_entry
            else Dict[UInt64, Tuple[String, UInt64]]()
        )
        try:
            self.scene.enter_pie(self.app.world)
        except:
            pass
        var w = BlobWriter()
        self.serialize(w)
        self.pie.checkpoint_lz4 = pack_checkpoint_lz4(w)
        self.pie.ticks_simulated = 0
        self.pie.last_restore_verified = False
        self.pie.mode = PIE_PLAYING
        _ = self.input_map.pop_context("StudioPlay")
        self.input_map.push_context("StudioPlay")
        var rps_play = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_play:
            rps_play.value()[unsafe_offset=0].editor_camera_override = False
        if (not self.hot_reload_host.is_loaded()) and self.gameplay_so_path.byte_length() > 0:
            var so_p = self.gameplay_so_path
            _ = self.load_gameplay_plugin(so_p)
        return len(self.pie.checkpoint_lz4) > 0

    def pie_pause(mut self) -> Bool:
        self._pie_frame_updated = False
        if self.pie.mode == PIE_PLAYING:
            self.pie.mode = PIE_PAUSED
            var rps_pause = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
            if rps_pause:
                rps_pause.value()[unsafe_offset=0].editor_camera_override = True
            return True
        if self.pie.mode == PIE_PAUSED:
            self.pie.mode = PIE_PLAYING
            var rps_res = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
            if rps_res:
                rps_res.value()[unsafe_offset=0].editor_camera_override = self.pie_camera_ejected
            return True
        return False

    def pie_step(mut self, ticks: Int = 1, dt: Float32 = Float32(1.0 / 60.0)) -> Int:
        if self.pie.mode == PIE_STOPPED:
            if not self.pie_play():
                return 0
            self.pie.mode = PIE_PAUSED
            var rps_step = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
            if rps_step:
                rps_step.value()[unsafe_offset=0].editor_camera_override = True
        if self.selected_entity_id >= 0 and self.is_alive_id(self.selected_entity_id):
            var st_opt = self.app.world.get_ptr[StudioStyleComponent](self.resolve_entity(self.selected_entity_id))
            if st_opt and st_opt.value()[unsafe_offset=0].anim_step_hz > Float32(0.0):
                self.stepped_clock.step_hz = st_opt.value()[unsafe_offset=0].anim_step_hz
        var done = self.step_simulation(ticks, dt, step_gameplay_plugin=True)
        for _ in range(done):
            _ = self.stepped_clock.advance(dt)
        self.pie.ticks_simulated += done
        return done

    def _abort_pie_session(mut self):
        """Cleanly aborts an active/paused PIE session without restoring the pre-PIE scene."""
        self._pie_frame_updated = False
        self._int_edit_wid = UInt64(0)
        self._int_edit_entity_id = -1
        self._int_edit_buf = ""
        self._reset_pie_trigger_buffer()
        if self.hot_reload_host.is_loaded():
            _ = self.hot_reload_host.unload(self.app.world, self.app.resources, is_reload=False)
        _ = self.input_map.pop_context("StudioPlay")
        self.release_pointer_lock()
        self.pie_camera_ejected = False
        var aud_opt = self.app.resources.get_ptr_mut[AudioServer]()
        if aud_opt:
            aud_opt.value()[unsafe_offset=0].stop_all()
        var async_opt = self.app.resources.get_ptr_mut[AudioSyncStateResource]()
        if async_opt:
            async_opt.value()[unsafe_offset=0].active_emitters.clear()
            async_opt.value()[unsafe_offset=0].emitter_to_slot.clear()
        var fix_rst = self.app.resources.get_ptr_mut[FixedTime]()
        if fix_rst:
            fix_rst.value()[unsafe_offset=0].accumulator = 0.0
            fix_rst.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)
            fix_rst.value()[unsafe_offset=0].clamped_last_frame = False
        var rps_stop = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_stop:
            rps_stop.value()[unsafe_offset=0].editor_camera_override = True
        self.pie.mode = PIE_STOPPED
        self.pie.checkpoint_lz4.clear()
        self.pie.pre_play_hash = 0
        self.pie.ticks_simulated = 0
        self.pie.last_restore_verified = False
        self.scene.has_pie_snapshot = False
        self.scene.pie_snapshot.clear()
        self._pie_entry_hot_reloads = 0
        self._pie_entry_epochs = PieResolverEpochSnapshot()
        self._pie_entry_style_epochs.clear()

    def pie_stop(mut self) raises -> Bool:
        """Restores exact pre-PIE `World` + `PhysicsWorld` from LZ4 checkpoint and verifies `fnv64` equality."""
        self._pie_frame_updated = False
        self._int_edit_wid = UInt64(0)
        self._int_edit_entity_id = -1
        self._int_edit_buf = ""
        self._reset_pie_trigger_buffer()
        if self.hot_reload_host.is_loaded():
            _ = self.hot_reload_host.unload(self.app.world, self.app.resources, is_reload=False)
        _ = self.input_map.pop_context("StudioPlay")
        self.release_pointer_lock()
        self.pie_camera_ejected = False
        var aud_opt = self.app.resources.get_ptr_mut[AudioServer]()
        if aud_opt:
            aud_opt.value()[unsafe_offset=0].stop_all()
        var async_opt = self.app.resources.get_ptr_mut[AudioSyncStateResource]()
        if async_opt:
            async_opt.value()[unsafe_offset=0].active_emitters.clear()
            async_opt.value()[unsafe_offset=0].emitter_to_slot.clear()
        var fix_rst = self.app.resources.get_ptr_mut[FixedTime]()
        if fix_rst:
            fix_rst.value()[unsafe_offset=0].accumulator = 0.0
            fix_rst.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)
            fix_rst.value()[unsafe_offset=0].clamped_last_frame = False
        var rps_stop = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_stop:
            rps_stop.value()[unsafe_offset=0].editor_camera_override = True
        if self.pie.mode == PIE_STOPPED or len(self.pie.checkpoint_lz4) == 0:
            self.pie.mode = PIE_STOPPED
            self.pie.checkpoint_lz4.clear()
            return False
        var play_end_hash = self.state_hash()
        var reader = unpack_checkpoint_lz4(self.pie.checkpoint_lz4)
        self.deserialize(reader)
        self.gpu_materials.clear_entity_slots()
        if play_end_hash != self.pie.pre_play_hash:
            self.push_toast("Stopped: changes made during Play were discarded", 0, 2.5)
        self.pie.mode = PIE_STOPPED
        self.pie.checkpoint_lz4.clear()
        self.scene.has_pie_snapshot = False
        self.scene.pie_snapshot.clear()
        # Hash the restored world now, but raise on a mismatch only after the material-style and asset
        # epochs are restored too: raising here used to skip them, leaving PIE-era bindings in Edit mode.
        var restored_hash = self.state_hash()
        self.pie.last_restore_verified = restored_hash == self.pie.pre_play_hash
        var style_sync_opt = self.app.resources.get_ptr_mut[StudioFlmatStyleSyncResource]()
        if style_sync_opt:
            style_sync_opt.value()[unsafe_offset=0].synced_entity_epochs = _copy_u64_style_epoch_dict(
                self._pie_entry_style_epochs
            )
        var srv_stop = self.app.resources.get_ptr_mut[AssetServer]()
        var cur_hr = srv_stop.value()[unsafe_offset=0].hot_reloads_applied if srv_stop else 0
        if srv_stop and cur_hr != self._pie_entry_hot_reloads:
            var res_stop = self.app.resources.get_ptr_mut[SceneAssetResolver]()
            if res_stop:
                res_stop.value()[unsafe_offset=0].restore_pie_epoch_snapshot(self._pie_entry_epochs)
                res_stop.value()[unsafe_offset=0].resolve_world(
                    self.app.world,
                    srv_stop.value()[unsafe_offset=0],
                    self.app.resources,
                )
            _sync_studio_flmat_styles(self.app.world, self.app.resources)
            self._sync_compat_view()
        self._pie_entry_hot_reloads = 0
        self._pie_entry_style_epochs.clear()
        if not self.pie.last_restore_verified:
            raise Error("PIE LZ4 world restore hash mismatch")
        return True

    def _resolve_default_gameplay_so_path(self) -> String:
        if self.project_root.byte_length() == 0:
            return String()
        var root = self.project_root
        while root.byte_length() > 1 and root.endswith("/"):
            var trimmed = String(root[byte=0 : root.byte_length() - 1])
            root = trimmed^
        var cand1 = root + "/build/gameplay.so"
        if file_size(cand1) > 0:
            return cand1
        var cand2 = root + "/build/libgameplay.so"
        if file_size(cand2) > 0:
            return cand2
        return String()

    def _sync_registered_pod_schemas(mut self):
        var res_opt = self.app.resources.get_ptr[PodSchemaRegistryResource]()
        if not res_opt:
            return
        ref res_reg = res_opt.value()[unsafe_offset=0]
        if res_reg.epoch == self._synced_pod_schema_epoch:
            return
        for i in range(len(res_reg.schemas)):
            _ = self.registry.register_pod_schema(res_reg.schemas[i])
        self.registry.ensure_registered_in_world(self.app.world)
        self._synced_pod_schema_epoch = res_reg.epoch

    def load_gameplay_plugin(mut self, so_path: String) -> Bool:
        if so_path.byte_length() == 0:
            self.push_toast("Gameplay .so load failed: empty path", 2, 3.0)
            return False
        var ok = self.hot_reload_host.load_or_reload(so_path, self.app.world, self.app.resources)
        if not ok:
            self.push_toast("Gameplay .so load failed: " + self.hot_reload_host.last_error, 2, 3.5)
            return False
        self.gameplay_so_path = so_path
        self._sync_registered_pod_schemas()
        self._sync_compat_view()
        self.push_toast("Loaded gameplay .so (gen " + String(self.hot_reload_host.reload_generation) + ")", 0, 2.0)
        return True

    def reload_gameplay_plugin(mut self, so_path: String = "") -> Bool:
        var target = so_path
        if target.byte_length() == 0:
            target = self.gameplay_so_path
        if target.byte_length() == 0:
            target = self.hot_reload_host.source_path
        if target.byte_length() == 0:
            target = self._resolve_default_gameplay_so_path()
        if target.byte_length() == 0:
            self.push_toast("Reload .so failed: no gameplay .so configured", 2, 3.0)
            return False
        return self.load_gameplay_plugin(target)

    # ---- `PhysicsWorld` + Static BVH Scene Raycast ----

    def raycast_scene(self, origin: Vec3, direction: Vec3, max_distance: Float32 = 1000.0) -> StudioRaycastHit:
        var best = StudioRaycastHit()
        var d_len = direction.length()
        if d_len <= Float32(1.0e-7) or max_distance <= Float32(0.0):
            return best
        var dir = direction * (Float32(1.0) / d_len)
        best.distance = max_distance

        var phys_opt = self.app.resources.get_ptr[PhysicsWorld]()
        var sync_opt = self.app.resources.get_ptr[PhysicsSyncStateResource]()
        if phys_opt and sync_opt:
            ref physics = phys_opt.value()[unsafe_offset=0]
            ref sync = sync_opt.value()[unsafe_offset=0]
            var hits = physics.raycast_all(origin, dir, max_distance, ignore_body_id=self._ground_body_id)
            for hi in range(len(hits)):
                ref h = hits[hi]
                var ent_bits_opt = sync.body_to_entity.get(h.body_id)
                if not ent_bits_opt:
                    continue
                var e = Entity.from_bits(ent_bits_opt.value())
                if not self.app.world.is_alive(e):
                    continue
                var mr_opt = self.app.world.get_ptr[MeshRendererComponent](e)
                if mr_opt and not mr_opt.value()[unsafe_offset=0].visible:
                    continue
                if h.distance < best.distance:
                    best.hit = True
                    best.entity_id = Int(e.index)
                    best.distance = h.distance
                    best.point = h.point
                    best.normal = h.normal
                break

        # Also check attached static BVH8 mesh colliders
        for i in range(len(self.app.world.entities)):
            if not self.app.world.entities[i].alive:
                continue
            var e = Entity(UInt32(i), self.app.world.entities[i].generation)
            var meta_opt = self.app.world.get_ptr[StudioEntityMetaComponent](e)
            if not meta_opt:
                continue
            ref meta = meta_opt.value()[unsafe_offset=0]
            if meta.collider_kind != COLLIDER_STATIC_MESH or meta.bvh_slot < 0 or meta.bvh_slot >= len(self.static_bvh_colliders):
                continue
            var mr_opt = self.app.world.get_ptr[MeshRendererComponent](e)
            if mr_opt and not mr_opt.value()[unsafe_offset=0].visible:
                continue
            var w_pos = Vec3()
            var w_rot = Quat()
            var w_scl = Vec3(1.0, 1.0, 1.0)
            var gt_opt = self.app.world.get_ptr[GlobalTransform](e)
            if gt_opt:
                w_pos = gt_opt.value()[unsafe_offset=0].translation
                w_rot = gt_opt.value()[unsafe_offset=0].rotation
                w_scl = gt_opt.value()[unsafe_offset=0].scale
            var inv_q = w_rot.conjugate()
            var inv_s = Vec3(
                Float32(1.0) / max(Float32(1.0e-4), w_scl.x),
                Float32(1.0) / max(Float32(1.0e-4), w_scl.y),
                Float32(1.0) / max(Float32(1.0e-4), w_scl.z),
            )
            var loc_o_raw = inv_q.rotate(origin - w_pos)
            var loc_o = Vec3(loc_o_raw.x * inv_s.x, loc_o_raw.y * inv_s.y, loc_o_raw.z * inv_s.z)
            var loc_d_raw = inv_q.rotate(dir)
            var loc_d_scaled = Vec3(loc_d_raw.x * inv_s.x, loc_d_raw.y * inv_s.y, loc_d_raw.z * inv_s.z)
            var d_scale = loc_d_scaled.length()
            if d_scale > Float32(1.0e-7):
                var loc_d_u = loc_d_scaled * (Float32(1.0) / d_scale)
                var bhit = self.static_bvh_colliders[meta.bvh_slot].raycast(loc_o, loc_d_u, best.distance * d_scale)
                if bhit.hit:
                    var world_t = bhit.t / d_scale
                    if world_t >= Float32(0.0) and world_t < best.distance:
                        best.hit = True
                        best.entity_id = i
                        best.distance = world_t
                        best.point = origin + dir * world_t
                        best.normal = w_rot.rotate(bhit.normal).normalized()

        if self.has_ground_plane and abs(dir.y) > Float32(1.0e-6):
            var tg = (self.ground_plane_y - origin.y) / dir.y
            if tg >= Float32(0.0) and tg < best.distance:
                best.hit = True
                best.entity_id = -2
                best.distance = tg
                best.point = origin + dir * tg
                best.normal = Vec3(0.0, 1.0, 0.0)

        return best

    def fill_render_instance_entity_maps(
        self,
        mut out_inst_to_entity: List[Int],
        mut out_inst_to_bits: List[UInt64],
    ):
        """Populates `out_inst_to_entity` / `out_inst_to_bits` matching `FlintAaaVulkanRenderer.draw_instance`
        and Tier-0 `_render_tier0_simd` filtering (`mesh_id >= 0 and mesh_id < mesh_count`, `max_instances`)."""
        out_inst_to_entity.clear()
        out_inst_to_bits.clear()
        var rw_opt = self.app.resources.get_ptr[RenderWorld]()
        if not rw_opt:
            return
        ref rw = rw_opt.value()[unsafe_offset=0]
        var rend_opt = self.app.resources.get_ptr[ScalableRenderer]()
        var mesh_count = -1
        var max_inst = -1
        if rend_opt and rend_opt.value()[unsafe_offset=0].uses_vulkan():
            ref gpu = rend_opt.value()[unsafe_offset=0].gpu_slot[0]
            mesh_count = len(gpu.geom.meshes)
            max_inst = gpu.max_instances
        elif rend_opt:
            mesh_count = len(rend_opt.value()[unsafe_offset=0].cpu_meshes)
        for i in range(len(rw.instances)):
            if max_inst >= 0 and len(out_inst_to_entity) >= max_inst:
                break
            var mid = rw.instances[i].mesh_id
            if mesh_count >= 0 and (mid < 0 or mid >= mesh_count):
                continue
            var bits = rw.instances[i].entity_id
            var e = Entity.from_bits(bits)
            out_inst_to_entity.append(Int(e.index))
            out_inst_to_bits.append(bits)


def fill_instance_entity_maps(
    studio: FlintStudioState,
    mut out_inst_to_entity: List[Int],
    mut out_inst_to_bits: List[UInt64],
):
    """Shared helper populating compact GPU instance-to-entity maps for `VisBufferGpuPicker`."""
    studio.fill_render_instance_entity_maps(out_inst_to_entity, out_inst_to_bits)


# ============================================================================
# 5. Procedural Forge Bake-to-`.glb` (`glb_writer.mojo`)
# ============================================================================

def _make_beveled_unit_box_mesh() -> NativeMesh:
    var mb = MeshBuilder(base_color=0xFFE0E6F0, roughness=0.35, metallic=0.15)
    var faces = [
        (Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0, 1, 0)),
        (Vec3(0, 0, -1), Vec3(-1, 0, 0), Vec3(0, 1, 0)),
        (Vec3(1, 0, 0), Vec3(0, 0, -1), Vec3(0, 1, 0)),
        (Vec3(-1, 0, 0), Vec3(0, 0, 1), Vec3(0, 1, 0)),
        (Vec3(0, 1, 0), Vec3(1, 0, 0), Vec3(0, 0, -1)),
        (Vec3(0, -1, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)),
    ]
    for fi in range(6):
        var n = faces[fi][0]
        var u = faces[fi][1]
        var v = faces[fi][2]
        var c = n * Float32(0.5)
        var i0 = mb.add_vertex(c - u * Float32(0.5) - v * Float32(0.5), n, 0.0, 1.0)
        var i1 = mb.add_vertex(c + u * Float32(0.5) - v * Float32(0.5), n, 1.0, 1.0)
        var i2 = mb.add_vertex(c + u * Float32(0.5) + v * Float32(0.5), n, 1.0, 0.0)
        var i3 = mb.add_vertex(c - u * Float32(0.5) + v * Float32(0.5), n, 0.0, 0.0)
        mb.add_tri(i0, i1, i2)
        mb.add_tri(i0, i2, i3)
    return mb^.finish()


def build_studio_reference_grid_mesh(span: Float32 = 36.0, divisions: Int = 24) -> NativeMesh:
    """Builds an engine-neutral architectural slate reference grid floor (`Studio_ReferenceGrid`)."""
    var n = max(4, divisions + 1)
    var step = span / Float32(n - 1)
    var half = span * Float32(0.5)
    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.58, metallic=0.06)
    for iz in range(n):
        var wz = -half + Float32(iz) * step
        for ix in range(n):
            var wx = -half + Float32(ix) * step
            var checker = ((ix + iz) & 1) == 0
            var col: UInt32 = 0xFF3E4A5C if checker else 0xFF2B3544
            _ = mb.add_vertex(
                Vec3(wx, 0.0, wz),
                Vec3(0.0, 1.0, 0.0),
                Float32(ix) * Float32(0.25),
                Float32(iz) * Float32(0.25),
                col,
            )
    for iz in range(n - 1):
        for ix in range(n - 1):
            var i00 = iz * n + ix
            var i10 = i00 + 1
            var i01 = (iz + 1) * n + ix
            var i11 = i01 + 1
            mb.add_tri(i00, i01, i10)
            mb.add_tri(i10, i01, i11)
    return mb^.finish()


def build_subd_material_orb(radius: Float32 = 0.88, levels: Int = 3) -> NativeMesh:
    """Builds an engine-neutral Catmull-Clark Subdivision PBR Material Preview Orb (`SubD_MaterialOrb`)
    with semi-sharp equatorial & base-pedestal studio creases."""
    var cage = SubdCage()
    var s = radius * Float32(1.18)
    var v0 = cage.add_vert(Vec3(-s, -s * Float32(0.92),  s))
    var v1 = cage.add_vert(Vec3( s, -s * Float32(0.92),  s))
    var v2 = cage.add_vert(Vec3( s,  s,  s))
    var v3 = cage.add_vert(Vec3(-s,  s,  s))
    var v4 = cage.add_vert(Vec3(-s, -s * Float32(0.92), -s))
    var v5 = cage.add_vert(Vec3( s, -s * Float32(0.92), -s))
    var v6 = cage.add_vert(Vec3( s,  s, -s))
    var v7 = cage.add_vert(Vec3(-s,  s, -s))

    cage.add_quad(v0, v1, v2, v3)
    cage.add_quad(v5, v4, v7, v6)
    cage.add_quad(v1, v5, v6, v2)
    cage.add_quad(v4, v0, v3, v7)
    cage.add_quad(v3, v2, v6, v7)
    cage.add_quad(v4, v5, v1, v0)

    cage.set_edge_sharpness(v4, v5, Float32(2.2))
    cage.set_edge_sharpness(v5, v1, Float32(2.2))
    cage.set_edge_sharpness(v1, v0, Float32(2.2))
    cage.set_edge_sharpness(v0, v4, Float32(2.2))
    return subdivide_to_mesh(cage, levels=max(2, min(4, levels)), split_sharp_threshold=Float32(1.6))


def build_subd_catmull_torus(
    major_r: Float32 = 0.82,
    minor_r: Float32 = 0.30,
    levels: Int = 2,
) -> NativeMesh:
    """Builds an engine-neutral `12x8` Catmull-Clark Subdivision Torus (`SubD_CatmullTorus`)."""
    comptime NU: Int = 12
    comptime NV: Int = 8
    var cage = SubdCage()
    var two_pi = Float32(6.28318530718)
    for iu in range(NU):
        var u = Float32(iu) * two_pi / Float32(NU)
        var cu = cos(u)
        var su = sin(u)
        for iv in range(NV):
            var v = Float32(iv) * two_pi / Float32(NV)
            var rr = major_r + minor_r * cos(v)
            var yy = minor_r * sin(v)
            _ = cage.add_vert(Vec3(rr * cu, yy, rr * su))

    for iu in range(NU):
        var iu1 = (iu + 1) % NU
        for iv in range(NV):
            var iv1 = (iv + 1) % NV
            var a = iu * NV + iv
            var b = iu * NV + iv1
            var c = iu1 * NV + iv1
            var d = iu1 * NV + iv
            cage.add_quad(a, b, c, d)
            if iv == 0 or iv == (NV // 2):
                cage.set_edge_sharpness(a, d, Float32(1.35))
    return subdivide_to_mesh(cage, levels=max(1, min(3, levels)), split_sharp_threshold=Float32(1.8))


def generate_procedural_forge_model(preset: String, seed: Int = 1) -> NativeModel:
    """Generates an engine-neutral `NativeModel` via `flint.procedural` (`subd`, `arch_kit`, `foliage`)."""
    var mdl = NativeModel()
    if preset == "material_orb" or preset == "orb" or preset == "SubD_MaterialOrb":
        mdl.meshes.append(build_subd_material_orb())
    elif preset == "catmull_torus" or preset == "torus" or preset == "SubD_CatmullTorus":
        mdl.meshes.append(build_subd_catmull_torus())
    elif preset == "arch_bay" or preset == "colonnade" or preset == "ArchKit_Colonnade":
        mdl.meshes.append(build_colonnade_arcade_bay(6.0, 4.8, 2.4))
    elif preset == "reference_grid" or preset == "grid" or preset == "Studio_ReferenceGrid":
        mdl.meshes.append(build_studio_reference_grid_mesh())
    elif preset == "foliage_sample" or preset == "pine" or preset == "Botanical_FoliageSample":
        var tree = build_botanical_pine(max(1, seed), 5.4)
        mdl.meshes.append(tree[0].copy())
        mdl.meshes.append(tree[1].copy())
    elif preset == "oak":
        var tree = build_botanical_oak(max(1, seed), 6.0)
        mdl.meshes.append(tree[0].copy())
        mdl.meshes.append(tree[1].copy())
    elif preset == "bollard":
        mdl.meshes.append(build_street_bollard_and_grate())
    else:
        mdl.meshes.append(_make_beveled_unit_box_mesh())
    return mdl^


def bake_procedural_to_glb(preset: String, output_glb_path: String, seed: Int = 1) -> Tuple[Bool, Int, Int, Int]:
    """Generates `preset` via Procedural Forge and serializes a binary glTF 2.0 `.glb` to `output_glb_path`.
    Returns `(ok, vertex_count, triangle_count, glb_byte_size)`."""
    var mdl = generate_procedural_forge_model(preset, seed)
    var v_sum = 0
    var t_sum = 0
    for i in range(len(mdl.meshes)):
        v_sum += mdl.meshes[i].vertex_count()
        t_sum += len(mdl.meshes[i].indices) // 3
    var glb_bytes = encode_native_glb(mdl)
    var ok = True
    if output_glb_path.byte_length() > 0:
        ok = write_native_glb(output_glb_path, mdl)
    return (ok, v_sum, t_sum, len(glb_bytes))


# ============================================================================
# 6. Docked `Flint Studio` Editor UI (`UiContext` + `DockSpace` -> `UiDrawList`)
# ============================================================================

def _fmt_f32_2dp(v: Float32) -> String:
    if isnan(v):
        return "nan"
    if isinf(v):
        return "-inf" if v < Float32(0.0) else "inf"
    var neg = v < Float32(0.0)
    var av = -v if neg else v
    if abs(Float64(av)) >= 1.0e15:
        return ("-" if neg else "") + String(av)
    var scaled = Int(av * Float32(100.0) + Float32(0.5))
    if scaled == 0:
        neg = False
    var whole = scaled // 100
    var frac = scaled % 100
    var sign = "-" if neg else ""
    var fstr = ("0" + String(frac)) if frac < 10 else String(frac)
    return sign + String(whole) + "." + fstr


def _quat_to_euler_deg(q_in: Quat) -> Vec3:
    var q = q_in.normalized()
    var rad_to_deg = Float32(180.0) / PI
    var sinr_cosp = Float32(2.0) * (q.w * q.x + q.y * q.z)
    var cosr_cosp = Float32(1.0) - Float32(2.0) * (q.x * q.x + q.y * q.y)
    var rx = det_atan2(sinr_cosp, cosr_cosp) * rad_to_deg

    var sinp = max(Float32(-1.0), min(Float32(1.0), Float32(2.0) * (q.w * q.y - q.z * q.x)))
    var ry = asin(sinp) * rad_to_deg

    var siny_cosp = Float32(2.0) * (q.w * q.z + q.x * q.y)
    var cosy_cosp = Float32(1.0) - Float32(2.0) * (q.y * q.y + q.z * q.z)
    var rz = det_atan2(siny_cosp, cosy_cosp) * rad_to_deg
    return Vec3(rx, ry, rz)


def _euler_deg_to_quat(deg: Vec3) -> Quat:
    var half_d2r = (PI / Float32(180.0)) * Float32(0.5)
    var cx = cos(deg.x * half_d2r)
    var sx = sin(deg.x * half_d2r)
    var cy = cos(deg.y * half_d2r)
    var sy = sin(deg.y * half_d2r)
    var cz = cos(deg.z * half_d2r)
    var sz = sin(deg.z * half_d2r)
    return Quat(
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    ).normalized()


def _studio_hierarchy_row(
    mut studio: FlintStudioState,
    row: HierarchyRow,
    ent_name: String,
    row_w: Float32,
    display_order: List[Int],
) -> Tuple[Bool, Int]:
    """Renders an indented collapsible hierarchy row with disclosure toggle, Ctrl/Shift selection,
    and drag-and-drop reparenting source/target. Returns `(dropped_here, dropped_child_id)`."""
    ref ui = studio.ui_ctx
    var row_h = ui.theme.row_height
    var wid = ui.widget_id("hier_row_" + String(row.entity_id))

    if len(ui.scroll_stack) > 0:
        var top_i = len(ui.box_stack) - 1
        var cy = ui.box_stack[top_i].cursor_y
        var sc = ui.scroll_stack[len(ui.scroll_stack) - 1]
        if cy + row_h < sc.view_rect.y or cy > sc.view_rect.bottom():
            _ = ui._alloc_item_rect(wid, row_w, row_h)
            ui.culled_items_this_frame += 1
            return (False, -1)

    var r = ui._alloc_item_rect(wid, row_w, row_h)
    var indent = Float32(row.depth * 14)
    var is_pri = studio.selected_entity_id == row.entity_id
    var is_sel = studio.is_selected(row.entity_id)
    var ptr_blocked = ui._blocked_by_modal() or ui._is_pointer_blocked_by_popup(wid)

    var bg_col = UInt32(0)
    if is_pri:
        bg_col = ui.theme.accent
    elif is_sel:
        bg_col = pack_rgba8(48, 86, 138, 220)
    elif (not ptr_blocked) and r.contains(ui.mouse_x, ui.mouse_y):
        bg_col = ui.theme.bg_widget_hover

    if bg_col != 0:
        ui.draw_list.add_rect(r.x, r.y, r.w, r.h, bg_col, 3.0)

    var disc_x = r.x + indent + Float32(4.0)
    var disc_r = _RectF(disc_x, r.y, Float32(16.0), r.h)
    var hovered = (not ptr_blocked) and r.contains(ui.mouse_x, ui.mouse_y)
    if hovered:
        ui.hovered_id = wid
        ui.hovered_rect = ui._clip_hovered_rect(r)

    if row.has_children:
        var ax = disc_x + Float32(7.0)
        var ay = r.center_y()
        if row.expanded:
            ui.draw_list.add_line(ax - Float32(4.0), ay - Float32(2.0), ax, ay + Float32(3.0), 1.6, ui.theme.text_primary)
            ui.draw_list.add_line(ax, ay + Float32(3.0), ax + Float32(4.0), ay - Float32(2.0), 1.6, ui.theme.text_primary)
        else:
            ui.draw_list.add_line(ax - Float32(2.0), ay - Float32(4.0), ax + Float32(3.0), ay, 1.6, ui.theme.text_primary)
            ui.draw_list.add_line(ax + Float32(3.0), ay, ax - Float32(2.0), ay + Float32(4.0), 1.6, ui.theme.text_primary)

    if hovered and ui.mouse_pressed_l:
        if row.has_children and disc_r.contains(ui.mouse_x, ui.mouse_y):
            _ = studio.toggle_hierarchy_expanded(row.entity_id)
        else:
            studio.select_entity(
                row.entity_id,
                multi_select=ui.key_ctrl,
                range_select=ui.key_shift,
                visible_order=display_order,
            )

    var mark = "* " if is_pri else ("+ " if is_sel else "")
    var txt_col = pack_rgba8(14, 18, 24, 255) if is_pri else ui.theme.text_primary
    ui._draw_text(
        mark + ent_name,
        disc_x + Float32(16.0),
        r.y + Float32(3.0),
        ui.theme.font_size,
        txt_col,
    )

    _ = ui.begin_drag_source("STUDIO_ENTITY", row.entity_id, ent_name)
    var drop = ui.accept_drag_target("STUDIO_ENTITY")
    if drop[0] and drop[1] != row.entity_id:
        return (True, drop[1])
    return (False, -1)


def populate_default_studio_scene(mut studio: FlintStudioState):
    """Populates the engine-neutral procedural SubD + ArchKit + Botanical Studio showcase into `app.world`."""
    _ = studio.spawn_entity(
        "Studio_ReferenceGrid",
        position=Vec3(0.0, 0.0, 0.0),
        scale=Vec3(1.0, 1.0, 1.0),
        collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(18.0, 0.05, 18.0),
        mass=0.0,
        base_color=Vec3(0.26, 0.31, 0.38),
        roughness=0.58,
        metallic=0.05,
        mesh_id=0,
        record_undo=False,
    )
    var orb_id = studio.spawn_entity(
        "SubD_MaterialOrb",
        position=Vec3(0.0, 0.92, 0.0),
        rotation=Quat(0.0, sin(Float32(-0.24)), 0.0, cos(Float32(-0.24))).normalized(),
        scale=Vec3(1.12, 1.12, 1.12),
        collider_kind=COLLIDER_SPHERE,
        collider_half_extents=Vec3(0.88, 0.88, 0.88),
        mass=0.0,
        base_color=Vec3(0.84, 0.22, 0.18),
        roughness=0.14,
        metallic=0.78,
        mesh_id=1,
        record_undo=False,
    )
    var orb_e = studio.resolve_entity(orb_id)
    if orb_e.valid():
        var st_opt = studio.app.world.get_ptr_mut[StudioStyleComponent](orb_e)
        if st_opt:
            st_opt.value()[unsafe_offset=0].clearcoat = Float32(0.88)
            st_opt.value()[unsafe_offset=0].clearcoat_roughness = Float32(0.05)
        var mo_orb = studio.app.world.get_ptr_mut[MaterialOverrideComponent](orb_e)
        if mo_orb:
            mo_orb.value()[unsafe_offset=0].clearcoat = Float32(0.88)
            mo_orb.value()[unsafe_offset=0].clearcoat_roughness = Float32(0.05)

    _ = studio.spawn_entity(
        "SubD_CatmullTorus",
        position=Vec3(3.1, 2.15, 1.2),
        rotation=Quat(sin(Float32(0.32)), sin(Float32(0.28)), 0.0, cos(Float32(0.42))).normalized(),
        scale=Vec3(1.05, 1.05, 1.05),
        collider_kind=COLLIDER_SPHERE,
        collider_half_extents=Vec3(0.85, 0.85, 0.85),
        mass=8.0,  # Dynamic body: drops & bounces onto reference grid on 1-Click PIE (`F5`)!
        base_color=Vec3(0.20, 0.64, 0.94),
        roughness=0.18,
        metallic=0.82,
        mesh_id=2,
        record_undo=False,
    )
    _ = studio.spawn_entity(
        "ArchKit_Colonnade",
        position=Vec3(0.0, 0.0, 6.0),
        scale=Vec3(1.35, 1.25, 1.25),
        collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(4.0, 3.0, 1.5),
        mass=0.0,
        base_color=Vec3(0.76, 0.73, 0.68),
        roughness=0.48,
        metallic=0.08,
        mesh_id=3,
        record_undo=False,
    )
    var foliage_id = studio.spawn_entity(
        "Botanical_FoliageSample",
        position=Vec3(-4.6, 0.0, 2.4),
        scale=Vec3(0.92, 0.92, 0.92),
        collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(0.9, 2.6, 0.9),
        mass=0.0,
        base_color=Vec3(0.34, 0.23, 0.14),
        roughness=0.84,
        metallic=0.0,
        mesh_id=4,
        record_undo=False,
    )
    _ = studio.spawn_entity(
        "Botanical_FoliageCanopy",
        position=Vec3(0.0, 0.0, 0.0),
        scale=Vec3(1.0, 1.0, 1.0),
        collider_kind=COLLIDER_NONE,
        mass=0.0,
        base_color=Vec3(0.18, 0.46, 0.22),
        roughness=0.68,
        metallic=0.0,
        mesh_id=5,
        parent_id=foliage_id,
        record_undo=False,
    )
    studio.select_entity(orb_id)
    studio.look_at(Vec3(-5.6, 3.75, -7.4), Vec3(0.15, 0.95, 1.1))
    upload_default_studio_gpu_assets(studio)


def upload_default_studio_gpu_assets(mut studio: FlintStudioState):
    """Registers dedicated non-recycled default mesh materials, showcase meshes (Tier 0 & Tier 1/2), and Bruneton sky onto `ScalableRenderer`."""
    var rend_opt = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    if not rend_opt:
        return
    ref renderer = rend_opt.value()[unsafe_offset=0]

    var n_ents = len(studio.app.world.entities)
    studio.gpu_materials.clear_entity_slots()

    if not studio.gpu_materials.uploaded:
        # Dedicated, immutable, non-recycled default mesh materials (never aliased to `entity_pbr_mat_ids`).
        var mat_grid_id = renderer.register_material(
            MaterialAsset(base_color=Vec3(0.26, 0.31, 0.38), roughness=0.58, metallic=0.05)
        )
        var mat_orb = MaterialAsset(base_color=Vec3(0.84, 0.22, 0.18), roughness=0.14, metallic=0.78)
        mat_orb.clearcoat = Float32(0.88)
        mat_orb.clearcoat_roughness = Float32(0.05)
        var mat_orb_id = renderer.register_material(mat_orb)
        var mat_torus_id = renderer.register_material(
            MaterialAsset(base_color=Vec3(0.20, 0.64, 0.94), roughness=0.18, metallic=0.82)
        )
        var mat_colonnade_id = renderer.register_material(
            MaterialAsset(base_color=Vec3(0.76, 0.73, 0.68), roughness=0.48, metallic=0.08)
        )
        var mat_trunk_id = renderer.register_material(
            MaterialAsset(base_color=Vec3(0.34, 0.23, 0.14), roughness=0.84, metallic=0.0)
        )
        var mat_foliage_id = renderer.register_material(
            MaterialAsset(base_color=Vec3(0.18, 0.46, 0.22), roughness=0.68, metallic=0.0)
        )
        studio.gpu_materials.mat_trunk_id = mat_trunk_id
        studio.gpu_materials.mat_foliage_id = mat_foliage_id

        studio.gpu_materials.grid_mesh_id = renderer.register_mesh(
            build_studio_reference_grid_mesh(36.0, 24), mat_grid_id, build_lods=False
        )
        studio.gpu_materials.orb_mesh_id = renderer.register_mesh(
            build_subd_material_orb(0.88, 3), mat_orb_id, build_lods=True
        )
        studio.gpu_materials.torus_mesh_id = renderer.register_mesh(
            build_subd_catmull_torus(0.82, 0.30, 2), mat_torus_id, build_lods=True
        )
        studio.gpu_materials.colonnade_mesh_id = renderer.register_mesh(
            build_colonnade_arcade_bay(6.0, 4.8, 2.4), mat_colonnade_id, build_lods=True
        )
        var pn = build_botanical_pine(7, 5.4)
        studio.gpu_materials.foliage_trunk_id = renderer.register_mesh(
            pn[0], studio.gpu_materials.mat_trunk_id, build_lods=False
        )
        studio.gpu_materials.foliage_canopy_id = renderer.register_mesh(
            pn[1], studio.gpu_materials.mat_foliage_id, build_lods=False
        )
        studio.gpu_materials.uploaded = True

    # Bind registered showcase mesh IDs back onto matching default showcase entities
    for i in range(n_ents):
        var e = studio.resolve_entity(i)
        if not e.valid():
            continue
        var mr_opt = studio.app.world.get_ptr_mut[MeshRendererComponent](e)
        if not mr_opt or not mr_opt.value()[unsafe_offset=0].mesh_ref.is_empty():
            continue
        var meta_opt = studio.app.world.get_ptr[StudioEntityMetaComponent](e)
        if not meta_opt:
            continue
        var nm = meta_opt.value()[unsafe_offset=0].name
        if nm == "Studio_ReferenceGrid":
            mr_opt.value()[unsafe_offset=0].mesh_id = studio.gpu_materials.grid_mesh_id
        elif nm == "SubD_MaterialOrb":
            mr_opt.value()[unsafe_offset=0].mesh_id = studio.gpu_materials.orb_mesh_id
        elif nm == "SubD_CatmullTorus":
            mr_opt.value()[unsafe_offset=0].mesh_id = studio.gpu_materials.torus_mesh_id
        elif nm == "ArchKit_Colonnade":
            mr_opt.value()[unsafe_offset=0].mesh_id = studio.gpu_materials.colonnade_mesh_id
        elif nm == "Botanical_FoliageSample":
            mr_opt.value()[unsafe_offset=0].mesh_id = studio.gpu_materials.foliage_trunk_id
        elif nm == "Botanical_FoliageCanopy":
            mr_opt.value()[unsafe_offset=0].mesh_id = studio.gpu_materials.foliage_canopy_id

    if renderer.uses_vulkan():
        ref gpu = renderer.gpu_slot[0]
        var atm = BrunetonAtmosphereParams()
        atm.sun_dir = Vec3(0.46, 0.58, -0.67).normalized()
        atm.cloud_coverage = Float32(0.28)
        _ = gpu.set_bruneton_atmosphere(atm, enable_aerial=True)

        gpu.display_mode = AAA_DISPLAY_AGX_REC709
        gpu.enable_vsm = True
        gpu.vsm_scene_center = Vec3(0.0, 1.5, 2.0)
        gpu.vsm_ortho_half_extent = Float32(24.0)
        gpu.enable_rt_shadows = gpu.vk.has_ray_query
        gpu.enable_rt_reflections = gpu.vk.has_ray_query
        gpu.enable_rt_gi = gpu.vk.has_ray_query
        gpu.gi_rays_per_pixel = 6
        gpu.svgf_passes = 1
        gpu.exposure = Float32(1.08)
        gpu.bloom_intensity = Float32(0.08)
    studio._sync_compat_view()


def _scene_display_name(path: String) -> String:
    if path.byte_length() == 0:
        return String("Untitled")
    var start = 0
    for i in range(path.byte_length()):
        if path.as_bytes()[i] == 47:
            start = i + 1
    return String(path[byte=start:])


def _emit_play_banner(mut studio: FlintStudioState, vp: RectF):
    """A strip across the top of the viewport while Play runs, so edits made now are never a surprise."""
    if studio.pie.mode == PIE_STOPPED:
        return
    var th = studio.studio_theme.copy()
    var paused = studio.pie.mode == PIE_PAUSED
    var col = th.play_paused if paused else th.play_mode
    var bh = Float32(22.0)
    studio.ui_ctx.draw_list.add_rect(vp.x, vp.y, vp.w, bh, col)
    var msg = String("PAUSED") if paused else String("PLAYING")
    msg += "  -  changes made during Play are discarded on Stop (Shift+F5)"
    studio.ui_ctx._draw_text(msg, vp.x + Float32(10.0), vp.y + Float32(4.0), Float32(13.0), th.text_on_accent)


def _emit_status_bar(mut studio: FlintStudioState, sw: Float32, sh: Float32):
    """Bottom status bar: scene + dirty state, mode, selection, tool, and the latest message
    (click it to open the console)."""
    var th = studio.studio_theme.copy()
    var y = sh - STUDIO_STATUS_BAR_H
    studio.ui_ctx.draw_list.add_rect(Float32(0.0), y, sw, STUDIO_STATUS_BAR_H, th.l1_shell, 0.0, 1.0, th.border_subtle)
    var fs = Float32(12.0)
    var ty = y + Float32(4.0)
    var left = _scene_display_name(studio.scene_file_path)
    if studio.dirty:
        left += "  *unsaved"
    var mode = String("Editing")
    if studio.pie.mode == PIE_PLAYING:
        mode = "Playing"
    elif studio.pie.mode == PIE_PAUSED:
        mode = "Paused"
    var tool = String("Move")
    if studio.gizmo_mode == GIZMO_MODE_ROTATE:
        tool = "Rotate"
    elif studio.gizmo_mode == GIZMO_MODE_SCALE:
        tool = "Scale"
    elif studio.gizmo_mode == GIZMO_MODE_SELECT:
        tool = "Select"
    var mid = mode + "   |   " + String(studio.selected_count()) + " selected   |   " + tool
    mid += " (" + ("Local" if studio.gizmo_local_space else "World") + ")"
    if studio.snap_enabled:
        mid += "   |   Snap " + String(studio.grid_snap_m) + " m"
    var x = Float32(10.0)
    studio.ui_ctx._draw_text(left, x, ty, fs, th.text_primary if studio.dirty else th.text_secondary)
    x += studio.ui_ctx._measure_text(left, fs) + Float32(24.0)
    studio.ui_ctx._draw_text(mid, x, ty, fs, th.text_secondary)
    x += studio.ui_ctx._measure_text(mid, fs) + Float32(24.0)
    var last = String("Console (F1 for shortcuts)")
    var last_col = th.text_muted
    if len(studio.console_log) > 0:
        last = studio.console_log[len(studio.console_log) - 1]
        if last.startswith("[error]"):
            last_col = th.status_danger
        elif last.startswith("[warn]"):
            last_col = th.status_warn
    var lw = studio.ui_ctx._measure_text(last, fs)
    var lx = max(x, sw - lw - Float32(12.0))
    studio.ui_ctx._draw_text(last, lx, ty, fs, last_col)
    var blocked = studio.ui_ctx.modal_depth > 0 or studio.ui_ctx.had_modal_prev_frame or studio.confirm_modal_open
    if (
        not blocked
        and studio.ui_ctx.mouse_pressed_l
        and studio.ui_ctx.mouse_y >= y
        and studio.ui_ctx.mouse_x >= lx - Float32(6.0)
    ):
        studio.console_open = True


def _emit_studio_modals(mut studio: FlintStudioState):
    """File (Open / Save As), Console, Keyboard shortcuts and first-run Welcome dialogs."""
    if studio.file_modal_open and not studio.confirm_modal_open:
        var title = String("Save Scene As") if studio.file_modal_save_as else String("Open Scene")
        if studio.ui_ctx.begin_modal("file_modal", title, Float32(520.0), Float32(330.0), studio.file_modal_open):
            studio.ui_ctx.label("Scene path (relative to the project, or absolute):", studio.studio_theme.text_secondary)
            var path_buf = studio.file_modal_path
            _ = studio.ui_ctx.text_input("file_modal_path", path_buf, Float32(488.0), "scenes/level.flscnb")
            studio.file_modal_path = path_buf
            if len(studio.recent_scenes) > 0:
                studio.ui_ctx.label("Recent:", studio.studio_theme.text_muted)
                var pick = -1
                for i in range(min(6, len(studio.recent_scenes))):
                    if studio.ui_ctx.button("recent_" + String(i), studio.recent_scenes[i], 488.0, 22.0):
                        pick = i
                if pick >= 0:
                    studio.file_modal_path = studio.recent_scenes[pick]
            studio.ui_ctx.begin_box("file_modal_btns", FlexStyle.row(gap=8.0, pad=2.0))
            var go = studio.ui_ctx.button("file_modal_ok", "Save" if studio.file_modal_save_as else "Open", 96.0, 26.0)
            if go or studio.ui_ctx.key_enter_pressed:
                _ = studio.submit_file_modal()
            if studio.ui_ctx.button("file_modal_cancel", "Cancel", 86.0, 26.0):
                studio.file_modal_open = False
            studio.ui_ctx.end_box()
            studio.ui_ctx.end_modal()

    if studio.console_open:
        if studio.ui_ctx.begin_modal("console_modal", "Console", Float32(640.0), Float32(380.0), studio.console_open):
            var n = len(studio.console_log)
            var row_h = studio.ui_ctx.theme.row_height
            studio.ui_ctx.begin_scroll_view("console_scroll", Float32(608.0), Float32(280.0), row_h * Float32(max(1, n)))
            if n == 0:
                studio.ui_ctx.label("No messages yet.", studio.studio_theme.text_muted)
            var i = n - 1
            while i >= 0:  # newest first
                var line = studio.console_log[i]
                var col = studio.studio_theme.text_primary
                if line.startswith("[error]"):
                    col = studio.studio_theme.status_danger
                elif line.startswith("[warn]"):
                    col = studio.studio_theme.status_warn
                studio.ui_ctx.label(line, col)
                i -= 1
            studio.ui_ctx.end_scroll_view()
            studio.ui_ctx.begin_box("console_btns", FlexStyle.row(gap=8.0, pad=2.0))
            if studio.ui_ctx.button("console_clear", "Clear", 86.0, 26.0):
                studio.console_log.clear()
            if studio.ui_ctx.button("console_close", "Close", 86.0, 26.0):
                studio.console_open = False
            studio.ui_ctx.end_box()
            studio.ui_ctx.end_modal()

    if studio.help_open or studio.welcome_open:
        var welcome = studio.welcome_open and not studio.help_open
        var open_flag = True
        var title = String("Welcome to Flint Studio") if welcome else String("Keyboard Shortcuts")
        if studio.ui_ctx.begin_modal("help_modal", title, Float32(560.0), Float32(470.0), open_flag):
            var mc = studio.studio_theme.text_muted
            if welcome:
                studio.ui_ctx.label("Build a scene, press F5 to play it, Shift+F5 to stop.")
                studio.ui_ctx.label("Ctrl+K opens the command palette: every command is there.", mc)
            var lines = List[String]()
            lines.append("Q / W / E / R      Select / Move / Rotate / Scale tool")
            lines.append("X                  Local / World gizmo space")
            lines.append("G                  Grid snap on/off")
            lines.append("F                  Frame selection")
            lines.append("Esc / right-click  Cancel a gizmo drag")
            lines.append("Click / drag       Select / marquee select (Shift or Ctrl adds)")
            lines.append("Arrows             Nudge selection; walk the Hierarchy when hovered")
            lines.append("Wheel / MMB drag   Dolly / pan     Alt+LMB orbit     RMB+WASDQE fly")
            lines.append("Ctrl+Z / Ctrl+Y    Undo / Redo")
            lines.append("Ctrl+D  Ctrl+C  Ctrl+V  Del   Duplicate  Copy  Paste  Delete")
            lines.append("Ctrl+A / F2        Select all / Rename")
            lines.append("Ctrl+S  Ctrl+Shift+S  Ctrl+O  Ctrl+Shift+O  Ctrl+N   Save / Save As / Load / Open / New")
            lines.append("F5 / F6 / F7 / Shift+F5   Play / Pause / Step / Stop")
            lines.append("Ctrl+B             Reload gameplay .so")
            lines.append("Ctrl+K             Command palette        F1  This list")
            for i in range(len(lines)):
                studio.ui_ctx.label(lines[i], mc if (welcome and i > 6) else studio.studio_theme.text_secondary)
            if studio.ui_ctx.button("help_close", "Got it" if welcome else "Close", 96.0, 26.0):
                open_flag = False
            studio.ui_ctx.end_modal()
        if not open_flag:
            if welcome:
                studio.dismiss_welcome()
            studio.help_open = False
            studio.welcome_open = False


def emit_studio_ui_draw_list(
    mut ui: UiDrawList,
    mut studio: FlintStudioState,
    font_slot: Int,
    font: SdfFont,
    screen_w: Int,
    screen_h: Int,
    gpu_ms: Float64 = 0.0,
    mounted_packs: Int = 0,
    mcp_clients: Int = 0,
    input: InputState = InputState(),
    dt: Float32 = 1.0 / 60.0,
):
    """Builds the `Flint Studio v5` visual editor UI using `DockSpace` + `UiContext`
    with multi-select, collapsible indented hierarchy tree + drag-and-drop reparenting,
    centroid multi-select gizmos, and single/bulk `FieldVisitor` Inspector."""
    studio._normalize_selection_from_primary()
    studio._prune_selection()

    var ui_sc = studio.effective_ui_scale()
    var sw = Float32(max(320, screen_w)) / ui_sc
    var sh = Float32(max(240, screen_h)) / ui_sc
    studio._last_screen_w = sw
    studio._last_screen_h = sh

    if (not studio.ui_ctx.font.valid) and font.valid:
        studio.ui_ctx.font = font.copy()
    studio.ui_ctx.font_slot = font_slot
    studio.ui_ctx.dock_style_v2 = True
    studio.ui_ctx.set_studio_theme(studio.studio_theme.copy())
    studio.ui_ctx.set_anim_mode(studio.anim_mode)
    studio.ui_ctx.theme.font_size = Float32(13.0)
    studio.ui_ctx.theme.row_height = Float32(22.0)

    var scaled_input = input.copy()
    if abs(ui_sc - Float32(1.0)) > Float32(1.0e-4):
        scaled_input.last_mouse_x = input.last_mouse_x / ui_sc
        scaled_input.last_mouse_y = input.last_mouse_y / ui_sc
        scaled_input.delta_x = input.delta_x / ui_sc
        scaled_input.delta_y = input.delta_y / ui_sc
    studio.ui_ctx.begin_frame(scaled_input, sw, sh, dt)
    if studio.ui_ctx.mouse_pressed_l or studio.ui_ctx.mouse_released_l:
        studio.ui_edit_seq += 1
        studio._edit_focus_id = UInt64(0)  # a click ends any typing interaction
    if studio.input_rebind_active:
        if input.key_pressed(UI_KEY_ESCAPE):
            studio.cancel_input_rebind()
        else:
            for k in range(512):
                if (k >= 340 and k <= 347) or k == UI_KEY_ESCAPE:
                    continue
                if input.key_pressed(k):
                    var mods = MOD_NONE
                    if input.key_down(341) or input.key_down(345):
                        mods |= MOD_CTRL
                    if input.key_down(340) or input.key_down(344):
                        mods |= MOD_SHIFT
                    if input.key_down(342) or input.key_down(346):
                        mods |= MOD_ALT
                    var tgt_ctx = studio.input_rebind_context
                    var tgt_act = studio.input_rebind_action
                    if studio.input_map.rebind_key(tgt_ctx, tgt_act, k, mods):
                        studio.push_toast("Rebound " + tgt_act + " -> " + _keycode_display_name(k), 0, 2.0)
                    studio.cancel_input_rebind()
                    break
    studio.ui_ctx.set_keyboard_capture(
        studio.input_rebind_active
        or studio.command_palette.is_open
        or studio.agent_panel.wants_keyboard()
        or studio.code_drawer.has_keyboard_focus
    )

    var overlay_blocked = (
        studio.confirm_modal_open
        or studio.ui_ctx.modal_depth > 0
        or studio.ui_ctx.had_modal_prev_frame
        or studio.command_palette.is_open
    )

    # 0. Slim 32px Unified Studio Header Bar (`0..STUDIO_HEADER_BAR_H`)
    var scene_title = studio.scene_file_path
    var last_sl = scene_title.rfind("/")
    if last_sl >= 0 and last_sl + 1 < scene_title.byte_length():
        var sub_title = String(scene_title[byte=last_sl + 1 :])
        scene_title = sub_title^
    var hdr_build_state = (
        2
        if studio.code_drawer.build_status == BUILD_STATUS_ERROR
        else (1 if studio.code_drawer.build_status == BUILD_STATUS_BUILDING else 0)
    )
    var show_win_ctrls = False
    var win_is_max = False
    # Window controls depend only on client-side decorations, not on the reduced-motion setting.
    var wc_opt = studio.app.resources.get_ptr[WindowStateResource]()
    if wc_opt and wc_opt.value()[unsafe_offset=0].initialized and len(wc_opt.value()[unsafe_offset=0].windows) > 0:
        show_win_ctrls = wc_opt.value()[unsafe_offset=0].windows[0].uses_client_decorations()
        win_is_max = wc_opt.value()[unsafe_offset=0].windows[0].maximized
    var hdr_act = studio.header_bar.draw(
        studio.ui_ctx.draw_list,
        font_slot,
        studio.ui_ctx.font,
        studio.studio_theme,
        studio.ui_ctx.animator,
        sw,
        scene_title,
        studio.dirty,
        studio.workspace_mgr.active_preset,
        studio._aggregate_agent_status(),
        studio.agent_bus.pending_approval_count(),
        hdr_build_state,
        studio.ui_ctx.mouse_x,
        studio.ui_ctx.mouse_y,
        studio.ui_ctx.mouse_down_l and (not overlay_blocked),
        studio.ui_ctx.mouse_pressed_l and (not overlay_blocked),
        Float32(0.0),
        show_window_controls=show_win_ctrls,
        window_maximized=win_is_max,
    )
    if hdr_act.kind == HDR_ACT_WORKSPACE:
        _ = studio.apply_workspace_preset(hdr_act.int_val, sw, sh)
    elif hdr_act.kind == HDR_ACT_COMMAND_PALETTE:
        studio.open_command_palette(PALETTE_ALL)
    elif hdr_act.kind == HDR_ACT_AGENT_TOGGLE:
        _ = studio.execute_palette_command(PAL_CMD_AGENT_TOGGLE_PANEL)
    elif hdr_act.kind == HDR_ACT_THEME_CYCLE:
        studio.set_studio_theme((studio.studio_theme_idx + 1) % 4)
    elif hdr_act.kind == HDR_ACT_SAVE:
        _ = studio.execute_palette_command(PAL_CMD_SAVE_SCENE)
    elif hdr_act.kind == HDR_ACT_BUILD:
        _ = studio.execute_palette_command(PAL_CMD_BUILD_HOT_RELOAD)
    elif hdr_act.kind == HDR_ACT_MENU:
        studio.menu.bar.group = 0 if studio.menu.bar.group < 0 else -1
    elif hdr_act.kind == HDR_ACT_WIN_MINIMIZE:
        var wm_opt = studio.app.resources.get_ptr_mut[WindowStateResource]()
        if wm_opt and len(wm_opt.value()[unsafe_offset=0].windows) > 0:
            wm_opt.value()[unsafe_offset=0].windows[0].minimize()
    elif hdr_act.kind == HDR_ACT_WIN_MAXIMIZE:
        var wm_opt = studio.app.resources.get_ptr_mut[WindowStateResource]()
        if wm_opt and len(wm_opt.value()[unsafe_offset=0].windows) > 0:
            wm_opt.value()[unsafe_offset=0].windows[0].toggle_maximize()
    elif hdr_act.kind == HDR_ACT_WIN_DRAG_MOVE:
        var wm_opt = studio.app.resources.get_ptr_mut[WindowStateResource]()
        if wm_opt and len(wm_opt.value()[unsafe_offset=0].windows) > 0:
            wm_opt.value()[unsafe_offset=0].windows[0].start_interactive_move()
    elif hdr_act.kind == HDR_ACT_WIN_CLOSE:
        _ = studio.request_quit()

    # 1. Obsidian v2 DockSpace below Header Bar
    var dock_rect = RectF(
        Float32(0.0),
        STUDIO_HEADER_BAR_H,
        sw,
        max(Float32(64.0), sh - STUDIO_HEADER_BAR_H - STUDIO_STATUS_BAR_H),
    )
    studio.ui_ctx.update_dock_space(studio.dock, dock_rect)

    var vp_info = studio.dock.find_panel("SceneViewport")
    var vp = (
        vp_info[3]
        if vp_info[0] >= 0
        else RectF(Float32(240.0), STUDIO_HEADER_BAR_H + Float32(12.0), sw - Float32(520.0), sh - Float32(190.0))
    )
    # The scene renders over the whole window; `vp` is only the clip/input hole (see `viewport_transform`).
    var vxf = ViewportTransform.from_logical(vp, sw, sh, ui_sc)

    # 2. 3D Viewport Active Selection Gizmo + Visual Agent Theatre HUD + Floating ViewportHud
    var show_edit_gizmo = (studio.pie.mode != PIE_PLAYING) or studio.pie_camera_ejected
    var giz_piv = studio.selection_gizmo_pivot()
    if show_edit_gizmo and giz_piv[0] and studio.gizmo_mode != GIZMO_MODE_SELECT:
        # `ui` is in physical pixels (only `ui_ctx.draw_list` is scaled later), so emit the gizmo
        # in physical space with a scaled target size; `main.mojo` picks with the same numbers.
        var gvx = vp.x * ui_sc
        var gvy = vp.y * ui_sc
        var gvw = vp.w * ui_sc
        var gvh = vp.h * ui_sc
        ui.push_clip(Int(gvx), Int(gvy), Int(gvx + gvw), Int(gvy + gvh))
        emit_gizmo_draw_list(
            ui,
            giz_piv[1],
            studio.gizmo_mode,
            studio.active_gizmo_handle,
            Float32(0.0),
            Float32(0.0),
            vxf.proj_phys_w(),
            vxf.proj_phys_h(),
            studio.cam_pos,
            studio.cam_fwd,
            studio.cam_right,
            studio.cam_up,
            studio.fov_y_deg,
            target_px=GIZMO_TARGET_PX * ui_sc,
            gizmo_rot=studio.selection_gizmo_rotation(),
        )
        ui.pop_clip()

    # Marquee selection rectangle (physical pixels, clipped to the scene panel).
    if studio.marquee_active:
        var mq_x = min(studio.marquee_x0, studio.marquee_x1)
        var mq_y = min(studio.marquee_y0, studio.marquee_y1)
        var mq_w = abs(studio.marquee_x1 - studio.marquee_x0)
        var mq_h = abs(studio.marquee_y1 - studio.marquee_y0)
        var pclip = studio.viewport_xf.panel_phys()
        ui.push_clip(Int(pclip.x), Int(pclip.y), Int(pclip.x + pclip.w), Int(pclip.y + pclip.h))
        ui.add_rect(mq_x, mq_y, mq_w, mq_h, UInt32(0x22428BF5), Float32(0.0), Float32(1.5), UInt32(0xCC428BF5))
        ui.pop_clip()

    # Visual Agent Theatre 2D Viewport Overlays (presence pills, Bezier flow beams, PRIM_ARC reticles, inline [✓]/[✗])
    studio._last_theatre_hits.clear()
    if studio.agent_bus.has_active_visuals() and vp.w > Float32(40.0) and vp.h > Float32(40.0):
        var vp_mat = vxf.hud_view_proj(studio.cam_pos, studio.cam_fwd, studio.fov_y_deg, studio.near_z, Float32(1000.0))
        _ = emit_agent_viewport_hud_overlays_with_font(
            studio.agent_bus,
            studio.ui_ctx.draw_list,
            font_slot,
            studio.ui_ctx.font,
            vp_mat,
            vp.x,
            vp.y,
            vp.w,
            vp.h,
            studio.ui_time_sec,
            studio._last_theatre_hits,
        )
        if studio.ui_ctx.mouse_pressed_l and (not overlay_blocked):
            for hi in range(len(studio._last_theatre_hits)):
                var hit = studio._last_theatre_hits[hi]
                if hit.contains(studio.ui_ctx.mouse_x, studio.ui_ctx.mouse_y):
                    if hit.kind == HIT_KIND_APPROVE:
                        var a = AgentPanelAction(PANEL_ACT_APPROVE)
                        a.agent_id = hit.agent_id
                        a.approval_id = hit.approval_id
                        studio.apply_agent_panel_action(a)
                    elif hit.kind == HIT_KIND_REJECT:
                        var a = AgentPanelAction(PANEL_ACT_REJECT)
                        a.agent_id = hit.agent_id
                        a.approval_id = hit.approval_id
                        studio.apply_agent_panel_action(a)
                    elif hit.kind == HIT_KIND_CALLOUT_FOCUS and hit.entity_id >= 0:
                        var a = AgentPanelAction(PANEL_ACT_FOCUS_ENTITY)
                        a.entity_id = hit.entity_id
                        studio.apply_agent_panel_action(a)
                    elif hit.kind == HIT_KIND_PRESENCE:
                        studio.agent_bus.selected_agent_id = hit.agent_id
                        _ = WorkspacePresetManager.ensure_panel_visible(studio.dock, PANEL_AGENTS)
                    break

    # Floating 4-Cluster Translucent Viewport HUD + 3D OrientationGizmo + GPU Sparkline
    if studio.anim_mode == UI_ANIM_INSTANT and studio.viewport_hud.ring_count == 0:
        var base_ms = Float32(gpu_ms) if gpu_ms > 0.1 else Float32(4.2)
        studio.viewport_hud.seed_baseline_sparkline(base_ms)
    studio.viewport_hud.push_gpu_ms(Float32(gpu_ms))
    var vhud_pie = (
        VHUD_PIE_PLAYING
        if studio.pie.mode == PIE_PLAYING
        else (VHUD_PIE_PAUSED if studio.pie.mode == PIE_PAUSED else VHUD_PIE_STOPPED)
    )
    var vhud_tool = (
        VHUD_TOOL_ROTATE
        if studio.gizmo_mode == GIZMO_MODE_ROTATE
        else (
            VHUD_TOOL_SCALE
            if studio.gizmo_mode == GIZMO_MODE_SCALE
            else (VHUD_TOOL_SELECT if studio.gizmo_mode == GIZMO_MODE_SELECT else VHUD_TOOL_MOVE)
        )
    )
    var vhud_act = studio.viewport_hud.draw(
        studio.ui_ctx.draw_list,
        font_slot,
        studio.ui_ctx.font,
        studio.studio_theme,
        studio.ui_ctx.animator,
        vp,
        vhud_tool,
        studio.gizmo_local_space,
        studio.snap_enabled,
        studio.grid_snap_m,
        vhud_pie,
        studio.pie_camera_ejected,
        studio.camera_rig.mode == CAM_MODE_FREE_FLY,
        studio.cam_ortho,
        studio.show_navmesh_overlay,
        studio.show_collider_overlay,
        studio.show_grid_overlay,
        studio.cam_right,
        studio.cam_up,
        studio.cam_fwd,
        studio.selected_count(),
        "",
        studio.ui_ctx.mouse_x,
        studio.ui_ctx.mouse_y,
        studio.ui_ctx.mouse_down_l and (not overlay_blocked),
        studio.ui_ctx.mouse_pressed_l and (not overlay_blocked),
    )
    if vhud_act.kind == VHUD_ACT_TOOL_SELECT:
        studio.active_gizmo_handle = GIZMO_AXIS_NONE
        studio.gizmo_mode = GIZMO_MODE_SELECT
    elif vhud_act.kind == VHUD_ACT_TOOL_MOVE:
        studio.gizmo_mode = GIZMO_MODE_TRANSLATE
    elif vhud_act.kind == VHUD_ACT_TOOL_ROTATE:
        studio.gizmo_mode = GIZMO_MODE_ROTATE
    elif vhud_act.kind == VHUD_ACT_TOOL_SCALE:
        studio.gizmo_mode = GIZMO_MODE_SCALE
    elif vhud_act.kind == VHUD_ACT_TOGGLE_SPACE:
        studio.toggle_gizmo_space()
    elif vhud_act.kind == VHUD_ACT_TOGGLE_SNAP:
        studio.snap_enabled = not studio.snap_enabled
    elif vhud_act.kind == VHUD_ACT_PIE_PLAY:
        _ = studio.pie_play()
    elif vhud_act.kind == VHUD_ACT_PIE_PAUSE:
        _ = studio.pie_pause()
    elif vhud_act.kind == VHUD_ACT_PIE_STEP:
        _ = studio.pie_step(1, Float32(1.0 / 60.0))
    elif vhud_act.kind == VHUD_ACT_PIE_STOP:
        try:
            _ = studio.pie_stop()
        except:
            studio._abort_pie_session()
    elif vhud_act.kind == VHUD_ACT_PIE_EJECT:
        _ = studio.execute_palette_command(PAL_CMD_PIE_EJECT_CAMERA)
    elif vhud_act.kind == VHUD_ACT_CAM_MODE:
        studio.camera_rig.mode = (
            CAM_MODE_ORBIT_FOLLOW if studio.camera_rig.mode == CAM_MODE_FREE_FLY else CAM_MODE_FREE_FLY
        )
    elif vhud_act.kind == VHUD_ACT_TOGGLE_NAVMESH:
        _ = studio.execute_palette_command(PAL_CMD_TOGGLE_NAVMESH_OVERLAY)
    elif vhud_act.kind == VHUD_ACT_TOGGLE_COLLIDERS:
        studio.show_collider_overlay = not studio.show_collider_overlay
    elif vhud_act.kind == VHUD_ACT_TOGGLE_GRID:
        studio.show_grid_overlay = not studio.show_grid_overlay
    elif vhud_act.kind == VHUD_ACT_TOGGLE_PERF:
        studio.viewport_hud.perf_expanded = not studio.viewport_hud.perf_expanded
    elif vhud_act.kind == VHUD_ACT_SNAP_VIEW:
        studio._snap_camera_direction(vhud_act.snap_dir)
    elif vhud_act.kind == VHUD_ACT_TOGGLE_ORTHO:
        studio.toggle_ortho_camera()

    # Optional Gameplay HUD overlay (`WORKSPACE_PLAY` / `--gallery 06_play_hud`) inside SceneViewport below header & transform HUD
    if studio.gallery_play_hud_active and vp.w >= Float32(180.0) and vp.h >= Float32(130.0):
        var hud_w = min(Float32(252.0), max(Float32(160.0), vp.w - Float32(32.0)))
        var hud_h = Float32(66.0)
        var hx = vp.x + Float32(16.0)
        var hy = vp.y + Float32(52.0)
        studio.ui_ctx.draw_list.add_glass_panel(
            hx,
            hy,
            hud_w,
            hud_h,
            studio.studio_theme.l4_overlay,
            studio.studio_theme.radius_md,
            Float32(3.0),
            Float32(1.2),
            rgba8_with_alpha(studio.studio_theme.play_mode, 185),
            True,
            True,
        )
        studio.ui_ctx._draw_text(
            "HP 84/100",
            hx + Float32(12.0),
            hy + Float32(9.0),
            Float32(13.5),
            studio.studio_theme.text_primary,
        )
        studio.ui_ctx._draw_text(
            "SCORE 1250",
            hx + hud_w * Float32(0.54),
            hy + Float32(9.0),
            Float32(13.5),
            studio.studio_theme.accent_primary,
        )
        hud_bar(
            studio.ui_ctx,
            hx + Float32(12.0),
            hy + Float32(34.0),
            hud_w - Float32(24.0),
            Float32(18.0),
            84,
            100,
            studio.studio_theme.play_mode,
        )

    var vp_drop = studio.ui_ctx.accept_drag_target_rect("ASSET_ENTRY", vp)
    if vp_drop[0]:
        var drop_pos = studio._compute_viewport_drop_pos(studio.ui_ctx.mouse_x, studio.ui_ctx.mouse_y, vp)
        var d_low = FlintStudioState._ascii_lower(vp_drop[2])
        var tgt_eid = -1
        if d_low.endswith(".flmat") or d_low.endswith(".flrig") or d_low.endswith(".flanim"):
            if vp.w > Float32(1.0) and vp.h > Float32(1.0) and vp.contains(studio.ui_ctx.mouse_x, studio.ui_ctx.mouse_y):
                var ray = vxf.ray_logical(
                    studio.ui_ctx.mouse_x, studio.ui_ctx.mouse_y,
                    studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
                )
                var hit = studio.raycast_scene(ray.origin, ray.dir, Float32(250.0))
                if hit.hit and studio.is_alive_id(hit.entity_id):
                    tgt_eid = hit.entity_id
            if tgt_eid < 0:
                tgt_eid = studio.selected_entity_id
        _ = studio.activate_asset_entry(vp_drop[2], target_entity_id=tgt_eid, drop_pos=drop_pos, from_drop=True)

    # 3. Left Scene Hierarchy Outliner (`DockSpace` -> `"Hierarchy"`, Obsidian v2 Outliner + Agent Badges)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "Hierarchy"):
        var h_info = studio.dock.find_panel("Hierarchy")
        # Search covers collapsed branches, and range-select (Shift+click) walks the filtered rows the
        # user sees, not the whole tree.
        var rows = studio.hierarchy_rows_for_query(studio.outliner_state.search_query)
        var disp_order = List[Int](capacity=len(rows))
        var out_rows = List[OutlinerRowView](capacity=len(rows))
        for ri in range(len(rows)):
            var row = rows[ri]
            var eid = row.entity_id
            disp_order.append(eid)
            var ename = studio.entities[eid].name if eid < len(studio.entities) else ("Entity_" + String(eid))
            var v_row = OutlinerRowView(eid, ename)
            v_row.parent_id = studio.parent_of(eid)
            v_row.depth = row.depth
            v_row.kind = studio._classify_entity_kind(eid)
            v_row.has_children = row.has_children
            v_row.is_expanded = row.expanded
            v_row.is_selected = studio.is_selected(eid)
            v_row.is_primary = studio.selected_entity_id == eid
            v_row.visible = studio.entities[eid].visible if eid < len(studio.entities) else True
            v_row.locked = studio.is_entity_locked(eid)
            out_rows.append(v_row^)

        var sel_has_parent = False
        if studio.selected_entity_id >= 0 and studio.parent_of(studio.selected_entity_id) >= 0:
            sel_has_parent = True
        else:
            for si in range(len(studio.selected_entity_ids)):
                if studio.parent_of(studio.selected_entity_ids[si]) >= 0:
                    sel_has_parent = True
                    break

        var out_act = draw_outliner_panel_v2(
            studio.ui_ctx,
            h_info[3],
            studio.studio_theme,
            out_rows,
            "StudioWorld",
            studio.live_entity_count(),
            studio.selected_count(),
            sel_has_parent,
            studio.agent_bus,
            studio.outliner_state,
        )
        if out_act.unparent_selected:
            _ = studio.reparent_selected_to(-1, preserve_world_transform=True, record_undo=True)
        if out_act.parent_to_primary and studio.selected_count() >= 2 and studio.selected_entity_id >= 0:
            _ = studio.reparent_selected_to(studio.selected_entity_id, preserve_world_transform=True, record_undo=True)
        if out_act.toggle_expand_entity_id >= 0:
            _ = studio.toggle_hierarchy_expanded(out_act.toggle_expand_entity_id)
        if out_act.toggle_visible_entity_id >= 0 and studio.is_alive_id(out_act.toggle_visible_entity_id):
            var vid = out_act.toggle_visible_entity_id
            var vc = studio.entities[vid].copy()
            vc.visible = not vc.visible  # editor visibility only: it no longer disables gameplay logic
            _ = studio.commit_node_edit(vid, vc^, "Toggle Visibility")
        if out_act.toggle_lock_entity_id >= 0:
            var lid = out_act.toggle_lock_entity_id
            _ = studio.set_entity_locked(lid, not studio.is_entity_locked(lid))
        if out_act.select_entity_id >= 0:
            studio.select_entity(
                out_act.select_entity_id,
                multi_select=out_act.select_additive,
                range_select=out_act.select_range,
                visible_order=disp_order,
            )
        if out_act.focus_entity_id >= 0 and studio.is_alive_id(out_act.focus_entity_id):
            studio.select_entity(out_act.focus_entity_id)
            studio.frame_selection()
        if out_act.quick_spawn_clicked:
            studio.open_command_palette(PALETTE_SPAWN)
        if out_act.reparent_child_id >= 0 and out_act.reparent_target_id >= -1:
            if studio.is_selected(out_act.reparent_child_id) and studio.selected_count() > 1:
                _ = studio.reparent_selected_to(out_act.reparent_target_id, preserve_world_transform=True, record_undo=True)
            else:
                _ = studio.reparent_entity(out_act.reparent_child_id, out_act.reparent_target_id, preserve_world_transform=True, record_undo=True)

        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "Outliner"):
        studio.ui_ctx.label("Live Entities: " + String(studio.live_entity_count()))
        studio.ui_ctx.label("Selected: " + String(studio.selected_count()))
        studio.ui_ctx.label("State Hash: " + String(studio.state_hash()), studio.ui_ctx.theme.text_muted)
        studio.ui_ctx.end_dock_panel()

    # 4. Right Inspector (`DockSpace` -> `"Inspector"`, single or bulk multi-select)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "Inspector"):
        var i_info = studio.dock.find_panel("Inspector")
        var panel_w = max(Float32(180.0), i_info[3].w - Float32(16.0))
        var insp_h = max(Float32(100.0), i_info[3].h - Float32(28.0))
        studio.ui_ctx.begin_scroll_view("insp_scroll", 0.0, insp_h)
        var sel_cnt = studio.selected_count()
        if sel_cnt > 1 and studio.selected_entity_id >= 0 and studio.is_alive_id(studio.selected_entity_id):
            var pri_id = studio.selected_entity_id
            var pri_view = studio.entities[pri_id].copy()
            studio.ui_ctx.label(
                "Multi-Select: " + String(sel_cnt) + " entities (Primary #" + String(pri_id) + ": " + pri_view.name + ")",
                studio.ui_ctx.theme.accent,
            )
            var bulk_delta = Vec3()
            var mat_w = max(Float32(160.0), panel_w - Float32(20.0))
            if studio.ui_ctx.drag_float3("bulk_delta", "Move Delta", bulk_delta, speed=0.05, w=mat_w):
                studio.begin_edit_interaction()
                _ = studio.bulk_translate_selected(bulk_delta, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "bulk_delta"))

            var b_col = pri_view.base_color
            if studio.ui_ctx.drag_float3("bulk_col", "Bulk Albedo", b_col, speed=0.01, min_val=0.0, max_val=1.0, w=mat_w):
                studio.begin_edit_interaction()
                var f_col = InspectedField("base_color", FIELD_COLOR_RGB)
                f_col.v3_val = b_col
                _ = studio.bulk_patch_field("base_color", f_col, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "bulk_col"))

            var b_rgh = pri_view.roughness
            if studio.ui_ctx.slider_float("bulk_rgh", "Bulk Rough", b_rgh, Float32(0.04), Float32(1.0), w=mat_w):
                studio.begin_edit_interaction()
                var f_rgh = InspectedField("roughness", FIELD_F32)
                f_rgh.f_val = b_rgh
                _ = studio.bulk_patch_field("roughness", f_rgh, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "bulk_rgh"))

            var b_met = pri_view.metallic
            if studio.ui_ctx.slider_float("bulk_met", "Bulk Metal", b_met, Float32(0.0), Float32(1.0), w=mat_w):
                studio.begin_edit_interaction()
                var f_met = InspectedField("metallic", FIELD_F32)
                f_met.f_val = b_met
                _ = studio.bulk_patch_field("metallic", f_met, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "bulk_met"))

            var b_vis = pri_view.visible
            if studio.ui_ctx.checkbox("bulk_vis", "Bulk Visible", b_vis):
                var f_vis = InspectedField("visible", FIELD_BOOL)
                f_vis.b_val = b_vis
                _ = studio.bulk_patch_field("visible", f_vis)

            studio.ui_ctx.begin_box("bulk_rep_row", FlexStyle.row(gap=6.0, pad=0.0))
            var act_w = max(Float32(80.0), (panel_w - Float32(12.0)) * Float32(0.5))
            if studio.ui_ctx.button("bulk_parent_pri", "Parent->Pri", act_w, 22.0):
                _ = studio.reparent_selected_to(pri_id, preserve_world_transform=True, record_undo=True)
            if studio.ui_ctx.button("bulk_unparent", "Unparent", act_w, 22.0):
                _ = studio.reparent_selected_to(-1, preserve_world_transform=True, record_undo=True)
            studio.ui_ctx.end_box()

            studio.ui_ctx.begin_box("bulk_actions_row", FlexStyle.row(gap=6.0, pad=0.0))
            if studio.ui_ctx.button("btn_dup_sel", "Dup Selected", act_w, 24.0):
                _ = studio.duplicate_selected(record_undo=True)
            if studio.ui_ctx.button("btn_del_sel", "Del Selected", act_w, 24.0):
                _ = studio.despawn_selected(record_undo=True)
            studio.ui_ctx.end_box()
        elif studio.selected_entity_id >= 0 and studio.is_alive_id(studio.selected_entity_id):
            var eid = studio.selected_entity_id
            var vis = FieldVisitor()
            studio.entities[eid].inspect_fields(vis)
            var cur = studio.entities[eid].copy()
            var changed = False

            var shd_lbl = (
                "NPR Cel"
                if cur.shading_mode == SHADING_NPR_CEL
                else ("Stylized VFX" if cur.shading_mode == SHADING_STYLIZED_VFX else "PBR")
            )
            var name_w = max(Float32(140.0), panel_w - Float32(24.0))
            if studio.rename_focus_requested:
                studio.rename_focus_requested = False
                studio.ui_ctx.focused_id = studio.ui_ctx.widget_id("insp_name")
            if studio.ui_ctx.text_input("insp_name", cur.name, name_w, "Entity Name"):
                changed = True
            if studio.ui_ctx.toggle_switch("ent_enabled", "Enabled (#" + String(eid) + ")", cur.enabled, w=name_w):
                cur.visible = cur.enabled
                changed = True
            var par_lbl = (
                studio.entities[cur.parent_id].name
                if (cur.parent_id >= 0 and cur.parent_id < len(studio.entities) and studio.entities[cur.parent_id].alive)
                else "Root"
            )
            studio.ui_ctx.label("Parent: " + par_lbl, studio.ui_ctx.theme.text_muted)
            if cur.parent_id >= 0:
                if studio.ui_ctx.button("insp_unparent", "Unparent to Root", 136.0, 20.0):
                    _ = studio.reparent_entity(eid, -1, preserve_world_transform=True, record_undo=True)
                    cur = studio.entities[eid].copy()

            var shd_summary = studio.ui_ctx._truncate_text_ellipsis(
                "Shading: " + shd_lbl + " | " + String(len(vis.fields)) + " fields",
                max(Float32(80.0), panel_w - Float32(18.0)),
                studio.ui_ctx.theme.font_size,
            )
            studio.ui_ctx.label(
                shd_summary,
                studio.ui_ctx.theme.accent if cur.shading_mode != SHADING_PBR else studio.ui_ctx.theme.text_muted,
            )
            if (
                cur.shading_mode != SHADING_PBR
                or cur.anim_step_hz > Float32(0.0)
                or studio.stepped_clock.hitstop_remaining_sec > Float32(0.0)
            ):
                var eff_hz = cur.anim_step_hz if cur.anim_step_hz > Float32(0.0) else studio.stepped_clock.step_hz
                studio.ui_ctx.label(
                    "SteppedClock: "
                    + _fmt_f32_2dp(studio.stepped_clock.stepped_time_sec)
                    + "s @ "
                    + _fmt_f32_2dp(eff_hz)
                    + " Hz | Hitstop "
                    + _fmt_f32_2dp(studio.stepped_clock.hitstop_remaining_sec)
                    + "s",
                    studio.ui_ctx.theme.accent,
                )

            if studio.ui_ctx.tree_node("insp_trs", "Transform", default_open=True):
                var tr_w = max(Float32(160.0), panel_w - Float32(28.0))
                if studio.ui_ctx.drag_float3_v2("tr_pos", "Pos", cur.translation, speed=0.04, w=tr_w):
                    changed = True
                var rot_deg = _quat_to_euler_deg(cur.rotation)
                if studio.ui_ctx.drag_float3_v2("tr_rot", "Rot", rot_deg, speed=0.5, min_val=-360.0, max_val=360.0, w=tr_w):
                    cur.rotation = _euler_deg_to_quat(rot_deg)
                    changed = True
                if studio.ui_ctx.drag_float3_v2("tr_scl", "Scale", cur.scale, speed=0.02, min_val=0.05, max_val=50.0, w=tr_w):
                    changed = True
                studio.ui_ctx.tree_pop()

            if studio.ui_ctx.tree_node("insp_pbr", "PBR Material & Physics", default_open=True):
                var mat_w = max(Float32(160.0), panel_w - Float32(28.0))
                if studio.ui_ctx.color_edit3("mat_col", "Albedo", cur.base_color, min_val=0.0, max_val=1.0, w=mat_w):
                    changed = True
                if studio.ui_ctx.color_edit3("mat_emissive", "Emissive", cur.emissive, min_val=0.0, max_val=20.0, w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("mat_rgh", "Rough", cur.roughness, Float32(0.04), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("mat_met", "Metal", cur.metallic, Float32(0.0), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("mat_opacity", "Opacity", cur.opacity, Float32(0.0), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("mat_alpha_cut", "Alpha Cutoff", cur.alpha_cutoff, Float32(0.0), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("mat_clearcoat", "Clearcoat", cur.clearcoat, Float32(0.0), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("mat_cc_rough", "CC Rough", cur.clearcoat_roughness, Float32(0.04), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("mat_sheen", "Sheen", cur.sheen, Float32(0.0), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("mat_sss", "Subsurface", cur.subsurface, Float32(0.0), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float_v2("phy_mass", "Mass", cur.mass, Float32(0.0), Float32(100.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.toggle_switch("ent_vis", "Visible", cur.visible, w=mat_w):
                    cur.enabled = cur.visible
                    changed = True
                studio.ui_ctx.tree_pop()

            if studio.ui_ctx.tree_node("insp_col", "Collider", default_open=True):
                var col_w = max(Float32(160.0), panel_w - Float32(28.0))
                var col_kinds = List[String]()
                col_kinds.append("None")
                col_kinds.append("Sphere")
                col_kinds.append("Box")
                col_kinds.append("Capsule")
                col_kinds.append("Static Mesh")
                var ck_idx = max(0, min(4, cur.collider_kind))
                if studio.ui_ctx.combo_box("col_kind", "Kind", ck_idx, col_kinds, w=col_w):
                    cur.collider_kind = ck_idx
                    changed = True
                if cur.collider_kind == COLLIDER_SPHERE:
                    if studio.ui_ctx.drag_float("col_radius", "Radius", cur.collider_half_extents.x, speed=0.02, min_val=0.01, max_val=500.0, w=col_w):
                        cur.collider_half_extents.y = cur.collider_half_extents.x
                        cur.collider_half_extents.z = cur.collider_half_extents.x
                        changed = True
                elif cur.collider_kind == COLLIDER_CAPSULE:
                    if studio.ui_ctx.drag_float("col_radius", "Radius", cur.collider_half_extents.x, speed=0.02, min_val=0.01, max_val=500.0, w=col_w):
                        cur.collider_half_extents.z = cur.collider_half_extents.x
                        changed = True
                    if studio.ui_ctx.drag_float("col_half_h", "Half Height", cur.collider_half_extents.y, speed=0.02, min_val=0.01, max_val=500.0, w=col_w):
                        changed = True
                elif cur.collider_kind != COLLIDER_NONE:
                    if studio.ui_ctx.drag_float3("col_he", "Half Extents", cur.collider_half_extents, speed=0.02, min_val=0.01, max_val=500.0, w=col_w):
                        changed = True
                if cur.collider_kind != COLLIDER_NONE:
                    if studio.ui_ctx.checkbox("col_trigger", "Is Trigger", cur.is_trigger):
                        changed = True
                    if studio.ui_ctx.drag_float("col_friction", "Friction", cur.friction, speed=0.01, min_val=0.0, max_val=2.0, w=col_w):
                        changed = True
                    if studio.ui_ctx.drag_float("col_rest", "Restitution", cur.restitution, speed=0.01, min_val=0.0, max_val=1.0, w=col_w):
                        changed = True
                studio.ui_ctx.tree_pop()

            var npr_open = cur.shading_mode != SHADING_PBR or cur.anim_step_hz > Float32(0.0)
            var npr_node_id = "insp_npr_on" if npr_open else "insp_npr_off"
            if studio.ui_ctx.tree_node(npr_node_id, "NPR Cel / Stylized VFX & Timeline", default_open=npr_open):
                var npr_w = max(Float32(160.0), panel_w - Float32(20.0))
                studio.ui_ctx.begin_box("npr_mode_row", FlexStyle.row(gap=4.0, pad=0.0))
                if studio.ui_ctx.button("shd_pbr", "PBR", 58.0, 20.0):
                    cur.shading_mode = SHADING_PBR
                    changed = True
                if studio.ui_ctx.button("shd_cel", "NPR Cel", 74.0, 20.0):
                    cur.shading_mode = SHADING_NPR_CEL
                    changed = True
                if studio.ui_ctx.button("shd_vfx", "Stylized VFX", 94.0, 20.0):
                    cur.shading_mode = SHADING_STYLIZED_VFX
                    changed = True
                studio.ui_ctx.end_box()
                if cur.shading_mode == SHADING_NPR_CEL:
                    if studio.ui_ctx.slider_float("cel_out_w", "Outline px", cur.cel_outline_width_px, Float32(0.0), Float32(8.0), w=npr_w):
                        changed = True
                    if studio.ui_ctx.slider_float("cel_halo", "Hair Halo", cur.cel_hair_halo, Float32(0.0), Float32(1.5), w=npr_w):
                        changed = True
                    if studio.ui_ctx.drag_float3("cel_shd", "SSS Tint", cur.cel_shadow_tint, speed=0.01, min_val=0.0, max_val=1.5, w=npr_w):
                        changed = True
                elif cur.shading_mode == SHADING_STYLIZED_VFX:
                    if studio.ui_ctx.slider_float("vfx_ero", "Erosion", cur.vfx_erosion, Float32(0.0), Float32(1.0), w=npr_w):
                        changed = True
                    if studio.ui_ctx.slider_float("vfx_scr", "Scroll", cur.vfx_scroll_speed, Float32(0.0), Float32(12.0), w=npr_w):
                        changed = True
                if studio.ui_ctx.slider_float("anm_hz", "Step Hz", cur.anim_step_hz, Float32(0.0), Float32(30.0), w=npr_w):
                    changed = True
                if studio.ui_ctx.slider_float("anm_smr", "Smear X", cur.smear_stretch, Float32(0.55), Float32(3.2), w=npr_w):
                    changed = True
                studio.ui_ctx.tree_pop()

            if changed:
                studio.begin_edit_interaction()
                _ = studio.commit_node_edit(eid, cur^, "Inspector UI Edit", coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "insp_main"))
                cur = studio.entities[eid].copy()

            # M1 Content Inspector Sections (`Mesh/Material/.flmat`, `Animator`, `Rig/Skeleton/BoneSocket`, `Audio`, `Prefab`)
            var sel_e = studio.resolve_entity(eid)
            if sel_e.valid():
                var insp_w = max(Float32(160.0), panel_w - Float32(28.0))
                if cur.has_mesh_renderer or (not cur.override_material_ref.is_empty()):
                    var show_asset_open = (
                        (not cur.mesh_ref.is_empty())
                        or (not cur.material_ref.is_empty())
                        or (not cur.override_material_ref.is_empty())
                    )
                    if studio.ui_ctx.tree_node("insp_asset_mr", "Mesh & .flmat Material", default_open=show_asset_open):
                        var mref_txt = cur.mesh_ref.rel_path if not cur.mesh_ref.is_empty() else String("(none)")
                        studio.ui_ctx.label("MeshRef: " + mref_txt + " (id=" + String(cur.mesh_id) + ")", studio.ui_ctx.theme.text_muted)
                        var mdrop = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                        if mdrop[0] and FlintStudioState._ascii_lower(mdrop[2]).endswith(".glb"):
                            var rel_glb = studio._project_rel_for_entry(mdrop[2])
                            var before_blob = studio._capture_world_blob()
                            var mr_m = studio.app.world.get_ptr_mut[MeshRendererComponent](sel_e)
                            if mr_m:
                                mr_m.value()[].mesh_ref = AssetRef.from_path(rel_glb, ASSET_KIND_MESH)
                                mr_m.value()[].mesh_id = -1
                            _ = studio.commit_ecs_edit(eid, "Assign MeshRef", before_blob^)
                            cur = studio.entities[eid].copy()

                        var act_mat_ref = cur.override_material_ref.copy() if not cur.override_material_ref.is_empty() else cur.material_ref.copy()
                        var mat_txt = act_mat_ref.rel_path if not act_mat_ref.is_empty() else String("(none)")
                        studio.ui_ctx.label("MaterialRef: " + mat_txt, studio.ui_ctx.theme.text_muted)
                        var mat_drop = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                        if mat_drop[0] and FlintStudioState._ascii_lower(mat_drop[2]).endswith(".flmat"):
                            _ = studio._assign_flmat_to_entity(eid, studio._project_rel_for_entry(mat_drop[2]))
                            cur = studio.entities[eid].copy()

                        if not act_mat_ref.is_empty():
                            var mat_disk = studio._resolve_asset_disk_path(act_mat_ref.rel_path)
                            if studio.flmat_pending_path.byte_length() > 0 and (
                                studio.flmat_pending_path != mat_disk or studio.flmat_pending_eid != eid
                            ):
                                _ = studio.commit_pending_flmat_edit()  # selection moved on mid-edit
                            var flm_loaded = False
                            var flm = FlmatAsset()
                            if studio.flmat_pending_path == mat_disk:
                                flm = studio.flmat_pending.copy()
                                flm_loaded = True
                                studio.flmat_pending_touch_sec = studio.ui_time_sec
                            else:
                                try:
                                    flm = load_flmat_file(mat_disk)
                                    flm_loaded = True
                                except:
                                    pass
                            if flm_loaded:
                                var flm_dirty = False
                                var flm_text_dirty = False
                                var sh_items = List[String]()
                                sh_items.append("PBR")
                                sh_items.append("NPR Cel")
                                sh_items.append("Stylized VFX")
                                var sh_idx = max(0, min(2, flm.shading_mode))
                                if studio.ui_ctx.combo_box("flmat_shading", ".flmat Shading", sh_idx, sh_items, w=insp_w):
                                    flm.shading_mode = sh_idx
                                    flm_dirty = True

                                var alb_s = flm.albedo_tex.rel_path
                                if studio.ui_ctx.text_input("flmat_alb_tex", alb_s, insp_w, "albedo_tex (.png/.jpg)"):
                                    flm.albedo_tex = AssetRef.from_path(alb_s, ASSET_KIND_TEXTURE) if alb_s.byte_length() > 0 else AssetRef()
                                    flm_text_dirty = True
                                var d_alb = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                                if d_alb[0]:
                                    var dl = FlintStudioState._ascii_lower(d_alb[2])
                                    if dl.endswith(".png") or dl.endswith(".jpg") or dl.endswith(".jpeg"):
                                        flm.albedo_tex = AssetRef.from_path(studio._project_rel_for_entry(d_alb[2]), ASSET_KIND_TEXTURE)
                                        flm_dirty = True

                                var nrm_s = flm.normal_tex.rel_path
                                if studio.ui_ctx.text_input("flmat_nrm_tex", nrm_s, insp_w, "normal_tex (.png/.jpg)"):
                                    flm.normal_tex = AssetRef.from_path(nrm_s, ASSET_KIND_TEXTURE) if nrm_s.byte_length() > 0 else AssetRef()
                                    flm_text_dirty = True
                                var d_nrm = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                                if d_nrm[0]:
                                    var dl = FlintStudioState._ascii_lower(d_nrm[2])
                                    if dl.endswith(".png") or dl.endswith(".jpg") or dl.endswith(".jpeg"):
                                        flm.normal_tex = AssetRef.from_path(studio._project_rel_for_entry(d_nrm[2]), ASSET_KIND_TEXTURE)
                                        flm_dirty = True

                                var orm_s = flm.orm_tex.rel_path
                                if studio.ui_ctx.text_input("flmat_orm_tex", orm_s, insp_w, "orm_tex (.png/.jpg)"):
                                    flm.orm_tex = AssetRef.from_path(orm_s, ASSET_KIND_TEXTURE) if orm_s.byte_length() > 0 else AssetRef()
                                    flm_text_dirty = True
                                var d_orm = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                                if d_orm[0]:
                                    var dl = FlintStudioState._ascii_lower(d_orm[2])
                                    if dl.endswith(".png") or dl.endswith(".jpg") or dl.endswith(".jpeg"):
                                        flm.orm_tex = AssetRef.from_path(studio._project_rel_for_entry(d_orm[2]), ASSET_KIND_TEXTURE)
                                        flm_dirty = True

                                var emi_s = flm.emissive_tex.rel_path
                                if studio.ui_ctx.text_input("flmat_emi_tex", emi_s, insp_w, "emissive_tex (.png/.jpg)"):
                                    flm.emissive_tex = AssetRef.from_path(emi_s, ASSET_KIND_TEXTURE) if emi_s.byte_length() > 0 else AssetRef()
                                    flm_text_dirty = True
                                var d_emi = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                                if d_emi[0]:
                                    var dl = FlintStudioState._ascii_lower(d_emi[2])
                                    if dl.endswith(".png") or dl.endswith(".jpg") or dl.endswith(".jpeg"):
                                        flm.emissive_tex = AssetRef.from_path(studio._project_rel_for_entry(d_emi[2]), ASSET_KIND_TEXTURE)
                                        flm_dirty = True

                                var tex_field_focused = False
                                var tex_field_ids = List[String]()
                                tex_field_ids.append("flmat_alb_tex")
                                tex_field_ids.append("flmat_nrm_tex")
                                tex_field_ids.append("flmat_orm_tex")
                                tex_field_ids.append("flmat_emi_tex")
                                for fi in range(len(tex_field_ids)):
                                    if studio.ui_ctx.focused_id == studio.ui_ctx.widget_id(tex_field_ids[fi]):
                                        tex_field_focused = True
                                if flm_text_dirty or (flm_dirty and studio.flmat_pending_path == mat_disk):
                                    if studio.flmat_pending_path != mat_disk:
                                        studio.flmat_pending_stamp = file_stamp(mat_disk)
                                    studio.flmat_pending = flm.copy()
                                    studio.flmat_pending_path = mat_disk
                                    studio.flmat_pending_eid = eid
                                    studio.flmat_pending_editor_id = studio.editor_id_of(eid)
                                    studio.flmat_pending_ref = act_mat_ref.rel_path
                                    studio.flmat_pending_touch_sec = studio.ui_time_sec
                                if flm_dirty or (studio.flmat_pending_path == mat_disk and not tex_field_focused):
                                    if studio.flmat_pending_path == mat_disk:
                                        _ = studio.commit_pending_flmat_edit()
                                    else:
                                        _ = save_flmat_file(mat_disk, flm)
                                        _ = studio._assign_flmat_to_entity(eid, act_mat_ref.rel_path)
                        studio.ui_ctx.tree_pop()

                if cur.has_animator or cur.has_skeleton:
                    if studio.ui_ctx.tree_node("insp_animator", "Animator & Viewport Preview", default_open=True):
                        _ = studio.ui_ctx.checkbox("insp_vp_anim_prev", "Viewport Idle Preview", studio.viewport_anim_preview)
                        if cur.has_animator:
                            var ac_opt = studio.app.world.get_ptr[AnimatorComponent](sel_e)
                            if ac_opt:
                                ref ac = ac_opt.value()[unsafe_offset=0]
                                var a_play = ac.playing
                                var a_loop = ac.loop
                                var a_spd = ac.speed
                                var a_cidx = ac.clip_index
                                var a_gref = ac.graph_ref.rel_path
                                var anim_dirty = False
                                var anim_spd_dirty = False

                                if studio.ui_ctx.checkbox("insp_anim_play", "Playing", a_play):
                                    anim_dirty = True
                                if studio.ui_ctx.checkbox("insp_anim_loop", "Loop", a_loop):
                                    anim_dirty = True
                                if studio.ui_ctx.slider_float("insp_anim_spd", "Speed", a_spd, Float32(0.0), Float32(3.0), w=insp_w):
                                    anim_dirty = True
                                    anim_spd_dirty = True

                                var clip_names = List[String]()
                                var ac_cache = studio.app.resources.get_ptr[AnimAssetCache]()
                                if ac_cache and cur.skeleton_model_id != 0:
                                    var midx = ac_cache.value()[unsafe_offset=0].find_model(cur.skeleton_model_id)
                                    if midx >= 0:
                                        ref mdl = ac_cache.value()[unsafe_offset=0].models[midx]
                                        for ci in range(len(mdl.animations)):
                                            var cn = mdl.animations[ci].name
                                            clip_names.append(cn if cn.byte_length() > 0 else ("Clip_" + String(ci)))
                                if len(clip_names) > 0:
                                    var sel_c = max(0, min(len(clip_names) - 1, a_cidx if a_cidx >= 0 else 0))
                                    if studio.ui_ctx.combo_box("insp_anim_clip", "Clip", sel_c, clip_names, w=insp_w):
                                        a_cidx = sel_c
                                        anim_dirty = True

                                studio.ui_ctx.label("GraphRef: " + (a_gref if a_gref.byte_length() > 0 else String("(none)")), studio.ui_ctx.theme.text_muted)
                                var gdrop = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                                if gdrop[0] and FlintStudioState._ascii_lower(gdrop[2]).endswith(".flanim"):
                                    a_gref = studio._project_rel_for_entry(gdrop[2])
                                    anim_dirty = True

                                if anim_dirty:
                                    if anim_spd_dirty and not studio.ui_ctx.mouse_down_l:
                                        studio.ui_edit_seq += 1
                                    var anim_ckey = (
                                        _ui_coalesce_key(studio.ui_edit_seq, "insp_anim_spd")
                                        if anim_spd_dirty
                                        else UInt64(0)
                                    )
                                    var before_blob = studio._capture_world_blob()
                                    var ac_m = studio.app.world.get_ptr_mut[AnimatorComponent](sel_e)
                                    if ac_m:
                                        ac_m.value()[].playing = a_play
                                        ac_m.value()[].loop = a_loop
                                        ac_m.value()[].speed = a_spd
                                        if a_cidx != ac_m.value()[].clip_index:
                                            ac_m.value()[].clip_index = a_cidx
                                            if a_cidx >= 0 and a_cidx < len(clip_names):
                                                ac_m.value()[].clip_name = clip_names[a_cidx]
                                            ac_m.value()[].time = Float32(0.0)
                                            studio.preview_anim_time = Float32(0.0)
                                        if a_gref != ac_m.value()[].graph_ref.rel_path:
                                            ac_m.value()[].graph_ref = AssetRef.from_path(a_gref, ASSET_KIND_RAW)
                                            ac_m.value()[].graph_id = 0
                                            ac_m.value()[].bound = False
                                    _ = studio.commit_ecs_edit(eid, "Edit Animator", before_blob^, coalesce_key=anim_ckey)
                        else:
                            if studio.ui_ctx.button("insp_add_animator", "Add AnimatorComponent", insp_w, 22.0):
                                var before_blob = studio._capture_world_blob()
                                _ = studio.app.world.insert[AnimatorComponent](sel_e, AnimatorComponent())
                                _ = studio.commit_ecs_edit(eid, "Add Animator", before_blob^)
                        studio.ui_ctx.tree_pop()

                if cur.has_rig or cur.has_skeleton or cur.has_bone_socket:
                    if studio.ui_ctx.tree_node("insp_rig_skel", "Rig, Skeleton & BoneSocket", default_open=True):
                        if cur.has_rig or cur.has_skeleton:
                            var r_txt = cur.rig_ref.rel_path if not cur.rig_ref.is_empty() else String("(none)")
                            studio.ui_ctx.label(
                                "Rig: " + r_txt + " (joints=" + String(cur.rig_joint_count if cur.has_rig else cur.skeleton_node_count) + ")",
                                studio.ui_ctx.theme.text_muted,
                            )
                            var rdrop = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                            if rdrop[0] and FlintStudioState._ascii_lower(rdrop[2]).endswith(".flrig"):
                                _ = studio.activate_asset_entry(rdrop[2], target_entity_id=eid, from_drop=True)
                        if cur.has_skeleton:
                            var sk_p = studio.app.world.get_ptr[SkeletonComponent](sel_e)
                            if sk_p:
                                var nb = sk_p.value()[].skeleton.node_count
                                studio.ui_ctx.label("Skeleton Nodes: " + String(nb) + " | ModelID: " + String(sk_p.value()[].model_id))
                        if cur.has_bone_socket:
                            var bs_p = studio.app.world.get_ptr[BoneSocketComponent](sel_e)
                            if bs_p:
                                ref bs = bs_p.value()[unsafe_offset=0]
                                var s_name = bs.attachment.name
                                var s_bnode = bs.attachment.bone_node
                                var s_lpos = bs.attachment.local_pos
                                var s_lscl = bs.attachment.local_scale
                                var s_sk_ent = bs.skeleton_entity
                                var bs_dirty = False
                                var bs_pos_dirty = False
                                var bs_scl_dirty = False

                                var sk_ids = List[Int]()
                                var sk_labels = List[String]()
                                var sel_sk_idx = 0
                                for si in range(len(studio.entities)):
                                    if studio.entities[si].alive and studio.entities[si].has_skeleton:
                                        if s_sk_ent.valid() and Int(s_sk_ent.index) == si:
                                            sel_sk_idx = len(sk_ids)
                                        sk_ids.append(si)
                                        sk_labels.append("#" + String(si) + " " + studio.entities[si].name)
                                if len(sk_labels) > 0:
                                    if studio.ui_ctx.combo_box("insp_bs_skel", "Target Skel", sel_sk_idx, sk_labels, w=insp_w):
                                        s_sk_ent = studio.resolve_entity(sk_ids[sel_sk_idx])
                                        bs_dirty = True

                                var bone_names = List[String]()
                                if s_sk_ent.valid() and studio.app.world.is_alive(s_sk_ent):
                                    var tgt_sk = studio.app.world.get_ptr[SkeletonComponent](s_sk_ent)
                                    var ac_c = studio.app.resources.get_ptr[AnimAssetCache]()
                                    if tgt_sk and ac_c and tgt_sk.value()[].model_id != 0:
                                        var mi = ac_c.value()[unsafe_offset=0].find_model(tgt_sk.value()[].model_id)
                                        if mi >= 0:
                                            ref mdl = ac_c.value()[unsafe_offset=0].models[mi]
                                            for bi in range(len(mdl.nodes)):
                                                var bn = mdl.nodes[bi].name
                                                bone_names.append(bn if bn.byte_length() > 0 else ("Bone_" + String(bi)))
                                    if len(bone_names) == 0 and tgt_sk:
                                        for bi in range(tgt_sk.value()[].skeleton.node_count):
                                            bone_names.append("Bone_" + String(bi))
                                if len(bone_names) > 0:
                                    var sel_b = max(0, min(len(bone_names) - 1, s_bnode if s_bnode >= 0 else 0))
                                    if studio.ui_ctx.combo_box("insp_bs_bone", "Bone", sel_b, bone_names, w=insp_w):
                                        s_bnode = sel_b
                                        bs_dirty = True

                                if studio.ui_ctx.text_input("insp_bs_name", s_name, insp_w, "Socket Name"):
                                    bs_dirty = True
                                if studio.ui_ctx.drag_float3("insp_bs_pos", "Socket Pos", s_lpos, speed=0.02, w=insp_w):
                                    bs_dirty = True
                                    bs_pos_dirty = True
                                if studio.ui_ctx.drag_float3("insp_bs_scl", "Socket Scale", s_lscl, speed=0.02, min_val=0.01, max_val=50.0, w=insp_w):
                                    bs_dirty = True
                                    bs_scl_dirty = True

                                if bs_dirty:
                                    if (bs_pos_dirty or bs_scl_dirty) and not studio.ui_ctx.mouse_down_l:
                                        studio.ui_edit_seq += 1
                                    var bs_ckey = (
                                        _ui_coalesce_key(studio.ui_edit_seq, "insp_bs_pos")
                                        if bs_pos_dirty
                                        else (_ui_coalesce_key(studio.ui_edit_seq, "insp_bs_scl") if bs_scl_dirty else UInt64(0))
                                    )
                                    var before_blob = studio._capture_world_blob()
                                    var bs_m = studio.app.world.get_ptr_mut[BoneSocketComponent](sel_e)
                                    if bs_m:
                                        bs_m.value()[].skeleton_entity = s_sk_ent
                                        bs_m.value()[].skeleton_slot = -1
                                        bs_m.value()[].attachment.name = s_name
                                        bs_m.value()[].attachment.bone_node = s_bnode
                                        bs_m.value()[].attachment.local_pos = s_lpos
                                        bs_m.value()[].attachment.local_scale = s_lscl
                                    _ = studio.commit_ecs_edit(eid, "Edit BoneSocket", before_blob^, coalesce_key=bs_ckey)
                        else:
                            if studio.ui_ctx.button("insp_add_bs", "Attach BoneSocketComponent", insp_w, 22.0):
                                var before_blob = studio._capture_world_blob()
                                _ = studio.app.world.insert[BoneSocketComponent](sel_e, BoneSocketComponent())
                                _ = studio.commit_ecs_edit(eid, "Attach BoneSocket", before_blob^)
                        studio.ui_ctx.tree_pop()

                if cur.has_audio:
                    if studio.ui_ctx.tree_node("insp_audio", "Audio Emitter", default_open=True):
                        var au_p = studio.app.world.get_ptr[AudioEmitterComponent](sel_e)
                        if au_p:
                            ref au = au_p.value()[unsafe_offset=0]
                            var au_ref_s = au.audio_ref.rel_path if not au.audio_ref.is_empty() else au.asset_path
                            studio.ui_ctx.label("AudioRef: " + (au_ref_s if au_ref_s.byte_length() > 0 else String("(none)")), studio.ui_ctx.theme.text_muted)
                            var adrop = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                            if adrop[0]:
                                var alow = FlintStudioState._ascii_lower(adrop[2])
                                if alow.endswith(".wav") or alow.endswith(".qoa"):
                                    _ = studio.activate_asset_entry(adrop[2], target_entity_id=eid, from_drop=True)
                            var au_vol = au.volume
                            var au_pch = au.pitch
                            var au_loop = au.looping
                            var au_spat = au.spatial
                            var au_play = au.playing
                            var au_dirty = False
                            var au_vol_dirty = False
                            var au_pch_dirty = False
                            if studio.ui_ctx.slider_float("insp_au_vol", "Volume", au_vol, Float32(0.0), Float32(2.0), w=insp_w):
                                au_dirty = True
                                au_vol_dirty = True
                            if studio.ui_ctx.slider_float("insp_au_pch", "Pitch", au_pch, Float32(0.1), Float32(3.0), w=insp_w):
                                au_dirty = True
                                au_pch_dirty = True
                            if studio.ui_ctx.checkbox("insp_au_loop", "Loop", au_loop):
                                au_dirty = True
                            if studio.ui_ctx.checkbox("insp_au_spat", "Spatial 3D", au_spat):
                                au_dirty = True
                            if studio.ui_ctx.checkbox("insp_au_play", "Playing", au_play):
                                au_dirty = True
                            if au_dirty:
                                if (au_vol_dirty or au_pch_dirty) and not studio.ui_ctx.mouse_down_l:
                                    studio.ui_edit_seq += 1
                                var au_ckey = (
                                    _ui_coalesce_key(studio.ui_edit_seq, "insp_au_vol")
                                    if au_vol_dirty
                                    else (_ui_coalesce_key(studio.ui_edit_seq, "insp_au_pch") if au_pch_dirty else UInt64(0))
                                )
                                var before_blob = studio._capture_world_blob()
                                var au_m = studio.app.world.get_ptr_mut[AudioEmitterComponent](sel_e)
                                if au_m:
                                    au_m.value()[].volume = au_vol
                                    au_m.value()[].pitch = au_pch
                                    au_m.value()[].looping = au_loop
                                    au_m.value()[].spatial = au_spat
                                    au_m.value()[].playing = au_play
                                    if au_play and not au_m.value()[].handle.is_valid():
                                        au_m.value()[].auto_play_pending = True
                                _ = studio.commit_ecs_edit(eid, "Edit AudioEmitter", before_blob^, coalesce_key=au_ckey)
                        studio.ui_ctx.tree_pop()

                if cur.has_prefab:
                    if studio.ui_ctx.tree_node("insp_prefab", "Prefab Instance", default_open=True):
                        var pf_p = studio.app.world.get_ptr[PrefabInstanceComponent](sel_e)
                        if pf_p:
                            ref pf = pf_p.value()[unsafe_offset=0]
                            studio.ui_ctx.label("Prefab: " + pf.prefab_name)
                            studio.ui_ctx.label("Ref: " + pf.prefab_ref.rel_path, studio.ui_ctx.theme.text_muted)
                            studio.ui_ctx.label("Slot: " + String(pf.node_slot) + " | Overrides: " + String(len(pf.overrides)), studio.ui_ctx.theme.text_muted)
                        studio.ui_ctx.tree_pop()

                var insp_panel_info = studio.dock.find_panel("Inspector")
                if insp_panel_info[0] >= 0:
                    var insp_drop = studio.ui_ctx.accept_drag_target_rect("ASSET_ENTRY", insp_panel_info[3])
                    if insp_drop[0]:
                        var ilow = FlintStudioState._ascii_lower(insp_drop[2])
                        if (
                            ilow.endswith(".flrig")
                            or ilow.endswith(".flmat")
                            or ilow.endswith(".flanim")
                            or ilow.endswith(".wav")
                            or ilow.endswith(".qoa")
                            or ilow.endswith(".png")
                            or ilow.endswith(".jpg")
                            or ilow.endswith(".jpeg")
                        ):
                            _ = studio.activate_asset_entry(insp_drop[2], target_entity_id=eid, from_drop=True)
                        elif ilow.endswith(".glb"):
                            var rel_glb = studio._project_rel_for_entry(insp_drop[2])
                            var before_blob = studio._capture_world_blob()
                            var mr_m = studio.app.world.get_ptr_mut[MeshRendererComponent](sel_e)
                            if mr_m:
                                mr_m.value()[].mesh_ref = AssetRef.from_path(rel_glb, ASSET_KIND_MESH)
                                mr_m.value()[].mesh_id = -1
                            else:
                                _ = studio.app.world.insert[MeshRendererComponent](
                                    sel_e,
                                    MeshRendererComponent(-1, -1, True, True, AssetRef.from_path(rel_glb, ASSET_KIND_MESH), AssetRef()),
                                )
                            _ = studio.commit_ecs_edit(eid, "Assign MeshRef", before_blob^)

                # M2 Generic Reflection Inspector (Add Component, POD Schemas, Built-ins & UnknownComponents)
                if studio.ui_ctx.tree_node("insp_reflect_add", "Components (Reflection)", default_open=True):
                    var addable_names = List[String]()
                    var sorted_cidx = studio.registry.sorted_entry_indices()
                    for si in range(len(sorted_cidx)):
                        var cidx = sorted_cidx[si]
                        var cname = studio.registry.entries[cidx].schema_name
                        if _is_internal_structural_component(cname):
                            continue
                        if not studio.registry.has_on_entity(cidx, studio.app.world, sel_e):
                            addable_names.append(cname)
                    if len(addable_names) > 0:
                        studio._insp_add_comp_idx = max(0, min(len(addable_names) - 1, studio._insp_add_comp_idx))
                        _ = studio.ui_ctx.combo_box("insp_add_comp_combo", "Add", studio._insp_add_comp_idx, addable_names, w=insp_w)
                        if studio.ui_ctx.button("insp_add_comp_btn", "Add Component", insp_w, 22.0):
                            var add_nm = addable_names[studio._insp_add_comp_idx]
                            if studio.add_component_by_name(eid, add_nm, record_undo=True):
                                if eid < len(studio.entities):
                                    cur = studio.entities[eid].copy()
                                sel_e = studio.resolve_entity(eid)
                    else:
                        studio.ui_ctx.label("All registered components attached", studio.ui_ctx.theme.text_muted)

                    # Reflected POD Components (`entry.is_pod == True`)
                    var sorted_pod = studio.registry.sorted_entry_indices()
                    for si in range(len(sorted_pod)):
                        var cidx = sorted_pod[si]
                        if not studio.registry.entries[cidx].is_pod:
                            continue
                        if not studio.registry.has_on_entity(cidx, studio.app.world, sel_e):
                            continue
                        var sidx = studio.registry.entries[cidx].pod_schema_idx
                        if sidx < 0 or sidx >= len(studio.registry.pod_schemas):
                            continue
                        var schema = studio.registry.pod_schemas[sidx].copy()
                        var tid = studio.app.world.find_component_by_hash(schema.component_hash)
                        if tid < 0:
                            continue
                        if studio.ui_ctx.tree_node("insp_pod_" + schema.name, schema.name, default_open=True):
                            if studio.ui_ctx.button("insp_pod_rem_" + schema.name, "Remove " + schema.name, insp_w, 20.0):
                                if studio.remove_component_by_name(eid, schema.name, record_undo=True):
                                    if eid < len(studio.entities):
                                        cur = studio.entities[eid].copy()
                                    sel_e = studio.resolve_entity(eid)
                                studio.ui_ctx.tree_pop()
                                continue
                            var ptr = studio.app.world.get_raw_ptr(sel_e, tid)
                            if ptr == 0 or len(schema.fields) == 0:
                                studio.ui_ctx.label("POD (" + String(schema.byte_size) + " B)", studio.ui_ctx.theme.text_muted)
                            else:
                                for fi in range(len(schema.fields)):
                                    var f = schema.fields[fi].copy()
                                    if f.offset < 0 or f.offset + f.size > schema.byte_size:
                                        continue
                                    var fid = "insp_pod_" + schema.name + "_" + f.name + "_" + String(fi)
                                    if f.kind == POD_FIELD_BOOL and f.size >= 1:
                                        var bv = (_studio_pod_read_u8(ptr, f.offset) != 0) if f.size < 4 else (_studio_pod_read_u32(ptr, f.offset) != 0)
                                        if studio.ui_ctx.checkbox(fid, f.name, bv):
                                            var before_blob = studio._capture_world_blob()
                                            var mptr = studio.app.world.get_raw_ptr_mut(sel_e, tid)
                                            if mptr != 0:
                                                if f.size < 4:
                                                    _studio_pod_write_u8(mptr, f.offset, UInt8(1) if bv else UInt8(0))
                                                else:
                                                    _studio_pod_write_u32(mptr, f.offset, UInt32(1) if bv else UInt32(0))
                                            _ = studio.commit_ecs_edit(eid, "Edit " + schema.name + "." + f.name, before_blob^)
                                            ptr = studio.app.world.get_raw_ptr(sel_e, tid)
                                    elif f.kind == POD_FIELD_INT and (f.size == 4 or f.size == 8):
                                        var iv = Int64(_studio_pod_read_i32(ptr, f.offset)) if f.size == 4 else _studio_pod_read_i64(ptr, f.offset)
                                        var int_changed = False
                                        var next_iv = iv
                                        studio.ui_ctx.label(f.name, studio.ui_ctx.theme.text_muted)
                                        studio.ui_ctx.begin_box(fid + "_row", FlexStyle.row(gap=4.0, pad=0.0))
                                        var txt_id = fid + "_txt"
                                        var txt_wid = studio.ui_ctx.widget_id(txt_id)
                                        if studio._int_edit_wid == txt_wid and (studio.ui_ctx.focused_id != txt_wid or studio._int_edit_entity_id != eid):
                                            studio._int_edit_wid = UInt64(0)
                                            studio._int_edit_entity_id = -1
                                            studio._int_edit_buf = ""
                                        var int_s = (
                                            studio._int_edit_buf
                                            if (studio._int_edit_wid == txt_wid and studio._int_edit_entity_id == eid and studio.ui_ctx.focused_id == txt_wid)
                                            else String(iv)
                                        )
                                        if studio.ui_ctx.button(fid + "_dec", "-", 24.0, 22.0):
                                            next_iv = iv - Int64(1)
                                            int_changed = True
                                            studio._int_edit_wid = UInt64(0)
                                            studio._int_edit_entity_id = -1
                                            studio._int_edit_buf = ""
                                        var txt_w = max(Float32(64.0), insp_w - Float32(60.0))
                                        if studio.ui_ctx.text_input(txt_id, int_s, txt_w, f.name):
                                            var parsed_i = _parse_signed_int64(int_s)
                                            if parsed_i[0]:
                                                next_iv = parsed_i[1]
                                                int_changed = True
                                                studio._int_edit_wid = txt_wid
                                                studio._int_edit_entity_id = eid
                                                studio._int_edit_buf = int_s
                                            else:
                                                var trimmed = String(int_s.strip())
                                                if trimmed == "" or trimmed == "-" or trimmed == "+":
                                                    studio._int_edit_wid = txt_wid
                                                    studio._int_edit_entity_id = eid
                                                    studio._int_edit_buf = int_s
                                        if studio.ui_ctx.button(fid + "_inc", "+", 24.0, 22.0):
                                            next_iv = iv + Int64(1)
                                            int_changed = True
                                            studio._int_edit_wid = UInt64(0)
                                            studio._int_edit_entity_id = -1
                                            studio._int_edit_buf = ""
                                        studio.ui_ctx.end_box()
                                        if int_changed:
                                            if f.max_val > f.min_val:
                                                next_iv = max(Int64(f.min_val), min(Int64(f.max_val), next_iv))
                                            if f.size == 4:
                                                next_iv = max(Int64(-2147483648), min(Int64(2147483647), next_iv))
                                            var before_blob = studio._capture_world_blob()
                                            var mptr = studio.app.world.get_raw_ptr_mut(sel_e, tid)
                                            if mptr != 0:
                                                if f.size == 4:
                                                    _studio_pod_write_i32(mptr, f.offset, Int32(next_iv))
                                                else:
                                                    _studio_pod_write_i64(mptr, f.offset, next_iv)
                                            _ = studio.commit_ecs_edit(eid, "Edit " + schema.name + "." + f.name, before_blob^)
                                            ptr = studio.app.world.get_raw_ptr(sel_e, tid)
                                    elif f.kind == POD_FIELD_F32 and f.size >= 4:
                                        var fv = _studio_pod_read_f32(ptr, f.offset)
                                        var lo_f = f.min_val if f.max_val > f.min_val else Float32(-1.0e6)
                                        var hi_f = f.max_val if f.max_val > f.min_val else Float32(1.0e6)
                                        var sp_f = f.step if f.step > Float32(0.0) else Float32(0.05)
                                        if studio.ui_ctx.drag_float(fid, f.name, fv, speed=sp_f, min_val=lo_f, max_val=hi_f, w=insp_w):
                                            studio.begin_edit_interaction()
                                            var before_blob = studio._capture_world_blob()
                                            var mptr = studio.app.world.get_raw_ptr_mut(sel_e, tid)
                                            if mptr != 0:
                                                _studio_pod_write_f32(mptr, f.offset, fv)
                                            _ = studio.commit_ecs_edit(eid, "Edit " + schema.name + "." + f.name, before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, fid))
                                            ptr = studio.app.world.get_raw_ptr(sel_e, tid)
                                    elif f.kind == POD_FIELD_VEC3 and f.size >= 12:
                                        var vv = Vec3(
                                            _studio_pod_read_f32(ptr, f.offset),
                                            _studio_pod_read_f32(ptr, f.offset + 4),
                                            _studio_pod_read_f32(ptr, f.offset + 8),
                                        )
                                        var lo_v = f.min_val if f.max_val > f.min_val else Float32(-1.0e6)
                                        var hi_v = f.max_val if f.max_val > f.min_val else Float32(1.0e6)
                                        var sp_v = f.step if f.step > Float32(0.0) else Float32(0.05)
                                        if studio.ui_ctx.drag_float3(fid, f.name, vv, speed=sp_v, min_val=lo_v, max_val=hi_v, w=insp_w):
                                            studio.begin_edit_interaction()
                                            var before_blob = studio._capture_world_blob()
                                            var mptr = studio.app.world.get_raw_ptr_mut(sel_e, tid)
                                            if mptr != 0:
                                                _studio_pod_write_f32(mptr, f.offset, vv.x)
                                                _studio_pod_write_f32(mptr, f.offset + 4, vv.y)
                                                _studio_pod_write_f32(mptr, f.offset + 8, vv.z)
                                            _ = studio.commit_ecs_edit(eid, "Edit " + schema.name + "." + f.name, before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, fid))
                                            ptr = studio.app.world.get_raw_ptr(sel_e, tid)
                                    elif f.kind == POD_FIELD_COLOR_RGB and f.size >= 12:
                                        var cv = Vec3(
                                            _studio_pod_read_f32(ptr, f.offset),
                                            _studio_pod_read_f32(ptr, f.offset + 4),
                                            _studio_pod_read_f32(ptr, f.offset + 8),
                                        )
                                        var lo_c = f.min_val if f.max_val > f.min_val else Float32(0.0)
                                        var hi_c = f.max_val if f.max_val > f.min_val else Float32(1.0)
                                        if studio.ui_ctx.color_edit3(fid, f.name, cv, min_val=lo_c, max_val=hi_c, w=insp_w):
                                            studio.begin_edit_interaction()
                                            var before_blob = studio._capture_world_blob()
                                            var mptr = studio.app.world.get_raw_ptr_mut(sel_e, tid)
                                            if mptr != 0:
                                                _studio_pod_write_f32(mptr, f.offset, cv.x)
                                                _studio_pod_write_f32(mptr, f.offset + 4, cv.y)
                                                _studio_pod_write_f32(mptr, f.offset + 8, cv.z)
                                            _ = studio.commit_ecs_edit(eid, "Edit " + schema.name + "." + f.name, before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, fid))
                                            ptr = studio.app.world.get_raw_ptr(sel_e, tid)
                                    elif f.kind == POD_FIELD_QUAT and f.size >= 16:
                                        var qv = Quat(
                                            _studio_pod_read_f32(ptr, f.offset),
                                            _studio_pod_read_f32(ptr, f.offset + 4),
                                            _studio_pod_read_f32(ptr, f.offset + 8),
                                            _studio_pod_read_f32(ptr, f.offset + 12),
                                        )
                                        var edeg = _quat_to_euler_deg(qv)
                                        if studio.ui_ctx.drag_float3(fid, f.name, edeg, speed=0.5, min_val=-360.0, max_val=360.0, w=insp_w):
                                            studio.begin_edit_interaction()
                                            var nq = _euler_deg_to_quat(edeg)
                                            var before_blob = studio._capture_world_blob()
                                            var mptr = studio.app.world.get_raw_ptr_mut(sel_e, tid)
                                            if mptr != 0:
                                                _studio_pod_write_f32(mptr, f.offset, nq.x)
                                                _studio_pod_write_f32(mptr, f.offset + 4, nq.y)
                                                _studio_pod_write_f32(mptr, f.offset + 8, nq.z)
                                                _studio_pod_write_f32(mptr, f.offset + 12, nq.w)
                                            _ = studio.commit_ecs_edit(eid, "Edit " + schema.name + "." + f.name, before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, fid))
                                            ptr = studio.app.world.get_raw_ptr(sel_e, tid)
                                    elif f.kind == POD_FIELD_STRING:
                                        studio.ui_ctx.label(f.name + ": (string)", studio.ui_ctx.theme.text_muted)
                            studio.ui_ctx.tree_pop()

                    # Reflected Built-in & AI Components
                    var sorted_bi = studio.registry.sorted_entry_indices()
                    for si in range(len(sorted_bi)):
                        var cidx = sorted_bi[si]
                        if studio.registry.entries[cidx].is_pod:
                            continue
                        var cname = studio.registry.entries[cidx].schema_name
                        if _is_internal_structural_component(cname):
                            continue
                        if not studio.registry.has_on_entity(cidx, studio.app.world, sel_e):
                            continue
                        var is_refl_open = (
                            cname == "BehaviorTreeComponent"
                            or cname == "PointLightComponent"
                            or cname == "SpotLightComponent"
                            or cname == "CameraComponent"
                            or cname == "EnvironmentComponent"
                        )
                        if studio.ui_ctx.tree_node("insp_comp_" + cname, cname, default_open=is_refl_open):
                            if cname != "LocalTransform":
                                if studio.ui_ctx.button("insp_comp_rem_" + cname, "Remove " + cname, insp_w, 20.0):
                                    if studio.remove_component_by_name(eid, cname, record_undo=True):
                                        if eid < len(studio.entities):
                                            cur = studio.entities[eid].copy()
                                        sel_e = studio.resolve_entity(eid)
                                    studio.ui_ctx.tree_pop()
                                    continue

                            if cname == "BehaviorTreeComponent":
                                var bt_p = studio.app.world.get_ptr[BehaviorTreeComponent](sel_e)
                                if bt_p:
                                    ref bt = bt_p.value()[unsafe_offset=0]
                                    var bt_en = bt.enabled
                                    var bt_iv = bt.tick_interval_sec
                                    var bt_ref_s = bt.tree_ref.rel_path
                                    var bt_bb_vals = List[Float32](capacity=len(bt.blackboard.floats))
                                    var bt_bb_names = List[String](capacity=len(bt.blackboard.names))
                                    for bbi in range(len(bt.blackboard.names)):
                                        bt_bb_names.append(bt.blackboard.names[bbi])
                                        bt_bb_vals.append(bt.blackboard.floats[bbi] if bbi < len(bt.blackboard.floats) else Float32(0.0))
                                    var bt_nodes_n = len(bt.tree.nodes)
                                    var bt_scorers_n = len(bt.tree.utility_scorers)
                                    var bt_dirty = False
                                    var bt_tree_loaded = False
                                    var bt_new_tree = bt.tree.copy()

                                    studio.ui_ctx.label(
                                        "Nodes: " + String(bt_nodes_n) + " | BB: " + String(len(bt_bb_names)) + " | Scorers: " + String(bt_scorers_n),
                                        studio.ui_ctx.theme.text_muted,
                                    )
                                    if studio.ui_ctx.checkbox("insp_bt_en", "Enabled", bt_en):
                                        bt_dirty = True
                                    if studio.ui_ctx.drag_float("insp_bt_iv", "Tick Interval s", bt_iv, speed=0.01, min_val=0.0, max_val=60.0, w=insp_w):
                                        bt_dirty = True
                                    if studio.ui_ctx.text_input("insp_bt_ref", bt_ref_s, insp_w, "BehaviorTree JSON path"):
                                        bt_dirty = True
                                    var bt_drop = studio.ui_ctx.accept_drag_target("ASSET_ENTRY")
                                    if bt_drop[0] and FlintStudioState._ascii_lower(bt_drop[2]).endswith(".json"):
                                        var rel_bt = studio._project_rel_for_entry(bt_drop[2])
                                        var disk_bt = studio._resolve_asset_disk_path(rel_bt)
                                        var raw_bt = read_bytes(disk_bt)
                                        if len(raw_bt) > 0 and bt_new_tree.load_json(String(unsafe_from_utf8=raw_bt)):
                                            bt_ref_s = rel_bt
                                            bt_tree_loaded = True
                                            bt_dirty = True
                                    var bb_dirty_idx = -1
                                    for bbi in range(len(bt_bb_names)):
                                        var bv = bt_bb_vals[bbi]
                                        if studio.ui_ctx.drag_float("insp_bt_bb_" + String(bbi), bt_bb_names[bbi], bv, speed=0.05, w=insp_w):
                                            bt_bb_vals[bbi] = bv
                                            bb_dirty_idx = bbi
                                            bt_dirty = True
                                    if bt_dirty:
                                        studio.begin_edit_interaction()
                                        var before_blob = studio._capture_world_blob()
                                        var bt_m = studio.app.world.get_ptr_mut[BehaviorTreeComponent](sel_e)
                                        if bt_m:
                                            bt_m.value()[].enabled = bt_en
                                            bt_m.value()[].tick_interval_sec = bt_iv
                                            bt_m.value()[].tree_ref = AssetRef.from_path(bt_ref_s, ASSET_KIND_RAW) if bt_ref_s.byte_length() > 0 else AssetRef()
                                            if bt_tree_loaded:
                                                bt_m.value()[].tree = bt_new_tree.copy()
                                                bt_m.value()[].state = BtExecutionState()
                                            if bb_dirty_idx >= 0 and bb_dirty_idx < len(bt_m.value()[].blackboard.floats):
                                                bt_m.value()[].blackboard.floats[bb_dirty_idx] = bt_bb_vals[bb_dirty_idx]
                                        _ = studio.commit_ecs_edit(eid, "Edit BehaviorTree", before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "insp_comp_BehaviorTree"))
                            elif cname == "PointLightComponent":
                                var pl_p = studio.app.world.get_ptr[PointLightComponent](sel_e)
                                if pl_p:
                                    ref pl = pl_p.value()[unsafe_offset=0]
                                    var pl_rad = pl.radiance
                                    var pl_rng = pl.radius
                                    var pl_en = pl.enabled
                                    var pl_dirty = False
                                    if studio.ui_ctx.checkbox("insp_pl_en", "Enabled", pl_en):
                                        pl_dirty = True
                                    if studio.ui_ctx.drag_float3("insp_pl_rad", "Radiance", pl_rad, speed=0.1, min_val=0.0, max_val=500.0, w=insp_w):
                                        pl_dirty = True
                                    if studio.ui_ctx.drag_float("insp_pl_rng", "Radius", pl_rng, speed=0.1, min_val=0.1, max_val=500.0, w=insp_w):
                                        pl_dirty = True
                                    if pl_dirty:
                                        studio.begin_edit_interaction()
                                        var before_blob = studio._capture_world_blob()
                                        var pl_m = studio.app.world.get_ptr_mut[PointLightComponent](sel_e)
                                        if pl_m:
                                            pl_m.value()[unsafe_offset=0].enabled = pl_en
                                            pl_m.value()[unsafe_offset=0].radiance = pl_rad
                                            pl_m.value()[unsafe_offset=0].radius = pl_rng
                                        _ = studio.commit_ecs_edit(eid, "Edit PointLight", before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "insp_comp_PointLight"))
                            elif cname == "SpotLightComponent":
                                var sl_p = studio.app.world.get_ptr[SpotLightComponent](sel_e)
                                if sl_p:
                                    ref sl = sl_p.value()[unsafe_offset=0]
                                    var sl_rad = sl.radiance
                                    var sl_rng = sl.range
                                    var sl_in = sl.inner_deg
                                    var sl_out = sl.outer_deg
                                    var sl_dir = sl.direction
                                    var sl_en = sl.enabled
                                    var sl_dirty = False
                                    if studio.ui_ctx.checkbox("insp_sl_en", "Enabled", sl_en):
                                        sl_dirty = True
                                    if studio.ui_ctx.drag_float3("insp_sl_rad", "Radiance", sl_rad, speed=0.1, min_val=0.0, max_val=500.0, w=insp_w):
                                        sl_dirty = True
                                    if studio.ui_ctx.drag_float("insp_sl_rng", "Range", sl_rng, speed=0.1, min_val=0.1, max_val=500.0, w=insp_w):
                                        sl_dirty = True
                                    if studio.ui_ctx.drag_float("insp_sl_in", "Inner Deg", sl_in, speed=0.25, min_val=1.0, max_val=89.0, w=insp_w):
                                        sl_dirty = True
                                    if studio.ui_ctx.drag_float("insp_sl_out", "Outer Deg", sl_out, speed=0.25, min_val=1.0, max_val=89.0, w=insp_w):
                                        sl_dirty = True
                                    if studio.ui_ctx.drag_float3("insp_sl_dir", "Direction", sl_dir, speed=0.02, min_val=-1.0, max_val=1.0, w=insp_w):
                                        sl_dirty = True
                                    if sl_dirty:
                                        studio.begin_edit_interaction()
                                        var before_blob = studio._capture_world_blob()
                                        var sl_m = studio.app.world.get_ptr_mut[SpotLightComponent](sel_e)
                                        if sl_m:
                                            sl_m.value()[unsafe_offset=0].enabled = sl_en
                                            sl_m.value()[unsafe_offset=0].radiance = sl_rad
                                            sl_m.value()[unsafe_offset=0].range = sl_rng
                                            sl_m.value()[unsafe_offset=0].inner_deg = sl_in
                                            sl_m.value()[unsafe_offset=0].outer_deg = max(sl_in, sl_out)
                                            sl_m.value()[unsafe_offset=0].direction = sl_dir
                                        _ = studio.commit_ecs_edit(eid, "Edit SpotLight", before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "insp_comp_SpotLight"))
                            elif cname == "CameraComponent":
                                var cm_p = studio.app.world.get_ptr[CameraComponent](sel_e)
                                if cm_p:
                                    ref cm = cm_p.value()[unsafe_offset=0]
                                    var cm_act = cm.active
                                    var cm_fov = cm.fov_y_deg
                                    var cm_near = cm.near_z
                                    var cm_far = cm.far_z
                                    var cm_exp = cm.exposure
                                    var cm_dirty = False
                                    if studio.ui_ctx.checkbox("insp_cm_act", "Active", cm_act):
                                        cm_dirty = True
                                    if studio.ui_ctx.drag_float("insp_cm_fov", "FOV Deg", cm_fov, speed=0.25, min_val=15.0, max_val=140.0, w=insp_w):
                                        cm_dirty = True
                                    if studio.ui_ctx.drag_float("insp_cm_near", "Near Z", cm_near, speed=0.01, min_val=0.01, max_val=100.0, w=insp_w):
                                        cm_dirty = True
                                    if studio.ui_ctx.drag_float("insp_cm_far", "Far Z", cm_far, speed=1.0, min_val=1.0, max_val=10000.0, w=insp_w):
                                        cm_dirty = True
                                    if studio.ui_ctx.drag_float("insp_cm_exp", "Exposure", cm_exp, speed=0.02, min_val=0.05, max_val=20.0, w=insp_w):
                                        cm_dirty = True
                                    if cm_dirty:
                                        studio.begin_edit_interaction()
                                        var before_blob = studio._capture_world_blob()
                                        var cm_m = studio.app.world.get_ptr_mut[CameraComponent](sel_e)
                                        if cm_m:
                                            cm_m.value()[unsafe_offset=0].active = cm_act
                                            cm_m.value()[unsafe_offset=0].fov_y_deg = cm_fov
                                            cm_m.value()[unsafe_offset=0].near_z = cm_near
                                            cm_m.value()[unsafe_offset=0].far_z = max(cm_near + Float32(0.1), cm_far)
                                            cm_m.value()[unsafe_offset=0].exposure = cm_exp
                                        _ = studio.commit_ecs_edit(eid, "Edit Camera", before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "insp_comp_Camera"))
                            elif cname == "EnvironmentComponent":
                                var ev_p = studio.app.world.get_ptr[EnvironmentComponent](sel_e)
                                if ev_p:
                                    ref ev = ev_p.value()[unsafe_offset=0]
                                    var ev_en = ev.enabled
                                    var ev_dir = ev.sun_dir
                                    var ev_rad = ev.sun_radiance
                                    var ev_exp = ev.exposure
                                    var ev_ibl = ev.ibl_intensity
                                    var ev_turb = ev.turbidity
                                    var ev_dirty = False
                                    if studio.ui_ctx.checkbox("insp_env_en", "Enabled", ev_en):
                                        ev_dirty = True
                                    if studio.ui_ctx.drag_float3("insp_env_dir", "Sun Dir", ev_dir, speed=0.02, min_val=-1.0, max_val=1.0, w=insp_w):
                                        ev_dirty = True
                                    if studio.ui_ctx.drag_float3("insp_env_rad", "Sun Rad", ev_rad, speed=0.1, min_val=0.0, max_val=100.0, w=insp_w):
                                        ev_dirty = True
                                    if studio.ui_ctx.drag_float("insp_env_exp", "Exposure", ev_exp, speed=0.02, min_val=0.05, max_val=20.0, w=insp_w):
                                        ev_dirty = True
                                    if studio.ui_ctx.drag_float("insp_env_ibl", "IBL", ev_ibl, speed=0.02, min_val=0.0, max_val=20.0, w=insp_w):
                                        ev_dirty = True
                                    if studio.ui_ctx.drag_float("insp_env_turb", "Turbidity", ev_turb, speed=0.05, min_val=1.0, max_val=20.0, w=insp_w):
                                        ev_dirty = True
                                    if ev_dirty:
                                        studio.begin_edit_interaction()
                                        var before_blob = studio._capture_world_blob()
                                        var ev_m = studio.app.world.get_ptr_mut[EnvironmentComponent](sel_e)
                                        if ev_m:
                                            ev_m.value()[unsafe_offset=0].enabled = ev_en
                                            ev_m.value()[unsafe_offset=0].sun_dir = ev_dir
                                            ev_m.value()[unsafe_offset=0].sun_radiance = ev_rad
                                            ev_m.value()[unsafe_offset=0].exposure = ev_exp
                                            ev_m.value()[unsafe_offset=0].ibl_intensity = ev_ibl
                                            ev_m.value()[unsafe_offset=0].turbidity = ev_turb
                                        _ = studio.commit_ecs_edit(eid, "Edit Environment", before_blob^, coalesce_key=_ui_coalesce_key(studio.ui_edit_seq, "insp_comp_Environment"))
                            else:
                                var comp_json = String()
                                _ = studio.registry.write_entity_json(cidx, studio.app.world, sel_e, comp_json)
                                var summary = comp_json
                                if summary.byte_length() > 80:
                                    summary = String(summary[byte=0:77]) + "..."
                                studio.ui_ctx.label(summary, studio.ui_ctx.theme.text_muted)
                            studio.ui_ctx.tree_pop()

                    var unk_p = studio.app.world.get_ptr[UnknownComponentsComponent](sel_e)
                    if unk_p and (
                        len(unk_p.value()[unsafe_offset=0].json_names) > 0
                        or len(unk_p.value()[unsafe_offset=0].bin_fourccs) > 0
                    ):
                        ref unk = unk_p.value()[unsafe_offset=0]
                        if studio.ui_ctx.tree_node("insp_unk", "Unknown Components (Preserved)", default_open=True):
                            studio.ui_ctx.label(
                                "Preserved: " + String(len(unk.json_names)) + " JSON, " + String(len(unk.bin_fourccs)) + " Bin",
                                studio.ui_ctx.theme.text_muted,
                            )
                            for ui_i in range(len(unk.json_names)):
                                studio.ui_ctx.label("JSON: " + unk.json_names[ui_i], studio.ui_ctx.theme.text_muted)
                            for bi_i in range(len(unk.bin_fourccs)):
                                var blen = len(unk.bin_payloads[bi_i]) if bi_i < len(unk.bin_payloads) else 0
                                studio.ui_ctx.label(
                                    "Bin FourCC " + String(unk.bin_fourccs[bi_i]) + " (" + String(blen) + " B)",
                                    studio.ui_ctx.theme.text_muted,
                                )
                            studio.ui_ctx.tree_pop()

                    studio.ui_ctx.tree_pop()

            studio.ui_ctx.begin_box("ent_actions_row", FlexStyle.row(gap=6.0, pad=0.0))
            var act_w = max(Float32(80.0), (panel_w - Float32(12.0)) * Float32(0.5))
            if studio.ui_ctx.button("btn_dup_ent", "Duplicate [Ctrl+D]", act_w, 24.0):
                _ = studio.duplicate_entity(eid, record_undo=True)
            if studio.ui_ctx.button("btn_del_ent", "Delete [Del]", act_w, 24.0):
                _ = studio.despawn_entity(eid, record_undo=True)
            studio.ui_ctx.end_box()
        else:
            studio.ui_ctx.label("No entity selected", studio.ui_ctx.theme.text_muted)
        studio.ui_ctx.end_scroll_view()
        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "Agents"):
        var ag_info = studio.dock.find_panel("Agents")
        var ag_r = ag_info[3]
        var ag_acts = studio.agent_panel.render_and_update(
            studio.ui_ctx.draw_list,
            studio.agent_bus,
            studio.studio_theme,
            font_slot,
            studio.ui_ctx.font,
            ag_r.x,
            ag_r.y,
            ag_r.w,
            ag_r.h,
            scaled_input,
            dt,
            studio.ui_time_sec,
        )
        if studio.agent_panel.hovered_entity_id >= 0 and studio.is_alive_id(studio.agent_panel.hovered_entity_id):
            var hid = studio.agent_panel.hovered_entity_id
            studio.agent_bus.set_hovered_entity(
                hid,
                studio.entities[hid].translation,
                studio.entities[hid].collider_half_extents,
            )
        else:
            studio.agent_bus.clear_hovered_entity()
        for ai in range(len(ag_acts)):
            studio.apply_agent_panel_action(ag_acts[ai])
        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "CodeDrawer"):
        var cd_info = studio.dock.find_panel("CodeDrawer")
        var cd_r = cd_info[3]
        var cd_act = studio.code_drawer.draw_and_handle(
            studio.ui_ctx.draw_list,
            font_slot,
            studio.ui_ctx.font,
            studio.studio_theme,
            cd_r.x,
            cd_r.y,
            cd_r.w,
            cd_r.h,
            scaled_input,
            studio.ui_time_sec,
        )
        if cd_act.kind == CODE_ACT_CLOSE_TAB:
            studio.code_drawer.close_tab(cd_act.tab_index)
        elif cd_act.kind == CODE_ACT_BUILD_HOT_RELOAD:
            _ = studio.execute_palette_command(PAL_CMD_BUILD_HOT_RELOAD)
        elif cd_act.kind == CODE_ACT_OPEN_EXTERNAL_IDE:
            studio.source_editor.requested = True
        elif cd_act.kind == CODE_ACT_ACCEPT_ALL_DIFF or cd_act.kind == CODE_ACT_REJECT_ALL_DIFF:
            var acc = cd_act.kind == CODE_ACT_ACCEPT_ALL_DIFF
            for pi in range(len(studio.agent_bus.pending_approvals)):
                if studio.agent_bus.pending_approvals[pi].file_path == cd_act.path:
                    var a = AgentPanelAction(PANEL_ACT_APPROVE if acc else PANEL_ACT_REJECT)
                    a.approval_id = studio.agent_bus.pending_approvals[pi].approval_id
                    studio.apply_agent_panel_action(a)
                    break
            studio.push_toast(
                ("Accepted diff: " if acc else "Rejected diff: ") + cd_act.path,
                0,
                2.2,
            )
        elif cd_act.kind == CODE_ACT_JUMP_DIAGNOSTIC and cd_act.path.byte_length() > 0:
            _ = studio.code_drawer.open_file(studio._resolve_asset_disk_path(cd_act.path), cd_act.line)
        elif cd_act.kind == CODE_ACT_ASK_AGENT_FIX:
            _ = studio.apply_workspace_preset(WORKSPACE_AGENT_STUDIO, sw, sh)
            var a_fix = AgentPanelAction(PANEL_ACT_RUN_PROMPT)
            a_fix.role = ROLE_GAMEPLAY_CODER
            a_fix.text = "Fix diagnostic in " + cd_act.path + ":" + String(cd_act.line) + " — " + cd_act.message
            studio.apply_agent_panel_action(a_fix)
        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "CameraRig"):
        studio.ui_ctx.label("Viewport CameraRig")
        if studio.ui_ctx.slider_float_v2("cam_fov", "FOV", studio.camera_rig.fov_deg, Float32(30.0), Float32(95.0), w=210.0):
            studio.fov_y_deg = studio.camera_rig.fov_deg
        if studio.ui_ctx.slider_float_v2("cam_arm", "Arm m", studio.camera_rig.spring_arm.target_length, Float32(1.5), Float32(28.0), w=210.0):
            studio.update_camera(Float32(0.0))
        studio.ui_ctx.end_dock_panel()

    # 5. Bottom Asset Server, Spawn Cards & Recursive Workspace/Pack Browser (`DockSpace` -> `"AssetBrowser"`)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "AssetBrowser"):
        if (not studio.asset_browser_scanned) and (
            studio.asset_browser_dir.byte_length() > 0 or studio.asset_browser_pack.byte_length() > 0
        ):
            _ = studio.scan_asset_browser()
        var ab_info = studio.dock.find_panel("AssetBrowser")
        var ab_w = max(Float32(220.0), ab_info[3].w - Float32(16.0))
        var ab_h = max(Float32(90.0), ab_info[3].h - Float32(16.0))
        studio.ui_ctx.label(
            "ASSET BROWSER (.flpk v3="
            + String(mounted_packs)
            + ", items="
            + String(len(studio.asset_browser_entries))
            + ")  |  MCP v2 CLIENTS="
            + String(mcp_clients),
            studio.ui_ctx.theme.accent,
        )
        var ab_cards_style = FlexStyle.row(gap=4.0, pad=1.0)
        ab_cards_style.fixed_h = Float32(26.0)
        ab_cards_style.min_h = Float32(26.0)
        ab_cards_style.shrink = Float32(0.0)
        studio.ui_ctx.begin_box("ab_spawn_cards", ab_cards_style)
        if studio.ui_ctx.button("card_empty", "Empty", 62.0, 24.0):
            _ = studio.spawn_card_entity(0)
        if studio.ui_ctx.button("card_cube", "Cube", 58.0, 24.0):
            _ = studio.spawn_card_entity(1)
        if studio.ui_ctx.button("card_point_light", "Point Light", 92.0, 24.0):
            _ = studio.spawn_card_entity(2)
        if studio.ui_ctx.button("card_spot_light", "Spot Light", 88.0, 24.0):
            _ = studio.spawn_card_entity(3)
        if studio.ui_ctx.button("card_camera", "Camera", 74.0, 24.0):
            _ = studio.spawn_card_entity(4)
        if studio.ui_ctx.button("card_audio_emitter", "Audio Emitter", 116.0, 24.0):
            _ = studio.spawn_card_entity(5)
        if studio.ui_ctx.button("ab_rescan", "Rescan", 66.0, 24.0):
            _ = studio.scan_asset_browser()
        if studio.ui_ctx.button("ab_import", "Import", 62.0, 24.0):
            if studio.import_src_input.byte_length() > 0:
                var imp_src = studio.import_src_input
                var imp_dst = studio.import_dst_subdir
                _ = studio.import_asset_to_project(imp_src, imp_dst)
            else:
                studio.push_toast("Set import_src_input or use MCP asset.import", 1, 2.5)
        studio.ui_ctx.end_box()

        var ab_scroll_h = max(Float32(40.0), ab_h - Float32(66.0))
        studio.ui_ctx.begin_scroll_view("ab_entries_scroll", 0.0, ab_scroll_h)
        var entry_bw = max(Float32(180.0), ab_w - Float32(28.0))
        for ai in range(len(studio.asset_browser_entries)):
            var entry_name = studio.asset_browser_entries[ai]
            var ab_clicked = studio.ui_ctx.button("ab_item_" + String(ai), entry_name, entry_bw, 22.0)
            _ = studio.ui_ctx.begin_drag_source("ASSET_ENTRY", ai, entry_name)
            if ab_clicked:
                var e_low = FlintStudioState._ascii_lower(entry_name)
                var tgt = studio.selected_entity_id if (e_low.endswith(".flmat") or e_low.endswith(".flrig") or e_low.endswith(".flanim")) else -1
                _ = studio.activate_asset_entry(
                    entry_name,
                    target_entity_id=tgt,
                    drop_pos=Vec3(Float32(ai % 6) * Float32(0.5), Float32(0.0), Float32(0.0)),
                    from_drop=False,
                )
        studio.ui_ctx.end_scroll_view()
        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "MCP v2"):
        studio.ui_ctx.label("MCP v2 Active Clients: " + String(mcp_clients))
        studio.ui_ctx.label("Sandbox: strict workspace_root + SO_PEERCRED + 0600 socket", studio.ui_ctx.theme.text_muted)
        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "InputMap"):
        var im_info = studio.dock.find_panel("InputMap")
        var im_w = max(Float32(200.0), im_info[3].w - Float32(16.0))
        var im_h = max(Float32(60.0), im_info[3].h - Float32(16.0))
        if studio.input_rebind_active:
            studio.ui_ctx.label(
                "Press key for " + studio.input_rebind_context + "." + studio.input_rebind_action + " (Esc cancels)",
                studio.ui_ctx.theme.accent,
            )
        else:
            studio.ui_ctx.label(
                "InputMap (" + String(len(studio.input_map.contexts)) + " contexts, " + String(len(studio.input_map.actions)) + " actions)",
                studio.ui_ctx.theme.text_muted,
            )
        var im_bar = FlexStyle.row(gap=6.0, pad=0.0)
        im_bar.fixed_h = Float32(24.0)
        im_bar.min_h = Float32(24.0)
        im_bar.shrink = Float32(0.0)
        studio.ui_ctx.begin_box("imap_actions_bar", im_bar)
        if studio.ui_ctx.button("imap_save_btn", "Save JSON", 86.0, 22.0):
            _ = studio.save_input_map_file()
        if studio.ui_ctx.button("imap_load_btn", "Load JSON", 86.0, 22.0):
            _ = studio.load_input_map_file()
        if studio.input_rebind_active:
            if studio.ui_ctx.button("imap_cancel_btn", "Cancel", 68.0, 22.0):
                studio.cancel_input_rebind()
        studio.ui_ctx.end_box()

        var im_scroll_h = max(Float32(30.0), im_h - Float32(52.0))
        studio.ui_ctx.begin_scroll_view("imap_scroll", 0.0, im_scroll_h)
        var row_lbl_w = max(Float32(110.0), im_w - Float32(118.0))
        for ci in range(len(studio.input_map.contexts)):
            var ctx_name = studio.input_map.contexts[ci].name
            var ctx_pri = studio.input_map.contexts[ci].priority
            var n_b = len(studio.input_map.contexts[ci].bindings)
            if studio.ui_ctx.tree_node("imap_ctx_" + String(ci), ctx_name + " (prio " + String(ctx_pri) + ")", default_open=True):
                for bi in range(n_b):
                    var b = studio.input_map.contexts[ci].bindings[bi]
                    studio.ui_ctx.begin_box("imap_row_" + String(ci) + "_" + String(bi), FlexStyle.row(gap=6.0, pad=0.0))
                    var key_str = _studio_format_input_binding(b)
                    var is_rebinding = (
                        studio.input_rebind_active
                        and studio.input_rebind_context == ctx_name
                        and studio.input_rebind_action == b.action_name
                        and b.source_kind == BIND_KEY
                    )
                    studio.ui_ctx.label(
                        b.action_name + ": " + key_str,
                        studio.ui_ctx.theme.accent if is_rebinding else studio.ui_ctx.theme.text_primary,
                    )
                    if b.source_kind == BIND_KEY:
                        var btn_txt = String("Listening...") if is_rebinding else String("Rebind")
                        if studio.ui_ctx.button("imap_reb_" + String(ci) + "_" + String(bi), btn_txt, min(Float32(92.0), row_lbl_w), 20.0):
                            _ = studio.begin_input_rebind(ctx_name, b.action_name)
                    studio.ui_ctx.end_box()
                studio.ui_ctx.tree_pop()
        studio.ui_ctx.end_scroll_view()
        studio.ui_ctx.end_dock_panel()

    # Draw legacy SceneMenu dropdown only when opened via FLINT brand menu button (or legacy Toolbar)
    var toolbar = studio.dock.find_panel("Toolbar")
    if (toolbar[0] >= 0 and toolbar[2]) or studio.menu.bar.group >= 0:
        ref bar = studio.menu.bar
        if toolbar[0] >= 0 and toolbar[2]:
            bar.x = Int(toolbar[3].x) + 4
            bar.y = Int(toolbar[3].y)
            bar.height = min(30, max(1, Int(toolbar[3].h)))
        else:
            bar.x = 4
            bar.y = 0
            bar.height = Int(STUDIO_HEADER_BAR_H)
        bar.screen_w = Int(sw)
        bar.screen_h = Int(sh)
        bar.draw(studio.ui_ctx.draw_list, studio.ui_ctx.font, font_slot)

    # Frosted-glass Toast Stack + Background Job Progress Pills (anchored top-right / bottom-right inside vp)
    var toast_act = draw_toast_stack(
        studio.toast_mgr,
        studio.ui_ctx.draw_list,
        font_slot,
        studio.ui_ctx.font,
        studio.studio_theme,
        studio.ui_ctx.animator,
        vp,
        studio.ui_ctx.mouse_x,
        studio.ui_ctx.mouse_y,
        studio.ui_ctx.mouse_pressed_l and (not overlay_blocked),
        top_offset=Float32(48.0),
    )
    if toast_act.action_entity_id >= 0 and studio.is_alive_id(toast_act.action_entity_id):
        studio.select_entity(toast_act.action_entity_id)
        studio.frame_selection()

    var job_act = draw_job_progress_pills(
        studio.job_tracker,
        studio.ui_ctx.draw_list,
        font_slot,
        studio.ui_ctx.font,
        studio.studio_theme,
        studio.ui_ctx.animator,
        vp,
        studio.ui_ctx.mouse_x,
        studio.ui_ctx.mouse_y,
        studio.ui_ctx.mouse_pressed_l and (not overlay_blocked),
        bottom_offset=Float32(122.0),
    )
    if job_act.cancel_job_id != 0:
        _ = studio.job_tracker.request_cancel(job_act.cancel_job_id)

    # Non-focus-stealing Key-Chord Hint Overlay (bottom-left inside vp)
    draw_key_chord_hud(
        studio.key_chord_hud,
        studio.ui_ctx.draw_list,
        font_slot,
        studio.ui_ctx.font,
        studio.studio_theme,
        studio.ui_ctx.animator,
        vp,
    )

    # Topmost Frosted-Glass Command Palette (`Ctrl+K` / `Shift+A`)
    var pal_res = draw_command_palette(
        studio.command_palette,
        studio.ui_ctx.draw_list,
        font_slot,
        studio.ui_ctx.font,
        studio.studio_theme,
        studio.ui_ctx.animator,
        sw,
        sh,
        studio.ui_ctx.mouse_x,
        studio.ui_ctx.mouse_y,
        studio.ui_ctx.mouse_down_l and (not studio.confirm_modal_open),
        studio.ui_ctx.mouse_pressed_l and (not studio.confirm_modal_open),
        studio.ui_ctx.key_shift,
    )
    if pal_res.executed:
        _ = studio.execute_palette_action(pal_res)

    _emit_play_banner(studio, vp)
    _emit_status_bar(studio, sw, sh)
    _emit_studio_modals(studio)

    # Unsaved changes confirmation modal (`Ctrl+Q`, window close, or `File -> New scene`)
    if studio.confirm_modal_open:
        if studio.ui_ctx.begin_modal("unsaved_confirm_modal", "Unsaved Changes", Float32(390.0), Float32(138.0), studio.confirm_modal_open):
            studio.ui_ctx.label("The current scene has unsaved changes.")
            studio.ui_ctx.begin_box("unsaved_modal_btns", FlexStyle.row(gap=8.0, pad=2.0))
            if studio.ui_ctx.button("modal_save_cont", "Save & Continue", 128.0, 26.0):
                if studio.pie.mode != PIE_STOPPED:
                    try:
                        _ = studio.pie_stop()
                    except:
                        studio._abort_pie_session()
                var saved_ok = studio.save_scene_file()
                if saved_ok:
                    var act = studio.confirm_modal_action
                    studio.confirm_modal_open = False
                    studio.confirm_modal_action = 0
                    if act == 1:
                        studio.quit_requested = True
                    elif act == 2:
                        _ = studio.new_scene(force=True)
                    elif act == 3:
                        var pend_path = studio.pending_load_path
                        _ = studio.request_load_scene(pend_path, force=True)
            if studio.ui_ctx.button("modal_discard", "Discard", 96.0, 26.0):
                var act = studio.confirm_modal_action
                studio.confirm_modal_open = False
                studio.confirm_modal_action = 0
                if act == 1:
                    studio.discard_autosave()  # Discard means discard: no recovery offer next launch
                    studio.quit_requested = True
                elif act == 2:
                    _ = studio.new_scene(force=True)
                elif act == 3:
                    var pend_path = studio.pending_load_path
                    _ = studio.request_load_scene(pend_path, force=True)
            if studio.ui_ctx.button("modal_cancel", "Cancel", 86.0, 26.0):
                studio.confirm_modal_open = False
                studio.confirm_modal_action = 0
            studio.ui_ctx.end_box()
            studio.ui_ctx.end_modal()
        if not studio.confirm_modal_open:
            studio.confirm_modal_action = 0

    studio.ui_ctx.end_frame()

    for i in range(len(studio.ui_ctx.draw_list.prims)):
        if (studio.ui_ctx.draw_list.prims[i].header & 0xF) == PRIM_GLYPH and studio.ui_ctx.draw_list.prims[i].param0 < Float32(0.5):
            studio.ui_ctx.draw_list.prims[i].param0 = max(Float32(1.5), studio.ui_ctx.draw_list.prims[i].param0 * Float32(255.0))
    if abs(ui_sc - Float32(1.0)) > Float32(1.0e-4):
        studio.ui_ctx.draw_list.scale_all(ui_sc)
    for i in range(len(studio.ui_ctx.draw_list.prims)):
        ui.prims.append(studio.ui_ctx.draw_list.prims[i])

"""Flint Studio Visual Scene Editor State, Reversible Undo/Redo, 1-Click LZ4 PIE & Procedural Forge (`studio_app.mojo`).

Implements:
  1. `StudioEntityNode` (`Serializable`, `Inspectable`): Authoritative Studio scene graph entity table
     (name, TRS, parent, generational mesh/material handles, PBR material overrides, collider, rigid-body mass, visibility/lock).
  2. `trait Inspectable` + `struct FieldVisitor`: Zero-macro reflection driving both the Studio Inspector UI
     and generic AI/editor property patching.
  3. Reversible Coalesced `EditorTransaction` Undo/Redo Stack (`undo()`, `redo()`, `state_hash()`):
     Exact algebraic invertibility (`fnv64` match at every step across spawn, despawn, and coalesced TRS/material edits).
  4. **1-Click Play-In-Editor (`PieSession`)** backed by LZ4 `checkpoint.mojo`:
     `pie_play()`, `pie_pause()`, `pie_step()`, `pie_stop()` with deterministic IEEE-754 physics stepping
     and verified bit-for-bit identical `fnv64` world restoration on stop!
  5. Analytical scene collider raycasting (`Sphere`, `OBB Box`, `Capsule`, `Bvh8MeshCollider`) & Procedural Forge
     bake-to-`.glb` (`flint.procedural` + `flint.assets.glb_writer`).
  6. Docked AAA Studio Editor UI builder (`emit_studio_ui_draw_list`) rendering Hierarchy, Viewport + 3D Gizmos,
     Asset/PIE Dock, and `FieldVisitor` Inspector into GPU `UiDrawList`.
"""
from std.math import abs, cos, max, min, sin, sqrt
from flint.core.detmath import PI, det_atan2, set_sim_float_mode
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobReader, BlobWriter, Serializable, fnv64, fourcc
from flint.core.transform import Mat4, trs
from flint.anim.flanim_ecs import SteppedAnimClock
from flint.assets.checkpoint import pack_checkpoint_lz4, unpack_checkpoint_lz4
from flint.assets.flpk_v3 import UntypedHandle
from flint.assets.glb_writer import encode_native_glb, write_native_glb
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh, NativeModel
from flint.camera.rig import CAM_MODE_FREE_FLY, CAM_MODE_ORBIT_FOLLOW, CameraRig
from flint.input.input_map import (
    ACTION_AXIS2D,
    ACTION_BOOL,
    InputContext,
    InputMap,
    MOD_CTRL,
)
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.world import PhysicsWorld
from flint.platform.input import InputState
from flint.procedural.arch_kit import (
    build_colonnade_arcade_bay,
    build_street_bollard_and_grate,
)
from flint.procedural.foliage import (
    build_botanical_oak,
    build_botanical_pine,
)
from flint.procedural.subd import SubdCage, subdivide_to_mesh
from flint.render.font import SdfFont
from flint.studio.gizmo import (
    GIZMO_AXIS_NONE,
    GIZMO_MODE_ROTATE,
    GIZMO_MODE_SCALE,
    GIZMO_MODE_TRANSLATE,
    emit_gizmo_draw_list,
)
from flint.ui.context import UiContext, pack_rgba8
from flint.ui.draw_list import PRIM_GLYPH, UiDrawList
from flint.ui.flexbox_dock import (
    DOCK_SPLIT_H,
    DOCK_SPLIT_V,
    DockSpace,
    FlexStyle,
    RectF,
)

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

comptime FIELD_BOOL: Int = 1
comptime FIELD_INT: Int = 2
comptime FIELD_F32: Int = 3
comptime FIELD_STRING: Int = 4
comptime FIELD_VEC3: Int = 5
comptime FIELD_QUAT: Int = 6
comptime FIELD_COLOR_RGB: Int = 7


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
# 2. `StudioEntityNode` Scene Table Entry
# ============================================================================

struct StudioEntityNode(Copyable, Movable, Serializable, Inspectable):
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
    var collider_kind: Int
    var collider_half_extents: Vec3
    var mass: Float32
    var restitution: Float32
    var friction: Float32
    var visible: Bool
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

    def __init__(out self, entity_id: Int = 0, name: String = "Entity"):
        self.entity_id = entity_id
        self.alive = True
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
        self.collider_kind = COLLIDER_BOX
        self.collider_half_extents = Vec3(0.5, 0.5, 0.5)
        self.mass = Float32(0.0)  # 0.0 = static, > 0.0 = dynamic rigid body
        self.restitution = Float32(0.32)
        self.friction = Float32(0.55)
        self.visible = True
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
        visitor.visit_int("collider_kind", self.collider_kind)
        visitor.visit_vec3("collider_half_extents", self.collider_half_extents)
        visitor.visit_f32("mass", self.mass)
        visitor.visit_bool("visible", self.visible)
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

    def serialize(self, mut out: BlobWriter):
        out.begin(fourcc("SENT"), 2)
        out.integer(self.entity_id)
        out.boolean(self.alive)
        out.text(self.name)
        out.integer(self.parent_id)
        out.vec3(self.translation)
        out.quat(self.rotation)
        out.vec3(self.scale)
        out.vec3(self.linear_velocity)
        out.vec3(self.angular_velocity)
        out.u64(self.mesh_handle.packed)
        out.integer(self.mesh_id)
        out.u64(self.material_handle.packed)
        out.integer(self.material_id)
        out.vec3(self.base_color)
        out.f32(self.roughness)
        out.f32(self.metallic)
        out.vec3(self.emissive)
        out.integer(self.collider_kind)
        out.vec3(self.collider_half_extents)
        out.f32(self.mass)
        out.f32(self.restitution)
        out.f32(self.friction)
        out.boolean(self.visible)
        out.boolean(self.locked)
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
        out.end()

    def deserialize(mut self, mut input: BlobReader) raises:
        var ver = input.enter(fourcc("SENT"), 2)
        self.entity_id = input.integer()
        self.alive = input.boolean()
        self.name = input.text()
        self.parent_id = input.integer()
        self.translation = input.vec3()
        self.rotation = input.quat()
        self.scale = input.vec3()
        self.linear_velocity = input.vec3()
        self.angular_velocity = input.vec3()
        self.mesh_handle = UntypedHandle(input.u64())
        self.mesh_id = input.integer()
        self.material_handle = UntypedHandle(input.u64())
        self.material_id = input.integer()
        self.base_color = input.vec3()
        self.roughness = input.f32()
        self.metallic = input.f32()
        self.emissive = input.vec3()
        self.collider_kind = input.integer()
        self.collider_half_extents = input.vec3()
        self.mass = input.f32()
        self.restitution = input.f32()
        self.friction = input.f32()
        self.visible = input.boolean()
        self.locked = input.boolean()
        if ver >= 2:
            self.shading_mode = input.integer()
            self.cel_shadow_tint = input.vec3()
            self.cel_outline_width_px = input.f32()
            self.cel_outline_color = input.vec3()
            self.cel_rim_color = input.vec3()
            self.cel_hair_halo = input.f32()
            self.vfx_erosion = input.f32()
            self.vfx_scroll_speed = input.f32()
            self.vfx_core_color = input.vec3()
            self.vfx_edge_color = input.vec3()
            self.anim_step_hz = input.f32()
            self.smear_stretch = input.f32()
        input.leave()


# ============================================================================
# 3. Reversible Coalesced `EditorTransaction` & `PieSession`
# ============================================================================

struct EditorTransaction(Copyable, Movable):
    var op: Int
    var entity_id: Int
    var appended_slot: Bool
    var before_node: StudioEntityNode
    var after_node: StudioEntityNode
    var prev_selected_id: Int
    var next_selected_id: Int
    var coalesce_key: UInt64
    var label: String

    def __init__(out self, op: Int = TX_PATCH_ENTITY, entity_id: Int = 0, label: String = "Edit"):
        self.op = op
        self.entity_id = entity_id
        self.appended_slot = False
        self.before_node = StudioEntityNode(entity_id)
        self.after_node = StudioEntityNode(entity_id)
        self.prev_selected_id = -1
        self.next_selected_id = -1
        self.coalesce_key = 0
        self.label = label


struct PieSession(Copyable, Movable):
    """1-Click Play-In-Editor session state backed by an LZ4 integrity-checked binary snapshot."""
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


# ============================================================================
# 4. `FlintStudioState`: Scene Graph, Undo/Redo, Deterministic Physics & PIE
# ============================================================================

def _create_studio_dockspace() -> DockSpace:
    """Builds the default 5-panel `DockSpace` layout for `Flint Studio v5`:
    Top `Toolbar`, Left `Hierarchy`, Central `SceneViewport`, Bottom `AssetBrowser`, Right `Inspector`."""
    var d = DockSpace()
    d.header_height = 22.0
    d.min_ratio = 0.05
    var vp = d.add_central_viewport("SceneViewport")
    var s_top = d.split_node(vp, DOCK_SPLIT_V, 0.082, [String("Toolbar")], place_new_first=True)
    var s_left = d.split_node(s_top[1], DOCK_SPLIT_H, 0.19, [String("Hierarchy"), String("Outliner")], place_new_first=True)
    var s_right = d.split_node(s_left[1], DOCK_SPLIT_H, 0.72, [String("Inspector"), String("CameraRig")], place_new_first=False)
    _ = d.split_node(s_right[0], DOCK_SPLIT_V, 0.77, [String("AssetBrowser"), String("MCP v2")], place_new_first=False)
    return d^


def _create_studio_input_map() -> InputMap:
    """Declarative `InputMap` for `Flint Studio`: `W`/`E`/`R` gizmos, `Ctrl+Z`/`Ctrl+Y` undo/redo, `F5` PIE."""
    var imap = InputMap()
    _ = imap.register_action("gizmo_translate", ACTION_BOOL)
    _ = imap.register_action("gizmo_rotate", ACTION_BOOL)
    _ = imap.register_action("gizmo_scale", ACTION_BOOL)
    _ = imap.register_action("editor_undo", ACTION_BOOL)
    _ = imap.register_action("editor_redo", ACTION_BOOL)
    _ = imap.register_action("pie_toggle", ACTION_BOOL)
    _ = imap.register_action("editor_quit", ACTION_BOOL)
    _ = imap.register_action("cam_orbit", ACTION_AXIS2D)

    var ctx = InputContext("StudioEditor", priority=10)
    ctx.bind_key("gizmo_translate", 87)                  # W
    ctx.bind_key("gizmo_rotate", 69)                     # E
    ctx.bind_key("gizmo_scale", 82)                      # R
    ctx.bind_key("editor_undo", 90, modifiers=MOD_CTRL)  # Ctrl+Z
    ctx.bind_key("editor_redo", 89, modifiers=MOD_CTRL)  # Ctrl+Y
    ctx.bind_key("pie_toggle", 294)                      # F5
    ctx.bind_key("pie_toggle", 80)                       # P (secondary shortcut)
    ctx.bind_key("editor_quit", 256)                     # Escape
    ctx.bind_key("cam_orbit", 263, scale_x=-1.0, scale_y=0.0, consume=False)  # Left
    ctx.bind_key("cam_orbit", 262, scale_x=1.0, scale_y=0.0, consume=False)   # Right
    ctx.bind_key("cam_orbit", 265, scale_x=0.0, scale_y=-1.0, consume=False)  # Up (look up)
    ctx.bind_key("cam_orbit", 264, scale_x=0.0, scale_y=1.0, consume=False)   # Down (look down)
    imap.add_context(ctx^)
    imap.push_context("StudioEditor")
    return imap^


struct FlintStudioState(Movable, Serializable):
    var entities: List[StudioEntityNode]
    var static_bvh_colliders: List[Bvh8MeshCollider]
    var entity_bvh_slot: List[Int]
    var selected_entity_id: Int
    var gizmo_mode: Int
    var active_gizmo_handle: Int
    var snap_enabled: Bool
    var grid_snap_m: Float32
    var angle_snap_deg: Float32
    var sim_tick: Int
    var gravity: Vec3
    var ground_plane_y: Float32
    var has_ground_plane: Bool

    var undo_stack: List[EditorTransaction]
    var redo_stack: List[EditorTransaction]
    var pie: PieSession

    # Studio-level Stepped Animation Clock & Preview NPR Post-FX State
    var stepped_clock: SteppedAnimClock
    var preview_impact_blend: Float32
    var preview_impact_invert: Bool
    var preview_shockwave_active: Bool
    var preview_shockwave_radius: Float32

    # Integrated v5 Subsystems: `DockSpace`, `UiContext`, `CameraRig`, and `InputMap`
    var dock: DockSpace
    var ui_ctx: UiContext
    var camera_rig: CameraRig
    var cam_world: PhysicsWorld
    var input_map: InputMap
    var cam_target: Vec3
    var ui_edit_seq: UInt64

    # Synced Editor Viewport Camera Vectors
    var cam_pos: Vec3
    var cam_fwd: Vec3
    var cam_right: Vec3
    var cam_up: Vec3
    var fov_y_deg: Float32
    var near_z: Float32

    def __init__(out self):
        self.entities = List[StudioEntityNode]()
        self.static_bvh_colliders = List[Bvh8MeshCollider]()
        self.entity_bvh_slot = List[Int]()
        self.selected_entity_id = -1
        self.gizmo_mode = GIZMO_MODE_TRANSLATE
        self.active_gizmo_handle = GIZMO_AXIS_NONE
        self.snap_enabled = True
        self.grid_snap_m = Float32(0.25)
        self.angle_snap_deg = Float32(15.0)
        self.sim_tick = 0
        self.gravity = Vec3(0.0, -9.81, 0.0)
        self.ground_plane_y = Float32(0.0)
        self.has_ground_plane = True

        self.undo_stack = List[EditorTransaction]()
        self.redo_stack = List[EditorTransaction]()
        self.pie = PieSession()

        self.stepped_clock = SteppedAnimClock()
        self.preview_impact_blend = Float32(0.0)
        self.preview_impact_invert = False
        self.preview_shockwave_active = False
        self.preview_shockwave_radius = Float32(0.22)

        self.dock = _create_studio_dockspace()
        self.ui_ctx = UiContext(preset_hud=False)
        self.ui_ctx.theme.font_size = Float32(13.0)
        self.ui_ctx.theme.row_height = Float32(22.0)
        self.camera_rig = CameraRig(CAM_MODE_ORBIT_FOLLOW)
        self.camera_rig.fov_deg = Float32(52.0)
        self.camera_rig.look_offset = Vec3(0.0, 0.0, 0.0)
        self.cam_world = PhysicsWorld()
        self.input_map = _create_studio_input_map()
        self.cam_target = Vec3(0.0, 0.9, 1.0)
        self.ui_edit_seq = UInt64(50000)

        self.cam_pos = Vec3(-6.5, 4.2, -9.5)
        self.cam_fwd = Vec3(0.52, -0.26, 0.81).normalized()
        self.cam_right = Vec3(0.0, 1.0, 0.0).cross(self.cam_fwd).normalized()
        self.cam_up = self.cam_fwd.cross(self.cam_right).normalized()
        self.fov_y_deg = Float32(52.0)
        self.near_z = Float32(0.15)
        self.look_at(Vec3(-6.5, 4.2, -9.5), Vec3(0.0, 0.9, 1.0))

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
        self.camera_rig.update(self.cam_world, target, Float32(0.0))
        self._sync_from_camera_rig()

    def update_camera(
        mut self,
        dt: Float32,
        orbit_delta_yaw_deg: Float32 = 0.0,
        orbit_delta_pitch_deg: Float32 = 0.0,
        fly_move_local: Vec3 = Vec3(),
        fly_speed: Float32 = 8.0,
    ):
        if orbit_delta_yaw_deg != Float32(0.0) or orbit_delta_pitch_deg != Float32(0.0):
            self.camera_rig.rotate_orbit(orbit_delta_yaw_deg, orbit_delta_pitch_deg)
        self.camera_rig.update(self.cam_world, self.cam_target, dt, fly_move_local=fly_move_local, fly_speed=fly_speed)
        self._sync_from_camera_rig()

    def viewport_rect(mut self, screen_w: Int, screen_h: Int) -> RectF:
        self.dock.solve_layout(RectF(0.0, 0.0, Float32(screen_w), Float32(screen_h)))
        var vp = self.dock.find_panel("SceneViewport")
        if vp[0] >= 0:
            return vp[3]
        return RectF(Float32(236.0), Float32(46.0), Float32(max(160, screen_w - 524)), Float32(max(120, screen_h - 184)))

    def process_editor_input(mut self, input: InputState, dt: Float32) -> Bool:
        """Evaluates `InputMap` editor hotkeys (`W/E/R`, `Ctrl+Z`, `Ctrl+Y`, `F5` PIE) & `CameraRig` orbit/fly.
        Returns `False` when `editor_quit` (`Esc`) is triggered."""
        self.input_map.update(input, dt)
        if self.input_map.just_pressed("editor_quit"):
            return False
        if self.input_map.just_pressed("editor_undo"):
            _ = self.undo()
        if self.input_map.just_pressed("editor_redo"):
            _ = self.redo()
        if self.input_map.just_pressed("pie_toggle"):
            if self.pie.mode == PIE_STOPPED:
                _ = self.pie_play()
            else:
                try:
                    _ = self.pie_stop()
                except:
                    self.pie.mode = PIE_STOPPED

        var rmb_fly = input.mouse_down(1)
        if not rmb_fly:
            if self.input_map.just_pressed("gizmo_translate"):
                self.gizmo_mode = GIZMO_MODE_TRANSLATE
            if self.input_map.just_pressed("gizmo_rotate"):
                self.gizmo_mode = GIZMO_MODE_ROTATE
            if self.input_map.just_pressed("gizmo_scale"):
                self.gizmo_mode = GIZMO_MODE_SCALE

        var orb = self.input_map.axis_2d("cam_orbit")
        var dyaw = orb[0] * Float32(80.0) * dt
        var dpitch = orb[1] * Float32(65.0) * dt
        var fly_move = Vec3()
        if rmb_fly:
            self.camera_rig.mode = CAM_MODE_FREE_FLY
            self.camera_rig.fly_pos = self.cam_pos
            dyaw += input.mouse_delta_x() * Float32(0.18)
            dpitch += input.mouse_delta_y() * Float32(0.18)
            if input.key_down(87): fly_move.z += Float32(1.0)   # W
            if input.key_down(83): fly_move.z -= Float32(1.0)   # S
            if input.key_down(68): fly_move.x += Float32(1.0)   # D
            if input.key_down(65): fly_move.x -= Float32(1.0)   # A
            if input.key_down(69): fly_move.y += Float32(1.0)   # E
            if input.key_down(81): fly_move.y -= Float32(1.0)   # Q
            self.update_camera(dt, dyaw, dpitch, fly_move_local=fly_move, fly_speed=Float32(8.5))
            self.cam_target = self.cam_pos + self.cam_fwd * self.camera_rig.spring_arm.target_length
            self.camera_rig.pivot_spring.value = self.cam_target
        else:
            self.camera_rig.mode = CAM_MODE_ORBIT_FOLLOW
            self.update_camera(dt, dyaw, dpitch)
        return True

    # ---- Authoritative Binary Serialization & `fnv64` State Hash ----

    def serialize(self, mut out: BlobWriter):
        out.begin(fourcc("FSTD"), 2)
        out.integer(self.selected_entity_id)
        out.integer(self.sim_tick)
        out.vec3(self.gravity)
        out.f32(self.ground_plane_y)
        out.boolean(self.has_ground_plane)
        out.u32(UInt32(len(self.entities)))
        for i in range(len(self.entities)):
            self.entities[i].serialize(out)
        out.f32(self.stepped_clock.raw_time_sec)
        out.f32(self.stepped_clock.stepped_time_sec)
        out.f32(self.stepped_clock.step_hz)
        out.f32(self.stepped_clock.hitstop_remaining_sec)
        out.boolean(self.stepped_clock.pose_changed_this_tick)
        out.f32(self.preview_impact_blend)
        out.boolean(self.preview_impact_invert)
        out.boolean(self.preview_shockwave_active)
        out.f32(self.preview_shockwave_radius)
        out.end()

    def deserialize(mut self, mut input: BlobReader) raises:
        var ver = input.enter(fourcc("FSTD"), 2)
        self.selected_entity_id = input.integer()
        self.sim_tick = input.integer()
        self.gravity = input.vec3()
        self.ground_plane_y = input.f32()
        self.has_ground_plane = input.boolean()
        var n_ent = Int(input.u32())
        var loaded = List[StudioEntityNode](capacity=n_ent)
        for _ in range(n_ent):
            var nd = StudioEntityNode()
            nd.deserialize(input)
            loaded.append(nd^)
        if ver >= 2:
            self.stepped_clock.raw_time_sec = input.f32()
            self.stepped_clock.stepped_time_sec = input.f32()
            self.stepped_clock.step_hz = input.f32()
            self.stepped_clock.hitstop_remaining_sec = input.f32()
            self.stepped_clock.pose_changed_this_tick = input.boolean()
            self.preview_impact_blend = input.f32()
            self.preview_impact_invert = input.boolean()
            self.preview_shockwave_active = input.boolean()
            self.preview_shockwave_radius = input.f32()
        input.leave()
        self.entities = loaded^
        while len(self.entity_bvh_slot) < len(self.entities):
            self.entity_bvh_slot.append(-1)

    def state_hash(self) -> UInt64:
        """Computes bit-exact `fnv64` digest over all authoritative Studio scene & simulation state."""
        var w = BlobWriter()
        self.serialize(w)
        return fnv64(w.data)

    def live_entity_count(self) -> Int:
        var cnt = 0
        for i in range(len(self.entities)):
            if self.entities[i].alive:
                cnt += 1
        return cnt

    def world_matrix(self, entity_id: Int) -> Mat4:
        if entity_id < 0 or entity_id >= len(self.entities):
            return Mat4()
        var local_m = self.entities[entity_id].local_matrix()
        var pid = self.entities[entity_id].parent_id
        var depth = 0
        while pid >= 0 and pid < len(self.entities) and depth < 16:
            var pm = self.entities[pid].local_matrix()
            local_m = pm.multiply(local_m)
            pid = self.entities[pid].parent_id
            depth += 1
        return local_m^

    # ---- Reversible Coalesced Transaction Stack ----

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
        var reuse_id = -1
        for i in range(len(self.entities)):
            if not self.entities[i].alive:
                reuse_id = i
                break

        var appended = reuse_id < 0
        var eid = len(self.entities) if appended else reuse_id

        var node = StudioEntityNode(eid, name)
        node.parent_id = parent_id
        node.translation = position
        node.rotation = rotation.normalized()
        node.scale = scale
        node.collider_kind = collider_kind
        node.collider_half_extents = collider_half_extents
        node.mass = max(Float32(0.0), mass)
        node.base_color = base_color
        node.roughness = max(Float32(0.04), min(Float32(1.0), roughness))
        node.metallic = max(Float32(0.0), min(Float32(1.0), metallic))
        node.mesh_id = mesh_id

        var tx = EditorTransaction(TX_SPAWN_ENTITY, eid, "Spawn " + name)
        tx.appended_slot = appended
        tx.prev_selected_id = self.selected_entity_id
        tx.next_selected_id = eid
        tx.after_node = node.copy()

        if appended:
            self.entities.append(node^)
            self.entity_bvh_slot.append(-1)
        else:
            tx.before_node = self.entities[eid].copy()
            self.entities[eid] = node^
            if eid < len(self.entity_bvh_slot):
                self.entity_bvh_slot[eid] = -1

        self.selected_entity_id = eid
        if record_undo:
            self.undo_stack.append(tx^)
            self.redo_stack.clear()
        return eid

    def despawn_entity(mut self, entity_id: Int, record_undo: Bool = True) -> Bool:
        if entity_id < 0 or entity_id >= len(self.entities) or not self.entities[entity_id].alive:
            return False
        var tx = EditorTransaction(TX_DESPAWN_ENTITY, entity_id, "Despawn " + self.entities[entity_id].name)
        tx.before_node = self.entities[entity_id].copy()
        tx.prev_selected_id = self.selected_entity_id
        self.entities[entity_id].alive = False
        if self.selected_entity_id == entity_id:
            self.selected_entity_id = -1
            for i in range(len(self.entities)):
                if self.entities[i].alive:
                    self.selected_entity_id = i
                    break
        tx.next_selected_id = self.selected_entity_id
        tx.after_node = self.entities[entity_id].copy()
        if record_undo:
            self.undo_stack.append(tx^)
            self.redo_stack.clear()
        return True

    def commit_node_edit(
        mut self,
        entity_id: Int,
        var updated_node: StudioEntityNode,
        label: String = "Edit Entity",
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        if entity_id < 0 or entity_id >= len(self.entities) or not self.entities[entity_id].alive:
            return False

        # Coalesce continuous interactive gizmo/slider drags with identical non-zero `coalesce_key`
        if coalesce_key != 0 and len(self.undo_stack) > 0 and len(self.redo_stack) == 0:
            ref top = self.undo_stack[len(self.undo_stack) - 1]
            if top.op == TX_PATCH_ENTITY and top.entity_id == entity_id and top.coalesce_key == coalesce_key:
                top.after_node = updated_node.copy()
                self.entities[entity_id] = updated_node^
                return True

        var tx = EditorTransaction(TX_PATCH_ENTITY, entity_id, label)
        tx.before_node = self.entities[entity_id].copy()
        tx.after_node = updated_node.copy()
        tx.prev_selected_id = self.selected_entity_id
        tx.next_selected_id = self.selected_entity_id
        tx.coalesce_key = coalesce_key
        self.entities[entity_id] = updated_node^
        self.undo_stack.append(tx^)
        self.redo_stack.clear()
        return True

    def set_entity_transform(
        mut self,
        entity_id: Int,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        if entity_id < 0 or entity_id >= len(self.entities) or not self.entities[entity_id].alive:
            return False
        var nd = self.entities[entity_id].copy()
        nd.translation = position
        nd.rotation = rotation.normalized()
        nd.scale = scale
        return self.commit_node_edit(entity_id, nd^, "Transform", coalesce_key)

    def patch_entity_field(
        mut self,
        entity_id: Int,
        field_name: String,
        new_val: InspectedField,
        coalesce_key: UInt64 = 0,
    ) -> Bool:
        if entity_id < 0 or entity_id >= len(self.entities) or not self.entities[entity_id].alive:
            return False
        var nd = self.entities[entity_id].copy()
        var vis = FieldVisitor.for_patch(field_name, new_val)
        nd.inspect_fields(vis)
        if not vis.mutated:
            return False
        return self.commit_node_edit(entity_id, nd^, "Patch " + field_name, coalesce_key)

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
        if entity_id < 0 or entity_id >= len(self.entities) or not self.entities[entity_id].alive:
            return False
        var nd = self.entities[entity_id].copy()
        nd.shading_mode = SHADING_NPR_CEL
        nd.cel_shadow_tint = cel_shadow_tint
        nd.cel_outline_width_px = max(Float32(0.0), min(Float32(24.0), cel_outline_width_px))
        nd.cel_outline_color = cel_outline_color
        nd.cel_rim_color = cel_rim_color
        nd.cel_hair_halo = max(Float32(0.0), cel_hair_halo)
        nd.anim_step_hz = max(Float32(0.0), anim_step_hz)
        nd.smear_stretch = max(Float32(0.55), min(Float32(3.2), smear_stretch))
        return self.commit_node_edit(entity_id, nd^, "Configure NPR Cel", coalesce_key)

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
        if entity_id < 0 or entity_id >= len(self.entities) or not self.entities[entity_id].alive:
            return False
        var nd = self.entities[entity_id].copy()
        nd.shading_mode = SHADING_STYLIZED_VFX
        nd.vfx_erosion = max(Float32(0.0), min(Float32(1.0), vfx_erosion))
        nd.vfx_scroll_speed = vfx_scroll_speed
        nd.vfx_core_color = vfx_core_color
        nd.vfx_edge_color = vfx_edge_color
        nd.anim_step_hz = max(Float32(0.0), anim_step_hz)
        nd.smear_stretch = max(Float32(0.55), min(Float32(3.2), smear_stretch))
        return self.commit_node_edit(entity_id, nd^, "Configure Stylized VFX", coalesce_key)

    def attach_static_bvh_collider(mut self, entity_id: Int, var bvh: Bvh8MeshCollider):
        if entity_id < 0 or entity_id >= len(self.entities):
            return
        while len(self.entity_bvh_slot) < len(self.entities):
            self.entity_bvh_slot.append(-1)
        var slot = len(self.static_bvh_colliders)
        self.static_bvh_colliders.append(bvh^)
        self.entity_bvh_slot[entity_id] = slot
        self.entities[entity_id].collider_kind = COLLIDER_STATIC_MESH

    def can_undo(self) -> Bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> Bool:
        return len(self.redo_stack) > 0

    def undo(mut self) -> Bool:
        if len(self.undo_stack) == 0:
            return False
        var tx = self.undo_stack.pop()
        if tx.op == TX_SPAWN_ENTITY:
            if tx.appended_slot and tx.entity_id == len(self.entities) - 1:
                _ = self.entities.pop()
                if len(self.entity_bvh_slot) > len(self.entities):
                    _ = self.entity_bvh_slot.pop()
            elif tx.entity_id >= 0 and tx.entity_id < len(self.entities):
                self.entities[tx.entity_id] = tx.before_node.copy()
            self.selected_entity_id = tx.prev_selected_id
        elif tx.op == TX_DESPAWN_ENTITY or tx.op == TX_PATCH_ENTITY:
            if tx.entity_id >= 0 and tx.entity_id < len(self.entities):
                self.entities[tx.entity_id] = tx.before_node.copy()
            self.selected_entity_id = tx.prev_selected_id
        self.redo_stack.append(tx^)
        return True

    def redo(mut self) -> Bool:
        if len(self.redo_stack) == 0:
            return False
        var tx = self.redo_stack.pop()
        if tx.op == TX_SPAWN_ENTITY:
            if tx.appended_slot and tx.entity_id == len(self.entities):
                self.entities.append(tx.after_node.copy())
                self.entity_bvh_slot.append(-1)
            elif tx.entity_id >= 0 and tx.entity_id < len(self.entities):
                self.entities[tx.entity_id] = tx.after_node.copy()
            self.selected_entity_id = tx.next_selected_id
        elif tx.op == TX_DESPAWN_ENTITY or tx.op == TX_PATCH_ENTITY:
            if tx.entity_id >= 0 and tx.entity_id < len(self.entities):
                self.entities[tx.entity_id] = tx.after_node.copy()
            self.selected_entity_id = tx.next_selected_id
        self.undo_stack.append(tx^)
        return True

    # ---- Deterministic Physics Step & Analytical Raycast ----

    def step_physics_deterministic(mut self, ticks: Int = 1, dt: Float32 = Float32(1.0 / 60.0)) -> Int:
        """Advances deterministic IEEE-754 rigid-body dynamics across `ticks` sub-stepped frames."""
        if ticks <= 0:
            return 0
        set_sim_float_mode()
        comptime SUBSTEPS: Int = 4
        var h = dt / Float32(SUBSTEPS)
        var n = len(self.entities)

        for _ in range(ticks):
            for _ in range(SUBSTEPS):
                # 1. Symplectic Euler velocity + position integration for dynamic bodies (`mass > 0`)
                for i in range(n):
                    if not self.entities[i].alive or self.entities[i].mass <= Float32(0.0):
                        continue
                    var lv = self.entities[i].linear_velocity + self.gravity * h
                    lv = lv * max(Float32(0.0), Float32(1.0) - Float32(0.04) * h)
                    var pos = self.entities[i].translation + lv * h

                    var w = self.entities[i].angular_velocity
                    if w.length_squared() > Float32(1.0e-10):
                        var q = self.entities[i].rotation
                        var dq = Quat(
                            Float32(0.5) * h * (w.x * q.w + w.y * q.z - w.z * q.y),
                            Float32(0.5) * h * (-w.x * q.z + w.y * q.w + w.z * q.x),
                            Float32(0.5) * h * (w.x * q.y - w.y * q.x + w.z * q.w),
                            Float32(0.5) * h * (-w.x * q.x - w.y * q.y - w.z * q.z),
                        )
                        self.entities[i].rotation = Quat(q.x + dq.x, q.y + dq.y, q.z + dq.z, q.w + dq.w).normalized()

                    # Ground plane contact solve
                    if self.has_ground_plane and self.entities[i].collider_kind != COLLIDER_NONE:
                        var half_y = self.entities[i].collider_half_extents.y * self.entities[i].scale.y
                        if self.entities[i].collider_kind == COLLIDER_SPHERE:
                            half_y = self.entities[i].collider_half_extents.x * self.entities[i].scale.x
                        var min_y = self.ground_plane_y + max(Float32(0.02), half_y)
                        if pos.y < min_y:
                            pos.y = min_y
                            if lv.y < Float32(0.0):
                                var rest = self.entities[i].restitution if (-lv.y) > Float32(0.5) else Float32(0.0)
                                lv.y = -lv.y * rest
                                var fric_damp = max(Float32(0.0), Float32(1.0) - self.entities[i].friction * Float32(4.0) * h)
                                lv.x *= fric_damp
                                lv.z *= fric_damp

                    self.entities[i].translation = pos
                    self.entities[i].linear_velocity = lv

                # 2. Pairwise dynamic-vs-dynamic / dynamic-vs-static sphere-proxy & BVH8 collision resolution
                for i in range(n):
                    if not self.entities[i].alive or self.entities[i].collider_kind == COLLIDER_NONE:
                        continue
                    for j in range(i + 1, n):
                        if not self.entities[j].alive or self.entities[j].collider_kind == COLLIDER_NONE:
                            continue
                        var dyn_a = self.entities[i].mass > Float32(0.0)
                        var dyn_b = self.entities[j].mass > Float32(0.0)
                        if not dyn_a and not dyn_b:
                            continue

                        var ra = max(Float32(0.05), self.entities[i].collider_half_extents.x * self.entities[i].scale.x)
                        var rb = max(Float32(0.05), self.entities[j].collider_half_extents.x * self.entities[j].scale.x)
                        var delta = self.entities[j].translation - self.entities[i].translation
                        var dist2 = delta.length_squared()
                        var r_sum = ra + rb
                        if dist2 < r_sum * r_sum:
                            var dist = sqrt(max(Float32(1.0e-8), dist2))
                            var nrm = (delta * (Float32(1.0) / dist)) if dist > Float32(1.0e-4) else Vec3(0.0, 1.0, 0.0)
                            var pen = r_sum - dist
                            var inv_ma = (Float32(1.0) / self.entities[i].mass) if dyn_a else Float32(0.0)
                            var inv_mb = (Float32(1.0) / self.entities[j].mass) if dyn_b else Float32(0.0)
                            var inv_sum = inv_ma + inv_mb
                            if inv_sum > Float32(0.0):
                                self.entities[i].translation = self.entities[i].translation - nrm * (pen * (inv_ma / inv_sum))
                                self.entities[j].translation = self.entities[j].translation + nrm * (pen * (inv_mb / inv_sum))
                                var rvel = (self.entities[j].linear_velocity - self.entities[i].linear_velocity).dot(nrm)
                                if rvel < Float32(0.0):
                                    var e = min(self.entities[i].restitution, self.entities[j].restitution)
                                    var j_imp = -(Float32(1.0) + e) * rvel / inv_sum
                                    self.entities[i].linear_velocity = self.entities[i].linear_velocity - nrm * (j_imp * inv_ma)
                                    self.entities[j].linear_velocity = self.entities[j].linear_velocity + nrm * (j_imp * inv_mb)

            self.sim_tick += 1
        return ticks

    # ---- 1-Click Play-In-Editor (`PieSession` + LZ4 `checkpoint.mojo`) ----

    def pie_play(mut self) -> Bool:
        if self.pie.mode == PIE_PLAYING:
            return True
        if self.pie.mode == PIE_PAUSED:
            self.pie.mode = PIE_PLAYING
            return True
        var w = BlobWriter()
        self.serialize(w)
        self.pie.pre_play_hash = fnv64(w.data)
        self.pie.checkpoint_lz4 = pack_checkpoint_lz4(w)
        self.pie.ticks_simulated = 0
        self.pie.last_restore_verified = False
        self.pie.mode = PIE_PLAYING
        return len(self.pie.checkpoint_lz4) > 0

    def pie_pause(mut self) -> Bool:
        if self.pie.mode == PIE_PLAYING:
            self.pie.mode = PIE_PAUSED
            return True
        if self.pie.mode == PIE_PAUSED:
            self.pie.mode = PIE_PLAYING
            return True
        return False

    def pie_step(mut self, ticks: Int = 1, dt: Float32 = Float32(1.0 / 60.0)) -> Int:
        if self.pie.mode == PIE_STOPPED:
            if not self.pie_play():
                return 0
            self.pie.mode = PIE_PAUSED
        if (
            self.selected_entity_id >= 0
            and self.selected_entity_id < len(self.entities)
            and self.entities[self.selected_entity_id].alive
            and self.entities[self.selected_entity_id].anim_step_hz > Float32(0.0)
        ):
            self.stepped_clock.step_hz = self.entities[self.selected_entity_id].anim_step_hz
        var done = self.step_physics_deterministic(ticks, dt)
        for _ in range(done):
            _ = self.stepped_clock.advance(dt)
        self.pie.ticks_simulated += done
        return done

    def pie_stop(mut self) raises -> Bool:
        """Restores the exact pre-PIE world from the LZ4 checkpoint and verifies bit-for-bit `fnv64` equality."""
        if self.pie.mode == PIE_STOPPED or len(self.pie.checkpoint_lz4) == 0:
            self.pie.mode = PIE_STOPPED
            return False
        var reader = unpack_checkpoint_lz4(self.pie.checkpoint_lz4)
        self.deserialize(reader)
        self.pie.mode = PIE_STOPPED
        var restored_hash = self.state_hash()
        self.pie.last_restore_verified = restored_hash == self.pie.pre_play_hash
        if not self.pie.last_restore_verified:
            raise Error("PIE LZ4 world restore hash mismatch")
        return True

    # ---- Analytical Scene Collider Raycast ----

    def raycast_scene(self, origin: Vec3, direction: Vec3, max_distance: Float32 = 1000.0) -> StudioRaycastHit:
        var best = StudioRaycastHit()
        var d_len = direction.length()
        if d_len <= Float32(1.0e-7) or max_distance <= Float32(0.0):
            return best
        var dir = direction * (Float32(1.0) / d_len)
        best.distance = max_distance

        for i in range(len(self.entities)):
            ref ent = self.entities[i]
            if not ent.alive or not ent.visible or ent.collider_kind == COLLIDER_NONE:
                continue

            if ent.collider_kind == COLLIDER_SPHERE:
                var rad = max(Float32(0.01), ent.collider_half_extents.x * ent.scale.x)
                var oc = origin - ent.translation
                var b = oc.dot(dir)
                var c = oc.dot(oc) - rad * rad
                var disc = b * b - c
                if disc >= Float32(0.0):
                    var s = sqrt(disc)
                    var t = -b - s
                    if t < Float32(0.0):
                        t = -b + s
                    if t >= Float32(0.0) and t < best.distance:
                        best.hit = True
                        best.entity_id = i
                        best.distance = t
                        best.point = origin + dir * t
                        best.normal = (best.point - ent.translation).normalized()

            elif ent.collider_kind == COLLIDER_BOX or ent.collider_kind == COLLIDER_CAPSULE:
                var inv_q = ent.rotation.conjugate()
                var loc_o = inv_q.rotate(origin - ent.translation)
                var loc_d = inv_q.rotate(dir)
                var he = Vec3(
                    max(Float32(0.01), ent.collider_half_extents.x * ent.scale.x),
                    max(Float32(0.01), ent.collider_half_extents.y * ent.scale.y),
                    max(Float32(0.01), ent.collider_half_extents.z * ent.scale.z),
                )
                var tmin = Float32(0.0)
                var tmax = best.distance
                var hit_n = Vec3(0.0, 1.0, 0.0)
                var ok = True

                var o_arr = [loc_o.x, loc_o.y, loc_o.z]
                var d_arr = [loc_d.x, loc_d.y, loc_d.z]
                var h_arr = [he.x, he.y, he.z]
                var n_axes = [Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0)]

                for ax in range(3):
                    if abs(d_arr[ax]) < Float32(1.0e-7):
                        if o_arr[ax] < -h_arr[ax] or o_arr[ax] > h_arr[ax]:
                            ok = False
                            break
                    else:
                        var inv_d = Float32(1.0) / d_arr[ax]
                        var t1 = (-h_arr[ax] - o_arr[ax]) * inv_d
                        var t2 = (h_arr[ax] - o_arr[ax]) * inv_d
                        var n_cand = -n_axes[ax] if d_arr[ax] > Float32(0.0) else n_axes[ax]
                        if t1 > t2:
                            var tmp = t1; t1 = t2; t2 = tmp
                        if t1 > tmin:
                            tmin = t1
                            hit_n = n_cand
                        if t2 < tmax:
                            tmax = t2
                        if tmin > tmax:
                            ok = False
                            break
                if ok and tmin >= Float32(0.0) and tmin < best.distance:
                    best.hit = True
                    best.entity_id = i
                    best.distance = tmin
                    best.point = origin + dir * tmin
                    best.normal = ent.rotation.rotate(hit_n).normalized()

            elif ent.collider_kind == COLLIDER_STATIC_MESH and i < len(self.entity_bvh_slot):
                var bslot = self.entity_bvh_slot[i]
                if bslot >= 0 and bslot < len(self.static_bvh_colliders):
                    var inv_q = ent.rotation.conjugate()
                    var inv_s = Vec3(
                        Float32(1.0) / max(Float32(1.0e-4), ent.scale.x),
                        Float32(1.0) / max(Float32(1.0e-4), ent.scale.y),
                        Float32(1.0) / max(Float32(1.0e-4), ent.scale.z),
                    )
                    var loc_o_raw = inv_q.rotate(origin - ent.translation)
                    var loc_o = Vec3(loc_o_raw.x * inv_s.x, loc_o_raw.y * inv_s.y, loc_o_raw.z * inv_s.z)
                    var loc_d_raw = inv_q.rotate(dir)
                    var loc_d_scaled = Vec3(loc_d_raw.x * inv_s.x, loc_d_raw.y * inv_s.y, loc_d_raw.z * inv_s.z)
                    var d_scale = loc_d_scaled.length()
                    if d_scale > Float32(1.0e-7):
                        var loc_d_u = loc_d_scaled * (Float32(1.0) / d_scale)
                        var bhit = self.static_bvh_colliders[bslot].raycast(loc_o, loc_d_u, best.distance * d_scale)
                        if bhit.hit:
                            var world_t = bhit.t / d_scale
                            if world_t >= Float32(0.0) and world_t < best.distance:
                                best.hit = True
                                best.entity_id = i
                                best.distance = world_t
                                best.point = origin + dir * world_t
                                best.normal = ent.rotation.rotate(bhit.normal).normalized()

        if self.has_ground_plane and abs(dir.y) > Float32(1.0e-6):
            var tg = (self.ground_plane_y - origin.y) / dir.y
            if tg >= Float32(0.0) and tg < best.distance:
                best.hit = True
                best.entity_id = -2  # ground plane sentinel
                best.distance = tg
                best.point = origin + dir * tg
                best.normal = Vec3(0.0, 1.0, 0.0)

        return best


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

    # Outward-wound closed 6-quad manifold control cage
    cage.add_quad(v0, v1, v2, v3)  # +Z
    cage.add_quad(v5, v4, v7, v6)  # -Z
    cage.add_quad(v1, v5, v6, v2)  # +X
    cage.add_quad(v4, v0, v3, v7)  # -X
    cage.add_quad(v3, v2, v6, v7)  # +Y
    cage.add_quad(v4, v5, v1, v0)  # -Y

    # Semi-sharp studio base-stand ring
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
    var neg = v < Float32(0.0)
    var av = -v if neg else v
    var scaled = Int(av * Float32(100.0) + Float32(0.5))
    var whole = scaled // 100
    var frac = scaled % 100
    var sign = "-" if neg else ""
    var fstr = ("0" + String(frac)) if frac < 10 else String(frac)
    return sign + String(whole) + "." + fstr


@always_inline
def _abgr_to_argb(c: UInt32) -> UInt32:
    """Converts `UiContext` `pack_rgba8` (`0xAABBGGRR`) to `FlintAaaVulkanRenderer` `0xAARRGGBB`."""
    return (c & UInt32(0xFF00FF00)) | ((c & UInt32(0x000000FF)) << 16) | ((c >> 16) & UInt32(0x000000FF))


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
    (`begin_dock_panel`, `tree_node`, `drag_float3`, `slider_float`, `checkbox`, `button`)
    plus 3D viewport selection gizmos into `ui`."""
    var sw = Float32(max(320, screen_w))
    var sh = Float32(max(240, screen_h))

    studio.ui_ctx.font = font.copy()
    studio.ui_ctx.font_slot = font_slot
    studio.ui_ctx.theme.font_size = Float32(13.0)
    studio.ui_ctx.theme.row_height = Float32(22.0)
    # Clear root-relative Flexbox cache so `DockSpace` panel coordinates govern every widget!
    studio.ui_ctx.cached_rects.clear()

    studio.ui_ctx.begin_frame(input, sw, sh, dt)
    studio.ui_ctx.update_dock_space(studio.dock, RectF(0.0, 0.0, sw, sh))

    var vp_info = studio.dock.find_panel("SceneViewport")
    var vp = vp_info[3] if vp_info[0] >= 0 else RectF(Float32(240.0), Float32(54.0), sw - Float32(520.0), sh - Float32(190.0))

    # 1. 3D Viewport Active Selection Gizmo (emitted directly to `ui` which uses `0xAARRGGBB` colors!)
    if (
        studio.selected_entity_id >= 0
        and studio.selected_entity_id < len(studio.entities)
        and studio.entities[studio.selected_entity_id].alive
    ):
        var wm = studio.world_matrix(studio.selected_entity_id)
        var pivot = Vec3(wm.values[12], wm.values[13], wm.values[14])
        ui.push_clip(Int(vp.x), Int(vp.y), Int(vp.right()), Int(vp.bottom()))
        emit_gizmo_draw_list(
            ui,
            pivot,
            studio.gizmo_mode,
            studio.active_gizmo_handle,
            vp.x,
            vp.y,
            vp.w,
            vp.h,
            studio.cam_pos,
            studio.cam_fwd,
            studio.cam_right,
            studio.cam_up,
            studio.fov_y_deg,
        )
        ui.pop_clip()

    var mode_txt = "TRANSLATE [W]" if studio.gizmo_mode == GIZMO_MODE_TRANSLATE else ("ROTATE [E]" if studio.gizmo_mode == GIZMO_MODE_ROTATE else "SCALE [R]")
    var cam_txt = "FLY" if studio.camera_rig.mode == CAM_MODE_FREE_FLY else "ORBIT"
    studio.ui_ctx.draw_list.add_rect(
        vp.x + Float32(10.0),
        vp.y + Float32(8.0),
        Float32(292.0),
        Float32(22.0),
        pack_rgba8(14, 20, 30, 210),
        4.0,
        1.0,
        studio.ui_ctx.theme.border,
    )
    studio.ui_ctx._draw_text(
        "CAM:" + cam_txt + "  |  " + mode_txt,
        vp.x + Float32(18.0),
        vp.y + Float32(11.0),
        Float32(12.5),
        studio.ui_ctx.theme.text_primary,
    )

    # 2. Top Command & 1-Click PIE Toolbar (`DockSpace` -> `"Toolbar"`)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "Toolbar"):
        studio.ui_ctx.begin_box("studio_toolbar_row", FlexStyle.row(gap=6.0, pad=0.0))
        studio.ui_ctx.label("FLINT STUDIO v5", pack_rgba8(255, 213, 79, 255))
        if studio.ui_ctx.button("tb_move", "W Move", 78.0, 22.0):
            studio.gizmo_mode = GIZMO_MODE_TRANSLATE
        if studio.ui_ctx.button("tb_rot", "E Rotate", 84.0, 22.0):
            studio.gizmo_mode = GIZMO_MODE_ROTATE
        if studio.ui_ctx.button("tb_scl", "R Scale", 78.0, 22.0):
            studio.gizmo_mode = GIZMO_MODE_SCALE

        var is_play = studio.pie.mode == PIE_PLAYING
        var pie_lbl = (
            "PIE STOP [F5] T=" + String(studio.pie.ticks_simulated)
            if is_play
            else ("PIE RESUME [F5]" if studio.pie.mode == PIE_PAUSED else "1-CLICK PIE [F5]")
        )
        if studio.ui_ctx.button("tb_pie", pie_lbl, 154.0, 22.0):
            if studio.pie.mode == PIE_STOPPED:
                _ = studio.pie_play()
            else:
                try:
                    _ = studio.pie_stop()
                except:
                    studio.pie.mode = PIE_STOPPED

        if studio.ui_ctx.button("tb_undo", "Undo [Ctrl+Z]", 108.0, 22.0):
            _ = studio.undo()
        if studio.ui_ctx.button("tb_redo", "Redo [Ctrl+Y]", 108.0, 22.0):
            _ = studio.redo()
        _ = studio.ui_ctx.checkbox("tb_snap", "Snap", studio.snap_enabled)
        studio.ui_ctx.label(
            "U:" + String(len(studio.undo_stack)) + " R:" + String(len(studio.redo_stack)) + " | GPU " + _fmt_f32_2dp(Float32(gpu_ms)) + " ms",
            studio.ui_ctx.theme.text_muted,
        )
        studio.ui_ctx.end_box()
        studio.ui_ctx.end_dock_panel()

    # 3. Left Scene Hierarchy Outliner (`DockSpace` -> `"Hierarchy"`, `tree_node` + `begin_scroll_view`)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "Hierarchy"):
        var h_info = studio.dock.find_panel("Hierarchy")
        var h_h = max(Float32(80.0), h_info[3].h - Float32(14.0))
        studio.ui_ctx.begin_scroll_view("hierarchy_scroll", 0.0, h_h)
        if studio.ui_ctx.tree_node("world_root", "StudioWorld (" + String(studio.live_entity_count()) + ")", default_open=True):
            for i in range(len(studio.entities)):
                if not studio.entities[i].alive:
                    continue
                var prefix = "* " if i == studio.selected_entity_id else "  "
                var row_w = max(Float32(120.0), h_info[3].w - Float32(48.0))
                if studio.ui_ctx.button("ent_sel_" + String(i), prefix + studio.entities[i].name, row_w, 22.0):
                    studio.selected_entity_id = i
            studio.ui_ctx.tree_pop()
        studio.ui_ctx.end_scroll_view()
        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "Outliner"):
        studio.ui_ctx.label("Live Entities: " + String(studio.live_entity_count()))
        studio.ui_ctx.label("State Hash: " + String(studio.state_hash()), studio.ui_ctx.theme.text_muted)
        studio.ui_ctx.end_dock_panel()

    # 4. Right Inspector (`DockSpace` -> `"Inspector"`, `FieldVisitor` + `drag_float3` + `slider_float` + `checkbox`)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "Inspector"):
        var i_info = studio.dock.find_panel("Inspector")
        var panel_w = max(Float32(180.0), i_info[3].w - Float32(16.0))
        if (
            studio.selected_entity_id >= 0
            and studio.selected_entity_id < len(studio.entities)
            and studio.entities[studio.selected_entity_id].alive
        ):
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
            studio.ui_ctx.label("Entity #" + String(eid) + ": " + cur.name, studio.ui_ctx.theme.accent)
            studio.ui_ctx.label(
                "Shading: " + shd_lbl + "  |  Reflected Fields: " + String(len(vis.fields)),
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
                var tr_w = max(Float32(160.0), panel_w - Float32(20.0))
                if studio.ui_ctx.drag_float3("tr_pos", "Pos", cur.translation, speed=0.04, w=tr_w):
                    changed = True
                if studio.ui_ctx.drag_float3("tr_scl", "Scale", cur.scale, speed=0.02, min_val=0.05, max_val=50.0, w=tr_w):
                    changed = True
                studio.ui_ctx.tree_pop()

            if studio.ui_ctx.tree_node("insp_pbr", "PBR Material & Physics", default_open=True):
                var mat_w = max(Float32(160.0), panel_w - Float32(20.0))
                if studio.ui_ctx.drag_float3("mat_col", "Albedo", cur.base_color, speed=0.01, min_val=0.0, max_val=1.0, w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float("mat_rgh", "Rough", cur.roughness, Float32(0.04), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float("mat_met", "Metal", cur.metallic, Float32(0.0), Float32(1.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.slider_float("phy_mass", "Mass", cur.mass, Float32(0.0), Float32(100.0), w=mat_w):
                    changed = True
                if studio.ui_ctx.checkbox("ent_vis", "Visible", cur.visible):
                    changed = True
                studio.ui_ctx.tree_pop()

            var npr_open = cur.shading_mode != SHADING_PBR or cur.anim_step_hz > Float32(0.0)
            if studio.ui_ctx.tree_node("insp_npr", "NPR Cel / Stylized VFX & Timeline", default_open=npr_open):
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
                if not studio.ui_ctx.mouse_down_l:
                    studio.ui_edit_seq += 1
                _ = studio.commit_node_edit(eid, cur^, "Inspector UI Edit", coalesce_key=studio.ui_edit_seq)

            if studio.ui_ctx.button("btn_dup_orb", "+ Spawn SubD MaterialOrb", panel_w, 24.0):
                _ = studio.spawn_entity(
                    "SubD_MaterialOrb",
                    position=Vec3(0.0, 1.2, 0.0),
                    collider_kind=COLLIDER_SPHERE,
                    collider_half_extents=Vec3(0.85, 0.85, 0.85),
                    mesh_id=1,
                    record_undo=True,
                )
        else:
            studio.ui_ctx.label("No entity selected", studio.ui_ctx.theme.text_muted)
        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "CameraRig"):
        studio.ui_ctx.label("Viewport CameraRig")
        if studio.ui_ctx.slider_float("cam_fov", "FOV", studio.camera_rig.fov_deg, Float32(30.0), Float32(95.0), w=210.0):
            studio.fov_y_deg = studio.camera_rig.fov_deg
        if studio.ui_ctx.slider_float("cam_arm", "Arm m", studio.camera_rig.spring_arm.target_length, Float32(1.5), Float32(28.0), w=210.0):
            studio.update_camera(Float32(0.0))
        studio.ui_ctx.end_dock_panel()

    # 5. Bottom Asset Server, Procedural Presets & MCP v2 Dock (`DockSpace` -> `"AssetBrowser"`)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "AssetBrowser"):
        studio.ui_ctx.label(
            "ASSET SERVER (.flpk v3=" + String(mounted_packs) + ")  |  PROCEDURAL STUDIO KIT  |  MCP v2 CLIENTS=" + String(mcp_clients),
            studio.ui_ctx.theme.accent,
        )
        studio.ui_ctx.begin_box("preset_cards_row", FlexStyle.row(gap=8.0, pad=2.0))
        if studio.ui_ctx.button("card_orb", "SubD_MaterialOrb", 142.0, 32.0):
            _ = studio.spawn_entity("SubD_MaterialOrb", position=Vec3(0.0, 1.1, 0.0), collider_kind=COLLIDER_SPHERE, mesh_id=1)
        if studio.ui_ctx.button("card_torus", "SubD_CatmullTorus", 148.0, 32.0):
            _ = studio.spawn_entity("SubD_CatmullTorus", position=Vec3(1.8, 2.2, 0.5), collider_kind=COLLIDER_SPHERE, mass=6.0, mesh_id=2)
        if studio.ui_ctx.button("card_arch", "ArchKit_Colonnade", 148.0, 32.0):
            _ = studio.spawn_entity("ArchKit_Colonnade", position=Vec3(0.0, 0.0, 5.8), collider_kind=COLLIDER_BOX, mesh_id=3)
        if studio.ui_ctx.button("card_fol", "Botanical_Foliage", 142.0, 32.0):
            _ = studio.spawn_entity("Botanical_FoliageSample", position=Vec3(-4.2, 0.0, 2.2), collider_kind=COLLIDER_BOX, mesh_id=4)
        if studio.ui_ctx.button("card_grid", "Studio_RefGrid", 132.0, 32.0):
            _ = studio.spawn_entity("Studio_ReferenceGrid", position=Vec3(0.0, 0.0, 0.0), collider_kind=COLLIDER_BOX, mesh_id=0)
        studio.ui_ctx.end_box()
        studio.ui_ctx.end_dock_panel()

    if studio.ui_ctx.begin_dock_panel(studio.dock, "MCP v2"):
        studio.ui_ctx.label("MCP v2 Active Clients: " + String(mcp_clients))
        studio.ui_ctx.label("Sandbox: strict workspace_root + SO_PEERCRED + 0600 socket", studio.ui_ctx.theme.text_muted)
        studio.ui_ctx.end_dock_panel()

    studio.ui_ctx.end_frame()

    # Transfer `UiContext` GPU primitives to `ui`, converting `0xAABBGGRR` -> `0xAARRGGBB` & SDF gain!
    for i in range(len(studio.ui_ctx.draw_list.prims)):
        var p = studio.ui_ctx.draw_list.prims[i]
        p.fill_rgba8 = _abgr_to_argb(p.fill_rgba8)
        p.secondary_rgba8 = _abgr_to_argb(p.secondary_rgba8)
        if (p.header & 0xF) == PRIM_GLYPH and p.param0 < Float32(0.5):
            p.param0 = max(Float32(1.5), p.param0 * Float32(255.0))
        ui.prims.append(p)

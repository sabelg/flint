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
from std.math import abs, asin, cos, max, min, sin, sqrt
from flint.anim.flanim_ecs import SteppedAnimClock, compute_volume_preserving_smear_scale
from flint.anim.plugin import register_anim_codecs
from flint.app.engine_app import (
    AppConfig,
    EngineApp,
    FixedTime,
    STAGE_FIRST,
    STAGE_FIXED_UPDATE,
    STAGE_LAST,
    STAGE_POST_UPDATE,
    STAGE_PRE_UPDATE,
    STAGE_UPDATE,
)
from flint.app.window_plugin import WindowPlugin
from flint.assets.checkpoint import pack_checkpoint_lz4, unpack_checkpoint_lz4
from flint.assets.flpk_v3 import FlpkV3MappedArchive, UntypedHandle
from flint.assets.glb_writer import encode_native_glb, write_native_glb
from flint.assets.json import JsonDocument
from flint.assets.json_write import json_quote
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeMesh, NativeModel
from flint.camera.rig import CAM_MODE_FREE_FLY, CAM_MODE_ORBIT_FOLLOW, CameraRig
from flint.core.detmath import PI, det_atan2, set_sim_float_mode
from flint.core.math import Quat, Vec3
from flint.core.serial import BlobReader, BlobWriter, fnv64, fourcc
from flint.core.transform import Mat4, trs
from flint.ecs.world import (
    Entity,
    NULL_ENTITY,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.input.input_map import (
    ACTION_AXIS2D,
    ACTION_BOOL,
    InputContext,
    InputMap,
    MOD_CTRL,
)
from flint.input.plugin import InputPlugin
from flint.physics.bvh import Bvh8MeshCollider
from flint.physics.filter import CollisionFilter
from flint.physics.plugin import (
    ColliderComponent,
    PhysicsPlugin,
    PhysicsSyncStateResource,
    RigidBodyComponent,
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
from flint.platform.input import InputState
from flint.platform.os import list_directory, make_dirs, parent_directory, read_bytes, write_bytes
from flint.procedural.arch_kit import (
    build_colonnade_arcade_bay,
    build_street_bollard_and_grate,
)
from flint.procedural.foliage import (
    build_botanical_oak,
    build_botanical_pine,
)
from flint.procedural.subd import SubdCage, subdivide_to_mesh
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_npr_cel import AaaCelParams
from flint.render.aaa_npr_vfx import AaaImpactFrameParams, AaaShockwaveRing, AaaStylizedVfxParams
from flint.render.font import SdfFont
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import (
    CameraComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    RenderPlugin,
    RenderPluginStateResource,
    register_render_codecs,
)
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RenderCameraPacket,
    RenderWorld,
    ScalableRenderer,
)
from flint.scene.scene import ComponentRegistry, Scene
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
    RectF as _RectF,
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
    comptime SCHEMA_VERSION: Int = 1

    var name: String
    var locked: Bool
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
    ):
        self.name = name
        self.locked = locked
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

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var nm = reader.text()
        var lk = reader.boolean()
        var mh = reader.u64()
        var mat_h = reader.u64()
        var ck = reader.integer()
        var che = reader.vec3()
        var rest = reader.f32()
        var fric = reader.f32()
        var bvh = reader.integer()
        return Self(nm, lk, mh, mat_h, ck, che, rest, fric, bvh)

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
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var nm = doc.text(doc.object_get(node, "name"), "Entity")
        var lk = doc.boolean(doc.object_get(node, "locked"), False)
        var mh = UInt64(max(0, doc.integer(doc.object_get(node, "mesh_handle_packed"), 0)))
        var mat_h = UInt64(max(0, doc.integer(doc.object_get(node, "material_handle_packed"), 0)))
        var ck = doc.integer(doc.object_get(node, "collider_kind"), COLLIDER_BOX)
        var che = _read_vec3_json(doc, doc.object_get(node, "collider_half_extents"), Vec3(0.5, 0.5, 0.5))
        var rest = Float32(doc.number(doc.object_get(node, "restitution"), 0.32))
        var fric = Float32(doc.number(doc.object_get(node, "friction"), 0.55))
        var bvh = doc.integer(doc.object_get(node, "bvh_slot"), -1)
        return Self(nm, lk, mh, mat_h, ck, che, rest, fric, bvh)


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
        return s

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
        self.collider_kind = COLLIDER_BOX
        self.collider_half_extents = Vec3(0.5, 0.5, 0.5)
        self.mass = Float32(0.0)
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
        self.uploaded = False


# ============================================================================
# 4. `FlintStudioState`: EngineApp + World + Scene + PhysicsWorld + Multi-Select
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
    _ = imap.register_action("source_editor", ACTION_BOOL)
    _ = imap.register_action("editor_save", ACTION_BOOL)
    _ = imap.register_action("editor_load", ACTION_BOOL)
    _ = imap.register_action("editor_duplicate", ACTION_BOOL)
    _ = imap.register_action("editor_delete", ACTION_BOOL)
    _ = imap.register_action("pie_toggle", ACTION_BOOL)
    _ = imap.register_action("editor_quit", ACTION_BOOL)
    _ = imap.register_action("cam_orbit", ACTION_AXIS2D)

    var ctx = InputContext("StudioEditor", priority=10)
    ctx.bind_key("gizmo_translate", 87)                  # W
    ctx.bind_key("gizmo_rotate", 69)                     # E
    ctx.bind_key("gizmo_scale", 82)                      # R
    ctx.bind_key("editor_undo", 90, modifiers=MOD_CTRL)  # Ctrl+Z
    ctx.bind_key("editor_redo", 89, modifiers=MOD_CTRL)  # Ctrl+Y
    ctx.bind_key("source_editor", 297)                    # F8
    ctx.bind_key("editor_save", 83, modifiers=MOD_CTRL)  # Ctrl+S
    ctx.bind_key("editor_load", 79, modifiers=MOD_CTRL)  # Ctrl+O
    ctx.bind_key("editor_duplicate", 68, modifiers=MOD_CTRL)  # Ctrl+D
    ctx.bind_key("editor_delete", 261)                   # Delete
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
    var active_gizmo_handle: Int
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

    var stepped_clock: SteppedAnimClock
    var preview_impact_blend: Float32
    var preview_impact_invert: Bool
    var preview_shockwave_active: Bool
    var preview_shockwave_radius: Float32

    var dock: DockSpace
    var ui_ctx: UiContext
    var camera_rig: CameraRig
    var _cam_dummy_world: PhysicsWorld
    var input_map: InputMap
    var cam_target: Vec3
    var ui_edit_seq: UInt64
    var menu: scene_menu.SceneMenu
    var source_editor: EmbeddedIde
    var scene_file_path: String
    var asset_browser_dir: String
    var asset_browser_pack: String
    var asset_browser_entries: List[String]
    var asset_browser_scanned: Bool

    var cam_pos: Vec3
    var cam_fwd: Vec3
    var cam_right: Vec3
    var cam_up: Vec3
    var fov_y_deg: Float32
    var near_z: Float32

    def __init__(
        out self,
        headless: Bool = True,
        render_tier: UInt32 = RENDER_TIER_0_HEADLESS_SIMD,
        width: Int = 1280,
        height: Int = 720,
        worker_threads: Int = 0,
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

        self.registry = ComponentRegistry.with_builtins()
        register_physics_codecs(self.registry)
        register_render_codecs(self.registry)
        register_anim_codecs(self.registry)
        self.registry.register[StudioEntityMetaComponent]()
        self.registry.register[StudioStyleComponent]()
        self.registry.ensure_registered_in_world(self.app.world)
        self.app.startup()

        self.scene = Scene("FlintStudioScene")
        self.entities = List[StudioEntityView]()
        self.static_bvh_colliders = List[Bvh8MeshCollider]()
        self.gpu_materials = StudioGpuMaterials()

        self.selected_entity_id = -1
        self.selected_entity_ids = List[Int]()
        self.selection_anchor_id = -1
        self.hierarchy_expanded = Dict[Int, Bool]()

        self.gizmo_mode = GIZMO_MODE_TRANSLATE
        self.active_gizmo_handle = GIZMO_AXIS_NONE
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
        self._cam_dummy_world = PhysicsWorld()
        self.input_map = _create_studio_input_map()
        self.cam_target = Vec3(0.0, 0.9, 1.0)
        self.ui_edit_seq = UInt64(50000)
        self.source_editor = EmbeddedIde()
        self.menu = scene_menu.SceneMenu()
        self.scene_file_path = "build/studio_scene.flscn"
        self.asset_browser_dir = ""
        self.asset_browser_pack = ""
        self.asset_browser_entries = List[String]()
        self.asset_browser_scanned = False

        self.cam_pos = Vec3(-6.5, 4.2, -9.5)
        self.cam_fwd = Vec3(0.52, -0.26, 0.81).normalized()
        self.cam_right = Vec3(0.0, 1.0, 0.0).cross(self.cam_fwd).normalized()
        self.cam_up = self.cam_fwd.cross(self.cam_right).normalized()
        self.fov_y_deg = Float32(52.0)
        self.near_z = Float32(0.15)
        self.look_at(Vec3(-6.5, 4.2, -9.5), Vec3(0.0, 0.9, 1.0))
        var fix_init = self.app.resources.get_ptr_mut[FixedTime]()
        if fix_init:
            fix_init.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)

    def shutdown(mut self):
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

    def _sync_compat_view(mut self):
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
                v.mesh_id = mr.mesh_id
                v.material_id = mr.material_id
                v.visible = mr.visible
            else:
                v.mesh_id = 0
                v.material_id = 0
                v.visible = True

            var mo_opt = self.app.world.get_ptr[MaterialOverrideComponent](e)
            if mo_opt:
                ref mo = mo_opt.value()[unsafe_offset=0]
                if mo.material_id >= 0:
                    v.material_id = mo.material_id
                v.base_color = mo.base_color
                v.roughness = mo.roughness
                v.metallic = mo.metallic
                v.emissive = mo.emissive

            var meta_opt = self.app.world.get_ptr[StudioEntityMetaComponent](e)
            if meta_opt:
                ref meta = meta_opt.value()[unsafe_offset=0]
                v.name = meta.name
                v.locked = meta.locked
                v.mesh_handle = UntypedHandle(meta.mesh_handle_packed)
                v.material_handle = UntypedHandle(meta.material_handle_packed)
                v.collider_kind = meta.collider_kind
                v.collider_half_extents = meta.collider_half_extents
                v.restitution = meta.restitution
                v.friction = meta.friction
            else:
                v.name = "Entity_" + String(i)
                v.locked = False
                v.collider_kind = COLLIDER_NONE

            var col_opt = self.app.world.get_ptr[ColliderComponent](e)
            if col_opt:
                v.collider_half_extents = col_opt.value()[unsafe_offset=0].half_extents

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

    def _sync_ecs_and_physics(mut self):
        self._propagate_transforms_immediate()
        self._sync_physics_pre_only()
        self._sync_compat_view()

    def _sync_ecs_physics_and_render(mut self):
        self._sync_ecs_and_physics()

    def _capture_world_blob(self) -> List[UInt8]:
        var w = BlobWriter()
        try:
            self.app.world.snapshot_blob(w)
        except:
            return List[UInt8]()
        var out = List[UInt8]()
        swap(out, w.data)
        return out^

    def _restore_world_blob(mut self, data: List[UInt8]) -> Bool:
        if len(data) == 0:
            return False
        try:
            var r = BlobReader(data.copy())
            self.app.world.restore_blob(r)
            if r.failed:
                return False
        except:
            return False
        self.registry.ensure_registered_in_world(self.app.world)
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
        if orbit_delta_yaw_deg != Float32(0.0) or orbit_delta_pitch_deg != Float32(0.0):
            self.camera_rig.rotate_orbit(orbit_delta_yaw_deg, orbit_delta_pitch_deg)
        self.camera_rig.update(self._cam_dummy_world, self.cam_target, dt, fly_move_local=fly_move_local, fly_speed=fly_speed)
        self._sync_from_camera_rig()

    def viewport_rect(mut self, screen_w: Int, screen_h: Int) -> RectF:
        self.dock.solve_layout(RectF(0.0, 0.0, Float32(screen_w), Float32(screen_h)))
        var vp = self.dock.find_panel("SceneViewport")
        if vp[0] >= 0:
            return vp[3]
        return RectF(Float32(236.0), Float32(46.0), Float32(max(160, screen_w - 524)), Float32(max(120, screen_h - 184)))

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
    ):
        if entity_id < 0 or entity_id >= len(visited) or visited[entity_id] or depth >= 64:
            return
        visited[entity_id] = True
        var ch = self.children_of(entity_id)
        var has_ch = len(ch) > 0
        var exp = self.is_hierarchy_expanded(entity_id)
        out.append(HierarchyRow(entity_id, parent_id, depth, has_ch, exp))
        if has_ch and exp:
            for i in range(len(ch)):
                self._collect_hierarchy_dfs(ch[i], entity_id, depth + 1, out, visited)

    def hierarchy_visible_rows(self) -> List[HierarchyRow]:
        var out = List[HierarchyRow]()
        var n = len(self.app.world.entities)
        if n == 0:
            return out^
        var visited = List[Bool](length=n, fill=False)
        for i in range(n):
            if self.app.world.entities[i].alive and self.parent_of(i) < 0:
                self._collect_hierarchy_dfs(i, -1, 0, out, visited)
        # Fallback for any orphan whose parent died externally (never emit children of collapsed live parents)
        for i in range(n):
            if self.app.world.entities[i].alive and not visited[i] and self.parent_of(i) < 0:
                self._collect_hierarchy_dfs(i, -1, 0, out, visited)
        return out^

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
        var ents = self.selected_entities()
        if len(ents) == 0:
            return (False, Vec3())
        var sum_p = Vec3()
        var cnt = 0
        for i in range(len(ents)):
            var gt_opt = self.app.world.get_ptr[GlobalTransform](ents[i])
            if gt_opt:
                sum_p = sum_p + gt_opt.value()[unsafe_offset=0].translation
                cnt += 1
            else:
                var lt_opt = self.app.world.get_ptr[LocalTransform](ents[i])
                if lt_opt:
                    sum_p = sum_p + lt_opt.value()[unsafe_offset=0].translation
                    cnt += 1
        if cnt == 0:
            return (False, Vec3())
        return (True, sum_p * (Float32(1.0) / Float32(cnt)))

    # ---- Editor Input ----

    def handle_command_menu(mut self, inp: InputState, width: Int, height: Int) -> Bool:
        self.menu.poll()
        self.dock.solve_layout(RectF(0,0,Float32(width),Float32(height)))
        var panel = self.dock.find_panel("Toolbar")
        if panel[0]<0 or not panel[2]:
            self.menu.bar.group = -1
            return False
        ref bar = self.menu.bar
        bar.x = Int(panel[3].x)+4
        bar.y = Int(panel[3].y)
        bar.height = min(30,max(1,Int(panel[3].h)))
        bar.screen_w = width
        bar.screen_h = height
        bar.groups[1].items[0].enabled = Bool(self.undo_stack)
        bar.groups[1].items[1].enabled = Bool(self.redo_stack)
        bar.groups[2].items[3].title = "Disable grid snapping" if self.snap_enabled else "Enable grid snapping"
        bar.groups[3].items[0].title = "Resume simulation" if self.pie.mode==PIE_PAUSED else "Play scene"
        bar.groups[3].items[0].enabled = self.pie.mode!=PIE_PLAYING
        bar.groups[3].items[1].enabled = self.pie.mode==PIE_PLAYING
        bar.groups[3].items[2].enabled = self.pie.mode!=PIE_STOPPED
        bar.groups[4].items[0].enabled = self.menu.has_project and self.menu.library_pid<=0
        var action = bar.input(inp,Int(inp.last_mouse_x),Int(inp.last_mouse_y))
        if action==scene_menu.SAVE:
            self.menu.message = "Scene saved" if self.save_scene_file() else "Could not save scene"
        elif action==scene_menu.LOAD:
            self.menu.message = "Scene loaded" if self.load_scene_file() else "Could not load scene"
        elif action==scene_menu.UNDO:
            _ = self.undo()
        elif action==scene_menu.REDO:
            _ = self.redo()
        elif action==scene_menu.MOVE:
            self.gizmo_mode = GIZMO_MODE_TRANSLATE
        elif action==scene_menu.ROTATE:
            self.gizmo_mode = GIZMO_MODE_ROTATE
        elif action==scene_menu.SCALE:
            self.gizmo_mode = GIZMO_MODE_SCALE
        elif action==scene_menu.SNAP:
            self.snap_enabled = not self.snap_enabled
        elif action==scene_menu.PLAY:
            _ = self.pie_play()
        elif action==scene_menu.PAUSE:
            _ = self.pie_pause()
        elif action==scene_menu.STOP:
            try:
                _ = self.pie_stop()
            except err:
                self.menu.message = String(err)
        elif action==scene_menu.SOURCE:
            self.source_editor.requested = True
        elif action==scene_menu.MOTION_TAKES:
            self.menu.open_takes()
        elif action==scene_menu.KIMODO:
            self.menu.open_kimodo()
        elif action==scene_menu.LIBRARY:
            self.menu.open_library()
        return action>=0

    def process_editor_input(mut self, input: InputState, dt: Float32) -> Bool:
        """Evaluates `InputMap` editor hotkeys (`W/E/R`, `Ctrl+Z`, `Ctrl+Y`, `F5` PIE) & `CameraRig` orbit/fly.
        Returns `False` when `editor_quit` (`Esc`) is triggered."""
        self.input_map.update(input, dt)
        if self.input_map.just_pressed("editor_quit"):
            return False
        if self.input_map.just_pressed("source_editor"):
            self.source_editor.requested = True
        if self.input_map.just_pressed("editor_undo"):
            _ = self.undo()
        if self.input_map.just_pressed("editor_redo"):
            _ = self.redo()
        if self.input_map.just_pressed("editor_save"):
            _ = self.save_scene_file()
        if self.input_map.just_pressed("editor_load"):
            _ = self.load_scene_file()
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
        _ = input.enter(STUDIO_CHECKPOINT_FOURCC, 1)
        self.selected_entity_id = input.integer()
        var n_sel = Int(input.u32())
        self.selected_entity_ids.clear()
        var rem_sel = max(0, input.limit() - input.offset)
        if input.failed or n_sel < 0 or n_sel > rem_sel // 8:
            input.failed = True
            input.leave()
            raise Error("Failed to read Studio checkpoint blob")
        self.selected_entity_ids.reserve(n_sel)
        for _ in range(n_sel):
            self.selected_entity_ids.append(input.integer())
        self.selection_anchor_id = input.integer()
        self.sim_tick = input.integer()
        self.gravity = input.vec3()
        self.ground_plane_y = input.f32()
        self.has_ground_plane = input.boolean()
        self._ground_body_id = input.integer()
        self.stepped_clock.raw_time_sec = input.f32()
        self.stepped_clock.stepped_time_sec = input.f32()
        self.stepped_clock.step_hz = input.f32()
        self.stepped_clock.hitstop_remaining_sec = input.f32()
        self.stepped_clock.pose_changed_this_tick = input.boolean()
        self.preview_impact_blend = input.f32()
        self.preview_impact_invert = input.boolean()
        self.preview_shockwave_active = input.boolean()
        self.preview_shockwave_radius = input.f32()

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
        var prev_gt_hash = PrevGlobalTransform.COMPONENT_ID

        for slot in range(len(ents)):
            var e = ents[slot]
            var p_slot = -1
            var p_opt = self.app.world.get_ptr[Parent](e)
            if p_opt:
                var pe = p_opt.value()[unsafe_offset=0].entity
                if self.app.world.is_alive(pe) and Int(pe.index) < len(ent_to_slot):
                    p_slot = ent_to_slot[Int(pe.index)]
            w.integer(p_slot)

            for ci in range(len(sorted_codecs)):
                var cidx = sorted_codecs[ci]
                var chash = self.registry.entries[cidx].component_hash
                if chash == prev_gt_hash:
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
                else:
                    _ = self.registry.write_entity_bin(cidx, self.app.world, e, w)
        w.end()
        return fnv64(w.data)

    def save_scene_file(mut self, path: String = "") -> Bool:
        """Saves the Studio scene via `Scene.save_flscnb_file` (`.flscnb`/`.flscene`) or `Scene.save_flscn_file` (`.flscn`)."""
        var target = path if path.byte_length() > 0 else self.scene_file_path
        if target.byte_length() == 0:
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
            return False
        self.scene_file_path = target
        return True

    def load_scene_file(mut self, path: String = "") -> Bool:
        """Loads a `.flscn`, `.flscnb`, or `.flscene` file into `app.world` and syncs transforms/physics."""
        var target = path if path.byte_length() > 0 else self.scene_file_path
        if target.byte_length() == 0:
            return False
        var raw = read_bytes(target)
        if len(raw) == 0:
            return False
        try:
            if target.endswith(".flscn") or raw[0] == UInt8(123):  # '{'
                _ = self.scene.load_flscn_bytes(self.app.world, self.registry, raw, clear_existing=True)
            else:
                _ = self.scene.load_flscnb_bytes(self.app.world, self.registry, raw, clear_existing=True)
        except:
            return False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_selection()
        if len(self.app.world.entities) > 0 and self.app.world.entities[0].alive:
            self.select_entity(0)
        self.scene_file_path = target
        self._reset_physics_world_and_sync()
        return True

    def scan_asset_browser(mut self, dir_path: String = "", pack_path: String = "") -> Int:
        """Scans a workspace directory and/or `.flpk` v3 archive to populate the Asset Browser panel."""
        self.asset_browser_entries.clear()
        var target_dir = dir_path if dir_path.byte_length() > 0 else self.asset_browser_dir
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

        if target_dir.byte_length() > 0:
            var names = list_directory(target_dir)
            for name in names:
                if (
                    name.endswith(".glb")
                    or name.endswith(".png")
                    or name.endswith(".jpg")
                    or name.endswith(".jpeg")
                    or name.endswith(".hdr")
                    or name.endswith(".exr")
                    or name.endswith(".flscn")
                    or name.endswith(".flscnb")
                    or name.endswith(".flscene")
                    or name.endswith(".flpk")
                    or name.endswith(".wav")
                    or name.endswith(".ttf")
                    or name.endswith(".otf")
                ):
                    var dup = False
                    for e in self.asset_browser_entries:
                        if e == name:
                            dup = True
                            break
                    if not dup:
                        self.asset_browser_entries.append(name)

        self.asset_browser_scanned = True
        return len(self.asset_browser_entries)

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

    def _push_undo(mut self, var tx: EditorTransaction):
        self.undo_stack.append(tx^)
        if len(self.undo_stack) > MAX_UNDO_STACK:
            var excess = len(self.undo_stack) - MAX_UNDO_STACK
            var trimmed = List[EditorTransaction](capacity=MAX_UNDO_STACK)
            for i in range(excess, len(self.undo_stack)):
                trimmed.append(self.undo_stack[i].copy())
            self.undo_stack = trimmed^

        var total_bytes = 0
        for i in range(len(self.undo_stack)):
            total_bytes += self.undo_stack[i].approx_bytes()
        if total_bytes > MAX_UNDO_BLOB_BYTES and len(self.undo_stack) > 32:
            var drop_n = 0
            while total_bytes > MAX_UNDO_BLOB_BYTES and len(self.undo_stack) - drop_n > 32:
                total_bytes -= self.undo_stack[drop_n].approx_bytes()
                drop_n += 1
            if drop_n > 0:
                var trimmed_b = List[EditorTransaction](capacity=len(self.undo_stack) - drop_n)
                for i in range(drop_n, len(self.undo_stack)):
                    trimmed_b.append(self.undo_stack[i].copy())
                self.undo_stack = trimmed_b^

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
        var after_blob = self._capture_world_blob()
        if coalesce_key != 0 and len(self.undo_stack) > 0 and len(self.redo_stack) == 0:
            ref top = self.undo_stack[len(self.undo_stack) - 1]
            if top.op == op and top.entity_id == primary_id and top.coalesce_key == coalesce_key:
                top.after_world_blob = after_blob^
                top.next_selected_id = self.selected_entity_id
                top.next_selected_ids = self.selected_entity_ids.copy()
                top.next_anchor_id = self.selection_anchor_id
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
        self._push_undo(tx^)
        self.redo_stack.clear()

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
        var q = rotation.normalized()
        var lt = LocalTransform(position, q, scale)
        var gt = GlobalTransform.from_local(lt)
        _ = self.app.world.insert[LocalTransform](e, lt)
        _ = self.app.world.insert[GlobalTransform](e, gt.copy())
        _ = self.app.world.insert[PrevGlobalTransform](e, PrevGlobalTransform(gt^))
        _ = self.app.world.insert[MeshRendererComponent](e, MeshRendererComponent(mesh_id, -1, True, True))
        var clamp_r = max(Float32(0.04), min(Float32(1.0), roughness))
        var clamp_m = max(Float32(0.0), min(Float32(1.0), metallic))
        _ = self.app.world.insert[MaterialOverrideComponent](
            e, MaterialOverrideComponent(-1, base_color, clamp_r, clamp_m, Vec3(), True)
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

    def _duplicate_entity_raw(mut self, src_id: Int) -> Int:
        var src_e = self.resolve_entity(src_id)
        if not src_e.valid():
            return -1
        var lt = self.app.world.get[LocalTransform](src_e) if self.app.world.has[LocalTransform](src_e) else LocalTransform()
        lt.translation = lt.translation + Vec3(Float32(0.5), Float32(0.0), Float32(0.0))
        var mr = self.app.world.get[MeshRendererComponent](src_e) if self.app.world.has[MeshRendererComponent](src_e) else MeshRendererComponent()
        var mo = self.app.world.get[MaterialOverrideComponent](src_e) if self.app.world.has[MaterialOverrideComponent](src_e) else MaterialOverrideComponent()
        var meta = StudioEntityMetaComponent()
        var meta_opt = self.app.world.get_ptr[StudioEntityMetaComponent](src_e)
        if meta_opt:
            ref m = meta_opt.value()[unsafe_offset=0]
            meta = StudioEntityMetaComponent(
                m.name + "_Copy",
                m.locked,
                m.mesh_handle_packed,
                m.material_handle_packed,
                m.collider_kind,
                m.collider_half_extents,
                m.restitution,
                m.friction,
                -1,
            )
        var style = self.app.world.get[StudioStyleComponent](src_e) if self.app.world.has[StudioStyleComponent](src_e) else StudioStyleComponent()
        var par_id = self.parent_of(src_id)

        var new_e = self.app.world.spawn_empty()
        var new_id = Int(new_e.index)
        var gt = GlobalTransform.from_local(lt)
        _ = self.app.world.insert[LocalTransform](new_e, lt)
        _ = self.app.world.insert[GlobalTransform](new_e, gt.copy())
        _ = self.app.world.insert[PrevGlobalTransform](new_e, PrevGlobalTransform(gt^))
        _ = self.app.world.insert[MeshRendererComponent](new_e, mr)
        _ = self.app.world.insert[MaterialOverrideComponent](new_e, mo)
        _ = self.app.world.insert[StudioEntityMetaComponent](new_e, meta^)
        _ = self.app.world.insert[StudioStyleComponent](new_e, style)

        var col_opt = self.app.world.get_ptr[ColliderComponent](src_e)
        if col_opt:
            ref sc = col_opt.value()[unsafe_offset=0]
            var nc = ColliderComponent(
                sc.shape, sc.radius, sc.half_height, sc.half_extents, sc.local_offset, sc.local_rot, sc.is_trigger, sc.filter, -1
            )
            nc.points = sc.points.copy()
            _ = self.app.world.insert[ColliderComponent](new_e, nc^)

        var rb_opt = self.app.world.get_ptr[RigidBodyComponent](src_e)
        if rb_opt:
            var nrb = rb_opt.value()[unsafe_offset=0]
            nrb.body_id = -1
            _ = self.app.world.insert[RigidBodyComponent](new_e, nrb)

        if par_id >= 0:
            var pe = self.resolve_entity(par_id)
            if pe.valid():
                _ = set_parent(self.app.world, new_e, pe)

        return new_id

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
        var srcs = List[Int]()
        for i in range(len(self.selected_entity_ids)):
            if self.is_alive_id(self.selected_entity_ids[i]):
                srcs.append(self.selected_entity_ids[i])
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
        # Detach live children and parent BEFORE World.despawn(e) so no dangling Children/Parent links remain!
        var ch = self.children_of(entity_id)
        for ci in range(len(ch)):
            var ce = self.resolve_entity(ch[ci])
            if ce.valid():
                _ = clear_parent(self.app.world, ce)
        _ = clear_parent(self.app.world, e)
        _ = self.app.world.despawn(e)
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

        _ = self.app.world.insert[LocalTransform](
            e, LocalTransform(updated.translation, updated.rotation.normalized(), updated.scale)
        )
        _ = self.app.world.insert[MeshRendererComponent](
            e, MeshRendererComponent(updated.mesh_id, updated.material_id, True, updated.visible)
        )
        var clamp_r = max(Float32(0.04), min(Float32(1.0), updated.roughness))
        var clamp_m = max(Float32(0.0), min(Float32(1.0), updated.metallic))
        _ = self.app.world.insert[MaterialOverrideComponent](
            e,
            MaterialOverrideComponent(
                updated.material_id,
                updated.base_color,
                clamp_r,
                clamp_m,
                updated.emissive,
                enabled=(updated.shading_mode == SHADING_PBR),
            ),
        )

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
            ),
        )

        var st = StudioStyleComponent()
        var old_st = self.app.world.get_ptr[StudioStyleComponent](e)
        if old_st:
            st = old_st.value()[unsafe_offset=0]
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

    def capture_selected_transforms(self) -> List[SelectedTransformSnap]:
        var top_ids = self._top_level_selected_ids()
        var snaps = List[SelectedTransformSnap](capacity=len(top_ids))
        for i in range(len(top_ids)):
            var eid = top_ids[i]
            var e = self.resolve_entity(eid)
            if not e.valid():
                continue
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
    ) -> Bool:
        if len(snaps) == 0:
            return False

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
                target_w_pos = sn.world_pos + world_delta_pos
            elif mode == GIZMO_MODE_ROTATE:
                var rel = sn.world_pos - pivot
                target_w_pos = pivot + world_dq_rot.rotate(rel)
                target_w_rot = world_dq_rot.multiply(sn.world_rot).normalized()
            elif mode == GIZMO_MODE_SCALE:
                target_local_scl = Vec3(
                    max(Float32(0.01), sn.local_scl.x * scale_factor.x),
                    max(Float32(0.01), sn.local_scl.y * scale_factor.y),
                    max(Float32(0.01), sn.local_scl.z * scale_factor.z),
                )

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
        v.cel_shadow_tint = cel_shadow_tint
        v.cel_outline_width_px = max(Float32(0.0), min(Float32(24.0), cel_outline_width_px))
        v.cel_outline_color = cel_outline_color
        v.cel_rim_color = cel_rim_color
        v.cel_hair_halo = max(Float32(0.0), cel_hair_halo)
        v.anim_step_hz = max(Float32(0.0), anim_step_hz)
        v.smear_stretch = max(Float32(0.55), min(Float32(3.2), smear_stretch))
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
        v.vfx_erosion = max(Float32(0.0), min(Float32(1.0), vfx_erosion))
        v.vfx_scroll_speed = vfx_scroll_speed
        v.vfx_core_color = vfx_core_color
        v.vfx_edge_color = vfx_edge_color
        v.anim_step_hz = max(Float32(0.0), anim_step_hz)
        v.smear_stretch = max(Float32(0.55), min(Float32(3.2), smear_stretch))
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
        if len(self.undo_stack) == 0:
            return False
        var tx = self.undo_stack.pop()
        if not self._restore_world_blob(tx.before_world_blob):
            self.undo_stack.append(tx^)
            return False
        self._restore_selection(tx.prev_selected_ids, tx.prev_selected_id, tx.prev_anchor_id)
        self.redo_stack.append(tx^)
        return True

    def redo(mut self) -> Bool:
        if len(self.redo_stack) == 0:
            return False
        var tx = self.redo_stack.pop()
        if not self._restore_world_blob(tx.after_world_blob):
            self.redo_stack.append(tx^)
            return False
        self._restore_selection(tx.next_selected_ids, tx.next_selected_id, tx.next_anchor_id)
        self._push_undo(tx^)
        return True

    # ---- Deterministic `PhysicsWorld` Step, Editor Frame Render & Analytical/Physics Raycast ----

    def step_simulation(mut self, ticks: Int = 1, dt: Float32 = Float32(1.0 / 60.0)) -> Int:
        """Advances `PhysicsWorld` via `PhysicsPlugin` (`STAGE_PRE_UPDATE` + `STAGE_FIXED_UPDATE` x `ticks`)
        and propagates `GlobalTransform` (`STAGE_POST_UPDATE`)."""
        if ticks <= 0:
            return 0
        set_sim_float_mode()
        self._ensure_ground_plane()
        var fix_opt = self.app.resources.get_ptr_mut[FixedTime]()
        var saved_dt = Float32(1.0 / 60.0)
        if fix_opt:
            saved_dt = fix_opt.value()[unsafe_offset=0].dt
            fix_opt.value()[unsafe_offset=0].dt = dt
            fix_opt.value()[unsafe_offset=0].dt_f64 = Float64(dt)

        self.app.schedule.run_stage(
            STAGE_PRE_UPDATE,
            self.app.world,
            self.app.resources,
            self.app.pool,
            self.app.commands,
        )
        self.app.world.flush_commands(self.app.commands)

        for _ in range(ticks):
            _ = self.app.world.advance_tick()
            self.app.schedule.run_stage(
                STAGE_FIXED_UPDATE,
                self.app.world,
                self.app.resources,
                self.app.pool,
                self.app.commands,
            )
            self.app.world.flush_commands(self.app.commands)
            self.sim_tick += 1

        if fix_opt:
            fix_opt.value()[unsafe_offset=0].dt = saved_dt
            fix_opt.value()[unsafe_offset=0].dt_f64 = Float64(saved_dt)
            fix_opt.value()[unsafe_offset=0].overstep_alpha = Float32(1.0)

        self.app.schedule.run_stage(
            STAGE_POST_UPDATE,
            self.app.world,
            self.app.resources,
            self.app.pool,
            self.app.commands,
        )
        self.app.world.flush_commands(self.app.commands)
        self._sync_compat_view()
        return ticks

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

        # 1. Sync camera when no active user CameraComponent entity exists
        var has_cam_ent = False
        var cam_mask = self.app.world.bit_for[CameraComponent]()
        var cam_ents = self.app.world.query_entities(cam_mask)
        for i in range(len(cam_ents)):
            var c_opt = self.app.world.get_ptr[CameraComponent](cam_ents[i])
            if c_opt and c_opt.value()[unsafe_offset=0].active:
                has_cam_ent = True
                break
        if not has_cam_ent:
            var rw_opt = self.app.resources.get_ptr_mut[RenderWorld]()
            if rw_opt:
                ref rw = rw_opt.value()[unsafe_offset=0]
                rw.camera = RenderCameraPacket(
                    self.cam_pos,
                    self.cam_pos + self.cam_fwd * Float32(10.0),
                    Vec3(0.0, 1.0, 0.0),
                    self.fov_y_deg,
                    self.near_z,
                    Float32(1000.0),
                    Float32(1.08),
                )

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

            if renderer.uses_vulkan():
                ref gpu = renderer.gpu_slot[0]
                font_slot = gpu.ui_font_atlas_slot
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
                    var mr_opt = self.app.world.get_ptr_mut[MeshRendererComponent](e)
                    var mo_opt = self.app.world.get_ptr_mut[MaterialOverrideComponent](e)
                    var st_opt = self.app.world.get_ptr[StudioStyleComponent](e)
                    if not mr_opt or not mo_opt or not st_opt:
                        continue
                    ref mr = mr_opt.value()[unsafe_offset=0]
                    ref mo = mo_opt.value()[unsafe_offset=0]
                    ref st = st_opt.value()[unsafe_offset=0]

                    var mat = AaaMaterial(base_color=mo.base_color, roughness=mo.roughness, metallic=mo.metallic)
                    mat.emissive = mo.emissive
                    mat.clearcoat = st.clearcoat
                    mat.clearcoat_roughness = st.clearcoat_roughness

                    if st.shading_mode == SHADING_NPR_CEL:
                        var cel = AaaCelParams(
                            sss_shadow_tint=st.cel_shadow_tint,
                            rim_color=st.cel_rim_color,
                            hair_halo_strength=st.cel_hair_halo,
                            outline_width_px=st.cel_outline_width_px,
                            outline_color=st.cel_outline_color,
                        )
                        if self.gpu_materials.entity_cel_mat_ids[i] < 0:
                            self.gpu_materials.entity_cel_mat_ids[i] = gpu.add_cel_material(mat, cel)
                        else:
                            gpu.update_material(self.gpu_materials.entity_cel_mat_ids[i], mat)
                            _ = gpu.update_cel_material(self.gpu_materials.entity_cel_mat_ids[i], cel)
                        mo.enabled = False
                        mr.material_id = max(0, self.gpu_materials.entity_cel_mat_ids[i])
                    elif st.shading_mode == SHADING_STYLIZED_VFX:
                        var vfx = AaaStylizedVfxParams.anime_flame(
                            core_color=st.vfx_core_color,
                            edge_color=st.vfx_edge_color,
                            erosion=st.vfx_erosion,
                            scroll_speed=st.vfx_scroll_speed,
                        )
                        if self.gpu_materials.entity_vfx_mat_ids[i] < 0:
                            self.gpu_materials.entity_vfx_mat_ids[i] = gpu.add_vfx_material(mat, vfx)
                        else:
                            var vfx_mat = mat.copy()
                            vfx_mat.alpha_cutoff = Float32(0.48)
                            gpu.update_material(self.gpu_materials.entity_vfx_mat_ids[i], vfx_mat)
                            _ = gpu.update_vfx_material(self.gpu_materials.entity_vfx_mat_ids[i], vfx)
                        mo.enabled = False
                        mr.material_id = max(0, self.gpu_materials.entity_vfx_mat_ids[i])
                    else:
                        if self.gpu_materials.entity_pbr_mat_ids[i] < 0:
                            var ma = MaterialAsset(
                                base_color=mo.base_color,
                                roughness=mo.roughness,
                                metallic=mo.metallic,
                            )
                            ma.emissive = mo.emissive
                            ma.clearcoat = st.clearcoat
                            ma.clearcoat_roughness = st.clearcoat_roughness
                            self.gpu_materials.entity_pbr_mat_ids[i] = renderer.register_material(ma)
                        var pbr_id = self.gpu_materials.entity_pbr_mat_ids[i]
                        if pbr_id >= 0:
                            mr.material_id = pbr_id
                            mo.material_id = pbr_id
                        mo.enabled = True
                        if pbr_id >= 0 and st.clearcoat > Float32(0.0):
                            gpu.update_material(pbr_id, mat)

                    # Apply smear stretch to GlobalTransform for render extraction if active
                    if st.smear_stretch != Float32(1.0):
                        var gt_opt = self.app.world.get_ptr_mut[GlobalTransform](e)
                        if gt_opt:
                            var sm = compute_volume_preserving_smear_scale(st.smear_stretch)
                            gt_opt.value()[unsafe_offset=0].matrix = gt_opt.value()[unsafe_offset=0].matrix.multiply(
                                trs(Vec3(0.0, 0.0, 0.0), Quat(0.0, 0.0, 0.0, 1.0), sm)
                            )

        # 3. Emit Studio DockSpace + Hierarchy + Inspector + Gizmos into RenderPluginStateResource.ui_draw_list
        var ui_copy = UiDrawList()
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
        var rps_opt = self.app.resources.get_ptr_mut[RenderPluginStateResource]()
        if rps_opt:
            rps_opt.value()[unsafe_offset=0].ui_draw_list = ui_copy^

        var rw_opt2 = self.app.resources.get_ptr_mut[RenderWorld]()
        if rw_opt2:
            rw_opt2.value()[unsafe_offset=0].cluster_lights.append(
                AaaClusterLight.point(Vec3(-2.8, 3.8, -2.5), Vec3(18.0, 14.0, 10.0), Float32(14.0))
            )
            rw_opt2.value()[unsafe_offset=0].cluster_lights.append(
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
        """Runs Edit/PIE update stages (`STAGE_PRE_UPDATE`..`STAGE_LAST` without fixed step in Edit mode)
        so `RenderPlugin` extracts ECS entities + Studio UI and renders via `ScalableRenderer`."""
        self._ensure_ground_plane()
        self.app.schedule.run_stage(STAGE_PRE_UPDATE, self.app.world, self.app.resources, self.app.pool, self.app.commands)
        self.app.world.flush_commands(self.app.commands)
        self.app.schedule.run_stage(STAGE_UPDATE, self.app.world, self.app.resources, self.app.pool, self.app.commands)
        self.app.world.flush_commands(self.app.commands)
        self.app.schedule.run_stage(STAGE_POST_UPDATE, self.app.world, self.app.resources, self.app.pool, self.app.commands)
        self.app.world.flush_commands(self.app.commands)

        self._sync_studio_pre_render(screen_w, screen_h, gpu_ms, mounted_packs, mcp_clients, input, dt)

        self.app.schedule.run_stage(STAGE_LAST, self.app.world, self.app.resources, self.app.pool, self.app.commands)
        self.app.world.flush_commands(self.app.commands)

        var res_ptr = Pointer[EngineApp, MutAnyOrigin](unsafe_from_address=Int(Pointer(to=self.app)))
        var r_ptr = Pointer(to=res_ptr[unsafe_offset=0].resources)
        for i in range(len(self.app.event_swap_hooks)):
            var hook = self.app.event_swap_hooks[i]
            hook(r_ptr)
        self._sync_compat_view()

    # ---- 1-Click Play-In-Editor (`PieSession` + `Scene` + `PhysicsWorld` + LZ4) ----

    def pie_play(mut self) -> Bool:
        if self.pie.mode == PIE_PLAYING:
            return True
        if self.pie.mode == PIE_PAUSED:
            self.pie.mode = PIE_PLAYING
            return True
        self.pie.pre_play_hash = self.state_hash()
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
        if self.selected_entity_id >= 0 and self.is_alive_id(self.selected_entity_id):
            var st_opt = self.app.world.get_ptr[StudioStyleComponent](self.resolve_entity(self.selected_entity_id))
            if st_opt and st_opt.value()[unsafe_offset=0].anim_step_hz > Float32(0.0):
                self.stepped_clock.step_hz = st_opt.value()[unsafe_offset=0].anim_step_hz
        var done = self.step_simulation(ticks, dt)
        for _ in range(done):
            _ = self.stepped_clock.advance(dt)
        self.pie.ticks_simulated += done
        return done

    def pie_stop(mut self) raises -> Bool:
        """Restores exact pre-PIE `World` + `PhysicsWorld` from LZ4 checkpoint and verifies `fnv64` equality."""
        if self.pie.mode == PIE_STOPPED or len(self.pie.checkpoint_lz4) == 0:
            self.pie.mode = PIE_STOPPED
            return False
        var reader = unpack_checkpoint_lz4(self.pie.checkpoint_lz4)
        self.deserialize(reader)
        self.pie.mode = PIE_STOPPED
        self.scene.has_pie_snapshot = False
        self.scene.pie_snapshot.clear()
        var restored_hash = self.state_hash()
        self.pie.last_restore_verified = restored_hash == self.pie.pre_play_hash
        if not self.pie.last_restore_verified:
            raise Error("PIE LZ4 world restore hash mismatch")
        return True

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
    var neg = v < Float32(0.0)
    var av = -v if neg else v
    var scaled = Int(av * Float32(100.0) + Float32(0.5))
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
            ui.box_stack[top_i].cursor_y = cy + row_h + ui.box_stack[top_i].gap
            ui.culled_items_this_frame += 1
            return (False, -1)

    var r = ui._alloc_item_rect(wid, row_w, row_h)
    var indent = Float32(row.depth * 14)
    var is_pri = studio.selected_entity_id == row.entity_id
    var is_sel = studio.is_selected(row.entity_id)

    var bg_col = UInt32(0)
    if is_pri:
        bg_col = ui.theme.accent
    elif is_sel:
        bg_col = pack_rgba8(48, 86, 138, 220)
    elif (not ui._blocked_by_modal()) and r.contains(ui.mouse_x, ui.mouse_y):
        bg_col = ui.theme.bg_widget_hover

    if bg_col != 0:
        ui.draw_list.add_rect(r.x, r.y, r.w, r.h, bg_col, 3.0)

    var disc_x = r.x + indent + Float32(4.0)
    var disc_r = _RectF(disc_x, r.y, Float32(16.0), r.h)
    var hovered = (not ui._blocked_by_modal()) and r.contains(ui.mouse_x, ui.mouse_y)

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

    var sw = Float32(max(320, screen_w))
    var sh = Float32(max(240, screen_h))

    if (not studio.ui_ctx.font.valid) and font.valid:
        studio.ui_ctx.font = font.copy()
    studio.ui_ctx.font_slot = font_slot
    studio.ui_ctx.theme.font_size = Float32(13.0)
    studio.ui_ctx.theme.row_height = Float32(22.0)

    studio.ui_ctx.begin_frame(input, sw, sh, dt)
    studio.ui_ctx.update_dock_space(studio.dock, RectF(0.0, 0.0, sw, sh))

    var vp_info = studio.dock.find_panel("SceneViewport")
    var vp = vp_info[3] if vp_info[0] >= 0 else RectF(Float32(240.0), Float32(54.0), sw - Float32(520.0), sh - Float32(190.0))

    # 1. 3D Viewport Active Selection Gizmo (centroid across multi-selected entities)
    var giz_piv = studio.selection_gizmo_pivot()
    if giz_piv[0]:
        ui.push_clip(Int(vp.x), Int(vp.y), Int(vp.right()), Int(vp.bottom()))
        emit_gizmo_draw_list(
            ui,
            giz_piv[1],
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
    var sel_txt = "SEL:" + String(studio.selected_count())
    studio.ui_ctx.draw_list.add_rect(
        vp.x + Float32(10.0),
        vp.y + Float32(8.0),
        Float32(348.0),
        Float32(22.0),
        pack_rgba8(14, 20, 30, 210),
        4.0,
        1.0,
        studio.ui_ctx.theme.border,
    )
    studio.ui_ctx._draw_text(
        "CAM:" + cam_txt + "  |  " + mode_txt + "  |  " + sel_txt,
        vp.x + Float32(18.0),
        vp.y + Float32(11.0),
        Float32(12.5),
        studio.ui_ctx.theme.text_primary,
    )

    # 3. Left Scene Hierarchy Outliner (`DockSpace` -> `"Hierarchy"`, collapsible tree + multi-select + DnD reparent)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "Hierarchy"):
        var h_info = studio.dock.find_panel("Hierarchy")
        var panel_w = max(Float32(140.0), h_info[3].w - Float32(16.0))
        studio.ui_ctx.begin_box("hier_actions", FlexStyle.row(gap=4.0, pad=0.0))
        var half_bw = max(Float32(60.0), (panel_w - Float32(8.0)) * Float32(0.5))
        if studio.ui_ctx.button("hier_unparent", "Unparent", half_bw, 20.0):
            _ = studio.reparent_selected_to(-1, preserve_world_transform=True, record_undo=True)
        if studio.ui_ctx.button("hier_parent_pri", "Parent->Pri", half_bw, 20.0):
            if studio.selected_count() >= 2 and studio.selected_entity_id >= 0:
                _ = studio.reparent_selected_to(studio.selected_entity_id, preserve_world_transform=True, record_undo=True)
        studio.ui_ctx.end_box()

        var h_h = max(Float32(80.0), h_info[3].h - Float32(42.0))
        studio.ui_ctx.begin_scroll_view("hierarchy_scroll", 0.0, h_h)
        var root_open = studio.ui_ctx.tree_node("world_root", "StudioWorld (" + String(studio.live_entity_count()) + ")", default_open=True)
        var root_drop = studio.ui_ctx.accept_drag_target("STUDIO_ENTITY")
        var pending_reparent = False
        var reparent_child = -1
        var reparent_target = -1
        if root_drop[0]:
            pending_reparent = True
            reparent_child = root_drop[1]
            reparent_target = -1

        if root_open:
            var rows = studio.hierarchy_visible_rows()
            var disp_order = List[Int](capacity=len(rows))
            for ri in range(len(rows)):
                disp_order.append(rows[ri].entity_id)
            var row_w = max(Float32(120.0), h_info[3].w - Float32(48.0))
            for ri in range(len(rows)):
                var row = rows[ri]
                var ename = studio.entities[row.entity_id].name if row.entity_id < len(studio.entities) else ("Entity_" + String(row.entity_id))
                var res = _studio_hierarchy_row(studio, row, ename, row_w, disp_order)
                if res[0]:
                    pending_reparent = True
                    reparent_child = res[1]
                    reparent_target = row.entity_id
            studio.ui_ctx.tree_pop()

        if pending_reparent and reparent_child >= 0:
            if studio.is_selected(reparent_child) and studio.selected_count() > 1:
                _ = studio.reparent_selected_to(reparent_target, preserve_world_transform=True, record_undo=True)
            else:
                _ = studio.reparent_entity(reparent_child, reparent_target, preserve_world_transform=True, record_undo=True)

        studio.ui_ctx.end_scroll_view()
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
                if not studio.ui_ctx.mouse_down_l:
                    studio.ui_edit_seq += 1
                _ = studio.bulk_translate_selected(bulk_delta, coalesce_key=studio.ui_edit_seq)

            var b_col = pri_view.base_color
            if studio.ui_ctx.drag_float3("bulk_col", "Bulk Albedo", b_col, speed=0.01, min_val=0.0, max_val=1.0, w=mat_w):
                if not studio.ui_ctx.mouse_down_l:
                    studio.ui_edit_seq += 1
                var f_col = InspectedField("base_color", FIELD_COLOR_RGB)
                f_col.v3_val = b_col
                _ = studio.bulk_patch_field("base_color", f_col, coalesce_key=studio.ui_edit_seq)

            var b_rgh = pri_view.roughness
            if studio.ui_ctx.slider_float("bulk_rgh", "Bulk Rough", b_rgh, Float32(0.04), Float32(1.0), w=mat_w):
                if not studio.ui_ctx.mouse_down_l:
                    studio.ui_edit_seq += 1
                var f_rgh = InspectedField("roughness", FIELD_F32)
                f_rgh.f_val = b_rgh
                _ = studio.bulk_patch_field("roughness", f_rgh, coalesce_key=studio.ui_edit_seq)

            var b_met = pri_view.metallic
            if studio.ui_ctx.slider_float("bulk_met", "Bulk Metal", b_met, Float32(0.0), Float32(1.0), w=mat_w):
                if not studio.ui_ctx.mouse_down_l:
                    studio.ui_edit_seq += 1
                var f_met = InspectedField("metallic", FIELD_F32)
                f_met.f_val = b_met
                _ = studio.bulk_patch_field("metallic", f_met, coalesce_key=studio.ui_edit_seq)

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
            studio.ui_ctx.label("Entity #" + String(eid) + ": " + cur.name, studio.ui_ctx.theme.accent)
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
                var rot_deg = _quat_to_euler_deg(cur.rotation)
                if studio.ui_ctx.drag_float3("tr_rot", "Rot", rot_deg, speed=0.5, min_val=-360.0, max_val=360.0, w=tr_w):
                    cur.rotation = _euler_deg_to_quat(rot_deg)
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

            studio.ui_ctx.begin_box("ent_actions_row", FlexStyle.row(gap=6.0, pad=0.0))
            var act_w = max(Float32(80.0), (panel_w - Float32(12.0)) * Float32(0.5))
            if studio.ui_ctx.button("btn_dup_ent", "Duplicate [Ctrl+D]", act_w, 24.0):
                _ = studio.duplicate_entity(eid, record_undo=True)
            if studio.ui_ctx.button("btn_del_ent", "Delete [Del]", act_w, 24.0):
                _ = studio.despawn_entity(eid, record_undo=True)
            studio.ui_ctx.end_box()
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

    # 5. Bottom Asset Server, Scanned Workspace/Pack Browser & Procedural Presets (`DockSpace` -> `"AssetBrowser"`)
    if studio.ui_ctx.begin_dock_panel(studio.dock, "AssetBrowser"):
        if (not studio.asset_browser_scanned) and (
            studio.asset_browser_dir.byte_length() > 0 or studio.asset_browser_pack.byte_length() > 0
        ):
            _ = studio.scan_asset_browser()
        studio.ui_ctx.label(
            "ASSET BROWSER (.flpk v3="
            + String(mounted_packs)
            + ", items="
            + String(len(studio.asset_browser_entries))
            + ")  |  MCP v2 CLIENTS="
            + String(mcp_clients),
            studio.ui_ctx.theme.accent,
        )
        studio.ui_ctx.begin_box("preset_cards_row", FlexStyle.row(gap=8.0, pad=2.0))
        if len(studio.asset_browser_entries) > 0:
            if studio.ui_ctx.button("ab_rescan", "Rescan", 72.0, 32.0):
                _ = studio.scan_asset_browser()
            var show_n = min(10, len(studio.asset_browser_entries))
            for ai in range(show_n):
                var entry_name = studio.asset_browser_entries[ai]
                if studio.ui_ctx.button("ab_item_" + String(ai), entry_name, 148.0, 32.0):
                    if entry_name.endswith(".flscn") or entry_name.endswith(".flscnb") or entry_name.endswith(".flscene"):
                        var fp = (
                            (studio.asset_browser_dir + "/" + entry_name)
                            if studio.asset_browser_dir.byte_length() > 0
                            else entry_name
                        )
                        _ = studio.load_scene_file(fp)
                    else:
                        _ = studio.spawn_entity(
                            entry_name,
                            position=Vec3(Float32(ai) * Float32(0.5), Float32(1.0), Float32(0.0)),
                            collider_kind=COLLIDER_BOX,
                            mesh_id=1,
                            record_undo=True,
                        )
        else:
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

    # Draw popovers after the dock panels so the viewport cannot cover them.
    var toolbar = studio.dock.find_panel("Toolbar")
    if toolbar[0]>=0 and toolbar[2]:
        ref bar = studio.menu.bar
        bar.x = Int(toolbar[3].x)+4
        bar.y = Int(toolbar[3].y)
        bar.height = min(30,max(1,Int(toolbar[3].h)))
        bar.screen_w = screen_w
        bar.screen_h = screen_h
        var right = bar.left(len(bar.groups))+14
        if right+200 < Int(toolbar[3].right()):
            var state = "Playing" if studio.pie.mode==PIE_PLAYING else ("Paused" if studio.pie.mode==PIE_PAUSED else "Scene editor")
            var message = studio.menu.message if studio.menu.message else String(state)+"   |   GPU "+_fmt_f32_2dp(Float32(gpu_ms))+" ms"
            studio.ui_ctx._draw_text(message,Float32(right),Float32(bar.y+3),13,studio.ui_ctx.theme.text_muted)
        bar.draw(studio.ui_ctx.draw_list,studio.ui_ctx.font,font_slot)

    studio.ui_ctx.end_frame()

    for i in range(len(studio.ui_ctx.draw_list.prims)):
        var p = studio.ui_ctx.draw_list.prims[i]
        if (p.header & 0xF) == PRIM_GLYPH and p.param0 < Float32(0.5):
            p.param0 = max(Float32(1.5), p.param0 * Float32(255.0))
        ui.prims.append(p)

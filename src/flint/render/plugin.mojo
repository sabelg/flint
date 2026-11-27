"""Flint v5 Render Plugin & ECS Component Schemas (`flint.render.plugin`).

Provides:
- `RenderPluginStateResource`: `Resource` owning `RenderExtractor`, `UiDrawList`,
  optional `TerrainClipmapPacket`, and frame render telemetry.
- 5 POD `SerializableComponent` ECS schemas:
  `MeshRendererComponent`, `CameraComponent`, `PointLightComponent`,
  `MaterialOverrideComponent`, and `VatCrowdComponent`.
- `register_render_components(mut world: World)` & `register_render_codecs(mut reg: ComponentRegistry)`.
- `RenderPlugin`: inserts `RenderWorld` directly as a `Resource`, `ScalableRenderer`
  directly as a `Resource`, and `RenderPluginStateResource`; extracts interpolated
  `PrevGlobalTransform -> GlobalTransform` poses, cameras, lights, materials, and
  VAT crowd instances in `STAGE_LAST`, calls `renderer.render_world(...)` and
  clears `ui_draw_list`; calls `renderer.shutdown()` deterministically in `STAGE_SHUTDOWN`.
"""
from std.collections import Dict, List
from std.math import isfinite, max, min
from flint.anim.plugin import AnimAssetCache, SkeletonComponent
from flint.anim.pose import SkeletonInstance
from flint.assets.model import NativeModel, NativeSkin
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FixedTime,
    FrameTime,
    Resource,
    Resources,
    STAGE_LAST,
    STAGE_SHUTDOWN,
)
from flint.assets.asset_ref import AssetRef
from flint.assets.json import JsonDocument
from flint.core.jobs import JobPool
from flint.core.math import Mat4, Vec3
from flint.core.serial import BlobReader, BlobWriter, fnv64, fourcc
from flint.ecs.world import (
    Entity,
    ParallelCommands,
    STORAGE_TABLE,
    World,
    comp_hash64,
)
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_renderer import AaaSkinnedMeshEntry
from flint.render.aaa_skinning_rt import DualQuat
from flint.render.material_graph import (
    MaterialAsset,
    MAT_DOMAIN_CLEARCOAT,
    MAT_DOMAIN_STANDARD_PBR,
    MAT_DOMAIN_SUBSURFACE_SKIN,
    MAT_FLAG_ALPHA_MASKED,
    MAT_FLAG_CLEARCOAT,
    MAT_FLAG_EMISSIVE_BLOOM,
    MAT_FLAG_SHEEN,
    MAT_FLAG_SUBSURFACE,
)
from flint.render.render_world import (
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    RenderCameraPacket,
    RenderExtractor,
    RenderWorld,
    ScalableRenderer,
    TerrainClipmapPacket,
)
from flint.scene.scene import ComponentRegistry
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    PrevGlobalTransform,
    SerializableComponent,
    interpolate_global_transform,
)
from flint.ui.draw_list import UiDrawList


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Resource & Binary Hook Helpers
# ---------------------------------------------------------------------------

def _render_bin_write[C: SerializableComponent](comp_addr: Int, writer_addr: Int):
    ref comp = Pointer[C, ImmutAnyOrigin](unsafe_from_address=comp_addr)[unsafe_offset=0]
    ref out = Pointer[BlobWriter, MutAnyOrigin](unsafe_from_address=writer_addr)[unsafe_offset=0]
    out.begin(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
    comp.write_bin(out)
    out.end()


def _render_bin_read[C: SerializableComponent](comp_addr: Int, reader_addr: Int) -> Bool:
    ref reader = Pointer[BlobReader, MutAnyOrigin](unsafe_from_address=reader_addr)[unsafe_offset=0]
    try:
        var ver = reader.enter(C.SCHEMA_FOURCC, C.SCHEMA_VERSION)
        var val = C.read_bin(reader, ver)
        reader.leave()
        var dst = Pointer[C, MutAnyOrigin](unsafe_from_address=comp_addr)
        __get_address_as_uninit_lvalue(dst._mlir_value) = val^
        return True
    except:
        return False


def _register_non_trivial_render[C: SerializableComponent](mut world: World):
    var tid = world.register_component[C]()
    comptime is_triv = C.__copy_ctor_is_trivial and C.__move_ctor_is_trivial and C.__del__is_trivial
    if not is_triv and tid >= 0 and not world.components[tid].has_custom_bin:
        world.set_component_binary_hooks(
            tid, _render_bin_write[C], _render_bin_read[C]
        )


struct _OverrideSlotCacheEntry(Copyable, Movable):
    var slot_id: Int
    var base_mat_id: Int
    var last_base_mat_sig: UInt64
    var last_ov: MaterialOverrideComponent
    var last_seen_frame: UInt64

    def __init__(
        out self,
        slot_id: Int,
        base_mat_id: Int,
        last_base_mat_sig: UInt64,
        var last_ov: MaterialOverrideComponent,
        last_seen_frame: UInt64,
    ):
        self.slot_id = slot_id
        self.base_mat_id = base_mat_id
        self.last_base_mat_sig = last_base_mat_sig
        self.last_ov = last_ov^
        self.last_seen_frame = last_seen_frame

    def __init__(out self, *, copy: Self):
        self.slot_id = copy.slot_id
        self.base_mat_id = copy.base_mat_id
        self.last_base_mat_sig = copy.last_base_mat_sig
        self.last_ov = copy.last_ov.copy()
        self.last_seen_frame = copy.last_seen_frame


@fieldwise_init
struct SkinnedPrimitive(Copyable, Movable):
    var node: Int
    var skin: Int
    var owns_material: Bool
    var entry: AaaSkinnedMeshEntry


struct RenderPluginStateResource(Resource, Movable):
    """Owns `RenderExtractor`, `UiDrawList`, optional `TerrainClipmapPacket`, and stats."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.render.RenderPluginStateResource")
    var extractor: RenderExtractor
    var ui_draw_list: UiDrawList
    var terrain_clipmap: TerrainClipmapPacket
    var rendered_frames: UInt64
    var last_render_ok: Bool
    var extra_lights: List[AaaClusterLight]
    var override_slots: Dict[UInt64, _OverrideSlotCacheEntry]
    var free_override_slots: List[Int]
    var editor_camera_override: Bool
    var editor_camera: RenderCameraPacket
    var has_editor_camera: Bool
    var studio_override_mat_ids: Dict[UInt64, Int]
    var entity_smear_scales: Dict[UInt64, Vec3]
    var skinned_sync_count: Int
    var skinned_entries: Dict[UInt64, AaaSkinnedMeshEntry]
    var skin_device_generation: Int
    var free_skin_materials: List[Int]
    var skinned_parts: Dict[UInt64, List[SkinnedPrimitive]]
    var preview_skeletons: Dict[UInt64, SkeletonInstance]
    var skinned_model_ids: Dict[UInt64, UInt64]
    var skinned_invalidated: Dict[UInt64, Bool]
    var preview_bone_palettes: Dict[UInt64, List[DualQuat]]
    var preview_prev_bone_palettes: Dict[UInt64, List[DualQuat]]
    var preview_world_transforms: Dict[UInt64, Mat4]

    def __init__(out self):
        self.extractor = RenderExtractor()
        self.ui_draw_list = UiDrawList()
        self.terrain_clipmap = TerrainClipmapPacket()
        self.rendered_frames = 0
        self.last_render_ok = True
        self.extra_lights = List[AaaClusterLight]()
        self.override_slots = Dict[UInt64, _OverrideSlotCacheEntry]()
        self.free_override_slots = List[Int]()
        self.editor_camera_override = False
        self.editor_camera = RenderCameraPacket()
        self.has_editor_camera = False
        self.studio_override_mat_ids = Dict[UInt64, Int]()
        self.entity_smear_scales = Dict[UInt64, Vec3]()
        self.skinned_sync_count = 0
        self.skinned_entries = Dict[UInt64, AaaSkinnedMeshEntry]()
        self.skin_device_generation = 0
        self.free_skin_materials = List[Int]()
        self.skinned_parts = Dict[UInt64, List[SkinnedPrimitive]]()
        self.preview_skeletons = Dict[UInt64, SkeletonInstance]()
        self.skinned_model_ids = Dict[UInt64, UInt64]()
        self.skinned_invalidated = Dict[UInt64, Bool]()
        self.preview_bone_palettes = Dict[UInt64, List[DualQuat]]()
        self.preview_prev_bone_palettes = Dict[UInt64, List[DualQuat]]()
        self.preview_world_transforms = Dict[UInt64, Mat4]()


# ---------------------------------------------------------------------------
# 5 Serializable ECS Render Components
# ---------------------------------------------------------------------------

struct MeshRendererComponent(SerializableComponent):
    """Renderable mesh instance referencing registered `mesh_id` & `material_id` + `AssetRef`s."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.MeshRendererComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "MeshRendererComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RMSH")
    comptime SCHEMA_VERSION: Int = 2

    var mesh_id: Int
    var material_id: Int
    var cast_shadows: Bool
    var visible: Bool
    var mesh_ref: AssetRef
    var material_ref: AssetRef

    def __init__(
        out self,
        mesh_id: Int = 0,
        material_id: Int = -1,
        cast_shadows: Bool = True,
        visible: Bool = True,
        var mesh_ref: AssetRef = AssetRef(),
        var material_ref: AssetRef = AssetRef(),
    ):
        self.mesh_id = mesh_id
        self.material_id = material_id
        self.cast_shadows = cast_shadows
        self.visible = visible
        self.mesh_ref = mesh_ref^
        self.material_ref = material_ref^

    def __init__(out self, *, copy: Self):
        self.mesh_id = copy.mesh_id
        self.material_id = copy.material_id
        self.cast_shadows = copy.cast_shadows
        self.visible = copy.visible
        self.mesh_ref = copy.mesh_ref.copy()
        self.material_ref = copy.material_ref.copy()

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.mesh_id)
        out.integer(self.material_id)
        out.boolean(self.cast_shadows)
        out.boolean(self.visible)
        self.mesh_ref.write_bin(out)
        self.material_ref.write_bin(out)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var mid = reader.integer()
        var mat = reader.integer()
        var cs = reader.boolean()
        var vis = reader.boolean()
        var mref = AssetRef()
        var matref = AssetRef()
        if version >= 2:
            mref = AssetRef.read_bin(reader, 1)
            matref = AssetRef.read_bin(reader, 1)
        if reader.failed:
            raise Error("MeshRendererComponent: truncated blob")
        return Self(mid, mat, cs, vis, mref^, matref^)

    def write_json(self) -> String:
        return (
            "{\"mesh_id\": " + String(self.mesh_id)
            + ", \"material_id\": " + String(self.material_id)
            + ", \"cast_shadows\": " + ("true" if self.cast_shadows else "false")
            + ", \"visible\": " + ("true" if self.visible else "false")
            + ", \"mesh_ref\": " + self.mesh_ref.write_json()
            + ", \"material_ref\": " + self.material_ref.write_json()
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var mid = doc.integer(doc.object_get(node, "mesh_id"), 0)
        var mat = doc.integer(doc.object_get(node, "material_id"), -1)
        var cs = doc.boolean(doc.object_get(node, "cast_shadows"), True)
        var vis = doc.boolean(doc.object_get(node, "visible"), True)
        var mr_node = doc.object_get(node, "mesh_ref")
        var mref = AssetRef.read_json(doc, mr_node) if mr_node >= 0 else AssetRef()
        var mat_node = doc.object_get(node, "material_ref")
        var matref = AssetRef.read_json(doc, mat_node) if mat_node >= 0 else AssetRef()
        return Self(mid, mat, cs, vis, mref^, matref^)


struct CameraComponent(SerializableComponent, ImplicitlyCopyable):
    """Perspective camera component driven by entity `GlobalTransform` (`-Z` forward)."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.CameraComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "CameraComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RCAM")
    comptime SCHEMA_VERSION: Int = 1

    var fov_y_deg: Float32
    var near_z: Float32
    var far_z: Float32
    var exposure: Float32
    var active: Bool
    var look_target_offset: Vec3

    def __init__(
        out self,
        fov_y_deg: Float32 = 55.0,
        near_z: Float32 = 0.10,
        far_z: Float32 = 1000.0,
        exposure: Float32 = 1.0,
        active: Bool = True,
        look_target_offset: Vec3 = Vec3(0.0, 0.0, -1.0),
    ):
        self.fov_y_deg = fov_y_deg
        self.near_z = near_z
        self.far_z = far_z
        self.exposure = exposure
        self.active = active
        self.look_target_offset = look_target_offset

    def write_bin(self, mut out: BlobWriter):
        out.f32(self.fov_y_deg)
        out.f32(self.near_z)
        out.f32(self.far_z)
        out.f32(self.exposure)
        out.boolean(self.active)
        out.vec3(self.look_target_offset)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var fov = reader.f32()
        var nz = reader.f32()
        var fz = reader.f32()
        var exp = reader.f32()
        var act = reader.boolean()
        var off = reader.vec3()
        if reader.failed:
            raise Error("CameraComponent: truncated blob")
        return Self(fov, nz, fz, exp, act, off)

    def write_json(self) -> String:
        return (
            "{\"fov_y_deg\": " + _f32_s(self.fov_y_deg)
            + ", \"near_z\": " + _f32_s(self.near_z)
            + ", \"far_z\": " + _f32_s(self.far_z)
            + ", \"exposure\": " + _f32_s(self.exposure)
            + ", \"active\": " + ("true" if self.active else "false")
            + ", \"look_target_offset\": " + _vec3_json(self.look_target_offset)
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var fov = Float32(doc.number(doc.object_get(node, "fov_y_deg"), 55.0))
        var nz = Float32(doc.number(doc.object_get(node, "near_z"), 0.10))
        var fz = Float32(doc.number(doc.object_get(node, "far_z"), 1000.0))
        var exp = Float32(doc.number(doc.object_get(node, "exposure"), 1.0))
        var act = doc.boolean(doc.object_get(node, "active"), True)
        var off = _read_vec3_json(doc, doc.object_get(node, "look_target_offset"), Vec3(0.0, 0.0, -1.0))
        return Self(fov, nz, fz, exp, act, off)


struct PointLightComponent(SerializableComponent, ImplicitlyCopyable):
    """Clustered point light extracted as `AaaClusterLight.point`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.PointLightComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "PointLightComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RLGT")
    comptime SCHEMA_VERSION: Int = 1

    var radiance: Vec3
    var radius: Float32
    var enabled: Bool

    def __init__(
        out self,
        radiance: Vec3 = Vec3(4.0, 4.0, 4.0),
        radius: Float32 = 12.0,
        enabled: Bool = True,
    ):
        self.radiance = radiance
        self.radius = radius
        self.enabled = enabled

    def write_bin(self, mut out: BlobWriter):
        out.vec3(self.radiance)
        out.f32(self.radius)
        out.boolean(self.enabled)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var rad = reader.vec3()
        var r = reader.f32()
        var en = reader.boolean()
        if reader.failed:
            raise Error("PointLightComponent: truncated blob")
        return Self(rad, r, en)

    def write_json(self) -> String:
        return (
            "{\"radiance\": " + _vec3_json(self.radiance)
            + ", \"radius\": " + _f32_s(self.radius)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var rad = _read_vec3_json(doc, doc.object_get(node, "radiance"), Vec3(4.0, 4.0, 4.0))
        var r = Float32(doc.number(doc.object_get(node, "radius"), 12.0))
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        return Self(rad, r, en)


struct MaterialOverrideComponent(SerializableComponent):
    """Per-entity material ID and PBR parameter override."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.MaterialOverrideComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "MaterialOverrideComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RMAT")
    comptime SCHEMA_VERSION: Int = 3

    var material_id: Int
    var base_color: Vec3
    var roughness: Float32
    var metallic: Float32
    var emissive: Vec3
    var enabled: Bool
    var opacity: Float32
    var alpha_cutoff: Float32
    var clearcoat: Float32
    var clearcoat_roughness: Float32
    var sheen: Float32
    var subsurface: Float32
    var material_ref: AssetRef

    def __init__(
        out self,
        material_id: Int = -1,
        base_color: Vec3 = Vec3(0.8, 0.8, 0.8),
        roughness: Float32 = 0.35,
        metallic: Float32 = 0.0,
        emissive: Vec3 = Vec3(),
        enabled: Bool = True,
        opacity: Float32 = 1.0,
        alpha_cutoff: Float32 = 0.0,
        clearcoat: Float32 = 0.0,
        clearcoat_roughness: Float32 = 0.08,
        sheen: Float32 = 0.0,
        subsurface: Float32 = 0.0,
        var material_ref: AssetRef = AssetRef(),
    ):
        self.material_id = material_id
        self.base_color = base_color
        self.roughness = roughness
        self.metallic = metallic
        self.emissive = emissive
        self.enabled = enabled
        self.opacity = opacity
        self.alpha_cutoff = alpha_cutoff
        self.clearcoat = clearcoat
        self.clearcoat_roughness = clearcoat_roughness
        self.sheen = sheen
        self.subsurface = subsurface
        self.material_ref = material_ref^

    def __init__(out self, *, copy: Self):
        self.material_id = copy.material_id
        self.base_color = copy.base_color
        self.roughness = copy.roughness
        self.metallic = copy.metallic
        self.emissive = copy.emissive
        self.enabled = copy.enabled
        self.opacity = copy.opacity
        self.alpha_cutoff = copy.alpha_cutoff
        self.clearcoat = copy.clearcoat
        self.clearcoat_roughness = copy.clearcoat_roughness
        self.sheen = copy.sheen
        self.subsurface = copy.subsurface
        self.material_ref = copy.material_ref.copy()

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.material_id)
        out.vec3(self.base_color)
        out.f32(self.roughness)
        out.f32(self.metallic)
        out.vec3(self.emissive)
        out.boolean(self.enabled)
        out.f32(self.opacity)
        out.f32(self.alpha_cutoff)
        out.f32(self.clearcoat)
        out.f32(self.clearcoat_roughness)
        out.f32(self.sheen)
        out.f32(self.subsurface)
        self.material_ref.write_bin(out)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        var mid = reader.integer()
        var bc = reader.vec3()
        var rg = reader.f32()
        var mt = reader.f32()
        var em = reader.vec3()
        var en = reader.boolean()
        var op = Float32(1.0)
        var ac = Float32(0.0)
        var cc = Float32(0.0)
        var ccr = Float32(0.08)
        var sh = Float32(0.0)
        var sss = Float32(0.0)
        var mref = AssetRef()
        if version >= 2:
            op = reader.f32()
            ac = reader.f32()
            cc = reader.f32()
            ccr = reader.f32()
            sh = reader.f32()
            sss = reader.f32()
        if version >= 3:
            mref = AssetRef.read_bin(reader, 1)
        if reader.failed:
            raise Error("MaterialOverrideComponent: truncated blob")
        return Self(mid, bc, rg, mt, em, en, op, ac, cc, ccr, sh, sss, mref^)

    def write_json(self) -> String:
        return (
            "{\"material_id\": " + String(self.material_id)
            + ", \"base_color\": " + _vec3_json(self.base_color)
            + ", \"roughness\": " + _f32_s(self.roughness)
            + ", \"metallic\": " + _f32_s(self.metallic)
            + ", \"emissive\": " + _vec3_json(self.emissive)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + ", \"opacity\": " + _f32_s(self.opacity)
            + ", \"alpha_cutoff\": " + _f32_s(self.alpha_cutoff)
            + ", \"clearcoat\": " + _f32_s(self.clearcoat)
            + ", \"clearcoat_roughness\": " + _f32_s(self.clearcoat_roughness)
            + ", \"sheen\": " + _f32_s(self.sheen)
            + ", \"subsurface\": " + _f32_s(self.subsurface)
            + ", \"material_ref\": " + self.material_ref.write_json()
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var mid = doc.integer(doc.object_get(node, "material_id"), -1)
        var bc = _read_vec3_json(doc, doc.object_get(node, "base_color"), Vec3(0.8, 0.8, 0.8))
        var rg = Float32(doc.number(doc.object_get(node, "roughness"), 0.35))
        var mt = Float32(doc.number(doc.object_get(node, "metallic"), 0.0))
        var em = _read_vec3_json(doc, doc.object_get(node, "emissive"), Vec3())
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        var op = Float32(doc.number(doc.object_get(node, "opacity"), 1.0))
        var ac = Float32(doc.number(doc.object_get(node, "alpha_cutoff"), 0.0))
        var cc = Float32(doc.number(doc.object_get(node, "clearcoat"), 0.0))
        var ccr = Float32(doc.number(doc.object_get(node, "clearcoat_roughness"), 0.08))
        var sh = Float32(doc.number(doc.object_get(node, "sheen"), 0.0))
        var sss = Float32(doc.number(doc.object_get(node, "subsurface"), 0.0))
        var mat_node = doc.object_get(node, "material_ref")
        var mref = AssetRef.read_json(doc, mat_node) if mat_node >= 0 else AssetRef()
        return Self(mid, bc, rg, mt, em, en, op, ac, cc, ccr, sh, sss, mref^)


struct VatCrowdComponent(SerializableComponent, ImplicitlyCopyable):
    """Baked Vertex-Animation-Texture crowd instance component."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.VatCrowdComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "VatCrowdComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RVAT")
    comptime SCHEMA_VERSION: Int = 1

    var clip_id: Int
    var phase_sec: Float32
    var prev_phase_sec: Float32
    var material_id: Int
    var visible: Bool

    def __init__(
        out self,
        clip_id: Int = 0,
        phase_sec: Float32 = 0.0,
        prev_phase_sec: Float32 = 0.0,
        material_id: Int = -1,
        visible: Bool = True,
    ):
        self.clip_id = clip_id
        self.phase_sec = phase_sec
        self.prev_phase_sec = prev_phase_sec
        self.material_id = material_id
        self.visible = visible

    def write_bin(self, mut out: BlobWriter):
        out.integer(self.clip_id)
        out.f32(self.phase_sec)
        out.f32(self.prev_phase_sec)
        out.integer(self.material_id)
        out.boolean(self.visible)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var cid = reader.integer()
        var ph = reader.f32()
        var pph = reader.f32()
        var mat = reader.integer()
        var vis = reader.boolean()
        if reader.failed:
            raise Error("VatCrowdComponent: truncated blob")
        return Self(cid, ph, pph, mat, vis)

    def write_json(self) -> String:
        return (
            "{\"clip_id\": " + String(self.clip_id)
            + ", \"phase_sec\": " + _f32_s(self.phase_sec)
            + ", \"material_id\": " + String(self.material_id)
            + ", \"visible\": " + ("true" if self.visible else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var cid = doc.integer(doc.object_get(node, "clip_id"), 0)
        var ph = Float32(doc.number(doc.object_get(node, "phase_sec"), 0.0))
        var mat = doc.integer(doc.object_get(node, "material_id"), -1)
        var vis = doc.boolean(doc.object_get(node, "visible"), True)
        return Self(cid, ph, ph, mat, vis)


struct EnvironmentComponent(SerializableComponent, ImplicitlyCopyable):
    """Directional sun, atmosphere, IBL intensity, and camera exposure settings."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.EnvironmentComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "EnvironmentComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RENV")
    comptime SCHEMA_VERSION: Int = 1

    var sun_dir: Vec3
    var sun_radiance: Vec3
    var exposure: Float32
    var ibl_intensity: Float32
    var turbidity: Float32
    var enabled: Bool

    def __init__(
        out self,
        sun_dir: Vec3 = Vec3(0.378803, 0.797480, -0.458551),
        sun_radiance: Vec3 = Vec3(3.2, 3.0, 2.7),
        exposure: Float32 = 1.0,
        ibl_intensity: Float32 = 1.0,
        turbidity: Float32 = 2.5,
        enabled: Bool = True,
    ):
        self.sun_dir = sun_dir
        self.sun_radiance = sun_radiance
        self.exposure = exposure
        self.ibl_intensity = ibl_intensity
        self.turbidity = turbidity
        self.enabled = enabled

    def write_bin(self, mut out: BlobWriter):
        out.vec3(self.sun_dir)
        out.vec3(self.sun_radiance)
        out.f32(self.exposure)
        out.f32(self.ibl_intensity)
        out.f32(self.turbidity)
        out.boolean(self.enabled)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var sd = reader.vec3()
        var sr = reader.vec3()
        var exp = reader.f32()
        var ibl = reader.f32()
        var turb = reader.f32()
        var en = reader.boolean()
        if reader.failed:
            raise Error("EnvironmentComponent: truncated blob")
        return Self(sd, sr, exp, ibl, turb, en)

    def write_json(self) -> String:
        return (
            "{\"sun_dir\": " + _vec3_json(self.sun_dir)
            + ", \"sun_radiance\": " + _vec3_json(self.sun_radiance)
            + ", \"exposure\": " + _f32_s(self.exposure)
            + ", \"ibl_intensity\": " + _f32_s(self.ibl_intensity)
            + ", \"turbidity\": " + _f32_s(self.turbidity)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var sd = _read_vec3_json(
            doc, doc.object_get(node, "sun_dir"), Vec3(0.378803, 0.797480, -0.458551)
        )
        var sr = _read_vec3_json(
            doc, doc.object_get(node, "sun_radiance"), Vec3(3.2, 3.0, 2.7)
        )
        var exp = Float32(doc.number(doc.object_get(node, "exposure"), 1.0))
        var ibl = Float32(doc.number(doc.object_get(node, "ibl_intensity"), 1.0))
        var turb = Float32(doc.number(doc.object_get(node, "turbidity"), 2.5))
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        return Self(sd, sr, exp, ibl, turb, en)


struct SpotLightComponent(SerializableComponent, ImplicitlyCopyable):
    """Clustered spot light extracted as `AaaClusterLight.spot`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64("flint.render.SpotLightComponent")
    comptime STORAGE_MODE: Int = STORAGE_TABLE
    comptime SCHEMA_NAME: StaticString = "SpotLightComponent"
    comptime SCHEMA_FOURCC: UInt32 = fourcc("RSPT")
    comptime SCHEMA_VERSION: Int = 1

    var radiance: Vec3
    var range: Float32
    var inner_deg: Float32
    var outer_deg: Float32
    var direction: Vec3
    var enabled: Bool

    def __init__(
        out self,
        radiance: Vec3 = Vec3(8.0, 6.0, 4.0),
        range: Float32 = 18.0,
        inner_deg: Float32 = 18.0,
        outer_deg: Float32 = 35.0,
        direction: Vec3 = Vec3(0.0, 0.0, -1.0),
        enabled: Bool = True,
    ):
        self.radiance = radiance
        self.range = range
        self.inner_deg = inner_deg
        self.outer_deg = outer_deg
        self.direction = direction
        self.enabled = enabled

    def write_bin(self, mut out: BlobWriter):
        out.vec3(self.radiance)
        out.f32(self.range)
        out.f32(self.inner_deg)
        out.f32(self.outer_deg)
        out.vec3(self.direction)
        out.boolean(self.enabled)

    @staticmethod
    def read_bin(mut reader: BlobReader, version: Int) raises -> Self:
        _ = version
        var rad = reader.vec3()
        var rng = reader.f32()
        var inn = reader.f32()
        var out_d = reader.f32()
        var dir = reader.vec3()
        var en = reader.boolean()
        if reader.failed:
            raise Error("SpotLightComponent: truncated blob")
        return Self(rad, rng, inn, out_d, dir, en)

    def write_json(self) -> String:
        return (
            "{\"radiance\": " + _vec3_json(self.radiance)
            + ", \"range\": " + _f32_s(self.range)
            + ", \"inner_deg\": " + _f32_s(self.inner_deg)
            + ", \"outer_deg\": " + _f32_s(self.outer_deg)
            + ", \"direction\": " + _vec3_json(self.direction)
            + ", \"enabled\": " + ("true" if self.enabled else "false")
            + "}"
        )

    @staticmethod
    def read_json(doc: JsonDocument, node: Int) raises -> Self:
        var rad = _read_vec3_json(doc, doc.object_get(node, "radiance"), Vec3(8.0, 6.0, 4.0))
        var rng = Float32(doc.number(doc.object_get(node, "range"), 18.0))
        var inn = Float32(doc.number(doc.object_get(node, "inner_deg"), 18.0))
        var out_d = Float32(doc.number(doc.object_get(node, "outer_deg"), 35.0))
        var dir = _read_vec3_json(doc, doc.object_get(node, "direction"), Vec3(0.0, 0.0, -1.0))
        var en = doc.boolean(doc.object_get(node, "enabled"), True)
        return Self(rad, rng, inn, out_d, dir, en)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_render_components(mut world: World):
    """Idempotently registers all 7 render components + custom binary hooks on `world`."""
    _register_non_trivial_render[MeshRendererComponent](world)
    _ = world.register_component[CameraComponent]()
    _ = world.register_component[PointLightComponent]()
    _register_non_trivial_render[MaterialOverrideComponent](world)
    _ = world.register_component[VatCrowdComponent]()
    _ = world.register_component[EnvironmentComponent]()
    _ = world.register_component[SpotLightComponent]()


def register_render_codecs(mut reg: ComponentRegistry):
    """Registers all 7 render `SerializableComponent` codecs on `ComponentRegistry`."""
    reg.register[MeshRendererComponent]()
    reg.register[CameraComponent]()
    reg.register[PointLightComponent]()
    reg.register[MaterialOverrideComponent]()
    reg.register[VatCrowdComponent]()
    reg.register[EnvironmentComponent]()
    reg.register[SpotLightComponent]()


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def _interpolated_gt(world: World, entity: Entity, alpha: Float32) -> GlobalTransform:
    var gt_opt = world.get_ptr[GlobalTransform](entity)
    if not gt_opt:
        return GlobalTransform()
    ref curr = gt_opt.value()[unsafe_offset=0]
    var prev_opt = world.get_ptr[PrevGlobalTransform](entity)
    if prev_opt:
        return interpolate_global_transform(prev_opt.value()[unsafe_offset=0].value, curr, alpha)
    return curr.copy()


def _ov_equals(a: MaterialOverrideComponent, b: MaterialOverrideComponent) -> Bool:
    return (
        a.material_id == b.material_id
        and a.base_color.x == b.base_color.x
        and a.base_color.y == b.base_color.y
        and a.base_color.z == b.base_color.z
        and a.roughness == b.roughness
        and a.metallic == b.metallic
        and a.emissive.x == b.emissive.x
        and a.emissive.y == b.emissive.y
        and a.emissive.z == b.emissive.z
        and a.enabled == b.enabled
        and a.opacity == b.opacity
        and a.alpha_cutoff == b.alpha_cutoff
        and a.clearcoat == b.clearcoat
        and a.clearcoat_roughness == b.clearcoat_roughness
        and a.sheen == b.sheen
        and a.subsurface == b.subsurface
        and a.material_ref == b.material_ref
    )


def _material_asset_sig(renderer: ScalableRenderer, base_mat_id: Int) -> UInt64:
    if base_mat_id < 0 or base_mat_id >= len(renderer.materials):
        return 0
    ref m = renderer.materials[base_mat_id]
    var w = BlobWriter()
    w.integer(Int(m.albedo_slot))
    w.integer(Int(m.normal_slot))
    w.integer(Int(m.orm_slot))
    w.integer(Int(m.emissive_slot))
    w.integer(Int(m.detail_normal_slot))
    w.u32(m.domain)
    w.u32(m.flags)
    w.vec3(m.base_color)
    w.f32(m.roughness)
    w.f32(m.metallic)
    w.vec3(m.emissive)
    w.f32(m.opacity)
    w.f32(m.alpha_cutoff)
    w.f32(m.clearcoat)
    w.f32(m.clearcoat_roughness)
    w.vec3(m.sheen_color)
    w.f32(m.subsurface_amount)
    var h = fnv64(w.data)
    return h if h != 0 else UInt64(1)


def build_overridden_material(
    renderer: ScalableRenderer, base_mat_id: Int, ov: MaterialOverrideComponent
) -> MaterialAsset:
    var mat = (
        renderer.materials[base_mat_id].copy()
        if (base_mat_id >= 0 and base_mat_id < len(renderer.materials))
        else MaterialAsset()
    )
    mat.base_color = Vec3(
        ov.base_color.x if isfinite(ov.base_color.x) else mat.base_color.x,
        ov.base_color.y if isfinite(ov.base_color.y) else mat.base_color.y,
        ov.base_color.z if isfinite(ov.base_color.z) else mat.base_color.z,
    )
    if isfinite(ov.roughness):
        mat.roughness = max(Float32(0.04), min(Float32(1.0), ov.roughness))
    if isfinite(ov.metallic):
        mat.metallic = max(Float32(0.0), min(Float32(1.0), ov.metallic))
    mat.emissive = Vec3(
        max(Float32(0.0), ov.emissive.x) if isfinite(ov.emissive.x) else mat.emissive.x,
        max(Float32(0.0), ov.emissive.y) if isfinite(ov.emissive.y) else mat.emissive.y,
        max(Float32(0.0), ov.emissive.z) if isfinite(ov.emissive.z) else mat.emissive.z,
    )
    if isfinite(ov.opacity):
        mat.opacity = max(Float32(0.0), min(Float32(1.0), ov.opacity))
    if isfinite(ov.alpha_cutoff):
        mat.alpha_cutoff = max(Float32(0.0), min(Float32(1.0), ov.alpha_cutoff))
    if isfinite(ov.clearcoat):
        mat.clearcoat = max(Float32(0.0), min(Float32(1.0), ov.clearcoat))
    if isfinite(ov.clearcoat_roughness):
        mat.clearcoat_roughness = max(Float32(0.04), min(Float32(1.0), ov.clearcoat_roughness))
    if isfinite(ov.sheen):
        var sh = max(Float32(0.0), min(Float32(1.0), ov.sheen))
        mat.sheen_color = Vec3(sh, sh, sh)
        if sh > Float32(0.01):
            mat.flags |= MAT_FLAG_SHEEN
        else:
            mat.flags &= ~MAT_FLAG_SHEEN
    if isfinite(ov.subsurface):
        var sss = max(Float32(0.0), min(Float32(1.0), ov.subsurface))
        mat.subsurface_amount = sss
        if sss > Float32(0.01):
            mat.flags |= MAT_FLAG_SUBSURFACE
            if mat.domain == MAT_DOMAIN_STANDARD_PBR:
                mat.domain = MAT_DOMAIN_SUBSURFACE_SKIN
        else:
            mat.flags &= ~MAT_FLAG_SUBSURFACE
    if mat.alpha_cutoff > Float32(0.01):
        mat.flags |= MAT_FLAG_ALPHA_MASKED
    else:
        mat.flags &= ~MAT_FLAG_ALPHA_MASKED
    if mat.clearcoat > Float32(0.01):
        mat.flags |= MAT_FLAG_CLEARCOAT
        if mat.domain == MAT_DOMAIN_STANDARD_PBR:
            mat.domain = MAT_DOMAIN_CLEARCOAT
    else:
        mat.flags &= ~MAT_FLAG_CLEARCOAT
    if mat.emissive.x + mat.emissive.y + mat.emissive.z > Float32(0.01):
        mat.flags |= MAT_FLAG_EMISSIVE_BLOOM
    else:
        mat.flags &= ~MAT_FLAG_EMISSIVE_BLOOM
    return mat^


def _skinned_primitives(model: NativeModel) -> List[Tuple[Int, Int, Int]]:
    var parts = List[Tuple[Int, Int, Int]]()
    for node_index in range(len(model.nodes)):
        ref node = model.nodes[node_index]
        if node.mesh < 0:
            continue
        var n = max(1, len(node.mesh_indices))
        for j in range(n):
            var mi = node.mesh_indices[j] if len(node.mesh_indices) > 0 else node.mesh
            if mi < 0 or mi >= len(model.meshes):
                continue
            ref mesh = model.meshes[mi]
            var verts = mesh.vertex_count()
            if verts == 0 or len(mesh.joints) < 4 * verts or len(mesh.weights) < 4 * verts:
                continue
            var skin = node.skin if node.skin >= 0 else mesh.skin
            if skin < 0 and len(model.skins) == 1:
                skin = 0
            if skin >= 0 and skin < len(model.skins) and len(model.skins[skin].joints) > 0:
                parts.append((mi, skin, node_index))
    return parts^


def _skin_palette(skin: NativeSkin, matrices: List[Mat4]) -> List[DualQuat]:
    var palette = List[DualQuat](capacity=len(skin.joints))
    for j in range(len(skin.joints)):
        var node = skin.joints[j]
        if node < 0 or node >= len(matrices):
            palette.append(DualQuat())
            continue
        var ibm = Mat4()
        if (j + 1) * 16 <= len(skin.inverse_bind):
            for k in range(16):
                ibm.values[k] = skin.inverse_bind[j * 16 + k]
        palette.append(DualQuat.from_mat4(matrices[node].multiply(ibm)))
    return palette^


def _sync_skin_parts(
    parts: List[SkinnedPrimitive], model: NativeModel, skel: SkeletonInstance,
    key: UInt64, mut state: RenderPluginStateResource, mut renderer: ScalableRenderer,
):
    for skin_id in range(len(model.skins)):
        var cur: List[DualQuat]
        var prev: List[DualQuat]
        var preview = state.preview_bone_palettes.get(key)
        if skin_id == 0 and preview:
            cur = preview.value().copy()
            var previous = state.preview_prev_bone_palettes.get(key)
            prev = previous.value().copy() if previous else cur.copy()
        else:
            ref skin = model.skins[skin_id]
            cur = _skin_palette(skin, skel.world_matrices)
            prev = (_skin_palette(skin, skel.prev_world_matrices)
                    if len(skel.prev_world_matrices) == len(skel.world_matrices) else cur.copy())
        for part in parts:
            if part.skin == skin_id:
                var morphs = skel.pose.morphs.get(part.node)
                if morphs:
                    var previous = skel.prev_morphs.get(part.node)
                    if previous and skel.has_prev:
                        renderer.update_skinned_mesh_bones(part.entry, cur, prev, True, morphs.value(), previous.value())
                    else:
                        renderer.update_skinned_mesh_bones(part.entry, cur, prev, True, morphs.value())
                else:
                    renderer.update_skinned_mesh_bones(part.entry, cur, prev, defer_to_frame=True)
                state.skinned_sync_count += 1


def _fits_skin_parts(renderer: ScalableRenderer, model: NativeModel, sources: List[Tuple[Int, Int, Int]]) -> Bool:
    if not renderer.uses_vulkan():
        return True
    var arena = renderer.gpu_slot[0].skin_arena.copy()
    for source in sources:
        ref mesh = model.meshes[source[0]]
        var vertices = mesh.vertex_count()
        if arena.alloc_vertices(vertices)[1] != vertices:
            return False
        if arena.alloc_bone_palette(len(model.skins[source[1]].joints)) < 0:
            return False
        if mesh.morph_count > 0 and arena.alloc_morph_targets(mesh.morph_count, vertices)[2] != mesh.morph_count:
            return False
    return True


def release_scene_skin(mut state: RenderPluginStateResource, mut renderer: ScalableRenderer, key: UInt64):
    var parts = state.skinned_parts.pop(key, List[SkinnedPrimitive]())
    for part in parts:
        renderer.remove_skinned_mesh(part.entry)
        if part.owns_material and part.entry.mesh_entry.material_id >= 0:
            state.free_skin_materials.append(part.entry.mesh_entry.material_id)
    _ = state.skinned_entries.pop(key, AaaSkinnedMeshEntry())
    _ = state.skinned_model_ids.pop(key, UInt64(0))
    _ = state.skinned_invalidated.pop(key, False)


def extract_scene_meshes(
    mut world: World,
    mut resources: Resources,
    mut render_world: RenderWorld,
    mut renderer: ScalableRenderer,
    mut state: RenderPluginStateResource,
    alpha: Float32,
):
    """Extract visible meshes and synchronize their material and skeleton state for this frame."""
    if renderer.uses_vulkan():
        var generation = renderer.gpu_slot[0].device_generation()
        if generation != state.skin_device_generation:
            for key in state.skinned_parts.keys():
                state.skinned_invalidated[key] = True
            state.skin_device_generation = generation
    var stale_skins = List[UInt64]()
    for key in state.skinned_parts.keys():
        var entity = Entity.from_bits(key)
        var sk = world.get_ptr[SkeletonComponent](entity)
        if not sk or sk.value()[].model_id == 0:
            stale_skins.append(key)
    for key in stale_skins:
        var entity = Entity.from_bits(key)
        var mc = world.get_ptr_mut[MeshRendererComponent](entity)
        var previous = state.skinned_entries.get(key)
        if mc and previous and mc.value()[].mesh_id == previous.value().mesh_entry.mesh_id:
            mc.value()[].mesh_id = -1
        release_scene_skin(state, renderer, key)
    # 1.5 Sync Skinned Meshes (SkeletonComponent -> per-entity DualQuat bone palette)
    if world.component_id[SkeletonComponent]() >= 0:
        var anim_cache_opt = resources.get_ptr[AnimAssetCache]()
        if anim_cache_opt:
            ref anim_cache = anim_cache_opt.value()[unsafe_offset=0]
            var sk_mask = world.bit_for[SkeletonComponent]()
            var sk_entities = world.query_entities(sk_mask)
            for si in range(len(sk_entities)):
                var se = sk_entities[si]
                var sk_opt = world.get_ptr[SkeletonComponent](se)
                if not sk_opt:
                    continue
                ref sk = sk_opt.value()[unsafe_offset=0]
                var skey = se.to_bits()
                var m_idx = anim_cache.find_model(sk.model_id)
                if sk.model_id == 0 or m_idx < 0:
                    continue
                ref model = anim_cache.models[m_idx]
                if len(model.skins) == 0 or len(model.meshes) == 0:
                    continue
                var invalid = state.skinned_invalidated.get(skey)
                var needs_reload = Bool(invalid) and invalid.value()
                var old_id = state.skinned_model_ids.get(skey)
                var mc = world.get_ptr_mut[MeshRendererComponent](se)
                if (not old_id or old_id.value() != sk.model_id or needs_reload
                    or skey not in state.skinned_parts):
                    release_scene_skin(state, renderer, skey)
                    var parts = List[SkinnedPrimitive]()
                    var sources = _skinned_primitives(model)
                    if not _fits_skin_parts(renderer, model, sources):
                        if mc:
                            mc.value()[].mesh_id = -1
                        continue
                    for source in sources:
                        var mi = source[0]
                        var skin = source[1]
                        var mat = -1
                        if mc and (len(parts) == 0 or not mc.value()[].material_ref.is_empty()
                                   or mc.value()[].mesh_ref.is_empty()):
                            mat = mc.value()[].material_id
                        var owns_material = mat < 0
                        if owns_material:
                            var material = renderer.build_native_mesh_material(model.meshes[mi], model)
                            if len(state.free_skin_materials) > 0:
                                mat = state.free_skin_materials.pop()
                                renderer.update_material_asset_deferred(mat, material)
                            else:
                                mat = renderer.register_material(material)
                        var entry = renderer.add_skinned_mesh(model.meshes[mi], mat, len(model.skins[skin].joints))
                        if entry.mesh_entry.mesh_id >= 0:
                            parts.append(SkinnedPrimitive(source[2], skin, owns_material, entry^))
                    if len(parts) > 0:
                        state.skinned_entries[skey] = parts[0].entry.copy()
                    state.skinned_parts[skey] = parts^
                    state.skinned_model_ids[skey] = sk.model_id
                    state.skinned_invalidated[skey] = False
                var bound_parts = state.skinned_parts.get(skey)
                if not bound_parts:
                    continue
                ref parts = bound_parts.value()
                if len(parts) == 0:
                    if mc:
                        mc.value()[].mesh_id = -1
                    continue
                if mc:
                    mc.value()[].mesh_id = parts[0].entry.mesh_entry.mesh_id
                var preview_skel = state.preview_skeletons.get(skey)
                if preview_skel:
                    _sync_skin_parts(parts, model, preview_skel.value(), skey, state, renderer)
                elif len(sk.skeleton.world_matrices) == 0 and sk.skeleton.node_count > 0:
                    var tmp = sk.skeleton.copy()
                    tmp.ensure_topology(model)
                    tmp.update_matrices(model)
                    tmp.snapshot_prev_matrices()
                    _sync_skin_parts(parts, model, tmp, skey, state, renderer)
                else:
                    _sync_skin_parts(parts, model, sk.skeleton, skey, state, renderer)

    # 2. Extract MeshRendererComponents (+ isolated per-entity MaterialOverrideComponent slots)
    var frame_tag = state.rendered_frames + 1
    var mesh_mask = world.bit_for[MeshRendererComponent]().union(world.bit_for[GlobalTransform]())
    var mesh_entities = world.query_entities(mesh_mask)
    for i in range(len(mesh_entities)):
        var e = mesh_entities[i]
        var mc_opt = world.get_ptr[MeshRendererComponent](e)
        if not mc_opt or not mc_opt.value()[unsafe_offset=0].visible:
            continue
        ref mc = mc_opt.value()[unsafe_offset=0]
        if mc.mesh_id < 0:
            continue
        var key = e.to_bits()
        var eff_mat = mc.material_id
        var override_all_parts = not mc.material_ref.is_empty() or mc.mesh_ref.is_empty()
        var studio_mat_opt = state.studio_override_mat_ids.get(key)
        if studio_mat_opt:
            override_all_parts = True
            eff_mat = studio_mat_opt.value()
        else:
            var ov_opt = world.get_ptr[MaterialOverrideComponent](e)
            if ov_opt and ov_opt.value()[unsafe_offset=0].enabled:
                override_all_parts = True
                ref ov = ov_opt.value()[unsafe_offset=0]
                var base_mat_id = ov.material_id if ov.material_id >= 0 else mc.material_id
                if base_mat_id < 0 and mc.mesh_id >= 0 and mc.mesh_id < len(renderer.cpu_mesh_materials):
                    base_mat_id = renderer.cpu_mesh_materials[mc.mesh_id]
                var base_mat_sig = _material_asset_sig(renderer, base_mat_id)
                var cached_opt = state.override_slots.get(key)
                if (
                    cached_opt
                    and cached_opt.value().slot_id >= 0
                    and cached_opt.value().slot_id < len(renderer.materials)
                ):
                    var entry = cached_opt.value().copy()
                    entry.last_seen_frame = frame_tag
                    if (
                        entry.base_mat_id != base_mat_id
                        or entry.last_base_mat_sig != base_mat_sig
                        or not _ov_equals(entry.last_ov, ov)
                    ):
                        var mat = build_overridden_material(renderer, base_mat_id, ov)
                        renderer.update_material_asset_deferred(entry.slot_id, mat)
                        entry.base_mat_id = base_mat_id
                        entry.last_base_mat_sig = base_mat_sig
                        entry.last_ov = ov.copy()
                    eff_mat = entry.slot_id
                    state.override_slots[key] = entry^
                else:
                    var mat = build_overridden_material(renderer, base_mat_id, ov)
                    var slot_id = -1
                    while len(state.free_override_slots) > 0:
                        var cand = state.free_override_slots.pop()
                        if cand >= 0 and cand < len(renderer.materials):
                            slot_id = cand
                            renderer.update_material_asset_deferred(slot_id, mat)
                            break
                    if slot_id < 0:
                        slot_id = renderer.register_material(mat)
                    if slot_id >= 0:
                        state.override_slots[key] = _OverrideSlotCacheEntry(
                            slot_id, base_mat_id, base_mat_sig, ov.copy(), frame_tag
                        )
                        eff_mat = slot_id

        var gt = _interpolated_gt(world, e, alpha)
        var prev_wm_opt = state.preview_world_transforms.get(key)
        var world_mat = prev_wm_opt.value().copy() if prev_wm_opt else gt.matrix.copy()
        var sm_opt = state.entity_smear_scales.get(key)
        if sm_opt:
            var sm = sm_opt.value()
            var sm_mat = Mat4()
            sm_mat.set(0, 0, sm.x)
            sm_mat.set(1, 1, sm.y)
            sm_mat.set(2, 2, sm.z)
            world_mat = world_mat.multiply(sm_mat)
        state.extractor.extract_entity(
            render_world,
            key,
            mc.mesh_id,
            world_mat,
            eff_mat,
            mc.cast_shadows,
        )

        # Sibling primitives share the entity transform, including its previous
        # matrix. Recording that transform again would erase its motion vector.
        var extra = state.skinned_parts.get(key)
        if extra and len(extra.value()) > 1 and extra.value()[0].entry.mesh_entry.mesh_id == mc.mesh_id:
            var first = len(render_world.instances) - 1
            for pi in range(1, len(extra.value())):
                ref entry = extra.value()[pi].entry.mesh_entry
                var packet = render_world.instances[first].copy()
                packet.mesh_id = entry.mesh_id
                if not override_all_parts:
                    packet.material_id = entry.material_id
                render_world.instances.append(packet^)

    var stale_ov_keys = List[UInt64]()
    for item in state.override_slots.items():
        if item.value.last_seen_frame != frame_tag:
            stale_ov_keys.append(item.key)
    for ki in range(len(stale_ov_keys)):
        var popped = state.override_slots.pop(
            stale_ov_keys[ki], _OverrideSlotCacheEntry(-1, -1, 0, MaterialOverrideComponent(), 0)
        )
        if popped.slot_id >= 0:
            state.free_override_slots.append(popped.slot_id)

def _sys_render_extract_and_draw(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var rw_opt = resources.get_ptr_mut[RenderWorld]()
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    var st_opt = resources.get_ptr_mut[RenderPluginStateResource]()
    if not rw_opt or not rend_opt or not st_opt:
        return

    ref render_world = rw_opt.value()[unsafe_offset=0]
    ref renderer = rend_opt.value()[unsafe_offset=0]
    ref state = st_opt.value()[unsafe_offset=0]

    var frame_idx = UInt64(0)
    var frame_dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        frame_idx = ft_opt.value()[unsafe_offset=0].frame_index
        frame_dt = ft_opt.value()[unsafe_offset=0].delta_seconds

    var alpha = Float32(1.0)
    var fix_opt = resources.get_ptr[FixedTime]()
    if fix_opt:
        alpha = fix_opt.value()[unsafe_offset=0].overstep_alpha

    state.extractor.begin_extraction(render_world, frame_idx, frame_dt)
    if state.terrain_clipmap.active:
        render_world.terrain_clipmap = state.terrain_clipmap.copy()

    # 1. Extract active CameraComponent (unless Edit-mode camera override is active)
    if state.editor_camera_override:
        if state.has_editor_camera:
            render_world.camera = state.editor_camera
    else:
        var cam_mask = world.bit_for[CameraComponent]().union(world.bit_for[GlobalTransform]())
        var cam_entities = world.query_entities(cam_mask)
        for i in range(len(cam_entities)):
            var e = cam_entities[i]
            var cam_opt = world.get_ptr[CameraComponent](e)
            if not cam_opt or not cam_opt.value()[unsafe_offset=0].active:
                continue
            ref cam = cam_opt.value()[unsafe_offset=0]
            var gt = _interpolated_gt(world, e, alpha)
            var eye = gt.translation
            var off = cam.look_target_offset
            if off.length_squared() <= Float32(1.0e-8):
                off = Vec3(0.0, 0.0, -1.0)
            var target = eye + gt.rotation.rotate(off)
            var up = gt.rotation.rotate(Vec3(0.0, 1.0, 0.0)).normalized()
            render_world.camera = RenderCameraPacket(
                eye, target, up, cam.fov_y_deg, cam.near_z, cam.far_z, cam.exposure
            )
            break

    extract_scene_meshes(world, resources, render_world, renderer, state, alpha)

    # 3. Extract VatCrowdComponents
    var vat_mask = world.bit_for[VatCrowdComponent]().union(world.bit_for[GlobalTransform]())
    var vat_entities = world.query_entities(vat_mask)
    for i in range(len(vat_entities)):
        var e = vat_entities[i]
        var vc_opt = world.get_ptr[VatCrowdComponent](e)
        if not vc_opt or not vc_opt.value()[unsafe_offset=0].visible:
            continue
        ref vc = vc_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt(world, e, alpha)
        state.extractor.extract_vat_crowd_entity(
            render_world,
            e.to_bits(),
            vc.clip_id,
            vc.phase_sec,
            vc.prev_phase_sec,
            gt.matrix.copy(),
            vc.material_id,
        )

    # 4. Extract PointLightComponents, SpotLightComponents & EnvironmentComponent
    var env_mask = world.bit_for[EnvironmentComponent]()
    var env_entities = world.query_entities(env_mask)
    for i in range(len(env_entities)):
        var ec_opt = world.get_ptr[EnvironmentComponent](env_entities[i])
        if not ec_opt or not ec_opt.value()[unsafe_offset=0].enabled:
            continue
        ref ec = ec_opt.value()[unsafe_offset=0]
        if ec.sun_dir.length_squared() > Float32(1.0e-8):
            render_world.sun_dir = ec.sun_dir.normalized()
        render_world.sun_radiance = Vec3(
            max(Float32(0.0), ec.sun_radiance.x) if isfinite(ec.sun_radiance.x) else render_world.sun_radiance.x,
            max(Float32(0.0), ec.sun_radiance.y) if isfinite(ec.sun_radiance.y) else render_world.sun_radiance.y,
            max(Float32(0.0), ec.sun_radiance.z) if isfinite(ec.sun_radiance.z) else render_world.sun_radiance.z,
        )
        if isfinite(ec.ibl_intensity):
            render_world.ibl_intensity = max(Float32(0.0), ec.ibl_intensity)
        if isfinite(ec.exposure):
            render_world.camera.exposure = max(Float32(0.01), ec.exposure)
        break

    var lgt_mask = world.bit_for[PointLightComponent]().union(world.bit_for[GlobalTransform]())
    var lgt_entities = world.query_entities(lgt_mask)
    for i in range(len(lgt_entities)):
        var e = lgt_entities[i]
        var lc_opt = world.get_ptr[PointLightComponent](e)
        if not lc_opt or not lc_opt.value()[unsafe_offset=0].enabled:
            continue
        ref lc = lc_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt(world, e, alpha)
        render_world.cluster_lights.append(
            AaaClusterLight.point(gt.translation, lc.radiance, lc.radius)
        )

    var spot_mask = world.bit_for[SpotLightComponent]().union(world.bit_for[GlobalTransform]())
    var spot_entities = world.query_entities(spot_mask)
    for i in range(len(spot_entities)):
        var e = spot_entities[i]
        var sc_opt = world.get_ptr[SpotLightComponent](e)
        if not sc_opt or not sc_opt.value()[unsafe_offset=0].enabled:
            continue
        ref sc = sc_opt.value()[unsafe_offset=0]
        var gt = _interpolated_gt(world, e, alpha)
        var local_dir = (
            sc.direction if sc.direction.length_squared() > Float32(1.0e-8) else Vec3(0.0, 0.0, -1.0)
        )
        var world_dir = gt.rotation.rotate(local_dir).normalized()
        render_world.cluster_lights.append(
            AaaClusterLight.spot(
                gt.translation,
                world_dir,
                sc.radiance,
                sc.range,
                sc.inner_deg,
                sc.outer_deg,
            )
        )

    for li in range(len(state.extra_lights)):
        render_world.cluster_lights.append(state.extra_lights[li].copy())

    state.extractor.end_extraction()
    state.last_render_ok = renderer.render_world(render_world, state.ui_draw_list)
    state.rendered_frames += 1
    state.ui_draw_list.clear()
    state.extra_lights.clear()
    state.studio_override_mat_ids.clear()
    state.entity_smear_scales.clear()
    state.preview_skeletons.clear()
    state.preview_bone_palettes.clear()
    state.preview_prev_bone_palettes.clear()
    state.preview_world_transforms.clear()
    render_world.overlay_lines.clear()


def _sys_render_shutdown(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    if rend_opt:
        rend_opt.value()[unsafe_offset=0].shutdown()


def _has_render_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct RenderPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `RenderWorld`, `ScalableRenderer`, render ECS schemas, and `STAGE_LAST` drawing."""
    var tier: UInt32
    var width: Int
    var height: Int
    var tsr_scale: Float32

    def __init__(
        out self,
        tier: UInt32 = RENDER_TIER_2_AAA_VISBUFFER_RT,
        width: Int = 640,
        height: Int = 360,
        tsr_scale: Float32 = 1.0,
    ):
        self.tier = tier
        self.width = width
        self.height = height
        self.tsr_scale = tsr_scale

    def build(self, mut app: EngineApp):
        register_render_components(app.world)
        if not app.resources.has[RenderWorld]():
            app.insert_resource[RenderWorld](RenderWorld())
        if not app.resources.has[ScalableRenderer]():
            app.insert_resource[ScalableRenderer](
                ScalableRenderer(
                    tier=self.tier,
                    width=self.width,
                    height=self.height,
                    tsr_scale=self.tsr_scale,
                )
            )
        if not app.resources.has[RenderPluginStateResource]():
            app.insert_resource[RenderPluginStateResource](RenderPluginStateResource())

        var b_gt = app.world.bit_for[GlobalTransform]()
        var b_prev = app.world.bit_for[PrevGlobalTransform]()
        var b_mesh = app.world.bit_for[MeshRendererComponent]()
        var b_cam = app.world.bit_for[CameraComponent]()
        var b_lgt = app.world.bit_for[PointLightComponent]()
        var b_spot = app.world.bit_for[SpotLightComponent]()
        var b_env = app.world.bit_for[EnvironmentComponent]()
        var b_mat = app.world.bit_for[MaterialOverrideComponent]()
        var b_vat = app.world.bit_for[VatCrowdComponent]()
        var all_reads = (
            b_gt.union(b_prev).union(b_mesh).union(b_cam)
            .union(b_lgt).union(b_spot).union(b_env).union(b_mat).union(b_vat)
        )

        if not _has_render_system(app, STAGE_LAST, "flint.render.extract_and_draw"):
            _ = app.add_system(
                STAGE_LAST,
                "flint.render.extract_and_draw",
                _sys_render_extract_and_draw,
                reads=all_reads,
                exclusive=True,
            )
        if not _has_render_system(app, STAGE_SHUTDOWN, "flint.render.shutdown"):
            var render_shutdown = app.add_system(
                STAGE_SHUTDOWN,
                "flint.render.shutdown",
                _sys_render_shutdown,
                exclusive=True,
            )
            # WSI needs a live wl_surface while it releases the swapchain and device.
            # Exclusive systems run in registration order unless ordered explicitly.
            ref stage = app.schedule.stages[STAGE_SHUTDOWN]
            var last = len(stage.systems)-1
            for i in range(last):
                if stage.systems[i].name == "flint.app.window_shutdown":
                    var system = stage.systems[last].copy()
                    for k in range(last, i, -1):
                        stage.systems[k] = stage.systems[k-1].copy()
                    stage.systems[i] = system^
                    stage.systems[i+1].explicit_after.append(render_shutdown)
                    stage.dirty = True
                    break

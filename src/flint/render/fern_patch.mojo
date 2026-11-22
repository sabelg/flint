"""GPU wind and screen-size LOD for instances of one reference fern."""
from std.math import sin
from flint.core.math import Vec3
from flint.core.transform import Mat4, Quat
from flint.procedural.reference_fern import FernReference, build_reference_fern
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_renderer import (
    AaaSkinnedMeshEntry,
    FlintAaaVulkanRenderer,
)
from flint.render.aaa_skinning_rt import DualQuat


def _wind_bone(time: Float32) -> DualQuat:
    var angle = 0.10 * sin(1.8 * time) + 0.035 * sin(3.1 * time + 0.4)
    var bend = Quat.from_axis_angle(Vec3(0, 0, 1), angle)
    return DualQuat.from_quat_trans(bend, Vec3())


struct FernPatch(Movable):
    var meshes: List[AaaSkinnedMeshEntry]
    var submitted: Int
    var current: List[DualQuat]
    var previous: List[DualQuat]

    def __init__(
        out self,
        mut renderer: FlintAaaVulkanRenderer,
        reference: FernReference,
        seed: Int = 1,
    ) raises:
        var material = AaaMaterial(roughness=0.74)
        material.albedo_tex = renderer.upload_bc7_texture(
            reference.atlas.pixels,
            reference.atlas.width,
            reference.atlas.height,
        )
        if material.albedo_tex < 0:
            raise Error("Could not upload fern atlas")
        material.two_sided_foliage = True
        material.subsurface_color = Vec3(0.36, 0.65, 0.14)
        material.subsurface_amount = 0.18
        var material_id = renderer.add_material(material)
        self.meshes = List[AaaSkinnedMeshEntry]()
        for lod in range(3):
            var mesh = build_reference_fern(reference, lod, seed)
            # Joint zero is fixed. Joint one bends the outer part of each frond.
            var joints = List[UInt32](length=mesh.vertex_count(), fill=0x0100)
            var weights = List[Vec3](capacity=mesh.vertex_count())
            for i in range(mesh.vertex_count()):
                var w = mesh.texcoords1[2 * i]
                weights.append(Vec3(1 - w, w, 0))
            self.meshes.append(
                renderer.add_skinned_mesh(
                    mesh, joints, weights, material_id, bone_count=2
                )
            )
        self.submitted = 0
        self.current = [DualQuat(), _wind_bone(0)]
        self.previous = self.current.copy()

    def begin_frame(mut self, seconds: Float32, previous_seconds: Float32):
        self.current[1] = _wind_bone(seconds)
        self.previous[1] = _wind_bone(previous_seconds)
        self.submitted = 0

    def draw(
        mut self,
        mut renderer: FlintAaaVulkanRenderer,
        transform: Mat4,
        scale: Float32,
        diameter_px: Float32,
    ):
        """Queue one stationary plant. Instances of the same LOD share one wind dispatch per frame.
        """
        var lod = 0 if diameter_px >= 110 else (1 if diameter_px >= 45 else 2)
        ref skin = self.meshes[lod]
        if (self.submitted & (1 << lod)) == 0:
            _ = renderer.update_skinned_mesh_bones(
                skin, self.current, self.previous, defer_to_frame=True
            )
            self.submitted |= 1 << lod
        renderer.draw_instance(
            skin.mesh_entry.mesh_id,
            transform,
            transform,
            uniform_scale=scale,
            has_prev=True,
        )

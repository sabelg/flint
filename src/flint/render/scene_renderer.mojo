"""High-level game-agnostic 3D PBR scene and volumetric VFX renderer.

Wraps `SoftwareFramebuffer`, `LightingEnvironment`, and `FxTextures` so any 3D
application can render glTF 2.0 `NativeModel` hierarchies, GGX PBR materials,
analytic ray-traced sphere shadows, dynamic point/event lights, volumetric
ribbons/beams/billboards, and SDF HUDs on Vulkan, AMDGPU, or CPU fallback.
"""
from std.math import sqrt, sin, cos
from flint.core.math import Vec3
from flint.core.transform import Mat4
from flint.assets.model import NativeMesh, NativeTexture, NativeModel
from flint.render.lighting import LightingProfile, LightingEnvironment, LocalLight
from flint.render.fx_atlas import (
    FxTextures,
    TEX_VIGNETTE,
    TEX_DROPLET,
    TEX_RIBBON,
    TEX_FLAME,
)
from flint.render.gpu_layout import (
    CONSTANT_KEY,
    CONSTANT_WORLD,
    CONSTANT_LIGHTS,
    CONSTANT_SPHERES,
    MAX_SHADOW_SPHERES,
)
from flint.render.gpu_raster import (
    BACKEND_CPU,
    BACKEND_AMDGPU,
    BACKEND_VULKAN,
    BACKEND_AUTOMATIC,
    FRAME_SETS,
    requested_backend,
    backend_name,
)
from flint.render.rasterizer import SoftwareFramebuffer, RenderVertex, dot, cross


comptime FX_SCORCH: Int = 1
comptime FX_BOLT: Int = 5
comptime FX_STRIPES: Int = 6
comptime FX_SOFT: Int = 7
comptime FX_RIBBON: Int = TEX_RIBBON
comptime FX_FLAME: Int = TEX_FLAME


def mul_argb(a: UInt32, b: UInt32) -> UInt32:
    var aa = ((a >> 24) & 255) * ((b >> 24) & 255) // 255
    var ar = ((a >> 16) & 255) * ((b >> 16) & 255) // 255
    var ag = ((a >> 8) & 255) * ((b >> 8) & 255) // 255
    var ab = (a & 255) * (b & 255) // 255
    return (aa << 24) | (ar << 16) | (ag << 8) | ab


struct GpuSceneRenderer(Movable):
    var fb: SoftwareFramebuffer
    var lighting: LightingEnvironment
    var fx: FxTextures
    var empty_tex: List[UInt32]
    var sphere_centers: List[Vec3]
    var sphere_radii: List[Float32]
    var active_profile: LightingProfile
    var active_quality: Int
    var active_reduced: Bool
    var frame_index: Int
    var scene_ended: Bool

    def __init__(out self, width: Int = 1280, height: Int = 720, prefer_gpu: Bool = True):
        self.fb = SoftwareFramebuffer(width, height)
        self.lighting = LightingEnvironment()
        self.fx = FxTextures()
        self.empty_tex = List[UInt32]()
        self.sphere_centers = List[Vec3]()
        self.sphere_radii = List[Float32]()
        self.active_profile = LightingProfile()
        self.active_quality = 2
        self.active_reduced = False
        self.frame_index = 0
        self.scene_ended = False

        if prefer_gpu:
            var wanted = requested_backend()
            if wanted == BACKEND_VULKAN:
                _ = self.fb.enable_gpu(BACKEND_VULKAN)
            elif wanted == BACKEND_AMDGPU:
                _ = self.fb.enable_gpu(BACKEND_AMDGPU)
            elif wanted == BACKEND_AUTOMATIC:
                # Try Vulkan first (works across NVIDIA, AMD, Intel, Mesa), then direct AMDGPU.
                if not self.fb.enable_gpu(BACKEND_VULKAN):
                    _ = self.fb.enable_gpu(BACKEND_AMDGPU)

    def active_backend_name(self) -> String:
        if self.fb.gpu.active:
            return backend_name(self.fb.gpu.backend)
        return "cpu"

    def using_gpu(self) -> Bool:
        return self.fb.gpu.active

    def begin_frame(
        mut self,
        cam_pos: Vec3,
        cam_target: Vec3,
        fov_y: Float32,
        profile: LightingProfile,
        dt: Float32 = 1.0 / 60.0,
        quality: Int = 2,
        reduced_effects: Bool = False,
    ):
        self.active_profile = profile
        self.active_quality = quality
        self.active_reduced = reduced_effects
        self.scene_ended = False
        self.sphere_centers.clear()
        self.sphere_radii.clear()
        self.fb.shadow_sphere_count = 0
        self.fb.cpu_sphere_centers.clear()
        self.fb.cpu_sphere_radii_sq.clear()
        self.fb.cpu_lights.clear()

        if self.fb.gpu.active:
            var slot = self.frame_index % FRAME_SETS
            if not self.fb.gpu.begin(slot):
                self.fb.stop_recording_if_gpu_lost()
            self.frame_index += 1

        self.fb.set_camera(cam_pos, cam_target, fov_y)
        self.fb.configure_post(profile, quality, reduced_effects)
        var fog_rgb = profile.fog_rgb()
        self.fb.clear(fog_rgb[0], fog_rgb[1], fog_rgb[2])

        self.lighting.time += max(Float32(0), dt)
        self.lighting.update_events(dt)
        self.lighting.begin_frame_with(profile, cam_pos, quality, reduced_effects)

    def add_point_light(
        mut self,
        position: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        priority: Int = 50,
        stable_id: Int = 0,
    ):
        self.lighting.register(position, color, radius, intensity, priority, stable_id)

    def emit_event_light(
        mut self,
        position: Vec3,
        color: Vec3,
        radius: Float32,
        intensity: Float32,
        duration: Float32,
        stable_id: Int = -1,
    ):
        self.lighting.emit_event(position, color, radius, intensity, duration, stable_id)

    def add_shadow_sphere(mut self, center: Vec3, radius: Float32):
        if radius <= 0.01 or len(self.sphere_centers) >= MAX_SHADOW_SPHERES:
            return
        self.sphere_centers.append(center)
        self.sphere_radii.append(radius)

    def commit_lighting(mut self):
        """Select priority lights and upload camera, key light, point lights, and shadow spheres."""
        var per_pixel = self.active_quality >= 1 and not self.active_reduced
        self.lighting.per_pixel = per_pixel
        self.lighting.finish_frame()

        var to_key = Vec3(
            -self.active_profile.key_direction.x,
            -self.active_profile.key_direction.y,
            -self.active_profile.key_direction.z,
        ).normalized()
        self.fb.cpu_to_key = to_key
        self.fb.cpu_key_color = self.active_profile.key_color
        self.fb.cpu_lights.clear()
        for i in range(self.lighting.selected_count):
            self.fb.cpu_lights.append(self.lighting.selected[i])

        var sphere_count = min(MAX_SHADOW_SPHERES, len(self.sphere_centers))
        self.fb.shadow_sphere_count = sphere_count
        self.fb.cpu_sphere_centers.clear()
        self.fb.cpu_sphere_radii_sq.clear()
        for i in range(sphere_count):
            self.fb.cpu_sphere_centers.append(self.sphere_centers[i])
            self.fb.cpu_sphere_radii_sq.append(self.sphere_radii[i] * self.sphere_radii[i])

        if not self.fb.gpu.active:
            return

        self.fb.upload_view()
        self.fb.gpu.write_float(CONSTANT_KEY + 0, to_key.x)
        self.fb.gpu.write_float(CONSTANT_KEY + 4, to_key.y)
        self.fb.gpu.write_float(CONSTANT_KEY + 8, to_key.z)
        self.fb.gpu.write_float(CONSTANT_KEY + 12, self.active_profile.key_color.x)
        self.fb.gpu.write_float(CONSTANT_KEY + 16, self.active_profile.key_color.y)
        self.fb.gpu.write_float(CONSTANT_KEY + 20, self.active_profile.key_color.z)

        var w_word = CONSTANT_WORLD // 4
        self.fb.gpu.constants.write32(w_word + 0, 0)
        self.fb.gpu.constants.write32(w_word + 1, 0)
        self.fb.gpu.constants.write32(w_word + 2, 0)
        self.fb.gpu.constants.write32(w_word + 3, UInt32(sphere_count))
        self.fb.gpu.constants.write32(w_word + 4, 0)
        self.fb.gpu.constants.write32(w_word + 5, 0)
        self.fb.gpu.constants.write32(w_word + 6, UInt32(self.lighting.selected_count))
        self.fb.gpu.constants.write32(w_word + 7, 0)

        self.fb.gpu.clear_tile_lights()
        for i in range(self.lighting.selected_count):
            var pl = self.lighting.selected[i]
            var base = CONSTANT_LIGHTS + i * 32
            var inv_r2 = 1.0 / max(Float32(0.01), pl.radius * pl.radius)
            self.fb.gpu.write_float(base + 0, pl.position.x)
            self.fb.gpu.write_float(base + 4, pl.position.y)
            self.fb.gpu.write_float(base + 8, pl.position.z)
            self.fb.gpu.write_float(base + 12, 0.0)
            self.fb.gpu.write_float(base + 16, pl.color.x * pl.intensity)
            self.fb.gpu.write_float(base + 20, pl.color.y * pl.intensity)
            self.fb.gpu.write_float(base + 24, pl.color.z * pl.intensity)
            self.fb.gpu.write_float(base + 28, inv_r2)
            self.fb.mark_light_on_screen(i, pl.position, pl.radius)

        for i in range(sphere_count):
            var sbase = CONSTANT_SPHERES + i * 16
            var c = self.sphere_centers[i]
            var r = self.sphere_radii[i]
            self.fb.gpu.write_float(sbase + 0, c.x)
            self.fb.gpu.write_float(sbase + 4, c.y)
            self.fb.gpu.write_float(sbase + 8, c.z)
            self.fb.gpu.write_float(sbase + 12, r * r)

    def make_lit_vertex(
        self,
        world_pos: Vec3,
        world_normal: Vec3,
        u: Float32,
        v: Float32,
        color: UInt32,
        ao: Float32,
        roughness: Float32,
        metallic: Float32,
        emissive: Float32,
        unlit: Bool = False,
        no_fog: Bool = False,
    ) -> RenderVertex:
        var rv = RenderVertex(world_pos, world_normal, u, v, color)
        rv.roughness = roughness
        rv.metallic = metallic
        rv.emissive = emissive
        if unlit:
            rv.light = Vec3(1.0, 1.0, 1.0)
            rv.occlusion = -1.0
        else:
            rv.light = self.lighting.sample(world_pos, world_normal, ao)
            rv.occlusion = self.lighting.ambient_occlusion(ao) if self.lighting.per_pixel else Float32(-1.0)
        if no_fog:
            rv.fog = 0.0
        else:
            var depth = max(Float32(0.0), dot(world_pos - self.fb.camera, self.fb.forward))
            rv.fog = self.lighting.fog_amount(world_pos, depth)
        return rv^

    def draw_mesh(
        mut self,
        mut mesh: NativeMesh,
        mut textures: List[NativeTexture],
        transform: Mat4,
        tint: UInt32 = 0xFFFFFFFF,
        emissive_boost: Float32 = 0.0,
        alpha_override: Int = -1,
        unlit: Bool = False,
        no_fog: Bool = False,
    ):
        var vcount = mesh.vertex_count()
        if vcount < 3:
            return
        var has_normals = len(mesh.normals) >= vcount * 3
        var has_uvs = len(mesh.texcoords) >= vcount * 2
        var has_colors = len(mesh.colors) >= vcount * 4
        var has_ao = len(mesh.ao) >= vcount
        var base_tint = mul_argb(mesh.base_color, tint)
        var alpha_byte = UInt8(255)
        if alpha_override >= 0:
            alpha_byte = UInt8(max(0, min(255, alpha_override)))
        elif mesh.alpha_mode == 2:
            alpha_byte = UInt8((base_tint >> 24) & 255)
            base_tint = 0xFF000000 | (base_tint & 0x00FFFFFF)

        var emissive = max(Float32(0.0), mesh.emissive_strength + emissive_boost)
        var verts = List[RenderVertex](capacity=vcount)
        for vi in range(vcount):
            var lp = Vec3(mesh.positions[vi * 3], mesh.positions[vi * 3 + 1], mesh.positions[vi * 3 + 2])
            var ln = Vec3(mesh.normals[vi * 3], mesh.normals[vi * 3 + 1], mesh.normals[vi * 3 + 2]) if has_normals else Vec3(0, 1, 0)
            var wp = transform.transform_point(lp)
            var wn = transform.transform_vector(ln).normalized()
            var u = mesh.texcoords[vi * 2] if has_uvs else Float32(0.0)
            var v = mesh.texcoords[vi * 2 + 1] if has_uvs else Float32(0.0)
            var vcol = base_tint
            if has_colors:
                var cr = UInt32(mesh.colors[vi * 4])
                var cg = UInt32(mesh.colors[vi * 4 + 1])
                var cb = UInt32(mesh.colors[vi * 4 + 2])
                var ca = UInt32(mesh.colors[vi * 4 + 3])
                vcol = mul_argb(base_tint, (ca << 24) | (cr << 16) | (cg << 8) | cb)
            var ao = Float32(mesh.ao[vi]) / 255.0 if has_ao else Float32(1.0)
            verts.append(
                self.make_lit_vertex(
                    wp,
                    wn,
                    u,
                    v,
                    vcol,
                    ao,
                    mesh.roughness,
                    mesh.metallic,
                    emissive,
                    unlit=unlit,
                    no_fog=no_fog,
                )
            )

        var has_tex = mesh.texture >= 0 and mesh.texture < len(textures) and textures[mesh.texture].width > 0 and textures[mesh.texture].height > 0
        var tri_count = mesh.triangle_count()
        var indexed = len(mesh.indices) >= tri_count * 3
        for ti in range(tri_count):
            var i0 = mesh.indices[ti * 3] if indexed else ti * 3
            var i1 = mesh.indices[ti * 3 + 1] if indexed else ti * 3 + 1
            var i2 = mesh.indices[ti * 3 + 2] if indexed else ti * 3 + 2
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= vcount or i1 >= vcount or i2 >= vcount:
                continue
            if has_tex:
                var tw = textures[mesh.texture].width
                var th = textures[mesh.texture].height
                self.fb.draw_triangle(
                    verts[i0],
                    verts[i1],
                    verts[i2],
                    textures[mesh.texture].pixels,
                    tw,
                    th,
                    alpha=alpha_byte,
                    double_sided=mesh.double_sided,
                )
            else:
                self.fb.draw_triangle(
                    verts[i0],
                    verts[i1],
                    verts[i2],
                    self.empty_tex,
                    0,
                    0,
                    alpha=alpha_byte,
                    double_sided=mesh.double_sided,
                )

    def draw_model(
        mut self,
        mut model: NativeModel,
        world_mat: Mat4,
        tint: UInt32 = 0xFFFFFFFF,
        emissive_boost: Float32 = 0.0,
        alpha_override: Int = -1,
        unlit: Bool = False,
        no_fog: Bool = False,
    ):
        if len(model.meshes) == 0:
            return
        if len(model.nodes) == 0:
            for pass_blend in range(2):
                var want_blend = pass_blend == 1
                for mi in range(len(model.meshes)):
                    var is_blend = model.meshes[mi].alpha_mode == 2 or (alpha_override >= 0 and alpha_override < 255)
                    if is_blend == want_blend:
                        self.draw_mesh(
                            model.meshes[mi],
                            model.textures,
                            world_mat,
                            tint=tint,
                            emissive_boost=emissive_boost,
                            alpha_override=alpha_override,
                            unlit=unlit,
                            no_fog=no_fog,
                        )
            return

        var globals = model.global_transforms()
        for pass_blend in range(2):
            var want_blend = pass_blend == 1
            for ni in range(len(model.nodes)):
                var node_world = world_mat.multiply(globals[ni])
                var count = len(model.nodes[ni].mesh_indices)
                if count > 0:
                    for k in range(count):
                        var mi = model.nodes[ni].mesh_indices[k]
                        if mi >= 0 and mi < len(model.meshes):
                            var is_blend = model.meshes[mi].alpha_mode == 2 or (alpha_override >= 0 and alpha_override < 255)
                            if is_blend == want_blend:
                                self.draw_mesh(
                                    model.meshes[mi],
                                    model.textures,
                                    node_world,
                                    tint=tint,
                                    emissive_boost=emissive_boost,
                                    alpha_override=alpha_override,
                                    unlit=unlit,
                                    no_fog=no_fog,
                                )
                elif model.nodes[ni].mesh >= 0 and model.nodes[ni].mesh < len(model.meshes):
                    var mi = model.nodes[ni].mesh
                    var is_blend = model.meshes[mi].alpha_mode == 2 or (alpha_override >= 0 and alpha_override < 255)
                    if is_blend == want_blend:
                        self.draw_mesh(
                            model.meshes[mi],
                            model.textures,
                            node_world,
                            tint=tint,
                            emissive_boost=emissive_boost,
                            alpha_override=alpha_override,
                            unlit=unlit,
                            no_fog=no_fog,
                        )

    def draw_planar_shadow_mesh(
        mut self,
        mesh: NativeMesh,
        transform: Mat4,
        plane_point: Vec3,
        plane_normal: Vec3 = Vec3(0.0, 1.0, 0.0),
        shadow_color: UInt32 = 0xFF050A14,
        max_alpha: UInt8 = 138,
        max_height: Float32 = 70.0,
        bias: Float32 = 0.06,
    ):
        """Projects `mesh` along the active key light onto `(plane_point, plane_normal)`."""
        var vcount = mesh.vertex_count()
        if vcount < 3 or max_alpha == 0:
            return
        if mesh.alpha_mode == 2 and mesh.emissive_strength > 0.8:
            return
        var n = plane_normal.normalized()
        var light_dir = self.active_profile.key_direction.normalized()
        var n_dot_l = dot(n, light_dir)
        if n_dot_l > -0.12:
            light_dir = Vec3(light_dir.x - n.x * 0.75, light_dir.y - n.y * 0.75, light_dir.z - n.z * 0.75).normalized()
            n_dot_l = min(Float32(-0.18), dot(n, light_dir))
        var inv_neg_ndl = Float32(1.0) / (-n_dot_l)
        var col = 0xFF000000 | (shadow_color & 0x00FFFFFF)
        var inv_max_h = Float32(1.0) / max(Float32(0.5), max_height)

        var proj_verts = List[RenderVertex](capacity=vcount)
        var dists = List[Float32](capacity=vcount)
        for vi in range(vcount):
            var lp = Vec3(mesh.positions[vi * 3], mesh.positions[vi * 3 + 1], mesh.positions[vi * 3 + 2])
            var wp = transform.transform_point(lp)
            var d = dot(wp - plane_point, n)
            dists.append(d)
            var ray_t = max(Float32(0.0), d) * inv_neg_ndl
            var p_proj = wp + light_dir * ray_t + n * bias
            proj_verts.append(
                self.make_lit_vertex(
                    p_proj,
                    n,
                    0.0,
                    0.0,
                    col,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    unlit=True,
                    no_fog=False,
                )
            )

        var tri_count = mesh.triangle_count()
        var indexed = len(mesh.indices) >= tri_count * 3
        for ti in range(tri_count):
            var i0 = mesh.indices[ti * 3] if indexed else ti * 3
            var i1 = mesh.indices[ti * 3 + 1] if indexed else ti * 3 + 1
            var i2 = mesh.indices[ti * 3 + 2] if indexed else ti * 3 + 2
            if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= vcount or i1 >= vcount or i2 >= vcount:
                continue
            var d0 = dists[i0]
            var d1 = dists[i1]
            var d2 = dists[i2]
            if max(d0, max(d1, d2)) < -0.5 or min(d0, min(d1, d2)) > max_height:
                continue
            var avg_d = max(Float32(0.0), (d0 + d1 + d2) * Float32(0.33333334))
            var atten = max(Float32(0.0), min(Float32(1.0), Float32(1.0) - avg_d * inv_max_h))
            var eff_a = UInt8(max(0, min(255, Int(Float32(max_alpha) * atten))))
            if eff_a < 4:
                continue
            self.fb.draw_triangle(
                proj_verts[i0],
                proj_verts[i1],
                proj_verts[i2],
                self.empty_tex,
                0,
                0,
                alpha=eff_a,
                double_sided=True,
            )

    def draw_planar_shadow_model(
        mut self,
        mut model: NativeModel,
        world_mat: Mat4,
        plane_point: Vec3,
        plane_normal: Vec3 = Vec3(0.0, 1.0, 0.0),
        shadow_color: UInt32 = 0xFF050A14,
        max_alpha: UInt8 = 138,
        max_height: Float32 = 70.0,
        bias: Float32 = 0.06,
    ):
        """Projects every articulated node of `model` onto a deck/ground shadow receiver plane."""
        if len(model.meshes) == 0 or max_alpha == 0:
            return
        if len(model.nodes) == 0:
            for mi in range(len(model.meshes)):
                self.draw_planar_shadow_mesh(
                    model.meshes[mi],
                    world_mat,
                    plane_point,
                    plane_normal=plane_normal,
                    shadow_color=shadow_color,
                    max_alpha=max_alpha,
                    max_height=max_height,
                    bias=bias,
                )
            return

        var globals = model.global_transforms()
        for ni in range(len(model.nodes)):
            var node_world = world_mat.multiply(globals[ni])
            var count = len(model.nodes[ni].mesh_indices)
            if count > 0:
                for k in range(count):
                    var mi = model.nodes[ni].mesh_indices[k]
                    if mi >= 0 and mi < len(model.meshes):
                        self.draw_planar_shadow_mesh(
                            model.meshes[mi],
                            node_world,
                            plane_point,
                            plane_normal=plane_normal,
                            shadow_color=shadow_color,
                            max_alpha=max_alpha,
                            max_height=max_height,
                            bias=bias,
                        )
            elif model.nodes[ni].mesh >= 0 and model.nodes[ni].mesh < len(model.meshes):
                self.draw_planar_shadow_mesh(
                    model.meshes[model.nodes[ni].mesh],
                    node_world,
                    plane_point,
                    plane_normal=plane_normal,
                    shadow_color=shadow_color,
                    max_alpha=max_alpha,
                    max_height=max_height,
                    bias=bias,
                )

    def draw_billboard(
        mut self,
        center: Vec3,
        half_size: Float32,
        fx_kind: Int,
        color: UInt32,
        alpha: UInt8,
        emissive: Float32 = 1.35,
        rotation: Float32 = 0.0,
    ):
        if half_size <= 0.001 or alpha == 0 or fx_kind < 0 or fx_kind >= len(self.fx.pixels):
            return
        var c_rot = cos(rotation)
        var s_rot = sin(rotation)
        var rx = (self.fb.right * c_rot + self.fb.up * s_rot) * half_size
        var ry = (self.fb.up * c_rot - self.fb.right * s_rot) * half_size
        var n = Vec3(-self.fb.forward.x, -self.fb.forward.y, -self.fb.forward.z)
        var col = 0xFF000000 | (color & 0x00FFFFFF)

        var p0 = center - rx + ry
        var p1 = center + rx + ry
        var p2 = center + rx - ry
        var p3 = center - rx - ry

        var v0 = self.make_lit_vertex(p0, n, 0.0, 0.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var v1 = self.make_lit_vertex(p1, n, 1.0, 0.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var v2 = self.make_lit_vertex(p2, n, 1.0, 1.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var v3 = self.make_lit_vertex(p3, n, 0.0, 1.0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)

        var sz = self.fx.sizes[fx_kind]
        self.fb.draw_triangle(v0, v1, v2, self.fx.pixels[fx_kind], sz, sz, alpha=alpha, double_sided=True)
        self.fb.draw_triangle(v0, v2, v3, self.fx.pixels[fx_kind], sz, sz, alpha=alpha, double_sided=True)

    def draw_beam_3d(
        mut self,
        start: Vec3,
        end: Vec3,
        half_width: Float32,
        fx_kind: Int,
        color: UInt32,
        alpha: UInt8,
        emissive: Float32 = 1.65,
        u0: Float32 = 0.0,
        u1: Float32 = 1.0,
        v0: Float32 = 0.0,
        v1: Float32 = 1.0,
    ):
        if half_width <= 0.001 or alpha == 0 or fx_kind < 0 or fx_kind >= len(self.fx.pixels):
            return
        var axis = end - start
        if axis.length_squared() < 0.0001:
            return
        var dir_v = axis.normalized()
        var mid = (start + end) * 0.5
        var to_cam = (self.fb.camera - mid).normalized()
        var side = cross(dir_v, to_cam)
        if side.length_squared() < 0.0001:
            side = self.fb.up
        else:
            side = side.normalized()
        var offset = side * half_width
        var col = 0xFF000000 | (color & 0x00FFFFFF)

        var p0 = start + offset
        var p1 = end + offset
        var p2 = end - offset
        var p3 = start - offset

        var rv0 = self.make_lit_vertex(p0, to_cam, u0, v0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var rv1 = self.make_lit_vertex(p1, to_cam, u1, v0, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var rv2 = self.make_lit_vertex(p2, to_cam, u1, v1, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var rv3 = self.make_lit_vertex(p3, to_cam, u0, v1, col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)

        var sz = self.fx.sizes[fx_kind]
        self.fb.draw_triangle(rv0, rv1, rv2, self.fx.pixels[fx_kind], sz, sz, alpha=alpha, double_sided=True)
        self.fb.draw_triangle(rv0, rv2, rv3, self.fx.pixels[fx_kind], sz, sz, alpha=alpha, double_sided=True)

    def draw_thruster_plume(
        mut self,
        nozzle: Vec3,
        exhaust_dir: Vec3,
        length: Float32,
        radius: Float32,
        core_color: UInt32,
        outer_color: UInt32,
        alpha: UInt8 = 235,
        emissive: Float32 = 2.15,
    ):
        """Volumetric afterburner plume with TEX_FLAME outer cone, hot core, and Mach shock diamonds."""
        if length <= 0.02 or radius <= 0.01 or alpha == 0:
            return
        var dir_n = exhaust_dir.normalized()
        var tip = nozzle + dir_n * length
        # Note: TEX_FLAME has tip at v=0 (top of texture) and wide base at v=1, so we map nozzle -> v=0.88, tip -> v=0.02 along v!
        var mid = (nozzle + tip) * 0.5
        var to_cam = (self.fb.camera - mid).normalized()
        var side = cross(dir_n, to_cam)
        if side.length_squared() < 0.0001:
            side = self.fb.right
        else:
            side = side.normalized()

        var sz = self.fx.sizes[FX_FLAME]
        var outer_col = 0xFF000000 | (outer_color & 0x00FFFFFF)
        var core_col = 0xFF000000 | (core_color & 0x00FFFFFF)

        # Outer flame envelope
        var o_off = side * (radius * 1.45)
        var oa = self.make_lit_vertex(tip - o_off, to_cam, 0.0, 0.0, outer_col, 1.0, 1.0, 0.0, emissive * 0.85, unlit=True, no_fog=True)
        var ob = self.make_lit_vertex(tip + o_off, to_cam, 1.0, 0.0, outer_col, 1.0, 1.0, 0.0, emissive * 0.85, unlit=True, no_fog=True)
        var oc = self.make_lit_vertex(nozzle + o_off, to_cam, 1.0, 0.90, outer_col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        var od = self.make_lit_vertex(nozzle - o_off, to_cam, 0.0, 0.90, outer_col, 1.0, 1.0, 0.0, emissive, unlit=True, no_fog=True)
        self.fb.draw_triangle(oa, ob, oc, self.fx.pixels[FX_FLAME], sz, sz, alpha=UInt8(Int(alpha) * 3 // 4), double_sided=True)
        self.fb.draw_triangle(oa, oc, od, self.fx.pixels[FX_FLAME], sz, sz, alpha=UInt8(Int(alpha) * 3 // 4), double_sided=True)

        # White-hot inner core
        var c_tip = nozzle + dir_n * (length * 0.62)
        var c_off = side * (radius * 0.72)
        var ca = self.make_lit_vertex(c_tip - c_off, to_cam, 0.0, 0.0, core_col, 1.0, 1.0, 0.0, emissive * 1.25, unlit=True, no_fog=True)
        var cb = self.make_lit_vertex(c_tip + c_off, to_cam, 1.0, 0.0, core_col, 1.0, 1.0, 0.0, emissive * 1.25, unlit=True, no_fog=True)
        var cc = self.make_lit_vertex(nozzle + c_off, to_cam, 1.0, 0.90, core_col, 1.0, 1.0, 0.0, emissive * 1.35, unlit=True, no_fog=True)
        var cd = self.make_lit_vertex(nozzle - c_off, to_cam, 0.0, 0.90, core_col, 1.0, 1.0, 0.0, emissive * 1.35, unlit=True, no_fog=True)
        self.fb.draw_triangle(ca, cb, cc, self.fx.pixels[FX_FLAME], sz, sz, alpha=alpha, double_sided=True)
        self.fb.draw_triangle(ca, cc, cd, self.fx.pixels[FX_FLAME], sz, sz, alpha=alpha, double_sided=True)

        # Mach shock diamonds along exhaust axis
        for d in range(3):
            var frac = Float32(0.14) + Float32(d) * 0.22
            var d_pos = nozzle + dir_n * (length * frac)
            var d_rad = radius * (0.72 - Float32(d) * 0.16)
            var d_alpha = UInt8(max(30, Int(alpha) - d * 55))
            self.draw_billboard(d_pos, d_rad, FX_SOFT, core_color, d_alpha, emissive * 1.3)

    def draw_ribbon_3d(
        mut self,
        points: List[Vec3],
        head_width: Float32,
        tail_width: Float32,
        head_color: UInt32,
        tail_color: UInt32,
        head_alpha: UInt8,
        tail_alpha: UInt8 = 12,
        emissive: Float32 = 0.45,
    ):
        """Camera-facing volumetric TEX_RIBBON strip from points[0] (newest/head) to points[n-1] (tail)."""
        var n = len(points)
        if n < 2 or head_alpha == 0:
            return
        var sz = self.fx.sizes[FX_RIBBON]
        var hr = Float32((head_color >> 16) & 255)
        var hg = Float32((head_color >> 8) & 255)
        var hb = Float32(head_color & 255)
        var tr = Float32((tail_color >> 16) & 255)
        var tg = Float32((tail_color >> 8) & 255)
        var tb = Float32(tail_color & 255)

        for i in range(n - 1):
            var p0 = points[i]
            var p1 = points[i + 1]
            var seg = p1 - p0
            if seg.length_squared() < 0.0001:
                continue
            var t0 = Float32(i) / Float32(max(1, n - 1))
            var t1 = Float32(i + 1) / Float32(max(1, n - 1))
            var w0 = head_width + (tail_width - head_width) * t0
            var w1 = head_width + (tail_width - head_width) * t1

            var dir0 = seg.normalized()
            var dir1 = (points[i + 2] - p1).normalized() if i + 2 < n and (points[i + 2] - p1).length_squared() > 0.0001 else dir0
            var cam0 = (self.fb.camera - p0).normalized()
            var cam1 = (self.fb.camera - p1).normalized()
            var side0 = cross(dir0, cam0)
            var side1 = cross(dir1, cam1)
            side0 = side0.normalized() if side0.length_squared() > 0.0001 else self.fb.up
            side1 = side1.normalized() if side1.length_squared() > 0.0001 else self.fb.up

            var c0 = 0xFF000000 | (UInt32(Int(hr + (tr - hr) * t0)) << 16) | (UInt32(Int(hg + (tg - hg) * t0)) << 8) | UInt32(Int(hb + (tb - hb) * t0))
            var c1 = 0xFF000000 | (UInt32(Int(hr + (tr - hr) * t1)) << 16) | (UInt32(Int(hg + (tg - hg) * t1)) << 8) | UInt32(Int(hb + (tb - hb) * t1))
            var em0 = emissive * (1.0 - t0 * 0.72)
            var em1 = emissive * (1.0 - t1 * 0.72)
            var a_seg = UInt8(max(0, min(255, Int(Float32(head_alpha) + Float32(Int(tail_alpha) - Int(head_alpha)) * ((t0 + t1) * 0.5)))))
            if a_seg == 0:
                continue

            var v0 = self.make_lit_vertex(p0 + side0 * w0, cam0, t0, 0.0, c0, 1.0, 1.0, 0.0, em0, unlit=True)
            var v1 = self.make_lit_vertex(p1 + side1 * w1, cam1, t1, 0.0, c1, 1.0, 1.0, 0.0, em1, unlit=True)
            var v2 = self.make_lit_vertex(p1 - side1 * w1, cam1, t1, 1.0, c1, 1.0, 1.0, 0.0, em1, unlit=True)
            var v3 = self.make_lit_vertex(p0 - side0 * w0, cam0, t0, 1.0, c0, 1.0, 1.0, 0.0, em0, unlit=True)
            self.fb.draw_triangle(v0, v1, v2, self.fx.pixels[FX_RIBBON], sz, sz, alpha=a_seg, double_sided=True)
            self.fb.draw_triangle(v0, v2, v3, self.fx.pixels[FX_RIBBON], sz, sz, alpha=a_seg, double_sided=True)

    def draw_colossal_beam_3d(
        mut self,
        start: Vec3,
        end: Vec3,
        radius: Float32,
        core_color: UInt32,
        sheath_color: UInt32,
        scroll: Float32 = 0.0,
        alpha: UInt8 = 245,
        emissive: Float32 = 3.2,
    ):
        """Capital-ship mega-particle cannon beam: scrolling plasma sheath + blinding core + shockwave flares."""
        if radius <= 0.01 or alpha == 0:
            return
        var axis = end - start
        var beam_len = axis.length()
        if beam_len < 0.05:
            return
        var dir_v = axis * (1.0 / beam_len)
        var repeats = max(Float32(2.0), beam_len / max(Float32(1.0), radius * 6.0))
        var u0 = -scroll
        var u1 = repeats - scroll

        # Outer wide atmospheric/plasma glow
        self.draw_beam_3d(
            start,
            end,
            radius * 2.15,
            FX_STRIPES,
            sheath_color,
            UInt8(Int(alpha) * 5 // 8),
            emissive=emissive * 0.78,
            u0=u0,
            u1=u1,
        )
        # Middle high-energy particle sheath
        self.draw_beam_3d(
            start,
            end,
            radius * 1.18,
            FX_BOLT,
            sheath_color,
            UInt8(Int(alpha) * 7 // 8),
            emissive=emissive,
            u0=u0 * 1.4,
            u1=u1 * 1.4,
        )
        # White-hot inner core
        self.draw_beam_3d(
            start,
            end,
            radius * 0.56,
            FX_BOLT,
            core_color,
            alpha,
            emissive=emissive * 1.35,
        )
        # Muzzle & terminal plasma flares + pulsing energy nodes along beam
        self.draw_billboard(start, radius * 3.4, FX_SOFT, sheath_color, UInt8(Int(alpha) * 3 // 4), emissive=emissive)
        self.draw_billboard(start, radius * 1.9, FX_SOFT, core_color, alpha, emissive=emissive * 1.4)
        self.draw_billboard(end, radius * 3.8, FX_SOFT, sheath_color, UInt8(Int(alpha) * 3 // 4), emissive=emissive)
        self.draw_billboard(end, radius * 2.1, FX_SOFT, core_color, alpha, emissive=emissive * 1.4)
        for k in range(3):
            var frac = Float32(((Float32(k) * 0.33 + scroll * 0.18) - Float32(Int(Float32(k) * 0.33 + scroll * 0.18))))
            var node_p = start + dir_v * (beam_len * max(Float32(0.08), min(Float32(0.92), frac)))
            self.draw_billboard(node_p, radius * 1.85, FX_SOFT, core_color, UInt8(Int(alpha) * 2 // 3), emissive=emissive * 1.15)

    def draw_cloud_deck(
        mut self,
        center: Vec3,
        altitude: Float32,
        span: Float32,
        count: Int,
        sunlit_color: UInt32 = 0xFFF8FAFF,
        shadow_color: UInt32 = 0xFF7D95B8,
        alpha: UInt8 = 168,
    ):
        """Volumetric cumulus cloud deck tiled deterministically around `center` for atmospheric flight."""
        if count <= 0 or span <= 1.0 or alpha == 0:
            return
        var half = span * 0.5
        var cell = max(Float32(20.0), span)
        var snap_x = Float32(Int(center.x / 64.0)) * 64.0
        var snap_z = Float32(Int(center.z / 64.0)) * 64.0
        for i in range(count):
            var h1 = UInt32(i * 747796405 + 2891336453)
            var h2 = UInt32(i * 1597334677 + 3812015801)
            var h3 = UInt32(i * 2246822519 + 3266489917)
            var ox = (Float32(h1 & 0xFFFF) / 65535.0) * cell - half
            var oz = (Float32(h2 & 0xFFFF) / 65535.0) * cell - half
            var oy = ((Float32(h3 & 0xFF) / 255.0) - 0.5) * 14.0
            var wx = snap_x + ox
            var wz = snap_z + oz
            # Wrap into [-half, +half] around camera center
            while wx - center.x > half:
                wx -= cell
            while wx - center.x < -half:
                wx += cell
            while wz - center.z > half:
                wz -= cell
            while wz - center.z < -half:
                wz += cell
            var base_r = Float32(13.0) + Float32((h3 >> 8) & 0xFF) / 255.0 * Float32(15.0)
            var c_pos = Vec3(wx, altitude + oy, wz)
            var dist_cam = (c_pos - self.fb.camera).length()
            if dist_cam < 10.0:
                continue
            var fade = min(Float32(1.0), max(Float32(0.0), (half - max(abs(wx - center.x), abs(wz - center.z))) / (half * 0.35)))
            var eff_a = UInt8(max(0, min(255, Int(Float32(alpha) * fade))))
            if eff_a < 8:
                continue
            var sh_a = UInt8(Int(eff_a) * 3 // 4)
            # Cool shadow underbelly + sunlit cumulus crown puffs
            self.draw_billboard(c_pos - Vec3(0.0, base_r * 0.22, 0.0), base_r * 1.15, FX_SOFT, shadow_color, sh_a, emissive=0.05)
            self.draw_billboard(c_pos + Vec3(-base_r * 0.28, base_r * 0.14, 0.0), base_r * 0.92, FX_SOFT, sunlit_color, eff_a, emissive=0.22)
            self.draw_billboard(c_pos + Vec3(base_r * 0.30, base_r * 0.18, base_r * 0.12), base_r * 0.84, FX_SOFT, sunlit_color, eff_a, emissive=0.24)

    def end_scene(mut self):
        """Complete 3D scene pass so 2D HUD widgets drawn afterward are not blurred by CPU bloom."""
        if self.scene_ended:
            return
        self.scene_ended = True
        if not self.fb.gpu.active:
            self.fb.apply_post(self.active_profile, self.active_quality, self.active_reduced)

    def present_headless(mut self) -> Bool:
        """Submit GPU passes (raster + bloom + smooth + HUD compose) and read back finished pixels."""
        if not self.scene_ended:
            self.end_scene()
        if self.fb.gpu.active:
            if self.fb.submit_gpu_frame():
                return self.fb.read_back_gpu()
            return False
        return True

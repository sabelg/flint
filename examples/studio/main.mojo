"""Flint v5 Visual Editor (`Flint Studio`, `examples/studio/main.mojo`).

Standalone AAA Visual Scene Editor, Procedural Kit & AI-Native MCP v2 Host:
  - **Interactive Window Mode** (`WaylandWindow` + `X11Window` fallback + `FlintAaaVulkanRenderer`):
    - $O(1)$ 64-bit `VisBuffer` viewport entity picking (`VisBufferGpuPicker`)
    - Analytical 3D Translate / Rotate (`< 0.15` grazing fallback) / Scale gizmos (`InputMap` `W` / `E` / `R`, grid/15° snap)
    - `DockSpace` + `UiContext` Scene Hierarchy (`tree_node`), `FieldVisitor` Inspector (`drag_float3` / `slider_float`),
      `CameraRig` orbit/fly viewport camera, and 1-Click LZ4 PIE (`F5`)
    - Reversible coalesced `EditorTransaction` Undo/Redo (`Ctrl+Z` / `Ctrl+Y`) & hardened `FlintMcpUnixListener` socket
  - **`--headless-smoke` / `--shots` (`--out <png>`)**:
    Renders a crisp `1280x720` Studio editor screenshot featuring engine-neutral procedural Studio primitives
    (`Studio_ReferenceGrid`, `SubD_MaterialOrb`, `SubD_CatmullTorus`, `ArchKit_Colonnade`, `Botanical_FoliageSample`)
    with `DockSpace` + `UiContext` panels and 3D Gizmo overlay.
  - **`--mcp-stdio`**:
    Runs `McpServerV2` in spec-compliant Model Context Protocol (`2024-11-05`) JSON-RPC 2.0 `stdio` mode.
"""
from std.math import abs, cos, max, min, sin
from std.sys import argv
from flint.anim.flanim_ecs import compute_volume_preserving_smear_scale
from flint.assets.image import write_png
from flint.assets.server import AssetServer
from flint.core.jobs import JobPool, default_worker_count
from flint.core.math import Quat, Vec3
from flint.core.transform import trs
from flint.devtools.mcp_v2 import FlintMcpUnixListener, McpServerV2, default_mcp_socket_path
from flint.platform.input import InputState
from flint.platform.os import get_time, make_dirs, parent_directory
from flint.platform.wayland import WaylandWindow
from flint.platform.x11 import X11Window
from flint.procedural.arch_kit import build_colonnade_arcade_bay
from flint.procedural.foliage import build_botanical_pine
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_clustered_lights import AaaClusterLight
from flint.render.aaa_npr_cel import AaaCelParams
from flint.render.aaa_npr_vfx import AaaImpactFrameParams, AaaShockwaveRing, AaaStylizedVfxParams
from flint.render.aaa_post_tsr import AAA_DISPLAY_AGX_REC709
from flint.render.aaa_renderer import FlintAaaVulkanRenderer
from flint.render.aaa_sky_atmosphere import BrunetonAtmosphereParams
from flint.studio import (
    COLLIDER_BOX,
    COLLIDER_SPHERE,
    GIZMO_AXIS_NONE,
    GIZMO_AXIS_X,
    GIZMO_AXIS_Y,
    GIZMO_AXIS_Z,
    GIZMO_MODE_ROTATE,
    GIZMO_MODE_SCALE,
    GIZMO_MODE_TRANSLATE,
    GIZMO_PLANE_XY,
    GIZMO_PLANE_XZ,
    GIZMO_PLANE_YZ,
    PIE_PLAYING,
    SHADING_NPR_CEL,
    SHADING_PBR,
    SHADING_STYLIZED_VFX,
    FlintStudioState,
    ViewportRay,
    VisBufferGpuPicker,
    build_studio_reference_grid_mesh,
    build_subd_catmull_torus,
    build_subd_material_orb,
    emit_studio_ui_draw_list,
    gizmo_rotation_delta_rad,
    gizmo_scale_drag,
    gizmo_world_scale,
    pick_gizmo_handle,
    ray_axis_closest_param,
    ray_plane_hit,
    snap_angle_deg,
    snap_translation,
)


def _populate_default_studio_scene(mut studio: FlintStudioState):
    """Populates the engine-neutral procedural SubD + ArchKit + Botanical Studio showcase into `FlintStudioState`."""
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
    _ = studio.spawn_entity(
        "Botanical_FoliageSample",
        position=Vec3(-4.6, 0.0, 2.4),
        scale=Vec3(0.92, 0.92, 0.92),
        collider_kind=COLLIDER_BOX,
        collider_half_extents=Vec3(0.9, 2.6, 0.9),
        mass=0.0,
        base_color=Vec3(0.20, 0.48, 0.24),
        roughness=0.72,
        metallic=0.0,
        mesh_id=4,
        record_undo=False,
    )
    studio.selected_entity_id = orb_id
    studio.look_at(Vec3(-5.6, 3.75, -7.4), Vec3(0.15, 0.95, 1.1))


struct _StudioGpuMeshes(Movable):
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


def _upload_studio_gpu_assets(mut ren: FlintAaaVulkanRenderer, studio: FlintStudioState) -> _StudioGpuMeshes:
    var gm = _StudioGpuMeshes()

    # Reserve 1 material per entity slot [0..4] + shared botanical trunk/foliage materials
    for i in range(len(studio.entities)):
        ref e = studio.entities[i]
        var mat = AaaMaterial(base_color=e.base_color, roughness=e.roughness, metallic=e.metallic)
        mat.emissive = e.emissive
        if i == 1:
            mat.clearcoat = Float32(0.88)
            mat.clearcoat_roughness = Float32(0.05)
        gm.entity_pbr_mat_ids.append(ren.add_material(mat))
        gm.entity_cel_mat_ids.append(-1)
        gm.entity_vfx_mat_ids.append(-1)

    var mat_trunk = AaaMaterial(base_color=Vec3(0.34, 0.23, 0.14), roughness=0.84, metallic=0.0)
    gm.mat_trunk_id = ren.add_material(mat_trunk)

    var mat_fol = AaaMaterial(base_color=Vec3(0.18, 0.46, 0.22), roughness=0.68, metallic=0.0)
    gm.mat_foliage_id = ren.add_material(mat_fol)

    gm.grid_mesh_id = ren.add_mesh(build_studio_reference_grid_mesh(36.0, 24), 0, build_lods=False).mesh_id
    gm.orb_mesh_id = ren.add_mesh(build_subd_material_orb(0.88, 3), 1, build_lods=True).mesh_id
    gm.torus_mesh_id = ren.add_mesh(build_subd_catmull_torus(0.82, 0.30, 2), 2, build_lods=True).mesh_id
    gm.colonnade_mesh_id = ren.add_mesh(build_colonnade_arcade_bay(6.0, 4.8, 2.4), 3, build_lods=True).mesh_id
    var pn = build_botanical_pine(7, 5.4)
    gm.foliage_trunk_id = ren.add_mesh(pn[0], gm.mat_trunk_id, build_lods=False).mesh_id
    gm.foliage_canopy_id = ren.add_mesh(pn[1], gm.mat_foliage_id, build_lods=False).mesh_id

    var atm = BrunetonAtmosphereParams()
    atm.sun_dir = Vec3(0.46, 0.58, -0.67).normalized()
    atm.cloud_coverage = Float32(0.28)
    _ = ren.set_bruneton_atmosphere(atm, enable_aerial=True)

    ren.display_mode = AAA_DISPLAY_AGX_REC709
    ren.enable_vsm = True
    ren.vsm_scene_center = Vec3(0.0, 1.5, 2.0)
    ren.vsm_ortho_half_extent = Float32(24.0)
    ren.enable_rt_shadows = ren.vk.has_ray_query
    ren.enable_rt_reflections = ren.vk.has_ray_query
    ren.enable_rt_gi = ren.vk.has_ray_query
    ren.gi_rays_per_pixel = 6
    ren.svgf_passes = 1
    ren.exposure = Float32(1.08)
    ren.bloom_intensity = Float32(0.08)
    return gm^


def _submit_studio_frame_draws(
    mut ren: FlintAaaVulkanRenderer,
    studio: FlintStudioState,
    mut gm: _StudioGpuMeshes,
    mut out_inst_to_entity: List[Int],
):
    out_inst_to_entity.clear()
    ren.add_cluster_light(AaaClusterLight.point(Vec3(-2.8, 3.8, -2.5), Vec3(18.0, 14.0, 10.0), Float32(14.0)))
    ren.add_cluster_light(AaaClusterLight.point(Vec3(4.8, 4.2, 0.5), Vec3(8.0, 14.0, 24.0), Float32(14.0)))

    if studio.preview_shockwave_active:
        ren.submit_shockwave(
            AaaShockwaveRing(
                center_u=Float32(0.5),
                center_v=Float32(0.5),
                radius_uv=studio.preview_shockwave_radius,
            )
        )
    if studio.preview_impact_blend > Float32(0.0):
        ren.set_impact_frame(
            AaaImpactFrameParams(
                blend=studio.preview_impact_blend,
                invert_polarity=studio.preview_impact_invert,
            )
        )
    else:
        ren.clear_impact_frame()

    while len(gm.entity_pbr_mat_ids) < len(studio.entities):
        gm.entity_pbr_mat_ids.append(-1)
        gm.entity_cel_mat_ids.append(-1)
        gm.entity_vfx_mat_ids.append(-1)

    for i in range(len(studio.entities)):
        ref e = studio.entities[i]
        if not e.alive or not e.visible:
            continue

        var mat = AaaMaterial(base_color=e.base_color, roughness=e.roughness, metallic=e.metallic)
        mat.emissive = e.emissive
        if i == 1 and e.shading_mode == SHADING_PBR:
            mat.clearcoat = Float32(0.88)
            mat.clearcoat_roughness = Float32(0.05)

        var mat_ov: Int
        if e.shading_mode == SHADING_NPR_CEL:
            var cel = AaaCelParams(
                sss_shadow_tint=e.cel_shadow_tint,
                rim_color=e.cel_rim_color,
                hair_halo_strength=e.cel_hair_halo,
                outline_width_px=e.cel_outline_width_px,
                outline_color=e.cel_outline_color,
            )
            if gm.entity_cel_mat_ids[i] < 0:
                gm.entity_cel_mat_ids[i] = ren.add_cel_material(mat, cel)
            else:
                ren.update_material(gm.entity_cel_mat_ids[i], mat)
                _ = ren.update_cel_material(gm.entity_cel_mat_ids[i], cel)
            mat_ov = gm.entity_cel_mat_ids[i] if gm.entity_cel_mat_ids[i] >= 0 else 0
        elif e.shading_mode == SHADING_STYLIZED_VFX:
            var vfx = AaaStylizedVfxParams.anime_flame(
                core_color=e.vfx_core_color,
                edge_color=e.vfx_edge_color,
                erosion=e.vfx_erosion,
                scroll_speed=e.vfx_scroll_speed,
            )
            if gm.entity_vfx_mat_ids[i] < 0:
                gm.entity_vfx_mat_ids[i] = ren.add_vfx_material(mat, vfx)
            else:
                var vfx_mat = mat.copy()
                vfx_mat.alpha_cutoff = Float32(0.48)
                ren.update_material(gm.entity_vfx_mat_ids[i], vfx_mat)
                _ = ren.update_vfx_material(gm.entity_vfx_mat_ids[i], vfx)
            mat_ov = gm.entity_vfx_mat_ids[i] if gm.entity_vfx_mat_ids[i] >= 0 else 0
        else:
            if gm.entity_pbr_mat_ids[i] < 0:
                gm.entity_pbr_mat_ids[i] = ren.add_material(mat)
            else:
                ren.update_material(gm.entity_pbr_mat_ids[i], mat)
            mat_ov = gm.entity_pbr_mat_ids[i] if gm.entity_pbr_mat_ids[i] >= 0 else 0

        var wm = studio.world_matrix(i)
        var us = max(e.scale.x, max(e.scale.y, e.scale.z))
        if e.smear_stretch != Float32(1.0):
            var sm = compute_volume_preserving_smear_scale(e.smear_stretch)
            wm = wm.multiply(trs(Vec3(0.0, 0.0, 0.0), Quat(0.0, 0.0, 0.0, 1.0), sm))
            us = max(e.scale.x * sm.x, max(e.scale.y * sm.y, e.scale.z * sm.z))

        if e.mesh_id == 0 and i == 0:
            ren.draw_instance(gm.grid_mesh_id, wm, material_override=(0 if e.shading_mode == SHADING_PBR else mat_ov), uniform_scale=us)
            out_inst_to_entity.append(i)
        elif e.mesh_id == 1:
            ren.draw_instance(gm.orb_mesh_id, wm, material_override=mat_ov, uniform_scale=us)
            out_inst_to_entity.append(i)
        elif e.mesh_id == 2:
            ren.draw_instance(gm.torus_mesh_id, wm, material_override=mat_ov, uniform_scale=us)
            out_inst_to_entity.append(i)
        elif e.mesh_id == 3:
            ren.draw_instance(gm.colonnade_mesh_id, wm, material_override=mat_ov, uniform_scale=us)
            out_inst_to_entity.append(i)
        elif e.mesh_id == 4:
            var trunk_ov = gm.mat_trunk_id if e.shading_mode == SHADING_PBR else mat_ov
            var fol_ov = gm.mat_foliage_id if e.shading_mode == SHADING_PBR else mat_ov
            ren.draw_instance(gm.foliage_trunk_id, wm, material_override=trunk_ov, uniform_scale=us)
            out_inst_to_entity.append(i)
            ren.draw_instance(gm.foliage_canopy_id, wm, material_override=fol_ov, uniform_scale=us)
            out_inst_to_entity.append(i)
        else:
            ren.draw_instance(gm.orb_mesh_id, wm, material_override=mat_ov, uniform_scale=us)
            out_inst_to_entity.append(i)


def main() raises:
    var mcp_stdio = False
    var headless_smoke = False
    var shots_mode = False
    var out_png = String("")

    var args = argv()
    var ai = 1
    while ai < len(args):
        var a = String(args[ai])
        if a == "--mcp-stdio":
            mcp_stdio = True
        elif a == "--headless-smoke":
            headless_smoke = True
        elif a == "--shots":
            shots_mode = True
        elif a == "--out" and ai + 1 < len(args):
            out_png = String(args[ai + 1])
            ai += 1
        ai += 1

    var pool = JobPool()
    _ = pool.start(max(1, default_worker_count()))
    var assets = AssetServer(enable_watcher=not (mcp_stdio or headless_smoke or shots_mode))
    var studio = FlintStudioState()
    _populate_default_studio_scene(studio)
    var mcp = McpServerV2(workspace_root=".")

    if mcp_stdio:
        _ = mcp.serve_stdio_loop(studio, assets, pool)
        assets.close()
        pool.stop()
        return

    var out_w = 1280
    var out_h = 720
    var ren = FlintAaaVulkanRenderer(width=out_w, height=out_h, tsr_scale=1.0, env_cube_dim=128)
    if not ren.active:
        print("[Flint Studio] ERROR: Vulkan 1.3 driver unavailable.")
        assets.close()
        pool.stop()
        return

    var gm = _upload_studio_gpu_assets(ren, studio)
    var inst_to_ent = List[Int]()
    var picker = VisBufferGpuPicker()

    if headless_smoke or shots_mode:
        var target_png = out_png
        if target_png.byte_length() == 0:
            target_png = "/tmp/flint_studio_smoke.png" if headless_smoke else "build/shots/flint_studio.png"
        var pdir = parent_directory(target_png)
        if pdir.byte_length() > 0 and pdir != ".":
            _ = make_dirs(pdir)

        var empty_input = InputState()
        for sub in range(4):
            studio.update_camera(Float32(0.0))
            var cam_target = studio.cam_pos + studio.cam_fwd * Float32(10.0)
            ren.tsr_history_weight = Float32(0.0) if sub == 0 else Float32(0.78)
            ren.begin_frame(studio.cam_pos, cam_target, fov_y_deg=studio.fov_y_deg, near_z=studio.near_z)
            _submit_studio_frame_draws(ren, studio, gm, inst_to_ent)
            ren.ui_draw_list.clear()
            emit_studio_ui_draw_list(
                ren.ui_draw_list,
                studio,
                ren.ui_font_atlas_slot,
                ren.ui_font,
                out_w,
                out_h,
                gpu_ms=ren.render_graph.total_gpu_ms,
                mounted_packs=len(assets.mounts),
                mcp_clients=0,
                input=empty_input,
                dt=Float32(1.0 / 60.0),
            )
            _ = ren.render_frame(capture_readback=(sub == 3))

        var px = ren.readback_pixels()
        _ = write_png(target_png, px, out_w, out_h)
        print(
            "[Flint Studio] Saved 1280x720 Studio frame -> "
            + target_png
            + " (UIPrims=" + String(ren.ui_draw_list.count())
            + ", Entities=" + String(studio.live_entity_count())
            + ", CulledDraws=" + String(ren.last_culled_meshlet_draws) + ")"
        )
        picker.shutdown(ren)
        ren.shutdown()
        assets.close()
        pool.stop()
        return

    # Interactive Window Mode
    var wl_win = WaylandWindow()
    var x11_win = X11Window()
    var use_wayland = wl_win.initialize(out_w, out_h)
    var use_x11 = False
    if not use_wayland:
        use_x11 = x11_win.initialize(out_w, out_h, "FLINT STUDIO v5 — AAA VISUAL EDITOR")
    if not use_wayland and not use_x11:
        print("[Flint Studio] No Wayland/X11 display detected. Use '--headless-smoke' or '--shots'.")
        picker.shutdown(ren)
        ren.shutdown()
        assets.close()
        pool.stop()
        return

    var mcp_sock = FlintMcpUnixListener(default_mcp_socket_path())
    var input = InputState()
    var dragging_gizmo = False
    var drag_start_ray = ViewportRay()
    var drag_start_pos = Vec3()
    var drag_start_rot = Quat()
    var drag_start_scl = Vec3(1.0, 1.0, 1.0)
    var drag_seq = UInt64(100)
    var t_last = get_time()

    while True:
        if use_wayland and wl_win.close_requested:
            break
        if use_x11 and not x11_win.active():
            break

        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - t_last)))
        t_last = now

        input.begin_frame()
        if use_wayland:
            wl_win.poll(input)
        else:
            x11_win.poll(input)

        # Process declarative `InputMap` hotkeys (`W/E/R`, `Ctrl+Z`, `Ctrl+Y`, `F5` PIE) & `CameraRig` orbit/fly
        if not studio.process_editor_input(input, dt):
            break

        # Query live `DockSpace` central viewport rectangle (supports interactive splitter dragging!)
        var vp_rect = studio.viewport_rect(out_w, out_h)
        var vp_x = vp_rect.x
        var vp_y = vp_rect.y
        var vp_w = vp_rect.w
        var vp_h = vp_rect.h

        # Viewport picking & analytical gizmo dragging
        var mx = input.last_mouse_x
        var my = input.last_mouse_y
        var in_vp = vp_rect.contains(mx, my)

        if input.mouse_pressed(0) and in_vp and studio.dock.dragging_node < 0:
            var g_hit = GIZMO_AXIS_NONE
            if studio.selected_entity_id >= 0 and studio.entities[studio.selected_entity_id].alive:
                g_hit = pick_gizmo_handle(
                    mx, my, studio.entities[studio.selected_entity_id].translation,
                    studio.gizmo_mode, vp_x, vp_y, vp_w, vp_h,
                    studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
                )
            if g_hit != GIZMO_AXIS_NONE:
                dragging_gizmo = True
                studio.active_gizmo_handle = g_hit
                drag_seq += 1
                drag_start_ray = ViewportRay.from_screen(
                    mx, my, vp_x, vp_y, vp_w, vp_h,
                    studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
                )
                drag_start_pos = studio.entities[studio.selected_entity_id].translation
                drag_start_rot = studio.entities[studio.selected_entity_id].rotation
                drag_start_scl = studio.entities[studio.selected_entity_id].scale
            else:
                var pick_res = picker.pick_viewport(ren, Int(mx), Int(my), inst_to_ent)
                if pick_res.hit and pick_res.entity_id >= 0 and pick_res.entity_id < len(studio.entities):
                    studio.selected_entity_id = pick_res.entity_id

        if dragging_gizmo and input.mouse_down(0) and studio.selected_entity_id >= 0:
            var cur_ray = ViewportRay.from_screen(
                mx, my, vp_x, vp_y, vp_w, vp_h,
                studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
            )
            var eid = studio.selected_entity_id
            var h_id = studio.active_gizmo_handle
            var wscale = gizmo_world_scale(drag_start_pos, studio.cam_pos, studio.cam_fwd, studio.fov_y_deg, Int(vp_h))
            if studio.gizmo_mode == GIZMO_MODE_TRANSLATE:
                var new_p = drag_start_pos
                if h_id == GIZMO_AXIS_X or h_id == GIZMO_AXIS_Y or h_id == GIZMO_AXIS_Z:
                    var ax = Vec3(1, 0, 0) if h_id == GIZMO_AXIS_X else (Vec3(0, 1, 0) if h_id == GIZMO_AXIS_Y else Vec3(0, 0, 1))
                    var t0 = ray_axis_closest_param(drag_start_ray, drag_start_pos, ax)
                    var t1 = ray_axis_closest_param(cur_ray, drag_start_pos, ax)
                    if t0[0] and t1[0]:
                        new_p = drag_start_pos + ax * (t1[1] - t0[1])
                elif h_id == GIZMO_PLANE_XY or h_id == GIZMO_PLANE_XZ or h_id == GIZMO_PLANE_YZ:
                    var pn = Vec3(0, 0, 1) if h_id == GIZMO_PLANE_XY else (Vec3(0, 1, 0) if h_id == GIZMO_PLANE_XZ else Vec3(1, 0, 0))
                    var p0 = ray_plane_hit(drag_start_ray, drag_start_pos, pn)
                    var p1 = ray_plane_hit(cur_ray, drag_start_pos, pn)
                    if p0[0] and p1[0]:
                        new_p = drag_start_pos + (p1[1] - p0[1])
                if studio.snap_enabled:
                    new_p = snap_translation(new_p, studio.grid_snap_m)
                _ = studio.set_entity_transform(eid, new_p, drag_start_rot, drag_start_scl, coalesce_key=drag_seq)
            elif studio.gizmo_mode == GIZMO_MODE_ROTATE:
                var rax = Vec3(1, 0, 0) if h_id == GIZMO_AXIS_X else (Vec3(0, 1, 0) if h_id == GIZMO_AXIS_Y else Vec3(0, 0, 1))
                var drad = gizmo_rotation_delta_rad(drag_start_ray, cur_ray, drag_start_pos, rax, studio.cam_fwd, wscale)
                if studio.snap_enabled:
                    drad = snap_angle_deg(drad, studio.angle_snap_deg)
                var dq = Quat(rax.x * sin(drad * Float32(0.5)), rax.y * sin(drad * Float32(0.5)), rax.z * sin(drad * Float32(0.5)), cos(drad * Float32(0.5)))
                _ = studio.set_entity_transform(eid, drag_start_pos, dq.multiply(drag_start_rot).normalized(), drag_start_scl, coalesce_key=drag_seq)
            elif studio.gizmo_mode == GIZMO_MODE_SCALE:
                var ns = gizmo_scale_drag(drag_start_ray, cur_ray, drag_start_pos, h_id, wscale, studio.cam_fwd, studio.cam_right, drag_start_scl)
                _ = studio.set_entity_transform(eid, drag_start_pos, drag_start_rot, ns, coalesce_key=drag_seq)

        if not input.mouse_down(0):
            dragging_gizmo = False
            studio.active_gizmo_handle = GIZMO_AXIS_NONE

        if studio.pie.mode == PIE_PLAYING:
            _ = studio.pie_step(1)

        _ = assets.tick(now)
        _ = mcp_sock.poll_once_with_renderer(mcp, studio, assets, pool, ren)

        ren.begin_frame(studio.cam_pos, studio.cam_pos + studio.cam_fwd * Float32(10.0), fov_y_deg=studio.fov_y_deg, near_z=studio.near_z)
        _submit_studio_frame_draws(ren, studio, gm, inst_to_ent)
        ren.ui_draw_list.clear()
        emit_studio_ui_draw_list(
            ren.ui_draw_list,
            studio,
            ren.ui_font_atlas_slot,
            ren.ui_font,
            out_w,
            out_h,
            gpu_ms=ren.render_graph.total_gpu_ms,
            mounted_packs=len(assets.mounts),
            mcp_clients=mcp_sock.client_count(),
            input=input,
            dt=dt,
        )
        _ = ren.render_frame(capture_readback=True)
        var frame_px = ren.readback_pixels()
        if use_wayland:
            wl_win.present(frame_px, out_w, out_h)
        else:
            x11_win.present(frame_px, out_w, out_h)

    mcp_sock.close()
    if use_wayland:
        wl_win.close()
    if use_x11:
        x11_win.close()
    picker.shutdown(ren)
    ren.shutdown()
    assets.close()
    pool.stop()

"""Flint v5 Visual Editor (`Flint Studio`, `examples/studio/main.mojo`).

Standalone AAA Visual Scene Editor, Procedural Kit & AI-Native MCP v2 Host powered by
the unified Engine v5 spine (`EngineApp`, `WindowPlugin`, `TransformPlugin`, `InputPlugin`,
`PhysicsPlugin`, `RenderPlugin`, `flint.ecs.World`, `flint.scene.Scene`, `PhysicsWorld`,
`ScalableRenderer`):
  - **Interactive Window Mode**:
    - Non-stalling `FRAMES_IN_FLIGHT` async 40-bit `VisBuffer` viewport entity picking (`VisBufferGpuPicker`)
    - Multi-select (`Ctrl`/`Shift`), centroid analytical 3D gizmos (`W` / `E` / `R`, grid/15° snap)
    - `DockSpace` + `UiContext` collapsible indented hierarchy tree with drag-and-drop reparenting,
      single/bulk `FieldVisitor` Inspector, `CameraRig` orbit/fly, and 1-Click LZ4 PIE (`F5`)
    - Reversible coalesced `EditorTransaction` Undo/Redo (`Ctrl+Z` / `Ctrl+Y`) & `FlintMcpUnixListener`
  - **`--headless-smoke` / `--shots` (`--out <png>`)**:
    Renders a crisp `1280x720` Studio editor frame (`Studio_ReferenceGrid`, `SubD_MaterialOrb`,
    `SubD_CatmullTorus`, `ArchKit_Colonnade`, `Botanical_FoliageSample` + `Botanical_FoliageCanopy`)
    with `DockSpace` + `UiContext` panels and 3D Gizmo overlay.
  - **`--mcp-stdio`**:
    Runs `McpServerV2` in spec-compliant Model Context Protocol (`2024-11-05`) JSON-RPC 2.0 `stdio` mode.
"""
from std.math import cos, max, min, sin
from std.sys import argv
from flint.app.engine_app import STAGE_FIRST
from flint.app.window_plugin import WindowStateResource
from flint.assets.image import write_png
from flint.assets.server import AssetServer
from flint.core.jobs import JobPool, default_worker_count
from flint.core.math import Quat, Vec3
from flint.devtools.mcp_v2 import FlintMcpUnixListener, McpServerV2, default_mcp_socket_path
from flint.ecs.world import Entity
from flint.input.plugin import InputStateResource
from flint.platform.input import InputState
from flint.platform.os import get_time, make_dirs, parent_directory
from flint.procedural.arch_kit import build_colonnade_arcade_bay
from flint.procedural.foliage import build_botanical_pine
from flint.render.aaa_post_tsr import AAA_DISPLAY_AGX_REC709
from flint.render.aaa_sky_atmosphere import BrunetonAtmosphereParams
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import MaterialOverrideComponent, MeshRendererComponent
from flint.render.render_world import (
    RENDER_TIER_0_HEADLESS_SIMD,
    RENDER_TIER_2_AAA_VISBUFFER_RT,
    ScalableRenderer,
)
from flint.studio import (
    COLLIDER_BOX,
    COLLIDER_NONE,
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
    GIZMO_SCALE_UNIFORM,
    PIE_PLAYING,
    FlintStudioState,
    SelectedTransformSnap,
    StudioStyleComponent,
    ViewportRay,
    VisBufferGpuPicker,
    build_studio_reference_grid_mesh,
    build_subd_catmull_torus,
    build_subd_material_orb,
    fill_instance_entity_maps,
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


def _upload_studio_gpu_assets(mut studio: FlintStudioState):
    var rend_opt = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    if not rend_opt:
        return
    ref renderer = rend_opt.value()[unsafe_offset=0]
    if not renderer.uses_vulkan():
        return

    var n_ents = len(studio.app.world.entities)
    studio.gpu_materials.entity_pbr_mat_ids.clear()
    studio.gpu_materials.entity_cel_mat_ids.clear()
    studio.gpu_materials.entity_vfx_mat_ids.clear()

    for i in range(n_ents):
        if not studio.app.world.entities[i].alive:
            studio.gpu_materials.entity_pbr_mat_ids.append(-1)
            studio.gpu_materials.entity_cel_mat_ids.append(-1)
            studio.gpu_materials.entity_vfx_mat_ids.append(-1)
            continue
        var e = Entity(UInt32(i), studio.app.world.entities[i].generation)
        var mo_opt = studio.app.world.get_ptr_mut[MaterialOverrideComponent](e)
        var mr_opt = studio.app.world.get_ptr_mut[MeshRendererComponent](e)
        var mat_id = 0
        if mo_opt:
            ref mo = mo_opt.value()[unsafe_offset=0]
            var ma = MaterialAsset(
                base_color=mo.base_color,
                roughness=mo.roughness,
                metallic=mo.metallic,
            )
            ma.emissive = mo.emissive
            var st_opt = studio.app.world.get_ptr[StudioStyleComponent](e)
            if st_opt:
                ma.clearcoat = st_opt.value()[unsafe_offset=0].clearcoat
                ma.clearcoat_roughness = st_opt.value()[unsafe_offset=0].clearcoat_roughness
            mat_id = renderer.register_material(ma)
            mo.material_id = mat_id
            if mr_opt:
                mr_opt.value()[unsafe_offset=0].material_id = mat_id
        studio.gpu_materials.entity_pbr_mat_ids.append(mat_id)
        studio.gpu_materials.entity_cel_mat_ids.append(-1)
        studio.gpu_materials.entity_vfx_mat_ids.append(-1)

    studio.gpu_materials.mat_trunk_id = (
        studio.gpu_materials.entity_pbr_mat_ids[4]
        if len(studio.gpu_materials.entity_pbr_mat_ids) > 4
        else renderer.register_material(MaterialAsset(base_color=Vec3(0.34, 0.23, 0.14), roughness=0.84, metallic=0.0))
    )
    studio.gpu_materials.mat_foliage_id = (
        studio.gpu_materials.entity_pbr_mat_ids[5]
        if len(studio.gpu_materials.entity_pbr_mat_ids) > 5
        else renderer.register_material(MaterialAsset(base_color=Vec3(0.18, 0.46, 0.22), roughness=0.68, metallic=0.0))
    )

    studio.gpu_materials.grid_mesh_id = renderer.register_mesh(build_studio_reference_grid_mesh(36.0, 24), 0, build_lods=False)
    studio.gpu_materials.orb_mesh_id = renderer.register_mesh(build_subd_material_orb(0.88, 3), 1, build_lods=True)
    studio.gpu_materials.torus_mesh_id = renderer.register_mesh(build_subd_catmull_torus(0.82, 0.30, 2), 2, build_lods=True)
    studio.gpu_materials.colonnade_mesh_id = renderer.register_mesh(build_colonnade_arcade_bay(6.0, 4.8, 2.4), 3, build_lods=True)
    var pn = build_botanical_pine(7, 5.4)
    studio.gpu_materials.foliage_trunk_id = renderer.register_mesh(pn[0], studio.gpu_materials.mat_trunk_id, build_lods=False)
    studio.gpu_materials.foliage_canopy_id = renderer.register_mesh(pn[1], studio.gpu_materials.mat_foliage_id, build_lods=False)

    # Bind registered GPU mesh IDs back onto each entity's MeshRendererComponent
    var mesh_ids = [
        studio.gpu_materials.grid_mesh_id,
        studio.gpu_materials.orb_mesh_id,
        studio.gpu_materials.torus_mesh_id,
        studio.gpu_materials.colonnade_mesh_id,
        studio.gpu_materials.foliage_trunk_id,
        studio.gpu_materials.foliage_canopy_id,
    ]
    for i in range(min(n_ents, len(mesh_ids))):
        var e = studio.resolve_entity(i)
        if e.valid():
            var mr_opt = studio.app.world.get_ptr_mut[MeshRendererComponent](e)
            if mr_opt:
                mr_opt.value()[unsafe_offset=0].mesh_id = mesh_ids[i]

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
    studio.gpu_materials.uploaded = True
    studio._sync_compat_view()


def _fill_instance_entity_maps(
    studio: FlintStudioState,
    mut out_inst_to_entity: List[Int],
    mut out_inst_to_bits: List[UInt64],
):
    fill_instance_entity_maps(studio, out_inst_to_entity, out_inst_to_bits)


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

    var out_w = 1280
    var out_h = 720
    var is_headless = mcp_stdio or headless_smoke or shots_mode
    var render_tier = RENDER_TIER_0_HEADLESS_SIMD if mcp_stdio else RENDER_TIER_2_AAA_VISBUFFER_RT
    var studio = FlintStudioState(
        headless=is_headless,
        render_tier=render_tier,
        width=out_w,
        height=out_h,
        worker_threads=max(1, default_worker_count()),
    )
    _populate_default_studio_scene(studio)

    var assets = AssetServer(enable_watcher=not is_headless)
    var mcp = McpServerV2(workspace_root=".")

    if mcp_stdio:
        var mcp_pool = JobPool()
        _ = mcp_pool.start(max(1, default_worker_count()))
        _ = mcp.serve_stdio_loop(studio, assets, mcp_pool)
        mcp_pool.stop()
        assets.close()
        studio.shutdown()
        return

    var ren_check = studio.app.resources.get_ptr[ScalableRenderer]()
    if not ren_check or not ren_check.value()[unsafe_offset=0].uses_vulkan():
        print("[Flint Studio] ERROR: Vulkan 1.3 driver unavailable.")
        assets.close()
        studio.shutdown()
        return

    _upload_studio_gpu_assets(studio)
    var picker = VisBufferGpuPicker()
    var inst_to_ent = List[Int]()
    var inst_to_bits = List[UInt64]()

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
            var rend_opt = studio.app.resources.get_ptr_mut[ScalableRenderer]()
            if rend_opt and rend_opt.value()[unsafe_offset=0].uses_vulkan():
                rend_opt.value()[unsafe_offset=0].gpu_slot[0].tsr_history_weight = (
                    Float32(0.0) if sub == 0 else Float32(0.78)
                )
            var gms = 0.0
            if rend_opt and rend_opt.value()[unsafe_offset=0].uses_vulkan():
                gms = rend_opt.value()[unsafe_offset=0].gpu_slot[0].render_graph.total_gpu_ms
            studio.render_editor_frame(
                out_w,
                out_h,
                gpu_ms=gms,
                mounted_packs=len(assets.mounts),
                mcp_clients=0,
                input=empty_input,
                dt=Float32(1.0 / 60.0),
            )

        var rend_fin = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        var px = List[UInt32]()
        var culled_draws = 0
        if rend_fin and rend_fin.value()[unsafe_offset=0].uses_vulkan():
            ref r = rend_fin.value()[unsafe_offset=0]
            px = r.readback_pixels()
            culled_draws = r.gpu_slot[0].last_culled_meshlet_draws
            picker.shutdown(r.gpu_slot[0])
        _ = write_png(target_png, px, out_w, out_h)

        var ui_cnt = len(studio.ui_ctx.draw_list.prims)
        print(
            "[Flint Studio] Saved 1280x720 Studio frame -> "
            + target_png
            + " (UIPrims=" + String(ui_cnt)
            + ", Entities=" + String(studio.live_entity_count())
            + ", CulledDraws=" + String(culled_draws) + ")"
        )
        assets.close()
        studio.shutdown()
        return

    # Interactive Window Mode via `WindowPlugin` + `ScalableRenderer`
    var win_ok = False
    var win_opt = studio.app.resources.get_ptr_mut[WindowStateResource]()
    if win_opt and win_opt.value()[unsafe_offset=0].initialized and len(win_opt.value()[unsafe_offset=0].windows) > 0:
        var rend_i = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if rend_i:
            _ = rend_i.value()[unsafe_offset=0].attach_window_surface(win_opt.value()[unsafe_offset=0].windows[0])
            if rend_i.value()[unsafe_offset=0].uses_vulkan():
                ref gpu = rend_i.value()[unsafe_offset=0].gpu_slot[0]
                if gpu.vk.swapchain == 0:
                    gpu.deferred_readback = True
            win_ok = True

    if not win_ok:
        print("[Flint Studio] No Wayland/X11/KMS display detected. Use '--headless-smoke' or '--shots'.")
        var rend_s = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if rend_s and rend_s.value()[unsafe_offset=0].uses_vulkan():
            picker.shutdown(rend_s.value()[unsafe_offset=0].gpu_slot[0])
        assets.close()
        studio.shutdown()
        return

    var mcp_sock = FlintMcpUnixListener(default_mcp_socket_path())
    var mcp_pool = JobPool()
    _ = mcp_pool.start(2)
    var dragging_gizmo = False
    var drag_start_ray = ViewportRay()
    var drag_start_pivot = Vec3()
    var drag_snaps = List[SelectedTransformSnap]()
    var drag_seq = UInt64(100)
    var pending_pick_ctrls = List[Tuple[UInt64, Bool]]()
    var t_last = get_time()

    while True:
        studio.app.schedule.run_stage(
            STAGE_FIRST, studio.app.world, studio.app.resources, studio.app.pool, studio.app.commands
        )
        studio.app.world.flush_commands(studio.app.commands)

        var w_opt = studio.app.resources.get_ptr_mut[WindowStateResource]()
        if not w_opt or not w_opt.value()[unsafe_offset=0].is_open():
            break
        ref win_res = w_opt.value()[unsafe_offset=0]
        if win_res.resized:
            out_w = max(64, win_res.width)
            out_h = max(64, win_res.height)
            var r_opt = studio.app.resources.get_ptr_mut[ScalableRenderer]()
            if r_opt:
                _ = r_opt.value()[unsafe_offset=0].resize(out_w, out_h)

        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - t_last)))
        t_last = now

        var inp_opt = studio.app.resources.get_ptr[InputStateResource]()
        var frame_input = inp_opt.value()[unsafe_offset=0].input.copy() if inp_opt else InputState()

        # Harvest completed non-stalling async GPU viewport picks from previous frames
        var r_pick = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if r_pick and r_pick.value()[unsafe_offset=0].uses_vulkan():
            while True:
                var harvested = picker.poll_completed_pick_entry(r_pick.value()[unsafe_offset=0].gpu_slot[0])
                if not harvested:
                    break
                var entry = harvested.value()
                var h_req_id = entry[0]
                var pres = entry[1]
                var pick_ctrl = False
                var kept = List[Tuple[UInt64, Bool]](capacity=len(pending_pick_ctrls))
                for pi in range(len(pending_pick_ctrls)):
                    if pending_pick_ctrls[pi][0] == h_req_id:
                        pick_ctrl = pending_pick_ctrls[pi][1]
                    else:
                        kept.append(pending_pick_ctrls[pi])
                pending_pick_ctrls = kept^
                if (
                    pres.hit
                    and pres.entity_id >= 0
                    and studio.is_alive_id(pres.entity_id)
                    and (pres.entity_bits == 0 or studio.resolve_entity(pres.entity_id).to_bits() == pres.entity_bits)
                ):
                    studio.select_entity(pres.entity_id, multi_select=pick_ctrl)
                elif not pick_ctrl:
                    studio.clear_selection()

        if not studio.process_editor_input(frame_input, dt):
            break

        var vp_rect = studio.viewport_rect(out_w, out_h)
        var vp_x = vp_rect.x
        var vp_y = vp_rect.y
        var vp_w = vp_rect.w
        var vp_h = vp_rect.h

        var mx = frame_input.last_mouse_x
        var my = frame_input.last_mouse_y
        var in_vp = vp_rect.contains(mx, my)

        if frame_input.mouse_pressed(0) and in_vp and studio.dock.dragging_node < 0:
            var g_piv = studio.selection_gizmo_pivot()
            var g_hit = GIZMO_AXIS_NONE
            if g_piv[0]:
                g_hit = pick_gizmo_handle(
                    mx, my, g_piv[1],
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
                drag_start_pivot = g_piv[1]
                drag_snaps = studio.capture_selected_transforms()
            else:
                var ctrl_down = frame_input.key_down(341) or frame_input.key_down(345)
                if picker.has_pending_pick:
                    var stale_id = picker.queued_request_id
                    var kept = List[Tuple[UInt64, Bool]](capacity=len(pending_pick_ctrls))
                    for pi in range(len(pending_pick_ctrls)):
                        if pending_pick_ctrls[pi][0] != stale_id:
                            kept.append(pending_pick_ctrls[pi])
                    pending_pick_ctrls = kept^
                var req_id = picker.request_pick(Int(mx), Int(my))
                pending_pick_ctrls.append((req_id, ctrl_down))

        if dragging_gizmo and frame_input.mouse_down(0) and len(drag_snaps) > 0:
            var cur_ray = ViewportRay.from_screen(
                mx, my, vp_x, vp_y, vp_w, vp_h,
                studio.cam_pos, studio.cam_fwd, studio.cam_right, studio.cam_up, studio.fov_y_deg,
            )
            var h_id = studio.active_gizmo_handle
            var wscale = gizmo_world_scale(drag_start_pivot, studio.cam_pos, studio.cam_fwd, studio.fov_y_deg, Int(vp_h))
            if studio.gizmo_mode == GIZMO_MODE_TRANSLATE:
                var delta_p = Vec3()
                if h_id == GIZMO_AXIS_X or h_id == GIZMO_AXIS_Y or h_id == GIZMO_AXIS_Z:
                    var ax = Vec3(1, 0, 0) if h_id == GIZMO_AXIS_X else (Vec3(0, 1, 0) if h_id == GIZMO_AXIS_Y else Vec3(0, 0, 1))
                    var t0 = ray_axis_closest_param(drag_start_ray, drag_start_pivot, ax)
                    var t1 = ray_axis_closest_param(cur_ray, drag_start_pivot, ax)
                    if t0[0] and t1[0]:
                        delta_p = ax * (t1[1] - t0[1])
                elif h_id == GIZMO_PLANE_XY or h_id == GIZMO_PLANE_XZ or h_id == GIZMO_PLANE_YZ:
                    var pn = Vec3(0, 0, 1) if h_id == GIZMO_PLANE_XY else (Vec3(0, 1, 0) if h_id == GIZMO_PLANE_XZ else Vec3(1, 0, 0))
                    var p0 = ray_plane_hit(drag_start_ray, drag_start_pivot, pn)
                    var p1 = ray_plane_hit(cur_ray, drag_start_pivot, pn)
                    if p0[0] and p1[0]:
                        delta_p = p1[1] - p0[1]
                if studio.snap_enabled:
                    var snapped_piv = snap_translation(drag_start_pivot + delta_p, studio.grid_snap_m)
                    delta_p = snapped_piv - drag_start_pivot
                _ = studio.apply_gizmo_drag_selected(
                    drag_snaps,
                    drag_start_pivot,
                    delta_p,
                    Quat(0.0, 0.0, 0.0, 1.0),
                    Vec3(1.0, 1.0, 1.0),
                    GIZMO_MODE_TRANSLATE,
                    coalesce_key=drag_seq,
                )
            elif studio.gizmo_mode == GIZMO_MODE_ROTATE:
                var rax = Vec3(1, 0, 0) if h_id == GIZMO_AXIS_X else (Vec3(0, 1, 0) if h_id == GIZMO_AXIS_Y else Vec3(0, 0, 1))
                var drad = gizmo_rotation_delta_rad(drag_start_ray, cur_ray, drag_start_pivot, rax, studio.cam_fwd, wscale)
                if studio.snap_enabled:
                    drad = snap_angle_deg(drad, studio.angle_snap_deg)
                var dq = Quat(rax.x * sin(drad * Float32(0.5)), rax.y * sin(drad * Float32(0.5)), rax.z * sin(drad * Float32(0.5)), cos(drad * Float32(0.5)))
                _ = studio.apply_gizmo_drag_selected(
                    drag_snaps,
                    drag_start_pivot,
                    Vec3(),
                    dq,
                    Vec3(1.0, 1.0, 1.0),
                    GIZMO_MODE_ROTATE,
                    coalesce_key=drag_seq,
                )
            elif studio.gizmo_mode == GIZMO_MODE_SCALE:
                var sf = gizmo_scale_drag(drag_start_ray, cur_ray, drag_start_pivot, h_id, wscale, studio.cam_fwd, studio.cam_right, Vec3(1.0, 1.0, 1.0))
                _ = studio.apply_gizmo_drag_selected(
                    drag_snaps,
                    drag_start_pivot,
                    Vec3(),
                    Quat(0.0, 0.0, 0.0, 1.0),
                    sf,
                    GIZMO_MODE_SCALE,
                    coalesce_key=drag_seq,
                )

        if not frame_input.mouse_down(0):
            dragging_gizmo = False
            studio.active_gizmo_handle = GIZMO_AXIS_NONE

        if studio.pie.mode == PIE_PLAYING:
            _ = studio.pie_step(1)

        _ = assets.tick(now)
        var r_mcp = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        var gms = 0.0
        if r_mcp and r_mcp.value()[unsafe_offset=0].uses_vulkan():
            _ = mcp_sock.poll_once_with_renderer(mcp, studio, assets, mcp_pool, r_mcp.value()[unsafe_offset=0].gpu_slot[0])
            gms = r_mcp.value()[unsafe_offset=0].gpu_slot[0].render_graph.total_gpu_ms

        studio.render_editor_frame(
            out_w,
            out_h,
            gpu_ms=gms,
            mounted_packs=len(assets.mounts),
            mcp_clients=mcp_sock.client_count(),
            input=frame_input,
            dt=dt,
        )
        _fill_instance_entity_maps(studio, inst_to_ent, inst_to_bits)

        var r_post = studio.app.resources.get_ptr_mut[ScalableRenderer]()
        if r_post:
            ref renderer = r_post.value()[unsafe_offset=0]
            if renderer.uses_vulkan():
                _ = picker.dispatch_pending_pick(renderer.gpu_slot[0], inst_to_ent, inst_to_bits)
            if not renderer.surface.is_zero_copy_gpu():
                var frame_px = renderer.readback_pixels()
                var w_p = studio.app.resources.get_ptr_mut[WindowStateResource]()
                if w_p and len(w_p.value()[unsafe_offset=0].windows) > 0:
                    w_p.value()[unsafe_offset=0].windows[0].present(frame_px, renderer.width, renderer.height)

    mcp_sock.close()
    mcp_pool.stop()
    var r_end = studio.app.resources.get_ptr_mut[ScalableRenderer]()
    if r_end and r_end.value()[unsafe_offset=0].uses_vulkan():
        picker.shutdown(r_end.value()[unsafe_offset=0].gpu_slot[0])
    assets.close()
    studio.shutdown()

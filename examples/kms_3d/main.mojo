"""GPU 3D straight to the screen: the AAA Vulkan renderer on the KMS backend.

No X11, no Wayland, no CPU copy: the window takes the display through DRM/KMS
(platform/kms.mojo), then `WaylandWindow.attach_vulkan` hands it to a Vulkan
swapchain on that connector (VK_KHR_display + VK_EXT_acquire_drm_display). From a text console
(Ctrl+Alt+F3):

    bash tools/flint.sh kms3d
    FLINT_KMS_MODE=1920x1080 bash tools/flint.sh kms3d   # a 60 Hz mode on a 4K@30 link

Drag with the left button or use WASD/arrows to orbit, wheel or Q/E to zoom.
Esc or Ctrl+C quits and gives the console back.
"""
from std.math import sin, cos
from flint.core.math import Vec3, Quat
from flint.core.transform import trs
from flint.assets.mesh_builder import MeshBuilder
from flint.platform.os import get_time, env_value
from flint.platform.input import InputState
from flint.platform.wayland import WaylandWindow
from flint.render.aaa_bsdf import AaaMaterial
from flint.render.aaa_renderer import FlintAaaVulkanRenderer


def main():
    var win = WaylandWindow()
    var input = InputState()
    if not win.initialize(1280, 720) or not win.using_kms:
        if win.backend_name() != "Headless":
            win.close()
        print("[Flint] kms_3d needs the bare display: run it from a text console (Ctrl+Alt+F3) with FLINT_BACKEND=kms.")
        return
    var out_w = win.kms.width
    var out_h = win.kms.height
    # Render below 4K and let TSR upscale; at 1080p and under, native.
    var tsr = Float32(0.5) if out_w * out_h > 2560 * 1440 else Float32(1.0)
    var tsr_text = env_value("FLINT_TSR")
    if tsr_text == "1" or tsr_text == "1.0":
        tsr = 1.0
    var ren = FlintAaaVulkanRenderer(width=out_w, height=out_h, tsr_scale=tsr, env_cube_dim=128)
    if not ren.active:
        print("[Flint] Vulkan renderer unavailable: " + ren.vk.error)
        ren.shutdown()
        win.close()
        return
    # The same call a game makes on any backend; on KMS Vulkan takes the display here.
    var sc = win.attach_vulkan(ren.vk, out_w, out_h)
    if sc.swapchain == 0:
        print("[Flint] Vulkan could not take the display: " + win.kms.last_error)
        ren.shutdown()
        win.close()
        return
    ren.enable_rt_shadows = ren.vk.has_ray_query
    ren.enable_rt_reflections = ren.vk.has_ray_query

    var m_floor = ren.add_material(AaaMaterial(base_color=Vec3(0.18, 0.20, 0.24), roughness=0.6))
    var m_gold = ren.add_material(AaaMaterial(base_color=Vec3(1.0, 0.78, 0.34), roughness=0.22, metallic=1.0))
    var m_amber = AaaMaterial(base_color=Vec3(0.96, 0.62, 0.22), roughness=0.4)
    m_amber.emissive = Vec3(2.4, 1.1, 0.2)
    var m_glow = ren.add_material(m_amber)
    var m_blue = ren.add_material(AaaMaterial(base_color=Vec3(0.20, 0.55, 0.95), roughness=0.08))
    var m_chrome = ren.add_material(AaaMaterial(base_color=Vec3(0.92, 0.93, 0.95), roughness=0.05, metallic=1.0))

    var floor_mb = MeshBuilder()
    floor_mb.add_beveled_box(Vec3(0.0, -0.1, 0.0), Vec3(9.0, 0.1, 9.0), 0.04)
    var floor_mesh = ren.add_mesh(floor_mb^.finish(), m_floor)
    var sphere_mb = MeshBuilder()
    sphere_mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0), 32, 48)
    var sphere_mesh = ren.add_mesh(sphere_mb^.finish(), m_gold)
    var box_mb = MeshBuilder()
    box_mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.6, 0.6, 0.6), 0.08)
    var box_mesh = ren.add_mesh(box_mb^.finish(), m_blue)

    var yaw = Float32(0.7)
    var pitch = Float32(0.32)
    var dist = Float32(9.0)
    var start = get_time()
    var last = start
    var frames = 0
    var fps = 0
    var fps_at = start
    var gpu_label = "VULKAN-RT" if ren.vk.has_ray_query else "VULKAN"
    while not win.close_requested:
        input.begin_frame()
        win.poll(input)
        if input.key_pressed(256):
            break
        var now = get_time()
        var dt = Float32(min(0.05, max(0.001, now - last)))
        last = now
        var t = Float32(now - start)
        if input.key_down(65) or input.key_down(263):
            yaw -= Float32(1.4) * dt
        if input.key_down(68) or input.key_down(262):
            yaw += Float32(1.4) * dt
        if input.key_down(87) or input.key_down(265):
            pitch = min(Float32(1.25), pitch + Float32(1.0) * dt)
        if input.key_down(83) or input.key_down(264):
            pitch = max(Float32(0.05), pitch - Float32(1.0) * dt)
        if input.key_down(81):
            dist = max(Float32(3.0), dist - Float32(6.0) * dt)
        if input.key_down(69):
            dist = min(Float32(25.0), dist + Float32(6.0) * dt)
        if input.mouse_down(0):
            yaw += input.mouse_delta_x() * Float32(0.006)
            pitch = max(Float32(0.05), min(Float32(1.25), pitch + input.mouse_delta_y() * Float32(0.006)))
        dist = max(Float32(3.0), min(Float32(25.0), dist - input.mouse_wheel() * Float32(0.6)))
        frames += 1
        if now - fps_at >= 1.0:
            fps = frames
            frames = 0
            fps_at = now

        var eye = Vec3(cos(yaw) * cos(pitch) * dist, sin(pitch) * dist + 0.6, sin(yaw) * cos(pitch) * dist)
        ren.begin_frame(eye, Vec3(0.0, 0.8, 0.0), fov_y_deg=50.0, near_z=0.1)
        ren.draw_instance(floor_mesh.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1, 1, 1)))
        ren.draw_instance(sphere_mesh.mesh_id, trs(Vec3(0.0, 1.2 + sin(t * 1.6) * 0.25, 0.0), Quat(), Vec3(1.2, 1.2, 1.2)))
        for k in range(6):
            var a = t * 0.5 + Float32(k) * 1.0472
            var spin = Quat.from_axis_angle(Vec3(0.3, 1.0, 0.2).normalized(), t * 1.3 + Float32(k))
            var material = m_glow if k % 3 == 0 else (m_chrome if k % 3 == 1 else m_blue)
            ren.draw_instance(box_mesh.mesh_id, trs(Vec3(cos(a) * 3.4, 0.7 + sin(t * 2.0 + Float32(k)) * 0.3, sin(a) * 3.4), spin, Vec3(1, 1, 1)), material_override=material)
        if ren.ui_font.valid:
            var hud = "FLINT KMS " + gpu_label + "  " + String(out_w) + "X" + String(out_h) + "@" + String(Int(win.kms.mode.refresh_hz() + 0.5)) + "  " + String(fps) + " FPS  |  DRAG/WASD ORBIT  WHEEL ZOOM  ESC QUITS"
            ren.ui_draw_list.add_rect(16.0, 14.0, Float32(min(out_w - 32, 1500)), 44.0, 0xC0101824, 6.0)
            ren.ui_draw_list.add_text(ren.ui_font_atlas_slot, ren.ui_font, hud, 30.0, 26.0, 22.0, 0xFFFFE082)
        if not ren.render_frame():
            print("[Flint] render_frame failed: " + ren.vk.error)
            break
    ren.vk.close_swapchain()  # releases the display back to our DRM fd
    ren.shutdown()
    win.close()
    print("[Flint] kms_3d closed after " + String(Int(get_time() - start)) + " s; console restored.")

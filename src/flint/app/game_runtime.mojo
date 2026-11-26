"""Standard game runtime for Flint games (`flint.app.game_runtime`).

A game made with `flint.sh new` used to install only Transform, Window and Input and draw every
entity as a CPU-rasterised box, so a scene built in Studio looked nothing like the shipped game,
and anything using physics, audio or animation silently did nothing. This module is the one place
that wires a real game:

- `add_game_plugins(app, cfg)`: Transform, Window, Input, Physics, Nav, AI, Audio, Anim, Asset and
  Render plugins, plus `GamePresentPlugin`.
- `register_game_codecs(reg)`: every engine component codec, so scene files load every component
  instead of keeping unknown ones as opaque blobs.
- `GamePresentPlugin`: after `RenderPlugin` draws, attaches the renderer to the game window on the
  first frame, follows window resizes, and presents the frame (zero-copy swapchain when available,
  read-back + `WaylandWindow.present` otherwise). It also registers the built-in primitive meshes
  scene files refer to (`mesh_id` 0 = unit cube, 1 = unit sphere) when the game has registered none.
"""
from flint.ai.behavior_tree import AiPlugin, register_ai_codecs
from flint.anim.plugin import AnimPlugin, register_anim_codecs
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    Resource,
    Resources,
    STAGE_LAST,
)
from flint.app.settings import GameSettings
from flint.app.window_plugin import WindowPlugin, WindowStateResource
from flint.assets.asset_plugin import AssetPlugin
from flint.assets.mesh_builder import MeshBuilder
from flint.audio.plugin import AudioPlugin, register_audio_codecs
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.ecs.world import ParallelCommands, World, comp_hash64
from flint.input.plugin import InputPlugin
from flint.nav.plugin import NavPlugin, register_nav_codecs
from flint.physics.plugin import PhysicsPlugin, register_physics_codecs
from flint.render.material_graph import MaterialAsset
from flint.render.plugin import RenderPlugin, register_render_codecs
from flint.render.render_world import RENDER_TIER_2_AAA_VISBUFFER_RT, ScalableRenderer
from flint.render.swapchain_surface import PRESENT_FIFO, PRESENT_MAILBOX
from flint.scene import ComponentRegistry, TransformPlugin

comptime GAME_MESH_CUBE: Int = 0
"""Built-in unit cube (1 m, centred) registered when a game registers no meshes of its own."""
comptime GAME_MESH_SPHERE: Int = 1
"""Built-in unit sphere (1 m diameter)."""


struct GameRuntimeConfig(Copyable, Movable):
    var title: String
    var width: Int
    var height: Int
    var headless: Bool
    var gravity: Float32
    var render_tier: UInt32
    var asset_root: String
    """Loose asset folder (`assets` by default); mounted packs are read before it."""
    var hot_reload_assets: Bool
    var vsync: Bool

    def __init__(
        out self,
        title: String = "Flint Game",
        width: Int = 1280,
        height: Int = 720,
        headless: Bool = False,
        gravity: Float32 = 9.81,
        render_tier: UInt32 = RENDER_TIER_2_AAA_VISBUFFER_RT,
        asset_root: String = "assets",
        hot_reload_assets: Bool = False,
        vsync: Bool = True,
    ):
        self.title = title
        self.width = width
        self.height = height
        self.headless = headless
        self.gravity = gravity
        self.render_tier = render_tier
        self.asset_root = asset_root
        self.hot_reload_assets = hot_reload_assets
        self.vsync = vsync


def add_game_plugins(mut app: EngineApp, cfg: GameRuntimeConfig):
    """Installs the standard plugin set a game needs, in dependency order."""
    app.add_plugin(TransformPlugin(snapshot_prev_in_fixed=True))
    app.add_plugin(WindowPlugin(headless=cfg.headless, width=cfg.width, height=cfg.height, title=cfg.title))
    app.add_plugin(InputPlugin(enable_evdev=not cfg.headless))
    app.add_plugin(AssetPlugin(loose_root=cfg.asset_root, enable_watcher=cfg.hot_reload_assets))
    app.add_plugin(PhysicsPlugin(gravity=cfg.gravity))
    app.add_plugin(NavPlugin(draw_debug=False))
    app.add_plugin(AiPlugin())
    app.add_plugin(AudioPlugin(headless=cfg.headless))
    app.add_plugin(AnimPlugin())
    app.add_plugin(RenderPlugin(tier=cfg.render_tier, width=cfg.width, height=cfg.height))
    app.add_plugin(GamePresentPlugin(vsync=cfg.vsync))


def register_game_codecs(mut reg: ComponentRegistry):
    """Registers every engine component codec on `reg` (call on `ComponentRegistry.with_builtins()`)."""
    register_render_codecs(reg)
    register_physics_codecs(reg)
    register_nav_codecs(reg)
    register_ai_codecs(reg)
    register_audio_codecs(reg)
    register_anim_codecs(reg)


struct GamePresentStateResource(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.GamePresentStateResource")
    var attached: Bool
    var attach_failed: Bool
    var primitives_registered: Bool
    var frames_presented: Int
    var vsync: Bool
    var device_recoveries: Int
    """Times the GPU device was lost and rebuilt (textures and skinned meshes need re-uploading then)."""
    var recovery_failed: Bool
    var applied_render_scale: Float32

    def __init__(out self, vsync: Bool = True):
        self.attached = False
        self.attach_failed = False
        self.primitives_registered = False
        self.frames_presented = 0
        self.vsync = vsync
        self.device_recoveries = 0
        self.recovery_failed = False
        self.applied_render_scale = Float32(1.0)


def register_builtin_primitives(mut renderer: ScalableRenderer) -> Bool:
    """Registers the unit cube (`GAME_MESH_CUBE`) and unit sphere (`GAME_MESH_SPHERE`) if the renderer has
    no meshes yet, so `mesh_id` 0/1 in a scene file draws something real. Returns True if it did."""
    if len(renderer.cpu_mesh_materials) > 0:
        return False
    var mat = renderer.register_material(MaterialAsset(base_color=Vec3(0.8, 0.8, 0.8), roughness=0.6, metallic=0.0))
    var cube = MeshBuilder()
    cube.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5), bevel=Float32(0.03))
    _ = renderer.register_mesh(cube^.finish(), mat, build_lods=False)
    var ball = MeshBuilder()
    ball.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5), rings=16, segments=24)
    _ = renderer.register_mesh(ball^.finish(), mat, build_lods=False)
    return True


def _sys_game_present(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[GamePresentStateResource]()
    var rend_opt = resources.get_ptr_mut[ScalableRenderer]()
    var win_opt = resources.get_ptr_mut[WindowStateResource]()
    if not st_opt or not rend_opt or not win_opt:
        return
    ref st = st_opt.value()[unsafe_offset=0]
    ref renderer = rend_opt.value()[unsafe_offset=0]
    ref win_res = win_opt.value()[unsafe_offset=0]
    if not st.primitives_registered:
        st.primitives_registered = True
        _ = register_builtin_primitives(renderer)
    if win_res.headless or not win_res.initialized or len(win_res.windows) == 0:
        return
    if not st.attached and not st.attach_failed:
        var mode = PRESENT_FIFO if st.vsync else PRESENT_MAILBOX
        if renderer.attach_window_surface(win_res.windows[0], present_mode=mode):
            st.attached = True
            if renderer.uses_vulkan() and renderer.gpu_slot[0].vk.swapchain == 0:
                renderer.gpu_slot[0].deferred_readback = True  # no swapchain: present through read-back
        else:
            st.attach_failed = True
    if renderer.uses_vulkan():
        ref gpu = renderer.gpu_slot[0]
        # A lost device (driver reset, GPU hang, eGPU unplug) is rebuilt instead of leaving a frozen window.
        if gpu.is_device_lost() and not st.recovery_failed:
            if gpu.recover_from_device_loss():
                st.device_recoveries += 1
            else:
                st.recovery_failed = True
        # The player's render-scale setting, applied when it changes.
        var set_opt = resources.get_ptr[GameSettings]()
        if set_opt:
            var want = set_opt.value()[unsafe_offset=0].render_scale
            if abs(want - st.applied_render_scale) > Float32(1.0e-3):
                if gpu.set_render_scale(want):
                    st.applied_render_scale = want
                else:
                    st.applied_render_scale = want  # keep the old scale; don't retry every frame
    if win_res.resized and win_res.width >= 64 and win_res.height >= 64:
        _ = renderer.resize(win_res.width, win_res.height)
    # `RenderPlugin` has drawn this frame by now (this system runs after it). With a zero-copy
    # swapchain the frame is already on screen; otherwise copy it into the window.
    if not renderer.surface.is_zero_copy_gpu() and not win_res.windows[0].presents_on_gpu():
        var px = renderer.readback_pixels()
        if len(px) >= renderer.width * renderer.height and renderer.width > 0:
            win_res.windows[0].present(px, renderer.width, renderer.height)
    st.frames_presented += 1


def _find_system_id(app: EngineApp, stage: Int, name: String) -> Int:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return sys[i].id
    return -1


struct GamePresentPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Presents `RenderPlugin`'s frame to the game window (see module docs). Add after `RenderPlugin`."""
    var vsync: Bool

    def __init__(out self, vsync: Bool = True):
        self.vsync = vsync

    def build(self, mut app: EngineApp):
        if not app.resources.has[GamePresentStateResource]():
            app.insert_resource[GamePresentStateResource](GamePresentStateResource(self.vsync))
        if _find_system_id(app, STAGE_LAST, "flint.app.game_present") >= 0:
            return
        var draw_id = _find_system_id(app, STAGE_LAST, "flint.render.extract_and_draw")
        _ = app.add_system(
            STAGE_LAST,
            "flint.app.game_present",
            _sys_game_present,
            exclusive=True,
            after_system_id=draw_id,
        )

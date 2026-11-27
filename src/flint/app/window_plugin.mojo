"""Flint v5 Window Plugin (`flint.app.window_plugin`).

Provides:
- `WindowStateResource`: RAII `Resource` managing `WaylandWindow` (with X11/KMS
  fallback when `not headless`; zero OS FDs opened when `headless=True`), resize/
  focus/scale tracking, and clean shutdown requests.
- `WindowPlugin`: `EnginePlugin` initializing the window in `STAGE_STARTUP`,
  polling window events into `InputStateResource` in `STAGE_FIRST` (before
  `InputPlugin` consumes `InputState`), and tearing down cleanly in `STAGE_SHUTDOWN`.
"""
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    Resource,
    Resources,
    STAGE_FIRST,
    STAGE_SHUTDOWN,
    STAGE_STARTUP,
)
from flint.core.jobs import JobPool
from flint.ecs.world import ParallelCommands, World, comp_hash64
from flint.input.plugin import InputStateResource, ensure_headless_input_swap_hook
from flint.platform.wayland import WaylandWindow


struct WindowStateResource(Resource, Movable):
    """Non-`Copyable` RAII `Resource` holding OS window handle and resize/close state."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.app.WindowStateResource")
    var headless: Bool
    var width: Int
    var height: Int
    var title: String
    var close_requested: Bool
    var resized: Bool
    var content_scale: Float32
    var focused: Bool
    var initialized: Bool
    var windows: List[WaylandWindow]
    # IME request from the app's UI: a text field has keyboard focus, with its caret rectangle in
    # framebuffer pixels. Sent to the backend (`WaylandWindow.set_text_input`) on every poll.
    var text_input_wanted: Bool
    var text_input_rect: Tuple[Float32, Float32, Float32, Float32]

    def __init__(
        out self,
        headless: Bool = False,
        width: Int = 1280,
        height: Int = 720,
        title: String = "Flint v5",
    ):
        self.headless = headless
        self.width = width
        self.height = height
        self.title = title
        self.close_requested = False
        self.resized = False
        self.content_scale = Float32(1.0)
        self.focused = True
        self.initialized = False
        self.windows = List[WaylandWindow]()
        self.text_input_wanted = False
        self.text_input_rect = (Float32(0.0), Float32(0.0), Float32(1.0), Float32(1.0))

    def set_text_input(mut self, wanted: Bool, x: Float32 = 0.0, y: Float32 = 0.0, w: Float32 = 1.0, h: Float32 = 1.0):
        """Asks for the platform IME while a text field has keyboard focus (caret in framebuffer px)."""
        self.text_input_wanted = wanted
        if wanted:
            self.text_input_rect = (x, y, w, h)

    def request_close(mut self):
        self.close_requested = True

    def is_open(self) -> Bool:
        return self.initialized and (not self.close_requested)

    def shutdown(mut self):
        for i in range(len(self.windows)):
            self.windows[i].shutdown()
        self.windows.clear()
        self.initialized = False

    def __deinit__(deinit self):
        self.shutdown()


def _sys_window_startup(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var win_opt = resources.get_ptr_mut[WindowStateResource]()
    if not win_opt:
        return
    ref win_res = win_opt.value()[unsafe_offset=0]
    if win_res.headless or win_res.initialized:
        return
    var win = WaylandWindow()
    win.window_title = win_res.title
    if win.initialize(win_res.width, win_res.height):
        var fb = win.framebuffer_size()
        if fb[0] > 0 and fb[1] > 0:
            win_res.width = fb[0]
            win_res.height = fb[1]
        win_res.content_scale = win.content_scale()
        win_res.focused = win.focused
        win_res.windows.append(win^)
        win_res.initialized = True
        var in_opt = resources.get_ptr_mut[InputStateResource]()
        if in_opt:
            in_opt.value()[unsafe_offset=0].has_os_window = True


def _sys_window_poll(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var win_opt = resources.get_ptr_mut[WindowStateResource]()
    var in_opt = resources.get_ptr_mut[InputStateResource]()
    if not win_opt:
        return
    ref win_res = win_opt.value()[unsafe_offset=0]
    win_res.resized = False
    if len(win_res.windows) == 0:
        if in_opt:
            in_opt.value()[unsafe_offset=0].has_os_window = False
        return
    ref win = win_res.windows[0]
    if in_opt:
        in_opt.value()[unsafe_offset=0].has_os_window = True
        ref input_state = in_opt.value()[unsafe_offset=0].input
        input_state.begin_frame()
        win.set_text_input(
            win_res.text_input_wanted,
            win_res.text_input_rect[0],
            win_res.text_input_rect[1],
            win_res.text_input_rect[2],
            win_res.text_input_rect[3],
        )
        win.poll_events(input_state)
    var fb = win.framebuffer_size()
    if fb[0] > 0 and fb[1] > 0 and (fb[0] != win_res.width or fb[1] != win_res.height):
        win_res.width = fb[0]
        win_res.height = fb[1]
        win_res.resized = True
    win_res.content_scale = win.content_scale()
    win_res.focused = win.focused
    if win.should_close():
        win_res.close_requested = True


def _sys_window_shutdown(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var win_opt = resources.get_ptr_mut[WindowStateResource]()
    if win_opt:
        win_opt.value()[unsafe_offset=0].shutdown()
    var in_opt = resources.get_ptr_mut[InputStateResource]()
    if in_opt:
        in_opt.value()[unsafe_offset=0].has_os_window = False


def _has_window_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct WindowPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers window lifecycle and `STAGE_FIRST` event polling into `EngineApp`."""
    var headless: Bool
    var width: Int
    var height: Int
    var title: String

    def __init__(
        out self,
        headless: Bool = False,
        width: Int = 1280,
        height: Int = 720,
        title: String = "Flint v5",
    ):
        self.headless = headless
        self.width = width
        self.height = height
        self.title = title

    def build(self, mut app: EngineApp):
        var eff_headless = self.headless or app.config.headless
        if not app.resources.has[WindowStateResource]():
            app.insert_resource[WindowStateResource](
                WindowStateResource(eff_headless, self.width, self.height, self.title)
            )
        if not app.resources.has[InputStateResource]():
            app.insert_resource[InputStateResource](InputStateResource())

        ensure_headless_input_swap_hook(app)

        if not _has_window_system(app, STAGE_STARTUP, "flint.app.window_startup"):
            _ = app.add_system(
                STAGE_STARTUP,
                "flint.app.window_startup",
                _sys_window_startup,
                exclusive=True,
            )
        if not _has_window_system(app, STAGE_FIRST, "flint.app.window_poll"):
            var win_poll_id = app.add_system(
                STAGE_FIRST,
                "flint.app.window_poll",
                _sys_window_poll,
                exclusive=True,
            )
            # Ensure `flint.app.window_poll` precedes `flint.input.update` in `STAGE_FIRST`
            # even if `InputPlugin` was added before `WindowPlugin`, while preserving any
            # systems registered after `flint.input.update` (e.g., dependent systems).
            ref st_first = app.schedule.stages[STAGE_FIRST]
            var found_inp_idx = -1
            for i in range(len(st_first.systems) - 1):
                if st_first.systems[i].name == "flint.input.update":
                    found_inp_idx = i
                    break
            if found_inp_idx >= 0:
                var last_idx = len(st_first.systems) - 1
                var win_desc = st_first.systems[last_idx].copy()
                var k = last_idx
                while k > found_inp_idx:
                    st_first.systems[k] = st_first.systems[k - 1].copy()
                    k -= 1
                st_first.systems[found_inp_idx] = win_desc^
                st_first.systems[found_inp_idx + 1].explicit_after.append(win_poll_id)
                st_first.dirty = True
        if not _has_window_system(app, STAGE_SHUTDOWN, "flint.app.window_shutdown"):
            _ = app.add_system(
                STAGE_SHUTDOWN,
                "flint.app.window_shutdown",
                _sys_window_shutdown,
                exclusive=True,
            )

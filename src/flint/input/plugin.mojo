"""Flint v5 Input Plugin (`flint.input.plugin`).

Provides:
- `InputStateResource`: `Resource` wrapper around `flint.platform.input.InputState`
- `EvdevHubResource`: RAII `Resource` wrapper around `flint.input.evdev.EvdevHub`
- `InputConfigResource`: configuration flags for `InputPlugin`
- `InputPlugin`: `EnginePlugin` inserting `InputMap`, `InputStateResource`, and
  `EvdevHubResource`, polling `EvdevHub` and updating `InputMap.update(...)` in
  `STAGE_FIRST` (strictly ordered after `WindowPlugin`'s window poll).
"""
from std.math import max
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    FrameTime,
    Resource,
    Resources,
    STAGE_FIRST,
    STAGE_SHUTDOWN,
)
from flint.core.jobs import JobPool
from flint.ecs.world import ParallelCommands, World, comp_hash64
from flint.input.evdev import EvdevHub
from flint.input.input_map import InputMap
from flint.platform.input import InputState


struct InputStateResource(Resource, Movable):
    """Singleton `Resource` holding the current frame's `InputState`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.input.InputStateResource")
    var input: InputState
    var has_os_window: Bool
    var _headless_hook_added: Bool

    def __init__(out self):
        self.input = InputState()
        self.has_os_window = False
        self._headless_hook_added = False


struct EvdevHubResource(Resource, Movable):
    """RAII `Resource` wrapping non-`Copyable` `EvdevHub` with deterministic close."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.input.EvdevHubResource")
    var hub: EvdevHub
    var enabled: Bool

    def __init__(out self, enabled: Bool = False, enable_hotplug: Bool = False):
        self.hub = EvdevHub(enable_hotplug=enable_hotplug)
        self.enabled = enabled

    def __deinit__(deinit self):
        self.hub.close()


@fieldwise_init
struct InputConfigResource(Resource, Copyable, ImplicitlyCopyable, Movable):
    """Configuration state for `InputPlugin`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.input.InputConfigResource")
    var enable_evdev: Bool


def _sys_input_update(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var dt = Float32(1.0 / 60.0)
    var ft_opt = resources.get_ptr[FrameTime]()
    if ft_opt:
        var fdt = ft_opt.value()[unsafe_offset=0].delta_seconds
        if fdt > Float32(0.0):
            dt = fdt

    var in_opt = resources.get_ptr_mut[InputStateResource]()
    if not in_opt:
        return
    ref input_state = in_opt.value()[unsafe_offset=0].input

    var ev_opt = resources.get_ptr_mut[EvdevHubResource]()
    if ev_opt:
        ref ev_res = ev_opt.value()[unsafe_offset=0]
        if ev_res.enabled:
            ev_res.hub.poll(input_state)

    var map_opt = resources.get_ptr_mut[InputMap]()
    if map_opt:
        map_opt.value()[unsafe_offset=0].update(input_state, dt)


def _sys_input_shutdown(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var ev_opt = resources.get_ptr_mut[EvdevHubResource]()
    if ev_opt:
        ev_opt.value()[unsafe_offset=0].hub.close()
    var in_opt = resources.get_ptr_mut[InputStateResource]()
    if in_opt:
        in_opt.value()[unsafe_offset=0].input.close()


def _swap_headless_input(res_ptr: Pointer[Resources, MutAnyOrigin]):
    ref resources = res_ptr[unsafe_offset=0]
    var in_opt = resources.get_ptr_mut[InputStateResource]()
    if in_opt:
        ref in_res = in_opt.value()[unsafe_offset=0]
        if not in_res.has_os_window:
            in_res.input.begin_frame()


def ensure_headless_input_swap_hook(mut app: EngineApp):
    """Idempotently registers end-of-frame `InputState.begin_frame()` when headless."""
    var in_opt = app.resources.get_ptr_mut[InputStateResource]()
    if in_opt and not in_opt.value()[unsafe_offset=0]._headless_hook_added:
        in_opt.value()[unsafe_offset=0]._headless_hook_added = True
        app.add_event_swap_hook(_swap_headless_input)


def _has_input_system(app: EngineApp, stage: Int, name: StaticString) -> Bool:
    ref sys = app.schedule.stages[stage].systems
    for i in range(len(sys)):
        if sys[i].name == name:
            return True
    return False


struct InputPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `InputMap`, `InputStateResource`, and `EvdevHubResource` in `EngineApp`."""
    var enable_evdev: Bool

    def __init__(out self, enable_evdev: Bool = True):
        self.enable_evdev = enable_evdev

    def build(self, mut app: EngineApp):
        var use_evdev = self.enable_evdev and (not app.config.headless)
        if not app.resources.has[InputMap]():
            app.insert_resource[InputMap](InputMap())
        if not app.resources.has[InputStateResource]():
            app.insert_resource[InputStateResource](InputStateResource())
        if not app.resources.has[EvdevHubResource]():
            app.insert_resource[EvdevHubResource](
                EvdevHubResource(enabled=use_evdev, enable_hotplug=use_evdev)
            )
        if not app.resources.has[InputConfigResource]():
            app.insert_resource[InputConfigResource](InputConfigResource(use_evdev))

        ensure_headless_input_swap_hook(app)

        if not _has_input_system(app, STAGE_FIRST, "flint.input.update"):
            var after_win_id = -1
            for i in range(len(app.schedule.stages[STAGE_FIRST].systems)):
                if app.schedule.stages[STAGE_FIRST].systems[i].name == "flint.app.window_poll":
                    after_win_id = app.schedule.stages[STAGE_FIRST].systems[i].id
                    break

            _ = app.add_system(
                STAGE_FIRST,
                "flint.input.update",
                _sys_input_update,
                exclusive=True,
                after_system_id=after_win_id,
            )
        if not _has_input_system(app, STAGE_SHUTDOWN, "flint.input.shutdown"):
            _ = app.add_system(
                STAGE_SHUTDOWN,
                "flint.input.shutdown",
                _sys_input_shutdown,
                exclusive=True,
            )

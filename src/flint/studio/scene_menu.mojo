from std.ffi import external_call
from flint.platform.os import env_value, file_size, last_errno
from flint.studio.command_menu import CommandMenu

comptime SAVE = 1
comptime LOAD = 2
comptime UNDO = 3
comptime REDO = 4
comptime MOVE = 5
comptime ROTATE = 6
comptime SCALE = 7
comptime SNAP = 8
comptime PLAY = 9
comptime PAUSE = 10
comptime STOP = 11
comptime SOURCE = 12
comptime LIBRARY = 13
comptime KIMODO = 14
comptime MOTION_TAKES = 15
comptime NEW_SCENE = 16
comptime DUPLICATE = 17
comptime DELETE_SEL = 18
comptime GIZMO_SPACE = 19
comptime FRAME_SEL = 20
comptime STEP = 21
comptime SPAWN_EMPTY = 22
comptime SPAWN_CUBE = 23
comptime SPAWN_POINT_LIGHT = 24
comptime SPAWN_SPOT_LIGHT = 25
comptime SPAWN_CAMERA = 26
comptime SPAWN_AUDIO = 27
comptime BAKE_NAVMESH = 28
comptime TOGGLE_NAV_OVERLAY = 29
comptime TOGGLE_COL_OVERLAY = 30
comptime RELOAD_SO = 31
comptime PACKAGE = 32
comptime PACKAGE_PROFILES = 33
comptime OPEN_SCENE = 34
comptime SAVE_AS = 35
comptime RECOVER_AUTOSAVE = 36
comptime SELECT_ALL = 37
comptime RENAME = 38
comptime WINDOW_RESET_LAYOUT = 39
comptime WINDOW_CONSOLE = 40
comptime HELP_SHORTCUTS = 41
comptime HELP_PALETTE = 42
comptime PIVOT_MODE = 43


struct SceneMenu(Movable):
    var bar: CommandMenu
    var takes_pid: Int32
    var kimodo_pid: Int32
    var library_pid: Int32
    var has_project: Bool
    var has_launcher: Bool
    var message: String

    def __init__(out self, project_root: String = ""):
        self.bar = CommandMenu()
        self.takes_pid = -1
        self.kimodo_pid = -1
        self.library_pid = -1
        var proj = project_root if project_root.byte_length() > 0 else env_value("FLINT_STUDIO_PROJECT")
        self.has_launcher = file_size(env_value("FLINT_STUDIO_IDE_LAUNCHER")) > 0
        self.has_project = proj.byte_length() > 0 and file_size(proj + "/flint.toml") >= 0
        self.message = ""
        self.bar.add_group("File", 48)
        self.bar.item(SAVE, "Save scene", "Ctrl+S")
        self.bar.item(OPEN_SCENE, "Open scene...", "Ctrl+O")
        self.bar.item(NEW_SCENE, "New scene", "Ctrl+N")
        self.bar.item(SAVE_AS, "Save scene as...", "Ctrl+Shift+S")
        self.bar.item(LOAD, "Revert to saved")
        self.bar.item(RECOVER_AUTOSAVE, "Recover autosave")
        self.bar.add_group("Edit", 48)
        self.bar.item(UNDO, "Undo", "Ctrl+Z")
        self.bar.item(REDO, "Redo", "Ctrl+Shift+Z")
        self.bar.item(DUPLICATE, "Duplicate", "Ctrl+D")
        self.bar.item(DELETE_SEL, "Delete", "Del")
        self.bar.item(SELECT_ALL, "Select all", "Ctrl+A")
        self.bar.item(RENAME, "Rename", "F2")
        self.bar.add_group("Transform", 100)
        self.bar.item(MOVE, "Move", "W")
        self.bar.item(ROTATE, "Rotate", "E")
        self.bar.item(SCALE, "Scale", "R")
        self.bar.item(SNAP, "Snap to grid", "G")
        self.bar.item(GIZMO_SPACE, "Gizmo: World Space", "X")
        self.bar.item(FRAME_SEL, "Frame selection", "F")
        self.bar.item(PIVOT_MODE, "Pivot: Bounds Centre")
        self.bar.add_group("Play", 50)
        self.bar.item(PLAY, "Play scene", "F5")
        self.bar.item(PAUSE, "Pause simulation", "F6")
        self.bar.item(STOP, "Stop and restore", "Shift+F5")
        self.bar.item(STEP, "Step frame", "F7")
        self.bar.item(RELOAD_SO, "Reload .so")
        self.bar.add_group("Tools", 58)
        self.bar.item(LIBRARY, "Move Library", "", self.has_project and self.has_launcher)
        self.bar.item(SOURCE, "Source editor", "F8")
        self.bar.item(KIMODO, "Kimodo motion", "", self.has_launcher)

        self.bar.item(MOTION_TAKES, "Motion takes", "", self.has_launcher)
        self.bar.add_group("Scene", 58)
        self.bar.item(SPAWN_EMPTY, "Spawn Empty")
        self.bar.item(SPAWN_CUBE, "Spawn Cube")
        self.bar.item(SPAWN_POINT_LIGHT, "Spawn Point Light")
        self.bar.item(SPAWN_SPOT_LIGHT, "Spawn Spot Light")
        self.bar.item(SPAWN_CAMERA, "Spawn Camera")
        self.bar.item(SPAWN_AUDIO, "Spawn Audio Emitter")
        self.bar.add_group("NavMesh", 76)
        self.bar.item(BAKE_NAVMESH, "Bake NavMesh")
        self.bar.item(TOGGLE_NAV_OVERLAY, "Hide NavMesh Overlay")
        self.bar.item(TOGGLE_COL_OVERLAY, "Show Collider Overlay")
        self.bar.add_group("Build", 56)
        self.bar.item(PACKAGE, "Package project", "", self.has_project)
        self.bar.item(PACKAGE_PROFILES, "Package with profile...", "", self.has_project)
        self.bar.add_group("Window", 66)
        self.bar.item(WINDOW_RESET_LAYOUT, "Reset layout")
        self.bar.item(WINDOW_CONSOLE, "Console")
        self.bar.add_group("Help", 46)
        self.bar.item(HELP_SHORTCUTS, "Keyboard shortcuts", "F1")
        self.bar.item(HELP_PALETTE, "Command palette", "Ctrl+K")

    def __moveinit__(out self, deinit existing: Self):
        self.bar = existing.bar^
        self.takes_pid = existing.takes_pid
        existing.takes_pid = -1
        self.kimodo_pid = existing.kimodo_pid
        existing.kimodo_pid = -1
        self.library_pid = existing.library_pid
        self.has_project = existing.has_project
        self.has_launcher = existing.has_launcher
        self.message = existing.message^
        existing.library_pid = -1

    def __deinit__(deinit self):
        self.close()

    def open_library(mut self):
        self.poll()
        if not self.has_project or not self.has_launcher or self.library_pid > 0:
            return
        self.library_pid = _launch('cd -- "$FLINT_STUDIO_PROJECT" && exec bash "$FLINT_STUDIO_IDE_LAUNCHER" studio --moves')
        self.message = "Move Library is open" if self.library_pid > 0 else "Could not open Move Library"

    def open_kimodo(mut self):
        self.poll()
        if self.kimodo_pid <= 0:
            self.kimodo_pid = _launch('cd -- "$FLINT_STUDIO_PROJECT" && exec bash "$FLINT_STUDIO_IDE_LAUNCHER" studio --kimodo')
            self.message = "Kimodo is open" if self.kimodo_pid > 0 else "Could not open Kimodo"

    def open_takes(mut self):
        self.poll()
        if self.takes_pid <= 0:
            self.takes_pid = _launch('cd -- "$FLINT_STUDIO_PROJECT" && exec bash "$FLINT_STUDIO_IDE_LAUNCHER" studio --takes')
            self.message = "Motion takes is open" if self.takes_pid > 0 else "Could not open Motion takes"

    def close(mut self):
        _close_tool(self.takes_pid)
        _close_tool(self.library_pid)
        _close_tool(self.kimodo_pid)

    def poll(mut self):
        _poll_tool(self.takes_pid, self.message, "Motion takes")
        _poll_tool(self.library_pid, self.message, "Move Library")
        _poll_tool(self.kimodo_pid, self.message, "Kimodo")


def _launch(var command: String) -> Int32:
    var shell = String("/bin/bash")
    var opt = String("-c")
    var child = external_call["fork", Int32]()
    if child == 0:
        _ = external_call["setpgid", Int32](Int32(0), Int32(0))
        _ = external_call["close_range", Int32](UInt32(3), UInt32(0xFFFFFFFF), UInt32(0))
        _ = external_call["execl", Int32](
            shell.as_c_string_span(),
            shell.as_c_string_span(),
            opt.as_c_string_span(),
            command.as_c_string_span(),
            Int(0),
        )
        external_call["_exit", NoneType](Int32(127))
    return child


def _close_tool(mut child: Int32):
    if child <= 0:
        return
    var status = Int32(0)
    var pid = external_call["waitpid", Int32](child, Pointer(to=status), Int32(1))
    if pid == child or (pid < 0 and last_errno() == 10):
        child = -1
        return
    _ = external_call["kill", Int32](-child, Int32(15))
    _ = external_call["kill", Int32](child, Int32(15))
    for _ in range(20):
        pid = external_call["waitpid", Int32](child, Pointer(to=status), Int32(1))
        if pid == child or (pid < 0 and last_errno() == 10):
            child = -1
            return
        _ = external_call["usleep", Int32](UInt32(5000))
    _ = external_call["kill", Int32](-child, Int32(9))
    _ = external_call["kill", Int32](child, Int32(9))
    while True:
        pid = external_call["waitpid", Int32](child, Pointer(to=status), Int32(0))
        if pid == child or (pid < 0 and last_errno() != 4):
            break
    child = -1



def _poll_tool(mut child: Int32, mut message: String, name: String):
    if child > 0:
        var status = Int32(0)
        var pid = external_call["waitpid", Int32](child, Pointer(to=status), Int32(1))
        if pid == child or (pid < 0 and last_errno() == 10):
            message = "" if status == 0 else name + " closed with an error"
            child = -1

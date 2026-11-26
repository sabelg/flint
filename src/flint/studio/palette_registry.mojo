"""Flint Studio Command Palette Catalog & Key-Chord Specifications (`flint.studio.palette_registry`).

Pure-Mojo 1.2 registry decoupling Studio's full command, quick-spawn, workspace, theme,
code/build, entity, asset, and Visual Agent Studio catalog from `studio_app.mojo`:
  - `PAL_CMD_*`: Stable typed action IDs across File/Edit, Transform, PIE, Workspaces,
    Themes, Code/Build/Nav, Quick-Spawn, and Visual Agent Theatre.
  - `populate_studio_palette(...)`: Registers 60+ built-in commands/spawns/agent actions
    plus live `@` scene entities and `#` project assets with PIE-safety flags and
    canonical `ICON_*` glyphs.
  - `studio_chord_spec(...)` & `begin_studio_chord(...)`: Populates `KeyChordHud`
    follow-up hints for `Space`, `W/E/R`, `Ctrl+K`, `F5..F10`, and `Alt+A`.
"""
from std.collections import List
from flint.studio.palette import (
    KeyChordEntry,
    KeyChordHud,
    PALETTE_ITEM_AGENT,
    PALETTE_ITEM_ASSET,
    PALETTE_ITEM_COMMAND,
    PALETTE_ITEM_ENTITY,
    PALETTE_ITEM_SPAWN,
    PaletteItem,
    StudioCommandPalette,
)
from flint.ui.theme import (
    ICON_AGENT_SPARK,
    ICON_AUDIO,
    ICON_CAMERA,
    ICON_CHECK,
    ICON_CLOCK,
    ICON_CLOSE,
    ICON_CODE,
    ICON_COLLIDER,
    ICON_CUBE,
    ICON_DIFF,
    ICON_EMPTY,
    ICON_EYE,
    ICON_FOCUS_CAM,
    ICON_FOLIAGE,
    ICON_GRID,
    ICON_LIGHT_POINT,
    ICON_LIGHT_SPOT,
    ICON_LOCAL,
    ICON_MATERIAL,
    ICON_MOON,
    ICON_MOVE,
    ICON_PAUSE,
    ICON_PLAY,
    ICON_PLUS,
    ICON_PREFAB,
    ICON_REFRESH,
    ICON_REVERT,
    ICON_ROTATE,
    ICON_SCALE,
    ICON_SCENE,
    ICON_SNAP,
    ICON_STEP,
    ICON_STOP,
    ICON_SUN,
    ICON_WARN,
    ICON_WORLD,
    icon_for_path,
)

# ============================================================================
# 1. Stable Command Palette Action IDs (`PAL_CMD_*`)
# ============================================================================

# File & Scene (100..119)
comptime PAL_CMD_SAVE_SCENE: Int = 100
comptime PAL_CMD_LOAD_SCENE: Int = 101
comptime PAL_CMD_NEW_SCENE: Int = 102
comptime PAL_CMD_RELOAD_SCENE: Int = 103
comptime PAL_CMD_RECOVER_AUTOSAVE: Int = 104

# Edit & Hierarchy (120..149)
comptime PAL_CMD_UNDO: Int = 120
comptime PAL_CMD_REDO: Int = 121
comptime PAL_CMD_DUPLICATE: Int = 122
comptime PAL_CMD_DELETE: Int = 123
comptime PAL_CMD_RENAME: Int = 124
comptime PAL_CMD_COPY: Int = 125
comptime PAL_CMD_PASTE: Int = 126
comptime PAL_CMD_SELECT_ALL: Int = 127
comptime PAL_CMD_CLEAR_SELECTION: Int = 128
comptime PAL_CMD_UNPARENT: Int = 129
comptime PAL_CMD_PARENT_TO_PRIMARY: Int = 130

# Transform & Gizmo (150..179)
comptime PAL_CMD_TOOL_SELECT: Int = 150
comptime PAL_CMD_TOOL_MOVE: Int = 151
comptime PAL_CMD_TOOL_ROTATE: Int = 152
comptime PAL_CMD_TOOL_SCALE: Int = 153
comptime PAL_CMD_TOGGLE_GIZMO_SPACE: Int = 154
comptime PAL_CMD_TOGGLE_SNAP: Int = 155
comptime PAL_CMD_FRAME_SELECTION: Int = 156
comptime PAL_CMD_RESET_TRANSFORM: Int = 157

# PIE & Simulation (200..219)
comptime PAL_CMD_PIE_PLAY: Int = 200
comptime PAL_CMD_PIE_PAUSE: Int = 201
comptime PAL_CMD_PIE_STEP: Int = 202
comptime PAL_CMD_PIE_STOP: Int = 203
comptime PAL_CMD_PIE_EJECT_CAMERA: Int = 204

# Workspace Presets (250..259)
comptime PAL_CMD_WORKSPACE_EDIT: Int = 250
comptime PAL_CMD_WORKSPACE_AGENT_STUDIO: Int = 251
comptime PAL_CMD_WORKSPACE_CODE_SCENE: Int = 252
comptime PAL_CMD_WORKSPACE_LOOKDEV: Int = 253
comptime PAL_CMD_WORKSPACE_PLAY: Int = 254

# Themes (260..269)
comptime PAL_CMD_THEME_OBSIDIAN: Int = 260
comptime PAL_CMD_THEME_MIDNIGHT: Int = 261
comptime PAL_CMD_THEME_OKABE_ITO: Int = 262
comptime PAL_CMD_THEME_PAPER_LIGHT: Int = 263
comptime PAL_CMD_THEME_CYCLE: Int = 264

# Code, Build, NavMesh & View Overlays (300..349)
comptime PAL_CMD_TOGGLE_CODE_DRAWER: Int = 300
comptime PAL_CMD_BUILD_HOT_RELOAD: Int = 301
comptime PAL_CMD_SHOW_DIAGNOSTICS: Int = 302
comptime PAL_CMD_BAKE_NAVMESH: Int = 303
comptime PAL_CMD_PACKAGE_PROJECT: Int = 304
comptime PAL_CMD_PACKAGE_PROFILES: Int = 305
comptime PAL_CMD_TOGGLE_NAVMESH_OVERLAY: Int = 310
comptime PAL_CMD_TOGGLE_COLLIDER_OVERLAY: Int = 311
comptime PAL_CMD_TOGGLE_GRID_OVERLAY: Int = 312
comptime PAL_CMD_TOGGLE_ORTHO_CAMERA: Int = 313
comptime PAL_CMD_TOGGLE_PERF_SPARKLINE: Int = 314
comptime PAL_CMD_RESET_CAMERA: Int = 315
comptime PAL_CMD_SNAP_VIEW_FRONT: Int = 316
comptime PAL_CMD_SNAP_VIEW_RIGHT: Int = 317
comptime PAL_CMD_SNAP_VIEW_TOP: Int = 318
comptime PAL_CMD_RESCAN_ASSETS: Int = 319

# Quick-Spawn Primitives & Objects (400..429)
comptime PAL_CMD_SPAWN_EMPTY: Int = 400
comptime PAL_CMD_SPAWN_CUBE: Int = 401
comptime PAL_CMD_SPAWN_SPHERE: Int = 402
comptime PAL_CMD_SPAWN_PLANE: Int = 403
comptime PAL_CMD_SPAWN_POINT_LIGHT: Int = 404
comptime PAL_CMD_SPAWN_SPOT_LIGHT: Int = 405
comptime PAL_CMD_SPAWN_SUN_LIGHT: Int = 406
comptime PAL_CMD_SPAWN_CAMERA: Int = 407
comptime PAL_CMD_SPAWN_AUDIO: Int = 408
comptime PAL_CMD_SPAWN_PHYSICS_BOX: Int = 409
comptime PAL_CMD_SPAWN_TRIGGER: Int = 410
comptime PAL_CMD_SPAWN_PREFAB: Int = 411

# Visual Agent Studio & Theatre (500..529)
comptime PAL_CMD_AGENT_SCENE_BUILDER: Int = 500
comptime PAL_CMD_AGENT_GAMEPLAY_CODER: Int = 501
comptime PAL_CMD_AGENT_LOOKDEV_ARTIST: Int = 502
comptime PAL_CMD_AGENT_QA_PLAYTESTER: Int = 503
comptime PAL_CMD_AGENT_TOGGLE_PANEL: Int = 504
comptime PAL_CMD_AGENT_TOGGLE_FOLLOW_CAM: Int = 505
comptime PAL_CMD_AGENT_TOGGLE_THEATRE_3D: Int = 506
comptime PAL_CMD_AGENT_WATCH_DEMO: Int = 507
comptime PAL_CMD_AGENT_APPROVE_STEP: Int = 508
comptime PAL_CMD_AGENT_REJECT_STEP: Int = 509
comptime PAL_CMD_AGENT_PAUSE_RESUME: Int = 510
comptime PAL_CMD_AGENT_REVERT_LAST_STEP: Int = 511
comptime PAL_CMD_AGENT_EXIT_TIME_TRAVEL: Int = 512

# Chord-only utility actions
comptime PAL_CMD_OPEN_PALETTE: Int = 600
comptime PAL_CMD_OPEN_QUICK_SPAWN: Int = 601


# ============================================================================
# 2. Internal String / Icon Helpers
# ============================================================================

def _ascii_lower(s: String) -> String:
    var src = s.as_bytes()
    var out = List[UInt8](capacity=len(src))
    for i in range(len(src)):
        var b = src[i]
        if b >= 65 and b <= 90:
            out.append(b + 32)
        else:
            out.append(b)
    return String(unsafe_from_utf8=out)


def _split_asset_path(path: String) -> Tuple[String, String]:
    """Splits `"assets/meshes/hero.glb"` into `("hero.glb", "assets/meshes")`."""
    var clean = String(path.strip())
    var bytes = clean.as_bytes()
    var last_sep = -1
    for i in range(len(bytes)):
        if bytes[i] == 47 or bytes[i] == 92:
            last_sep = i
    if last_sep < 0:
        return (clean, String(""))
    var base = String(clean[byte=last_sep + 1:])
    var dir = String(clean[byte=0:last_sep])
    return (base, dir)


def _icon_for_entity_name(name: String) -> Int:
    var low = _ascii_lower(name)
    if low.find("spot") >= 0:
        return ICON_LIGHT_SPOT
    if low.find("light") >= 0 or low.find("sun") >= 0 or low.find("lamp") >= 0:
        return ICON_LIGHT_POINT
    if low.find("cam") >= 0:
        return ICON_CAMERA
    if low.find("audio") >= 0 or low.find("sound") >= 0 or low.find("sfx") >= 0:
        return ICON_AUDIO
    if low.find("foliage") >= 0 or low.find("tree") >= 0 or low.find("canopy") >= 0 or low.find("oak") >= 0:
        return ICON_FOLIAGE
    if low.find("grid") >= 0 or low.find("floor") >= 0 or low.find("ground") >= 0:
        return ICON_GRID
    if low.find("prefab") >= 0:
        return ICON_PREFAB
    if low.find("trigger") >= 0 or low.find("collider") >= 0:
        return ICON_COLLIDER
    if low.find("empty") >= 0 or low.find("root") >= 0:
        return ICON_EMPTY
    return ICON_CUBE


# ============================================================================
# 3. Studio Command Palette Catalog Populator (60+ Built-ins + Live Entities & Assets)
# ============================================================================

def populate_studio_palette(
    mut pal: StudioCommandPalette,
    entity_names: List[String],
    entity_ids: List[Int],
    asset_paths: List[String],
    pie_active: Bool,
    can_undo: Bool = True,
    can_redo: Bool = True,
    has_selection: Bool = True,
    has_project: Bool = True,
    gizmo_local: Bool = False,
    snap_enabled: Bool = True,
    show_navmesh: Bool = True,
    show_colliders: Bool = False,
    follow_agent_cam: Bool = False,
    show_agent_overlays: Bool = True,
):
    """Populates `pal` with Flint Studio's full command, quick-spawn, agent, entity & asset catalog."""
    pal.clear_items()
    pal.pie_active = pie_active

    # ---- File & Scene ----
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_SAVE_SCENE, "Save Scene", "File", "Ctrl+S", ICON_SCENE, "cmd:save", True, True, "Save Scene File Ctrl+S"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_LOAD_SCENE, "Load Saved Scene", "File", "Ctrl+O", ICON_SCENE, "cmd:load", True, True, "Load Open Scene File"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_NEW_SCENE, "New Scene", "File", "Ctrl+N", ICON_PLUS, "cmd:new", True, True, "New Empty Scene File"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_RELOAD_SCENE, "Reload Scene from Disk", "File", "", ICON_REFRESH, "cmd:reload_scene", True, True, "Reload Revert Scene Disk"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_RECOVER_AUTOSAVE, "Recover Autosave", "File", "", ICON_REFRESH, "cmd:recover_autosave", True, True, "Recover Autosave Crash Restore Backup"))

    # ---- Edit & Hierarchy ----
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_UNDO, "Undo", "Edit", "Ctrl+Z", ICON_REVERT, "cmd:undo", True, can_undo, "Undo Edit Transaction"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_REDO, "Redo", "Edit", "Ctrl+Shift+Z", ICON_REFRESH, "cmd:redo", True, can_redo, "Redo Edit Transaction"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_DUPLICATE, "Duplicate Selected", "Edit", "Ctrl+D", ICON_CUBE, "cmd:dup", True, has_selection, "Duplicate Clone Entity"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_DELETE, "Delete Selected", "Edit", "Del", ICON_CLOSE, "cmd:del", True, has_selection, "Delete Remove Despawn Entity"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_RENAME, "Rename Selected Entity", "Edit", "F2", ICON_CODE, "cmd:rename", True, has_selection, "Rename Entity Quick F2"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_COPY, "Copy Selected", "Edit", "Ctrl+C", ICON_DIFF, "cmd:copy", False, has_selection, "Copy Entity Clipboard"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PASTE, "Paste Entities", "Edit", "Ctrl+V", ICON_PLUS, "cmd:paste", True, True, "Paste Entity Clipboard"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_SELECT_ALL, "Select All Entities", "Edit", "Ctrl+A", ICON_CHECK, "cmd:sel_all", False, True, "Select All Scene Entities"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_CLEAR_SELECTION, "Clear Selection", "Edit", "Esc", ICON_CLOSE, "cmd:sel_clear", False, has_selection, "Deselect Clear Selection"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_UNPARENT, "Unparent Selected to Root", "Hierarchy", "", ICON_EMPTY, "cmd:unparent", True, has_selection, "Unparent Detach Root Hierarchy"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PARENT_TO_PRIMARY, "Parent Selected to Primary", "Hierarchy", "", ICON_PREFAB, "cmd:parent_pri", True, has_selection, "Parent Attach Reparent Hierarchy"))

    # ---- Transform & Gizmo ----
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOOL_SELECT, "Select Tool", "Transform", "Q", ICON_EMPTY, "cmd:tool_select", False, True, "Select Cursor Pick Q"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOOL_MOVE, "Move / Translate Tool", "Transform", "W", ICON_MOVE, "cmd:tool_move", False, True, "Move Translate Gizmo W"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOOL_ROTATE, "Rotate Tool", "Transform", "E", ICON_ROTATE, "cmd:tool_rotate", False, True, "Rotate Orbit Gizmo E"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOOL_SCALE, "Scale Tool", "Transform", "R", ICON_SCALE, "cmd:tool_scale", False, True, "Scale Resize Gizmo R"))
    var space_lbl = String("Gizmo Space: Switch to World") if gizmo_local else String("Gizmo Space: Switch to Local")
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOGGLE_GIZMO_SPACE, space_lbl, "Transform", "X", ICON_WORLD if gizmo_local else ICON_LOCAL, "cmd:gizmo_space", False, True, "Gizmo Space World Local X"))
    var snap_lbl = String("Disable Grid Snapping") if snap_enabled else String("Enable Grid Snapping")
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOGGLE_SNAP, snap_lbl, "Transform", "G", ICON_SNAP, "cmd:snap", False, True, "Snap Grid Angle Magnet G"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_FRAME_SELECTION, "Frame Selected Entity", "Transform", "F", ICON_FOCUS_CAM, "cmd:frame", False, has_selection, "Frame Focus Camera Selection F"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_RESET_TRANSFORM, "Reset Selected Transform", "Transform", "", ICON_REVERT, "cmd:reset_trs", True, has_selection, "Reset Transform Position Rotation Scale"))

    # ---- Play-In-Editor (PIE) ----
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PIE_PLAY, "Play / Resume Scene (PIE)", "Play", "F5", ICON_PLAY, "cmd:pie_play", False, True, "Play Start Resume PIE Simulation F5"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PIE_PAUSE, "Pause Simulation (PIE)", "Play", "F6", ICON_PAUSE, "cmd:pie_pause", False, pie_active, "Pause PIE Simulation F6"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PIE_STEP, "Step Single Frame (PIE)", "Play", "F7", ICON_STEP, "cmd:pie_step", False, True, "Step Frame PIE Simulation F7"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PIE_STOP, "Stop Simulation & Restore", "Play", "Shift+F5", ICON_STOP, "cmd:pie_stop", False, pie_active, "Stop Restore PIE Simulation"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PIE_EJECT_CAMERA, "Eject / Possess Game Camera", "Play", "Shift+F1", ICON_CAMERA, "cmd:pie_eject", False, pie_active, "Eject Possess PIE Camera"))

    # ---- Workspace Presets (All 5) ----
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_WORKSPACE_EDIT, "Workspace: Edit", "Layout", "Ctrl+1", ICON_SCENE, "cmd:ws_edit", False, True, "Workspace Preset Edit Layout"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_WORKSPACE_AGENT_STUDIO, "Workspace: Agent Studio", "Layout", "Ctrl+2", ICON_AGENT_SPARK, "cmd:ws_agent", False, True, "Workspace Preset Agent Studio AI"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_WORKSPACE_CODE_SCENE, "Workspace: Code + Scene", "Layout", "Ctrl+3", ICON_CODE, "cmd:ws_code", False, True, "Workspace Preset Code Scene Split IDE"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_WORKSPACE_LOOKDEV, "Workspace: Lookdev", "Layout", "Ctrl+4", ICON_MATERIAL, "cmd:ws_lookdev", False, True, "Workspace Preset Lookdev Lighting Material"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_WORKSPACE_PLAY, "Workspace: Play Focus", "Layout", "Ctrl+5", ICON_PLAY, "cmd:ws_play", False, True, "Workspace Preset Play Viewport"))

    # ---- Themes (All 4 + Cycle) ----
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_THEME_OBSIDIAN, "Theme: Flint Obsidian", "Appearance", "", ICON_MOON, "cmd:theme_obsidian", False, True, "Theme Dark Flint Obsidian"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_THEME_MIDNIGHT, "Theme: Midnight Studio", "Appearance", "", ICON_MOON, "cmd:theme_midnight", False, True, "Theme Dark Midnight Studio Neon"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_THEME_OKABE_ITO, "Theme: Okabe-Ito High-Clarity", "Appearance", "", ICON_EYE, "cmd:theme_okabe", False, True, "Theme Colorblind CVD Okabe Ito"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_THEME_PAPER_LIGHT, "Theme: Paper Studio Light", "Appearance", "", ICON_SUN, "cmd:theme_light", False, True, "Theme Light Paper Daylight"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_THEME_CYCLE, "Cycle Studio Theme", "Appearance", "", ICON_SUN, "cmd:theme_cycle", False, True, "Theme Switch Toggle Cycle"))

    # ---- Code, Build, NavMesh & View Overlays ----
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOGGLE_CODE_DRAWER, "Toggle Code & Diff Drawer", "Code", "F8", ICON_CODE, "cmd:code_drawer", False, True, "Code Drawer Editor Source Diff F8"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_BUILD_HOT_RELOAD, "Build & Hot-Reload Plugin", "Build", "Ctrl+B", ICON_REFRESH, "cmd:build_reload", False, has_project, "Build Compile Hot Reload SO Plugin"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_SHOW_DIAGNOSTICS, "Show Compiler Diagnostics", "Code", "", ICON_WARN, "cmd:diagnostics", False, True, "Problems Errors Warnings Diagnostics"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_BAKE_NAVMESH, "Bake NavMesh", "Navigation", "", ICON_GRID, "cmd:bake_nav", True, True, "Bake NavMesh Recast Pathfinding"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PACKAGE_PROJECT, "Package Project", "Build", "", ICON_PREFAB, "cmd:package", True, has_project, "Package Build Ship Release Project"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_PACKAGE_PROFILES, "Package Project with Profiles", "Build", "", ICON_PREFAB, "cmd:package_prof", True, has_project, "Package Profiles Ship Release"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOGGLE_NAVMESH_OVERLAY, "Hide NavMesh Overlay" if show_navmesh else "Show NavMesh Overlay", "View", "", ICON_GRID, "cmd:nav_ov", False, True, "Toggle NavMesh Debug Overlay"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOGGLE_COLLIDER_OVERLAY, "Hide Collider Overlay" if show_colliders else "Show Collider Overlay", "View", "", ICON_COLLIDER, "cmd:col_ov", False, True, "Toggle Physics Collider Overlay"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOGGLE_GRID_OVERLAY, "Toggle Reference Grid", "View", "", ICON_GRID, "cmd:grid_ov", False, True, "Toggle Viewport Reference Grid"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOGGLE_ORTHO_CAMERA, "Toggle Perspective / Orthographic", "Camera", "", ICON_CAMERA, "cmd:cam_ortho", False, True, "Toggle Perspective Orthographic Projection"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_TOGGLE_PERF_SPARKLINE, "Toggle GPU Performance Details", "View", "", ICON_CLOCK, "cmd:perf", False, True, "GPU Perf FPS Sparkline Stats"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_RESET_CAMERA, "Reset Viewport Camera", "Camera", "", ICON_CAMERA, "cmd:cam_reset", False, True, "Reset Orbit Camera Default"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_SNAP_VIEW_FRONT, "Snap Camera: Front View", "Camera", "", ICON_FOCUS_CAM, "cmd:view_front", False, True, "Camera Snap Front View Z"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_SNAP_VIEW_RIGHT, "Snap Camera: Right View", "Camera", "", ICON_FOCUS_CAM, "cmd:view_right", False, True, "Camera Snap Right View X"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_SNAP_VIEW_TOP, "Snap Camera: Top View", "Camera", "", ICON_FOCUS_CAM, "cmd:view_top", False, True, "Camera Snap Top Down View Y"))
    pal.add_item(PaletteItem(PALETTE_ITEM_COMMAND, PAL_CMD_RESCAN_ASSETS, "Rescan Project Assets", "Assets", "", ICON_REFRESH, "cmd:rescan_assets", False, True, "Rescan Refresh Asset Browser"))

    # ---- Quick-Spawn (`+`) ----
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_EMPTY, "Spawn Empty Entity", "Quick Spawn", "", ICON_EMPTY, "spawn:empty", True, True, "Spawn Create Empty Node"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_CUBE, "Spawn Cube", "Quick Spawn", "", ICON_CUBE, "spawn:cube", True, True, "Spawn Create Cube Box Mesh"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_SPHERE, "Spawn Sphere", "Quick Spawn", "", ICON_MATERIAL, "spawn:sphere", True, True, "Spawn Create Sphere Orb Mesh"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_PLANE, "Spawn Ground Plane", "Quick Spawn", "", ICON_GRID, "spawn:plane", True, True, "Spawn Create Floor Plane Ground"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_POINT_LIGHT, "Spawn Point Light", "Quick Spawn", "", ICON_LIGHT_POINT, "spawn:point_light", True, True, "Spawn Create Point Light"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_SPOT_LIGHT, "Spawn Spot Light", "Quick Spawn", "", ICON_LIGHT_SPOT, "spawn:spot_light", True, True, "Spawn Create Spot Light Cone"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_SUN_LIGHT, "Spawn Directional Sun", "Quick Spawn", "", ICON_SUN, "spawn:sun", True, True, "Spawn Create Sun Directional Environment Light"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_CAMERA, "Spawn Camera", "Quick Spawn", "", ICON_CAMERA, "spawn:camera", True, True, "Spawn Create Game Camera"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_AUDIO, "Spawn Audio Emitter", "Quick Spawn", "", ICON_AUDIO, "spawn:audio", True, True, "Spawn Create Audio Sound Emitter"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_PHYSICS_BOX, "Spawn Physics RigidBody Box", "Quick Spawn", "", ICON_COLLIDER, "spawn:phys_box", True, True, "Spawn Create Dynamic Physics RigidBody Box"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_TRIGGER, "Spawn Trigger Volume", "Quick Spawn", "", ICON_COLLIDER, "spawn:trigger", True, True, "Spawn Create Trigger Sensor Volume"))
    pal.add_item(PaletteItem(PALETTE_ITEM_SPAWN, PAL_CMD_SPAWN_PREFAB, "Spawn Prefab Instance", "Quick Spawn", "", ICON_PREFAB, "spawn:prefab", True, True, "Spawn Instantiate Prefab"))

    # ---- Visual Agent Studio & Theatre ----
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_SCENE_BUILDER, "Agent: Launch Scene Builder", "Agent Studio", "", ICON_AGENT_SPARK, "agent:scene_builder", False, True, "AI Agent Launch Scene Builder Level Layout"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_GAMEPLAY_CODER, "Agent: Launch Gameplay Coder", "Agent Studio", "", ICON_CODE, "agent:code_author", False, True, "AI Agent Launch Gameplay Coder Mojo Script"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_LOOKDEV_ARTIST, "Agent: Launch Lookdev Artist", "Agent Studio", "", ICON_MATERIAL, "agent:lookdev", False, True, "AI Agent Launch Lookdev Lighting Material Artist"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_QA_PLAYTESTER, "Agent: Launch QA Playtester", "Agent Studio", "", ICON_PLAY, "agent:qa", False, True, "AI Agent Launch QA Playtester Verify"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_TOGGLE_PANEL, "Agent: Toggle Mission Control", "Agent Studio", "Alt+A", ICON_AGENT_SPARK, "agent:panel", False, True, "AI Agent Toggle Mission Control Panel Alt+A"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_TOGGLE_FOLLOW_CAM, "Agent: Disable Follow Camera" if follow_agent_cam else "Agent: Enable Follow Camera", "Agent Studio", "", ICON_FOCUS_CAM, "agent:follow_cam", False, True, "AI Agent Follow Camera Auto Frame"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_TOGGLE_THEATRE_3D, "Agent: Hide 3D Theatre Overlays" if show_agent_overlays else "Agent: Show 3D Theatre Overlays", "Agent Studio", "", ICON_EYE, "agent:theatre_3d", False, True, "AI Agent Toggle 3D Viewport Theatre Overlays"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_WATCH_DEMO, "Agent: Watch Visual Agent Demo", "Agent Studio", "", ICON_AGENT_SPARK, "agent:demo", False, True, "AI Agent Watch Live Visual Demo Showcase"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_APPROVE_STEP, "Agent: Approve Pending Step", "Agent Studio", "", ICON_CHECK, "agent:approve", False, True, "AI Agent Approve Pending Tool Step"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_REJECT_STEP, "Agent: Reject Pending Step", "Agent Studio", "", ICON_CLOSE, "agent:reject", False, True, "AI Agent Reject Deny Pending Step"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_PAUSE_RESUME, "Agent: Pause / Resume Run", "Agent Studio", "", ICON_PAUSE, "agent:pause", False, True, "AI Agent Pause Resume Execution"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_REVERT_LAST_STEP, "Agent: Revert Last Step", "Agent Studio", "", ICON_REVERT, "agent:revert", True, True, "AI Agent Undo Revert Last Step Checkpoint"))
    pal.add_item(PaletteItem(PALETTE_ITEM_AGENT, PAL_CMD_AGENT_EXIT_TIME_TRAVEL, "Agent: Return to Live Tip", "Agent Studio", "", ICON_CLOCK, "agent:live", False, True, "AI Agent Exit Time Travel Scrubber Live"))

    # ---- Live Scene Entities (`@`) ----
    var n_ent = min(len(entity_names), len(entity_ids))
    for i in range(n_ent):
        var ename = entity_names[i]
        var eid = entity_ids[i]
        var disp = ename if ename.byte_length() > 0 else ("Entity_" + String(eid))
        pal.add_item(
            PaletteItem(
                PALETTE_ITEM_ENTITY,
                eid,
                disp,
                "Entity #" + String(eid),
                "",
                _icon_for_entity_name(disp),
                "ent:" + String(eid),
                False,
                True,
                disp + "  #" + String(eid),
            )
        )

    # ---- Project Assets (`#`) ----
    for i in range(len(asset_paths)):
        var apath = asset_paths[i]
        if apath.byte_length() == 0:
            continue
        var split = _split_asset_path(apath)
        var base = split[0]
        var folder = split[1]
        pal.add_item(
            PaletteItem(
                PALETTE_ITEM_ASSET,
                i,
                base,
                folder,
                "",
                icon_for_path(apath, False),
                "asset:" + apath,
                False,
                True,
                base + "  " + apath,
            )
        )

    pal.refilter()


# ============================================================================
# 4. Non-Blocking Key-Chord Hint Specifications (`KeyChordHud`)
# ============================================================================

def studio_chord_spec(
    trigger_key: String,
    gizmo_local: Bool = False,
    snap_enabled: Bool = True,
    follow_agent_cam: Bool = False,
    show_agent_overlays: Bool = True,
) -> List[KeyChordEntry]:
    """Builds `KeyChordHud` entries for `Space`, `W/E/R`, `Ctrl+K`, `F5..F10`, and `Alt+A`."""
    var out = List[KeyChordEntry]()
    var key = _ascii_lower(String(trigger_key.strip()))

    if key == "space":
        out.append(KeyChordEntry("K", "Command Palette", PAL_CMD_OPEN_PALETTE))
        out.append(KeyChordEntry("S", "Quick Spawn", PAL_CMD_OPEN_QUICK_SPAWN))
        out.append(KeyChordEntry("A", "Agent Studio", PAL_CMD_WORKSPACE_AGENT_STUDIO))
        out.append(KeyChordEntry("C", "Code + Scene", PAL_CMD_WORKSPACE_CODE_SCENE))
        out.append(KeyChordEntry("P", "Play Scene", PAL_CMD_PIE_PLAY))
        out.append(KeyChordEntry("F", "Frame Selection", PAL_CMD_FRAME_SELECTION))
        out.append(KeyChordEntry("G", "Snap to Grid", PAL_CMD_TOGGLE_SNAP, 1 if snap_enabled else 0))
        return out^

    if key == "w" or key == "e" or key == "r":
        out.append(KeyChordEntry("W", "Move Tool", PAL_CMD_TOOL_MOVE))
        out.append(KeyChordEntry("E", "Rotate Tool", PAL_CMD_TOOL_ROTATE))
        out.append(KeyChordEntry("R", "Scale Tool", PAL_CMD_TOOL_SCALE))
        out.append(KeyChordEntry("X", "Local Space", PAL_CMD_TOGGLE_GIZMO_SPACE, 1 if gizmo_local else 0))
        out.append(KeyChordEntry("G", "Grid Snap", PAL_CMD_TOGGLE_SNAP, 1 if snap_enabled else 0))
        out.append(KeyChordEntry("F", "Frame Selection", PAL_CMD_FRAME_SELECTION))
        out.append(KeyChordEntry("0", "Reset Transform", PAL_CMD_RESET_TRANSFORM))
        return out^

    if key == "ctrl+k":
        out.append(KeyChordEntry("1", "Workspace: Edit", PAL_CMD_WORKSPACE_EDIT))
        out.append(KeyChordEntry("2", "Workspace: Agent Studio", PAL_CMD_WORKSPACE_AGENT_STUDIO))
        out.append(KeyChordEntry("3", "Workspace: Code + Scene", PAL_CMD_WORKSPACE_CODE_SCENE))
        out.append(KeyChordEntry("4", "Workspace: Lookdev", PAL_CMD_WORKSPACE_LOOKDEV))
        out.append(KeyChordEntry("5", "Workspace: Play", PAL_CMD_WORKSPACE_PLAY))
        out.append(KeyChordEntry("T", "Cycle Theme", PAL_CMD_THEME_CYCLE))
        out.append(KeyChordEntry("S", "Save Scene", PAL_CMD_SAVE_SCENE))
        return out^

    if key == "f5" or key == "f6" or key == "f7" or key == "f8" or key == "f9" or key == "f10":
        out.append(KeyChordEntry("F5", "Play / Stop PIE", PAL_CMD_PIE_PLAY))
        out.append(KeyChordEntry("F6", "Pause / Resume PIE", PAL_CMD_PIE_PAUSE))
        out.append(KeyChordEntry("F7", "Step Frame", PAL_CMD_PIE_STEP))
        out.append(KeyChordEntry("F8", "Code & Diff Drawer", PAL_CMD_TOGGLE_CODE_DRAWER))
        out.append(KeyChordEntry("S-F1", "Eject Camera", PAL_CMD_PIE_EJECT_CAMERA))
        out.append(KeyChordEntry("C-B", "Build & Hot-Reload", PAL_CMD_BUILD_HOT_RELOAD))
        return out^

    if key == "alt+a":
        out.append(KeyChordEntry("1", "Scene Builder Agent", PAL_CMD_AGENT_SCENE_BUILDER))
        out.append(KeyChordEntry("2", "Gameplay Coder Agent", PAL_CMD_AGENT_GAMEPLAY_CODER))
        out.append(KeyChordEntry("3", "Lookdev Artist Agent", PAL_CMD_AGENT_LOOKDEV_ARTIST))
        out.append(KeyChordEntry("4", "QA Playtester Agent", PAL_CMD_AGENT_QA_PLAYTESTER))
        out.append(KeyChordEntry("F", "Follow-Agent Camera", PAL_CMD_AGENT_TOGGLE_FOLLOW_CAM, 1 if follow_agent_cam else 0))
        out.append(KeyChordEntry("O", "3D Theatre Overlays", PAL_CMD_AGENT_TOGGLE_THEATRE_3D, 1 if show_agent_overlays else 0))
        out.append(KeyChordEntry("D", "Watch Demo Agent", PAL_CMD_AGENT_WATCH_DEMO))
        out.append(KeyChordEntry("Y", "Approve Step", PAL_CMD_AGENT_APPROVE_STEP))
        out.append(KeyChordEntry("N", "Reject Step", PAL_CMD_AGENT_REJECT_STEP))
        out.append(KeyChordEntry("Z", "Revert Last Step", PAL_CMD_AGENT_REVERT_LAST_STEP))
        return out^

    return out^


def begin_studio_chord(
    mut hud: KeyChordHud,
    trigger_key: String,
    gizmo_local: Bool = False,
    snap_enabled: Bool = True,
    follow_agent_cam: Bool = False,
    show_agent_overlays: Bool = True,
) -> Bool:
    """Starts `hud` with the chord spec for `trigger_key`. Returns `True` if hints exist."""
    var entries = studio_chord_spec(
        trigger_key,
        gizmo_local=gizmo_local,
        snap_enabled=snap_enabled,
        follow_agent_cam=follow_agent_cam,
        show_agent_overlays=show_agent_overlays,
    )
    if len(entries) == 0:
        hud.cancel()
        return False
    hud.begin_chord(trigger_key, entries^)
    return True

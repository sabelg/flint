# Flint Studio Roadmap: Making a Whole Game in Studio

**Goal:** a person can make a complete game in Flint Studio and ship it, without leaving Studio and without writing scene-setup code. That means importing art, placing a level, writing and hot-reloading gameplay, playtesting in the editor, and packaging a build.

**Source:** an expert review of Studio on `main` (`7f64fbb`), written 2026-11-21; checkboxes last updated 2026-11-26 on `studio-redesign`. Five reviewers took part: an editor UX lead, a gameplay programmer, a technical artist and animator, a level designer who also covered UI and audio, and a build/release engineer. Code references are approximate (`~`) because the files change often.

---

## 1. Where we are

**Verdict:** you can't make a whole game in Studio yet. Studio is a strong editor for laying out scenes, testing physics and checking the look. The engine underneath is mostly real code, not stubs, but Studio isn't connected to it.

**What works today:**
- **Editor core:** undo/redo that merges rapid edits and keeps up to 512 steps or 64 MB; GPU picking without stalling the frame; move/rotate/scale gizmos that work on multi-selections and snap to grid and angle; a hierarchy you can reparent by drag-and-drop.
- **Scene files and play-in-editor:** deterministic `.flscn` JSON and LZ4 `.flscnb` scenes. Play-in-editor (PIE) saves the world when you press play and restores it on stop, checked with an fnv64 hash.
- **Automation:** the MCP v2 server with 31 tools (`devtools/mcp_v2.mojo`; count the `{"name":...}` entries in its `tools/list` reply).
- **Engine runtimes** (not yet exposed in Studio):
  - **Assets and rendering:** GLB loading with skins, morph targets and animation (`assets/glb_native.mojo`); GPU skinning; a one-call model uploader (`render/aaa_renderer.mojo:~4114`).
  - **Animation:** animation graphs (`anim/graph.mojo`), sockets, ragdolls, and the MotionBricks inbetweening runtime.
  - **Gameplay systems:** the ECS scheduler (`app/engine_app.mojo:~569`), `Events[T]`, trigger events (`physics/events.mojo`), `InputMap` with rebinding and JSON save/load, `BehaviorTree` and `UtilityAiScorer`, and world snapshots.
  - **Level systems:** NavMesh built from physics, including heightfields (`nav/navmesh.mojo:~574`), and `HeightfieldCollider`.
  - **Audio and UI:** `AudioServer` (buses, reverb zones, HRTF, LUFS limiter) and the `UiContext` toolkit (Flexbox, nine-slice, SDF text, gamepad focus).
  - **Code reload:** the `.so` hot-reload host (`app/hot_reload.mojo`).
  - **Shipping:** `flint.sh package` (Linux bundle plus `.flpk` and manifest) and the incremental SHA-256 cooker.

### The three breaks

1. **Content can't come in.**
   - Scenes store runtime GPU slot numbers, not asset references (`render/plugin.mojo:~112`).
   - Clicking a `.glb`, `.png` or `.wav` in the asset browser spawns a placeholder box with `mesh_id=1` (`studio/studio_app.mojo:~4150-4170`).
   - Studio's GPU meshes are a fixed table of 6 built at startup (`examples/studio/main.mojo:~227-250`).
   - The cooked mesh format drops skins and animations (`assets/server.mojo:~94`).
2. **Game logic can't run.**
   - PIE runs only Studio's physics stages and never `STAGE_UPDATE` (`studio_app.mojo:~3054`).
   - The animation and audio plugins aren't installed.
   - `HotReloadHost` is used only by a test.
   - The in-Studio IDE is the external `tmide` running in a terminal; its Build/Run buttons only send keystrokes to it.
   - Editor hotkeys stay live during play, including Esc, which quits Studio. There's no game camera.
3. **Scenes don't reach the shipped game.**
   - The `flint.sh new` scaffold builds its world in code and never loads `.flscn`.
   - Studio runs from the engine repo (`cd "$ROOT"` in `flint.sh`) and saves to `build/studio_scene.flscn`.
   - There's no Build or Package command in Studio or MCP.
   - Schemas aren't versioned: `read_bin` ignores its version argument, and an unknown component is a hard error when loading.

### Game names leaking into the engine (breaks the standalone rule in `AGENTS.md`)
- `core/cvars.mojo:~37-53` mentions "Maroon".
- `anim/graph.mojo:~41` defaults to `assets/deepsong/anim/maroon.json`.
- About 11 crash and log paths in `src/` hard-code `sharddepth`.
- The legacy ragdoll is a 19-joint template sized for one game's character.
- `core/crash.mojo` refers to `tools/symbolize` and `tools/replay.mojo`, which don't exist. The crash handler is off by default.

---

## 2. Ranked gaps

| # | Gap | Effort |
|---|---|---|
| 1 | **Asset references in scenes plus real GLB import.** Add an `AssetRef` (path or uuid) on mesh and material components, upload through the model uploader, resolve references when a scene loads, and keep skins and animations in the cooked format. Everything else depends on this. | L |
| 2 | **Gameplay in PIE.** Run `STAGE_UPDATE`. Load a project gameplay `.so` through `HotReloadHost`, with a `flint.sh` target that builds it. Add a play-mode input context and game-camera possession, and install the animation and audio plugins. | L |
| 3 | **Studio → game.** Make Studio project-rooted (`<project>/scenes/`) and have the scaffold load `assets/main.flscn`. Add a Build/Package menu and MCP tools `scene.save`, `scene.load` and `project.package`. | M |
| 4 | **Generic inspector.** Drive it from `ComponentRegistry` codecs and field reflection (`scene/scene.mojo:~147-243`) instead of today's hard-coded `StudioEntityView`. Add Add/Remove Component, and let project components register with Studio. | M |
| 5 | **Version every format and migrate old files.** Make `read_bin` honour its version argument, keep unknown components as opaque blobs, and add migration hooks. | M |
| 6 | **World panels.** Make lights entities, replacing the two hard-coded per-frame point lights (`studio_app.mojo:~3279`). Add sun, sky and exposure controls, an audio emitter and bus mixer, a NavMesh bake with an overlay, and trigger-volume editing. | S each |
| 7 | **Prefabs in Studio.** Add a prefab file format, links from instances to their prefab, and apply/revert of overrides. The engine already supports nesting and overrides (`scene.mojo:~245-465`). | M |
| 8 | **Animation graph editor.** Edit states, transitions, conditions, blend spaces and layers, with a JSON writer and clip and socket pickers. Add hitbox/hurtbox authoring with animation event tracks inside Flint, not through game `flint.toml` entries. | L |
| 9 | **Game flow and level scripting.** Add `entity_from_body`, OnEnter/OnExit helpers, BehaviorTree save plus a component, a game-state machine, scene transitions, save slots, and timers and tweens. | M each |
| 10 | **Larger authoring tools.** Terrain sculpt and paint (rendering lives only in `examples/open_world`), scatter and drop-to-surface, UI documents with a HUD editor, world streaming and sub-levels, and a camera sequencer. | L |
| 11 | **TurboMojo IDE integration.** The in-Studio IDE is the external `tmide` running in a pseudo-terminal (`studio/ide_pty.mojo`, `source_editor.mojo`, `tools/flint.sh` `FLINT_TURBOMOJO`). Its Build/Run buttons only send F-key presses, so edited code never reaches Studio. Connect it to the gameplay `.so` hot reload, compiler errors and code navigation (see M2 below). | M |

---

## 3. Milestones

Each milestone ends with an acceptance test played on a **test-harness game**: a small, original 3D action level built only in Studio. Fixes found while building it go into Flint, not into the game.

### M0: Studio as a safe daily tool (1–2 weeks)

**Editor safety**
- [x] Esc no longer quits Studio. Add an unsaved-changes guard on quit and a "*" marker for unsaved changes.
  Done: `request_quit` opens the unsaved-changes modal ([`studio_app.mojo:~4729`](../src/flint/studio/studio_app.mojo)) and `mark_dirty` adds the " *" title marker ([`studio_app.mojo:~3474-3479`](../src/flint/studio/studio_app.mojo)); covered by `test_m3b_f11_input_context_hotkeys_dirty_modal_and_layout` ([`tests/test_studio_core_bugs.mojo:~2197`](../tests/test_studio_core_bugs.mojo)).
- [x] Turn off editor hotkeys during PIE (use a play-mode input context).
  Done: the `StudioPlay` input context blocks the editor's bindings, and Esc in PIE only releases the pointer (`pie_release_cursor`, [`studio_app.mojo:~1882`](../src/flint/studio/studio_app.mojo)); covered in [`tests/test_studio_core_bugs.mojo:~4621`](../tests/test_studio_core_bugs.mojo).
- [x] Save and restore the dock layout with the existing `DockSpace.save_json` / `load_json`.
  Done: `_load_studio_dockspace` / `_save_studio_dockspace` ([`studio_app.mojo:~1703-1738`](../src/flint/studio/studio_app.mojo)); JSON validation covered by `test_m3a_dockspace_json_validation_and_env_spotlight_tier0` ([`tests/test_studio_core_bugs.mojo:~2084`](../tests/test_studio_core_bugs.mojo)).
- [ ] Add F to frame the selection, Ctrl+Shift+Z for redo, Ctrl+C/V built on duplicate, and a local/world gizmo toggle.

**Inspector**
- [ ] Rename entities with the existing `text_input` widget.
- [ ] Show the collider kind and size, `is_trigger`, friction and restitution.
- [ ] Show the material fields that already exist: emissive, clearcoat, sheen, subsurface and opacity.

**Spawning, plugins and PIE**
- [ ] Add spawn cards: Empty, Cube, Point Light, Camera and Audio Emitter. Light entities replace the hard-coded lights.
- [ ] Install the animation and audio plugins and register their codecs.
- [ ] Run `STAGE_UPDATE` in PIE.
- [ ] Add a "Bake NavMesh" menu item with a debug overlay.
- [ ] Add sun direction, sun radiance and exposure sliders.

**Asset browser**
- [ ] Point the asset browser at the project directory and remove the 10-entry limit.

**Engine hygiene**
- [ ] Remove "Maroon", the "deepsong" default and "sharddepth" from the engine. Use the project name for crash and log paths.
- [ ] Turn on the crash handler in the `new` template.

**Acceptance:** the gate passes; a scene with lights, a trigger and a baked NavMesh saves and reloads unchanged; Esc and Delete in PIE don't touch the editor.

### M1: Content comes in (3–4 weeks)
- [ ] Add an `AssetRef` component field, with paths relative to the project, serialized in `.flscn` and `.flscnb`.
- [ ] Clicking or dropping a `.glb` loads the model, uploads it and spawns real mesh and material entities. Apply the same fix to MCP `procedural.generate_asset`.
- [ ] When a scene loads, resolve its asset references through `AssetServer`, including hot reload.
- [ ] Keep skins, morphs and animations in the cooked mesh format.
- [ ] Skinned characters animate in the viewport and in PIE, with clip and graph pickers and a bone/socket inspector.
- [ ] Add shared material assets with texture slots.

**Acceptance:** drop a rigged character and props into a scene; the character plays an idle; save, restart Studio and reload, and it comes back identical.

### M2: Behaviour runs (3–4 weeks)
- [ ] Add `flint.sh gameplay-so`, which builds the project's gameplay `.so`. PIE loads and hot-swaps it through `HotReloadHost`.
- [ ] Project components register with Studio, and the inspector becomes generic with Add/Remove Component.
- [ ] Add a game camera: the player can possess it in PIE and leave with Shift+F1.
- [ ] Add trigger helpers (`entity_from_body`, OnEnter/OnExit), BehaviorTree save plus a component, and an InputMap editor panel.
- [ ] Add a game-state machine, scene transitions, save slots, and timers and tweens.

**Acceptance:** a playable loop in PIE: move, fight one enemy that uses a BehaviorTree, pick something up, reach a goal trigger. Editing gameplay code reloads it without restarting Studio.

#### M2 (IDE): TurboMojo integration
Gameplay code is written in TurboMojo (`tmide`, opened in Studio with F8). This track makes the edit → build → play loop work without leaving Studio.

- [ ] **Build calls the real build.** TurboMojo's Build in a Flint project runs `flint.sh gameplay-so`. Studio hot-swaps the resulting `.so` into PIE, or into the next PIE session if it isn't running. Use the project's `.turbomojo.toml` `[run]` entries, not F-key presses sent from `source_editor.mojo`.
- [ ] **Run plays in the editor.** TurboMojo's Run starts or restarts PIE in the open Studio instead of launching a separate program.
- [ ] **Compiler errors.** Build output is parsed into file:line:col diagnostics. Studio shows them in a Problems list and as a toast. Clicking one opens TurboMojo at that line. A failed build keeps the last good `.so` loaded.
- [ ] **Code navigation from the editor.** In the inspector, "Open source" on a project component or system jumps to its definition in TurboMojo. Add an MCP tool `ide.open` (path, line) so agents can do the same.
- [ ] **Reliable launch.** Resolve `tmide` and `tmide-gui` from the project, then `FLINT_TURBOMOJO`, then the pixi env, then `PATH`. If it is missing, Studio shows a message with the install command instead of failing, and Studio stays usable.
- [ ] **Keyboard ownership.** While the IDE panel has focus, every key goes to TurboMojo and no Studio hotkey fires, using the same capture rules as `UiContext.wants_keyboard()`.
- [ ] **New projects come IDE-ready.** `flint.sh new` writes a `.turbomojo.toml` with `gameplay` (build `.so`), `game` (run) and `test` targets and the Flint include paths.
- [ ] **Tests.** A headless test uses a fake `tmide` on `PATH` (extending `tests/test_studio_ide.sh`). It covers launch, Build → `.so` reload, error parsing and jump-to-line.

**Acceptance:** with PIE running, change a value in a gameplay system in TurboMojo and press Build. The running game picks it up within a few seconds, with world state kept. Introduce a type error: Studio lists it, clicking it opens the line, and the game keeps running the last good build.

### M3: It ships (about 2 weeks)
- [ ] Studio runs from the project directory; scenes live in `<project>/scenes/`.
- [ ] The `new` scaffold loads `assets/main.flscn` and pins the engine commit; `flint_version` is enforced.
- [x] MCP tools `scene.save`, `scene.load` and `project.package` ([`mcp_v2.mojo:~2703, ~2731, ~3407`](../src/flint/devtools/mcp_v2.mojo)); covered by [`tests/test_flint_v5_assets_studio_mcp.mojo:~4146`](../tests/test_flint_v5_assets_studio_mcp.mojo) and [`tests/e2e_studio/tier1/t1_m3_shipping_vfs.sh`](../tests/e2e_studio/tier1/t1_m3_shipping_vfs.sh) (case 4).
- [ ] Add a Build → Package menu item using the profile picker.
- [ ] `package` uses the incremental cooker with a cache directory.
- [ ] Every schema has a version and migration hooks; unknown components load as opaque blobs.
- [ ] Add a game-content test harness: headless golden-image and input-replay checks run from `run.sh`.

**Acceptance:** a level made in Studio, packaged from inside Studio, plays from the bundle's `run.sh` on a clean machine.

### M4: Depth (ongoing)
- **Scene and animation authoring:** prefab assets with apply/revert of overrides; an animation graph node editor; hitbox and animation event tracks; folding the motion and take tools into Studio panels.
- **Interface:** UI documents with a HUD and menu editor (progress bars, images, world markers, state bindings).
- **World building:** terrain sculpt and paint with heightmap import; scatter and drop-to-surface.
- **Large worlds and cinematics:** world streaming and sub-levels (additive load, cells); a camera sequencer on `CameraRig` rails and `curve_editor`.
- **Platforms and AI tooling:** windowing beyond Wayland; MCP resources and prompts so an agent can build a whole game.

---

## 4. Principles
- **Wire before building.** Most gaps are about connecting existing engine systems to Studio, so prefer that over new runtimes.
- **Studio output is the game's input.** Anything authored in Studio must round-trip through files the shipped game loads.
- **Standalone.** No game names in `src/flint/`, `bash tools/flint.sh gate` passes with zero warnings, and the work stays 100% pure Mojo, as `AGENTS.md` requires.
- **Every Studio feature gets an MCP tool**, so agents can build and test games headlessly.

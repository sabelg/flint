# Character course

From the repository root, run `bash tools/flint.sh character-course`. The course loads a two-skin GLB and a binary animation graph through Flint's asset resolver. The same actor can use graph locomotion or query Flint's motion database. Both modes use the standard controller, foot placement and world-hosted ragdoll. The review camera draws the final skeleton with Flint's CPU skinning path.

| Control | Action |
|---|---|
| W, A/D | Walk forward, steer left/right |
| P | Pivot 90 degrees |
| F, G | Fall, begin a get-up |
| M, C | Switch graph/motion matching, change camera |
| Space, period | Pause the simulation, advance one fixed tick |
| Esc | Exit |

Walk toward the low deck. It travels 1 m and turns 30 degrees, then returns. A wall stands at its far end. The amber crosses show planted contacts and the turquoise line shows desired travel over the next 0.8 s.

## Repeatable review

`bash tools/flint.sh character-course --capture` replays identical input in both modes and writes contact sheets and per-tick CSVs under `build/character-course/`. Each sheet pairs a side view with a three-quarter view at the same walk, stop, pivot and recovery ticks. The CSVs record clip, source frame, feet, support body, deck transform, actor position, actual heading and desired facing at 60 Hz. Pressing P commits the character to its authored quarter-turn through the landing. The replay checks the turn angle and the progression of the clip. To export both views of every tick as PNGs, use `--frames`.

Run `bash tools/flint.sh character-course --match-tick 125` to inspect a tick from 1 through 465 in motion-matching mode. The replay reports the clip playing at that tick and the latest search that preceded it. Compare the winning frame, the continuation of the current clip, and the best other eligible frame. Each row lists the trajectory, facing, feet, linear velocity, angular velocity, stance and signed continuation discount that make up its total cost. Lower totals score better. During a committed pivot or a ragdoll fall, the latest locomotion search can be several ticks earlier.

The character and its six clips were authored for Flint and are dedicated to the public domain under CC0 1.0. The GLB carries its rest hierarchy, inverse bind matrices, materials, clip metadata and foot-strike events. Body and jacket use different joint palette orders. Edit the GLB and graph JSON, then run `pixi run mojo run -I src examples/character_course/bake_graph.mojo` from the repository root to rebuild the checked-in graph.

To move a foot strike, edit the corresponding glTF animation's `extras.events` array. The walk uses `{"kind":"footstep_L","arg":"sole","time":0.05}` for the left foot and a right-foot event at 0.55 seconds. The same animation's `extras.loop` and `extras.speed` specify looping and ground speed in metres per second. Recook packaged assets after editing a take.

## Physical recovery benchmark

Run `bash tools/flint.sh character-course --recovery-bench` to compare passive motion and joint motors on the original and calibrated elbow rigs. All four conditions start from the same supine pose with the same lateral shove. The CSV includes the original standing criterion, ankle drift, elbow and whole-body error, ticks with a floor impulse through either foot, and the fraction of floor load transmitted by the feet. Three identical runs per condition check repeatability. Use `--recovery-bench --check` to run just the elbow calibration and loaded-versus-hovering foot checks.

The Sunspire bridge player loads `assets/bridge_dust.json` from the same content pack. Edit its particle count, size in metres, colour, opacity and lifetime in seconds, then recook and reopen the bridge. Each walk marker emits one burst at the named foot joint. The traversal benchmark reports the number of bursts.

Run the 1080p and 2160p dust readback checks on a Vulkan device with `pixi run mojo build -I src -I tests -O3 --fp-mode contract=off --Werror examples/open_world/qualify_dust.mojo -o /tmp/flint-qualify-dust && /tmp/flint-qualify-dust`. It also checks Tier 0 rendering, effect expiry, and two fresh pack-only opens around an edit.

## Play the bridge in Sunspire Valley

Cook the level, then launch the regular Vulkan player from the repository root.

```bash
bash tools/flint.sh pack --cook examples/character_course/assets build/bridge.flpk --cache-dir build/bridge-cache
bash tools/flint.sh world --bridge --pack build/bridge.flpk
```

To build a relocatable Linux player, run the following commands on the target CPU architecture. The release includes the content pack, engine runtime libraries and toolchain lock. The player uses the system Vulkan driver.

```bash
bash tools/flint.sh package --target world --name sunspire --assets examples/character_course/assets --out build/releases/sunspire
build/releases/sunspire/run.sh --bridge --pack assets.flpk
```

Walk with W/S or the D-pad. Ring the guide’s bell with E/A, pivot with P/X, fall with F/B and recover with R/Y. Hold C/Back for a close-up. K/L and the left/right shoulder buttons save and load a checkpoint. Esc/Start closes the level. The level streams the nearby lanterns and guide from five authored cells. The terrain and architecture are built when Sunspire starts.

Use `--locale ar-EG --text-size 34 --mute` to play with Arabic captions at 34 pixels and muted audio. The language and caption size persist for the next launch. The pack includes the font outlines and their license. The game draws the speaker and sound cues during the encounter.

```bash
bash tools/flint.sh world --bridge --pack build/bridge.flpk --offscreen --frames 1050 --save-checkpoint --out build/bridge.png
bash tools/flint.sh world --bridge --pack build/bridge.flpk --resume --offscreen --frames 12
bash tools/flint.sh world --bridge --pack build/bridge.flpk --width 1920 --height 1080 --frames 1800
bash tools/flint.sh world --bridge --pack build/bridge.flpk --width 1280 --height 800 --soak-seconds 1800
bash tools/flint.sh lab world_stream
```

A frame-count run replays recorded actions with a fixed simulation step. A duration run repeats the crossing for the requested wall time. `--bridge-baseline` follows the same camera without the encounter. The player reports game-loop and GPU timestamps separately. Read the display mode and presentation path before using these numbers as a refresh-rate qualification. Set `FLINT_SAVE_DIR` to a disposable directory for automated runs.

To repeat the six desktop presentation runs, use `bash tools/qualify_bridge_present.sh` in a native graphical session with Vulkan WSI. It builds a strict standalone player and cooker, then collects at least 1,800 visible presentations per trial at 1920×1080 and 1280×800, with one cold and two warm launches at each extent. It lengthens short trials and preserves their logs. A system without display feedback records submitted frames and labels the visibility measurement unavailable. Each profile gets new engine and driver cache directories. The script leaves the desktop mode and OS filesystem cache intact. Logs under `build/production-qualification/presented-*` record Vulkan GPU times, loop times, compositor or driver presentation intervals, the measured refresh period and reported discards when available. A 30 Hz desktop can only qualify its own display cadence.

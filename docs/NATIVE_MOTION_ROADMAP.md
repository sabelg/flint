# Native Mojo motion tools for Flint Studio

Studio now has twelve **Author tools** templates. Each creates an editable recipe that can be run from the take editor or the command line. The table records what each template does today. The acceptance scenes in the sections below are longer-term targets. See the [recipe guide and capture setup](../examples/kimodo/WORKFLOWS.md).

| Studio tool | Current implementation | Remaining work |
|---|---|---|
| Bone mapping | Rest-pose retargeting, editable SOMA-to-skin names, textured preview and skinned GLB. Tested on Mozu and a renamed Shrike enemy rig. | Calibrate non-unit-scale skeletons. |
| Foot cleanup | Detect stance from ankle height and velocity, or mark a plant interval and solve it with two-bone IK. | Direct viewport handles for interval editing. |
| Ledge warp | Move the root over a timed approach to align a joint with a world target. Reject corrections outside the specified reach. | Bind a marker to a moving level object. |
| Optimize GLB | Reduce synchronized keys with a selectable world-space tolerance. Check joint origins and offset markers at keys and midpoints. | Per-joint tolerances. |
| Path markers | Guide a take through timed ground markers, blend heading across turns and retain the source gait. | Obstacle-aware routing. |
| Joint pin | Pin a hand or foot in world space with a frame range, fade and weight. Report residual error. | Multiple simultaneous constraints on the same limb. |
| Beat timeline | Combine BVHs or generated cues using MotionBricks. Cache beats and adjacent joins by input. | Graphical rearrangement of beats. |
| Prompt providers | Native Kimodo, HY-Motion Lite with Qwen3 and CLIP text conditioning, and a provider that blends captioned BVHs. | Longer HY-Motion windows and validation across a wider movement library. |
| Two actors | Scrub two synchronized skeletons, mark hand contacts and export separate actor tracks. A partner can use its own GLB and bone map. | Learned two-person generation. |
| Single camera | Native RTMPose detection and nine-frame MotionBERT body recovery on fixed-camera footage. | Moving cameras, absolute depth and tracking through occlusion. |
| Stereo capture | Reconstruct a performer from synchronized, calibrated views. Optionally measure error against a held-out camera. | Physical multi-camera validation, live synchronization and calibration. |
| Capture to GLB | Recover a take, clean contacts, retarget and bake a playable skinned character in one recipe. | Finger and face tracking. |

The following research references informed the scope. Each feature uses Flint’s own runtime. Weights for particular models are prepared separately.

## 1. Make current takes playable

| Reference | Studio feature and implementation | First acceptance scene |
|---|---|---|
| **1. Rokoko retargeting** | **Character preview and bake.** Use the SOMA and Mixamo rest poses to map joint rotation and scale hip travel by leg length. Use the existing native GLB loader, skinning and clip sampler. First slice covers Mozu's rigid, unit-scale hierarchy. Follow with a user-editable bone map, per-rig rest calibration and profile persistence. [Rokoko retargeting](https://github.com/Rokoko/rokoko-studio-live-blender#retargeting) | Play all 148 frames of the walk-to-crouch take on Mozu. Reopen the exported GLB, compare sampled joint transforms with the editor, and verify that the original skin, materials and clips still load. |
| **2. Cascadeur cleanup** | **Contact editor.** Use ankle height and velocity to propose a stance interval. Let the animator move its boundaries, set a ground plane and bake corrections through `FootPlantSolver` and two-bone IK. Start with foot locking and a displacement overlay. Airborne posing and whole-body physics are later work. [Cascadeur](https://cascadeur.com/) | On one landing and one crouch, measure planted-foot travel before and after the edit. Aim for less than 2 cm of horizontal travel during the marked interval with correction fading out at toe-off. |
| **3. Unreal Motion Warping** | **Reusable ledge targets.** Flint already has `MotionWarp`, a contact time, and choreography ledge placement. Expose draggable approach and contact targets for imported takes, preview the reach limit, and save the target relative to the level object. [Epic motion warping](https://dev.epicgames.com/documentation/en-us/unreal-engine/motion-warping-in-unreal-engine) | Place the same vault on three reachable ledges. Measure contact error in world space, aiming for 3 cm or less, and show when a target exceeds the configured reach. |
| **4. ozz-animation Optimizer** | **Clip optimization preview.** Remove candidate keys offline, resample through Flint's native interpolation, and compare world-space joint positions. Give wrists, feet and sockets tighter tolerances. Show measured byte savings before export. [ozz optimizer](https://guillaumeblanc.github.io/ozz-animation/samples/optimize/) | Compare a walk, crouch and idle at source keys and at intermediate samples. An exported clip passes when measured hand and foot error stays under the selected tolerance and the file gets smaller. |

## 2. Direct the choreography

| Reference | Studio feature and implementation | First acceptance scene |
|---|---|---|
| **5. DART** | **Path-guided takes.** Draw timed waypoints on the stage. First fit a retrieved locomotion take with motion matching and bounded root warping. For learned synthesis, evaluate DART's recurrent motion history, checkpoint operators and body representation in a separate native inference spike. [DART](https://github.com/zkf1997/DART) | Mozu rounds a rooftop obstacle, faces the lookout and stops at the final marker. Measure cross-track and arrival errors, obstacle intersections and stance drift. |
| **6. OmniControl** | **Joint pins over time.** Add a named joint, reference space, frame interval and weight to a take. Start with an arm or leg IK solve with smooth release. A future diffusion model could condition on the same pins. Per-limb IK is a smaller first deliverable than arbitrary-joint generation. [OmniControl](https://github.com/neu-3dvi/OmniControl) | Pin one hand to a rail during a turn. Plot wrist error for every constrained frame and verify motion before and after the interval. Test an unreachable pin separately. |
| **7. FlowMDM** | **Multi-beat timeline.** Store an ordered list of cues, durations, seeds and transition boundaries. Build the first playable timeline using Kimodo takes and MotionBricks joins. Cache individual takes so editing one beat regenerates only that take and its adjacent joins. Evaluate long-sequence diffusion only if the baseline has visible continuity problems. [FlowMDM](https://github.com/BarqueroGerman/FlowMDM) | Author “walk to the edge, kneel, look back, stand.” Change the third beat, replay the result on Mozu, and compare the untouched first beat frame for frame. |
| **8. HY-Motion** | **Alternative text-to-motion generator.** Define a prompt-to-take interface around the existing Kimodo workflow. Audit HY-Motion's tokenizer, conditioning, denoiser and skeleton conversion before attempting its Mojo port. Compare reference intermediates and peak memory before adding a Studio provider. [HY-Motion](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) | Use identical prompts for a sidestep, stop, crouch and recovery. Compare instruction accuracy, foot drift, generation time, peak memory and manual cleanup on the same machine. |
| **9. InterGen** | **Two-character takes.** Start with two existing clips on a shared clock, separate root tracks, and relative hand-contact targets. Reuse the character mapper and joint pins. Learned two-person generation is a separate research track. InterGen publishes its material under noncommercial terms, so a shipping generator needs an independently cleared model and data. [InterGen licensing and demos](https://github.com/tr3e/InterGen#licenses) | An ally helps Mozu onto a roof. Scrub both actors together, measure hand separation through the marked contact, and preserve relative placement when reopening the scene. |

## 3. Bring a performance into Studio

| Reference | Studio feature and implementation | First acceptance scene |
|---|---|---|
| **10. GVHMR** | **Single-camera motion recovery.** Begin with an image-sequence input. Add native inference for person tracking, 2D keypoints and temporal body recovery, then map the result onto a supported skeleton. Check operator support, body-model dependencies and ground alignment with reference data before committing to a checkpoint. [GVHMR](https://github.com/zju3dv/GVHMR) | From a fixed-camera recording of a performer turning and crouching, recover a continuous take. Compare projected joints to annotated frames and inspect foot drift on Mozu. Add moving-camera footage as a separate test. |
| **11. FreeMoCap** | **Calibrated multi-camera capture.** Share the 2D detector with the single-camera pipeline. Implement camera calibration, timestamp alignment, confidence-weighted triangulation and skeleton fitting. A first release can accept pre-calibrated, synchronized recordings. Live camera discovery can follow. [FreeMoCap](https://docs.freemocap.org/) | Record one performer from two cameras and hold back a third view for validation. Plot reprojection error, compare reconstructed bone lengths and flag frames where too few views see a joint. |
| **12. DeepMotion Animate 3D** | **Local video-to-character workflow.** Combine video recovery, identity selection, retargeting, contact cleanup and GLB export in one Studio job. Share the inference from item 10. This produces an independently implemented equivalent of the workflow. Hand and face tracking need their own models and tests. [DeepMotion Animate 3D](https://www.deepmotion.com/animate-3d) | Load a short performance, choose Mozu, inspect low-confidence frames, correct a planted foot and export. Reopen the GLB in a fresh Studio session and play it without running inference again. |

## Follow-up milestones

1. Extend the tested Mozu and enemy-rig maps to non-unit-scale hierarchies and more rest poses.
2. Add draggable ground, ledge and limb targets in the viewport. Display residual reach errors as the artist scrubs.
3. Add per-joint optimization tolerances and direct timeline editing. Test paired contacts on additional character proportions.
4. Evaluate learned path control and longer HY-Motion windows against authored traversal scenes.
5. Validate two synchronized physical recordings with a held-out third camera. Add calibration and live capture after measuring lens distortion and timing error on that setup.

The engine-side starting points are `src/flint/anim/procedural.mojo` for contact and warping, `choreography.mojo` for timing, `motion_matching.mojo` for retrieval, `src/flint/ml/` for native inference, and `examples/kimodo/` for the Studio take editor.

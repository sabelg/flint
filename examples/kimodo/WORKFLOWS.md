# Author motion in Flint Studio

Open **Tools → Motion takes**, load a SOMA30 BVH, then click **Author tools**. Pick a template to open a numbered recipe under `build/motion-recipes/` in Studio’s embedded TurboMojo editor. Save the file, return to Motion takes and click **Open**. **Edit recipe** reopens it. Drag the slider to scrub and press Space to resume. **Use as start** remembers a second take for templates that need two clips.

**Save new BVH** exports the edited skeleton. **Character preview → Save character GLB** exports the skin with an appended animation. A paired recipe writes a second file with `.partner` in its name. Use the skeleton view to see both actors together. The same recipe can run without opening a window.

```sh
bash /path/to/flint/tools/flint.sh studio --takes --recipe rooftop.motionrecipe.json rooftop.glb rooftop.png
```

Choose new filenames. The optional PNG shows three poses of one actor or both actors together at the midpoint. Asset paths in a recipe resolve from the recipe’s directory. World coordinates use metres and Y points up. Frames are zero-based and end frames are inclusive. `character` defaults to the project’s `assets/mozu.glb` or the path in `FLINT_TAKE_CHARACTER`. Target bones use their name after any colon namespace.

## Direct a take

This recipe applies a hand target to a 60-frame SOMA30 clip, then reduces its baked keys using a 4 mm world-space tolerance. Set `bones` when a destination joint has a different name.

```json
{
  "version": 1,
  "source": "walk.bvh",
  "character": "mozu.glb",
  "bones": {"Hips": "Hips"},
  "edits": [
    {"op": "warp", "joint": "LeftHand", "start": 10, "end": 28,
     "target": [0.35, 1.05, 0.4], "reach": 0.65},
    {"op": "pin", "joint": "LeftHand", "start": 28, "end": 34,
     "target": [0.35, 1.05, 0.4], "fade": 3, "weight": 1}
  ],
  "tolerance": 0.004
}
```

Edits run in list order. A warp translates the root and keeps the resulting displacement through the rest of the take. A pin solves a hand or foot and reports the maximum error during the selected interval. An unreachable pin extends the limb to its reach. `plant` holds the named foot at its first interval position, with an optional absolute `height`. `cleanup` detects stance using `speed` in m/s, `height` above the lowest observed ankle and `minimum` consecutive frames. `path` takes an ordered `markers` array of objects such as `{"frame": 0, "position": [0, 0, 0]}`, spanning the first through last frames. X and Z set world coordinates, Y adds elevation to the original hip motion.

For **Two actors**, set `partner.source`, `partner.offset` in metres and optionally `partner.delay` in seconds. `partner.contacts` takes entries such as `{"joint":"LeftHand","primaryJoint":"RightHand","start":18,"end":24}`. The partner follows the primary clock and holds the nearest endpoint outside its own duration. Use `partner.character` and `partner.bones` to export different characters. Studio solves contact on the SOMA skeleton, then solves it again against the actual partner rig when baking both GLBs. The console reports peak separation on each rig. If contact is outside the partner’s reach, adjust the actor offset or contact interval.

## Compose and generate

A `beats` array replaces the source take. Each entry specifies a local BVH or a generation prompt. MotionBricks joins require matching 30 Hz rigs and at least four frames per clip. Generated beats and transitions are reused from `build/motion-cache` under the recipe directory. Change a prompt, seed or source clip to rebuild the affected parts.

```json
{
  "version": 1,
  "beats": [
    {"source": "walk.bvh"},
    {"provider": "kimodo", "prompt": "A person kneels down.", "frames": 45, "seed": 7},
    {"source": "look-back.bvh"}
  ]
}
```

A top-level `prompt` produces one take. The `kimodo` provider samples the diffusion model. The `library` provider embeds the prompt and each authored caption, selects the closest two takes and blends them over normalized time. Provide at least two `{"caption":"…","source":"…bvh"}` entries in `library`. Its range of motion depends on your clips. Kimodo and the library provider use the LLM2Vec text encoder. See [Kimodo and MotionBricks setup](README.md).

### HY-Motion Lite

The third provider runs the released HY-Motion Lite flow model, Qwen3-8B and CLIP-L in Mojo. With its weights present, **Prompt providers** starts with a sidestep recipe. Edit the plain-English prompt and reopen it to generate another take. A prepared cue is optional. The initial Studio version supports 2–60 frames at 30 Hz and up to 30 Qwen tokens of cue text, plus the upstream template suffix. Set `guidance` (default 5), `steps` (default 20) and `seed`. The exported upper body and limbs use the SMPL 22 body rotations on the reference SOMA skeleton, with floor alignment.

```json
{
  "version": 1,
  "provider": "hymotion-lite",
  "skeleton": "walk.bvh",
  "prompt": "A person takes a deliberate step sideways to the left.",
  "frames": 45,
  "steps": 20,
  "seed": 4
}
```

For a fresh installation, obtain the upstream HY-Motion checkout, Lite checkpoint directory (including `config.yml`), Qwen3-8B and CLIP-L checkpoints. These one-time converters use PyTorch, Transformers, ONNX, NumPy, PyYAML and einops. The native model cache occupies about 17 GiB. Override its location using `modelDirectory` in the recipe. The Mojo runtime selects GPU matrix products when available. Set `FLINT_HYMOTION_CPU=1` for CPU mode.

```sh
python tools/prepare-hymotion.py model /path/to/HY-Motion-1.0 /path/to/HY-Motion-1.0-Lite ~/.cache/flint/hymotion-lite
python tools/prepare-hymotion-text.py /path/to/HY-Motion-1.0 /path/to/Qwen3-8B /path/to/clip-vit-large-patch14 ~/.cache/flint/hymotion-lite
```

For a cue prepared with the upstream text encoders, use `tools/prepare-hymotion.py cue --help` and set `cue` to that file in place of `prompt`. This path needs only the flow model during authoring. Compared with upstream inference using the same noise and prepared sidestep cue, the largest of 990 local joint rotation differences was 0.00030° and the maximum relative hip travel difference was 0.005 mm. These measurements cover one 45-frame clip before upstream optional smoothing. [HY-Motion](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) · [Lite checkpoint](https://huggingface.co/tencent/HY-Motion-1.0) · [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) · [CLIP-L](https://huggingface.co/openai/clip-vit-large-patch14)

## Capture a performer

Use one fixed camera for MotionBERT, or two synchronized pre-calibrated cameras for triangulation. Each view accepts an MJPEG AVI or a directory of zero-padded JPEG/PNG frames. Set `fps` for image sequences. `source` supplies the reference skeleton. An optional pixel crop `roi` selects one performer throughout each view.

```json
{
  "version": 1,
  "source": "walk.bvh",
  "character": "mozu.glb",
  "capture": {"views": [{"frames": "performance.avi", "roi": [124, 76, 171, 295]}]},
  "edits": [{"op": "cleanup"}]
}
```

For stereo, give each view `eye`, `lookAt` and `up` in a shared world coordinate system, plus focal lengths `fx` and `fy` in pixels. The principal point defaults to image centre and can be overridden with `cx` and `cy`. Use undistorted recordings with a matching frame count and cadence. `rayTolerance` is the maximum accepted gap between the two rays in metres (default 0.12). A third calibrated `validationView` reports mean and maximum pixel error against independently detected joints. Monocular capture estimates hip travel in the image plane and floor height from the visible performance. Stereo estimates the hip in the calibrated world frame.

### Prepare the capture models

On this machine the prepared files live in `~/.cache/flint/rtmpose/`. A fresh machine needs two one-time conversions. The converter requires Python, NumPy and ONNX. The lifter converter also requires PyTorch and the dependencies for importing DSTformer from the upstream MotionBERT checkout.

```sh
mkdir -p ~/.cache/flint/rtmpose
python tools/prepare-rtmpose.py /path/to/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx ~/.cache/flint/rtmpose/body17.onnx
python tools/prepare-motionbert.py /path/to/MotionBERT /path/to/FT_MB_release_MB_ft_h36m/best_epoch.bin ~/.cache/flint/rtmpose/lift9.onnx
```

Override the prepared files with `capture.detector` and `capture.lifter`. The AVI/image decoder, inference, skeleton fitting and GLB exporter run in Mojo. The stereo path only uses the detector. [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) · [MotionBERT](https://github.com/Walter0807/MotionBERT) · [Tennis footage used for the local test](https://github.com/zju3dv/GVHMR/tree/main/docs/example_video)

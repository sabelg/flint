# Kimodo in Flint Studio

Use **Tools → Kimodo motion** in the scene editor. Type an English movement cue and press **Generate**. Studio saves a 60-frame, 30 Hz animation under `build/kimodo-takes` in the current project and plays the generated SOMA skeleton. The generator runs in its own process so Studio stays usable.

Install the pinned model assets once from the Flint checkout:

```sh
bash tools/kimodo-models.sh
```

The installer verifies SHA-256 checksums. Allow about 7 GB for the weights in `~/.cache/flint/kimodo-soma-rp-v1.1`. Set `FLINT_KIMODO_DIR` to choose another directory for installation and inference.

Open the generator directly with `bash tools/flint.sh studio --kimodo`. For a repeatable headless take:

```sh
bash tools/flint.sh studio --kimodo "A person walks forward." build/walk.bvh 60 20 7
bash tools/flint.sh studio --kimodo --sheet build/walk.bvh build/walk.png
```

The optional numeric arguments are frame count (2–300), denoising steps (2–1000) and seed. The defaults are 60, 20 and 1. BVH uses centimetres, local joint rotations and Y up. Flint and Blender can import the take. Game characters with other skeletons need retargeting.

## Runtime

The authoring path runs in Mojo. A byte-pair tokenizer and bidirectional LLM2Vec encoder produce one 4096-value pooled embedding from printable English text, up to 64 tokens including BOS. The encoder memory-maps the pinned Q4/128 weights and uses the native CPU job pool. The SOMA RP v1.1 two-stage denoiser runs on Flint's ONNX interpreter with MAX GPU acceleration for supported matrix products and CPU fallback. The host applies text guidance and deterministic DDIM, undoes training normalization and recovers local rotations on the SOMA30 skeleton. This release supports single-prompt takes. Trajectory constraints, contact cleanup and character-specific retargeting are separate workflows.

The denoiser and model statistics come from [NVIDIA Kimodo SOMA RP v1.1](https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1). The packaged weights use the [KitsuMate denoiser conversion](https://huggingface.co/KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX) and [Q4 LLM2Vec conversion](https://huggingface.co/KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX). Model terms are published with those artifacts. Kimodo's SOMA layout and sampler follow [NVIDIA's implementation](https://github.com/nv-tlabs/kimodo).

## Edit and connect motion takes

Open **Tools → Motion takes**. Use **Next project take** to cycle through BVHs in the project's `build/kimodo-takes`, `build/motion-takes`, `build/kimodo` and top-level directory. You can also enter a BVH path and press **Open**. The editor reads SOMA30 takes with a translating hip and fixed bone offsets.

- **Mirror L/R** swaps paired joint rotations and reflects hip travel across the starting X position. **In place** holds horizontal hip position while retaining height and rotation. **Reverse** plays the selected frames backwards.
- **Find loop** searches for similar poses moving in the same direction, between half a second and two seconds apart. It compares the hip, head, hands, shins and feet and leaves the repeated endpoint out. Preview the wrap before saving, especially on one-off gestures. **Full range** restores the whole clip.
- **Use as start** remembers the current preview, including its edits. Open a second take, then click **Join to this take**. MotionBricks generates the intervening frames and places the second take at the first take's ending floor position. This requires two 30 Hz takes with the same skeleton and at least four frames each.
- **Save new BVH** writes a numbered variant under `build/motion-takes`. Joining automatically saves and opens the combined clip there. The source files remain available.

Editing an existing take needs no weights. Joining uses the separate SOMA FP32 ONNX model from [NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks). Set `FLINT_MOTIONBRICKS_MODEL` to the downloaded file, set `MOTIONBRICKS_DIR` to the upstream checkout, or place `mmm_v6_soma_fp32.onnx` under `~/.cache/flint/motionbricks/`. See [Flint's model notices](../../licenses/README.md).

From the Flint checkout, these commands run the same operations headlessly and require new output filenames.

```sh
bash tools/flint.sh studio --takes --export walk.bvh walk-cycle.bvh --loop --mirror --in-place
bash tools/flint.sh studio --takes --join walk.bvh crouch.bvh walk-to-crouch.bvh
```

## Preview on a game character

In Shrike's **Tools → Motion takes**, open a SOMA30 BVH and choose **Character preview**. The editor loads `assets/mozu.glb` from the current project. To use a different unit-scale Mixamo character, set `FLINT_TAKE_CHARACTER` to its GLB path before starting Studio. Toggle **Character preview** again to see the source skeleton.

The preview follows the current edits, including mirrored, reversed and in-place takes. **Save character GLB** writes a new, self-contained model with an appended animation named **Retargeted take** under `build/motion-takes/`. The profile drives the major body joints and hip travel. Fingers inherit the hand pose. Recipes can override the Mixamo destination names. The destination hierarchy must use unit-scale TRS transforms.

The same bake and a three-frame preview are available headlessly. Choose a new output GLB filename.

```sh
bash /path/to/flint/tools/flint.sh studio --takes --retarget walk.bvh assets/mozu.glb mozu-walk.glb mozu-walk.png
```

Open **Author tools** for editable recipes, including camera capture and two-actor choreography. See the [motion recipe guide](WORKFLOWS.md).

The [native motion roadmap](../../docs/NATIVE_MOTION_ROADMAP.md) covers twelve workflows, including contact cleanup, ledge warping and local performance capture.

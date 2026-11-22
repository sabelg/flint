# Reference fern

From the Flint repository root, build a plant using the included reference:

```sh
bash tools/flint.sh fern examples/fern/frond.png build/fern
```

The command writes `build/fern-lod0.glb`, `build/fern-lod1.glb` and `build/fern-lod2.glb`. Each file includes its material and color atlas. The example image comes from Flint’s procedural palm-frond baker.

To use a photo, crop to one frond, viewed face-on with the stem at the bottom. Use a transparent PNG or a PNG/JPEG with an even, contrasting background visible in the top-left corner. The frond needs at least 16 foreground pixels of height. The generator samples its width and color along the stem and arranges seven curved fronds around a common root. Each LOD retains all seven stems. The generator models a fern growth pattern, so a picture of an entire garden is unsuitable.

## Animated instances in Flint

Create one `flint.render.fern_patch.FernPatch(renderer, reference, seed)` per plant variant at load time. Build `reference` with `FernReference(image)` from `flint.procedural.reference_fern`, using a decoded `NativeImage`. Each render frame, call `patch.begin_frame(time_seconds, previous_time_seconds)`, then `patch.draw(renderer, world_transform, uniform_scale, projected_diameter_pixels)` for each stationary instance. For the first frame use the same time twice.

Instances share one material, one atlas and one wind update per visible LOD. Projected diameters of 110 pixels and above use 560 triangles, 45–109 pixels use 168, and smaller plants use 84. The renderer bends the tips with current and previous GPU poses for temporal reconstruction. The exported GLBs contain the static geometry and wind influence in `TEXCOORD_1.x`. Wind playback and automatic LOD selection are provided by the Flint runtime helper.

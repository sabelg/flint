# Desktop Workstation Setup & GPU Realism Roadmap

## 1. Moving to Your Desktop GPU Workstation

Clone `flint` and `robotech` side-by-side into any directory:

```bash
git clone https://github.com/sabelg/flint.git
git clone https://github.com/sabelg/robotech.git   # (if pushed to GitHub)

cd flint
pixi install
bash tools/flint.sh gate
bash tools/flint.sh play
```

### Syncing New Libraries from `Deepsong`
Whenever new pure-Mojo packages land in `https://github.com/sabelg/Deepsong.git`:

```bash
git clone https://github.com/sabelg/Deepsong.git /tmp/Deepsong   # or git -C /tmp/Deepsong pull
cd flint
bash tools/flint.sh sync /tmp/Deepsong
bash tools/flint.sh gate
```

---

## 2. Discrete GPU Verification (Radeon RX 9070 GFX12 & Vulkan 1.2+)

Flint ships two zero-C-wrapper GPU backends under `src/flint/platform/` and `src/flint/render/`:
1. **Vulkan Compute + Swapchain (`vulkan.mojo` + `vulkan_shaders.mojo` + `spirv.mojo`)**:
   - Emits SPIR-V 1.3 instruction streams directly from Mojo (`SPIRV_MAGIC = 0x07230203`).
2. **Direct AMD RDNA GFX11/GFX12 (`amdgpu.mojo` + `amd_isa.mojo`)**:
   - Talks straight to `/dev/dri/renderD*` DRM `ioctl`s and assembles RDNA wave32 ISA dwords directly in Mojo.
3. **Write-Once Dual-GPU Compute Builder (`render/kernel.mojo`)**:
   - Single Mojo `Kernel` definition emits **both** Radeon GFX12 ISA and Vulkan SPIR-V 1.3.

---

## 3. Next-Gen Visual Realism Roadmap in Flint (`flint.render`)

1. **Unified High-Level `GpuSceneRenderer` Wrapper (`src/flint/render/scene_renderer.mojo`)**:
   - Wrap `gpu_raster.mojo` (`1,000,000` textured triangles, `2 GiB` VRAM texture heap), `glb_native.mojo` (`.glb` models + skeletal CoR skinning), `lighting.mojo` (48 dynamic point lights + hemisphere IBL), and `grading.mojo` (AgX filmic curve + ASC CDL) behind an ergonomic API matching `SimdCanvas`.
2. **Cook-Torrance Microfacet PBR + Tangent-Space Normal Maps (`kernel.mojo`)**:
   - GGX normal distribution, Smith geometry masking, Schlick Fresnel, metallic/roughness/normal/emissive texture channels, and pre-convolved HDR environment reflections.
3. **Cascaded Solar Shadows & Self-Shadowing**:
   - Crisp armor self-shadowing across articulated mecha limbs and wings.
4. **Volumetric Exhaust, Smoke Ribbons & TAA / Per-Object Motion Blur**:
   - Soft alpha-blended GPU particle ribbons (`frame_passes.mojo` + `fx_atlas.mojo`), Mach shock-diamonds, HDR bloom, and velocity-buffer motion blur.

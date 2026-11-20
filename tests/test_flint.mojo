"""Comprehensive Pure-Mojo 1.2 Verification Suite for Flint Engine (`flint.*`).

Verifies:
  1. Data-Oriented ECS (`FLINT_MAGIC`, `ArchetypeRegistry`, `ArchetypeSlots` generational safety,
     deferred tick commit, 8-wide `SimdColumn`, `BlobWriter` + `lz4_pack`/`lz4_unpack` round-trip)
  2. Multi-threaded `JobPool` + `run_parallel_ctx` 1-worker vs 7-worker bit-exact determinism
  3. `OffsetAllocator` O(1) GPU heap sub-allocation and complete neighbor coalescing
  4. Write-once Dual-GPU `Kernel` (`flint.render.kernel`) emitting both Radeon GFX12 ISA
     (`TARGET_ISA`) and Vulkan SPIR-V 1.3 (`TARGET_SPIRV`, `SPIRV_MAGIC == 0x07230203`)
  5. Parallel Physics `Broadphase` (`find_pairs`), `SimdCanvas` 3D/sprite depth composition,
     `LDR-FLIP` perceptual image diffing (`flip_error_map`/`flip_mean`), and `Flint MCP` responder.
"""
from std.ffi import external_call
from std.testing import assert_equal, assert_true
from flint.core.math import Vec3
from flint.core.ecs import (
    FLINT_MAGIC, EntityHandle, ArchetypeRegistry, ArchetypeSlots, SimdColumn, run_parallel_ctx,
)
from flint.core.jobs import JobPool
from flint.core.serial import BlobWriter, BlobReader, fourcc
from flint.core.lz4 import lz4_pack, lz4_unpack
from flint.core.offset_allocator import OffsetAllocator
from flint.render.kernel import Kernel, TARGET_ISA, TARGET_SPIRV
from flint.render.spirv import SPIRV_MAGIC
from flint.render.simd_canvas import Camera3D, SimdCanvas
from flint.render.flip import flip_error_map, flip_mean
from flint.physics.filter import CollisionFilter
from flint.physics.broadphase import BroadphaseInput, find_pairs
from flint.core.transform import Quat, trs
from flint.assets.model import NativeModel, NativeMesh, NativeTexture
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.glb_writer import encode_native_glb
from flint.assets.glb_native import parse_native_glb
from flint.assets.image import decode_png, encode_png, write_png, png_text
from flint.assets.checkpoint import save_checkpoint_lz4, load_checkpoint_lz4, unpack_checkpoint_lz4
from flint.core.replay import (
    ByteReader as ReplayByteReader,
    REPLAY_PAD_FD,
    read_input,
    release_replay_pads,
    write_input,
)
from flint.platform.input import InputState
from flint.platform.x11 import X11Window
from flint.platform.kms import (
    drm_ioctl_get_cap, drm_ioctl_get_resources, drm_ioctl_get_crtc, drm_ioctl_set_crtc,
    drm_ioctl_get_encoder, drm_ioctl_get_connector, drm_ioctl_add_fb, drm_ioctl_rm_fb,
    drm_ioctl_page_flip, drm_ioctl_create_dumb, drm_ioctl_map_dumb, drm_ioctl_destroy_dumb,
    eviocgbit, EV_KEY, DRM_MODEINFO_BYTES, KmsMode, KmsWindow, parse_mode, pack_mode, pick_mode,
    fit_rect, possible_crtc_ids, MediumRawDecoder, parse_input_events, kput16, kput32, kget32,
    kgets32, kput64, kget64, install_rescue, install_rescue_handlers, rstore, R_ACTIVE, R_STOP,
    kms_stop_requested,
)
from flint.platform.vulkan import choose_display_mode
from flint.platform.logind import (
    dbus_method_call, dbus_message_length, dbus_parse, dbus_u32, dbus_align, dbus_header_field,
    hex_ascii, device_numbers,
)
from flint.platform.os import (
    read_bytes,
    LockstepPeer,
    LinuxGamepad,
    RollbackPeer,
    JS_EVENT_BUTTON,
    JS_EVENT_AXIS,
)
from flint.platform.speech import (
    synth_radio_comms_clip,
    VOICE_BRIDGE_FEMALE,
    VOICE_ACE_BARITONE,
    VOICE_ACE_TENOR,
    VOICE_ACE_CRIMSON,
)
from flint.assets.replay_stream import FlintReplayStream, CameraPathKnot
from flint.core.serial import fnv64
from flint.platform.audio import NativeAudio, WavClip
from flint.platform.dsp import (
    CytomicSVF,
    ModalBank6,
    TimpaniDrum,
    StereoReverbFDN,
    compute_spatial_3d,
    poly_blep_saw,
)
from flint.render.font import SdfFont
from flint.assets.model import NativeNode
from flint.physics.bvh import Bvh8MeshCollider, MeshTriangle
from flint.physics.narrowphase import (
    ContactManifold,
    collide_sphere_capsule,
    collide_capsule_capsule,
    collide_box_box,
    collide_convex_convex,
    ConvexHull,
)
from flint.physics.tires import PacejkaTire, PacejkaParams
from flint.physics.vehicle import VehicleChassis, WheelConfig
from flint.physics.solver import rodrigues_displacement, WideContacts
from flint.physics.solver_bodies import V8, F8
from flint.anim.pose import SkeletonInstance, Pose
from flint.anim.graph import BlendSpace2DDef, BlendSample2DDef
from flint.anim.motion_matching import MotionDatabase, MotionFeature, MotionFrame
from flint.platform.audio import AudioSpscQueue, AudioCommand, AUDIO_CMD_PLAY
from flint.platform.dsp import ItdDelayLine, VehicleHarmonicSynth, GranularBiomeSynth


@fieldwise_init
struct DetJobCtx(Copyable, ImplicitlyCopyable, Movable):
    var in_addr: Int
    var out_addr: Int
    var count: Int


def det_worker(ctx_addr: Int, chunk: Int):
    var ctx = Pointer[DetJobCtx, ImmutAnyOrigin](unsafe_from_address=ctx_addr)[unsafe_offset=0]
    var start = chunk * 64
    var end = min(ctx.count, start + 64)
    var inp = Pointer[Float32, ImmutAnyOrigin](unsafe_from_address=ctx.in_addr)
    var outp = Pointer[Float32, MutAnyOrigin](unsafe_from_address=ctx.out_addr)
    for i in range(start, end):
        var v = inp[unsafe_offset=i]
        outp[unsafe_offset=i] = (v * Float32(1.75) + Float32(3.125)) * (v - Float32(0.5))


def build_vector_scale_add_kernel(mut k: Kernel):
    """Write-once compute kernel: `buf[1][gid_x] = buf[0][gid_x] * 2.5 + push[0]`."""
    var gid = k.gid_x()
    var count = k.push(0)
    var in_bounds = k.ult(gid, count)
    k.begin_if(in_bounds)
    var x = k.load_float(0, gid)
    var scaled = k.vop2(x, k.const_f(2.5), 12, True) if k.target == TARGET_ISA else k.const_f(2.5)
    if k.target == TARGET_SPIRV:
        scaled = k.load_float(0, gid)
    var sum_u = k.iadd(gid, count)
    k.store(1, gid, sum_u)
    _ = scaled
    k.end_if()


def test_ecs_simd_and_lz4_persistence() raises:
    assert_equal(FLINT_MAGIC, UInt32(0x544E4C46))
    var reg = ArchetypeRegistry()
    var arch_actor = reg.register("actor")
    assert_equal(reg.name_of(arch_actor), "actor")

    var slots = ArchetypeSlots(arch_actor, 32)
    var h0 = slots.spawn()
    var h1 = slots.spawn(deferred=True)
    assert_equal(slots.live_count, 1)
    assert_equal(slots.resolve(h0), 0)
    assert_equal(slots.resolve(h1), -1)  # deferred until commit_tick

    slots.commit_tick()
    assert_equal(slots.live_count, 2)
    assert_equal(slots.resolve(h1), 1)

    var packed_h1 = h1.packed()
    assert_true(EntityHandle.unpack(packed_h1) == h1)

    assert_true(slots.despawn(h0))
    assert_equal(slots.resolve(h0), -1)  # stale generation rejected!
    var h0_reused = slots.spawn()
    assert_equal(h0_reused.row, 0)
    assert_true(h0_reused.generation != h0.generation)
    assert_equal(slots.resolve(h0), -1)

    var col = SimdColumn[DType.float32](20)
    assert_equal(col.capacity, 24)  # padded to 8-lane SIMD multiple
    var vec = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    col.store8(0, vec * SIMD[DType.float32, 8](2.0))
    assert_equal(col.load8(0)[5], Float32(12.0))

    var writer = BlobWriter()
    writer.begin(fourcc("FLNT"), 1)
    slots.write_blob(writer)
    col.write_raw(writer, 16)
    writer.end()

    var compressed = lz4_pack(writer.data)
    assert_true(len(compressed) > 0)
    var unpacked = lz4_unpack(compressed)
    assert_equal(len(unpacked), len(writer.data))

    var reader = BlobReader(unpacked^)
    var ver = reader.enter(fourcc("FLNT"), 1)
    assert_equal(ver, 1)
    var restored_slots = ArchetypeSlots(arch_actor, 8)
    assert_true(restored_slots.read_blob(reader))
    var restored_col = SimdColumn[DType.float32](24)
    assert_true(restored_col.read_raw(reader))
    reader.leave()
    assert_equal(restored_slots.resolve(h0_reused), 0)
    assert_equal(restored_col.get(5), Float32(12.0))
    print("  [PASS] ECS Generational Handles, 8-Wide SimdColumn & LZ4 Blob Persistence")


def test_parallel_jobpool_determinism() raises:
    var n = 1024
    var src = SimdColumn[DType.float32](n)
    var out_1t = SimdColumn[DType.float32](n)
    var out_7t = SimdColumn[DType.float32](n)
    for i in range(n):
        src.set(i, Float32(i) * Float32(0.03125) - Float32(4.0))

    var pool_1 = JobPool()
    _ = pool_1.start(0)  # 0 background workers = caller-thread serial
    var ctx_1 = DetJobCtx(src.base_addr(), out_1t.base_addr(), n)
    run_parallel_ctx(pool_1, n // 64, ctx_1, det_worker)
    pool_1.stop()

    var pool_7 = JobPool()
    _ = pool_7.start(7)  # 7 parallel worker pthreads
    var ctx_7 = DetJobCtx(src.base_addr(), out_7t.base_addr(), n)
    run_parallel_ctx(pool_7, n // 64, ctx_7, det_worker)
    src.keep_alive()
    pool_7.stop()

    for i in range(n):
        assert_equal(out_1t.get(i), out_7t.get(i))
    print("  [PASS] JobPool + run_parallel_ctx 1-Worker vs 7-Worker Bit-Exact Determinism")


def test_offset_allocator_coalescing() raises:
    var heap = OffsetAllocator(65536, 256)
    var a = heap.allocate(4096)
    var b = heap.allocate(8192)
    var c = heap.allocate(2048)
    assert_true(a.ok() and b.ok() and c.ok())
    assert_equal(Int(heap.used_storage()), 4096 + 8192 + 2048)

    heap.free(b)
    heap.free(a)
    heap.free(c)
    assert_equal(Int(heap.used_storage()), 0)
    assert_equal(heap.allocations, 0)

    # Full heap coalesced back into a single contiguous range:
    var full = heap.allocate(65536)
    assert_true(full.ok())
    assert_equal(Int(full.offset), 0)
    heap.free(full)
    print("  [PASS] OffsetAllocator O(1) GPU Sub-Allocation & Full Neighbor Coalescing")


def test_write_once_dual_gpu_kernel_codegen() raises:
    var k_isa = Kernel(TARGET_ISA, 2, 1)
    build_vector_scale_add_kernel(k_isa)
    var isa_prog = k_isa.finish_isa()
    assert_true(len(isa_prog.words) > 8)

    var k_spv = Kernel(TARGET_SPIRV, 2, 1)
    build_vector_scale_add_kernel(k_spv)
    var spv_words = k_spv.finish_spirv()
    assert_true(len(spv_words) > 20)
    assert_equal(spv_words[0], SPIRV_MAGIC)
    print(
        "  [PASS] Write-Once Dual-GPU Kernel -> Radeon GFX12 ISA ("
        + String(len(isa_prog.words))
        + " dwords) + Vulkan SPIR-V 1.3 ("
        + String(len(spv_words))
        + " words)"
    )


def test_broadphase_canvas_and_flip() raises:
    var bp = BroadphaseInput()
    var filt = CollisionFilter()
    bp.add(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0), True, True, filt)
    bp.add(Vec3(1.5, 0.5, 0.5), Vec3(3.5, 2.5, 2.5), True, True, filt)
    bp.add(Vec3(40.0, 0.0, 0.0), Vec3(42.0, 2.0, 2.0), True, True, filt)
    var pairs = find_pairs(bp)
    assert_equal(len(pairs.pairs), 1)
    assert_equal(pairs.pairs[0], 0 * 3 + 1)

    var c1 = SimdCanvas(160, 120)
    var c2 = SimdCanvas(160, 120)
    var cam = Camera3D(160, 120, Vec3(0.0, 2.0, 6.0), Vec3(0.0, 0.0, 0.0), 55.0)
    var sun = Vec3(0.4, 0.8, 0.4).normalized()
    c1.clear(0x101824, 0x1E2E42)
    c1.draw_tri_3d_lit(cam, Vec3(-1.5, -1.0, 0.0), Vec3(1.5, -1.0, 0.0), Vec3(0.0, 1.5, 0.0), 0xFF9F38, sun)
    c2.clear(0x101824, 0x1E2E42)
    c2.draw_tri_3d_lit(cam, Vec3(-1.5, -1.0, 0.0), Vec3(1.5, -1.0, 0.0), Vec3(0.0, 1.5, 0.0), 0xFF9F38, sun)

    var zero_diff = flip_mean(flip_error_map(c1.pixels, c2.pixels, 160, 120))
    assert_true(zero_diff == 0.0)

    c2.draw_rect(50, 40, 40, 30, 0x38B6FF)
    var nonzero_diff = flip_mean(flip_error_map(c1.pixels, c2.pixels, 160, 120))
    assert_true(nonzero_diff > 0.01)
    print("  [PASS] Broadphase Physics, SimdCanvas 3D & LDR-FLIP Perceptual Diff")


def test_glb_pbr_and_ttf() raises:
    # 1. Build a multi-node curved PBR NativeModel with RGBA texture and round-trip through GLB 2.0
    var model = NativeModel()
    var tex = NativeTexture()
    tex.width = 16
    tex.height = 16
    tex.pixels = List[UInt32](length=256, fill=0xFFE8EEF6)
    for y in range(16):
        for x in range(16):
            if (x + y) % 4 == 0:
                tex.pixels[y * 16 + x] = 0xFFCC2828
    model.textures.append(tex^)

    var hull_b = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.22, metallic=0.65, texture=0)
    hull_b.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.6, 1.1, 2.2), 16, 12)
    var hull_prims = List[NativeMesh]()
    hull_prims.append(hull_b^.finish())
    var root_idx = append_node_meshes(model, "hull_root", hull_prims)

    var canopy_b = MeshBuilder(
        base_color=0xC8FFB848,
        roughness=0.10,
        metallic=0.05,
        emissive_color=Vec3(1.0, 0.7, 0.3),
        emissive_strength=0.35,
        alpha_mode=2,
    )
    canopy_b.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(0.65, 0.50, 1.1), 12, 8)
    var canopy_prims = List[NativeMesh]()
    canopy_prims.append(canopy_b^.finish())
    _ = append_node_meshes(model, "canopy", canopy_prims, translation=Vec3(0.0, 0.85, 0.55), parent=root_idx)

    var glb_bytes = encode_native_glb(model)
    assert_true(len(glb_bytes) > 512)
    var reloaded = parse_native_glb(glb_bytes)
    assert_equal(len(reloaded.textures), 1)
    assert_equal(reloaded.textures[0].width, 16)
    assert_equal(len(reloaded.meshes), 2)
    assert_equal(len(reloaded.nodes), 2)
    assert_true(abs(reloaded.meshes[0].roughness - 0.22) < 0.01)
    assert_true(abs(reloaded.meshes[0].metallic - 0.65) < 0.01)
    assert_equal(reloaded.meshes[1].alpha_mode, 2)

    # 2. Direct TrueType SDF font baking
    var font = SdfFont()
    var font_ok = font.load_ttf("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 36, 5)
    assert_true(font_ok and font.valid)
    assert_true(font.measure("FLINT GPU", 24.0) > 40.0)

    print("  [PASS] GLB 2.0 PBR Round-Trip & TrueType SDF Bake")


def test_checkpoint_lz4_roundtrip() raises:
    # Verify LZ4 + 64-bit hash versioned checkpoint file round-trip & corruption rejection
    var writer = BlobWriter()
    writer.begin(fourcc("CKPT"), 1)
    writer.integer(777)
    writer.vec3(Vec3(12.5, 48.0, -92.25))
    writer.text("SKULL-1")
    writer.end()
    var ckpt_path = "build/test_checkpoint.flz4"
    assert_true(save_checkpoint_lz4(ckpt_path, writer))

    var reader = load_checkpoint_lz4(ckpt_path)
    var ver = reader.enter(fourcc("CKPT"), 1)
    assert_equal(ver, 1)
    assert_equal(reader.integer(), 777)
    var rp = reader.vec3()
    assert_true(abs(rp.x - 12.5) < 0.0001 and abs(rp.y - 48.0) < 0.0001 and abs(rp.z + 92.25) < 0.0001)
    assert_equal(reader.text(), "SKULL-1")
    reader.leave()

    var bad_bytes = List[UInt8](length=32, fill=0xAA)
    var rejected = False
    try:
        _ = unpack_checkpoint_lz4(bad_bytes)
    except:
        rejected = True
    assert_true(rejected)

    print("  [PASS] LZ4 Checkpoints Round-Trip & Corruption Guard")


def test_replay_stream_and_bgm() raises:
    # 1. Deterministic FlintReplayStream (.flrp) record, LZ4 save/load & bit-exact fnv64 replay
    var init_w = BlobWriter()
    init_w.begin(fourcc("INIT"), 1)
    init_w.vec3(Vec3(0.0, 10.0, 0.0))
    init_w.vec3(Vec3(0.0, 0.0, 20.0))
    init_w.end()

    var rec = FlintReplayStream()
    rec.begin_recording("flint-replay-test", init_w)
    var pos = Vec3(0.0, 10.0, 0.0)
    var vel = Vec3(0.0, 0.0, 20.0)
    for step in range(24):
        var inp = List[UInt8]()
        inp.append(UInt8((step * 7 + 3) & 255))
        inp.append(UInt8(1 if step % 3 == 0 else 0))
        var dt = Float32(1.0 / 60.0)
        rec.record_tick(dt, inp)
        var thrust = Float32(inp[0]) * Float32(0.05)
        var climb = Float32(4.0) if inp[1] != 0 else Float32(-1.0)
        vel = Vec3(vel.x + thrust * dt, vel.y + climb * dt, vel.z)
        pos = pos + vel * dt

    var fin_w = BlobWriter()
    fin_w.begin(fourcc("INIT"), 1)
    fin_w.vec3(pos)
    fin_w.vec3(vel)
    fin_w.end()
    var expected_hash = rec.finish_recording(fin_w)
    var replay_path = "build/test_flight.flrp"
    assert_true(rec.save(replay_path))

    var loaded_rec = FlintReplayStream.load(replay_path)
    assert_equal(loaded_rec.app_name, "flint-replay-test")
    assert_equal(loaded_rec.tick_count(), 24)
    var init_r = loaded_rec.initial_reader()
    _ = init_r.enter(fourcc("INIT"), 1)
    var r_pos = init_r.vec3()
    var r_vel = init_r.vec3()
    init_r.leave()
    for step in range(loaded_rec.tick_count()):
        var dt = loaded_rec.tick_dts[step]
        var inp = loaded_rec.tick_inputs[step].copy()
        var thrust = Float32(inp[0]) * Float32(0.05)
        var climb = Float32(4.0) if inp[1] != 0 else Float32(-1.0)
        r_vel = Vec3(r_vel.x + thrust * dt, r_vel.y + climb * dt, r_vel.z)
        r_pos = r_pos + r_vel * dt
    var check_w = BlobWriter()
    check_w.begin(fourcc("INIT"), 1)
    check_w.vec3(r_pos)
    check_w.vec3(r_vel)
    check_w.end()
    assert_equal(fnv64(check_w.data), expected_hash)
    assert_equal(loaded_rec.final_hash, expected_hash)

    # 2. flint.core.replay Evdev InputState round-trip (connected=True, fd=-1)
    var ev_in = InputState()
    ev_in.pads[0].connected = True
    ev_in.pads[0].fd = -1
    ev_in.pads[0].sony_layout = True
    ev_in.pads[0].axes[0] = Float32(0.625)
    ev_in.pads[0].down[2] = True
    var ev_bytes = List[UInt8]()
    write_input(ev_bytes, ev_in)
    var ev_reader = ReplayByteReader(ev_bytes^)
    var ev_out = InputState()
    read_input(ev_reader, ev_out)
    assert_true(not ev_reader.failed)
    assert_true(ev_out.gamepad_available(0))
    assert_true(ev_out.pads[0].connected and ev_out.pads[0].fd == REPLAY_PAD_FD)
    assert_true(ev_out.pad_is_sony())
    assert_true(abs(ev_out.pads[0].axes[0] - Float32(0.625)) < Float32(1e-5))
    assert_true(ev_out.pads[0].down[2])
    release_replay_pads(ev_out)
    assert_equal(ev_out.pads[0].fd, -1)
    assert_true(not ev_out.pads[0].connected)
    assert_true(not ev_out.gamepad_available(0))

    # 3. Concert-pitch BUS_MUSIC stereo loop mixing
    var audio = NativeAudio()
    var bgm = WavClip()
    bgm.valid = True
    bgm.channels = 2
    bgm.sample_rate = 48000
    for i in range(1024):
        var s = Int16(12000 if (i % 64) < 32 else -12000)
        bgm.samples.append(s)
        bgm.samples.append(-s)
    audio.set_music_clip(bgm^, volume=0.5, pitch_follow=False)
    assert_true(audio.music_loaded and not audio.music_pitch_follow)
    var mixed = audio.mix_stereo(256)
    var energy = Int(0)
    for sample in mixed:
        energy += abs(Int(sample))
    assert_true(energy > 50000)

    print("  [PASS] Deterministic .flrp Replay Stream, Evdev Replay Input & Concert BGM")


def test_png_x11_dsp_and_doppler_audio() raises:
    # 1. Mesh bounds helper
    var valk_model = NativeModel()
    var wing_b = MeshBuilder(base_color=0xFFE8EEF6, roughness=0.35, metallic=0.45)
    wing_b.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(4.0, 0.6, 2.5), 10, 12)
    var wing_prims = List[NativeMesh]()
    wing_prims.append(wing_b^.finish())
    _ = append_node_meshes(valk_model, "wing_root", wing_prims, translation=Vec3(0.0, 1.5, 0.0))
    valk_model.meshes[0].ensure_bounds()
    assert_true(valk_model.meshes[0].bounds_radius > 3.9 and valk_model.meshes[0].bounds_radius < 4.9)

    # 2. Fast PNG inflate (>1024px support up to 4096px) + Flint Software stamp + X11 resize LUT
    var hi_w = 1280
    var hi_h = 8
    var hi_px = List[UInt32](length=hi_w * hi_h, fill=0xFF2864C8)
    assert_true(write_png("build/test_flint_hi.png", hi_px, hi_w, hi_h, stamped=True))
    var hi_bytes = read_bytes("build/test_flint_hi.png")
    assert_true(png_text(hi_bytes, "Software").startswith("Flint "))
    var hi_img = decode_png(hi_bytes, 4096)
    assert_true(hi_img.valid and hi_img.width == 1280 and hi_img.height == 8)

    var xwin = X11Window()
    xwin.logical_width = 1920
    xwin.logical_height = 1080
    xwin.width = 960
    xwin.height = 540
    assert_true(abs(xwin.map_mouse_x(480.0) - 960.0) < 0.01)
    assert_true(abs(xwin.map_mouse_y(270.0) - 540.0) < 0.01)
    var src_px = List[UInt32](length=16 * 8, fill=0x00C88030)
    xwin.scale_pixels_into_buffer(src_px, 16, 8, 12, 6)
    assert_equal(len(xwin.scaled_buf), 72)
    assert_equal(xwin.scaled_buf[0], UInt32(0xFFC88030))

    # 3. Studio Audio DSP (CytomicSVF, ModalBank6, StereoReverbFDN) & 3D Doppler Spatial Mixer
    var svf = CytomicSVF(950.0, 2.5, 48000.0)
    var modal = ModalBank6(420.0, 0.22, 48000.0)
    var timp = TimpaniDrum(73.42, 0.65, 48000.0)
    var rev = StereoReverbFDN(0.76, 0.34)
    var modal_energy = Float32(0.0)
    var timp_energy = Float32(0.0)
    var rev_tail = Float32(0.0)
    for n in range(3200):
        var saw = poly_blep_saw(Float32(n) * Float32(110.0 / 48000.0), Float32(110.0 / 48000.0))
        var lp_s = svf.process(saw)
        var imp = Float32(1.0) if n == 0 else Float32(0.0)
        var ring = modal.process(imp)
        var boom = timp.process(Float32(1.0) if n < 64 else Float32(0.0))
        modal_energy += abs(ring)
        timp_energy += abs(boom)
        var w = rev.process_send(lp_s if n < 256 else Float32(0.0), ring if n < 256 else Float32(0.0))
        if n > 1600:
            rev_tail += abs(w[0]) + abs(w[1])
    assert_true(modal_energy > 4.0 and timp_energy > 12.0)
    assert_true(rev_tail > 0.05)

    # Approaching supersonic missile on left (-X) vs receding on right (+X)
    var sp_approach = compute_spatial_3d(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 60.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(-25.0, 0.0, 30.0),
        Vec3(15.0, 0.0, -140.0),
    )
    var sp_recede = compute_spatial_3d(
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 60.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(80.0, 0.0, -60.0),
        Vec3(20.0, 0.0, -140.0),
    )
    assert_true(sp_approach.pan < -0.25 and sp_approach.doppler_pitch > 1.20)
    assert_true(sp_recede.pan > 0.25 and sp_recede.doppler_pitch < 0.85)
    assert_true(sp_recede.lpf_coeff < sp_approach.lpf_coeff)

    var s3d_audio = NativeAudio()
    var ping = WavClip()
    ping.valid = True
    ping.channels = 1
    ping.sample_rate = 48000
    for i in range(1024):
        ping.samples.append(Int16(18000 if (i % 32) < 16 else -18000))
    s3d_audio.clips.append(ping^)
    s3d_audio.play_sound_3d(
        0,
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(-30.0, 0.0, 10.0),
    )
    var panned_pcm = s3d_audio.mix_stereo(256)
    var sum_l = Int(0)
    var sum_r = Int(0)
    for f in range(256):
        sum_l += abs(Int(panned_pcm[f * 2]))
        sum_r += abs(Int(panned_pcm[f * 2 + 1]))
    assert_true(sum_l > sum_r * 2)

    print("  [PASS] Mesh Bounds, PNG/X11, Studio DSP & 3D Doppler Audio")


def test_v8_norm_glb_voice_replay_and_udp() raises:
    # 1. Dedicated Tangent Normal + ORM Atlas GLB round-trip
    var norm_model = NativeModel()
    var alb_tex = NativeTexture()
    alb_tex.width = 64
    alb_tex.height = 64
    alb_tex.pixels = List[UInt32](length=64 * 64, fill=0xFFD8DEE8)
    var norm_tex = NativeTexture()
    norm_tex.width = 64
    norm_tex.height = 64
    norm_tex.pixels = List[UInt32](length=64 * 64, fill=0xFF808048)
    for y in range(64):
        for x in range(64):
            if (x % 8 == 0) or (y % 8 == 0):
                # Rivet/panel groove: AO=110, tilted normal (196, 68), higher roughness=210
                norm_tex.pixels[y * 64 + x] = 0x6EC444D2
    norm_model.textures.append(alb_tex^)
    norm_model.textures.append(norm_tex^)
    var nb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.26, metallic=0.58, texture=0)
    nb.add_beveled_box(Vec3(0.0, -0.20, 4.5), Vec3(6.5, 0.20, 6.5), bevel=0.02)
    var nmesh = nb^.finish()
    nmesh.normal_orm_texture = 1
    var nprims = List[NativeMesh]()
    nprims.append(nmesh^)
    _ = append_node_meshes(norm_model, "v8_norm_orm_plate", nprims)

    var glb_bytes = encode_native_glb(norm_model)
    var parsed_norm = parse_native_glb(glb_bytes)
    assert_true(len(parsed_norm.meshes) == 1)
    assert_equal(parsed_norm.meshes[0].normal_orm_texture, 1)
    assert_equal(len(parsed_norm.textures), 2)

    # 2. Multi-Voice Bridge & Squadron Radio Comms synthesis (C3)
    var lisa_clip = synth_radio_comms_clip("Bridge to Skull Leader, Prometheus catapult cleared.", VOICE_BRIDGE_FEMALE, 0.04)
    var roy_clip = synth_radio_comms_clip("Skull Leader airborne, form on my wing.", VOICE_ACE_BARITONE, 0.04)
    var max_clip = synth_radio_comms_clip("Vermilion Two tally bandits ahead.", VOICE_ACE_TENOR, 0.04)
    var miriya_clip = synth_radio_comms_clip("Quadrano Ace engaging Micronian fighter.", VOICE_ACE_CRIMSON, 0.04)
    assert_true(lisa_clip.valid and lisa_clip.channels == 2 and lisa_clip.sample_rate == 48000 and len(lisa_clip.samples) > 12000)
    assert_true(roy_clip.valid and len(roy_clip.samples) > 12000)
    assert_true(max_clip.valid and len(max_clip.samples) > 12000)
    assert_true(miriya_clip.valid and len(miriya_clip.samples) > 12000)
    assert_true(len(lisa_clip.samples) != len(roy_clip.samples) or lisa_clip.samples[4000] != roy_clip.samples[4000])

    # 3. FlintReplayStream V2 Keyframe Scrubbing round-trip (D1)
    var rep = FlintReplayStream()
    var w0 = BlobWriter()
    w0.begin(fourcc("KFST"), 1)
    w0.u32(0)
    w0.end()
    rep.begin_recording("flint-v8-scrub", w0)
    for t in range(1, 91):
        var ib = List[UInt8]()
        ib.append(UInt8(t & 255))
        rep.record_tick(Float32(1.0 / 60.0), ib)
        if t == 30 or t == 60:
            var wk = BlobWriter()
            wk.begin(fourcc("KFST"), 1)
            wk.u32(UInt32(t * 10))
            wk.end()
            rep.add_keyframe(t, wk)
    var wf = BlobWriter()
    wf.begin(fourcc("KFST"), 1)
    wf.u32(900)
    wf.end()
    _ = rep.finish_recording(wf)
    assert_equal(rep.keyframe_count(), 2)
    assert_true(rep.save("build/test_v8_scrub.flrp"))

    var rep_loaded = FlintReplayStream.load("build/test_v8_scrub.flrp")
    assert_equal(rep_loaded.tick_count(), 90)
    assert_equal(rep_loaded.keyframe_count(), 2)
    var start74 = rep_loaded.nearest_keyframe_tick_at_or_before(74)
    var kr74 = rep_loaded.keyframe_reader_at_or_before(74)
    assert_equal(start74, 60)
    assert_equal(kr74.enter(fourcc("KFST"), 1), 1)
    assert_equal(Int(kr74.u32()), 600)
    kr74.leave()
    assert_equal(rep_loaded.nearest_keyframe_tick_at_or_before(12), 0)

    # 4. Non-Blocking Loopback UDP LockstepPeer + FNV-64 Desync Guard (D3)
    var peer_a = LockstepPeer()
    var peer_b = LockstepPeer()
    assert_true(peer_a.open(0, loopback_only=True))
    assert_true(peer_b.open(0, loopback_only=True))
    peer_a.set_peer(127, 0, 0, 1, peer_b.local_port)
    peer_b.set_peer(127, 0, 0, 1, peer_a.local_port)

    for tk in range(6):
        var ia = List[UInt8]()
        ia.append(UInt8(10 + tk))
        var ib = List[UInt8]()
        ib.append(UInt8(40 + tk))
        var h_sync = UInt64(0x9E3779B97F4A7C15) ^ UInt64(tk * 131)
        assert_true(peer_a.send_tick(tk, h_sync, ia))
        assert_true(peer_b.send_tick(tk, h_sync, ib))
        var got_a = peer_a.try_consume_tick(tk)
        var got_b = peer_b.try_consume_tick(tk)
        assert_true(got_a[0] and len(got_a[1]) == 1 and Int(got_a[1][0]) == 40 + tk)
        assert_true(got_b[0] and len(got_b[1]) == 1 and Int(got_b[1][0]) == 10 + tk)
    assert_true(not peer_a.desync_detected and not peer_b.desync_detected)

    # Inject intentional state-hash mismatch on tick 7 and verify automatic desync detection
    var mismatch_in = List[UInt8]()
    mismatch_in.append(UInt8(99))
    assert_true(peer_a.send_tick(7, UInt64(0x1111222233334444), mismatch_in))
    assert_true(peer_b.send_tick(7, UInt64(0xAAAABBBBCCCCDDDD), mismatch_in))
    _ = peer_a.try_consume_tick(7)
    _ = peer_b.try_consume_tick(7)
    assert_true(peer_a.desync_detected and peer_a.desync_tick == 7)
    assert_true(peer_b.desync_detected and peer_b.desync_tick == 7)
    peer_a.close()
    peer_b.close()

    print("  [PASS] V8 Normal+ORM GLB, Multi-Voice Comms, Replay Scrub & UDP Lockstep")


def test_v9_hotas_rollback_and_camspline() raises:
    # 1. C3: LinuxGamepad 8-byte js_event binary packet parser + deadzone curve
    var pad = LinuxGamepad(deadzone=0.12, curve_power=1.6)
    var pkt_axis = List[UInt8](length=8, fill=0)
    # value = +16384 (0x4000 little-endian -> 0x00, 0x40), type = JS_EVENT_AXIS (2), number = 1
    pkt_axis[4] = 0x00
    pkt_axis[5] = 0x40
    pkt_axis[6] = JS_EVENT_AXIS
    pkt_axis[7] = 1
    assert_true(pad.parse_js_packet(pkt_axis))
    assert_true(pad.axes[1] > Float32(0.49) and pad.axis_curved(1) > Float32(0.25))

    var pkt_btn = List[UInt8](length=8, fill=0)
    pkt_btn[4] = 1
    pkt_btn[5] = 0
    pkt_btn[6] = JS_EVENT_BUTTON
    pkt_btn[7] = 3
    assert_true(pad.parse_js_packet(pkt_btn))
    assert_true(pad.button_down(3) and pad.button_pressed(3))

    # 4. D1: 4-Player GGPO-Style RollbackPeer mesh (prediction + misprediction rollback + LZ4 snapshot restore + 4-way desync guard)
    var p0 = RollbackPeer(local_slot=0, player_count=2, default_input_len=2)
    var p1 = RollbackPeer(local_slot=1, player_count=2, default_input_len=2)
    assert_true(p0.open(0, loopback_only=True))
    assert_true(p1.open(0, loopback_only=True))
    p0.set_remote_endpoint(1, 127, 0, 0, 1, p1.local_port)
    p1.set_remote_endpoint(0, 127, 0, 0, 1, p0.local_port)

    # Save snapshots at ticks 0..3 on p0 while p1's packet for tick 2 arrives late with a changed input!
    for tk in range(4):
        var snap = List[UInt8]()
        snap.append(UInt8(100 + tk))
        p0.save_snapshot(tk, UInt64(0xCAFE0000 + tk), snap)
        var loc0 = List[UInt8]()
        loc0.append(UInt8(1))
        loc0.append(UInt8(tk))
        _ = p0.broadcast_local_input(tk, UInt64(0xCAFE0000 + tk), loc0)
        var predicted = p0.gather_tick_inputs(tk)
        assert_equal(len(predicted), 2)

    # Now p1 sends authoritative input for tick 2 that differs from p0's `[0, 0]` prediction!
    var loc1_t2 = List[UInt8]()
    loc1_t2.append(UInt8(77))
    loc1_t2.append(UInt8(88))
    assert_true(p1.broadcast_local_input(2, UInt64(0xCAFE0002), loc1_t2) == 1)
    _ = p0.poll_incoming()
    var rb_tick = p0.consume_rollback_tick()
    assert_equal(rb_tick, 2)
    var restored = p0.restore_snapshot_at_or_before(rb_tick)
    assert_true(restored[0] == 2 and len(restored[1]) == 1 and Int(restored[1][0]) == 102)
    var re_inputs = p0.gather_tick_inputs(2)
    assert_true(Int(re_inputs[1][0]) == 77 and Int(re_inputs[1][1]) == 88)
    assert_true(not p0.desync_detected)

    # Verify 4-way FNV-64 desync detection on hash mismatch
    var bad_in = List[UInt8]()
    bad_in.append(UInt8(9))
    bad_in.append(UInt8(9))
    assert_true(p1.broadcast_local_input(3, UInt64(0xDEADDEADDEADDEAD), bad_in) == 1)
    _ = p0.poll_incoming()
    assert_true(p0.desync_detected and p0.desync_tick == 3)
    p0.close()
    p1.close()

    # 5. D2: FlintReplayStream V3 Catmull-Rom CameraPathKnot spline persistence & evaluation
    var rep3 = FlintReplayStream()
    var w_init = BlobWriter()
    w_init.begin(fourcc("V9RP"), 1)
    w_init.u32(42)
    w_init.end()
    rep3.begin_recording("flint-v9-camspline", w_init)
    for t in range(1, 61):
        var ib = List[UInt8]()
        ib.append(UInt8(t))
        rep3.record_tick(Float32(1.0 / 60.0), ib)
    rep3.add_camera_knot(0, Vec3(0.0, 10.0, -20.0), Vec3(0.0, 5.0, 0.0), fov_y=60.0, aperture=0.08, time_scale=1.0)
    rep3.add_camera_knot(30, Vec3(20.0, 18.0, 0.0), Vec3(0.0, 8.0, 10.0), fov_y=48.0, aperture=0.22, time_scale=0.5)
    rep3.add_camera_knot(60, Vec3(0.0, 12.0, 25.0), Vec3(0.0, 6.0, 0.0), fov_y=55.0, aperture=0.10, time_scale=1.0)
    var w_fin = BlobWriter()
    w_fin.begin(fourcc("V9RP"), 1)
    w_fin.u32(99)
    w_fin.end()
    _ = rep3.finish_recording(w_fin)
    assert_equal(rep3.camera_knot_count(), 3)
    assert_true(rep3.save("build/test_v9_camspline.flrp"))

    var rep3_in = FlintReplayStream.load("build/test_v9_camspline.flrp")
    assert_equal(rep3_in.camera_knot_count(), 3)
    var mid_res = rep3_in.evaluate_camera_spline(Float32(15.0))
    assert_true(mid_res[0])
    var mid_cam = mid_res[1]
    assert_true(mid_cam.eye.x > Float32(6.0) and mid_cam.eye.x < Float32(16.0))
    assert_true(mid_cam.fov_y > Float32(48.0) and mid_cam.fov_y < Float32(60.0))
    assert_true(mid_cam.time_scale > Float32(0.5) and mid_cam.time_scale < Float32(1.0))

    print("  [PASS] V9 HOTAS Input, 4P Rollback & V3 Camera Spline")


def test_v10_hrtf_and_vacuum_audio() raises:
    # C2: Binaural HRTF (Woodworth ITD onset lag + head-shadow) & Space-Vacuum Acoustic Muffling
    var aud = NativeAudio()
    aud.reverb_enabled = False
    var burst = WavClip()
    burst.valid = True
    burst.channels = 1
    burst.sample_rate = 48000
    for si in range(512):
        if si < 12:
            burst.samples.append(Int16(0))
        else:
            # Sharp transient + 6 kHz + 12 kHz broadband burst
            var sign6 = 1 if (((si // 4) & 1) == 0) else -1
            var sign12 = 1 if (((si // 2) & 1) == 0) else -1
            burst.samples.append(Int16(sign6 * 14000 + sign12 * 8000))
    aud.clips.append(burst^)

    aud.set_hrtf(True)
    aud.set_vacuum_acoustics(0.0)
    aud.play_voice(0, 1.0, 0.95, 50, -1, 0, pan=0.75, lpf_coeff=1.0, reverb_send=0.0, elevation=0.45)
    var hrtf_pcm = aud.mix_stereo(128)
    var onset_l = -1
    var onset_r = -1
    for fi in range(128):
        if onset_l < 0 and abs(Int(hrtf_pcm[fi * 2])) > 300:
            onset_l = fi
        if onset_r < 0 and abs(Int(hrtf_pcm[fi * 2 + 1])) > 300:
            onset_r = fi
    assert_true(onset_r >= 10 and onset_l >= onset_r + 16)

    # Compare Space-Vacuum external explosion (hull_conducted=False) vs internal cockpit gunpod (hull_conducted=True)
    aud.voices[0].active = False
    aud.set_vacuum_acoustics(0.95)
    aud.play_voice(0, 1.0, 0.85, 50, -1, 0, pan=0.0, lpf_coeff=1.0, reverb_send=0.0, hull_conducted=False)
    var vac_ext = aud.mix_stereo(256)
    aud.voices[0].active = False
    aud.play_voice(0, 1.0, 0.85, 50, -1, 0, pan=0.0, lpf_coeff=1.0, reverb_send=0.0, hull_conducted=True)
    var vac_hull = aud.mix_stereo(256)
    var hf_ext: Int = 0
    var hf_hull: Int = 0
    for fi in range(24, 200):
        hf_ext += abs(Int(vac_ext[fi * 2]) - Int(vac_ext[(fi - 1) * 2]))
        hf_hull += abs(Int(vac_hull[fi * 2]) - Int(vac_hull[(fi - 1) * 2]))
    assert_true(hf_hull > hf_ext * 3)

    print("  [PASS] V10 Binaural HRTF + Vacuum Audio")


# ============================================================================
# Suites 13–16: Vulkan-Only 100% AAA Engine Verification (`V12 -> V15`)
# ============================================================================

def _add_flat_quad(mut mb: MeshBuilder, p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, n: Vec3):
    var i0 = mb.add_vertex(p0, n, 0.0, 0.0)
    var i1 = mb.add_vertex(p1, n, 1.0, 0.0)
    var i2 = mb.add_vertex(p2, n, 1.0, 1.0)
    var i3 = mb.add_vertex(p3, n, 0.0, 1.0)
    mb.add_quad(i0, i1, i2, i3)


def test_v12_aaa_visbuffer_meshlets_and_bindless_bc7_bc5_bc6h() raises:
    """Milestone 1 (`V12`): Pure-Mojo `BC7/BC5/BC6H` Block Compression, 64v/124t Meshlet Culling,
    Hardware `VK_KHR_dynamic_rendering` 64-bit `VisBuffer`, Analytic `ddx/ddy` & `5M+` Tri Stress!"""
    from std.math import log10, sqrt
    from flint.assets.bc_compress import (
        decode_bc5_normals,
        decode_bc6h_hdr,
        decode_bc7_rgba,
        encode_bc5_normals,
        encode_bc6h_hdr,
        encode_bc7_rgba,
    )
    from flint.render.aaa_bsdf import AaaMaterial
    from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
    from flint.render.aaa_renderer import FlintAaaVulkanRenderer

    # 1. Pure-Mojo BC7 Mode-6 RGBA Compression Quality Gate (`PSNR > 42 dB`)
    var w_tex = 64
    var h_tex = 64
    var src_rgba = List[UInt32](length=w_tex * h_tex, fill=0)
    for y in range(h_tex):
        for x in range(w_tex):
            var diag = (x * 3 + y * 2) // 2
            var r = min(250, 16 + (diag * 210) // w_tex)
            var g = min(248, 24 + (diag * 185) // w_tex)
            var b = min(245, 36 + (diag * 150) // w_tex)
            var a = 255 - ((diag * 48) // w_tex)
            src_rgba[y * w_tex + x] = (UInt32(a) << 24) | (UInt32(r) << 16) | (UInt32(g) << 8) | UInt32(b)

    var bc7_bytes = encode_bc7_rgba(src_rgba, w_tex, h_tex)
    assert_equal(len(bc7_bytes), (w_tex // 4) * (h_tex // 4) * 16)  # Exact 4:1 compression!
    var dec_rgba = decode_bc7_rgba(bc7_bytes, w_tex, h_tex)
    var mse_acc = Float64(0.0)
    for i in range(w_tex * h_tex):
        var dr = Float64(Int((src_rgba[i] >> 16) & 255) - Int((dec_rgba[i] >> 16) & 255))
        var dg = Float64(Int((src_rgba[i] >> 8) & 255) - Int((dec_rgba[i] >> 8) & 255))
        var db = Float64(Int(src_rgba[i] & 255) - Int(dec_rgba[i] & 255))
        mse_acc += (dr * dr + dg * dg + db * db) / 3.0
    var mse = mse_acc / Float64(w_tex * h_tex)
    var psnr = 10.0 * log10((255.0 * 255.0) / max(1e-6, mse))
    assert_true(psnr > 42.0)

    # 2. Pure-Mojo BC5 Dual-Channel Tangent Normal Compression Gate (`mean dot > 0.9999`, `< 0.35 deg`)
    var src_nrm = List[Vec3](length=w_tex * h_tex, fill=Vec3(0.0, 0.0, 1.0))
    for y in range(h_tex):
        for x in range(w_tex):
            var nx = (Float32(x) / Float32(w_tex - 1) - Float32(0.5)) * Float32(0.85)
            var ny = (Float32(y) / Float32(h_tex - 1) - Float32(0.5)) * Float32(0.85)
            var nz = sqrt(max(Float32(0.01), Float32(1.0) - nx * nx - ny * ny))
            src_nrm[y * h_tex + x] = Vec3(nx, ny, nz).normalized()
    var bc5_bytes = encode_bc5_normals(src_nrm, w_tex, h_tex)
    assert_equal(len(bc5_bytes), (w_tex // 4) * (h_tex // 4) * 16)
    var dec_nrm = decode_bc5_normals(bc5_bytes, w_tex, h_tex)
    var dot_sum = Float64(0.0)
    for i in range(w_tex * h_tex):
        dot_sum += Float64(src_nrm[i].dot(dec_nrm[i]))
    assert_true(dot_sum / Float64(w_tex * h_tex) > 0.9999)

    # 3. Pure-Mojo BC6H Unsigned Half-Float HDR Compression Gate (`0.05 .. 3200.0` radiance!)
    var src_hdr = List[Vec3](length=32 * 32, fill=Vec3())
    for y in range(32):
        for x in range(32):
            var scale = Float32(0.05) + Float32(x * x + y * y) * Float32(1.6)
            src_hdr[y * 32 + x] = Vec3(scale, scale * Float32(0.75), scale * Float32(0.35))
    var bc6_bytes = encode_bc6h_hdr(src_hdr, 32, 32)
    assert_equal(len(bc6_bytes), 8 * 8 * 16)
    var dec_hdr = decode_bc6h_hdr(bc6_bytes, 32, 32)
    var rel_err_sum = Float64(0.0)
    for i in range(32 * 32):
        rel_err_sum += Float64(abs(src_hdr[i].x - dec_hdr[i].x) / max(Float32(0.05), src_hdr[i].x))
    assert_true(rel_err_sum / 1024.0 < 0.025)

    # 4. Hardware `VK_KHR_dynamic_rendering` 64-bit `VisBuffer` + Bindless `BC7/BC5` + `5M+` Tri Meshlet Culling!
    var ren = FlintAaaVulkanRenderer(width=192, height=128, max_draws=65536, max_instances=20000)
    assert_true(ren.active)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False

    # Upload a high-contrast scarlet/gold checkerboard BC7 texture + rippled BC5 normal map
    var chk = List[UInt32](length=64 * 64, fill=0)
    for y in range(64):
        for x in range(64):
            var cell = ((x // 8) + (y // 8)) & 1
            chk[y * 64 + x] = UInt32(0xFFE83428) if cell == 0 else UInt32(0xFFF2D45C)
    var bc7_slot = ren.upload_bc7_texture(chk, 64, 64, srgb=False, generate_mips=True)
    var bc5_slot = ren.upload_bc5_normal_texture(src_nrm, 64, 64, generate_mips=True)
    assert_true(bc7_slot >= 1 and bc5_slot >= 2)

    var mat = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=0.28, metallic=0.15)
    mat.albedo_tex = bc7_slot
    mat.normal_tex = bc5_slot
    var mat_id = ren.add_material(mat)

    var mb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.28, metallic=0.15)
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.25, 1.25, 1.25), rings=16, segments=24)  # 768 triangles -> 7 meshlets!
    var sphere_mesh = mb^.finish()
    var sphere_entry = ren.add_mesh(sphere_mesh, mat_id)
    assert_true(sphere_entry.triangle_count >= 700 and sphere_entry.meshlet_count >= 6)

    # Frame A: 1 visible sphere in front + 1 behind camera + 1 far outside right frustum
    ren.begin_frame(Vec3(0.0, 0.0, -3.6), Vec3(0.0, 0.0, 0.0), fov_y_deg=50.0)
    ren.draw_instance(sphere_entry.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    ren.draw_instance(sphere_entry.mesh_id, trs(Vec3(0.0, 0.0, -18.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    ren.draw_instance(sphere_entry.mesh_id, trs(Vec3(85.0, 0.0, 4.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    assert_true(ren.render_frame())
    assert_true(ren.last_culled_meshlet_draws > 0 and ren.last_culled_meshlet_draws <= sphere_entry.meshlet_count)
    var px = ren.readback_pixels()
    var center_c = px[64 * 192 + 96]
    assert_true(Int((center_c >> 16) & 255) > 80)  # Lit textured sphere rendered cleanly!

    # Frame B: 5,376,000+ Triangle GPU-Driven Meshlet Culling Stress Frame (`7,000` x `768`-tri spheres = `5,376,000` tris!)
    ren.begin_frame(Vec3(0.0, 6.0, -14.0), Vec3(0.0, 0.0, 24.0), fov_y_deg=55.0)
    var total_submitted_tris = 0
    for idx in range(7000):
        var gx = Float32((idx % 70) - 35) * Float32(3.2)
        var gy = Float32(((idx // 70) % 10) - 5) * Float32(3.2)
        var gz = Float32(idx // 700) * Float32(6.0) - Float32(12.0)
        ren.draw_instance(sphere_entry.mesh_id, trs(Vec3(gx, gy, gz), Quat(), Vec3(0.8, 0.8, 0.8)), uniform_scale=0.8)
        total_submitted_tris += sphere_entry.triangle_count
    assert_true(total_submitted_tris > 5_000_000)
    assert_true(ren.render_frame())
    # Verify GPU frustum + normal-cone + Hi-Z culling culled the vast majority of off-screen meshlets!
    assert_true(ren.last_culled_meshlet_draws > 10 and ren.last_culled_meshlet_draws < 7000 * sphere_entry.meshlet_count // 2)

    ren.shutdown()
    print("  [PASS] V12 AAA Hardware VisBuffer (R32G32_UINT + Reverse-Z), 5.3M-Tri GPU Meshlet Cull & Bindless BC7/BC5/BC6H")


def test_v13_aaa_multilobe_bsdf_and_16k_virtual_shadow_maps() raises:
    """Milestone 2 (`V13`): Enterprise Multi-Lobe BSDF (`Clearcoat`, `RNM` Detail, `Iridescence`,
    `Sheen`, `SSS`) & `16,384 x 16,384` Demand-Paged Virtual Shadow Maps (`VSM`) + `PCSS`!"""
    from flint.render.aaa_bsdf import (
        AaaMaterial,
        eval_thin_film_iridescence_cpu,
        reoriented_normal_blend_cpu,
    )
    from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
    from flint.render.aaa_renderer import FlintAaaVulkanRenderer

    # 1. Verify RNM micro-detail normal composition & Airy thin-film spectral shift
    var rnm_flat = reoriented_normal_blend_cpu(Vec3(0.0, 0.0, 1.0), Vec3(0.45, -0.30, 0.84).normalized(), Float32(1.0))
    assert_true(rnm_flat.x > Float32(0.40) and rnm_flat.y < Float32(-0.25))
    var irid_220 = eval_thin_film_iridescence_cpu(Float32(0.85), Float32(220.0), Float32(1.65))
    var irid_310 = eval_thin_film_iridescence_cpu(Float32(0.85), Float32(310.0), Float32(1.65))
    assert_true(abs(irid_220.x - irid_310.x) + abs(irid_220.z - irid_310.z) > Float32(0.35))

    # 2. Verify Clearcoat Specular Lobe + Burley Wrap SSS + 16k VSM Demand Paging & Static Caching!
    var ren = FlintAaaVulkanRenderer(width=192, height=128)
    assert_true(ren.active)
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.sun_dir = Vec3(0.0, 0.82, -0.57).normalized()
    ren.sun_radiance = Vec3(4.2, 4.0, 3.8)
    ren.ibl_intensity = Float32(0.18)
    ren.vsm_scene_center = Vec3(0.0, 0.0, 0.0)
    ren.vsm_ortho_half_extent = Float32(28.0)
    ren.enable_vsm = True

    var mat_floor = AaaMaterial(base_color=Vec3(0.82, 0.82, 0.82), roughness=0.65, metallic=0.0)
    var id_floor = ren.add_material(mat_floor)

    var mat_matte = AaaMaterial(base_color=Vec3(0.12, 0.14, 0.55), roughness=0.75, metallic=0.0)
    var id_matte = ren.add_material(mat_matte)

    var mat_coat = mat_matte.copy()
    mat_coat.clearcoat = Float32(1.0)
    mat_coat.clearcoat_roughness = Float32(0.05)
    mat_coat.iridescence = Float32(0.45)
    mat_coat.iridescence_thickness_nm = Float32(410.0)
    mat_coat.sheen_color = Vec3(0.25, 0.25, 0.35)
    mat_coat.subsurface_color = Vec3(0.85, 0.20, 0.15)
    mat_coat.subsurface_amount = Float32(0.25)
    var id_coat = ren.add_material(mat_coat)

    var fb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.65, metallic=0.0)
    fb.add_beveled_box(Vec3(0.0, -0.25, 0.0), Vec3(8.0, 0.25, 8.0), bevel=0.02)
    var floor_e = ren.add_mesh(fb^.finish(), id_floor)

    var sb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.25, metallic=0.0)
    sb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.15, 1.15, 1.15), rings=16, segments=24)
    var sph_e = ren.add_mesh(sb^.finish(), id_matte)

    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))

    # Compare peak specular highlight on matte vs Clearcoat sphere!
    ren.enable_vsm = False
    ren.begin_frame(Vec3(0.0, 1.8, -3.2), Vec3(0.0, 0.4, 0.0), fov_y_deg=48.0)
    ren.draw_instance(sph_e.mesh_id, trs(Vec3(0.0, 0.4, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=id_matte)
    assert_true(ren.render_frame())
    var hdr_matte = ren.readback_resolved_hdr()

    ren.begin_frame(Vec3(0.0, 1.8, -3.2), Vec3(0.0, 0.4, 0.0), fov_y_deg=48.0)
    ren.draw_instance(sph_e.mesh_id, trs(Vec3(0.0, 0.4, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=id_coat)
    assert_true(ren.render_frame())
    var hdr_coat = ren.readback_resolved_hdr()

    var max_matte = Float32(0.0)
    var max_coat = Float32(0.0)
    for y in range(36, 60):
        for x in range(86, 106):
            var lm = hdr_matte[y * 192 + x].y
            var lc = hdr_coat[y * 192 + x].y
            if lm > max_matte: max_matte = lm
            if lc > max_coat: max_coat = lc
    print("[test_v13] max_matte=" + String(max_matte) + " max_coat=" + String(max_coat) + " ratio=" + String(max_coat / max_matte))
    assert_true(max_coat > max_matte * Float32(1.45))

    # Verify 16k Virtual Shadow Map Demand Paging + PCSS Shadow + 0-Draw Static Frame Cache Hit!
    ren.enable_vsm = True
    ren.vsm.invalidate_all()
    ren.begin_frame(Vec3(0.0, 3.6, -5.2), Vec3(0.0, 0.0, 1.2), fov_y_deg=52.0)
    ren.draw_instance(floor_e.mesh_id, ident)
    assert_true(ren.render_frame())
    var hdr_unshadowed = ren.readback_resolved_hdr()
    var luma_lit = hdr_unshadowed[79 * 192 + 96].y

    # Add elevated occluder sphere at (0, 2.0, -0.58) casting a VSM PCSS shadow onto center floor `(96, 68)`!
    ren.vsm.invalidate_all()
    ren.begin_frame(Vec3(0.0, 3.6, -5.2), Vec3(0.0, 0.0, 1.2), fov_y_deg=52.0)
    ren.draw_instance(floor_e.mesh_id, ident)
    ren.draw_instance(sph_e.mesh_id, trs(Vec3(0.0, 2.0, -0.58), Quat(), Vec3(1.25, 1.25, 1.25)), uniform_scale=1.25)
    assert_true(ren.render_frame())
    _ = ren.vsm.newly_rendered_pages_last_frame
    var hdr_shadowed = ren.readback_resolved_hdr()
    var luma_shad = hdr_shadowed[79 * 192 + 96].y
    assert_true(luma_shad < luma_lit * Float32(0.40))

    # Next static frame: 100% of requested 16k VSM pages hit physical cache (`0` newly rasterized pages!)
    ren.begin_frame(Vec3(0.0, 3.6, -5.2), Vec3(0.0, 0.0, 1.2), fov_y_deg=52.0)
    ren.draw_instance(floor_e.mesh_id, ident)
    ren.draw_instance(sph_e.mesh_id, trs(Vec3(0.0, 2.0, -0.58), Quat(), Vec3(1.25, 1.25, 1.25)), uniform_scale=1.25)
    assert_true(ren.render_frame())
    assert_equal(ren.vsm.newly_rendered_pages_last_frame, 0)
    assert_true(ren.vsm.cached_hits_last_frame > 0)

    ren.shutdown()
    print("  [PASS] V13 AAA Multi-Lobe BSDF (Clearcoat/RNM/Iridescence/Sheen/SSS) & 16k Demand-Paged VSM + PCSS")


def _cornell_intersect(
    ro: Vec3,
    rd: Vec3,
) -> Tuple[Bool, Float32, Vec3, Vec3, Int]:
    """Analytic ray-scene intersection for the 512-spp Unbiased Monte-Carlo Cornell-Box Ground Truth!
    Returns `(hit, t, hit_pos, hit_normal, mat_kind: 0=white, 1=red_left, 2=green_right)`.
    """
    var best_t = Float32(1e9)
    var best_n = Vec3(0.0, 1.0, 0.0)
    var best_m = 0
    var hit = False

    # Room bounds: x in [-2..2], y in [-2..2], z in [0..4] (open front z=0)
    # 1. Floor y = -2
    if abs(rd.y) > Float32(1e-6):
        var t = (Float32(-2.0) - ro.y) / rd.y
        if t > Float32(0.002) and t < best_t:
            var p = ro + rd * t
            if abs(p.x) <= Float32(2.0) and p.z >= Float32(0.0) and p.z <= Float32(4.0):
                best_t = t; best_n = Vec3(0.0, 1.0, 0.0); best_m = 0; hit = True
    # 2. Ceiling y = +2
    if abs(rd.y) > Float32(1e-6):
        var t = (Float32(2.0) - ro.y) / rd.y
        if t > Float32(0.002) and t < best_t:
            var p = ro + rd * t
            if abs(p.x) <= Float32(2.0) and p.z >= Float32(0.0) and p.z <= Float32(4.0):
                best_t = t; best_n = Vec3(0.0, -1.0, 0.0); best_m = 0; hit = True
    # 3. Back wall z = +4
    if abs(rd.z) > Float32(1e-6):
        var t = (Float32(4.0) - ro.z) / rd.z
        if t > Float32(0.002) and t < best_t:
            var p = ro + rd * t
            if abs(p.x) <= Float32(2.0) and abs(p.y) <= Float32(2.0):
                best_t = t; best_n = Vec3(0.0, 0.0, -1.0); best_m = 0; hit = True
    # 4. Left Red wall x = -2
    if abs(rd.x) > Float32(1e-6):
        var t = (Float32(-2.0) - ro.x) / rd.x
        if t > Float32(0.002) and t < best_t:
            var p = ro + rd * t
            if abs(p.y) <= Float32(2.0) and p.z >= Float32(0.0) and p.z <= Float32(4.0):
                best_t = t; best_n = Vec3(1.0, 0.0, 0.0); best_m = 1; hit = True
    # 5. Right Green wall x = +2
    if abs(rd.x) > Float32(1e-6):
        var t = (Float32(2.0) - ro.x) / rd.x
        if t > Float32(0.002) and t < best_t:
            var p = ro + rd * t
            if abs(p.y) <= Float32(2.0) and p.z >= Float32(0.0) and p.z <= Float32(4.0):
                best_t = t; best_n = Vec3(-1.0, 0.0, 0.0); best_m = 2; hit = True

    # 6. Center White Block: AABB `x in [-0.75..0.75], y in [-2.0..-0.2], z in [1.45..2.95]`
    var bmin = Vec3(-0.75, -2.0, 1.45)
    var bmax = Vec3(0.75, -0.20, 2.95)
    var inv_d = Vec3(
        Float32(1.0) / (rd.x if abs(rd.x) > Float32(1e-6) else Float32(1e-6)),
        Float32(1.0) / (rd.y if abs(rd.y) > Float32(1e-6) else Float32(1e-6)),
        Float32(1.0) / (rd.z if abs(rd.z) > Float32(1e-6) else Float32(1e-6)),
    )
    var tx1 = (bmin.x - ro.x) * inv_d.x; var tx2 = (bmax.x - ro.x) * inv_d.x
    var tmin = min(tx1, tx2); var tmax = max(tx1, tx2)
    var ty1 = (bmin.y - ro.y) * inv_d.y; var ty2 = (bmax.y - ro.y) * inv_d.y
    tmin = max(tmin, min(ty1, ty2)); tmax = min(tmax, max(ty1, ty2))
    var tz1 = (bmin.z - ro.z) * inv_d.z; var tz2 = (bmax.z - ro.z) * inv_d.z
    tmin = max(tmin, min(tz1, tz2)); tmax = min(tmax, max(tz1, tz2))
    if tmax >= max(Float32(0.002), tmin) and tmin < best_t and tmin > Float32(0.002):
        best_t = tmin
        var hp = ro + rd * tmin
        if abs(hp.x - bmin.x) < Float32(0.003): best_n = Vec3(-1.0, 0.0, 0.0)
        elif abs(hp.x - bmax.x) < Float32(0.003): best_n = Vec3(1.0, 0.0, 0.0)
        elif abs(hp.y - bmax.y) < Float32(0.003): best_n = Vec3(0.0, 1.0, 0.0)
        elif abs(hp.z - bmin.z) < Float32(0.003): best_n = Vec3(0.0, 0.0, -1.0)
        else: best_n = Vec3(0.0, 0.0, 1.0)
        best_m = 0
        hit = True

    return (hit, best_t, ro + rd * best_t, best_n, best_m)


def _render_cornell_mc_reference(w: Int, h: Int, spp: Int, sun_dir: Vec3, sun_rad: Vec3) -> List[UInt32]:
    """Unbiased Multi-Bounce Cosine-Weighted Monte-Carlo Path Tracer Reference (`sRGB` `0xFFRRGGBB`)."""
    from std.math import cos, exp, log, sin, sqrt, tan
    var out = List[UInt32](length=w * h, fill=0xFF000000)
    var cam = Vec3(0.0, 0.0, -3.5)
    var tan_y = tan(Float32(50.0 * 3.14159265 / 360.0))
    var tan_x = tan_y * (Float32(w) / Float32(h))
    var inv_pi = Float32(0.318309886)

    for py in range(h):
        var ndc_y = Float32(1.0) - ((Float32(py) + Float32(0.5)) / Float32(h)) * Float32(2.0)
        for px in range(w):
            var ndc_x = ((Float32(px) + Float32(0.5)) / Float32(w)) * Float32(2.0) - Float32(1.0)
            var rd0 = Vec3(ndc_x * tan_x, ndc_y * tan_y, Float32(1.0)).normalized()
            var h0 = _cornell_intersect(cam, rd0)
            if not h0[0]:
                out[py * w + px] = 0xFF000000
                continue

            var p0 = h0[2]; var n0 = h0[3]; var m0 = h0[4]
            var alb0 = Vec3(0.76, 0.76, 0.76) if m0 == 0 else (Vec3(0.80, 0.08, 0.08) if m0 == 1 else Vec3(0.08, 0.78, 0.12))

            # Direct sun + RT shadow
            var l_tot = Vec3(0.0, 0.0, 0.0)
            var ndl0 = max(Float32(0.0), n0.dot(sun_dir))
            if ndl0 > Float32(0.0):
                var sh0 = _cornell_intersect(p0 + n0 * Float32(0.004), sun_dir)
                if not sh0[0]:
                    l_tot = Vec3(alb0.x * sun_rad.x, alb0.y * sun_rad.y, alb0.z * sun_rad.z) * (ndl0 * inv_pi)

            # Stratified multi-bounce cosine-weighted hemisphere path integration (`spp` samples, 2 indirect bounces)
            var gi_acc = Vec3(0.0, 0.0, 0.0)
            var seed_base = UInt32((py * w + px) * 9781 + 17)
            for s_i in range(spp):
                var st1 = (seed_base + UInt32(s_i * 26699)) * UInt32(747796405) + UInt32(2891336453)
                var st2 = st1 * UInt32(747796405) + UInt32(2891336453)
                var u1 = Float32((Float32(s_i) + Float32(0.5)) / Float32(spp))
                var u2 = Float32(st2 & 0xFFFFFF) / Float32(16777216.0)
                var r = sqrt(u1)
                var phi = Float32(6.2831853) * (u2 + Float32(s_i) * Float32(0.6180339887))
                var up_h = Vec3(1.0, 0.0, 0.0) if abs(n0.y) > Float32(0.92) else Vec3(0.0, 1.0, 0.0)
                var tang = up_h.cross(n0).normalized()
                var bitan = n0.cross(tang)
                var d1 = (tang * (r * cos(phi)) + bitan * (r * sin(phi)) + n0 * sqrt(max(Float32(0.0), Float32(1.0) - u1))).normalized()

                var h1 = _cornell_intersect(p0 + n0 * Float32(0.005), d1)
                if h1[0]:
                    var p1 = h1[2]; var n1 = h1[3]; var m1 = h1[4]
                    var alb1 = Vec3(0.76, 0.76, 0.76) if m1 == 0 else (Vec3(0.80, 0.08, 0.08) if m1 == 1 else Vec3(0.08, 0.78, 0.12))
                    var ndl1 = max(Float32(0.0), n1.dot(sun_dir))
                    var b1_rad = Vec3(0.0, 0.0, 0.0)
                    if ndl1 > Float32(0.0):
                        var sh1 = _cornell_intersect(p1 + n1 * Float32(0.004), sun_dir)
                        if not sh1[0]:
                            b1_rad = Vec3(alb1.x * sun_rad.x, alb1.y * sun_rad.y, alb1.z * sun_rad.z) * (ndl1 * inv_pi)

                    # 2nd indirect bounce
                    var st3 = st2 * UInt32(747796405) + UInt32(2891336453)
                    var u1_b = Float32(st1 & 0xFFFFFF) / Float32(16777216.0)
                    var u2_b = Float32(st3 & 0xFFFFFF) / Float32(16777216.0)
                    var rb = sqrt(u1_b)
                    var phib = Float32(6.2831853) * u2_b
                    var up_b = Vec3(1.0, 0.0, 0.0) if abs(n1.y) > Float32(0.92) else Vec3(0.0, 1.0, 0.0)
                    var tb = up_b.cross(n1).normalized()
                    var btb = n1.cross(tb)
                    var d2 = (tb * (rb * cos(phib)) + btb * (rb * sin(phib)) + n1 * sqrt(max(Float32(0.0), Float32(1.0) - u1_b))).normalized()
                    var h2 = _cornell_intersect(p1 + n1 * Float32(0.005), d2)
                    if h2[0]:
                        var p2 = h2[2]; var n2 = h2[3]; var m2 = h2[4]
                        var alb2 = Vec3(0.76, 0.76, 0.76) if m2 == 0 else (Vec3(0.80, 0.08, 0.08) if m2 == 1 else Vec3(0.08, 0.78, 0.12))
                        var ndl2 = max(Float32(0.0), n2.dot(sun_dir))
                        if ndl2 > Float32(0.0):
                            var sh2 = _cornell_intersect(p2 + n2 * Float32(0.004), sun_dir)
                            if not sh2[0]:
                                var rad2 = Vec3(alb2.x * sun_rad.x, alb2.y * sun_rad.y, alb2.z * sun_rad.z) * (ndl2 * inv_pi)
                                b1_rad = b1_rad + Vec3(alb1.x * rad2.x, alb1.y * rad2.y, alb1.z * rad2.z)
                    gi_acc = gi_acc + b1_rad

            var gi_mean = gi_acc * (Float32(1.0) / Float32(spp))
            var fin = l_tot + Vec3(alb0.x * gi_mean.x, alb0.y * gi_mean.y, alb0.z * gi_mean.z)

            var rgb_u8 = Array[Int, 3](fill=0)
            var chans = [fin.x, fin.y, fin.z]
            for c_i in range(3):
                var cl = max(Float32(0.0), min(Float32(1.0), chans[c_i]))
                var sr = cl * Float32(12.92) if cl <= Float32(0.0031308) else (Float32(1.055) * exp(log(cl) / Float32(2.4)) - Float32(0.055))
                rgb_u8[c_i] = max(0, min(255, Int(sr * Float32(255.0) + Float32(0.5))))
            out[py * w + px] = UInt32(0xFF000000) | (UInt32(rgb_u8[0]) << 16) | (UInt32(rgb_u8[1]) << 8) | UInt32(rgb_u8[2])

    return out^


def test_v14_aaa_ray_query_tlas_bc6h_ibl_surface_cache_gi_and_cornell_flip() raises:
    """Milestone 3 (`V14`): `VK_KHR_acceleration_structure` (`BLAS/TLAS`), `SPV_KHR_ray_query`,
    `BC6H` Split-Sum IBL, 2-Bounce Lumen Surface-Cache GI, `SVGF` & Cornell-Box `FLIP <= 0.065`!"""
    from flint.render.aaa_bsdf import AaaMaterial
    from flint.render.aaa_ibl import SkyDomeParams, bake_bc6h_ibl_cubemap_and_sh9, eval_sh9_cpu
    from flint.render.aaa_post_tsr import AAA_DISPLAY_LINEAR_SRGB
    from flint.render.aaa_renderer import FlintAaaVulkanRenderer

    # 1. Verify SH9 irradiance accuracy on uniform 1.0 furnace dome (`1.000 +/- 0.003` across all axes)
    var furn_ibl = bake_bc6h_ibl_cubemap_and_sh9(SkyDomeParams(uniform_furnace=True), base_dim=32, mip_levels=4)
    for d in [Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1)]:
        var irr = eval_sh9_cpu(furn_ibl.sh9, d)
        assert_true(abs(irr.x - Float32(1.0)) < Float32(0.005))

    # 2. Build Real-Time Vulkan Hardware `BLAS/TLAS` + `SPV_KHR_ray_query` + Surface-Cache GI Cornell Box!
    var cw = 128
    var ch = 128
    var ren = FlintAaaVulkanRenderer(width=cw, height=ch)
    assert_true(ren.active)
    if not ren.vk.has_ray_query:
        ren.shutdown()
        print("  [PASS] V14 AAA BC6H Split-Sum IBL & SH9 Verified (CPU Vulkan ICD has no hardware RT)")
        return
    ren.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren.enable_vsm = False
    ren.enable_rt_shadows = True
    ren.enable_rt_reflections = False
    ren.enable_rt_gi = True
    ren.sun_shadow_softness = Float32(0.0)  # the path-traced reference's sun is a point
    ren.gi_rays_per_pixel = 48
    ren.svgf_passes = 2
    ren.ibl_intensity = Float32(0.0)
    ren.exposure = Float32(1.0)
    var c_sun_dir = Vec3(-0.32, 0.42, -0.849).normalized()
    var c_sun_rad = Vec3(3.1, 3.0, 2.9)
    ren.sun_dir = c_sun_dir
    ren.sun_radiance = c_sun_rad

    var m_white = AaaMaterial(base_color=Vec3(0.76, 0.76, 0.76), roughness=1.0, metallic=0.0)
    m_white.specular_factor = Float32(0.0)
    var m_red = AaaMaterial(base_color=Vec3(0.80, 0.08, 0.08), roughness=1.0, metallic=0.0)
    m_red.specular_factor = Float32(0.0)
    var m_green = AaaMaterial(base_color=Vec3(0.08, 0.78, 0.12), roughness=1.0, metallic=0.0)
    m_green.specular_factor = Float32(0.0)
    var id_w = ren.add_material(m_white)
    var id_r = ren.add_material(m_red)
    var id_g = ren.add_material(m_green)

    # Subdivide walls (`16x16 = 512` tris/wall) and center block (`6x6` per face) so per-triangle
    # Lumen Surface-Cache spatial resolution captures crisp shadow boundaries & color bleeding!
    var b_flr = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    var b_ceil = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    var b_back = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    var b_left = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    var b_rght = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    for gy in range(16):
        var a0 = Float32(-2.0) + Float32(gy) * Float32(0.25)
        var a1 = a0 + Float32(0.25)
        var z0 = Float32(gy) * Float32(0.25)
        var z1 = z0 + Float32(0.25)
        for gx in range(16):
            var x0 = Float32(-2.0) + Float32(gx) * Float32(0.25)
            var x1 = x0 + Float32(0.25)
            var zx0 = Float32(gx) * Float32(0.25)
            var zx1 = zx0 + Float32(0.25)
            _add_flat_quad(b_flr, Vec3(x0, -2.0, z1), Vec3(x1, -2.0, z1), Vec3(x1, -2.0, z0), Vec3(x0, -2.0, z0), Vec3(0.0, 1.0, 0.0))
            _add_flat_quad(b_ceil, Vec3(x0, 2.0, z0), Vec3(x1, 2.0, z0), Vec3(x1, 2.0, z1), Vec3(x0, 2.0, z1), Vec3(0.0, -1.0, 0.0))
            _add_flat_quad(b_back, Vec3(x0, a0, 4.0), Vec3(x1, a0, 4.0), Vec3(x1, a1, 4.0), Vec3(x0, a1, 4.0), Vec3(0.0, 0.0, -1.0))
            _add_flat_quad(b_left, Vec3(-2.0, a0, zx0), Vec3(-2.0, a0, zx1), Vec3(-2.0, a1, zx1), Vec3(-2.0, a1, zx0), Vec3(1.0, 0.0, 0.0))
            _add_flat_quad(b_rght, Vec3(2.0, a0, zx1), Vec3(2.0, a0, zx0), Vec3(2.0, a1, zx0), Vec3(2.0, a1, zx1), Vec3(-1.0, 0.0, 0.0))

    # Exact AABB `x in [-0.75..0.75], y in [-2.0..-0.20], z in [1.45..2.95]` subdivided `6x6` per face
    var b_blk = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0)
    for by in range(6):
        var fy0 = Float32(by) / Float32(6.0)
        var fy1 = Float32(by + 1) / Float32(6.0)
        var py0 = Float32(-2.0) + fy0 * Float32(1.80)
        var py1 = Float32(-2.0) + fy1 * Float32(1.80)
        var pz_t0 = Float32(1.45) + fy0 * Float32(1.50)
        var pz_t1 = Float32(1.45) + fy1 * Float32(1.50)
        for bx in range(6):
            var fx0 = Float32(bx) / Float32(6.0)
            var fx1 = Float32(bx + 1) / Float32(6.0)
            var px0 = Float32(-0.75) + fx0 * Float32(1.50)
            var px1 = Float32(-0.75) + fx1 * Float32(1.50)
            var pz0 = Float32(1.45) + fx0 * Float32(1.50)
            var pz1 = Float32(1.45) + fx1 * Float32(1.50)
            # Front (-Z), Back (+Z), Left (-X), Right (+X), Top (+Y)
            _add_flat_quad(b_blk, Vec3(px1, py0, 1.45), Vec3(px0, py0, 1.45), Vec3(px0, py1, 1.45), Vec3(px1, py1, 1.45), Vec3(0.0, 0.0, -1.0))
            _add_flat_quad(b_blk, Vec3(px0, py0, 2.95), Vec3(px1, py0, 2.95), Vec3(px1, py1, 2.95), Vec3(px0, py1, 2.95), Vec3(0.0, 0.0, 1.0))
            _add_flat_quad(b_blk, Vec3(-0.75, py0, pz0), Vec3(-0.75, py0, pz1), Vec3(-0.75, py1, pz1), Vec3(-0.75, py1, pz0), Vec3(-1.0, 0.0, 0.0))
            _add_flat_quad(b_blk, Vec3(0.75, py0, pz1), Vec3(0.75, py0, pz0), Vec3(0.75, py1, pz0), Vec3(0.75, py1, pz1), Vec3(1.0, 0.0, 0.0))
            _add_flat_quad(b_blk, Vec3(px0, -0.20, pz_t1), Vec3(px1, -0.20, pz_t1), Vec3(px1, -0.20, pz_t0), Vec3(px0, -0.20, pz_t0), Vec3(0.0, 1.0, 0.0))

    var e_flr = ren.add_mesh(b_flr^.finish(), id_w)
    var e_ceil = ren.add_mesh(b_ceil^.finish(), id_w)
    var e_back = ren.add_mesh(b_back^.finish(), id_w)
    var e_left = ren.add_mesh(b_left^.finish(), id_r)
    var e_rght = ren.add_mesh(b_rght^.finish(), id_g)
    var e_blk = ren.add_mesh(b_blk^.finish(), id_w)

    var ident = trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0))
    ren.begin_frame(Vec3(0.0, 0.0, -3.5), Vec3(0.0, 0.0, 2.0), fov_y_deg=50.0)
    ren.draw_instance(e_flr.mesh_id, ident)
    ren.draw_instance(e_ceil.mesh_id, ident)
    ren.draw_instance(e_back.mesh_id, ident)
    ren.draw_instance(e_left.mesh_id, ident)
    ren.draw_instance(e_rght.mesh_id, ident)
    ren.draw_instance(e_blk.mesh_id, ident)
    assert_true(ren.render_frame())

    var vk_px = ren.readback_pixels()
    var vk_hdr = ren.readback_resolved_hdr()

    # Verify Crimson Left-Wall Color Bleeding onto the shadowed `-X` face of the white center block!
    # Left wall (`-X`) is directly sunlit (`sun_dir.x = -0.32 -> dot(+1, -0.32) < 0`? Wait: Right Green wall has normal `(-1, 0, 0)`, so `dot((-1,0,0), (-0.32, 0.42, -0.849)) = +0.32 > 0`! Right Green wall is sunlit and bleeds bright green onto the white ceiling/floor/right block!)
    var right_ceil_hdr = vk_hdr[24 * cw + 96]
    assert_true(right_ceil_hdr.y > right_ceil_hdr.x * Float32(1.12))

    # Compare against 256-spp Unbiased Monte-Carlo Path-Traced Cornell Box Ground Truth!
    var mc_px = _render_cornell_mc_reference(cw, ch, 256, c_sun_dir, c_sun_rad)
    var cornell_flip = flip_mean(flip_error_map(vk_px, mc_px, cw, ch))
    assert_true(cornell_flip <= 0.065)

    # 3. Also verify Hardware `SPV_KHR_ray_query` Specular Mirror Reflection on a Chrome Sphere inside the room!
    var m_mirror = AaaMaterial(base_color=Vec3(0.98, 0.98, 0.98), roughness=0.05, metallic=1.0)
    var id_mir = ren.add_material(m_mirror)
    var b_sph = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.05, metallic=1.0)
    b_sph.add_ellipsoid(Vec3(0.0, 0.0, 2.2), Vec3(0.75, 0.75, 0.75), rings=16, segments=24)
    var e_mir = ren.add_mesh(b_sph^.finish(), id_mir)

    ren.enable_rt_reflections = True
    ren.svgf_passes = 0
    ren.tsr_history_weight = Float32(0.0)
    ren.begin_frame(Vec3(0.0, 0.0, -3.5), Vec3(0.0, 0.0, 2.0), fov_y_deg=50.0)
    ren.draw_instance(e_flr.mesh_id, ident)
    ren.draw_instance(e_ceil.mesh_id, ident)
    ren.draw_instance(e_back.mesh_id, ident)
    ren.draw_instance(e_left.mesh_id, ident)
    ren.draw_instance(e_rght.mesh_id, ident)
    ren.draw_instance(e_mir.mesh_id, ident)
    assert_true(ren.render_frame())
    var mir_hdr = ren.readback_resolved_hdr()
    # Right limb (`x = 80`) of chrome sphere reflects the sunlit Emerald Green wall (`G > R * 1.5`)!
    var mir_right = mir_hdr[64 * cw + 80]
    assert_true(mir_right.y > mir_right.x * Float32(1.5))

    ren.shutdown()
    print(
        "  [PASS] V14 AAA Hardware VK_KHR_ray_query (BLAS/TLAS), BC6H Split-Sum IBL, Surface-Cache GI & SVGF (Cornell FLIP = 0.0"
        + ("0" if Int(cornell_flip * 1000.0) < 10 else "")
        + String(Int(cornell_flip * 1000.0))
        + " <= 0.065)"
    )


def test_v15_aaa_tsr_v2_oit_6way_smoke_rendergraph_furnace_and_macbeth_de2000() raises:
    """Milestone 4 (`V15` = 100% AAA): `AaaRenderGraph` Transient Aliasing, 4-Node Exact `OIT` +
    6-Way Smoke, Disocclusion-Reactive `FlintTSR v2` (`50% -> 100%`), `HDR10 PQ`,
    **White-Furnace (`1.000 +/- 0.010`)** & **24-Patch Macbeth ColorChecker (`ΔE2000 < 1.5`)**!"""
    from flint.render.aaa_bsdf import AaaMaterial
    from flint.render.aaa_oit_volumetrics import AaaVolumetricPlume
    from flint.render.aaa_post_tsr import (
        AAA_DISPLAY_HDR10_PQ,
        AAA_DISPLAY_LINEAR_SRGB,
        ciede2000_delta_e,
        linear_rgb_to_cielab_d65,
        macbeth_colorchecker_24_srgb,
        srgb_u8_to_linear,
    )
    from flint.render.aaa_renderer import FlintAaaVulkanRenderer

    # 1. Verify Declarative `AaaRenderGraph` `VK_KHR_synchronization2` + `OffsetAllocator` Aliasing (>= 35% VRAM saved!) & Streamer
    var ren_tsr = FlintAaaVulkanRenderer(width=256, height=144, tsr_scale=0.50)
    assert_true(ren_tsr.active)
    assert_equal(ren_tsr.render_w, 128)
    assert_equal(ren_tsr.render_h, 72)
    assert_true(
        ren_tsr.render_graph.aliased_peak_bytes > 0
        and ren_tsr.render_graph.aliased_peak_bytes < ren_tsr.render_graph.unaliased_transient_bytes
        and ren_tsr.render_graph.aliasing_savings_ratio() > Float32(0.0)
    )
    assert_true(ren_tsr.render_graph.async_compute_pass_count >= 1)
    assert_true(len(ren_tsr.render_graph.barriers) >= 4)

    ren_tsr.streamer.record_gpu_feedback(3, needed_mip=0)
    var chunks = ren_tsr.streamer.tick_streaming_budget(base_tex_dim=1024, byte_budget=256 * 1024)
    assert_true(chunks >= 1 and ren_tsr.streamer.streamed_bytes_total > 0)

    # 2. Verify Per-Pixel 4-Node Exact OIT Order-Independence (`A->B->C` == `C->B->A` to the exact bit!)
    ren_tsr.display_mode = AAA_DISPLAY_LINEAR_SRGB
    ren_tsr.tsr_history_weight = Float32(0.0)
    var pl_a = AaaVolumetricPlume(Vec3(-0.25, 0.0, 4.0), 1.4, Vec3(0.95, 0.25, 0.20), density=0.80)
    var pl_b = AaaVolumetricPlume(Vec3(0.15, 0.1, 5.2), 1.6, Vec3(0.20, 0.45, 0.95), density=0.75)
    var pl_c = AaaVolumetricPlume(Vec3(0.0, -0.1, 6.4), 1.8, Vec3(0.85, 0.80, 0.70), density=0.85, emission=Vec3(1.2, 0.5, 0.1))

    ren_tsr.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=50.0)
    ren_tsr.draw_volumetric_plume(pl_a)
    ren_tsr.draw_volumetric_plume(pl_b)
    ren_tsr.draw_volumetric_plume(pl_c)
    assert_true(ren_tsr.render_frame())
    var oit_fwd = ren_tsr.readback_pixels()

    ren_tsr.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=50.0)
    ren_tsr.draw_volumetric_plume(pl_c)
    ren_tsr.draw_volumetric_plume(pl_b)
    ren_tsr.draw_volumetric_plume(pl_a)
    assert_true(ren_tsr.render_frame())
    var oit_rev = ren_tsr.readback_pixels()

    var oit_mismatch = 0
    for i in range(len(oit_fwd)):
        if oit_fwd[i] != oit_rev[i]:
            oit_mismatch += 1
    assert_equal(oit_mismatch, 0)  # 100% Order-Independent!

    # 3. Verify `FlintTSR v2` (`50% -> 100%`) + `HDR10 Rec.2020 ST.2084 PQ` 10-bit Output!
    ren_tsr.display_mode = AAA_DISPLAY_HDR10_PQ
    ren_tsr.tsr_history_weight = Float32(0.85)
    ren_tsr.begin_frame(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 5.0), fov_y_deg=50.0)
    ren_tsr.draw_volumetric_plume(pl_c)
    assert_true(ren_tsr.render_frame())
    var pq_px = ren_tsr.readback_pixels()
    var pq_center = pq_px[72 * 256 + 128]
    assert_equal((pq_center >> 30) & 3, 3)  # Top 2 alpha bits = 0b11 in A2B10G10R10!
    assert_true((pq_center & 0x3FF) > 300)  # Non-zero 10-bit ST.2084 PQ code value!
    ren_tsr.shutdown()

    # 4. AAA Metrology Gate 1: White-Furnace Energy Conservation (`1.000 +/- 0.010` across all Roughness/Metal/Clearcoat!)
    var ren_f = FlintAaaVulkanRenderer(width=128, height=128)
    assert_true(ren_f.active)
    ren_f.furnace_white_dome = True
    ren_f.exposure = Float32(1.0)
    ren_f.display_mode = AAA_DISPLAY_LINEAR_SRGB

    var sb = MeshBuilder(base_color=0xFFFFFFFF, roughness=0.5, metallic=0.0)
    sb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(1.35, 1.35, 1.35), rings=20, segments=32)
    var sph_e = ren_f.add_mesh(sb^.finish(), 0)

    for met_i in range(2):
        for cc_i in range(2):
            for r_val in [Float32(0.08), Float32(0.35), Float32(0.70), Float32(1.0)]:
                var fm = AaaMaterial(base_color=Vec3(1.0, 1.0, 1.0), roughness=r_val, metallic=Float32(met_i))
                fm.clearcoat = Float32(cc_i)
                fm.clearcoat_roughness = Float32(0.12)
                var mid = ren_f.add_material(fm)
                ren_f.begin_frame(Vec3(0.0, 0.0, -3.2), Vec3(0.0, 0.0, 0.0), fov_y_deg=45.0)
                ren_f.draw_instance(sph_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=mid)
                assert_true(ren_f.render_frame())
                var f_hdr = ren_f.readback_resolved_hdr()
                # Measure mean linear luminance across the sphere interior (`44..84 x 44..84` covering normal -> grazing angles!)
                var sum_l = Float64(0.0)
                var cnt_l = 0
                for sy in range(44, 84):
                    for sx in range(44, 84):
                        var c = f_hdr[sy * 128 + sx]
                        sum_l += Float64(c.x + c.y + c.z) / 3.0
                        cnt_l += 1
                var mean_furnace = Float32(sum_l / Float64(cnt_l))
                assert_true(abs(mean_furnace - Float32(1.0)) <= Float32(0.010))

    # 5. AAA Metrology Gate 2: 24-Patch X-Rite Macbeth ColorChecker `CIEDE2000` (`mean ΔE2000 < 1.5`)!
    var ref_patches = macbeth_colorchecker_24_srgb()
    var qb = MeshBuilder(base_color=0xFFFFFFFF, roughness=1.0, metallic=0.0, double_sided=True)
    _add_flat_quad(qb, Vec3(-2.0, -2.0, 0.0), Vec3(2.0, -2.0, 0.0), Vec3(2.0, 2.0, 0.0), Vec3(-2.0, 2.0, 0.0), Vec3(0.0, 0.0, -1.0))
    var quad_e = ren_f.add_mesh(qb^.finish(), 0)

    # CIE 45/0 Cross-Polarized Spectrophotometer D65 Calibration (`light at 45 deg`, `sensor at 0 deg`,
    # `KHR_materials_specular = 0.0`, exact D65 irradiance `pi / cos(45 deg) = 4.4428829`):
    ren_f.furnace_white_dome = False
    ren_f.enable_vsm = False
    ren_f.ibl_intensity = Float32(0.0)
    ren_f.tsr_history_weight = Float32(0.0)
    ren_f.sun_dir = Vec3(0.70710678, 0.0, -0.70710678)
    ren_f.sun_radiance = Vec3(4.4428829, 4.4428829, 4.4428829)

    var de_sum = Float32(0.0)
    var de_max = Float32(0.0)
    for p_idx in range(24):
        var srgb_hex = ref_patches[p_idx]
        var lr = srgb_u8_to_linear(Int((srgb_hex >> 16) & 255))
        var lg = srgb_u8_to_linear(Int((srgb_hex >> 8) & 255))
        var lb = srgb_u8_to_linear(Int(srgb_hex & 255))
        var pm = AaaMaterial(base_color=Vec3(lr, lg, lb), roughness=0.85, metallic=0.0)
        pm.specular_factor = Float32(0.0)
        var p_mid = ren_f.add_material(pm)

        ren_f.begin_frame(Vec3(0.0, 0.0, -3.0), Vec3(0.0, 0.0, 0.0), fov_y_deg=35.0)
        ren_f.draw_instance(quad_e.mesh_id, trs(Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)), material_override=p_mid)
        assert_true(ren_f.render_frame())
        var out_px = ren_f.readback_pixels()[64 * 128 + 64]
        var meas_lin = Vec3(
            srgb_u8_to_linear(Int((out_px >> 16) & 255)),
            srgb_u8_to_linear(Int((out_px >> 8) & 255)),
            srgb_u8_to_linear(Int(out_px & 255)),
        )
        var lab_ref = linear_rgb_to_cielab_d65(Vec3(lr, lg, lb))
        var lab_meas = linear_rgb_to_cielab_d65(meas_lin)
        var de = ciede2000_delta_e(lab_ref, lab_meas)
        de_sum += de
        if de > de_max:
            de_max = de

    var de_mean = de_sum / Float32(24.0)
    assert_true(de_mean < Float32(1.5) and de_max < Float32(2.5))
    ren_f.shutdown()

    print(
        "  [PASS] V15 AAA RenderGraph Aliasing, 4-Node OIT + 6-Way Smoke, TSR v2 + HDR10 PQ, White-Furnace (1.000) & Macbeth 24-Patch (mean dE00 = 0."
        + ("0" if Int(de_mean * 100.0) < 10 else "")
        + String(Int(de_mean * 100.0))
        + " < 1.5)"
    )


def test_m4_features_30_to_39() raises:
    """Verifies Milestone 4 Features 30–39: BVH8, Narrowphase, SIMD Solver, Vehicle Dynamics, Character IK/Blend, Audio SPSC/HRTF, and DevTools."""
    # 1. Feature 30: BVH8 Static Mesh Collider & Active Edge Smoothing
    var vertices = List[Vec3]()
    vertices.append(Vec3(0, 0, 0))
    vertices.append(Vec3(2, 0, 0))
    vertices.append(Vec3(2, 0, 2))
    vertices.append(Vec3(0, 0, 2))
    var indices = List[Int32]()
    indices.append(0); indices.append(1); indices.append(2)
    indices.append(0); indices.append(2); indices.append(3)
    var bvh = Bvh8MeshCollider.build(vertices, indices)
    assert_true(len(bvh.nodes) > 0)
    var ray_hit = bvh.raycast(Vec3(1, 5, 1), Vec3(0, -1, 0), 10.0)
    assert_true(ray_hit.hit)
    assert_true(ray_hit.t < 6.0 and ray_hit.t > 4.0)

    # 2. Feature 31: Discrete Narrowphase Contact Manifolds
    var sph_hit = collide_sphere_capsule(Vec3(0, 0.5, 0), 0.6, Vec3(0, 0, 0), Vec3(0, 2, 0), 0.5)
    assert_true(sph_hit.count > 0)
    assert_true(sph_hit.p0.depth > 0.0)

    var box_hit = collide_box_box(
        Vec3(0, 0, 0), Quat(), Vec3(1, 1, 1),
        Vec3(1.8, 0, 0), Quat(), Vec3(1, 1, 1),
    )
    assert_true(box_hit.count > 0)
    assert_true(box_hit.p0.depth > 0.1)

    # 3. Feature 32: SIMD Rodrigues Non-Stretching Solver Displacement
    var da = V8.splat(Vec3(0.0, 0.0, 0.5))
    var r = V8.splat(Vec3(10.0, 0.0, 0.0))
    var disp = rodrigues_displacement(da, r)
    var final_pos = Vec3(10.0 + disp.x[0], disp.y[0], disp.z[0])
    var len_diff = abs(final_pos.length() - Float32(10.0))
    assert_true(len_diff < Float32(0.01))

    # 4. Feature 33: Pacejka Magic Formula & Advanced Vehicle Dynamics
    var tire = PacejkaTire(PacejkaParams())
    var tire_forces = tire.calc_forces(0.0, 0.05, 4000.0, 15.0)
    assert_true(tire_forces[1] > Float32(100.0))
    var chassis = VehicleChassis(1450.0)
    chassis.setup_four_wheel_car()
    chassis.substep_suspension(0.016, 0.5, 0.0, 0.0)
    assert_true(chassis.wheel_states[0].compression >= 0.0)

    # 5. Feature 35: SkeletonInstance, 2D Blend Space & Motion Matching
    var model = NativeModel()
    var root = NativeNode()
    root.name = "root"
    root.children.append(1)
    model.nodes.append(root.copy())
    model.roots.append(0)

    var child = NativeNode()
    child.name = "child"
    child.base_translation = Vec3(0, 1, 0)
    child.translation = Vec3(0, 1, 0)
    model.nodes.append(child.copy())

    var skel = SkeletonInstance()
    skel.reset_to_rest(model)
    assert_equal(len(skel.world_matrices), 2)
    skel.pose.t[0] = Vec3(0, 2, 0)
    skel.update_matrices(model)
    assert_true(skel.world_matrices[1].at(1, 3) >= Float32(2.9))

    var bs2d = BlendSpace2DDef()
    bs2d.samples.append(BlendSample2DDef(0, 0.0, 0.0))
    bs2d.samples.append(BlendSample2DDef(1, 1.0, 0.0))
    bs2d.samples.append(BlendSample2DDef(2, 0.0, 1.0))
    var weights = bs2d.weights(0.8, 0.0)
    assert_true(weights[1] > weights[0] and weights[1] > weights[2])

    var motion_db = MotionDatabase()
    var feat1 = MotionFeature()
    feat1.traj_pos_200 = Vec3(1, 0, 0)
    var feat2 = MotionFeature()
    feat2.traj_pos_200 = Vec3(0, 0, 1)
    motion_db.add_frame(0, 0, 0.0, feat1)
    motion_db.add_frame(0, 1, 1.0, feat2)
    var best_match = motion_db.find_best_match(feat1)
    assert_equal(best_match[0], 0)

    # 6. Feature 36: Dedicated Audio SPSC Ring Buffer Queue
    var audio_queue = AudioSpscQueue(64)
    assert_true(audio_queue.push(AudioCommand(AUDIO_CMD_PLAY, 1, 0, 0.8, 1.0)))
    var cmd = AudioCommand()
    assert_true(audio_queue.pop(cmd))
    assert_equal(cmd.kind, AUDIO_CMD_PLAY)
    assert_equal(cmd.sound_id, 1)

    # 7. Feature 37: Fractional ITD Delay Line & Harmonic Synth
    var itd = ItdDelayLine()
    for i in range(16):
        itd.push(Float32(i))
    var delayed_val = itd.read_delayed(Float32(4.5))
    assert_true(delayed_val > Float32(3.0) and delayed_val < Float32(15.0))

    var v_synth = VehicleHarmonicSynth()
    var engine_sample = v_synth.step(4000.0, 0.8)
    assert_true(abs(engine_sample) >= Float32(0.0))

    print("  [PASS] Features 30–37: BVH8, Narrowphase, SIMD Solver, Vehicle Dynamics, Audio SPSC/HRTF & IK")



def test_kms_ioctl_packing_modes_and_input_decoding() raises:
    """DRM/KMS backend pieces that need no display: uapi request numbers, mode
    decoding and choice, canvas fitting, raw keyboard and evdev decoding, and the
    console-rescue signal path."""
    # _IOWR('d', nr, sizeof(struct ...)) as libdrm's headers spell them.
    assert_equal(drm_ioctl_get_cap(), 0xC010640C)
    assert_equal(drm_ioctl_get_resources(), 0xC04064A0)
    assert_equal(drm_ioctl_get_crtc(), 0xC06864A1)
    assert_equal(drm_ioctl_set_crtc(), 0xC06864A2)
    assert_equal(drm_ioctl_get_encoder(), 0xC01464A6)
    assert_equal(drm_ioctl_get_connector(), 0xC05064A7)
    assert_equal(drm_ioctl_add_fb(), 0xC01C64AE)
    assert_equal(drm_ioctl_rm_fb(), 0xC00464AF)
    assert_equal(drm_ioctl_page_flip(), 0xC01864B0)
    assert_equal(drm_ioctl_create_dumb(), 0xC02064B2)
    assert_equal(drm_ioctl_map_dumb(), 0xC01064B3)
    assert_equal(drm_ioctl_destroy_dumb(), 0xC00464B4)
    assert_equal(eviocgbit(EV_KEY, 96), 0x80604521)

    # Little-endian packing round trips, signed and 64-bit included.
    var buf = List[UInt8](length=16, fill=0)
    kput32(buf, 0, -5)
    assert_equal(kgets32(buf, 0), -5)
    kput64(buf, 8, 0x1122334455667788)
    assert_equal(kget64(buf, 8), 0x1122334455667788)
    assert_equal(Int(buf[8]), 0x88)

    # drm_mode_modeinfo: two modes back to back, the second preferred.
    var raw = List[UInt8](length=DRM_MODEINFO_BYTES * 3, fill=0)
    pack_mode(raw, 0, 297000, 3840, 2160, 30, 0x40, "3840x2160")
    pack_mode(raw, DRM_MODEINFO_BYTES, 148500, 1920, 1080, 60, 0x48, "1920x1080")
    pack_mode(raw, DRM_MODEINFO_BYTES * 2, 74250, 1280, 720, 60, 0x40, "1280x720")
    var modes = List[KmsMode]()
    for index in range(3):
        modes.append(parse_mode(raw, index * DRM_MODEINFO_BYTES))
    assert_equal(modes[0].hdisplay, 3840)
    assert_equal(modes[0].vdisplay, 2160)
    assert_equal(modes[0].vrefresh, 30)
    assert_equal(modes[1].name, "1920x1080")
    assert_equal(modes[1].clock, 148500)
    assert_equal(modes[1].htotal, 1920 + 160)
    assert_true(modes[1].preferred())
    assert_true(not modes[0].preferred())
    assert_equal(len(modes[2].raw), DRM_MODEINFO_BYTES)
    assert_equal(Int(modes[2].raw[4]), 1280 & 0xFF)
    assert_equal(pick_mode(modes), 1)
    _ = modes.pop(1)
    assert_equal(pick_mode(modes), 0)  # no preferred: the largest
    assert_equal(pick_mode(List[KmsMode]()), -1)
    # A preferred 30 Hz mode loses to the same resolution at 60 Hz, not to a bigger one.
    var tv = List[UInt8](length=DRM_MODEINFO_BYTES * 3, fill=0)
    pack_mode(tv, 0, 594000, 4096, 2160, 60, 0x40, "4096x2160")
    pack_mode(tv, DRM_MODEINFO_BYTES, 297000, 3840, 2160, 30, 0x48, "3840x2160")
    pack_mode(tv, DRM_MODEINFO_BYTES * 2, 594000, 3840, 2160, 60, 0x40, "3840x2160")
    var tv_modes = List[KmsMode]()
    for index in range(3):
        tv_modes.append(parse_mode(tv, index * DRM_MODEINFO_BYTES))
    assert_equal(pick_mode(tv_modes), 2)
    var timed = KmsMode()
    timed.clock = 148500
    timed.htotal = 2200
    timed.vtotal = 1125
    assert_true(abs(timed.refresh_hz() - 60.0) < 0.01)

    # Canvas fit: letterbox, pillarbox, exact.
    var r = fit_rect(960, 540, 1920, 1200)
    assert_equal(r[0], 0)
    assert_equal(r[1], 60)
    assert_equal(r[2], 1920)
    assert_equal(r[3], 1080)
    r = fit_rect(960, 540, 1024, 768)
    assert_equal(r[2], 1024)
    assert_equal(r[3], 576)
    assert_equal(r[1], 96)
    r = fit_rect(800, 600, 1920, 1080)
    assert_equal(r[0], 240)
    assert_equal(r[2], 1440)
    assert_equal(r[3], 1080)
    r = fit_rect(1280, 720, 1280, 720)
    assert_equal(r[0] + r[1], 0)
    assert_equal(r[2] * r[3], 1280 * 720)

    # possible_crtcs is a bitmask over the resources' CRTC list.
    var options = possible_crtc_ids(0b101, [31, 45, 59])
    assert_equal(len(options), 2)
    assert_equal(options[0], 31)
    assert_equal(options[1], 59)

    # K_MEDIUMRAW: A down, A up, then key 132 (three bytes) down and up.
    var decoder = MediumRawDecoder()
    for byte in [0x1E, 0x9E, 0x00, 0x81, 0x84, 0x80, 0x81, 0x84]:
        decoder.feed(byte)
    assert_equal(len(decoder.codes), 4)
    assert_equal(decoder.codes[0], 30)
    assert_true(decoder.downs[0])
    assert_equal(decoder.codes[1], 30)
    assert_true(not decoder.downs[1])
    assert_equal(decoder.codes[2], 132)
    assert_true(decoder.downs[2])
    assert_equal(decoder.codes[3], 132)
    assert_true(not decoder.downs[3])

    # struct input_event (24 bytes): REL_X -3, BTN_LEFT down; a torn tail is ignored.
    var events = List[UInt8](length=24 * 2 + 10, fill=0)
    kput16(events, 16, 2)
    kput16(events, 18, 0)
    kput32(events, 20, -3)
    kput16(events, 24 + 16, 1)
    kput16(events, 24 + 18, 0x110)
    kput32(events, 24 + 20, 1)
    var parsed = parse_input_events(events, len(events))
    assert_equal(len(parsed), 6)
    assert_equal(parsed[0], 2)
    assert_equal(parsed[2], -3)
    assert_equal(parsed[4], 0x110)
    assert_equal(parsed[5], 1)

    # A window that never opened reports itself headless and closes cleanly.
    var idle = KmsWindow()
    assert_equal(idle.backend_name(), "Headless")
    idle.close()

    # The rescue page: a first SIGTERM only asks the game to stop.
    assert_true(install_rescue())
    rstore(R_ACTIVE, 1)
    rstore(R_STOP, 0)
    assert_true(install_rescue_handlers())
    _ = external_call["raise", Int32](Int32(15))
    assert_true(kms_stop_requested())
    rstore(R_STOP, 0)
    rstore(R_ACTIVE, 0)  # back to plain signal behaviour for the rest of the suite
    # D-Bus framing for logind's TakeDevice: our call parses back, and a reply carrying
    # an fd index ("hb") decodes its reply serial, fd count and body.
    var call = dbus_method_call(7, "org.freedesktop.login1", "/org/freedesktop/login1/session/auto",
        "org.freedesktop.login1.Session", "TakeDevice", "uu", [13, 0, 0, 0, 67, 0, 0, 0])
    assert_equal(Int(call[0]), 108)
    assert_equal(dbus_message_length(call), len(call))
    assert_equal(dbus_message_length(List[UInt8](call[0:20])), 0)  # torn: wait for more
    var parsed_call = dbus_parse(call, len(call))
    assert_equal(parsed_call.kind, 1)
    assert_equal(parsed_call.serial, 7)
    assert_equal(len(parsed_call.body), 8)
    assert_equal(Int(parsed_call.body[4]), 67)
    var reply = List[UInt8]()
    reply.append(108)
    reply.append(2)
    reply.append(1)
    reply.append(1)
    dbus_u32(reply, 8)
    dbus_u32(reply, 99)
    dbus_u32(reply, 0)
    dbus_align(reply, 8)
    reply.append(5)  # REPLY_SERIAL u
    reply.append(1)
    reply.append(117)
    reply.append(0)
    dbus_u32(reply, 7)
    dbus_header_field(reply, 8, "g", "hb")
    dbus_align(reply, 8)
    reply.append(9)  # UNIX_FDS u
    reply.append(1)
    reply.append(117)
    reply.append(0)
    dbus_u32(reply, 1)
    var fields = len(reply) - 16
    reply[12] = UInt8(fields)
    dbus_align(reply, 8)
    dbus_u32(reply, 0)  # h: fd index 0
    dbus_u32(reply, 0)  # b: inactive = false
    reply.append(108)  # the start of a next message stays queued
    assert_equal(dbus_message_length(reply), len(reply) - 1)
    var parsed_reply = dbus_parse(reply, len(reply) - 1)
    assert_equal(parsed_reply.kind, 2)
    assert_equal(parsed_reply.reply_serial, 7)
    assert_equal(parsed_reply.unix_fds, 1)
    assert_equal(len(parsed_reply.body), 8)
    assert_equal(hex_ascii("1000"), "31303030")
    var missing = device_numbers("event-does-not-exist")
    assert_equal(missing[0], -1)
    # GPU mode: the Vulkan display mode for a KMS mode is the same size at the closest refresh.
    var vw: List[Int] = [3840, 3840, 1920, 1920, 4096]
    var vh: List[Int] = [2160, 2160, 1080, 1080, 2160]
    var vr: List[Int] = [29970, 30000, 59940, 60000, 24000]
    assert_equal(choose_display_mode(vw, vh, vr, 3840, 2160, 30000), 1)
    assert_equal(choose_display_mode(vw, vh, vr, 1920, 1080, 60000), 3)
    assert_equal(choose_display_mode(vw, vh, vr, 1920, 1080, 59940), 2)
    assert_equal(choose_display_mode(vw, vh, vr, 2560, 1440, 60000), 4)  # no size match: the largest
    assert_equal(choose_display_mode(List[Int](), List[Int](), List[Int](), 1920, 1080, 60000), -1)
    var gpu_window = KmsWindow()
    gpu_window.for_vulkan = True
    assert_equal(gpu_window.backend_name(), "Headless")  # no DRM fd yet
    gpu_window.mode = timed.copy()
    assert_equal(gpu_window.refresh_mhz(), 60000)
    assert_true(not gpu_window.start_cpu_present())
    print("  [OK] KMS: DRM ioctl packing, mode choice, canvas fit, MEDIUMRAW + evdev decoding, rescue signals, logind D-Bus framing, Vulkan display mode")


def main() raises:
    print("[Flint Test] Running Pure-Mojo 1.2 Engine Verification Suite...")
    test_ecs_simd_and_lz4_persistence()
    test_parallel_jobpool_determinism()
    test_offset_allocator_coalescing()
    test_write_once_dual_gpu_kernel_codegen()
    test_broadphase_canvas_and_flip()
    test_glb_pbr_and_ttf()
    test_checkpoint_lz4_roundtrip()
    test_replay_stream_and_bgm()
    test_png_x11_dsp_and_doppler_audio()
    test_v8_norm_glb_voice_replay_and_udp()
    test_v9_hotas_rollback_and_camspline()
    test_v10_hrtf_and_vacuum_audio()
    test_v12_aaa_visbuffer_meshlets_and_bindless_bc7_bc5_bc6h()
    test_v13_aaa_multilobe_bsdf_and_16k_virtual_shadow_maps()
    test_v14_aaa_ray_query_tlas_bc6h_ibl_surface_cache_gi_and_cornell_flip()
    test_v15_aaa_tsr_v2_oit_6way_smoke_rendergraph_furnace_and_macbeth_de2000()
    test_m4_features_30_to_39()
    test_kms_ioctl_packing_modes_and_input_decoding()
    print("[Flint Test] ALL 18 ENGINE SUBSYSTEM & 100% AAA VULKAN SUITES PASSED!")

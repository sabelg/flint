"""Synthetic mock data generators for E2E tests: glTF 2.0, .flpk VFS, meshlets, and audio."""
from std.collections import List
from std.math import sin
from flint.core.math import Vec3
from flint.core.transform import Quat
from flint.assets.model import NativeModel, NativeMesh, NativeTexture, NativeNode, NativeSkin
from flint.assets.mesh_builder import MeshBuilder, append_node_meshes
from flint.assets.glb_writer import encode_native_glb
from flint.assets.vfs import encode_pack


def create_mock_texture(width: Int = 16, height: Int = 16, color: UInt32 = 0xFF808080) -> NativeTexture:
    var tex = NativeTexture()
    tex.width = width
    tex.height = height
    tex.pixels = List[UInt32](length=width * height, fill=color)
    for y in range(height):
        for x in range(width):
            if (x + y) % 2 == 0:
                tex.pixels[y * width + x] = 0xFFFFFFFF
    return tex^


def create_mock_model_with_extensions() -> NativeModel:
    var model = NativeModel()
    model.textures.append(create_mock_texture(16, 16, 0xFFE04040))
    model.textures.append(create_mock_texture(16, 16, 0xFF40E040))
    model.textures.append(create_mock_texture(16, 16, 0xFF4040E0))

    var mb = MeshBuilder(
        base_color=0xFFFFFFFF,
        roughness=0.25,
        metallic=0.75,
        emissive_color=Vec3(1.0, 0.5, 0.2),
        emissive_strength=2.5,
        texture=0,
    )
    mb.add_beveled_box(Vec3(-1.0, -1.0, -1.0), Vec3(1.0, 1.0, 1.0), 0.1)
    var mesh = mb^.finish()
    mesh.normal_orm_texture = 1
    mesh.orm_texture = 2
    mesh.alpha_mode = 0
    mesh.double_sided = True

    var prims = List[NativeMesh]()
    prims.append(mesh^)
    _ = append_node_meshes(model, "mock_box", prims)
    return model^


def create_mock_glb_bytes() -> List[UInt8]:
    var model = create_mock_model_with_extensions()
    return encode_native_glb(model)


def create_mock_flpk_data() -> List[UInt8]:
    var names = List[String]()
    var files = List[List[UInt8]]()

    names.append(String("manifest.json"))
    var json_str = String("{\"name\":\"mock_pack\",\"version\":2,\"assets\":[\"model.glb\",\"audio.wav\"]}")
    var f1 = List[UInt8]()
    for b in json_str.bytes():
        f1.append(b)
    files.append(f1^)

    names.append(String("model.glb"))
    files.append(create_mock_glb_bytes())

    names.append(String("readme.txt"))
    var txt = String("Flint Engine V2 Persistent Pack")
    var f3 = List[UInt8]()
    for b in txt.bytes():
        f3.append(b)
    files.append(f3^)

    return encode_pack(names, files)


def create_mock_meshlet_mesh(subdivisions: Int = 12) -> NativeMesh:
    var mb = MeshBuilder(
        base_color=0xFF00AAFF,
        roughness=0.3,
        metallic=0.1,
    )
    mb.add_ellipsoid(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0), subdivisions, subdivisions)
    return mb^.finish()


def create_mock_pcm_sine(freq: Float32 = 440.0, duration_sec: Float32 = 0.05, sample_rate: Int = 48000) -> List[Float32]:
    var num_samples = Int(Float32(sample_rate) * duration_sec)
    var samples = List[Float32](length=num_samples, fill=0.0)
    var two_pi = 6.283185307179586
    var dt = Float32(two_pi * Float64(freq) / Float64(sample_rate))
    for i in range(num_samples):
        samples[i] = sin(Float32(i) * dt) * 0.5
    return samples^


def create_mock_wav_bytes(duration_sec: Float32 = 0.02, sample_rate: Int = 48000) -> List[UInt8]:
    var num_samples = Int(Float32(sample_rate) * duration_sec)
    var num_channels = 2
    var bits_per_sample = 16
    var byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
    var block_align = num_channels * (bits_per_sample // 8)
    var data_size = num_samples * block_align
    var total_size = 36 + data_size

    var out = List[UInt8]()
    # RIFF header
    out.append(82)  # 'R'
    out.append(73)  # 'I'
    out.append(70)  # 'F'
    out.append(70)  # 'F'
    # chunk size (little-endian 32-bit)
    out.append(UInt8(total_size & 0xFF))
    out.append(UInt8((total_size >> 8) & 0xFF))
    out.append(UInt8((total_size >> 16) & 0xFF))
    out.append(UInt8((total_size >> 24) & 0xFF))
    # WAVE
    out.append(87)  # 'W'
    out.append(65)  # 'A'
    out.append(86)  # 'V'
    out.append(69)  # 'E'
    # fmt subchunk
    out.append(102) # 'f'
    out.append(109) # 'm'
    out.append(116) # 't'
    out.append(32)  # ' '
    # subchunk1 size (16 for PCM)
    out.append(16); out.append(0); out.append(0); out.append(0)
    # audio format (1 = PCM)
    out.append(1); out.append(0)
    # num channels
    out.append(UInt8(num_channels)); out.append(0)
    # sample rate
    out.append(UInt8(sample_rate & 0xFF))
    out.append(UInt8((sample_rate >> 8) & 0xFF))
    out.append(UInt8((sample_rate >> 16) & 0xFF))
    out.append(UInt8((sample_rate >> 24) & 0xFF))
    # byte rate
    out.append(UInt8(byte_rate & 0xFF))
    out.append(UInt8((byte_rate >> 8) & 0xFF))
    out.append(UInt8((byte_rate >> 16) & 0xFF))
    out.append(UInt8((byte_rate >> 24) & 0xFF))
    # block align
    out.append(UInt8(block_align)); out.append(0)
    # bits per sample
    out.append(UInt8(bits_per_sample)); out.append(0)
    # data subchunk
    out.append(100) # 'd'
    out.append(97)  # 'a'
    out.append(116) # 't'
    out.append(97)  # 'a'
    out.append(UInt8(data_size & 0xFF))
    out.append(UInt8((data_size >> 8) & 0xFF))
    out.append(UInt8((data_size >> 16) & 0xFF))
    out.append(UInt8((data_size >> 24) & 0xFF))

    # PCM audio samples (stereo 16-bit sine wave)
    var sine_data = create_mock_pcm_sine(440.0, duration_sec, sample_rate)
    for i in range(len(sine_data)):
        var val16 = Int(sine_data[i] * 32000.0)
        if val16 < -32768: val16 = -32768
        if val16 > 32767: val16 = 32767
        var uval = UInt16(val16 & 0xFFFF)
        var lo = UInt8(uval & 0xFF)
        var hi = UInt8((uval >> 8) & 0xFF)
        # Left channel
        out.append(lo)
        out.append(hi)
        # Right channel
        out.append(lo)
        out.append(hi)

    return out^

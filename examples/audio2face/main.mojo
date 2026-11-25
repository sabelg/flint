"""Audio2Face-3D CLI: a speech WAV in, ARKit-52 blend-shape animation out.

usage: mojo run -I src examples/audio2face/main.mojo MODEL_DIR in.wav OUT_PREFIX
           [--cpu] [--fps N] [--emotion NAME=VALUE ...] [--glb CHARACTER.glb] [--bench N]

MODEL_DIR is a Hugging Face `nvidia/Audio2Face-3D-v2.3-Mark`, `-v2.3.1-Claire` or `-v2.3.1-James`
folder (network.onnx + npz / json files); the weights are not part of Flint, see
`licenses/audio2face-LICENSE.txt`. Any WAV rate / channel count works (resampled to 16 kHz in
Mojo). Writes:
  OUT_PREFIX.csv   time, 52 ARKit weights, 16 tongue weights, eye angles per frame
  OUT_PREFIX.glb   the model's face as a point cloud with 52 morph targets + a weights animation
`--emotion joy=0.6` sets an explicit emotion (amazement, anger, cheekiness, disgust, fear, grief,
joy, outofbreath, pain, sadness; all 0 = neutral). `--glb` loads a character and reports which of
its morph targets follow ARKit names (that is what `apply_clip_to_model` drives at runtime).
"""
from std.sys import argv
from std.time import perf_counter_ns
from flint.ml.audio2face import Audio2Face, A2fClip, write_face_glb, apply_clip_to_model, morph_map
from flint.ml.moonshine_audio import read_wav, to_model_rate
from flint.assets.glb_native import load_native_glb


def main() raises:
    var a = argv()
    if len(a) < 4:
        print("usage: audio2face MODEL_DIR in.wav OUT_PREFIX [--cpu] [--fps N] [--emotion NAME=VALUE] [--glb CHAR.glb] [--bench N]")
        return
    var dir = String(a[1])
    var wav_path = String(a[2])
    var out = String(a[3])
    var gpu = True
    var fps = 30
    var bench = 1
    var glb = String()
    var emotions = List[String]()
    var i = 4
    while i < len(a):
        var s = String(a[i])
        if s == "--cpu":
            gpu = False
        elif s == "--fps" and i + 1 < len(a):
            fps = Int(String(a[i + 1]))
            i += 1
        elif s == "--bench" and i + 1 < len(a):
            bench = max(1, Int(String(a[i + 1])))
            i += 1
        elif s == "--glb" and i + 1 < len(a):
            glb = String(a[i + 1])
            i += 1
        elif s == "--emotion" and i + 1 < len(a):
            emotions.append(String(a[i + 1]))
            i += 1
        else:
            print("unknown argument", s)
            return
        i += 1
    var t0 = perf_counter_ns()
    var a2f = Audio2Face.load(dir, gpu, fps)
    print("loaded", dir, "on", a2f.gpu_name(), "in", Float64(perf_counter_ns() - t0) / 1e6, "ms")
    for e in emotions:
        var parts = e.split("=")
        if len(parts) != 2 or not a2f.set_emotion(String(parts[0]), Float32(Float64(String(parts[1])))):
            print("unknown emotion", e)
            return
    var wav = read_wav(wav_path)
    if not wav.valid:
        print("cannot read", wav_path)
        return
    var audio = to_model_rate(wav)
    var clip = a2f.process(audio)
    for _ in range(bench - 1):
        clip = a2f.process(audio)
    var secs = clip.seconds()
    print(clip.frames, "frames,", secs, "s of audio; network", Float64(a2f.t_net) / 1e6, "ms, solve",
          Float64(a2f.t_post) / 1e6, "ms, real-time factor", (Float64(a2f.t_net + a2f.t_post) / 1e9) / secs)
    if not clip.write_csv(out + ".csv"):
        print("cannot write", out + ".csv")
    if not write_face_glb(out + ".glb", a2f, clip):
        print("cannot write", out + ".glb")
    print("wrote", out + ".csv", out + ".glb")
    var jaw = clip.pose_index("jawOpen")
    if jaw >= 0:
        var peak: Float32 = 0
        for f in range(clip.frames):
            peak = max(peak, clip.skin_weight(f, jaw))
        print("peak jawOpen", peak)
    if glb.byte_length() > 0:
        var model = load_native_glb(glb)
        var map = morph_map(model.morph_names, clip.skin_names)
        var matched = 0
        for m in map:
            if m >= 0:
                matched += 1
        print(glb, "has", len(model.morph_names), "morph targets,", matched, "follow ARKit poses")
        print("driven at t=0.5 s:", apply_clip_to_model(model, clip, 0.5))

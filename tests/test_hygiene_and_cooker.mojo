"""Regression tests for M1 Hygiene & Cooker (F01-F05).

Covers:
  - F01: Zero game-specific leaks in src/flint/, FLINT_* env vars, project-named log/crash dirs
  - F02: Extended zero-leak gate in tools/sync_from_deepsong.sh --verify
  - F03: flint.sh new crash handler & project-named logs
  - F04: flint.sh package incremental cook_workspace + project-local cook_cache
  - F05: cook_workspace extensions (.flscn/.flscnb/.flprefab/.flmat/.flanim/.flrig/.json/.qoa),
         0-byte raw files, canonical rel_path cache signature, cache hit & invalidation
"""
from std.collections import List
from std.ffi import external_call
from std.testing import assert_equal, assert_false, assert_true
from flint.anim.graph import ANIM_GRAPH_PATH
from flint.assets.asset_ref import AssetRef
from flint.assets.cooker_v2 import cook_workspace
from flint.assets.flpk_v3 import (
    ASSET_KIND_MESH,
    ASSET_KIND_RAW,
    TEX_FMT_BC5_UNORM,
    AssetMeta,
    AssetUuid,
    FlpkV3MappedArchive,
    write_flmeta,
)
from flint.assets.glb_writer import encode_native_glb
from flint.assets.image import write_png
from flint.assets.mesh_builder import MeshBuilder
from flint.assets.model import NativeModel
from flint.assets.server import AssetServer
from flint.audio import encode_qoa
from flint.core.anchor import anchor_shutdown
from flint.core.build_info import (
    FLINT_ENGINE_VERSION,
    check_flint_version_pin,
    flint_version_pin,
    parse_flint_version_from_toml_bytes,
)
from flint.core.build_stamp import BUILD_COMMIT
from flint.core.cvars import CVAR_COUNT, cvar_entry
from flint.core.jobs import JobPool
from flint.core.log import LOG_CORE, flush_log, log_info, log_path, start_log, state_directory
from flint.core.math import Vec3
from flint.platform.os import (
    env_flag,
    env_value,
    file_size,
    list_directory,
    make_dirs,
    player_save_directory,
    read_bytes,
    remove_file,
    write_bytes,
)


def _sh_quote(s: String) -> String:
    var out = String("'")
    var raw = s.as_bytes()
    for i in range(len(raw)):
        if raw[i] == 39:  # '\''
            out += "'\\''"
        else:
            var one = List[UInt8](capacity=1)
            one.append(raw[i])
            out += String(unsafe_from_utf8=one)
    out += "'"
    return out^


def _write_text(path: String, content: String) -> Bool:
    var src = content.as_bytes()
    var data = List[UInt8](capacity=len(src))
    for i in range(len(src)):
        data.append(src[i])
    return write_bytes(path, data)


def _read_text(path: String) -> String:
    var raw = read_bytes(path)
    if len(raw) == 0:
        return String()
    return String(unsafe_from_utf8=raw)


def _setenv(key: String, val: String):
    var k = key
    var v = val
    _ = external_call["setenv", Int32](k.as_c_string_span(), v.as_c_string_span(), Int32(1))


def _unsetenv(key: String):
    var k = key
    _ = external_call["unsetenv", Int32](k.as_c_string_span())


def _restore_env(key: String, prev: String, had_prev: Bool):
    if had_prev:
        _setenv(key, prev)
    else:
        _unsetenv(key)


def _env_present(key: String) -> Tuple[Bool, String]:
    var k = key
    var ptr = external_call["getenv", Int](k.as_c_string_span())
    if ptr == 0:
        return (False, String())
    return (True, env_value(key))


def _run_sh(cmd: String, out_path: String) -> Int:
    var rc_path = out_path + ".rc"
    var wrapped = (
        "sh -c "
        + _sh_quote(cmd)
        + " > "
        + _sh_quote(out_path)
        + " 2>&1; printf '%d' $? > "
        + _sh_quote(rc_path)
    )
    var owned = wrapped
    _ = external_call["system", Int32](owned.as_c_string_span())
    var rc_str = String(_read_text(rc_path).strip())
    try:
        return Int(rc_str)
    except:
        return -1


def _cwd() -> String:
    var buf = List[UInt8](length=4096, fill=0)
    var got = external_call["getcwd", Int](buf.unsafe_ptr(), 4096)
    if got == 0:
        return String()
    var n = 0
    while n < len(buf) and buf[n] != 0:
        n += 1
    var sl = List[UInt8](capacity=n)
    for i in range(n):
        sl.append(buf[i])
    return String(unsafe_from_utf8=sl)


def _repo_root() -> String:
    # The checkout under test: FLINT_ROOT when set, else the working directory
    # (tools/flint.sh runs tests from the repo root). Never a fixed path, or a
    # worktree run would scaffold against a different checkout's commit.
    var env = _env_present("FLINT_ROOT")
    if env[0] and file_size(env[1] + "/tools/flint.sh") > 0:
        return env[1]
    var cwd = _cwd()
    if cwd.byte_length() > 0 and file_size(cwd + "/tools/flint.sh") > 0:
        return cwd
    return "/home/user/flint"


def _rm_rf_tmp(path: String):
    if not path.startswith("/tmp/flint-hygiene-cooker-"):
        return
    var cmd = "rm -rf " + _sh_quote(path)
    var owned = cmd
    _ = external_call["system", Int32](owned.as_c_string_span())


def _make_minimal_wav_16bit(sample_rate: Int = 22050, num_samples: Int = 64) -> List[UInt8]:
    var data_bytes = num_samples * 2
    var riff_size = 36 + data_bytes
    var buf = List[UInt8](capacity=44 + data_bytes)
    # "RIFF"
    buf.append(82); buf.append(73); buf.append(70); buf.append(70)
    buf.append(UInt8(riff_size & 0xFF))
    buf.append(UInt8((riff_size >> 8) & 0xFF))
    buf.append(UInt8((riff_size >> 16) & 0xFF))
    buf.append(UInt8((riff_size >> 24) & 0xFF))
    # "WAVEfmt "
    buf.append(87); buf.append(65); buf.append(86); buf.append(69)
    buf.append(102); buf.append(109); buf.append(116); buf.append(32)
    # subchunk1 size = 16
    buf.append(16); buf.append(0); buf.append(0); buf.append(0)
    # PCM = 1, channels = 1
    buf.append(1); buf.append(0); buf.append(1); buf.append(0)
    # sample rate
    buf.append(UInt8(sample_rate & 0xFF))
    buf.append(UInt8((sample_rate >> 8) & 0xFF))
    buf.append(UInt8((sample_rate >> 16) & 0xFF))
    buf.append(UInt8((sample_rate >> 24) & 0xFF))
    # byte rate = sample_rate * 2
    var byte_rate = sample_rate * 2
    buf.append(UInt8(byte_rate & 0xFF))
    buf.append(UInt8((byte_rate >> 8) & 0xFF))
    buf.append(UInt8((byte_rate >> 16) & 0xFF))
    buf.append(UInt8((byte_rate >> 24) & 0xFF))
    # block align = 2, bits per sample = 16
    buf.append(2); buf.append(0); buf.append(16); buf.append(0)
    # "data"
    buf.append(100); buf.append(97); buf.append(116); buf.append(97)
    buf.append(UInt8(data_bytes & 0xFF))
    buf.append(UInt8((data_bytes >> 8) & 0xFF))
    buf.append(UInt8((data_bytes >> 16) & 0xFF))
    buf.append(UInt8((data_bytes >> 24) & 0xFF))
    for i in range(num_samples):
        var s = Int16((i % 16 - 8) * 1000)
        var u = UInt16(s)
        buf.append(UInt8(u & 0xFF))
        buf.append(UInt8((u >> 8) & 0xFF))
    return buf^


def test_f01_hygiene_env_and_paths(tmp_root: String, repo_root: String) raises:
    print("  [1/6] F01: Zero game leaks, FLINT_* env vars, and project-named log paths...")
    var grep_out = tmp_root + "/f01_grep.out"
    var rc = _run_sh(
        "grep -rnE 'Maroon|maroon|deepsong|Deepsong|sharddepth|Sharddepth|SHARDDEPTH|SD_[A-Z0-9_]+' "
        + _sh_quote(repo_root + "/src/flint/"),
        grep_out,
    )
    assert_equal(rc, 1)
    assert_equal(_read_text(grep_out).byte_length(), 0)

    # Preserve non-leak math locals and binary magic constants
    assert_true(_read_text(repo_root + "/src/flint/physics/narrowphase.mojo").find("sd_pt") >= 0)
    assert_true(_read_text(repo_root + "/src/flint/render/ui_overlay.mojo").find("sd_line") >= 0)
    assert_true(_read_text(repo_root + "/src/flint/core/anchor.mojo").find("SDSAFE1") >= 0)
    assert_true(_read_text(repo_root + "/src/flint/core/replay.mojo").find("SDRP") >= 0)

    # Active constants & cvar table hygiene
    assert_equal(ANIM_GRAPH_PATH, "assets/anim/default.json")
    assert_true(CVAR_COUNT > 0)
    for i in range(CVAR_COUNT):
        var e = cvar_entry(i)
        assert_true(e.name.byte_length() > 0)
        assert_equal(e.help.find("Maroon"), -1)
        assert_equal(e.help.find("SD_LOG"), -1)

    var prev_home = _env_present("HOME")
    var prev_xdg_state = _env_present("XDG_STATE_HOME")
    var prev_xdg_data = _env_present("XDG_DATA_HOME")
    var prev_flint_state = _env_present("FLINT_STATE_DIR")
    var prev_flint_save = _env_present("FLINT_SAVE_DIR")
    var prev_sd_state = _env_present("SD_STATE_DIR")
    var prev_sd_save = _env_present("SD_SAVE_DIR")
    var prev_vk_val = _env_present("FLINT_VULKAN_VALIDATION")

    try:
        _unsetenv("FLINT_STATE_DIR")
        _unsetenv("XDG_STATE_HOME")
        _unsetenv("FLINT_SAVE_DIR")
        _unsetenv("XDG_DATA_HOME")
        _setenv("SD_STATE_DIR", tmp_root + "/ignored_sd_state")
        _setenv("SD_SAVE_DIR", tmp_root + "/ignored_sd_save")
        _setenv("HOME", tmp_root + "/home")

        var st_home = state_directory()
        assert_equal(st_home, tmp_root + "/home/.local/state/flint")
        assert_equal(st_home.find("sharddepth"), -1)
        assert_equal(player_save_directory(), tmp_root + "/home/.local/share/flint")

        _setenv("XDG_STATE_HOME", tmp_root + "/xdg_state")
        _setenv("XDG_DATA_HOME", tmp_root + "/xdg_data")
        assert_equal(state_directory(), tmp_root + "/xdg_state/flint")
        assert_equal(state_directory("mygame"), tmp_root + "/xdg_state/mygame")
        assert_equal(state_directory("../bad/name"), tmp_root + "/xdg_state/.._bad_name")
        assert_equal(state_directory(".."), tmp_root + "/xdg_state/flint")
        assert_equal(player_save_directory(), tmp_root + "/xdg_data/flint")

        _setenv("FLINT_STATE_DIR", tmp_root + "/custom_state")
        _setenv("FLINT_SAVE_DIR", tmp_root + "/custom_save")
        assert_equal(state_directory("mygame"), tmp_root + "/custom_state")
        assert_equal(player_save_directory(), tmp_root + "/custom_save")

        _setenv("FLINT_VULKAN_VALIDATION", "1")
        assert_true(env_flag("FLINT_VULKAN_VALIDATION"))

        _unsetenv("FLINT_STATE_DIR")
        _setenv("XDG_STATE_HOME", tmp_root + "/log_xdg")
        assert_true(start_log("test_app_log"))
        log_info(LOG_CORE, "hello_hygiene_marker")
        _ = flush_log()
        var lp = log_path()
        assert_equal(lp, tmp_root + "/log_xdg/test_app_log/logs/test_app_log.log")
        assert_true(file_size(lp) > 0)
        assert_true(_read_text(lp).find("hello_hygiene_marker") >= 0)
        anchor_shutdown()
    finally:
        anchor_shutdown()
        _restore_env("HOME", prev_home[1], prev_home[0])
        _restore_env("XDG_STATE_HOME", prev_xdg_state[1], prev_xdg_state[0])
        _restore_env("XDG_DATA_HOME", prev_xdg_data[1], prev_xdg_data[0])
        _restore_env("FLINT_STATE_DIR", prev_flint_state[1], prev_flint_state[0])
        _restore_env("FLINT_SAVE_DIR", prev_flint_save[1], prev_flint_save[0])
        _restore_env("SD_STATE_DIR", prev_sd_state[1], prev_sd_state[0])
        _restore_env("SD_SAVE_DIR", prev_sd_save[1], prev_sd_save[0])
        _restore_env("FLINT_VULKAN_VALIDATION", prev_vk_val[1], prev_vk_val[0])


def test_f02_extended_zero_leak_gate(tmp_root: String, repo_root: String) raises:
    print("  [2/6] F02: Extended Zero-Leak Gate in sync_from_deepsong.sh --verify...")
    var verify_out = tmp_root + "/f02_real_verify.out"
    var rc_real = _run_sh(
        "bash " + _sh_quote(repo_root + "/tools/sync_from_deepsong.sh") + " --verify",
        verify_out,
    )
    assert_equal(rc_real, 0)
    assert_true(_read_text(verify_out).find("[PASS]") >= 0)

    var fake_repo = tmp_root + "/fake_repo"
    assert_true(make_dirs(fake_repo + "/tools"))
    assert_true(make_dirs(fake_repo + "/src/flint"))
    var sync_src = _read_text(repo_root + "/tools/sync_from_deepsong.sh")
    assert_true(sync_src.byte_length() > 0)
    assert_true(_write_text(fake_repo + "/tools/sync_from_deepsong.sh", sync_src))

    var stub_out = tmp_root + "/f02_stub_gen.out"
    var py_cmd = (
        "python3 -c "
        + _sh_quote(
            "import ast, os, re, sys\n"
            "sync_p, dst = sys.argv[1], sys.argv[2]\n"
            "s = open(sync_p).read()\n"
            "enh = ast.literal_eval(re.search(r'FLINT_ENHANCED\\s*=\\s*(\\{.*?\\n\\})', s, re.DOTALL).group(1))\n"
            "mods = ast.literal_eval(re.search(r'MODULES\\s*=\\s*(\\{.*?\\n\\})', s, re.DOTALL).group(1))\n"
            "for pkg, fname in enh:\n"
            "    os.makedirs(os.path.join(dst, pkg), exist_ok=True)\n"
            "    open(os.path.join(dst, pkg, fname), 'w').write('def ok():\\n    pass\\n')\n"
            "for pkg, files in mods.items():\n"
            "    os.makedirs(os.path.join(dst, pkg), exist_ok=True)\n"
            "    for fname in files:\n"
            "        open(os.path.join(dst, pkg, fname), 'w').write('def ok():\\n    pass\\n')\n"
        )
        + " "
        + _sh_quote(fake_repo + "/tools/sync_from_deepsong.sh")
        + " "
        + _sh_quote(fake_repo + "/src/flint")
    )
    assert_equal(_run_sh(py_cmd, stub_out), 0)

    var fake_pass_out = tmp_root + "/f02_fake_pass.out"
    assert_equal(
        _run_sh("bash " + _sh_quote(fake_repo + "/tools/sync_from_deepsong.sh") + " --verify", fake_pass_out),
        0,
    )

    var leak_target = fake_repo + "/src/flint/core/log.mojo"
    var leaks = [
        String("# SD_FOO_BAR"),
        String("# sharddepth"),
        String("# Maroon"),
        String("# deepsong"),
        String("from gameplay.foo import bar"),
    ]
    for i in range(len(leaks)):
        assert_true(_write_text(leak_target, leaks[i] + "\n"))
        var fail_out = tmp_root + "/f02_leak_" + String(i) + ".out"
        var rc_leak = _run_sh(
            "bash " + _sh_quote(fake_repo + "/tools/sync_from_deepsong.sh") + " --verify",
            fail_out,
        )
        assert_true(rc_leak != 0)
        assert_true(_read_text(fail_out).find("[FAIL]") >= 0)

    # Allowed math locals must NOT fail the zero-leak gate
    assert_true(_write_text(leak_target, "def f():\n    var sd_pt = 0.0\n    var sd_line = 0.0\n"))
    var allowed_out = tmp_root + "/f02_allowed.out"
    assert_equal(
        _run_sh("bash " + _sh_quote(fake_repo + "/tools/sync_from_deepsong.sh") + " --verify", allowed_out),
        0,
    )


def test_f03_flint_new_crash_handler_and_logs(tmp_root: String, repo_root: String) raises:
    print("  [3/6] F03: flint.sh new crash handler, project-named logs, and crash report...")
    var proj = tmp_root + "/hygiene_sample_game"
    var new_out = tmp_root + "/f03_new.out"
    var rc_new = _run_sh(
        "bash " + _sh_quote(repo_root + "/tools/flint.sh") + " new " + _sh_quote(proj),
        new_out,
    )
    if rc_new != 0:
        print("  [F03] flint.sh new rc=" + String(rc_new) + " output:\n" + _read_text(new_out))
    assert_equal(rc_new, 0)

    var main_src = _read_text(proj + "/src/main.mojo")
    assert_true(main_src.find("start_log(cfg.app_name)") >= 0)
    assert_true(main_src.find("enable_crash_handler=True") >= 0)
    assert_true(main_src.find("flush_log()") >= 0)
    assert_equal(main_src.find("sharddepth"), -1)
    assert_equal(main_src.find("SD_"), -1)
    assert_equal(main_src.find("Maroon"), -1)
    assert_equal(main_src.find("deepsong"), -1)

    var game_xdg = tmp_root + "/game_xdg_state"
    var smoke_out = tmp_root + "/f03_smoke.out"
    var rc_smoke = _run_sh(
        "cd "
        + _sh_quote(proj)
        + " && env -u FLINT_STATE_DIR XDG_STATE_HOME="
        + _sh_quote(game_xdg)
        + " FLINT_HEADLESS=1 FLINT_ROOT="
        + _sh_quote(repo_root)
        + " bash "
        + _sh_quote(proj + "/tools/run.sh")
        + " --headless-smoke",
        smoke_out,
    )
    if rc_smoke != 0:
        # Show what the scaffolded game printed.
        print("  [F03] headless smoke rc=" + String(rc_smoke) + " output:\n" + _read_text(smoke_out))
    assert_equal(rc_smoke, 0)

    var expected_log = game_xdg + "/hygiene_sample_game/logs/hygiene_sample_game.log"
    assert_true(file_size(expected_log) > 0)
    assert_equal(len(list_directory(game_xdg + "/sharddepth")), 0)

    # Crash report probe
    var probe_src = tmp_root + "/crash_probe.mojo"
    var probe_bin = tmp_root + "/crash_probe_bin"
    var probe_code = (
        "from std.ffi import external_call\n"
        "from flint.app import AppConfig, EngineApp\n"
        "from flint.core.log import flush_log, start_log\n"
        "def main():\n"
        "    _ = start_log(\"crash_probe_app\")\n"
        "    var app = EngineApp(AppConfig(title=\"crash_probe_app\", headless=True, enable_crash_handler=True))\n"
        "    app.startup()\n"
        "    _ = flush_log()\n"
        "    external_call[\"raise\", NoneType](Int32(6))\n"
    )
    assert_true(_write_text(probe_src, probe_code))
    var mojo_prefix = repo_root + "/.pixi/envs/default"
    var mojo_bin = mojo_prefix + "/bin/mojo"
    var build_probe_out = tmp_root + "/f03_build_probe.out"
    var rc_build_probe = _run_sh(
        "MODULAR_HOME="
        + _sh_quote(mojo_prefix + "/share/max")
        + " LD_LIBRARY_PATH="
        + _sh_quote(mojo_prefix + "/lib")
        + " "
        + _sh_quote(mojo_bin)
        + " build "
        + _sh_quote(probe_src)
        + " -o "
        + _sh_quote(probe_bin)
        + " -I "
        + _sh_quote(repo_root + "/src")
        + " -O1 --fp-mode contract=off --Werror",
        build_probe_out,
    )
    assert_equal(rc_build_probe, 0)

    var crash_xdg = tmp_root + "/crash_xdg"
    var crash_out_path = tmp_root + "/f03_crash_run.out"
    var rc_crash = _run_sh(
        "env -u FLINT_STATE_DIR XDG_STATE_HOME="
        + _sh_quote(crash_xdg)
        + " "
        + _sh_quote(probe_bin),
        crash_out_path,
    )
    assert_true(rc_crash != 0)
    var crash_stderr = _read_text(crash_out_path)
    assert_true(crash_stderr.find("FLINT_CRASH ") >= 0)
    assert_equal(crash_stderr.find("SHARDDEPTH_CRASH"), -1)

    var crashes_dir = crash_xdg + "/crash_probe_app/crashes"
    var crash_entries = list_directory(crashes_dir)
    assert_true(len(crash_entries) >= 1)
    var report_txt = _read_text(crashes_dir + "/" + crash_entries[0] + "/crash.txt")
    assert_true(report_txt.startswith("Flint crash report"))
    assert_equal(report_txt.find("Sharddepth crash report"), -1)


def test_f04_f05_cook_workspace_extensions_cache_and_package(tmp_root: String, repo_root: String) raises:
    print("  [4/6] F04 & F05: cook_workspace extensions, 0-byte raw, cache hits, and flint.sh package...")
    var ws = tmp_root + "/cook_ws"
    var cache = tmp_root + "/cook_cache"
    var pack = tmp_root + "/game.flpk"
    assert_true(make_dirs(ws + "/textures"))
    assert_true(make_dirs(ws + "/audio"))
    assert_true(make_dirs(ws + "/scenes"))
    assert_true(make_dirs(ws + "/prefabs"))
    assert_true(make_dirs(ws + "/materials"))
    assert_true(make_dirs(ws + "/anim"))
    assert_true(make_dirs(ws + "/rigs"))
    assert_true(make_dirs(ws + "/data"))

    # Populate all supported extensions including 0-byte raw JSON
    var px = List[UInt32](length=16 * 16, fill=UInt32(0xFF336699))
    assert_true(write_png(ws + "/textures/tex.png", px, 16, 16, stamped=False))
    var wav_bytes = _make_minimal_wav_16bit()
    assert_true(write_bytes(ws + "/audio/beep.wav", wav_bytes))

    var sc_json = String('{"version":1,"entities":[{"name":"Hero","pos":[1,2,3]}]}')
    var sc_bin = List[UInt8](capacity=8)
    for bi in range(8):
        sc_bin.append(UInt8(0x40 + bi))
    var prefab_json = String('{"name":"crate","scale":1.0}')
    var mat_json = String('{"roughness":0.2,"metallic":0.8,"albedo":[0.9,0.2,0.1]}')
    var anim_json = String('{"states":["Idle","Run"]}')
    var rig_json = String('{"joints":["Root","Spine","Head"]}')
    var data_json = String('{"hp":100,"speed":5.5,"title":"Flint"}')
    var pcm = List[Int16](capacity=20)
    for si in range(20):
        pcm.append(Int16((si - 10) * 500))
    var qoa_bytes = encode_qoa(pcm, 1, 22050)
    assert_true(len(qoa_bytes) > 0)
    var empty_bytes = List[UInt8]()

    assert_true(_write_text(ws + "/scenes/main.flscn", sc_json))
    assert_true(write_bytes(ws + "/scenes/main.flscnb", sc_bin))
    assert_true(_write_text(ws + "/prefabs/crate.flprefab", prefab_json))
    assert_true(_write_text(ws + "/materials/metal.flmat", mat_json))
    assert_true(_write_text(ws + "/anim/hero.flanim", anim_json))
    assert_true(_write_text(ws + "/rigs/hero.flrig", rig_json))
    assert_true(_write_text(ws + "/data/config.json", data_json))
    assert_true(write_bytes(ws + "/audio/music.qoa", qoa_bytes))
    assert_true(write_bytes(ws + "/data/empty.json", empty_bytes))
    assert_equal(file_size(ws + "/data/empty.json"), 0)

    var pool = JobPool()
    assert_true(pool.start(4))

    # Run 1: cold cook
    var rep1 = cook_workspace(ws, pack, pool, cache_dir=cache)
    assert_true(rep1.ok)
    assert_equal(rep1.total_assets, 11)
    assert_equal(rep1.cooked_count, 11)
    assert_equal(rep1.cache_hits, 0)
    assert_equal(rep1.failed_count, 0)

    # Verify archive TOC, CRC64, ASSET_KIND_RAW payloads & AssetServer.load_raw
    var arch = FlpkV3MappedArchive.open(pack)
    assert_true(arch.valid)
    var expected_paths = [
        String("textures/tex.png"),
        String("audio/beep.wav"),
        String("scenes/main.flscn"),
        String("scenes/main.flscnb"),
        String("prefabs/crate.flprefab"),
        String("materials/metal.flmat"),
        String("anim/hero.flanim"),
        String("rigs/hero.flrig"),
        String("data/config.json"),
        String("audio/music.qoa"),
        String("data/empty.json"),
    ]
    for i in range(len(expected_paths)):
        var tidx = arch.find_by_path(expected_paths[i], 0)
        assert_true(tidx >= 0)
        assert_true(arch.verify_entry_crc64(tidx))
        if i >= 2:
            assert_equal(Int(arch.toc_entry(tidx).asset_kind), Int(ASSET_KIND_RAW))

    var sc_read = arch.read_by_path("scenes/main.flscn", 0, verify_crc=True)
    assert_true(sc_read[0])
    assert_equal(String(unsafe_from_utf8=sc_read[1]), sc_json)

    var empty_read = arch.read_by_path("data/empty.json", 0, verify_crc=True)
    assert_true(empty_read[0])
    assert_equal(len(empty_read[1]), 0)

    var qoa_read = arch.read_by_path("audio/music.qoa", 0, verify_crc=True)
    assert_true(qoa_read[0])
    assert_equal(len(qoa_read[1]), len(qoa_bytes))
    arch.close()

    var srv = AssetServer(enable_watcher=False)
    assert_true(srv.mount_flpk(pack))
    var h_sc = srv.load_raw("scenes/main.flscn")
    var p_sc = srv.raws.get_ptr(h_sc)
    assert_true(Bool(p_sc))
    assert_equal(String(unsafe_from_utf8=p_sc.value()[unsafe_offset=0].bytes), sc_json)
    var h_emp = srv.load_raw("data/empty.json")
    var p_emp = srv.raws.get_ptr(h_emp)
    assert_true(Bool(p_emp))
    assert_equal(len(p_emp.value()[unsafe_offset=0].bytes), 0)
    srv.close()

    # Run 2: unchanged re-cook -> 100% cache hit
    var rep2 = cook_workspace(ws, pack, pool, cache_dir=cache)
    assert_true(rep2.ok)
    assert_equal(rep2.total_assets, 11)
    assert_equal(rep2.cache_hits, 11)
    assert_equal(rep2.cooked_count, 0)

    # Run 3: equivalent source_dir spelling hits canonical rel_path fast signature
    var rep3 = cook_workspace(ws + "/.", pack, pool, cache_dir=cache)
    assert_true(rep3.ok)
    assert_equal(rep3.total_assets, 11)
    assert_equal(rep3.cache_hits, 11)
    assert_equal(rep3.cooked_count, 0)

    # Run 4: edit 1 file -> 1 cooked, 10 cache hits
    var mat_v2 = String('{"roughness":0.85,"metallic":0.1}')
    assert_true(_write_text(ws + "/materials/metal.flmat", mat_v2))
    var rep4 = cook_workspace(ws, pack, pool, cache_dir=cache)
    assert_true(rep4.ok)
    assert_equal(rep4.total_assets, 11)
    assert_equal(rep4.cooked_count, 1)
    assert_equal(rep4.cache_hits, 10)

    var arch2 = FlpkV3MappedArchive.open(pack)
    assert_true(arch2.valid)
    var mat_read = arch2.read_by_path("materials/metal.flmat", 0, verify_crc=True)
    assert_true(mat_read[0])
    assert_equal(String(unsafe_from_utf8=mat_read[1]), mat_v2)
    arch2.close()

    # Run 5: unchanged again -> 100% cache hit
    var rep5 = cook_workspace(ws, pack, pool, cache_dir=cache)
    assert_true(rep5.ok)
    assert_equal(rep5.total_assets, 11)
    assert_equal(rep5.cache_hits, 11)
    assert_equal(rep5.cooked_count, 0)

    # Run 6: rename/move with .flmeta preserving UUID updates rel_path even on cache hit
    var mv_ws = tmp_root + "/mv_flmeta_ws"
    var mv_cache = tmp_root + "/mv_flmeta_cache"
    var mv_pack = tmp_root + "/mv_flmeta.flpk"
    assert_true(make_dirs(mv_ws + "/data"))
    var tracked_body = String('{"color":"cyan","roughness":0.3}')
    assert_true(_write_text(mv_ws + "/data/tracked_a.flmat", tracked_body))
    var tracked_meta = AssetMeta()
    tracked_meta.uuid = AssetUuid.from_path("custom/preserved_uuid_mat")
    tracked_meta.asset_kind = ASSET_KIND_RAW
    tracked_meta.source_path = "data/tracked_a.flmat"
    assert_true(write_flmeta(mv_ws + "/data/tracked_a.flmat.flmeta", tracked_meta))

    var mv_rep1 = cook_workspace(mv_ws, mv_pack, pool, cache_dir=mv_cache)
    assert_true(mv_rep1.ok)
    assert_equal(mv_rep1.cooked_count, 1)

    assert_true(remove_file(mv_ws + "/data/tracked_a.flmat"))
    assert_true(remove_file(mv_ws + "/data/tracked_a.flmat.flmeta"))
    assert_true(make_dirs(mv_ws + "/data/moved"))
    assert_true(_write_text(mv_ws + "/data/moved/tracked_b.flmat", tracked_body))
    tracked_meta.source_path = "data/moved/tracked_b.flmat"
    assert_true(write_flmeta(mv_ws + "/data/moved/tracked_b.flmat.flmeta", tracked_meta))

    var mv_rep2 = cook_workspace(mv_ws, mv_pack, pool, cache_dir=mv_cache)
    assert_true(mv_rep2.ok)
    assert_equal(mv_rep2.total_assets, 1)
    assert_equal(mv_rep2.cache_hits, 1)
    assert_equal(mv_rep2.cooked_count, 0)
    assert_equal(mv_rep2.rel_paths[0], "data/moved/tracked_b.flmat")

    var mv_arch = FlpkV3MappedArchive.open(mv_pack)
    assert_true(mv_arch.valid)
    assert_true(mv_arch.find_by_path("data/moved/tracked_b.flmat", 0) >= 0)
    assert_equal(mv_arch.find_by_path("data/tracked_a.flmat", 0), -1)
    var mv_u_idx = mv_arch.find_by_uuid(tracked_meta.uuid, 0)
    assert_true(mv_u_idx >= 0)
    assert_equal(mv_arch.entry_path(mv_u_idx), "data/moved/tracked_b.flmat")
    mv_arch.close()

    var srv_mv = AssetServer(enable_watcher=False)
    assert_true(srv_mv.mount_flpk(mv_pack))
    var h_mv = srv_mv.load_raw("data/moved/tracked_b.flmat")
    var p_mv = srv_mv.raws.get_ptr(h_mv)
    assert_true(Bool(p_mv))
    assert_equal(String(unsafe_from_utf8=p_mv.value()[unsafe_offset=0].bytes), tracked_body)
    srv_mv.close()

    pool.stop()

    # F04: flint.sh package incremental cook_workspace + project-local cook_cache
    var pkg_proj = tmp_root + "/pkg_proj"
    var pkg_new_out = tmp_root + "/f04_new.out"
    assert_equal(
        _run_sh(
            "bash " + _sh_quote(repo_root + "/tools/flint.sh") + " new " + _sh_quote(pkg_proj),
            pkg_new_out,
        ),
        0,
    )
    assert_true(make_dirs(pkg_proj + "/assets"))
    assert_true(_write_text(pkg_proj + "/assets/main.flscn", sc_json))
    assert_true(_write_text(pkg_proj + "/assets/data.json", data_json))
    assert_true(_write_text(pkg_proj + "/assets/mat.flmat", mat_json))
    assert_true(write_png(pkg_proj + "/assets/icon.png", px, 16, 16, stamped=False))
    assert_true(_write_text(pkg_proj + "/assets/readme.txt", "loose readme\n"))

    var pkg_dist = tmp_root + "/pkg_dist"
    var pkg1_out = tmp_root + "/f04_pkg1.out"
    assert_equal(
        _run_sh(
            "cd "
            + _sh_quote(pkg_proj)
            + " && bash "
            + _sh_quote(repo_root + "/tools/flint.sh")
            + " package "
            + _sh_quote(pkg_dist),
            pkg1_out,
        ),
        0,
    )
    assert_true(file_size(pkg_dist + "/assets.flpk") > 64)
    assert_true(file_size(pkg_dist + "/assets/readme.txt") > 0)
    assert_true(_read_text(pkg1_out).find("cooked=4, cache_hits=0") >= 0)
    assert_true(len(list_directory(pkg_proj + "/build/cook_cache")) > 0)

    var inspect_out = tmp_root + "/f04_inspect.out"
    assert_equal(
        _run_sh(
            _sh_quote(repo_root + "/build/flint-pack")
            + " --inspect "
            + _sh_quote(pkg_dist + "/assets.flpk"),
            inspect_out,
        ),
        0,
    )
    var inspect_txt = _read_text(inspect_out)
    assert_true(inspect_txt.find("main.flscn") >= 0)
    assert_true(inspect_txt.find("data.json") >= 0)
    assert_true(inspect_txt.find("mat.flmat") >= 0)
    assert_true(inspect_txt.find("icon.png") >= 0)

    # Second run into same dist -> 100% cache hit
    var pkg2_out = tmp_root + "/f04_pkg2.out"
    assert_equal(
        _run_sh(
            "cd "
            + _sh_quote(pkg_proj)
            + " && bash "
            + _sh_quote(repo_root + "/tools/flint.sh")
            + " package "
            + _sh_quote(pkg_dist),
            pkg2_out,
        ),
        0,
    )
    assert_true(_read_text(pkg2_out).find("cooked=0, cache_hits=4") >= 0)

    # Edit 1 asset -> only 1 recooked, 3 cache hits, updated payload in assets.flpk
    var data_v2 = String('{"hp":250,"speed":7.0}')
    assert_true(_write_text(pkg_proj + "/assets/data.json", data_v2))
    var pkg3_out = tmp_root + "/f04_pkg3.out"
    assert_equal(
        _run_sh(
            "cd "
            + _sh_quote(pkg_proj)
            + " && bash "
            + _sh_quote(repo_root + "/tools/flint.sh")
            + " package "
            + _sh_quote(pkg_dist),
            pkg3_out,
        ),
        0,
    )
    assert_true(_read_text(pkg3_out).find("cooked=1, cache_hits=3") >= 0)

    var srv_pkg = AssetServer(enable_watcher=False)
    assert_true(srv_pkg.mount_flpk(pkg_dist + "/assets.flpk"))
    var h_data = srv_pkg.load_raw("data.json")
    var p_data = srv_pkg.raws.get_ptr(h_data)
    assert_true(Bool(p_data))
    assert_equal(String(unsafe_from_utf8=p_data.value()[unsafe_offset=0].bytes), data_v2)
    srv_pkg.close()


def test_f26b_flint_version_pin_and_toml(tmp_root: String, repo_root: String) raises:
    _ = repo_root
    print("  [5/6] F26b: flint_version_pin, check_flint_version_pin, and TOML parsing...")
    var pin = flint_version_pin()
    assert_true(pin.startswith(String(FLINT_ENGINE_VERSION) + "+"))
    assert_true(check_flint_version_pin("")[0])
    assert_true(check_flint_version_pin("5.0.0")[0])
    assert_true(check_flint_version_pin(">=5.0")[0])
    assert_true(check_flint_version_pin(pin)[0])
    assert_false(check_flint_version_pin("99.0.0+deadbeef")[0])
    assert_false(check_flint_version_pin("6.0.0")[0])
    assert_false(check_flint_version_pin("not-a-version")[0])
    if String(BUILD_COMMIT) != "dev":
        assert_false(check_flint_version_pin("5.0.0+deadbeef")[0])

    var sample_toml = String(
        '[project]\nname = "demo"\n\n[engine]\nflint_version = "' + pin + '" # pinned\nwerror = true\n'
    )
    var toml_path = tmp_root + "/f26b_flint.toml"
    assert_true(_write_text(toml_path, sample_toml))
    var raw = read_bytes(toml_path)
    assert_equal(parse_flint_version_from_toml_bytes(raw), pin)


def test_f26c_asset_server_flpk_assets_prefix_alias(tmp_root: String, repo_root: String) raises:
    _ = repo_root
    print("  [6/6] F26c: AssetServer .flpk 'assets/' prefix alias & loose fallback...")
    var ws = tmp_root + "/alias_ws"
    var cache = tmp_root + "/alias_cache"
    var pack = tmp_root + "/alias.flpk"
    assert_true(make_dirs(ws + "/data"))
    assert_true(make_dirs(ws + "/meshes"))
    assert_true(_write_text(ws + "/main.flscn", '{"format":"flscn","version":2,"name":"main","entities":[]}\n'))
    assert_true(_write_text(ws + "/data/cfg.json", '{"ok":true}\n'))

    var mb = MeshBuilder()
    mb.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5))
    var mdl = NativeModel()
    mdl.meshes.append(mb^.finish())
    var glb_bytes = encode_native_glb(mdl)
    assert_true(len(glb_bytes) > 12)
    assert_true(write_bytes(ws + "/meshes/tri.glb", glb_bytes))

    var pool = JobPool()
    _ = pool.start(2)
    var rep = cook_workspace(ws, pack, pool, cache_dir=cache)
    pool.stop()
    assert_true(rep.ok)

    var srv = AssetServer(enable_watcher=False)
    assert_true(srv.mount_flpk(pack))

    # Raw lookup via both "assets/main.flscn" and "main.flscn" hits the same slot (no duplicate slot leak)
    var h_raw1 = srv.load_raw("assets/main.flscn")
    var h_raw2 = srv.load_raw("main.flscn")
    assert_true(srv.raws.is_loaded(h_raw1))
    assert_true(srv.raws.is_loaded(h_raw2))
    assert_equal(srv.raws.live_count, 1)

    # Mesh resolution via AssetRef("assets/meshes/tri.glb") caches cleanly across repeated frames
    var mref = AssetRef.from_path("assets/meshes/tri.glb", ASSET_KIND_MESH)
    var hm1 = srv.resolve_mesh(mref)
    assert_true(srv.meshes.is_loaded(hm1))
    var hm2 = srv.resolve_mesh(mref)
    assert_true(srv.meshes.is_loaded(hm2))
    assert_equal(srv.meshes.live_count, 1)
    srv.close()

    # Loose FS fallback with loose_root ending in '/assets' + "assets/..." rel_path
    var loose_proj = tmp_root + "/alias_loose_proj"
    assert_true(make_dirs(loose_proj + "/assets"))
    assert_true(_write_text(loose_proj + "/assets/only_loose.json", '{"loose":1}\n'))
    var srv_loose = AssetServer(enable_watcher=False)
    _ = srv_loose.watch_loose_directory(loose_proj + "/assets")
    assert_true(srv_loose._resolve_loose_disk_path("assets/only_loose.json").endswith("/assets/only_loose.json"))
    var h_l = srv_loose.load_raw("assets/only_loose.json")
    assert_true(srv_loose.raws.is_loaded(h_l))
    srv_loose.close()

    # patch.flpk > base.flpk precedence across "assets/" prefix differences for resolve_raw & resolve_mesh
    var pool2 = JobPool()
    _ = pool2.start(2)
    var ws_base_pref = tmp_root + "/ov_base_pref"
    var ws_patch_nopref = tmp_root + "/ov_patch_nopref"
    assert_true(make_dirs(ws_base_pref + "/assets/data"))
    assert_true(make_dirs(ws_base_pref + "/assets/meshes"))
    assert_true(make_dirs(ws_patch_nopref + "/data"))
    assert_true(make_dirs(ws_patch_nopref + "/meshes"))

    assert_true(_write_text(ws_base_pref + "/assets/data/override.json", '{"ver":"base"}'))
    assert_true(_write_text(ws_patch_nopref + "/data/override.json", '{"ver":"patch"}'))

    var mb1 = MeshBuilder()
    mb1.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5))
    var mdl1 = NativeModel()
    mdl1.meshes.append(mb1^.finish())
    var glb1 = encode_native_glb(mdl1)
    assert_true(write_bytes(ws_base_pref + "/assets/meshes/hero.glb", glb1))

    var mb2 = MeshBuilder()
    mb2.add_beveled_box(Vec3(0.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5))
    mb2.add_beveled_box(Vec3(2.0, 0.0, 0.0), Vec3(0.5, 0.5, 0.5))
    var mdl2 = NativeModel()
    mdl2.meshes.append(mb2^.finish())
    var glb2 = encode_native_glb(mdl2)
    assert_true(write_bytes(ws_patch_nopref + "/meshes/hero.glb", glb2))

    var pack_base_pref = tmp_root + "/ov_base_pref.flpk"
    var pack_patch_nopref = tmp_root + "/ov_patch_nopref.flpk"
    assert_true(cook_workspace(ws_base_pref, pack_base_pref, pool2, cache_dir=tmp_root + "/ov_c1").ok)
    assert_true(cook_workspace(ws_patch_nopref, pack_patch_nopref, pool2, cache_dir=tmp_root + "/ov_c2").ok)

    # Case A: base.flpk has "assets/..." and patch.flpk has "..."
    var srv_ov_a = AssetServer(enable_watcher=False)
    assert_true(srv_ov_a.mount_flpk(pack_base_pref))
    assert_true(srv_ov_a.mount_flpk(pack_patch_nopref))
    var h_ra = srv_ov_a.resolve_raw(AssetRef.from_path("assets/data/override.json", ASSET_KIND_RAW))
    var p_ra = srv_ov_a.raws.get_ptr(h_ra)
    assert_true(Bool(p_ra))
    assert_equal(String(unsafe_from_utf8=p_ra.value()[unsafe_offset=0].bytes), '{"ver":"patch"}')
    var h_ra2 = srv_ov_a.resolve_raw(AssetRef.from_path("data/override.json", ASSET_KIND_RAW))
    assert_true(srv_ov_a.raws.is_loaded(h_ra2))
    assert_equal(srv_ov_a.raws.live_count, 1)

    var h_ma = srv_ov_a.resolve_mesh(AssetRef.from_path("assets/meshes/hero.glb", ASSET_KIND_MESH))
    var p_ma = srv_ov_a.meshes.get_ptr(h_ma)
    assert_true(Bool(p_ma))
    var verts_patch = p_ma.value()[unsafe_offset=0].geom.total_vertices()
    var h_ma2 = srv_ov_a.resolve_mesh(AssetRef.from_path("meshes/hero.glb", ASSET_KIND_MESH))
    assert_true(srv_ov_a.meshes.is_loaded(h_ma2))
    assert_equal(srv_ov_a.meshes.live_count, 1)
    srv_ov_a.close()

    # Case B: base.flpk has "..." and patch.flpk has "assets/..."
    var srv_ov_b = AssetServer(enable_watcher=False)
    assert_true(srv_ov_b.mount_flpk(pack_patch_nopref))
    assert_true(srv_ov_b.mount_flpk(pack_base_pref))
    var h_rb = srv_ov_b.resolve_raw(AssetRef.from_path("data/override.json", ASSET_KIND_RAW))
    var p_rb = srv_ov_b.raws.get_ptr(h_rb)
    assert_true(Bool(p_rb))
    assert_equal(String(unsafe_from_utf8=p_rb.value()[unsafe_offset=0].bytes), '{"ver":"base"}')

    var h_mb = srv_ov_b.resolve_mesh(AssetRef.from_path("meshes/hero.glb", ASSET_KIND_MESH))
    var p_mb = srv_ov_b.meshes.get_ptr(h_mb)
    assert_true(Bool(p_mb))
    var verts_base = p_mb.value()[unsafe_offset=0].geom.total_vertices()
    assert_true(verts_patch > verts_base)
    srv_ov_b.close()
    pool2.stop()


def test_t5_hot_reload_texture_role_and_zero_byte_loose_parity(
    tmp_root: String, repo_root: String
) raises:
    _ = repo_root
    # 1. Texture role (BC5 normal map role=1 -> fmt=3) preserved across _hot_reload_path normalization
    var ws_hr = tmp_root + "/hr_tex_role"
    assert_true(make_dirs(ws_hr + "/assets"))
    var px1 = List[UInt32](length=4 * 4, fill=UInt32(0xFFFF8080))
    assert_true(write_png(ws_hr + "/assets/wall.png", px1, 4, 4, stamped=False))

    var srv_hr = AssetServer(enable_watcher=False)
    srv_hr.loose_root = ws_hr
    var th = srv_hr.load_texture_with_role("assets/wall.png", role=1, progressive=False)
    assert_true(srv_hr.textures.is_loaded(th))
    var p_before = srv_hr.textures.get_ptr(th)
    assert_true(Bool(p_before))
    assert_equal(Int(p_before.value()[unsafe_offset=0].tex_format), Int(TEX_FMT_BC5_UNORM))

    var px2 = List[UInt32](length=4 * 4, fill=UInt32(0xFFFA8C8C))
    assert_true(write_png(ws_hr + "/assets/wall.png", px2, 4, 4, stamped=False))
    assert_true(srv_hr._hot_reload_path("wall.png"))
    var p_after = srv_hr.textures.get_ptr(th)
    assert_true(Bool(p_after))
    assert_equal(Int(p_after.value()[unsafe_offset=0].tex_format), Int(TEX_FMT_BC5_UNORM))
    srv_hr.close()

    # 2. 0-byte loose raw asset parity with .flpk (without weakening 0-byte .flpk archive rejection)
    var ws_z = tmp_root + "/zero_raw_parity"
    assert_true(make_dirs(ws_z + "/cfg"))
    var empty_bytes = List[UInt8]()
    assert_true(write_bytes(ws_z + "/cfg/zero.json", empty_bytes))
    var z_pack = tmp_root + "/zero_raw.flpk"
    var z_cache = tmp_root + "/zero_raw_cache"
    var z_pool = JobPool()
    assert_true(z_pool.allocate())
    var z_rep = cook_workspace(ws_z, z_pack, z_pool, cache_dir=z_cache)
    z_pool.stop()
    assert_true(z_rep.ok)

    var srv_pk = AssetServer(enable_watcher=False)
    assert_true(srv_pk.mount_flpk(z_pack))
    var hp = srv_pk.load_raw("cfg/zero.json")
    var pp = srv_pk.raws.get_ptr(hp)
    assert_true(Bool(pp))
    assert_equal(len(pp.value()[unsafe_offset=0].bytes), 0)
    srv_pk.close()

    var srv_ls = AssetServer(enable_watcher=False)
    srv_ls.loose_root = ws_z
    var hl = srv_ls.load_raw("cfg/zero.json")
    var pl = srv_ls.raws.get_ptr(hl)
    assert_true(Bool(pl), "0-byte loose raw file must load cleanly to match .flpk")
    assert_equal(len(pl.value()[unsafe_offset=0].bytes), 0)
    # Missing file still returns invalid handle
    assert_false(srv_ls.raws.is_loaded(srv_ls.load_raw("cfg/missing.json")))
    # Hot-reload 0-byte -> non-empty -> 0-byte preserves Handle[RawBytesAsset]
    assert_true(_write_text(ws_z + "/cfg/zero.json", '{"v":1}'))
    assert_true(srv_ls._hot_reload_path("cfg/zero.json"))
    assert_equal(len(srv_ls.raws.get_ptr(hl).value()[unsafe_offset=0].bytes), 7)
    var empty_bytes2 = List[UInt8]()
    assert_true(write_bytes(ws_z + "/cfg/zero.json", empty_bytes2))
    assert_true(srv_ls._hot_reload_path("cfg/zero.json"))
    assert_equal(len(srv_ls.raws.get_ptr(hl).value()[unsafe_offset=0].bytes), 0)
    srv_ls.close()

    # 0-byte .flpk archive file must STILL be rejected!
    var bad_flpk = tmp_root + "/empty_archive.flpk"
    var empty_arch = List[UInt8]()
    assert_true(write_bytes(bad_flpk, empty_arch))
    var srv_bad = AssetServer(enable_watcher=False)
    assert_false(srv_bad.mount_flpk(bad_flpk), "0-byte .flpk archive must still be rejected")
    srv_bad.close()


def main() raises:
    var pid = Int(external_call["getpid", Int32]())
    var tmp_root = "/tmp/flint-hygiene-cooker-" + String(pid)
    assert_true(make_dirs(tmp_root))
    var repo_root = _repo_root()
    try:
        print("=== Flint M1 Hygiene & Cooker Test Suite ===")
        test_f01_hygiene_env_and_paths(tmp_root, repo_root)
        test_f02_extended_zero_leak_gate(tmp_root, repo_root)
        test_f03_flint_new_crash_handler_and_logs(tmp_root, repo_root)
        test_f04_f05_cook_workspace_extensions_cache_and_package(tmp_root, repo_root)
        test_f26b_flint_version_pin_and_toml(tmp_root, repo_root)
        test_f26c_asset_server_flpk_assets_prefix_alias(tmp_root, repo_root)
        test_t5_hot_reload_texture_role_and_zero_byte_loose_parity(tmp_root, repo_root)
        print("[PASS] All M1 Hygiene & Cooker tests passed.")
    finally:
        _rm_rf_tmp(tmp_root)

# AREA: m0
"""Tier 1 Pending M0 Public Contract Tests: Area B — EnvironmentComponent, SpotLightComponent, NavPlugin, AudioPlugin, AudioServer.stop_all() (F14)."""

from std.ffi import external_call
from std.math import abs
from flint.app.engine_app import (
    AppConfig,
    EngineApp,
    STAGE_FIRST,
    STAGE_FIXED_UPDATE,
    STAGE_LAST,
    STAGE_POST_UPDATE,
    STAGE_PRE_UPDATE,
    STAGE_STARTUP,
    STAGE_UPDATE,
)
from flint.audio.plugin import (
    AudioEmitterComponent,
    AudioListenerComponent,
    AudioPlugin,
    register_audio_codecs,
)
from flint.audio.qoa import encode_qoa
from flint.audio.server import AudioServer, BUS_SFX
from flint.core.math import Vec3
from flint.core.serial import BlobReader, BlobWriter
from flint.ecs.world import World
from flint.nav.navmesh import NavMesh
from flint.nav.plugin import NavPlugin
from flint.physics.plugin import ColliderComponent
from flint.physics.world import PhysicsWorld
from flint.platform.audio import decode_wav
from flint.render.plugin import (
    EnvironmentComponent,
    SpotLightComponent,
)
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, RenderWorld, ScalableRenderer
from flint.scene.scene import ComponentRegistry, Scene
from flint.scene.transform_hierarchy import (
    GlobalTransform,
    LocalTransform,
    PrevGlobalTransform,
    TransformPlugin,
)
from flint.studio.studio_app import COLLIDER_BOX, FlintStudioState
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    read_bytes,
    rm_rf_tmp,
    write_bytes,
    write_minimal_wav,
)


def _studio_has_nav_installed(studio: FlintStudioState) -> Bool:
    var stages = [
        STAGE_STARTUP,
        STAGE_FIRST,
        STAGE_PRE_UPDATE,
        STAGE_FIXED_UPDATE,
        STAGE_UPDATE,
        STAGE_POST_UPDATE,
        STAGE_LAST,
    ]
    for si in range(len(stages)):
        ref sys = studio.app.schedule.stages[stages[si]].systems
        for i in range(len(sys)):
            var nm = String(sys[i].name)
            if nm.find("nav") >= 0 or nm.find("Nav") >= 0:
                return True
    for gi in range(len(studio.menu.bar.groups)):
        for ii in range(len(studio.menu.bar.groups[gi].items)):
            var mt = studio.menu.bar.groups[gi].items[ii].title
            if mt.find("NavMesh") >= 0 or mt.find("Nav") >= 0:
                return True
    return False


def main() raises:
    var tmp = make_tmp_dir("t1_m0_contracts")
    var runner = StudioSuiteRunner("t1_m0_studio_m0_contracts")

    try:
        # 1. EnvironmentComponent + SpotLightComponent + Trigger + NavPlugin (installed by FlintStudioState) + NavMesh bake & wireframe
        var studio = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=320,
            height=180,
            worker_threads=0,
        )

        var floor_id = studio.spawn_entity(
            "GroundFloor",
            position=Vec3(0.0, -0.5, 0.0),
            scale=Vec3(20.0, 1.0, 20.0),
            collider_kind=COLLIDER_BOX,
            collider_half_extents=Vec3(10.0, 0.5, 10.0),
            mass=0.0,
            record_undo=False,
        )
        var obs_id = studio.spawn_entity(
            "NavObstacleBox",
            position=Vec3(0.0, 0.75, 0.0),
            scale=Vec3(2.0, 1.5, 4.0),
            collider_kind=COLLIDER_BOX,
            collider_half_extents=Vec3(1.0, 0.75, 2.0),
            mass=0.0,
            record_undo=False,
        )
        var env_e = studio.app.world.spawn()
        _ = studio.app.world.insert[LocalTransform](env_e, LocalTransform(Vec3(0.0, 5.0, 0.0)))
        _ = studio.app.world.insert[GlobalTransform](env_e, GlobalTransform())
        _ = studio.app.world.insert[PrevGlobalTransform](env_e, PrevGlobalTransform())
        _ = studio.app.world.insert[EnvironmentComponent](
            env_e,
            EnvironmentComponent(
                sun_dir=Vec3(0.0, -1.0, 0.0),
                sun_radiance=Vec3(0.4, 0.4, 0.4),
                exposure=0.5,
            ),
        )

        # Spawn ONLY SpotLightComponent (no PointLightComponent) to distinguish from Studio's 2 fallback lights
        var spot_e = studio.app.world.spawn()
        _ = studio.app.world.insert[LocalTransform](spot_e, LocalTransform(Vec3(2.0, 4.0, 2.0)))
        _ = studio.app.world.insert[GlobalTransform](spot_e, GlobalTransform())
        _ = studio.app.world.insert[PrevGlobalTransform](spot_e, PrevGlobalTransform())
        _ = studio.app.world.insert[SpotLightComponent](
            spot_e,
            SpotLightComponent(
                radiance=Vec3(8.0, 6.0, 4.0),
                range=18.0,
                inner_deg=18.0,
                outer_deg=35.0,
                enabled=True,
            ),
        )

        var trig_e = studio.app.world.spawn()
        _ = studio.app.world.insert[LocalTransform](trig_e, LocalTransform(Vec3(0.0, 1.0, 0.0)))
        _ = studio.app.world.insert[GlobalTransform](trig_e, GlobalTransform())
        _ = studio.app.world.insert[ColliderComponent](trig_e, ColliderComponent.box(Vec3(1.5, 1.5, 1.5), is_trigger=True))

        # Verify FlintStudioState itself installs NavPlugin at startup (not added post-startup!)
        var nav_probe_app = EngineApp(AppConfig(title="m0_nav_probe", headless=True, worker_threads=0))
        nav_probe_app.add_plugin(TransformPlugin())
        nav_probe_app.add_plugin(NavPlugin())
        nav_probe_app.startup()
        nav_probe_app.shutdown()

        var has_nav_plugin = (
            _studio_has_nav_installed(studio)
            and studio.registry.find_by_name("EnvironmentComponent") >= 0
            and studio.registry.find_by_name("SpotLightComponent") >= 0
        )

        studio.render_editor_frame(320, 180)
        var rw_ptr = studio.app.resources.get_ptr[RenderWorld]()
        var spot_lights_ok = (
            Bool(rw_ptr)
            and len(rw_ptr.value()[unsafe_offset=0].cluster_lights) == 1
            and abs(rw_ptr.value()[unsafe_offset=0].cluster_lights[0].radiance.x - Float32(8.0)) < Float32(1.0e-4)
            and abs(rw_ptr.value()[unsafe_offset=0].cluster_lights[0].radiance.y - Float32(6.0)) < Float32(1.0e-4)
            and abs(rw_ptr.value()[unsafe_offset=0].cluster_lights[0].radiance.z - Float32(4.0)) < Float32(1.0e-4)
        )

        # Verify EnvironmentComponent sun/exposure change alters rendered pixels
        var px_env_1 = studio.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
        var env_mut = studio.app.world.get_ptr_mut[EnvironmentComponent](env_e)
        if env_mut:
            env_mut.value()[unsafe_offset=0].sun_dir = Vec3(-0.4, -0.8, 0.3).normalized()
            env_mut.value()[unsafe_offset=0].sun_radiance = Vec3(8.0, 6.0, 4.0)
            env_mut.value()[unsafe_offset=0].exposure = 2.5
        studio.render_editor_frame(320, 180)
        var px_env_2 = studio.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
        var env_diffs = 0
        for i in range(len(px_env_1)):
            if px_env_1[i] != px_env_2[i]:
                env_diffs += 1
        var env_render_ok = env_diffs > 50

        # Bake NavMesh from Studio physics over floor + obstacle + verify pathfinding & binary round-trip
        var pw_ptr = studio.app.resources.get_ptr[PhysicsWorld]()
        var nav_polys_ok = False
        var nav_path_ok = False
        var nav_blob_rt_ok = False
        if pw_ptr:
            var nm = NavMesh.build_from_physics(pw_ptr.value()[unsafe_offset=0])
            nav_polys_ok = len(nm.polys) > 0
            if nav_polys_ok:
                var path = nm.find_path(Vec3(-6.0, 0.0, 0.0), Vec3(6.0, 0.0, 0.0))
                nav_path_ok = path.found and len(path.waypoints) >= 2
                var nbw = BlobWriter()
                nm.serialize(nbw)
                try:
                    var nbr = BlobReader(nbw.data.copy())
                    var nm2 = NavMesh()
                    nm2.deserialize(nbr)
                    nav_blob_rt_ok = len(nm2.polys) == len(nm.polys)
                except:
                    nav_blob_rt_ok = False

        _ = runner.check(
            "t1_m0_contract_env_spotlight_navplugin_lighting",
            floor_id >= 0
            and obs_id >= 0
            and has_nav_plugin
            and spot_lights_ok
            and env_render_ok
            and nav_polys_ok
            and nav_path_ok
            and nav_blob_rt_ok,
            "EnvironmentComponent/SpotLightComponent/NavPlugin bake contract failed: floor="
            + String(floor_id >= 0) + " obs=" + String(obs_id >= 0) + " nav_plugin=" + String(has_nav_plugin)
            + " spot_lights=" + String(spot_lights_ok) + " env_render=" + String(env_render_ok)
            + " (" + String(env_diffs) + " px)" + " nav_polys=" + String(nav_polys_ok)
            + " nav_path=" + String(nav_path_ok) + " nav_blob=" + String(nav_blob_rt_ok),
        )

        # 2. Save & reload round-trip of EnvironmentComponent + SpotLightComponent + Trigger + NavMesh
        var sc_json = tmp + "/m0_scene.flscn"
        var sc_bin = tmp + "/m0_scene.flscnb"
        var saved_j = studio.save_scene_file(sc_json)
        var saved_b = studio.save_scene_file(sc_bin)
        var hash_before = studio.state_hash()
        studio.shutdown()

        var studio2 = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=320,
            height=180,
            worker_threads=0,
        )
        var loaded_j = studio2.load_scene_file(sc_json)
        var hash_after_j = studio2.state_hash()
        var pw_j = studio2.app.resources.get_ptr[PhysicsWorld]()
        var nav_j_ok = Bool(pw_j) and len(NavMesh.build_from_physics(pw_j.value()[unsafe_offset=0]).polys) > 0
        var loaded_b = studio2.load_scene_file(sc_bin)
        var hash_after_b = studio2.state_hash()
        var pw_b = studio2.app.resources.get_ptr[PhysicsWorld]()
        var nav_b_ok = Bool(pw_b) and len(NavMesh.build_from_physics(pw_b.value()[unsafe_offset=0]).polys) > 0
        studio2.shutdown()
        _ = runner.check(
            "t1_m0_contract_env_spotlight_nav_trigger_scene_roundtrip",
            saved_j and saved_b and loaded_j and loaded_b and hash_after_j == hash_before and hash_after_b == hash_before and nav_j_ok and nav_b_ok,
            "M0 scene + NavMesh save/reload hash mismatch",
        )

        # 3. AudioPlugin + register_audio_codecs + AudioEmitterComponent/AudioListenerComponent roundtrip & Studio wiring + WAV/QOA path cache
        var wav_path = tmp + "/sfx_cache.wav"
        var wav_written = write_minimal_wav(wav_path, 0.25, 48000, 2)
        var wav = decode_wav(read_bytes(wav_path))
        var qoa_bytes = encode_qoa(wav.samples, wav.channels, wav.sample_rate)
        var qoa_path = tmp + "/sfx_cache.qoa"
        var qoa_written = write_bytes(qoa_path, qoa_bytes)

        var reg = ComponentRegistry.with_builtins()
        register_audio_codecs(reg)
        var audio_codecs_ok = (
            reg.find_by_name("AudioEmitterComponent") >= 0
            and reg.find_by_name("AudioListenerComponent") >= 0
        )

        var app_aud = EngineApp(AppConfig(title="e2e_audio_plugin", headless=True, worker_threads=0))
        app_aud.add_plugin(TransformPlugin())
        app_aud.add_plugin(AudioPlugin(headless=True, sample_rate=48000, quantum_frames=256, ring_capacity=512))
        app_aud.startup()
        var has_aud_res = app_aud.resources.has[AudioServer]()
        var ae = app_aud.world.spawn()
        _ = app_aud.world.insert[LocalTransform](ae, LocalTransform(Vec3(0.0, 1.0, 0.0)))
        _ = app_aud.world.insert[GlobalTransform](ae, GlobalTransform())
        _ = app_aud.world.insert[PrevGlobalTransform](ae, PrevGlobalTransform())
        _ = app_aud.world.insert[AudioEmitterComponent](ae, AudioEmitterComponent(asset_id=-1, asset_path=wav_path, volume=0.8, play_on_spawn=True))
        var ae2 = app_aud.world.spawn()
        _ = app_aud.world.insert[LocalTransform](ae2, LocalTransform(Vec3(1.0, 1.0, 0.0)))
        _ = app_aud.world.insert[GlobalTransform](ae2, GlobalTransform())
        _ = app_aud.world.insert[PrevGlobalTransform](ae2, PrevGlobalTransform())
        _ = app_aud.world.insert[AudioEmitterComponent](ae2, AudioEmitterComponent(asset_id=-1, asset_path=wav_path, volume=0.8, play_on_spawn=True))
        var aq1 = app_aud.world.spawn()
        _ = app_aud.world.insert[LocalTransform](aq1, LocalTransform(Vec3(2.0, 1.0, 0.0)))
        _ = app_aud.world.insert[GlobalTransform](aq1, GlobalTransform())
        _ = app_aud.world.insert[PrevGlobalTransform](aq1, PrevGlobalTransform())
        _ = app_aud.world.insert[AudioEmitterComponent](aq1, AudioEmitterComponent(asset_id=-1, asset_path=qoa_path, volume=0.8, play_on_spawn=True))
        var aq2 = app_aud.world.spawn()
        _ = app_aud.world.insert[LocalTransform](aq2, LocalTransform(Vec3(3.0, 1.0, 0.0)))
        _ = app_aud.world.insert[GlobalTransform](aq2, GlobalTransform())
        _ = app_aud.world.insert[PrevGlobalTransform](aq2, PrevGlobalTransform())
        _ = app_aud.world.insert[AudioEmitterComponent](aq2, AudioEmitterComponent(asset_id=-1, asset_path=qoa_path, volume=0.8, play_on_spawn=True))
        var al = app_aud.world.spawn()
        _ = app_aud.world.insert[LocalTransform](al, LocalTransform(Vec3(0.0, 1.0, 0.0)))
        _ = app_aud.world.insert[GlobalTransform](al, GlobalTransform())
        _ = app_aud.world.insert[PrevGlobalTransform](al, PrevGlobalTransform())
        _ = app_aud.world.insert[AudioListenerComponent](al, AudioListenerComponent(active=True))
        app_aud.step_one_frame(Float32(1.0 / 60.0))

        var wav_aid_1 = app_aud.world.get_ptr[AudioEmitterComponent](ae).value()[unsafe_offset=0].asset_id
        var wav_aid_2 = app_aud.world.get_ptr[AudioEmitterComponent](ae2).value()[unsafe_offset=0].asset_id
        var qoa_aid_1 = app_aud.world.get_ptr[AudioEmitterComponent](aq1).value()[unsafe_offset=0].asset_id
        var qoa_aid_2 = app_aud.world.get_ptr[AudioEmitterComponent](aq2).value()[unsafe_offset=0].asset_id
        var path_cache_ok = (
            wav_aid_1 >= 0
            and wav_aid_1 == wav_aid_2
            and qoa_aid_1 >= 0
            and qoa_aid_1 == qoa_aid_2
        )

        var sc_aud = Scene("audio_rt")
        var aud_scene_ok: Bool
        try:
            var jb = sc_aud.save_flscn_bytes(app_aud.world, reg)
            var bb = sc_aud.save_flscnb_bytes(app_aud.world, reg)
            var wj = World()
            reg.ensure_registered_in_world(wj)
            var nj = sc_aud.load_flscn_bytes(wj, reg, jb, clear_existing=True)
            var wb = World()
            reg.ensure_registered_in_world(wb)
            var nb = sc_aud.load_flscnb_bytes(wb, reg, bb, clear_existing=True)
            aud_scene_ok = len(nj) == 5 and len(nb) == 5 and wj.entity_count() == 5 and wb.entity_count() == 5
        except:
            aud_scene_ok = False
        app_aud.shutdown()

        var studio_aud_plg = FlintStudioState(headless=True, render_tier=RENDER_TIER_0_HEADLESS_SIMD, width=160, height=120, worker_threads=0)
        var studio_has_aud = (
            studio_aud_plg.registry.find_by_name("AudioEmitterComponent") >= 0
            and studio_aud_plg.registry.find_by_name("AudioListenerComponent") >= 0
            and studio_aud_plg.app.resources.has[AudioServer]()
        )
        studio_aud_plg.shutdown()
        _ = runner.check(
            "t1_m0_contract_audio_plugin_and_codecs",
            wav_written and qoa_written and audio_codecs_ok and has_aud_res and path_cache_ok and aud_scene_ok and studio_has_aud,
            "AudioPlugin / AudioEmitterComponent codec, path cache, or Studio install failed",
        )

        # 4. AudioServer.stop_all() on PIE stop + direct stop_all()
        var studio_aud = FlintStudioState(
            headless=True,
            render_tier=RENDER_TIER_0_HEADLESS_SIMD,
            width=160,
            height=120,
            worker_threads=0,
        )
        var srv_opt = studio_aud.app.resources.get_ptr_mut[AudioServer]()
        var pie_stop_all_ok = False
        if srv_opt:
            ref srv = srv_opt.value()[unsafe_offset=0]
            var aid_w = srv.register_wav_asset(wav)
            if studio_aud.pie_play():
                _ = srv.play_2d(aid_w, volume=0.5, bus_id=BUS_SFX, looping=True)
                for _ in range(15):
                    _ = studio_aud.step_pie_frame(Float32(1.0 / 60.0))
                    if srv.active_logical_count() > 0:
                        break
                    if srv.rt_running:
                        _ = external_call["usleep", Int32](UInt32(2000))
                var active_during_pie = srv.active_logical_count() > 0
                var stopped_pie: Bool
                try:
                    stopped_pie = studio_aud.pie_stop()
                except:
                    stopped_pie = False
                pie_stop_all_ok = active_during_pie and stopped_pie and srv.active_logical_count() == 0

                # Also verify direct AudioServer.stop_all()
                _ = srv.play_2d(aid_w, volume=0.5, bus_id=BUS_SFX, looping=True)
                srv.stop_all()
                pie_stop_all_ok = pie_stop_all_ok and srv.active_logical_count() == 0
        studio_aud.shutdown()
        _ = runner.check(
            "t1_m0_contract_audioserver_stop_all",
            pie_stop_all_ok,
            "AudioServer.stop_all() on pie_stop() or direct stop_all() failed",
        )
    finally:
        rm_rf_tmp(tmp)
    runner.finish()

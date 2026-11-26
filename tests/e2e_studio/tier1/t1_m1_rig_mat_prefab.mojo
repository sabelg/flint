# AREA: m1
"""Tier 1 Opaque API Tests: Area C — M1 FRIG/Skinned Playback, .flmat & .flprefab (F19, F20)."""

from std.collections import List
from flint.anim.animator import Animator
from flint.anim.flanim_ecs import (
    BoneSocketAttachment,
    load_flanim_bytes,
    save_flanim_bytes,
)
from flint.anim.flrig import (
    FRIG_FOURCC,
    FrigJointDef,
    FrigRigDef,
    FrigSocketDef,
    RagdollBoneConfig,
    RagdollChain,
    load_flrig_bytes,
    load_flrig_file,
    save_flrig_bytes,
    save_flrig_file,
)
from flint.anim.graph import (
    OP_GT,
    AnimGraphDef,
    Condition,
    StateDef,
    TransitionDef,
)
from flint.anim.plugin import (
    AnimAssetCache,
    AnimPlugin,
    AnimatorComponent,
    BoneSocketComponent,
    RigComponent,
    SkeletonComponent,
    register_anim_codecs,
)
from flint.anim.pose import SkeletonInstance
from flint.anim.sync import ClipInfo
from flint.app.engine_app import AppConfig, EngineApp
from flint.assets.asset_plugin import AssetPlugin
from flint.assets.asset_ref import AssetRef
from flint.assets.flmat import (
    FlmatAsset,
    MAT_SHADING_NPR_CEL,
    MAT_SHADING_PBR,
    MAT_SHADING_STYLIZED_VFX,
    load_flmat_file,
    save_flmat_file,
)
from flint.assets.flpk_v3 import ASSET_KIND_RAW, ASSET_KIND_TEXTURE, AssetUuid
from flint.assets.model import (
    NativeAnimation,
    NativeChannel,
    NativeMesh,
    NativeModel,
    NativeNode,
    NativeSkin,
)
from flint.core.math import Vec3
from flint.core.serial import fourcc
from flint.core.transform import Quat
from flint.ecs.world import World
from flint.platform.os import make_dirs
from flint.render.aaa_npr_cel import AaaCelParams
from flint.render.aaa_npr_vfx import AaaStylizedVfxParams
from flint.render.plugin import (
    MaterialOverrideComponent,
    MeshRendererComponent,
    RenderPlugin,
    RenderPluginStateResource,
    register_render_codecs,
)
from flint.render.render_world import RENDER_TIER_0_HEADLESS_SIMD, RenderWorld
from flint.scene import (
    ComponentRegistry,
    GlobalTransform,
    LocalTransform,
    NestedPrefabRef,
    Parent,
    Prefab,
    PrefabOverride,
    Scene,
    TransformPlugin,
    load_flprefab_file,
    save_flprefab_file,
)
from e2e_studio.common.studio_test_utils import (
    StudioSuiteRunner,
    make_tmp_dir,
    rm_rf_tmp,
)


def _build_rigged_native_model() -> NativeModel:
    var model = NativeModel()

    var n0 = NativeNode()
    n0.name = "Root"
    n0.children.append(1)
    n0.mesh = 0
    model.nodes.append(n0^)

    var n1 = NativeNode()
    n1.name = "Spine"
    n1.translation = Vec3(0.0, 1.0, 0.0)
    n1.base_translation = Vec3(0.0, 1.0, 0.0)
    n1.children.append(2)
    model.nodes.append(n1^)

    var n2 = NativeNode()
    n2.name = "Hand.R"
    n2.translation = Vec3(0.5, 0.4, 0.0)
    n2.base_translation = Vec3(0.5, 0.4, 0.0)
    model.nodes.append(n2^)
    model.roots.append(0)

    var skin = NativeSkin()
    skin.joints.append(0)
    skin.joints.append(1)
    skin.joints.append(2)
    for _ in range(3):
        # Identity 4x4 column-major matrix
        skin.inverse_bind.append(1.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0)
        skin.inverse_bind.append(0.0); skin.inverse_bind.append(1.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0)
        skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(1.0); skin.inverse_bind.append(0.0)
        skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(0.0); skin.inverse_bind.append(1.0)
    model.skins.append(skin^)

    var mesh = NativeMesh()
    mesh.skin = 0
    mesh.positions.append(0.0); mesh.positions.append(0.0); mesh.positions.append(0.0)
    mesh.positions.append(0.5); mesh.positions.append(0.0); mesh.positions.append(0.0)
    mesh.positions.append(0.0); mesh.positions.append(1.0); mesh.positions.append(0.0)
    for _ in range(3):
        mesh.normals.append(0.0); mesh.normals.append(0.0); mesh.normals.append(1.0)
        mesh.texcoords.append(0.0); mesh.texcoords.append(0.0)
        mesh.joints.append(0); mesh.joints.append(1); mesh.joints.append(2); mesh.joints.append(0)
        mesh.weights.append(0.5); mesh.weights.append(0.3); mesh.weights.append(0.2); mesh.weights.append(0.0)
    mesh.indices.append(0); mesh.indices.append(1); mesh.indices.append(2)
    model.meshes.append(mesh^)

    # Clip 0: Idle (hand slightly raised)
    var idle_anim = NativeAnimation()
    idle_anim.name = "Idle"
    idle_anim.duration = Float32(1.0)
    var ch_idle = NativeChannel()
    ch_idle.node = 2
    ch_idle.path = 0  # translation
    ch_idle.times.append(Float32(0.0))
    ch_idle.times.append(Float32(1.0))
    ch_idle.values.append(Float32(0.5)); ch_idle.values.append(Float32(0.4)); ch_idle.values.append(Float32(0.0))
    ch_idle.values.append(Float32(0.5)); ch_idle.values.append(Float32(0.45)); ch_idle.values.append(Float32(0.0))
    idle_anim.channels.append(ch_idle^)
    model.animations.append(idle_anim^)

    # Clip 1: Run (hand swings forward in +Z)
    var run_anim = NativeAnimation()
    run_anim.name = "Run"
    run_anim.duration = Float32(0.5)
    var ch_run = NativeChannel()
    ch_run.node = 2
    ch_run.path = 0
    ch_run.times.append(Float32(0.0))
    ch_run.times.append(Float32(0.5))
    ch_run.values.append(Float32(0.5)); ch_run.values.append(Float32(0.8)); ch_run.values.append(Float32(0.9))
    ch_run.values.append(Float32(0.5)); ch_run.values.append(Float32(0.8)); ch_run.values.append(Float32(0.9))
    run_anim.channels.append(ch_run^)
    model.animations.append(run_anim^)

    return model^


def main() raises:
    var tmp = make_tmp_dir("t1_m1_rig_mat_prefab")
    var runner = StudioSuiteRunner("t1_m1_rig_mat_prefab")

    # -----------------------------------------------------------------------
    # 1. F19: FRIG (.flrig) Slab Round-Trip (Joints, RagdollChain, Sockets)
    # -----------------------------------------------------------------------
    var rig = FrigRigDef("HeroRig")
    rig.joints.append(FrigJointDef("Root", -1, Vec3(0.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    rig.joints.append(FrigJointDef("Spine", 0, Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    rig.joints.append(FrigJointDef("Hand.R", 1, Vec3(0.5, 0.4, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))

    var chain = RagdollChain("RightArm")
    chain.bones.append(RagdollBoneConfig("Spine", Float32(0.18), Float32(0.45), Float32(8.0)))
    chain.bones.append(RagdollBoneConfig("Hand.R", Float32(0.08), Float32(0.25), Float32(2.0)))
    rig.ragdoll_chains.append(chain^)

    rig.sockets.append(FrigSocketDef("WeaponSocket", "Hand.R", Vec3(0.0, 0.1, 0.25), Quat(), Vec3(1.0, 1.0, 1.0)))

    var rig_bytes = save_flrig_bytes(rig)
    var flrig_path = tmp + "/hero.flrig"
    var saved_rig = save_flrig_file(flrig_path, rig)
    var rig_rt = load_flrig_file(flrig_path)
    var rig_rt_mem = load_flrig_bytes(rig_bytes)

    var frig_ok = (
        saved_rig
        and len(rig_bytes) >= 12
        and FRIG_FOURCC == fourcc("FRIG")
        and rig_rt.name == "HeroRig"
        and len(rig_rt.joints) == 3
        and rig_rt.joints[2].name == "Hand.R"
        and rig_rt.joints[2].parent_index == 1
        and len(rig_rt.ragdoll_chains) == 1
        and len(rig_rt.ragdoll_chains[0].bones) == 2
        and rig_rt.ragdoll_chains[0].bones[1].bone_name == "Hand.R"
        and len(rig_rt.sockets) == 1
        and rig_rt.sockets[0].name == "WeaponSocket"
        and rig_rt.sockets[0].bone_name == "Hand.R"
        and len(rig_rt_mem.joints) == 3
    )
    _ = runner.check("t1_m1_frig_slab_roundtrip_joints_ragdoll_sockets", frig_ok, "FRIG .flrig roundtrip failed")

    # -----------------------------------------------------------------------
    # 2. F19: Skinned Mesh Playback, Clip/State Transition & Socket Sync
    # -----------------------------------------------------------------------
    var g = AnimGraphDef()
    g.name = "HeroLocomotion"
    g.params.append("speed")

    var st_run = StateDef()
    st_run.name = "Run"
    st_run.clip = 1
    st_run.when.append(Condition(0, OP_GT, Float32(0.5)))
    g.states.append(st_run^)

    var st_idle = StateDef()
    st_idle.name = "Idle"
    st_idle.clip = 0
    g.states.append(st_idle^)
    g.transitions.append(TransitionDef(1, 0, Float32(0.05)))

    var flanim_bytes = save_flanim_bytes(g)
    var g_rt = load_flanim_bytes(flanim_bytes)

    var clips = List[ClipInfo]()
    clips.append(ClipInfo("Idle", Float32(1.0), True, Float32(1.0)))
    clips.append(ClipInfo("Run", Float32(0.5), True, Float32(1.0)))

    var rig_ws = tmp + "/rig_ws"
    _ = make_dirs(rig_ws + "/assets/rigs")
    _ = save_flrig_file(rig_ws + "/assets/rigs/hero.flrig", rig)

    var app = EngineApp(AppConfig(title="M1SkinnedPlayback", headless=True, worker_threads=0, fixed_hz=60.0))
    app.add_plugin(TransformPlugin())
    app.add_plugin(AssetPlugin(loose_root=rig_ws, enable_watcher=False))
    app.add_plugin(AnimPlugin())
    app.add_plugin(RenderPlugin(tier=RENDER_TIER_0_HEADLESS_SIMD, width=64, height=64))

    var e_char = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](e_char, LocalTransform(Vec3(0.0, 0.0, -4.0), Quat(), Vec3(1.0, 1.0, 1.0)))

    var rc = RigComponent()
    rc.rig_ref = AssetRef(AssetUuid.from_path("assets/rigs/hero.flrig"), "assets/rigs/hero.flrig", ASSET_KIND_RAW)
    _ = app.world.insert[RigComponent](e_char, rc^)

    var skel_c = SkeletonComponent(UInt64(101), SkeletonInstance(3), True)
    _ = app.world.insert[SkeletonComponent](e_char, skel_c^)

    var anim_c = AnimatorComponent(UInt64(201))
    _ = app.world.insert[AnimatorComponent](e_char, anim_c^)

    var mr_c = MeshRendererComponent(mesh_id=0, material_id=0, cast_shadows=True, visible=True)
    _ = app.world.insert[MeshRendererComponent](e_char, mr_c^)

    var e_weapon = app.world.spawn_empty()
    _ = app.world.insert[LocalTransform](e_weapon, LocalTransform.identity())
    var sock = BoneSocketAttachment("WeaponSocket", 2, -1, Vec3(0.0, 0.1, 0.25))
    _ = app.world.insert[BoneSocketComponent](e_weapon, BoneSocketComponent(e_char, sock^))

    app.startup()
    if app.resources.has[AnimAssetCache]():
        var acache_ptr = app.resources.get_ptr_mut[AnimAssetCache]().value()
        acache_ptr[unsafe_offset=0].register_model(UInt64(101), _build_rigged_native_model())
        acache_ptr[unsafe_offset=0].register_graph(UInt64(201), g_rt^, clips^)

    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))

    var state_idle = String()
    var z_idle = Float32(0.0)
    var bone_mat_count = 0
    var rig_bound_ok = False
    if app.world.has[RigComponent](e_char):
        var rptr = app.world.get_ptr[RigComponent](e_char).value()
        rig_bound_ok = (
            rptr[].rig_ref.rel_path == "assets/rigs/hero.flrig"
            and rptr[].bound
            and rptr[].joint_count == 3
        )
    if app.world.has[AnimatorComponent](e_char):
        state_idle = app.world.get_ptr[AnimatorComponent](e_char).value()[].animator.state_name()
    if app.world.has[SkeletonComponent](e_char):
        bone_mat_count = len(app.world.get_ptr[SkeletonComponent](e_char).value()[].skeleton.world_matrices)
    if app.world.has[GlobalTransform](e_weapon):
        z_idle = app.world.get_ptr[GlobalTransform](e_weapon).value()[].translation.z

    # Trigger transition Idle -> Run
    if app.world.has[AnimatorComponent](e_char):
        var aptr = app.world.get_ptr_mut[AnimatorComponent](e_char).value()
        var p_speed = aptr[].animator.param("speed")
        aptr[].animator.set(p_speed, Float32(2.0))

    for _ in range(10):
        app.step_one_frame(Float32(1.0 / 60.0))

    var state_run = String()
    var z_run = Float32(0.0)
    if app.world.has[AnimatorComponent](e_char):
        state_run = app.world.get_ptr[AnimatorComponent](e_char).value()[].animator.state_name()
    if app.world.has[GlobalTransform](e_weapon):
        z_run = app.world.get_ptr[GlobalTransform](e_weapon).value()[].translation.z

    var render_skin_ok = False
    if app.resources.has[RenderPluginStateResource]() and app.resources.has[RenderWorld]():
        var rps = app.resources.get_ptr[RenderPluginStateResource]().value()
        var rw = app.resources.get_ptr[RenderWorld]().value()
        render_skin_ok = (
            rps[].rendered_frames >= 13
            and rps[].last_render_ok
            and rps[].skinned_sync_count >= 1
            and rw[].instance_count() >= 1
        )

    app.shutdown()
    var skin_ok = (
        rig_bound_ok
        and render_skin_ok
        and state_idle == "Idle"
        and state_run == "Run"
        and bone_mat_count == 3
        and abs(z_run - z_idle) > Float32(0.1)
    )
    _ = runner.check(
        "t1_m1_skinned_playback_state_transition_and_socket_sync",
        skin_ok,
        "rig_bound=" + String(rig_bound_ok) + " render_skin=" + String(render_skin_ok) + " idle=" + state_idle + " run=" + state_run + " bones=" + String(bone_mat_count) + " dz=" + String(z_run - z_idle),
    )

    # -----------------------------------------------------------------------
    # 3. F20: Standalone .flmat Shared Material Assets (PBR/Cel/VFX + Texture AssetRefs)
    # -----------------------------------------------------------------------
    var mat_pbr = FlmatAsset("PbrGold")
    mat_pbr.shading_mode = MAT_SHADING_PBR
    mat_pbr.base_color = Vec3(0.95, 0.78, 0.28)
    mat_pbr.roughness = Float32(0.22)
    mat_pbr.metallic = Float32(0.95)
    mat_pbr.clearcoat = Float32(0.4)
    mat_pbr.albedo_tex = AssetRef(AssetUuid.from_path("assets/textures/gold_albedo.png"), "assets/textures/gold_albedo.png", ASSET_KIND_TEXTURE)
    mat_pbr.normal_tex = AssetRef(AssetUuid.from_path("assets/textures/gold_normal.png"), "assets/textures/gold_normal.png", ASSET_KIND_TEXTURE)
    mat_pbr.orm_tex = AssetRef(AssetUuid.from_path("assets/textures/gold_orm.png"), "assets/textures/gold_orm.png", ASSET_KIND_TEXTURE)
    mat_pbr.emissive_tex = AssetRef(AssetUuid.from_path("assets/textures/gold_emit.png"), "assets/textures/gold_emit.png", ASSET_KIND_TEXTURE)

    var mat_cel = FlmatAsset("ToonHero")
    mat_cel.shading_mode = MAT_SHADING_NPR_CEL
    mat_cel.cel_params = AaaCelParams()
    mat_cel.cel_params.shadow_threshold = Float32(0.42)
    mat_cel.cel_params.shadow_feather = Float32(0.025)
    mat_cel.cel_params.outline_width_px = Float32(3.0)
    mat_cel.cel_params.inner_ink_strength = Float32(0.85)

    var mat_vfx = FlmatAsset("EnergyShield")
    mat_vfx.shading_mode = MAT_SHADING_STYLIZED_VFX
    mat_vfx.vfx_params = AaaStylizedVfxParams()
    mat_vfx.opacity = Float32(0.65)

    var pbr_path = tmp + "/gold.flmat"
    var cel_path = tmp + "/toon.flmat"
    var vfx_path = tmp + "/shield.flmat"
    var s1 = save_flmat_file(pbr_path, mat_pbr)
    var s2 = save_flmat_file(cel_path, mat_cel)
    var s3 = save_flmat_file(vfx_path, mat_vfx)

    var pbr_rt = load_flmat_file(pbr_path)
    var cel_rt = load_flmat_file(cel_path)
    var vfx_rt = load_flmat_file(vfx_path)

    var flmat_ok = (
        s1 and s2 and s3
        and pbr_rt.shading_mode == MAT_SHADING_PBR
        and abs(pbr_rt.metallic - Float32(0.95)) < Float32(0.001)
        and abs(pbr_rt.clearcoat - Float32(0.4)) < Float32(0.001)
        and pbr_rt.albedo_tex.rel_path == "assets/textures/gold_albedo.png"
        and pbr_rt.normal_tex.rel_path == "assets/textures/gold_normal.png"
        and pbr_rt.orm_tex.rel_path == "assets/textures/gold_orm.png"
        and pbr_rt.emissive_tex.rel_path == "assets/textures/gold_emit.png"
        and cel_rt.shading_mode == MAT_SHADING_NPR_CEL
        and abs(cel_rt.cel_params.shadow_threshold - Float32(0.42)) < Float32(0.001)
        and abs(cel_rt.cel_params.shadow_feather - Float32(0.025)) < Float32(0.001)
        and abs(cel_rt.cel_params.outline_width_px - Float32(3.0)) < Float32(0.001)
        and abs(cel_rt.cel_params.inner_ink_strength - Float32(0.85)) < Float32(0.001)
        and vfx_rt.shading_mode == MAT_SHADING_STYLIZED_VFX
        and abs(vfx_rt.opacity - Float32(0.65)) < Float32(0.001)
    )
    _ = runner.check("t1_m1_flmat_pbr_cel_vfx_and_texture_refs_roundtrip", flmat_ok, ".flmat save/load roundtrip failed")

    # -----------------------------------------------------------------------
    # 4. F20: Standalone .flprefab (save_flprefab/load_flprefab, Nested + Sparse Override)
    # -----------------------------------------------------------------------
    var reg = ComponentRegistry.with_builtins()
    register_render_codecs(reg)
    register_anim_codecs(reg)

    var child_pf = Prefab("TurretBarrel")
    var c_slot = child_pf.add_node("Barrel")
    child_pf.set_component(c_slot, LocalTransform(Vec3(0.0, 0.5, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    var barrel_mat = MaterialOverrideComponent()
    barrel_mat.material_ref = AssetRef(AssetUuid.from_path("assets/materials/steel.flmat"), "assets/materials/steel.flmat", ASSET_KIND_RAW)
    barrel_mat.base_color = Vec3(0.4, 0.4, 0.45)
    child_pf.set_component(c_slot, barrel_mat)

    var parent_pf = Prefab("TurretBase")
    var p_slot = parent_pf.add_node("Base")
    parent_pf.set_component(p_slot, LocalTransform(Vec3(10.0, 0.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    var nested_ref = NestedPrefabRef("TurretBarrel", p_slot, LocalTransform(Vec3(0.0, 1.0, 0.0), Quat(), Vec3(1.0, 1.0, 1.0)))
    parent_pf.add_nested(nested_ref^)

    var child_pf_path = tmp + "/turret_barrel.flprefab"
    var parent_pf_path = tmp + "/turret_base.flprefab"
    var spf1 = save_flprefab_file(child_pf_path, child_pf)
    var spf2 = save_flprefab_file(parent_pf_path, parent_pf)

    var child_rt = load_flprefab_file(child_pf_path)
    var parent_rt = load_flprefab_file(parent_pf_path)

    var lib = List[Prefab]()
    lib.append(child_rt^)
    var inst_overrides = List[PrefabOverride]()
    var gold_ov = MaterialOverrideComponent()
    gold_ov.material_ref = AssetRef(AssetUuid.from_path("assets/materials/gold.flmat"), "assets/materials/gold.flmat", ASSET_KIND_RAW)
    gold_ov.base_color = Vec3(0.95, 0.78, 0.28)
    inst_overrides.append(PrefabOverride.for_name("Barrel", gold_ov))

    var w_pf = World()
    var spawned_pf = parent_rt.instantiate(w_pf, reg, lib, inst_overrides)

    var scn_pf = Scene("PrefabRoundTrip")
    var scn_pf_str = scn_pf.save_flscn_string(w_pf, reg)
    var w_pf_reload = World()
    var bytes_pf = List[UInt8]()
    for b in scn_pf_str.as_bytes():
        bytes_pf.append(b)
    var reloaded_pf = scn_pf.load_flscn_bytes(w_pf_reload, reg, bytes_pf)

    var pf_ok = spf1 and spf2 and len(spawned_pf) == 2 and len(reloaded_pf) == 2
    if pf_ok:
        var e_barrel = spawned_pf[1]
        var e_barrel_rt = reloaded_pf[1]
        pf_ok = (
            w_pf.has[Parent](e_barrel)
            and w_pf.get_ptr[LocalTransform](e_barrel).value()[].translation.y == Float32(1.5)
            and w_pf.get_ptr[MaterialOverrideComponent](e_barrel).value()[].material_ref.rel_path == "assets/materials/gold.flmat"
            and w_pf_reload.get_ptr[LocalTransform](e_barrel_rt).value()[].translation.y == Float32(1.5)
            and w_pf_reload.get_ptr[MaterialOverrideComponent](e_barrel_rt).value()[].material_ref.rel_path == "assets/materials/gold.flmat"
        )
    _ = runner.check("t1_m1_flprefab_nested_and_sparse_override_roundtrip", pf_ok, ".flprefab save/load/nested/override failed")

    rm_rf_tmp(tmp)
    runner.finish()

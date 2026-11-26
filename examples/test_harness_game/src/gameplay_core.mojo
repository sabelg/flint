"""Core Pure-Mojo 1.2 gameplay, SimdCanvas cooked-mesh rendering, and SDRP checksum logic."""
from std.collections import Dict, List
from std.math import abs, max, min
from std.memory import bitcast
from std.sys import size_of
from flint.ai.behavior_tree import (
    BT_RUNNING,
    BT_SUCCESS,
    BehaviorTreeComponent,
)
from flint.anim.flanim_ecs import load_flanim_bytes
from flint.anim.flrig import FrigRigDef, load_flrig_bytes, validate_rig_bindings
from flint.anim.plugin import (
    AnimAssetCache,
    AnimatorComponent,
    BoneSocketComponent,
    RigComponent,
    SkeletonComponent,
)
from flint.anim.pose import SkeletonInstance
from flint.anim.sync import ClipInfo
from flint.app import (
    EngineApp,
    FixedTime,
    FrameTime,
    Resource,
    Resources,
    STAGE_POST_UPDATE,
    STAGE_PRE_UPDATE,
    STAGE_UPDATE,
)
from flint.app.flow import GameFlow
from flint.app.hot_reload import HotReloadHost, PodSchemaRegistryResource
from flint.app.window_plugin import WindowStateResource
from flint.assets.asset_ref import AssetRef, normalize_asset_rel_path
from flint.assets.flmat import FlmatAsset, load_flmat_bytes
from flint.assets.flpk_v3 import Handle, decode_cooked_rig_slab
from flint.assets.glb_native import parse_native_glb
from flint.assets.json import JsonDocument, parse_json
from flint.assets.meshlet import AAA_VERTEX_WORDS
from flint.assets.model import (
    NativeAnimation,
    NativeChannel,
    NativeMesh,
    NativeModel,
    NativeNode,
    NativeSkin,
)
from flint.assets.server import AssetServer, CookedMeshAsset
from flint.audio.plugin import AudioEmitterComponent, AudioListenerComponent
from flint.audio.server import AudioServer, BUS_SFX
from flint.core.jobs import JobPool
from flint.core.math import Quat, Vec3
from flint.core.replay import (
    CHECK_SECTIONS,
    Fnv64,
    SECTION_BODIES,
    SECTION_CAMERA,
    SECTION_ENEMIES,
    SECTION_FLOW,
    SECTION_OBJECTS,
    SECTION_PLAYER,
    SECTION_PROGRESSION,
    SECTION_TERRAIN,
)
from flint.ecs import Entity, NULL_ENTITY, ParallelCommands, World, comp_hash64
from flint.input.input_map import ACTION_AXIS1D, ACTION_BOOL, InputContext, InputMap
from flint.nav.plugin import NavAgentComponent, NavMeshComponent, NavPluginStateResource
from flint.physics.plugin import (
    CharacterControllerComponent,
    ColliderComponent,
    OnEnter,
    RigidBodyComponent,
    TriggerEventsResource,
    entity_from_body,
)
from flint.physics.world import PhysicsWorld
from flint.platform.os import file_size, read_bytes
from flint.render.plugin import (
    CameraComponent,
    EnvironmentComponent,
    MaterialOverrideComponent,
    MeshRendererComponent,
    PointLightComponent,
    SpotLightComponent,
)
from flint.render.simd_canvas import Camera3D, SimdCanvas, pack_argb
from flint.scene import GlobalTransform, LocalTransform, Parent, UnknownComponentsComponent
from flint.ui.context import UiContext
from flint.ui.draw_list import PRIM_SHAPE, UI_FLAG_BORDER
from flint.ui.hud import game_hud
from gameplay_shared import (
    HARNESS_ATTACK_DAMAGE,
    HARNESS_ATTACK_RANGE,
    HARNESS_GOAL_SCORE,
    HARNESS_HIT_SCORE,
    HARNESS_KILL_SCORE,
    HARNESS_PICKUP_SCORE,
    HARNESS_PLAYER_SPEED,
    HarnessStats,
    build_harness_stats_schema,
)


def _norm_asset_key(raw: String) -> String:
    var res = normalize_asset_rel_path(raw)
    if res[0]:
        return res[1]
    return raw


def _parse_rig_bytes(raw: List[UInt8]) raises -> FrigRigDef:
    var slab = decode_cooked_rig_slab(raw)
    if slab[0] and len(slab[1]) > 0:
        return load_flrig_bytes(slab[1])
    return load_flrig_bytes(raw)


def _compute_skeleton_topo_order(mut si: SkeletonInstance):
    var n = si.node_count
    si.topo_order.clear()
    var visited = List[Bool](length=n, fill=False)
    for i in range(n):
        if si.parents[i] < 0 or si.parents[i] >= n:
            visited[i] = True
            si.topo_order.append(i)
    var progress = True
    while progress and len(si.topo_order) < n:
        progress = False
        for i in range(n):
            if not visited[i]:
                var p = si.parents[i]
                if p >= 0 and p < n and visited[p]:
                    visited[i] = True
                    si.topo_order.append(i)
                    progress = True
    for i in range(n):
        if not visited[i]:
            si.topo_order.append(i)


def _build_fallback_hero_model_from_rig(rig: FrigRigDef) -> NativeModel:
    var model = NativeModel()
    var nj = len(rig.joints)
    for ji in range(nj):
        var nd = NativeNode()
        nd.name = rig.joints[ji].name
        nd.translation = rig.joints[ji].translation
        nd.base_translation = rig.joints[ji].translation
        nd.rotation = rig.joints[ji].rotation
        nd.base_rotation = rig.joints[ji].rotation
        nd.scale = rig.joints[ji].scale
        nd.base_scale = rig.joints[ji].scale
        if ji == 0:
            nd.mesh = 0
        for ci in range(nj):
            if rig.joints[ci].parent_index == ji:
                nd.children.append(ci)
        model.nodes.append(nd^)
    if nj > 0:
        model.roots.append(0)
        var skin = NativeSkin()
        for ji in range(nj):
            skin.joints.append(ji)
            for r in range(4):
                for c in range(4):
                    skin.inverse_bind.append(Float32(1.0) if r == c else Float32(0.0))
        model.skins.append(skin^)
    var idle_anim = NativeAnimation()
    idle_anim.name = "Idle"
    idle_anim.duration = Float32(1.0)
    if nj >= 3:
        var ch = NativeChannel()
        ch.node = 2
        ch.path = 0
        ch.times.append(Float32(0.0))
        ch.times.append(Float32(0.5))
        ch.times.append(Float32(1.0))
        var base_t = rig.joints[2].translation
        ch.values.append(base_t.x)
        ch.values.append(base_t.y)
        ch.values.append(base_t.z)
        ch.values.append(base_t.x)
        ch.values.append(base_t.y + Float32(0.06))
        ch.values.append(base_t.z)
        ch.values.append(base_t.x)
        ch.values.append(base_t.y)
        ch.values.append(base_t.z)
        idle_anim.channels.append(ch^)
    model.animations.append(idle_anim^)
    var run_anim = NativeAnimation()
    run_anim.name = "Run"
    run_anim.duration = Float32(0.6)
    if nj >= 2:
        var ch2 = NativeChannel()
        ch2.node = 1
        ch2.path = 0
        ch2.times.append(Float32(0.0))
        ch2.times.append(Float32(0.3))
        ch2.times.append(Float32(0.6))
        var sp_t = rig.joints[1].translation
        ch2.values.append(sp_t.x)
        ch2.values.append(sp_t.y)
        ch2.values.append(sp_t.z)
        ch2.values.append(sp_t.x)
        ch2.values.append(sp_t.y + Float32(0.08))
        ch2.values.append(sp_t.z)
        ch2.values.append(sp_t.x)
        ch2.values.append(sp_t.y)
        ch2.values.append(sp_t.z)
        run_anim.channels.append(ch2^)
    model.animations.append(run_anim^)
    return model^


@always_inline
def _hash_str_u64(s: String) -> UInt64:
    var h = UInt64(14695981039346656037)
    var b = s.as_bytes()
    for i in range(len(b)):
        h = (h ^ UInt64(b[i])) * UInt64(1099511628211)
    return h if h != 0 else UInt64(1)


struct HarnessStateResource(Resource, Movable):
    """Runtime gameplay, render cache, and verification state for `test_harness_game`."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("test_harness_game.HarnessStateResource")
    var headless: Bool
    var canvas: SimdCanvas
    var ui_ctx: UiContext
    var hot_reload_host: HotReloadHost
    var mesh_handles: Dict[String, Handle[CookedMeshAsset]]
    var flmat_colors: Dict[String, Vec3]
    var flmat_emissive: Dict[String, Vec3]
    var player_e: Entity
    var enemy_e: Entity
    var pickup_e: Entity
    var goal_e: Entity
    var floor_e: Entity
    var weapon_e: Entity
    var cam_e: Entity
    var player_hp: Int
    var player_max_hp: Int
    var score: Int
    var pickup_collected: Bool
    var enemy_hp: Int
    var enemy_hits: Int
    var goal_reached: Bool
    var trigger_cursor: Int
    var last_trigger_count: Int
    var last_nav_path_found: Bool
    var last_nav_waypoints: Int
    var nav_path_verified: Bool
    var audio_played_count: Int
    var kcc_synced: Bool
    var stats_tid: Int

    def __init__(out self, headless: Bool, width: Int = 1280, height: Int = 720):
        self.headless = headless
        self.canvas = SimdCanvas(width, height)
        self.ui_ctx = UiContext(preset_hud=True)
        self.ui_ctx.screen_w = Float32(width)
        self.ui_ctx.screen_h = Float32(height)
        self.hot_reload_host = HotReloadHost()
        self.mesh_handles = Dict[String, Handle[CookedMeshAsset]]()
        self.flmat_colors = Dict[String, Vec3]()
        self.flmat_emissive = Dict[String, Vec3]()
        self.player_e = NULL_ENTITY
        self.enemy_e = NULL_ENTITY
        self.pickup_e = NULL_ENTITY
        self.goal_e = NULL_ENTITY
        self.floor_e = NULL_ENTITY
        self.weapon_e = NULL_ENTITY
        self.cam_e = NULL_ENTITY
        self.player_hp = 100
        self.player_max_hp = 100
        self.score = 0
        self.pickup_collected = False
        self.enemy_hp = 30
        self.enemy_hits = 0
        self.goal_reached = False
        self.trigger_cursor = 0
        self.last_trigger_count = 0
        self.last_nav_path_found = False
        self.last_nav_waypoints = 0
        self.nav_path_verified = False
        self.audio_played_count = 0
        self.kcc_synced = False
        self.stats_tid = -1


def _studio_entity_name(world: World, e: Entity) -> String:
    var unk_opt = world.get_ptr[UnknownComponentsComponent](e)
    if not unk_opt:
        return String("")
    ref unk = unk_opt.value()[unsafe_offset=0]
    for i in range(len(unk.json_names)):
        if unk.json_names[i] == "StudioEntityMetaComponent":
            var payload = unk.json_payloads[i]
            var bytes = List[UInt8](capacity=payload.byte_length())
            for b in payload.as_bytes():
                bytes.append(b)
            var doc = parse_json(bytes)
            if doc.valid and doc.root >= 0:
                return doc.text(doc.object_get(doc.root, "name"), "")
    return String("")


def resolve_harness_entities(mut world: World, mut state: HarnessStateResource):
    state.player_e = NULL_ENTITY
    state.enemy_e = NULL_ENTITY
    state.pickup_e = NULL_ENTITY
    state.goal_e = NULL_ENTITY
    state.floor_e = NULL_ENTITY
    state.weapon_e = NULL_ENTITY
    state.cam_e = NULL_ENTITY

    for i in range(len(world.entities)):
        if not world.entities[i].alive:
            continue
        var e = Entity(UInt32(i), world.entities[i].generation)
        var ename = _studio_entity_name(world, e)
        if ename == "Player":
            state.player_e = e
        elif ename == "Enemy":
            state.enemy_e = e
        elif ename == "Pickup":
            state.pickup_e = e
        elif ename == "Goal":
            state.goal_e = e
        elif ename == "Floor":
            state.floor_e = e
        elif ename == "Weapon":
            state.weapon_e = e
        elif ename == "MainCamera":
            state.cam_e = e

    # Component-based fallbacks if names differ
    if not state.player_e.valid():
        var kcc_ents = world.query_entities(world.bit_for[CharacterControllerComponent]())
        if len(kcc_ents) > 0:
            state.player_e = kcc_ents[0]
    if not state.enemy_e.valid():
        var bt_ents = world.query_entities(world.bit_for[BehaviorTreeComponent]())
        if len(bt_ents) > 0:
            state.enemy_e = bt_ents[0]
    if not state.floor_e.valid():
        var nm_ents = world.query_entities(world.bit_for[NavMeshComponent]())
        if len(nm_ents) > 0:
            state.floor_e = nm_ents[0]
    if not state.weapon_e.valid():
        var bs_ents = world.query_entities(world.bit_for[BoneSocketComponent]())
        if len(bs_ents) > 0:
            state.weapon_e = bs_ents[0]
    if not state.cam_e.valid():
        var cam_ents = world.query_entities(world.bit_for[CameraComponent]())
        if len(cam_ents) > 0:
            state.cam_e = cam_ents[0]
    if not state.pickup_e.valid() or not state.goal_e.valid():
        var col_ents = world.query_entities(world.bit_for[ColliderComponent]())
        for ci in range(len(col_ents)):
            var ce = col_ents[ci]
            var col_opt = world.get_ptr[ColliderComponent](ce)
            if not col_opt or not col_opt.value()[unsafe_offset=0].is_trigger:
                continue
            if world.has[AudioEmitterComponent](ce) and not state.pickup_e.valid():
                state.pickup_e = ce
            elif ce != state.pickup_e and not state.goal_e.valid():
                state.goal_e = ce

    # Sync CharacterControllerComponent.controller.pos to LocalTransform.translation once on load
    if state.player_e.valid() and not state.kcc_synced:
        var kc_opt = world.get_ptr_mut[CharacterControllerComponent](state.player_e)
        var lt_opt = world.get_ptr[LocalTransform](state.player_e)
        if kc_opt and lt_opt:
            kc_opt.value()[unsafe_offset=0].controller.pos = lt_opt.value()[unsafe_offset=0].translation
            state.kcc_synced = True

    # Ensure HarnessStats POD component is registered & attached to player_e
    var stats_schema = build_harness_stats_schema()
    var tid = world.register_pod_component(
        stats_schema.component_hash, stats_schema.byte_size, stats_schema.alignment, stats_schema.storage_mode
    )
    state.stats_tid = tid
    if tid >= 0 and state.player_e.valid() and not world.has_by_type_id(state.player_e, tid):
        if world.insert_raw_pod(state.player_e, tid, 0):
            var dst_ptr = world.get_raw_ptr_mut(state.player_e, tid)
            if dst_ptr != 0:
                Pointer[HarnessStats, MutAnyOrigin](unsafe_from_address=dst_ptr)[unsafe_offset=0] = HarnessStats()


def resolve_harness_assets(
    mut world: World,
    mut resources: Resources,
    mut state: HarnessStateResource,
):
    var assets_opt = resources.get_ptr_mut[AssetServer]()
    if not assets_opt:
        return
    ref assets = assets_opt.value()[unsafe_offset=0]

    # 1. Meshes & .flmat materials
    var mr_ents = world.query_entities(world.bit_for[MeshRendererComponent]())
    for i in range(len(mr_ents)):
        var e = mr_ents[i]
        var mr_opt = world.get_ptr[MeshRendererComponent](e)
        if not mr_opt:
            continue
        ref mr = mr_opt.value()[unsafe_offset=0]
        if not mr.mesh_ref.is_empty():
            var key = _norm_asset_key(mr.mesh_ref.rel_path)
            if not state.mesh_handles.get(key):
                var mh = assets.resolve_mesh(mr.mesh_ref)
                if assets.meshes.is_loaded(mh):
                    state.mesh_handles[key] = mh
        if not mr.material_ref.is_empty():
            var mkey = _norm_asset_key(mr.material_ref.rel_path)
            if not state.flmat_colors.get(mkey):
                var rh = assets.resolve_raw(mr.material_ref)
                if assets.raws.is_loaded(rh):
                    var rptr = assets.raws.get_ptr(rh)
                    if rptr:
                        try:
                            var flm = load_flmat_bytes(rptr.value()[unsafe_offset=0].bytes)
                            state.flmat_colors[mkey] = flm.base_color
                            state.flmat_emissive[mkey] = flm.emissive
                        except:
                            pass

    var mo_ents = world.query_entities(world.bit_for[MaterialOverrideComponent]())
    for i in range(len(mo_ents)):
        var e = mo_ents[i]
        var mo_opt = world.get_ptr_mut[MaterialOverrideComponent](e)
        if not mo_opt:
            continue
        ref mo = mo_opt.value()[unsafe_offset=0]
        if not mo.material_ref.is_empty():
            var mkey = _norm_asset_key(mo.material_ref.rel_path)
            if not state.flmat_colors.get(mkey):
                var rh = assets.resolve_raw(mo.material_ref)
                if assets.raws.is_loaded(rh):
                    var rptr = assets.raws.get_ptr(rh)
                    if rptr:
                        try:
                            var flm = load_flmat_bytes(rptr.value()[unsafe_offset=0].bytes)
                            state.flmat_colors[mkey] = flm.base_color
                            state.flmat_emissive[mkey] = flm.emissive
                        except:
                            pass
            var col_hit = state.flmat_colors.get(mkey)
            if col_hit:
                mo.base_color = col_hit.value()
            var em_hit = state.flmat_emissive.get(mkey)
            if em_hit:
                mo.emissive = em_hit.value()

    # 2. RigComponent & SkeletonComponent & AnimatorComponent -> AnimAssetCache
    var anim_cache_opt = resources.get_ptr_mut[AnimAssetCache]()
    var loaded_rigs = List[FrigRigDef]()
    var rig_ents = world.query_entities(world.bit_for[RigComponent]())
    for i in range(len(rig_ents)):
        var rc_opt = world.get_ptr_mut[RigComponent](rig_ents[i])
        if not rc_opt or rc_opt.value()[unsafe_offset=0].rig_ref.is_empty():
            continue
        var rh = assets.resolve_raw(rc_opt.value()[unsafe_offset=0].rig_ref)
        if assets.raws.is_loaded(rh):
            var bptr = assets.raws.get_ptr(rh)
            if bptr:
                try:
                    var rdef = _parse_rig_bytes(bptr.value()[unsafe_offset=0].bytes)
                    validate_rig_bindings(rdef)
                    rc_opt.value()[unsafe_offset=0].joint_count = len(rdef.joints)
                    rc_opt.value()[unsafe_offset=0].bound = True
                    loaded_rigs.append(rdef^)
                except:
                    pass

    var sk_ents = world.query_entities(world.bit_for[SkeletonComponent]())
    for i in range(len(sk_ents)):
        var e = sk_ents[i]
        var sk_opt = world.get_ptr_mut[SkeletonComponent](e)
        if not sk_opt:
            continue
        ref sk = sk_opt.value()[unsafe_offset=0]
        var resolved_rig: Optional[FrigRigDef] = None
        if not sk.rig_ref.is_empty():
            var rh = assets.resolve_raw(sk.rig_ref)
            if assets.raws.is_loaded(rh):
                var bptr = assets.raws.get_ptr(rh)
                if bptr:
                    try:
                        var rdef = _parse_rig_bytes(bptr.value()[unsafe_offset=0].bytes)
                        validate_rig_bindings(rdef)
                        resolved_rig = rdef^
                    except:
                        pass
        if not resolved_rig and len(loaded_rigs) > 0:
            resolved_rig = loaded_rigs[0].copy()
        if resolved_rig:
            ref rdef = resolved_rig.value()
            var nj = len(rdef.joints)
            var si = SkeletonInstance(nj)
            for ji in range(nj):
                si.parents[ji] = rdef.joints[ji].parent_index
                si.pose.t[ji] = rdef.joints[ji].translation
                si.pose.r[ji] = rdef.joints[ji].rotation
                si.pose.s[ji] = rdef.joints[ji].scale
            _compute_skeleton_topo_order(si)
            sk.skeleton = si^
            sk.dirty_topology = False
            if sk.model_id == 0:
                sk.model_id = _hash_str_u64(
                    _norm_asset_key(sk.rig_ref.rel_path) if not sk.rig_ref.is_empty() else String("assets/chars/hero.flrig")
                )
            if anim_cache_opt and anim_cache_opt.value()[unsafe_offset=0].find_model(sk.model_id) < 0:
                var mdl_loaded = False
                var mr_opt = world.get_ptr[MeshRendererComponent](e)
                if mr_opt and not mr_opt.value()[unsafe_offset=0].mesh_ref.is_empty():
                    var disk = assets._resolve_loose_disk_path(_norm_asset_key(mr_opt.value()[unsafe_offset=0].mesh_ref.rel_path))
                    if disk.byte_length() > 0 and file_size(disk) > 0:
                        var glb_raw = read_bytes(disk)
                        if len(glb_raw) > 0:
                            var mdl = parse_native_glb(glb_raw)
                            if len(mdl.nodes) > 0:
                                anim_cache_opt.value()[unsafe_offset=0].register_model(sk.model_id, mdl^)
                                mdl_loaded = True
                if not mdl_loaded:
                    var fb_mdl = _build_fallback_hero_model_from_rig(rdef)
                    anim_cache_opt.value()[unsafe_offset=0].register_model(sk.model_id, fb_mdl^)

    var an_ents = world.query_entities(world.bit_for[AnimatorComponent]())
    for i in range(len(an_ents)):
        var e = an_ents[i]
        var an_opt = world.get_ptr_mut[AnimatorComponent](e)
        if not an_opt or an_opt.value()[unsafe_offset=0].graph_ref.is_empty():
            continue
        ref ac = an_opt.value()[unsafe_offset=0]
        var ah = assets.resolve_raw(ac.graph_ref)
        if assets.raws.is_loaded(ah):
            var bptr = assets.raws.get_ptr(ah)
            if bptr:
                try:
                    var gdef = load_flanim_bytes(bptr.value()[unsafe_offset=0].bytes)
                    var gid = _hash_str_u64(_norm_asset_key(ac.graph_ref.rel_path))
                    if anim_cache_opt:
                        var clips = List[ClipInfo]()
                        var sk_opt = world.get_ptr[SkeletonComponent](e)
                        if sk_opt:
                            var mi = anim_cache_opt.value()[unsafe_offset=0].find_model(sk_opt.value()[unsafe_offset=0].model_id)
                            if mi >= 0:
                                ref mdl = anim_cache_opt.value()[unsafe_offset=0].models[mi]
                                for ci in range(len(mdl.animations)):
                                    clips.append(
                                        ClipInfo(mdl.animations[ci].name, max(Float32(0.1), mdl.animations[ci].duration), True, Float32(1.0))
                                    )
                        if len(clips) == 0:
                            for si in range(len(gdef.states)):
                                clips.append(ClipInfo(gdef.states[si].name, Float32(1.0), True, Float32(1.0)))
                        anim_cache_opt.value()[unsafe_offset=0].register_graph(gid, gdef^, clips^)
                    ac.graph_id = gid
                except:
                    pass

    # 3. AudioEmitterComponent (.flpk-first PCM16 audio -> AudioServer)
    var audio_srv_opt = resources.get_ptr_mut[AudioServer]()
    var au_ents = world.query_entities(world.bit_for[AudioEmitterComponent]())
    for i in range(len(au_ents)):
        var au_opt = world.get_ptr_mut[AudioEmitterComponent](au_ents[i])
        if not au_opt or au_opt.value()[unsafe_offset=0].audio_ref.is_empty():
            continue
        ref au = au_opt.value()[unsafe_offset=0]
        var auh = assets.resolve_audio(au.audio_ref)
        if assets.audios.is_loaded(auh):
            var dptr = assets.audios.get_ptr(auh)
            if dptr and audio_srv_opt:
                au.asset_id = audio_srv_opt.value()[unsafe_offset=0].register_pcm16_asset(
                    dptr.value()[unsafe_offset=0].samples,
                    dptr.value()[unsafe_offset=0].channels,
                    dptr.value()[unsafe_offset=0].sample_rate,
                )


def setup_harness_input(mut app: EngineApp):
    var imap_opt = app.resources.get_ptr_mut[InputMap]()
    if not imap_opt:
        return
    ref imap = imap_opt.value()[unsafe_offset=0]
    _ = imap.register_action("move_x", ACTION_AXIS1D)
    _ = imap.register_action("move_z", ACTION_AXIS1D)
    _ = imap.register_action("attack", ACTION_BOOL)
    _ = imap.register_action("quit", ACTION_BOOL)
    var ctx = InputContext("gameplay", priority=10)
    ctx.bind_key("move_x", 68, scale_x=Float32(1.0))    # D
    ctx.bind_key("move_x", 262, scale_x=Float32(1.0))   # Right
    ctx.bind_key("move_x", 65, scale_x=Float32(-1.0))   # A
    ctx.bind_key("move_x", 263, scale_x=Float32(-1.0))  # Left
    ctx.bind_key("move_z", 83, scale_x=Float32(1.0))    # S
    ctx.bind_key("move_z", 264, scale_x=Float32(1.0))   # Down
    ctx.bind_key("move_z", 87, scale_x=Float32(-1.0))   # W
    ctx.bind_key("move_z", 265, scale_x=Float32(-1.0))  # Up
    ctx.bind_key("attack", 32)   # Space
    ctx.bind_key("quit", 256)    # Esc
    imap.add_context(ctx^)
    imap.push_context("gameplay")


def _sys_harness_pre_input(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[HarnessStateResource]()
    if not st_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]

    var dt = Float32(1.0 / 60.0)
    var fix_opt = resources.get_ptr[FixedTime]()
    if fix_opt and fix_opt.value()[unsafe_offset=0].dt > Float32(1.0e-5):
        dt = fix_opt.value()[unsafe_offset=0].dt

    if state.hot_reload_host.is_loaded():
        _ = state.hot_reload_host.step(world, resources, dt)

    var in_play = True
    var flow_opt = resources.get_ptr[GameFlow]()
    if flow_opt:
        in_play = flow_opt.value()[unsafe_offset=0].fsm.current_state() == "Play"

    var imap_opt = resources.get_ptr[InputMap]()
    if not state.headless and imap_opt and imap_opt.value()[unsafe_offset=0].just_pressed("quit"):
        var win_opt = resources.get_ptr_mut[WindowStateResource]()
        if win_opt:
            win_opt.value()[unsafe_offset=0].request_close()
        return

    if state.player_e.valid():
        var kc_opt = world.get_ptr_mut[CharacterControllerComponent](state.player_e)
        if kc_opt:
            ref kc = kc_opt.value()[unsafe_offset=0]
            if not in_play or not imap_opt:
                kc.desired_vel = Vec3()
            else:
                ref imap = imap_opt.value()[unsafe_offset=0]
                var mx = imap.axis_1d("move_x")
                var mz = imap.axis_1d("move_z")
                var len2 = mx * mx + mz * mz
                if len2 > Float32(1.0):
                    var inv_l = Float32(1.0) / (Vec3(mx, Float32(0.0), mz).length())
                    mx *= inv_l
                    mz *= inv_l
                kc.desired_vel = Vec3(mx * HARNESS_PLAYER_SPEED, Float32(0.0), mz * HARNESS_PLAYER_SPEED)


def _sys_harness_update(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[HarnessStateResource]()
    if not st_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]

    # 1. Drain trigger events
    var trig_opt = resources.get_ptr[TriggerEventsResource]()
    state.last_trigger_count = 0
    if trig_opt:
        var tevs = trig_opt.value()[unsafe_offset=0].events.read_since(state.trigger_cursor)
        state.last_trigger_count = len(tevs)
        for ti in range(len(tevs)):
            ref ev = tevs[ti]
            var actor = entity_from_body(ev.body_id, resources)
            if actor != state.player_e:
                continue
            if (
                not state.pickup_collected
                and state.pickup_e.valid()
                and OnEnter(ev, state.pickup_e, world, resources)
            ):
                state.pickup_collected = True
                state.score += HARNESS_PICKUP_SCORE
                var mr_opt = world.get_ptr_mut[MeshRendererComponent](state.pickup_e)
                if mr_opt:
                    mr_opt.value()[unsafe_offset=0].visible = False
                var au_opt = world.get_ptr[AudioEmitterComponent](state.pickup_e)
                var asrv_opt = resources.get_ptr_mut[AudioServer]()
                if au_opt and asrv_opt and au_opt.value()[unsafe_offset=0].asset_id >= 0:
                    _ = asrv_opt.value()[unsafe_offset=0].play_2d(
                        au_opt.value()[unsafe_offset=0].asset_id,
                        volume=Float32(0.75),
                        bus_id=BUS_SFX,
                    )
                    state.audio_played_count += 1

            if (
                not state.goal_reached
                and state.goal_e.valid()
                and OnEnter(ev, state.goal_e, world, resources)
            ):
                state.goal_reached = True
                state.score += HARNESS_GOAL_SCORE
                var flow_opt = resources.get_ptr_mut[GameFlow]()
                if flow_opt:
                    _ = flow_opt.value()[unsafe_offset=0].fsm.transition_to("Victory")

    # 2. Enemy BehaviorTree + NavMesh pathfinding + Player attack
    if state.player_e.valid() and state.enemy_e.valid():
        var plt_opt = world.get_ptr[LocalTransform](state.player_e)
        var elt_opt = world.get_ptr[LocalTransform](state.enemy_e)
        if plt_opt and elt_opt:
            var ppos = plt_opt.value()[unsafe_offset=0].translation
            var epos = elt_opt.value()[unsafe_offset=0].translation
            var dist = (ppos - epos).length()

            var nav_opt = resources.get_ptr[NavPluginStateResource]()
            if nav_opt and nav_opt.value()[unsafe_offset=0].baked:
                var np = nav_opt.value()[unsafe_offset=0].navmesh.find_path(
                    Vec3(epos.x, Float32(0.0), epos.z),
                    Vec3(ppos.x, Float32(0.0), ppos.z),
                    Float32(15.0),
                )
                state.last_nav_path_found = np.found
                state.last_nav_waypoints = len(np.waypoints)
                if np.found and len(np.waypoints) >= 2:
                    state.nav_path_verified = True

            var imap_opt = resources.get_ptr[InputMap]()
            if (
                imap_opt
                and imap_opt.value()[unsafe_offset=0].just_pressed("attack")
                and state.enemy_hp > 0
                and dist <= HARNESS_ATTACK_RANGE
            ):
                state.enemy_hp = max(0, state.enemy_hp - HARNESS_ATTACK_DAMAGE)
                state.enemy_hits += 1
                state.score += HARNESS_HIT_SCORE
                if state.enemy_hp == 0:
                    state.score += HARNESS_KILL_SCORE

            var bt_opt = world.get_ptr_mut[BehaviorTreeComponent](state.enemy_e)
            var na_opt = world.get_ptr_mut[NavAgentComponent](state.enemy_e)
            if bt_opt:
                ref btc = bt_opt.value()[unsafe_offset=0]
                var k_dist = btc.blackboard.find_key("dist_to_player")
                if k_dist >= 0:
                    btc.blackboard.set_float(k_dist, dist)
                var k_ehp = btc.blackboard.find_key("enemy_hp")
                if k_ehp >= 0:
                    btc.blackboard.set_float(k_ehp, Float32(state.enemy_hp))
                var k_alert = btc.blackboard.find_key("alert")
                if k_alert >= 0:
                    btc.blackboard.set_float(k_alert, Float32(1.0) if dist < Float32(6.0) else Float32(0.0))

                if state.enemy_hp <= 0:
                    btc.state.set_action_status(10, BT_SUCCESS)
                    btc.state.set_action_status(2, BT_SUCCESS)
                    if na_opt:
                        na_opt.value()[unsafe_offset=0].enabled = False
                    var emr_opt = world.get_ptr_mut[MeshRendererComponent](state.enemy_e)
                    if emr_opt:
                        emr_opt.value()[unsafe_offset=0].visible = False
                elif btc.state.active_action_id == 10 and na_opt:
                    na_opt.value()[unsafe_offset=0].goal = Vec3(ppos.x, Float32(0.0), ppos.z)
                    na_opt.value()[unsafe_offset=0].has_goal = True
                    na_opt.value()[unsafe_offset=0].enabled = True

    # 3. Sync HarnessStats POD component on player_e
    if state.stats_tid >= 0 and state.player_e.valid():
        var raw_ptr = world.get_raw_ptr_mut(state.player_e, state.stats_tid)
        if raw_ptr != 0:
            ref hs = Pointer[HarnessStats, MutAnyOrigin](unsafe_from_address=raw_ptr)[unsafe_offset=0]
            hs.hp = Int32(state.player_hp)
            hs.max_hp = Int32(state.player_max_hp)
            hs.score = Int32(state.score)
            hs.pickup_collected = Int32(1 if state.pickup_collected else 0)
            hs.enemy_hp = Int32(state.enemy_hp)
            hs.goal_reached = Int32(1 if state.goal_reached else 0)
            hs.enemy_hits = Int32(state.enemy_hits)
            hs.nav_waypoints = Int32(state.last_nav_waypoints)


def render_harness_scene_simd(
    mut world: World,
    mut resources: Resources,
    mut state: HarnessStateResource,
):
    var cam = Camera3D(
        screen_w=state.canvas.width,
        screen_h=state.canvas.height,
        eye=Vec3(Float32(0.0), Float32(5.8), Float32(10.5)),
        target=Vec3(Float32(0.5), Float32(0.8), Float32(0.0)),
        fov_deg=Float32(52.0),
    )
    var cam_ents = world.query_entities(world.bit_for[CameraComponent]())
    for i in range(len(cam_ents)):
        var ce = cam_ents[i]
        var cc_opt = world.get_ptr[CameraComponent](ce)
        var lt_opt = world.get_ptr[LocalTransform](ce)
        if cc_opt and lt_opt and cc_opt.value()[unsafe_offset=0].active:
            var eye = lt_opt.value()[unsafe_offset=0].translation
            cam.eye = eye
            cam.target = eye + cc_opt.value()[unsafe_offset=0].look_target_offset
            cam.fov_rad = cc_opt.value()[unsafe_offset=0].fov_y_deg * Float32(0.0174532925)
            break
    cam.update_basis()

    var sun_dir = Vec3(Float32(0.40), Float32(0.85), Float32(-0.35))
    var exposure = Float32(1.0)
    var sun_boost = Float32(1.0)
    var env_ents = world.query_entities(world.bit_for[EnvironmentComponent]())
    for i in range(len(env_ents)):
        var eo = world.get_ptr[EnvironmentComponent](env_ents[i])
        if eo and eo.value()[unsafe_offset=0].enabled:
            ref env = eo.value()[unsafe_offset=0]
            sun_dir = env.sun_dir
            exposure = max(Float32(0.2), min(Float32(2.5), env.exposure))
            sun_boost = max(
                Float32(0.5),
                min(Float32(1.6), (env.sun_radiance.x + env.sun_radiance.y + env.sun_radiance.z) * Float32(0.11)),
            )
            break

    # Collect active PointLight & SpotLight for per-entity radiance modulation
    var light_positions = List[Vec3]()
    var light_radiances = List[Vec3]()
    var light_ranges = List[Float32]()
    var pt_ents = world.query_entities(world.bit_for[PointLightComponent]())
    for i in range(len(pt_ents)):
        var pe = pt_ents[i]
        var pl_opt = world.get_ptr[PointLightComponent](pe)
        var lt_opt = world.get_ptr[LocalTransform](pe)
        if pl_opt and lt_opt and pl_opt.value()[unsafe_offset=0].enabled:
            light_positions.append(lt_opt.value()[unsafe_offset=0].translation)
            light_radiances.append(pl_opt.value()[unsafe_offset=0].radiance)
            light_ranges.append(max(Float32(1.0), pl_opt.value()[unsafe_offset=0].radius))
    var sp_ents = world.query_entities(world.bit_for[SpotLightComponent]())
    for i in range(len(sp_ents)):
        var se = sp_ents[i]
        var sl_opt = world.get_ptr[SpotLightComponent](se)
        var lt_opt = world.get_ptr[LocalTransform](se)
        if sl_opt and lt_opt and sl_opt.value()[unsafe_offset=0].enabled:
            light_positions.append(lt_opt.value()[unsafe_offset=0].translation)
            light_radiances.append(sl_opt.value()[unsafe_offset=0].radiance)
            light_ranges.append(max(Float32(1.0), sl_opt.value()[unsafe_offset=0].range))

    var sky_top = pack_argb(Int(Float32(14.0) * exposure), Int(Float32(24.0) * exposure), Int(Float32(42.0) * exposure))
    var sky_bot = pack_argb(Int(Float32(28.0) * exposure), Int(Float32(46.0) * exposure), Int(Float32(72.0) * exposure))
    state.canvas.clear(sky_top, sky_bot)

    var assets_opt = resources.get_ptr[AssetServer]()
    var mesh_mask = world.bit_for[LocalTransform]().union(world.bit_for[MeshRendererComponent]())
    var mesh_ents = world.query_entities(mesh_mask)
    for i in range(len(mesh_ents)):
        var me = mesh_ents[i]
        var mr_opt = world.get_ptr[MeshRendererComponent](me)
        var lt_opt = world.get_ptr[LocalTransform](me)
        if not mr_opt or not lt_opt or not mr_opt.value()[unsafe_offset=0].visible:
            continue
        ref mr = mr_opt.value()[unsafe_offset=0]
        var gt_opt = world.get_ptr[GlobalTransform](me)
        var pos = gt_opt.value()[unsafe_offset=0].translation if gt_opt else lt_opt.value()[unsafe_offset=0].translation
        var rot = gt_opt.value()[unsafe_offset=0].rotation if gt_opt else lt_opt.value()[unsafe_offset=0].rotation
        var scl = gt_opt.value()[unsafe_offset=0].scale if gt_opt else lt_opt.value()[unsafe_offset=0].scale

        var bc = Vec3(Float32(0.75), Float32(0.78), Float32(0.84))
        var em = Vec3()
        var mo_opt = world.get_ptr[MaterialOverrideComponent](me)
        if mo_opt and mo_opt.value()[unsafe_offset=0].enabled:
            bc = mo_opt.value()[unsafe_offset=0].base_color
            em = mo_opt.value()[unsafe_offset=0].emissive
        elif not mr.material_ref.is_empty():
            var mkey = _norm_asset_key(mr.material_ref.rel_path)
            var fc = state.flmat_colors.get(mkey)
            if fc:
                bc = fc.value()
            var fe = state.flmat_emissive.get(mkey)
            if fe:
                em = fe.value()

        # Modulate by exposure + nearby PointLight/SpotLight contribution
        var light_tint = Vec3(sun_boost, sun_boost, sun_boost)
        for li in range(len(light_positions)):
            var d = (pos - light_positions[li]).length()
            var r = light_ranges[li]
            if d < r:
                var atten = (Float32(1.0) - d / r) * Float32(0.08)
                light_tint = light_tint + light_radiances[li] * atten
        var cr = (bc.x * light_tint.x + em.x * Float32(0.4)) * exposure
        var cg = (bc.y * light_tint.y + em.y * Float32(0.4)) * exposure
        var cb = (bc.z * light_tint.z + em.z * Float32(0.4)) * exposure
        var col = pack_argb(
            Int(max(Float32(0.0), min(Float32(255.0), cr * Float32(255.0)))),
            Int(max(Float32(0.0), min(Float32(255.0), cg * Float32(255.0)))),
            Int(max(Float32(0.0), min(Float32(255.0), cb * Float32(255.0)))),
        )

        # Rasterize cooked mesh LOD 0 triangles from AssetServer.resolve_mesh -> CookedMeshAsset.geom
        var drew_cooked = False
        if assets_opt and not mr.mesh_ref.is_empty():
            var mkey = _norm_asset_key(mr.mesh_ref.rel_path)
            var mh_opt = state.mesh_handles.get(mkey)
            if mh_opt:
                var cma_opt = assets_opt.value()[unsafe_offset=0].meshes.get_ptr(mh_opt.value())
                if cma_opt:
                    ref geom = cma_opt.value()[unsafe_offset=0].geom
                    if len(geom.meshes) > 0:
                        ref gm = geom.meshes[0]
                        var n_tris = gm.triangle_count
                        var total_v = len(geom.vertex_words) // AAA_VERTEX_WORDS
                        if gm.first_index + n_tris * 3 <= len(geom.index_words):
                            for ti in range(n_tris):
                                var i0 = Int(geom.index_words[gm.first_index + ti * 3 + 0])
                                var i1 = Int(geom.index_words[gm.first_index + ti * 3 + 1])
                                var i2 = Int(geom.index_words[gm.first_index + ti * 3 + 2])
                                if i0 < 0 or i1 < 0 or i2 < 0 or i0 >= total_v or i1 >= total_v or i2 >= total_v:
                                    continue
                                var b0 = i0 * AAA_VERTEX_WORDS
                                var b1 = i1 * AAA_VERTEX_WORDS
                                var b2 = i2 * AAA_VERTEX_WORDS
                                var lv0 = Vec3(
                                    bitcast[DType.float32](geom.vertex_words[b0 + 0]) * scl.x,
                                    bitcast[DType.float32](geom.vertex_words[b0 + 1]) * scl.y,
                                    bitcast[DType.float32](geom.vertex_words[b0 + 2]) * scl.z,
                                )
                                var lv1 = Vec3(
                                    bitcast[DType.float32](geom.vertex_words[b1 + 0]) * scl.x,
                                    bitcast[DType.float32](geom.vertex_words[b1 + 1]) * scl.y,
                                    bitcast[DType.float32](geom.vertex_words[b1 + 2]) * scl.z,
                                )
                                var lv2 = Vec3(
                                    bitcast[DType.float32](geom.vertex_words[b2 + 0]) * scl.x,
                                    bitcast[DType.float32](geom.vertex_words[b2 + 1]) * scl.y,
                                    bitcast[DType.float32](geom.vertex_words[b2 + 2]) * scl.z,
                                )
                                var wp0 = pos + rot.rotate(lv0)
                                var wp1 = pos + rot.rotate(lv1)
                                var wp2 = pos + rot.rotate(lv2)
                                state.canvas.draw_tri_3d_lit(cam, wp0, wp1, wp2, col, sun_dir)
                            drew_cooked = n_tris > 0
        if not drew_cooked:
            # Minimal fallback wireframe box if mesh asset was missing
            var hx = scl.x * Float32(0.5)
            var hy = scl.y * Float32(0.5)
            var hz = scl.z * Float32(0.5)
            var a = pos + rot.rotate(Vec3(-hx, -hy, -hz))
            var b = pos + rot.rotate(Vec3(hx, hy, hz))
            state.canvas.draw_line_3d(cam, a, b, col, 1)

    # NavMesh debug wireframe overlay when NavMeshComponent.draw_debug is True
    var nav_opt = resources.get_ptr[NavPluginStateResource]()
    if nav_opt and nav_opt.value()[unsafe_offset=0].baked:
        var should_draw = nav_opt.value()[unsafe_offset=0].draw_debug
        var nm_ents = world.query_entities(world.bit_for[NavMeshComponent]())
        for ni in range(len(nm_ents)):
            var nm_opt = world.get_ptr[NavMeshComponent](nm_ents[ni])
            if nm_opt and nm_opt.value()[unsafe_offset=0].enabled:
                should_draw = should_draw and nm_opt.value()[unsafe_offset=0].draw_debug
                break
        if should_draw:
            ref polys = nav_opt.value()[unsafe_offset=0].navmesh.polys
            var lift = Vec3(Float32(0.0), Float32(0.04), Float32(0.0))
            for pi in range(len(polys)):
                ref poly = polys[pi]
                var nv = len(poly.verts)
                if nv < 2:
                    continue
                for vi in range(nv):
                    var va = poly.verts[vi] + lift
                    var vb = poly.verts[(vi + 1) % nv] + lift
                    state.canvas.draw_line_3d(cam, va, vb, UInt32(0xFF38E898), 1)

    # HUD via game_hud(ui_ctx, hp, max_hp, score) -> SimdCanvas
    state.ui_ctx.screen_w = Float32(state.canvas.width)
    state.ui_ctx.screen_h = Float32(state.canvas.height)
    state.ui_ctx.draw_list.clear()
    _ = game_hud(state.ui_ctx, state.player_hp, state.player_max_hp, state.score)
    for pi in range(len(state.ui_ctx.draw_list.prims)):
        ref p = state.ui_ctx.draw_list.prims[pi]
        var kind = p.header & UInt32(0xF)
        if kind == PRIM_SHAPE:
            var rx = Int(p.x0)
            var ry = Int(p.y0)
            var rw = max(1, Int(p.x1 - p.x0))
            var rh = max(1, Int(p.y1 - p.y0))
            var alpha = Int((p.fill_rgba8 >> UInt32(24)) & UInt32(0xFF))
            if alpha < 245:
                state.canvas.draw_rect_dim(rx, ry, rw, rh, p.fill_rgba8)
            else:
                state.canvas.draw_rect(rx, ry, rw, rh, p.fill_rgba8)
            var flags = (p.header >> UInt32(16)) & UInt32(0xFFFF)
            if (flags & UI_FLAG_BORDER) != 0 and p.param1 > Float32(0.0):
                var bw = max(1, Int(p.param1))
                state.canvas.draw_rect(rx, ry, rw, bw, p.secondary_rgba8)
                state.canvas.draw_rect(rx, ry + rh - bw, rw, bw, p.secondary_rgba8)
                state.canvas.draw_rect(rx, ry, bw, rh, p.secondary_rgba8)
                state.canvas.draw_rect(rx + rw - bw, ry, bw, rh, p.secondary_rgba8)
    var flow_label = String("PLAY")
    var flow_opt = resources.get_ptr[GameFlow]()
    if flow_opt:
        flow_label = flow_opt.value()[unsafe_offset=0].fsm.current_state()
    state.canvas.draw_text(
        24,
        22,
        "HP " + String(state.player_hp) + "/" + String(state.player_max_hp) + "  SCORE " + String(state.score),
        UInt32(0xFFF0F8F6),
        2,
        True,
    )
    state.canvas.draw_text(274, 22, "STATE: " + flow_label, UInt32(0xFF4EE2C8), 2, True)


def _sys_harness_render(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref world = world_ptr[unsafe_offset=0]
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[HarnessStateResource]()
    if not st_opt:
        return
    ref state = st_opt.value()[unsafe_offset=0]
    var win_opt = resources.get_ptr[WindowStateResource]()
    if win_opt and win_opt.value()[unsafe_offset=0].resized:
        var ww = win_opt.value()[unsafe_offset=0].width
        var wh = win_opt.value()[unsafe_offset=0].height
        if ww > 0 and wh > 0:
            state.canvas.resize(ww, wh)
    render_harness_scene_simd(world, resources, state)


def _sys_harness_present(
    _world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    _pool_ptr: Pointer[JobPool, MutAnyOrigin],
    _cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    _ctx: Int,
):
    ref resources = res_ptr[unsafe_offset=0]
    var st_opt = resources.get_ptr_mut[HarnessStateResource]()
    var win_opt = resources.get_ptr_mut[WindowStateResource]()
    if not st_opt or not win_opt:
        return
    if st_opt.value()[unsafe_offset=0].headless:
        return
    ref state = st_opt.value()[unsafe_offset=0]
    ref win_res = win_opt.value()[unsafe_offset=0]
    if len(win_res.windows) > 0:
        win_res.windows[0].present(state.canvas.pixels, state.canvas.width, state.canvas.height)


def compute_sim_checksums(
    world: World, resources: Resources, state: HarnessStateResource
) -> List[UInt64]:
    var out = List[UInt64](capacity=CHECK_SECTIONS)

    # 0. SECTION_PLAYER
    var h_player = Fnv64()
    if state.player_e.valid():
        var kc_opt = world.get_ptr[CharacterControllerComponent](state.player_e)
        if kc_opt:
            h_player.add_vec3(kc_opt.value()[unsafe_offset=0].controller.pos)
            h_player.add_vec3(kc_opt.value()[unsafe_offset=0].desired_vel)
        var lt_opt = world.get_ptr[LocalTransform](state.player_e)
        if lt_opt:
            h_player.add_vec3(lt_opt.value()[unsafe_offset=0].translation)
    h_player.add_int(state.player_hp)
    h_player.add_int(state.player_max_hp)
    out.append(h_player.value)

    # 1. SECTION_CAMERA
    var h_cam = Fnv64()
    if state.cam_e.valid():
        var cc_opt = world.get_ptr[CameraComponent](state.cam_e)
        var lt_opt = world.get_ptr[LocalTransform](state.cam_e)
        if cc_opt:
            h_cam.add_f32(cc_opt.value()[unsafe_offset=0].fov_y_deg)
            h_cam.add_vec3(cc_opt.value()[unsafe_offset=0].look_target_offset)
        if lt_opt:
            h_cam.add_vec3(lt_opt.value()[unsafe_offset=0].translation)
    out.append(h_cam.value)

    # 2. SECTION_ENEMIES
    var h_enemy = Fnv64()
    if state.enemy_e.valid():
        var elt_opt = world.get_ptr[LocalTransform](state.enemy_e)
        if elt_opt:
            h_enemy.add_vec3(elt_opt.value()[unsafe_offset=0].translation)
        var bt_opt = world.get_ptr[BehaviorTreeComponent](state.enemy_e)
        if bt_opt:
            h_enemy.add_int(bt_opt.value()[unsafe_offset=0].state.active_action_id)
            h_enemy.add_f32(bt_opt.value()[unsafe_offset=0].state.time_s)
        var na_opt = world.get_ptr[NavAgentComponent](state.enemy_e)
        if na_opt:
            h_enemy.add_bool(na_opt.value()[unsafe_offset=0].enabled)
            h_enemy.add_vec3(na_opt.value()[unsafe_offset=0].goal)
    h_enemy.add_int(state.enemy_hp)
    h_enemy.add_int(state.enemy_hits)
    h_enemy.add_int(state.last_nav_waypoints)
    out.append(h_enemy.value)

    # 3. SECTION_BODIES
    var h_bodies = Fnv64()
    var pw_opt = resources.get_ptr[PhysicsWorld]()
    if pw_opt:
        ref pw = pw_opt.value()[unsafe_offset=0]
        h_bodies.add_int(len(pw.bodies))
        for bi in range(len(pw.bodies)):
            ref b = pw.bodies[bi]
            h_bodies.add_vec3(b.pos)
            h_bodies.add_vec3(b.linear_vel)
    out.append(h_bodies.value)

    # 4. SECTION_TERRAIN (NavMesh polygons)
    var h_terrain = Fnv64()
    var nav_opt = resources.get_ptr[NavPluginStateResource]()
    if nav_opt:
        h_terrain.add_bool(nav_opt.value()[unsafe_offset=0].baked)
        h_terrain.add_int(nav_opt.value()[unsafe_offset=0].last_poly_count)
    out.append(h_terrain.value)

    # 5. SECTION_OBJECTS
    var h_obj = Fnv64()
    h_obj.add_bool(state.pickup_collected)
    h_obj.add_bool(state.goal_reached)
    if state.pickup_e.valid():
        var pmr = world.get_ptr[MeshRendererComponent](state.pickup_e)
        if pmr:
            h_obj.add_bool(pmr.value()[unsafe_offset=0].visible)
    out.append(h_obj.value)

    # 6. SECTION_PROGRESSION
    var h_prog = Fnv64()
    h_prog.add_int(state.score)
    h_prog.add_bool(state.pickup_collected)
    h_prog.add_int(state.enemy_hits)
    h_prog.add_bool(state.goal_reached)
    out.append(h_prog.value)

    # 7. SECTION_FLOW
    var h_flow = Fnv64()
    h_flow.add_int(Int(world.current_tick))
    h_flow.add_int(state.last_trigger_count)
    var flow_opt = resources.get_ptr[GameFlow]()
    if flow_opt:
        var st_name = flow_opt.value()[unsafe_offset=0].fsm.current_state()
        for b in st_name.as_bytes():
            h_flow.add_byte(b)
    out.append(h_flow.value)

    return out^


def register_harness_systems(mut app: EngineApp):
    _ = app.add_system(STAGE_PRE_UPDATE, "harness.pre_input", _sys_harness_pre_input, exclusive=True)
    _ = app.add_system(STAGE_UPDATE, "harness.update", _sys_harness_update, exclusive=True)
    _ = app.add_system(STAGE_POST_UPDATE, "harness.render", _sys_harness_render, exclusive=True)
    _ = app.add_system(STAGE_POST_UPDATE, "harness.present", _sys_harness_present, exclusive=True)

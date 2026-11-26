"""Unified `AssetPlugin` & `SceneAssetResolver` binding `AssetServer` to ECS runtime components (`flint.assets.asset_plugin`)."""
from std.collections import Dict, List
from std.math import isfinite, max, min
from flint.anim.flanim_ecs import load_flanim_bytes
from flint.anim.sync import ClipInfo
from flint.anim.flrig import FrigRigDef, load_flrig_bytes, validate_rig_bindings
from flint.anim.plugin import AnimAssetCache, AnimatorComponent, RigComponent, SkeletonComponent
from flint.anim.pose import SkeletonInstance
from flint.app.engine_app import (
    EngineApp,
    EnginePlugin,
    Resource,
    Resources,
    STAGE_PRE_UPDATE,
    STAGE_STARTUP,
)
from flint.assets.asset_ref import AssetRef, normalize_asset_rel_path
from flint.assets.flmat import (
    FlmatAsset,
    MAT_SHADING_NPR_CEL,
    MAT_SHADING_PBR,
    MAT_SHADING_STYLIZED_VFX,
    load_flmat_bytes,
)
from flint.assets.flpk_v3 import decode_cooked_rig_slab
from flint.assets.glb_native import parse_native_glb
from flint.assets.model import NativeMesh, NativeModel
from flint.assets.server import AssetServer
from flint.audio.plugin import AudioEmitterComponent
from flint.audio.server import AudioServer
from flint.core.jobs import JobPool
from flint.core.math import Vec3
from flint.ecs.world import Entity, ParallelCommands, World, comp_hash64
from flint.platform.os import read_bytes
from flint.render.material_graph import MAT_DOMAIN_STANDARD_PBR, MAT_DOMAIN_UNLIT_UI, MaterialAsset
from flint.render.plugin import (
    MaterialOverrideComponent,
    MeshRendererComponent,
    RenderPluginStateResource,
)
from flint.render.render_world import ScalableRenderer, material_asset_from_native_mesh


@always_inline
def _ref_cache_key(aref: AssetRef) -> String:
    if aref.rel_path.byte_length() > 0:
        var norm = normalize_asset_rel_path(aref.rel_path)
        return norm[1] if norm[0] else aref.rel_path
    if not aref.uuid.is_nil():
        return String(aref.uuid.hi) + ":" + String(aref.uuid.lo)
    return String()


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


@always_inline
def _hash_str_u64(s: String) -> UInt64:
    var h = UInt64(14695981039346656037)
    var b = s.as_bytes()
    for i in range(len(b)):
        h = (h ^ UInt64(b[i])) * UInt64(1099511628211)
    return h if h != 0 else UInt64(1)


def _flmat_to_material_asset(mat: FlmatAsset) -> MaterialAsset:
    var domain = MAT_DOMAIN_STANDARD_PBR
    if mat.shading_mode == MAT_SHADING_STYLIZED_VFX:
        domain = MAT_DOMAIN_UNLIT_UI
    var m = MaterialAsset(mat.base_color, mat.roughness, mat.metallic, domain)
    m.opacity = max(Float32(0.0), min(Float32(1.0), mat.opacity)) if isfinite(mat.opacity) else Float32(1.0)
    m.alpha_cutoff = max(Float32(0.0), min(Float32(1.0), mat.alpha_cutoff)) if isfinite(mat.alpha_cutoff) else Float32(0.0)
    m.emissive = mat.emissive
    m.clearcoat = max(Float32(0.0), min(Float32(1.0), mat.clearcoat)) if isfinite(mat.clearcoat) else Float32(0.0)
    m.clearcoat_roughness = (
        max(Float32(0.04), min(Float32(1.0), mat.clearcoat_roughness))
        if isfinite(mat.clearcoat_roughness)
        else Float32(0.08)
    )
    var sh = max(Float32(0.0), min(Float32(1.0), mat.sheen)) if isfinite(mat.sheen) else Float32(0.0)
    m.sheen_color = Vec3(sh, sh, sh)
    m.subsurface_amount = (
        max(Float32(0.0), min(Float32(1.0), mat.subsurface))
        if isfinite(mat.subsurface)
        else Float32(0.0)
    )
    return m^


def _parse_rig_bytes(raw: List[UInt8]) raises -> FrigRigDef:
    var slab = decode_cooked_rig_slab(raw)
    if slab[0] and len(slab[1]) > 0:
        return load_flrig_bytes(slab[1])
    return load_flrig_bytes(raw)


def _copy_u64_dict(src: Dict[String, UInt64]) -> Dict[String, UInt64]:
    var dst = Dict[String, UInt64]()
    for item in src.items():
        dst[item.key] = item.value
    return dst^


def _restore_u64_epoch_dict(mut cur: Dict[String, UInt64], snap: Dict[String, UInt64]):
    var next = Dict[String, UInt64]()
    for item in snap.items():
        next[item.key] = item.value
    for item in cur.items():
        if not next.get(item.key) and item.value > UInt64(0):
            next[item.key] = UInt64(0)
    cur = next^


struct PieResolverEpochSnapshot(Movable):
    var mesh_epochs: Dict[String, UInt64]
    var mat_epochs: Dict[String, UInt64]
    var mat_tex_epochs: Dict[String, Tuple[UInt64, UInt64, UInt64, UInt64]]
    var glb_model_epochs: Dict[String, UInt64]
    var rig_comp_epochs: Dict[String, UInt64]
    var skel_comp_epochs: Dict[String, UInt64]
    var graph_epochs: Dict[String, UInt64]
    var audio_epochs: Dict[String, UInt64]

    def __init__(out self):
        self.mesh_epochs = Dict[String, UInt64]()
        self.mat_epochs = Dict[String, UInt64]()
        self.mat_tex_epochs = Dict[String, Tuple[UInt64, UInt64, UInt64, UInt64]]()
        self.glb_model_epochs = Dict[String, UInt64]()
        self.rig_comp_epochs = Dict[String, UInt64]()
        self.skel_comp_epochs = Dict[String, UInt64]()
        self.graph_epochs = Dict[String, UInt64]()
        self.audio_epochs = Dict[String, UInt64]()


struct SceneAssetResolver(Resource, Movable):
    """Resolves `AssetRef` fields on scene/runtime ECS components via `AssetServer` and propagates hot-reloads."""
    comptime RESOURCE_ID: UInt64 = comp_hash64("flint.assets.SceneAssetResolver")

    var mesh_ids: Dict[String, Int]
    var mesh_epochs: Dict[String, UInt64]
    var mesh_default_mat_ids: Dict[String, Int]
    var mat_ids: Dict[String, Int]
    var mat_epochs: Dict[String, UInt64]
    var mat_tex_epochs: Dict[String, Tuple[UInt64, UInt64, UInt64, UInt64]]
    var mat_tex_slots: Dict[String, Tuple[Int, Int, Int, Int]]
    var glb_model_epochs: Dict[String, UInt64]
    var glb_model_ids: Dict[String, UInt64]
    var rig_comp_epochs: Dict[String, UInt64]
    var skel_comp_epochs: Dict[String, UInt64]
    var skel_entity_rig_keys: Dict[UInt64, String]
    var graph_ids: Dict[String, UInt64]
    var graph_epochs: Dict[String, UInt64]
    var audio_ids: Dict[String, Int]
    var audio_epochs: Dict[String, UInt64]
    var hot_reloads_applied: Int
    var reload_epoch: UInt64

    def __init__(out self):
        self.mesh_ids = Dict[String, Int]()
        self.mesh_epochs = Dict[String, UInt64]()
        self.mesh_default_mat_ids = Dict[String, Int]()
        self.mat_ids = Dict[String, Int]()
        self.mat_epochs = Dict[String, UInt64]()
        self.mat_tex_epochs = Dict[String, Tuple[UInt64, UInt64, UInt64, UInt64]]()
        self.mat_tex_slots = Dict[String, Tuple[Int, Int, Int, Int]]()
        self.glb_model_epochs = Dict[String, UInt64]()
        self.glb_model_ids = Dict[String, UInt64]()
        self.rig_comp_epochs = Dict[String, UInt64]()
        self.skel_comp_epochs = Dict[String, UInt64]()
        self.skel_entity_rig_keys = Dict[UInt64, String]()
        self.graph_ids = Dict[String, UInt64]()
        self.graph_epochs = Dict[String, UInt64]()
        self.audio_ids = Dict[String, Int]()
        self.audio_epochs = Dict[String, UInt64]()
        self.hot_reloads_applied = 0
        self.reload_epoch = 0

    def capture_pie_epoch_snapshot(self) -> PieResolverEpochSnapshot:
        var snap = PieResolverEpochSnapshot()
        snap.mesh_epochs = _copy_u64_dict(self.mesh_epochs)
        snap.mat_epochs = _copy_u64_dict(self.mat_epochs)
        for item in self.mat_tex_epochs.items():
            snap.mat_tex_epochs[item.key] = item.value
        snap.glb_model_epochs = _copy_u64_dict(self.glb_model_epochs)
        snap.rig_comp_epochs = _copy_u64_dict(self.rig_comp_epochs)
        snap.skel_comp_epochs = _copy_u64_dict(self.skel_comp_epochs)
        snap.graph_epochs = _copy_u64_dict(self.graph_epochs)
        snap.audio_epochs = _copy_u64_dict(self.audio_epochs)
        return snap^

    def restore_pie_epoch_snapshot(mut self, snap: PieResolverEpochSnapshot):
        _restore_u64_epoch_dict(self.mesh_epochs, snap.mesh_epochs)
        _restore_u64_epoch_dict(self.mat_epochs, snap.mat_epochs)
        var next_tex = Dict[String, Tuple[UInt64, UInt64, UInt64, UInt64]]()
        for item in snap.mat_tex_epochs.items():
            next_tex[item.key] = item.value
        self.mat_tex_epochs = next_tex^
        _restore_u64_epoch_dict(self.glb_model_epochs, snap.glb_model_epochs)
        _restore_u64_epoch_dict(self.rig_comp_epochs, snap.rig_comp_epochs)
        _restore_u64_epoch_dict(self.skel_comp_epochs, snap.skel_comp_epochs)
        _restore_u64_epoch_dict(self.graph_epochs, snap.graph_epochs)
        _restore_u64_epoch_dict(self.audio_epochs, snap.audio_epochs)

    def _resolve_flmat(
        mut self,
        aref: AssetRef,
        mut server: AssetServer,
        mut renderer_opt: Optional[Pointer[ScalableRenderer, MutAnyOrigin]],
        mut reloaded_mat_keys: Dict[String, Bool],
    ) -> Tuple[Int, Bool, FlmatAsset]:
        if aref.is_empty():
            return (-1, False, FlmatAsset())
        var h = server.resolve_raw(aref)
        if not server.raws.is_loaded(h):
            return (-1, False, FlmatAsset())
        var rptr = server.raws.get_ptr(h)
        if not rptr:
            return (-1, False, FlmatAsset())
        var cur_epoch = server.raws.reload_epoch(h)
        var key = _ref_cache_key(aref)
        var prev_id = self.mat_ids.get(key)
        var prev_ep = self.mat_epochs.get(key)
        var prev_rel = reloaded_mat_keys.get(key)
        var first_load = (not prev_id) or (not prev_ep)
        var epoch_changed = Bool(prev_ep) and (prev_ep.value() != cur_epoch)
        try:
            var flmat = load_flmat_bytes(rptr.value()[].bytes)
            var alb_ep = UInt64(0)
            var nrm_ep = UInt64(0)
            var orm_ep = UInt64(0)
            var emi_ep = UInt64(0)
            var alb_slot = -1
            var nrm_slot = -1
            var orm_slot = -1
            var emi_slot = -1

            if not flmat.albedo_tex.is_empty():
                var th = server.resolve_texture_with_role(flmat.albedo_tex, role=0, progressive=False)
                if server.textures.is_loaded(th):
                    alb_ep = server.textures.reload_epoch(th)
                    alb_slot = server.ensure_texture_gpu_slot(th)
            if not flmat.normal_tex.is_empty():
                var th = server.resolve_texture_with_role(flmat.normal_tex, role=1, progressive=False)
                if server.textures.is_loaded(th):
                    nrm_ep = server.textures.reload_epoch(th)
                    nrm_slot = server.ensure_texture_gpu_slot(th)
            if not flmat.orm_tex.is_empty():
                var th = server.resolve_texture_with_role(flmat.orm_tex, role=2, progressive=False)
                if server.textures.is_loaded(th):
                    orm_ep = server.textures.reload_epoch(th)
                    orm_slot = server.ensure_texture_gpu_slot(th)
            if not flmat.emissive_tex.is_empty():
                var th = server.resolve_texture_with_role(flmat.emissive_tex, role=0, progressive=False)
                if server.textures.is_loaded(th):
                    emi_ep = server.textures.reload_epoch(th)
                    emi_slot = server.ensure_texture_gpu_slot(th)

            var cur_tex_eps = (alb_ep, nrm_ep, orm_ep, emi_ep)
            var cur_tex_slots = (alb_slot, nrm_slot, orm_slot, emi_slot)
            var prev_tex_eps = self.mat_tex_epochs.get(key)
            var prev_tex_slots = self.mat_tex_slots.get(key)
            var tex_changed = False
            if (not prev_tex_eps) or (not prev_tex_slots):
                if alb_ep != 0 or nrm_ep != 0 or orm_ep != 0 or emi_ep != 0 or alb_slot >= 0 or nrm_slot >= 0 or orm_slot >= 0 or emi_slot >= 0:
                    tex_changed = True
            else:
                var pe = prev_tex_eps.value()
                var ps = prev_tex_slots.value()
                if (
                    pe[0] != cur_tex_eps[0]
                    or pe[1] != cur_tex_eps[1]
                    or pe[2] != cur_tex_eps[2]
                    or pe[3] != cur_tex_eps[3]
                    or ps[0] != cur_tex_slots[0]
                    or ps[1] != cur_tex_slots[1]
                    or ps[2] != cur_tex_slots[2]
                    or ps[3] != cur_tex_slots[3]
                ):
                    tex_changed = True

            var needs_update = first_load or epoch_changed or tex_changed
            var is_reloaded = needs_update or (Bool(prev_rel) and prev_rel.value())
            if needs_update:
                var mat_asset = _flmat_to_material_asset(flmat)
                mat_asset.set_albedo_slot(alb_slot)
                mat_asset.set_normal_slot(nrm_slot)
                mat_asset.set_orm_slot(orm_slot)
                mat_asset.set_emissive_slot(emi_slot)
                self.mat_tex_epochs[key] = cur_tex_eps
                self.mat_tex_slots[key] = cur_tex_slots
                if prev_id and prev_id.value() >= 0:
                    if renderer_opt:
                        renderer_opt.value()[].update_material_asset_deferred(prev_id.value(), mat_asset)
                    self.mat_epochs[key] = cur_epoch
                    reloaded_mat_keys[key] = True
                    return (prev_id.value(), True, flmat^)
                var new_id = 0
                if renderer_opt:
                    new_id = renderer_opt.value()[].register_material(mat_asset)
                else:
                    new_id = len(self.mat_ids)
                self.mat_ids[key] = new_id
                self.mat_epochs[key] = cur_epoch
                reloaded_mat_keys[key] = True
                return (new_id, True, flmat^)
            return (prev_id.value(), is_reloaded, flmat^)
        except:
            return (prev_id.value() if prev_id else -1, False, FlmatAsset())

    @staticmethod
    def _read_glb_bytes_for_ref(server: AssetServer, aref: AssetRef) -> List[UInt8]:
        var norm = normalize_asset_rel_path(aref.rel_path)
        if norm[0] and norm[1].byte_length() > 0:
            var full = server._resolve_loose_disk_path(norm[1])
            if full.byte_length() > 0:
                var raw = read_bytes(full)
                if len(raw) > 0:
                    return raw^
        if not aref.uuid.is_nil():
            var s = server.meshes.find_slot_by_uuid(aref.uuid)
            if s >= 0 and s < len(server.meshes.rel_paths):
                var full = server._resolve_loose_disk_path(server.meshes.rel_paths[s])
                if full.byte_length() > 0:
                    var raw = read_bytes(full)
                    if len(raw) > 0:
                        return raw^
        if norm[0] and norm[1].byte_length() > 0:
            var loc = server._find_in_mounts_by_path(norm[1], 0)
            if loc[0] >= 0:
                var pay = server.mounts[loc[0]].read_payload_by_toc(loc[1], verify_crc=True)
                if (
                    pay[0]
                    and len(pay[1]) >= 4
                    and pay[1][0] == 0x67
                    and pay[1][1] == 0x6C
                    and pay[1][2] == 0x54
                    and pay[1][3] == 0x46
                ):
                    return pay[1].copy()
        return List[UInt8]()

    def _sync_glb_anim_cache(
        mut self,
        mut world: World,
        e: Entity,
        mkey: String,
        mesh_ref: AssetRef,
        cur_ep: UInt64,
        is_reloaded: Bool,
        mut server: AssetServer,
        mut anim_cache_opt: Optional[Pointer[AnimAssetCache, MutAnyOrigin]],
    ):
        if not anim_cache_opt:
            return
        var sk_ptr = world.get_ptr_mut[SkeletonComponent](e)
        if not sk_ptr:
            return
        ref sk = sk_ptr.value()[unsafe_offset=0]
        ref anim_cache = anim_cache_opt.value()[unsafe_offset=0]
        var prev_glb_ep = self.glb_model_epochs.get(mkey)
        var glb_epoch_changed = is_reloaded or (Bool(prev_glb_ep) and prev_glb_ep.value() != cur_ep)
        var need_model_load = (
            sk.model_id == 0
            or anim_cache.find_model(sk.model_id) < 0
            or glb_epoch_changed
        )
        if not need_model_load:
            if not prev_glb_ep:
                self.glb_model_epochs[mkey] = cur_ep
            if sk.model_id != 0:
                self.glb_model_ids[mkey] = sk.model_id
            return
        var raw_glb = Self._read_glb_bytes_for_ref(server, mesh_ref)
        if len(raw_glb) == 0:
            return
        var mdl = parse_native_glb(raw_glb)
        if not mdl.loaded() or len(mdl.meshes) == 0:
            return
        var target_mid = sk.model_id
        if target_mid == 0:
            var cached_mid = self.glb_model_ids.get(mkey)
            if cached_mid and cached_mid.value() != 0:
                target_mid = cached_mid.value()
            else:
                target_mid = _hash_str_u64(mkey)
        self.glb_model_ids[mkey] = target_mid
        self.glb_model_epochs[mkey] = cur_ep
        var existing_idx = anim_cache.find_model(target_mid)
        if existing_idx >= 0 and len(anim_cache.models[existing_idx].skins) > 0 and len(mdl.skins) == 0:
            return
        sk.model_id = target_mid
        if sk.skeleton.node_count == 0 or (glb_epoch_changed and sk.rig_ref.is_empty()):
            var n_nodes = len(mdl.nodes)
            if n_nodes > 0:
                sk.skeleton.ensure_topology(mdl)
                for ji in range(n_nodes):
                    sk.skeleton.pose.t[ji] = mdl.nodes[ji].translation
                    sk.skeleton.pose.r[ji] = mdl.nodes[ji].rotation
                    sk.skeleton.pose.s[ji] = mdl.nodes[ji].scale
                sk.dirty_topology = False
        elif sk.rig_ref.is_empty():
            sk.dirty_topology = False
        var ac_ptr = world.get_ptr_mut[AnimatorComponent](e)
        if ac_ptr:
            ref ac = ac_ptr.value()[unsafe_offset=0]
            if len(mdl.animations) == 0:
                if ac.clip_index >= 0:
                    ac.clip_index = -1
                    ac.time = Float32(0.0)
            elif ac.clip_index >= len(mdl.animations):
                ac.clip_index = 0
                ac.time = Float32(0.0)
        anim_cache.register_model(target_mid, mdl^)

    def _resolve_glb_default_material(
        mut self,
        mkey: String,
        mesh_ref: AssetRef,
        mesh_idx: Int,
        prev_def_mat: Int,
        needs_update: Bool,
        server: AssetServer,
        mut renderer_opt: Optional[Pointer[ScalableRenderer, MutAnyOrigin]],
    ) -> Int:
        if (not needs_update) and prev_def_mat >= 0:
            if not renderer_opt or prev_def_mat < len(renderer_opt.value()[].materials):
                return prev_def_mat
        if not renderer_opt:
            var new_id = prev_def_mat if prev_def_mat >= 0 else len(self.mesh_default_mat_ids)
            self.mesh_default_mat_ids[mkey] = new_id
            return new_id
        ref renderer = renderer_opt.value()[unsafe_offset=0]
        var ma = material_asset_from_native_mesh(NativeMesh())
        var raw_glb = Self._read_glb_bytes_for_ref(server, mesh_ref)
        if len(raw_glb) > 0:
            var mdl = parse_native_glb(raw_glb)
            if mdl.loaded() and len(mdl.meshes) > 0:
                var src_mi = mesh_idx if (mesh_idx >= 0 and mesh_idx < len(mdl.meshes)) else 0
                ma = renderer.build_native_mesh_material(mdl.meshes[src_mi], mdl)
        if prev_def_mat >= 0 and prev_def_mat < len(renderer.materials):
            renderer.update_material_asset_deferred(prev_def_mat, ma)
            self.mesh_default_mat_ids[mkey] = prev_def_mat
            return prev_def_mat
        var new_mat_id = renderer.register_material(ma)
        self.mesh_default_mat_ids[mkey] = new_mat_id
        return new_mat_id

    def resolve_world(mut self, mut world: World, mut server: AssetServer, mut resources: Resources):
        if server.hot_reloads_applied > self.hot_reloads_applied:
            self.reload_epoch += UInt64(server.hot_reloads_applied - self.hot_reloads_applied)
            self.hot_reloads_applied = server.hot_reloads_applied

        var renderer_opt = resources.get_ptr_mut[ScalableRenderer]()
        var render_state_opt = resources.get_ptr_mut[RenderPluginStateResource]()
        var anim_cache_opt = resources.get_ptr_mut[AnimAssetCache]()
        var audio_srv_opt = resources.get_ptr_mut[AudioServer]()
        var has_skel_comp = world.component_id[SkeletonComponent]() >= 0

        # 1. MeshRendererComponent (mesh_ref + material_ref)
        var reloaded_mkeys = Dict[String, Bool]()
        var reloaded_mat_keys = Dict[String, Bool]()
        var mr_ents = world.query_entities(world.bit_for[MeshRendererComponent]())
        for i in range(len(mr_ents)):
            var e = mr_ents[i]
            var mr_ptr = world.get_ptr_mut[MeshRendererComponent](e)
            if not mr_ptr:
                continue
            var has_skel = has_skel_comp and world.has[SkeletonComponent](e)
            var resolved_default_mat = -1
            if not mr_ptr.value()[].mesh_ref.is_empty():
                var mh = server.resolve_mesh(mr_ptr.value()[].mesh_ref)
                if server.meshes.is_loaded(mh):
                    var mkey = _ref_cache_key(mr_ptr.value()[].mesh_ref)
                    var cur_ep = server.meshes.reload_epoch(mh)
                    var prev_mid = self.mesh_ids.get(mkey)
                    var prev_ep = self.mesh_epochs.get(mkey)
                    var prev_def_mat_opt = self.mesh_default_mat_ids.get(mkey)
                    var prev_def_mat = prev_def_mat_opt.value() if prev_def_mat_opt else -1
                    var def_mat_valid = prev_def_mat >= 0
                    if def_mat_valid and renderer_opt:
                        if prev_def_mat >= len(renderer_opt.value()[].materials):
                            def_mat_valid = False
                    var epoch_changed = Bool(prev_ep) and (prev_ep.value() != cur_ep)
                    var prev_reloaded = reloaded_mkeys.get(mkey)
                    var is_reloaded = epoch_changed or (Bool(prev_reloaded) and prev_reloaded.value())
                    if epoch_changed:
                        reloaded_mkeys[mkey] = True

                    var mptr = server.meshes.get_ptr(mh)
                    var src_mi = 0
                    if mptr and len(mptr.value()[].geom.meshes) > 0:
                        src_mi = mptr.value()[].geom.meshes[0].material_id
                    if (not prev_mid) or (not prev_ep) or epoch_changed or (not def_mat_valid):
                        resolved_default_mat = self._resolve_glb_default_material(
                            mkey,
                            mr_ptr.value()[].mesh_ref,
                            src_mi,
                            prev_def_mat if def_mat_valid else -1,
                            epoch_changed or (not def_mat_valid),
                            server,
                            renderer_opt,
                        )
                    else:
                        resolved_default_mat = prev_def_mat

                    if (not prev_mid) or (not prev_ep) or epoch_changed:
                        if mptr:
                            var resolved_mid = -1
                            if renderer_opt:
                                resolved_mid = renderer_opt.value()[].register_cooked_mesh_asset(
                                    mptr.value()[].geom, mat_id=resolved_default_mat
                                )
                            elif prev_mid:
                                resolved_mid = prev_mid.value()
                            else:
                                resolved_mid = len(self.mesh_ids)
                            self.mesh_ids[mkey] = resolved_mid
                            self.mesh_epochs[mkey] = cur_ep
                            if (not has_skel) or mr_ptr.value()[].mesh_id < 0 or is_reloaded:
                                mr_ptr.value()[].mesh_id = resolved_mid
                            if has_skel:
                                self._sync_glb_anim_cache(
                                    world, e, mkey, mr_ptr.value()[].mesh_ref, cur_ep, is_reloaded, server, anim_cache_opt
                                )
                            if has_skel and is_reloaded and render_state_opt:
                                render_state_opt.value()[].skinned_invalidated[e.to_bits()] = True
                    elif prev_mid:
                        var target_mid = prev_mid.value()
                        if (
                            renderer_opt
                            and resolved_default_mat >= 0
                            and target_mid >= 0
                            and target_mid < len(renderer_opt.value()[].cpu_mesh_materials)
                        ):
                            renderer_opt.value()[].cpu_mesh_materials[target_mid] = resolved_default_mat
                        if is_reloaded:
                            mr_ptr.value()[].mesh_id = target_mid
                            if has_skel and render_state_opt:
                                render_state_opt.value()[].skinned_invalidated[e.to_bits()] = True
                        elif mr_ptr.value()[].mesh_id < 0:
                            mr_ptr.value()[].mesh_id = target_mid
                        elif (not has_skel) and mr_ptr.value()[].mesh_id != target_mid:
                            mr_ptr.value()[].mesh_id = target_mid
                        if has_skel:
                            self._sync_glb_anim_cache(
                                world, e, mkey, mr_ptr.value()[].mesh_ref, cur_ep, is_reloaded, server, anim_cache_opt
                            )
            if not mr_ptr.value()[].material_ref.is_empty():
                var mres = self._resolve_flmat(
                    mr_ptr.value()[].material_ref, server, renderer_opt, reloaded_mat_keys
                )
                if mres[0] >= 0 and (
                    mr_ptr.value()[].material_id < 0
                    or mres[1]
                    or mr_ptr.value()[].material_id != mres[0]
                ):
                    mr_ptr.value()[].material_id = mres[0]
            elif resolved_default_mat >= 0 and mr_ptr.value()[].material_id != resolved_default_mat:
                mr_ptr.value()[].material_id = resolved_default_mat

        # 2. MaterialOverrideComponent (material_ref -> resolved shared material + defaults)
        var mo_ents = world.query_entities(world.bit_for[MaterialOverrideComponent]())
        for i in range(len(mo_ents)):
            var e = mo_ents[i]
            var mo_ptr = world.get_ptr_mut[MaterialOverrideComponent](e)
            if not mo_ptr:
                continue
            if not mo_ptr.value()[].material_ref.is_empty():
                var mres = self._resolve_flmat(
                    mo_ptr.value()[].material_ref, server, renderer_opt, reloaded_mat_keys
                )
                if mres[0] >= 0 and (
                    mo_ptr.value()[].material_id < 0
                    or mres[1]
                    or mo_ptr.value()[].material_id != mres[0]
                ):
                    mo_ptr.value()[].material_id = mres[0]
                    mo_ptr.value()[].base_color = mres[2].base_color
                    mo_ptr.value()[].roughness = mres[2].roughness
                    mo_ptr.value()[].metallic = mres[2].metallic
                    mo_ptr.value()[].emissive = mres[2].emissive
                    mo_ptr.value()[].opacity = mres[2].opacity
                    mo_ptr.value()[].alpha_cutoff = mres[2].alpha_cutoff
                    mo_ptr.value()[].clearcoat = mres[2].clearcoat
                    mo_ptr.value()[].clearcoat_roughness = mres[2].clearcoat_roughness
                    mo_ptr.value()[].sheen = mres[2].sheen
                    mo_ptr.value()[].subsurface = mres[2].subsurface

        # 3. RigComponent (rig_ref -> validate & bind joint_count)
        var reloaded_rig_keys = Dict[String, Bool]()
        var rig_ents = world.query_entities(world.bit_for[RigComponent]())
        for i in range(len(rig_ents)):
            var e = rig_ents[i]
            var r_ptr = world.get_ptr_mut[RigComponent](e)
            if not r_ptr:
                continue
            if not r_ptr.value()[].rig_ref.is_empty():
                var rh = server.resolve_raw(r_ptr.value()[].rig_ref)
                if server.raws.is_loaded(rh):
                    var cur_ep = server.raws.reload_epoch(rh)
                    var rkey = _ref_cache_key(r_ptr.value()[].rig_ref)
                    var prev_ep = self.rig_comp_epochs.get(rkey)
                    var prev_rel = reloaded_rig_keys.get(rkey)
                    var is_reloaded = (Bool(prev_ep) and prev_ep.value() != cur_ep) or (Bool(prev_rel) and prev_rel.value())
                    if (not r_ptr.value()[].bound) or (not prev_ep) or is_reloaded:
                        var bptr = server.raws.get_ptr(rh)
                        if bptr:
                            try:
                                var rig_def = _parse_rig_bytes(bptr.value()[].bytes)
                                validate_rig_bindings(rig_def)
                                r_ptr.value()[].joint_count = len(rig_def.joints)
                                r_ptr.value()[].bound = True
                                if prev_ep and prev_ep.value() != cur_ep:
                                    reloaded_rig_keys[rkey] = True
                                self.rig_comp_epochs[rkey] = cur_ep
                            except:
                                pass

        # 4. SkeletonComponent (rig_ref -> populate rest joints if unpopulated, changed, dirty, or reloaded)
        var reloaded_skel_keys = Dict[String, Bool]()
        var sk_ents = world.query_entities(world.bit_for[SkeletonComponent]())
        for i in range(len(sk_ents)):
            var e = sk_ents[i]
            var sk_ptr = world.get_ptr_mut[SkeletonComponent](e)
            if not sk_ptr:
                continue
            var ebits = e.to_bits()
            if not sk_ptr.value()[].rig_ref.is_empty():
                var rh = server.resolve_raw(sk_ptr.value()[].rig_ref)
                if server.raws.is_loaded(rh):
                    var cur_ep = server.raws.reload_epoch(rh)
                    var rkey = _ref_cache_key(sk_ptr.value()[].rig_ref)
                    var prev_ep = self.skel_comp_epochs.get(rkey)
                    var prev_rel = reloaded_skel_keys.get(rkey)
                    var prev_ent_rkey = self.skel_entity_rig_keys.get(ebits)
                    var rig_changed_on_ent = Bool(prev_ent_rkey) and (prev_ent_rkey.value() != rkey)
                    var epoch_changed = Bool(prev_ep) and (prev_ep.value() != cur_ep)
                    var is_reloaded = epoch_changed or (Bool(prev_rel) and prev_rel.value())
                    if (
                        sk_ptr.value()[].skeleton.node_count == 0
                        or sk_ptr.value()[].dirty_topology
                        or rig_changed_on_ent
                        or is_reloaded
                    ):
                        var bptr = server.raws.get_ptr(rh)
                        if bptr:
                            try:
                                var rig_def = _parse_rig_bytes(bptr.value()[].bytes)
                                validate_rig_bindings(rig_def)
                                var nj = len(rig_def.joints)
                                var si = SkeletonInstance(nj)
                                for ji in range(nj):
                                    si.parents[ji] = rig_def.joints[ji].parent_index
                                    si.pose.t[ji] = rig_def.joints[ji].translation
                                    si.pose.r[ji] = rig_def.joints[ji].rotation
                                    si.pose.s[ji] = rig_def.joints[ji].scale
                                _compute_skeleton_topo_order(si)
                                sk_ptr.value()[].skeleton = si^
                                sk_ptr.value()[].dirty_topology = False
                                if epoch_changed:
                                    reloaded_skel_keys[rkey] = True
                                self.skel_comp_epochs[rkey] = cur_ep
                            except:
                                pass
                    elif not prev_ep:
                        self.skel_comp_epochs[rkey] = cur_ep
                    self.skel_entity_rig_keys[ebits] = rkey
            else:
                self.skel_entity_rig_keys[ebits] = String()

        # 5. AnimatorComponent (graph_ref -> AnimAssetCache)
        var an_ents = world.query_entities(world.bit_for[AnimatorComponent]())
        for i in range(len(an_ents)):
            var e = an_ents[i]
            var an_ptr = world.get_ptr_mut[AnimatorComponent](e)
            if not an_ptr:
                continue
            if not an_ptr.value()[].graph_ref.is_empty():
                var ah = server.resolve_raw(an_ptr.value()[].graph_ref)
                if server.raws.is_loaded(ah):
                    var cur_ep = server.raws.reload_epoch(ah)
                    var akey = _ref_cache_key(an_ptr.value()[].graph_ref)
                    var prev_gid = self.graph_ids.get(akey)
                    var prev_ep = self.graph_epochs.get(akey)
                    if an_ptr.value()[].graph_id == 0 or (not prev_gid) or (not prev_ep) or (prev_ep.value() != cur_ep):
                        var bptr = server.raws.get_ptr(ah)
                        if bptr:
                            try:
                                var gdef = load_flanim_bytes(bptr.value()[].bytes)
                                var gid = _hash_str_u64(akey)
                                if anim_cache_opt:
                                    var empty_clips = List[ClipInfo]()
                                    anim_cache_opt.value()[].register_graph(gid, gdef^, empty_clips^)
                                self.graph_ids[akey] = gid
                                self.graph_epochs[akey] = cur_ep
                                an_ptr.value()[].graph_id = gid
                            except:
                                pass

        # 6. AudioEmitterComponent (audio_ref -> AudioServer PCM clip)
        var reloaded_audio_keys = Dict[String, Bool]()
        var au_ents = world.query_entities(world.bit_for[AudioEmitterComponent]())
        for i in range(len(au_ents)):
            var e = au_ents[i]
            var au_ptr = world.get_ptr_mut[AudioEmitterComponent](e)
            if not au_ptr:
                continue
            if not au_ptr.value()[].audio_ref.is_empty():
                var auh = server.resolve_audio(au_ptr.value()[].audio_ref)
                if server.audios.is_loaded(auh):
                    var cur_ep = server.audios.reload_epoch(auh)
                    var aukey = _ref_cache_key(au_ptr.value()[].audio_ref)
                    var prev_aid = self.audio_ids.get(aukey)
                    var prev_ep = self.audio_epochs.get(aukey)
                    var first_load = (not prev_aid) or (not prev_ep)
                    var epoch_changed = Bool(prev_ep) and (prev_ep.value() != cur_ep)
                    var prev_rel = reloaded_audio_keys.get(aukey)
                    var is_reloaded = first_load or epoch_changed or (Bool(prev_rel) and prev_rel.value())
                    if (
                        au_ptr.value()[].asset_id < 0
                        or is_reloaded
                        or (Bool(prev_aid) and au_ptr.value()[].asset_id != prev_aid.value())
                    ):
                        var dptr = server.audios.get_ptr(auh)
                        if dptr:
                            if audio_srv_opt:
                                var aid = -1
                                if first_load or epoch_changed:
                                    aid = audio_srv_opt.value()[].register_pcm16_asset(
                                        dptr.value()[].samples,
                                        dptr.value()[].channels,
                                        dptr.value()[].sample_rate,
                                    )
                                    self.audio_ids[aukey] = aid
                                    self.audio_epochs[aukey] = cur_ep
                                    reloaded_audio_keys[aukey] = True
                                else:
                                    aid = prev_aid.value()
                                au_ptr.value()[].asset_id = aid
                            elif prev_aid:
                                au_ptr.value()[].asset_id = prev_aid.value()
                            if au_ptr.value()[].audio_ref.rel_path.byte_length() > 0:
                                au_ptr.value()[].asset_path = au_ptr.value()[].audio_ref.rel_path


def _sys_asset_tick_and_resolve(
    world_ptr: Pointer[World, MutAnyOrigin],
    res_ptr: Pointer[Resources, MutAnyOrigin],
    pool_ptr: Pointer[JobPool, MutAnyOrigin],
    cmds_ptr: Pointer[ParallelCommands, MutAnyOrigin],
    ctx: Int,
):
    _ = pool_ptr
    _ = cmds_ptr
    _ = ctx
    var srv_opt = res_ptr[].get_ptr_mut[AssetServer]()
    if not srv_opt:
        return
    if srv_opt.value()[].renderer_ptr == 0:
        var rend_opt = res_ptr[].get_ptr_mut[ScalableRenderer]()
        if rend_opt and rend_opt.value()[].uses_vulkan():
            srv_opt.value()[].attach_renderer_residency_bridge(rend_opt.value()[].gpu_slot[0])
    _ = srv_opt.value()[].tick()
    var res_opt = res_ptr[].get_ptr_mut[SceneAssetResolver]()
    if res_opt:
        res_opt.value()[].resolve_world(world_ptr[], srv_opt.value()[], res_ptr[])


struct AssetPlugin(EnginePlugin, Copyable, ImplicitlyCopyable, Movable):
    """Registers `AssetServer` + `SceneAssetResolver` and ticks streaming/hot-reload & ECS resolution."""
    var loose_root: String
    var enable_watcher: Bool
    var enable_resolver: Bool

    def __init__(
        out self,
        loose_root: String = "",
        enable_watcher: Bool = False,
        enable_resolver: Bool = True,
    ):
        self.loose_root = loose_root
        self.enable_watcher = enable_watcher
        self.enable_resolver = enable_resolver

    def build(self, mut app: EngineApp):
        if not app.resources.has[AssetServer]():
            var srv = AssetServer(enable_watcher=self.enable_watcher)
            if self.loose_root.byte_length() > 0:
                if self.enable_watcher:
                    _ = srv.watch_loose_directory(self.loose_root)
                else:
                    srv.loose_root = self.loose_root
                    srv.index_loose_directory()
            app.resources.insert(srv^)
        else:
            var existing = app.resources.get_ptr_mut[AssetServer]()
            if existing and self.loose_root.byte_length() > 0:
                if self.enable_watcher:
                    _ = existing.value()[].watch_loose_directory(self.loose_root)
                else:
                    existing.value()[].loose_root = self.loose_root
                    existing.value()[].index_loose_directory()
        if self.enable_resolver and not app.resources.has[SceneAssetResolver]():
            app.resources.insert(SceneAssetResolver())
        _ = app.add_system(STAGE_STARTUP, "asset_tick_resolve_startup", _sys_asset_tick_and_resolve)
        _ = app.add_system(STAGE_PRE_UPDATE, "asset_tick_resolve_pre", _sys_asset_tick_and_resolve)

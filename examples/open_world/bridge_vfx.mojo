"""Attach the authored stone dust burst to the performer's animated foot contacts."""
from flint.anim.plugin import AnimAssetCache, AnimEventsResource, SkeletonComponent
from flint.app import EngineApp, Resource, Resources
from flint.assets.server import AssetServer
from flint.core.math import Vec3
from flint.ecs import Entity, World, comp_hash64
from flint.render.aaa_gpu_particles import AaaParticleEmitter
from flint.render.particle_effect import load_particle_burst
from flint.render.render_world import RenderWorld
from flint.scene import GlobalTransform


struct BridgeDust(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("example.sunspire.BridgeDust")
    var effect: AaaParticleEmitter
    var cursor: Int
    var bursts: Int

    def __init__(out self, var effect: AaaParticleEmitter):
        self.effect = effect^
        self.cursor = 0
        self.bursts = 0


def load_bridge_dust(mut app: EngineApp) raises:
    ref assets = app.resources.get_ptr_mut[AssetServer]().value()[]
    var handle = assets.load_raw("bridge_dust.json")
    var raw = assets.raws.get_ptr(handle)
    if not raw:
        raise Error("Missing bridge_dust.json")
    var bytes = raw.value()[].bytes.copy()
    _ = assets.raws.release(handle)
    app.insert_resource(BridgeDust(load_particle_burst(bytes)))


def extract_bridge_dust(world: World, mut resources: Resources, mut frame: RenderWorld, actor: Entity, walking: Bool):
    var opt = resources.get_ptr_mut[BridgeDust]()
    var events = resources.get_ptr[AnimEventsResource]()
    if not opt or not events:
        return
    ref dust = opt.value()[]
    var records = events.value()[].events.read_since(dust.cursor)
    if not walking or not len(records):
        return
    var sk = world.get_ptr[SkeletonComponent](actor)
    var gt = world.get_ptr[GlobalTransform](actor)
    if not sk or not gt:
        return
    ref cache = resources.get_ptr[AnimAssetCache]().value()[]
    var slot = cache.find_model(sk.value()[].model_id)
    if slot < 0:
        return
    for record in records:
        if record.entity.to_bits() != actor.to_bits():
            continue
        var left = record.event.kind == "footstep_L"
        if not left and record.event.kind != "footstep_R":
            continue
        var node = cache.models[slot].find_node("Foot.L" if left else "Foot.R")
        ref pose = sk.value()[].skeleton.world_matrices
        if node < 0 or node >= len(pose):
            continue
        var burst = dust.effect.copy()
        burst.origin = gt.value()[].transform_point(pose[node].transform_point(Vec3()))
        burst.bounce_ground_y = burst.origin.y
        frame.particle_emitters.append(burst^)
        dust.bursts += 1

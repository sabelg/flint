"""Rebuild the checked-in binary graph from the editable JSON and GLB clip order."""
from flint.anim.flanim_ecs import save_flanim_bytes
from flint.anim.graph import read_anim_graph
from flint.assets.glb_native import load_native_glb
from flint.platform.os import write_bytes
from flint.anim.plugin import RagdollComponent, register_anim_codecs
from flint.ecs import World
from flint.scene import ComponentRegistry, Scene
from course import make_ragdoll

def main() raises:
    var folder = String("examples/character_course/assets/")
    var model = load_native_glb(folder + "performer.glb")
    var names = List[String]()
    for anim in model.animations:
        names.append(anim.name)
    var graph = read_anim_graph(folder + "locomotion.json", names)
    var data = save_flanim_bytes(graph)
    if not write_bytes(folder + "locomotion.flanim", data):
        raise Error("Failed to write the course graph")

    var world = World()
    var registry = ComponentRegistry.with_builtins()
    register_anim_codecs(registry)
    registry.ensure_registered_in_world(world)
    var actor = world.spawn()
    var rig = make_ragdoll(model)
    _ = world.insert[RagdollComponent](actor, RagdollComponent(rig^, auto_recover=False))
    var scene = Scene("Bridge performer physics")
    if not scene.save_flscnb_file(folder + "bridge_rig.flscnb", world, registry):
        raise Error("Failed to write the course physical rig")

"""Physical rig shared by the playable character and the recovery benchmark."""
from flint.anim.ragdoll import ArticulatedRagdoll
from flint.assets.model import NativeModel
from flint.core.math import Vec3


def physical_performer(model: NativeModel) raises -> ArticulatedRagdoll:
    var rig = ArticulatedRagdoll()
    rig.bind_skeleton(model)
    for node in ["Root", "Spine", "Chest", "Head", "Hip.L", "Knee.L", "Foot.L", "Hip.R", "Knee.R", "Foot.R",
                 "Shoulder.L", "Elbow.L", "Hand.L", "Shoulder.R", "Elbow.R", "Hand.R"]:
        if model.find_node(String(node)) < 0:
            raise Error("Performer is missing bone " + String(node))
    var pelvis = rig.add_limb_from_model(model, model.find_node("Root"), -1, 9, 0.13, 0.18, Vec3(0, 1, 0))
    var spine = rig.add_limb_from_model(model, model.find_node("Spine"), pelvis, 6, 0.12)
    var chest = rig.add_limb_from_model(model, model.find_node("Chest"), spine, 10, 0.14)
    _ = rig.add_limb_from_model(model, model.find_node("Head"), chest, 4, 0.10, 0.14, Vec3(0, 1, 0))
    for side in [String(".L"), String(".R")]:
        var hip = rig.add_limb_from_model(model, model.find_node("Hip" + side), pelvis, 7, 0.08)
        var knee = rig.add_limb_from_model(model, model.find_node("Knee" + side), hip, 4, 0.065, is_hinge=True)
        _ = rig.add_limb_from_model(model, model.find_node("Foot" + side), knee, 1, 0.05, 0.16, Vec3(0, 0, 1))
        var shoulder = rig.add_limb_from_model(model, model.find_node("Shoulder" + side), chest, 2.5, 0.06)
        # The performer flexes its elbows about local -X and its knees about +X.
        var elbow = rig.add_limb_from_model(
            model, model.find_node("Elbow" + side), shoulder, 1.4, 0.05,
            is_hinge=True, hinge_axis_local=Vec3(-1, 0, 0),
        )
        _ = rig.add_limb_from_model(model, model.find_node("Hand" + side), elbow, 0.5, 0.04, 0.08, Vec3(0, -1, 0))
    return rig^

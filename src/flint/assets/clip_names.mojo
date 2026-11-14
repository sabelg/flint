"""The clips the game needs, by name. The game's clip numbers are positions in
these lists; everything else about a clip comes from its manifest. A form's
mesh is refused when one of these is missing. (Echo's list left with her in
package 3.21; her export stays in assets/deepsong/characters, out of the build.)

Maroon's list only grows at its end: the simulation stores clip numbers
(`Game.action_clip`), so a name's position is part of saves and replays.
Package 3.6 added Sprint (the gallop on all fours; Run is the upright run),
Swim, Surf, Idle_Sniff, Idle_Scratch, Idle_Look and Slam_Dive. Package 3.7
added Climb_Down (down a trunk or a wall),
Scramble (on all fours up steep ground) and Ceiling (hand over hand). Package
3.8 added the new bear skills: Roar, Clap, Belly_Slide, Burrow, Bear_Hug,
Tree_Shake, Uproot, Sniff (nose to the ground; Idle_Sniff stands) and
Trunk_Swing.

Maroon_Moves is a second clip file on the same skeleton: every v6 move whole,
for the packages that cut skills from them (tools/blender/ds_moves_v6.py).
Since package 3.23 the game loads it once, bound to no mesh, and Maroon's
acting plays windows of its moves and its face poses from there
(gameplay/acting.mojo, anim/acting.mojo): the simulation numbers none of them.
`maroon_move_names` lists the moves the plan and the acting count on, so an
export that lost one is refused."""


def maroon_clip_names() -> List[String]:
    var names = List[String]()
    for name in [
        "Idle", "Idle_Variation", "Walk", "Run", "Roll", "Jump", "Fall",
        "Glide", "Climb", "Landing", "Punch_1", "Punch_2", "Punch_3",
        "Tear", "Carry", "Throw", "Absorb", "Place", "Ground_Slam",
        "Transform_Activate", "Transform_Cancel", "Hurt", "Defeat",
        "Recovery", "Interaction", "Victory",
        "Sprint", "Swim", "Surf", "Idle_Sniff", "Idle_Scratch", "Idle_Look", "Slam_Dive",
        "Climb_Down", "Scramble", "Ceiling",
        "Roar", "Clap", "Belly_Slide", "Burrow", "Bear_Hug", "Tree_Shake", "Uproot", "Sniff", "Trunk_Swing",
    ]:
        names.append(name)
    return names^


def maroon_move_names() -> List[String]:
    """Clips of Maroon_Moves the plan counts on: skills for 3.7, 3.8 and 3.1,
    then sitting, sleeping, eating and gestures for idles and the 5.4
    sequencer, then the face poses."""
    var names = List[String]()
    for name in [
        "climb", "climb_rock", "climb_down", "roar", "clap", "dig", "dig_ground", "grapple", "shove",
        "push_heavy", "sneak", "crawl", "crawl_under", "dodge", "kick", "block", "lift_overhead",
        "gallop_fours", "walk_fours", "wade", "carry_heavy", "throw", "pull_rope",
        "sit", "sit_down", "sit_stand", "stand_from_sit", "lie_down", "lie_sleep", "get_up",
        "eat_paw", "drink_paws", "forage", "wave_hello", "nod_yes", "shake_no", "shrug",
        "point_forward", "beckon", "think", "cheer", "chest_beat", "bow",
        "face_angry", "face_blink", "face_happy", "face_look_left", "face_roar", "face_sad",
        "face_sleepy", "face_smirk", "face_snarl", "face_surprise",
        # Package 3.23: the acting plays these straight from the file (gameplay/acting.mojo `act_defs`).
        "stretch", "shiver", "sneeze", "shake_off", "cover_ears", "facepalm", "belly_laugh", "dance", "skip_happy",
        "march_proud", "angry_stomp", "huff", "cry", "scared_look", "back_away", "wipe_brow", "cross_arms",
        "look_around", "hind_scratch", "rub_back_tree", "scent_mark", "walk_slow", "limp", "step_over", "stumble",
    ]:
        names.append(name)
    return names^


def needed_clips(asset: String) -> List[String]:
    if asset == "Maroon_Anim":
        return maroon_clip_names()
    if asset == "Maroon_Moves":
        return maroon_move_names()
    return List[String]()


def clip_event_kinds() -> List[String]:
    """Event markers the game reads from a clip (Blender markers event:<kind>).
    tools/blender/pipeline.py refuses any other kind, and the Blender add-on's
    event palette lists these from build/game_metrics.json."""
    var kinds = List[String]()
    for kind in ["footstep_L", "footstep_R", "hit", "vfx", "sfx", "cancel_from", "cancel_until"]:
        kinds.append(kind)
    return kinds^


def known_event_kind(kind: String) -> Bool:
    for known in clip_event_kinds():
        if known == kind:
            return True
    return False

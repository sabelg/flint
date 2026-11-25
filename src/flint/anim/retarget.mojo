"""Skeleton retargeting for Flint (`flint.anim.retarget`).

Maps motion between humanoid skeletons with different proportions, joint counts and bone names
(SOMA / MotionBricks / GEM-X clips, Mixamo / Make-It-Animatable rigs, UniRig's `bone_N` rigs):

  1. Role detection (`detect_humanoid_roles`): every skeleton is labelled with canonical humanoid
     roles. Names are tried first (normalised: prefixes such as `mixamorig:` / `DEF-` stripped,
     side from left/right/l/r tokens, SOMA and Mixamo spellings); whatever the names leave open
     is solved from topology and rest geometry - hips = the first joint whose subtree splits into
     legs + spine, legs = the two lowest-reaching branches, chest = the spine joint where the
     arms branch off, knees / elbows by path length, ankles where the leg chain stops going
     down, hands where the arm chain fans into fingers, left = up x forward with forward from the
     toes.
  2. Motion transfer (`retarget`): for each mapped role the source's world-space rotation delta
     from its rest pose is applied to the target's rest, after a per-bone rest alignment (the
     shortest arc that turns the target's rest bone direction onto the source's), so A-pose and
     T-pose rigs map onto each other. Unmapped target joints keep their rest local rotation.
     Root motion is scaled by the hip-height ratio.
  3. IK correction: feet (and optionally hands) are moved onto the source effector paths scaled
     about the ground origin, with an analytic two-bone solve (law of cosines in the FK knee
     plane, exact whenever the target is in reach), keeping the FK foot orientation. Contact
     frames are detected from the source (low and slow feet) and pinned so planted feet do not
     slide.

`foot_contact_error` measures what matters for retargeting quality: how far each target foot
drifts during the source's contact phases, and its height error against the scaled source.
"""
from std.math import abs, max, min, sqrt
from flint.core.math import Quat, Vec3
from flint.core.ik import shortest_arc_quat
from flint.assets.model import NativeAnimation, NativeChannel, NativeModel


# ------------------------------------------------------------------------------- roles

comptime R_HIPS: Int = 0
comptime R_SPINE: Int = 1
comptime R_CHEST: Int = 2
comptime R_NECK: Int = 3
comptime R_HEAD: Int = 4
comptime R_L_SHOULDER: Int = 5
comptime R_L_UPPER_ARM: Int = 6
comptime R_L_LOWER_ARM: Int = 7
comptime R_L_HAND: Int = 8
comptime R_R_SHOULDER: Int = 9
comptime R_R_UPPER_ARM: Int = 10
comptime R_R_LOWER_ARM: Int = 11
comptime R_R_HAND: Int = 12
comptime R_L_UPPER_LEG: Int = 13
comptime R_L_LOWER_LEG: Int = 14
comptime R_L_FOOT: Int = 15
comptime R_L_TOES: Int = 16
comptime R_R_UPPER_LEG: Int = 17
comptime R_R_LOWER_LEG: Int = 18
comptime R_R_FOOT: Int = 19
comptime R_R_TOES: Int = 20
comptime ROLE_COUNT: Int = 21


def role_name(r: Int) -> String:
    var names = [
        String("Hips"), "Spine", "Chest", "Neck", "Head",
        "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
        "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
        "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
    ]
    return names[r] if r >= 0 and r < ROLE_COUNT else String("?")


def _role_parent(r: Int) -> Int:
    """The role whose bone ends at role r's joint (for rest alignment), -1 for none."""
    if r == R_SPINE or r == R_L_UPPER_LEG or r == R_R_UPPER_LEG:
        return R_HIPS
    if r == R_CHEST:
        return R_SPINE
    if r == R_NECK or r == R_L_SHOULDER or r == R_R_SHOULDER:
        return R_CHEST
    if r == R_HEAD:
        return R_NECK
    if r == R_L_UPPER_ARM or r == R_L_LOWER_ARM or r == R_L_HAND:
        return r - 1
    if r == R_R_UPPER_ARM or r == R_R_LOWER_ARM or r == R_R_HAND:
        return r - 1
    if r == R_L_LOWER_LEG or r == R_L_FOOT or r == R_L_TOES or r == R_R_LOWER_LEG or r == R_R_FOOT or r == R_R_TOES:
        return r - 1
    return -1


# ------------------------------------------------------------------------------- skeleton / clip

struct RetargetSkeleton(Copyable, Movable):
    var names: List[String]
    var parents: List[Int]       # parents precede children
    var rest_t: List[Vec3]       # local translation
    var rest_r: List[Quat]       # local rotation
    var node_of_joint: List[Int] # source NativeModel node per joint (-1 when built by hand)

    def __init__(out self):
        self.names = List[String]()
        self.parents = List[Int]()
        self.rest_t = List[Vec3]()
        self.rest_r = List[Quat]()
        self.node_of_joint = List[Int]()

    def count(self) -> Int:
        return len(self.names)

    def add(mut self, name: String, parent: Int, t: Vec3, r: Quat = Quat()) -> Int:
        self.names.append(name)
        self.parents.append(parent)
        self.rest_t.append(t)
        self.rest_r.append(r)
        self.node_of_joint.append(-1)
        return len(self.names) - 1

    def index_of(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def world(self, local_r: List[Quat], root_t: Vec3, mut out_p: List[Vec3], mut out_r: List[Quat]):
        """World positions / rotations for local rotations (root translation overridden)."""
        var n = self.count()
        out_p = List[Vec3](length=n, fill=Vec3(0, 0, 0))
        out_r = List[Quat](length=n, fill=Quat())
        for j in range(n):
            var p = self.parents[j]
            if p < 0:
                out_p[j] = root_t
                out_r[j] = local_r[j]
            else:
                out_p[j] = out_p[p] + out_r[p].rotate(self.rest_t[j])
                out_r[j] = out_r[p].multiply(local_r[j]).normalized()

    def rest_world(self, mut out_p: List[Vec3], mut out_r: List[Quat]):
        var root = Vec3(0, 0, 0)
        for j in range(self.count()):
            if self.parents[j] < 0:
                root = self.rest_t[j]
                break
        self.world(self.rest_r, root, out_p, out_r)


def skeleton_from_model(model: NativeModel, skin: Int = 0) -> RetargetSkeleton:
    """The joints of `model.skins[skin]` (topologically ordered) with node rest TRS."""
    var s = RetargetSkeleton()
    if skin >= len(model.skins):
        return s^
    ref joints = model.skins[skin].joints
    var parent_node = List[Int](length=len(model.nodes), fill=-1)
    for n in range(len(model.nodes)):
        for c in model.nodes[n].children:
            parent_node[c] = n
    var in_skin = List[Int](length=len(model.nodes), fill=-1)
    for j in range(len(joints)):
        in_skin[joints[j]] = j
    # order: repeatedly emit joints whose joint-parent is already emitted
    var emitted = List[Int](length=len(joints), fill=-1)
    var remaining = len(joints)
    var guard = 0
    while remaining > 0 and guard < len(joints) + 2:
        guard += 1
        for j in range(len(joints)):
            if emitted[j] >= 0:
                continue
            var nd = joints[j]
            var pn = parent_node[nd]
            while pn >= 0 and in_skin[pn] < 0:
                pn = parent_node[pn]
            var pj = in_skin[pn] if pn >= 0 else -1
            if pj >= 0 and emitted[pj] < 0:
                continue
            var t = model.nodes[nd].base_translation
            var r = model.nodes[nd].base_rotation
            var idx = s.add(model.nodes[nd].name, emitted[pj] if pj >= 0 else -1, t, r)
            s.node_of_joint[idx] = nd
            emitted[j] = idx
            remaining -= 1
    return s^


struct RetargetClip(Copyable, Movable):
    var skeleton: RetargetSkeleton
    var sample_rate: Float32
    var rotations: List[List[Quat]]   # [frame][joint] local rotations
    var root: List[Vec3]              # [frame] root joint translation (parent space)

    def __init__(out self, var skeleton: RetargetSkeleton, sample_rate: Float32):
        self.skeleton = skeleton^
        self.sample_rate = sample_rate
        self.rotations = List[List[Quat]]()
        self.root = List[Vec3]()

    def frame_count(self) -> Int:
        return len(self.rotations)

    def world(self, f: Int, mut p: List[Vec3], mut r: List[Quat]):
        self.skeleton.world(self.rotations[f], self.root[f], p, r)


def clip_to_animation(clip: RetargetClip, name: String = "retarget") -> NativeAnimation:
    """A `NativeAnimation` driving the clip skeleton's `node_of_joint` nodes (rotation per joint,
    translation on the root)."""
    var anim = NativeAnimation()
    anim.name = name
    var n = clip.frame_count()
    var times = List[Float32](capacity=n)
    for f in range(n):
        times.append(Float32(f) / clip.sample_rate)
    anim.duration = Float32(max(n - 1, 0)) / clip.sample_rate
    ref sk = clip.skeleton
    for j in range(sk.count()):
        var nd = sk.node_of_joint[j]
        if nd < 0:
            continue
        var ch = NativeChannel()
        ch.node = nd
        ch.path = 1
        ch.times = times.copy()
        for f in range(n):
            var q = clip.rotations[f][j]
            ch.values.append(q.x); ch.values.append(q.y); ch.values.append(q.z); ch.values.append(q.w)
        anim.channels.append(ch^)
        if sk.parents[j] < 0:
            var tc = NativeChannel()
            tc.node = nd
            tc.path = 0
            tc.times = times.copy()
            for f in range(n):
                var p = clip.root[f]
                tc.values.append(p.x); tc.values.append(p.y); tc.values.append(p.z)
            anim.channels.append(tc^)
    return anim^


# ------------------------------------------------------------------------------- name matching

def _lower_alnum(s: String) -> String:
    var out = String()
    for i in range(s.byte_length()):
        var c = Int(s.as_bytes()[i])
        if c >= 65 and c <= 90:
            out += chr(c + 32)
        elif (c >= 97 and c <= 122) or (c >= 48 and c <= 57):
            out += chr(c)
        else:
            out += " "
    return out


def _tokens(name: String) -> List[String]:
    """Lower-case word tokens, splitting camelCase, digits and separators."""
    var out = List[String]()
    var cur = String()
    var prev_lower = False
    for i in range(name.byte_length()):
        var c = Int(name.as_bytes()[i])
        var is_up = c >= 65 and c <= 90
        var is_lo = c >= 97 and c <= 122
        var is_dg = c >= 48 and c <= 57
        if is_up and prev_lower and cur.byte_length() > 0:
            out.append(cur)
            cur = String()
        if is_up:
            cur += chr(c + 32)
        elif is_lo:
            cur += chr(c)
        elif is_dg:
            if cur.byte_length() > 0 and not prev_lower and False:
                pass
            cur += chr(c)
        else:
            if cur.byte_length() > 0:
                out.append(cur)
            cur = String()
        prev_lower = is_lo or is_dg
    if cur.byte_length() > 0:
        out.append(cur)
    return out^


def _has(toks: List[String], w: String) -> Bool:
    for t in toks:
        if t == w:
            return True
    return False


def _has_prefix(toks: List[String], w: String) -> Bool:
    for t in toks:
        if t.startswith(w):
            return True
    return False


def name_side(name: String) -> Int:
    """+1 left, -1 right, 0 unknown, from name tokens (left/l/lf, right/r/rt, `.L`, `_r`...)."""
    var toks = _tokens(name)
    for t in toks:
        if t == "left" or t == "l" or t == "lf" or t == "lft":
            return 1
        if t == "right" or t == "r" or t == "rt" or t == "rgt":
            return -1
    # glued forms such as "lthigh" / "rhand"
    if len(toks) > 0:
        var t0 = toks[0]
        if t0.byte_length() > 3 and (t0.startswith("l") or t0.startswith("r")):
            var rest = String(t0[byte=1:])
            for k in ["thigh", "calf", "shin", "foot", "toe", "hand", "arm", "forearm", "shoulder", "clavicle", "leg", "knee", "elbow", "wrist", "ankle", "upleg", "uparm"]:
                if rest.startswith(k):
                    return 1 if t0.startswith("l") else -1
    return 0


def _name_role(name: String) -> Int:
    """A side-free role guess from the name: 0 hips, 1 spine, 2 chest, 3 neck, 4 head,
    5 shoulder, 6 upper arm, 7 lower arm, 8 hand, 13 upper leg, 14 lower leg, 15 foot,
    16 toes, -2 ambiguous 'leg' (SOMA thigh / Mixamo shin), -1 unknown."""
    var toks = _tokens(name)
    var low = _lower_alnum(name)
    _ = low
    if _has(toks, "hips") or _has(toks, "pelvis") or _has(toks, "hip") and not _has(toks, "left") and not _has(toks, "right"):
        return R_HIPS
    if _has_prefix(toks, "toe") or _has(toks, "ball"):
        return 16
    if _has(toks, "foot") or _has(toks, "ankle"):
        return 15
    if _has(toks, "shin") or _has(toks, "calf") or _has(toks, "knee") or _has(toks, "lowerleg"):
        return 14
    if _has(toks, "thigh") or _has(toks, "upleg") or _has(toks, "upperleg"):
        return 13
    if _has(toks, "up") and _has(toks, "leg"):
        return 13
    if _has(toks, "lower") and _has(toks, "leg"):
        return 14
    if _has(toks, "upper") and _has(toks, "leg"):
        return 13
    if _has(toks, "leg"):
        return -2
    if _has(toks, "hand") or _has(toks, "wrist"):
        return 8
    if _has(toks, "forearm") or _has(toks, "elbow") or _has(toks, "lowerarm") or (_has(toks, "fore") and _has(toks, "arm")) or (_has(toks, "lower") and _has(toks, "arm")):
        return 7
    if _has(toks, "upperarm") or _has(toks, "uparm") or (_has(toks, "upper") and _has(toks, "arm")) or _has(toks, "arm"):
        return 6
    if _has(toks, "shoulder") or _has(toks, "clavicle") or _has(toks, "collar"):
        return 5
    if _has(toks, "head"):
        return 4
    if _has_prefix(toks, "neck"):
        return 3
    if _has_prefix(toks, "chest") or _has(toks, "upperchest") or _has(toks, "thorax"):
        return 2
    if _has_prefix(toks, "spine") or _has(toks, "abdomen") or _has(toks, "torso") or _has(toks, "waist"):
        return 1
    return -1


# ------------------------------------------------------------------------------- role detection

struct HumanoidMap(Copyable, Movable, Writable):
    var joint: List[Int]   # per role: joint index or -1
    var from_names: Int    # roles settled by name
    var from_topology: Int # roles settled by topology/geometry
    var forward: Vec3
    var up: Vec3

    def __init__(out self):
        self.joint = List[Int](length=ROLE_COUNT, fill=-1)
        self.from_names = 0
        self.from_topology = 0
        self.forward = Vec3(0, 0, 1)
        self.up = Vec3(0, 1, 0)

    def mapped(self) -> Int:
        var n = 0
        for r in range(ROLE_COUNT):
            if self.joint[r] >= 0:
                n += 1
        return n

    def has_legs(self) -> Bool:
        return self.joint[R_L_UPPER_LEG] >= 0 and self.joint[R_L_LOWER_LEG] >= 0 and self.joint[R_L_FOOT] >= 0 and self.joint[R_R_UPPER_LEG] >= 0 and self.joint[R_R_LOWER_LEG] >= 0 and self.joint[R_R_FOOT] >= 0

    def write_to[W: Writer](self, mut writer: W):
        writer.write("roles=", self.mapped(), "/", ROLE_COUNT, " (names ", self.from_names, ", topology ", self.from_topology, ")")


def _children(s: RetargetSkeleton) -> List[List[Int]]:
    var out = List[List[Int]]()
    for _ in range(s.count()):
        out.append(List[Int]())
    for j in range(s.count()):
        if s.parents[j] >= 0:
            out[s.parents[j]].append(j)
    return out^


def _subtree_size(ch: List[List[Int]], j: Int) -> Int:
    var n = 0
    var stack = List[Int]()
    stack.append(j)
    while len(stack) > 0:
        var x = stack.pop()
        n += 1
        for c in ch[x]:
            stack.append(c)
    return n


def _subtree_extreme(ch: List[List[Int]], p: List[Vec3], j: Int, up: Vec3, lowest: Bool) -> Float32:
    var v = p[j].dot(up)
    var stack = List[Int]()
    stack.append(j)
    while len(stack) > 0:
        var x = stack.pop()
        var h = p[x].dot(up)
        v = min(v, h) if lowest else max(v, h)
        for c in ch[x]:
            stack.append(c)
    return v


def _deepest_chain(ch: List[List[Int]], j: Int) -> List[Int]:
    """j followed by its descendants along the largest subtree at each step."""
    var out = List[Int]()
    var cur = j
    while True:
        out.append(cur)
        if len(ch[cur]) == 0:
            break
        var best = ch[cur][0]
        var bs = _subtree_size(ch, best)
        for c in ch[cur]:
            var s = _subtree_size(ch, c)
            if s > bs:
                bs = s
                best = c
        cur = best
    return out^


def detect_humanoid_roles(s: RetargetSkeleton) -> HumanoidMap:
    var m = HumanoidMap()
    var n = s.count()
    if n == 0:
        return m^
    var p = List[Vec3]()
    var r = List[Quat]()
    s.rest_world(p, r)
    var ch = _children(s)
    var up = Vec3(0, 1, 0)
    # ---- names
    var name_roles = List[Int](length=n, fill=-1)
    var sides = List[Int](length=n, fill=0)
    for j in range(n):
        name_roles[j] = _name_role(s.names[j])
        sides[j] = name_side(s.names[j])
    # ---- topology: hips = first joint (from the root) with >= 3 significant branches, or 2
    #      downward branches + one upward
    var hips = -1
    for j in range(n):
        var big = 0
        for c in ch[j]:
            if _subtree_size(ch, c) >= 3:
                big += 1
        if big >= 3:
            hips = j
            break
    if hips < 0:
        for j in range(n):
            if name_roles[j] == R_HIPS:
                hips = j
                break
    if hips < 0:
        hips = 0
    m.joint[R_HIPS] = hips
    var hip_h = p[hips].dot(up)
    # branches of the hips: legs are the two that reach lowest, spine the highest
    var branches = ch[hips].copy()
    var legs = List[Int]()
    var spine_root = -1
    var best_hi: Float32 = -3.0e38
    var lows = List[Float32]()
    for b in branches:
        lows.append(_subtree_extreme(ch, p, b, up, True))
    for _ in range(2):
        var bi = -1
        for i in range(len(branches)):
            var used = False
            for l in legs:
                if l == branches[i]:
                    used = True
            if used or _subtree_size(ch, branches[i]) < 3:
                continue
            if lows[i] < hip_h and (bi < 0 or lows[i] < lows[bi]):
                bi = i
        if bi >= 0:
            legs.append(branches[bi])
    for b in branches:
        var is_leg = False
        for l in legs:
            if l == b:
                is_leg = True
        if is_leg:
            continue
        var hi = _subtree_extreme(ch, p, b, up, False)
        if hi > best_hi:
            best_hi = hi
            spine_root = b
    # forward from the feet: ankle -> toe direction, horizontal
    var leg_chains = List[List[Int]]()
    for l in legs:
        leg_chains.append(_deepest_chain(ch, l))
    var fwd = Vec3(0, 0, 0)
    for lc in leg_chains:
        if len(lc) >= 2:
            var d = p[lc[len(lc) - 1]] - p[lc[len(lc) - 2]]
            fwd = fwd + Vec3(d.x, 0, d.z)
    var fl = fwd.length()
    m.forward = fwd * (1.0 / fl) if fl > 1e-6 else Vec3(0, 0, 1)
    var left_axis = up.cross(m.forward)
    # legs: assign side (names win), then joints along the chain
    for li in range(len(leg_chains)):
        var lc = leg_chains[li].copy()
        var side = 0
        for j in lc:
            if sides[j] != 0:
                side = sides[j]
                break
        if side == 0:
            side = 1 if (p[lc[0]] - p[hips]).dot(left_axis) + (p[lc[len(lc) - 1]] - p[hips]).dot(left_axis) > 0 else -1
        if li == 1 and len(leg_chains) == 2:
            # make the two legs opposite
            var other = m.joint[R_L_UPPER_LEG] if m.joint[R_L_UPPER_LEG] >= 0 else m.joint[R_R_UPPER_LEG]
            if other >= 0 and ((side == 1 and m.joint[R_L_UPPER_LEG] >= 0) or (side == -1 and m.joint[R_R_UPPER_LEG] >= 0)):
                side = -side
        var base = R_L_UPPER_LEG if side == 1 else R_R_UPPER_LEG
        _assign_leg(m, lc, p, up, name_roles, base)
    # spine chain up to the arm branch point
    if spine_root >= 0:
        var cur = spine_root
        var spine_joints = List[Int]()
        var chest = -1
        while True:
            spine_joints.append(cur)
            var big = List[Int]()
            for c in ch[cur]:
                if _subtree_size(ch, c) >= 2:
                    big.append(c)
            if len(big) >= 3 or (len(big) == 2 and _lateral_pair(p, big, cur, left_axis)):
                chest = cur
                break
            if len(ch[cur]) == 0:
                break
            var nxt = ch[cur][0]
            var hi: Float32 = -3.0e38
            for c in ch[cur]:
                var e = _subtree_extreme(ch, p, c, up, False)
                if e > hi:
                    hi = e
                    nxt = c
            cur = nxt
        m.joint[R_SPINE] = spine_joints[0]
        m.from_topology += 1
        if chest >= 0:
            m.joint[R_CHEST] = chest
            m.from_topology += 1
            # neck/head: the branch going up; arms: the two most lateral
            var arm_br = List[Int]()
            var neck_br = -1
            var neck_hi: Float32 = -3.0e38
            for c in ch[chest]:
                var e = _subtree_extreme(ch, p, c, up, False)
                var lat = abs((p[_deepest_chain(ch, c)[len(_deepest_chain(ch, c)) - 1]] - p[chest]).dot(left_axis))
                _ = lat
                if e > neck_hi:
                    neck_hi = e
                    neck_br = c
            for c in ch[chest]:
                if c != neck_br and _subtree_size(ch, c) >= 2:
                    arm_br.append(c)
            if neck_br >= 0:
                var nc = _deepest_chain(ch, neck_br)
                m.joint[R_NECK] = nc[0]
                # head: the joint just before the chain tip when the tip is a short end bone
                var head = nc[len(nc) - 1]
                if len(nc) >= 3:
                    head = nc[len(nc) - 2] if name_roles[nc[len(nc) - 1]] != 4 else nc[len(nc) - 1]
                for j in nc:
                    if name_roles[j] == 4:
                        head = j
                        break
                if head == nc[0] and len(nc) > 1:
                    head = nc[1]
                m.joint[R_HEAD] = head
                m.from_topology += 2
            # keep the two arms most opposite along left_axis
            for c in arm_br:
                var ac = _deepest_chain(ch, c)
                var side = 0
                for j in ac:
                    if sides[j] != 0:
                        side = sides[j]
                        break
                if side == 0:
                    side = 1 if (p[ac[len(ac) - 1]] - p[chest]).dot(left_axis) > 0 else -1
                var base = R_L_SHOULDER if side == 1 else R_R_SHOULDER
                if m.joint[base + 1] < 0:
                    _assign_arm(m, ac, p, ch, name_roles, base)
    return m^


def _lateral_pair(p: List[Vec3], big: List[Int], cur: Int, left_axis: Vec3) -> Bool:
    var a = (p[big[0]] - p[cur]).dot(left_axis)
    var b = (p[big[1]] - p[cur]).dot(left_axis)
    return a * b < 0 and abs(a) > 1e-4 and abs(b) > 1e-4


def _assign_leg(mut m: HumanoidMap, lc: List[Int], p: List[Vec3], up: Vec3, name_roles: List[Int], base: Int):
    # names first: thigh/shin/foot/toe tokens (SOMA "Leg" = thigh when a "Shin" follows)
    var thigh = -1
    var shin = -1
    var foot = -1
    var toe = -1
    var has_shin_name = False
    for j in lc:
        if name_roles[j] == 14:
            has_shin_name = True
    for j in lc:
        var nr = name_roles[j]
        if nr == 13 or (nr == -2 and has_shin_name):
            if thigh < 0:
                thigh = j
        elif nr == 14 or (nr == -2 and not has_shin_name and thigh >= 0):
            if shin < 0:
                shin = j
        elif nr == 15:
            if foot < 0:
                foot = j
        elif nr == 16:
            if toe < 0:
                toe = j
    var named = (1 if thigh >= 0 else 0) + (1 if shin >= 0 else 0) + (1 if foot >= 0 else 0) + (1 if toe >= 0 else 0)
    if thigh < 0 or shin < 0 or foot < 0:
        # geometry: the downward run of the chain
        var start = 0
        while start + 1 < len(lc):
            var d = p[lc[start + 1]] - p[lc[start]]
            var l = d.length()
            if l > 1e-6 and -d.dot(up) > 0.6 * l:
                break
            start += 1
        var end_ = start
        while end_ + 1 < len(lc):
            var d = p[lc[end_ + 1]] - p[lc[end_]]
            var l = d.length()
            if l > 1e-6 and -d.dot(up) > 0.5 * l:
                end_ += 1
            else:
                break
        var a = start
        var c = end_
        # knee: the joint nearest the middle of the downward path
        var total: Float32 = 0
        for i in range(a, c):
            total += (p[lc[i + 1]] - p[lc[i]]).length()
        var acc: Float32 = 0
        var best = a + 1
        var best_d: Float32 = 3.0e38
        for i in range(a + 1, c):
            acc += (p[lc[i]] - p[lc[i - 1]]).length()
            var dd = abs(acc - total * 0.5)
            if dd < best_d:
                best_d = dd
                best = i
        if c - a >= 2:
            thigh = lc[a]
            shin = lc[best]
            foot = lc[c]
            toe = lc[c + 1] if c + 1 < len(lc) else -1
            m.from_topology += 3 + (1 if toe >= 0 else 0)
        named = 0
    m.from_names += named
    m.joint[base] = thigh
    m.joint[base + 1] = shin
    m.joint[base + 2] = foot
    m.joint[base + 3] = toe


def _assign_arm(mut m: HumanoidMap, ac: List[Int], p: List[Vec3], ch: List[List[Int]], name_roles: List[Int], base: Int):
    var sh = -1
    var ua = -1
    var la = -1
    var hand = -1
    for j in ac:
        var nr = name_roles[j]
        if nr == 5 and sh < 0:
            sh = j
        elif nr == 6 and ua < 0:
            ua = j
        elif nr == 7 and la < 0:
            la = j
        elif nr == 8 and hand < 0:
            hand = j
    if ua >= 0 and la >= 0 and hand >= 0:
        m.from_names += 3 + (1 if sh >= 0 else 0)
    else:
        # geometry: hand = first joint that fans out (fingers) or the chain tip; a short first
        # bone next to the chest is the clavicle
        var h = len(ac) - 1
        for i in range(1, len(ac)):
            if len(ch[ac[i]]) >= 2:
                h = i
                break
        var s0 = 0
        var reach = (p[ac[h]] - p[ac[0]]).length()
        if len(ac) >= 4 and h >= 3 and (p[ac[1]] - p[ac[0]]).length() < 0.3 * reach:
            s0 = 1
            sh = ac[0]
        if h - s0 < 2:
            return
        var total: Float32 = 0
        for i in range(s0, h):
            total += (p[ac[i + 1]] - p[ac[i]]).length()
        var acc: Float32 = 0
        var best = s0 + 1
        var best_d: Float32 = 3.0e38
        for i in range(s0 + 1, h):
            acc += (p[ac[i]] - p[ac[i - 1]]).length()
            var dd = abs(acc - total * 0.5)
            if dd < best_d:
                best_d = dd
                best = i
        ua = ac[s0]
        la = ac[best]
        hand = ac[h]
        m.from_topology += 3 + (1 if sh >= 0 else 0)
    m.joint[base] = sh
    m.joint[base + 1] = ua
    m.joint[base + 2] = la
    m.joint[base + 3] = hand


# ------------------------------------------------------------------------------- retarget

struct RetargetConfig(Copyable, Movable):
    var foot_ik: Bool
    var hand_ik: Bool
    var contact_height: Float32     # fraction of source hip height
    var contact_speed: Float32      # source hip heights per second
    var pin_contacts: Bool

    def __init__(out self):
        self.foot_ik = True
        self.hand_ik = False
        self.contact_height = 0.06
        self.contact_speed = 0.35
        self.pin_contacts = True


struct RetargetResult(Movable):
    var clip: RetargetClip
    var source_map: HumanoidMap
    var target_map: HumanoidMap
    var scale: Float32
    var ground_offset: Float32
    var src_origin: Vec3            # source floor point under the rest hips
    var tgt_origin: Vec3            # target floor point under the rest hips
    var face: Quat                  # source facing -> target facing
    var ankle_lift: Float32         # target ankle-above-sole minus the scaled source one
    var contacts: List[List[Bool]]  # [foot 0 = left, 1 = right][frame]

    def map_point(self, p: Vec3) -> Vec3:
        """A source world point in the target's world (floor-anchored, scaled, re-faced)."""
        return self.face.rotate(p - self.src_origin) * self.scale + self.tgt_origin

    def __init__(out self, var clip: RetargetClip):
        self.clip = clip^
        self.source_map = HumanoidMap()
        self.target_map = HumanoidMap()
        self.scale = 1
        self.ground_offset = 0
        self.src_origin = Vec3(0, 0, 0)
        self.tgt_origin = Vec3(0, 0, 0)
        self.face = Quat()
        self.ankle_lift = 0
        self.contacts = List[List[Bool]]()


def _bone_dir(m: HumanoidMap, p: List[Vec3], role: Int) -> Tuple[Bool, Vec3]:
    """Rest direction of the bone that starts at `role`: toward its child role's joint."""
    var nxt = -1
    if role == R_HIPS:
        nxt = R_SPINE
    elif role == R_SPINE:
        nxt = R_CHEST
    elif role == R_CHEST:
        nxt = R_NECK
    elif role == R_NECK:
        nxt = R_HEAD
    elif role == R_L_SHOULDER or role == R_L_UPPER_ARM or role == R_L_LOWER_ARM or role == R_R_SHOULDER or role == R_R_UPPER_ARM or role == R_R_LOWER_ARM:
        nxt = role + 1
    elif role == R_L_UPPER_LEG or role == R_L_LOWER_LEG or role == R_L_FOOT or role == R_R_UPPER_LEG or role == R_R_LOWER_LEG or role == R_R_FOOT:
        nxt = role + 1
    if nxt < 0 or m.joint[role] < 0 or m.joint[nxt] < 0:
        return (False, Vec3(0, 0, 0))
    var d = p[m.joint[nxt]] - p[m.joint[role]]
    var l = d.length()
    if l < 1e-8:
        return (False, Vec3(0, 0, 0))
    return (True, d * (1.0 / l))


def _foot_height(m: HumanoidMap, p: List[Vec3], up: Vec3) -> Float32:
    var h: Float32 = 3.0e38
    for r in [R_L_FOOT, R_R_FOOT, R_L_TOES, R_R_TOES]:
        if m.joint[r] >= 0:
            h = min(h, p[m.joint[r]].dot(up))
    return h if h < 1.0e38 else 0


def retarget(src: RetargetClip, target: RetargetSkeleton, cfg: RetargetConfig) -> RetargetResult:
    var sm = detect_humanoid_roles(src.skeleton)
    var tm = detect_humanoid_roles(target)
    var nt = target.count()
    var out = RetargetResult(RetargetClip(target.copy(), src.sample_rate))
    out.source_map = sm.copy()
    out.target_map = tm.copy()
    var up = Vec3(0, 1, 0)
    # rest poses
    var sp0 = List[Vec3]()
    var sr0 = List[Quat]()
    src.skeleton.rest_world(sp0, sr0)
    var tp0 = List[Vec3]()
    var tr0 = List[Quat]()
    target.rest_world(tp0, tr0)
    var sh = sm.joint[R_HIPS]
    var th = tm.joint[R_HIPS]
    var s_ground = _foot_height(sm, sp0, up)
    var t_ground = _foot_height(tm, tp0, up)
    var s_hip_h = max(Float32(1e-4), sp0[sh].dot(up) - s_ground)
    var t_hip_h = max(Float32(1e-4), tp0[th].dot(up) - t_ground)
    var scale = t_hip_h / s_hip_h
    out.scale = scale
    # target joint -> source joint, and per-role rest alignment (target rest dir -> source rest dir)
    var src_of = List[Int](length=nt, fill=-1)
    var align = List[Quat](length=nt, fill=Quat())
    for r in range(ROLE_COUNT):
        var tj = tm.joint[r]
        var sj = sm.joint[r]
        if tj < 0 or sj < 0:
            continue
        src_of[tj] = sj
        var ds = _bone_dir(sm, sp0, r)
        var dt = _bone_dir(tm, tp0, r)
        if ds[0] and dt[0] and r != R_HIPS:
            align[tj] = shortest_arc_quat(dt[1], ds[1])
    # facing alignment: rotate source world deltas into the target's facing frame
    var face = shortest_arc_quat(Vec3(sm.forward.x, 0, sm.forward.z), Vec3(tm.forward.x, 0, tm.forward.z))
    var face_inv = face.conjugate()
    var sp = List[Vec3]()
    var sr = List[Quat]()
    var nf = src.frame_count()
    # contacts from the source
    var feet = [R_L_FOOT, R_R_FOOT]
    var src_feet = List[List[Vec3]]()
    for _ in range(2):
        out.contacts.append(List[Bool](length=nf, fill=False))
        src_feet.append(List[Vec3](length=nf, fill=Vec3(0, 0, 0)))
    var src_hips = List[Vec3](length=nf, fill=Vec3(0, 0, 0))
    var floor_s: Float32 = 3.0e38
    var src_hands = List[List[Vec3]]()
    for _ in range(2):
        src_hands.append(List[Vec3](length=nf, fill=Vec3(0, 0, 0)))
    for f in range(nf):
        src.world(f, sp, sr)
        src_hips[f] = sp[sh]
        for r in [R_L_FOOT, R_R_FOOT, R_L_TOES, R_R_TOES]:
            if sm.joint[r] >= 0:
                floor_s = min(floor_s, sp[sm.joint[r]].dot(up))
        for fi in range(2):
            var j = sm.joint[feet[fi]]
            if j >= 0:
                src_feet[fi][f] = sp[j]
            var hj = sm.joint[R_L_HAND if fi == 0 else R_R_HAND]
            if hj >= 0:
                src_hands[fi][f] = sp[hj]
    if floor_s > 1.0e38:
        floor_s = s_ground
    out.src_origin = Vec3(sp0[sh].x, floor_s, sp0[sh].z)
    out.tgt_origin = Vec3(tp0[th].x, t_ground, tp0[th].z)
    out.face = face
    var dt_s = 1.0 / max(Float32(1e-3), src.sample_rate)
    for fi in range(2):
        if sm.joint[feet[fi]] < 0:
            continue
        # sole clearance: ankle height above the floor minus its standing (rest) clearance
        var rest_clear = sp0[sm.joint[feet[fi]]].dot(up) - s_ground
        for f in range(nf):
            var a = src_feet[fi][max(0, f - 1)]
            var b = src_feet[fi][min(nf - 1, f + 1)]
            var span = Float32(min(nf - 1, f + 1) - max(0, f - 1)) * dt_s
            var v = Vec3(b.x - a.x, 0, b.z - a.z).length() / max(span, Float32(1e-6))
            var h = src_feet[fi][f].dot(up) - floor_s - rest_clear
            out.contacts[fi][f] = h < cfg.contact_height * s_hip_h and v < cfg.contact_speed * s_hip_h
    var ground_off = t_ground - floor_s * scale
    out.ground_offset = ground_off
    # per-frame transfer
    var tw_r = List[Quat](length=nt, fill=Quat())
    var tw_p = List[Vec3](length=nt, fill=Vec3(0, 0, 0))
    var pin = List[Vec3](length=2, fill=Vec3(0, 0, 0))
    var pinned = List[Bool](length=2, fill=False)
    for f in range(nf):
        src.world(f, sp, sr)
        # world rotations: mapped joints take the source delta, others follow their parents
        for j in range(nt):
            var pj = target.parents[j]
            var parent_r = tw_r[pj] if pj >= 0 else Quat()
            var sj = src_of[j]
            if sj >= 0:
                var delta = sr[sj].multiply(sr0[sj].conjugate()).normalized()
                delta = face.multiply(delta).multiply(face_inv).normalized()
                tw_r[j] = delta.multiply(align[j]).multiply(tr0[j]).normalized()
            else:
                tw_r[j] = parent_r.multiply(target.rest_r[j]).normalized()
        # root translation: scaled source hips motion about the ground origin
        var hips_world = out.map_point(src_hips[f])
        # locals and FK
        var local = List[Quat](length=nt, fill=Quat())
        for j in range(nt):
            var pj = target.parents[j]
            local[j] = tw_r[pj].conjugate().multiply(tw_r[j]).normalized() if pj >= 0 else tw_r[j]
        # the root may sit above the hips: place it so the hips land on hips_world
        var root_t = Vec3(0, 0, 0)
        for j in range(nt):
            if target.parents[j] < 0:
                root_t = target.rest_t[j]
                break
        target.world(local, root_t, tw_p, tw_r)
        root_t = root_t + (hips_world - tw_p[th])
        target.world(local, root_t, tw_p, tw_r)
        # IK: feet (and hands) onto the scaled source paths
        if cfg.foot_ik and tm.has_legs() and sm.has_legs():
            for fi in range(2):
                var base = R_L_UPPER_LEG if fi == 0 else R_R_UPPER_LEG
                var goal = out.map_point(src_feet[fi][f])
                goal = goal + up * ((tp0[tm.joint[base + 2]].dot(up) - t_ground) - (sp0[sm.joint[base + 2]].dot(up) - s_ground) * scale)
                if cfg.pin_contacts:
                    if out.contacts[fi][f]:
                        if not pinned[fi]:
                            pin[fi] = goal
                            pinned[fi] = True
                        goal = Vec3(pin[fi].x, goal.y, pin[fi].z)
                    else:
                        pinned[fi] = False
                _two_bone(target, local, root_t, tm.joint[base], tm.joint[base + 1], tm.joint[base + 2], goal, tw_p, tw_r)
        if cfg.hand_ik:
            for fi in range(2):
                var base = R_L_UPPER_ARM if fi == 0 else R_R_UPPER_ARM
                if tm.joint[base] < 0 or sm.joint[base] < 0 or tm.joint[base + 2] < 0:
                    continue
                var goal = out.map_point(src_hands[fi][f])
                _two_bone(target, local, root_t, tm.joint[base], tm.joint[base + 1], tm.joint[base + 2], goal, tw_p, tw_r)
        out.clip.rotations.append(local^)
        out.clip.root.append(root_t)
    return out^


def _two_bone(skel: RetargetSkeleton, mut local: List[Quat], root_t: Vec3, a: Int, b: Int, c: Int, goal: Vec3,
              mut wp: List[Vec3], mut wr: List[Quat]):
    """Analytic two-bone IK in world space: moves c onto `goal` (exact when in reach) in the
    current knee plane, keeps c's world rotation. Joints between a and b (twist bones) are
    treated as part of the upper bone."""
    var pa = wp[a]
    var pb = wp[b]
    var pc = wp[c]
    var l1 = (pb - pa).length()
    var l2 = (pc - pb).length()
    if l1 < 1e-6 or l2 < 1e-6:
        return
    var to = goal - pa
    var d = to.length()
    if d < 1e-6:
        return
    var dir = to * (1.0 / d)
    d = min(max(d, abs(l1 - l2) + 1e-5), l1 + l2 - 1e-5)
    var x = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    var h = sqrt(max(Float32(0), l1 * l1 - x * x))
    var bend = (pb - pa) - dir * (pb - pa).dot(dir)
    if bend.length() < 1e-6:
        bend = (pb - pa).cross(Vec3(1, 0, 0)).cross(dir)
        if bend.length() < 1e-6:
            bend = Vec3(0, 0, 1)
    bend = bend * (1.0 / bend.length())
    var knee = pa + dir * x + bend * h
    var end_ = pa + dir * d
    var keep_c = wr[c]
    # rotate the upper chain so b lands on the knee
    var q1 = shortest_arc_quat(pb - pa, knee - pa)
    var pa_parent = skel.parents[a]
    var parent_r = wr[pa_parent] if pa_parent >= 0 else Quat()
    var new_a = q1.multiply(wr[a]).normalized()
    local[a] = parent_r.conjugate().multiply(new_a).normalized()
    skel.world(local, root_t, wp, wr)
    # rotate the lower bone so c lands on the goal
    var q2 = shortest_arc_quat(wp[c] - wp[b], end_ - wp[b])
    var pb_parent = skel.parents[b]
    var new_b = q2.multiply(wr[b]).normalized()
    local[b] = wr[pb_parent].conjugate().multiply(new_b).normalized()
    skel.world(local, root_t, wp, wr)
    # restore the end effector's world orientation
    var pc_parent = skel.parents[c]
    local[c] = wr[pc_parent].conjugate().multiply(keep_c).normalized()
    skel.world(local, root_t, wp, wr)


# ------------------------------------------------------------------------------- metrics

@fieldwise_init
struct FootContactError(Copyable, ImplicitlyCopyable, Movable, Writable):
    var contact_frames: Int
    var max_slide: Float32        # largest horizontal drift of a planted target foot from where it planted
    var max_height_error: Float32 # vs the scaled source foot height during contact
    var max_goal_error: Float32   # vs the scaled source foot position (swing frames)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("contact_frames=", self.contact_frames, " max_slide=", self.max_slide,
                     " max_height_err=", self.max_height_error, " max_goal_err=", self.max_goal_error)


def foot_contact_error(src: RetargetClip, res: RetargetResult) -> FootContactError:
    var sm = res.source_map.copy()
    var tm = res.target_map.copy()
    var up = Vec3(0, 1, 0)
    var sp0 = List[Vec3]()
    var sr0 = List[Quat]()
    src.skeleton.rest_world(sp0, sr0)
    var tp0 = List[Vec3]()
    var tr0 = List[Quat]()
    res.clip.skeleton.rest_world(tp0, tr0)
    var s_ground = _foot_height(sm, sp0, up)
    var t_ground = _foot_height(tm, tp0, up)
    var frames = 0
    var slide: Float32 = 0
    var herr: Float32 = 0
    var gerr: Float32 = 0
    var sp = List[Vec3]()
    var sr = List[Quat]()
    var tp = List[Vec3]()
    var tr = List[Quat]()
    var anchor = List[Vec3](length=2, fill=Vec3(0, 0, 0))
    var planted = List[Bool](length=2, fill=False)
    for f in range(res.clip.frame_count()):
        src.world(f, sp, sr)
        res.clip.world(f, tp, tr)
        for fi in range(2):
            var sf = sm.joint[R_L_FOOT if fi == 0 else R_R_FOOT]
            var tf = tm.joint[R_L_FOOT if fi == 0 else R_R_FOOT]
            if sf < 0 or tf < 0:
                continue
            var goal = res.map_point(sp[sf])
            goal = goal + up * ((tp0[tf].dot(up) - t_ground) - (sp0[sf].dot(up) - s_ground) * res.scale)
            if not res.contacts[fi][f]:
                gerr = max(gerr, (tp[tf] - goal).length())
            if res.contacts[fi][f]:
                frames += 1
                herr = max(herr, abs(tp[tf].dot(up) - goal.dot(up)))
                if not planted[fi]:
                    anchor[fi] = tp[tf]
                    planted[fi] = True
                var d = tp[tf] - anchor[fi]
                slide = max(slide, Vec3(d.x, 0, d.z).length())
            else:
                planted[fi] = False
    return FootContactError(frames, slide, herr, gerr)

"""Minimal MJCF (MuJoCo XML) importer and articulated-robot builder (`flint.physics.mjcf`).

`MjcfModel.load(path)` / `MjcfModel.parse(text)` read the subset of MJCF a legged robot needs, in
MuJoCo's own frame (z up, quaternions stored xyzw here):

  - `<compiler angle="radian|degree" eulerseq>`, `<option timestep gravity>`
  - `<default>` classes (nested, `class` / `childclass`) for joint, geom, motor, position, general
  - `<body pos quat|euler|axisangle childclass>` trees with `<inertial pos quat mass diaginertia |
    fullinertia>`, `<joint type=hinge|slide|ball|free pos axis range limited armature damping
    frictionloss stiffness>` / `<freejoint>`, `<geom type=plane|sphere|capsule|cylinder|box|mesh
    size pos quat|euler fromto contype conaffinity friction>` (mesh geoms are kept as visual)
  - `<actuator>` `motor` / `position` / `velocity` / `general` (joint, gear, kp, kv,
    ctrlrange, forcerange), `<contact>` `pair` / `exclude`, `<tendon><fixed>` (recorded only)

`qpos` / `qvel` use MuJoCo's layout (free joint: xyz + wxyz quaternion / linear world + angular
local velocity; hinge: one value). `forward_kinematics(qpos)` and `body_velocities` give world body
poses / velocities without any simulator.

`MjcfArticulation.build(world, model, ...)` turns the tree into Flint `PhysicsWorld` bodies
(y up: MuJoCo (x, y, z) -> Flint (x, z, -y)) joined by `JOINT_REVOLUTE` joints with the joint
limits, MJCF armature as a geared rotor, frictionloss, and a joint-space PD drive per actuator
(`set_command(targets, kp, kd)`, explicit like MuJoCo's position actuator, torque-capped by
forcerange or `set_efforts`). Drive stiff chains with small steps and stiff joint rows: the G1
under Metatron matches MuJoCo best at 1 kHz, 8 substeps, `joint_hertz` 2000, contact_hertz 250;
without armature the explicit kd on light links (wrists, ankles) is unstable at 1 kHz. Jointless bodies are
welded into their parent (composite mass, COM and inertia, re-diagonalised). Each Flint body sits
at its link's centre of mass in the principal inertia frame. Collision geoms (contype or
conaffinity non-zero) become compound children; cylinders become capsules. Self-collision is
off (MuJoCo's parent/child exclusion has no Flint equivalent; enable with `self_collide`).
Not modelled: tendons (e.g. the G1 ankle-coupling limits), slide / ball joints in the
articulation, solref / solimp, condim, margins.
"""

from std.math import sqrt, sin, cos, atan2
from flint.core.math import Vec3, Quat
from flint.platform.os import read_bytes
from flint.physics.world import PhysicsWorld
from flint.physics.shapes_v5 import CompoundShape, CompoundChild
from flint.physics.filter import CollisionFilter, CAT_DEBRIS, CAT_ALL
from flint.physics.joints import JointDef, JOINT_REVOLUTE, twist_angle, world_axis_a

comptime MJ_JNT_FREE: Int = 0
comptime MJ_JNT_BALL: Int = 1
comptime MJ_JNT_SLIDE: Int = 2
comptime MJ_JNT_HINGE: Int = 3

comptime MJ_GEOM_PLANE: Int = 0
comptime MJ_GEOM_SPHERE: Int = 2
comptime MJ_GEOM_CAPSULE: Int = 3
comptime MJ_GEOM_ELLIPSOID: Int = 4
comptime MJ_GEOM_CYLINDER: Int = 5
comptime MJ_GEOM_BOX: Int = 6
comptime MJ_GEOM_MESH: Int = 7

comptime MJ_ACT_MOTOR: Int = 0
comptime MJ_ACT_POSITION: Int = 1
comptime MJ_ACT_VELOCITY: Int = 2
comptime MJ_ACT_GENERAL: Int = 3


# ------------------------------------------------------------------------------------- XML

struct XmlDoc(Movable):
    """A parsed XML tree: element k has `tag[k]`, `parent[k]` (-1 for the root) and attributes
    `attr_names` / `attr_values` [first_attr[k], first_attr[k] + n_attr[k]). Text, comments,
    processing instructions and DOCTYPE are skipped; the five predefined entities are decoded."""
    var tag: List[String]
    var parent: List[Int]
    var first_attr: List[Int]
    var n_attr: List[Int]
    var attr_names: List[String]
    var attr_values: List[String]

    def __init__(out self):
        self.tag = List[String]()
        self.parent = List[Int]()
        self.first_attr = List[Int]()
        self.n_attr = List[Int]()
        self.attr_names = List[String]()
        self.attr_values = List[String]()

    @staticmethod
    def parse(text: String) raises -> XmlDoc:
        var d = XmlDoc()
        var b = text.as_bytes()
        var n = len(b)
        var stack = List[Int]()
        var i = 0
        while i < n:
            if b[i] != 60:  # '<'
                i += 1
                continue
            if i + 3 < n and b[i + 1] == 33 and b[i + 2] == 45 and b[i + 3] == 45:  # <!--
                var j = i + 4
                while j + 2 < n and not (b[j] == 45 and b[j + 1] == 45 and b[j + 2] == 62):
                    j += 1
                i = j + 3
                continue
            if i + 1 < n and (b[i + 1] == 63 or b[i + 1] == 33):  # <? or <!
                var j = i + 2
                while j < n and b[j] != 62:
                    j += 1
                i = j + 1
                continue
            if i + 1 < n and b[i + 1] == 47:  # </tag>
                var j = i + 2
                while j < n and b[j] != 62:
                    j += 1
                if len(stack) == 0:
                    raise Error("XML: unbalanced closing tag at byte " + String(i))
                _ = stack.pop()
                i = j + 1
                continue
            # an element
            var j = i + 1
            while j < n and not _is_space(b[j]) and b[j] != 62 and b[j] != 47:
                j += 1
            var k = len(d.tag)
            d.tag.append(String(text[byte=i + 1:j]))
            d.parent.append(stack[len(stack) - 1] if len(stack) > 0 else -1)
            d.first_attr.append(len(d.attr_names))
            var count = 0
            var closed = False
            while j < n:
                while j < n and _is_space(b[j]):
                    j += 1
                if j >= n:
                    break
                if b[j] == 62:  # '>'
                    j += 1
                    break
                if b[j] == 47:  # '/>'
                    closed = True
                    j += 2
                    break
                var ns = j
                while j < n and b[j] != 61 and not _is_space(b[j]) and b[j] != 62:
                    j += 1
                var name = String(text[byte=ns:j])
                while j < n and _is_space(b[j]):
                    j += 1
                if j >= n or b[j] != 61:
                    raise Error("XML: attribute " + name + " without a value in <" + d.tag[k] + ">")
                j += 1
                while j < n and _is_space(b[j]):
                    j += 1
                var q = b[j]
                if q != 34 and q != 39:
                    raise Error("XML: unquoted attribute " + name)
                j += 1
                var vs = j
                while j < n and b[j] != q:
                    j += 1
                d.attr_names.append(name)
                d.attr_values.append(_unescape(String(text[byte=vs:j])))
                j += 1
                count += 1
            d.n_attr.append(count)
            if not closed:
                stack.append(k)
            i = j
        if len(d.tag) == 0:
            raise Error("XML: no element")
        return d^

    def count(self) -> Int:
        return len(self.tag)

    def children(self, k: Int, tag: String = "") -> List[Int]:
        var out = List[Int]()
        for c in range(k + 1, len(self.tag)):
            if self.parent[c] == k and (tag.byte_length() == 0 or self.tag[c] == tag):
                out.append(c)
        return out^

    def find_attr(self, k: Int, name: String) -> Int:
        for a in range(self.first_attr[k], self.first_attr[k] + self.n_attr[k]):
            if self.attr_names[a] == name:
                return a
        return -1

    def has(self, k: Int, name: String) -> Bool:
        return self.find_attr(k, name) >= 0

    def attr(self, k: Int, name: String, default: String = "") -> String:
        var a = self.find_attr(k, name)
        return self.attr_values[a] if a >= 0 else default


@always_inline
def _is_space(c: UInt8) -> Bool:
    return c == 32 or c == 9 or c == 10 or c == 13


def _unescape(s: String) -> String:
    if not ("&" in s):
        return s
    return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'").replace("&amp;", "&")


def _floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    var b = s.as_bytes()
    var start = -1
    for k in range(len(b) + 1):
        var sp = k == len(b) or _is_space(b[k])
        if sp:
            if start >= 0:
                out.append(atof(String(s[byte=start:k])))
                start = -1
        elif start < 0:
            start = k
    return out^


# ---------------------------------------------------------------------------------- 3-D math

def mj_quat_mul(a: Quat, b: Quat) -> Quat:
    """a * b (apply b, then a), xyzw."""
    return a.multiply(b)


def mj_rotate(q: Quat, v: Vec3) -> Vec3:
    var u = Vec3(q.x, q.y, q.z)
    var t = u.cross(v) * 2.0
    return v + t * q.w + u.cross(t)


def mj_quat_from_two(a: Vec3, b: Vec3) -> Quat:
    """The shortest rotation taking unit `a` onto unit `b`."""
    var c = a.cross(b)
    var d = a.dot(b)
    if d < -0.999999:
        var p = Vec3(1, 0, 0).cross(a)
        if p.length() < 1.0e-6:
            p = Vec3(0, 1, 0).cross(a)
        p = p.normalized()
        return Quat(p.x, p.y, p.z, 0.0)
    var q = Quat(c.x, c.y, c.z, 1.0 + d)
    return q.normalized()


def mj_axis_angle(axis: Vec3, angle: Float64) -> Quat:
    var a = axis.normalized()
    var s = Float32(sin(angle * 0.5))
    return Quat(a.x * s, a.y * s, a.z * s, Float32(cos(angle * 0.5)))


struct Mat3(Copyable, ImplicitlyCopyable, Movable):
    """Row-major 3x3 (inertia tensors, rotation matrices)."""
    var m: SIMD[DType.float64, 16]

    def __init__(out self):
        self.m = SIMD[DType.float64, 16](0)

    @staticmethod
    def diag(x: Float64, y: Float64, z: Float64) -> Mat3:
        var r = Mat3()
        r.m[0] = x
        r.m[4] = y
        r.m[8] = z
        return r

    @staticmethod
    def rotation(q: Quat) -> Mat3:
        var r = Mat3()
        var x = Float64(q.x); var y = Float64(q.y); var z = Float64(q.z); var w = Float64(q.w)
        r.m[0] = 1 - 2 * (y * y + z * z); r.m[1] = 2 * (x * y - z * w); r.m[2] = 2 * (x * z + y * w)
        r.m[3] = 2 * (x * y + z * w); r.m[4] = 1 - 2 * (x * x + z * z); r.m[5] = 2 * (y * z - x * w)
        r.m[6] = 2 * (x * z - y * w); r.m[7] = 2 * (y * z + x * w); r.m[8] = 1 - 2 * (x * x + y * y)
        return r

    def at(self, i: Int, j: Int) -> Float64:
        return self.m[i * 3 + j]

    def mul(self, o: Mat3) -> Mat3:
        var r = Mat3()
        for i in range(3):
            for j in range(3):
                var s = 0.0
                for k in range(3):
                    s += self.m[i * 3 + k] * o.m[k * 3 + j]
                r.m[i * 3 + j] = s
        return r

    def transposed(self) -> Mat3:
        var r = Mat3()
        for i in range(3):
            for j in range(3):
                r.m[i * 3 + j] = self.m[j * 3 + i]
        return r

    def add(self, o: Mat3) -> Mat3:
        var r = Mat3()
        for k in range(9):
            r.m[k] = self.m[k] + o.m[k]
        return r


def mat3_to_quat(r: Mat3) -> Quat:
    var tr = r.at(0, 0) + r.at(1, 1) + r.at(2, 2)
    var x: Float64
    var y: Float64
    var z: Float64
    var w: Float64
    if tr > 0:
        var s = sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r.at(2, 1) - r.at(1, 2)) / s
        y = (r.at(0, 2) - r.at(2, 0)) / s
        z = (r.at(1, 0) - r.at(0, 1)) / s
    elif r.at(0, 0) > r.at(1, 1) and r.at(0, 0) > r.at(2, 2):
        var s = sqrt(1.0 + r.at(0, 0) - r.at(1, 1) - r.at(2, 2)) * 2.0
        w = (r.at(2, 1) - r.at(1, 2)) / s
        x = 0.25 * s
        y = (r.at(0, 1) + r.at(1, 0)) / s
        z = (r.at(0, 2) + r.at(2, 0)) / s
    elif r.at(1, 1) > r.at(2, 2):
        var s = sqrt(1.0 + r.at(1, 1) - r.at(0, 0) - r.at(2, 2)) * 2.0
        w = (r.at(0, 2) - r.at(2, 0)) / s
        x = (r.at(0, 1) + r.at(1, 0)) / s
        y = 0.25 * s
        z = (r.at(1, 2) + r.at(2, 1)) / s
    else:
        var s = sqrt(1.0 + r.at(2, 2) - r.at(0, 0) - r.at(1, 1)) * 2.0
        w = (r.at(1, 0) - r.at(0, 1)) / s
        x = (r.at(0, 2) + r.at(2, 0)) / s
        y = (r.at(1, 2) + r.at(2, 1)) / s
        z = 0.25 * s
    return Quat(Float32(x), Float32(y), Float32(z), Float32(w)).normalized()


def symmetric_eigen(a: Mat3) -> Tuple[Vec3, Quat]:
    """Eigenvalues and the rotation R (columns = eigenvectors, right-handed) with a = R diag R^T
    (cyclic Jacobi)."""
    var m = a
    var v = Mat3.diag(1, 1, 1)
    for _ in range(50):
        var off = m.at(0, 1) * m.at(0, 1) + m.at(0, 2) * m.at(0, 2) + m.at(1, 2) * m.at(1, 2)
        if off < 1.0e-30:
            break
        for p in range(2):
            for q in range(p + 1, 3):
                var apq = m.at(p, q)
                if abs(apq) < 1.0e-300:
                    continue
                var theta = (m.at(q, q) - m.at(p, p)) / (2.0 * apq)
                var t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                var j = Mat3.diag(1, 1, 1)
                j.m[p * 3 + p] = c
                j.m[q * 3 + q] = c
                j.m[p * 3 + q] = s
                j.m[q * 3 + p] = -s
                m = j.transposed().mul(m).mul(j)
                v = v.mul(j)
    # right-handed
    var c0 = Vec3(Float32(v.at(0, 0)), Float32(v.at(1, 0)), Float32(v.at(2, 0)))
    var c1 = Vec3(Float32(v.at(0, 1)), Float32(v.at(1, 1)), Float32(v.at(2, 1)))
    var c2 = Vec3(Float32(v.at(0, 2)), Float32(v.at(1, 2)), Float32(v.at(2, 2)))
    if c0.cross(c1).dot(c2) < 0:
        for i in range(3):
            v.m[i * 3 + 2] = -v.m[i * 3 + 2]
    return (Vec3(Float32(m.at(0, 0)), Float32(m.at(1, 1)), Float32(m.at(2, 2))), mat3_to_quat(v))


# -------------------------------------------------------------------------------------- model

struct MjBody(Copyable, Movable):
    var name: String
    var parent: Int
    var pos: Vec3
    var quat: Quat
    var mass: Float64
    var ipos: Vec3
    var iquat: Quat
    var inertia: Vec3        # principal moments in the (ipos, iquat) frame
    var has_inertial: Bool
    var joints: List[Int]
    var geoms: List[Int]

    def __init__(out self):
        self.name = String()
        self.parent = -1
        self.pos = Vec3()
        self.quat = Quat()
        self.mass = 0
        self.ipos = Vec3()
        self.iquat = Quat()
        self.inertia = Vec3()
        self.has_inertial = False
        self.joints = List[Int]()
        self.geoms = List[Int]()


struct MjJoint(Copyable, Movable):
    var name: String
    var body: Int
    var kind: Int
    var pos: Vec3
    var axis: Vec3
    var limited: Bool
    var lower: Float64
    var upper: Float64
    var armature: Float64
    var damping: Float64
    var frictionloss: Float64
    var stiffness: Float64
    var qpos_adr: Int
    var dof_adr: Int

    def __init__(out self):
        self.name = String()
        self.body = 0
        self.kind = MJ_JNT_HINGE
        self.pos = Vec3()
        self.axis = Vec3(0, 0, 1)
        self.limited = False
        self.lower = 0
        self.upper = 0
        self.armature = 0
        self.damping = 0
        self.frictionloss = 0
        self.stiffness = 0
        self.qpos_adr = 0
        self.dof_adr = 0


struct MjGeom(Copyable, Movable):
    var name: String
    var body: Int
    var kind: Int
    var size: Vec3
    var pos: Vec3
    var quat: Quat
    var contype: Int
    var conaffinity: Int
    var friction: Float64

    def __init__(out self):
        self.name = String()
        self.body = 0
        self.kind = MJ_GEOM_SPHERE
        self.size = Vec3()
        self.pos = Vec3()
        self.quat = Quat()
        self.contype = 1
        self.conaffinity = 1
        self.friction = 1.0

    def collides(self) -> Bool:
        return self.contype != 0 or self.conaffinity != 0


struct MjActuator(Copyable, Movable):
    var name: String
    var joint: Int
    var kind: Int
    var gear: Float64
    var kp: Float64
    var kv: Float64
    var ctrl_lower: Float64
    var ctrl_upper: Float64
    var ctrl_limited: Bool
    var force_lower: Float64
    var force_upper: Float64
    var force_limited: Bool

    def __init__(out self):
        self.name = String()
        self.joint = -1
        self.kind = MJ_ACT_MOTOR
        self.gear = 1
        self.kp = 0
        self.kv = 0
        self.ctrl_lower = 0
        self.ctrl_upper = 0
        self.ctrl_limited = False
        self.force_lower = 0
        self.force_upper = 0
        self.force_limited = False


struct MjTendon(Copyable, Movable):
    """A fixed tendon L = sum coef_i q_i (recorded, not simulated by the articulation)."""
    var joints: List[Int]
    var coefs: List[Float64]
    var limited: Bool
    var lower: Float64
    var upper: Float64

    def __init__(out self):
        self.joints = List[Int]()
        self.coefs = List[Float64]()
        self.limited = False
        self.lower = 0
        self.upper = 0


struct MjcfModel(Movable):
    """A robot read from MJCF. Body 0 is the world. Bodies are in document order (parents first)."""
    var name: String
    var timestep: Float64
    var gravity: Vec3
    var bodies: List[MjBody]
    var joints: List[MjJoint]
    var geoms: List[MjGeom]
    var actuators: List[MjActuator]
    var tendons: List[MjTendon]
    var pair_geom1: List[Int]      # -1: a geom not defined in the file
    var pair_geom2: List[Int]
    var pair_friction: List[Float64]
    var exclude_body1: List[Int]
    var exclude_body2: List[Int]
    var nq: Int
    var nv: Int
    # parse state
    var _degree: Bool
    var _eulerseq: String
    var _class_names: List[String]
    var _class_parent: List[Int]
    var _class_node: List[Int]

    def __init__(out self):
        self.name = String()
        self.timestep = 0.002
        self.gravity = Vec3(0, 0, -9.81)
        self.bodies = List[MjBody]()
        self.joints = List[MjJoint]()
        self.geoms = List[MjGeom]()
        self.actuators = List[MjActuator]()
        self.tendons = List[MjTendon]()
        self.pair_geom1 = List[Int]()
        self.pair_geom2 = List[Int]()
        self.pair_friction = List[Float64]()
        self.exclude_body1 = List[Int]()
        self.exclude_body2 = List[Int]()
        self.nq = 0
        self.nv = 0
        self._degree = True
        self._eulerseq = "xyz"
        self._class_names = List[String]()
        self._class_parent = List[Int]()
        self._class_node = List[Int]()

    @staticmethod
    def load(path: String) raises -> MjcfModel:
        var b = read_bytes(path)
        if len(b) == 0:
            raise Error("MJCF: cannot read " + path)
        return MjcfModel.parse(String(unsafe_from_utf8=b))

    @staticmethod
    def parse(text: String) raises -> MjcfModel:
        var d = XmlDoc.parse(text)
        if d.tag[0] != "mujoco":
            raise Error("MJCF: the root element is <" + d.tag[0] + ">, not <mujoco>")
        var m = MjcfModel()
        m.name = d.attr(0, "model")
        for c in d.children(0, "compiler"):
            m._degree = d.attr(c, "angle", "degree") != "radian"
            m._eulerseq = d.attr(c, "eulerseq", "xyz")
        for c in d.children(0, "option"):
            if d.has(c, "timestep"):
                m.timestep = atof(d.attr(c, "timestep"))
            if d.has(c, "gravity"):
                m.gravity = _vec(_floats(d.attr(c, "gravity")))
        for c in d.children(0, "default"):
            m._read_defaults(d, c, -1)
        var world = MjBody()
        world.name = "world"
        m.bodies.append(world^)
        for wb in d.children(0, "worldbody"):
            for g in d.children(wb, "geom"):
                m._read_geom(d, g, 0, "main")
            for bnode in d.children(wb, "body"):
                m._read_body(d, bnode, 0, "main")
        for a in d.children(0, "actuator"):
            for c in d.children(a):
                m._read_actuator(d, c)
        for c in d.children(0, "contact"):
            for p in d.children(c, "pair"):
                var g1 = m.geom_index(d.attr(p, "geom1"))
                var g2 = m.geom_index(d.attr(p, "geom2"))
                # a geom the file does not define (e.g. a floor added by the host) stays -1
                m.pair_geom1.append(g1)
                m.pair_geom2.append(g2)
                var fr = _floats(d.attr(p, "friction", "1"))
                m.pair_friction.append(fr[0])
            for e in d.children(c, "exclude"):
                m.exclude_body1.append(m.body_index(d.attr(e, "body1")))
                m.exclude_body2.append(m.body_index(d.attr(e, "body2")))
        for t in d.children(0, "tendon"):
            for f in d.children(t, "fixed"):
                var td = MjTendon()
                for jn in d.children(f, "joint"):
                    td.joints.append(m.joint_index(d.attr(jn, "joint")))
                    td.coefs.append(atof(d.attr(jn, "coef", "1")))
                if d.has(f, "range"):
                    var r = _floats(d.attr(f, "range"))
                    td.lower = r[0]
                    td.upper = r[1]
                    td.limited = d.attr(f, "limited", "true") == "true"
                m.tendons.append(td^)
        return m^

    # -------------------------------------------------------------------------------- lookup

    def body_index(self, name: String) -> Int:
        for b in range(len(self.bodies)):
            if self.bodies[b].name == name:
                return b
        return -1

    def joint_index(self, name: String) -> Int:
        for j in range(len(self.joints)):
            if self.joints[j].name == name:
                return j
        return -1

    def geom_index(self, name: String) -> Int:
        for g in range(len(self.geoms)):
            if self.geoms[g].name == name:
                return g
        return -1

    def actuator_index(self, name: String) -> Int:
        for a in range(len(self.actuators)):
            if self.actuators[a].name == name:
                return a
        return -1

    def total_mass(self) -> Float64:
        var s = 0.0
        for b in self.bodies:
            s += b.mass
        return s

    def qpos0(self) -> List[Float64]:
        """The reference configuration: free joints at their body's pose, other joints at 0."""
        var q = List[Float64](length=self.nq, fill=0.0)
        for j in self.joints:
            if j.kind == MJ_JNT_FREE:
                ref b = self.bodies[j.body]
                q[j.qpos_adr] = Float64(b.pos.x)
                q[j.qpos_adr + 1] = Float64(b.pos.y)
                q[j.qpos_adr + 2] = Float64(b.pos.z)
                q[j.qpos_adr + 3] = Float64(b.quat.w)
                q[j.qpos_adr + 4] = Float64(b.quat.x)
                q[j.qpos_adr + 5] = Float64(b.quat.y)
                q[j.qpos_adr + 6] = Float64(b.quat.z)
            elif j.kind == MJ_JNT_BALL:
                q[j.qpos_adr] = 1.0
        return q^

    # ------------------------------------------------------------------------------ defaults

    def _read_defaults(mut self, d: XmlDoc, root: Int, root_parent: Int):
        var nodes = List[Int]()
        nodes.append(root)
        var parents = List[Int]()
        parents.append(root_parent)
        while len(nodes) > 0:
            var node = nodes.pop()
            var parent = parents.pop()
            var k = len(self._class_names)
            self._class_names.append(d.attr(node, "class", "main"))
            self._class_parent.append(parent)
            self._class_node.append(node)
            for c in d.children(node, "default"):
                nodes.append(c)
                parents.append(k)

    def _class_index(self, name: String) -> Int:
        for k in range(len(self._class_names)):
            if self._class_names[k] == name:
                return k
        return -1

    def _get(self, d: XmlDoc, node: Int, cls: String, name: String, default: String) -> String:
        """An element attribute, else its class chain's default for that element kind."""
        if d.has(node, name):
            return d.attr(node, name)
        var c = self._class_index(d.attr(node, "class", cls))
        var tag = d.tag[node]
        while c >= 0:
            for e in d.children(self._class_node[c], tag):
                if d.has(e, name):
                    return d.attr(e, name)
            c = self._class_parent[c]
        return default

    def _angle(self, v: Float64) -> Float64:
        return v * 3.141592653589793 / 180.0 if self._degree else v

    def _orientation(self, d: XmlDoc, node: Int, cls: String) raises -> Quat:
        var qs = self._get(d, node, cls, "quat", "")
        if qs.byte_length() > 0:
            var q = _floats(qs)
            return Quat(Float32(q[1]), Float32(q[2]), Float32(q[3]), Float32(q[0])).normalized()
        var aa = self._get(d, node, cls, "axisangle", "")
        if aa.byte_length() > 0:
            var a = _floats(aa)
            return mj_axis_angle(Vec3(Float32(a[0]), Float32(a[1]), Float32(a[2])), self._angle(a[3]))
        var eu = self._get(d, node, cls, "euler", "")
        if eu.byte_length() > 0:
            var e = _floats(eu)
            var q = Quat()
            var seq = self._eulerseq.as_bytes()
            for k in range(3):
                var c = seq[k]
                var axis = Vec3(1, 0, 0) if (c == 120 or c == 88) else (Vec3(0, 1, 0) if (c == 121 or c == 89) else Vec3(0, 0, 1))
                var r = mj_axis_angle(axis, self._angle(e[k]))
                # lowercase: intrinsic (post-multiply), uppercase: extrinsic (pre-multiply)
                q = q.multiply(r) if c >= 97 else r.multiply(q)
            return q.normalized()
        var zs = self._get(d, node, cls, "zaxis", "")
        if zs.byte_length() > 0:
            return mj_quat_from_two(Vec3(0, 0, 1), _vec(_floats(zs)).normalized())
        return Quat()

    # -------------------------------------------------------------------------------- bodies

    def _read_body(mut self, d: XmlDoc, root: Int, root_parent: Int, root_cls: String) raises:
        """Depth-first, document order (parents before children, like MuJoCo)."""
        var nodes = List[Int]()
        nodes.append(root)
        var parents = List[Int]()
        parents.append(root_parent)
        var classes = List[String]()
        classes.append(root_cls)
        while len(nodes) > 0:
            var node = nodes.pop()
            var parent = parents.pop()
            var cls_in = classes.pop()
            var kids = self._read_one_body(d, node, parent, cls_in)
            var k = len(self.bodies) - 1
            var cls = d.attr(node, "childclass", cls_in)
            for c in reversed(kids):
                nodes.append(c)
                parents.append(k)
                classes.append(cls)

    def _read_one_body(mut self, d: XmlDoc, node: Int, parent: Int, cls_in: String) raises -> List[Int]:
        var cls = d.attr(node, "childclass", cls_in)
        var b = MjBody()
        b.name = d.attr(node, "name", "body" + String(len(self.bodies)))
        b.parent = parent
        b.pos = _vec(_floats(d.attr(node, "pos", "0 0 0")))
        b.quat = self._orientation(d, node, cls)
        var k = len(self.bodies)
        for i in d.children(node, "inertial"):
            b.has_inertial = True
            b.mass = atof(d.attr(i, "mass", "0"))
            b.ipos = _vec(_floats(d.attr(i, "pos", "0 0 0")))
            b.iquat = self._orientation(d, i, cls)
            if d.has(i, "diaginertia"):
                b.inertia = _vec(_floats(d.attr(i, "diaginertia")))
            elif d.has(i, "fullinertia"):
                var f = _floats(d.attr(i, "fullinertia"))  # ixx iyy izz ixy ixz iyz
                var t = Mat3()
                t.m[0] = f[0]; t.m[4] = f[1]; t.m[8] = f[2]
                t.m[1] = f[3]; t.m[3] = f[3]
                t.m[2] = f[4]; t.m[6] = f[4]
                t.m[5] = f[5]; t.m[7] = f[5]
                var e = symmetric_eigen(t)
                b.inertia = e[0]
                b.iquat = b.iquat.multiply(e[1]).normalized()
        self.bodies.append(b^)
        for j in d.children(node):
            if d.tag[j] == "joint" or d.tag[j] == "freejoint":
                self._read_joint(d, j, k, cls)
        for g in d.children(node, "geom"):
            self._read_geom(d, g, k, cls)
        return d.children(node, "body")

    def _read_joint(mut self, d: XmlDoc, node: Int, body: Int, cls: String) raises:
        var j = MjJoint()
        j.body = body
        j.name = d.attr(node, "name", "joint" + String(len(self.joints)))
        var t = "free" if d.tag[node] == "freejoint" else self._get(d, node, cls, "type", "hinge")
        if t == "free":
            j.kind = MJ_JNT_FREE
        elif t == "ball":
            j.kind = MJ_JNT_BALL
        elif t == "slide":
            j.kind = MJ_JNT_SLIDE
        elif t == "hinge":
            j.kind = MJ_JNT_HINGE
        else:
            raise Error("MJCF: joint type " + t + " is not supported")
        j.pos = _vec(_floats(self._get(d, node, cls, "pos", "0 0 0")))
        j.axis = _vec(_floats(self._get(d, node, cls, "axis", "0 0 1"))).normalized()
        var rs = self._get(d, node, cls, "range", "")
        if rs.byte_length() > 0 and j.kind != MJ_JNT_FREE:
            var r = _floats(rs)
            var scale = j.kind != MJ_JNT_SLIDE
            j.lower = self._angle(r[0]) if scale else r[0]
            j.upper = self._angle(r[1]) if scale else r[1]
            var lim = self._get(d, node, cls, "limited", "auto")
            j.limited = lim == "true" or (lim == "auto" and j.upper > j.lower)
        j.armature = atof(self._get(d, node, cls, "armature", "0"))
        j.damping = atof(self._get(d, node, cls, "damping", "0"))
        j.frictionloss = atof(self._get(d, node, cls, "frictionloss", "0"))
        j.stiffness = atof(self._get(d, node, cls, "stiffness", "0"))
        j.qpos_adr = self.nq
        j.dof_adr = self.nv
        if j.kind == MJ_JNT_FREE:
            self.nq += 7
            self.nv += 6
        elif j.kind == MJ_JNT_BALL:
            self.nq += 4
            self.nv += 3
        else:
            self.nq += 1
            self.nv += 1
        self.bodies[body].joints.append(len(self.joints))
        self.joints.append(j^)

    def _read_geom(mut self, d: XmlDoc, node: Int, body: Int, cls: String) raises:
        var g = MjGeom()
        g.body = body
        g.name = d.attr(node, "name", "")
        var t = self._get(d, node, cls, "type", "sphere")
        if t == "plane": g.kind = MJ_GEOM_PLANE
        elif t == "sphere": g.kind = MJ_GEOM_SPHERE
        elif t == "capsule": g.kind = MJ_GEOM_CAPSULE
        elif t == "ellipsoid": g.kind = MJ_GEOM_ELLIPSOID
        elif t == "cylinder": g.kind = MJ_GEOM_CYLINDER
        elif t == "box": g.kind = MJ_GEOM_BOX
        elif t == "mesh": g.kind = MJ_GEOM_MESH
        else:
            raise Error("MJCF: geom type " + t + " is not supported")
        var sz = _floats(self._get(d, node, cls, "size", "0 0 0"))
        while len(sz) < 3:
            sz.append(0.0)
        g.size = Vec3(Float32(sz[0]), Float32(sz[1]), Float32(sz[2]))
        var ft = self._get(d, node, cls, "fromto", "")
        if ft.byte_length() > 0:
            var f = _floats(ft)
            var a = Vec3(Float32(f[0]), Float32(f[1]), Float32(f[2]))
            var b = Vec3(Float32(f[3]), Float32(f[4]), Float32(f[5]))
            g.pos = (a + b) * 0.5
            var dvec = b - a
            g.size.y = dvec.length() * 0.5
            g.quat = mj_quat_from_two(Vec3(0, 0, 1), dvec.normalized()) if dvec.length() > 1.0e-9 else Quat()
        else:
            g.pos = _vec(_floats(self._get(d, node, cls, "pos", "0 0 0")))
            g.quat = self._orientation(d, node, cls)
        g.contype = Int(atof(self._get(d, node, cls, "contype", "1")))
        g.conaffinity = Int(atof(self._get(d, node, cls, "conaffinity", "1")))
        g.friction = _floats(self._get(d, node, cls, "friction", "1"))[0]
        self.bodies[body].geoms.append(len(self.geoms))
        self.geoms.append(g^)

    def _read_actuator(mut self, d: XmlDoc, node: Int) raises:
        var tag = d.tag[node]
        var a = MjActuator()
        if tag == "motor": a.kind = MJ_ACT_MOTOR
        elif tag == "position": a.kind = MJ_ACT_POSITION
        elif tag == "velocity": a.kind = MJ_ACT_VELOCITY
        elif tag == "general": a.kind = MJ_ACT_GENERAL
        else:
            return
        var cls = "main"
        a.name = d.attr(node, "name", "")
        a.joint = self.joint_index(self._get(d, node, cls, "joint", ""))
        if a.joint < 0:
            raise Error("MJCF: actuator " + a.name + " has no known joint")
        a.gear = _floats(self._get(d, node, cls, "gear", "1"))[0]
        a.kp = atof(self._get(d, node, cls, "kp", "1" if a.kind == MJ_ACT_POSITION else "0"))
        a.kv = atof(self._get(d, node, cls, "kv", "1" if a.kind == MJ_ACT_VELOCITY else "0"))
        var cr = self._get(d, node, cls, "ctrlrange", "")
        if cr.byte_length() > 0:
            var r = _floats(cr)
            a.ctrl_lower = r[0]
            a.ctrl_upper = r[1]
            var lim = self._get(d, node, cls, "ctrllimited", "auto")
            a.ctrl_limited = lim == "true" or (lim == "auto" and r[1] > r[0])
        var fr = self._get(d, node, cls, "forcerange", "")
        if fr.byte_length() > 0:
            var r = _floats(fr)
            a.force_lower = r[0]
            a.force_upper = r[1]
            var lim = self._get(d, node, cls, "forcelimited", "auto")
            a.force_limited = lim == "true" or (lim == "auto" and r[1] > r[0])
        self.actuators.append(a^)

    # ----------------------------------------------------------------------------- kinematics

    def forward_kinematics(self, qpos: List[Float64]) -> Tuple[List[Vec3], List[Quat]]:
        """World position and orientation (xyzw) of every body origin, body 0 = world."""
        var pos = List[Vec3](length=len(self.bodies), fill=Vec3())
        var rot = List[Quat](length=len(self.bodies), fill=Quat())
        for b in range(1, len(self.bodies)):
            ref body = self.bodies[b]
            var p = pos[body.parent] + mj_rotate(rot[body.parent], body.pos)
            var q = rot[body.parent].multiply(body.quat)
            for ji in body.joints:
                ref j = self.joints[ji]
                var a = j.qpos_adr
                if j.kind == MJ_JNT_FREE:
                    p = Vec3(Float32(qpos[a]), Float32(qpos[a + 1]), Float32(qpos[a + 2]))
                    q = Quat(Float32(qpos[a + 4]), Float32(qpos[a + 5]), Float32(qpos[a + 6]), Float32(qpos[a + 3])).normalized()
                elif j.kind == MJ_JNT_HINGE:
                    var anchor = p + mj_rotate(q, j.pos)
                    var r = mj_axis_angle(j.axis, qpos[a])
                    q = q.multiply(r)
                    p = anchor - mj_rotate(q, j.pos)
                elif j.kind == MJ_JNT_SLIDE:
                    p = p + mj_rotate(q, j.axis) * Float32(qpos[a])
                else:
                    var r = Quat(Float32(qpos[a + 1]), Float32(qpos[a + 2]), Float32(qpos[a + 3]), Float32(qpos[a])).normalized()
                    var anchor = p + mj_rotate(q, j.pos)
                    q = q.multiply(r)
                    p = anchor - mj_rotate(q, j.pos)
            pos[b] = p
            rot[b] = q.normalized()
        return (pos^, rot^)

    def body_velocities(self, qpos: List[Float64], qvel: List[Float64]) -> Tuple[List[Vec3], List[Vec3]]:
        """World linear velocity of every body origin and world angular velocity."""
        var fk = self.forward_kinematics(qpos)
        var pos = fk[0].copy()
        var rot = fk[1].copy()
        var v = List[Vec3](length=len(self.bodies), fill=Vec3())
        var w = List[Vec3](length=len(self.bodies), fill=Vec3())
        for b in range(1, len(self.bodies)):
            ref body = self.bodies[b]
            var pp = pos[body.parent]
            var wb = w[body.parent]
            var vb = v[body.parent] + wb.cross(pos[b] - pp)
            for ji in body.joints:
                ref j = self.joints[ji]
                var a = j.dof_adr
                if j.kind == MJ_JNT_FREE:
                    vb = Vec3(Float32(qvel[a]), Float32(qvel[a + 1]), Float32(qvel[a + 2]))
                    wb = mj_rotate(rot[b], Vec3(Float32(qvel[a + 3]), Float32(qvel[a + 4]), Float32(qvel[a + 5])))
                elif j.kind == MJ_JNT_HINGE:
                    var axis = mj_rotate(rot[b], j.axis)
                    var anchor = pos[b] + mj_rotate(rot[b], j.pos)
                    var dw = axis * Float32(qvel[a])
                    wb = wb + dw
                    vb = vb + dw.cross(pos[b] - anchor)
                elif j.kind == MJ_JNT_SLIDE:
                    vb = vb + mj_rotate(rot[b], j.axis) * Float32(qvel[a])
                else:
                    var dw = mj_rotate(rot[b], Vec3(Float32(qvel[a]), Float32(qvel[a + 1]), Float32(qvel[a + 2])))
                    var anchor = pos[b] + mj_rotate(rot[b], j.pos)
                    wb = wb + dw
                    vb = vb + dw.cross(pos[b] - anchor)
            v[b] = vb
            w[b] = wb
        return (v^, w^)


def _vec(f: List[Float64]) -> Vec3:
    if len(f) < 3:
        return Vec3()
    return Vec3(Float32(f[0]), Float32(f[1]), Float32(f[2]))


# ---------------------------------------------------------------------------------- frames

comptime _S2: Float32 = 0.70710678118654752


def mj_to_flint(v: Vec3) -> Vec3:
    """MuJoCo z-up to Flint y-up: (x, y, z) -> (x, z, -y)."""
    return Vec3(v.x, v.z, -v.y)


def flint_to_mj(v: Vec3) -> Vec3:
    return Vec3(v.x, -v.z, v.y)


def mj_to_flint_quat(q: Quat) -> Quat:
    """A body orientation: Flint world from the body's own (MuJoCo-axis) local frame, so local
    offsets and inertia axes keep their MJCF values."""
    return Quat(-_S2, 0, 0, _S2).multiply(q).normalized()


def flint_to_mj_quat(q: Quat) -> Quat:
    return Quat(_S2, 0, 0, _S2).multiply(q).normalized()


# ----------------------------------------------------------------------------- articulation

struct MjLink(Copyable, Movable):
    """One Flint rigid body: an MJCF body with a joint (or the root) plus the jointless bodies
    welded to it. `com` / `frame` are in the link body's frame (MuJoCo axes)."""
    var body: Int          # MJCF body index
    var parent_link: Int   # -1 for a root
    var slot: Int          # Flint body slot
    var joint: Int         # Flint joint index (-1 for a root)
    var mj_joint: Int      # MJCF joint index (-1 for a root)
    var mass: Float64
    var com: Vec3
    var frame: Quat        # principal axes
    var moments: Vec3

    def __init__(out self):
        self.body = 0
        self.parent_link = -1
        self.slot = -1
        self.joint = -1
        self.mj_joint = -1
        self.mass = 0
        self.com = Vec3()
        self.frame = Quat()
        self.moments = Vec3()


struct MjcfArticulation(Movable):
    """An MJCF robot in a Flint `PhysicsWorld`, with joint-space state and PD commands.

    `body_link[b]` maps an MJCF body to its link; `body_offset_pos/rot[b]` place the body origin
    in its link body's frame. `act_link[a]` is the link driven by actuator a."""
    var links: List[MjLink]
    var body_link: List[Int]
    var body_offset_pos: List[Vec3]
    var body_offset_rot: List[Quat]
    var act_link: List[Int]
    var act_effort: List[Float32]
    var root_link: Int
    var free_joint: Int    # MJCF joint index of the floating base (-1: fixed base)
    var ground_slot: Int

    def __init__(out self):
        self.links = List[MjLink]()
        self.body_link = List[Int]()
        self.body_offset_pos = List[Vec3]()
        self.body_offset_rot = List[Quat]()
        self.act_link = List[Int]()
        self.act_effort = List[Float32]()
        self.root_link = -1
        self.free_joint = -1
        self.ground_slot = -1

    @staticmethod
    def build(mut world: PhysicsWorld, model: MjcfModel, friction: Float32 = 0.8, joint_hertz: Float32 = 0.0,
              armature: Bool = True, frictionloss: Bool = True, self_collide: Bool = False,
              ground: Bool = True, armature_scale: Float32 = 1.0) raises -> MjcfArticulation:
        """Add the robot (at qpos0) and, with `ground`, a static ground box whose top is z = 0.
        `joint_hertz` > 0 sets the softness of the joints' rigid rows (0: the world default)."""
        var art = MjcfArticulation()
        var q0 = model.qpos0()
        var fk = model.forward_kinematics(q0)
        var bpos = fk[0].copy()
        var brot = fk[1].copy()
        var nb = len(model.bodies)
        # 1. links: every body with a joint (or a child of the world) starts one
        art.body_link = List[Int](length=nb, fill=-1)
        art.body_offset_pos = List[Vec3](length=nb, fill=Vec3())
        art.body_offset_rot = List[Quat](length=nb, fill=Quat())
        for b in range(1, nb):
            ref body = model.bodies[b]
            var starts = len(body.joints) > 0 or body.parent == 0
            if len(body.joints) > 1:
                raise Error("MJCF articulation: body " + body.name + " has several joints")
            if starts:
                var l = MjLink()
                l.body = b
                l.parent_link = art.body_link[body.parent] if body.parent > 0 else -1
                if len(body.joints) == 1:
                    var jk = model.joints[body.joints[0]].kind
                    if jk == MJ_JNT_FREE:
                        art.free_joint = body.joints[0]
                        l.parent_link = -1
                    elif jk == MJ_JNT_HINGE:
                        l.mj_joint = body.joints[0]
                    else:
                        raise Error("MJCF articulation: only hinge and free joints are supported (" + body.name + ")")
                art.body_link[b] = len(art.links)
                art.links.append(l^)
            else:
                art.body_link[b] = art.body_link[body.parent]
            var lb = art.links[art.body_link[b]].body
            var inv = brot[lb].conjugate()
            art.body_offset_pos[b] = mj_rotate(inv, bpos[b] - bpos[lb])
            art.body_offset_rot[b] = inv.multiply(brot[b]).normalized()
        # 2. composite inertia per link (link body frame)
        for li in range(len(art.links)):
            var mass = 0.0
            var msum = Vec3()
            for b in range(1, nb):
                if art.body_link[b] != li or not model.bodies[b].has_inertial:
                    continue
                ref body = model.bodies[b]
                var c = art.body_offset_pos[b] + mj_rotate(art.body_offset_rot[b], body.ipos)
                mass += body.mass
                msum = msum + c * Float32(body.mass)
            var com = msum * Float32(1.0 / mass) if mass > 0 else Vec3()
            var tensor = Mat3()
            for b in range(1, nb):
                if art.body_link[b] != li or not model.bodies[b].has_inertial:
                    continue
                ref body = model.bodies[b]
                var c = art.body_offset_pos[b] + mj_rotate(art.body_offset_rot[b], body.ipos)
                var r = Mat3.rotation(art.body_offset_rot[b].multiply(body.iquat))
                var local = r.mul(Mat3.diag(Float64(body.inertia.x), Float64(body.inertia.y), Float64(body.inertia.z))).mul(r.transposed())
                var dd = c - com
                var dx = Float64(dd.x); var dy = Float64(dd.y); var dz = Float64(dd.z)
                var pa = Mat3()
                pa.m[0] = body.mass * (dy * dy + dz * dz); pa.m[4] = body.mass * (dx * dx + dz * dz); pa.m[8] = body.mass * (dx * dx + dy * dy)
                pa.m[1] = -body.mass * dx * dy; pa.m[3] = pa.m[1]
                pa.m[2] = -body.mass * dx * dz; pa.m[6] = pa.m[2]
                pa.m[5] = -body.mass * dy * dz; pa.m[7] = pa.m[5]
                tensor = tensor.add(local).add(pa)
            var e = symmetric_eigen(tensor)
            art.links[li].mass = mass
            art.links[li].com = com
            art.links[li].frame = e[1]
            art.links[li].moments = Vec3(max(e[0].x, Float32(1.0e-6)), max(e[0].y, Float32(1.0e-6)), max(e[0].z, Float32(1.0e-6)))
        # 3. Flint bodies
        var filt = CollisionFilter(CAT_DEBRIS, CAT_ALL, 0 if self_collide else -1)
        for li in range(len(art.links)):
            ref l = art.links[li]
            var lb = l.body
            var cw = bpos[lb] + mj_rotate(brot[lb], l.com)
            var fw = brot[lb].multiply(l.frame).normalized()
            var inv = l.frame.conjugate()
            var shape = CompoundShape()
            var nchild = 0
            for b in range(1, nb):
                if art.body_link[b] != li:
                    continue
                for gi in model.bodies[b].geoms:
                    ref g = model.geoms[gi]
                    if not g.collides():
                        continue
                    # geom pose in the link's principal (Flint body) frame, MuJoCo axes
                    var gp = art.body_offset_pos[b] + mj_rotate(art.body_offset_rot[b], g.pos)
                    var gq = art.body_offset_rot[b].multiply(g.quat)
                    var lp = mj_rotate(inv, gp - l.com)
                    var lq = inv.multiply(gq).normalized()
                    if g.kind == MJ_GEOM_SPHERE:
                        shape.add_child(CompoundChild.sphere(lp, g.size.x))
                        nchild += 1
                    elif g.kind == MJ_GEOM_CAPSULE or g.kind == MJ_GEOM_CYLINDER:
                        # Flint capsules lie along local +y; MuJoCo's along the geom's z
                        var cq = lq.multiply(Quat(_S2, 0, 0, _S2)).normalized()
                        shape.add_child(CompoundChild.capsule(lp, cq, g.size.x, g.size.y))
                        nchild += 1
                    elif g.kind == MJ_GEOM_BOX:
                        shape.add_child(CompoundChild.box(lp, lq, g.size))
                        nchild += 1
            var slot: Int
            var wpos = mj_to_flint(cw)
            var wrot = mj_to_flint_quat(fw)
            if nchild > 0:
                slot = world.add_compound(wpos, shape, Float32(max(l.mass, 1.0e-3)), wrot, friction, 0.0, filt)
            else:
                slot = world.add_sphere(wpos, 0.01, Float32(max(l.mass, 1.0e-3)), wrot, friction, 0.0,
                                        CollisionFilter(CAT_DEBRIS, 0, -1))
            world.bodies[slot].mass = Float32(max(l.mass, 1.0e-3))
            world.bodies[slot].inv_mass = 1.0 / world.bodies[slot].mass
            # principal moments: MuJoCo frame axes -> Flint body axes are the same body axes
            world.bodies[slot].local_inertia = l.moments
            world.bodies[slot].linear_damping = 0.0
            world.bodies[slot].angular_damping = 0.0
            world.bodies[slot].can_sleep = False
            world.bodies[slot].restitution = 0.0
            art.links[li].slot = slot
        # 4. joints
        for li in range(len(art.links)):
            var mj = art.links[li].mj_joint
            if mj < 0:
                if art.links[li].parent_link < 0 and art.free_joint < 0:
                    var anchor = mj_to_flint(bpos[art.links[li].body])
                    var jd = JointDef(JOINT_REVOLUTE, -1, art.links[li].slot, anchor, Vec3(0, 1, 0))
                    jd.limits(0.0, 0.0)
                    art.links[li].joint = world.add_joint(jd)
                if art.root_link < 0:
                    art.root_link = li
                continue
            ref j = model.joints[mj]
            var lb = art.links[li].body
            var anchor = bpos[lb] + mj_rotate(brot[lb], j.pos)
            var axis = mj_rotate(brot[lb], j.axis)
            var pslot = art.links[art.links[li].parent_link].slot if art.links[li].parent_link >= 0 else -1
            var jd = JointDef(JOINT_REVOLUTE, pslot, art.links[li].slot, mj_to_flint(anchor), mj_to_flint(axis).normalized())
            if j.limited:
                jd.limits(Float32(j.lower), Float32(j.upper))
            if joint_hertz > 0:
                jd.hertz = joint_hertz
            var ji = world.add_joint(jd)
            if armature:
                world.joints[ji].armature = Float32(j.armature) * armature_scale
            if frictionloss:
                world.joints[ji].friction_torque = Float32(j.frictionloss)
            art.links[li].joint = ji
        # 5. actuators
        for a in model.actuators:
            var li = art.body_link[model.joints[a.joint].body]
            if art.links[li].mj_joint != a.joint:
                raise Error("MJCF articulation: actuator " + a.name + " drives an unsupported joint")
            art.act_link.append(li)
            art.act_effort.append(Float32(a.force_upper) if a.force_limited else Float32(0))
            var ji = art.links[li].joint
            if a.kind == MJ_ACT_POSITION:
                world.joints[ji].enable_drive = True
                world.joints[ji].drive_kp = Float32(a.kp)
                world.joints[ji].drive_kd = Float32(a.kv)
                world.joints[ji].drive_max_torque = art.act_effort[len(art.act_effort) - 1]
        if ground:
            art.ground_slot = world.add_box(Vec3(0, -0.5, 0), Vec3(50, 0.5, 50), 0.0, Quat(), friction, 0.0)
        world.enable_sleeping = False
        return art^

    # ------------------------------------------------------------------------------- state

    def set_state(self, mut world: PhysicsWorld, model: MjcfModel, qpos: List[Float64], qvel: List[Float64]):
        """Place every link at the MuJoCo state (qpos, qvel) and sync the rotors."""
        var fk = model.forward_kinematics(qpos)
        var vel = model.body_velocities(qpos, qvel)
        var fpos = fk[0].copy()
        var frot = fk[1].copy()
        var bv = vel[0].copy()
        var bw = vel[1].copy()
        for li in range(len(self.links)):
            ref l = self.links[li]
            var b = l.body
            var c = fpos[b] + mj_rotate(frot[b], l.com)
            var r = frot[b].multiply(l.frame).normalized()
            var w = bw[b]
            var v = bv[b] + w.cross(c - fpos[b])
            world.reset_body(l.slot, mj_to_flint(c), mj_to_flint_quat(r), mj_to_flint(v), mj_to_flint(w), True)
            if l.mj_joint >= 0 and l.joint >= 0:
                world.joints[l.joint].rotor_speed = Float32(qvel[model.joints[l.mj_joint].dof_adr])
                world.joints[l.joint].axial_impulse = 0
                world.joints[l.joint].lower_impulse = 0
                world.joints[l.joint].upper_impulse = 0
                world.joints[l.joint].linear_impulse = Vec3()
                world.joints[l.joint].angular_impulse = Vec3()

    def link_pose(self, world: PhysicsWorld, li: Int) -> Tuple[Vec3, Quat]:
        """The link body's origin pose in MuJoCo axes."""
        ref l = self.links[li]
        var r = flint_to_mj_quat(world.bodies[l.slot].rot).multiply(l.frame.conjugate()).normalized()
        var c = flint_to_mj(world.bodies[l.slot].pos)
        return (c - mj_rotate(r, l.com), r)

    def body_poses(self, world: PhysicsWorld, model: MjcfModel) -> Tuple[List[Vec3], List[Quat]]:
        """World pose of every MJCF body origin (MuJoCo axes, xyzw), body 0 = world."""
        var nb = len(model.bodies)
        var pos = List[Vec3](length=nb, fill=Vec3())
        var rot = List[Quat](length=nb, fill=Quat())
        var lp = List[Tuple[Vec3, Quat]]()
        for li in range(len(self.links)):
            lp.append(self.link_pose(world, li))
        for b in range(1, nb):
            var t = lp[self.body_link[b]]
            pos[b] = t[0] + mj_rotate(t[1], self.body_offset_pos[b])
            rot[b] = t[1].multiply(self.body_offset_rot[b]).normalized()
        return (pos^, rot^)

    def qpos(self, world: PhysicsWorld, model: MjcfModel) -> List[Float64]:
        var q = List[Float64](length=model.nq, fill=0.0)
        if self.free_joint >= 0:
            var t = self.link_pose(world, self.body_link[model.joints[self.free_joint].body])
            var a = model.joints[self.free_joint].qpos_adr
            q[a] = Float64(t[0].x); q[a + 1] = Float64(t[0].y); q[a + 2] = Float64(t[0].z)
            q[a + 3] = Float64(t[1].w); q[a + 4] = Float64(t[1].x); q[a + 5] = Float64(t[1].y); q[a + 6] = Float64(t[1].z)
        for l in self.links:
            if l.mj_joint < 0:
                continue
            ref jt = world.joints[l.joint]
            var ta = world.bodies[self.links[l.parent_link].slot].rot if l.parent_link >= 0 else Quat()
            q[model.joints[l.mj_joint].qpos_adr] = Float64(twist_angle(jt, ta, world.bodies[l.slot].rot))
        return q^

    def qvel(self, world: PhysicsWorld, model: MjcfModel) -> List[Float64]:
        var v = List[Float64](length=model.nv, fill=0.0)
        if self.free_joint >= 0:
            var li = self.body_link[model.joints[self.free_joint].body]
            ref l = self.links[li]
            var t = self.link_pose(world, li)
            var w = flint_to_mj(world.bodies[l.slot].angular_vel)
            var vc = flint_to_mj(world.bodies[l.slot].linear_vel)
            var c = flint_to_mj(world.bodies[l.slot].pos)
            var vo = vc + w.cross(t[0] - c)
            var wl = mj_rotate(t[1].conjugate(), w)
            var a = model.joints[self.free_joint].dof_adr
            v[a] = Float64(vo.x); v[a + 1] = Float64(vo.y); v[a + 2] = Float64(vo.z)
            v[a + 3] = Float64(wl.x); v[a + 4] = Float64(wl.y); v[a + 5] = Float64(wl.z)
        for l in self.links:
            if l.mj_joint < 0:
                continue
            ref jt = world.joints[l.joint]
            var ta = world.bodies[self.links[l.parent_link].slot].rot if l.parent_link >= 0 else Quat()
            var wa = world.bodies[self.links[l.parent_link].slot].angular_vel if l.parent_link >= 0 else Vec3()
            var axis = world_axis_a(jt, ta)
            v[model.joints[l.mj_joint].dof_adr] = Float64(axis.dot(world.bodies[l.slot].angular_vel - wa))
        return v^

    def set_command(self, mut world: PhysicsWorld, targets: List[Float32], kp: List[Float32], kd: List[Float32]):
        """Per actuator (model order): PD target (rad), stiffness (N m / rad), damping (N m s / rad)."""
        for a in range(len(self.act_link)):
            var ji = self.links[self.act_link[a]].joint
            world.joints[ji].enable_drive = True
            world.joints[ji].drive_target = targets[a]
            world.joints[ji].drive_velocity = 0
            world.joints[ji].drive_kp = kp[a]
            world.joints[ji].drive_kd = kd[a]
            world.joints[ji].drive_max_torque = self.act_effort[a]

    def set_efforts(mut self, effort: List[Float32]):
        """Per-actuator torque caps (N m), for files whose actuators carry no forcerange."""
        for a in range(min(len(effort), len(self.act_effort))):
            self.act_effort[a] = effort[a]

    def drive_torque(self, world: PhysicsWorld, a: Int, h: Float32) -> Float32:
        """Actuator a's torque over the last substep of length h."""
        return world.joints[self.links[self.act_link[a]].joint].drive_impulse / h

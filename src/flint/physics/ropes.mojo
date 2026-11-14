"""Rope chains in the simulation (package 2.12).

A sim rope is a few link bodies in a row, hung from an anchor (a point of
the world, or a body) down to an end body (a lantern, a bucket), each link
held to the next by a ball joint (a swing-twist joint with no limits). The
links are sub-stepped with everything else by the Soft Step, so a punch
swings the lantern and the rope follows. Links never collide with each
other or with the end (one negative collision group per rope) but do with
the terrain and other bodies.

Cut to drop: `PhysicsWorld.cut_rope` breaks the joint below a link (a joint
break event), and a world anchor that was in rock when the rope was made
lets go (the top joint breaks) when that rock is dug away: the lantern
falls. The record keeps the ids of its links, joints and
end, all state (the FRCE blob).

The rope as drawn is a cosmetic XPBD chain (physics/xpbd.mojo) pinned to the
links' joint points: smooth between few sim links, swaying in the wind
mirror. It is presentation only.
"""
from flint.core.math import Vec3
from flint.core.serial import BlobWriter, BlobReader

comptime MAX_ROPE_LINKS: Int = 12
comptime ROPE_GROUP_BASE: Int = 1 << 20  # rope r's collision group is -(ROPE_GROUP_BASE + r)


struct Rope(Copyable, Movable):
    var id: Int
    var anchor_body: Int   # 0: the world
    var anchor: Vec3       # world point (world anchor), or in the anchor body's frame from its centre of mass
    var end_body: Int
    var links: List[Int]   # body ids, top first
    var joints: List[Int]  # joint ids, top first: anchor-link0, link0-link1, ..., last link-end
    var segment: Float32   # link length
    var in_rock: Bool      # a world anchor that was in rock when made (it lets go when that rock is dug out)

    def __init__(out self):
        self.id = 0
        self.anchor_body = 0
        self.anchor = Vec3()
        self.end_body = 0
        self.links = List[Int]()
        self.joints = List[Int]()
        self.segment = 0
        self.in_rock = False

    def group(self) -> Int:
        return -(ROPE_GROUP_BASE + self.id)

    def write(self, mut out: BlobWriter):
        out.integer(self.id)
        out.integer(self.anchor_body)
        out.vec3(self.anchor)
        out.integer(self.end_body)
        out.f32(self.segment)
        out.boolean(self.in_rock)
        out.integer(len(self.links))
        for id in self.links:
            out.integer(id)
        out.integer(len(self.joints))
        for id in self.joints:
            out.integer(id)

    @staticmethod
    def read(mut input: BlobReader) raises -> Self:
        var r = Self()
        r.id = input.integer()
        r.anchor_body = input.integer()
        r.anchor = input.vec3()
        r.end_body = input.integer()
        r.segment = input.f32()
        r.in_rock = input.boolean()
        var links = input.integer()
        if links < 0 or links > MAX_ROPE_LINKS:
            raise Error("force blob: rope " + String(r.id) + " has " + String(links) + " links")
        for _ in range(links):
            r.links.append(input.integer())
        var joints = input.integer()
        if joints < 0 or joints > MAX_ROPE_LINKS + 1:
            raise Error("force blob: rope " + String(r.id) + " has " + String(joints) + " joints")
        for _ in range(joints):
            r.joints.append(input.integer())
        return r^

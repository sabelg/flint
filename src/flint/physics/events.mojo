"""Contact cache, per-tick physics events and sensors (package 2.11).

Contact cache. Every contact point has a key: the lower body id, the other
side (the higher body id, OTHER_TERRAIN, or a static box as -(1 + index)) and
the sample feature that made it. A key seen again keeps its persistent contact
id, age and last impulse, unless it moved too far to be the same contact: its
normal turned past CACHE_NORMAL_DOT, or its anchor (the point in the frame of
what it touches: the world for terrain) slid more than CACHE_DRIFT, as
Bullet's persistent manifolds do. Then it is recycled: a new id, age 0.
Entries not seen during a tick are dropped at its end, except where both
sides sleep (a sleeping body makes no contacts but still touches). The entries
are kept sorted by key and the ids handed out in the order contacts are
refreshed, which is the solver's fixed contact order, so the cache is the same
on every run. The solver (package 2.2) warm-starts from the stored impulses:
each entry keeps the last substep's normal impulse and its friction impulse as
a world vector, so it projects onto the next tick's tangents whatever they are.

Events. A tick's events go into one buffer in a fixed order: begin (a pair
starts touching), end (stops), hit (it touched approaching faster than
HIT_EVENT_SPEED, with that speed and the tick's summed normal impulse), sensor
begin and end, sleep and wake, joint break (a joint that broke this tick, with
the joint's id as a and its second body as b); since package 2.12 water
entry and exit (body id, the water volume's id; the point on the surface,
the vertical speed, mass x speed and the body's material: what a splash
needs) and kill (a kill volume took the body); since package 2.14 stick,
unstick and overload (physics/mechanics.mojo: magnets).
Inside each kind they are sorted by (a, b). Pairs are (lower id, higher id)
for two bodies and (body id, other) for terrain and static boxes.

Sensors are volumes (a box or a sphere) with a filter: a body whose box
overlaps one, and whose filter it accepts, is inside; entering and leaving
are events. Sensors never push anything.
"""
from flint.core.math import Vec3
from flint.core.serial import BlobWriter, BlobReader
from flint.physics.filter import CollisionFilter, CAT_ALL, CAT_SENSOR

comptime OTHER_TERRAIN: Int = 0

comptime EVENT_BEGIN: Int = 0
comptime EVENT_END: Int = 1
comptime EVENT_HIT: Int = 2
comptime EVENT_SENSOR_BEGIN: Int = 3
comptime EVENT_SENSOR_END: Int = 4
comptime EVENT_SLEEP: Int = 5
comptime EVENT_WAKE: Int = 6
comptime EVENT_JOINT_BREAK: Int = 7
comptime EVENT_WATER_ENTER: Int = 8   # package 2.12: a body went into water (a splash)
comptime EVENT_WATER_EXIT: Int = 9
comptime EVENT_KILL: Int = 10         # a kill volume took a body
# Package 2.14 (physics/mechanics.mojo): a: the body; b: the magnet's id
# (MAGNET_LODESTONE for lodestone rock); point: where; impulse: its mass.
comptime EVENT_STICK: Int = 11        # a ferrous body stuck to a magnet
comptime EVENT_UNSTICK: Int = 12      # it let go (switched off, broke, or the magnet's overload)
comptime EVENT_OVERLOAD: Int = 13     # a: the magnet's id; impulse: the load it held (kg); it drops everything
comptime EVENT_KINDS: Int = 14

comptime CACHE_NORMAL_DOT: Float32 = 0.95  # recycle a contact whose normal turned more (~18 degrees)
comptime CACHE_DRIFT: Float32 = 0.1        # or whose anchor moved more (voxels, in the body frame)
comptime HIT_EVENT_SPEED: Float32 = 1.0    # m/s of approach for a hit event (Box2D's default)


def event_name(kind: Int) -> String:
    var names = ["begin", "end", "hit", "sensor_begin", "sensor_end", "sleep", "wake", "joint_break", "water_enter", "water_exit", "kill", "stick", "unstick", "overload"]
    if kind < 0 or kind >= len(names):
        return "unknown"
    return String(names[kind])


def static_other(index: Int) -> Int:
    """The `other` of a contact with static box `index`."""
    return -(1 + index)


struct PhysicsEvent(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var a: Int           # body id (a sensor's id for sensor events)
    var b: Int           # other side: body id, OTHER_TERRAIN, static -(1 + i); the body for sensor events
    var point: Vec3
    var normal: Vec3     # from b toward a
    var speed: Float32   # approach speed (hit)
    var impulse: Float32 # summed normal impulse this tick (hit); mass x speed (water entry and exit)
    var material: Int    # the body's material (water entry and exit, kill)

    def __init__(out self, kind: Int, a: Int, b: Int, point: Vec3 = Vec3(), normal: Vec3 = Vec3(), speed: Float32 = 0, impulse: Float32 = 0):
        self.kind = kind
        self.a = a
        self.b = b
        self.point = point
        self.normal = normal
        self.speed = speed
        self.impulse = impulse
        self.material = 0


def event_before(x: PhysicsEvent, y: PhysicsEvent) -> Bool:
    if x.kind != y.kind:
        return x.kind < y.kind
    if x.a != y.a:
        return x.a < y.a
    return x.b < y.b


def sort_events(mut events: List[PhysicsEvent]):
    """Stable insertion sort by (kind, a, b): short lists, fixed order."""
    for index in range(1, len(events)):
        var item = events[index]
        var at = index - 1
        while at >= 0 and event_before(item, events[at]):
            events[at + 1] = events[at]
            at -= 1
        events[at + 1] = item


struct CachedContact(Copyable, ImplicitlyCopyable, Movable):
    var a: Int
    var b: Int
    var feature: Int
    var id: Int          # persistent contact id
    var ticks: Int       # ticks it has lasted
    var normal: Vec3
    var anchor: Vec3     # the point in the other side's frame (the world for terrain and static boxes)
    var impulse: Float32 # last normal impulse
    var tangent: Vec3    # last friction impulse, world (package 2.2)
    var seen: Bool       # refreshed this tick

    def __init__(out self, a: Int, b: Int, feature: Int, id: Int, normal: Vec3, anchor: Vec3, impulse: Float32):
        self.a = a
        self.b = b
        self.feature = feature
        self.id = id
        self.ticks = 0
        self.normal = normal
        self.anchor = anchor
        self.impulse = impulse
        self.tangent = Vec3()
        self.seen = True


def key_before(a: Int, b: Int, feature: Int, entry: CachedContact) -> Bool:
    if a != entry.a:
        return a < entry.a
    if b != entry.b:
        return b < entry.b
    return feature < entry.feature


def pair_key_before(a: Int, b: Int, c: Int, d: Int) -> Bool:
    return a < c or (a == c and b < d)


struct PairTouch(Copyable, ImplicitlyCopyable, Movable):
    """One pair over this tick: contacts seen, the fastest approach and the
    summed impulse, and where (for hit events)."""
    var a: Int
    var b: Int
    var speed: Float32
    var impulse: Float32
    var point: Vec3
    var normal: Vec3

    def __init__(out self, a: Int, b: Int):
        self.a = a
        self.b = b
        self.speed = 0
        self.impulse = 0
        self.point = Vec3()
        self.normal = Vec3()


struct ContactCache(Copyable, Movable):
    var entries: List[CachedContact]  # sorted by (a, b, feature)
    var next_id: Int
    var touching_a: List[Int]         # pairs touching at the end of the last tick, sorted
    var touching_b: List[Int]
    var tick_pairs: List[PairTouch]   # this tick so far, sorted by (a, b); not state between ticks
    var recycled: Int                 # diagnostics: contacts recycled this tick
    var persisted: Int                # and kept

    def __init__(out self):
        self.entries = List[CachedContact]()
        self.next_id = 0
        self.touching_a = List[Int]()
        self.touching_b = List[Int]()
        self.tick_pairs = List[PairTouch]()
        self.recycled = 0
        self.persisted = 0

    def clear(mut self):
        self.entries.clear()
        self.next_id = 0
        self.touching_a.clear()
        self.touching_b.clear()
        self.tick_pairs.clear()

    def is_default(self) -> Bool:
        return len(self.entries) == 0 and len(self.touching_a) == 0 and self.next_id == 0

    def begin_tick(mut self):
        for index in range(len(self.entries)):
            self.entries[index].seen = False
        self.tick_pairs.clear()
        self.recycled = 0
        self.persisted = 0

    def find(self, a: Int, b: Int, feature: Int) -> Int:
        """Index of the key, or -(insert position) - 1."""
        var lo = 0
        var hi = len(self.entries)
        while lo < hi:
            var mid = (lo + hi) // 2
            if key_before(a, b, feature, self.entries[mid]):
                hi = mid
            elif self.entries[mid].a == a and self.entries[mid].b == b and self.entries[mid].feature == feature:
                return mid
            else:
                lo = mid + 1
        return -lo - 1

    def refresh(mut self, a: Int, b: Int, feature: Int, normal: Vec3, anchor: Vec3, impulse: Float32) -> Int:
        """A contact seen in a substep; returns its persistent id."""
        var at = self.find(a, b, feature)
        if at >= 0:
            var entry = self.entries[at]
            var turned = entry.normal.x * normal.x + entry.normal.y * normal.y + entry.normal.z * normal.z < CACHE_NORMAL_DOT
            var drifted = (entry.anchor - anchor).length_squared() > CACHE_DRIFT * CACHE_DRIFT
            if turned or drifted:
                entry.id = self.next_id
                self.next_id += 1
                entry.ticks = 0
                self.recycled += 1
            elif not entry.seen:
                self.persisted += 1
            entry.normal = normal
            entry.anchor = anchor
            entry.impulse = impulse
            entry.seen = True
            self.entries[at] = entry
            return entry.id
        var position = -at - 1
        var made = CachedContact(a, b, feature, self.next_id, normal, anchor, impulse)
        self.next_id += 1
        self.entries.insert(position, made)
        return made.id

    def warm(mut self, a: Int, b: Int, feature: Int, normal: Vec3, anchor: Vec3) -> CachedContact:
        """A contact found this tick (package 2.2: once per tick, before the
        solve): refreshes its entry as `refresh` does and returns it, with
        the impulses to warm-start from (zero for a new or recycled contact)."""
        var at = self.find(a, b, feature)
        var fresh = at < 0
        var impulse = Float32(0)
        var tangent = Vec3()
        if not fresh:
            var entry = self.entries[at]
            fresh = (entry.normal.x * normal.x + entry.normal.y * normal.y + entry.normal.z * normal.z < CACHE_NORMAL_DOT
                     or (entry.anchor - anchor).length_squared() > CACHE_DRIFT * CACHE_DRIFT)
            impulse = entry.impulse
            tangent = entry.tangent
        _ = self.refresh(a, b, feature, normal, anchor, 0)
        at = self.find(a, b, feature)
        var entry = self.entries[at]
        entry.impulse = Float32(0) if fresh else impulse
        entry.tangent = Vec3() if fresh else tangent
        self.entries[at] = entry
        return entry

    def store(mut self, a: Int, b: Int, feature: Int, impulse: Float32, tangent: Vec3):
        """The solved impulses of a contact `warm` returned, for next tick."""
        var at = self.find(a, b, feature)
        if at >= 0:
            self.entries[at].impulse = impulse
            self.entries[at].tangent = tangent

    def touch(mut self, a: Int, b: Int, speed: Float32, impulse: Float32, point: Vec3, normal: Vec3):
        """Pair (a, b) touched in a substep."""
        var lo = 0
        var hi = len(self.tick_pairs)
        while lo < hi:
            var mid = (lo + hi) // 2
            if pair_key_before(a, b, self.tick_pairs[mid].a, self.tick_pairs[mid].b):
                hi = mid
            elif self.tick_pairs[mid].a == a and self.tick_pairs[mid].b == b:
                var pair = self.tick_pairs[mid]
                pair.impulse += impulse
                if speed > pair.speed:
                    pair.speed = speed
                    pair.point = point
                    pair.normal = normal
                self.tick_pairs[mid] = pair
                return
            else:
                lo = mid + 1
        var made = PairTouch(a, b)
        made.speed = speed
        made.impulse = impulse
        made.point = point
        made.normal = normal
        self.tick_pairs.insert(lo, made)

    def end_tick(mut self, mut events: List[PhysicsEvent]):
        """Drop unseen contacts, age the rest, and add the begin, end and hit
        events (already in (a, b) order within each kind)."""
        var kept = List[CachedContact](capacity=len(self.entries))
        for entry in self.entries:
            if entry.seen:
                var aged = entry
                aged.ticks += 1
                kept.append(aged)
        self.entries = kept^
        # Begin: in this tick's pairs, not in last tick's. End: the other way.
        var old = 0
        for pair in self.tick_pairs:
            while old < len(self.touching_a) and pair_key_before(self.touching_a[old], self.touching_b[old], pair.a, pair.b):
                events.append(PhysicsEvent(EVENT_END, self.touching_a[old], self.touching_b[old]))
                old += 1
            if old < len(self.touching_a) and self.touching_a[old] == pair.a and self.touching_b[old] == pair.b:
                old += 1
            else:
                events.append(PhysicsEvent(EVENT_BEGIN, pair.a, pair.b, pair.point, pair.normal, pair.speed, pair.impulse))
        while old < len(self.touching_a):
            events.append(PhysicsEvent(EVENT_END, self.touching_a[old], self.touching_b[old]))
            old += 1
        for pair in self.tick_pairs:
            if pair.speed >= HIT_EVENT_SPEED:
                events.append(PhysicsEvent(EVENT_HIT, pair.a, pair.b, pair.point, pair.normal, pair.speed, pair.impulse))
        self.touching_a.clear()
        self.touching_b.clear()
        for pair in self.tick_pairs:
            self.touching_a.append(pair.a)
            self.touching_b.append(pair.b)

    def keep_resting(mut self, sleeping: List[Int]):
        """Sleeping bodies make no contacts, but they still touch: a contact
        or pair whose body sleeps and whose other side is terrain, a static box
        or another sleeper is kept as it was (no end event, then a begin when
        it wakes). `sleeping` lists the ids of the bodies asleep."""
        for index in range(len(self.entries)):
            if not self.entries[index].seen and resting(self.entries[index].a, self.entries[index].b, sleeping):
                self.entries[index].seen = True
        for index in range(len(self.touching_a)):
            var a = self.touching_a[index]
            var b = self.touching_b[index]
            if resting(a, b, sleeping):
                self.touch(a, b, 0, 0, Vec3(), Vec3())

    def serialize(self, mut out: BlobWriter):
        out.integer(self.next_id)
        out.integer(len(self.entries))
        for entry in self.entries:
            out.integer(entry.a)
            out.integer(entry.b)
            out.integer(entry.feature)
            out.integer(entry.id)
            out.integer(entry.ticks)
            out.vec3(entry.normal)
            out.vec3(entry.anchor)
            out.f32(entry.impulse)
            out.vec3(entry.tangent)
        out.integer(len(self.touching_a))
        for index in range(len(self.touching_a)):
            out.integer(self.touching_a[index])
            out.integer(self.touching_b[index])

    def deserialize(mut self, mut input: BlobReader, version: Int = 2) raises:
        """Version 1 (2.11) had no friction impulse."""
        self.clear()
        self.next_id = input.integer()
        var count = input.integer()
        if count < 0 or count > 1 << 20:
            raise Error("contact cache blob lists " + String(count) + " contacts")
        for _ in range(count):
            var a = input.integer()
            var b = input.integer()
            var feature = input.integer()
            var id = input.integer()
            var ticks = input.integer()
            var normal = input.vec3()
            var anchor = input.vec3()
            var entry = CachedContact(a, b, feature, id, normal, anchor, input.f32())
            if version >= 2:
                entry.tangent = input.vec3()
            entry.ticks = ticks
            self.entries.append(entry)
        var pairs = input.integer()
        if pairs < 0 or pairs > 1 << 20:
            raise Error("contact cache blob lists " + String(pairs) + " touching pairs")
        for _ in range(pairs):
            self.touching_a.append(input.integer())
            self.touching_b.append(input.integer())


def asleep(id: Int, sleeping: List[Int]) -> Bool:
    for other in sleeping:
        if other == id:
            return True
    return False


def resting(a: Int, b: Int, sleeping: List[Int]) -> Bool:
    if b <= 0:
        return asleep(a, sleeping)
    return asleep(a, sleeping) and asleep(b, sleeping)


struct Sensor(Copyable, ImplicitlyCopyable, Movable):
    """A volume that reports bodies entering and leaving. `radius` > 0 makes
    it a sphere, otherwise it is the box `center` +- `half`."""
    var id: Int
    var center: Vec3
    var half: Vec3
    var radius: Float32
    var filter: CollisionFilter

    def __init__(out self, id: Int, center: Vec3, half: Vec3, radius: Float32, filter: CollisionFilter):
        self.id = id
        self.center = center
        self.half = half
        self.radius = radius
        self.filter = filter

    def overlaps_box(self, lo: Vec3, hi: Vec3) -> Bool:
        if self.radius > 0:
            var x = max(lo.x, min(self.center.x, hi.x)) - self.center.x
            var y = max(lo.y, min(self.center.y, hi.y)) - self.center.y
            var z = max(lo.z, min(self.center.z, hi.z)) - self.center.z
            return x * x + y * y + z * z <= self.radius * self.radius
        return (
            lo.x <= self.center.x + self.half.x and hi.x >= self.center.x - self.half.x
            and lo.y <= self.center.y + self.half.y and hi.y >= self.center.y - self.half.y
            and lo.z <= self.center.z + self.half.z and hi.z >= self.center.z - self.half.z
        )


def sensor_filter(mask: UInt32 = CAT_ALL) -> CollisionFilter:
    return CollisionFilter(CAT_SENSOR, mask, 0)

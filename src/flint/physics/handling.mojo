"""What package 2.3 adds to the physics: flight without drag, landings, and
bodies that split when their grid comes apart.

Flight. A thrown body is `drag_free`: no linear damping and no air drag (the
2.12 wind) until its first touching contact, so a throw keeps the speed the
ballistic aim solved for (with material damping a stone loses about 15% a
second). Water still acts on it. The first contact that pushed it this tick
(a predicted contact stops a fast body a hair before it touches) or that
touches ends the flight and is reported as a
`Landing` in that tick's output: which body, against what, where, the
approach speed when the tick began, and the velocity it flew with.

Splits. An edit that empties a cell of a body's grid (surf wear now, 2.5's
impact fracture later) queues the body; at the next update, before contacts,
a grid whose voxels are no longer 6-connected becomes one body per component.
Children take new ids in a fixed order: largest first, ties to the component
whose first cell (y, z, x order) comes first. Each keeps its voxels where they
were in the world, the parent's turn, the parent's velocity at its own centre
of mass (v + w x r) and spin, the parent's flight and filter. The parent
leaves; the split is reported (`BodySplit`) so a holder re-targets the largest
child. Children that find no free row are dropped as chips.
"""
from flint.core.math import Vec3


struct Landing(Copyable, ImplicitlyCopyable, Movable):
    """A thrown body's first touch (tick output, not state)."""
    var id: Int        # the body
    var other: Int     # the other body's id, or 0 for terrain and static boxes
    var point: Vec3
    var normal: Vec3
    var speed: Float32     # approach speed along the normal when the tick began
    var velocity: Vec3     # the body's velocity before the tick's solve

    def __init__(out self, id: Int, other: Int, point: Vec3, normal: Vec3, speed: Float32, velocity: Vec3):
        self.id = id
        self.other = other
        self.point = point
        self.normal = normal
        self.speed = speed
        self.velocity = velocity


struct BodySplit(Copyable, Movable):
    """A body that came apart this tick and the ids of its children, largest
    first (tick output, not state). Children spawned during a tick join the
    simulation at the tick's commit."""
    var parent: Int
    var children: List[Int]
    var voxels: List[Int]   # each child's voxel count, same order
    var dropped: Int        # voxels of components that found no row

    def __init__(out self, parent: Int):
        self.parent = parent
        self.children = List[Int]()
        self.voxels = List[Int]()
        self.dropped = 0

    def largest(self) -> Int:
        return self.children[0] if len(self.children) > 0 else 0


struct GridComponent(Copyable, Movable):
    """One 6-connected part of a grid: its cells (indices into the grid, in
    fill order) and inclusive bounds."""
    var cells: List[Int]
    var first: Int
    var lo_x: Int
    var lo_y: Int
    var lo_z: Int
    var hi_x: Int
    var hi_y: Int
    var hi_z: Int

    def __init__(out self, first: Int):
        self.cells = List[Int]()
        self.first = first
        self.lo_x = 1 << 30
        self.lo_y = 1 << 30
        self.lo_z = 1 << 30
        self.hi_x = -1
        self.hi_y = -1
        self.hi_z = -1

    def size_x(self) -> Int:
        return self.hi_x - self.lo_x + 1

    def size_y(self) -> Int:
        return self.hi_y - self.lo_y + 1

    def size_z(self) -> Int:
        return self.hi_z - self.lo_z + 1


def grid_components(cells: List[UInt8], base: Int, size_x: Int, size_y: Int, size_z: Int) -> List[GridComponent]:
    """The 6-connected components of a grid's solid cells (ordered y, z, x
    from `base`), largest first; ties keep the order of their first cell."""
    var count = size_x * size_y * size_z
    var seen = List[Bool](length=count, fill=False)
    var found = List[GridComponent]()
    var queue = List[Int]()
    for start in range(count):
        if seen[start] or cells[base + start] == 0:
            continue
        var part = GridComponent(start)
        queue.clear()
        queue.append(start)
        seen[start] = True
        var cursor = 0
        while cursor < len(queue):
            var cell = queue[cursor]
            cursor += 1
            part.cells.append(cell)
            var x = cell % size_x
            var z = (cell // size_x) % size_z
            var y = cell // (size_x * size_z)
            part.lo_x = min(part.lo_x, x)
            part.lo_y = min(part.lo_y, y)
            part.lo_z = min(part.lo_z, z)
            part.hi_x = max(part.hi_x, x)
            part.hi_y = max(part.hi_y, y)
            part.hi_z = max(part.hi_z, z)
            # Neighbours in a fixed order: -x, +x, -z, +z, -y, +y.
            if x > 0:
                _visit(cells, base, cell - 1, seen, queue)
            if x + 1 < size_x:
                _visit(cells, base, cell + 1, seen, queue)
            if z > 0:
                _visit(cells, base, cell - size_x, seen, queue)
            if z + 1 < size_z:
                _visit(cells, base, cell + size_x, seen, queue)
            if y > 0:
                _visit(cells, base, cell - size_x * size_z, seen, queue)
            if y + 1 < size_y:
                _visit(cells, base, cell + size_x * size_z, seen, queue)
        found.append(part^)
    # Largest first, stable (insertion sort: a split has few parts).
    var ordered = List[GridComponent]()
    for part in found:
        var at = len(ordered)
        while at > 0 and len(ordered[at - 1].cells) < len(part.cells):
            at -= 1
        ordered.insert(at, part.copy())
    return ordered^


@always_inline
def _visit(cells: List[UInt8], base: Int, cell: Int, mut seen: List[Bool], mut queue: List[Int]):
    if not seen[cell] and cells[base + cell] != 0:
        seen[cell] = True
        queue.append(cell)


def component_cells(cells: List[UInt8], base: Int, size_x: Int, size_z: Int, part: GridComponent) -> List[UInt8]:
    """A component's own grid over its bounds (ordered y, z, x), air elsewhere."""
    var sx = part.size_x()
    var sz = part.size_z()
    var out = List[UInt8](length=sx * part.size_y() * sz, fill=0)
    for cell in part.cells:
        var x = cell % size_x
        var z = (cell // size_x) % size_z
        var y = cell // (size_x * size_z)
        out[((y - part.lo_y) * sz + (z - part.lo_z)) * sx + (x - part.lo_x)] = cells[base + cell]
    return out^

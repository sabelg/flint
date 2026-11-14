"""Persistent islands and the persistent constraint graph colouring (package 2.2).

Islands. Every body row has an island label. A body starts as an island of its
own, labelled with its body id. A *link* is a touching contact between two
awake bodies, or a joint that is not broken. A link that appears merges the
two islands: the higher label's bodies take the lower label, so a label is
always the lowest body id the island had when it formed, and labels never
repeat (body ids never do). A link that disappears does not split anything at
once: the island is marked *pending* (Box2D v3's constraintRemoveCount).
Splitting is deferred and rate-limited: at most ISLAND_SPLITS_PER_TICK pending
islands are split per tick, the sleepiest first (the largest smallest
sleep_ticks among its bodies; ties go to the lower label), by a breadth-first
walk over the current links in sorted order; each piece is labelled with its
lowest body id. An island sleeps as a whole once every body in it has rested
SLEEP_TICKS ticks and it is not pending; a pending island that is ready to sleep
is the first to be split. Links between two sleeping bodies are kept as they
were (sleeping bodies make no contacts but still touch), so an island that
falls asleep does not look as if it came apart.

Constraint graph. Contact manifolds are coloured so that no colour holds two
constraints on one moving body (a pair with two manifolds needs two colours); a colour's constraints are then solved in
parallel, in batches of eight, without locks (Box2D v3's graph colouring).
The colouring persists: a pair keeps last tick's colour while it keeps
touching. Each tick, the pairs kept from last tick claim their colours first,
in (id, other) order, then new pairs (and kept ones whose colour is no longer
free, because the other side woke) take the first free colour, in the same
order. Pairs of two bodies search up from colour 0; pairs with terrain, a
static box or a sleeping body (one moving body) search down from the last
colour, so they are solved last in every pass: Gauss-Seidel gives the last
word to the constraints solved last, and the terrain is what everything rests
on (the "priority colours"). A pair that finds no free colour among
GRAPH_COLOURS goes to the overflow colour, which one worker solves alone, in
order, before the coloured stages. "Pair" below means one manifold's key:
the pair and the manifold's place among the pair's.
"""
from flint.core.detsort import stable_order
from flint.core.serial import BlobWriter, BlobReader

comptime GRAPH_COLOURS: Int = 24
comptime OVERFLOW_COLOUR: Int = GRAPH_COLOURS  # solved serially, in constraint order
comptime ISLAND_SPLITS_PER_TICK: Int = 1
comptime NO_ISLAND: Int = -1


@always_inline
def pair_key(a: Int, b: Int) -> Int:
    """One sortable integer for (a, b): a body id first, then the other side
    (a body id, 0 for terrain, -(1 + i) for static box i). Ids stay far below
    2^31."""
    return (a << 32) | ((b + (1 << 31)) & 0xFFFFFFFF)


@always_inline
def manifold_key(pair: Int, ordinal: Int) -> Int:
    """A constraint's key in the graph: its pair's key and which of the
    pair's manifolds it is (at most four normal groups per pair). Body ids
    stay below 2^29."""
    return (pair << 2) | (ordinal & 3)


struct IslandSet(Copyable, Movable):
    """Island labels per body row, the links seen last tick (sorted body id
    pairs), and the labels waiting for a split."""
    var label: List[Int]        # per row; NO_ISLAND for a row without a body
    var link_a: List[Int]       # last tick's links, sorted by (a, b), a < b (body ids)
    var link_b: List[Int]
    var pending: List[Int]      # labels with removed links, sorted
    var splits: Int             # diagnostics: islands split this tick
    var merges: Int             # diagnostics: merges this tick

    def __init__(out self):
        self.label = List[Int]()
        self.link_a = List[Int]()
        self.link_b = List[Int]()
        self.pending = List[Int]()
        self.splits = 0
        self.merges = 0

    def clear(mut self):
        self.label.clear()
        self.link_a.clear()
        self.link_b.clear()
        self.pending.clear()
        self.splits = 0
        self.merges = 0

    def is_default(self, ids: List[Int], active: List[Bool]) -> Bool:
        """True when every body is an island of its own and nothing is linked
        or pending: the state a world without the 2.2 blob loads into."""
        if len(self.link_a) > 0 or len(self.pending) > 0:
            return False
        for row in range(len(self.label)):
            var expected = ids[row] if row < len(active) and active[row] else NO_ISLAND
            if self.label[row] != expected:
                return False
        return True

    def ensure_rows(mut self, rows: Int):
        while len(self.label) < rows:
            self.label.append(NO_ISLAND)

    def set_single(mut self, row: Int, id: Int):
        """A new body: an island of its own."""
        self.ensure_rows(row + 1)
        self.label[row] = id

    def drop(mut self, row: Int):
        """A body left: links to it are dropped next tick, which marks its old
        island pending (what stays may have come apart)."""
        if row < len(self.label):
            self.label[row] = NO_ISLAND

    def mark_pending(mut self, label: Int):
        if label == NO_ISLAND:
            return
        var at = 0
        while at < len(self.pending) and self.pending[at] < label:
            at += 1
        if at < len(self.pending) and self.pending[at] == label:
            return
        self.pending.insert(at, label)

    def is_pending(self, label: Int) -> Bool:
        for item in self.pending:
            if item == label:
                return True
            if item > label:
                return False
        return False

    def relabel(mut self, old: Int, new: Int):
        for row in range(len(self.label)):
            if self.label[row] == old:
                self.label[row] = new
        # A pending mark follows the bodies.
        var was = False
        var kept = List[Int]()
        for item in self.pending:
            if item == old:
                was = True
            else:
                kept.append(item)
        self.pending = kept^
        if was:
            self.mark_pending(new)

    def update_links(mut self, now_a: List[Int], now_b: List[Int], now_row_a: List[Int], now_row_b: List[Int], ids: List[Int]):
        """This tick's links (body id pairs with a < b, in any order, and the
        rows of their two ends) against last tick's: new links merge islands,
        missing ones mark islands pending. `ids` maps a row to its body id."""
        self.merges = 0
        var count = len(now_a)
        var keys = List[Int](capacity=count)
        for k in range(count):
            keys.append(pair_key(now_a[k], now_b[k]))
        var order = stable_order(keys)
        var sorted_a = List[Int](capacity=count)
        var sorted_b = List[Int](capacity=count)
        var rows_a = List[Int](capacity=count)
        var rows_b = List[Int](capacity=count)
        var last = -1
        for k in order:
            if len(sorted_a) > 0 and keys[k] == last:
                continue  # a pair seen twice (two manifolds, or a contact and a joint)
            last = keys[k]
            sorted_a.append(now_a[k])
            sorted_b.append(now_b[k])
            rows_a.append(now_row_a[k])
            rows_b.append(now_row_b[k])
        # Removed links: in last tick's list, not in this one.
        var old = 0
        for k in range(len(sorted_a)):
            var key = pair_key(sorted_a[k], sorted_b[k])
            while old < len(self.link_a) and pair_key(self.link_a[old], self.link_b[old]) < key:
                self.removed_link(self.link_a[old], self.link_b[old], ids)
                old += 1
            if old < len(self.link_a) and pair_key(self.link_a[old], self.link_b[old]) == key:
                old += 1
        while old < len(self.link_a):
            self.removed_link(self.link_a[old], self.link_b[old], ids)
            old += 1
        # Merges, in sorted order.
        for k in range(len(sorted_a)):
            var la = self.label[rows_a[k]]
            var lb = self.label[rows_b[k]]
            if la != lb and la != NO_ISLAND and lb != NO_ISLAND:
                self.relabel(max(la, lb), min(la, lb))
                self.merges += 1
        self.link_a = sorted_a^
        self.link_b = sorted_b^

    def removed_link(mut self, a: Int, b: Int, ids: List[Int]):
        """A link went away: whichever island its ends are in now may have
        come apart."""
        for row in range(len(self.label)):
            if row < len(ids) and (ids[row] == a or ids[row] == b) and self.label[row] != NO_ISLAND:
                self.mark_pending(self.label[row])

    def split(mut self, label: Int, ids: List[Int]):
        """Splits one island into its connected pieces over the current links;
        each piece takes its lowest body id as its label."""
        var members = List[Int]()
        for row in range(len(self.label)):
            if self.label[row] == label:
                members.append(row)
        var kept = List[Int]()
        for item in self.pending:
            if item != label:
                kept.append(item)
        self.pending = kept^
        if len(members) <= 1:
            return
        var piece = List[Int](length=len(members), fill=-1)
        var pieces = 0
        for start in range(len(members)):
            if piece[start] >= 0:
                continue
            piece[start] = pieces
            var queue = List[Int]()
            queue.append(start)
            var cursor = 0
            while cursor < len(queue):
                var here = ids[members[queue[cursor]]]
                cursor += 1
                for k in range(len(self.link_a)):
                    var other = -1
                    if self.link_a[k] == here:
                        other = self.link_b[k]
                    elif self.link_b[k] == here:
                        other = self.link_a[k]
                    if other < 0:
                        continue
                    for m in range(len(members)):
                        if piece[m] < 0 and ids[members[m]] == other:
                            piece[m] = pieces
                            queue.append(m)
            pieces += 1
        if pieces <= 1:
            return
        self.splits += 1
        for p in range(pieces):
            var lowest = -1
            for m in range(len(members)):
                if piece[m] == p and (lowest < 0 or ids[members[m]] < lowest):
                    lowest = ids[members[m]]
            for m in range(len(members)):
                if piece[m] == p:
                    self.label[members[m]] = lowest

    def serialize(self, mut out: BlobWriter):
        out.integer(len(self.label))
        for value in self.label:
            out.integer(value)
        out.integer(len(self.link_a))
        for k in range(len(self.link_a)):
            out.integer(self.link_a[k])
            out.integer(self.link_b[k])
        out.integer(len(self.pending))
        for value in self.pending:
            out.integer(value)

    def deserialize(mut self, mut input: BlobReader) raises:
        self.clear()
        var rows = input.integer()
        if rows < 0 or rows > 1 << 16:
            raise Error("island blob lists " + String(rows) + " rows")
        for _ in range(rows):
            self.label.append(input.integer())
        var links = input.integer()
        if links < 0 or links > 1 << 20:
            raise Error("island blob lists " + String(links) + " links")
        for _ in range(links):
            self.link_a.append(input.integer())
            self.link_b.append(input.integer())
        var pending = input.integer()
        if pending < 0 or pending > 1 << 16:
            raise Error("island blob lists " + String(pending) + " pending islands")
        for _ in range(pending):
            self.pending.append(input.integer())


struct ConstraintGraph(Copyable, Movable):
    """The persistent colouring: last tick's pairs (sorted keys) and their
    colours."""
    var keys: List[Int]
    var colours: List[Int]
    var overflow: Int   # diagnostics: pairs in the overflow colour this tick
    var used: Int       # diagnostics: colours with a pair this tick

    def __init__(out self):
        self.keys = List[Int]()
        self.colours = List[Int]()
        self.overflow = 0
        self.used = 0

    def clear(mut self):
        self.keys.clear()
        self.colours.clear()
        self.overflow = 0
        self.used = 0

    def is_default(self) -> Bool:
        return len(self.keys) == 0

    def previous(self, key: Int) -> Int:
        var lo = 0
        var hi = len(self.keys)
        while lo < hi:
            var mid = (lo + hi) // 2
            if self.keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(self.keys) and self.keys[lo] == key:
            return self.colours[lo]
        return -1

    def assign(mut self, keys: List[Int], rows_a: List[Int], rows_b: List[Int], rows: Int) -> List[Int]:
        """Colours for this tick's pairs. `keys` are the pairs' sorted unique
        keys; `rows_a`/`rows_b` the moving rows of each pair (-1 for a side
        that does not move: terrain, a static box, a sleeping body). Returns
        each pair's colour (OVERFLOW_COLOUR for the overflow). The debug cvar
        that forces the serial path does not change the colours: the solver
        then runs every pair on one worker in colour order, which gives the
        same result."""
        var count = len(keys)
        var result = List[Int](length=count, fill=-1)
        var body_bits = List[UInt32](length=rows, fill=0)
        # 1. Pairs kept from last tick claim their colour, if it is still free.
        for k in range(count):
            var colour = self.previous(keys[k])
            if colour < 0 or colour >= GRAPH_COLOURS:
                continue
            var bit = UInt32(1) << UInt32(colour)
            var a = rows_a[k]
            var b = rows_b[k]
            if (a >= 0 and (body_bits[a] & bit) != 0) or (b >= 0 and (body_bits[b] & bit) != 0):
                continue
            result[k] = colour
            if a >= 0:
                body_bits[a] |= bit
            if b >= 0:
                body_bits[b] |= bit
        # 2. The rest take the first free colour: pairs of two moving bodies
        # from colour 0 up, pairs with one moving body from the last down.
        var overflow = 0
        for k in range(count):
            if result[k] >= 0:
                continue
            var a = rows_a[k]
            var b = rows_b[k]
            var taken = UInt32(0)
            if a >= 0:
                taken |= body_bits[a]
            if b >= 0:
                taken |= body_bits[b]
            var colour = -1
            if a >= 0 and b >= 0:
                for c in range(GRAPH_COLOURS):
                    if (taken & (UInt32(1) << UInt32(c))) == 0:
                        colour = c
                        break
            else:
                var c = GRAPH_COLOURS - 1
                while c >= 0:
                    if (taken & (UInt32(1) << UInt32(c))) == 0:
                        colour = c
                        break
                    c -= 1
            if colour < 0:
                result[k] = OVERFLOW_COLOUR
                overflow += 1
                continue
            result[k] = colour
            var bit = UInt32(1) << UInt32(colour)
            if a >= 0:
                body_bits[a] |= bit
            if b >= 0:
                body_bits[b] |= bit
        self.keys = keys.copy()
        self.colours = result.copy()
        self.overflow = overflow
        var seen = UInt32(0)
        for colour in result:
            if colour < GRAPH_COLOURS:
                seen |= UInt32(1) << UInt32(colour)
        var used = 0
        for c in range(GRAPH_COLOURS):
            if (seen & (UInt32(1) << UInt32(c))) != 0:
                used += 1
        self.used = used
        return result^

    def serialize(self, mut out: BlobWriter):
        out.integer(len(self.keys))
        for k in range(len(self.keys)):
            out.integer(self.keys[k])
            out.integer(self.colours[k])

    def deserialize(mut self, mut input: BlobReader) raises:
        self.clear()
        var count = input.integer()
        if count < 0 or count > 1 << 20:
            raise Error("colour blob lists " + String(count) + " pairs")
        for _ in range(count):
            self.keys.append(input.integer())
            self.colours.append(input.integer())

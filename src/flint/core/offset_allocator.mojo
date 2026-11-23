"""An O(1) offset allocator for sub-allocating one big GPU buffer (package 1.6).

After Sebastian Aaltonen's OffsetAllocator (MIT, https://github.com/sebbbi/OffsetAllocator;
see THIRD_PARTY_NOTICES). It hands out ranges of abstract *units* inside a heap
of `size` units; the caller decides what a unit is (the chunk-mesh heap uses
16 bytes). Nothing here touches the memory itself: the bookkeeping lives on
the CPU, so the heap can be VRAM the CPU never reads.

Free ranges sit in 256 bins named by an 8-bit float code of their size (3-bit
mantissa, 5-bit exponent): exact below 8, then 8 steps per power of two, so a
range is never more than 12.5% bigger than its bin's floor. A two-level bitfield
(one 32-bit word of non-empty top bins, one 8-bit word of non-empty leaf bins
per top bin) finds the smallest non-empty bin that certainly fits with two
bit scans, whatever the heap holds. An allocation rounds its size *up* to a bin
code and takes a range from that bin or higher; a free range is filed *down*,
so any range in a bin is at least the bin's floor. The rest of the range taken
goes back as a new free range.

Every range, used or free, is a node in a doubly linked list of neighbours in
address order, so a free merges with free neighbours on both sides at once:
freeing everything always leaves one range of the whole heap. Nodes come from a
fixed array of `max_allocations` (a free list of node indices), so nothing
allocates after construction. Free nodes of one bin are a second linked list.

Presentation only: GPU memory placement never feeds the simulation. It is still
fully deterministic (no hashing, no time), which keeps tests repeatable.
"""
from std.bit import count_trailing_zeros, count_leading_zeros

comptime OA_NUM_TOP_BINS: Int = 32
comptime OA_BINS_PER_LEAF: Int = 8
comptime OA_TOP_BINS_INDEX_SHIFT: Int = 3
comptime OA_LEAF_BINS_INDEX_MASK: Int = 7
comptime OA_NUM_LEAF_BINS: Int = OA_NUM_TOP_BINS * OA_BINS_PER_LEAF

comptime OA_MANTISSA_BITS: Int = 3
comptime OA_MANTISSA_VALUE: Int = 1 << OA_MANTISSA_BITS
comptime OA_MANTISSA_MASK: Int = OA_MANTISSA_VALUE - 1

comptime OA_UNUSED: UInt32 = 0xFFFFFFFF  # no node
comptime OA_NO_SPACE: UInt32 = 0xFFFFFFFF  # an allocation's offset when it failed


def highest_bit(value: UInt32) -> Int:
    return 31 - Int(count_leading_zeros(value))


def size_to_bin_round_up(size: UInt32) -> Int:
    """The 8-bit float code of `size`, rounded up: the smallest bin whose
    every range is at least `size`."""
    var exponent = 0
    var mantissa: Int
    if Int(size) < OA_MANTISSA_VALUE:
        mantissa = Int(size)
    else:
        var mantissa_start = highest_bit(size) - OA_MANTISSA_BITS
        exponent = mantissa_start + 1
        mantissa = Int(size >> UInt32(mantissa_start)) & OA_MANTISSA_MASK
        var low_bits = (UInt32(1) << UInt32(mantissa_start)) - 1
        if (size & low_bits) != 0:
            mantissa += 1  # may carry into the exponent, which is still right
    return (exponent << OA_MANTISSA_BITS) + mantissa


def size_to_bin_round_down(size: UInt32) -> Int:
    """The 8-bit float code of `size`, rounded down: the bin a free range of
    this size is filed under."""
    var exponent = 0
    var mantissa: Int
    if Int(size) < OA_MANTISSA_VALUE:
        mantissa = Int(size)
    else:
        var mantissa_start = highest_bit(size) - OA_MANTISSA_BITS
        exponent = mantissa_start + 1
        mantissa = Int(size >> UInt32(mantissa_start)) & OA_MANTISSA_MASK
    return (exponent << OA_MANTISSA_BITS) | mantissa


def bin_to_size(code: Int) -> UInt32:
    """The smallest size a bin's code stands for."""
    var exponent = code >> OA_MANTISSA_BITS
    var mantissa = code & OA_MANTISSA_MASK
    if exponent == 0:
        return UInt32(mantissa)
    return UInt32(mantissa | OA_MANTISSA_VALUE) << UInt32(exponent - 1)


def lowest_set_bit_after(mask: UInt32, start: Int) -> UInt32:
    """Index of the lowest set bit at or above `start`, or OA_NO_SPACE."""
    if start >= 32:
        return OA_NO_SPACE
    var after = mask & ~((UInt32(1) << UInt32(start)) - 1)
    if after == 0:
        return OA_NO_SPACE
    return UInt32(count_trailing_zeros(after))


struct OffsetAllocation(Copyable, ImplicitlyCopyable, Movable):
    """A range: its first unit, and the node to hand back to `free`."""
    var offset: UInt32
    var metadata: UInt32

    def __init__(out self, offset: UInt32 = OA_NO_SPACE, metadata: UInt32 = OA_UNUSED):
        self.offset = offset
        self.metadata = metadata

    def ok(self) -> Bool:
        return self.offset != OA_NO_SPACE


struct OffsetNode(Copyable, ImplicitlyCopyable, Movable):
    var data_offset: UInt32
    var data_size: UInt32
    var bin_prev: UInt32
    var bin_next: UInt32
    var neighbor_prev: UInt32
    var neighbor_next: UInt32
    var used: Bool

    def __init__(out self):
        self.data_offset = 0
        self.data_size = 0
        self.bin_prev = OA_UNUSED
        self.bin_next = OA_UNUSED
        self.neighbor_prev = OA_UNUSED
        self.neighbor_next = OA_UNUSED
        self.used = False


struct OffsetAllocator(Movable):
    """Ranges of a heap of `size` units, at most `max_allocations` nodes (used
    and free ranges together; a heap cut into n used ranges needs at most
    2n + 1)."""
    var size: UInt32
    var max_allocations: Int
    var free_storage: UInt32
    var used_bins_top: UInt32
    var used_bins: List[UInt8]   # OA_NUM_TOP_BINS
    var bin_heads: List[UInt32]  # OA_NUM_LEAF_BINS: first free node of each bin
    var nodes: List[OffsetNode]
    var free_nodes: List[UInt32]  # a stack of node indices
    var free_count: Int           # how many of free_nodes are free
    var allocations: Int          # used ranges now
    var peak_used: UInt32         # most units in use at once

    def __init__(out self, size: UInt32, max_allocations: Int = 128 * 1024):
        self.size = size
        self.max_allocations = max(2, max_allocations)
        self.free_storage = 0
        self.used_bins_top = 0
        self.used_bins = List[UInt8](length=OA_NUM_TOP_BINS, fill=0)
        self.bin_heads = List[UInt32](length=OA_NUM_LEAF_BINS, fill=OA_UNUSED)
        self.nodes = List[OffsetNode](length=self.max_allocations, fill=OffsetNode())
        self.free_nodes = List[UInt32](length=self.max_allocations, fill=0)
        self.free_count = 0
        self.allocations = 0
        self.peak_used = 0
        self.reset()

    def reset(mut self):
        """Everything free again: one range of the whole heap."""
        self.free_storage = 0
        self.used_bins_top = 0
        self.allocations = 0
        for top in range(OA_NUM_TOP_BINS):
            self.used_bins[top] = 0
        for code in range(OA_NUM_LEAF_BINS):
            self.bin_heads[code] = OA_UNUSED
        for index in range(self.max_allocations):
            self.nodes[index] = OffsetNode()
            # Popped from the end, so node 0 goes first.
            self.free_nodes[index] = UInt32(self.max_allocations - index - 1)
        self.free_count = self.max_allocations
        if self.size > 0:
            _ = self.insert_node_into_bin(self.size, 0)

    def used_storage(self) -> UInt32:
        return self.size - self.free_storage

    def largest_free(self) -> UInt32:
        """A lower bound on the largest range `allocate` can still return
        (the floor of the highest non-empty bin, as OffsetAllocator reports it)."""
        if self.used_bins_top == 0:
            return 0
        var top = highest_bit(self.used_bins_top)
        var leaf = highest_bit(UInt32(self.used_bins[top]))
        return bin_to_size((top << OA_TOP_BINS_INDEX_SHIFT) | leaf)

    def allocate(mut self, size: UInt32) -> OffsetAllocation:
        """A range of `size` units (at least 1), or one with offset OA_NO_SPACE
        when no free range is certainly big enough or the nodes ran out."""
        if size == 0 or self.free_count < 1:
            return OffsetAllocation()  # a split needs a spare node for the rest
        var min_bin = size_to_bin_round_up(size)
        var min_top = min_bin >> OA_TOP_BINS_INDEX_SHIFT
        var min_leaf = min_bin & OA_LEAF_BINS_INDEX_MASK
        if min_top >= OA_NUM_TOP_BINS:
            return OffsetAllocation()
        var top = min_top
        var leaf = OA_NO_SPACE
        if (self.used_bins_top & (UInt32(1) << UInt32(top))) != 0:
            leaf = lowest_set_bit_after(UInt32(self.used_bins[top]), min_leaf)
        if leaf == OA_NO_SPACE:
            var higher = lowest_set_bit_after(self.used_bins_top, min_top + 1)
            if higher == OA_NO_SPACE:
                return OffsetAllocation()
            top = Int(higher)
            leaf = UInt32(count_trailing_zeros(UInt32(self.used_bins[top])))
        var code = (top << OA_TOP_BINS_INDEX_SHIFT) | Int(leaf)
        var node_index = self.bin_heads[code]
        var node = self.nodes[Int(node_index)]
        var total = node.data_size
        node.data_size = size
        node.used = True
        self.bin_heads[code] = node.bin_next
        if node.bin_next != OA_UNUSED:
            self.nodes[Int(node.bin_next)].bin_prev = OA_UNUSED
        node.bin_prev = OA_UNUSED
        node.bin_next = OA_UNUSED
        self.free_storage -= total
        if self.bin_heads[code] == OA_UNUSED:
            self.used_bins[top] &= ~UInt8(1 << Int(leaf))
            if self.used_bins[top] == 0:
                self.used_bins_top &= ~(UInt32(1) << UInt32(top))
        self.nodes[Int(node_index)] = node
        var rest = total - size
        if rest > 0:
            var new_index = self.insert_node_into_bin(rest, node.data_offset + size)
            if node.neighbor_next != OA_UNUSED:
                self.nodes[Int(node.neighbor_next)].neighbor_prev = new_index
            self.nodes[Int(new_index)].neighbor_prev = node_index
            self.nodes[Int(new_index)].neighbor_next = node.neighbor_next
            self.nodes[Int(node_index)].neighbor_next = new_index
        self.allocations += 1
        var used = self.used_storage()
        if used > self.peak_used:
            self.peak_used = used
        return OffsetAllocation(node.data_offset, node_index)

    def free(mut self, allocation: OffsetAllocation):
        """Hand a range back, merging it with free neighbours."""
        if allocation.metadata == OA_UNUSED:
            return
        var node_index = allocation.metadata
        var idx = Int(node_index)
        if idx < 0 or idx >= len(self.nodes):
            return
        var node = self.nodes[idx]
        if not node.used:
            return  # a double free is a caller bug; refuse rather than corrupt
        var offset = node.data_offset
        var size = node.data_size
        if node.neighbor_prev != OA_UNUSED and Int(node.neighbor_prev) < len(self.nodes) and not self.nodes[Int(node.neighbor_prev)].used:
            var prev = self.nodes[Int(node.neighbor_prev)]
            offset = prev.data_offset
            size += prev.data_size
            self.remove_node_from_bin(node.neighbor_prev)
            node.neighbor_prev = prev.neighbor_prev
        if node.neighbor_next != OA_UNUSED and Int(node.neighbor_next) < len(self.nodes) and not self.nodes[Int(node.neighbor_next)].used:
            var next = self.nodes[Int(node.neighbor_next)]
            size += next.data_size
            self.remove_node_from_bin(node.neighbor_next)
            node.neighbor_next = next.neighbor_next
        var neighbor_next = node.neighbor_next
        var neighbor_prev = node.neighbor_prev
        self.nodes[idx] = OffsetNode()
        self.free_nodes[self.free_count] = node_index
        self.free_count += 1
        var combined = self.insert_node_into_bin(size, offset)
        if neighbor_next != OA_UNUSED and Int(neighbor_next) < len(self.nodes):
            self.nodes[Int(combined)].neighbor_next = neighbor_next
            self.nodes[Int(neighbor_next)].neighbor_prev = combined
        if neighbor_prev != OA_UNUSED and Int(neighbor_prev) < len(self.nodes):
            self.nodes[Int(combined)].neighbor_prev = neighbor_prev
            self.nodes[Int(neighbor_prev)].neighbor_next = combined
        self.allocations -= 1

    def allocation_size(self, allocation: OffsetAllocation) -> UInt32:
        if allocation.metadata == OA_UNUSED:
            return 0
        var idx = Int(allocation.metadata)
        if idx < 0 or idx >= len(self.nodes):
            return 0
        if not self.nodes[idx].used:
            return 0
        return self.nodes[idx].data_size

    def insert_node_into_bin(mut self, size: UInt32, data_offset: UInt32) -> UInt32:
        var code = size_to_bin_round_down(size)
        var top = code >> OA_TOP_BINS_INDEX_SHIFT
        var leaf = code & OA_LEAF_BINS_INDEX_MASK
        if self.bin_heads[code] == OA_UNUSED:
            self.used_bins[top] |= UInt8(1 << leaf)
            self.used_bins_top |= UInt32(1) << UInt32(top)
        var head = self.bin_heads[code]
        self.free_count -= 1
        var node_index = self.free_nodes[self.free_count]
        var node = OffsetNode()
        node.data_offset = data_offset
        node.data_size = size
        node.bin_next = head
        self.nodes[Int(node_index)] = node
        if head != OA_UNUSED:
            self.nodes[Int(head)].bin_prev = node_index
        self.bin_heads[code] = node_index
        self.free_storage += size
        return node_index

    def remove_node_from_bin(mut self, node_index: UInt32):
        var node = self.nodes[Int(node_index)]
        if node.bin_prev != OA_UNUSED:
            self.nodes[Int(node.bin_prev)].bin_next = node.bin_next
            if node.bin_next != OA_UNUSED:
                self.nodes[Int(node.bin_next)].bin_prev = node.bin_prev
        else:
            var code = size_to_bin_round_down(node.data_size)
            var top = code >> OA_TOP_BINS_INDEX_SHIFT
            var leaf = code & OA_LEAF_BINS_INDEX_MASK
            self.bin_heads[code] = node.bin_next
            if node.bin_next != OA_UNUSED:
                self.nodes[Int(node.bin_next)].bin_prev = OA_UNUSED
            if self.bin_heads[code] == OA_UNUSED:
                self.used_bins[top] &= ~UInt8(1 << leaf)
                if self.used_bins[top] == 0:
                    self.used_bins_top &= ~(UInt32(1) << UInt32(top))
        self.nodes[Int(node_index)] = OffsetNode()
        self.free_nodes[self.free_count] = node_index
        self.free_count += 1
        self.free_storage -= node.data_size

    def free_ranges(self) -> List[Tuple[UInt32, UInt32]]:
        """Every free range as (offset, size), by walking the bins: for tests
        and the memory page, not for the frame."""
        var ranges = List[Tuple[UInt32, UInt32]]()
        for code in range(OA_NUM_LEAF_BINS):
            var index = self.bin_heads[code]
            while index != OA_UNUSED:
                ref node = self.nodes[Int(index)]
                ranges.append((node.data_offset, node.data_size))
                index = node.bin_next
        return ranges^

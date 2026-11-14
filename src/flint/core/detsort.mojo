"""Our own stable sort, heap and hash for simulation code (package 0.12).

The standard library's sort is not promised to be stable or to stay the same
between versions, its hash is seeded per process, and iterating a hash map
visits entries in an order that depends on capacity and insertion history. Any
of those in the simulation would make two machines (or two runs) disagree, so
simulation code uses these instead (docs/DETERMINISM.md):

- `stable_order(keys)`: the indices of `keys` in ascending order, equal keys
  in their original order (a bottom-up merge sort, O(n log n), no recursion).
- `MinHeap`: a binary heap of (key, value) whose ties pop in push order, so a
  flood fill or an event queue visits cells in one fixed order.
- `mix64` and `hash_words`: a fixed integer hash (splitmix64's finaliser) for
  hashing positions or ids into buckets; buckets are iterated by index, never
  by a map's own order.
"""


def stable_order(keys: List[Int]) -> List[Int]:
    """Indices that visit `keys` in ascending order; ties keep their order."""
    var count = len(keys)
    var order = List[Int](capacity=count)
    for index in range(count):
        order.append(index)
    if count < 2:
        return order^
    var scratch = List[Int](length=count, fill=0)
    var width = 1
    while width < count:
        var start = 0
        while start < count:
            var middle = min(start + width, count)
            var end = min(start + 2 * width, count)
            var left = start
            var right = middle
            var out = start
            while left < middle and right < end:
                # `<=` takes the left (earlier) one on a tie: stable.
                if keys[order[left]] <= keys[order[right]]:
                    scratch[out] = order[left]
                    left += 1
                else:
                    scratch[out] = order[right]
                    right += 1
                out += 1
            while left < middle:
                scratch[out] = order[left]
                left += 1
                out += 1
            while right < end:
                scratch[out] = order[right]
                right += 1
                out += 1
            start = end
        for index in range(count):
            order[index] = scratch[index]
        width *= 2
    return order^


def stable_sort(mut values: List[Int]):
    """Sorts integers ascending (equal integers are indistinguishable)."""
    var order = stable_order(values)
    var sorted = List[Int](capacity=len(values))
    for index in order:
        sorted.append(values[index])
    values = sorted^


struct HeapEntry(Copyable, ImplicitlyCopyable, Movable):
    var key: Int
    var sequence: Int  # push order: ties pop first-in first-out
    var value: Int

    def __init__(out self, key: Int, sequence: Int, value: Int):
        self.key = key
        self.sequence = sequence
        self.value = value

    @always_inline
    def before(self, other: Self) -> Bool:
        if self.key != other.key:
            return self.key < other.key
        return self.sequence < other.sequence


struct MinHeap(Movable, Sized):
    """Smallest key first; equal keys in the order they were pushed."""
    var entries: List[HeapEntry]
    var pushed: Int

    def __init__(out self):
        self.entries = List[HeapEntry]()
        self.pushed = 0

    def __len__(self) -> Int:
        return len(self.entries)

    def clear(mut self):
        self.entries.clear()
        self.pushed = 0

    def push(mut self, key: Int, value: Int):
        self.entries.append(HeapEntry(key, self.pushed, value))
        self.pushed += 1
        var at = len(self.entries) - 1
        while at > 0:
            var parent = (at - 1) // 2
            if not self.entries[at].before(self.entries[parent]):
                break
            var swap = self.entries[at]
            self.entries[at] = self.entries[parent]
            self.entries[parent] = swap
            at = parent

    def peek(self) -> HeapEntry:
        return self.entries[0]

    def pop(mut self) -> HeapEntry:
        """The smallest entry. The heap must not be empty."""
        var top = self.entries[0]
        var last = self.entries.pop()
        var count = len(self.entries)
        if count == 0:
            return top
        self.entries[0] = last
        var at = 0
        while True:
            var left = at * 2 + 1
            if left >= count:
                break
            var child = left
            var right = left + 1
            if right < count and self.entries[right].before(self.entries[left]):
                child = right
            if not self.entries[child].before(self.entries[at]):
                break
            var swap = self.entries[at]
            self.entries[at] = self.entries[child]
            self.entries[child] = swap
            at = child
        return top


@always_inline
def mix64(value: UInt64) -> UInt64:
    """splitmix64's finaliser: a fixed, well-mixed 64-bit hash of a word."""
    var z = value + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def hash_words(words: List[Int]) -> UInt64:
    """A fixed hash of a sequence of integers (cell coordinates, ids)."""
    var value = UInt64(0x51ED270B27A0F1C3)
    for word in words:
        value = mix64(value ^ UInt64(word))
    return value

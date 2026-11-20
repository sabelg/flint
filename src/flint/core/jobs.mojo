"""A pool of worker threads running a graph of jobs.

A job is a function, a context address, and a number of chunks; every chunk
runs once, on any thread. Jobs can depend on other jobs: a job becomes ready
when its dependency counter reaches zero, and a job is finished when its last
chunk has. Any thread that calls `wait` runs ready jobs until the job it wants
is finished, so a waiting thread is never idle while work remains, a job body
may itself wait, and a pool with no workers runs the whole graph on the caller.

    var meshes = pool.create(run_mesh_chunk, Int(Pointer(to=work)), len(chunks))
    var skins = pool.create(run_skin_chunk, Int(Pointer(to=skin_work)), count)
    pool.depend(skins, meshes)        # skins start after meshes finish
    pool.submit(meshes)
    pool.submit(skins)
    pool.wait(skins)

The context must stay alive until the job is finished, so a job on the stack
is waited for in the same scope. A job waiting on another job from inside its
body must not be something that job in turn needs; `depend` is the tool for
ordering, and a body that only waits for its own children is always safe.

Nothing here allocates once the pool has started. Slots hold jobs, and the
ready ring holds handles. A handle is a slot and a generation, and the ticket
word packs a generation, chunk total and next chunk, so a thread that took a
handle from the ring late can never claim a chunk of a newer job in that slot.
Workers spin a little, then sleep on a futex until a job is pushed.
"""
from std.atomic import Atomic, Ordering, fence
from std.ffi import external_call
from std.sys.info import CompilationTarget
from flint.core.crash import install_thread_stack
from flint.core.detmath import set_sim_float_mode
from flint.core.profiler import Profiler, ZONE_NONE

comptime JobBody = def(Int, Int) thin -> None
comptime NO_JOB: Int = -1

comptime JOB_PRIO_HIGH: Int = 0
comptime JOB_PRIO_NORMAL: Int = 1
comptime JOB_PRIO_LOW: Int = 2
comptime JOB_PRIO_COUNT: Int = 3

comptime CACHE_LINE_WORDS: Int = 8
comptime EPOCH_WORD: Int = 0
comptime SLEEPERS_WORD: Int = CACHE_LINE_WORDS
comptime NEXT_SLOT_WORD: Int = CACHE_LINE_WORDS * 2
comptime STOP_WORD: Int = CACHE_LINE_WORDS * 3
comptime PROFILER_WORD: Int = STOP_WORD + 1  # shares a line with a word that never changes in play
comptime WORKERS_WORD: Int = STOP_WORD + 2
comptime START_GATE_WORD: Int = STOP_WORD + 3
comptime MAIN_TID_WORD: Int = STOP_WORD + 4
comptime COMPLETED_WORD: Int = CACHE_LINE_WORDS * 4
comptime HELPED_WORD: Int = CACHE_LINE_WORDS * 5
comptime THREADS_WORD: Int = CACHE_LINE_WORDS * 6
comptime MAX_WORKERS: Int = 32
comptime WORKER_BOOT_WORDS: Int = 2
comptime WORKER_BOOT_WORD: Int = THREADS_WORD + MAX_WORKERS

# 3-tier global bounded MPMC priority rings (fallback for external threads & deque overflow)
comptime GLOBAL_RING_CELLS: Int = 1024
comptime GLOBAL_RING_MASK: Int = GLOBAL_RING_CELLS - 1
comptime GLOBAL_RING_CELL_WORDS: Int = 2
comptime GLOBAL_META_STRIDE: Int = CACHE_LINE_WORDS * 2  # push line + pop line
comptime GLOBAL_META_WORD: Int = WORKER_BOOT_WORD + MAX_WORKERS * WORKER_BOOT_WORDS
comptime GLOBAL_RINGS_WORD: Int = GLOBAL_META_WORD + JOB_PRIO_COUNT * GLOBAL_META_STRIDE

# 64-byte cache-line-aligned per-worker Chase-Lev deques: (MAX_WORKERS + 1) slots x 3 priorities
comptime DEQUE_SLOTS: Int = MAX_WORKERS + 1
comptime DEQUE_CAP: Int = 1024
comptime DEQUE_MASK: Int = DEQUE_CAP - 1
comptime DEQUE_BOTTOM_OFF: Int = 0
comptime DEQUE_TOP_OFF: Int = CACHE_LINE_WORDS
comptime DEQUE_BUF_OFF: Int = CACHE_LINE_WORDS * 2
comptime DEQUE_STRIDE_WORDS: Int = CACHE_LINE_WORDS * 2 + DEQUE_CAP
comptime DEQUES_WORD: Int = GLOBAL_RINGS_WORD + JOB_PRIO_COUNT * GLOBAL_RING_CELLS * GLOBAL_RING_CELL_WORDS

comptime SLOTS_WORD: Int = DEQUES_WORD + DEQUE_SLOTS * JOB_PRIO_COUNT * DEQUE_STRIDE_WORDS

comptime SLOT_BITS: Int = 10
comptime JOB_SLOTS: Int = 1 << SLOT_BITS
comptime SLOT_WORDS: Int = CACHE_LINE_WORDS * 4
comptime POOL_WORDS: Int = SLOTS_WORD + JOB_SLOTS * SLOT_WORDS
# First line: written once while the job is built, then only read.
comptime SLOT_BODY: Int = 0
comptime SLOT_CONTEXT: Int = 1
comptime SLOT_CHUNKS: Int = 2
comptime SLOT_GENERATION: Int = 3
comptime SLOT_STATE: Int = 4
comptime SLOT_ZONE: Int = 5
comptime SLOT_PRIORITY: Int = 6
# Second line: the words threads fight over.
comptime SLOT_TICKET: Int = CACHE_LINE_WORDS
comptime SLOT_FINISHED: Int = CACHE_LINE_WORDS + 1
comptime SLOT_PENDING: Int = CACHE_LINE_WORDS + 2
comptime SLOT_LOCK: Int = CACHE_LINE_WORDS + 3
comptime SLOT_DONE: Int = CACHE_LINE_WORDS + 4
comptime SLOT_DEPENDENT_COUNT: Int = CACHE_LINE_WORDS + 5
comptime SLOT_DEP_OVERFLOW_HEAD: Int = CACHE_LINE_WORDS + 6
comptime SLOT_DEP_OVERFLOW_TAIL: Int = CACHE_LINE_WORDS + 7
comptime SLOT_DEPENDENTS: Int = CACHE_LINE_WORDS * 2
comptime MAX_DEPENDENTS: Int = CACHE_LINE_WORDS * 2

# Dynamic overflow block for jobs with > MAX_DEPENDENTS dependents:
# [0] = next block addr, [1] = count in block, [2..33] = dependent job handles
comptime _DEP_OVERFLOW_CAP: Int = 32
comptime _DEP_OVERFLOW_WORDS: Int = 2 + _DEP_OVERFLOW_CAP

comptime STATE_FREE: Int = 0
comptime STATE_BUILDING: Int = 1
comptime STATE_LIVE: Int = 2

comptime MAX_CHUNKS: Int = 1048575
comptime TICKET_GENERATIONS: Int = 0x7FFFFF  # keeps the ticket word positive
comptime SPINS_BEFORE_SLEEP: Int = 64  # low-latency yielding before sleeping on futex
comptime WORKER_STACK_BYTES: Int = 8 << 20  # a simulation tick needs the main thread's stack

comptime SYS_FUTEX: Int = 202 if CompilationTarget.is_x86() else 98
comptime FUTEX_WAIT_PRIVATE: Int = 128
comptime FUTEX_WAKE_PRIVATE: Int = 129
comptime FUTEX_WAKE_EVERYONE: Int = 0x7FFFFFFF


def job_slot(handle: Int) -> Int:
    return handle & (JOB_SLOTS - 1)


def job_generation(handle: Int) -> Int:
    return handle >> SLOT_BITS


def job_worker(boot_ptr: Int) abi("C") -> Int:
    """Thread body: run ready jobs until the pool stops, sleeping when there are none."""
    # Its own signal stack, so a crash in a job is reported even when it is a stack overflow.
    _ = install_thread_stack()
    # One floating-point mode for every thread that may run a tick
    # (docs/DETERMINISM.md): round to nearest, FTZ and DAZ off.
    set_sim_float_mode()
    var boot = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=boot_ptr)
    var pool_addr = Int(boot[unsafe_offset=0])
    var worker_slot = Int(boot[unsafe_offset=1])
    var pool = _JobPoolRef(pool_addr, worker_slot)
    while pool.word(START_GATE_WORD)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]() == 0 and not pool.stopping():
        _ = external_call["sched_yield", Int32]()
    var spins = 0
    while True:
        if pool.help():
            spins = 0
            continue
        if pool.stopping():
            return 0
        spins += 1
        if spins < SPINS_BEFORE_SLEEP:
            _ = external_call["sched_yield", Int32]()
        else:
            pool.sleep()
            spins = 0


struct _JobPoolRef(Copyable, Movable):
    """Non-owning view over JobPool shared memory (no __deinit__; safe for worker threads)."""
    var address: Int
    var owner_slot: Int  # 0 = main, 1..MAX_WORKERS = worker slot, -1 = dynamic lookup

    @always_inline
    def __init__(out self, address: Int, owner_slot: Int = -1):
        self.address = address
        self.owner_slot = owner_slot

    @always_inline
    def word(self, index: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=self.address + index * 8)

    @always_inline
    def slot_word(self, slot: Int, index: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return self.word(SLOTS_WORD + slot * SLOT_WORDS + index)

    @always_inline
    def worker_count(self) -> Int:
        return Int(self.word(WORKERS_WORD)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())

    @always_inline
    def stopping(self) -> Bool:
        return self.word(STOP_WORD)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]() != 0

    @always_inline
    def _clamp_prio(self, prio: Int) -> Int:
        if prio < JOB_PRIO_HIGH:
            return JOB_PRIO_HIGH
        if prio > JOB_PRIO_LOW:
            return JOB_PRIO_LOW
        return prio

    @always_inline
    def resolve_owner_slot(self) -> Int:
        if self.owner_slot >= 0:
            return self.owner_slot
        var self_tid = Int64(external_call["pthread_self", UInt64]())
        var main_word = self.word(MAIN_TID_WORD)
        var tid = main_word[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
        if tid == 0:
            var expected = Int64(0)
            if main_word[unsafe_offset=0].compare_exchange[
                failure_ordering=Ordering.ACQUIRE, success_ordering=Ordering.SEQUENTIAL
            ](expected, self_tid):
                return 0
            tid = expected
        if tid == self_tid:
            return 0
        var n_workers = self.worker_count()
        for index in range(n_workers):
            if self.word(THREADS_WORD + index)[unsafe_offset=0].load[ordering=Ordering.RELAXED]() == self_tid:
                return index + 1
        return -1

    # ---- Per-worker 64B-aligned Chase-Lev lock-free work-stealing deques ----

    @always_inline
    def deque_word(self, slot: Int, prio: Int, offset: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        var base = DEQUES_WORD + (slot * JOB_PRIO_COUNT + prio) * DEQUE_STRIDE_WORDS
        return self.word(base + offset)

    @always_inline
    def push_bottom(self, slot: Int, prio: Int, handle: Int) -> Bool:
        var b = self.deque_word(slot, prio, DEQUE_BOTTOM_OFF)[unsafe_offset=0].load[ordering=Ordering.RELAXED]()
        var t = self.deque_word(slot, prio, DEQUE_TOP_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
        if b - t >= Int64(DEQUE_CAP):
            return False
        self.deque_word(slot, prio, DEQUE_BUF_OFF + (Int(b) & DEQUE_MASK))[unsafe_offset=0].store[ordering=Ordering.RELAXED](Int64(handle))
        self.deque_word(slot, prio, DEQUE_BOTTOM_OFF)[unsafe_offset=0].store[ordering=Ordering.RELEASE](b + 1)
        return True

    @always_inline
    def pop_bottom(self, slot: Int, prio: Int) -> Int:
        var b = self.deque_word(slot, prio, DEQUE_BOTTOM_OFF)[unsafe_offset=0].load[ordering=Ordering.RELAXED]() - 1
        self.deque_word(slot, prio, DEQUE_BOTTOM_OFF)[unsafe_offset=0].store[ordering=Ordering.RELAXED](b)
        fence[Ordering.SEQUENTIAL]()
        var t = self.deque_word(slot, prio, DEQUE_TOP_OFF)[unsafe_offset=0].load[ordering=Ordering.RELAXED]()
        if t <= b:
            var item = Int(self.deque_word(slot, prio, DEQUE_BUF_OFF + (Int(b) & DEQUE_MASK))[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
            if t == b:
                var expected = t
                if not self.deque_word(slot, prio, DEQUE_TOP_OFF)[unsafe_offset=0].compare_exchange[
                    failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
                ](expected, t + 1):
                    item = NO_JOB
                self.deque_word(slot, prio, DEQUE_BOTTOM_OFF)[unsafe_offset=0].store[ordering=Ordering.RELAXED](b + 1)
            return item
        self.deque_word(slot, prio, DEQUE_BOTTOM_OFF)[unsafe_offset=0].store[ordering=Ordering.RELAXED](b + 1)
        return NO_JOB

    @always_inline
    def steal(self, slot: Int, prio: Int) -> Int:
        while True:
            var t = self.deque_word(slot, prio, DEQUE_TOP_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
            fence[Ordering.SEQUENTIAL]()
            var b = self.deque_word(slot, prio, DEQUE_BOTTOM_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
            if t >= b:
                return NO_JOB
            var item = Int(self.deque_word(slot, prio, DEQUE_BUF_OFF + (Int(t) & DEQUE_MASK))[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
            var expected = t
            if self.deque_word(slot, prio, DEQUE_TOP_OFF)[unsafe_offset=0].compare_exchange[
                failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
            ](expected, t + 1):
                return item

    # ---- 3-tier global bounded MPMC priority rings ----

    @always_inline
    def global_push_word(self, prio: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return self.word(GLOBAL_META_WORD + prio * GLOBAL_META_STRIDE)

    @always_inline
    def global_pop_word(self, prio: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return self.word(GLOBAL_META_WORD + prio * GLOBAL_META_STRIDE + CACHE_LINE_WORDS)

    @always_inline
    def global_cell_sequence(self, prio: Int, position: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        var base = GLOBAL_RINGS_WORD + prio * GLOBAL_RING_CELLS * GLOBAL_RING_CELL_WORDS
        return self.word(base + (position & GLOBAL_RING_MASK) * GLOBAL_RING_CELL_WORDS)

    @always_inline
    def global_cell_value(self, prio: Int, position: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        var base = GLOBAL_RINGS_WORD + prio * GLOBAL_RING_CELLS * GLOBAL_RING_CELL_WORDS
        return self.word(base + (position & GLOBAL_RING_MASK) * GLOBAL_RING_CELL_WORDS + 1)

    def push_global(self, prio: Int, handle: Int) -> Bool:
        var head = self.global_push_word(prio)
        var position = Int(head[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
        while True:
            var sequence = Int(self.global_cell_sequence(prio, position)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
            if sequence == position:
                var expected = Int64(position)
                if head[unsafe_offset=0].compare_exchange[
                    failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
                ](expected, Int64(position + 1)):
                    break
                position = Int(head[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
            elif sequence < position:
                return False
            else:
                position = Int(head[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
        self.global_cell_value(prio, position)[unsafe_offset=0].store[ordering=Ordering.RELAXED](Int64(handle))
        self.global_cell_sequence(prio, position)[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int64(position + 1))
        return True

    def pop_global(self, prio: Int) -> Int:
        var tail = self.global_pop_word(prio)
        var position = Int(tail[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
        while True:
            var sequence = Int(self.global_cell_sequence(prio, position)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
            if sequence == position + 1:
                var expected = Int64(position)
                if tail[unsafe_offset=0].compare_exchange[
                    failure_ordering=Ordering.RELAXED, success_ordering=Ordering.SEQUENTIAL
                ](expected, Int64(position + 1)):
                    break
                position = Int(tail[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
            elif sequence <= position:
                return NO_JOB
            else:
                position = Int(tail[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
        var handle = Int(self.global_cell_value(prio, position)[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
        self.global_cell_sequence(prio, position)[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int64(position + GLOBAL_RING_CELLS))
        return handle

    # ---- unified priority scheduling ----

    def push_ready(self, handle: Int, priority: Int = JOB_PRIO_NORMAL, wake: Int = 1) -> Bool:
        var prio = self._clamp_prio(priority)
        var owner = self.resolve_owner_slot()
        var pushed = False
        if owner >= 0:
            pushed = self.push_bottom(owner, prio, handle)
        if not pushed:
            pushed = self.push_global(prio, handle)
        if not pushed:
            return False
        _ = self.word(EPOCH_WORD)[unsafe_offset=0].fetch_add(1)
        if self.word(SLEEPERS_WORD)[unsafe_offset=0].load[ordering=Ordering.SEQUENTIAL]() > 0:
            self.futex_wake(wake)
        return True

    def pop_next_job(self) -> Int:
        var owner = self.resolve_owner_slot()
        var n_workers = self.worker_count()
        var total_slots = n_workers + 1
        for prio in range(JOB_PRIO_COUNT):
            if owner >= 0:
                var own = self.pop_bottom(owner, prio)
                if own != NO_JOB:
                    return own
            var g = self.pop_global(prio)
            if g != NO_JOB:
                return g
            if total_slots > 1:
                var start = (owner + 1) % total_slots if owner >= 0 else 0
                for off in range(total_slots):
                    var victim = (start + off) % total_slots
                    if victim == owner:
                        continue
                    var stolen = self.steal(victim, prio)
                    if stolen != NO_JOB:
                        return stolen
        return NO_JOB

    def has_any_work(self) -> Bool:
        var total_slots = self.worker_count() + 1
        for prio in range(JOB_PRIO_COUNT):
            if self.global_push_word(prio)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]() != self.global_pop_word(prio)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]():
                return True
            for slot in range(total_slots):
                var b = self.deque_word(slot, prio, DEQUE_BOTTOM_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
                var t = self.deque_word(slot, prio, DEQUE_TOP_OFF)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]()
                if b > t:
                    return True
        return False

    def sleep(self):
        """Block until a push. Sleepers rises with seq_cst before checking all queues."""
        _ = self.word(SLEEPERS_WORD)[unsafe_offset=0].fetch_add(1)
        fence[Ordering.SEQUENTIAL]()
        var epoch = Int(self.word(EPOCH_WORD)[unsafe_offset=0].load[ordering=Ordering.SEQUENTIAL]())
        var empty = not self.has_any_work()
        if empty and not self.stopping():
            _ = external_call["syscall", Int](SYS_FUTEX, self.address + EPOCH_WORD * 8, FUTEX_WAIT_PRIVATE, epoch & 0xFFFFFFFF, Int(0), Int(0), Int(0))
        _ = self.word(SLEEPERS_WORD)[unsafe_offset=0].fetch_sub(1)

    def futex_wake(self, count: Int):
        _ = external_call["syscall", Int](SYS_FUTEX, self.address + EPOCH_WORD * 8, FUTEX_WAKE_PRIVATE, count, Int(0), Int(0), Int(0))

    # ---- execution & completion ----

    def help(self) -> Bool:
        """Run one ready job if there is one."""
        var handle = self.pop_next_job()
        if handle == NO_JOB:
            return False
        _ = self.work_on(handle, True)
        return True

    def claim(self, job: Int) -> Int:
        """Take the job's next chunk, or -1 when none remain or the handle is stale."""
        var ticket_word = self.slot_word(job_slot(job), SLOT_TICKET)
        var generation = job_generation(job) & TICKET_GENERATIONS
        while True:
            var ticket = ticket_word[unsafe_offset=0].load()
            if (Int(ticket) >> 40) != generation:
                return -1
            var next = Int(ticket) & MAX_CHUNKS
            var total = (Int(ticket) >> 20) & MAX_CHUNKS
            if next >= total:
                return -1
            if ticket_word[unsafe_offset=0].compare_exchange(ticket, ticket + 1):
                return next

    def work_on(self, job: Int, spread: Bool) -> Bool:
        """Run the job's chunks until none are left to claim. False if there were none."""
        set_sim_float_mode()
        var chunk = self.claim(job)
        if chunk < 0:
            return False
        var slot = job_slot(job)
        var prio = Int(self.slot_word(slot, SLOT_PRIORITY)[unsafe_offset=0].load[ordering=Ordering.RELAXED]())
        if spread and chunk + 1 < Int(self.slot_word(slot, SLOT_CHUNKS)[unsafe_offset=0].load()):
            _ = self.push_ready(job, prio, 1)
        var body = Pointer[JobBody, origin=MutAnyOrigin](unsafe_from_address=self.address + (SLOTS_WORD + slot * SLOT_WORDS + SLOT_BODY) * 8)[unsafe_offset=0]
        var context = Int(self.slot_word(slot, SLOT_CONTEXT)[unsafe_offset=0].load())
        var chunks = Int(self.slot_word(slot, SLOT_CHUNKS)[unsafe_offset=0].load())
        var zone = Int(self.slot_word(slot, SLOT_ZONE)[unsafe_offset=0].load())
        var profiler = Profiler(Int(self.word(PROFILER_WORD)[unsafe_offset=0].load()) if zone != ZONE_NONE else 0)
        var started = profiler.now()
        var completed = 0
        while chunk >= 0:
            body(context, chunk)
            completed += 1
            chunk = self.claim(job)
        profiler.record(zone, started, profiler.now())
        if Int(self.slot_word(slot, SLOT_FINISHED)[unsafe_offset=0].fetch_add(Int64(completed))) + completed == chunks:
            self.complete(job)
        return True

    def release(self, job: Int):
        """Remove one reason the job could not run."""
        var slot = job_slot(job)
        if self.slot_word(slot, SLOT_PENDING)[unsafe_offset=0].fetch_sub(1) != 1:
            return
        var chunks = Int(self.slot_word(slot, SLOT_CHUNKS)[unsafe_offset=0].load())
        if chunks == 0:
            self.complete(job)
            return
        var generation = job_generation(job) & TICKET_GENERATIONS
        self.slot_word(slot, SLOT_TICKET)[unsafe_offset=0].store(Int64((generation << 40) | (chunks << 20)))
        var prio = Int(self.slot_word(slot, SLOT_PRIORITY)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
        if not self.push_ready(job, prio, chunks):
            _ = self.work_on(job, False)

    def complete(self, job: Int):
        """The job is over: publish that, then let its dependents go."""
        var slot = job_slot(job)
        self.lock(slot)
        self.slot_word(slot, SLOT_DONE)[unsafe_offset=0].store(Int64(job_generation(job)))
        var count = Int(self.slot_word(slot, SLOT_DEPENDENT_COUNT)[unsafe_offset=0].load())
        var ov_head = Int(self.slot_word(slot, SLOT_DEP_OVERFLOW_HEAD)[unsafe_offset=0].load())
        self.slot_word(slot, SLOT_DEP_OVERFLOW_HEAD)[unsafe_offset=0].store(0)
        self.slot_word(slot, SLOT_DEP_OVERFLOW_TAIL)[unsafe_offset=0].store(0)
        self.unlock(slot)
        var inline_count = min(count, MAX_DEPENDENTS)
        for index in range(inline_count):
            self.release(Int(self.slot_word(slot, SLOT_DEPENDENTS + index)[unsafe_offset=0].load()))
        var cur = ov_head
        while cur != 0:
            var blk = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=cur)
            var nxt = Int(blk[unsafe_offset=0])
            var b_cnt = Int(blk[unsafe_offset=1])
            for idx in range(b_cnt):
                self.release(Int(blk[unsafe_offset=2 + idx]))
            external_call["free", NoneType](cur)
            cur = nxt
        _ = self.word(COMPLETED_WORD)[unsafe_offset=0].fetch_add(1)
        self.slot_word(slot, SLOT_STATE)[unsafe_offset=0].store(Int64(STATE_FREE))

    def lock(self, slot: Int):
        """Held for a few instructions around one job's dependent list."""
        var word = self.slot_word(slot, SLOT_LOCK)
        while True:
            var free = Int64(0)
            if word[unsafe_offset=0].compare_exchange(free, 1):
                return
            _ = external_call["sched_yield", Int32]()

    def unlock(self, slot: Int):
        self.slot_word(slot, SLOT_LOCK)[unsafe_offset=0].store(0)


struct JobPool(Movable):
    """Owns shared memory and worker threads for lock-free work-stealing job graphs."""
    var address: Int
    var block: Int
    var workers: Int

    def __init__(out self):
        self.address = 0
        self.block = 0
        self.workers = 0

    def __init__(out self, *, deinit move: Self):
        self.address = move.address
        self.block = move.block
        self.workers = move.workers

    def __deinit__(deinit self):
        self.stop()

    @always_inline
    def _ref(self) -> _JobPoolRef:
        return _JobPoolRef(self.address, -1)

    def ready(self) -> Bool:
        """True when jobs can run in parallel, not merely on the calling thread."""
        return self.workers > 0

    def allocated(self) -> Bool:
        return self.address != 0

    def word(self, index: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=self.address + index * 8)

    def slot_word(self, slot: Int, index: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return self.word(SLOTS_WORD + slot * SLOT_WORDS + index)

    def allocate(mut self) -> Bool:
        """Make the shared memory. Safe to repeat; a pool with no workers still runs jobs."""
        if self.address != 0:
            return True
        var memory = external_call["calloc", Int](Int(1), POOL_WORDS * 8 + 64)
        if memory == 0:
            return False
        self.block = memory
        self.address = (memory + 63) & ~63
        var r = self._ref()
        for prio in range(JOB_PRIO_COUNT):
            for cell in range(GLOBAL_RING_CELLS):
                r.global_cell_sequence(prio, cell)[unsafe_offset=0].store[ordering=Ordering.RELAXED](Int64(cell))
        return True

    def start(mut self, workers: Int) -> Bool:
        """Start the worker threads once. Zero workers is a valid, serial pool."""
        if not self.allocate() or self.workers != 0:
            return False
        self.word(START_GATE_WORD)[unsafe_offset=0].store[ordering=Ordering.RELEASE](0)
        var attribute_words = List[UInt64](length=8, fill=0)  # pthread_attr_t: 56 bytes
        var attributes = Int(attribute_words.unsafe_ptr())
        _ = external_call["pthread_attr_init", Int32](attributes)
        _ = external_call["pthread_attr_setstacksize", Int32](attributes, UInt64(WORKER_STACK_BYTES))
        var target = min(workers, MAX_WORKERS)
        for idx in range(target):
            var boot_addr = self.address + (WORKER_BOOT_WORD + idx * WORKER_BOOT_WORDS) * 8
            var boot = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=boot_addr)
            boot[unsafe_offset=0] = Int64(self.address)
            boot[unsafe_offset=1] = Int64(idx + 1)
            var handle = UInt64(0)
            var code = external_call["pthread_create", Int32](Pointer(to=handle), attributes, job_worker, boot_addr)
            if code != 0:
                break
            self.word(THREADS_WORD + self.workers)[unsafe_offset=0].store[ordering=Ordering.RELAXED](Int64(handle))
            self.workers += 1
        _ = external_call["pthread_attr_destroy", Int32](attributes)
        _ = attribute_words^
        self.word(WORKERS_WORD)[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int64(self.workers))
        self.word(START_GATE_WORD)[unsafe_offset=0].store[ordering=Ordering.RELEASE](1)
        return True

    def stop(mut self):
        """Finish every thread and release the memory. Idempotent. All jobs must be finished."""
        if self.address == 0:
            return
        self.word(STOP_WORD)[unsafe_offset=0].store[ordering=Ordering.RELEASE](1)
        _ = self.word(EPOCH_WORD)[unsafe_offset=0].fetch_add(1)
        self._ref().futex_wake(FUTEX_WAKE_EVERYONE)
        for index in range(self.workers):
            var handle = UInt64(self.word(THREADS_WORD + index)[unsafe_offset=0].load())
            if handle != 0:
                _ = external_call["pthread_join", Int32](handle, Int(0))
        for slot in range(JOB_SLOTS):
            var ov = Int(self.slot_word(slot, SLOT_DEP_OVERFLOW_HEAD)[unsafe_offset=0].load())
            while ov != 0:
                var nxt = Int(Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=ov)[unsafe_offset=0])
                external_call["free", NoneType](ov)
                ov = nxt
            self.slot_word(slot, SLOT_DEP_OVERFLOW_HEAD)[unsafe_offset=0].store(0)
            self.slot_word(slot, SLOT_DEP_OVERFLOW_TAIL)[unsafe_offset=0].store(0)
        if self.block != 0:
            external_call["free", NoneType](self.block)
        self.address = 0
        self.block = 0
        self.workers = 0

    def stopping(self) -> Bool:
        return self._ref().stopping()

    def set_profiler(self, profiler: Profiler):
        """Every thread records the time it spends in a job created with a zone."""
        if self.address != 0:
            self.word(PROFILER_WORD)[unsafe_offset=0].store(Int64(profiler.address))

    def jobs_completed(self) -> Int:
        return Int(self.word(COMPLETED_WORD)[unsafe_offset=0].load())

    def jobs_helped(self) -> Int:
        """How many times a thread inside `wait` ran a job it was not waiting for."""
        return Int(self.word(HELPED_WORD)[unsafe_offset=0].load())

    # ---- building the graph ----

    def create(self, run: JobBody, context: Int, chunks: Int = 1, zone: Int = ZONE_NONE) -> Int:
        """Reserve a job. It cannot run until `submit`, so dependencies can be added."""
        var r = self._ref()
        var slot = 0
        var tries = 0
        while True:
            slot = Int(self.word(NEXT_SLOT_WORD)[unsafe_offset=0].fetch_add(1)) & (JOB_SLOTS - 1)
            var state = self.slot_word(slot, SLOT_STATE)[unsafe_offset=0].load()
            if state == Int64(STATE_FREE) and self.slot_word(slot, SLOT_STATE)[unsafe_offset=0].compare_exchange(state, Int64(STATE_BUILDING)):
                break
            tries += 1
            if tries % JOB_SLOTS == 0 and not r.help():
                _ = external_call["sched_yield", Int32]()
        var generation = Int(self.slot_word(slot, SLOT_GENERATION)[unsafe_offset=0].load()) + 1
        self.slot_word(slot, SLOT_GENERATION)[unsafe_offset=0].store(Int64(generation))
        var body = Pointer[JobBody, origin=MutAnyOrigin](unsafe_from_address=self.address + (SLOTS_WORD + slot * SLOT_WORDS + SLOT_BODY) * 8)
        body[unsafe_offset=0] = run
        self.slot_word(slot, SLOT_CONTEXT)[unsafe_offset=0].store(Int64(context))
        self.slot_word(slot, SLOT_CHUNKS)[unsafe_offset=0].store(Int64(max(0, min(chunks, MAX_CHUNKS))))
        self.slot_word(slot, SLOT_ZONE)[unsafe_offset=0].store(Int64(zone))
        self.slot_word(slot, SLOT_PRIORITY)[unsafe_offset=0].store(Int64(JOB_PRIO_NORMAL))
        self.slot_word(slot, SLOT_FINISHED)[unsafe_offset=0].store(0)
        self.slot_word(slot, SLOT_DEPENDENT_COUNT)[unsafe_offset=0].store(0)
        self.slot_word(slot, SLOT_DEP_OVERFLOW_HEAD)[unsafe_offset=0].store(0)
        self.slot_word(slot, SLOT_DEP_OVERFLOW_TAIL)[unsafe_offset=0].store(0)
        self.slot_word(slot, SLOT_PENDING)[unsafe_offset=0].store(1)  # held until submit
        self.slot_word(slot, SLOT_STATE)[unsafe_offset=0].store(Int64(STATE_LIVE))
        return (generation << SLOT_BITS) | slot

    def depend(self, job: Int, prerequisite: Int):
        """Make `job`, not yet submitted, wait for `prerequisite`, in any state."""
        if prerequisite == NO_JOB:
            return
        var r = self._ref()
        var slot = job_slot(prerequisite)
        r.lock(slot)
        if Int(self.slot_word(slot, SLOT_DONE)[unsafe_offset=0].load()) >= job_generation(prerequisite):
            r.unlock(slot)
            return
        _ = self.slot_word(job_slot(job), SLOT_PENDING)[unsafe_offset=0].fetch_add(1)
        var count = Int(self.slot_word(slot, SLOT_DEPENDENT_COUNT)[unsafe_offset=0].load())
        if count < MAX_DEPENDENTS:
            self.slot_word(slot, SLOT_DEPENDENTS + count)[unsafe_offset=0].store(Int64(job))
        else:
            var tail = Int(self.slot_word(slot, SLOT_DEP_OVERFLOW_TAIL)[unsafe_offset=0].load())
            var need_new = tail == 0
            if not need_new:
                var tblk = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=tail)
                if Int(tblk[unsafe_offset=1]) >= _DEP_OVERFLOW_CAP:
                    need_new = True
            if need_new:
                var new_blk = external_call["malloc", Int](_DEP_OVERFLOW_WORDS * 8)
                if new_blk == 0:
                    r.unlock(slot)
                    print("JOB_GRAPH_ERROR out of memory allocating dependent overflow block")
                    external_call["abort", NoneType]()
                var nblk = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=new_blk)
                nblk[unsafe_offset=0] = 0
                nblk[unsafe_offset=1] = 0
                if tail == 0:
                    self.slot_word(slot, SLOT_DEP_OVERFLOW_HEAD)[unsafe_offset=0].store(Int64(new_blk))
                else:
                    Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=tail)[unsafe_offset=0] = Int64(new_blk)
                self.slot_word(slot, SLOT_DEP_OVERFLOW_TAIL)[unsafe_offset=0].store(Int64(new_blk))
                tail = new_blk
            var wblk = Pointer[Int64, origin=MutAnyOrigin](unsafe_from_address=tail)
            var b_cnt = Int(wblk[unsafe_offset=1])
            wblk[unsafe_offset=2 + b_cnt] = Int64(job)
            wblk[unsafe_offset=1] = Int64(b_cnt + 1)
        self.slot_word(slot, SLOT_DEPENDENT_COUNT)[unsafe_offset=0].store(Int64(count + 1))
        r.unlock(slot)

    def submit(self, job: Int, priority: Int = JOB_PRIO_NORMAL):
        """Let the job go: it runs once everything it depends on has finished."""
        var r = self._ref()
        var prio = r._clamp_prio(priority)
        self.slot_word(job_slot(job), SLOT_PRIORITY)[unsafe_offset=0].store[ordering=Ordering.RELEASE](Int64(prio))
        r.release(job)

    def run(self, body: JobBody, context: Int, chunks: Int = 1, zone: Int = ZONE_NONE):
        """Run one fan-out to the end, on this thread and any free worker."""
        var job = self.create(body, context, chunks, zone)
        self.submit(job)
        self.wait(job)

    def rouse(self):
        """Wake every sleeping worker ahead of known work."""
        if self.allocated() and self.word(SLEEPERS_WORD)[unsafe_offset=0].load[ordering=Ordering.SEQUENTIAL]() > 0:
            _ = self.word(EPOCH_WORD)[unsafe_offset=0].fetch_add(1)
            self._ref().futex_wake(FUTEX_WAKE_EVERYONE)

    def chain(self, body: JobBody, context: Int, count: Int, zone: Int = ZONE_NONE) -> Int:
        """Submit `count` single-chunk jobs that run strictly one after another."""
        var last = NO_JOB
        for _ in range(count):
            var job = self.create(body, context, 1, zone)
            self.depend(job, last)
            self.submit(job)
            last = job
        return last

    def wait(self, job: Int):
        """Return when the job is finished, running ready jobs in the meantime."""
        set_sim_float_mode()
        if job == NO_JOB:
            return
        var r = self._ref()
        while not self.finished(job):
            if r.work_on(job, False):
                continue
            var handle = r.pop_next_job()
            if handle != NO_JOB:
                if r.work_on(handle, True) and handle != job:
                    _ = self.word(HELPED_WORD)[unsafe_offset=0].fetch_add(1)
            else:
                _ = external_call["sched_yield", Int32]()

    def finished(self, job: Int) -> Bool:
        if job == NO_JOB:
            return True
        return Int(self.slot_word(job_slot(job), SLOT_DONE)[unsafe_offset=0].load()) >= job_generation(job)

    def help(self) -> Bool:
        """Run one ready job if there is one."""
        return self._ref().help()

    def claim(self, job: Int) -> Int:
        return self._ref().claim(job)

    def work_on(self, job: Int, spread: Bool) -> Bool:
        return self._ref().work_on(job, spread)

    def release(self, job: Int):
        self._ref().release(job)

    def complete(self, job: Int):
        self._ref().complete(job)

    def lock(self, slot: Int):
        self._ref().lock(slot)

    def unlock(self, slot: Int):
        self._ref().unlock(slot)

    def sleep(self):
        self._ref().sleep()

    def futex_wake(self, count: Int):
        self._ref().futex_wake(count)

    def current_thread_index(self) -> Int:
        """Returns 0 on the calling thread, or 1..workers for pool worker threads."""
        if self.address == 0:
            return 0
        var n_workers = Int(self.word(WORKERS_WORD)[unsafe_offset=0].load[ordering=Ordering.ACQUIRE]())
        if n_workers <= 0:
            return 0
        var self_tid = external_call["pthread_self", UInt64]()
        for index in range(n_workers):
            var handle = UInt64(self.word(THREADS_WORD + index)[unsafe_offset=0].load())
            if handle == self_tid:
                return index + 1
        return 0

    def run_affine(self, body: JobBody, context: Int, chunks: Int = 1, zone: Int = ZONE_NONE):
        """Affine fan-out: assigns each worker a deterministic home chunk slice first
        (keeping SoA lines L2-cache-hot across ticks), with lock-free work-stealing tail fallback."""
        if chunks <= 0:
            return
        if not self.allocated() or self.workers <= 0 or chunks == 1:
            set_sim_float_mode()
            for chunk in range(chunks):
                body(context, chunk)
            return
        var t_count = min(chunks, min(MAX_WORKERS + 1, 1 + self.workers))
        var disp = _AffineDispatch(Int(Pointer(to=self)), body, context, t_count, chunks)
        self.rouse()
        self.run(_run_affine_task, Int(Pointer(to=disp)), t_count, zone)
        _ = disp.threads


comptime _AFFINE_SLOTS: Int = MAX_WORKERS + 1
comptime _AFFINE_WORDS: Int = _AFFINE_SLOTS * CACHE_LINE_WORDS


struct _AffineDispatch:
    var pool_addr: Int
    var user_body: JobBody
    var user_ctx: Int
    var threads: Int
    var words: Array[Int64, _AFFINE_WORDS]

    def __init__(out self, pool_addr: Int, user_body: JobBody, user_ctx: Int, threads: Int, chunks: Int):
        self.pool_addr = pool_addr
        self.user_body = user_body
        self.user_ctx = user_ctx
        self.threads = threads
        self.words = Array[Int64, _AFFINE_WORDS](fill=0)
        for t in range(threads):
            var start = (t * chunks) // threads
            var end = ((t + 1) * chunks) // threads
            self.words[t * CACHE_LINE_WORDS] = Int64(start)
            self.words[t * CACHE_LINE_WORDS + 1] = Int64(end)

    @always_inline
    def cursor_ptr(self, slot: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](
            unsafe_from_address=Int(Pointer(to=self.words)) + slot * CACHE_LINE_WORDS * 8
        )

    @always_inline
    def end_of(self, slot: Int) -> Int:
        return Int(self.words[slot * CACHE_LINE_WORDS + 1])


def _run_affine_task(context: Int, chunk: Int):
    ref disp = Pointer[_AffineDispatch, origin=MutAnyOrigin](unsafe_from_address=context)[unsafe_offset=0]
    ref pool = Pointer[JobPool, origin=MutAnyOrigin](unsafe_from_address=disp.pool_addr)[unsafe_offset=0]
    var t_count = disp.threads
    var home = pool.current_thread_index()
    if home < 0 or home >= t_count:
        home = chunk % t_count
    var body = disp.user_body
    var uctx = disp.user_ctx

    var own_cur = disp.cursor_ptr(home)
    var own_end = disp.end_of(home)
    while True:
        var c = Int(own_cur[unsafe_offset=0].fetch_add(1))
        if c >= own_end:
            break
        body(uctx, c)

    for offset in range(1, t_count):
        var victim = (home + offset) % t_count
        var v_cur = disp.cursor_ptr(victim)
        var v_end = disp.end_of(victim)
        while True:
            var c = Int(v_cur[unsafe_offset=0].fetch_add(1))
            if c >= v_end:
                break
            body(uctx, c)


def default_worker_count() -> Int:
    """One worker per physical core, leaving one for the main thread."""
    var logical = Int(external_call["sysconf", Int64](Int32(84)))  # _SC_NPROCESSORS_ONLN
    return max(0, min(31, logical // 2 - 1))


def run_jobs(pool_address: Int, body: JobBody, context: Int, chunks: Int, zone: Int = ZONE_NONE):
    """`JobPool.run` for code that holds a pool by address (the simulation's
    physics gets the runtime's pool this way, package 2.1). Address 0, or a
    pool never allocated (tests without a runtime), runs every chunk here in
    order. The chunks must not depend on which thread runs them or in which
    order, so both ways give the same result."""
    if chunks <= 0:
        return
    if pool_address != 0:
        ref pool = Pointer[JobPool, origin=MutAnyOrigin](unsafe_from_address=pool_address)[unsafe_offset=0]
        if pool.allocated():
            pool.run(body, context, chunks, zone)
            return
    set_sim_float_mode()
    for chunk in range(chunks):
        body(context, chunk)


def run_jobs_affine(pool_address: Int, body: JobBody, context: Int, chunks: Int, zone: Int = ZONE_NONE):
    """Home-chunk L2-cache-affine dispatch per worker with work-stealing tail fallback."""
    if chunks <= 0:
        return
    if pool_address != 0:
        ref pool = Pointer[JobPool, origin=MutAnyOrigin](unsafe_from_address=pool_address)[unsafe_offset=0]
        if pool.allocated():
            pool.run_affine(body, context, chunks, zone)
            return
    set_sim_float_mode()
    for chunk in range(chunks):
        body(context, chunk)

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
from std.atomic import Atomic
from std.ffi import external_call
from std.sys.info import CompilationTarget
from flint.core.crash import install_thread_stack
from flint.core.detmath import set_sim_float_mode
from flint.core.profiler import Profiler, ZONE_NONE

comptime JobBody = def(Int, Int) thin -> None
comptime NO_JOB: Int = -1

comptime CACHE_LINE_WORDS: Int = 8
comptime PUSH_WORD: Int = 0
comptime POP_WORD: Int = CACHE_LINE_WORDS
comptime EPOCH_WORD: Int = CACHE_LINE_WORDS * 2
comptime SLEEPERS_WORD: Int = CACHE_LINE_WORDS * 3
comptime NEXT_SLOT_WORD: Int = CACHE_LINE_WORDS * 4
comptime STOP_WORD: Int = CACHE_LINE_WORDS * 5
comptime PROFILER_WORD: Int = STOP_WORD + 1  # shares a line with a word that never changes in play
comptime COMPLETED_WORD: Int = CACHE_LINE_WORDS * 6
comptime HELPED_WORD: Int = CACHE_LINE_WORDS * 7
comptime THREADS_WORD: Int = CACHE_LINE_WORDS * 8
comptime MAX_WORKERS: Int = 32
comptime RING_WORD: Int = THREADS_WORD + MAX_WORKERS
comptime RING_CELLS: Int = 4096
comptime RING_CELL_WORDS: Int = 2
comptime SLOTS_WORD: Int = RING_WORD + RING_CELLS * RING_CELL_WORDS

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
# Second line: the words threads fight over.
comptime SLOT_TICKET: Int = CACHE_LINE_WORDS
comptime SLOT_FINISHED: Int = CACHE_LINE_WORDS + 1
comptime SLOT_PENDING: Int = CACHE_LINE_WORDS + 2
comptime SLOT_LOCK: Int = CACHE_LINE_WORDS + 3
comptime SLOT_DONE: Int = CACHE_LINE_WORDS + 4
comptime SLOT_DEPENDENT_COUNT: Int = CACHE_LINE_WORDS + 5
comptime SLOT_DEPENDENTS: Int = CACHE_LINE_WORDS * 2
comptime MAX_DEPENDENTS: Int = CACHE_LINE_WORDS * 2

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


def job_worker(argument: Int) abi("C") -> Int:
    """Thread body: run ready jobs until the pool stops, sleeping when there are none."""
    # Its own signal stack, so a crash in a job is reported even when it is a stack overflow.
    _ = install_thread_stack()
    # One floating-point mode for every thread that may run a tick
    # (docs/DETERMINISM.md): round to nearest, FTZ and DAZ off.
    set_sim_float_mode()
    var pool = JobPool()
    pool.address = argument
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


struct JobPool(Copyable, Movable):
    """Points at shared memory that every worker thread reads."""
    var address: Int
    var block: Int
    var workers: Int

    def __init__(out self):
        self.address = 0
        self.block = 0
        self.workers = 0

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
        for cell in range(RING_CELLS):
            self.word(RING_WORD + cell * RING_CELL_WORDS)[unsafe_offset=0].store(Int64(cell))
        return True

    def start(mut self, workers: Int) -> Bool:
        """Start the worker threads once. Zero workers is a valid, serial pool."""
        if not self.allocate() or self.workers != 0:
            return False
        var attribute_words = List[UInt64](length=8, fill=0)  # pthread_attr_t: 56 bytes
        var attributes = Int(attribute_words.unsafe_ptr())
        _ = external_call["pthread_attr_init", Int32](attributes)
        _ = external_call["pthread_attr_setstacksize", Int32](attributes, UInt64(WORKER_STACK_BYTES))
        for _ in range(min(workers, MAX_WORKERS)):
            var handle = UInt64(0)
            var code = external_call["pthread_create", Int32](Pointer(to=handle), attributes, job_worker, self.address)
            if code != 0:
                break
            self.word(THREADS_WORD + self.workers)[unsafe_offset=0].store(Int64(handle))
            self.workers += 1
        _ = external_call["pthread_attr_destroy", Int32](attributes)
        _ = attribute_words^
        return True

    def stop(mut self):
        """Finish every thread and release the memory. All jobs must be finished."""
        if self.address == 0:
            return
        self.word(STOP_WORD)[unsafe_offset=0].store(1)
        _ = self.word(EPOCH_WORD)[unsafe_offset=0].fetch_add(1)
        self.futex_wake(FUTEX_WAKE_EVERYONE)
        for index in range(self.workers):
            var handle = UInt64(self.word(THREADS_WORD + index)[unsafe_offset=0].load())
            _ = external_call["pthread_join", Int32](handle, Int(0))
        external_call["free", NoneType](self.block)
        self.address = 0
        self.block = 0
        self.workers = 0

    def stopping(self) -> Bool:
        return self.word(STOP_WORD)[unsafe_offset=0].load() != 0

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
        """Reserve a job. It cannot run until `submit`, so dependencies can be added.

        With a profiler zone, each thread's stretch of work on it is one event."""
        var slot = 0
        var tries = 0
        while True:
            slot = Int(self.word(NEXT_SLOT_WORD)[unsafe_offset=0].fetch_add(1)) & (JOB_SLOTS - 1)
            var state = self.slot_word(slot, SLOT_STATE)[unsafe_offset=0].load()
            if state == Int64(STATE_FREE) and self.slot_word(slot, SLOT_STATE)[unsafe_offset=0].compare_exchange(state, Int64(STATE_BUILDING)):
                break
            tries += 1
            if tries % JOB_SLOTS == 0 and not self.help():
                # Every slot is live: running a job here is what frees one.
                _ = external_call["sched_yield", Int32]()
        var generation = Int(self.slot_word(slot, SLOT_GENERATION)[unsafe_offset=0].load()) + 1
        self.slot_word(slot, SLOT_GENERATION)[unsafe_offset=0].store(Int64(generation))
        var body = Pointer[JobBody, origin=MutAnyOrigin](unsafe_from_address=self.address + (SLOTS_WORD + slot * SLOT_WORDS + SLOT_BODY) * 8)
        body[unsafe_offset=0] = run
        self.slot_word(slot, SLOT_CONTEXT)[unsafe_offset=0].store(Int64(context))
        self.slot_word(slot, SLOT_CHUNKS)[unsafe_offset=0].store(Int64(max(0, min(chunks, MAX_CHUNKS))))
        self.slot_word(slot, SLOT_ZONE)[unsafe_offset=0].store(Int64(zone))
        self.slot_word(slot, SLOT_FINISHED)[unsafe_offset=0].store(0)
        self.slot_word(slot, SLOT_DEPENDENT_COUNT)[unsafe_offset=0].store(0)
        self.slot_word(slot, SLOT_PENDING)[unsafe_offset=0].store(1)  # held until submit
        self.slot_word(slot, SLOT_STATE)[unsafe_offset=0].store(Int64(STATE_LIVE))
        return (generation << SLOT_BITS) | slot

    def depend(self, job: Int, prerequisite: Int):
        """Make `job`, not yet submitted, wait for `prerequisite`, in any state.

        A prerequisite that already finished is simply satisfied, even if its
        slot has since been reused, because the done word only rises.
        """
        if prerequisite == NO_JOB:
            return
        var slot = job_slot(prerequisite)
        self.lock(slot)
        if Int(self.slot_word(slot, SLOT_DONE)[unsafe_offset=0].load()) >= job_generation(prerequisite):
            self.unlock(slot)
            return
        var count = Int(self.slot_word(slot, SLOT_DEPENDENT_COUNT)[unsafe_offset=0].load())
        if count >= MAX_DEPENDENTS:
            self.unlock(slot)
            print("JOB_GRAPH_ERROR too many dependents on one job; add a job between them")
            external_call["abort", NoneType]()
        _ = self.slot_word(job_slot(job), SLOT_PENDING)[unsafe_offset=0].fetch_add(1)
        self.slot_word(slot, SLOT_DEPENDENTS + count)[unsafe_offset=0].store(Int64(job))
        self.slot_word(slot, SLOT_DEPENDENT_COUNT)[unsafe_offset=0].store(Int64(count + 1))
        self.unlock(slot)

    def submit(self, job: Int):
        """Let the job go: it runs once everything it depends on has finished."""
        self.release(job)

    def run(self, body: JobBody, context: Int, chunks: Int = 1, zone: Int = ZONE_NONE):
        """Run one fan-out to the end, on this thread and any free worker."""
        var job = self.create(body, context, chunks, zone)
        self.submit(job)
        self.wait(job)

    def rouse(self):
        """Wake every sleeping worker ahead of known work. They spin briefly, so
        a fan-out that follows soon starts on cores that are already awake."""
        if self.allocated() and self.word(SLEEPERS_WORD)[unsafe_offset=0].load() > 0:
            _ = self.word(EPOCH_WORD)[unsafe_offset=0].fetch_add(1)
            self.futex_wake(FUTEX_WAKE_EVERYONE)

    def chain(self, body: JobBody, context: Int, count: Int, zone: Int = ZONE_NONE) -> Int:
        """Submit `count` single-chunk jobs that run strictly one after another,
        on any thread. Returns the last one, or NO_JOB for a count of zero."""
        var last = NO_JOB
        for _ in range(count):
            var job = self.create(body, context, 1, zone)
            self.depend(job, last)
            self.submit(job)
            last = job
        return last

    def wait(self, job: Int):
        """Return when the job is finished, running ready jobs in the meantime.

        The job's own chunks come first, so a frame is not held up by unrelated
        work while what it is waiting for could be running here.
        """
        if job == NO_JOB:
            return
        while not self.finished(job):
            if self.work_on(job, False):
                continue
            var handle = self.pop()
            if handle != NO_JOB:
                if self.work_on(handle, True) and handle != job:
                    _ = self.word(HELPED_WORD)[unsafe_offset=0].fetch_add(1)
            else:
                _ = external_call["sched_yield", Int32]()

    def finished(self, job: Int) -> Bool:
        if job == NO_JOB:
            return True
        return Int(self.slot_word(job_slot(job), SLOT_DONE)[unsafe_offset=0].load()) >= job_generation(job)

    def help(self) -> Bool:
        """Run one ready job if there is one."""
        var handle = self.pop()
        if handle == NO_JOB:
            return False
        _ = self.work_on(handle, True)
        return True

    # ---- running ----

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
        """Run the job's chunks until none are left to claim. False if there were none.

        A thread that took a fan-out from the ring puts it back for the next
        thread, so the job holds one ring cell however many threads work on it.
        Losing that push to a full ring costs parallelism, never work.
        """
        var chunk = self.claim(job)
        if chunk < 0:
            return False
        var slot = job_slot(job)
        if spread and chunk + 1 < Int(self.slot_word(slot, SLOT_CHUNKS)[unsafe_offset=0].load()):
            _ = self.push(job)
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
        # Before the add below: once a job is finished, its events must be published.
        profiler.record(zone, started, profiler.now())
        # One add per batch: the last add sees every chunk accounted for.
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
            self.complete(job)  # a job with no chunks only joins others
            return
        var generation = job_generation(job) & TICKET_GENERATIONS
        self.slot_word(slot, SLOT_TICKET)[unsafe_offset=0].store(Int64((generation << 40) | (chunks << 20)))
        # Wake enough threads at once: waking them one from another adds a wake
        # latency per thread to the first fan-out of every frame.
        if not self.push(job, chunks):
            _ = self.work_on(job, False)  # ring full: this thread runs it, nothing is lost

    def complete(self, job: Int):
        """The job is over: publish that, then let its dependents go."""
        var slot = job_slot(job)
        self.lock(slot)
        self.slot_word(slot, SLOT_DONE)[unsafe_offset=0].store(Int64(job_generation(job)))
        var count = Int(self.slot_word(slot, SLOT_DEPENDENT_COUNT)[unsafe_offset=0].load())
        self.unlock(slot)
        # Nothing can be added now and the slot stays ours until it is freed below.
        for index in range(count):
            self.release(Int(self.slot_word(slot, SLOT_DEPENDENTS + index)[unsafe_offset=0].load()))
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

    # ---- the ready ring: Vyukov's bounded multi-producer, multi-consumer queue ----

    def cell_sequence(self, position: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return self.word(RING_WORD + (position & (RING_CELLS - 1)) * RING_CELL_WORDS)

    def cell_value(self, position: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return self.word(RING_WORD + (position & (RING_CELLS - 1)) * RING_CELL_WORDS + 1)

    def push(self, handle: Int, wake: Int = 1) -> Bool:
        var head = self.word(PUSH_WORD)
        var position = Int(head[unsafe_offset=0].load())
        while True:
            var sequence = Int(self.cell_sequence(position)[unsafe_offset=0].load())
            if sequence == position:
                var expected = Int64(position)
                if head[unsafe_offset=0].compare_exchange(expected, Int64(position + 1)):
                    break
                position = Int(head[unsafe_offset=0].load())
            elif sequence < position:
                return False
            else:
                position = Int(head[unsafe_offset=0].load())
        self.cell_value(position)[unsafe_offset=0].store(Int64(handle))
        self.cell_sequence(position)[unsafe_offset=0].store(Int64(position + 1))
        _ = self.word(EPOCH_WORD)[unsafe_offset=0].fetch_add(1)
        if self.word(SLEEPERS_WORD)[unsafe_offset=0].load() > 0:
            self.futex_wake(wake)
        return True

    def pop(self) -> Int:
        var tail = self.word(POP_WORD)
        var position = Int(tail[unsafe_offset=0].load())
        while True:
            var sequence = Int(self.cell_sequence(position)[unsafe_offset=0].load())
            if sequence == position + 1:
                var expected = Int64(position)
                if tail[unsafe_offset=0].compare_exchange(expected, Int64(position + 1)):
                    break
                position = Int(tail[unsafe_offset=0].load())
            elif sequence <= position:
                return NO_JOB
            else:
                position = Int(tail[unsafe_offset=0].load())
        var handle = Int(self.cell_value(position)[unsafe_offset=0].load())
        self.cell_sequence(position)[unsafe_offset=0].store(Int64(position + RING_CELLS))
        return handle

    # ---- sleeping ----

    def sleep(self):
        """Block until a push. The order matters: sleepers rises before the
        ring is checked, and a push bumps the epoch before reading sleepers, so
        either this thread sees the job or the push sees this thread."""
        _ = self.word(SLEEPERS_WORD)[unsafe_offset=0].fetch_add(1)
        var epoch = Int(self.word(EPOCH_WORD)[unsafe_offset=0].load())
        var empty = self.word(PUSH_WORD)[unsafe_offset=0].load() == self.word(POP_WORD)[unsafe_offset=0].load()
        if empty and not self.stopping():
            _ = external_call["syscall", Int](SYS_FUTEX, self.address + EPOCH_WORD * 8, FUTEX_WAIT_PRIVATE, epoch & 0xFFFFFFFF, Int(0), Int(0), Int(0))
        _ = self.word(SLEEPERS_WORD)[unsafe_offset=0].fetch_sub(1)

    def futex_wake(self, count: Int):
        _ = external_call["syscall", Int](SYS_FUTEX, self.address + EPOCH_WORD * 8, FUTEX_WAKE_PRIVATE, count, Int(0), Int(0), Int(0))

    def current_thread_index(self) -> Int:
        """Returns 0 on the calling thread, or 1..workers for pool worker threads."""
        if self.address == 0 or self.workers <= 0:
            return 0
        var self_tid = external_call["pthread_self", UInt64]()
        for index in range(self.workers):
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
    for chunk in range(chunks):
        body(context, chunk)

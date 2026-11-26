"""Load-aware timing helpers for gate tests (sota:perf r3).

The gate runs while other builds and GPU jobs share the host, so a single
wall-clock sample says more about the neighbours than about the code. A timing
check built from these helpers keeps its meaning and stops failing on noise:

- Interleave the variants (A, B, A, B, ...) so load drift hits both alike.
- Compare the best (minimum) sample of each variant: the least-disturbed run is
  the closest estimate of the code's own cost, and noise only ever adds time.
- Measure in rounds and stop at the first passing round (`timing_rounds()`,
  default 3, `FLINT_TIMING_ROUNDS` overrides). A real regression fails every
  round; a burst of load rarely lasts through all of them.
- Scale only the margin that load legitimately changes (a parallel speed-up
  shrinks when the other cores are busy) with `load_aware_ratio`, which never
  drops the check: it interpolates between the idle and the loaded bound.
- For single-threaded budgets, `thread_cpu_s()` (CLOCK_THREAD_CPUTIME_ID) does
  not count time the thread spent descheduled.
"""
from std.collections import List
from std.ffi import external_call
from std.math import max, min
from flint.platform.os import env_value, read_stream


def host_cores() -> Int:
    """Online logical CPUs (sysconf _SC_NPROCESSORS_ONLN)."""
    return max(1, Int(external_call["sysconf", Int64](Int32(84))))


def _parse_num(b: List[UInt8], mut i: Int) -> Float64:
    """Parses an unsigned decimal at `i` and leaves `i` on the first byte after it."""
    var v = Float64(0.0)
    var scale = Float64(0.0)
    while i < len(b):
        var c = Int(b[i])
        if c == 46 and scale == 0.0:
            scale = 1.0
        elif c >= 48 and c <= 57:
            if scale > 0.0:
                scale *= 0.1
                v += Float64(c - 48) * scale
            else:
                v = v * 10.0 + Float64(c - 48)
        else:
            break
        i += 1
    return v


def host_load() -> Float64:
    """Competing runnable threads: the larger of the 1-minute load average and
    the instantaneous runnable count from /proc/loadavg (minus this thread).
    The average lags a burst that just started; the count catches it."""
    var b = read_stream("/proc/loadavg", 256)
    var i = 0
    var avg1 = _parse_num(b, i)
    # skip to the fourth field "running/total"
    var field = 0
    while i < len(b) and field < 3:
        if Int(b[i]) == 32:
            field += 1
        i += 1
    var running = _parse_num(b, i)
    return max(avg1, running - 1.0)


def host_busy() -> Float64:
    """Load per core: 0 idle, 1 every core has a competing runnable thread."""
    return host_load() / Float64(host_cores())


def load_aware_ratio(idle_ratio: Float64, loaded_ratio: Float64) -> Float64:
    """The bound a ratio check needs at the current host load: `idle_ratio`
    up to 0.5 load per core, `loaded_ratio` from 1.5, linear in between."""
    var busy = host_busy()
    var t = min(1.0, max(0.0, busy - 0.5))
    return idle_ratio + (loaded_ratio - idle_ratio) * t


def timing_rounds(default: Int = 3) -> Int:
    """Measurement rounds a timing check may take before it fails."""
    var s = env_value("FLINT_TIMING_ROUNDS")
    if s.byte_length() == 0:
        return default
    var b = List[UInt8]()
    for k in range(s.byte_length()):
        b.append(s.as_bytes()[k])
    var i = 0
    return max(1, Int(_parse_num(b, i)))


def min_of(xs: List[Float64]) -> Float64:
    var m = Float64(1.0e300)
    for i in range(len(xs)):
        m = min(m, xs[i])
    return m


def quantile(xs: List[Float64], q: Float64) -> Float64:
    """Nearest-rank quantile (q in [0, 1]); 0 for an empty list."""
    if len(xs) == 0:
        return 0.0
    var s = xs.copy()
    for i in range(1, len(s)):
        var v = s[i]
        var j = i - 1
        while j >= 0 and s[j] > v:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v
    var k = Int(q * Float64(len(s) - 1) + 0.5)
    return s[max(0, min(len(s) - 1, k))]


def thread_cpu_s() -> Float64:
    """CPU seconds this thread has run (CLOCK_THREAD_CPUTIME_ID): unlike the
    wall clock it does not grow while the thread waits for a core."""
    var ts = List[Int64](length=2, fill=0)
    _ = external_call["clock_gettime", Int32](Int32(3), ts.unsafe_ptr())
    var s = Float64(ts[0]) + Float64(ts[1]) * 1.0e-9
    _ = ts^
    return s


def timing_note(label: String, ok: Bool, round: Int, rounds: Int, detail: String):
    """One line per measurement round, so a gate log shows how a check passed."""
    var tag = String("ok") if ok else (String("retry") if round + 1 < rounds else String("FAIL"))
    print("    [timing] " + label + " round " + String(round + 1) + "/" + String(rounds) + " (host load "
          + String(Float64(Int(host_load() * 10.0)) / 10.0) + " on " + String(host_cores()) + " cores): " + detail + " -> " + tag)

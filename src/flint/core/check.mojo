"""check, ensure and dev_assert: three ways to state what must be true.

    check(count <= MAX_BODIES, LOG_PHYSICS, "body pool overflow")
        Always on. A failure is a bug the game cannot continue past: it logs a
        fatal entry and stops through the crash handler, which writes a report
        (with the replay up to this tick) and the reason.
    if not ensure(model.valid, LOG_ASSETS, "model failed to load"):
        return
        Always on. A failure is logged as an error and counted; the caller
        handles it and the game goes on. Use it where a fallback exists.
    dev_assert(is_finite(position), LOG_SIM, "player position is not finite")
        A check that costs time on hot paths. On in every build of this repo;
        a build made with `-D SD_RELEASE` drops it at compile time.

All three only read a Bool someone already computed and write to the log:
none can change the simulation, so a replay is the same with or without
them (tests/test_safety.mojo replays with failures forced to show it).
Messages are static strings, so a passing check builds no text.
"""
from std.sys import is_defined
from std.ffi import external_call
from flint.core.anchor import (
    ANCHOR, A_CHECK_FAILURES, A_ENSURE_FAILURES, A_HANDLER_INSTALLED, SYS_GETPID, SYS_GETTID, SYS_TGKILL,
    T_REASON, anchor_add, anchor_get, anchor_ready, load, set_text, sys,
)
from flint.core.log import LOG_ERROR, LOG_FATAL, category_name, flush_log, log_message

comptime SIGABRT: Int = 6
comptime DEV_ASSERTS: Bool = not is_defined["SD_RELEASE"]()


def fail_hard(category: Int, kind: StaticString, message: StaticString):
    """Log, record the reason for the crash report, and stop through the handler."""
    var reason = String(kind) + " failed [" + category_name(category) + "]: " + String(message)
    log_message(category, LOG_FATAL, reason)
    if anchor_ready():
        _ = anchor_add(A_CHECK_FAILURES)
        set_text(T_REASON, reason)
        if load(ANCHOR + A_HANDLER_INSTALLED) != 0:
            _ = sys(SYS_TGKILL, sys(SYS_GETPID), sys(SYS_GETTID), SIGABRT)
        else:
            _ = flush_log()
    print("SHARDDEPTH_ABORT " + reason)
    external_call["abort", NoneType]()


@always_inline
def check(condition: Bool, category: Int, message: StaticString):
    """Must hold in every build; a failure stops the game with a crash report."""
    if not condition:
        fail_hard(category, "check", message)


def ensure_failed(category: Int, message: StaticString):
    if anchor_ready():
        _ = anchor_add(A_ENSURE_FAILURES)
    log_message(category, LOG_ERROR, "ensure failed: " + String(message))


@always_inline
def ensure(condition: Bool, category: Int, message: StaticString) -> Bool:
    """Should hold; a failure is logged and counted, and the caller copes. Returns `condition`."""
    if not condition:
        ensure_failed(category, message)
    return condition


@always_inline
def dev_assert(condition: Bool, category: Int, message: StaticString):
    """A check for development builds; compiled out with -D SD_RELEASE."""
    comptime if DEV_ASSERTS:
        if not condition:
            fail_hard(category, "dev_assert", message)


def ensure_failures() -> Int:
    return anchor_get(A_ENSURE_FAILURES) if anchor_ready() else 0

"""Menu narration (package 3.15, a Could): the focused menu line spoken by
speech-dispatcher's `spd-say` when it is installed.

`speak` starts `spd-say -C` (stop what it was saying) and then `spd-say
<text>` with posix_spawn, no shell, never waiting; children that have
finished are reaped (waitpid WNOHANG) on each call. Without
/usr/bin/spd-say it does nothing. Presentation only: render/interface.mojo
calls it from drawing, never from a tick, and never in automated runs.
"""
from std.ffi import external_call
from flint.platform.os import file_size, env_value

comptime SPD_SAY = "/usr/bin/spd-say"
comptime WNOHANG: Int32 = 1


def c_bytes(text: String) -> List[UInt8]:
    var out = List[UInt8](capacity=text.byte_length() + 1)
    for byte in text.as_bytes():
        out.append(byte)
    out.append(0)
    return out^


def spawn(args: List[String]) -> Bool:
    """Starts SPD_SAY with `args` (args[0] is the program name) and a small
    environment (what spd-say needs to reach the speech server)."""
    var storage = List[List[UInt8]]()
    for arg in args:
        storage.append(c_bytes(arg))
    var names = ["HOME", "PATH", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "LANG", "SPEECHD_ADDRESS"]
    var env_count = 0
    for name in names:
        var value = env_value(String(name))
        if value.byte_length() > 0:
            storage.append(c_bytes(String(name) + "=" + value))
            env_count += 1
    var argv = List[Int]()
    for index in range(len(args)):
        argv.append(Int(storage[index].unsafe_ptr()))
    argv.append(0)
    var envp = List[Int]()
    for index in range(len(args), len(args) + env_count):
        envp.append(Int(storage[index].unsafe_ptr()))
    envp.append(0)
    var path = c_bytes(SPD_SAY)
    var pid = List[Int32](length=1, fill=0)
    var result = external_call["posix_spawn", Int32](pid.unsafe_ptr(), path.unsafe_ptr(), Int(0), Int(0), argv.unsafe_ptr(), envp.unsafe_ptr())
    _ = storage^
    _ = path^
    return result == 0


def reap():
    """Collects finished children (never blocks)."""
    var status = List[Int32](length=1, fill=0)
    for _ in range(8):
        if external_call["waitpid", Int32](Int32(-1), status.unsafe_ptr(), WNOHANG) <= 0:
            return


def speak(text: String):
    if file_size(SPD_SAY) < 0 or text.byte_length() == 0:
        return
    reap()
    _ = spawn(["spd-say", "-C"])
    _ = spawn(["spd-say", text])

"""Private controlling terminal for a Studio-owned TurboMojo process on Linux."""
from std.ffi import external_call
from flint.devtools.job_runner import (
    ChildExecPrelude,
    child_exec_prelude,
    reap_child,
    signal_process_group,
)


def _c_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length() + 1)
    for b in s.as_bytes():
        out.append(b)
    out.append(0)
    return out^


struct IdePty(Movable):
    var fd: Int32
    var pid: Int32
    var pending: List[UInt8]

    def __init__(out self):
        self.fd = -1
        self.pid = -1
        self.pending = List[UInt8]()

    def __moveinit__(out self, deinit existing: Self):
        self.fd = existing.fd
        self.pid = existing.pid
        self.pending = existing.pending^
        existing.fd = -1
        existing.pid = -1

    def __deinit__(deinit self):
        self.stop()

    def start(mut self, cols: Int, rows: Int) -> Bool:
        self.stop()
        var size: Array[UInt16, 4] = [UInt16(max(1, rows)), UInt16(max(1, cols)), UInt16(0), UInt16(0)]
        var master = Int32(-1)
        # Bash expands the launcher's paths as individual arguments, after exec.
        # Every C string and the argv array are built before fork: the child must not allocate.
        var program = _c_bytes("/bin/bash")
        var option = _c_bytes("-c")
        var command = _c_bytes(
            'cd -- "$FLINT_STUDIO_PROJECT" && exec bash "$FLINT_STUDIO_IDE_LAUNCHER" studio'
            ' --ide-embedded "$FLINT_STUDIO_IDE_FILE"'
        )
        var argv = List[Int](length=4, fill=0)
        argv[0] = Int(program.unsafe_ptr())
        argv[1] = Int(option.unsafe_ptr())
        argv[2] = Int(command.unsafe_ptr())
        var prelude = ChildExecPrelude()
        var pid = external_call["forkpty", Int32](
            Pointer(to=master), Int(0), Int(0), Pointer(to=size)
        )
        if pid == 0:
            # forkpty already made the child a session (and process-group) leader.
            child_exec_prelude(prelude, new_group=False)
            _ = external_call["close_range", Int32](UInt32(3), UInt32(0xFFFFFFFF), UInt32(0))
            _ = external_call["execv", Int32](program.unsafe_ptr(), argv.unsafe_ptr())
            external_call["_exit", NoneType](Int32(127))
        _ = program[0]
        _ = option[0]
        _ = command[0]
        _ = argv[0]
        _ = prelude.sigdfl[0]
        if pid < 0:
            return False
        self.fd = master
        self.pid = pid
        _ = external_call["fcntl", Int32](master, Int32(2), Int32(1))
        var fl = external_call["fcntl", Int32](master, Int32(3), Int32(0))
        if fl < 0 or external_call["fcntl", Int32](master, Int32(4), fl | Int32(2048)) < 0:
            self.stop()
            return False
        self.pending.clear()
        return True

    def resize(self, cols: Int, rows: Int):
        if self.fd < 0 or cols <= 0 or rows <= 0:
            return
        var size: Array[UInt16, 4] = [UInt16(rows), UInt16(cols), UInt16(0), UInt16(0)]
        _ = external_call["ioctl", Int32](self.fd, UInt64(0x5414), Pointer(to=size))

    def send(mut self, s: String):
        for b in s.as_bytes():
            self.pending.append(b)

    def flush(mut self):
        if self.fd < 0 or not len(self.pending):
            return
        var n = external_call["write", Int](self.fd, self.pending.unsafe_ptr(), len(self.pending))
        if n > 0:
            var rest = List[UInt8]()
            for i in range(n, len(self.pending)):
                rest.append(self.pending[i])
            self.pending = rest^

    def reap(mut self, wait: Bool = False) -> Bool:
        if self.pid <= 0:
            return True
        var rr = reap_child(self.pid, block=wait)
        if rr[0]:
            self.pid = -1
            if self.fd >= 0:
                _ = external_call["close", Int32](self.fd)
                self.fd = -1
            return True
        return False

    def stop(mut self):
        """Ends the session without sleeping on the UI thread: closing the master hangs up the
        terminal, SIGHUP goes to the whole process group (the launcher, TurboMojo and anything they
        started), and whatever is still alive at the following non-blocking check is SIGKILLed
        (group-wide) and reaped. SIGKILL is immediate, so the final reap does not wait on the child."""
        if self.reap():
            if self.fd >= 0:
                _ = external_call["close", Int32](self.fd)
                self.fd = -1
            return
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = -1
        if self.pid > 0:
            signal_process_group(self.pid, Int32(1))
            if self.reap():
                return
            signal_process_group(self.pid, Int32(9))
            _ = self.reap(True)

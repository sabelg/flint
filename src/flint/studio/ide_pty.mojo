"""Private controlling terminal for a Studio-owned TurboMojo process on Linux."""
from std.ffi import external_call
from flint.platform.os import last_errno


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
        var program = String("/bin/bash")
        var option = String("-c")
        var command = String(
            'cd -- "$FLINT_STUDIO_PROJECT" && exec bash "$FLINT_STUDIO_IDE_LAUNCHER" studio'
            ' --ide-embedded "$FLINT_STUDIO_IDE_FILE"'
        )
        var pid = external_call["forkpty", Int32](
            Pointer(to=master), Int(0), Int(0), Pointer(to=size)
        )
        if pid == 0:
            _ = external_call["close_range", Int32](UInt32(3), UInt32(0xFFFFFFFF), UInt32(0))
            _ = external_call["execl", Int32](
                program.as_c_string_span(),
                program.as_c_string_span(),
                option.as_c_string_span(),
                command.as_c_string_span(),
                Int(0),
            )
            external_call["_exit", NoneType](Int32(127))
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
        var status = Int32(0)
        var r = external_call["waitpid", Int32](
            self.pid, Pointer(to=status), Int32(0 if wait else 1)
        )
        if r == self.pid or (r < 0 and last_errno() == 10):
            self.pid = -1
            if self.fd >= 0:
                _ = external_call["close", Int32](self.fd)
                self.fd = -1
            return True
        return False

    def stop(mut self):
        if self.reap():
            if self.fd >= 0:
                _ = external_call["close", Int32](self.fd)
                self.fd = -1
            return
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = -1
        if self.pid > 0:
            _ = external_call["kill", Int32](self.pid, Int32(1))
            for _ in range(40):
                if self.reap():
                    return
                _ = external_call["usleep", Int32](UInt32(10000))
            _ = external_call["kill", Int32](self.pid, Int32(9))
            while not self.reap(True):
                pass

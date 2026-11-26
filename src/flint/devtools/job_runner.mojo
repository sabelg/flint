"""Pure-Mojo 1.2 non-blocking background job runner (`flint.devtools.job_runner`).

Runs build, gameplay `.so` compilation, cook, test, and shell commands without
stalling Studio's render loop. Streams combined stdout/stderr incrementally,
tracks elapsed duration & exit codes, supports safe cancellation (`SIGTERM`/`SIGKILL`),
and parses Mojo compiler diagnostics automatically via `flint.devtools.problems`.
"""
from std.collections import List
from std.ffi import external_call
from std.math import max
from flint.devtools.problems import DIAG_SRC_BUILD, Diagnostic, parse_compiler_output
from flint.platform.os import close_fd, get_time, last_errno

comptime JOB_IDLE: Int = 0
comptime JOB_RUNNING: Int = 1
comptime JOB_SUCCEEDED: Int = 2
comptime JOB_FAILED: Int = 3
comptime JOB_CANCELLED: Int = 4

comptime JOB_KIND_BUILD_SO: Int = 1
comptime JOB_KIND_TEST: Int = 2
comptime JOB_KIND_COOK: Int = 3
comptime JOB_KIND_TASK: Int = 4

comptime DEFAULT_JOB_OUTPUT_LIMIT: Int = 2 * 1024 * 1024

comptime _O_NONBLOCK: Int32 = 2048
comptime _EINTR: Int32 = 4
comptime _EAGAIN: Int32 = 11
comptime _ECHILD: Int32 = 10
comptime _SIGTERM: Int32 = 15
comptime _SIGKILL: Int32 = 9


def _c_bytes(s: String) -> List[UInt8]:
    var sb = s.as_bytes()
    var out = List[UInt8](capacity=len(sb) + 1)
    for i in range(len(sb)):
        out.append(sb[i])
    out.append(0)
    return out^


def _set_fd_nonblock_cloexec(fd: Int32):
    if fd < 0:
        return
    _ = external_call["fcntl", Int32](fd, Int32(2), Int32(1))
    var fl = external_call["fcntl", Int32](fd, Int32(3), Int32(0))
    if fl >= 0:
        _ = external_call["fcntl", Int32](fd, Int32(4), fl | _O_NONBLOCK)


struct BackgroundJob(Movable):
    """Single non-blocking subprocess job with streaming output and parsed diagnostics."""
    var id: Int
    var kind: Int
    var title: String
    var cwd: String
    var command_display: String
    var state: Int
    var pid: Int32
    var pipe_fd: Int32
    var exit_code: Int
    var started_sec: Float64
    var finished_sec: Float64
    var max_output_bytes: Int
    var truncated: Bool
    var output_bytes: List[UInt8]
    var streamed_line_cursor: Int
    var diagnostics: List[Diagnostic]

    def __init__(
        out self,
        id: Int = 0,
        kind: Int = JOB_KIND_TASK,
        title: String = "",
        cwd: String = ".",
        max_output_bytes: Int = DEFAULT_JOB_OUTPUT_LIMIT,
    ):
        self.id = id
        self.kind = kind
        self.title = title
        self.cwd = cwd
        self.command_display = ""
        self.state = JOB_IDLE
        self.pid = -1
        self.pipe_fd = -1
        self.exit_code = -1
        self.started_sec = 0.0
        self.finished_sec = 0.0
        self.max_output_bytes = max(4096, max_output_bytes)
        self.truncated = False
        self.output_bytes = List[UInt8]()
        self.streamed_line_cursor = 0
        self.diagnostics = List[Diagnostic]()

    def __deinit__(deinit self):
        self.cancel()

    def is_running(self) -> Bool:
        return self.state == JOB_RUNNING

    def is_finished(self) -> Bool:
        return (
            self.state == JOB_SUCCEEDED
            or self.state == JOB_FAILED
            or self.state == JOB_CANCELLED
        )

    def elapsed_sec(self, now_sec: Float64 = -1.0) -> Float64:
        if self.state == JOB_IDLE:
            return 0.0
        if self.is_finished():
            return max(0.0, self.finished_sec - self.started_sec)
        var t = now_sec if now_sec >= 0.0 else get_time()
        return max(0.0, t - self.started_sec)

    def output_text(self) -> String:
        return String(unsafe_from_utf8=self.output_bytes)

    def _append_bytes(mut self, src: List[UInt8], count: Int):
        var avail = self.max_output_bytes - len(self.output_bytes)
        if avail <= 0:
            self.truncated = True
            return
        var take = count if count <= avail else avail
        if take < count:
            self.truncated = True
        for i in range(take):
            self.output_bytes.append(src[i])

    def start_argv(mut self, argv: List[String], cwd: String = "", now_sec: Float64 = -1.0) -> Bool:
        self.cancel()
        if len(argv) == 0:
            return False
        if cwd.byte_length() > 0:
            self.cwd = cwd

        var disp = String("")
        for i in range(len(argv)):
            if i > 0:
                disp += " "
            disp += argv[i]
        self.command_display = disp

        var fds = List[Int32](length=2, fill=-1)
        if external_call["pipe", Int32](fds.unsafe_ptr()) != 0:
            return False
        var rd = fds[0]
        var wr = fds[1]

        var args = List[List[UInt8]](capacity=len(argv))
        for i in range(len(argv)):
            args.append(_c_bytes(argv[i]))
        var ptrs = List[Int](length=len(args) + 1, fill=0)
        for i in range(len(args)):
            ptrs[i] = Int(args[i].unsafe_ptr())
        var dir_c = _c_bytes(self.cwd)

        var pid = external_call["fork", Int32]()
        if pid == 0:
            var devnull = external_call["openat", Int32](
                Int32(-100), _c_bytes("/dev/null").unsafe_ptr(), Int32(0), Int32(0)
            )
            if devnull >= 0:
                _ = external_call["dup2", Int32](devnull, Int32(0))
            _ = external_call["dup2", Int32](wr, Int32(1))
            _ = external_call["dup2", Int32](wr, Int32(2))
            _ = external_call["close_range", Int32](UInt32(3), UInt32(0xFFFFFFFF), UInt32(0))
            if self.cwd.byte_length() > 0:
                _ = external_call["chdir", Int32](dir_c.unsafe_ptr())
            _ = external_call["execvp", Int32](args[0].unsafe_ptr(), ptrs.unsafe_ptr())
            external_call["_exit", NoneType](Int32(127))

        _ = len(args)
        _ = ptrs[0]
        _ = dir_c[0]

        close_fd(Int(wr))
        if pid < 0:
            close_fd(Int(rd))
            return False

        _set_fd_nonblock_cloexec(rd)
        self.pid = pid
        self.pipe_fd = rd
        self.state = JOB_RUNNING
        self.exit_code = -1
        self.started_sec = now_sec if now_sec >= 0.0 else get_time()
        self.finished_sec = self.started_sec
        self.truncated = False
        self.output_bytes.clear()
        self.streamed_line_cursor = 0
        self.diagnostics.clear()
        return True

    def start_shell(mut self, shell_cmd: String, cwd: String = "", now_sec: Float64 = -1.0) -> Bool:
        var argv = List[String]()
        argv.append("/bin/sh")
        argv.append("-c")
        argv.append(shell_cmd)
        var ok = self.start_argv(argv, cwd, now_sec)
        if ok:
            self.command_display = shell_cmd
        return ok

    def _drain_pipe(mut self) -> Bool:
        if self.pipe_fd < 0:
            return False
        var got_any = False
        var scratch = List[UInt8](length=16384, fill=0)
        for _ in range(32):
            var n = Int(external_call["read", Int64](self.pipe_fd, scratch.unsafe_ptr(), Int64(16384)))
            if n > 0:
                self._append_bytes(scratch, n)
                got_any = True
                continue
            if n == 0:
                close_fd(Int(self.pipe_fd))
                self.pipe_fd = -1
                break
            var err = last_errno()
            if err == _EINTR:
                continue
            if err == _EAGAIN:
                break
            close_fd(Int(self.pipe_fd))
            self.pipe_fd = -1
            break
        return got_any

    def poll(mut self, now_sec: Float64 = -1.0) -> Bool:
        """Drains output and checks process exit. Returns True if output arrived or state changed."""
        if self.state != JOB_RUNNING:
            return False
        var changed = self._drain_pipe()
        if self.pid > 0:
            var st = Int32(0)
            var r = external_call["waitpid", Int32](self.pid, Pointer(to=st), Int32(1))
            if r == self.pid or (r < 0 and last_errno() == _ECHILD):
                _ = self._drain_pipe()
                if self.pipe_fd >= 0:
                    close_fd(Int(self.pipe_fd))
                    self.pipe_fd = -1
                var wst = Int(st)
                if r == self.pid:
                    self.exit_code = ((wst >> 8) & 255) if (wst & 127) == 0 else (128 + (wst & 127))
                else:
                    self.exit_code = 0
                self.pid = -1
                self.finished_sec = now_sec if now_sec >= 0.0 else get_time()
                self.state = JOB_SUCCEEDED if self.exit_code == 0 else JOB_FAILED
                self.diagnostics = parse_compiler_output(self.output_text(), self.cwd, DIAG_SRC_BUILD)
                changed = True
        return changed

    def take_new_lines(mut self) -> List[String]:
        """Returns complete output lines received since the previous `take_new_lines()` call."""
        var out = List[String]()
        var n = len(self.output_bytes)
        var start = self.streamed_line_cursor
        var i = start
        while i < n:
            if self.output_bytes[i] == 10:
                var end = i
                if end > start and self.output_bytes[end - 1] == 13:
                    end -= 1
                var buf = List[UInt8](capacity=end - start)
                for k in range(start, end):
                    buf.append(self.output_bytes[k])
                out.append(String(unsafe_from_utf8=buf))
                start = i + 1
            i += 1
        self.streamed_line_cursor = start
        if self.is_finished() and self.streamed_line_cursor < n:
            var buf = List[UInt8](capacity=n - self.streamed_line_cursor)
            for k in range(self.streamed_line_cursor, n):
                if self.output_bytes[k] != 13:
                    buf.append(self.output_bytes[k])
            if len(buf) > 0:
                out.append(String(unsafe_from_utf8=buf))
            self.streamed_line_cursor = n
        return out^

    def cancel(mut self):
        if self.pid > 0:
            _ = external_call["kill", Int32](self.pid, _SIGTERM)
            var exited = False
            for _ in range(20):
                var st = Int32(0)
                var r = external_call["waitpid", Int32](self.pid, Pointer(to=st), Int32(1))
                if r == self.pid or (r < 0 and last_errno() == _ECHILD):
                    exited = True
                    break
                _ = external_call["usleep", Int32](UInt32(10000))
            if not exited:
                _ = external_call["kill", Int32](self.pid, _SIGKILL)
                var st = Int32(0)
                _ = external_call["waitpid", Int32](self.pid, Pointer(to=st), Int32(0))
            self.pid = -1
            if self.state == JOB_RUNNING:
                self.state = JOB_CANCELLED
                self.finished_sec = get_time()
        if self.pipe_fd >= 0:
            close_fd(Int(self.pipe_fd))
            self.pipe_fd = -1

    def feed_mock_output(mut self, text: String, finish_exit_code: Int = -999, now_sec: Float64 = 0.0):
        """Injects synthetic output and optional completion status for deterministic headless tests."""
        var sb = text.as_bytes()
        var buf = List[UInt8](capacity=len(sb))
        for i in range(len(sb)):
            buf.append(sb[i])
        self._append_bytes(buf, len(buf))
        if finish_exit_code != -999:
            self.exit_code = finish_exit_code
            self.finished_sec = now_sec
            self.state = JOB_SUCCEEDED if finish_exit_code == 0 else JOB_FAILED
            self.diagnostics = parse_compiler_output(self.output_text(), self.cwd, DIAG_SRC_BUILD)


struct JobRunner(Movable):
    """Manages non-blocking Studio background jobs (gameplay `.so` builds, cooks, tests, tasks)."""
    var jobs: List[BackgroundJob]
    var next_job_id: Int
    var last_completed_id: Int

    def __init__(out self):
        self.jobs = List[BackgroundJob]()
        self.next_job_id = 1
        self.last_completed_id = 0

    def find_index(self, job_id: Int) -> Int:
        for i in range(len(self.jobs)):
            if self.jobs[i].id == job_id:
                return i
        return -1

    def start_argv_job(
        mut self,
        kind: Int,
        title: String,
        argv: List[String],
        cwd: String = ".",
        now_sec: Float64 = -1.0,
    ) -> Int:
        var jid = self.next_job_id
        self.next_job_id += 1
        var job = BackgroundJob(jid, kind, title, cwd)
        if not job.start_argv(argv, cwd, now_sec):
            return -1
        self.jobs.append(job^)
        return jid

    def start_shell_job(
        mut self,
        kind: Int,
        title: String,
        shell_cmd: String,
        cwd: String = ".",
        now_sec: Float64 = -1.0,
    ) -> Int:
        var jid = self.next_job_id
        self.next_job_id += 1
        var job = BackgroundJob(jid, kind, title, cwd)
        if not job.start_shell(shell_cmd, cwd, now_sec):
            return -1
        self.jobs.append(job^)
        return jid

    def start_gameplay_so_build(
        mut self,
        project_root: String,
        entry_mojo: String,
        out_so: String,
        include_paths: List[String],
        compiler: String = "mojo",
        opt_flag: String = "-O1",
        werror: Bool = True,
    ) -> Int:
        """Starts an async `mojo build --emit shared-lib` job for 1-click gameplay `.so` hot-reload."""
        var argv = List[String]()
        argv.append(compiler)
        argv.append("build")
        argv.append("--emit")
        argv.append("shared-lib")
        argv.append(opt_flag)
        if werror:
            argv.append("--Werror")
        for i in range(len(include_paths)):
            argv.append("-I")
            argv.append(include_paths[i])
        argv.append(entry_mojo)
        argv.append("-o")
        argv.append(out_so)
        return self.start_argv_job(JOB_KIND_BUILD_SO, "Build gameplay.so", argv, project_root)

    def add_mock_job(mut self, kind: Int, title: String, cwd: String = ".") -> Int:
        """Creates a running synthetic job for deterministic headless unit tests."""
        var jid = self.next_job_id
        self.next_job_id += 1
        var job = BackgroundJob(jid, kind, title, cwd)
        job.state = JOB_RUNNING
        self.jobs.append(job^)
        return jid

    def poll_all(mut self, now_sec: Float64 = -1.0) -> List[Int]:
        """Polls all running jobs and returns IDs of jobs that finished on this tick."""
        var newly_done = List[Int]()
        for i in range(len(self.jobs)):
            var was_running = self.jobs[i].is_running()
            _ = self.jobs[i].poll(now_sec)
            if was_running and self.jobs[i].is_finished():
                self.last_completed_id = self.jobs[i].id
                newly_done.append(self.jobs[i].id)
        return newly_done^

    def cancel(mut self, job_id: Int) -> Bool:
        var idx = self.find_index(job_id)
        if idx < 0:
            return False
        self.jobs[idx].cancel()
        return True

    def cancel_all(mut self):
        for i in range(len(self.jobs)):
            self.jobs[i].cancel()

    def active_count(self) -> Int:
        var n = 0
        for i in range(len(self.jobs)):
            if self.jobs[i].is_running():
                n += 1
        return n

    def has_running_build(self) -> Bool:
        for i in range(len(self.jobs)):
            if self.jobs[i].is_running() and self.jobs[i].kind == JOB_KIND_BUILD_SO:
                return True
        return False

    def latest_diagnostics(self) -> List[Diagnostic]:
        var idx = self.find_index(self.last_completed_id)
        if idx < 0:
            return List[Diagnostic]()
        return self.jobs[idx].diagnostics.copy()

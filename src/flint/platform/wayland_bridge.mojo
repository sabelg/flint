"""One Wayland connection shared by our client and libwayland, for VK_KHR_wayland_surface.

That extension takes a wl_display and a wl_surface as libwayland objects, and the
driver speaks to them through libwayland, which it links for itself. The window and
its input stay ours, and a surface belongs to one connection, so both must use the
same one. The driver's library is not given the compositor's socket. It gets one end
of a socketpair, and a thread here passes bytes along:

    compositor <-> our socket <-> this thread <-> socketpair <-> libwayland

Events are sorted by the object they are for, which needs only the wire header, so
no interface tables or C listeners exist here. Requests from libwayland go out
unchanged, a whole message at a time, so they never split around one of ours.

Compositors reject a new id that skips ahead, and two allocators would soon do
that. So every id comes from libwayland's allocator: `allocate_id` reserves one with
wl_proxy_create, and first pushes out anything libwayland has already numbered.

File descriptors come with no marker for their message. The only ones that reach
this client are wl_keyboard.keymap, which is ours and unused, so it is closed, and
the dmabuf feedback table, which the driver reads. Others stay in order for libwayland.
"""
from std.ffi import external_call
from flint.platform.os import close_fd
from flint.platform.vulkan import call1, get32, get64, put32, put64

comptime WORD_REAL: Int = 0
comptime WORD_LIBRARY: Int = 1
comptime WORD_NOTIFY: Int = 2
comptime WORD_STOP: Int = 3
comptime WORD_IN: Int = 4
comptime WORD_IN_USED: Int = 5
comptime WORD_OUT: Int = 6
comptime WORD_OUT_USED: Int = 7
comptime WORD_GAME: Int = 8
comptime WORD_GAME_USED: Int = 9
comptime WORD_OWNED: Int = 10
comptime WORD_KEYBOARD: Int = 11
comptime WORD_MUTEX: Int = 12
comptime WORD_IN_FDS: Int = 13
comptime WORD_IN_FD_COUNT: Int = 14
comptime WORD_OUT_FDS: Int = 15
comptime WORD_OUT_FD_COUNT: Int = 16
comptime WORD_THREAD: Int = 17
comptime WORD_FAILED: Int = 18
comptime STATE_WORDS: Int = 24

comptime STREAM_BYTES: Int = 256 * 1024
comptime GAME_BYTES: Int = 1024 * 1024
comptime MAX_OWNED_ID: Int = 1 << 22
comptime MAX_FDS: Int = 64
comptime CONTROL_BYTES: Int = 16 + MAX_FDS * 4
comptime POLL_MS: Int32 = 100
comptime DONT_WAIT: Int32 = 64
comptime SOCKET_LEVEL: UInt32 = 1
comptime SCM_RIGHTS: UInt32 = 1
comptime POLL_IN: UInt32 = 1
comptime WL_DISPLAY_ID: Int = 1
comptime OPCODE_ERROR: Int = 0
comptime OPCODE_DELETE_ID: Int = 1
comptime SURFACE_VERSION: Int = 4


def raw(address: Int) -> Pointer[UInt8, origin=MutAnyOrigin]:
    """external_call fixes a symbol's signature program-wide, and the rest of the
    engine passes these as pointers."""
    return Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=address)


def word(state: Int, index: Int) -> Int:
    return get64(state + index * 8)


def set_word(state: Int, index: Int, value: Int):
    put64(state + index * 8, value)


def is_owned(state: Int, id: Int) -> Bool:
    if id >= MAX_OWNED_ID:
        return False
    var byte = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=word(state, WORD_OWNED) + id // 8)[unsafe_offset=0]
    return (byte >> UInt8(id % 8)) & 1 != 0


def set_owned(state: Int, id: Int):
    if id < MAX_OWNED_ID:
        var pointer = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=word(state, WORD_OWNED) + id // 8)
        pointer[unsafe_offset=0] = pointer[unsafe_offset=0] | (UInt8(1) << UInt8(id % 8))


def lock(state: Int):
    _ = external_call["pthread_mutex_lock", Int32](word(state, WORD_MUTEX))


def unlock(state: Int):
    _ = external_call["pthread_mutex_unlock", Int32](word(state, WORD_MUTEX))


def receive_some(fd: Int, buffer: Int, capacity: Int, fds: Int, mut fd_count: Int) -> Int:
    """recvmsg without waiting. Passed descriptors join the end of `fds`."""
    var iov = external_call["malloc", Int](16 + 56 + CONTROL_BYTES)
    var message = iov + 16
    var control = message + 56
    _ = external_call["memset", Int](iov, Int32(0), 16 + 56 + CONTROL_BYTES)
    put64(iov, buffer)
    put64(iov + 8, capacity)
    put64(message + 16, iov)
    put64(message + 24, 1)
    put64(message + 32, control)
    put64(message + 40, CONTROL_BYTES)
    var count = Int(external_call["recvmsg", Int64](Int32(fd), raw(message), DONT_WAIT))
    if count >= 0 and get32(control + 8) == SOCKET_LEVEL and get32(control + 12) == SCM_RIGHTS:
        var carried = (get64(control) - 16) // 4
        for index in range(carried):
            var descriptor = Int(get32(control + 16 + index * 4))
            if fd_count < MAX_FDS:
                put64(fds + fd_count * 8, descriptor)
                fd_count += 1
            else:
                close_fd(descriptor)
    external_call["free", NoneType](iov)
    return count


def send_some(fd: Int, buffer: Int, size: Int, fds: Int, fd_count: Int) -> Bool:
    """sendmsg all of `size`, with the descriptors on its first byte."""
    var iov = external_call["malloc", Int](16 + 56 + CONTROL_BYTES)
    var message = iov + 16
    var control = message + 56
    _ = external_call["memset", Int](iov, Int32(0), 16 + 56 + CONTROL_BYTES)
    var sent = 0
    var ok = True
    while sent < size and ok:
        put64(iov, buffer + sent)
        put64(iov + 8, size - sent)
        put64(message + 16, iov)
        put64(message + 24, 1)
        if sent == 0 and fd_count > 0:
            put64(control, 16 + fd_count * 4)
            put32(control + 8, SOCKET_LEVEL)
            put32(control + 12, SCM_RIGHTS)
            for index in range(fd_count):
                put32(control + 16 + index * 4, UInt32(get64(fds + index * 8)))
            put64(message + 32, control)
            put64(message + 40, (16 + fd_count * 4 + 7) & ~7)
        else:
            put64(message + 32, 0)
            put64(message + 40, 0)
        var count = Int(external_call["sendmsg", Int64](Int32(fd), raw(message), Int32(0)))
        if count <= 0:
            ok = False
        else:
            sent += count
    external_call["free", NoneType](iov)
    return ok


def take_incoming(state: Int) -> Bool:
    """Compositor to here: sort every whole event. True if the game got something."""
    var used = word(state, WORD_IN_USED)
    var buffer = word(state, WORD_IN)
    var fds = word(state, WORD_IN_FDS)
    var fd_count = word(state, WORD_IN_FD_COUNT)
    var count = receive_some(word(state, WORD_REAL), buffer + used, STREAM_BYTES - used, fds, fd_count)
    if count == 0:
        set_word(state, WORD_FAILED, 1)
    used += max(count, 0)
    var offset = 0
    var game_got = False
    while used - offset >= 8:
        var object = Int(get32(buffer + offset))
        var header = Int(get32(buffer + offset + 4))
        var opcode = header & 0xFFFF
        var size = header >> 16
        if size < 8 or used - offset < size:
            break
        var for_game = is_owned(state, object)
        var for_library = not for_game
        if object == WL_DISPLAY_ID:
            for_game = opcode == OPCODE_ERROR
            for_library = opcode == OPCODE_ERROR or not (opcode == OPCODE_DELETE_ID and is_owned(state, Int(get32(buffer + offset + 8))))
        if for_game:
            var game_used = word(state, WORD_GAME_USED)
            if game_used + size <= GAME_BYTES:
                _ = external_call["memcpy", Int](word(state, WORD_GAME) + game_used, buffer + offset, size)
                set_word(state, WORD_GAME_USED, game_used + size)
                game_got = True
            if object == word(state, WORD_KEYBOARD) and opcode == 0 and fd_count > 0:
                close_fd(get64(fds))
                fd_count -= 1
                _ = external_call["memmove", Int](fds, fds + 8, fd_count * 8)
        if for_library:
            if not send_some(word(state, WORD_LIBRARY), buffer + offset, size, fds, fd_count):
                set_word(state, WORD_FAILED, 1)
            for index in range(fd_count):
                close_fd(get64(fds + index * 8))  # the kernel holds copies now
            fd_count = 0
        offset += size
    if offset > 0:
        _ = external_call["memmove", Int](buffer, buffer + offset, used - offset)
    set_word(state, WORD_IN_USED, used - offset)
    set_word(state, WORD_IN_FD_COUNT, fd_count)
    return game_got


def take_outgoing(state: Int):
    """libwayland to the compositor: every whole request, in order, descriptors with it."""
    var buffer = word(state, WORD_OUT)
    var fds = word(state, WORD_OUT_FDS)
    while True:
        var used = word(state, WORD_OUT_USED)
        var fd_count = word(state, WORD_OUT_FD_COUNT)
        var count = receive_some(word(state, WORD_LIBRARY), buffer + used, STREAM_BYTES - used, fds, fd_count)
        set_word(state, WORD_OUT_FD_COUNT, fd_count)
        if count <= 0:
            return
        used += count
        var whole = 0
        while used - whole >= 8:
            var size = Int(get32(buffer + whole + 4)) >> 16
            if size < 8 or used - whole < size:
                break
            whole += size
        if whole > 0:
            if not send_some(word(state, WORD_REAL), buffer, whole, fds, fd_count):
                set_word(state, WORD_FAILED, 1)
            for index in range(fd_count):
                close_fd(get64(fds + index * 8))
            set_word(state, WORD_OUT_FD_COUNT, 0)
            _ = external_call["memmove", Int](buffer, buffer + whole, used - whole)
        set_word(state, WORD_OUT_USED, used - whole)


def bridge_thread(state: Int) abi("C") -> Int:
    var descriptors = external_call["malloc", Int](16)
    put32(descriptors, UInt32(word(state, WORD_REAL)))
    put32(descriptors + 4, POLL_IN)
    put32(descriptors + 8, UInt32(word(state, WORD_LIBRARY)))
    put32(descriptors + 12, POLL_IN)
    while word(state, WORD_STOP) == 0:
        var ready = external_call["poll", Int32](raw(descriptors), UInt64(2), POLL_MS)
        if ready <= 0:
            continue
        var from_compositor = (get32(descriptors + 4) >> 16) != 0
        var from_library = (get32(descriptors + 12) >> 16) != 0
        lock(state)
        if from_library:
            take_outgoing(state)
        var game_got = False
        if from_compositor:
            game_got = take_incoming(state)
        unlock(state)
        if game_got:
            var one = external_call["malloc", Int](8)
            put64(one, 1)
            _ = external_call["write", Int64](Int32(word(state, WORD_NOTIFY)), raw(one), Int64(8))
            external_call["free", NoneType](one)
    external_call["free", NoneType](descriptors)
    return 0


struct WaylandBridge(Copyable, Movable):
    var state: Int
    var library: Int
    var display: Int  # wl_display*
    var surface_proxy: Int  # wl_surface*
    var compositor_proxy: Int
    var fn_flush: Int
    var fn_proxy_create: Int
    var fn_proxy_id: Int
    var fn_constructor: Int
    var fn_disconnect: Int
    var fn_proxy_destroy: Int
    var callback_interface: Int
    var compositor_interface: Int
    var surface_interface: Int
    var error: String

    def __init__(out self):
        self.state = 0
        self.library = 0
        self.display = 0
        self.surface_proxy = 0
        self.compositor_proxy = 0
        self.fn_flush = 0
        self.fn_proxy_create = 0
        self.fn_proxy_id = 0
        self.fn_constructor = 0
        self.fn_disconnect = 0
        self.fn_proxy_destroy = 0
        self.callback_interface = 0
        self.compositor_interface = 0
        self.surface_interface = 0
        self.error = ""

    def active(self) -> Bool:
        return self.state != 0

    def symbol(mut self, var name: String) -> Int:
        var address = external_call["dlsym", Int](self.library, name.as_c_string_span())
        if address == 0:
            self.error = "libwayland-client has no " + name
        return address

    def start(mut self, real_fd: Int) -> Bool:
        """Take over a connected socket on which nothing has been said yet."""
        var library_name = "libwayland-client.so.0"
        self.library = external_call["dlopen", Int](library_name.as_c_string_span(), Int32(2))
        if self.library == 0:
            self.error = "no libwayland-client on this system"
            return False
        var connect = self.symbol("wl_display_connect_to_fd")
        self.fn_flush = self.symbol("wl_display_flush")
        self.fn_disconnect = self.symbol("wl_display_disconnect")
        self.fn_proxy_create = self.symbol("wl_proxy_create")
        self.fn_proxy_id = self.symbol("wl_proxy_get_id")
        self.fn_proxy_destroy = self.symbol("wl_proxy_destroy")
        self.fn_constructor = self.symbol("wl_proxy_marshal_array_constructor_versioned")
        self.callback_interface = self.symbol("wl_callback_interface")
        self.compositor_interface = self.symbol("wl_compositor_interface")
        self.surface_interface = self.symbol("wl_surface_interface")
        if self.error.byte_length() > 0:
            return False
        var pair = external_call["malloc", Int](8)
        if external_call["socketpair", Int32](Int32(1), Int32(1), Int32(0), pair) != 0:
            self.error = "socketpair failed"
            return False
        var ours = Int(get32(pair))
        var theirs = Int(get32(pair + 4))
        external_call["free", NoneType](pair)
        self.display = self.connect_display(connect, theirs)
        if self.display == 0:
            self.error = "wl_display_connect_to_fd failed"
            return False
        var state = external_call["calloc", Int](Int(STATE_WORDS), Int(8))
        set_word(state, WORD_REAL, real_fd)
        set_word(state, WORD_LIBRARY, ours)
        set_word(state, WORD_NOTIFY, Int(external_call["eventfd", Int32](UInt32(0), Int32(0x80800))))
        set_word(state, WORD_IN, external_call["malloc", Int](STREAM_BYTES))
        set_word(state, WORD_OUT, external_call["malloc", Int](STREAM_BYTES))
        set_word(state, WORD_GAME, external_call["malloc", Int](GAME_BYTES))
        set_word(state, WORD_OWNED, external_call["calloc", Int](Int(MAX_OWNED_ID // 8), Int(1)))
        set_word(state, WORD_MUTEX, external_call["calloc", Int](Int(1), Int(64)))
        set_word(state, WORD_IN_FDS, external_call["malloc", Int](MAX_FDS * 8))
        set_word(state, WORD_OUT_FDS, external_call["malloc", Int](MAX_FDS * 8))
        set_owned(state, WL_DISPLAY_ID)
        var thread = 0
        if external_call["pthread_create", Int32](Pointer(to=thread), Int(0), bridge_thread, state) != 0:
            self.error = "the bridge thread did not start"
            return False
        set_word(state, WORD_THREAD, thread)
        self.state = state
        return True

    def connect_display(self, connect: Int, fd: Int) -> Int:
        comptime Connect = def(Int32) thin abi("C") -> Int
        var slot = connect
        return Pointer(to=slot).unsafe_bitcast[Connect]()[unsafe_offset=0](Int32(fd))

    def create_proxy(self, interface: Int) -> Int:
        comptime Create = def(Int, Int) thin abi("C") -> Int
        var slot = self.fn_proxy_create
        return Pointer(to=slot).unsafe_bitcast[Create]()[unsafe_offset=0](self.display, interface)

    def proxy_id(self, proxy: Int) -> Int:
        comptime GetId = def(Int) thin abi("C") -> UInt32
        var slot = self.fn_proxy_id
        return Int(Pointer(to=slot).unsafe_bitcast[GetId]()[unsafe_offset=0](proxy))

    def construct(self, proxy: Int, opcode: Int, arguments: Int, interface: Int, version: Int) -> Int:
        comptime Construct = def(Int, UInt32, Int, Int, UInt32) thin abi("C") -> Int
        var slot = self.fn_constructor
        return Pointer(to=slot).unsafe_bitcast[Construct]()[unsafe_offset=0](proxy, UInt32(opcode), arguments, interface, UInt32(version))

    def push_out(self):
        """Everything libwayland has numbered goes first, or the compositor sees a gap."""
        _ = call1(self.fn_flush, self.display)
        lock(self.state)
        take_outgoing(self.state)
        unlock(self.state)

    def allocate_id(self) -> Int:
        self.push_out()
        var id = self.proxy_id(self.create_proxy(self.callback_interface))
        set_owned(self.state, id)
        return id

    def reserve_compositor(mut self) -> Int:
        """The id to bind wl_compositor to, with a proxy that can make surfaces."""
        self.push_out()
        self.compositor_proxy = self.create_proxy(self.compositor_interface)
        var id = self.proxy_id(self.compositor_proxy)
        set_owned(self.state, id)
        return id

    def create_surface(mut self) -> Int:
        """wl_compositor.create_surface through libwayland, so the driver has a real proxy."""
        var arguments = external_call["calloc", Int](Int(2), Int(8))
        self.surface_proxy = self.construct(self.compositor_proxy, 0, arguments, self.surface_interface, SURFACE_VERSION)
        external_call["free", NoneType](arguments)
        if self.surface_proxy == 0:
            self.error = "wl_compositor.create_surface failed"
            return 0
        var id = self.proxy_id(self.surface_proxy)
        set_owned(self.state, id)
        self.push_out()
        return id

    def note_keyboard(self, id: Int):
        lock(self.state)
        set_word(self.state, WORD_KEYBOARD, id)
        unlock(self.state)

    def write_all(self, address: Int, size: Int) -> Bool:
        self.push_out()
        lock(self.state)
        var ok = send_some(word(self.state, WORD_REAL), address, size, 0, 0)
        unlock(self.state)
        return ok

    def write_with_fd(self, address: Int, size: Int, fd: Int) -> Bool:
        self.push_out()
        var fds = external_call["malloc", Int](8)
        put64(fds, fd)
        lock(self.state)
        var ok = send_some(word(self.state, WORD_REAL), address, size, fds, 1)
        unlock(self.state)
        external_call["free", NoneType](fds)
        return ok

    def read_into(self, mut incoming: List[UInt8]):
        lock(self.state)
        var used = word(self.state, WORD_GAME_USED)
        var pointer = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=word(self.state, WORD_GAME))
        for index in range(used):
            incoming.append(pointer[unsafe_offset=index])
        set_word(self.state, WORD_GAME_USED, 0)
        unlock(self.state)

    def wait(self, timeout_ms: Int) -> Bool:
        """True once the game has events to read."""
        var descriptor = external_call["malloc", Int](16)
        put32(descriptor, UInt32(word(self.state, WORD_NOTIFY)))
        put32(descriptor + 4, POLL_IN)
        var ready = external_call["poll", Int32](raw(descriptor), UInt64(1), Int32(timeout_ms)) > 0
        if ready:
            _ = external_call["read", Int64](Int32(word(self.state, WORD_NOTIFY)), raw(descriptor + 8), Int64(8))
        external_call["free", NoneType](descriptor)
        return ready

    def failed(self) -> Bool:
        return word(self.state, WORD_FAILED) != 0

    def stop(mut self):
        """Stop the thread and let go of the library's connection. The socket stays ours."""
        if self.state == 0:
            return
        self.push_out()
        set_word(self.state, WORD_STOP, 1)
        _ = external_call["pthread_join", Int32](word(self.state, WORD_THREAD), Int(0))
        _ = call1(self.fn_disconnect, self.display)
        close_fd(word(self.state, WORD_LIBRARY))
        close_fd(word(self.state, WORD_NOTIFY))
        self.state = 0

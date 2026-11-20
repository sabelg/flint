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
from std.atomic import Atomic
from std.ffi import external_call
from flint.platform.dynlib import dl_open, dl_sym, dl_close
from flint.platform.os import close_fd, last_errno, EINTR
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
comptime WORD_IO_SCRATCH: Int = 19
comptime WORD_POLL_SCRATCH: Int = 20
comptime WORD_WAIT_SCRATCH: Int = 21
comptime WORD_GAME_FDS: Int = 22
comptime WORD_GAME_FD_COUNT: Int = 23
comptime WORD_DATA_DEVICE: Int = 24
comptime WORD_DATA_SOURCE: Int = 25
comptime WORD_OFFER_IDS: Int = 26
comptime WORD_OFFER_COUNT: Int = 27
comptime WORD_LIB_IFACES: Int = 28
comptime WORD_STRICT_IFACE: Int = 29
comptime STATE_WORDS: Int = 30
comptime MAX_OFFERS: Int = 32
comptime LIB_IFACE_CAP: Int = 1024
comptime IO_SCRATCH_BYTES: Int = 16 + 56 + CONTROL_BYTES

comptime LIB_IFACE_UNKNOWN: Int = 0
comptime LIB_IFACE_NO_FD: Int = 1
comptime LIB_IFACE_REGISTRY: Int = 2
comptime LIB_IFACE_DMABUF: Int = 3
comptime LIB_IFACE_DMABUF_FEEDBACK: Int = 4
comptime LIB_IFACE_EXPLICIT_SYNC: Int = 5
comptime LIB_IFACE_SURFACE_SYNC: Int = 6
comptime LIB_IFACE_BUFFER_RELEASE: Int = 7
comptime LIB_IFACE_COLOR_MANAGER: Int = 8
comptime LIB_IFACE_COLOR_SURFACE_FEEDBACK: Int = 9
comptime LIB_IFACE_IMAGE_DESC_CREATOR: Int = 10
comptime LIB_IFACE_IMAGE_DESC: Int = 11
comptime LIB_IFACE_IMAGE_DESC_INFO: Int = 12
comptime LIB_IFACE_DRM_LEASE_DEV: Int = 13
comptime LIB_IFACE_DRM_LEASE_REQ: Int = 14
comptime LIB_IFACE_DRM_LEASE: Int = 15

comptime STREAM_BYTES: Int = 256 * 1024
comptime GAME_BYTES: Int = 1024 * 1024
comptime MAX_OWNED_ID: Int = 1 << 22
comptime MAX_FDS: Int = 64
comptime CONTROL_BYTES: Int = 16 + MAX_FDS * 4
comptime POLL_MS: Int32 = 100
comptime DONT_WAIT: Int32 = 64
comptime MSG_NOSIGNAL: Int32 = 0x4000
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


@always_inline
def _stop_ptr(state: Int) -> Pointer[Atomic[Int64], MutAnyOrigin]:
    return Pointer[Atomic[Int64], MutAnyOrigin](unsafe_from_address=state + WORD_STOP * 8)


def is_owned(state: Int, id: Int) -> Bool:
    if id >= MAX_OWNED_ID:
        return False
    var byte = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=word(state, WORD_OWNED) + id // 8)[unsafe_offset=0]
    return (byte >> UInt8(id % 8)) & 1 != 0


def set_owned(state: Int, id: Int):
    if id < MAX_OWNED_ID:
        var pointer = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=word(state, WORD_OWNED) + id // 8)
        pointer[unsafe_offset=0] = pointer[unsafe_offset=0] | (UInt8(1) << UInt8(id % 8))


def is_server_offer(state: Int, id: Int) -> Bool:
    if id == 0:
        return False
    var offers = word(state, WORD_OFFER_IDS)
    var count = min(MAX_OFFERS, word(state, WORD_OFFER_COUNT))
    if offers == 0 or count <= 0:
        return False
    for i in range(count):
        if Int(get32(offers + i * 4)) == id:
            return True
    return False


def register_server_offer(state: Int, id: Int):
    if id == 0:
        return
    var offers = word(state, WORD_OFFER_IDS)
    if offers == 0:
        return
    var count = word(state, WORD_OFFER_COUNT)
    for i in range(min(MAX_OFFERS, count)):
        if Int(get32(offers + i * 4)) == id:
            return
    if count < MAX_OFFERS:
        put32(offers + count * 4, UInt32(id))
        set_word(state, WORD_OFFER_COUNT, count + 1)
    else:
        for i in range(MAX_OFFERS - 1):
            put32(offers + i * 4, get32(offers + (i + 1) * 4))
        put32(offers + (MAX_OFFERS - 1) * 4, UInt32(id))


def lock(state: Int):
    _ = external_call["pthread_mutex_lock", Int32](word(state, WORD_MUTEX))


def unlock(state: Int):
    _ = external_call["pthread_mutex_unlock", Int32](word(state, WORD_MUTEX))


def init_lib_ifaces(state: Int):
    if word(state, WORD_LIB_IFACES) == 0:
        set_word(state, WORD_LIB_IFACES, external_call["calloc", Int](Int(LIB_IFACE_CAP), Int(8)))


def free_lib_ifaces(state: Int):
    if state == 0:
        return
    var tbl = word(state, WORD_LIB_IFACES)
    if tbl != 0:
        external_call["free", NoneType](tbl)
        set_word(state, WORD_LIB_IFACES, 0)


def register_lib_iface(state: Int, id: Int, iface: Int):
    if id <= 0:
        return
    init_lib_ifaces(state)
    set_word(state, WORD_STRICT_IFACE, 1)
    var tbl = word(state, WORD_LIB_IFACES)
    if tbl == 0:
        return
    var mask = LIB_IFACE_CAP - 1
    var idx = Int((UInt32(id) * UInt32(2654435761)) & UInt32(mask))
    for _ in range(LIB_IFACE_CAP):
        var slot = tbl + idx * 8
        var cur = Int(get32(slot))
        if cur == 0 or cur == id:
            put32(slot, UInt32(id))
            put32(slot + 4, UInt32(iface))
            return
        idx = (idx + 1) & mask


def lookup_lib_iface(state: Int, id: Int) -> Int:
    if id <= 0:
        return LIB_IFACE_UNKNOWN
    var tbl = word(state, WORD_LIB_IFACES)
    if tbl == 0:
        return LIB_IFACE_UNKNOWN
    var mask = LIB_IFACE_CAP - 1
    var idx = Int((UInt32(id) * UInt32(2654435761)) & UInt32(mask))
    for _ in range(LIB_IFACE_CAP):
        var slot = tbl + idx * 8
        var cur = Int(get32(slot))
        if cur == 0:
            return LIB_IFACE_UNKNOWN
        if cur == id:
            return Int(get32(slot + 4))
        idx = (idx + 1) & mask
    return LIB_IFACE_UNKNOWN


def unregister_lib_iface(state: Int, id: Int):
    if id <= 0:
        return
    var tbl = word(state, WORD_LIB_IFACES)
    if tbl == 0:
        return
    var mask = LIB_IFACE_CAP - 1
    var i = Int((UInt32(id) * UInt32(2654435761)) & UInt32(mask))
    var found = False
    for _ in range(LIB_IFACE_CAP):
        var slot = tbl + i * 8
        var cur = Int(get32(slot))
        if cur == 0:
            return
        if cur == id:
            found = True
            break
        i = (i + 1) & mask
    if not found:
        return
    var j = i
    while True:
        j = (j + 1) & mask
        if j == i:
            put32(tbl + i * 8, UInt32(0))
            put32(tbl + i * 8 + 4, UInt32(0))
            return
        var j_slot = tbl + j * 8
        var j_id = Int(get32(j_slot))
        if j_id == 0:
            put32(tbl + i * 8, UInt32(0))
            put32(tbl + i * 8 + 4, UInt32(0))
            return
        var k = Int((UInt32(j_id) * UInt32(2654435761)) & UInt32(mask))
        var can_move = (k <= i or k > j) if i < j else (k <= i and k > j)
        if can_move:
            put32(tbl + i * 8, UInt32(j_id))
            put32(tbl + i * 8 + 4, get32(j_slot + 4))
            i = j


@always_inline
def _lib_event_has_fd(iface: Int, opcode: Int) -> Bool:
    return (
        (iface == LIB_IFACE_DMABUF_FEEDBACK and opcode == 1)
        or (iface == LIB_IFACE_BUFFER_RELEASE and opcode == 0)
        or (iface == LIB_IFACE_IMAGE_DESC_INFO and opcode == 1)
        or (iface == LIB_IFACE_DRM_LEASE_DEV and opcode == 0)
        or (iface == LIB_IFACE_DRM_LEASE and opcode == 0)
    )


def _wire_name_eq(addr: Int, name_len: Int, target: String) -> Bool:
    if name_len != target.byte_length():
        return False
    var ptr = Pointer[UInt8, ImmutAnyOrigin](unsafe_from_address=addr)
    var tbytes = target.as_bytes()
    for i in range(name_len):
        if ptr[unsafe_offset=i] != tbytes[i]:
            return False
    return True


def _track_outgoing_request(state: Int, msg: Int, obj: Int, op: Int, size: Int):
    if obj == WL_DISPLAY_ID:
        if op == 0 and size >= 12:
            register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_NO_FD)
        elif op == 1 and size >= 12:
            register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_REGISTRY)
        return
    var iface = lookup_lib_iface(state, obj)
    if iface == LIB_IFACE_UNKNOWN:
        return
    if iface == LIB_IFACE_REGISTRY and op == 0 and size >= 24:
        var slen = Int(get32(msg + 12))
        var padded = (slen + 3) & ~3
        if slen > 0 and size >= 24 + padded:
            var name_len = slen - 1
            var name_addr = msg + 16
            var new_id = Int(get32(msg + 20 + padded))
            var kind = LIB_IFACE_NO_FD
            if _wire_name_eq(name_addr, name_len, "zwp_linux_dmabuf_v1"):
                kind = LIB_IFACE_DMABUF
            elif _wire_name_eq(name_addr, name_len, "zwp_linux_explicit_synchronization_v1"):
                kind = LIB_IFACE_EXPLICIT_SYNC
            elif _wire_name_eq(name_addr, name_len, "wp_color_manager_v1"):
                kind = LIB_IFACE_COLOR_MANAGER
            elif _wire_name_eq(name_addr, name_len, "wp_drm_lease_device_v1"):
                kind = LIB_IFACE_DRM_LEASE_DEV
            register_lib_iface(state, new_id, kind)
    elif iface == LIB_IFACE_DMABUF and size >= 12:
        if op == 1:
            register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_NO_FD)
        elif op == 2 or op == 3:
            register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_DMABUF_FEEDBACK)
    elif iface == LIB_IFACE_EXPLICIT_SYNC and op == 1 and size >= 12:
        register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_SURFACE_SYNC)
    elif iface == LIB_IFACE_SURFACE_SYNC and op == 2 and size >= 12:
        register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_BUFFER_RELEASE)
    elif iface == LIB_IFACE_COLOR_MANAGER and size >= 12:
        if op == 1 or op == 2:
            register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_NO_FD)
        elif op == 3:
            register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_COLOR_SURFACE_FEEDBACK)
        elif op == 4 or op == 5:
            register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_IMAGE_DESC_CREATOR)
        elif op == 6 or op == 7:
            register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_IMAGE_DESC)
    elif iface == LIB_IFACE_COLOR_SURFACE_FEEDBACK and (op == 1 or op == 2) and size >= 12:
        register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_IMAGE_DESC)
    elif iface == LIB_IFACE_IMAGE_DESC_CREATOR and op == 0 and size >= 12:
        register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_IMAGE_DESC)
    elif iface == LIB_IFACE_IMAGE_DESC and op == 1 and size >= 12:
        register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_IMAGE_DESC_INFO)
    elif iface == LIB_IFACE_DRM_LEASE_DEV and op == 0 and size >= 12:
        register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_DRM_LEASE_REQ)
    elif iface == LIB_IFACE_DRM_LEASE_REQ and op == 1 and size >= 12:
        register_lib_iface(state, Int(get32(msg + 8)), LIB_IFACE_DRM_LEASE)


def receive_some(fd: Int, buffer: Int, capacity: Int, fds: Int, mut fd_count: Int, scratch: Int) -> Int:
    """recvmsg without waiting using pre-allocated `scratch`. Passed descriptors join the end of `fds`."""
    var iov = scratch
    var message = iov + 16
    var control = message + 56
    _ = external_call["memset", Int](iov, Int32(0), IO_SCRATCH_BYTES)
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
    return count


def send_some(fd: Int, buffer: Int, size: Int, fds: Int, fd_count: Int, scratch: Int) -> Bool:
    """sendmsg all of `size` using pre-allocated `scratch`, with the descriptors on its first byte."""
    var iov = scratch
    var message = iov + 16
    var control = message + 56
    _ = external_call["memset", Int](iov, Int32(0), IO_SCRATCH_BYTES)
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
        var count = Int(external_call["sendmsg", Int64](Int32(fd), raw(message), MSG_NOSIGNAL))
        if count < 0:
            if last_errno() == EINTR:
                continue
            ok = False
        elif count == 0:
            ok = False
        else:
            sent += count
    return ok


def _count_remaining_game_fds(state: Int, buffer: Int, start_offset: Int, used: Int) -> Int:
    """Counts FD-carrying game events (`wl_keyboard.keymap`, `wl_data_source.send`) from `start_offset` onward."""
    var kb_id = word(state, WORD_KEYBOARD)
    var ds_id = word(state, WORD_DATA_SOURCE)
    if kb_id == 0 and ds_id == 0:
        return 0
    var scan = start_offset
    var count = 0
    while used - scan >= 8:
        var obj = Int(get32(buffer + scan))
        var hdr = Int(get32(buffer + scan + 4))
        var op = hdr & 0xFFFF
        var sz = hdr >> 16
        if is_owned(state, obj) and (
            (kb_id != 0 and obj == kb_id and op == 0)
            or (ds_id != 0 and obj == ds_id and op == 1)
        ):
            count += 1
        if sz < 8 or used - scan < sz:
            break
        scan += sz
    if used - scan > 0 and used - scan < 8:
        if used - scan < 4:
            count += 1
        else:
            var partial_obj = Int(get32(buffer + scan))
            if (kb_id != 0 and partial_obj == kb_id) or (ds_id != 0 and partial_obj == ds_id):
                if used - scan >= 6:
                    var op_ptr = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=buffer + scan + 4)
                    var partial_op = Int(op_ptr[unsafe_offset=0]) | (Int(op_ptr[unsafe_offset=1]) << 8)
                    if (kb_id != 0 and partial_obj == kb_id and partial_op == 0) or (
                        ds_id != 0 and partial_obj == ds_id and partial_op == 1
                    ):
                        count += 1
                else:
                    count += 1
    return count


def _in_new_offers(new_offers: Array[Int, MAX_OFFERS], count: Int, obj: Int) -> Bool:
    if obj == 0:
        return False
    for i in range(count):
        if new_offers[i] == obj:
            return True
    return False


def _has_library_msg_after_game_fd(state: Int, buffer: Int, start_offset: Int, used: Int) -> Bool:
    """Legacy fallback (WORD_STRICT_IFACE == 0): returns True if any `for_library` message appears after a game FD event."""
    var kb_id = word(state, WORD_KEYBOARD)
    var ds_id = word(state, WORD_DATA_SOURCE)
    if kb_id == 0 and ds_id == 0:
        return False
    var data_dev_id = word(state, WORD_DATA_DEVICE)
    var new_offers = Array[Int, MAX_OFFERS](fill=0)
    var new_offer_count = 0
    var scan = start_offset
    var seen_game_fd = False
    while used - scan >= 8:
        var obj = Int(get32(buffer + scan))
        var hdr = Int(get32(buffer + scan + 4))
        var op = hdr & 0xFFFF
        var sz = hdr >> 16
        if sz < 8 or used - scan < sz:
            if seen_game_fd:
                var partial_for_game = is_owned(state, obj) or is_server_offer(state, obj) or _in_new_offers(new_offers, new_offer_count, obj)
                if not partial_for_game:
                    return True
            break
        if data_dev_id != 0 and obj == data_dev_id and op == 0 and sz >= 12:
            var oid = Int(get32(buffer + scan + 8))
            if oid != 0 and new_offer_count < MAX_OFFERS and not _in_new_offers(new_offers, new_offer_count, oid):
                new_offers[new_offer_count] = oid
                new_offer_count += 1
        var for_game = is_owned(state, obj) or is_server_offer(state, obj) or _in_new_offers(new_offers, new_offer_count, obj)
        var for_library = not for_game
        if obj == WL_DISPLAY_ID:
            for_library = op == OPCODE_ERROR or not (
                op == OPCODE_DELETE_ID and sz >= 12 and is_owned(state, Int(get32(buffer + scan + 8)))
            )
        if for_library and seen_game_fd:
            return True
        if for_game and (
            (kb_id != 0 and obj == kb_id and op == 0)
            or (ds_id != 0 and obj == ds_id and op == 1)
        ):
            seen_game_fd = True
        scan += sz
    if seen_game_fd and used - scan >= 4 and used - scan < 8:
        var partial_obj = Int(get32(buffer + scan))
        var partial_for_game = is_owned(state, partial_obj) or is_server_offer(state, partial_obj) or _in_new_offers(new_offers, new_offer_count, partial_obj)
        if not partial_for_game:
            return True
    return False


def take_incoming(state: Int) -> Bool:
    """Compositor to here: sort every whole event. True if the game got something."""
    var used = word(state, WORD_IN_USED)
    var buffer = word(state, WORD_IN)
    var fds = word(state, WORD_IN_FDS)
    var fd_count = word(state, WORD_IN_FD_COUNT)
    var io_scratch = word(state, WORD_IO_SCRATCH)
    var count = receive_some(word(state, WORD_REAL), buffer + used, STREAM_BYTES - used, fds, fd_count, io_scratch)
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
        var for_game = is_owned(state, object) or is_server_offer(state, object)
        var for_library = not for_game
        var dead_lib_id = 0
        if object == WL_DISPLAY_ID:
            for_game = opcode == OPCODE_ERROR
            if opcode == OPCODE_ERROR:
                for_library = True
            elif opcode == OPCODE_DELETE_ID and size >= 12:
                var dead_id = Int(get32(buffer + offset + 8))
                var owned_dead = is_owned(state, dead_id)
                for_library = not owned_dead
                if not owned_dead:
                    dead_lib_id = dead_id
            else:
                for_library = True
        if for_game:
            var data_dev_id = word(state, WORD_DATA_DEVICE)
            if data_dev_id != 0 and object == data_dev_id and opcode == 0 and size >= 12:
                register_server_offer(state, Int(get32(buffer + offset + 8)))
            var game_used = word(state, WORD_GAME_USED)
            if game_used + size <= GAME_BYTES:
                _ = external_call["memcpy", Int](word(state, WORD_GAME) + game_used, buffer + offset, size)
                set_word(state, WORD_GAME_USED, game_used + size)
                game_got = True
            var kb_id = word(state, WORD_KEYBOARD)
            var ds_id = word(state, WORD_DATA_SOURCE)
            var carries_game_fd = (
                (kb_id != 0 and object == kb_id and opcode == 0)
                or (ds_id != 0 and object == ds_id and opcode == 1)
            )
            if carries_game_fd and fd_count > 0:
                var carried_fd = Int(get64(fds))
                fd_count -= 1
                _ = external_call["memmove", Int](fds, fds + 8, fd_count * 8)
                var g_fds = word(state, WORD_GAME_FDS)
                var g_cnt = word(state, WORD_GAME_FD_COUNT)
                if g_fds != 0 and g_cnt < MAX_FDS:
                    put64(g_fds + g_cnt * 8, carried_fd)
                    set_word(state, WORD_GAME_FD_COUNT, g_cnt + 1)
                else:
                    close_fd(carried_fd)
        if for_library:
            var lib_fd_count = 0
            if word(state, WORD_STRICT_IFACE) != 0:
                var iface = lookup_lib_iface(state, object)
                if fd_count > 0 and _lib_event_has_fd(iface, opcode):
                    lib_fd_count = 1
            else:
                var rem_game_fds = _count_remaining_game_fds(state, buffer, offset, used)
                if rem_game_fds == 0:
                    lib_fd_count = fd_count
                elif fd_count > rem_game_fds:
                    if not _has_library_msg_after_game_fd(state, buffer, offset + size, used):
                        lib_fd_count = fd_count - rem_game_fds
            if not send_some(word(state, WORD_LIBRARY), buffer + offset, size, fds, lib_fd_count, io_scratch):
                set_word(state, WORD_FAILED, 1)
            if lib_fd_count > 0:
                for index in range(lib_fd_count):
                    close_fd(get64(fds + index * 8))  # the kernel holds copies now
                fd_count -= lib_fd_count
                if fd_count > 0:
                    _ = external_call["memmove", Int](fds, fds + lib_fd_count * 8, fd_count * 8)
            if dead_lib_id != 0:
                unregister_lib_iface(state, dead_lib_id)
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
    var io_scratch = word(state, WORD_IO_SCRATCH)
    while True:
        var used = word(state, WORD_OUT_USED)
        var fd_count = word(state, WORD_OUT_FD_COUNT)
        var count = receive_some(word(state, WORD_LIBRARY), buffer + used, STREAM_BYTES - used, fds, fd_count, io_scratch)
        set_word(state, WORD_OUT_FD_COUNT, fd_count)
        if count <= 0:
            return
        used += count
        var whole = 0
        while used - whole >= 8:
            var obj = Int(get32(buffer + whole))
            var hdr = Int(get32(buffer + whole + 4))
            var op = hdr & 0xFFFF
            var size = hdr >> 16
            if size < 8 or used - whole < size:
                break
            _track_outgoing_request(state, buffer + whole, obj, op, size)
            whole += size
        if whole > 0:
            if not send_some(word(state, WORD_REAL), buffer, whole, fds, fd_count, io_scratch):
                set_word(state, WORD_FAILED, 1)
            for index in range(fd_count):
                close_fd(get64(fds + index * 8))
            set_word(state, WORD_OUT_FD_COUNT, 0)
            _ = external_call["memmove", Int](buffer, buffer + whole, used - whole)
        set_word(state, WORD_OUT_USED, used - whole)


def bridge_thread(state: Int) abi("C") -> Int:
    var descriptors = word(state, WORD_POLL_SCRATCH)
    var notify_one = descriptors + 16
    put64(notify_one, 1)
    put32(descriptors, UInt32(word(state, WORD_REAL)))
    put32(descriptors + 4, POLL_IN)
    put32(descriptors + 8, UInt32(word(state, WORD_LIBRARY)))
    put32(descriptors + 12, POLL_IN)
    while _stop_ptr(state)[unsafe_offset=0].load() == 0:
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
            _ = external_call["write", Int64](Int32(word(state, WORD_NOTIFY)), raw(notify_one), Int64(8))
    return 0


struct WaylandBridge(Movable):
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

    def __moveinit__(out self, deinit existing: Self):
        self.state = existing.state
        self.library = existing.library
        self.display = existing.display
        self.surface_proxy = existing.surface_proxy
        self.compositor_proxy = existing.compositor_proxy
        self.fn_flush = existing.fn_flush
        self.fn_proxy_create = existing.fn_proxy_create
        self.fn_proxy_id = existing.fn_proxy_id
        self.fn_constructor = existing.fn_constructor
        self.fn_disconnect = existing.fn_disconnect
        self.fn_proxy_destroy = existing.fn_proxy_destroy
        self.callback_interface = existing.callback_interface
        self.compositor_interface = existing.compositor_interface
        self.surface_interface = existing.surface_interface
        self.error = existing.error^
        existing.state = 0
        existing.library = 0
        existing.display = 0
        existing.surface_proxy = 0
        existing.compositor_proxy = 0
        existing.fn_flush = 0
        existing.fn_proxy_create = 0
        existing.fn_proxy_id = 0
        existing.fn_constructor = 0
        existing.fn_disconnect = 0
        existing.fn_proxy_destroy = 0
        existing.callback_interface = 0
        existing.compositor_interface = 0
        existing.surface_interface = 0

    def __deinit__(deinit self):
        self.stop()

    def active(self) -> Bool:
        return self.state != 0

    def symbol(mut self, var name: String) -> Int:
        var address = dl_sym(self.library, name)
        if address == 0:
            self.error = "libwayland-client has no " + name
        return address

    def start(mut self, real_fd: Int) -> Bool:
        """Take over a connected socket on which nothing has been said yet."""
        var library_name = "libwayland-client.so.0"
        self.library = dl_open(library_name)
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
            self.stop()
            return False
        var pair = external_call["malloc", Int](8)
        if external_call["socketpair", Int32](Int32(1), Int32(1), Int32(0), pair) != 0:
            external_call["free", NoneType](pair)
            self.error = "socketpair failed"
            self.stop()
            return False
        var ours = Int(get32(pair))
        var theirs = Int(get32(pair + 4))
        external_call["free", NoneType](pair)
        self.display = self.connect_display(connect, theirs)
        if self.display == 0:
            close_fd(ours)
            close_fd(theirs)
            self.error = "wl_display_connect_to_fd failed"
            self.stop()
            return False
        var state = external_call["calloc", Int](Int(STATE_WORDS), Int(8))
        set_word(state, WORD_REAL, real_fd)
        set_word(state, WORD_LIBRARY, ours)
        set_word(state, WORD_NOTIFY, Int(external_call["eventfd", Int32](UInt32(0), Int32(0x80800))))
        set_word(state, WORD_IN, external_call["malloc", Int](STREAM_BYTES))
        set_word(state, WORD_OUT, external_call["malloc", Int](STREAM_BYTES))
        set_word(state, WORD_GAME, external_call["malloc", Int](GAME_BYTES))
        set_word(state, WORD_OWNED, external_call["calloc", Int](Int(MAX_OWNED_ID // 8), Int(1)))
        var mutex_ptr = external_call["calloc", Int](Int(1), Int(64))
        _ = external_call["pthread_mutex_init", Int32](mutex_ptr, Int(0))
        set_word(state, WORD_MUTEX, mutex_ptr)
        set_word(state, WORD_IN_FDS, external_call["malloc", Int](MAX_FDS * 8))
        set_word(state, WORD_OUT_FDS, external_call["malloc", Int](MAX_FDS * 8))
        set_word(state, WORD_GAME_FDS, external_call["malloc", Int](MAX_FDS * 8))
        set_word(state, WORD_OFFER_IDS, external_call["calloc", Int](Int(MAX_OFFERS), Int(4)))
        set_word(state, WORD_IO_SCRATCH, external_call["calloc", Int](Int(1), IO_SCRATCH_BYTES))
        set_word(state, WORD_POLL_SCRATCH, external_call["calloc", Int](Int(1), Int(32)))
        set_word(state, WORD_WAIT_SCRATCH, external_call["calloc", Int](Int(1), Int(24)))
        init_lib_ifaces(state)
        set_word(state, WORD_STRICT_IFACE, 1)
        set_owned(state, WL_DISPLAY_ID)
        var thread = 0
        if external_call["pthread_create", Int32](Pointer(to=thread), Int(0), bridge_thread, state) != 0:
            self.error = "the bridge thread did not start"
            if self.display != 0 and self.fn_disconnect != 0:
                _ = call1(self.fn_disconnect, self.display)
            self.display = 0
            close_fd(ours)
            close_fd(word(state, WORD_NOTIFY))
            _ = external_call["pthread_mutex_destroy", Int32](mutex_ptr)
            external_call["free", NoneType](mutex_ptr)
            external_call["free", NoneType](word(state, WORD_IN))
            external_call["free", NoneType](word(state, WORD_OUT))
            external_call["free", NoneType](word(state, WORD_GAME))
            external_call["free", NoneType](word(state, WORD_OWNED))
            external_call["free", NoneType](word(state, WORD_IN_FDS))
            external_call["free", NoneType](word(state, WORD_OUT_FDS))
            external_call["free", NoneType](word(state, WORD_GAME_FDS))
            external_call["free", NoneType](word(state, WORD_OFFER_IDS))
            external_call["free", NoneType](word(state, WORD_IO_SCRATCH))
            external_call["free", NoneType](word(state, WORD_POLL_SCRATCH))
            external_call["free", NoneType](word(state, WORD_WAIT_SCRATCH))
            free_lib_ifaces(state)
            external_call["free", NoneType](state)
            self.stop()
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
        lock(self.state)
        set_owned(self.state, id)
        unlock(self.state)
        return id

    def reserve_compositor(mut self) -> Int:
        """The id to bind wl_compositor to, with a proxy that can make surfaces."""
        self.push_out()
        self.compositor_proxy = self.create_proxy(self.compositor_interface)
        var id = self.proxy_id(self.compositor_proxy)
        lock(self.state)
        set_owned(self.state, id)
        unlock(self.state)
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
        lock(self.state)
        set_owned(self.state, id)
        unlock(self.state)
        self.push_out()
        return id

    def note_keyboard(self, id: Int):
        if self.state == 0:
            return
        lock(self.state)
        set_word(self.state, WORD_KEYBOARD, id)
        unlock(self.state)

    def note_data_device(self, id: Int):
        if self.state == 0:
            return
        lock(self.state)
        set_word(self.state, WORD_DATA_DEVICE, id)
        unlock(self.state)

    def note_data_source(self, id: Int):
        if self.state == 0:
            return
        lock(self.state)
        set_word(self.state, WORD_DATA_SOURCE, id)
        unlock(self.state)

    def write_all(self, address: Int, size: Int) -> Bool:
        self.push_out()
        lock(self.state)
        var ok = send_some(word(self.state, WORD_REAL), address, size, 0, 0, word(self.state, WORD_IO_SCRATCH))
        unlock(self.state)
        return ok

    def write_with_fd(self, address: Int, size: Int, fd: Int) -> Bool:
        self.push_out()
        var fds = word(self.state, WORD_WAIT_SCRATCH) + 16
        put64(fds, fd)
        lock(self.state)
        var ok = send_some(word(self.state, WORD_REAL), address, size, fds, 1, word(self.state, WORD_IO_SCRATCH))
        unlock(self.state)
        return ok

    def read_into(self, mut incoming: List[UInt8]):
        var unused_fds = List[Int]()
        self.read_into(incoming, unused_fds)
        for fd in unused_fds:
            close_fd(fd)

    def read_into(self, mut incoming: List[UInt8], mut incoming_fds: List[Int]):
        lock(self.state)
        var used = word(self.state, WORD_GAME_USED)
        var pointer = Pointer[UInt8, origin=MutAnyOrigin](unsafe_from_address=word(self.state, WORD_GAME))
        for index in range(used):
            incoming.append(pointer[unsafe_offset=index])
        set_word(self.state, WORD_GAME_USED, 0)
        var g_fds = word(self.state, WORD_GAME_FDS)
        var g_cnt = word(self.state, WORD_GAME_FD_COUNT)
        if g_fds != 0 and g_cnt > 0:
            for i in range(g_cnt):
                incoming_fds.append(Int(get64(g_fds + i * 8)))
            set_word(state=self.state, index=WORD_GAME_FD_COUNT, value=0)
        unlock(self.state)

    def wait(self, timeout_ms: Int) -> Bool:
        """True once the game has events to read."""
        var descriptor = word(self.state, WORD_WAIT_SCRATCH)
        put32(descriptor, UInt32(word(self.state, WORD_NOTIFY)))
        put32(descriptor + 4, POLL_IN)
        var ready = external_call["poll", Int32](raw(descriptor), UInt64(1), Int32(timeout_ms)) > 0
        if ready:
            _ = external_call["read", Int64](Int32(word(self.state, WORD_NOTIFY)), raw(descriptor + 8), Int64(8))
        return ready

    def failed(self) -> Bool:
        return word(self.state, WORD_FAILED) != 0

    def stop(mut self):
        """Stop the thread and let go of the library's connection. The socket stays ours."""
        if self.state != 0:
            self.push_out()
            _stop_ptr(self.state)[unsafe_offset=0].store(1)
            _ = external_call["pthread_join", Int32](word(self.state, WORD_THREAD), Int(0))
            if self.display != 0 and self.fn_disconnect != 0:
                _ = call1(self.fn_disconnect, self.display)
            self.display = 0
            self.surface_proxy = 0
            self.compositor_proxy = 0
            close_fd(word(self.state, WORD_LIBRARY))
            close_fd(word(self.state, WORD_NOTIFY))
            var in_fds = word(self.state, WORD_IN_FDS)
            var in_fd_count = word(self.state, WORD_IN_FD_COUNT)
            for idx in range(in_fd_count):
                close_fd(get64(in_fds + idx * 8))
            var out_fds = word(self.state, WORD_OUT_FDS)
            var out_fd_count = word(self.state, WORD_OUT_FD_COUNT)
            for idx in range(out_fd_count):
                close_fd(get64(out_fds + idx * 8))
            var g_fds = word(self.state, WORD_GAME_FDS)
            var g_fd_count = word(self.state, WORD_GAME_FD_COUNT)
            for idx in range(g_fd_count):
                close_fd(get64(g_fds + idx * 8))
            var mutex_ptr = word(self.state, WORD_MUTEX)
            if mutex_ptr != 0:
                _ = external_call["pthread_mutex_destroy", Int32](mutex_ptr)
                external_call["free", NoneType](mutex_ptr)
            external_call["free", NoneType](word(self.state, WORD_IN))
            external_call["free", NoneType](word(self.state, WORD_OUT))
            external_call["free", NoneType](word(self.state, WORD_GAME))
            external_call["free", NoneType](word(self.state, WORD_OWNED))
            external_call["free", NoneType](in_fds)
            external_call["free", NoneType](out_fds)
            external_call["free", NoneType](g_fds)
            external_call["free", NoneType](word(self.state, WORD_OFFER_IDS))
            external_call["free", NoneType](word(self.state, WORD_IO_SCRATCH))
            external_call["free", NoneType](word(self.state, WORD_POLL_SCRATCH))
            external_call["free", NoneType](word(self.state, WORD_WAIT_SCRATCH))
            free_lib_ifaces(self.state)
            external_call["free", NoneType](self.state)
            self.state = 0
        elif self.display != 0 and self.fn_disconnect != 0:
            _ = call1(self.fn_disconnect, self.display)
            self.display = 0
        if self.library != 0:
            _ = dl_close(self.library)
            self.library = 0
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

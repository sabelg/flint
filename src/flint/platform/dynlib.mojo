"""dlopen/dlsym with the stdlib's own declarations.

A module may hold only one declaration per C symbol. The stdlib declares `dlopen` and
`dlsym` in std.ffi (its DLHandle, also used by std.python), so any code that also links
those (a Python extension built from Flint, for one) fails to build if Flint declares
them with different types. These helpers use exactly the stdlib's shapes and hand back
plain addresses, which is what the rest of Flint stores.
"""
from std.ffi import external_call

comptime RTLD_LAZY: Int32 = 1
comptime RTLD_NOW: Int32 = 2
comptime RTLD_LOCAL: Int32 = 0
comptime RTLD_GLOBAL: Int32 = 0x100

comptime _Raw = Pointer[NoneType, MutUntrackedOrigin]
comptime _Handle = Optional[_Raw]


def _handle(address: Int) -> _Handle:
    if address == 0:
        return None
    return _Raw(unsafe_from_address=address)


def _address(handle: _Handle) -> Int:
    return Int(handle.value()) if handle else 0


def dl_open(var name: String, flags: Int32 = RTLD_NOW) -> Int:
    """The library's handle, or 0 if it can't be loaded."""
    var c_name = name.as_c_string_span()
    return _address(external_call["dlopen", _Handle](_handle(Int(c_name.unsafe_ptr())), flags))


def dl_sym(library: Int, var name: String) -> Int:
    """The symbol's address in `library`, or 0."""
    if library == 0:
        return 0
    return _address(external_call["dlsym", _Handle](_handle(library), name.as_c_string_span()))

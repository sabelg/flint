# Flint Engine — Agent & Developer Guidelines (`AGENTS.md`)

## 1. Engine Mission & Invariants
- **100% Pure Mojo 1.2**: Zero C/C++ wrapper libraries (no SDL, GLFW, OpenGL, shaderc, miniaudio, freetype, or Box2D/Jolt). Talk directly to OS protocols/syscalls and Vulkan/DRM drivers.
- **Strict Zero-Game-Leak Gate**: `src/flint/` must never import `gameplay.*` or `world.*` or hardcode game-specific rules. Run `bash tools/flint.sh gate` before committing.
- **Mojo 1.2 Syntax Rules**:
  - Use `def` for all functions/methods (`fn` was removed); place `raises` before `->`.
  - Use `comptime` (`alias` and `@parameter` were removed).
  - Declare variables with `var` (`let` was removed).
  - Argument conventions: immutable borrow default, `mut`, `var`, `out self` in `__init__`.
  - Structs: `@fieldwise_init` with explicit `Copyable`, `Movable`; transfer non-`ImplicitlyCopyable` values with `^`.
  - Stdlib imports use `std.` prefix (`from std.testing import assert_equal, assert_true`).
  - UTF-8 strings: use `s.byte_length()` and `s[byte=i]`.
- **Parallel Job Lifetime Safety**: Always dispatch `JobPool` work via `flint.core.ecs.run_parallel_ctx` and call `.keep_alive()` on any `SimdColumn` passed by raw `base_addr()` right after the barrier so Mojo's ASAP destructor cannot reclaim buffers early.

## 2. Common Commands
```bash
pixi install                              # Install Mojo 1.2 toolchain on a new machine
bash tools/flint.sh gate                  # Sync check + compile + 5 test suites + headless PNG
bash tools/flint.sh play                  # Run interactive minimal_spark 2D+3D showcase
bash tools/flint.sh sync /path/to/Deepsong # Harvest latest game-agnostic pure-Mojo libraries
bash tools/flint.sh new ../my-game        # Scaffold a standalone game repo
```

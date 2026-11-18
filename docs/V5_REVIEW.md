# Flint v5 review (2026-11-18): uncommitted work in ~/flint

Scope: about 40k new lines in 18 areas plus about 5k changed lines, all uncommitted and still being
written. Three reviewers read it, and every v5 suite was built and run. All pass, and the full
gate passes. I re-checked the starred (★) items myself.

## Fix first (memory safety and security)
1. ★ **`assets/flpk_v3.mojo` crashes on a hostile file.** `_validate_mapped_header` (~:876) never
   range-checks `toc_offset` or `uuid_table_offset`, and `off + csz > size` can overflow.
   A crafted `.flpk` segfaults the reader (probe: /tmp/v5rev/probe/flpk_probe.mojo + evil.flpk).
   CRC64 doesn't help, because an attacker can recompute it.
2. ★ **`ecs/world.mojo` `restore_blob` maps components wrongly.** It rebuilds archetypes from raw
   type-id masks (~:1998) and never remaps them by the stored hashes. With a different
   registration order, components swap type, and a missing type means a write to address 0.
3. **ECS snapshot and checksum memcpy heap-owning components.** Strings and Lists in components
   without custom serialization hooks get their raw pointer bytes copied, which gives a
   use-after-free and double free in PIE enter/exit.
4. ★ **`app/engine_app.mojo` `insert_pod_raw` (~:129) overflows the heap.** When a reloaded
   plugin re-inserts a bigger resource, the copy is skipped but the old, smaller pointer is
   returned, and `byte_size` is never updated.
5. **`assets/server.mojo` `_unreachable_empty_payload` (~:360) returns uninitialized `T`.** The
   destructor then frees garbage. LRU eviction never frees payloads.
6. **The MCP v2 socket `/tmp/flint_studio_mcp_v2.sock` has no auth or permissions.** Its
   `output_path`/`output_flpk` arguments allow arbitrary file writes. Sends truncate on EAGAIN,
   a bad header wedges the stream, and receive buffers are unbounded.
7. ★ **`net/rollback.mojo` trusts the `sender_id` byte in the packet (~:584).** It never checks
   the socket's source address against `peer_endpoints`, so any host can inject inputs or
   forge checksums.
8. **The `audio/server.mojo` RT thread races with the caller.** It holds `Pointer(to=self)` on a
   Movable struct, and handle allocation, asset registration and bus Lists are mutated while the
   RT thread iterates them.
9. **The HDR scRGB swapchain fallback copies 4-byte PQ words into an 8-byte format.**

## Claims that don't hold
- **ORCA "exact 0-overlap":** 88–126 overlapping pairs once agents meet. The test runs 120 ticks,
  before any agent interacts.
- **Navmesh:** carving takes 27 ms (claimed <0.25 ms, and the test discards its timing). ★ "Binary-heap A*"
  is a linear-scan list. Paths zig-zag (91 m vs about 80 m optimal) and cut the agent-radius
  margin. Regions past 512 silently get no polygons.
- **ECS default command order** depends on thread scheduling despite its docstring. The test only
  passes because every command sets `order_key`.
- **6-DOF joint:** applied outside the solver's substeps with the full dt, so it acts a frame late.
- **material_graph:** a screen-space preview kernel that no renderer dispatches. Binding 4 uses
  stride 32 while the renderer packs 24 words.
- **4-cascade VSM:** only cascade 0 is ever used for shading.

## Structure
- Mostly islands. scene, net, nav, ai, input, audio and camera are used only by their own tests
  and labs. Studio doesn't use `ui/context`. No real `PhysicsWorld` implements `RollbackWorld`.
- `flint.ecs` duplicates `core/ecs`, with different handles and blob formats. Existing examples
  still use `core/ecs`.
- AGENTS.md rules: parallel code calls `run_jobs`/`run_affine` directly, not `run_parallel_ctx`.
  `swapchain_surface.mojo` dlopens `libxcb.so.1`. `studio_app.mojo` hardcodes supercar and
  hovercar presets (borderline game content).
- Flaky test: `elapsed_ms < 0.1` (test_flint_v5_audio_ui_input.mojo:674). The hot-reload test
  hardcodes `/home/user/flint`.

## Genuinely strong
- **Networking:** GGPO-style rollback over a deterministic lossy network simulator (4 peers,
  10k ticks, more than 100 rollbacks, hash-checked), and zero-allocation `recvmmsg`/`sendmmsg` UDP.
- **Audio:** real measured mastering (LUFS ±0.08, true-peak limiting, ITD) and a QOA codec
  from scratch.
- **Engine core:** the schedule compiler (transitive reduction plus relay splitting), real `.so`
  hot reload, the flexbox/dock solver with a fuzz test, and a pure-Mojo SHA-256 and mmap archive.
- **Physics:** a character controller with behavioral tests.
- **Studio:** renders on the GPU with VisBuffer picking and bit-exact PIE restore.

---
**Update 2026-11-18 09:35:** the 9 "fix first" items are fixed. One new crash and two minor items are open; see [V5_RECHECK.md](V5_RECHECK.md).

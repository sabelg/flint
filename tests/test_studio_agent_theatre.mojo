"""End-to-End Deterministic Visual Agent Theatre & Gallery Suite (`tests/test_studio_agent_theatre.mojo`).

Tests the fully wired `FlintStudioState` Phase 3 cutover end-to-end (pure Mojo 1.2, Tier-0 headless):
  1. Headless ACP (`start_acp_headless()` + `feed_bytes()`) + `ingest_mcp_events()` populating
     `agent_bus` + `agent_panel`, 3D Theatre non-zero `DebugLineBatch` & `UiDrawList` overlays
     when active and exact `0` overlays when idle.
  2. Approval gate (`APPROVAL_EDITS` / `APPROVAL_STEP`): mutating MCP call queues `PendingApproval`
     + 3D ghost; panel Reject resolves cleanly and in-viewport `TheatrePillHit` (`HIT_KIND_APPROVE`)
     click commits the transform and clears the queue.
  3. Follow-Agent Camera smooth convergence + automatic yield on user camera orbit + resume.
  4. Non-destructive Time-Travel scrub (`step 0` -> exact `post_hash`, `[Live]` -> live hash)
     AND 1-click Revert Step & Revert Turn restoring exact `studio.state_hash()`.
  5. Agent file proposal in `CodeDrawer` -> Accept hunk patches source + mock compiler output
     populates `code_drawer.problems` (`ProblemStore`).
  6. All 6 `configure_gallery_shot(0..5)` render deterministically under `UI_ANIM_INSTANT`
     (two renders produce identical prim counts and bit-identical pixel hashes).
"""
from std.collections import List
from std.math import abs
from std.testing import assert_equal, assert_false, assert_true

from flint.core.math import Quat, Vec3
from flint.core.serial import fnv64
from flint.devtools.acp_client import ACP_READY
from flint.devtools.job_runner import JOB_KIND_BUILD_SO
from flint.devtools.problems import DIAG_SRC_BUILD
from flint.platform.input import InputState
from flint.render.render_world import (
    DebugLineBatch,
    RENDER_TIER_0_HEADLESS_SIMD,
    ScalableRenderer,
)
from flint.studio.agent_bridge import BridgedMcpEvent
from flint.studio.agent_bus import (
    APPROVAL_EDITS,
    APPROVAL_SRC_MCP_TOOL,
    APPROVAL_STEP,
    AgentGhostPreview,
    GHOST_TRANSFORM,
    PendingApproval,
    ROLE_GAMEPLAY_CODER,
    ROLE_SCENE_BUILDER,
    STEP_REVERTED,
)
from flint.studio.agent_panel import (
    AgentPanelAction,
    PANEL_ACT_REJECT,
    PANEL_ACT_REVERT_STEP,
    PANEL_ACT_REVERT_TURN,
    PANEL_ACT_TIME_TRAVEL_EXIT,
    PANEL_ACT_TIME_TRAVEL_SCRUB,
    PANEL_ACT_TOGGLE_FOLLOW_CAMERA,
)
from flint.studio.agent_theatre_3d import (
    HIT_KIND_APPROVE,
    emit_agent_viewport_3d_overlays,
)
from flint.studio.code_drawer import CODE_MODE_DIFF_UNIFIED
from flint.studio.studio_app import (
    FlintStudioState,
    populate_default_studio_scene,
)
from flint.studio.viewport_hud import WORKSPACE_AGENT_STUDIO
from flint.ui.animator import UI_ANIM_INSTANT
from flint.ui.draw_list import (
    PRIM_ARC,
    UI_FLAG_DASHED_FLOW,
    UI_FLAG_GLASS_SCRIM,
)


def _str_bytes(s: String) -> List[UInt8]:
    var sb = s.as_bytes()
    var out = List[UInt8](capacity=len(sb))
    for i in range(len(sb)):
        out.append(sb[i])
    return out^


def _hash_pixels(pixels: List[UInt32]) -> UInt64:
    var raw = List[UInt8](capacity=len(pixels) * 4)
    for i in range(len(pixels)):
        var p = pixels[i]
        raw.append(UInt8(p & UInt32(0xFF)))
        raw.append(UInt8((p >> 8) & UInt32(0xFF)))
        raw.append(UInt8((p >> 16) & UInt32(0xFF)))
        raw.append(UInt8((p >> 24) & UInt32(0xFF)))
    return fnv64(raw)


def _make_headless_studio(w: Int = 1280, h: Int = 720) -> FlintStudioState:
    var studio = FlintStudioState(
        headless=True,
        render_tier=RENDER_TIER_0_HEADLESS_SIMD,
        width=w,
        height=h,
        worker_threads=0,
    )
    populate_default_studio_scene(studio)
    studio.set_anim_mode(UI_ANIM_INSTANT)
    return studio^


# ---------------------------------------------------------------------------
# 1. Headless ACP + ingest_mcp_events + Idle/Active 3D & HUD Overlays
# ---------------------------------------------------------------------------

def test_e2e_acp_mcp_ingest_and_overlays() raises:
    print("  [1/6] Headless ACP + ingest_mcp_events + 3D/HUD Theatre overlays...")
    var studio = _make_headless_studio(1280, 720)
    _ = studio.apply_workspace_preset(WORKSPACE_AGENT_STUDIO, 1280.0, 720.0)

    # Idle invariant: zero active visuals -> zero 3D overlay segments & zero theatre hit rects
    assert_false(studio.agent_bus.has_active_visuals())
    var idle_lines = DebugLineBatch()
    _ = emit_agent_viewport_3d_overlays(studio.agent_bus, idle_lines, Float32(0.0))
    assert_equal(idle_lines.segment_count(), 0)
    studio.render_editor_frame(1280, 720, dt=Float32(1.0 / 60.0))
    assert_equal(len(studio._last_theatre_hits), 0)

    # Start headless ACP session, complete initialize + session/new, stream thought/plan/tool
    var aid = studio.agent_runtime.start_acp_headless(
        studio.agent_bus, ROLE_GAMEPLAY_CODER, ".", 0.0
    )
    assert_true(aid > 0)
    studio.agent_runtime.acp.feed_bytes(
        _str_bytes('{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":1}}\n')
    )
    studio.poll_agent_runtime(Float32(1.0 / 60.0))
    studio.agent_runtime.acp.feed_bytes(
        _str_bytes('{"jsonrpc":"2.0","id":2,"result":{"sessionId":"studio-e2e-1"}}\n')
    )
    studio.poll_agent_runtime(Float32(1.0 / 60.0))
    assert_equal(studio.agent_runtime.acp.state, ACP_READY)

    assert_true(
        studio.agent_runtime.send_prompt(
            studio.agent_bus, aid, "Add torch and inspect hero orb", now_sec=0.5
        )
    )
    studio.agent_runtime.acp.feed_bytes(
        _str_bytes(
            '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"studio-e2e-1","update":{"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":"Inspecting hero orb placement"}}}}\n'
            + '{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"studio-e2e-1","update":{"sessionUpdate":"plan","entries":[{"content":"Spawn key light","status":"in_progress"},{"content":"Tune material","status":"pending"}]}}}\n'
        )
    )
    studio.poll_agent_runtime(Float32(1.0 / 60.0))

    # Ingest a live MCP tool event targeting a real Studio entity
    var target_eid = studio.spawn_entity("AgentLamp", position=Vec3(0.0, 1.2, 0.0), record_undo=False)
    var evs = List[BridgedMcpEvent]()
    var ev = BridgedMcpEvent(
        seq=1,
        tool_name="entity.set_transform",
        params_json='{"entity_id":' + String(target_eid) + ',"translation":[0.0,1.2,0.0]}',
        result_json='{"ok":true}',
        elapsed_ms=4.5,
        ok=True,
        is_mutating=True,
        blocked_by_gate=False,
        agent_id=aid,
    )
    ev.affected_entities.append(target_eid)
    evs.append(ev^)
    var ingested = studio.ingest_mcp_events(evs^)
    assert_equal(ingested, 1)

    var si = studio.agent_bus.find_session_index(aid)
    assert_true(si >= 0)
    assert_true(len(studio.agent_bus.sessions[si].steps) >= 2)
    assert_true(len(studio.agent_bus.sessions[si].plan_entries) == 2)
    assert_true(studio.agent_bus.has_active_visuals())

    # Active 3D overlay batch + rendered Studio frame must emit non-zero 3D lines & GPU SDF Theatre prims
    var act_lines = DebugLineBatch()
    _ = emit_agent_viewport_3d_overlays(studio.agent_bus, act_lines, Float32(0.5))
    assert_true(act_lines.segment_count() > 0)

    studio.render_editor_frame(1280, 720, dt=Float32(1.0 / 60.0))
    assert_true(len(studio._last_theatre_hits) > 0)
    var saw_arc = False
    var saw_flow = False
    var saw_glass = False
    for pi in range(studio.ui_ctx.draw_list.count()):
        var hdr = studio.ui_ctx.draw_list.prims[pi].header
        var kind = hdr & UInt32(0xF)
        var flags = hdr >> 16
        if kind == PRIM_ARC:
            saw_arc = True
        if (flags & UI_FLAG_DASHED_FLOW) != 0:
            saw_flow = True
        if (flags & UI_FLAG_GLASS_SCRIM) != 0:
            saw_glass = True
    assert_true(saw_arc and saw_flow and saw_glass)

    studio.agent_runtime.acp.stop()
    studio.shutdown()


# ---------------------------------------------------------------------------
# 2. Approval Gate (APPROVAL_EDITS / APPROVAL_STEP) + Panel & Viewport Clicks
# ---------------------------------------------------------------------------

def test_e2e_approval_gate_and_viewport_hit() raises:
    print("  [2/6] Approval gate (APPROVAL_STEP/EDITS) + Panel Reject & Viewport Approve...")
    var studio = _make_headless_studio(1280, 720)
    _ = studio.apply_workspace_preset(WORKSPACE_AGENT_STUDIO, 1280.0, 720.0)

    var eid = studio.spawn_entity("TargetPedestal", position=Vec3(0.0, 1.0, 0.0), record_undo=False)
    var aid = studio.agent_bus.create_session(ROLE_SCENE_BUILDER, "Scene Builder", APPROVAL_STEP)
    assert_true(studio.agent_bus.should_gate_mcp_tool(aid, "entity.set_transform"))
    studio.agent_bus.set_session_policy(aid, APPROVAL_EDITS)
    assert_true(studio.agent_bus.should_gate_mcp_tool(aid, "ecs.despawn_entity"))
    assert_false(studio.agent_bus.should_gate_mcp_tool(aid, "entity.set_transform"))
    assert_false(studio.agent_bus.should_gate_mcp_tool(aid, "scene.inspect"))
    studio.agent_bus.set_session_policy(aid, APPROVAL_STEP)

    # A. Gated MCP event queues PendingApproval + ghost -> Reject via panel action
    var blocked_evs = List[BridgedMcpEvent]()
    var bev = BridgedMcpEvent(
        seq=10,
        tool_name="entity.set_transform",
        params_json='{"entity_id":' + String(eid) + ',"translation":[2.0,1.5,0.0]}',
        result_json='{"blocked":true}',
        elapsed_ms=1.0,
        ok=False,
        is_mutating=True,
        blocked_by_gate=True,
        agent_id=aid,
    )
    bev.affected_entities.append(eid)
    blocked_evs.append(bev^)
    _ = studio.ingest_mcp_events(blocked_evs^)
    assert_equal(studio.agent_bus.pending_approval_count(), 1)
    var rej_id = studio.agent_bus.pending_approvals[0].approval_id

    var rej_act = AgentPanelAction(PANEL_ACT_REJECT)
    rej_act.agent_id = aid
    rej_act.approval_id = rej_id
    studio.apply_agent_panel_action(rej_act)
    assert_equal(studio.agent_bus.pending_approval_count(), 0)
    # Rejected move must NOT have altered entity translation
    assert_true(abs(studio.entities[eid].translation.x - Float32(0.0)) < Float32(1.0e-4))

    # B. Queue transform approval with 3D ghost -> Approve by clicking in-viewport HIT_KIND_APPROVE
    var new_pos = Vec3(0.8, 1.4, 0.2)
    var sid = studio.agent_bus.begin_mcp_tool_step(
        aid,
        "entity.set_transform",
        "Move TargetPedestal",
        '{"entity_id":' + String(eid) + '}',
        entity_id=eid,
        world_pos=new_pos,
        has_world_pos=True,
        awaiting_approval=True,
        now_sec=1.0,
    )
    var pa = PendingApproval(0, aid, sid, APPROVAL_SRC_MCP_TOOL, "Move TargetPedestal", "Shift pedestal")
    pa.ghost = AgentGhostPreview(
        GHOST_TRANSFORM, aid, sid, eid, "TargetPedestal", new_pos, Quat.identity(), Vec3(0.5, 0.5, 0.5)
    )
    _ = studio.agent_bus.queue_approval(pa^)
    assert_equal(studio.agent_bus.pending_approval_count(), 1)

    # Frame 1: render to populate studio._last_theatre_hits and locate HIT_KIND_APPROVE
    studio.look_at(Vec3(-4.5, 2.8, -5.5), new_pos)
    studio.render_editor_frame(1280, 720, dt=Float32(1.0 / 60.0))
    var found_hit = False
    var hx = Float32(0.0)
    var hy = Float32(0.0)
    for hi in range(len(studio._last_theatre_hits)):
        if studio._last_theatre_hits[hi].kind == HIT_KIND_APPROVE:
            ref h = studio._last_theatre_hits[hi]
            hx = h.x + h.w * Float32(0.5)
            hy = h.y + h.h * Float32(0.5)
            found_hit = True
            break
    assert_true(found_hit)

    # Frame 2: click the exact in-viewport [✓] pill button -> resolves & applies transform!
    var clk = InputState()
    clk.begin_frame()
    clk.move_mouse(hx, hy)
    clk.set_mouse(0, True)
    studio.render_editor_frame(1280, 720, input=clk, dt=Float32(1.0 / 60.0))
    assert_equal(studio.agent_bus.pending_approval_count(), 0)
    assert_true(abs(studio.entities[eid].translation.x - new_pos.x) < Float32(1.0e-3))
    assert_true(abs(studio.entities[eid].translation.y - new_pos.y) < Float32(1.0e-3))

    studio.shutdown()


# ---------------------------------------------------------------------------
# 3. Follow-Agent Camera + Auto-Yield on User Input + Resume
# ---------------------------------------------------------------------------

def test_e2e_follow_agent_camera_and_yield() raises:
    print("  [3/6] Follow-Agent Camera smooth converge + user orbit yield + resume...")
    var studio = _make_headless_studio(640, 360)
    studio.set_anim_mode(0)  # UI_ANIM_NORMAL to verify smooth convergence
    studio.look_at(Vec3(0.0, 3.0, -8.0), Vec3(0.0, 0.5, 0.0))

    var aid = studio.agent_bus.create_session(ROLE_SCENE_BUILDER, "Scene Builder", APPROVAL_EDITS)
    studio.agent_bus.follow_agent_camera = True
    var far_target = Vec3(10.0, 2.5, -6.0)
    _ = studio.agent_bus.begin_mcp_tool_step(
        aid,
        "ecs.spawn_entity",
        "Spawn Distant Pillar",
        "{}",
        entity_id=99,
        world_pos=far_target,
        has_world_pos=True,
        now_sec=1.0,
    )
    assert_true(studio.agent_bus.should_follow_camera())

    var dist_before = (studio.cam_target - far_target).length()
    for _ in range(30):
        studio.poll_agent_runtime(Float32(1.0 / 60.0))
    var dist_after = (studio.cam_target - far_target).length()
    assert_true(dist_after < dist_before * Float32(0.35))

    # User camera orbit input MUST automatically yield Follow-Agent Camera!
    studio.update_camera(Float32(1.0 / 60.0), orbit_delta_yaw_deg=Float32(18.0))
    assert_false(studio.agent_bus.should_follow_camera())
    var yielded_target = studio.cam_target
    var new_far = Vec3(-14.0, 4.0, 9.0)
    _ = studio.agent_bus.begin_mcp_tool_step(
        aid,
        "ecs.spawn_entity",
        "Spawn Another Pillar",
        "{}",
        entity_id=100,
        world_pos=new_far,
        has_world_pos=True,
        now_sec=2.0,
    )
    for _ in range(10):
        studio.poll_agent_runtime(Float32(1.0 / 60.0))
    # Camera must NOT have been hijacked toward new_far while yielded
    assert_true((studio.cam_target - yielded_target).length() < Float32(1.0e-3))

    # Resume follow camera -> converges toward new_far
    studio.agent_bus.resume_follow_camera()
    assert_true(studio.agent_bus.should_follow_camera())
    var d0 = (studio.cam_target - new_far).length()
    for _ in range(35):
        studio.poll_agent_runtime(Float32(1.0 / 60.0))
    var d1 = (studio.cam_target - new_far).length()
    assert_true(d1 < d0 * Float32(0.35))

    # Toggle off via PANEL_ACT_TOGGLE_FOLLOW_CAMERA
    var tog = AgentPanelAction(PANEL_ACT_TOGGLE_FOLLOW_CAMERA)
    studio.apply_agent_panel_action(tog)
    assert_false(studio.agent_bus.follow_agent_camera)

    studio.shutdown()


# ---------------------------------------------------------------------------
# 4. Non-Destructive Time-Travel Scrub + 1-Click Revert Step & Revert Turn
# ---------------------------------------------------------------------------

def test_e2e_time_travel_and_step_turn_revert() raises:
    print("  [4/6] Non-destructive Time-Travel scrub + 1-click Revert Step & Revert Turn...")
    var studio = _make_headless_studio(640, 360)
    var aid = studio.agent_bus.create_session(ROLE_SCENE_BUILDER, "Scene Builder", APPROVAL_EDITS)

    var h0 = studio.state_hash()
    var b0 = studio._capture_world_blob()
    var turn_id = studio.agent_bus.begin_turn(aid, "Spawn and elevate statue", b0, h0, now_sec=1.0)

    # Step 1: spawn Statue at (1, 1, 1) -> h1
    var s1 = studio.agent_bus.begin_mcp_tool_step(
        aid, "ecs.spawn_entity", "Spawn Statue", "{}",
        pre_world_lz4=b0, pre_hash=h0, now_sec=1.1,
    )
    var statue_id = studio.spawn_entity("AgentStatue", position=Vec3(1.0, 1.0, 1.0), record_undo=True)
    var h1 = studio.state_hash()
    var b1 = studio._capture_world_blob()
    assert_true(h1 != h0)
    var tx1 = studio.undo_stack[len(studio.undo_stack) - 1].tx_id if len(studio.undo_stack) > 0 else UInt64(1)
    _ = studio.record_agent_step_checkpoint(tx1, Int(aid), Int(turn_id), Int(s1), b0, b1, "Spawn Statue")
    assert_true(studio.agent_bus.complete_mcp_tool_step(
        aid, s1, True, '{"ok":true}', 3.0, tx1,
        entity_id=statue_id, world_pos=Vec3(1.0, 1.0, 1.0), has_world_pos=True,
        post_world_lz4=b1, post_hash=h1,
    ))

    # Step 2: elevate Statue to (1, 5, 1) -> h2
    var s2 = studio.agent_bus.begin_mcp_tool_step(
        aid, "entity.set_transform", "Elevate Statue", "{}",
        entity_id=statue_id, world_pos=Vec3(1.0, 5.0, 1.0), has_world_pos=True,
        pre_world_lz4=b1, pre_hash=h1, now_sec=1.2,
    )
    var upd = studio.entities[statue_id].copy()
    upd.translation = Vec3(1.0, 5.0, 1.0)
    assert_true(studio.commit_node_edit(statue_id, upd^, "Elevate Statue"))
    var h2 = studio.state_hash()
    var b2 = studio._capture_world_blob()
    assert_true(h2 != h1)
    var tx2 = studio.undo_stack[len(studio.undo_stack) - 1].tx_id if len(studio.undo_stack) > 0 else UInt64(2)
    _ = studio.record_agent_step_checkpoint(tx2, Int(aid), Int(turn_id), Int(s2), b1, b2, "Elevate Statue")
    assert_true(studio.agent_bus.complete_mcp_tool_step(
        aid, s2, True, '{"ok":true}', 2.0, tx2,
        entity_id=statue_id, world_pos=Vec3(1.0, 5.0, 1.0), has_world_pos=True,
        post_world_lz4=b2, post_hash=h2,
    ))

    # A. Non-destructive Time-Travel scrub to step index 0 -> exact h1
    var scrub_act = AgentPanelAction(PANEL_ACT_TIME_TRAVEL_SCRUB)
    scrub_act.agent_id = aid
    scrub_act.step_index = 0
    studio.apply_agent_panel_action(scrub_act)
    assert_true(studio.agent_bus.time_travel_active)
    assert_equal(studio.state_hash(), h1)

    # Exit Time-Travel [Live] -> exact h2 restored!
    var exit_act = AgentPanelAction(PANEL_ACT_TIME_TRAVEL_EXIT)
    studio.apply_agent_panel_action(exit_act)
    assert_false(studio.agent_bus.time_travel_active)
    assert_equal(studio.state_hash(), h2)

    # B. 1-click Revert Step (s2) -> exact h1 restored!
    var rev_s2 = AgentPanelAction(PANEL_ACT_REVERT_STEP)
    rev_s2.agent_id = aid
    rev_s2.step_id = s2
    studio.apply_agent_panel_action(rev_s2)
    assert_equal(studio.state_hash(), h1)
    var si = studio.agent_bus.find_session_index(aid)
    assert_equal(studio.agent_bus.sessions[si].steps[1].status, STEP_REVERTED)

    # C. 1-click Revert Turn -> exact pre-turn h0 restored!
    var rev_turn = AgentPanelAction(PANEL_ACT_REVERT_TURN)
    rev_turn.agent_id = aid
    rev_turn.turn_id = turn_id
    studio.apply_agent_panel_action(rev_turn)
    assert_equal(studio.state_hash(), h0)
    assert_equal(studio.agent_bus.sessions[si].steps[0].status, STEP_REVERTED)

    studio.shutdown()


# ---------------------------------------------------------------------------
# 5. Agent File Proposal in CodeDrawer + Compiler Diagnostics -> ProblemStore
# ---------------------------------------------------------------------------

def test_e2e_code_drawer_proposal_and_problems() raises:
    print("  [5/6] Agent file proposal in CodeDrawer + compiler diagnostics -> ProblemStore...")
    var studio = _make_headless_studio(1280, 720)
    var old_code = String("def dash_speed() -> Float32:\n    return 6.0\n\ndef gravity() -> Float32:\n    return -9.8\n")
    var new_code = String("def dash_speed() -> Float32:\n    return 11.5\n\ndef gravity() -> Float32:\n    return -14.0\n")

    var tab_idx = studio.code_drawer.show_agent_proposal(
        "src/hero_controller.mojo", old_code, new_code, "Boost dash & crisp gravity"
    )
    assert_true(tab_idx >= 0)
    studio.code_drawer.view_mode = CODE_MODE_DIFF_UNIFIED
    assert_equal(studio.code_drawer.tabs[tab_idx].diff_doc.hunk_count(), 2)

    # Accept hunk 0 & reject hunk 1 -> CodeDrawer commits patched source into tab.text
    assert_true(studio.code_drawer.apply_hunk_decision(0, True))
    assert_true(studio.code_drawer.apply_hunk_decision(1, False))
    assert_true(studio.code_drawer.tabs[tab_idx].dirty)
    assert_false(studio.code_drawer.tabs[tab_idx].has_proposal)
    var patched = studio.code_drawer.tabs[tab_idx].text
    assert_true(patched.find("11.5") >= 0)
    assert_true(patched.find("-9.8") >= 0)
    assert_true(patched.find("-14.0") < 0)

    # Mock compiler build output -> ProblemStore populated inside Studio's CodeDrawer
    var jid = studio.agent_runtime.jobs.add_mock_job(JOB_KIND_BUILD_SO, "Build gameplay.so", ".")
    var jidx = studio.agent_runtime.jobs.find_index(jid)
    assert_true(jidx >= 0)
    var mock_err = String(
        "src/hero_controller.mojo:12:5: error: cannot implicitly convert 'Float64' to 'Float32'\n"
        + "src/hero_controller.mojo:19:9: warning: unused local variable 'prev_y'\n"
    )
    studio.agent_runtime.jobs.jobs[jidx].feed_mock_output(mock_err, finish_exit_code=1, now_sec=1.0)
    var parsed = studio.agent_runtime.jobs.jobs[jidx].diagnostics.copy()
    assert_equal(len(parsed), 2)
    studio.code_drawer.problems.replace_source(DIAG_SRC_BUILD, parsed)
    assert_equal(studio.code_drawer.problems.error_count(), 1)
    assert_equal(studio.code_drawer.problems.warning_count(), 1)

    studio.shutdown()


# ---------------------------------------------------------------------------
# 6. All 6 configure_gallery_shot(0..5) Deterministic Under UI_ANIM_INSTANT
# ---------------------------------------------------------------------------

def test_e2e_all_6_gallery_shots_deterministic() raises:
    print("  [6/6] All 6 configure_gallery_shot(0..5) deterministic under UI_ANIM_INSTANT...")
    var studio = _make_headless_studio(640, 360)
    var shot_hashes = List[UInt64](capacity=6)

    for shot_idx in range(6):
        # Pass A: configure + 1 warm-up frame (populates retain-cached FlexTree & glyph atlas) + steady-state capture
        studio.configure_gallery_shot(shot_idx, 640, 360)
        studio.render_editor_frame(640, 360, dt=Float32(1.0 / 60.0))
        studio.render_editor_frame(640, 360, dt=Float32(1.0 / 60.0))
        var cnt_a = studio.ui_ctx.draw_list.count()
        var ren_a = studio.app.resources.get_ptr[ScalableRenderer]()
        assert_true(Bool(ren_a))
        var px_a = ren_a.value()[unsafe_offset=0].readback_pixels()
        var hash_a = _hash_pixels(px_a)

        # Pass B: re-configure + identical 2 frames -> MUST match exact prim count & pixel hash!
        studio.configure_gallery_shot(shot_idx, 640, 360)
        studio.render_editor_frame(640, 360, dt=Float32(1.0 / 60.0))
        studio.render_editor_frame(640, 360, dt=Float32(1.0 / 60.0))
        var cnt_b = studio.ui_ctx.draw_list.count()
        var px_b = studio.app.resources.get_ptr[ScalableRenderer]().value()[unsafe_offset=0].readback_pixels()
        var hash_b = _hash_pixels(px_b)

        assert_true(cnt_a > 20)
        assert_equal(cnt_a, cnt_b)
        assert_equal(hash_a, hash_b)
        shot_hashes.append(hash_a)

    # Verify shots are genuinely distinct (Edit vs Palette vs AgentTheatre vs CodeSplit vs PIE vs PlayHUD)
    assert_true(shot_hashes[0] != shot_hashes[1])
    assert_true(shot_hashes[0] != shot_hashes[2])
    assert_true(shot_hashes[0] != shot_hashes[3])
    assert_true(shot_hashes[0] != shot_hashes[4])
    assert_true(shot_hashes[0] != shot_hashes[5])
    assert_true(shot_hashes[2] != shot_hashes[3])

    studio.shutdown()


def main() raises:
    print("============================================================")
    print(" FLINT STUDIO E2E VISUAL AGENT THEATRE & GALLERY SUITE")
    print("============================================================")
    test_e2e_acp_mcp_ingest_and_overlays()
    test_e2e_approval_gate_and_viewport_hit()
    test_e2e_follow_agent_camera_and_yield()
    test_e2e_time_travel_and_step_turn_revert()
    test_e2e_code_drawer_proposal_and_problems()
    test_e2e_all_6_gallery_shots_deterministic()
    print("[PASS] All 6 E2E Visual Agent Theatre & Gallery suites PASSED!")

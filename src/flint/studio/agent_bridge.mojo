"""Flint Studio Unified Agent Runtime Bridge (`agent_bridge.mojo`).

Pure-Mojo 1.2 runtime orchestrator connecting:
  - `AcpClient` (`flint.devtools.acp_client`) — coding-agent JSON-RPC 2.0 streams
  - `BridgedMcpEvent` (decoupled mirror of `McpToolEvent` so there is zero circular import of `studio_app.mojo`)
  - `JobRunner` (`flint.devtools.job_runner`) — non-blocking build/test background jobs
  - `AgentTelemetryBus` (`flint.studio.agent_bus`) — unified timeline, approvals, trails, time-travel
  - `CodeDrawer` (`flint.studio.code_drawer`) — inline Mojo diffs & compiler diagnostics
  - `ToastManager` (`flint.studio.notifications`) — non-blocking Studio notifications

Also provides:
  - `seed_demo_agent_showcase(...)`: deterministic showcase fixture for gallery shots
    (`03_agent_theatre.png`, `04_code_scene_split.png`) and instant inspection
  - `start_scripted_demo(...)` + `tick_scripted_demo(...)`: deterministic fixed-dt live
    multi-step playback for interactive "Watch Demo Agent" sessions
"""
from std.collections import List
from std.math import isfinite, max
from flint.assets.json import JsonDocument, parse_json
from flint.core.math import Quat, Vec3
from flint.devtools.acp_client import (
    AE_CMD_APPROVAL,
    AE_DONE,
    AE_ERROR,
    AE_FILE_PROPOSAL,
    AE_MESSAGE,
    AE_PERMISSION,
    AE_PLAN,
    AE_READY,
    AE_THOUGHT,
    AE_TOOL,
    AE_TOOL_UPDATE,
    AcpClient,
    AcpEvent,
)
from flint.devtools.job_runner import (
    JOB_FAILED,
    JOB_KIND_BUILD_SO,
    JOB_SUCCEEDED,
    JobRunner,
)
from flint.devtools.problems import (
    DIAG_INFO,
    DIAG_SRC_AGENT,
    DIAG_SRC_BUILD,
    DIAG_WARNING,
    Diagnostic,
)
from flint.platform.os import read_bytes
from flint.studio.agent_bus import (
    ACT_BUILD_HOTRELOAD,
    ACT_ENTITY_DESPAWN,
    ACT_ENTITY_SPAWN,
    ACT_ENTITY_TRANSFORM,
    ACT_MATERIAL_STYLE,
    APPROVAL_EDITS,
    APPROVAL_SRC_ACP_PERM,
    APPROVAL_SRC_ACP_RUN,
    APPROVAL_SRC_ACP_WRITE,
    APPROVAL_SRC_MCP_TOOL,
    APPROVAL_STEP,
    GHOST_DESPAWN,
    GHOST_MATERIAL,
    GHOST_NONE,
    GHOST_SPAWN,
    GHOST_TRANSFORM,
    ROLE_GAMEPLAY_CODER,
    ROLE_SCENE_BUILDER,
    AgentGhostPreview,
    AgentTelemetryBus,
    PendingApproval,
    action_kind_from_mcp_tool,
    agent_role_accent_rgba8,
)
from flint.studio.code_drawer import (
    BUILD_STATUS_BUILDING,
    BUILD_STATUS_ERROR,
    BUILD_STATUS_OK,
    CODE_MODE_DIFF_UNIFIED,
    CodeDrawer,
)
from flint.studio.notifications import (
    TOAST_AGENT,
    TOAST_ERROR,
    TOAST_SUCCESS,
    TOAST_WARN,
    ToastManager,
)


# ============================================================================
# 1. Decoupled MCP Event Mirror & JSON Spatial Extractors
# ============================================================================

struct BridgedMcpEvent(Copyable, Movable):
    """Decoupled mirror of `McpToolEvent` (avoids importing `mcp_v2.mojo` / `studio_app.mojo`)."""
    var seq: UInt64
    var agent_id: UInt32
    var tool_name: String
    var params_json: String
    var result_json: String
    var affected_entities: List[Int]
    var elapsed_ms: Float64
    var ok: Bool
    var is_mutating: Bool
    var blocked_by_gate: Bool
    var mcp_client_key: Int
    """MCP connection that made the call (0 = in-process / stdio)."""
    var mcp_client_name: String

    # Optional spatial hints supplied by Studio when entity world positions are known
    var world_pos: Vec3
    var has_world_pos: Bool
    var prev_world_pos: Vec3
    var has_prev_world_pos: Bool
    var half_extents: Vec3

    def __init__(
        out self,
        seq: UInt64 = 0,
        tool_name: String = "",
        params_json: String = "{}",
        result_json: String = "{}",
        elapsed_ms: Float64 = 0.0,
        ok: Bool = True,
        is_mutating: Bool = False,
        blocked_by_gate: Bool = False,
        agent_id: UInt32 = 0,
    ):
        self.seq = seq
        self.agent_id = agent_id
        self.tool_name = tool_name
        self.params_json = params_json
        self.result_json = result_json
        self.affected_entities = List[Int]()
        self.elapsed_ms = elapsed_ms
        self.ok = ok
        self.is_mutating = is_mutating
        self.blocked_by_gate = blocked_by_gate
        self.mcp_client_key = 0
        self.mcp_client_name = String("")
        self.world_pos = Vec3()
        self.has_world_pos = False
        self.prev_world_pos = Vec3()
        self.has_prev_world_pos = False
        self.half_extents = Vec3(0.5, 0.5, 0.5)


def _str_to_bytes(s: String) -> List[UInt8]:
    var out = List[UInt8](capacity=s.byte_length())
    for b in s.as_bytes():
        out.append(b)
    return out^


def _parse_json_str(s: String) -> JsonDocument:
    if s.byte_length() == 0:
        return JsonDocument()
    return parse_json(_str_to_bytes(s))


def _extract_vec3_from_obj(doc: JsonDocument, obj_node: Int, key: String) -> Tuple[Bool, Vec3]:
    if not doc.valid or obj_node < 0 or doc.nodes[obj_node].kind != 5:
        return (False, Vec3())
    var arr = doc.object_get(obj_node, key)
    if arr < 0 or doc.nodes[arr].kind != 4 or doc.array_size(arr) < 3:
        return (False, Vec3())
    var c0 = doc.at(arr, 0)
    var c1 = doc.at(arr, 1)
    var c2 = doc.at(arr, 2)
    if doc.nodes[c0].kind != 2 or doc.nodes[c1].kind != 2 or doc.nodes[c2].kind != 2:
        return (False, Vec3())
    var x = Float32(doc.nodes[c0].number)
    var y = Float32(doc.nodes[c1].number)
    var z = Float32(doc.nodes[c2].number)
    if not (isfinite(x) and isfinite(y) and isfinite(z)):
        return (False, Vec3())
    return (True, Vec3(x, y, z))


def _first_option_of_kind(kinds: List[String], ids: List[String], want: String) -> String:
    var n = min(len(kinds), len(ids))
    for i in range(n):
        if kinds[i] == want:
            return ids[i]
    return String()


def _extract_path_from_location(loc: String) -> String:
    var colon = loc.rfind(":")
    if colon > 0:
        return String(loc[byte=0:colon])
    return loc


def _ghost_kind_for_mcp_tool(tool_name: String) -> UInt8:
    var act = action_kind_from_mcp_tool(tool_name)
    if act == ACT_ENTITY_SPAWN:
        return GHOST_SPAWN
    if act == ACT_ENTITY_TRANSFORM:
        return GHOST_TRANSFORM
    if act == ACT_ENTITY_DESPAWN:
        return GHOST_DESPAWN
    if act == ACT_MATERIAL_STYLE:
        return GHOST_MATERIAL
    if tool_name == "ecs.patch_components":
        return GHOST_TRANSFORM
    return GHOST_NONE


def _summarize_mcp_event(ev: BridgedMcpEvent, pdoc: JsonDocument, rdoc: JsonDocument) -> String:
    var name = pdoc.text(pdoc.object_get(pdoc.root, "name"), "") if pdoc.valid else String("")
    var eid = ev.affected_entities[0] if len(ev.affected_entities) > 0 else (
        rdoc.integer(rdoc.object_get(rdoc.root, "entity_id"), -1) if rdoc.valid else -1
    )
    if ev.tool_name == "ecs.spawn_entity":
        return "Spawn " + (name if name.byte_length() > 0 else ("#" + String(eid)))
    if ev.tool_name == "prefab.spawn":
        var p = pdoc.text(pdoc.object_get(pdoc.root, "prefab_path"), "prefab") if pdoc.valid else String("prefab")
        return "Spawn prefab " + p
    if ev.tool_name == "entity.set_transform":
        return "Transform #" + String(eid)
    if ev.tool_name == "ecs.patch_components":
        return "Patch " + (name if name.byte_length() > 0 else ("#" + String(eid)))
    if ev.tool_name == "ecs.despawn_entity":
        return "Despawn #" + String(eid)
    if ev.tool_name == "material.assign" or ev.tool_name == "render.configure_npr_cel":
        return "Style #" + String(eid)
    if ev.tool_name == "render.capture_viewport":
        var out_p = pdoc.text(pdoc.object_get(pdoc.root, "output_path"), "viewport.png") if pdoc.valid else String("viewport.png")
        return "Capture " + out_p
    if ev.tool_name == "studio.pie_control":
        var a = pdoc.text(pdoc.object_get(pdoc.root, "action"), "play") if pdoc.valid else String("play")
        return "PIE: " + a
    if ev.tool_name == "nav.bake" or ev.tool_name == "nav.query_path":
        return "NavMesh " + ev.tool_name
    if ev.tool_name == "code.build_hot_reload":
        return "Build & Hot-Reload gameplay"
    if ev.tool_name == "procedural.generate_asset":
        var pre = pdoc.text(pdoc.object_get(pdoc.root, "preset"), "asset") if pdoc.valid else String("asset")
        return "Forge " + pre
    return ev.tool_name


# ============================================================================
# 2. Deterministic Showcase Diff Content & Scripted Demo State
# ============================================================================

def _demo_controller_old_source() -> String:
    return (
        "from flint.core.math import Vec3\n"
        "\n"
        "struct HeroMotionState(Copyable, Movable):\n"
        "    var velocity: Vec3\n"
        "    var coyote_timer: Float32\n"
        "    var dash_cooldown: Float32\n"
        "    var grounded: Bool\n"
        "\n"
        "    def __init__(out self):\n"
        "        self.velocity = Vec3()\n"
        "        self.coyote_timer = Float32(0.0)\n"
        "        self.dash_cooldown = Float32(0.0)\n"
        "        self.grounded = True\n"
        "\n"
        "def step_hero_motion(mut st: HeroMotionState, move_x: Float32, jump_pressed: Bool, dt: Float32):\n"
        "    if st.grounded:\n"
        "        st.coyote_timer = Float32(0.06)\n"
        "    else:\n"
        "        st.coyote_timer = max(Float32(0.0), st.coyote_timer - dt)\n"
        "    st.velocity.x = move_x * Float32(5.5)\n"
        "    if jump_pressed and st.coyote_timer > Float32(0.0):\n"
        "        st.velocity.y = Float32(7.2)\n"
        "        st.coyote_timer = Float32(0.0)\n"
        "        st.grounded = False\n"
    )


def _demo_controller_new_source() -> String:
    return (
        "from flint.core.math import Vec3\n"
        "\n"
        "comptime HERO_COYOTE_SEC: Float32 = 0.12\n"
        "comptime HERO_RUN_SPEED: Float32 = 6.8\n"
        "comptime HERO_JUMP_IMPULSE: Float32 = 8.4\n"
        "\n"
        "struct HeroMotionState(Copyable, Movable):\n"
        "    var velocity: Vec3\n"
        "    var coyote_timer: Float32\n"
        "    var dash_cooldown: Float32\n"
        "    var grounded: Bool\n"
        "\n"
        "    def __init__(out self):\n"
        "        self.velocity = Vec3()\n"
        "        self.coyote_timer = Float32(0.0)\n"
        "        self.dash_cooldown = Float32(0.0)\n"
        "        self.grounded = True\n"
        "\n"
        "def step_hero_motion(mut st: HeroMotionState, move_x: Float32, jump_pressed: Bool, dt: Float32):\n"
        "    if st.grounded:\n"
        "        st.coyote_timer = HERO_COYOTE_SEC\n"
        "    else:\n"
        "        st.coyote_timer = max(Float32(0.0), st.coyote_timer - dt)\n"
        "    st.dash_cooldown = max(Float32(0.0), st.dash_cooldown - dt)\n"
        "    st.velocity.x = move_x * HERO_RUN_SPEED\n"
        "    if jump_pressed and st.coyote_timer > Float32(0.0):\n"
        "        st.velocity.y = HERO_JUMP_IMPULSE\n"
        "        st.coyote_timer = Float32(0.0)\n"
        "        st.grounded = False\n"
    )


struct ScriptedDemoState(Copyable, Movable):
    """Deterministic multi-step playback controller for interactive Visual Agent Theatre demos."""
    var active: Bool
    var elapsed_sec: Float32
    var stage: Int
    var scene_agent_id: UInt32
    var coder_agent_id: UInt32
    var target_eid: Int
    var target_pos: Vec3
    var prev_pos: Vec3
    var last_approval_id: UInt64
    var spawn_step_id: UInt64
    var move_step_id: UInt64

    def __init__(out self):
        self.active = False
        self.elapsed_sec = Float32(0.0)
        self.stage = 0
        self.scene_agent_id = 0
        self.coder_agent_id = 0
        self.target_eid = 1
        self.target_pos = Vec3(0.0, 1.2, 0.0)
        self.prev_pos = Vec3(-2.4, 0.9, -1.2)
        self.last_approval_id = 0
        self.spawn_step_id = 0
        self.move_step_id = 0


# ============================================================================
# 3. `StudioAgentRuntime`: Unified ACP + MCP + Jobs + Theatre + CodeDrawer Bridge
# ============================================================================

struct StudioAgentRuntime(Movable):
    """Coordinates live ACP agents, MCP v2 tool events, background builds, and demo playback."""
    var acp: AcpClient
    var acp_agent_id: UInt32
    var acp_command: String
    var workspace_root: String
    var mcp_socket_path: String
    var flint_bin_path: String
    var jobs: JobRunner
    var demo: ScriptedDemoState
    var build_generation: Int
    var mcp_events_processed: UInt64
    var acp_events_processed: UInt64
    var _last_mcp_toast_sec: Float64

    def __init__(out self):
        self.acp = AcpClient()
        self.acp_agent_id = 0
        self.acp_command = String()
        self.workspace_root = String(".")
        self.mcp_socket_path = String()
        self.flint_bin_path = String("flint-studio")
        self.jobs = JobRunner()
        self.demo = ScriptedDemoState()
        self.build_generation = 0
        self.mcp_events_processed = 0
        self.acp_events_processed = 0
        self._last_mcp_toast_sec = -10.0

    def configure_acp(
        mut self,
        command: String,
        workspace_root: String,
        mcp_socket_path: String = "",
        flint_bin_path: String = "flint-studio",
    ):
        self.acp_command = command
        self.workspace_root = workspace_root if workspace_root.byte_length() > 0 else String(".")
        self.mcp_socket_path = mcp_socket_path
        self.flint_bin_path = flint_bin_path if flint_bin_path.byte_length() > 0 else String("flint-studio")
        if self.mcp_socket_path.byte_length() > 0:
            self.acp.configure_studio_mcp_socket(self.flint_bin_path, self.mcp_socket_path)

    def start_acp(
        mut self,
        mut bus: AgentTelemetryBus,
        role: UInt8 = ROLE_GAMEPLAY_CODER,
        now_sec: Float64 = 0.0,
    ) -> Bool:
        if self.acp_command.byte_length() == 0:
            return False
        var aid = self._ensure_role_session(bus, role)
        self.acp_agent_id = aid
        if self.mcp_socket_path.byte_length() > 0:
            self.acp.configure_studio_mcp_socket(self.flint_bin_path, self.mcp_socket_path)
        return self.acp.start(self.acp_command, self.workspace_root, "/dev/null", now_sec)

    def start_acp_headless(
        mut self,
        mut bus: AgentTelemetryBus,
        role: UInt8 = ROLE_GAMEPLAY_CODER,
        cwd: String = ".",
        now_sec: Float64 = 0.0,
    ) -> UInt32:
        """Starts an in-memory headless ACP session for deterministic tests without spawning a subprocess."""
        var aid = self._ensure_role_session(bus, role)
        self.acp_agent_id = aid
        self.workspace_root = cwd
        self.acp.start_headless(cwd, now_sec)
        return aid

    def send_prompt(
        mut self,
        mut bus: AgentTelemetryBus,
        agent_id: UInt32,
        prompt: String,
        pre_world_lz4: List[UInt8] = List[UInt8](),
        pre_hash: UInt64 = 0,
        now_sec: Float64 = 0.0,
    ) -> Bool:
        if prompt.byte_length() == 0:
            return False
        var aid = agent_id
        if bus.find_session_index(aid) < 0:
            aid = self._ensure_role_session(bus, ROLE_SCENE_BUILDER)
        _ = bus.begin_turn(aid, prompt, pre_world_lz4, pre_hash, now_sec)
        bus.selected_agent_id = aid
        if self.acp.active() and (self.acp_agent_id == 0 or self.acp_agent_id == aid):
            self.acp_agent_id = aid
            return self.acp.send_prompt(prompt, now_sec)
        return True

    def cancel_active_turn(
        mut self,
        mut bus: AgentTelemetryBus,
        agent_id: UInt32,
        now_sec: Float64 = 0.0,
    ):
        if self.acp.busy() and (self.acp_agent_id == 0 or self.acp_agent_id == agent_id):
            self.acp.cancel()
        if self.demo.active and (agent_id == self.demo.scene_agent_id or agent_id == self.demo.coder_agent_id):
            self.demo.active = False
        bus.end_turn(agent_id, "cancelled", now_sec)

    def answer_acp_approval(
        mut self,
        pa: PendingApproval,
        approved: Bool,
        allow_turn: Bool = False,
    ) -> Bool:
        """Replies to a parked ACP permission, file-write proposal, or terminal run request."""
        if not self.acp.active() or pa.request_id_raw.byte_length() == 0:
            return False
        if pa.source == APPROVAL_SRC_ACP_PERM:
            var opt: String
            if approved:
                opt = pa.option_allow_always if (allow_turn and pa.option_allow_always.byte_length() > 0) else pa.option_allow_once
                if opt.byte_length() == 0:
                    opt = pa.option_allow_always
            else:
                opt = pa.option_reject
            return self.acp.answer_permission(pa.request_id_raw, opt)
        if pa.source == APPROVAL_SRC_ACP_WRITE or pa.source == APPROVAL_SRC_ACP_RUN:
            if approved:
                self.acp.answer_service(pa.request_id_raw, "{}")
            else:
                var why = pa.reject_reason if pa.reject_reason.byte_length() > 0 else String("Rejected by user")
                self.acp.answer_service_error(pa.request_id_raw, -32000, why)
            return True
        return False

    def _ensure_role_session(mut self, mut bus: AgentTelemetryBus, role: UInt8) -> UInt32:
        for i in range(len(bus.sessions)):
            if bus.sessions[i].role == role:
                return bus.sessions[i].agent_id
        var pol = APPROVAL_STEP if role == ROLE_SCENE_BUILDER else APPROVAL_EDITS
        return bus.create_session(role, "", pol)

    # ---- ACP Event Drain -> AgentTelemetryBus + CodeDrawer + ToastManager ----

    def drain_acp_events(
        mut self,
        mut bus: AgentTelemetryBus,
        mut code_drawer: CodeDrawer,
        mut toasts: ToastManager,
        now_sec: Float64 = 0.0,
    ) -> Int:
        var evs = self.acp.take_events()
        if len(evs) == 0:
            return 0
        var aid = self.acp_agent_id
        if bus.find_session_index(aid) < 0:
            aid = self._ensure_role_session(bus, ROLE_GAMEPLAY_CODER)
            self.acp_agent_id = aid

        var handled = 0
        for i in range(len(evs)):
            ref e = evs[i]
            self.acp_events_processed += 1
            handled += 1

            if e.kind == AE_READY:
                var si = bus.find_session_index(aid)
                if si >= 0:
                    bus.sessions[si].acp_session_id = self.acp.session_id
                var aname = self.acp.agent_name if self.acp.agent_name.byte_length() > 0 else String("Agent")
                _ = toasts.push(TOAST_AGENT, aname + " Connected", "Session ready for tasks")

            elif e.kind == AE_THOUGHT:
                _ = bus.record_acp_thought(aid, e.text, now_sec)

            elif e.kind == AE_MESSAGE:
                var si = bus.find_session_index(aid)
                if si >= 0 and e.text.byte_length() > 0:
                    bus.sessions[si].active_callout = e.text

            elif e.kind == AE_PLAN:
                _ = bus.record_acp_plan(aid, e.entries, now_sec)
                _ = toasts.push(TOAST_AGENT, "Agent Plan Updated", String(len(e.entries)) + " step(s) planned")

            elif e.kind == AE_TOOL:
                var fpath = _extract_path_from_location(e.locations[0]) if len(e.locations) > 0 else e.path
                var ttl = e.title if e.title.byte_length() > 0 else String("Tool call")
                _ = bus.record_acp_tool_start(aid, e.id, ttl, e.tool_kind, fpath, now_sec)

            elif e.kind == AE_TOOL_UPDATE:
                var updated = bus.record_acp_tool_update(aid, e.id, e.status, e.text, e.text)
                if not updated and e.id.byte_length() > 0:
                    _ = bus.record_acp_tool_start(aid, e.id, e.title, e.tool_kind, e.path, now_sec)
                    _ = bus.record_acp_tool_update(aid, e.id, e.status, e.text, e.text)
                if e.status == "completed" and len(e.diffs) > 0:
                    _ = toasts.push(TOAST_AGENT, "Agent Edited File", e.diffs[0])

            elif e.kind == AE_FILE_PROPOSAL:
                var old_txt = String("")
                var existed = False
                var tab_i = code_drawer.find_tab(e.path)
                if tab_i >= 0:
                    old_txt = code_drawer.tabs[tab_i].text
                    existed = True
                elif e.path.byte_length() > 0:
                    var disk_bytes = read_bytes(e.path)
                    if len(disk_bytes) > 0:
                        old_txt = String(unsafe_from_utf8=disk_bytes)
                        existed = True
                _ = code_drawer.show_agent_proposal(e.path, old_txt, e.text, e.request_id, split_mode=False)

                var sid = bus.record_acp_file_edit(aid, e.path, old_txt, e.text, existed, now_sec)
                var pa = PendingApproval(0, aid, sid, APPROVAL_SRC_ACP_WRITE, "Write " + e.path, "Proposed file diff ready in Code Drawer")
                pa.request_id_raw = e.request_id
                pa.file_path = e.path
                pa.old_text = old_txt
                pa.new_text = e.text
                pa.file_existed = existed
                _ = bus.queue_approval(pa^)
                _ = toasts.push(TOAST_WARN, "Agent Proposes File Edit", e.path, own_job=True)

            elif e.kind == AE_PERMISSION:
                var ttl = e.title if e.title.byte_length() > 0 else String("Permission Request")
                var sid = bus.record_acp_tool_start(aid, e.id, ttl, e.tool_kind, e.path, now_sec)
                var pa = PendingApproval(0, aid, sid, APPROVAL_SRC_ACP_PERM, ttl, "Agent requests permission (" + e.tool_kind + ")")
                pa.request_id_raw = e.request_id
                pa.tool_kind = e.tool_kind
                pa.option_allow_once = _first_option_of_kind(e.option_kinds, e.option_ids, "allow_once")
                pa.option_allow_always = _first_option_of_kind(e.option_kinds, e.option_ids, "allow_always")
                pa.option_reject = _first_option_of_kind(e.option_kinds, e.option_ids, "reject_once")
                if pa.option_reject.byte_length() == 0:
                    pa.option_reject = _first_option_of_kind(e.option_kinds, e.option_ids, "reject_always")
                _ = bus.queue_approval(pa^)
                _ = toasts.push(TOAST_WARN, "Agent Permission Required", ttl, own_job=True)

            elif e.kind == AE_CMD_APPROVAL:
                var cmd_title = e.title
                if cmd_title.byte_length() == 0 and len(e.argv) > 0:
                    for ai in range(len(e.argv)):
                        if ai > 0:
                            cmd_title += " "
                        cmd_title += e.argv[ai]
                var sid = bus.record_acp_tool_start(aid, e.request_id, "$ " + cmd_title, "execute", e.path, now_sec)
                var pa = PendingApproval(0, aid, sid, APPROVAL_SRC_ACP_RUN, "Run Command", cmd_title)
                pa.request_id_raw = e.request_id
                pa.tool_kind = "execute"
                _ = bus.queue_approval(pa^)
                _ = toasts.push(TOAST_WARN, "Agent Wants to Run Command", cmd_title, own_job=True)

            elif e.kind == AE_DONE:
                var reason = e.stop_reason if e.stop_reason.byte_length() > 0 else String("end_turn")
                bus.end_turn(aid, reason, now_sec)
                if reason == "end_turn":
                    _ = toasts.push(TOAST_SUCCESS, "Agent Turn Complete", "Ready for review or next task")
                else:
                    _ = toasts.push(TOAST_WARN, "Agent Turn Ended", reason)

            elif e.kind == AE_ERROR:
                bus.end_turn(aid, "error", now_sec)
                _ = toasts.push(TOAST_ERROR, "Agent Error", e.text, own_job=True)

        return handled

    # ---- MCP v2 Event Drain -> AgentTelemetryBus + 3D Halos/Trails + Toasts ----

    def drain_mcp_events(
        mut self,
        mut bus: AgentTelemetryBus,
        events: List[BridgedMcpEvent],
        mut toasts: ToastManager,
        now_sec: Float64 = 0.0,
    ) -> Int:
        var count = 0
        for i in range(len(events)):
            ref ev = events[i]
            self.mcp_events_processed += 1
            count += 1

            var aid = ev.agent_id
            if bus.find_session_index(aid) < 0:
                if self.acp_agent_id != 0 and bus.find_session_index(self.acp_agent_id) >= 0:
                    aid = self.acp_agent_id
                else:
                    aid = bus.ensure_external_mcp_session()

            var pdoc = _parse_json_str(ev.params_json)
            var rdoc = _parse_json_str(ev.result_json)

            var eid = ev.affected_entities[0] if len(ev.affected_entities) > 0 else -1
            if eid < 0 and rdoc.valid:
                eid = rdoc.integer(rdoc.object_get(rdoc.root, "entity_id"), -1)
            if eid < 0 and rdoc.valid:
                eid = rdoc.integer(rdoc.object_get(rdoc.root, "spawned_entity_id"), -1)
            if eid < 0 and pdoc.valid:
                eid = pdoc.integer(pdoc.object_get(pdoc.root, "entity_id"), -1)

            var has_pos = ev.has_world_pos
            var wpos = ev.world_pos
            if not has_pos and pdoc.valid:
                var pv = _extract_vec3_from_obj(pdoc, pdoc.root, "position")
                if pv[0]:
                    has_pos = True
                    wpos = pv[1]
                else:
                    var sv = _extract_vec3_from_obj(pdoc, pdoc.root, "goal")
                    if sv[0]:
                        has_pos = True
                        wpos = sv[1]

            var summary = _summarize_mcp_event(ev, pdoc, rdoc)
            var sid = bus.begin_mcp_tool_step(
                agent_id=aid,
                tool_name=ev.tool_name,
                summary=summary,
                params_json=ev.params_json,
                entity_id=eid,
                world_pos=wpos,
                has_world_pos=has_pos,
                prev_world_pos=ev.prev_world_pos,
                has_prev_world_pos=ev.has_prev_world_pos,
                half_extents=ev.half_extents,
                awaiting_approval=ev.blocked_by_gate,
                now_sec=now_sec,
            )

            if ev.blocked_by_gate:
                var pa = PendingApproval(0, aid, sid, APPROVAL_SRC_MCP_TOOL, summary, "Gated MCP tool: " + ev.tool_name)
                pa.tool_name = ev.tool_name
                pa.raw_payload_json = ev.params_json
                pa.mcp_client_key = ev.mcp_client_key
                var gk = _ghost_kind_for_mcp_tool(ev.tool_name)
                if gk != GHOST_NONE and has_pos:
                    var accent = agent_role_accent_rgba8(ROLE_SCENE_BUILDER)
                    var sidx = bus.find_session_index(aid)
                    if sidx >= 0:
                        accent = bus.sessions[sidx].accent_rgba8
                    pa.ghost = AgentGhostPreview(
                        kind=gk,
                        agent_id=aid,
                        step_id=sid,
                        entity_id=eid,
                        label=summary,
                        target_pos=wpos,
                        half_extents=ev.half_extents,
                        accent_rgba8=accent,
                        from_pos=ev.prev_world_pos,
                        has_from_pos=ev.has_prev_world_pos,
                    )
                _ = bus.queue_approval(pa^)
                _ = toasts.push(TOAST_WARN, "Approval Needed", summary, own_job=True, action_entity_id=eid)
                continue

            var cap_path = String("")
            var flip_val = Float32(-1.0)
            if ev.tool_name == "render.capture_viewport" and rdoc.valid:
                cap_path = rdoc.text(rdoc.object_get(rdoc.root, "output_path"), "")
                var flip_node = rdoc.object_get(rdoc.root, "flip_mean")
                if flip_node >= 0 and rdoc.nodes[flip_node].kind == 2:
                    flip_val = Float32(rdoc.nodes[flip_node].number)

            _ = bus.complete_mcp_tool_step(
                agent_id=aid,
                step_id=sid,
                ok=ev.ok,
                result_json=ev.result_json,
                elapsed_ms=Float32(ev.elapsed_ms),
                entity_id=eid,
                world_pos=wpos,
                has_world_pos=has_pos,
                half_extents=ev.half_extents,
                capture_png_path=cap_path,
                flip_score=flip_val,
            )

            if not ev.ok:
                _ = toasts.push(TOAST_ERROR, "MCP Tool Failed", summary, own_job=True, action_entity_id=eid)
            elif ev.is_mutating:
                var is_milestone = (
                    ev.tool_name == "scene.save"
                    or ev.tool_name == "scene.load"
                    or ev.tool_name == "code.build_hot_reload"
                    or ev.tool_name == "nav.bake"
                    or ev.tool_name == "project.package"
                )
                if is_milestone or (now_sec - self._last_mcp_toast_sec) >= 1.5 or now_sec < self._last_mcp_toast_sec:
                    _ = toasts.push(TOAST_AGENT, summary, ev.tool_name, action_entity_id=eid)
                    self._last_mcp_toast_sec = now_sec

        return count

    # ---- Background Build/Test Job Polling -> CodeDrawer + Bus + Toasts ----

    def poll_jobs(
        mut self,
        mut bus: AgentTelemetryBus,
        mut code_drawer: CodeDrawer,
        mut toasts: ToastManager,
        now_sec: Float64 = -1.0,
    ) -> Int:
        var finished = self.jobs.poll_all(now_sec)
        if self.jobs.has_running_build():
            code_drawer.set_build_status(BUILD_STATUS_BUILDING, self.build_generation)

        for fi in range(len(finished)):
            var jid = finished[fi]
            var idx = self.jobs.find_index(jid)
            if idx < 0:
                continue
            ref job = self.jobs.jobs[idx]
            if job.kind == JOB_KIND_BUILD_SO:
                var diags = self.jobs.latest_diagnostics()
                code_drawer.problems.clear_source(DIAG_SRC_BUILD)
                for di in range(len(diags)):
                    code_drawer.problems.add(diags[di])
                var errs = code_drawer.problems.error_count()
                var warns = code_drawer.problems.warning_count()
                if job.state == JOB_SUCCEEDED and errs == 0:
                    self.build_generation += 1
                    code_drawer.set_build_status(BUILD_STATUS_OK, self.build_generation, 0, warns)
                    _ = toasts.push(TOAST_SUCCESS, "Hot-Reload Complete", "Build gen " + String(self.build_generation), own_job=True)
                elif job.state == JOB_FAILED or errs > 0:
                    code_drawer.set_build_status(BUILD_STATUS_ERROR, self.build_generation, max(1, errs), warns)
                    _ = toasts.push(TOAST_ERROR, "Build Failed", String(max(1, errs)) + " error(s) in Problems", own_job=True)

            # Mirror build job completion into active coder session timeline if one exists
            var coder_si = -1
            for si in range(len(bus.sessions)):
                if bus.sessions[si].role == ROLE_GAMEPLAY_CODER:
                    coder_si = si
                    break
            if coder_si >= 0:
                var cid = bus.sessions[coder_si].agent_id
                var st_id = bus.begin_mcp_tool_step(
                    cid, "code.build_hot_reload", job.title, "{}",
                    now_sec=now_sec if now_sec >= 0.0 else 0.0,
                )
                var ok = job.state == JOB_SUCCEEDED
                _ = bus.complete_mcp_tool_step(
                    cid, st_id, ok, '{"exit_code":' + String(job.exit_code) + "}",
                    elapsed_ms=Float32(job.elapsed_sec(now_sec) * 1000.0),
                )
                var s_idx = bus.sessions[coder_si].find_step_index(st_id)
                if s_idx >= 0:
                    bus.sessions[coder_si].steps[s_idx].kind = ACT_BUILD_HOTRELOAD
                    bus.sessions[coder_si].steps[s_idx].terminal_tail = job.output_text()

        return len(finished)

    # ---- Deterministic Visual Agent Showcase Fixture (Gallery & Instant Demo) ----

    def seed_demo_agent_showcase(
        mut self,
        mut bus: AgentTelemetryBus,
        mut code_drawer: CodeDrawer,
        target_eid: Int,
        target_pos: Vec3,
        prev_pos: Vec3,
    ):
        """Populates a deterministic, visually rich Visual Agent Theatre state:
        - Active Scene Builder with 4-item plan (2 done, 1 active), 3D breadcrumb trail,
          glowing ghost transform preview, Bézier tether, world-space callout & [Approve]/[Reject] banner
        - Completed Gameplay Coder turn with a real multi-hunk Mojo diff in CodeDrawer + diagnostics."""
        bus = AgentTelemetryBus()
        bus.theatre_overlays_enabled = True
        bus.theatre_time_sec = Float32(1.45)

        # 1. Active Scene Builder session
        var scene_aid = bus.create_session(ROLE_SCENE_BUILDER, "Scene Builder", APPROVAL_STEP)
        _ = bus.begin_turn(
            scene_aid,
            "Stage warm key lighting, align hero prop, and preview rim accent placement",
            _str_to_bytes("FLINT_DEMO_PRE_TURN"),
            UInt64(0x1001),
            0.0,
        )
        _ = bus.record_acp_thought(
            scene_aid,
            "Framing hero MaterialOrb against colonnade silhouette; adding warm key + cyan rim accent.",
            0.1,
        )
        var plan = List[String]()
        plan.append("[completed] Inspect scene bounds & hero silhouette")
        plan.append("[completed] Align hero prop & warm key light")
        plan.append("[in_progress] Preview Rim_Cyan accent placement (awaiting approval)")
        plan.append("[pending] Capture viewport & verify FLIP < 0.01")
        _ = bus.record_acp_plan(scene_aid, plan, 0.2)

        # Completed spawn step + transform step so 3D breadcrumb trail & Time-Travel scrubber shine
        var st_spawn = bus.begin_mcp_tool_step(
            agent_id=scene_aid,
            tool_name="ecs.spawn_entity",
            summary="Spawned Key_Warm SpotLight",
            params_json='{"name":"Key_Warm","position":[' + String(Float64(prev_pos.x)) + "," + String(Float64(prev_pos.y)) + "," + String(Float64(prev_pos.z)) + "]}",
            entity_id=target_eid,
            world_pos=prev_pos,
            has_world_pos=True,
            half_extents=Vec3(0.55, 0.55, 0.55),
            pre_world_lz4=_str_to_bytes("FLINT_DEMO_PRE_1"),
            pre_hash=UInt64(0x1001),
            now_sec=0.4,
        )
        _ = bus.complete_mcp_tool_step(
            agent_id=scene_aid,
            step_id=st_spawn,
            ok=True,
            result_json='{"ok":true,"entity_id":' + String(target_eid) + "}",
            elapsed_ms=Float32(4.2),
            tx_id=UInt64(101),
            entity_id=target_eid,
            world_pos=prev_pos,
            has_world_pos=True,
            half_extents=Vec3(0.55, 0.55, 0.55),
            post_world_lz4=_str_to_bytes("FLINT_DEMO_POST_1"),
            post_hash=UInt64(0x1002),
        )

        var st_move = bus.begin_mcp_tool_step(
            agent_id=scene_aid,
            tool_name="entity.set_transform",
            summary="Aligned Hero Prop #" + String(target_eid),
            params_json='{"entity_id":' + String(target_eid) + ',"position":[' + String(Float64(target_pos.x)) + "," + String(Float64(target_pos.y)) + "," + String(Float64(target_pos.z)) + "]}",
            entity_id=target_eid,
            world_pos=target_pos,
            has_world_pos=True,
            prev_world_pos=prev_pos,
            has_prev_world_pos=True,
            half_extents=Vec3(0.65, 0.65, 0.65),
            pre_world_lz4=_str_to_bytes("FLINT_DEMO_POST_1"),
            pre_hash=UInt64(0x1002),
            now_sec=0.9,
        )
        _ = bus.complete_mcp_tool_step(
            agent_id=scene_aid,
            step_id=st_move,
            ok=True,
            result_json='{"ok":true,"entity_id":' + String(target_eid) + "}",
            elapsed_ms=Float32(2.6),
            tx_id=UInt64(102),
            entity_id=target_eid,
            world_pos=target_pos,
            has_world_pos=True,
            half_extents=Vec3(0.65, 0.65, 0.65),
            post_world_lz4=_str_to_bytes("FLINT_DEMO_POST_2"),
            post_hash=UInt64(0x1003),
        )

        # Pending gated step with glowing 3D ghost wireframe + approval banner
        var ghost_pos = target_pos + Vec3(Float32(1.25), Float32(0.55), Float32(-0.55))
        var st_gate = bus.begin_mcp_tool_step(
            agent_id=scene_aid,
            tool_name="entity.set_transform",
            summary="Stage Rim_Cyan Accent Light",
            params_json='{"entity_id":' + String(target_eid) + ',"position":[' + String(Float64(ghost_pos.x)) + "," + String(Float64(ghost_pos.y)) + "," + String(Float64(ghost_pos.z)) + '],"intensity":4.2}',
            entity_id=target_eid,
            world_pos=ghost_pos,
            has_world_pos=True,
            prev_world_pos=target_pos,
            has_prev_world_pos=True,
            half_extents=Vec3(0.60, 0.60, 0.60),
            pre_world_lz4=_str_to_bytes("FLINT_DEMO_POST_2"),
            pre_hash=UInt64(0x1003),
            awaiting_approval=True,
            now_sec=1.3,
        )
        var pa = PendingApproval(
            0, scene_aid, st_gate, APPROVAL_SRC_MCP_TOOL,
            "Stage Rim_Cyan Light",
            "Place accent light +1.25m X, +0.55m Y above hero prop",
        )
        pa.tool_name = "entity.set_transform"
        pa.raw_payload_json = '{"entity_id":' + String(target_eid) + ',"delta":[+1.25, +0.55, -0.55],"rim_color":[0.22,0.74,0.97]}'
        pa.ghost = AgentGhostPreview(
            kind=GHOST_TRANSFORM,
            agent_id=scene_aid,
            step_id=st_gate,
            entity_id=target_eid,
            label="Rim_Cyan",
            target_pos=ghost_pos,
            target_rot=Quat.identity(),
            half_extents=Vec3(0.60, 0.60, 0.60),
            accent_rgba8=agent_role_accent_rgba8(ROLE_SCENE_BUILDER),
            from_pos=target_pos,
            has_from_pos=True,
        )
        _ = bus.queue_approval(pa^)

        # 2. Gameplay Coder completed turn + multi-hunk Mojo diff in CodeDrawer + diagnostics
        var coder_aid = bus.create_session(ROLE_GAMEPLAY_CODER, "Gameplay Coder", APPROVAL_EDITS)
        _ = bus.begin_turn(coder_aid, "Tune coyote time & dash cooldown in hero_controller.mojo", _str_to_bytes("FLINT_DEMO_POST_2"), UInt64(0x1003), 0.5)
        var old_src = _demo_controller_old_source()
        var new_src = _demo_controller_new_source()
        var script_path = "scripts/hero_controller.mojo"
        _ = bus.record_acp_file_edit(coder_aid, script_path, old_src, new_src, True, 0.8, Float32(18.4))
        bus.end_turn(coder_aid, "end_turn", 1.1)

        _ = code_drawer.show_agent_proposal(script_path, old_src, new_src, "demo_proposal_1", split_mode=False)
        code_drawer.view_mode = CODE_MODE_DIFF_UNIFIED
        code_drawer.problems.clear()
        code_drawer.problems.add(
            Diagnostic(
                DIAG_SRC_BUILD,
                DIAG_WARNING,
                script_path,
                22,
                5,
                "dash_cooldown clamped to 0.0; consider exposing dash_duration as a comptime constant",
            )
        )
        code_drawer.problems.add(
            Diagnostic(
                DIAG_SRC_AGENT,
                DIAG_INFO,
                script_path,
                3,
                1,
                "Gameplay Coder: extracted HERO_COYOTE_SEC (0.12s) and HERO_JUMP_IMPULSE (8.4) for Inspector tuning",
            )
        )
        self.build_generation = 2
        code_drawer.set_build_status(BUILD_STATUS_OK, 2, 0, 1)

        # Keep Scene Builder selected so Mission Control opens directly onto active turn + approval banner
        bus.selected_agent_id = scene_aid

    # ---- Interactive Scripted Live Demo Playback ----

    def start_scripted_demo(
        mut self,
        mut bus: AgentTelemetryBus,
        mut code_drawer: CodeDrawer,
        mut toasts: ToastManager,
        target_eid: Int,
        target_pos: Vec3,
        prev_pos: Vec3,
    ):
        """Starts a deterministic live multi-step visual agent demo playable in real time."""
        bus = AgentTelemetryBus()
        bus.theatre_overlays_enabled = True
        self.demo = ScriptedDemoState()
        self.demo.active = True
        self.demo.elapsed_sec = Float32(0.0)
        self.demo.stage = 0
        self.demo.target_eid = target_eid
        self.demo.target_pos = target_pos
        self.demo.prev_pos = prev_pos
        self.demo.scene_agent_id = bus.create_session(ROLE_SCENE_BUILDER, "Scene Builder", APPROVAL_STEP)
        self.demo.coder_agent_id = bus.create_session(ROLE_GAMEPLAY_CODER, "Gameplay Coder", APPROVAL_EDITS)
        bus.selected_agent_id = self.demo.scene_agent_id
        _ = code_drawer.open_source("scripts/hero_controller.mojo", _demo_controller_old_source(), _demo_controller_old_source())
        _ = toasts.push(TOAST_AGENT, "Visual Agent Demo Started", "Watch Scene Builder & Gameplay Coder live")

    def tick_scripted_demo(
        mut self,
        dt: Float32,
        mut bus: AgentTelemetryBus,
        mut code_drawer: CodeDrawer,
        mut toasts: ToastManager,
    ) -> Bool:
        """Advances scripted demo playback by `dt`; returns True while demo is still progressing."""
        if not self.demo.active:
            return False
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(0.1))
        self.demo.elapsed_sec += safe_dt
        var t = self.demo.elapsed_sec

        if self.demo.stage == 0:
            _ = bus.begin_turn(self.demo.scene_agent_id, "Stage hero lighting & align prop", _str_to_bytes("DEMO_T0"), UInt64(5001), Float64(t))
            _ = bus.record_acp_thought(self.demo.scene_agent_id, "Scanning scene bounds & selecting hero focal point...", Float64(t))
            var plan = List[String]()
            plan.append("[in_progress] Spawn Key_Warm spotlight")
            plan.append("[pending] Align hero prop transform")
            plan.append("[pending] Propose Rim_Cyan accent placement")
            plan.append("[pending] Tune hero controller & hot-reload")
            _ = bus.record_acp_plan(self.demo.scene_agent_id, plan, Float64(t))
            self.demo.stage = 1
            return True

        if self.demo.stage == 1 and t >= Float32(0.6):
            var sid = bus.begin_mcp_tool_step(
                self.demo.scene_agent_id, "ecs.spawn_entity", "Spawned Key_Warm SpotLight",
                '{"name":"Key_Warm"}', entity_id=self.demo.target_eid,
                world_pos=self.demo.prev_pos, has_world_pos=True,
                pre_world_lz4=_str_to_bytes("DEMO_T0"), pre_hash=UInt64(5001), now_sec=Float64(t),
            )
            _ = bus.complete_mcp_tool_step(
                self.demo.scene_agent_id, sid, True, '{"ok":true}', Float32(3.4), UInt64(201),
                entity_id=self.demo.target_eid, world_pos=self.demo.prev_pos, has_world_pos=True,
                post_world_lz4=_str_to_bytes("DEMO_T1"), post_hash=UInt64(5002),
            )
            self.demo.spawn_step_id = sid
            _ = toasts.push(TOAST_AGENT, "Spawned Key_Warm SpotLight", "Scene Builder")
            self.demo.stage = 2
            return True

        if self.demo.stage == 2 and t >= Float32(1.4):
            var plan = List[String]()
            plan.append("[completed] Spawn Key_Warm spotlight")
            plan.append("[completed] Align hero prop transform")
            plan.append("[in_progress] Propose Rim_Cyan accent placement")
            plan.append("[pending] Tune hero controller & hot-reload")
            _ = bus.record_acp_plan(self.demo.scene_agent_id, plan, Float64(t))
            var sid = bus.begin_mcp_tool_step(
                self.demo.scene_agent_id, "entity.set_transform", "Aligned Hero Prop",
                '{"entity_id":' + String(self.demo.target_eid) + "}",
                entity_id=self.demo.target_eid, world_pos=self.demo.target_pos, has_world_pos=True,
                prev_world_pos=self.demo.prev_pos, has_prev_world_pos=True,
                pre_world_lz4=_str_to_bytes("DEMO_T1"), pre_hash=UInt64(5002), now_sec=Float64(t),
            )
            _ = bus.complete_mcp_tool_step(
                self.demo.scene_agent_id, sid, True, '{"ok":true}', Float32(2.1), UInt64(202),
                entity_id=self.demo.target_eid, world_pos=self.demo.target_pos, has_world_pos=True,
                post_world_lz4=_str_to_bytes("DEMO_T2"), post_hash=UInt64(5003),
            )
            self.demo.move_step_id = sid
            self.demo.stage = 3
            return True

        if self.demo.stage == 3 and t >= Float32(2.2):
            var gpos = self.demo.target_pos + Vec3(Float32(1.2), Float32(0.5), Float32(-0.5))
            var sid = bus.begin_mcp_tool_step(
                self.demo.scene_agent_id, "entity.set_transform", "Stage Rim_Cyan Light",
                '{"entity_id":' + String(self.demo.target_eid) + ',"rim_intensity":4.2}',
                entity_id=self.demo.target_eid, world_pos=gpos, has_world_pos=True,
                prev_world_pos=self.demo.target_pos, has_prev_world_pos=True,
                awaiting_approval=True, now_sec=Float64(t),
            )
            var pa = PendingApproval(0, self.demo.scene_agent_id, sid, APPROVAL_SRC_MCP_TOOL, "Stage Rim_Cyan Light", "Approve ghost placement in 3D viewport")
            pa.tool_name = "entity.set_transform"
            pa.raw_payload_json = '{"delta":[+1.2,+0.5,-0.5],"rim_intensity":4.2}'
            pa.ghost = AgentGhostPreview(
                kind=GHOST_TRANSFORM, agent_id=self.demo.scene_agent_id, step_id=sid,
                entity_id=self.demo.target_eid, label="Rim_Cyan", target_pos=gpos,
                accent_rgba8=agent_role_accent_rgba8(ROLE_SCENE_BUILDER),
                from_pos=self.demo.target_pos, has_from_pos=True,
            )
            self.demo.last_approval_id = bus.queue_approval(pa^)
            _ = toasts.push(TOAST_WARN, "Approval Needed", "Stage Rim_Cyan Light", own_job=True)
            self.demo.stage = 4
            return True

        if self.demo.stage == 4 and t >= Float32(3.2):
            var old_s = _demo_controller_old_source()
            var new_s = _demo_controller_new_source()
            var spath = "scripts/hero_controller.mojo"
            _ = bus.begin_turn(self.demo.coder_agent_id, "Tune coyote time & dash cooldown", _str_to_bytes("DEMO_T2"), UInt64(5003), Float64(t))
            _ = bus.record_acp_file_edit(self.demo.coder_agent_id, spath, old_s, new_s, True, Float64(t), Float32(14.0))
            bus.end_turn(self.demo.coder_agent_id, "end_turn", Float64(t))
            _ = code_drawer.show_agent_proposal(spath, old_s, new_s, "demo_live_1", split_mode=False)
            var jid = self.jobs.add_mock_job(JOB_KIND_BUILD_SO, "Build game library", self.workspace_root)
            var ji = self.jobs.find_index(jid)
            if ji >= 0:
                self.jobs.jobs[ji].feed_mock_output("scripts/hero_controller.mojo:22:5: warning: dash_cooldown clamped\n", 0, Float64(t))
            _ = self.poll_jobs(bus, code_drawer, toasts, Float64(t))
            self.demo.stage = 5
            self.demo.active = False
            return False

        return self.demo.active

    def poll_runtime(
        mut self,
        mut bus: AgentTelemetryBus,
        mut code_drawer: CodeDrawer,
        mut toasts: ToastManager,
        now_sec: Float64,
        dt: Float32,
    ):
        """Per-frame Studio runtime pump: polls ACP, background jobs, scripted demo, and trail decay."""
        if self.acp.active():
            self.acp.pump(now_sec)
            _ = self.drain_acp_events(bus, code_drawer, toasts, now_sec)
        _ = self.poll_jobs(bus, code_drawer, toasts, now_sec)
        if self.demo.active:
            _ = self.tick_scripted_demo(dt, bus, code_drawer, toasts)
        bus.tick(dt)

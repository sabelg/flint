"""Flint Studio Visual Agent Telemetry Bus, Step History & Unified Approval Queue (`agent_bus.mojo`).

Pure-Mojo 1.2 structured event bus unifying:
  - Coding-agent ACP streams (thoughts, plans, file reads/writes, terminal runs, permissions)
  - Live Flint MCP v2 tool invocations (entity spawn/transform/patch/despawn/reparent,
    materials/lighting/NPR-cel/VFX, NavMesh bake, PIE play/step, viewport captures)

Every mutating step links cleanly to a monotonic `EditorTransaction` `tx_id` plus LZ4-compressed
pre/post world snapshots (and per-step file text snapshots) enabling:
  - 1-click individual step revert or full turn rollback
  - Non-destructive Time-Travel scrubbing across agent steps
  - Unified non-blocking Step-by-Step Approval with 3D ghost placement previews
  - Follow-Agent camera targeting and Outliner/Viewport entity activity queries
"""
from std.collections import List
from std.math import isfinite, max, min
from flint.core.math import Quat, Vec3
from flint.ui.context import pack_rgba8

# ============================================================================
# 1. Roles, Statuses, Action Kinds, Approval Policies & Constants
# ============================================================================

comptime ROLE_SCENE_BUILDER: UInt8 = 0
comptime ROLE_GAMEPLAY_CODER: UInt8 = 1
comptime ROLE_LOOKDEV_ARTIST: UInt8 = 2
comptime ROLE_QA_PLAYTESTER: UInt8 = 3
comptime ROLE_EXTERNAL_MCP: UInt8 = 4

comptime AGENT_STATUS_IDLE: UInt8 = 0
comptime AGENT_STATUS_THINKING: UInt8 = 1
comptime AGENT_STATUS_PLANNING: UInt8 = 2
comptime AGENT_STATUS_EDITING_SCENE: UInt8 = 3
comptime AGENT_STATUS_WRITING_CODE: UInt8 = 4
comptime AGENT_STATUS_BUILDING: UInt8 = 5
comptime AGENT_STATUS_TESTING_PIE: UInt8 = 6
comptime AGENT_STATUS_WAITING_APPROVAL: UInt8 = 7
comptime AGENT_STATUS_ERROR: UInt8 = 8

comptime ACT_THOUGHT: UInt8 = 0
comptime ACT_PLAN: UInt8 = 1
comptime ACT_FILE_READ: UInt8 = 2
comptime ACT_CODE_EDIT: UInt8 = 3
comptime ACT_BUILD_HOTRELOAD: UInt8 = 4
comptime ACT_TERMINAL: UInt8 = 5
comptime ACT_ENTITY_SPAWN: UInt8 = 6
comptime ACT_ENTITY_TRANSFORM: UInt8 = 7
comptime ACT_ENTITY_PATCH: UInt8 = 8
comptime ACT_ENTITY_DESPAWN: UInt8 = 9
comptime ACT_ENTITY_REPARENT: UInt8 = 10
comptime ACT_MATERIAL_STYLE: UInt8 = 11
comptime ACT_LIGHT_ENV: UInt8 = 12
comptime ACT_NAVMESH_BAKE: UInt8 = 13
comptime ACT_PIE_CONTROL: UInt8 = 14
comptime ACT_VIEWPORT_CAPTURE: UInt8 = 15
comptime ACT_ASSET_PROCEDURAL: UInt8 = 16
comptime ACT_OTHER: UInt8 = 17

comptime STEP_PENDING: UInt8 = 0
comptime STEP_AWAITING_APPROVAL: UInt8 = 1
comptime STEP_RUNNING: UInt8 = 2
comptime STEP_COMPLETED: UInt8 = 3
comptime STEP_FAILED: UInt8 = 4
comptime STEP_REJECTED: UInt8 = 5
comptime STEP_REVERTED: UInt8 = 6

comptime STEP_SOURCE_ACP: UInt8 = 0
comptime STEP_SOURCE_MCP: UInt8 = 1

comptime APPROVAL_AUTO: UInt8 = 0
comptime APPROVAL_EDITS: UInt8 = 1
comptime APPROVAL_STEP: UInt8 = 2

comptime APPROVAL_SRC_MCP_TOOL: UInt8 = 0
comptime APPROVAL_SRC_ACP_PERM: UInt8 = 1
comptime APPROVAL_SRC_ACP_WRITE: UInt8 = 2
comptime APPROVAL_SRC_ACP_RUN: UInt8 = 3

comptime GHOST_NONE: UInt8 = 0
comptime GHOST_SPAWN: UInt8 = 1
comptime GHOST_TRANSFORM: UInt8 = 2
comptime GHOST_DESPAWN: UInt8 = 3
comptime GHOST_MATERIAL: UInt8 = 4

comptime MAX_TRAIL_MARKS: Int = 32
comptime MAX_STEPS_PER_SESSION: Int = 512
comptime TRAIL_HEAT_DURATION_SEC: Float32 = 4.0
comptime TRAIL_ARC_DURATION_SEC: Float32 = 3.5
comptime FOLLOW_SETTLE_HOLD_SEC: Float32 = 0.45


def agent_role_name(role: UInt8) -> String:
    if role == ROLE_SCENE_BUILDER:
        return "Scene Builder"
    if role == ROLE_GAMEPLAY_CODER:
        return "Gameplay Coder"
    if role == ROLE_LOOKDEV_ARTIST:
        return "LookDev Artist"
    if role == ROLE_QA_PLAYTESTER:
        return "QA Playtester"
    return "External MCP"


def agent_role_short_tag(role: UInt8) -> String:
    if role == ROLE_SCENE_BUILDER:
        return "SCENE"
    if role == ROLE_GAMEPLAY_CODER:
        return "CODE"
    if role == ROLE_LOOKDEV_ARTIST:
        return "LOOKDEV"
    if role == ROLE_QA_PLAYTESTER:
        return "QA"
    return "MCP"


def agent_role_accent_rgba8(role: UInt8) -> UInt32:
    if role == ROLE_SCENE_BUILDER:
        return pack_rgba8(56, 189, 248, 255)   # Cyan #38BDF8
    if role == ROLE_GAMEPLAY_CODER:
        return pack_rgba8(167, 139, 250, 255)  # Electric Violet #A78BFA
    if role == ROLE_LOOKDEV_ARTIST:
        return pack_rgba8(245, 158, 11, 255)   # Warm Amber #F59E0B
    if role == ROLE_QA_PLAYTESTER:
        return pack_rgba8(52, 211, 153, 255)   # Emerald #34D399
    return pack_rgba8(96, 165, 250, 255)       # Slate Blue #60A5FA


def agent_status_label(status: UInt8) -> String:
    if status == AGENT_STATUS_IDLE:
        return "Idle"
    if status == AGENT_STATUS_THINKING:
        return "Thinking"
    if status == AGENT_STATUS_PLANNING:
        return "Planning"
    if status == AGENT_STATUS_EDITING_SCENE:
        return "Editing Scene"
    if status == AGENT_STATUS_WRITING_CODE:
        return "Writing Code"
    if status == AGENT_STATUS_BUILDING:
        return "Building"
    if status == AGENT_STATUS_TESTING_PIE:
        return "Testing PIE"
    if status == AGENT_STATUS_WAITING_APPROVAL:
        return "Waiting Approval"
    return "Error"


def action_kind_label(kind: UInt8) -> String:
    if kind == ACT_THOUGHT:
        return "Thought"
    if kind == ACT_PLAN:
        return "Plan"
    if kind == ACT_FILE_READ:
        return "Read File"
    if kind == ACT_CODE_EDIT:
        return "Edit Code"
    if kind == ACT_BUILD_HOTRELOAD:
        return "Build & Reload"
    if kind == ACT_TERMINAL:
        return "Terminal"
    if kind == ACT_ENTITY_SPAWN:
        return "Spawn Entity"
    if kind == ACT_ENTITY_TRANSFORM:
        return "Transform"
    if kind == ACT_ENTITY_PATCH:
        return "Patch Entity"
    if kind == ACT_ENTITY_DESPAWN:
        return "Despawn"
    if kind == ACT_ENTITY_REPARENT:
        return "Reparent"
    if kind == ACT_MATERIAL_STYLE:
        return "Material / Style"
    if kind == ACT_LIGHT_ENV:
        return "Light / Env"
    if kind == ACT_NAVMESH_BAKE:
        return "NavMesh"
    if kind == ACT_PIE_CONTROL:
        return "PIE Control"
    if kind == ACT_VIEWPORT_CAPTURE:
        return "Capture"
    if kind == ACT_ASSET_PROCEDURAL:
        return "Asset / Forge"
    return "Action"


def action_kind_from_mcp_tool(tool_name: String) -> UInt8:
    if tool_name == "ecs.spawn_entity" or tool_name == "prefab.spawn":
        return ACT_ENTITY_SPAWN
    if tool_name == "entity.set_transform":
        return ACT_ENTITY_TRANSFORM
    if tool_name == "ecs.patch_components" or tool_name == "scene.set_entity_enabled":
        return ACT_ENTITY_PATCH
    if tool_name == "ecs.despawn_entity":
        return ACT_ENTITY_DESPAWN
    if tool_name == "material.assign" or tool_name == "render.configure_npr_cel":
        return ACT_MATERIAL_STYLE
    if tool_name == "render.trigger_post_fx":
        return ACT_LIGHT_ENV
    if tool_name == "nav.query_path":
        return ACT_NAVMESH_BAKE
    if (
        tool_name == "studio.pie_control"
        or tool_name == "physics.step_deterministic"
        or tool_name == "physics.trigger_events"
        or tool_name == "anim.set_state"
        or tool_name == "audio.play"
    ):
        return ACT_PIE_CONTROL
    if tool_name == "render.capture_viewport" or tool_name == "render.inspect_gpu_frame":
        return ACT_VIEWPORT_CAPTURE
    if (
        tool_name == "procedural.generate_asset"
        or tool_name == "assets.list_and_cook"
        or tool_name == "asset.import"
        or tool_name == "asset.list"
        or tool_name == "project.package"
    ):
        return ACT_ASSET_PROCEDURAL
    return ACT_OTHER


def action_kind_is_mutating(kind: UInt8) -> Bool:
    return (
        kind == ACT_CODE_EDIT
        or kind == ACT_ENTITY_SPAWN
        or kind == ACT_ENTITY_TRANSFORM
        or kind == ACT_ENTITY_PATCH
        or kind == ACT_ENTITY_DESPAWN
        or kind == ACT_ENTITY_REPARENT
        or kind == ACT_MATERIAL_STYLE
        or kind == ACT_LIGHT_ENV
        or kind == ACT_NAVMESH_BAKE
        or kind == ACT_ASSET_PROCEDURAL
    )


def action_kind_is_destructive(kind: UInt8, tool_name: String = "") -> Bool:
    if kind == ACT_ENTITY_DESPAWN:
        return True
    if tool_name == "scene.load" or tool_name == "project.package":
        return True
    return False


def _status_for_action(kind: UInt8) -> UInt8:
    if kind == ACT_THOUGHT:
        return AGENT_STATUS_THINKING
    if kind == ACT_PLAN:
        return AGENT_STATUS_PLANNING
    if kind == ACT_CODE_EDIT or kind == ACT_FILE_READ:
        return AGENT_STATUS_WRITING_CODE
    if kind == ACT_BUILD_HOTRELOAD or kind == ACT_TERMINAL:
        return AGENT_STATUS_BUILDING
    if kind == ACT_PIE_CONTROL or kind == ACT_VIEWPORT_CAPTURE:
        return AGENT_STATUS_TESTING_PIE
    if (
        kind == ACT_ENTITY_SPAWN
        or kind == ACT_ENTITY_TRANSFORM
        or kind == ACT_ENTITY_PATCH
        or kind == ACT_ENTITY_DESPAWN
        or kind == ACT_ENTITY_REPARENT
        or kind == ACT_MATERIAL_STYLE
        or kind == ACT_LIGHT_ENV
        or kind == ACT_NAVMESH_BAKE
        or kind == ACT_ASSET_PROCEDURAL
    ):
        return AGENT_STATUS_EDITING_SCENE
    return AGENT_STATUS_THINKING


@always_inline
def _safe_vec3(v: Vec3, fallback: Vec3 = Vec3()) -> Vec3:
    if isfinite(v.x) and isfinite(v.y) and isfinite(v.z):
        return v
    return fallback


@always_inline
def _safe_half_extents(he: Vec3) -> Vec3:
    var sx = he.x if (isfinite(he.x) and he.x > Float32(0.05)) else Float32(0.5)
    var sy = he.y if (isfinite(he.y) and he.y > Float32(0.05)) else Float32(0.5)
    var sz = he.z if (isfinite(he.z) and he.z > Float32(0.05)) else Float32(0.5)
    return Vec3(sx, sy, sz)


@always_inline
def _safe_quat(q: Quat) -> Quat:
    var lsq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
    if isfinite(lsq) and lsq > Float32(1.0e-8):
        return q.normalized()
    return Quat.identity()


# ============================================================================
# 2. Ghost Previews, Trail Marks, Steps, Turns, Approvals & Sessions
# ============================================================================

struct AgentGhostPreview(Copyable, Movable):
    """Pre-commit 3D ghost placement / transform / despawn preview shown during approval."""
    var kind: UInt8
    var agent_id: UInt32
    var step_id: UInt64
    var entity_id: Int
    var editor_id: UInt64
    """Stable `editor_id` of `entity_id` when the preview was queued (0 = not stamped yet)."""
    var label: String
    var from_pos: Vec3
    var has_from_pos: Bool
    var target_pos: Vec3
    var target_rot: Quat
    var half_extents: Vec3
    var accent_rgba8: UInt32

    def __init__(
        out self,
        kind: UInt8 = GHOST_NONE,
        agent_id: UInt32 = 0,
        step_id: UInt64 = 0,
        entity_id: Int = -1,
        label: String = "",
        target_pos: Vec3 = Vec3(),
        target_rot: Quat = Quat.identity(),
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        accent_rgba8: UInt32 = 0xFF38BDF8,
        from_pos: Vec3 = Vec3(),
        has_from_pos: Bool = False,
    ):
        self.kind = kind
        self.agent_id = agent_id
        self.step_id = step_id
        self.entity_id = entity_id
        self.editor_id = UInt64(0)
        self.label = label
        self.from_pos = _safe_vec3(from_pos)
        self.has_from_pos = has_from_pos
        self.target_pos = _safe_vec3(target_pos)
        self.target_rot = _safe_quat(target_rot)
        self.half_extents = _safe_half_extents(half_extents)
        self.accent_rgba8 = accent_rgba8


struct AgentTrailMark(Copyable, ImplicitlyCopyable, Movable):
    """Decaying footprint / breadcrumb record for recently modified entities in the 3D viewport."""
    var entity_id: Int
    var agent_id: UInt32
    var step_id: UInt64
    var action_kind: UInt8
    var world_pos: Vec3
    var prev_world_pos: Vec3
    var has_prev_pos: Bool
    var half_extents: Vec3
    var rotation: Quat
    var accent_rgba8: UInt32
    var created_sec: Float32
    var last_touched_sec: Float32
    var heat: Float32

    def __init__(
        out self,
        entity_id: Int = -1,
        agent_id: UInt32 = 0,
        step_id: UInt64 = 0,
        action_kind: UInt8 = ACT_ENTITY_PATCH,
        world_pos: Vec3 = Vec3(),
        prev_world_pos: Vec3 = Vec3(),
        has_prev_pos: Bool = False,
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        rotation: Quat = Quat.identity(),
        accent_rgba8: UInt32 = 0xFF38BDF8,
        created_sec: Float32 = 0.0,
    ):
        self.entity_id = entity_id
        self.agent_id = agent_id
        self.step_id = step_id
        self.action_kind = action_kind
        self.world_pos = _safe_vec3(world_pos)
        self.prev_world_pos = _safe_vec3(prev_world_pos)
        self.has_prev_pos = has_prev_pos
        self.half_extents = _safe_half_extents(half_extents)
        self.rotation = _safe_quat(rotation)
        self.accent_rgba8 = accent_rgba8
        self.created_sec = created_sec
        self.last_touched_sec = created_sec
        self.heat = Float32(1.0)


struct AgentStep(Copyable, Movable):
    """Single structured step captured from ACP or MCP v2 with exact undo/checkpoint links."""
    var step_id: UInt64
    var agent_id: UInt32
    var turn_id: UInt32
    var source: UInt8
    var external_call_id: String
    var kind: UInt8
    var status: UInt8
    var title: String
    var summary: String
    var started_sec: Float64
    var elapsed_ms: Float32
    var progress: Float32

    var affected_entity_ids: List[Int]
    var world_pos: Vec3
    var has_world_pos: Bool
    var prev_world_pos: Vec3
    var has_prev_world_pos: Bool
    var entity_half_extents: Vec3
    var entity_rotation: Quat

    var file_path: String
    var file_existed_before: Bool
    var old_text: String
    var new_text: String

    var params_json: String
    var result_json: String
    var terminal_tail: String
    var capture_png_path: String
    var flip_score: Float32

    var tx_id: UInt64
    var pre_world_lz4: List[UInt8]
    var post_world_lz4: List[UInt8]
    var pre_hash: UInt64
    var post_hash: UInt64
    var expanded: Bool

    def __init__(
        out self,
        step_id: UInt64 = 0,
        agent_id: UInt32 = 0,
        turn_id: UInt32 = 0,
        source: UInt8 = STEP_SOURCE_MCP,
        kind: UInt8 = ACT_OTHER,
        title: String = "",
        summary: String = "",
    ):
        self.step_id = step_id
        self.agent_id = agent_id
        self.turn_id = turn_id
        self.source = source
        self.external_call_id = String()
        self.kind = kind
        self.status = STEP_PENDING
        self.title = title
        self.summary = summary
        self.started_sec = 0.0
        self.elapsed_ms = Float32(0.0)
        self.progress = Float32(0.0)

        self.affected_entity_ids = List[Int]()
        self.world_pos = Vec3()
        self.has_world_pos = False
        self.prev_world_pos = Vec3()
        self.has_prev_world_pos = False
        self.entity_half_extents = Vec3(0.5, 0.5, 0.5)
        self.entity_rotation = Quat.identity()

        self.file_path = String()
        self.file_existed_before = False
        self.old_text = String()
        self.new_text = String()

        self.params_json = String()
        self.result_json = String()
        self.terminal_tail = String()
        self.capture_png_path = String()
        self.flip_score = Float32(-1.0)

        self.tx_id = 0
        self.pre_world_lz4 = List[UInt8]()
        self.post_world_lz4 = List[UInt8]()
        self.pre_hash = 0
        self.post_hash = 0
        self.expanded = False

    def primary_entity_id(self) -> Int:
        if len(self.affected_entity_ids) > 0:
            return self.affected_entity_ids[0]
        return -1

    def is_revertible(self) -> Bool:
        if self.status != STEP_COMPLETED:
            return False
        if len(self.pre_world_lz4) > 0 or self.tx_id > 0:
            return True
        if self.kind == ACT_CODE_EDIT and self.file_path.byte_length() > 0:
            return True
        return False


struct AgentTurn(Copyable, Movable):
    """Logical user-prompt turn grouping all ACP + MCP steps and pre-turn world checkpoint."""
    var turn_id: UInt32
    var agent_id: UInt32
    var prompt: String
    var started_sec: Float64
    var ended_sec: Float64
    var stop_reason: String
    var active: Bool
    var step_ids: List[UInt64]
    var pre_world_lz4: List[UInt8]
    var pre_hash: UInt64
    var reverted: Bool

    def __init__(
        out self,
        turn_id: UInt32 = 0,
        agent_id: UInt32 = 0,
        prompt: String = "",
        started_sec: Float64 = 0.0,
    ):
        self.turn_id = turn_id
        self.agent_id = agent_id
        self.prompt = prompt
        self.started_sec = started_sec
        self.ended_sec = 0.0
        self.stop_reason = String()
        self.active = True
        self.step_ids = List[UInt64]()
        self.pre_world_lz4 = List[UInt8]()
        self.pre_hash = 0
        self.reverted = False


struct PendingApproval(Copyable, Movable):
    """Unified deferred approval item for both ACP permission/write/run and gated MCP calls."""
    var approval_id: UInt64
    var agent_id: UInt32
    var step_id: UInt64
    var source: UInt8
    var title: String
    var summary: String
    var tool_name: String
    var tool_kind: String

    var mcp_client_key: Int
    """MCP connection whose gated call this is; an approval grants the retry to that client only."""
    # Deferred MCP socket response framing (for APPROVAL_SRC_MCP_TOOL)
    var client_fd: Int
    var request_id_raw: String
    var use_content_length: Bool
    var raw_payload_json: String

    # Deferred ACP proposal / permission payload
    var file_path: String
    var old_text: String
    var new_text: String
    var file_existed: Bool
    var option_allow_once: String
    var option_allow_always: String
    var option_reject: String

    var ghost: AgentGhostPreview
    var created_sec: Float64
    var resolved: Bool
    var approved: Bool
    var allow_turn: Bool
    var reject_reason: String

    def __init__(
        out self,
        approval_id: UInt64 = 0,
        agent_id: UInt32 = 0,
        step_id: UInt64 = 0,
        source: UInt8 = APPROVAL_SRC_MCP_TOOL,
        title: String = "",
        summary: String = "",
    ):
        self.approval_id = approval_id
        self.agent_id = agent_id
        self.step_id = step_id
        self.source = source
        self.title = title
        self.summary = summary
        self.tool_name = String()
        self.tool_kind = String()
        self.mcp_client_key = 0
        self.client_fd = -1
        self.request_id_raw = String()
        self.use_content_length = False
        self.raw_payload_json = String()
        self.file_path = String()
        self.old_text = String()
        self.new_text = String()
        self.file_existed = False
        self.option_allow_once = String()
        self.option_allow_always = String()
        self.option_reject = String()
        self.ghost = AgentGhostPreview()
        self.created_sec = 0.0
        self.resolved = False
        self.approved = False
        self.allow_turn = False
        self.reject_reason = String()


struct AgentRevertPlan(Copyable, Movable):
    """Self-contained revert instructions executed by Studio state (world blob + file rollbacks)."""
    var valid: Bool
    var is_turn_revert: Bool
    var agent_id: UInt32
    var step_id: UInt64
    var turn_id: UInt32
    var target_tx_id: UInt64
    var restore_world_lz4: List[UInt8]
    var expected_hash: UInt64
    var file_paths: List[String]
    var file_existed: List[Bool]
    var file_old_texts: List[String]
    var reverted_step_ids: List[UInt64]
    var reason: String

    def __init__(out self):
        self.valid = False
        self.is_turn_revert = False
        self.agent_id = 0
        self.step_id = 0
        self.turn_id = 0
        self.target_tx_id = 0
        self.restore_world_lz4 = List[UInt8]()
        self.expected_hash = 0
        self.file_paths = List[String]()
        self.file_existed = List[Bool]()
        self.file_old_texts = List[String]()
        self.reverted_step_ids = List[UInt64]()
        self.reason = String()


struct AgentSession(Copyable, Movable):
    """Single in-Studio agent instance with role preset, timeline steps, and active 3D target."""
    var agent_id: UInt32
    var role: UInt8
    var name: String
    var accent_rgba8: UInt32
    var status: UInt8
    var approval_policy: UInt8
    var auto_approve_turn_id: UInt32
    var active_turn_id: UInt32
    var mcp_auth_token: String
    var acp_session_id: String
    var plan_entries: List[String]
    var turns: List[AgentTurn]
    var steps: List[AgentStep]

    var active_entity_id: Int
    var active_world_pos: Vec3
    var has_active_world_pos: Bool
    var active_half_extents: Vec3
    var active_rotation: Quat
    var active_callout: String
    var active_progress: Float32
    var last_event_sec: Float32
    var paused: Bool

    def __init__(
        out self,
        agent_id: UInt32 = 1,
        role: UInt8 = ROLE_SCENE_BUILDER,
        name: String = "",
        approval_policy: UInt8 = APPROVAL_EDITS,
    ):
        self.agent_id = agent_id
        self.role = role
        self.name = name if name.byte_length() > 0 else agent_role_name(role)
        self.accent_rgba8 = agent_role_accent_rgba8(role)
        self.status = AGENT_STATUS_IDLE
        self.approval_policy = approval_policy
        self.auto_approve_turn_id = 0
        self.active_turn_id = 0
        self.mcp_auth_token = "flint-agent-" + String(Int(agent_id))
        self.acp_session_id = String()
        self.plan_entries = List[String]()
        self.turns = List[AgentTurn]()
        self.steps = List[AgentStep]()

        self.active_entity_id = -1
        self.active_world_pos = Vec3()
        self.has_active_world_pos = False
        self.active_half_extents = Vec3(0.5, 0.5, 0.5)
        self.active_rotation = Quat.identity()
        self.active_callout = String()
        self.active_progress = Float32(0.0)
        self.last_event_sec = Float32(0.0)
        self.paused = False

    def find_step_index(self, step_id: UInt64) -> Int:
        for i in range(len(self.steps) - 1, -1, -1):
            if self.steps[i].step_id == step_id:
                return i
        return -1

    def find_step_by_external_id(self, external_call_id: String) -> Int:
        if external_call_id.byte_length() == 0:
            return -1
        for i in range(len(self.steps) - 1, -1, -1):
            if self.steps[i].external_call_id == external_call_id:
                return i
        return -1

    def find_turn_index(self, turn_id: UInt32) -> Int:
        for i in range(len(self.turns) - 1, -1, -1):
            if self.turns[i].turn_id == turn_id:
                return i
        return -1

    def is_busy(self) -> Bool:
        return self.status != AGENT_STATUS_IDLE and self.status != AGENT_STATUS_ERROR


# ============================================================================
# 3. `AgentTelemetryBus`: Unified Event, Approval, Time-Travel & Trail Engine
# ============================================================================

struct AgentTelemetryBus(Copyable, Movable):
    """Central Studio AI telemetry bus feeding Mission Control UI, Outliner badges, and 3D Theatre."""
    var sessions: List[AgentSession]
    var selected_agent_id: UInt32
    var pending_approvals: List[PendingApproval]
    var trails: List[AgentTrailMark]

    var next_agent_id: UInt32
    var next_step_id: UInt64
    var next_turn_id: UInt32
    var next_approval_id: UInt64
    var event_seq: UInt64

    # Time-travel scrub state (non-destructive preview over historical step snapshots)
    var time_travel_active: Bool
    var time_travel_agent_id: UInt32
    var time_travel_step_idx: Int
    var live_tip_lz4: List[UInt8]
    var live_tip_hash: UInt64

    # Follow-Agent camera targeting (drives Studio CameraRig smoothly, auto-yields on user input)
    var follow_agent_camera: Bool
    var follow_paused_by_user: Bool
    var focus_target_pos: Vec3
    var focus_target_radius: Float32
    var has_focus_target: Bool
    var focus_entity_id: Int
    var focus_settle_timer_sec: Float32

    # Interactive visual theatre state
    var theatre_overlays_enabled: Bool
    var hovered_step_id: UInt64
    var hovered_entity_id: Int
    var hovered_entity_pos: Vec3
    var hovered_entity_half_extents: Vec3
    var hovered_entity_rot: Quat
    var has_hovered_entity: Bool
    var theatre_time_sec: Float32

    def __init__(out self):
        self.sessions = List[AgentSession]()
        self.selected_agent_id = 0
        self.pending_approvals = List[PendingApproval]()
        self.trails = List[AgentTrailMark]()

        self.next_agent_id = 1
        self.next_step_id = 1
        self.next_turn_id = 1
        self.next_approval_id = 1
        self.event_seq = 0

        self.time_travel_active = False
        self.time_travel_agent_id = 0
        self.time_travel_step_idx = -1
        self.live_tip_lz4 = List[UInt8]()
        self.live_tip_hash = 0

        self.follow_agent_camera = True
        self.follow_paused_by_user = False
        self.focus_target_pos = Vec3()
        self.focus_target_radius = Float32(1.5)
        self.has_focus_target = False
        self.focus_entity_id = -1
        self.focus_settle_timer_sec = Float32(0.0)

        self.theatre_overlays_enabled = True
        self.hovered_step_id = 0
        self.hovered_entity_id = -1
        self.hovered_entity_pos = Vec3()
        self.hovered_entity_half_extents = Vec3(0.5, 0.5, 0.5)
        self.hovered_entity_rot = Quat.identity()
        self.has_hovered_entity = False
        self.theatre_time_sec = Float32(0.0)

    # ---- Session Management ----

    def create_session(
        mut self,
        role: UInt8,
        name: String = "",
        approval_policy: UInt8 = APPROVAL_EDITS,
    ) -> UInt32:
        var aid = self.next_agent_id
        self.next_agent_id += 1
        var sess = AgentSession(aid, role, name, approval_policy)
        sess.last_event_sec = self.theatre_time_sec
        self.sessions.append(sess^)
        if self.selected_agent_id == 0:
            self.selected_agent_id = aid
        self.event_seq += 1
        return aid

    def find_session_index(self, agent_id: UInt32) -> Int:
        for i in range(len(self.sessions)):
            if self.sessions[i].agent_id == agent_id:
                return i
        return -1

    def find_session_by_token(self, token: String) -> Int:
        if token.byte_length() == 0:
            return -1
        for i in range(len(self.sessions)):
            if self.sessions[i].mcp_auth_token == token:
                return i
        return -1

    def ensure_external_mcp_session(mut self) -> UInt32:
        for i in range(len(self.sessions)):
            if self.sessions[i].role == ROLE_EXTERNAL_MCP:
                return self.sessions[i].agent_id
        return self.create_session(ROLE_EXTERNAL_MCP, "External MCP", APPROVAL_AUTO)

    def set_session_policy(mut self, agent_id: UInt32, policy: UInt8):
        var si = self.find_session_index(agent_id)
        if si >= 0:
            self.sessions[si].approval_policy = policy
            self.event_seq += 1

    def set_session_paused(mut self, agent_id: UInt32, paused: Bool):
        var si = self.find_session_index(agent_id)
        if si >= 0:
            self.sessions[si].paused = paused
            self.event_seq += 1

    def has_active_visuals(self) -> Bool:
        """Fast check so idle Studio emits zero 3D overlay or HUD primitives."""
        if not self.theatre_overlays_enabled:
            return False
        if self.time_travel_active or self.has_hovered_entity:
            return True
        if len(self.trails) > 0:
            return True
        for i in range(len(self.pending_approvals)):
            if not self.pending_approvals[i].resolved:
                return True
        for i in range(len(self.sessions)):
            ref s = self.sessions[i]
            if s.is_busy() or s.has_active_world_pos:
                return True
        return False

    # ---- Turn Lifecycle ----

    def begin_turn(
        mut self,
        agent_id: UInt32,
        prompt: String,
        pre_world_lz4: List[UInt8] = List[UInt8](),
        pre_hash: UInt64 = 0,
        now_sec: Float64 = 0.0,
    ) -> UInt32:
        var si = self.find_session_index(agent_id)
        if si < 0:
            return 0
        var tid = self.next_turn_id
        self.next_turn_id += 1
        var turn = AgentTurn(tid, agent_id, prompt, now_sec)
        turn.pre_world_lz4 = pre_world_lz4.copy()
        turn.pre_hash = pre_hash
        self.sessions[si].turns.append(turn^)
        self.sessions[si].active_turn_id = tid
        self.sessions[si].auto_approve_turn_id = 0
        self.sessions[si].status = AGENT_STATUS_THINKING
        self.sessions[si].active_callout = "Thinking..."
        self.sessions[si].active_progress = Float32(0.0)
        self.sessions[si].last_event_sec = self.theatre_time_sec
        self.event_seq += 1
        return tid

    def end_turn(
        mut self,
        agent_id: UInt32,
        stop_reason: String = "end_turn",
        now_sec: Float64 = 0.0,
    ):
        var si = self.find_session_index(agent_id)
        if si < 0:
            return
        var tid = self.sessions[si].active_turn_id
        var ti = self.sessions[si].find_turn_index(tid)
        if ti >= 0:
            self.sessions[si].turns[ti].active = False
            self.sessions[si].turns[ti].ended_sec = now_sec
            self.sessions[si].turns[ti].stop_reason = stop_reason
        self.sessions[si].status = AGENT_STATUS_ERROR if stop_reason == "error" else AGENT_STATUS_IDLE
        self.sessions[si].has_active_world_pos = False
        self.sessions[si].active_entity_id = -1
        self.sessions[si].active_callout = String()
        self.sessions[si].active_progress = Float32(0.0)
        self.sessions[si].last_event_sec = self.theatre_time_sec
        self.event_seq += 1

    # ---- Step Recording Helpers (Internal + Public ACP & MCP Hooks) ----

    def _append_step(mut self, si: Int, var step: AgentStep) -> UInt64:
        var sid = step.step_id
        var tid = step.turn_id
        self.sessions[si].steps.append(step^)
        if len(self.sessions[si].steps) > MAX_STEPS_PER_SESSION:
            var drop = len(self.sessions[si].steps) - MAX_STEPS_PER_SESSION
            var kept = List[AgentStep](capacity=MAX_STEPS_PER_SESSION)
            for i in range(drop, len(self.sessions[si].steps)):
                kept.append(self.sessions[si].steps[i].copy())
            self.sessions[si].steps = kept^
        var ti = self.sessions[si].find_turn_index(tid)
        if ti >= 0:
            self.sessions[si].turns[ti].step_ids.append(sid)
        self.sessions[si].last_event_sec = self.theatre_time_sec
        self.event_seq += 1
        return sid

    def _update_follow_focus(mut self, entity_id: Int, world_pos: Vec3, half_extents: Vec3):
        var safe_p = _safe_vec3(world_pos)
        var he = _safe_half_extents(half_extents)
        var rad = max(Float32(0.75), max(he.x, max(he.y, he.z)) * Float32(1.8))
        self.focus_entity_id = entity_id
        self.focus_target_pos = safe_p
        self.focus_target_radius = rad
        self.has_focus_target = True
        self.focus_settle_timer_sec = FOLLOW_SETTLE_HOLD_SEC

    def record_trail_mark(
        mut self,
        entity_id: Int,
        agent_id: UInt32,
        step_id: UInt64,
        action_kind: UInt8,
        world_pos: Vec3,
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        rotation: Quat = Quat.identity(),
        prev_world_pos: Vec3 = Vec3(),
        has_prev_pos: Bool = False,
    ):
        var accent = pack_rgba8(56, 189, 248, 255)
        var si = self.find_session_index(agent_id)
        if si >= 0:
            accent = self.sessions[si].accent_rgba8
        var now_t = self.theatre_time_sec
        # Update existing mark for same entity so trails stay clean
        if entity_id >= 0:
            for i in range(len(self.trails)):
                if self.trails[i].entity_id == entity_id:
                    var old_p = self.trails[i].world_pos
                    self.trails[i].agent_id = agent_id
                    self.trails[i].step_id = step_id
                    self.trails[i].action_kind = action_kind
                    self.trails[i].world_pos = _safe_vec3(world_pos)
                    if has_prev_pos:
                        self.trails[i].prev_world_pos = _safe_vec3(prev_world_pos)
                        self.trails[i].has_prev_pos = True
                    else:
                        self.trails[i].prev_world_pos = old_p
                        self.trails[i].has_prev_pos = (self.trails[i].world_pos - old_p).length_squared() > Float32(1.0e-4)
                    self.trails[i].half_extents = _safe_half_extents(half_extents)
                    self.trails[i].rotation = _safe_quat(rotation)
                    self.trails[i].accent_rgba8 = accent
                    self.trails[i].last_touched_sec = now_t
                    self.trails[i].heat = Float32(1.0)
                    return
        var mark = AgentTrailMark(
            entity_id=entity_id,
            agent_id=agent_id,
            step_id=step_id,
            action_kind=action_kind,
            world_pos=world_pos,
            prev_world_pos=prev_world_pos,
            has_prev_pos=has_prev_pos,
            half_extents=half_extents,
            rotation=rotation,
            accent_rgba8=accent,
            created_sec=now_t,
        )
        self.trails.append(mark)
        if len(self.trails) > MAX_TRAIL_MARKS:
            _ = self.trails.pop(0)

    # ---- ACP Event Ingestion ----

    def record_acp_thought(mut self, agent_id: UInt32, chunk: String, now_sec: Float64 = 0.0) -> UInt64:
        var si = self.find_session_index(agent_id)
        if si < 0:
            return 0
        self.sessions[si].status = AGENT_STATUS_THINKING
        self.sessions[si].active_callout = "Thinking..."
        var n = len(self.sessions[si].steps)
        if (
            n > 0
            and self.sessions[si].steps[n - 1].kind == ACT_THOUGHT
            and self.sessions[si].steps[n - 1].turn_id == self.sessions[si].active_turn_id
            and self.sessions[si].steps[n - 1].status == STEP_RUNNING
        ):
            self.sessions[si].steps[n - 1].summary += chunk
            self.sessions[si].last_event_sec = self.theatre_time_sec
            self.event_seq += 1
            return self.sessions[si].steps[n - 1].step_id
        var sid = self.next_step_id
        self.next_step_id += 1
        var st = AgentStep(sid, agent_id, self.sessions[si].active_turn_id, STEP_SOURCE_ACP, ACT_THOUGHT, "Thinking", chunk)
        st.status = STEP_RUNNING
        st.started_sec = now_sec
        return self._append_step(si, st^)

    def record_acp_plan(mut self, agent_id: UInt32, entries: List[String], now_sec: Float64 = 0.0) -> UInt64:
        var si = self.find_session_index(agent_id)
        if si < 0:
            return 0
        self.sessions[si].plan_entries = entries.copy()
        self.sessions[si].status = AGENT_STATUS_PLANNING
        var summary = String(len(entries)) + " plan step(s)"
        self.sessions[si].active_callout = "Plan (" + String(len(entries)) + " steps)"
        var sid = self.next_step_id
        self.next_step_id += 1
        var st = AgentStep(sid, agent_id, self.sessions[si].active_turn_id, STEP_SOURCE_ACP, ACT_PLAN, "Updated Plan", summary)
        st.status = STEP_COMPLETED
        st.progress = Float32(1.0)
        st.started_sec = now_sec
        return self._append_step(si, st^)

    def record_acp_tool_start(
        mut self,
        agent_id: UInt32,
        external_call_id: String,
        title: String,
        tool_kind: String,
        file_path: String = "",
        now_sec: Float64 = 0.0,
    ) -> UInt64:
        var si = self.find_session_index(agent_id)
        if si < 0:
            return 0
        var kind = ACT_OTHER
        if tool_kind == "read":
            kind = ACT_FILE_READ
        elif tool_kind == "edit" or tool_kind == "write":
            kind = ACT_CODE_EDIT
        elif tool_kind == "execute" or tool_kind == "run" or tool_kind == "terminal":
            kind = ACT_TERMINAL
        var sid = self.next_step_id
        self.next_step_id += 1
        var st = AgentStep(sid, agent_id, self.sessions[si].active_turn_id, STEP_SOURCE_ACP, kind, title, file_path)
        st.external_call_id = external_call_id
        st.file_path = file_path
        st.status = STEP_RUNNING
        st.started_sec = now_sec
        self.sessions[si].status = _status_for_action(kind)
        self.sessions[si].active_callout = title
        self.sessions[si].active_progress = Float32(0.25)
        return self._append_step(si, st^)

    def record_acp_tool_update(
        mut self,
        agent_id: UInt32,
        external_call_id: String,
        status_str: String,
        summary_text: String = "",
        terminal_tail: String = "",
        elapsed_ms: Float32 = 0.0,
    ) -> Bool:
        var si = self.find_session_index(agent_id)
        if si < 0:
            return False
        var idx = self.sessions[si].find_step_by_external_id(external_call_id)
        if idx < 0:
            return False
        ref st = self.sessions[si].steps[idx]
        if summary_text.byte_length() > 0:
            st.summary = summary_text
        if terminal_tail.byte_length() > 0:
            st.terminal_tail = terminal_tail
        if isfinite(elapsed_ms) and elapsed_ms > Float32(0.0):
            st.elapsed_ms = elapsed_ms
        if status_str == "completed":
            st.status = STEP_COMPLETED
            st.progress = Float32(1.0)
        elif status_str == "failed":
            st.status = STEP_FAILED
            st.progress = Float32(1.0)
        elif status_str == "cancelled":
            st.status = STEP_REJECTED
            st.progress = Float32(1.0)
        elif status_str == "in_progress":
            st.status = STEP_RUNNING
            st.progress = Float32(0.55)
        self.sessions[si].last_event_sec = self.theatre_time_sec
        self.event_seq += 1
        return True

    def record_acp_file_edit(
        mut self,
        agent_id: UInt32,
        file_path: String,
        old_text: String,
        new_text: String,
        existed_before: Bool,
        now_sec: Float64 = 0.0,
        elapsed_ms: Float32 = 0.0,
    ) -> UInt64:
        var si = self.find_session_index(agent_id)
        if si < 0:
            return 0
        var sid = self.next_step_id
        self.next_step_id += 1
        var title = ("Edit " if existed_before else "Create ") + file_path
        var st = AgentStep(sid, agent_id, self.sessions[si].active_turn_id, STEP_SOURCE_ACP, ACT_CODE_EDIT, title, file_path)
        st.file_path = file_path
        st.file_existed_before = existed_before
        st.old_text = old_text
        st.new_text = new_text
        st.status = STEP_COMPLETED
        st.progress = Float32(1.0)
        st.started_sec = now_sec
        st.elapsed_ms = elapsed_ms if isfinite(elapsed_ms) else Float32(0.0)
        self.sessions[si].status = AGENT_STATUS_WRITING_CODE
        self.sessions[si].active_callout = title
        self.sessions[si].active_progress = Float32(1.0)
        return self._append_step(si, st^)

    # ---- MCP v2 Tool Hooks ----

    def should_gate_mcp_tool(self, agent_id: UInt32, tool_name: String) -> Bool:
        """Returns True when an incoming MCP call must park for visual user approval."""
        var si = self.find_session_index(agent_id)
        if si < 0:
            return False
        ref sess = self.sessions[si]
        if sess.paused:
            return True
        var kind = action_kind_from_mcp_tool(tool_name)
        if not action_kind_is_mutating(kind) and not action_kind_is_destructive(kind, tool_name):
            return False
        if sess.auto_approve_turn_id != 0 and sess.auto_approve_turn_id == sess.active_turn_id:
            return False
        if sess.approval_policy == APPROVAL_AUTO:
            return False
        if sess.approval_policy == APPROVAL_STEP:
            return True
        # APPROVAL_EDITS gates destructive scene ops (despawn / scene.load / package)
        return action_kind_is_destructive(kind, tool_name)

    def begin_mcp_tool_step(
        mut self,
        agent_id: UInt32,
        tool_name: String,
        summary: String,
        params_json: String,
        entity_id: Int = -1,
        world_pos: Vec3 = Vec3(),
        has_world_pos: Bool = False,
        prev_world_pos: Vec3 = Vec3(),
        has_prev_world_pos: Bool = False,
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        rotation: Quat = Quat.identity(),
        pre_world_lz4: List[UInt8] = List[UInt8](),
        pre_hash: UInt64 = 0,
        awaiting_approval: Bool = False,
        now_sec: Float64 = 0.0,
    ) -> UInt64:
        var si = self.find_session_index(agent_id)
        if si < 0:
            return 0
        var kind = action_kind_from_mcp_tool(tool_name)
        var sid = self.next_step_id
        self.next_step_id += 1
        var st = AgentStep(sid, agent_id, self.sessions[si].active_turn_id, STEP_SOURCE_MCP, kind, tool_name, summary)
        st.params_json = params_json
        st.started_sec = now_sec
        st.status = STEP_AWAITING_APPROVAL if awaiting_approval else STEP_RUNNING
        st.progress = Float32(0.0) if awaiting_approval else Float32(0.35)
        if entity_id >= 0:
            st.affected_entity_ids.append(entity_id)
        st.world_pos = _safe_vec3(world_pos)
        st.has_world_pos = has_world_pos
        st.prev_world_pos = _safe_vec3(prev_world_pos)
        st.has_prev_world_pos = has_prev_world_pos
        st.entity_half_extents = _safe_half_extents(half_extents)
        st.entity_rotation = _safe_quat(rotation)
        st.pre_world_lz4 = pre_world_lz4.copy()
        st.pre_hash = pre_hash

        self.sessions[si].status = AGENT_STATUS_WAITING_APPROVAL if awaiting_approval else _status_for_action(kind)
        self.sessions[si].last_event_sec = self.theatre_time_sec
        if entity_id >= 0:
            self.sessions[si].active_entity_id = entity_id
        if has_world_pos:
            self.sessions[si].active_world_pos = st.world_pos
            self.sessions[si].has_active_world_pos = True
            self.sessions[si].active_half_extents = st.entity_half_extents
            self.sessions[si].active_rotation = st.entity_rotation
        self.sessions[si].active_callout = summary if summary.byte_length() > 0 else tool_name
        self.sessions[si].active_progress = st.progress

        if has_world_pos:
            self._update_follow_focus(entity_id, st.world_pos, st.entity_half_extents)

        return self._append_step(si, st^)

    def complete_mcp_tool_step(
        mut self,
        agent_id: UInt32,
        step_id: UInt64,
        ok: Bool,
        result_json: String,
        elapsed_ms: Float32 = 0.0,
        tx_id: UInt64 = 0,
        entity_id: Int = -1,
        world_pos: Vec3 = Vec3(),
        has_world_pos: Bool = False,
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        rotation: Quat = Quat.identity(),
        post_world_lz4: List[UInt8] = List[UInt8](),
        post_hash: UInt64 = 0,
        capture_png_path: String = "",
        flip_score: Float32 = -1.0,
    ) -> Bool:
        var si = self.find_session_index(agent_id)
        if si < 0:
            return False
        var idx = self.sessions[si].find_step_index(step_id)
        if idx < 0:
            return False

        var kind: UInt8
        var eff_eid = entity_id
        var eff_has_pos = has_world_pos
        var eff_pos = _safe_vec3(world_pos)
        var eff_prev_pos: Vec3
        var eff_has_prev: Bool
        var eff_he = _safe_half_extents(half_extents)
        var eff_rot = _safe_quat(rotation)

        # Mutate the step in place
        ref st = self.sessions[si].steps[idx]
        st.status = STEP_COMPLETED if ok else STEP_FAILED
        st.result_json = result_json
        st.elapsed_ms = elapsed_ms if (isfinite(elapsed_ms) and elapsed_ms >= Float32(0.0)) else Float32(0.0)
        st.progress = Float32(1.0)
        st.tx_id = tx_id
        st.post_world_lz4 = post_world_lz4.copy()
        st.post_hash = post_hash
        if capture_png_path.byte_length() > 0:
            st.capture_png_path = capture_png_path
        if isfinite(flip_score) and flip_score >= Float32(0.0):
            st.flip_score = flip_score
        if entity_id >= 0:
            var found_eid = False
            for k in range(len(st.affected_entity_ids)):
                if st.affected_entity_ids[k] == entity_id:
                    found_eid = True
                    break
            if not found_eid:
                st.affected_entity_ids.append(entity_id)
        elif len(st.affected_entity_ids) > 0:
            eff_eid = st.affected_entity_ids[0]

        if has_world_pos:
            st.world_pos = eff_pos
            st.has_world_pos = True
            st.entity_half_extents = eff_he
            st.entity_rotation = eff_rot
        elif st.has_world_pos:
            eff_has_pos = True
            eff_pos = st.world_pos
            eff_he = st.entity_half_extents
            eff_rot = st.entity_rotation

        kind = st.kind
        eff_prev_pos = st.prev_world_pos
        eff_has_prev = st.has_prev_world_pos

        self.sessions[si].active_progress = Float32(1.0)
        self.sessions[si].last_event_sec = self.theatre_time_sec
        if eff_has_pos:
            self.sessions[si].active_entity_id = eff_eid
            self.sessions[si].active_world_pos = eff_pos
            self.sessions[si].has_active_world_pos = True
            self.sessions[si].active_half_extents = eff_he
            self.sessions[si].active_rotation = eff_rot
            self._update_follow_focus(eff_eid, eff_pos, eff_he)
            if ok and action_kind_is_mutating(kind):
                self.record_trail_mark(
                    entity_id=eff_eid,
                    agent_id=agent_id,
                    step_id=step_id,
                    action_kind=kind,
                    world_pos=eff_pos,
                    half_extents=eff_he,
                    rotation=eff_rot,
                    prev_world_pos=eff_prev_pos,
                    has_prev_pos=eff_has_prev,
                )
        self.event_seq += 1
        return True

    # ---- Unified Approval Queue ----

    def queue_approval(mut self, var item: PendingApproval) -> UInt64:
        var aid = self.next_approval_id
        self.next_approval_id += 1
        item.approval_id = aid
        item.resolved = False
        var si = self.find_session_index(item.agent_id)
        if si >= 0:
            self.sessions[si].status = AGENT_STATUS_WAITING_APPROVAL
            self.sessions[si].active_callout = "Approve: " + item.title
            if item.ghost.kind != GHOST_NONE:
                self.sessions[si].active_world_pos = item.ghost.target_pos
                self.sessions[si].has_active_world_pos = True
                self.sessions[si].active_entity_id = item.ghost.entity_id
                self.sessions[si].active_half_extents = item.ghost.half_extents
                self.sessions[si].active_rotation = item.ghost.target_rot
                self._update_follow_focus(item.ghost.entity_id, item.ghost.target_pos, item.ghost.half_extents)
        self.pending_approvals.append(item^)
        self.event_seq += 1
        return aid

    def resolve_approval(
        mut self,
        approval_id: UInt64,
        approved: Bool,
        allow_turn: Bool = False,
        reject_reason: String = "Rejected by user",
    ) -> Tuple[Bool, PendingApproval]:
        for i in range(len(self.pending_approvals)):
            if self.pending_approvals[i].approval_id == approval_id and not self.pending_approvals[i].resolved:
                var out = self.pending_approvals.pop(i)
                out.resolved = True
                out.approved = approved
                out.allow_turn = allow_turn
                out.reject_reason = String() if approved else reject_reason

                var si = self.find_session_index(out.agent_id)
                if si >= 0:
                    if approved and allow_turn:
                        self.sessions[si].auto_approve_turn_id = self.sessions[si].active_turn_id
                    var sidx = self.sessions[si].find_step_index(out.step_id)
                    if sidx >= 0:
                        self.sessions[si].steps[sidx].status = STEP_RUNNING if approved else STEP_REJECTED
                        if not approved:
                            self.sessions[si].steps[sidx].result_json = '{"ok":false,"rejected":true}'
                    # If no more pending approvals for this agent, restore working status
                    if self.first_pending_for_agent(out.agent_id) < 0:
                        self.sessions[si].status = AGENT_STATUS_EDITING_SCENE if approved else AGENT_STATUS_THINKING
                self.event_seq += 1
                return (True, out^)
        return (False, PendingApproval())

    def pending_approval_count(self) -> Int:
        var c = 0
        for i in range(len(self.pending_approvals)):
            if not self.pending_approvals[i].resolved:
                c += 1
        return c

    def first_pending_for_agent(self, agent_id: UInt32) -> Int:
        for i in range(len(self.pending_approvals)):
            if not self.pending_approvals[i].resolved and self.pending_approvals[i].agent_id == agent_id:
                return i
        return -1

    # ---- Time-Travel Scrubbing ----

    def enter_time_travel(
        mut self,
        agent_id: UInt32,
        step_idx: Int,
        live_tip_lz4: List[UInt8],
        live_tip_hash: UInt64,
    ) -> Tuple[Bool, List[UInt8], UInt64]:
        var si = self.find_session_index(agent_id)
        if si < 0 or step_idx < 0 or step_idx >= len(self.sessions[si].steps):
            return (False, List[UInt8](), UInt64(0))
        if not self.time_travel_active:
            self.live_tip_lz4 = live_tip_lz4.copy()
            self.live_tip_hash = live_tip_hash
        self.time_travel_active = True
        self.time_travel_agent_id = agent_id
        return self.scrub_time_travel(step_idx)

    def scrub_time_travel(mut self, step_idx: Int) -> Tuple[Bool, List[UInt8], UInt64]:
        if not self.time_travel_active:
            return (False, List[UInt8](), UInt64(0))
        var si = self.find_session_index(self.time_travel_agent_id)
        if si < 0 or step_idx < 0 or step_idx >= len(self.sessions[si].steps):
            return (False, List[UInt8](), UInt64(0))
        self.time_travel_step_idx = step_idx
        self.event_seq += 1
        # Search backward from step_idx for the nearest step carrying a world snapshot
        for k in range(step_idx, -1, -1):
            if len(self.sessions[si].steps[k].post_world_lz4) > 0:
                return (True, self.sessions[si].steps[k].post_world_lz4.copy(), self.sessions[si].steps[k].post_hash)
            if len(self.sessions[si].steps[k].pre_world_lz4) > 0:
                return (True, self.sessions[si].steps[k].pre_world_lz4.copy(), self.sessions[si].steps[k].pre_hash)
        return (False, List[UInt8](), UInt64(0))

    def exit_time_travel(mut self) -> Tuple[Bool, List[UInt8], UInt64]:
        if not self.time_travel_active:
            return (False, List[UInt8](), UInt64(0))
        var blob = self.live_tip_lz4.copy()
        var h = self.live_tip_hash
        self.time_travel_active = False
        self.time_travel_agent_id = 0
        self.time_travel_step_idx = -1
        self.live_tip_lz4.clear()
        self.live_tip_hash = 0
        self.event_seq += 1
        return (True, blob^, h)

    # ---- Step & Turn Revert Planners ----

    def plan_revert_step(self, agent_id: UInt32, step_id: UInt64) -> AgentRevertPlan:
        var plan = AgentRevertPlan()
        var si = self.find_session_index(agent_id)
        if si < 0:
            plan.reason = "agent session not found"
            return plan^
        var idx = self.sessions[si].find_step_index(step_id)
        if idx < 0:
            plan.reason = "step not found"
            return plan^
        ref st = self.sessions[si].steps[idx]
        if not st.is_revertible():
            plan.reason = "step has no revertible checkpoint"
            return plan^
        plan.valid = True
        plan.is_turn_revert = False
        plan.agent_id = agent_id
        plan.step_id = step_id
        plan.turn_id = st.turn_id
        plan.target_tx_id = st.tx_id
        if len(st.pre_world_lz4) > 0:
            plan.restore_world_lz4 = st.pre_world_lz4.copy()
            plan.expected_hash = st.pre_hash
        if st.kind == ACT_CODE_EDIT and st.file_path.byte_length() > 0:
            plan.file_paths.append(st.file_path)
            plan.file_existed.append(st.file_existed_before)
            plan.file_old_texts.append(st.old_text)
        plan.reverted_step_ids.append(step_id)
        return plan^

    def plan_revert_turn(self, agent_id: UInt32, turn_id: UInt32) -> AgentRevertPlan:
        var plan = AgentRevertPlan()
        var si = self.find_session_index(agent_id)
        if si < 0:
            plan.reason = "agent session not found"
            return plan^
        var ti = self.sessions[si].find_turn_index(turn_id)
        if ti < 0:
            plan.reason = "turn not found"
            return plan^
        ref turn = self.sessions[si].turns[ti]
        if turn.reverted:
            plan.reason = "turn already reverted"
            return plan^
        plan.valid = True
        plan.is_turn_revert = True
        plan.agent_id = agent_id
        plan.turn_id = turn_id
        if len(turn.pre_world_lz4) > 0:
            plan.restore_world_lz4 = turn.pre_world_lz4.copy()
            plan.expected_hash = turn.pre_hash
        # Collect earliest pre-edit text for each file touched in this turn
        for k in range(len(self.sessions[si].steps)):
            ref st = self.sessions[si].steps[k]
            if st.turn_id != turn_id or st.status == STEP_REVERTED:
                continue
            plan.reverted_step_ids.append(st.step_id)
            if len(plan.restore_world_lz4) == 0 and len(st.pre_world_lz4) > 0:
                plan.restore_world_lz4 = st.pre_world_lz4.copy()
                plan.expected_hash = st.pre_hash
            if st.kind == ACT_CODE_EDIT and st.file_path.byte_length() > 0:
                var already = False
                for fi in range(len(plan.file_paths)):
                    if plan.file_paths[fi] == st.file_path:
                        already = True
                        break
                if not already:
                    plan.file_paths.append(st.file_path)
                    plan.file_existed.append(st.file_existed_before)
                    plan.file_old_texts.append(st.old_text)
        if len(plan.restore_world_lz4) == 0 and len(plan.file_paths) == 0:
            plan.valid = False
            plan.reason = "turn has no world or file changes to revert"
        return plan^

    def commit_revert_plan(mut self, plan: AgentRevertPlan):
        if not plan.valid:
            return
        var si = self.find_session_index(plan.agent_id)
        if si < 0:
            return
        for ri in range(len(plan.reverted_step_ids)):
            var sidx = self.sessions[si].find_step_index(plan.reverted_step_ids[ri])
            if sidx >= 0:
                self.sessions[si].steps[sidx].status = STEP_REVERTED
        if plan.is_turn_revert:
            var ti = self.sessions[si].find_turn_index(plan.turn_id)
            if ti >= 0:
                self.sessions[si].turns[ti].reverted = True
        self.event_seq += 1

    # ---- Follow-Agent Camera & Hover Controls ----

    def notify_user_camera_input(mut self):
        if self.follow_agent_camera and not self.follow_paused_by_user:
            self.follow_paused_by_user = True
            self.event_seq += 1

    def resume_follow_camera(mut self):
        self.follow_agent_camera = True
        self.follow_paused_by_user = False
        self.event_seq += 1

    def should_follow_camera(self) -> Bool:
        return self.follow_agent_camera and (not self.follow_paused_by_user) and self.has_focus_target

    def set_hovered_entity(
        mut self,
        entity_id: Int,
        world_pos: Vec3,
        half_extents: Vec3 = Vec3(0.5, 0.5, 0.5),
        rotation: Quat = Quat.identity(),
    ):
        self.hovered_entity_id = entity_id
        self.hovered_entity_pos = _safe_vec3(world_pos)
        self.hovered_entity_half_extents = _safe_half_extents(half_extents)
        self.hovered_entity_rot = _safe_quat(rotation)
        self.has_hovered_entity = entity_id >= 0

    def clear_hovered_entity(mut self):
        self.hovered_entity_id = -1
        self.has_hovered_entity = False

    # ---- Deterministic Fixed-dt Tick ----

    def tick(mut self, dt: Float32):
        var safe_dt = Float32(0.0) if (not isfinite(dt) or dt < Float32(0.0)) else min(dt, Float32(0.1))
        self.theatre_time_sec += safe_dt
        if self.focus_settle_timer_sec > Float32(0.0):
            self.focus_settle_timer_sec = max(Float32(0.0), self.focus_settle_timer_sec - safe_dt)
        var inv_dur = Float32(1.0) / max(Float32(0.1), TRAIL_HEAT_DURATION_SEC)
        var i = 0
        while i < len(self.trails):
            var age = self.theatre_time_sec - self.trails[i].last_touched_sec
            var h = Float32(1.0) - age * inv_dur
            if h <= Float32(0.005):
                _ = self.trails.pop(i)
            else:
                self.trails[i].heat = min(Float32(1.0), h)
                i += 1
        # Auto-settle MCP sessions that finished a burst (3.5s idle with no pending approval)
        for si in range(len(self.sessions)):
            if (
                self.sessions[si].is_busy()
                and self.sessions[si].status != AGENT_STATUS_WAITING_APPROVAL
                and self.first_pending_for_agent(self.sessions[si].agent_id) < 0
                and self.sessions[si].active_progress >= Float32(0.999)
                and (self.theatre_time_sec - self.sessions[si].last_event_sec) > Float32(3.5)
            ):
                self.sessions[si].status = AGENT_STATUS_IDLE
                self.sessions[si].has_active_world_pos = False

    # ---- Outliner & Viewport Queries ----

    def active_entity_ids(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.sessions)):
            var eid = self.sessions[i].active_entity_id
            if eid >= 0 and self.sessions[i].is_busy():
                var seen = False
                for k in range(len(out)):
                    if out[k] == eid:
                        seen = True
                        break
                if not seen:
                    out.append(eid)
        for i in range(len(self.pending_approvals)):
            ref pa = self.pending_approvals[i]
            if not pa.resolved and pa.ghost.entity_id >= 0:
                var eid = pa.ghost.entity_id
                var seen = False
                for k in range(len(out)):
                    if out[k] == eid:
                        seen = True
                        break
                if not seen:
                    out.append(eid)
        return out^

    def recent_entity_ids(self) -> List[Int]:
        var out = List[Int]()
        for i in range(len(self.trails)):
            var eid = self.trails[i].entity_id
            if eid >= 0 and self.trails[i].heat > Float32(0.01):
                out.append(eid)
        return out^

    def entity_badge_info(self, entity_id: Int) -> Tuple[Bool, UInt32, Bool]:
        """Returns `(has_badge, accent_rgba8, is_actively_editing)` for Outliner row rendering."""
        if entity_id < 0:
            return (False, UInt32(0), False)
        for i in range(len(self.sessions)):
            if self.sessions[i].is_busy() and self.sessions[i].active_entity_id == entity_id:
                return (True, self.sessions[i].accent_rgba8, True)
        for i in range(len(self.pending_approvals)):
            ref pa = self.pending_approvals[i]
            if not pa.resolved and pa.ghost.entity_id == entity_id:
                return (True, pa.ghost.accent_rgba8, True)
        for i in range(len(self.trails)):
            if self.trails[i].entity_id == entity_id and self.trails[i].heat > Float32(0.05):
                return (True, self.trails[i].accent_rgba8, False)
        return (False, UInt32(0), False)

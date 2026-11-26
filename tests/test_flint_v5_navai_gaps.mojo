"""Flint v5 navigation / AI / camera gap verification (`flint.nav`, `flint.ai`, `flint.camera`).

Covers: Recast contour polygons on diagonal walls, area costs + query filters, multi-radius
navmesh sets, sliced A*, path corridor following with repath after carving, off-mesh link
traversal events, perception memory, behaviour-tree decorators + task library, EQS point queries,
cover points, and virtual-camera blending.
"""
from std.collections import List
from std.math import abs, sqrt
from std.testing import assert_true, assert_equal
from flint.core.math import Vec3, Quat
from flint.core.ik import angle_axis
from flint.core.serial import BlobWriter, BlobReader
from flint.physics.world import PhysicsWorld
from flint.nav.navmesh import (
    NavMesh,
    NavMeshConfig,
    NavQueryFilter,
    NavConvexVolume,
    NavObstacle,
    NAV_POLY_RECT,
    NAV_POLY_CONTOUR,
    NAV_QUERY_IN_PROGRESS,
    NAV_QUERY_SUCCESS,
    OffMeshLink,
    OFFMESH_JUMP,
)
from flint.nav.pathfollow import (
    NavPathFollower,
    NavMeshSet,
    NAV_FOLLOW_MOVING,
    NAV_FOLLOW_ARRIVED,
    NAV_FOLLOW_OFFMESH_WAIT,
    NAV_FOLLOW_OFFMESH_TRAVERSE,
    NAV_EVT_REPATH,
    NAV_EVT_OFFMESH_REACHED,
    NAV_EVT_OFFMESH_END,
    NAV_EVT_ARRIVED,
    NAV_REPATH_MESH_CHANGED,
)
from flint.ai.behavior_tree import (
    Blackboard,
    BehaviorTree,
    BtExecutionState,
    BT_SUCCESS,
    BT_FAILURE,
    BT_RUNNING,
)
from flint.ai.tasks import BtTaskLibrary, BtAgentContext
from flint.ai.perception import (
    PerceptionSystem,
    Perceiver,
    PERCEPT_EVT_SIGHTED,
    PERCEPT_EVT_FORGOT,
    PERCEPT_EVT_HEARD,
)
from flint.ai.perception_plugin import PerceptionPlugin, PerceptionResource, PerceiverComponent, StimulusComponent
from flint.ai.eqs import EnvQuery, generate_cover_points, find_best_cover
from flint.camera.vcam import VirtualCamera, CameraBrain, VCAM_FIXED, VCAM_LOCK_ON
from flint.camera.plugin import CameraPlugin, CameraBrainResource, VirtualCameraComponent
from flint.app import AppConfig, EngineApp
from flint.scene.transform_hierarchy import LocalTransform


def _diag_world() -> PhysicsWorld:
    var w = PhysicsWorld()
    _ = w.add_box(Vec3(0.0, -0.25, 0.0), Vec3(8.0, 0.25, 8.0), mass=0.0)
    # 45-degree wall across the floor
    _ = w.add_box(Vec3(0.0, 1.0, 0.0), Vec3(5.0, 1.0, 0.3), mass=0.0, rot=angle_axis(Float32(0.785398), Vec3(0, 1, 0)))
    return w^


def _flat_world(half: Float32 = 6.0) -> PhysicsWorld:
    var w = PhysicsWorld()
    _ = w.add_box(Vec3(0.0, -0.25, 0.0), Vec3(half, 0.25, half * Float32(0.6)), mass=0.0)
    return w^


def test_diagonal_wall_few_polys() raises:
    var w = _diag_world()
    var cfg = NavMeshConfig()
    cfg.cell_size = 0.25
    cfg.tile_size = 32
    cfg.poly_mode = NAV_POLY_RECT
    var nav_rect = NavMesh.build_from_physics(w, cfg)
    cfg.poly_mode = NAV_POLY_CONTOUR
    var nav_c = NavMesh.build_from_physics(w, cfg)
    print("  rect polys:", nav_rect.poly_count(), " contour polys:", nav_c.poly_count())
    assert_true(nav_c.poly_count() > 0)
    assert_true(nav_c.poly_count() * 3 < nav_rect.poly_count())
    var max_v = 0
    for i in range(nav_c.poly_count()):
        max_v = max(max_v, len(nav_c.polys[i].verts))
    assert_true(max_v <= cfg.max_verts_per_poly)
    # Path around the diagonal wall still works and is taut-ish
    var p = nav_c.find_path(Vec3(-5.0, 0.0, 3.0), Vec3(5.0, 0.0, -3.0))
    assert_true(p.found)
    assert_true(len(p.waypoints) >= 2)
    var p2 = nav_c.find_path(Vec3(-5.0, 0.0, 3.0), Vec3(-3.0, 0.0, 5.0))
    assert_true(p2.found)
    # Round-trip serialization (v2 blob keeps poly mode + areas)
    var bw = BlobWriter()
    nav_c.serialize(bw)
    var br = BlobReader(bw.data.copy())
    var nav_d = NavMesh()
    nav_d.deserialize(br)
    assert_equal(nav_d.poly_count(), nav_c.poly_count())
    assert_equal(nav_d.cfg.poly_mode, NAV_POLY_CONTOUR)
    print("[PASS] diagonal wall yields few contour polys")


def _seg_hits_box_xz(a: Vec3, b: Vec3, x0: Float32, x1: Float32, z0: Float32, z1: Float32) -> Bool:
    for k in range(33):
        var t = Float32(k) / Float32(32.0)
        var p = a + (b - a) * t
        if p.x > x0 and p.x < x1 and p.z > z0 and p.z < z1:
            return True
    return False


def _path_hits_box_xz(wps: List[Vec3], x0: Float32, x1: Float32, z0: Float32, z1: Float32) -> Bool:
    for i in range(len(wps) - 1):
        if _seg_hits_box_xz(wps[i], wps[i + 1], x0, x1, z0, z1):
            return True
    return False


def test_area_cost_filters_and_radii() raises:
    var w = _flat_world()
    var cfg = NavMeshConfig()
    var nav = NavMesh.build_from_physics(w, cfg)
    var a = Vec3(-5.0, 0.0, 0.0)
    var b = Vec3(5.0, 0.0, 0.0)
    var p_plain = nav.find_path(a, b)
    assert_true(p_plain.found)
    # Mud strip in the middle (area 2)
    var mud_idx = nav.mark_area(NavConvexVolume.box(Vec3(-1.5, -1.0, -2.0), Vec3(1.5, 1.0, 2.0), 2))
    assert_equal(mud_idx, 0)
    assert_true(nav.polys_with_area(2) > 0)
    var p_cheap = nav.find_path(a, b)
    assert_true(p_cheap.found)
    assert_true(p_cheap.total_length < Float32(10.2))
    var f = NavQueryFilter()
    f.set_area_cost(2, 25.0)
    var p_cost = nav.find_path_filtered(a, b, f)
    assert_true(p_cost.found)
    print("  through-mud length:", p_cheap.total_length, " costed length:", p_cost.total_length)
    assert_true(p_cost.total_length > p_cheap.total_length + Float32(0.3))
    assert_true(not _path_hits_box_xz(p_cost.waypoints, -1.4, 1.4, -1.9, 1.9))
    var fx = NavQueryFilter()
    fx.exclude_area(2)
    var p_ex = nav.find_path_filtered(a, b, fx)
    assert_true(p_ex.found)
    assert_true(not _path_hits_box_xz(p_ex.waypoints, -1.4, 1.4, -1.9, 1.9))
    # Excluded area also filters nearest-poly queries
    var np_ex = nav.find_nearest_poly_filtered(Vec3(0.0, 0.0, 0.0), fx, 0.5)
    assert_true(np_ex[0] < 0 or nav.polys[np_ex[0]].area != 2)
    # A full-width excluded band disconnects the mesh
    var nav2 = NavMesh.build_from_physics(w, cfg)
    _ = nav2.mark_area(NavConvexVolume.box(Vec3(-0.5, -1.0, -5.0), Vec3(0.5, 1.0, 5.0), 3))
    var fx3 = NavQueryFilter()
    fx3.exclude_area(3)
    assert_true(not nav2.find_path_filtered(a, b, fx3).found)
    assert_true(nav2.find_path(a, b).found)
    # Area flags: flag-excluded areas behave like exclusions
    nav2.set_area_flags(3, 2)
    var ff = NavQueryFilter()
    ff.exclude_flags = 2
    assert_true(not nav2.find_path_filtered(a, b, ff).found)
    print("[PASS] area cost changes chosen path (+ exclude filters)")

    # Multiple agent radii: a 1.2 m doorway admits the thin agent only
    var wr = PhysicsWorld()
    _ = wr.add_box(Vec3(0.0, -0.25, 0.0), Vec3(6.0, 0.25, 4.0), mass=0.0)
    _ = wr.add_box(Vec3(0.0, 1.0, -2.3), Vec3(0.25, 1.0, 1.7), mass=0.0)
    _ = wr.add_box(Vec3(0.0, 1.0, 2.3), Vec3(0.25, 1.0, 1.7), mass=0.0)
    var radii = List[Float32]()
    radii.append(0.9)
    radii.append(0.3)
    var set = NavMeshSet.build_from_physics(wr, NavMeshConfig(), radii)
    assert_equal(set.count(), 2)
    assert_equal(set.mesh_index_for_radius(0.25), 0)
    assert_equal(set.mesh_index_for_radius(0.5), 1)
    var thin = set.find_path(0.3, Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0))
    var fat = set.find_path(0.9, Vec3(-4.0, 0.0, 0.0), Vec3(4.0, 0.0, 0.0))
    assert_true(thin.found)
    assert_true(not fat.found)
    print("[PASS] per-radius navmesh set (thin agent fits doorway, wide agent does not)")


def test_sliced_search() raises:
    var w = _diag_world()
    var nav = NavMesh.build_from_physics(w, NavMeshConfig())
    var a = Vec3(-6.0, 0.0, 5.0)
    var b = Vec3(6.0, 0.0, -5.0)
    var sync = nav.find_path(a, b)
    assert_true(sync.found)
    var q = nav.begin_path_query(a, b, NavQueryFilter())
    assert_equal(q.status, NAV_QUERY_IN_PROGRESS)
    var frames = 0
    while q.in_progress() and frames < 10000:
        _ = nav.update_path_query(q, 2)
        frames += 1
    assert_equal(q.status, NAV_QUERY_SUCCESS)
    assert_true(frames > 1)
    var sliced = nav.finish_path_query(q)
    assert_true(sliced.found)
    assert_true(abs(sliced.total_length - sync.total_length) < Float32(1.0e-3))
    # A rebuild between slices fails the stale query
    var q2 = nav.begin_path_query(a, b, NavQueryFilter())
    _ = nav.add_obstacle(NavObstacle.cylinder(7, Vec3(5.0, 1.0, 5.0), 0.5, 1.0))
    _ = nav.update_path_query(q2, 1)
    assert_true(q2.done() and q2.status != NAV_QUERY_SUCCESS)
    # Sliced follower plans over several frames, then moves
    var fol = NavPathFollower(3.5, 0.3, sliced=True)
    fol.max_iters_per_update = 1
    var pos = a
    fol.set_goal(nav, pos, b)
    var plan_frames = 0
    while fol.state != NAV_FOLLOW_MOVING and plan_frames < 1000:
        fol.step_kinematic(nav, pos, 0.016)
        plan_frames += 1
    assert_true(plan_frames >= 1)
    assert_equal(fol.state, NAV_FOLLOW_MOVING)
    print("  sliced query frames:", frames, " follower plan frames:", plan_frames)
    print("[PASS] sliced search completes over frames")


def test_repath_after_carve() raises:
    var w = _flat_world()
    var nav = NavMesh.build_from_physics(w, NavMeshConfig())
    var fol = NavPathFollower(3.5, 0.3)
    var pos = Vec3(-5.0, 0.0, 0.0)
    var goal = Vec3(5.0, 0.0, 0.0)
    fol.set_goal(nav, pos, goal)
    assert_equal(fol.state, NAV_FOLLOW_MOVING)
    for _ in range(10):
        fol.step_kinematic(nav, pos, 0.05)
    assert_equal(fol.repath_count, 0)
    _ = nav.add_obstacle(NavObstacle.box(1, Vec3(1.0, 1.0, 0.0), Vec3(0.5, 1.0, 2.2)))
    var hit_obstacle = False
    var frames = 0
    while fol.state != NAV_FOLLOW_ARRIVED and frames < 800:
        fol.step_kinematic(nav, pos, 0.02)
        if pos.x > Float32(0.5) and pos.x < Float32(1.5) and abs(pos.z) < Float32(2.2):
            hit_obstacle = True
        frames += 1
    assert_true(fol.has_event(NAV_EVT_REPATH))
    assert_true(fol.repath_count >= 1)
    assert_equal(fol.last_repath_reason, NAV_REPATH_MESH_CHANGED)
    assert_equal(fol.state, NAV_FOLLOW_ARRIVED)
    assert_true(not hit_obstacle)
    assert_true((pos - goal).length() < Float32(0.5))
    print("  repaths:", fol.repath_count, " frames:", frames)
    print("[PASS] agent repaths after carve and arrives")


def test_offmesh_traversal_events() raises:
    var w = PhysicsWorld()
    _ = w.add_box(Vec3(-4.0, -0.25, 0.0), Vec3(3.0, 0.25, 3.0), mass=0.0)
    _ = w.add_box(Vec3(4.0, -0.25, 0.0), Vec3(3.0, 0.25, 3.0), mass=0.0)
    var nav = NavMesh.build_from_physics(w, NavMeshConfig())
    _ = nav.add_off_mesh_link(OffMeshLink(5, Vec3(-1.4, 0.0, 0.0), Vec3(1.4, 0.0, 0.0), 0.5, OFFMESH_JUMP))
    var fol = NavPathFollower(3.5, 0.3)
    fol.auto_offmesh = False
    var pos = Vec3(-5.0, 0.0, 1.0)
    var goal = Vec3(5.0, 0.0, -1.0)
    fol.set_goal(nav, pos, goal)
    assert_equal(fol.state, NAV_FOLLOW_MOVING)
    var frames = 0
    while fol.state != NAV_FOLLOW_OFFMESH_WAIT and frames < 400:
        fol.step_kinematic(nav, pos, 0.02)
        frames += 1
    assert_equal(fol.state, NAV_FOLLOW_OFFMESH_WAIT)
    assert_true(fol.has_event(NAV_EVT_OFFMESH_REACHED))
    assert_equal(fol.offmesh_link_id, 5)
    # Game plays its jump animation, then reports completion
    fol.begin_offmesh()
    assert_equal(fol.state, NAV_FOLLOW_OFFMESH_TRAVERSE)
    fol.offmesh_t = 0.5
    assert_true(fol.offmesh_position().y > Float32(0.3))
    fol.complete_offmesh()
    assert_true(fol.has_event(NAV_EVT_OFFMESH_END))
    frames = 0
    while fol.state != NAV_FOLLOW_ARRIVED and frames < 400:
        fol.step_kinematic(nav, pos, 0.02)
        frames += 1
    assert_equal(fol.state, NAV_FOLLOW_ARRIVED)
    assert_true(fol.has_event(NAV_EVT_ARRIVED))
    assert_true(pos.x > Float32(3.0))
    # Auto mode crosses the link by itself
    var fol2 = NavPathFollower(3.5, 0.3)
    var pos2 = Vec3(-5.0, 0.0, 1.0)
    fol2.set_goal(nav, pos2, goal)
    frames = 0
    while fol2.state != NAV_FOLLOW_ARRIVED and frames < 800:
        fol2.step_kinematic(nav, pos2, 0.02)
        frames += 1
    assert_equal(fol2.state, NAV_FOLLOW_ARRIVED)
    print("[PASS] off-mesh link traversal states + events")


def test_perception_memory() raises:
    var w = PhysicsWorld()
    _ = w.add_box(Vec3(0.0, -0.25, 0.0), Vec3(20.0, 0.25, 20.0), mass=0.0)
    _ = w.add_box(Vec3(0.0, 1.5, 4.0), Vec3(2.0, 1.5, 0.2), mass=0.0)  # wall between guard and enemy
    var ps = PerceptionSystem()
    var guard = Perceiver(UInt64(100), 1, 90.0, 20.0, 2.0)
    var gi = ps.add_perceiver(guard^)
    ps.set_perceiver_pose(gi, Vec3(0.0, 1.6, 0.0), Vec3(0.0, 0.0, 1.0))
    var enemy = UInt64(200)
    var si = ps.register_stimulus(enemy, Vec3(0.0, 1.0, 8.0), 2)
    ps.update(w, 0.1)
    assert_true(not ps.perceivers[gi].knows(enemy))  # wall blocks LOS
    ps.update_stimulus(si, Vec3(5.0, 1.0, 8.0))
    ps.update(w, 0.1)
    assert_true(ps.perceivers[gi].can_see(enemy))
    var mi = ps.perceivers[gi].find_memory(enemy)
    assert_true(ps.perceivers[gi].memories[mi].awareness > Float32(0.0))
    # Enemy slips behind the guard: remembered at last known position
    ps.update_stimulus(si, Vec3(0.0, 1.0, -8.0))
    ps.update(w, 0.1)
    assert_true(ps.perceivers[gi].knows(enemy))
    assert_true(not ps.perceivers[gi].can_see(enemy))
    mi = ps.perceivers[gi].find_memory(enemy)
    assert_true((ps.perceivers[gi].memories[mi].last_known_pos - Vec3(5.0, 1.0, 8.0)).length() < Float32(1.0e-3))
    var evs = ps.take_events()
    var sighted = False
    for i in range(len(evs)):
        if evs[i].kind == PERCEPT_EVT_SIGHTED and evs[i].entity == enemy:
            sighted = True
    assert_true(sighted)
    # ... and forgotten after memory_duration
    for _ in range(25):
        ps.update(w, 0.1)
    assert_true(not ps.perceivers[gi].knows(enemy))
    var forgot = False
    evs = ps.take_events()
    for i in range(len(evs)):
        if evs[i].kind == PERCEPT_EVT_FORGOT and evs[i].entity == enemy:
            forgot = True
    assert_true(forgot)
    # Hearing: a noise behind the guard creates a memory at the noise position
    ps.report_noise(Vec3(1.0, 0.0, -6.0), 10.0, UInt64(300), 2)
    ps.update(w, 0.1)
    var hm = ps.perceivers[gi].find_memory(UInt64(300))
    assert_true(hm >= 0)
    assert_true(ps.perceivers[gi].memories[hm].heard)
    assert_true(len(ps.noises) == 0)
    print("[PASS] perception sees (LOS), remembers, then forgets; hearing works")

    # ECS plugin smoke
    var app = EngineApp(AppConfig(title="PerceptionPluginTest", headless=True, worker_threads=0))
    app.add_plugin(PerceptionPlugin())
    app.startup()
    var g_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](g_e, LocalTransform(Vec3(0.0, 0.0, 0.0)))
    _ = app.world.insert[PerceiverComponent](g_e, PerceiverComponent(team=1))
    var e_e = app.world.spawn()
    _ = app.world.insert[LocalTransform](e_e, LocalTransform(Vec3(1.0, 0.0, 6.0)))
    _ = app.world.insert[StimulusComponent](e_e, StimulusComponent(team=2))
    app.step_one_frame(Float32(1.0 / 60.0))
    var pr_opt = app.resources.get_ptr[PerceptionResource]()
    assert_true(Bool(pr_opt))
    var mems = pr_opt.value()[unsafe_offset=0].memories_of(g_e.to_bits())
    assert_equal(len(mems), 1)
    assert_equal(mems[0].entity, e_e.to_bits())
    print("[PASS] PerceptionPlugin senses ECS stimuli")


def test_bt_decorators_and_tasks() raises:
    var bb = Blackboard()
    # Repeat(3) over an always-succeeding action
    var t_rep = BehaviorTree()
    _ = t_rep.begin_repeat(3)
    _ = t_rep.add_action(0)
    t_rep.end_composite()
    var st = BtExecutionState()
    st.set_action_status(0, BT_SUCCESS)
    assert_equal(t_rep.tick(bb, st, 0.1), BT_RUNNING)
    assert_equal(t_rep.tick(bb, st, 0.1), BT_RUNNING)
    assert_equal(t_rep.tick(bb, st, 0.1), BT_SUCCESS)
    assert_equal(st.action_exec_counts[0], 3)
    # Repeat fails when the child fails
    st.set_action_status(0, BT_FAILURE)
    assert_equal(t_rep.tick(bb, st, 0.1), BT_FAILURE)

    # Timeout(1.0) over a never-ending action
    var t_to = BehaviorTree()
    _ = t_to.begin_timeout(1.0)
    _ = t_to.add_action(1)
    t_to.end_composite()
    var st2 = BtExecutionState()
    st2.set_action_status(1, BT_RUNNING)
    for _ in range(4):
        assert_equal(t_to.tick(bb, st2, 0.25), BT_RUNNING)
    assert_equal(t_to.tick(bb, st2, 0.25), BT_FAILURE)
    # Child finishing in time passes through
    st2.set_action_status(1, BT_SUCCESS)
    assert_equal(t_to.tick(bb, st2, 0.25), BT_SUCCESS)
    print("[PASS] BT Repeat/Timeout behave")

    # Wait leaf, Succeeder / ForceFailure, Bool & entity keys, subtree, service
    var t_w = BehaviorTree()
    _ = t_w.begin_sequence()
    _ = t_w.add_wait(0.5)
    _ = t_w.add_action(2)
    t_w.end_composite()
    var st3 = BtExecutionState()
    assert_equal(t_w.tick(bb, st3, 0.25), BT_RUNNING)
    assert_equal(t_w.tick(bb, st3, 0.25), BT_RUNNING)
    assert_equal(t_w.tick(bb, st3, 0.25), BT_SUCCESS)

    var t_sf = BehaviorTree()
    _ = t_sf.begin_sequence()
    _ = t_sf.begin_succeeder()
    _ = t_sf.add_action(3)
    t_sf.end_composite()
    _ = t_sf.begin_force_failure()
    _ = t_sf.add_action(4)
    t_sf.end_composite()
    t_sf.end_composite()
    var st4 = BtExecutionState()
    st4.set_action_status(3, BT_FAILURE)
    st4.set_action_status(4, BT_SUCCESS)
    assert_equal(t_sf.tick(bb, st4, 0.1), BT_FAILURE)
    assert_equal(st4.action_exec_counts[4], 1)

    var k_alert = bb.def_bool_key("alert")
    var k_target = bb.def_entity_key("target")
    var sub = BehaviorTree()
    _ = sub.begin_sequence()
    _ = sub.add_blackboard_entity_set(k_target)
    _ = sub.add_action(6)
    sub.end_composite()
    var t_main = BehaviorTree()
    _ = t_main.begin_selector()
    _ = t_main.begin_sequence()
    _ = t_main.add_blackboard_bool(k_alert, True)
    _ = t_main.add_subtree(sub, "engage")
    t_main.end_composite()
    _ = t_main.begin_service(9, 0.5)
    _ = t_main.add_action(7)
    t_main.end_composite()
    t_main.end_composite()
    var st5 = BtExecutionState()
    st5.set_action_status(6, BT_SUCCESS)
    st5.set_action_status(7, BT_RUNNING)
    for _ in range(4):
        _ = t_main.tick(bb, st5, 0.25)
    assert_equal(len(st5.take_services()), 2)
    assert_equal(st5.action_exec_counts[6], 0)
    bb.set_bool(k_alert, True)
    bb.set_entity(k_target, UInt64(0xABCD00000001))
    st5.clear_running_in_range(0, len(t_main.nodes))
    assert_equal(t_main.tick(bb, st5, 0.25), BT_SUCCESS)
    assert_equal(st5.action_exec_counts[6], 1)
    assert_equal(bb.get_entity(k_target), UInt64(0xABCD00000001))

    # Task library: SetBlackboard -> MoveTo (navmesh) -> Wait
    var w = _flat_world()
    var nav = NavMesh.build_from_physics(w, NavMeshConfig())
    var bb2 = Blackboard()
    var k_goal = bb2.def_key("goal", initial_vec=Vec3(4.0, 0.0, 1.5))
    var k_done = bb2.def_bool_key("moving")
    var tree = BehaviorTree()
    _ = tree.begin_sequence()
    _ = tree.add_action(10)
    _ = tree.add_action(11)
    _ = tree.add_action(12)
    tree.end_composite()
    var lib = BtTaskLibrary()
    _ = lib.add_set_bool(10, k_done, True)
    _ = lib.add_move_to(11, k_goal, 0.3)
    _ = lib.add_wait(12, 0.2)
    var st6 = BtExecutionState()
    lib.prime(st6)
    var ctx = BtAgentContext(Vec3(-4.0, 0.0, -1.0), 4.0)
    var status = BT_RUNNING
    var ticks = 0
    while status != BT_SUCCESS and ticks < 600:
        status = lib.tick(tree, bb2, st6, ctx, nav, 0.02)
        ticks += 1
    assert_equal(status, BT_SUCCESS)
    assert_true(bb2.get_bool(k_done))
    assert_true((ctx.position - Vec3(4.0, 0.0, 1.5)).length() < Float32(0.5))
    print("  task tree ticks:", ticks)
    print("[PASS] BT Wait/Succeeder/ForceFailure/subtree/service/bool+entity keys + task library")


def test_eqs_and_cover() raises:
    var w = _flat_world()
    var nav = NavMesh.build_from_physics(w, NavMeshConfig())
    var q = EnvQuery.grid(Vec3(0.0, 0.0, 0.0), 4.0, 1.0)
    _ = q.add_distance_test(Vec3(3.0, 0.0, 2.0), 1.0, False)
    _ = q.add_path_length_test(Vec3(3.0, 0.0, 2.0), 0.5, True)
    var res = q.run(nav, w)
    assert_true(res.found())
    assert_true(res.valid_count() > 20)
    assert_true((res.best() - Vec3(3.0, 0.0, 2.0)).length() < Float32(0.8))
    # Ring + LOS filter: only points hidden from the threat by a wall survive
    var w2 = _flat_world()
    _ = w2.add_box(Vec3(0.0, 1.0, 0.0), Vec3(2.0, 1.0, 0.3), mass=0.0)
    var nav2 = NavMesh.build_from_physics(w2, NavMeshConfig())
    var threat = Vec3(0.0, 1.6, 3.0)
    var q2 = EnvQuery.ring(Vec3(0.0, 0.0, 0.0), 1.0, 2.5, 3, 16)
    _ = q2.add_los_test(threat, False, True)
    _ = q2.add_distance_test(Vec3(0.0, 0.0, -1.0), 1.0, False)
    var r2 = q2.run(nav2, w2)
    assert_true(r2.found())
    assert_true(r2.best().z < Float32(0.0))
    # Cover points from navmesh boundary edges along the wall
    var covers = generate_cover_points(nav2, w2, 0.75)
    assert_true(len(covers) > 0)
    var ci = find_best_cover(covers, Vec3(0.0, 0.0, -3.0), threat, w2)
    assert_true(ci >= 0)
    assert_true(covers[ci].pos.z < Float32(0.0))
    assert_true(covers[ci].wall_normal.z > Float32(0.5))
    print("  eqs best:", res.best().x, res.best().z, " covers:", len(covers))
    print("[PASS] EQS picks the best point; cover points from navmesh edges")


def test_camera_blend() raises:
    var w = PhysicsWorld()
    var brain = CameraBrain()
    var a = VirtualCamera.fixed(Vec3(0.0, 3.0, -6.0), 10)
    var b = VirtualCamera.fixed(Vec3(6.0, 3.0, 0.0), 5)
    b.blend_in_time = 1.0
    var ida = brain.add_vcam(a^)
    var idb = brain.add_vcam(b^)
    for _ in range(3):
        brain.update(w, 0.1)
    assert_equal(brain.active_id, ida)
    assert_true((brain.pos - Vec3(0.0, 3.0, -6.0)).length() < Float32(1.0e-4))
    brain.set_priority(idb, 20)
    for _ in range(5):
        brain.update(w, 0.1)
    assert_equal(brain.active_id, idb)
    assert_true(brain.blending)
    var d_a = (brain.pos - Vec3(0.0, 3.0, -6.0)).length()
    var d_b = (brain.pos - Vec3(6.0, 3.0, 0.0)).length()
    assert_true(d_a > Float32(1.0) and d_b > Float32(1.0))
    for _ in range(7):
        brain.update(w, 0.1)
    assert_true(not brain.blending)
    assert_true((brain.pos - Vec3(6.0, 3.0, 0.0)).length() < Float32(1.0e-3))
    # Event-driven shake
    brain.shake_event(Vec3(6.0, 3.0, 0.0), 1.0, 10.0)
    brain.update(w, 0.05)
    assert_true(brain.trauma > Float32(0.0))
    assert_true((brain.pos - brain.unshaken_pos).length() > Float32(0.0) or brain.rot.dot(brain.unshaken_rot) < Float32(0.99999))
    # Lock-on framing: camera behind the player, looking toward the target
    var brain2 = CameraBrain()
    var lk = VirtualCamera(VCAM_LOCK_ON, 1, 0.0)
    var idl = brain2.add_vcam(lk^)
    brain2.set_follow_target(idl, Vec3(0.0, 0.0, 0.0))
    brain2.set_lock_target(idl, Vec3(0.0, 0.0, 10.0))
    brain2.update(w, 0.1)
    assert_true(brain2.pos.z < Float32(-1.0))
    assert_true(brain2.forward().z > Float32(0.8))
    print("[PASS] camera blends between two vcams (+ shake, lock-on)")

    # ECS plugin smoke: follow vcam bound to an entity
    var app = EngineApp(AppConfig(title="CamPluginTest", headless=True, worker_threads=0))
    app.add_plugin(CameraPlugin())
    app.startup()
    var player = app.world.spawn()
    _ = app.world.insert[LocalTransform](player, LocalTransform(Vec3(2.0, 0.0, 3.0)))
    var cam_e = app.world.spawn()
    _ = app.world.insert[VirtualCameraComponent](cam_e, VirtualCameraComponent(priority=3, follow_entity=player.to_bits()))
    for _ in range(3):
        app.step_one_frame(Float32(1.0 / 60.0))
    var res_opt = app.resources.get_ptr[CameraBrainResource]()
    assert_true(Bool(res_opt))
    ref br = res_opt.value()[unsafe_offset=0].brain
    assert_true(br.active_id >= 0)
    assert_true((br.pos - Vec3(2.0, 1.5, 3.0)).length() > Float32(1.0))
    assert_true((br.pos - Vec3(2.0, 1.5, 3.0)).length() < Float32(6.0))
    print("[PASS] CameraPlugin drives the brain from ECS entities")


def main() raises:
    test_diagonal_wall_few_polys()
    test_area_cost_filters_and_radii()
    test_sliced_search()
    test_repath_after_carve()
    test_offmesh_traversal_events()
    test_perception_memory()
    test_bt_decorators_and_tasks()
    test_eqs_and_cover()
    test_camera_blend()
    print("ALL FLINT V5 NAV/AI/CAMERA GAP TESTS PASSED")

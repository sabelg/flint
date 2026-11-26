"""Shared POD component schema and gameplay constants for `test_harness_game`."""
from std.sys import size_of
from flint.app.hot_reload import (
    POD_FIELD_INT,
    PodComponentSchema,
    PodFieldDesc,
)
from flint.ecs.world import STORAGE_TABLE, comp_hash64_str

comptime HARNESS_PLAYER_SPEED: Float32 = 4.5
comptime HARNESS_ATTACK_RANGE: Float32 = 3.6
comptime HARNESS_ATTACK_DAMAGE: Int = 15
comptime HARNESS_PICKUP_SCORE: Int = 100
comptime HARNESS_HIT_SCORE: Int = 50
comptime HARNESS_KILL_SCORE: Int = 100
comptime HARNESS_GOAL_SCORE: Int = 200
comptime HARNESS_PLUGIN_RES_ID: UInt64 = comp_hash64_str("test_harness_game.HarnessPluginTickStats")


@fieldwise_init
struct HarnessStats(Copyable, ImplicitlyCopyable, Movable):
    """Hot-reloadable POD gameplay component synced across `gameplay.so` and `main.mojo`."""
    comptime COMPONENT_ID: UInt64 = comp_hash64_str("HarnessStats")
    var hp: Int32
    var max_hp: Int32
    var score: Int32
    var pickup_collected: Int32
    var enemy_hp: Int32
    var goal_reached: Int32
    var enemy_hits: Int32
    var nav_waypoints: Int32

    def __init__(out self):
        self.hp = Int32(100)
        self.max_hp = Int32(100)
        self.score = Int32(0)
        self.pickup_collected = Int32(0)
        self.enemy_hp = Int32(30)
        self.goal_reached = Int32(0)
        self.enemy_hits = Int32(0)
        self.nav_waypoints = Int32(0)


@fieldwise_init
struct HarnessPluginTickStats(Copyable, ImplicitlyCopyable, Movable):
    """POD resource updated by `gameplay.so` on load and step."""
    var plugin_steps: Int32
    var entities_seen: Int32
    var last_score: Int32
    var reload_count: Int32


def build_harness_stats_schema() -> PodComponentSchema:
    var s = PodComponentSchema("HarnessStats", size_of[HarnessStats](), 4, STORAGE_TABLE)
    _ = s.add_field(PodFieldDesc("hp", POD_FIELD_INT, 0, 4, Float32(0.0), Float32(1000.0), Float32(1.0)))
    _ = s.add_field(PodFieldDesc("max_hp", POD_FIELD_INT, 4, 4, Float32(1.0), Float32(1000.0), Float32(1.0)))
    _ = s.add_field(PodFieldDesc("score", POD_FIELD_INT, 8, 4, Float32(0.0), Float32(100000.0), Float32(10.0)))
    _ = s.add_field(PodFieldDesc("pickup_collected", POD_FIELD_INT, 12, 4, Float32(0.0), Float32(1.0), Float32(1.0)))
    _ = s.add_field(PodFieldDesc("enemy_hp", POD_FIELD_INT, 16, 4, Float32(0.0), Float32(1000.0), Float32(1.0)))
    _ = s.add_field(PodFieldDesc("goal_reached", POD_FIELD_INT, 20, 4, Float32(0.0), Float32(1.0), Float32(1.0)))
    _ = s.add_field(PodFieldDesc("enemy_hits", POD_FIELD_INT, 24, 4, Float32(0.0), Float32(1000.0), Float32(1.0)))
    _ = s.add_field(PodFieldDesc("nav_waypoints", POD_FIELD_INT, 28, 4, Float32(0.0), Float32(1000.0), Float32(1.0)))
    return s^

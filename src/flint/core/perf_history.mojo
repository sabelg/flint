"""Performance history for the regression gate (package 0.7).

Every gate run appends one row to `docs/PERF_HISTORY.csv`: the date, commit,
package, result, the profiled replay's frame times at 3440x1440 on the RX 9070,
and a chosen number from each benchmark. A number more than 10% worse than the
median of the same column in the last three rows that passed (or were accepted)
fails the gate. Every number here is a time or a size, so higher is worse.

Each number also has a noise floor: a change smaller than the floor never counts,
however large it is in percent. A floor is 1.5 times the range the column's gate
value covered over 8 repeated gate measurements on rog (`SD_GATE_ONLY=noise
bash tools/regression_gate.sh`; the measurements are in docs/PERF_HISTORY.md).
On every column but the two voxel meshing times that is under 10% of the
number, so the floor changes nothing and the plain 10% rule decides.

Memory peaks (`mem_*` columns, package 0.11) are exact for a replay, so they
are judged at 2% (MEMORY_GROWTH_LIMIT) instead of 10%: the memory-growth rule.

A package that knowingly trades speed for a feature adds a line
`- accept <package> <column|all>: <why>` to docs/PERF_HISTORY.md, and the gate
records that row as `accepted` instead of failing.

An accepted column restarts its baseline (package 0.13): for the rows after it,
that column is judged against the median of the last three passing rows from
the accepting row onward (that row alone until more exist), never against rows
from before the accepted cost. A column nobody accepted keeps the plain rule.
Which rows accepted which columns is read from the history itself: a row whose
result is `accepted`, of a package with an accept line for the column.

A column over its limit is measured a second time before it fails (`settle`),
and every row reports each column's drift since the first passing row of its
phase and since the 0.7 baseline (`drift`); see docs/PERF_HISTORY.md.
"""

comptime REGRESSION_LIMIT: Float64 = 0.10
# Tracked memory peaks (0.11) are exact: the same replay charges the same bytes
# every run. Growth past 2% fails, however the times look.
comptime MEMORY_GROWTH_LIMIT: Float64 = 0.02
comptime BASELINE_ROWS: Int = 3
# 1.5 x the range of dig_remesh_p99_ms over six golden-perf runs on rog (docs/PERF_HISTORY.md).
comptime DIG_REMESH_FLOOR: Float64 = 0.22
comptime FIXED_COLUMNS: Int = 4  # date, commit, package, result
# Drift (0.13): a column this much above the first passing row of its phase is
# a warning in the gate summary and the runner's notification, never a failure.
comptime DRIFT_WARNING: Float64 = 0.25
comptime BASELINE_PACKAGE = "0.7"


struct PerfMetric(Copyable, Movable):
    var name: String      # CSV column
    var source: String    # which log: perf, voxel, physics, ...
    var prefix: String    # the number follows this text on the last line that has it
    var floor: Float64    # changes smaller than this never count
    var budget: Float64   # an absolute limit (0 = none)
    var limit: Float64    # how much worse than the baseline fails, as a fraction

    def __init__(out self, name: String, source: String, prefix: String, floor: Float64, budget: Float64 = 0, limit: Float64 = REGRESSION_LIMIT):
        self.name = name
        self.source = source
        self.prefix = prefix
        self.floor = floor
        self.budget = budget
        self.limit = limit


def perf_metrics() -> List[PerfMetric]:
    # Floors: 1.5 x the measured range (docs/PERF_HISTORY.md, "Noise floors").
    # Re-measure and update both when a column or the machine changes.
    var list = List[PerfMetric]()
    # The profiled replay of the heaviest golden recording at 3440x1440.
    list.append(PerfMetric("frame_p50_ms", "perf", "frame_p50_ms=", 0.034))
    list.append(PerfMetric("frame_p99_ms", "perf", "frame_p99_ms=", 0.26))
    list.append(PerfMetric("tick_p99_ms", "perf", "tick_p99_ms=", 0.003))
    # Physics sub-budget from the 2026-11 physics research: 4 ms p99 per tick.
    list.append(PerfMetric("physics_tick_p99_ms", "perf", "physics_tick_p99_ms=", 0.0034, 4.0))
    list.append(PerfMetric("peak_rss_mib", "perf", "peak_rss_mib=", 38))
    # The memory-growth rule (0.11): tracked peaks of the same replay. The
    # floor is a quarter MiB (the frame arena's in KiB), the limit 2%.
    list.append(PerfMetric("mem_cpu_peak_mib", "perf", "mem_cpu_peak_mib=", 0.25, 0, MEMORY_GROWTH_LIMIT))
    list.append(PerfMetric("mem_gpu_peak_mib", "perf", "mem_gpu_peak_mib=", 0.25, 0, MEMORY_GROWTH_LIMIT))
    list.append(PerfMetric("mem_terrain_peak_mib", "perf", "mem_terrain_peak_mib=", 0.25, 0, MEMORY_GROWTH_LIMIT))
    list.append(PerfMetric("mem_frame_peak_kib", "perf", "mem_frame_peak_kib=", 4, 0, MEMORY_GROWTH_LIMIT))
    # Digging (package 1.6): the profiled destruction replay's remesh time on
    # frames with dirty chunks, and the meshing memory peaks (the CPU chunk
    # meshes and the GPU chunk-mesh heap), held to the memory rule.
    list.append(PerfMetric("dig_remesh_p99_ms", "perf", "dig_remesh_p99_ms=", DIG_REMESH_FLOOR))
    list.append(PerfMetric("mem_mesh_peak_mib", "perf", "mem_mesh_peak_mib=", 0.25, 0, MEMORY_GROWTH_LIMIT))
    list.append(PerfMetric("mesh_heap_peak_mib", "perf", "mesh_heap_peak_mib=", 0.25, 0, MEMORY_GROWTH_LIMIT))
    # Benchmarks (tools/run.sh benchmark and benchmark-smooth).
    list.append(PerfMetric("voxel_initial_mesh_ms", "voxel", "initial_mesh_ms=", 0.57))
    list.append(PerfMetric("voxel_dirty_remesh_ms", "voxel", "dirty_remesh_ms=", 0.47))
    list.append(PerfMetric("voxel_carve_ms", "voxel", "carve_ms=", 0.0031))
    list.append(PerfMetric("physics_debris_step_p50_ms", "physics", "debris_step_p50_ms=", 0.00078))
    list.append(PerfMetric("physics_debris_step_p99_ms", "physics", "debris_step_p99_ms=", 0.0034))
    list.append(PerfMetric("physics_destruction_step_p95_ms", "physics", "destruction_step_p95_ms=", 0.0013))
    list.append(PerfMetric("physics_projectile_sweep_p50_ms", "physics", "projectile_sweep_p50_ms=", 0.0000016))
    list.append(PerfMetric("lighting_vertex_q2_p50_ms", "lighting", "vertex_lighting_q2_p50_ms=", 0.00032))
    list.append(PerfMetric("lighting_ao_mesh_p50_ms", "lighting", "voxel_ao_mesh_p50_ms=", 0.0064))
    list.append(PerfMetric("lighting_post_q2_p50_ms", "lighting", "scene_post_q2_p50_ms=", 0.016))
    list.append(PerfMetric("systemic_r2_update_p50_ms", "systemic", "heavy_r2_standard_update_p50_ms=", 0.00076))
    list.append(PerfMetric("systemic_r2_physics_p50_ms", "systemic", "heavy_r2_standard_physics_step_p50_ms=", 0.00038))
    list.append(PerfMetric("smooth_exposed_chunk_ms", "smooth", "exposed=", 0.0015))
    list.append(PerfMetric("smooth_smooth_chunk_ms", "smooth", "smooth=", 0.015))
    list.append(PerfMetric("jobs_chain_ns", "jobs", "empty job in a chain", 3.9))
    list.append(PerfMetric("jobs_fan_out_us", "jobs", "fan-out of 64 chunks:", 0.45))
    list.append(PerfMetric("profiler_zone_ns", "profiler", "zone on, one thread", 0.3))
    list.append(PerfMetric("profiler_end_frame_file_ns", "profiler", "end_frame with 1000 events, file", 450))
    list.append(PerfMetric("debug_draw_scene_ms", "debug-draw", "a scene: 40 boxes, 40 spheres, 40 labels :", 0.0037))
    list.append(PerfMetric("debug_draw_busy_ms", "debug-draw", "busy: 200 boxes, 200 spheres, 200 labels :", 0.014))
    # Collision scenes (package 2.11, tools/run.sh benchmark-collision). Floors
    # are 1.5 x the range of six runs on the dev machine (docs/PERF_HISTORY.md).
    list.append(PerfMetric("collision_rain_step_p50_ms", "collision", "rain_step_p50_ms=", 0.03))
    list.append(PerfMetric("collision_pile_step_p50_ms", "collision", "pile_step_p50_ms=", 0.02))
    list.append(PerfMetric("collision_throw_step_p50_ms", "collision", "throw_step_p50_ms=", 0.003))
    list.append(PerfMetric("collision_query_ray_us", "collision", "query_ray_us=", 0.012))
    # Solver scenes (package 2.2, tools/run.sh benchmark-solver; the 0.7
    # physics scenes). Floors are 1.5 x the range of six runs on rog
    # (docs/PERF_HISTORY.md).
    list.append(PerfMetric("solver_pyramid_step_p50_ms", "solver", "pyramid_step_p50_ms=", 0.0005))
    list.append(PerfMetric("solver_pile_step_p50_ms", "solver", "pile_step_p50_ms=", 0.016))
    list.append(PerfMetric("solver_pile_step_p99_ms", "solver", "pile_step_p99_ms=", 0.052))
    list.append(PerfMetric("solver_rain_step_p50_ms", "solver", "rain_step_p50_ms=", 0.034))
    list.append(PerfMetric("solver_smash_step_p50_ms", "solver", "smash_step_p50_ms=", 0.018))
    list.append(PerfMetric("solver_throw_step_p50_ms", "solver", "throw_step_p50_ms=", 0.0029))
    list.append(PerfMetric("solver_tumbler_step_p50_ms", "solver", "tumbler_step_p50_ms=", 0.0025))
    list.append(PerfMetric("solver_chain_step_p50_ms", "solver", "chain_step_p50_ms=", 0.0006))
    list.append(PerfMetric("solver_pile_7w_step_p50_ms", "solver", "pile_7w_step_p50_ms=", 0.0069))
    # Forces, wind and floating (package 2.12, tools/run.sh benchmark-forces).
    # Floors are 1.5 x the range of six runs on rog (docs/PERF_HISTORY.md).
    list.append(PerfMetric("forces_flotilla_step_p50_ms", "forces", "flotilla_step_p50_ms=", 0.0017))
    list.append(PerfMetric("forces_fields_step_p50_ms", "forces", "fields_step_p50_ms=", 0.0004))
    list.append(PerfMetric("forces_joints_step_p50_ms", "forces", "joints_step_p50_ms=", 0.0018))
    list.append(PerfMetric("forces_ropes_step_p50_ms", "forces", "ropes_step_p50_ms=", 0.0033))
    list.append(PerfMetric("forces_wind_eval_ns", "forces", "wind_eval_ns=", 12.0))
    list.append(PerfMetric("forces_wind_mirror_bake_p50_ms", "forces", "wind_mirror_bake_p50_ms=", 0.034))
    list.append(PerfMetric("forces_calm_step_p50_ms", "forces", "calm_step_p50_ms=", 0.069))
    # Grab, carry, throw and surf (package 2.3, tools/run.sh benchmark-carry).
    # Floors are 1.5 x the range of six runs on rog (docs/PERF_HISTORY.md).
    list.append(PerfMetric("carry_toss_step_p50_ms", "carry", "toss_step_p50_ms=", 0.095))
    list.append(PerfMetric("carry_split_step_p50_ms", "carry", "split_step_p50_ms=", 0.011))
    list.append(PerfMetric("carry_surf_tick_p50_ms", "carry", "surf_tick_p50_ms=", 0.0001))
    # The effects library's heavy scene (package 2.13, tools/run.sh
    # benchmark-fx: 200 decals, 20 ambient areas with rain, beams, 3440x1440
    # on the RX 9070). Floors are 1.5 x the range of six runs on rog
    # (docs/PERF_HISTORY.md).
    list.append(PerfMetric("fx_heavy_frame_p50_ms", "fx", "heavy_frame_p50_ms=", 0.2))
    list.append(PerfMetric("fx_cpu_p50_ms", "fx", "fx_cpu_p50_ms=", 0.011))
    list.append(PerfMetric("fx_decal_spawn_us", "fx", "decal_spawn_us=", 0.21))
    # Movers and layer mechanics (package 2.14, tools/run.sh
    # benchmark-mechanics: mechanics.demo's game tick, its own work, lifts,
    # belts, magnets and a three-mirror beam). Floors are 1.5 x the range of
    # six runs on rog, rounded up (docs/PERF_HISTORY.md).
    list.append(PerfMetric("mech_demo_tick_p50_ms", "mechanics", "demo_tick_p50_ms=", 0.002))
    list.append(PerfMetric("mech_plain_tick_p50_ms", "mechanics", "plain_tick_p50_ms=", 0.0008))
    list.append(PerfMetric("mech_systems_us", "mechanics", "systems_us=", 0.08))
    list.append(PerfMetric("mech_movers_step_p50_ms", "mechanics", "movers_step_p50_ms=", 0.002))
    list.append(PerfMetric("mech_belts_step_p50_ms", "mechanics", "belts_step_p50_ms=", 0.0017))
    list.append(PerfMetric("mech_magnets_step_p50_ms", "mechanics", "magnets_step_p50_ms=", 0.001))
    list.append(PerfMetric("mech_beam_trace_us", "mechanics", "beam_trace_us=", 0.08))
    # The hum (package 3.24, tools/run.sh benchmark-hum): a whole frame of the
    # heaviest restored layer (gold lit, every ambient emitter at full rate)
    # at 3440x1440 on the RX 9070. The floor is the one of
    # fx_heavy_frame_p50_ms times 1.5 (the same kind of frame in the same
    # layer); the machine was busy with the package's builds when the column
    # was added, so its own six runs are still to be measured.
    list.append(PerfMetric("hum_restored_frame_p50_ms", "hum", "hum_restored_frame_p50_ms=", 0.3))
    return list^


# ---------------------------------------------------------------------------
# Reading numbers out of logs
# ---------------------------------------------------------------------------

def is_number_byte(byte: UInt8) -> Bool:
    return (byte >= 48 and byte <= 57) or byte == 46 or byte == 45 or byte == 43 or byte == 101 or byte == 69


def parse_number(text: String) -> Tuple[Bool, Float64]:
    try:
        return (True, Float64(text))
    except:
        return (False, Float64(0))


def is_word_byte(byte: UInt8) -> Bool:
    return (byte >= 48 and byte <= 57) or (byte >= 65 and byte <= 90) or (byte >= 97 and byte <= 122) or byte == 95


def word_start(piece: String, prefix: String) -> Int:
    """Where `prefix` starts a word in the line (so `tick_p99_ms=` does not
    match inside `physics_tick_p99_ms=`), or -1."""
    var bytes = piece.as_bytes()
    var start = 0
    while True:
        var at = piece.find(prefix, start)
        if at < 0:
            return -1
        if at == 0 or not is_word_byte(bytes[at - 1]):
            return at
        start = at + 1


def number_after(text: String, prefix: String) -> Tuple[Bool, Float64]:
    """The number after `prefix` on the last line where it starts a word."""
    var found = False
    var value = Float64(0)
    for line in text.split("\n"):
        var piece = String(line)
        var at = word_start(piece, prefix)
        if at < 0:
            continue
        var bytes = piece.as_bytes()
        var index = at + prefix.byte_length()
        while index < len(bytes) and (bytes[index] == 32 or bytes[index] == 9):
            index += 1
        var start = index
        while index < len(bytes) and is_number_byte(bytes[index]):
            index += 1
        if index > start:
            var parsed = parse_number(String(piece[byte=start:index]))
            if parsed[0]:
                found = True
                value = parsed[1]
    return (found, value)


# ---------------------------------------------------------------------------
# The CSV file
# ---------------------------------------------------------------------------

struct PerfTable(Movable):
    var header: List[String]
    var rows: List[List[String]]

    def __init__(out self):
        self.header = ["date", "commit", "package", "result"]
        self.rows = List[List[String]]()

    def column(self, name: String) -> Int:
        for index in range(len(self.header)):
            if self.header[index] == name:
                return index
        return -1

    def ensure_column(mut self, name: String) -> Int:
        """Adds a column at the end; older rows get an empty cell."""
        var index = self.column(name)
        if index >= 0:
            return index
        self.header.append(name)
        for row_index in range(len(self.rows)):
            self.rows[row_index].append(String())
        return len(self.header) - 1

    def cell(self, row: Int, column: Int) -> String:
        if column < 0 or column >= len(self.rows[row]):
            return String()
        return self.rows[row][column]

    def text(self) -> String:
        var out = String()
        out += ",".join(self.header) + "\n"
        for row in self.rows:
            out += ",".join(row) + "\n"
        return out


def parse_table(text: String) -> PerfTable:
    var table = PerfTable()
    var first = True
    for line in text.split("\n"):
        var piece = String(String(line).strip())
        if piece.byte_length() == 0:
            continue
        var cells = List[String]()
        for cell in piece.split(","):
            cells.append(String(cell))
        if first:
            table.header = cells^
            first = False
            continue
        while len(cells) < len(table.header):
            cells.append(String())
        table.rows.append(cells^)
    return table^


def median(values: List[Float64]) -> Float64:
    var ordered = List[Float64]()
    for value in values:
        var position = len(ordered)
        ordered.append(value)
        while position > 0 and ordered[position - 1] > value:
            ordered[position] = ordered[position - 1]
            position -= 1
        ordered[position] = value
    var count = len(ordered)
    if count == 0:
        return 0
    if count % 2 == 1:
        return ordered[count // 2]
    return (ordered[count // 2 - 1] + ordered[count // 2]) / 2


struct AcceptLine(Copyable, Movable):
    """One `- accept <package> <column|all>: <why>` line of docs/PERF_HISTORY.md."""
    var package: String
    var column: String

    def __init__(out self, package: String, column: String):
        self.package = package
        self.column = column


def accept_lines(notes: String) -> List[AcceptLine]:
    """Every accept line in the notes, for any package, in file order."""
    var out = List[AcceptLine]()
    for line in notes.split("\n"):
        var piece = String(String(line).strip())
        if not piece.startswith("- accept "):
            continue
        var words = List[String]()
        for word in piece.split(" "):
            if String(word).byte_length() > 0:
                words.append(String(word))
        if len(words) < 4:
            continue
        var word = words[3]
        if word.endswith(":"):
            out.append(AcceptLine(words[2], String(word[byte=0:word.byte_length() - 1])))
        else:
            out.append(AcceptLine(words[2], word))
    return out^


def is_good_row(table: PerfTable, row: Int) -> Bool:
    var result = table.cell(row, table.column("result"))
    return result == "pass" or result == "accepted"


def value_at(table: PerfTable, row: Int, column: Int) -> Tuple[Bool, Float64]:
    """The number in a cell, if the cell has one."""
    var text = table.cell(row, column)
    if text.byte_length() == 0:
        return (False, Float64(0))
    return parse_number(text)


def row_accepts(table: PerfTable, accepts: List[AcceptLine], row: Int, name: String) -> Bool:
    """Whether this row accepted a cost in this column: its result is
    `accepted` and its package has an accept line for the column (or `all`)."""
    if table.cell(row, table.column("result")) != "accepted":
        return False
    var package = table.cell(row, table.column("package"))
    for line in accepts:
        if line.package == package and (line.column == name or line.column == "all"):
            return True
    return False


def restart_row(table: PerfTable, accepts: List[AcceptLine], name: String, before: Int) -> Int:
    """The last of the first `before` rows that accepted a cost in this column
    and has a number in it: where the column's baseline restarts. -1 if none."""
    var column = table.column(name)
    if column < 0:
        return -1
    var row = min(before, len(table.rows)) - 1
    while row >= 0:
        if row_accepts(table, accepts, row, name) and value_at(table, row, column)[0]:
            return row
        row -= 1
    return -1


def baseline_from(table: PerfTable, accepts: List[AcceptLine], name: String, before: Int) -> Tuple[Int, Float64, Int]:
    """The baseline a new row is judged against, from the first `before` rows:
    how many rows were used (0 to 3), the median of their values, and the row
    the baseline restarts at (-1 if the column was never accepted). The rows
    are the last three that passed or were accepted and have this number, and
    none from before the column's last accepted row."""
    var column = table.column(name)
    var values = List[Float64]()
    if column < 0:
        return (0, Float64(0), -1)
    var restart = restart_row(table, accepts, name, before)
    var row = min(before, len(table.rows)) - 1
    while row >= 0 and row >= restart and len(values) < BASELINE_ROWS:
        if is_good_row(table, row):
            var parsed = value_at(table, row, column)
            if parsed[0]:
                values.append(parsed[1])
        row -= 1
    return (len(values), median(values), restart)


def baseline(table: PerfTable, name: String) -> Tuple[Int, Float64]:
    """The plain rule (0.7), which a column nobody accepted still gets: how many
    earlier rows were used (0 to 3), and the median of their values: the last
    three rows that passed or were accepted and have this number."""
    var base = baseline_from(table, List[AcceptLine](), name, len(table.rows))
    return (base[0], base[1])


struct Finding(Copyable, Movable):
    var metric: String
    var value: Float64
    var baseline: Float64
    var rows: Int
    var worse: Bool        # more than 10% and more than the floor above the baseline
    var over_budget: Bool
    var accepted: Bool
    var restart: String    # the package whose accept the baseline restarts at ("" = plain rule)
    var remeasured: Bool   # over its limit at first, so measured a second time
    var first: Float64     # the first measurement (best of the gate's runs)
    var second: Float64    # the second (best of the runs made again)

    def __init__(out self, metric: String, value: Float64, baseline: Float64, rows: Int):
        self.metric = metric
        self.value = value
        self.baseline = baseline
        self.rows = rows
        self.worse = False
        self.over_budget = False
        self.accepted = False
        self.restart = String()
        self.remeasured = False
        self.first = value
        self.second = value

    def needs_again(self) -> Bool:
        """Over its limit and not accepted: measure it again before failing."""
        return self.over_budget or (self.worse and not self.accepted)

    def change(self) -> Float64:
        if self.baseline <= 0:
            return 0
        return (self.value - self.baseline) / self.baseline


def judge_from(metric: PerfMetric, value: Float64, table: PerfTable, accepts: List[AcceptLine], before: Int) -> Finding:
    """Judges a value against the first `before` rows of the history, with
    baselines restarted at accepted costs (`baseline_from`)."""
    var base = baseline_from(table, accepts, metric.name, before)
    var finding = Finding(metric.name, value, base[1], base[0])
    if base[2] >= 0:
        finding.restart = table.cell(base[2], table.column("package"))
    if base[0] > 0:
        var limit = base[1] * (1.0 + metric.limit)
        finding.worse = value > limit and value - base[1] > metric.floor
    finding.over_budget = metric.budget > 0 and value > metric.budget
    return finding^


def judge(metric: PerfMetric, value: Float64, table: PerfTable) -> Finding:
    """The plain rule: no accept lines, so no baseline restarts."""
    return judge_from(metric, value, table, List[AcceptLine](), len(table.rows))


def settle(metric: PerfMetric, first: Finding, second: Float64, table: PerfTable, accepts: List[AcceptLine], before: Int) -> Finding:
    """A column that was over its limit, measured again (0.13): one noisy run
    must not fail the gate. The column keeps the better of the two measurements
    and fails only if that is still over, which is the same as both being over.
    Both measurements stay in the finding and go into the row."""
    var finding = judge_from(metric, min(first.value, second), table, accepts, before)
    finding.remeasured = True
    finding.first = first.value
    finding.second = second
    return finding^


def remeasured_cell(findings: List[Finding]) -> String:
    """The row's `remeasured` cell: `<column>=<first>><second>` for every
    column that was measured twice, separated by spaces."""
    var parts = List[String]()
    for finding in findings:
        if finding.remeasured:
            parts.append(finding.metric + "=" + String(finding.first) + ">" + String(finding.second))
    return " ".join(parts)


def build_seconds_cell(text: String) -> String:
    """The row's `build_s` cell (package 0.14): how long the gate's build step
    took, whole seconds, or empty when it is not a whole number (the gate
    reused build/, or ran only the perf step). Written, never judged: no
    PerfMetric is named build_s."""
    if text.byte_length() == 0:
        return String()
    for byte in text.as_bytes():
        if byte < 48 or byte > 57:
            return String()
    return text


def accepted_columns(notes: String, package: String) -> List[String]:
    """Columns docs/PERF_HISTORY.md accepts for this package: lines of the form
    `- accept <package> <column|all>: <why>`."""
    var out = List[String]()
    for line in accept_lines(notes):
        if line.package == package:
            out.append(line.column)
    return out^


def is_accepted(columns: List[String], name: String) -> Bool:
    for column in columns:
        if column == name or column == "all":
            return True
    return False


# ---------------------------------------------------------------------------
# Drift (package 0.13)
# ---------------------------------------------------------------------------

def package_phase(package: String) -> String:
    """The phase of a plan package: "3" for "3.3". Empty for anything else
    (`manual` rows)."""
    var at = package.find(".")
    if at <= 0:
        return String()
    var bytes = package.as_bytes()
    for index in range(at):
        if bytes[index] < 48 or bytes[index] > 57:
            return String()
    return String(package[byte=0:at])


def signed_percent(fraction: Float64) -> String:
    """`+12.3%` or `-4.0%`, to a tenth of a percent."""
    var tenths = Int(abs(fraction) * 1000.0 + 0.5)
    var sign = String("-") if fraction < 0 and tenths > 0 else String("+")
    return sign + String(tenths // 10) + "." + String(tenths % 10) + "%"


struct Drift(Copyable, Movable):
    """How far one column of a new row has moved since the first passing row of
    the package's phase and since the 0.7 baseline row, and which packages
    accepted a cost in it on the way. A report, never a failure."""
    var metric: String
    var value: Float64
    var floor: Float64
    var phase: String
    var phase_rows: Int             # 0: this row is the phase's first with the column
    var phase_base: Float64
    var phase_accepts: List[String]
    var origin: String              # the package of the row the long drift is from
    var origin_rows: Int
    var origin_base: Float64
    var origin_accepts: List[String]

    def __init__(out self, metric: String, value: Float64, floor: Float64, phase: String):
        self.metric = metric
        self.value = value
        self.floor = floor
        self.phase = phase
        self.phase_rows = 0
        self.phase_base = value
        self.phase_accepts = List[String]()
        self.origin = String()
        self.origin_rows = 0
        self.origin_base = value
        self.origin_accepts = List[String]()

    def phase_change(self) -> Float64:
        if self.phase_rows == 0 or self.phase_base <= 0:
            return 0
        return (self.value - self.phase_base) / self.phase_base

    def origin_change(self) -> Float64:
        if self.origin_rows == 0 or self.origin_base <= 0:
            return 0
        return (self.value - self.origin_base) / self.origin_base

    def warns(self) -> Bool:
        """More than 25% (and more than the noise floor) worse in one phase."""
        return self.phase_change() > DRIFT_WARNING and self.value - self.phase_base > self.floor

    def line(self) -> String:
        """`<column> +N% since phase <P> (accepted in ...); +M% since 0.7 (...)`."""
        var text = self.metric + " " + signed_percent(self.phase_change()) + " since phase " + self.phase
        if len(self.phase_accepts) > 0:
            text += " (accepted in " + ", ".join(self.phase_accepts) + ")"
        if self.origin_rows > 0:
            text += "; " + signed_percent(self.origin_change()) + " since " + self.origin
            if self.origin != BASELINE_PACKAGE:
                text += ", the column's first row"
            if len(self.origin_accepts) > 0:
                text += " (accepted in " + ", ".join(self.origin_accepts) + ")"
        return text


def accepting_packages(table: PerfTable, accepts: List[AcceptLine], name: String, after: Int, before: Int) -> List[String]:
    """The packages that accepted a cost in this column in rows after..before
    (both exclusive), each once, in row order."""
    var out = List[String]()
    for row in range(max(after + 1, 0), min(before, len(table.rows))):
        if not row_accepts(table, accepts, row, name):
            continue
        var package = table.cell(row, table.column("package"))
        var known = False
        for seen in out:
            if seen == package:
                known = True
        if not known:
            out.append(package)
    return out^


def drift(
    table: PerfTable, accepts: List[AcceptLine], metric: PerfMetric, value: Float64,
    package: String, accepted_now: Bool, before: Int,
) -> Drift:
    """The drift of `value`, a new row of `package` judged after the first
    `before` rows. The phase is the package's (a `manual` row takes the last
    plan row's); its start is the first passing row of that phase with the
    column, the long drift is from the last passing 0.7 row with it, or from
    the column's first passing row if it is younger than 0.7. Nothing counts
    from before the 0.7 baseline row."""
    var rows = min(before, len(table.rows))
    var column = table.column(metric.name)
    var package_column = table.column("package")
    var phase = package_phase(package)
    var row = rows - 1
    while phase.byte_length() == 0 and row >= 0:
        phase = package_phase(table.cell(row, package_column))
        row -= 1
    var out = Drift(metric.name, value, metric.floor, phase)
    if column < 0:
        return out^
    var origin = -1
    var first = -1
    var phase_start = -1
    for index in range(rows):
        if not is_good_row(table, index) or not value_at(table, index, column)[0]:
            continue
        var row_package = table.cell(index, package_column)
        if first < 0:
            first = index
        if row_package == BASELINE_PACKAGE:
            origin = index
        if phase_start < 0 and package_phase(row_package) == phase:
            phase_start = index
    if origin < 0:
        origin = first
    if origin >= 0:
        out.origin = table.cell(origin, package_column)
        out.origin_rows = 1
        out.origin_base = value_at(table, origin, column)[1]
        out.origin_accepts = accepting_packages(table, accepts, metric.name, origin, rows)
        if accepted_now:
            append_once(out.origin_accepts, package)
    if phase_start >= 0:
        phase_start = max(phase_start, origin)
        out.phase_rows = 1
        out.phase_base = value_at(table, phase_start, column)[1]
        out.phase_accepts = accepting_packages(table, accepts, metric.name, phase_start, rows)
        if accepted_now:
            append_once(out.phase_accepts, package)
    return out^


def append_once(mut list: List[String], item: String):
    for seen in list:
        if seen == item:
            return
    list.append(item)


# ---------------------------------------------------------------------------
# Measuring the noise floors
# ---------------------------------------------------------------------------

comptime NOISE_RANGES: Float64 = 1.5
"""A floor is this many times the range (highest minus lowest) of a column's
gate value over repeated gate measurements. Over 8 sets the range is about 2.8
standard deviations, so the floor is about 4: a gate value is compared with the
median of three earlier ones, and 4 standard deviations keeps a false failure
rare across all the columns of a run."""


struct NoiseSummary(Copyable, Movable):
    """One column measured over repeated gate runs, each a best-of-N value."""
    var sets: Int
    var median: Float64
    var low: Float64
    var high: Float64

    def __init__(out self, values: List[Float64]):
        self.sets = len(values)
        self.median = median(values)
        self.low = Float64(0)
        self.high = Float64(0)
        for index in range(len(values)):
            if index == 0 or values[index] < self.low:
                self.low = values[index]
            if index == 0 or values[index] > self.high:
                self.high = values[index]

    def spread(self) -> Float64:
        """The range as a fraction of the median."""
        if self.median <= 0:
            return 0
        return (self.high - self.low) / self.median

    def floor(self) -> Float64:
        """The noise floor the measurement supports, in the column's unit."""
        return NOISE_RANGES * (self.high - self.low)


def effective_limit(floor: Float64, base: Float64) -> Float64:
    """How much worse than `base` a number may get before the gate fails, as a
    fraction: 10%, or the floor if that is larger."""
    if base <= 0:
        return REGRESSION_LIMIT
    return max(REGRESSION_LIMIT, floor / base)

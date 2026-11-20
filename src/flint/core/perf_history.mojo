"""Performance history for the Flint Engine v5 regression gate (`v5.0`).

Every recorded gate run appends one row to `docs/PERF_HISTORY.csv`: the date,
commit, package, result, and the p50/p99 subsystem benchmark measurements from
`tools/flint.sh lab all`. A number more than 10% worse than the median of the
same column in the last three rows that passed (or were accepted) fails the
gate, and any metric exceeding its absolute budget also fails. Every number
here is a time or a size, so higher is worse.

Each metric also has a noise floor: a change smaller than the floor never
counts, however large it is in percent. A floor is 1.5 times the range the
metric covered across repeated `-O3` calibration runs on the reference host
(recorded in `docs/PERF_HISTORY.md`).

Memory-growth metrics use `MEMORY_GROWTH_LIMIT` (2%) instead of 10%.

A release that knowingly trades speed for a feature adds a line
`- accept <package> <column|all>: <why>` to `docs/PERF_HISTORY.md`, and the
gate records that row as `accepted` instead of failing.

An accepted column restarts its baseline: for the rows after it, that column is
judged against the median of the last three passing rows from the accepting row
onward (that row alone until more exist), never against rows from before the
accepted cost. A column nobody accepted keeps the plain rule.

A column over its limit is measured a second time before it fails (`settle`),
and every row reports each column's drift since the first passing row of its
phase and since the `v5.0` baseline (`drift`); see `docs/PERF_HISTORY.md`.
"""

comptime REGRESSION_LIMIT: Float64 = 0.10
# Tracked memory peaks are exact: growth past 2% fails.
comptime MEMORY_GROWTH_LIMIT: Float64 = 0.02
comptime BASELINE_ROWS: Int = 3
comptime FIXED_COLUMNS: Int = 4  # date, commit, package, result
# Drift: a column this much above the first passing row of its phase is a
# warning in the gate summary, never a failure.
comptime DRIFT_WARNING: Float64 = 0.25
comptime BASELINE_PACKAGE = "v5.0"


struct PerfMetric(Copyable, Movable):
    var name: String      # CSV column
    var source: String    # which lab: lab_physics, lab_anim, lab_render, lab_audio_ui, lab_net
    var prefix: String    # text prefix if parsed from stdout logs
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
    # Floors: >= 1.5 x the measured `-O3` range (docs/PERF_HISTORY.md, "Noise floors").
    # Column names match `EngineBenchSuite.to_perf_table()` `<case>_p50_ms` / `<case>_p99_ms` 1-to-1.
    var list = List[PerfMetric]()
    # Physics, KCC, Shapes & Nav/ORCA (`labs/lab_physics.mojo`)
    list.append(PerfMetric("physics_200body_step_p50_ms", "lab_physics", "physics_200body_step_p50_ms=", 0.05, 2.0))
    list.append(PerfMetric("physics_200body_step_p99_ms", "lab_physics", "physics_200body_step_p99_ms=", 0.50, 4.0))
    list.append(PerfMetric("physics_rollback_det_120_300_p99_ms", "lab_physics", "physics_rollback_det_120_300_p99_ms=", 90.0, 1000.0))
    list.append(PerfMetric("physics_shapes_vhacd_dda_p99_ms", "lab_physics", "physics_shapes_vhacd_dda_p99_ms=", 0.05, 2.0))
    list.append(PerfMetric("physics_kcc_stair_90t_p99_ms", "lab_physics", "physics_kcc_stair_90t_p99_ms=", 0.02, 0.25))
    list.append(PerfMetric("nav_obstacle_carve_p99_ms", "lab_physics", "nav_obstacle_carve_p99_ms=", 0.02, 0.25))
    list.append(PerfMetric("nav_orca_256_tick_p50_ms", "lab_physics", "nav_orca_256_tick_p50_ms=", 0.10, 2.0))
    list.append(PerfMetric("nav_orca_256_tick_p99_ms", "lab_physics", "nav_orca_256_tick_p99_ms=", 0.15, 3.0))
    # Animation, IK, Ragdoll, AI & Camera (`labs/lab_anim.mojo`)
    list.append(PerfMetric("anim_flanim_roundtrip_p99_ms", "lab_anim", "anim_flanim_roundtrip_p99_ms=", 0.05, 0.5))
    list.append(PerfMetric("anim_socket_foot_ik_p99_ms", "lab_anim", "anim_socket_foot_ik_p99_ms=", 0.05, 0.5))
    list.append(PerfMetric("anim_ragdoll_sm_tick_p99_ms", "lab_anim", "anim_ragdoll_sm_tick_p99_ms=", 0.09, 0.5))
    list.append(PerfMetric("ai_bt_utility_perc_cam_p99_ms", "lab_anim", "ai_bt_utility_perc_cam_p99_ms=", 0.02, 0.5))
    list.append(PerfMetric("anim_ai_state_det_p99_ms", "lab_anim", "anim_ai_state_det_p99_ms=", 7.0, 50.0))
    # Vulkan VisBuffer/RT Render, SPIR-V & VAT (`labs/lab_render.mojo`)
    list.append(PerfMetric("render_spirv_mat_compile_p99_ms", "lab_render", "render_spirv_mat_compile_p99_ms=", 0.20, 25.0))
    list.append(PerfMetric("render_vat_crowd_bake_p99_ms", "lab_render", "render_vat_crowd_bake_p99_ms=", 0.10, 25.0))
    list.append(PerfMetric("render_world_256x192_frame_p50_ms", "lab_render", "render_world_256x192_frame_p50_ms=", 0.25, 25.0))
    list.append(PerfMetric("render_world_256x192_frame_p99_ms", "lab_render", "render_world_256x192_frame_p99_ms=", 0.30, 50.0))
    list.append(PerfMetric("render_ui_readback_check_p99_ms", "lab_render", "render_ui_readback_check_p99_ms=", 0.05, 50.0))
    # Audio QOA/BVH/Server, Flexbox DockSpace UI & InputMap (`labs/lab_audio_ui.mojo`)
    list.append(PerfMetric("audio_qoa_codec_p99_ms", "lab_audio_ui", "audio_qoa_codec_p99_ms=", 0.90, 5.0))
    list.append(PerfMetric("audio_bvh_acoustics_p99_ms", "lab_audio_ui", "audio_bvh_acoustics_p99_ms=", 0.02, 1.0))
    list.append(PerfMetric("audio_server_quantum_p99_ms", "lab_audio_ui", "audio_server_quantum_p99_ms=", 0.15, 2.0))
    list.append(PerfMetric("ui_dockspace_100k_frame_p99_ms", "lab_audio_ui", "ui_dockspace_100k_frame_p99_ms=", 0.02, 0.5))
    list.append(PerfMetric("input_lockstep_det_p99_ms", "lab_audio_ui", "input_lockstep_det_p99_ms=", 0.02, 5.0))
    # Networking, BitPack, GGPO Rollback & AOI Replicator (`labs/lab_net.mojo`)
    list.append(PerfMetric("udp_dual_stack_32_mmsg_p99_ms", "lab_net", "udp_dual_stack_32_mmsg_p99_ms=", 0.30, 50.0))
    list.append(PerfMetric("bitpack_10k_quat32_fuzz_p99_ms", "lab_net", "bitpack_10k_quat32_fuzz_p99_ms=", 6.0, 50.0))
    list.append(PerfMetric("channel_reliable_8kib_loss12_p99_ms", "lab_net", "channel_reliable_8kib_loss12_p99_ms=", 0.50, 50.0))
    list.append(PerfMetric("rollback_8tick_resim_latency_p99_ms", "lab_net", "rollback_8tick_resim_latency_p99_ms=", 0.02, 1.2))
    list.append(PerfMetric("rollback_4peer_10000_ticks_p99_ms", "lab_net", "rollback_4peer_10000_ticks_p99_ms=", 25.0, 1500.0))
    list.append(PerfMetric("physics_world_rollback_spoof_auth_p99_ms", "lab_net", "physics_world_rollback_spoof_auth_p99_ms=", 1.0, 1000.0))
    list.append(PerfMetric("replicator_16c_256e_tick_p50_ms", "lab_net", "replicator_16c_256e_tick_p50_ms=", 0.05, 3.0))
    list.append(PerfMetric("replicator_16c_256e_tick_p99_ms", "lab_net", "replicator_16c_256e_tick_p99_ms=", 0.05, 5.0))
    list.append(PerfMetric("replicator_bandwidth_gate_24kbs_p99_ms", "lab_net", "replicator_bandwidth_gate_24kbs_p99_ms=", 1.5, 100.0))
    list.append(PerfMetric("lab_harness_flip_and_csv_p99_ms", "lab_net", "lab_harness_flip_and_csv_p99_ms=", 0.10, 50.0))
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
    """The plain rule, which a column nobody accepted still gets: how many
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
    """A column that was over its limit, measured again: one noisy run
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
    """The row's `build_s` cell: how long the gate's build step
    took, whole seconds, or empty when it is not a whole number. Written,
    never judged: no PerfMetric is named build_s."""
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
# Drift
# ---------------------------------------------------------------------------

def package_phase(package: String) -> String:
    """The phase of a plan package: "5" for "v5.0" or "5.0". Empty for anything else."""
    var start = 0
    var bytes = package.as_bytes()
    if len(bytes) > 0 and (bytes[0] == 118 or bytes[0] == 86):  # 'v' or 'V'
        start = 1
    var at = package.find(".", start)
    if at <= start:
        return String()
    for index in range(start, at):
        if bytes[index] < 48 or bytes[index] > 57:
            return String()
    return String(package[byte=start:at])


def signed_percent(fraction: Float64) -> String:
    """`+12.3%` or `-4.0%`, to a tenth of a percent."""
    var tenths = Int(abs(fraction) * 1000.0 + 0.5)
    var sign = String("-") if fraction < 0 and tenths > 0 else String("+")
    return sign + String(tenths // 10) + "." + String(tenths % 10) + "%"


struct Drift(Copyable, Movable):
    """How far one column of a new row has moved since the first passing row of
    the package's phase and since the baseline row, and which packages
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
        """`<column> +N% since phase <P> (accepted in ...); +M% since v5.0 (...)`."""
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
    `before` rows. The phase is the package's; its start is the first passing
    row of that phase with the column, and the long drift is from the last
    passing baseline row with it, or from the column's first passing row if it
    is younger than the baseline. Nothing counts from before the baseline row."""
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


# ---------------------------------------------------------------------------
# Lab CSV merging and row-level regression gating
# ---------------------------------------------------------------------------


def merge_lab_csvs(
    csv_texts: List[String],
    package_name: String = "v5-gate",
    date_str: String = "2026-11-20",
    commit_str: String = "",
) -> PerfTable:
    """Merges the latest row from each per-lab CSV into a single unified PerfTable row."""
    var out = PerfTable()
    var row = List[String](length=len(out.header), fill=String())
    var all_pass = True
    var resolved_date = date_str
    var resolved_commit = commit_str

    for i in range(len(csv_texts)):
        var raw = csv_texts[i]
        if raw.byte_length() == 0:
            all_pass = False
            continue
        var t = parse_table(raw)
        if len(t.rows) == 0:
            all_pass = False
            continue
        var r_idx = len(t.rows) - 1
        var c_res = t.column("result")
        if c_res >= 0 and t.cell(r_idx, c_res) != "pass":
            all_pass = False
        var c_date = t.column("date")
        if resolved_date.byte_length() == 0 and c_date >= 0:
            resolved_date = t.cell(r_idx, c_date)
        var c_commit = t.column("commit")
        if resolved_commit.byte_length() == 0 and c_commit >= 0:
            resolved_commit = t.cell(r_idx, c_commit)

        for col_idx in range(len(t.header)):
            var col_name = t.header[col_idx]
            if (
                col_name == "date"
                or col_name == "commit"
                or col_name == "package"
                or col_name == "result"
            ):
                continue
            var dst_col = out.ensure_column(col_name)
            while len(row) < len(out.header):
                row.append(String())
            row[dst_col] = t.cell(r_idx, col_idx)

    var cd = out.column("date")
    var cc = out.column("commit")
    var cp = out.column("package")
    var cr = out.column("result")
    if cd >= 0:
        row[cd] = resolved_date
    if cc >= 0:
        row[cc] = resolved_commit
    if cp >= 0:
        row[cp] = package_name
    if cr >= 0:
        row[cr] = "pass" if all_pass else "fail"
    out.rows.append(row^)
    return out^


def append_candidate_row(mut history: PerfTable, candidate: PerfTable, candidate_row: Int = 0):
    """Appends `candidate.rows[candidate_row]` into `history`, aligning and adding columns."""
    if candidate_row < 0 or candidate_row >= len(candidate.rows):
        return
    for col_idx in range(len(candidate.header)):
        _ = history.ensure_column(candidate.header[col_idx])
    var row = List[String](length=len(history.header), fill=String())
    for col_idx in range(len(candidate.header)):
        var dst = history.column(candidate.header[col_idx])
        if dst >= 0:
            row[dst] = candidate.cell(candidate_row, col_idx)
    history.rows.append(row^)


def judge_lab_row(
    metrics: List[PerfMetric],
    candidate: PerfTable,
    candidate_row: Int,
    history: PerfTable,
    accepts: List[AcceptLine],
) -> List[Finding]:
    """Judges all matching metrics in `candidate.rows[candidate_row]` against `history`."""
    var findings = List[Finding]()
    if candidate_row < 0 or candidate_row >= len(candidate.rows):
        return findings^
    var pkg_col = candidate.column("package")
    var pkg = candidate.cell(candidate_row, pkg_col) if pkg_col >= 0 else String()
    for i in range(len(metrics)):
        ref m = metrics[i]
        var col = candidate.column(m.name)
        if col < 0:
            continue
        var parsed = value_at(candidate, candidate_row, col)
        if not parsed[0]:
            continue
        var finding = judge_from(m, parsed[1], history, accepts, len(history.rows))
        for a in range(len(accepts)):
            if accepts[a].package == pkg and (accepts[a].column == m.name or accepts[a].column == "all"):
                finding.accepted = True
                break
        findings.append(finding^)
    return findings^

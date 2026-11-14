"""Flaky-test history for the regression gate (package 0.12).

The gate runs every test binary on its own (tools/regression_gate.sh). A test
that fails is run once more: failing twice makes it FAILED, passing the
second time makes it FLAKY. Both fail the gate (a flaky test is a race or a
timing assumption somewhere, as 0.10's crash-probe test was), but the gate
says which, and every gate run appends one row to docs/TEST_HISTORY.csv:

    date,commit,package,tests,failed,flaky,slowest

`failed` and `flaky` are space-separated test names (`-` for none), `slowest`
is `name:seconds`. From the history, `flaky_count` says how often a test was
flaky in the last runs, so the summary can say "flaky 3 times in the last 10
gate runs" and the developer knows a test, not the package, needs fixing.
SD_GATE_FLAKY_OK=1 (the developer's choice, never the runner's) lets a test
that is flaky for the first time in the window pass the gate.
"""

comptime TEST_HISTORY_HEADER = "date,commit,package,tests,failed,flaky,slowest"
comptime FLAKY_WINDOW: Int = 10

comptime OUTCOME_PASS: Int = 0
comptime OUTCOME_FLAKY: Int = 1
comptime OUTCOME_FAIL: Int = 2


struct TestResult(Copyable, Movable):
    var kind: String     # cpu or gpu
    var name: String
    var outcome: Int
    var seconds: Int

    def __init__(out self, kind: String, name: String, outcome: Int, seconds: Int):
        self.kind = kind
        self.name = name
        self.outcome = outcome
        self.seconds = seconds


def outcome_from_text(text: String) -> Int:
    if text == "pass":
        return OUTCOME_PASS
    if text == "flaky":
        return OUTCOME_FLAKY
    return OUTCOME_FAIL


def parse_results(text: String) raises -> List[TestResult]:
    """Lines of `kind name outcome seconds`, as the gate writes them."""
    var results = List[TestResult]()
    for raw in text.split("\n"):
        var line = String(String(raw).strip())
        if line.byte_length() == 0:
            continue
        var parts = line.split(" ")
        if len(parts) < 4:
            raise Error("bad test result line: " + line)
        results.append(TestResult(String(parts[0]), String(parts[1]), outcome_from_text(String(parts[2])), Int(String(parts[3]))))
    return results^


def names_with(results: List[TestResult], outcome: Int) -> String:
    var out = String()
    for result in results:
        if result.outcome == outcome:
            out += (" " if out.byte_length() > 0 else "") + result.name
    return out if out.byte_length() > 0 else String("-")


def history_row(date: String, commit: String, package: String, results: List[TestResult]) -> String:
    var slowest = String("-")
    var longest = -1
    for result in results:
        if result.seconds > longest:
            longest = result.seconds
            slowest = result.name + ":" + String(result.seconds)
    return (date + "," + commit + "," + package + "," + String(len(results)) + "," + names_with(results, OUTCOME_FAIL)
            + "," + names_with(results, OUTCOME_FLAKY) + "," + slowest)


def flaky_column(row: String) -> List[String]:
    var out = List[String]()
    var fields = row.split(",")
    if len(fields) < 7:
        return out^
    for name in String(fields[5]).split(" "):
        var clean = String(String(name).strip())
        if clean.byte_length() > 0 and clean != "-":
            out.append(clean)
    return out^


def flaky_count(rows: List[String], name: String, window: Int = FLAKY_WINDOW) -> Int:
    """How many of the last `window` rows list `name` as flaky."""
    var count = 0
    var start = max(0, len(rows) - window)
    for index in range(start, len(rows)):
        for flaky in flaky_column(rows[index]):
            if flaky == name:
                count += 1
    return count


def history_rows(text: String) -> List[String]:
    """The data rows of a TEST_HISTORY.csv, header and blank lines left out."""
    var rows = List[String]()
    for raw in text.split("\n"):
        var line = String(String(raw).strip())
        if line.byte_length() == 0 or line.startswith("date,"):
            continue
        rows.append(line)
    return rows^

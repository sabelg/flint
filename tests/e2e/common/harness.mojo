"""Common E2E assertion harness and test context for Flint Engine."""
from std.collections import List
from std.time import monotonic
from flint.core.math import Vec3
from flint.platform.os import remove_file


struct TestContext:
    var total_tests: Int
    var passed_tests: Int
    var failed_tests: Int
    var current_suite: String
    var failures: List[String]
    var start_time_ns: Int

    def __init__(out self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.current_suite = ""
        self.failures = List[String]()
        self.start_time_ns = monotonic()

    def start_suite(mut self, suite_name: String):
        self.current_suite = suite_name

    def record_pass(mut self, test_name: String):
        self.total_tests += 1
        self.passed_tests += 1

    def record_fail(mut self, test_name: String, err_msg: String):
        self.total_tests += 1
        self.failed_tests += 1
        var f = "[" + self.current_suite + "] " + test_name + " FAILED: " + err_msg
        self.failures.append(f)

    def assert_true(mut self, cond: Bool, msg: String = "") raises:
        if not cond:
            raise Error(msg if msg.byte_length() > 0 else "Assertion failed: condition was False")

    def assert_false(mut self, cond: Bool, msg: String = "") raises:
        if cond:
            raise Error(msg if msg.byte_length() > 0 else "Assertion failed: condition was True")

    def assert_equal[T: Equatable](mut self, actual: T, expected: T, msg: String = "") raises:
        if actual != expected:
            raise Error(msg if msg.byte_length() > 0 else "Assertion failed: values are not equal")

    def assert_approx(mut self, actual: Float32, expected: Float32, eps: Float32 = 0.001, msg: String = "") raises:
        var diff = actual - expected
        if diff < 0:
            diff = -diff
        if diff > eps:
            raise Error(msg if msg.byte_length() > 0 else "Assertion failed: approx difference exceeded epsilon")

    def assert_vec3_approx(mut self, actual: Vec3, expected: Vec3, eps: Float32 = 0.001, msg: String = "") raises:
        var dx = actual.x - expected.x
        if dx < 0: dx = -dx
        var dy = actual.y - expected.y
        if dy < 0: dy = -dy
        var dz = actual.z - expected.z
        if dz < 0: dz = -dz
        if dx > eps or dy > eps or dz > eps:
            raise Error(msg if msg.byte_length() > 0 else "Assertion failed: Vec3 difference exceeded epsilon")

    def cleanup_artifact(self, path: String):
        _ = remove_file(path)

    def elapsed_ms(self) -> Float64:
        var now_ns = monotonic()
        return Float64(now_ns - self.start_time_ns) / 1000000.0

    def print_summary(self):
        print("------------------------------------------------------------")
        print("E2E Test Results: " + String(self.passed_tests) + " / " + String(self.total_tests) + " passed (" + String(self.failed_tests) + " failed) in " + String(Int(self.elapsed_ms())) + " ms")
        if self.failed_tests > 0:
            print("Failures:")
            for i in range(len(self.failures)):
                print("  - " + self.failures[i])
        print("------------------------------------------------------------")

"""Client-Side Input Prediction, Authoritative Reconciliation & Visual Smoothing (`flint.net.prediction`).

Implements:
- `PredictedState`: position, velocity, orientation, state flags, and deterministic 64-bit hash
- `PredictedInputEntry[InputT]`: circular-ring entry pairing a sequenced `RollbackInput`
  command with its predicted post-step state and relative motion delta
- `trait PredictionStepper`: optional deterministic single-input re-simulation hook
- `ClientPredictionBuffer[InputT]` (aliased as `PredictionBuffer`): bounded input history ring
  supporting monotonic ACK trimming, misprediction detection, rewind + replay, and
  `VisualSmoother` error absorption.
"""
from std.collections import List
from std.math import abs, max, min
from flint.core.detmath import f32_bits
from flint.core.detsort import mix64
from flint.core.math import Quat, Vec3
from flint.net.bitpack import quat_angular_error_deg
from flint.net.rollback import RollbackInput, VisualSmoother


struct PredictedState(Copyable, ImplicitlyCopyable, Movable):
    """Client-predicted entity/player kinematic state."""
    var pos: Vec3
    var vel: Vec3
    var rot: Quat
    var state_flags: UInt32
    var state_hash: UInt64

    def __init__(
        out self,
        pos: Vec3 = Vec3(),
        vel: Vec3 = Vec3(),
        rot: Quat = Quat.identity(),
        state_flags: UInt32 = 0,
        state_hash: UInt64 = 0,
    ):
        self.pos = pos
        self.vel = vel
        self.rot = rot.normalized()
        self.state_flags = state_flags
        self.state_hash = state_hash

    def compute_hash(self) -> UInt64:
        var h = mix64(
            UInt64(f32_bits(self.pos.x)) | (UInt64(f32_bits(self.pos.y)) << 32)
        )
        h = mix64(
            h
            ^ (
                UInt64(f32_bits(self.pos.z))
                | (UInt64(f32_bits(self.vel.x)) << 32)
            )
        )
        h = mix64(
            h
            ^ (
                UInt64(f32_bits(self.vel.y))
                | (UInt64(f32_bits(self.vel.z)) << 32)
            )
        )
        h = mix64(
            h
            ^ (
                UInt64(f32_bits(self.rot.x))
                | (UInt64(f32_bits(self.rot.y)) << 32)
            )
        )
        h = mix64(
            h
            ^ (
                UInt64(f32_bits(self.rot.z))
                | (UInt64(f32_bits(self.rot.w)) << 32)
            )
        )
        return mix64(h ^ UInt64(self.state_flags))


@fieldwise_init
struct PredictedInputEntry[InputT: RollbackInput](
    Copyable, ImplicitlyCopyable, Movable
):
    """Single sequenced input command and resulting predicted state in the client ring."""
    var valid: Bool
    var input_seq: Int
    var tick: Int
    var dt: Float32
    var input: Self.InputT
    var predicted_state: PredictedState
    var state_hash: UInt64
    var delta_pos: Vec3
    var delta_vel: Vec3
    var delta_rot: Quat

    def __init__(out self):
        self.valid = False
        self.input_seq = -1
        self.tick = -1
        self.dt = Float32(0.0)
        self.input = Self.InputT()
        self.predicted_state = PredictedState()
        self.state_hash = 0
        self.delta_pos = Vec3()
        self.delta_vel = Vec3()
        self.delta_rot = Quat.identity()


trait PredictionStepper(Movable):
    """Deterministic single-input simulation callback for client reconciliation replay."""
    def step_prediction(
        mut self,
        mut state: PredictedState,
        input_ptr: Int,
        input_seq: Int,
        tick: Int,
        dt: Float32,
    ): ...


@always_inline
def _pred_is_finite32(v: Float32) -> Bool:
    return (v - v) == Float32(0.0)


@always_inline
def _pred_vec3_finite(v: Vec3) -> Bool:
    return (
        _pred_is_finite32(v.x)
        and _pred_is_finite32(v.y)
        and _pred_is_finite32(v.z)
    )


@always_inline
def _pred_quat_finite(q: Quat) -> Bool:
    return (
        _pred_is_finite32(q.x)
        and _pred_is_finite32(q.y)
        and _pred_is_finite32(q.z)
        and _pred_is_finite32(q.w)
    )


@always_inline
def _pred_state_finite(s: PredictedState) -> Bool:
    return (
        _pred_vec3_finite(s.pos)
        and _pred_vec3_finite(s.vel)
        and _pred_quat_finite(s.rot)
    )


struct ClientPredictionBuffer[InputT: RollbackInput](Movable):
    """Circular client-side input prediction & authoritative reconciliation ring buffer."""
    var capacity: Int
    var entries: List[PredictedInputEntry[Self.InputT]]
    var head: Int
    var count: Int
    var has_acked: Bool
    var last_acked_input_seq: Int
    var latest_state: PredictedState
    var has_latest_state: Bool
    var pos_epsilon: Float32
    var vel_epsilon: Float32
    var rot_epsilon_deg: Float32
    var reconciliations: Int
    var mispredictions: Int
    var replayed_inputs: Int
    var overflow_evictions: Int

    def __init__(
        out self,
        capacity: Int = 64,
        pos_epsilon: Float32 = 0.005,
        vel_epsilon: Float32 = 0.01,
        rot_epsilon_deg: Float32 = 0.15,
    ):
        self.capacity = max(1, capacity)
        self.entries = List[PredictedInputEntry[Self.InputT]](
            capacity=self.capacity
        )
        for _ in range(self.capacity):
            self.entries.append(PredictedInputEntry[Self.InputT]())
        self.head = 0
        self.count = 0
        self.has_acked = False
        self.last_acked_input_seq = 0
        self.latest_state = PredictedState()
        self.has_latest_state = False
        self.pos_epsilon = max(Float32(0.0), pos_epsilon)
        self.vel_epsilon = max(Float32(0.0), vel_epsilon)
        self.rot_epsilon_deg = max(Float32(0.0), rot_epsilon_deg)
        self.reconciliations = 0
        self.mispredictions = 0
        self.replayed_inputs = 0
        self.overflow_evictions = 0

    def _clear_pending(mut self):
        for i in range(self.capacity):
            self.entries[i].valid = False
        self.head = 0
        self.count = 0

    def clear(mut self):
        self._clear_pending()
        self.has_acked = False
        self.last_acked_input_seq = 0
        self.latest_state = PredictedState()
        self.has_latest_state = False
        self.reconciliations = 0
        self.mispredictions = 0
        self.replayed_inputs = 0
        self.overflow_evictions = 0

    @always_inline
    def pending_count(self) -> Int:
        return self.count

    @always_inline
    def latest_predicted_state(self) -> PredictedState:
        return self.latest_state

    def push_predicted(
        mut self,
        input_seq: Int,
        input: Self.InputT,
        predicted_state: PredictedState,
        tick: Int = -1,
        dt: Float32 = Float32(0.0),
    ):
        """Appends a locally predicted `(input_seq, input, predicted_state)` entry."""
        if input_seq < 0:
            return
        if not _pred_state_finite(predicted_state):
            return
        if self.has_acked and input_seq <= self.last_acked_input_seq:
            return

        var prev_state = PredictedState()
        if self.count > 0:
            var tail = (self.head + self.count - 1) % self.capacity
            if input_seq <= self.entries[tail].input_seq:
                return
            prev_state = self.entries[tail].predicted_state
        elif self.has_latest_state:
            prev_state = self.latest_state

        if self.count == self.capacity:
            self.entries[self.head].valid = False
            self.head = (self.head + 1) % self.capacity
            self.count -= 1
            self.overflow_evictions += 1

        var slot = (self.head + self.count) % self.capacity
        var state_copy = predicted_state
        state_copy.rot = state_copy.rot.normalized()
        if state_copy.state_hash == 0:
            state_copy.state_hash = state_copy.compute_hash()

        var entry = PredictedInputEntry[Self.InputT]()
        entry.valid = True
        entry.input_seq = input_seq
        entry.tick = tick
        entry.dt = dt
        entry.input = input
        entry.predicted_state = state_copy
        entry.state_hash = state_copy.state_hash
        var d_pos = state_copy.pos - prev_state.pos
        var d_vel = state_copy.vel - prev_state.vel
        var d_rot = (state_copy.rot * prev_state.rot.conjugate()).normalized()
        entry.delta_pos = d_pos if _pred_vec3_finite(d_pos) else Vec3()
        entry.delta_vel = d_vel if _pred_vec3_finite(d_vel) else Vec3()
        entry.delta_rot = d_rot if _pred_quat_finite(d_rot) else Quat.identity()

        self.entries[slot] = entry
        self.count += 1
        self.latest_state = state_copy
        self.has_latest_state = True

    def _states_match(self, a: PredictedState, b: PredictedState) -> Bool:
        if not _pred_state_finite(a) or not _pred_state_finite(b):
            return False
        var dist_sq = (a.pos - b.pos).length_squared()
        if not (dist_sq <= self.pos_epsilon * self.pos_epsilon):
            return False
        var vel_sq = (a.vel - b.vel).length_squared()
        if not (vel_sq <= self.vel_epsilon * self.vel_epsilon):
            return False
        var ang = quat_angular_error_deg(a.rot, b.rot)
        if not (ang <= self.rot_epsilon_deg):
            return False
        if a.state_flags != b.state_flags:
            return False
        return True

    def _trim_up_to(mut self, seq: Int):
        while self.count > 0 and self.entries[self.head].input_seq <= seq:
            self.entries[self.head].valid = False
            self.head = (self.head + 1) % self.capacity
            self.count -= 1

    def reconcile(
        mut self, acked_seq: Int, auth_state: PredictedState
    ) -> PredictedState:
        """Reconciles against authoritative `auth_state` at `acked_seq`, replaying unacked deltas on mismatch."""
        if acked_seq < 0 or (
            self.has_acked and acked_seq <= self.last_acked_input_seq
        ):
            if self.has_latest_state:
                return self.latest_state
            return auth_state if _pred_state_finite(auth_state) else PredictedState()

        if not _pred_state_finite(auth_state):
            self.reconciliations += 1
            self.mispredictions += 1
            return self.latest_state if self.has_latest_state else PredictedState()

        var auth = auth_state
        auth.rot = auth.rot.normalized()
        if auth.state_hash == 0:
            auth.state_hash = auth.compute_hash()

        var found_offset = -1
        for k in range(self.count):
            var slot = (self.head + k) % self.capacity
            if self.entries[slot].input_seq == acked_seq:
                found_offset = k
                break

        self.has_acked = True
        self.last_acked_input_seq = acked_seq
        self.reconciliations += 1

        if found_offset < 0:
            if (
                self.count > 0
                and acked_seq < self.entries[self.head].input_seq
            ):
                return self.latest_state if self.has_latest_state else auth
            if self.count > 0:
                var newest_seq = self.entries[
                    (self.head + self.count - 1) % self.capacity
                ].input_seq
                if acked_seq < newest_seq:
                    var prev_tail = self.latest_state
                    self._trim_up_to(acked_seq)
                    var cur = auth
                    for i in range(self.count):
                        var idx = (self.head + i) % self.capacity
                        var next_cur = cur
                        next_cur.pos = cur.pos + self.entries[idx].delta_pos
                        next_cur.vel = cur.vel + self.entries[idx].delta_vel
                        next_cur.rot = (
                            self.entries[idx].delta_rot * cur.rot
                        ).normalized()
                        next_cur.state_flags = (
                            self.entries[idx].predicted_state.state_flags
                        )
                        if not _pred_state_finite(next_cur):
                            if _pred_state_finite(self.entries[idx].predicted_state):
                                next_cur = self.entries[idx].predicted_state
                            else:
                                next_cur = cur
                        next_cur.state_hash = next_cur.compute_hash()
                        cur = next_cur
                        self.entries[idx].predicted_state = cur
                        self.entries[idx].state_hash = cur.state_hash
                        self.replayed_inputs += 1
                    if self.has_latest_state and not self._states_match(
                        prev_tail, cur
                    ):
                        self.mispredictions += 1
                    self.latest_state = cur
                    self.has_latest_state = True
                    return cur
            self._clear_pending()
            self.latest_state = auth
            self.has_latest_state = True
            return auth

        var matched_slot = (self.head + found_offset) % self.capacity
        var matched = self._states_match(
            self.entries[matched_slot].predicted_state, auth
        )
        self._trim_up_to(acked_seq)

        if matched:
            if self.count == 0:
                self.latest_state = auth
                self.has_latest_state = True
            return self.latest_state

        self.mispredictions += 1
        var cur = auth
        for i in range(self.count):
            var idx = (self.head + i) % self.capacity
            var next_cur = cur
            next_cur.pos = cur.pos + self.entries[idx].delta_pos
            next_cur.vel = cur.vel + self.entries[idx].delta_vel
            next_cur.rot = (self.entries[idx].delta_rot * cur.rot).normalized()
            next_cur.state_flags = self.entries[idx].predicted_state.state_flags
            if not _pred_state_finite(next_cur):
                if _pred_state_finite(self.entries[idx].predicted_state):
                    next_cur = self.entries[idx].predicted_state
                else:
                    next_cur = cur
            next_cur.state_hash = next_cur.compute_hash()
            cur = next_cur
            self.entries[idx].predicted_state = cur
            self.entries[idx].state_hash = cur.state_hash
            self.replayed_inputs += 1

        self.latest_state = cur
        self.has_latest_state = True
        return cur

    def reconcile_with[
        StepperT: PredictionStepper
    ](
        mut self,
        acked_seq: Int,
        auth_state: PredictedState,
        mut stepper: StepperT,
        default_dt: Float32 = Float32(1.0 / 60.0),
    ) -> PredictedState:
        """Reconciles against `auth_state` and replays unacked inputs via deterministic `stepper`."""
        if acked_seq < 0 or (
            self.has_acked and acked_seq <= self.last_acked_input_seq
        ):
            if self.has_latest_state:
                return self.latest_state
            return auth_state if _pred_state_finite(auth_state) else PredictedState()

        if not _pred_state_finite(auth_state):
            self.reconciliations += 1
            self.mispredictions += 1
            return self.latest_state if self.has_latest_state else PredictedState()

        var auth = auth_state
        auth.rot = auth.rot.normalized()
        if auth.state_hash == 0:
            auth.state_hash = auth.compute_hash()

        var found_offset = -1
        for k in range(self.count):
            var slot = (self.head + k) % self.capacity
            if self.entries[slot].input_seq == acked_seq:
                found_offset = k
                break

        self.has_acked = True
        self.last_acked_input_seq = acked_seq
        self.reconciliations += 1

        if found_offset < 0:
            if (
                self.count > 0
                and acked_seq < self.entries[self.head].input_seq
            ):
                return self.latest_state if self.has_latest_state else auth
            if self.count > 0:
                var newest_seq = self.entries[
                    (self.head + self.count - 1) % self.capacity
                ].input_seq
                if acked_seq < newest_seq:
                    var prev_tail = self.latest_state
                    self._trim_up_to(acked_seq)
                    var cur = auth
                    var stepper_failed = False
                    for i in range(self.count):
                        var idx = (self.head + i) % self.capacity
                        var prev_cur = cur
                        var inp = self.entries[idx].input
                        var dt = (
                            self.entries[idx].dt
                            if self.entries[idx].dt > Float32(0.0)
                            else default_dt
                        )
                        stepper.step_prediction(
                            cur,
                            Int(Pointer(to=inp)),
                            self.entries[idx].input_seq,
                            self.entries[idx].tick,
                            dt,
                        )
                        _ = inp.same_input(inp)
                        if not _pred_state_finite(cur):
                            cur = prev_cur
                            stepper_failed = True
                            break
                        cur.rot = cur.rot.normalized()
                        if not _pred_state_finite(cur):
                            cur = prev_cur
                            stepper_failed = True
                            break
                        cur.state_hash = cur.compute_hash()
                        self.entries[idx].predicted_state = cur
                        self.entries[idx].state_hash = cur.state_hash
                        var d_pos = cur.pos - prev_cur.pos
                        var d_vel = cur.vel - prev_cur.vel
                        var d_rot = (cur.rot * prev_cur.rot.conjugate()).normalized()
                        self.entries[idx].delta_pos = (
                            d_pos if _pred_vec3_finite(d_pos) else Vec3()
                        )
                        self.entries[idx].delta_vel = (
                            d_vel if _pred_vec3_finite(d_vel) else Vec3()
                        )
                        self.entries[idx].delta_rot = (
                            d_rot if _pred_quat_finite(d_rot) else Quat.identity()
                        )
                        self.replayed_inputs += 1
                    if stepper_failed or (
                        self.has_latest_state and not self._states_match(prev_tail, cur)
                    ):
                        self.mispredictions += 1
                    self.latest_state = cur
                    self.has_latest_state = True
                    return cur
            self._clear_pending()
            self.latest_state = auth
            self.has_latest_state = True
            return auth

        var matched_slot = (self.head + found_offset) % self.capacity
        var matched = self._states_match(
            self.entries[matched_slot].predicted_state, auth
        )
        self._trim_up_to(acked_seq)

        if matched:
            if self.count == 0:
                self.latest_state = auth
                self.has_latest_state = True
            return self.latest_state

        self.mispredictions += 1
        var cur = auth
        for i in range(self.count):
            var idx = (self.head + i) % self.capacity
            var prev_cur = cur
            var inp = self.entries[idx].input
            var dt = (
                self.entries[idx].dt
                if self.entries[idx].dt > Float32(0.0)
                else default_dt
            )
            stepper.step_prediction(
                cur,
                Int(Pointer(to=inp)),
                self.entries[idx].input_seq,
                self.entries[idx].tick,
                dt,
            )
            _ = inp.same_input(inp)
            if not _pred_state_finite(cur):
                cur = prev_cur
                break
            cur.rot = cur.rot.normalized()
            if not _pred_state_finite(cur):
                cur = prev_cur
                break
            cur.state_hash = cur.compute_hash()
            self.entries[idx].predicted_state = cur
            self.entries[idx].state_hash = cur.state_hash
            var d_pos = cur.pos - prev_cur.pos
            var d_vel = cur.vel - prev_cur.vel
            var d_rot = (cur.rot * prev_cur.rot.conjugate()).normalized()
            self.entries[idx].delta_pos = (
                d_pos if _pred_vec3_finite(d_pos) else Vec3()
            )
            self.entries[idx].delta_vel = (
                d_vel if _pred_vec3_finite(d_vel) else Vec3()
            )
            self.entries[idx].delta_rot = (
                d_rot if _pred_quat_finite(d_rot) else Quat.identity()
            )
            self.replayed_inputs += 1

        self.latest_state = cur
        self.has_latest_state = True
        return cur

    def reconcile_smoothed(
        mut self,
        acked_seq: Int,
        auth_state: PredictedState,
        mut smoother: VisualSmoother,
        snap_threshold: Float32 = Float32(-1.0),
    ) -> PredictedState:
        """Reconciles state and feeds any correction delta into `VisualSmoother`."""
        if not _pred_state_finite(auth_state):
            smoother.pos_offset = Vec3(0.0, 0.0, 0.0)
            smoother.rot_offset = Quat.identity()
            return self.reconcile(acked_seq, auth_state)
        var had_prev = self.has_latest_state
        var prev_state = self.latest_state
        var prev_vis_pos = (
            smoother.smooth_pos(prev_state.pos)
            if had_prev
            else auth_state.pos
        )
        var prev_vis_rot = (
            smoother.smooth_rot(prev_state.rot)
            if had_prev
            else auth_state.rot
        )
        var prev_mis = self.mispredictions
        var new_state = self.reconcile(acked_seq, auth_state)
        if not _pred_state_finite(new_state):
            smoother.pos_offset = Vec3(0.0, 0.0, 0.0)
            smoother.rot_offset = Quat.identity()
            return new_state
        if had_prev and (
            self.mispredictions > prev_mis
            or not self._states_match(prev_state, new_state)
        ):
            smoother.record_correction(
                prev_vis_pos,
                new_state.pos,
                prev_vis_rot,
                new_state.rot,
                snap_threshold=snap_threshold,
            )
        return new_state


comptime PredictionBuffer = ClientPredictionBuffer

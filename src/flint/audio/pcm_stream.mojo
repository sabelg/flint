"""Lock-free single-producer / single-consumer stereo PCM stream (`flint.audio.pcm_stream`).

A `PcmStreamRing` carries interleaved float32 stereo frames from a generator thread (a music
model, a decoder, a network voice stream) to the `AudioServer` mixer thread, which mixes it
into a bus (`AudioServer.add_pcm_stream`). The handle is a plain address and can be copied to
both threads; the producer calls `write`, the mixer `mix_into`. Underruns play silence and are
counted. Call `destroy()` once, after the server has stopped.
"""

from std.atomic import Atomic
from std.ffi import external_call

comptime _PCM_WRITE: Int = 0
comptime _PCM_READ: Int = 64
comptime _PCM_UNDERRUN: Int = 128
comptime _PCM_DATA: Int = 192


struct PcmStreamRing(Copyable, ImplicitlyCopyable, Movable):
    var raw: Int
    var base: Int
    var capacity: Int     # frames (power of two)

    def __init__(out self):
        self.raw = 0
        self.base = 0
        self.capacity = 0

    @staticmethod
    def create(min_frames: Int) -> PcmStreamRing:
        var r = PcmStreamRing()
        var cap = 256
        while cap < min_frames:
            cap *= 2
        var blk = external_call["calloc", Int](Int(1), _PCM_DATA + cap * 8 + 64)
        if blk == 0:
            return r^
        r.raw = blk
        r.base = (blk + 63) & ~63
        r.capacity = cap
        return r^

    def destroy(mut self):
        if self.raw != 0:
            external_call["free", NoneType](self.raw)
            self.raw = 0
            self.base = 0
            self.capacity = 0

    @always_inline
    def _ctr(self, off: Int) -> Pointer[Atomic[Int64], origin=MutAnyOrigin]:
        return Pointer[Atomic[Int64], origin=MutAnyOrigin](unsafe_from_address=self.base + off)

    @always_inline
    def _data(self) -> Pointer[Float32, MutAnyOrigin]:
        return Pointer[Float32, MutAnyOrigin](unsafe_from_address=self.base + _PCM_DATA)

    def allocated(self) -> Bool:
        return self.base != 0

    def buffered_frames(self) -> Int:
        if self.base == 0:
            return 0
        return Int(self._ctr(_PCM_WRITE)[unsafe_offset=0].load() - self._ctr(_PCM_READ)[unsafe_offset=0].load())

    def free_frames(self) -> Int:
        return self.capacity - self.buffered_frames()

    def underrun_frames(self) -> Int:
        if self.base == 0:
            return 0
        return Int(self._ctr(_PCM_UNDERRUN)[unsafe_offset=0].load())

    def write(self, src: Int, frames: Int) -> Int:
        """Producer: copies up to `frames` interleaved stereo frames from `src`; returns how
        many fit."""
        if self.base == 0:
            return 0
        var w = Int(self._ctr(_PCM_WRITE)[unsafe_offset=0].load())
        var r = Int(self._ctr(_PCM_READ)[unsafe_offset=0].load())
        var n = min(frames, self.capacity - (w - r))
        var mask = self.capacity - 1
        var d = self._data()
        var s = Pointer[Float32, MutAnyOrigin](unsafe_from_address=src)
        for i in range(n):
            var slot = ((w + i) & mask) * 2
            d.unsafe_offset(slot).unsafe_store(s.unsafe_offset(2 * i).unsafe_load())
            d.unsafe_offset(slot + 1).unsafe_store(s.unsafe_offset(2 * i + 1).unsafe_load())
        self._ctr(_PCM_WRITE)[unsafe_offset=0].store(Int64(w + n))
        return n

    def mix_into(self, mut bus_lr: List[Float32], frames: Int, gain: Float32):
        """Consumer (mixer thread): adds up to `frames` frames times `gain` to `bus_lr`."""
        if self.base == 0:
            return
        var r = Int(self._ctr(_PCM_READ)[unsafe_offset=0].load())
        var w = Int(self._ctr(_PCM_WRITE)[unsafe_offset=0].load())
        var n = min(frames, w - r)
        var mask = self.capacity - 1
        var d = self._data()
        for i in range(n):
            var slot = ((r + i) & mask) * 2
            bus_lr[2 * i] += d.unsafe_offset(slot).unsafe_load() * gain
            bus_lr[2 * i + 1] += d.unsafe_offset(slot + 1).unsafe_load() * gain
        self._ctr(_PCM_READ)[unsafe_offset=0].store(Int64(r + n))
        if n < frames:
            _ = self._ctr(_PCM_UNDERRUN)[unsafe_offset=0].fetch_add(Int64(frames - n))

    def read(self, dst: Int, frames: Int) -> Int:
        """Consumer without a mixer (offline rendering, tests): copies up to `frames` frames."""
        if self.base == 0:
            return 0
        var r = Int(self._ctr(_PCM_READ)[unsafe_offset=0].load())
        var w = Int(self._ctr(_PCM_WRITE)[unsafe_offset=0].load())
        var n = min(frames, w - r)
        var mask = self.capacity - 1
        var d = self._data()
        var o = Pointer[Float32, MutAnyOrigin](unsafe_from_address=dst)
        for i in range(n):
            var slot = ((r + i) & mask) * 2
            o.unsafe_offset(2 * i).unsafe_store(d.unsafe_offset(slot).unsafe_load())
            o.unsafe_offset(2 * i + 1).unsafe_store(d.unsafe_offset(slot + 1).unsafe_load())
        self._ctr(_PCM_READ)[unsafe_offset=0].store(Int64(r + n))
        return n

"""Skinning as a GPU compute pass (package 3.4), on amdgpu and on Vulkan.

One kernel (render/kernel.mojo, so one description for both backends) skins a
mesh's vertices: morph targets, then linear blend, dual quaternions or
centers of rotation, exactly the steps of `skin_vertex` in anim/skinning.mojo.
Linear blend gives the CPU's bits; the quaternion methods differ in the last
places because the kernel's reciprocal square root is the GPU's.

What lives on the GPU (all under the `gpu.meshes` tag):

- The static heap: every skinned mesh once, 16 words a vertex: bind position,
  bind normal, center of rotation, the four joints as bytes, the four weights
  (zero where the CPU would skip the influence) and one over their sum.
- The morph heap: for a mesh with morph targets, eight entries a vertex of
  (target, position delta, normal delta), ascending by target, zero padded.
  It is allocated when the first such mesh is drawn.
- Two palette blocks (one per batch in turn): each instance's joints,
  PALETTE_FLOATS floats a joint, then its morph weights.
- The output heap: per instance two ranges of 8 words a vertex (position,
  normal, in model space). A batch writes the range of its own parity, so the
  other still holds what the batch before wrote: this frame's skinned
  positions and the previous frame's, for velocity (4.1, 4.5, 4.10).

Who reads the output today is the CPU: the frame is still built from triangle
records the CPU writes (render/rasterizer.mojo), so the runtime takes a
finished batch back and lights, projects and bins its vertices. Waiting for a
batch inside the frame that queued it would wait behind the raster waves of
the frames in flight (4 to 7 ms, measured for GPU meshing in 1.3), so, as
there, a frame queues its poses, the batch runs beside the frames on the side
queue, and the next frame adopts it: the pose in model space is one frame
old, the placement in the world is this frame's. `adopt` refuses a result
whose pose is far from the one asked for (a teleport, a snap, another
character's turn in the draw order): that draw is skinned on the CPU, as is
any draw with no batch behind it (the first frame, a new level of detail),
and every draw while frames are more than 25 ms apart (the runtime's
SKIN_ADOPT_SECONDS), so a pose on screen is never older than that.
Once the raster reads vertices on the GPU (4.1) the same pass runs in the
frame's own stream and nothing comes back.

Presentation only.
"""
from std.ffi import external_call
from flint.platform.amdgpu import GpuBuffer, DOMAIN_GTT, PAGE_EXECUTE, PAGE_READ, PAGE_WRITE
from flint.platform.amd_isa import ISA_GFX12, CommandBuilder
from flint.platform.vulkan import ACCESS_HOST_READ, ACCESS_SHADER_WRITE, PIPELINE_COMPUTE, PIPELINE_HOST
from flint.platform.os import get_time
from flint.core.memory import MEM_GPU_MESHES, MEM_GPU_SHADERS, MEM_GPU_FRAMES
from flint.core.offset_allocator import OffsetAllocator, OffsetAllocation
from flint.render.kernel import Kernel, KVal, KVar, TARGET_ISA, TARGET_SPIRV, kernel_compute_state, KERNEL_GROUP
from flint.render.gpu_raster import GpuRaster, BACKEND_AMDGPU, BACKEND_VULKAN, SIDE_RING
from flint.assets.model import NativeMesh
from flint.anim.skinning import (
    PALETTE_FLOATS, PALETTE_QUAT, SKIN_LBS, SKIN_DQS, SKIN_COR, SKIN_MAX_MORPHS,
    SKIN_AVERAGE_STEPS, SKIN_SIDE_SHARPNESS, SKIN_AGREE_FULL, SKIN_AGREE_NONE,
)

comptime SKIN_BIND_STATIC: Int = 0
comptime SKIN_BIND_MORPH: Int = 1
comptime SKIN_BIND_PALETTE: Int = 2
comptime SKIN_BIND_OUT: Int = 3
comptime SKIN_BUFFERS: Int = 4

comptime SKIN_PUSH_STATIC: Int = 0         # the mesh's first word in the static heap
comptime SKIN_PUSH_OUT: Int = 1            # the instance's first output word
comptime SKIN_PUSH_PALETTE: Int = 2        # its first palette word
comptime SKIN_PUSH_COUNT: Int = 3          # vertices
comptime SKIN_PUSH_METHOD: Int = 4         # SKIN_LBS, SKIN_DQS or SKIN_COR
comptime SKIN_PUSH_JOINT_MAX: Int = 5      # joints - 1
comptime SKIN_PUSH_MORPH: Int = 6          # the mesh's first word in the morph heap
comptime SKIN_PUSH_MORPH_WEIGHTS: Int = 7  # the instance's first morph weight word (in the palette block)
comptime SKIN_PUSH_MORPH_ON: Int = 8       # 1 when any morph weight is not zero
comptime SKIN_PUSH_MORPH_MAX: Int = 9      # morph targets - 1
comptime SKIN_PUSH_WORDS: Int = 10

comptime SKIN_STATIC_WORDS: Int = 16       # per vertex
comptime SKIN_STATIC_CENTER: Int = 6
comptime SKIN_STATIC_JOINTS: Int = 9
comptime SKIN_STATIC_WEIGHTS: Int = 10
comptime SKIN_STATIC_INVERSE: Int = 14
comptime SKIN_OUT_WORDS: Int = 8           # per vertex: position, pad, normal, pad
comptime SKIN_MORPH_ENTRY_WORDS: Int = 8   # target, three position deltas, three normal deltas, pad
comptime SKIN_MORPH_WORDS: Int = SKIN_MAX_MORPHS * SKIN_MORPH_ENTRY_WORDS  # per vertex

comptime SKIN_STATIC_BYTES: Int = 8 * 1024 * 1024   # 131,072 vertices: Maroon's LOD0 six times over
comptime SKIN_MORPH_BYTES: Int = 8 * 1024 * 1024    # allocated when the first mesh with morph targets is drawn
comptime SKIN_OUT_BYTES: Int = 8 * 1024 * 1024      # 131,072 vertices, twice each
comptime SKIN_PALETTE_BYTES: Int = 64 * 1024        # per batch parity: 819 joints
comptime SKIN_UNIT: Int = 64                   # bytes per allocator unit
comptime SKIN_NODES: Int = 4096
comptime SKIN_CODE_DWORDS: Int = 16384
comptime SKIN_MAX_JOINTS: Int = 256            # joints ride as bytes
comptime SKIN_MAX_VERTICES: Int = 200000
comptime SKIN_IDLE_FRAMES: Int = 600           # an instance or mesh unused this long gives its ranges back
comptime SKIN_STAMP_BYTE: Int = 256            # in the address table buffer, past the 8 addresses
comptime SKIN_QUAT_AGREE: Float32 = 0.9        # |dot| of a joint's rotation now and when the batch was queued


def add_scaled(mut k: Kernel, sum: KVar, a: KVal, b: KVal):
    """sum = sum + a * b."""
    var mark = k.mark()
    var product = k.fmul(a, b)
    var before = k.get(sum)
    var after = k.fadd(before, product)
    k.set(sum, after)
    k.release(mark)


def row3(mut k: Kernel, a: KVal, b: KVal, c: KVal, v: List[KVal]) -> KVal:
    """(a x + b y) + c z, in the CPU's order."""
    var mark = k.mark()
    var first = k.fmul(a, v[0])
    var second = k.fmul(b, v[1])
    var sum = k.fadd(first, second)
    var third = k.fmul(c, v[2])
    sum = k.fadd(sum, third)
    return k.keep(mark, sum)


def row4(mut k: Kernel, a: KVal, b: KVal, c: KVal, d: KVal, v: List[KVal]) -> KVal:
    var mark = k.mark()
    var sum = row3(k, a, b, c, v)
    sum = k.fadd(sum, d)
    return k.keep(mark, sum)


def skin_kernel(target: Int, isa: Int = ISA_GFX12) raises -> Kernel:
    """One vertex per invocation: `skin_vertex` and `morphed` of anim/skinning.mojo."""
    var k = Kernel(target, SKIN_BUFFERS, SKIN_PUSH_WORDS, isa)
    k.exact_floats = True  # linear blend must round as the CPU's does
    var v = k.gid_x()
    k.begin_if(k.ult(v, k.push(SKIN_PUSH_COUNT)))
    var at = k.iadd(k.push(SKIN_PUSH_STATIC), k.shl(v, 4))
    var load_mark = k.mark()
    k.defer_waits()
    var loaded = List[KVal]()
    for n in range(15):
        loaded.append(k.load(SKIN_BIND_STATIC, k.iadd_c(at, UInt32(n))))
    k.wait_loads()
    var s = k.compact(load_mark, loaded)
    var position = List[KVar]()
    var normal = List[KVar]()
    for axis in range(3):
        position.append(k.var_f(k.as_float(s[axis])))
        normal.append(k.var_f(k.as_float(s[3 + axis])))
    # Morph targets move the bind position and normal first.
    k.begin_if(k.push(SKIN_PUSH_MORPH_ON))
    var morph_at = k.iadd(k.push(SKIN_PUSH_MORPH), k.shl(v, 6))
    for slot in range(SKIN_MAX_MORPHS):
        var mark = k.mark()
        var entry = k.iadd_c(morph_at, UInt32(slot * SKIN_MORPH_ENTRY_WORDS))
        k.defer_waits()
        var words = List[KVal]()
        for n in range(7):
            words.append(k.load(SKIN_BIND_MORPH, k.iadd_c(entry, UInt32(n))))
        k.wait_loads()
        var target_index = k.umin(words[0], k.push(SKIN_PUSH_MORPH_MAX))
        var weight = k.load_float(SKIN_BIND_PALETTE, k.iadd(k.push(SKIN_PUSH_MORPH_WEIGHTS), target_index))
        for axis in range(3):
            add_scaled(k, position[axis], k.as_float(words[1 + axis]), weight)
            add_scaled(k, normal[axis], k.as_float(words[4 + axis]), weight)
        k.release(mark)
    k.end_if()
    var p = List[KVal]()
    var n = List[KVal]()
    var c = List[KVal]()
    for axis in range(3):
        p.append(k.get(position[axis]))
    for axis in range(3):
        n.append(k.get(normal[axis]))
    for axis in range(3):
        c.append(k.as_float(s[SKIN_STATIC_CENTER + axis]))
    var method = k.push(SKIN_PUSH_METHOD)
    var is_lbs = k.ieq_c(method, UInt32(SKIN_LBS))
    var is_cor = k.ieq_c(method, UInt32(SKIN_COR))
    var zero = k.const_f(0.0)
    var one = k.const_f(1.0)
    var joint_max = k.push(SKIN_PUSH_JOINT_MAX)
    var joints = List[KVal]()
    var weights = List[KVal]()
    for i in range(4):
        var mark = k.mark()
        var joint = k.band_c(k.shr(s[SKIN_STATIC_JOINTS], i * 8), 255)
        joint = k.umin(joint, joint_max)
        joints.append(k.keep(mark, joint))
        weights.append(k.as_float(s[SKIN_STATIC_WEIGHTS + i]))
    var inverse = k.as_float(s[SKIN_STATIC_INVERSE])
    # The heaviest influence's quaternion starts the average.
    var best_weight = k.var_f(weights[0])
    var best_slot = k.var_u(k.const_u(0))
    for i in range(1, 4):
        var mark = k.mark()
        var better = k.fgt(weights[i], k.get(best_weight))
        k.set(best_weight, k.select(better, weights[i], k.get(best_weight)))
        k.set(best_slot, k.select(better, k.const_u(UInt32(i)), k.get(best_slot)))
        k.release(mark)
    var palette = k.push(SKIN_PUSH_PALETTE)
    # Every influence: its matrix moves the vertex, the normal and the center
    # (linear blend); its quaternion and dual part are kept for the average.
    var linear = List[KVar]()
    var linear_normal = List[KVar]()
    var moved = List[KVar]()
    for _ in range(3):
        linear.append(k.var_f(zero))
        linear_normal.append(k.var_f(zero))
        moved.append(k.var_f(zero))
    var quats = List[KVar]()   # four influences x (quaternion, dual part)
    for _ in range(32):
        quats.append(k.var_f(zero))
    var average = List[KVar]()
    for _ in range(4):
        average.append(k.var_f(zero))
    for i in range(4):
        var mark = k.mark()
        var base = k.iadd(palette, k.imul_c(joints[i], UInt32(PALETTE_FLOATS)))
        k.defer_waits()
        var words = List[KVal]()
        for index in range(PALETTE_FLOATS):
            words.append(k.load(SKIN_BIND_PALETTE, k.iadd_c(base, UInt32(index))))
        k.wait_loads()
        var m = List[KVal]()
        for index in range(PALETTE_FLOATS):
            m.append(k.as_float(words[index]))
        for axis in range(3):
            var inner = k.mark()
            add_scaled(k, linear[axis], row4(k, m[axis * 4], m[axis * 4 + 1], m[axis * 4 + 2], m[axis * 4 + 3], p), weights[i])
            add_scaled(k, linear_normal[axis], row3(k, m[axis * 4], m[axis * 4 + 1], m[axis * 4 + 2], n), weights[i])
            add_scaled(k, moved[axis], row4(k, m[axis * 4], m[axis * 4 + 1], m[axis * 4 + 2], m[axis * 4 + 3], c), weights[i])
            k.release(inner)
        var is_best = k.ieq_c(k.get(best_slot), UInt32(i))
        for index in range(8):
            k.set(quats[i * 8 + index], m[PALETTE_QUAT + index])
        for index in range(4):
            var inner = k.mark()
            k.set(average[index], k.select(is_best, m[PALETTE_QUAT + index], k.get(average[index])))
            k.release(inner)
        k.release(mark)
    # The average rotation by power iteration: b <- normalize(sum w (q . b) q).
    for _ in range(SKIN_AVERAGE_STEPS):
        var mark = k.mark()
        var b = List[KVal]()
        for index in range(4):
            b.append(k.get(average[index]))
        var next = List[KVar]()
        for _ in range(4):
            next.append(k.var_f(zero))
        for i in range(4):
            var inner = k.mark()
            var side = k.fmul(b[0], k.get(quats[i * 8]))
            for index in range(1, 4):
                side = k.fadd(side, k.fmul(b[index], k.get(quats[i * 8 + index])))
            var scaled = k.fmul(side, weights[i])
            for index in range(4):
                add_scaled(k, next[index], k.get(quats[i * 8 + index]), scaled)
            k.release(inner)
        var size = k.fmul(k.get(next[0]), k.get(next[0]))
        for index in range(1, 4):
            size = k.fadd(size, k.fmul(k.get(next[index]), k.get(next[index])))
        var stuck = k.flt(size, k.const_f(1.0e-20))
        var scale = k.rsq(size)
        for index in range(4):
            k.set(average[index], k.select(stuck, b[index], k.fmul(k.get(next[index]), scale)))
        k.release(mark)
    # How much of the weight agrees with it, and the dual quaternion sum on its side.
    var agreement = k.var_f(zero)
    var real = List[KVar]()
    var dual = List[KVar]()
    for _ in range(4):
        real.append(k.var_f(zero))
        dual.append(k.var_f(zero))
    for i in range(4):
        var mark = k.mark()
        var side = k.fmul(k.get(average[0]), k.get(quats[i * 8]))
        for index in range(1, 4):
            side = k.fadd(side, k.fmul(k.get(average[index]), k.get(quats[i * 8 + index])))
        add_scaled(k, agreement, k.fmul(side, side), weights[i])
        var sharp = k.fmul_c(side, SKIN_SIDE_SHARPNESS)
        var signed_weight = k.fmul(k.fmax(k.const_f(-1.0), k.fmin(one, sharp)), weights[i])
        for index in range(4):
            add_scaled(k, real[index], k.get(quats[i * 8 + index]), signed_weight)
            add_scaled(k, dual[index], k.get(quats[i * 8 + 4 + index]), signed_weight)
        k.release(mark)
    var fade_mark = k.mark()
    var agreed = k.fmul(k.get(agreement), inverse)
    var fade = k.fmul_c(k.fsub(k.const_f(SKIN_AGREE_FULL), agreed), 1.0 / (SKIN_AGREE_FULL - SKIN_AGREE_NONE))
    fade = k.fmax(zero, k.fmin(one, fade))
    fade = k.fmul(k.fmul(fade, fade), k.fsub(k.const_f(3.0), k.fmul_c(fade, 2.0)))
    fade = k.keep(fade_mark, fade)
    # Dual quaternions use their own normalized sum; centers of rotation the average.
    var blend_mark = k.mark()
    var size = k.fmul(k.get(real[0]), k.get(real[0]))
    for index in range(1, 4):
        size = k.fadd(size, k.fmul(k.get(real[index]), k.get(real[index])))
    var no_sum = k.flt(size, k.const_f(0.00000001))
    var scale = k.rsq(size)
    var keep_blend = List[KVal]()
    keep_blend.append(k.band(no_sum, k.ieq_c(method, UInt32(SKIN_DQS))))
    for index in range(4):
        keep_blend.append(k.select(is_cor, k.get(average[index]), k.fmul(k.get(real[index]), scale)))
    for index in range(4):
        keep_blend.append(k.fmul(k.get(dual[index]), scale))
    var blended = k.compact(blend_mark, keep_blend)
    var linear_only = k.bor(is_lbs, blended[0])
    var rx = blended[1]
    var ry = blended[2]
    var rz = blended[3]
    var rw = blended[4]
    var matrix_mark = k.mark()
    var two = k.const_f(2.0)
    var xx = k.fmul(rx, rx)
    var yy = k.fmul(ry, ry)
    var zz = k.fmul(rz, rz)
    var xy = k.fmul(rx, ry)
    var xz = k.fmul(rx, rz)
    var yz = k.fmul(ry, rz)
    var wx = k.fmul(rw, rx)
    var wy = k.fmul(rw, ry)
    var wz = k.fmul(rw, rz)
    var rows = List[KVal]()
    rows.append(k.fsub(one, k.fmul(two, k.fadd(yy, zz))))
    rows.append(k.fmul(two, k.fsub(xy, wz)))
    rows.append(k.fmul(two, k.fadd(xz, wy)))
    rows.append(k.fmul(two, k.fadd(xy, wz)))
    rows.append(k.fsub(one, k.fmul(two, k.fadd(xx, zz))))
    rows.append(k.fmul(two, k.fsub(yz, wx)))
    rows.append(k.fmul(two, k.fsub(xz, wy)))
    rows.append(k.fmul(two, k.fadd(yz, wx)))
    rows.append(k.fsub(one, k.fmul(two, k.fadd(xx, yy))))
    var r = k.compact(matrix_mark, rows)
    var unweighted = k.ieq_c(k.as_uint(inverse), 0)
    var out_at = k.iadd(k.push(SKIN_PUSH_OUT), k.shl(v, 3))
    var two_again = k.const_f(2.0)
    # The dual quaternion's translation: twice dual x conjugate(real).
    var dx = blended[5]
    var dy = blended[6]
    var dz = blended[7]
    var dw = blended[8]
    var t_mark = k.mark()
    var tx = k.fsub(k.fmul(dx, rw), k.fmul(dw, rx))
    tx = k.fsub(tx, k.fmul(dy, rz))
    tx = k.fadd(tx, k.fmul(dz, ry))
    tx = k.fmul(two_again, tx)
    var ty = k.fsub(k.fmul(dy, rw), k.fmul(dw, ry))
    ty = k.fsub(ty, k.fmul(dz, rx))
    ty = k.fadd(ty, k.fmul(dx, rz))
    ty = k.fmul(two_again, ty)
    var tz = k.fsub(k.fmul(dz, rw), k.fmul(dw, rz))
    tz = k.fsub(tz, k.fmul(dx, ry))
    tz = k.fadd(tz, k.fmul(dy, rx))
    tz = k.fmul(two_again, tz)
    var translation = k.compact(t_mark, [tx, ty, tz])
    var offset = List[KVal]()
    for axis in range(3):
        offset.append(k.fsub(p[axis], c[axis]))
    for axis in range(3):
        var mark = k.mark()
        var linear_p = k.fmul(k.get(linear[axis]), inverse)
        var center_moved = k.fmul(k.get(moved[axis]), inverse)
        var about_center = k.fadd(row3(k, r[axis * 3], r[axis * 3 + 1], r[axis * 3 + 2], offset), center_moved)
        var rigid = k.fadd(row3(k, r[axis * 3], r[axis * 3 + 1], r[axis * 3 + 2], p), translation[axis])
        var result = k.select(is_cor, about_center, rigid)
        # Fade to linear blend where the rotations have no sensible average.
        result = k.fadd(result, k.fmul(k.fsub(linear_p, result), fade))
        result = k.select(linear_only, linear_p, result)
        result = k.select(unweighted, p[axis], result)
        k.store(SKIN_BIND_OUT, k.iadd_c(out_at, UInt32(axis)), result)
        var linear_n = k.fmul(k.get(linear_normal[axis]), inverse)
        var result_n = row3(k, r[axis * 3], r[axis * 3 + 1], r[axis * 3 + 2], n)
        result_n = k.fadd(result_n, k.fmul(k.fsub(linear_n, result_n), fade))
        result_n = k.select(linear_only, linear_n, result_n)
        result_n = k.select(unweighted, n[axis], result_n)
        k.store(SKIN_BIND_OUT, k.iadd_c(out_at, UInt32(4 + axis)), result_n)
        k.release(mark)
    k.end_if()
    return k^


struct SkinMesh(Copyable, Movable):
    """A mesh in the static heap."""
    var model: Int
    var mesh: Int
    var data: Int           # address of its positions, to notice a reload
    var vertices: Int
    var joints: Int
    var morphs: Int         # morph targets (0: no morph block)
    var static_range: OffsetAllocation
    var morph_range: OffsetAllocation
    var used_frame: Int
    var draws: Int          # draws of it so far this frame
    var refused: Bool       # it found no room: its draws stay on the CPU

    def __init__(out self):
        self.refused = False
        self.model = -1
        self.mesh = -1
        self.data = 0
        self.vertices = 0
        self.joints = 0
        self.morphs = 0
        self.static_range = OffsetAllocation()
        self.morph_range = OffsetAllocation()
        self.used_frame = 0
        self.draws = 0


struct SkinInstance(Copyable, Movable):
    """One drawn character: a mesh, its place in the draw order, two output ranges."""
    var mesh: Int           # index into `meshes`, -1 when free
    var ordinal: Int
    var ranges: List[OffsetAllocation]  # per batch parity
    var written: List[Int]  # the batch that last wrote each range, -1 never
    var pose: List[Float32]  # joint positions (3) and rotations (4) of the newest queued batch
    var method: Int
    var used_frame: Int
    var queued_frame: Int   # the frame that queued `pose`

    def __init__(out self):
        self.queued_frame = -2
        self.mesh = -1
        self.ordinal = 0
        self.ranges = [OffsetAllocation(), OffsetAllocation()]
        self.written = [-1, -1]
        self.pose = List[Float32]()
        self.method = SKIN_LBS
        self.used_frame = 0


struct SkinStats(Copyable, ImplicitlyCopyable, Movable):
    var batches: Int
    var queued: Int       # instances skinned on the GPU
    var adopted: Int      # draws that took a GPU result
    var refused: Int      # draws whose GPU result was too far from the pose asked for
    var missed: Int       # draws with no finished batch behind them
    var uploads: Int
    var refused_meshes: Int  # meshes that could not go on the GPU
    var wait_seconds: Float64

    def __init__(out self):
        self.batches = 0
        self.queued = 0
        self.adopted = 0
        self.refused = 0
        self.missed = 0
        self.uploads = 0
        self.refused_meshes = 0
        self.wait_seconds = 0


struct GpuSkinner(Movable):
    var ready: Bool
    var backend: Int
    var recoveries: Int  # the raster's at creation: a recovered raster has a new device
    var code: GpuBuffer
    var commands: GpuBuffer
    var addresses: GpuBuffer  # amdgpu: the buffer table, then two GPU clock stamps
    var static_heap: GpuBuffer
    var morph_heap: GpuBuffer
    var palettes: GpuBuffer
    var output: GpuBuffer
    var static_alloc: OffsetAllocator
    var morph_alloc: OffsetAllocator
    var out_alloc: OffsetAllocator
    var vgprs: UInt32
    var pipeline: Int
    var descriptor: Int
    var command_index: Int
    var meshes: List[SkinMesh]
    var instances: List[SkinInstance]
    var frame: Int
    var batch: Int             # the batch being queued
    var finished: Int          # the newest batch known to have finished, -1 none
    var fence: UInt64          # of batch - 1, 0 when it has been waited for (or never ran)
    var fence_batch: Int
    var pushes: List[List[UInt32]]  # this batch's dispatches
    var palette_words: Int     # used in this batch's palette block
    var error: String
    var stats: SkinStats

    def __init__(out self):
        self.ready = False
        self.backend = BACKEND_AMDGPU
        self.recoveries = -1
        self.code = GpuBuffer()
        self.commands = GpuBuffer()
        self.addresses = GpuBuffer()
        self.static_heap = GpuBuffer()
        self.morph_heap = GpuBuffer()
        self.palettes = GpuBuffer()
        self.output = GpuBuffer()
        self.static_alloc = OffsetAllocator(UInt32(SKIN_STATIC_BYTES // SKIN_UNIT), SKIN_NODES)
        self.morph_alloc = OffsetAllocator(UInt32(SKIN_MORPH_BYTES // SKIN_UNIT), SKIN_NODES)
        self.out_alloc = OffsetAllocator(UInt32(SKIN_OUT_BYTES // SKIN_UNIT), SKIN_NODES)
        self.vgprs = 0
        self.pipeline = -1
        self.descriptor = 0
        self.command_index = -1
        self.meshes = List[SkinMesh]()
        self.instances = List[SkinInstance]()
        self.frame = 0
        self.batch = 0
        self.finished = -1
        self.fence = 0
        self.fence_batch = -1
        self.pushes = List[List[UInt32]]()
        self.palette_words = 0
        self.error = String()
        self.stats = SkinStats()

    def usable(self, raster: GpuRaster) -> Bool:
        """Ready on this raster's device: a recovered or moved raster needs `initialize` again."""
        return self.ready and raster.active and raster.recoveries == self.recoveries and raster.backend == self.backend

    def stopped_on(self, raster: GpuRaster) -> Bool:
        """It started on this raster's device and then failed there (a batch that
        did not finish, a submission refused): do not start it there again."""
        return not self.ready and self.static_heap.size > 0 and raster.recoveries == self.recoveries and raster.backend == self.backend

    def fail(mut self, message: String) -> Bool:
        self.error = message
        self.ready = False
        return False

    def initialize(mut self, mut raster: GpuRaster) -> Bool:
        """Buffers and the kernel on the raster's device. False leaves skinning on the CPU."""
        self.ready = False
        self.forget_all()
        if not raster.active or (raster.backend != BACKEND_AMDGPU and raster.backend != BACKEND_VULKAN):
            return self.fail("no GPU backend")
        self.backend = raster.backend
        self.recoveries = raster.recoveries
        self.error = String()
        self.morph_heap = GpuBuffer()
        var local = raster.local_domain
        self.static_heap = raster.allocate(SKIN_STATIC_BYTES, local, PAGE_READ, MEM_GPU_MESHES)
        self.palettes = raster.allocate(SKIN_PALETTE_BYTES * 2, DOMAIN_GTT, PAGE_READ, MEM_GPU_MESHES)
        self.output = raster.allocate(SKIN_OUT_BYTES, DOMAIN_GTT, PAGE_READ | PAGE_WRITE, MEM_GPU_MESHES, False, True)
        if self.static_heap.size == 0 or self.palettes.size == 0 or self.output.size == 0:
            return self.fail("GPU skinning buffers: " + raster.backend_error())
        try:
            if self.backend == BACKEND_AMDGPU:
                self.code = raster.allocate(SKIN_CODE_DWORDS * 4, DOMAIN_GTT, PAGE_READ | PAGE_EXECUTE, MEM_GPU_SHADERS)
                self.commands = raster.allocate(65536, DOMAIN_GTT, PAGE_READ | PAGE_EXECUTE, MEM_GPU_FRAMES)
                self.addresses = raster.allocate(4096, DOMAIN_GTT, PAGE_READ | PAGE_WRITE, MEM_GPU_FRAMES)
                if self.code.size == 0 or self.commands.size == 0 or self.addresses.size == 0:
                    return self.fail("GPU skinning code: " + raster.backend_error())
                var kernel = skin_kernel(TARGET_ISA, raster.gpu.isa())
                self.vgprs = kernel.vgprs()
                var shader = kernel.finish_isa()
                if len(shader.words) > SKIN_CODE_DWORDS:
                    return self.fail("the skin kernel is " + String(len(shader.words)) + " dwords")
                _ = shader.write_to(self.code, 0)
                # No morph heap yet: its slot reads the static heap, and no dispatch asks for morphs.
                var buffers = [self.static_heap.gpu_address, self.static_heap.gpu_address, self.palettes.gpu_address, self.output.gpu_address, UInt64(0), UInt64(0), UInt64(0), UInt64(0)]
                for index in range(8):
                    self.addresses.write32(index * 2, UInt32(buffers[index] & 0xFFFFFFFF))
                    self.addresses.write32(index * 2 + 1, UInt32(buffers[index] >> 32))
            else:
                var kernel = skin_kernel(TARGET_SPIRV)
                var words = kernel.finish_spirv()
                if not raster.vk.create_pipeline(words):
                    return self.fail("the skin kernel: " + raster.vk.error)
                self.pipeline = len(raster.vk.pipelines) - 1
                var sets = raster.vk.create_side_descriptor_sets(1)
                if len(sets) != 1:
                    return self.fail("GPU skinning descriptors: " + raster.vk.error)
                self.descriptor = sets[0]
                raster.vk.write_descriptor(self.descriptor, SKIN_BIND_STATIC, self.static_heap)
                raster.vk.write_descriptor(self.descriptor, SKIN_BIND_MORPH, self.static_heap)
                raster.vk.write_descriptor(self.descriptor, SKIN_BIND_PALETTE, self.palettes)
                raster.vk.write_descriptor(self.descriptor, SKIN_BIND_OUT, self.output)
                self.command_index = raster.vk.allocate_side_command_buffers(1)
                if self.command_index < 0:
                    return self.fail("GPU skinning commands: " + raster.vk.error)
        except e:
            return self.fail("the skin kernel: " + String(e))
        self.ready = True
        return True

    def forget_all(mut self):
        """Empty heaps, no instances, no batch (a new device, or shutdown)."""
        self.static_alloc.reset()
        self.morph_alloc.reset()
        self.out_alloc.reset()
        self.meshes.clear()
        self.instances.clear()
        self.pushes.clear()
        self.palette_words = 0
        self.fence = 0
        self.fence_batch = -1
        self.finished = -1

    def begin_frame(mut self):
        """Once a frame, before any draw: draw counts start over, and what has
        not been drawn for a long while gives its memory back."""
        self.frame += 1
        for index in range(len(self.meshes)):
            self.meshes[index].draws = 0
        if self.frame % 64 != 0:
            return
        for index in range(len(self.instances)):
            if self.instances[index].mesh >= 0 and self.frame - self.instances[index].used_frame > SKIN_IDLE_FRAMES:
                self.free_instance(index)
        for index in range(len(self.meshes)):
            if self.meshes[index].model >= 0 and self.frame - self.meshes[index].used_frame > SKIN_IDLE_FRAMES:
                self.free_mesh(index)

    def free_instance(mut self, index: Int):
        for parity in range(2):
            if self.instances[index].ranges[parity].ok():
                self.out_alloc.free(self.instances[index].ranges[parity])
        self.instances[index] = SkinInstance()

    def free_mesh(mut self, index: Int):
        """The mesh's ranges and every instance of it. Not while a batch that reads
        them is queued: callers run between frames."""
        for instance in range(len(self.instances)):
            if self.instances[instance].mesh == index:
                self.free_instance(instance)
        if self.meshes[index].static_range.ok():
            self.static_alloc.free(self.meshes[index].static_range)
        if self.meshes[index].morph_range.ok():
            self.morph_alloc.free(self.meshes[index].morph_range)
        self.meshes[index] = SkinMesh()

    def forget_model(mut self, mut raster: GpuRaster, model: Int):
        """The model behind this index is going away or being replaced (hot
        reload, unload). The batch in flight may still read its vertices, so it
        is waited for first; nothing queued this frame survives."""
        var found = False
        for index in range(len(self.meshes)):
            if self.meshes[index].model == model:
                found = True
        if not found:
            return
        _ = self.wait(raster)
        self.pushes.clear()
        self.palette_words = 0
        for index in range(len(self.meshes)):
            if self.meshes[index].model == model:
                self.free_mesh(index)

    def static_word(self, mesh: Int) -> Int:
        return Int(self.meshes[mesh].static_range.offset) * SKIN_UNIT // 4

    def ensure_morph_heap(mut self, mut raster: GpuRaster) -> Bool:
        """The morph heap, made when the first mesh with morph targets arrives.
        The batch in flight is waited for: its buffer table is about to change."""
        if self.morph_heap.size > 0:
            return True
        if not self.wait(raster):
            return False
        self.morph_heap = raster.allocate(SKIN_MORPH_BYTES, raster.local_domain, PAGE_READ, MEM_GPU_MESHES)
        if self.morph_heap.size == 0:
            return False
        if self.backend == BACKEND_AMDGPU:
            self.addresses.write32(SKIN_BIND_MORPH * 2, UInt32(self.morph_heap.gpu_address & 0xFFFFFFFF))
            self.addresses.write32(SKIN_BIND_MORPH * 2 + 1, UInt32(self.morph_heap.gpu_address >> 32))
        else:
            raster.vk.write_descriptor(self.descriptor, SKIN_BIND_MORPH, self.morph_heap)
        return True

    def keep_entry(mut self, var entry: SkinMesh) -> Int:
        for index in range(len(self.meshes)):
            if self.meshes[index].model < 0:
                self.meshes[index] = entry^
                return index
        self.meshes.append(entry^)
        return len(self.meshes) - 1

    def upload(mut self, mut raster: GpuRaster, mesh: NativeMesh, model: Int, mesh_index: Int, joints: Int) -> Int:
        """Put a mesh in the static heap (and its morph targets in the morph
        heap). Returns its index in `meshes`; the entry is `refused` when the
        mesh cannot go on the GPU (too big, too many joints, no room)."""
        var count = mesh.vertex_count()
        var entry = SkinMesh()
        entry.model = model
        entry.mesh = mesh_index
        entry.data = Int(mesh.positions.unsafe_ptr())
        entry.vertices = count
        entry.joints = joints
        entry.used_frame = self.frame
        var has_morphs = mesh.morph_count > 0 and len(mesh.morph_positions) == mesh.morph_count * count * 3
        entry.refused = count == 0 or count > SKIN_MAX_VERTICES or joints <= 0 or joints > SKIN_MAX_JOINTS or len(mesh.weights) != count * 4 or len(mesh.joints) != count * 4 or len(mesh.normals) != count * 3
        if not entry.refused and has_morphs:
            entry.refused = not self.ensure_morph_heap(raster)
            if not entry.refused:
                entry.morph_range = self.morph_alloc.allocate(UInt32((count * SKIN_MORPH_WORDS * 4 + SKIN_UNIT - 1) // SKIN_UNIT))
                entry.refused = not entry.morph_range.ok()
        if not entry.refused:
            entry.static_range = self.static_alloc.allocate(UInt32((count * SKIN_STATIC_WORDS * 4 + SKIN_UNIT - 1) // SKIN_UNIT))
            if not entry.static_range.ok():
                entry.refused = True
                if entry.morph_range.ok():
                    self.morph_alloc.free(entry.morph_range)
                    entry.morph_range = OffsetAllocation()
        if entry.refused:
            self.stats.refused_meshes += 1
            return self.keep_entry(entry^)
        var has_centers = len(mesh.centers) == count * 3
        var floats = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=self.static_heap.cpu_address + Int(entry.static_range.offset) * SKIN_UNIT)
        var words = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.static_heap.cpu_address + Int(entry.static_range.offset) * SKIN_UNIT)
        for vertex in range(count):
            var at = vertex * SKIN_STATIC_WORDS
            for axis in range(3):
                floats[unsafe_offset=at + axis] = mesh.positions[vertex * 3 + axis]
                floats[unsafe_offset=at + 3 + axis] = mesh.normals[vertex * 3 + axis]
                floats[unsafe_offset=at + SKIN_STATIC_CENTER + axis] = mesh.centers[vertex * 3 + axis] if has_centers else mesh.positions[vertex * 3 + axis]
            var packed = UInt32(0)
            var total = Float32(0)
            for i in range(4):
                var weight = mesh.weights[vertex * 4 + i]
                var joint = Int(mesh.joints[vertex * 4 + i])
                # As skin_vertex: an influence counts when its weight is positive and its joint exists.
                if weight > 0 and joint < joints:
                    packed |= UInt32(joint) << UInt32(i * 8)
                    total += weight
                else:
                    weight = 0
                floats[unsafe_offset=at + SKIN_STATIC_WEIGHTS + i] = weight
            words[unsafe_offset=at + SKIN_STATIC_JOINTS] = packed
            floats[unsafe_offset=at + SKIN_STATIC_INVERSE] = 1.0 / total if total > 0.00001 else Float32(0)
            words[unsafe_offset=at + 15] = 0
        if has_morphs:
            entry.morphs = mesh.morph_count
            var has_normals = len(mesh.morph_normals) == len(mesh.morph_positions)
            var morph_floats = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=self.morph_heap.cpu_address + Int(entry.morph_range.offset) * SKIN_UNIT)
            var morph_words = Pointer[UInt32, origin=MutAnyOrigin](unsafe_from_address=self.morph_heap.cpu_address + Int(entry.morph_range.offset) * SKIN_UNIT)
            var too_many = False
            for vertex in range(count):
                var slot = 0
                for morph in range(mesh.morph_count):
                    var source = (morph * count + vertex) * 3
                    var moves = mesh.morph_positions[source] != 0 or mesh.morph_positions[source + 1] != 0 or mesh.morph_positions[source + 2] != 0
                    if has_normals:
                        moves = moves or mesh.morph_normals[source] != 0 or mesh.morph_normals[source + 1] != 0 or mesh.morph_normals[source + 2] != 0
                    if not moves:
                        continue
                    if slot >= SKIN_MAX_MORPHS:
                        too_many = True  # the import check refuses such a model; the CPU skins it
                        continue
                    var at = vertex * SKIN_MORPH_WORDS + slot * SKIN_MORPH_ENTRY_WORDS
                    morph_words[unsafe_offset=at] = UInt32(morph)
                    for axis in range(3):
                        morph_floats[unsafe_offset=at + 1 + axis] = mesh.morph_positions[source + axis]
                        morph_floats[unsafe_offset=at + 4 + axis] = mesh.morph_normals[source + axis] if has_normals else Float32(0)
                    morph_words[unsafe_offset=at + 7] = 0
                    slot += 1
                for rest in range(slot, SKIN_MAX_MORPHS):
                    var at = vertex * SKIN_MORPH_WORDS + rest * SKIN_MORPH_ENTRY_WORDS
                    for word in range(SKIN_MORPH_ENTRY_WORDS):
                        morph_words[unsafe_offset=at + word] = 0
            if too_many:
                self.static_alloc.free(entry.static_range)
                self.morph_alloc.free(entry.morph_range)
                entry.static_range = OffsetAllocation()
                entry.morph_range = OffsetAllocation()
                entry.refused = True
                self.stats.refused_meshes += 1
                return self.keep_entry(entry^)
        self.stats.uploads += 1
        return self.keep_entry(entry^)

    def instance_for(mut self, mut raster: GpuRaster, mesh: NativeMesh, model: Int, mesh_index: Int, joints: Int) -> Int:
        """The instance this draw is: the mesh (uploaded on first sight) and how
        many times it was drawn before this frame. -1: this draw stays on the CPU."""
        if not self.ready:
            return -1
        var found = -1
        for index in range(len(self.meshes)):
            if self.meshes[index].model == model and self.meshes[index].mesh == mesh_index:
                found = index
        if found >= 0 and (self.meshes[found].data != Int(mesh.positions.unsafe_ptr()) or self.meshes[found].vertices != mesh.vertex_count() or self.meshes[found].joints != joints):
            # Another mesh behind the same handle, and no one said (forget_model).
            return -1
        if found < 0:
            found = self.upload(raster, mesh, model, mesh_index, joints)
        if self.meshes[found].refused:
            self.meshes[found].used_frame = self.frame
            return -1
        var ordinal = self.meshes[found].draws
        self.meshes[found].draws += 1
        self.meshes[found].used_frame = self.frame
        for index in range(len(self.instances)):
            if self.instances[index].mesh == found and self.instances[index].ordinal == ordinal:
                self.instances[index].used_frame = self.frame
                return index
        var instance = SkinInstance()
        instance.mesh = found
        instance.ordinal = ordinal
        instance.used_frame = self.frame
        var units = UInt32((self.meshes[found].vertices * SKIN_OUT_WORDS * 4 + SKIN_UNIT - 1) // SKIN_UNIT)
        for parity in range(2):
            instance.ranges[parity] = self.out_alloc.allocate(units)
        if not instance.ranges[0].ok() or not instance.ranges[1].ok():
            for parity in range(2):
                if instance.ranges[parity].ok():
                    self.out_alloc.free(instance.ranges[parity])
            return -1
        for index in range(len(self.instances)):
            if self.instances[index].mesh < 0:
                self.instances[index] = instance^
                return index
        self.instances.append(instance^)
        return len(self.instances) - 1

    def output_address(self, instance: Int, parity: Int) -> Int:
        """Where a batch of this parity wrote the instance: SKIN_OUT_WORDS floats a vertex."""
        return self.output.cpu_address + Int(self.instances[instance].ranges[parity].offset) * SKIN_UNIT

    def wait(mut self, mut raster: GpuRaster) -> Bool:
        """Wait for the batch in flight. False: the GPU did not finish it."""
        if self.fence == 0:
            return True
        var started = get_time()
        var done = raster.wait_fence(self.fence, True)
        self.stats.wait_seconds += get_time() - started
        self.fence = 0
        if done:
            self.finished = self.fence_batch
        else:
            self.error = "GPU skinning did not finish: " + raster.backend_error()
            self.ready = False
        return done

    def adopt(mut self, mut raster: GpuRaster, instance: Int, pose: List[Float32], method: Int, max_move: Float32) -> Int:
        """The address of the instance's vertices as the last batch skinned them
        (SKIN_OUT_WORDS floats each), or 0 when this draw must be skinned on the
        CPU: no finished batch wrote it, with another method, or in a pose more
        than `max_move` (a joint's position) or a wide turn away from `pose`
        (seven floats a joint: position, rotation)."""
        if instance < 0 or not self.usable(raster):
            return 0
        var last = self.batch - 1
        var parity = last & 1
        if last < 0 or self.instances[instance].written[parity] != last or self.frame - self.instances[instance].queued_frame > 1:
            self.stats.missed += 1  # nothing skinned it last frame
            return 0
        if not self.wait(raster) or self.finished != last:
            self.stats.missed += 1
            return 0
        ref then = self.instances[instance].pose
        if self.instances[instance].method != method or len(then) != len(pose):
            self.stats.refused += 1
            return 0
        var limit = max_move * max_move
        for joint in range(len(pose) // 7):
            var at = joint * 7
            var dx = pose[at] - then[at]
            var dy = pose[at + 1] - then[at + 1]
            var dz = pose[at + 2] - then[at + 2]
            var agree = pose[at + 3] * then[at + 3] + pose[at + 4] * then[at + 4] + pose[at + 5] * then[at + 5] + pose[at + 6] * then[at + 6]
            if dx * dx + dy * dy + dz * dz > limit or abs(agree) < SKIN_QUAT_AGREE:
                self.stats.refused += 1
                return 0
        self.stats.adopted += 1
        return self.output_address(instance, parity)

    def queue(mut self, mut raster: GpuRaster, instance: Int, palette_address: Int, joints: Int, pose: List[Float32], method: Int, morph_weights: List[Float32]) -> Bool:
        """Add the instance, in this pose, to the batch `dispatch` sends."""
        if instance < 0 or not self.usable(raster):
            return False
        var mesh = self.instances[instance].mesh
        var parity = self.batch & 1
        if self.instances[instance].written[parity] == self.batch:
            return False  # queued already
        var morphs = self.meshes[mesh].morphs
        var palette_floats = joints * PALETTE_FLOATS
        if joints != self.meshes[mesh].joints or (self.palette_words + palette_floats + morphs) * 4 > SKIN_PALETTE_BYTES:
            return False
        var block = self.palettes.cpu_address + parity * SKIN_PALETTE_BYTES
        var first = self.palette_words
        _ = external_call["memcpy", Int](block + first * 4, palette_address, palette_floats * 4)
        var morph_on = False
        var weights = Pointer[Float32, origin=MutAnyOrigin](unsafe_from_address=block + (first + palette_floats) * 4)
        for morph in range(morphs):
            var weight = morph_weights[morph] if morph < len(morph_weights) else Float32(0)
            weights[unsafe_offset=morph] = weight
            morph_on = morph_on or weight != 0
        self.palette_words += palette_floats + morphs
        var palette_base = parity * SKIN_PALETTE_BYTES // 4 + first
        var push = List[UInt32](length=SKIN_PUSH_WORDS, fill=0)
        push[SKIN_PUSH_STATIC] = UInt32(self.static_word(mesh))
        push[SKIN_PUSH_OUT] = UInt32(Int(self.instances[instance].ranges[parity].offset) * SKIN_UNIT // 4)
        push[SKIN_PUSH_PALETTE] = UInt32(palette_base)
        push[SKIN_PUSH_COUNT] = UInt32(self.meshes[mesh].vertices)
        push[SKIN_PUSH_METHOD] = UInt32(method)
        push[SKIN_PUSH_JOINT_MAX] = UInt32(joints - 1)
        push[SKIN_PUSH_MORPH] = UInt32(Int(self.meshes[mesh].morph_range.offset) * SKIN_UNIT // 4) if morphs > 0 else UInt32(0)
        push[SKIN_PUSH_MORPH_WEIGHTS] = UInt32(palette_base + palette_floats)
        push[SKIN_PUSH_MORPH_ON] = UInt32(1) if morph_on else UInt32(0)
        push[SKIN_PUSH_MORPH_MAX] = UInt32(max(morphs - 1, 0))
        self.pushes.append(push^)
        self.instances[instance].written[parity] = self.batch
        self.instances[instance].pose = pose.copy()
        self.instances[instance].method = method
        self.instances[instance].queued_frame = self.frame
        self.stats.queued += 1
        return True

    def dispatch(mut self, mut raster: GpuRaster) -> Bool:
        """Send this frame's batch to the side queue. The batch before it is
        done by then (its command buffer and palette block are reused in turn)."""
        if len(self.pushes) == 0:
            return True
        if not self.usable(raster):
            self.pushes.clear()
            self.palette_words = 0
            return False
        if not self.wait(raster):
            self.pushes.clear()
            self.palette_words = 0
            return False
        var ok = self.record_amdgpu(raster) if self.backend == BACKEND_AMDGPU else self.record_vulkan(raster)
        self.pushes.clear()
        self.palette_words = 0
        if not ok:
            self.ready = False
            return False
        self.fence_batch = self.batch
        self.batch += 1
        self.stats.batches += 1
        return True

    def groups(self, push: List[UInt32]) -> Int:
        return (Int(push[SKIN_PUSH_COUNT]) + KERNEL_GROUP - 1) // KERNEL_GROUP

    def record_amdgpu(mut self, mut raster: GpuRaster) -> Bool:
        var builder = CommandBuilder(raster.gpu.isa())
        builder.write_timestamp(self.addresses.gpu_address + UInt64(SKIN_STAMP_BYTE))
        kernel_compute_state(builder, self.code.gpu_address, self.vgprs, SKIN_PUSH_WORDS)
        builder.user_address(0, self.addresses.gpu_address)
        for push in self.pushes:
            builder.user_values(2, push.copy())
            builder.dispatch(UInt32(self.groups(push)), 1, 1)
        builder.pass_barrier()
        builder.write_timestamp(self.addresses.gpu_address + UInt64(SKIN_STAMP_BYTE + 8))
        var dwords = builder.write_to(self.commands)
        var handles: List[UInt32] = [self.commands.handle, self.code.handle, self.addresses.handle, self.static_heap.handle, self.palettes.handle, self.output.handle]
        if self.morph_heap.size > 0:
            handles.append(self.morph_heap.handle)
        # On the side entity, so a batch runs beside queued frames, not after them.
        self.fence = raster.gpu.submit(self.commands, raster.gpu.pad_commands(self.commands, dwords), handles, UInt32(SIDE_RING))
        if self.fence == 0:
            self.error = raster.gpu.error
            return False
        return True

    def record_vulkan(mut self, mut raster: GpuRaster) -> Bool:
        ref vk = raster.vk
        if not vk.begin_commands(self.command_index):
            self.error = vk.error
            return False
        for push in self.pushes:
            vk.dispatch(self.command_index, self.pipeline, self.descriptor, push.copy(), self.groups(push), 1)
        vk.barrier(self.command_index, PIPELINE_COMPUTE, ACCESS_SHADER_WRITE, PIPELINE_HOST, ACCESS_HOST_READ)
        self.fence = vk.submit_side(self.command_index)
        if self.fence == 0:
            self.error = vk.error
            return False
        return True

    def gpu_microseconds(self, raster: GpuRaster) -> Float64:
        """amdgpu: how long the last finished batch took on the GPU (0 on Vulkan)."""
        var hertz = raster.clock_hertz()
        if self.backend != BACKEND_AMDGPU or hertz <= 0 or self.addresses.size == 0:
            return 0
        var words = Pointer[UInt64, origin=MutAnyOrigin](unsafe_from_address=self.addresses.cpu_address + SKIN_STAMP_BYTE)
        return Float64(words[unsafe_offset=1] - words[unsafe_offset=0]) / hertz * 1e6

    def used_bytes(self) -> Int:
        return (Int(self.static_alloc.used_storage()) + Int(self.morph_alloc.used_storage()) + Int(self.out_alloc.used_storage())) * SKIN_UNIT

    def release(mut self, mut raster: GpuRaster):
        """Give the buffers back while the raster's device is the one they came from."""
        if self.recoveries != raster.recoveries or self.backend != raster.backend:
            self.ready = False
            self.forget_all()
            return
        if self.fence != 0:
            _ = raster.wait_fence(self.fence, True)
            self.fence = 0
        var owned = List[GpuBuffer]()
        owned.append(self.static_heap.copy())
        owned.append(self.morph_heap.copy())
        owned.append(self.palettes.copy())
        owned.append(self.output.copy())
        owned.append(self.code.copy())
        owned.append(self.commands.copy())
        owned.append(self.addresses.copy())
        for buffer in owned:
            if buffer.size > 0:
                raster.release(buffer)
        self.static_heap = GpuBuffer()
        self.morph_heap = GpuBuffer()
        self.palettes = GpuBuffer()
        self.output = GpuBuffer()
        self.code = GpuBuffer()
        self.commands = GpuBuffer()
        self.addresses = GpuBuffer()
        self.forget_all()
        self.ready = False

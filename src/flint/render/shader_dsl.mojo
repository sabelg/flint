"""Phase 1 (`V16`): Pure-Mojo Typed GPU Shader eDSL (`shader_dsl.mojo`).

Replaces external HLSL / Slang + DXC / shaderc C++ toolchains with zero-overhead
Mojo 1.2 operator-overloaded GPU expression types (`GpuF32`, `GpuU32`, `GpuBool`,
`GpuVec2`, `GpuVec3`, `GpuVec4`, `GpuMat4`), typed `BufferRef[StrideWords]` &
`BindlessTex2D` resource handles, structured control flow (`if_then` / `loop`),
and compile-time `[PERM: Bool]` shader permutation specialization emitting
SPIR-V 1.5 directly through `AaaSpirvModule`.
"""
from flint.render.spirv import (
    AaaSpirvModule,
    IMAGE_FORMAT_RGBA16F,
    IMAGE_FORMAT_R11F_G11F_B10F,
    MODEL_FRAGMENT,
    MODEL_GL_COMPUTE,
    MODEL_VERTEX,
    OP_LOAD,
    OP_STORE,
    STORAGE_INPUT,
    STORAGE_OUTPUT,
    emit,
)


@always_inline
def _mod(addr: Int) -> Pointer[AaaSpirvModule, MutAnyOrigin]:
    return Pointer[AaaSpirvModule, MutAnyOrigin](unsafe_from_address=addr)


# ============================================================================
# 1. Core Scalar Types (`GpuBool`, `GpuU32`, `GpuF32`)
# ============================================================================

@fieldwise_init
struct GpuBool(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var id: UInt32

    @always_inline
    def __and__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].logical_and(self.id, rhs.id))

    @always_inline
    def __or__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].logical_or(self.id, rhs.id))

    @always_inline
    def __invert__(self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].logical_not(self.id))

    @always_inline
    def not_(self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].logical_not(self.id))


@fieldwise_init
struct GpuU32(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var id: UInt32

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].iadd(self.id, rhs.id))

    @always_inline
    def __add__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].iadd(self.id, m[].uint(rhs)))

    @always_inline
    def __radd__(self, lhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].iadd(m[].uint(lhs), self.id))

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].isub(self.id, rhs.id))

    @always_inline
    def __sub__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].isub(self.id, m[].uint(rhs)))

    @always_inline
    def __rsub__(self, lhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].isub(m[].uint(lhs), self.id))

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].imul(self.id, rhs.id))

    @always_inline
    def __mul__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].imul(self.id, m[].uint(rhs)))

    @always_inline
    def __rmul__(self, lhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].imul(m[].uint(lhs), self.id))

    @always_inline
    def __floordiv__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].udiv(self.id, rhs.id))

    @always_inline
    def __floordiv__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].udiv(self.id, m[].uint(rhs)))

    @always_inline
    def __mod__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].umod(self.id, rhs.id))

    @always_inline
    def __mod__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].umod(self.id, m[].uint(rhs)))

    @always_inline
    def __and__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].bit_and(self.id, rhs.id))

    @always_inline
    def __and__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].bit_and(self.id, m[].uint(rhs)))

    @always_inline
    def __or__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].bit_or(self.id, rhs.id))

    @always_inline
    def __or__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].bit_or(self.id, m[].uint(rhs)))

    @always_inline
    def __xor__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].bit_xor(self.id, rhs.id))

    @always_inline
    def __xor__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].bit_xor(self.id, m[].uint(rhs)))

    @always_inline
    def __lshift__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].shl(self.id, rhs.id))

    @always_inline
    def __lshift__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].shl(self.id, m[].uint(rhs)))

    @always_inline
    def __rshift__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].shr(self.id, rhs.id))

    @always_inline
    def __rshift__(self, rhs: UInt32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].shr(self.id, m[].uint(rhs)))

    @always_inline
    def eq(self, rhs: Self) -> GpuBool:
        return GpuBool(self.mod_addr, _mod(self.mod_addr)[].ieq(self.id, rhs.id))

    @always_inline
    def eq(self, rhs: UInt32) -> GpuBool:
        var m = _mod(self.mod_addr)
        return GpuBool(self.mod_addr, m[].ieq(self.id, m[].uint(rhs)))

    @always_inline
    def ne(self, rhs: Self) -> GpuBool:
        return GpuBool(self.mod_addr, _mod(self.mod_addr)[].ine(self.id, rhs.id))

    @always_inline
    def ne(self, rhs: UInt32) -> GpuBool:
        var m = _mod(self.mod_addr)
        return GpuBool(self.mod_addr, m[].ine(self.id, m[].uint(rhs)))

    @always_inline
    def lt(self, rhs: Self) -> GpuBool:
        return GpuBool(self.mod_addr, _mod(self.mod_addr)[].ult(self.id, rhs.id))

    @always_inline
    def lt(self, rhs: UInt32) -> GpuBool:
        var m = _mod(self.mod_addr)
        return GpuBool(self.mod_addr, m[].ult(self.id, m[].uint(rhs)))

    @always_inline
    def ge(self, rhs: Self) -> GpuBool:
        return GpuBool(self.mod_addr, _mod(self.mod_addr)[].uge(self.id, rhs.id))

    @always_inline
    def ge(self, rhs: UInt32) -> GpuBool:
        var m = _mod(self.mod_addr)
        return GpuBool(self.mod_addr, m[].uge(self.id, m[].uint(rhs)))

    @always_inline
    def to_f32(self) -> GpuF32:
        return GpuF32(self.mod_addr, _mod(self.mod_addr)[].float_from_uint(self.id))

    @always_inline
    def bitcast_f32(self) -> GpuF32:
        return GpuF32(self.mod_addr, _mod(self.mod_addr)[].bitcast_float(self.id))


@fieldwise_init
struct GpuF32(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var id: UInt32

    @always_inline
    def __neg__(self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].fneg(self.id))

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].fadd(self.id, rhs.id))

    @always_inline
    def __add__(self, rhs: Float32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].fadd(self.id, m[].float(rhs)))

    @always_inline
    def __radd__(self, lhs: Float32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].fadd(m[].float(lhs), self.id))

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].fsub(self.id, rhs.id))

    @always_inline
    def __sub__(self, rhs: Float32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].fsub(self.id, m[].float(rhs)))

    @always_inline
    def __rsub__(self, lhs: Float32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].fsub(m[].float(lhs), self.id))

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].fmul(self.id, rhs.id))

    @always_inline
    def __mul__(self, rhs: Float32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].fmul(self.id, m[].float(rhs)))

    @always_inline
    def __rmul__(self, lhs: Float32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].fmul(m[].float(lhs), self.id))

    @always_inline
    def __truediv__(self, rhs: Self) -> Self:
        return Self(self.mod_addr, _mod(self.mod_addr)[].fdiv(self.id, rhs.id))

    @always_inline
    def __truediv__(self, rhs: Float32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].fdiv(self.id, m[].float(rhs)))

    @always_inline
    def __rtruediv__(self, lhs: Float32) -> Self:
        var m = _mod(self.mod_addr)
        return Self(self.mod_addr, m[].fdiv(m[].float(lhs), self.id))

    @always_inline
    def lt(self, rhs: Self) -> GpuBool:
        return GpuBool(self.mod_addr, _mod(self.mod_addr)[].flt(self.id, rhs.id))

    @always_inline
    def lt(self, rhs: Float32) -> GpuBool:
        var m = _mod(self.mod_addr)
        return GpuBool(self.mod_addr, m[].flt(self.id, m[].float(rhs)))

    @always_inline
    def gt(self, rhs: Self) -> GpuBool:
        return GpuBool(self.mod_addr, _mod(self.mod_addr)[].fgt(self.id, rhs.id))

    @always_inline
    def gt(self, rhs: Float32) -> GpuBool:
        var m = _mod(self.mod_addr)
        return GpuBool(self.mod_addr, m[].fgt(self.id, m[].float(rhs)))

    @always_inline
    def ge(self, rhs: Self) -> GpuBool:
        return GpuBool(self.mod_addr, _mod(self.mod_addr)[].fge(self.id, rhs.id))

    @always_inline
    def ge(self, rhs: Float32) -> GpuBool:
        var m = _mod(self.mod_addr)
        return GpuBool(self.mod_addr, m[].fge(self.id, m[].float(rhs)))

    @always_inline
    def saturate_u32(self) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].saturate_uint(self.id))

    @always_inline
    def bitcast_u32(self) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].bitcast_uint(self.id))


# ============================================================================
# 2. Vector & Matrix Types (`GpuVec2`, `GpuVec3`, `GpuVec4`, `GpuMat4`)
# ============================================================================

@fieldwise_init
struct GpuVec2(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var x_id: UInt32
    var y_id: UInt32

    def __init__(out self, x: GpuF32, y: GpuF32):
        self.mod_addr = x.mod_addr
        self.x_id = x.id
        self.y_id = y.id

    @always_inline
    def x(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.x_id)

    @always_inline
    def y(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.y_id)

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        return Self(self.x() + rhs.x(), self.y() + rhs.y())

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        return Self(self.x() - rhs.x(), self.y() - rhs.y())

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        return Self(self.x() * rhs.x(), self.y() * rhs.y())

    @always_inline
    def __mul__(self, s: GpuF32) -> Self:
        return Self(self.x() * s, self.y() * s)

    @always_inline
    def __mul__(self, s: Float32) -> Self:
        return Self(self.x() * s, self.y() * s)

    @always_inline
    def dot(self, rhs: Self) -> GpuF32:
        var m = _mod(self.mod_addr)
        return GpuF32(self.mod_addr, m[].fma(self.x_id, rhs.x_id, m[].fmul(self.y_id, rhs.y_id)))

    @always_inline
    def to_spv(self) -> UInt32:
        return _mod(self.mod_addr)[].vec2(self.x_id, self.y_id)

    @staticmethod
    def from_spv(mod_addr: Int, v_id: UInt32) -> Self:
        var m = _mod(mod_addr)
        return Self(mod_addr, m[].extract_f(v_id, 0), m[].extract_f(v_id, 1))


@fieldwise_init
struct GpuVec3(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var x_id: UInt32
    var y_id: UInt32
    var z_id: UInt32

    def __init__(out self, x: GpuF32, y: GpuF32, z: GpuF32):
        self.mod_addr = x.mod_addr
        self.x_id = x.id
        self.y_id = y.id
        self.z_id = z.id

    @always_inline
    def x(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.x_id)

    @always_inline
    def y(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.y_id)

    @always_inline
    def z(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.z_id)

    @always_inline
    def __neg__(self) -> Self:
        return Self(-self.x(), -self.y(), -self.z())

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        return Self(self.x() + rhs.x(), self.y() + rhs.y(), self.z() + rhs.z())

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        return Self(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z())

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        return Self(self.x() * rhs.x(), self.y() * rhs.y(), self.z() * rhs.z())

    @always_inline
    def __mul__(self, s: GpuF32) -> Self:
        return Self(self.x() * s, self.y() * s, self.z() * s)

    @always_inline
    def __rmul__(self, s: GpuF32) -> Self:
        return Self(self.x() * s, self.y() * s, self.z() * s)

    @always_inline
    def __mul__(self, s: Float32) -> Self:
        return Self(self.x() * s, self.y() * s, self.z() * s)

    @always_inline
    def __rmul__(self, s: Float32) -> Self:
        return Self(self.x() * s, self.y() * s, self.z() * s)

    @always_inline
    def __truediv__(self, s: GpuF32) -> Self:
        var inv = Float32(1.0) / s
        return self * inv

    @always_inline
    def dot(self, rhs: Self) -> GpuF32:
        var m = _mod(self.mod_addr)
        return GpuF32(
            self.mod_addr,
            m[].fma(self.x_id, rhs.x_id, m[].fma(self.y_id, rhs.y_id, m[].fmul(self.z_id, rhs.z_id))),
        )

    @always_inline
    def cross(self, rhs: Self) -> Self:
        var m = _mod(self.mod_addr)
        var cx = m[].fsub(m[].fmul(self.y_id, rhs.z_id), m[].fmul(self.z_id, rhs.y_id))
        var cy = m[].fsub(m[].fmul(self.z_id, rhs.x_id), m[].fmul(self.x_id, rhs.z_id))
        var cz = m[].fsub(m[].fmul(self.x_id, rhs.y_id), m[].fmul(self.y_id, rhs.x_id))
        return Self(self.mod_addr, cx, cy, cz)

    @always_inline
    def length_squared(self) -> GpuF32:
        return self.dot(self)

    @always_inline
    def length(self) -> GpuF32:
        var m = _mod(self.mod_addr)
        return GpuF32(self.mod_addr, m[].fsqrt(m[].nmax(m[].float(1e-12), self.dot(self).id)))

    @always_inline
    def normalized(self) -> Self:
        var m = _mod(self.mod_addr)
        var len2 = m[].nmax(m[].float(1e-12), self.dot(self).id)
        var inv_l = GpuF32(self.mod_addr, m[].inverse_sqrt(len2))
        return self * inv_l

    @always_inline
    def to_spv(self) -> UInt32:
        return _mod(self.mod_addr)[].vec3(self.x_id, self.y_id, self.z_id)

    @staticmethod
    def from_spv(mod_addr: Int, v_id: UInt32) -> Self:
        var m = _mod(mod_addr)
        return Self(mod_addr, m[].extract_f(v_id, 0), m[].extract_f(v_id, 1), m[].extract_f(v_id, 2))


@fieldwise_init
struct GpuVec4(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var x_id: UInt32
    var y_id: UInt32
    var z_id: UInt32
    var w_id: UInt32

    def __init__(out self, x: GpuF32, y: GpuF32, z: GpuF32, w: GpuF32):
        self.mod_addr = x.mod_addr
        self.x_id = x.id
        self.y_id = y.id
        self.z_id = z.id
        self.w_id = w.id

    def __init__(out self, xyz: GpuVec3, w: GpuF32):
        self.mod_addr = xyz.mod_addr
        self.x_id = xyz.x_id
        self.y_id = xyz.y_id
        self.z_id = xyz.z_id
        self.w_id = w.id

    @always_inline
    def x(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.x_id)

    @always_inline
    def y(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.y_id)

    @always_inline
    def z(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.z_id)

    @always_inline
    def w(self) -> GpuF32:
        return GpuF32(self.mod_addr, self.w_id)

    @always_inline
    def xyz(self) -> GpuVec3:
        return GpuVec3(self.mod_addr, self.x_id, self.y_id, self.z_id)

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        return Self(self.x() + rhs.x(), self.y() + rhs.y(), self.z() + rhs.z(), self.w() + rhs.w())

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        return Self(self.x() - rhs.x(), self.y() - rhs.y(), self.z() - rhs.z(), self.w() - rhs.w())

    @always_inline
    def __mul__(self, s: GpuF32) -> Self:
        return Self(self.x() * s, self.y() * s, self.z() * s, self.w() * s)

    @always_inline
    def to_spv(self) -> UInt32:
        return _mod(self.mod_addr)[].vec4(self.x_id, self.y_id, self.z_id, self.w_id)

    @staticmethod
    def from_spv(mod_addr: Int, v_id: UInt32) -> Self:
        var m = _mod(mod_addr)
        return Self(
            mod_addr,
            m[].extract_f(v_id, 0),
            m[].extract_f(v_id, 1),
            m[].extract_f(v_id, 2),
            m[].extract_f(v_id, 3),
        )


struct GpuMat4(Copyable, Movable):
    """Column-major 4x4 GPU matrix backed by 16 SSA float IDs."""
    var mod_addr: Int
    var cols: Array[UInt32, 16]

    def __init__(out self, mod_addr: Int):
        self.mod_addr = mod_addr
        self.cols = Array[UInt32, 16](fill=0)

    def mul_point(self, p: GpuVec3) -> GpuVec4:
        var m = _mod(self.mod_addr)
        var rx = m[].fma(self.cols[0], p.x_id, m[].fma(self.cols[4], p.y_id, m[].fma(self.cols[8], p.z_id, self.cols[12])))
        var ry = m[].fma(self.cols[1], p.x_id, m[].fma(self.cols[5], p.y_id, m[].fma(self.cols[9], p.z_id, self.cols[13])))
        var rz = m[].fma(self.cols[2], p.x_id, m[].fma(self.cols[6], p.y_id, m[].fma(self.cols[10], p.z_id, self.cols[14])))
        var rw = m[].fma(self.cols[3], p.x_id, m[].fma(self.cols[7], p.y_id, m[].fma(self.cols[11], p.z_id, self.cols[15])))
        return GpuVec4(self.mod_addr, rx, ry, rz, rw)

    def mul_dir(self, d: GpuVec3) -> GpuVec3:
        var m = _mod(self.mod_addr)
        var rx = m[].fma(self.cols[0], d.x_id, m[].fma(self.cols[4], d.y_id, m[].fmul(self.cols[8], d.z_id)))
        var ry = m[].fma(self.cols[1], d.x_id, m[].fma(self.cols[5], d.y_id, m[].fmul(self.cols[9], d.z_id)))
        var rz = m[].fma(self.cols[2], d.x_id, m[].fma(self.cols[6], d.y_id, m[].fmul(self.cols[10], d.z_id)))
        return GpuVec3(self.mod_addr, rx, ry, rz)


# ============================================================================
# 3. Mutable Function Variables (`GpuVarF32`, `GpuVarU32`, `GpuVarVec3`)
# ============================================================================

@fieldwise_init
struct GpuVarU32(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var var_id: UInt32

    @always_inline
    def get(self) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].load_u(self.var_id))

    @always_inline
    def set(self, val: GpuU32):
        _mod(self.mod_addr)[].store(self.var_id, val.id)


@fieldwise_init
struct GpuVarF32(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var var_id: UInt32

    @always_inline
    def get(self) -> GpuF32:
        return GpuF32(self.mod_addr, _mod(self.mod_addr)[].load_f(self.var_id))

    @always_inline
    def set(self, val: GpuF32):
        _mod(self.mod_addr)[].store(self.var_id, val.id)


@fieldwise_init
struct GpuVarVec3(Copyable, ImplicitlyCopyable, Movable):
    var vx: GpuVarF32
    var vy: GpuVarF32
    var vz: GpuVarF32

    @always_inline
    def get(self) -> GpuVec3:
        return GpuVec3(self.vx.get(), self.vy.get(), self.vz.get())

    @always_inline
    def set(self, v: GpuVec3):
        self.vx.set(v.x())
        self.vy.set(v.y())
        self.vz.set(v.z())


# ============================================================================
# 4. Typed Resource Handles (`BufferRef[StrideWords]`, `SampledTex2D`, `BindlessTex2D`)
# ============================================================================

@fieldwise_init
struct BufferRef[StrideWords: Int = 1](Copyable, ImplicitlyCopyable, Movable):
    """Typed storage buffer handle with compile-time element word stride specialization."""
    var mod_addr: Int
    var buf_id: UInt32

    @always_inline
    def base_word(self, elem_idx: GpuU32) -> GpuU32:
        if Self.StrideWords == 1:
            return elem_idx
        return elem_idx * UInt32(Self.StrideWords)

    @always_inline
    def load_u32(self, word_idx: GpuU32) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].load_word(self.buf_id, word_idx.id))

    @always_inline
    def load_f32(self, word_idx: GpuU32) -> GpuF32:
        return GpuF32(self.mod_addr, _mod(self.mod_addr)[].load_float(self.buf_id, word_idx.id))

    @always_inline
    def load_field_f32(self, elem_idx: GpuU32, field_word: Int) -> GpuF32:
        var off = self.base_word(elem_idx)
        if field_word == 0:
            return self.load_f32(off)
        return self.load_f32(off + UInt32(field_word))

    @always_inline
    def load_field_u32(self, elem_idx: GpuU32, field_word: Int) -> GpuU32:
        var off = self.base_word(elem_idx)
        if field_word == 0:
            return self.load_u32(off)
        return self.load_u32(off + UInt32(field_word))

    @always_inline
    def load_vec3(self, word_idx: GpuU32) -> GpuVec3:
        return GpuVec3(
            self.load_f32(word_idx),
            self.load_f32(word_idx + UInt32(1)),
            self.load_f32(word_idx + UInt32(2)),
        )

    @always_inline
    def load_vec4(self, word_idx: GpuU32) -> GpuVec4:
        return GpuVec4(
            self.load_f32(word_idx),
            self.load_f32(word_idx + UInt32(1)),
            self.load_f32(word_idx + UInt32(2)),
            self.load_f32(word_idx + UInt32(3)),
        )

    def load_mat4(self, word_idx: GpuU32) -> GpuMat4:
        var mat = GpuMat4(self.mod_addr)
        mat.cols[0] = self.load_f32(word_idx).id
        for k in range(1, 16):
            mat.cols[k] = self.load_f32(word_idx + UInt32(k)).id
        return mat^

    @always_inline
    def store_u32(self, word_idx: GpuU32, val: GpuU32):
        _mod(self.mod_addr)[].store_word(self.buf_id, word_idx.id, val.id)

    @always_inline
    def store_f32(self, word_idx: GpuU32, val: GpuF32):
        _mod(self.mod_addr)[].store_float(self.buf_id, word_idx.id, val.id)

    @always_inline
    def store_vec3(self, word_idx: GpuU32, v: GpuVec3):
        self.store_f32(word_idx, v.x())
        self.store_f32(word_idx + UInt32(1), v.y())
        self.store_f32(word_idx + UInt32(2), v.z())

    @always_inline
    def store_vec4(self, word_idx: GpuU32, v: GpuVec4):
        self.store_f32(word_idx, v.x())
        self.store_f32(word_idx + UInt32(1), v.y())
        self.store_f32(word_idx + UInt32(2), v.z())
        self.store_f32(word_idx + UInt32(3), v.w())

    @always_inline
    def atomic_add(self, word_idx: GpuU32, delta: GpuU32) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].atomic_add_word(self.buf_id, word_idx.id, delta.id))


@fieldwise_init
struct SampledTex2D(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var samp_type: UInt32
    var var_id: UInt32

    @always_inline
    def sample_lod(self, uv: GpuVec2, lod: GpuF32) -> GpuVec4:
        var v4 = _mod(self.mod_addr)[].sample_2d_lod((self.samp_type, self.var_id), uv.to_spv(), lod.id)
        return GpuVec4.from_spv(self.mod_addr, v4)

    @always_inline
    def fetch_f32(self, x: GpuU32, y: GpuU32, lod: GpuU32) -> GpuVec4:
        var m = _mod(self.mod_addr)
        var v4 = m[].texel_fetch_f2d((self.samp_type, self.var_id), m[].uvec2(x.id, y.id), lod.id)
        return GpuVec4.from_spv(self.mod_addr, v4)


@fieldwise_init
struct SampledCube(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var samp_type: UInt32
    var var_id: UInt32

    @always_inline
    def sample_lod(self, dir_v: GpuVec3, lod: GpuF32) -> GpuVec4:
        var v4 = _mod(self.mod_addr)[].sample_cube_lod((self.samp_type, self.var_id), dir_v.to_spv(), lod.id)
        return GpuVec4.from_spv(self.mod_addr, v4)


@fieldwise_init
struct BindlessTex2D(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var elem_ptr_type: UInt32
    var array_var_id: UInt32

    @always_inline
    def sample_grad(self, tex_idx: GpuU32, uv: GpuVec2, ddx: GpuVec2, ddy: GpuVec2) -> GpuVec4:
        var v4 = _mod(self.mod_addr)[].sample_bindless_grad(
            (self.elem_ptr_type, self.array_var_id),
            tex_idx.id,
            uv.to_spv(),
            ddx.to_spv(),
            ddy.to_spv(),
        )
        return GpuVec4.from_spv(self.mod_addr, v4)

    @always_inline
    def sample_lod(self, tex_idx: GpuU32, uv: GpuVec2, lod: GpuF32) -> GpuVec4:
        var v4 = _mod(self.mod_addr)[].sample_bindless_lod(
            (self.elem_ptr_type, self.array_var_id),
            tex_idx.id,
            uv.to_spv(),
            lod.id,
        )
        return GpuVec4.from_spv(self.mod_addr, v4)


# ============================================================================
# 4b. Workgroup LDS Arrays & 2D Storage Images
# ============================================================================

@fieldwise_init
struct GpuSharedArrayF32[Size: Int](Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var ptr_elem_t: UInt32
    var var_id: UInt32

    @always_inline
    def load(self, index: GpuU32) -> GpuF32:
        return GpuF32(self.mod_addr, _mod(self.mod_addr)[].load_workgroup_f32(self.ptr_elem_t, self.var_id, index.id))

    @always_inline
    def store(self, index: GpuU32, value: GpuF32):
        _mod(self.mod_addr)[].store_workgroup_f32(self.ptr_elem_t, self.var_id, index.id, value.id)


@fieldwise_init
struct GpuSharedArrayU32[Size: Int](Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var ptr_elem_t: UInt32
    var var_id: UInt32

    @always_inline
    def load(self, index: GpuU32) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].load_workgroup_u32(self.ptr_elem_t, self.var_id, index.id))

    @always_inline
    def store(self, index: GpuU32, value: GpuU32):
        _mod(self.mod_addr)[].store_workgroup_u32(self.ptr_elem_t, self.var_id, index.id, value.id)

    @always_inline
    def atomic_add(self, index: GpuU32, value: GpuU32) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].atomic_add_workgroup_u32(self.ptr_elem_t, self.var_id, index.id, value.id))


@fieldwise_init
struct StorageImage2D(Copyable, ImplicitlyCopyable, Movable):
    var mod_addr: Int
    var image_type: UInt32
    var var_id: UInt32

    @always_inline
    def read(self, coord: GpuVec2) -> GpuVec4:
        var m = _mod(self.mod_addr)
        var ucoord = m[].uvec2(m[].uint_from_float(coord.x().id), m[].uint_from_float(coord.y().id))
        var res = m[].image_read((self.image_type, self.var_id), ucoord)
        return GpuVec4.from_spv(self.mod_addr, res)

    @always_inline
    def read_u(self, coord_x: GpuU32, coord_y: GpuU32) -> GpuVec4:
        var m = _mod(self.mod_addr)
        var ucoord = m[].uvec2(coord_x.id, coord_y.id)
        var res = m[].image_read((self.image_type, self.var_id), ucoord)
        return GpuVec4.from_spv(self.mod_addr, res)

    @always_inline
    def write(self, coord: GpuVec2, color: GpuVec4):
        var m = _mod(self.mod_addr)
        var ucoord = m[].uvec2(m[].uint_from_float(coord.x().id), m[].uint_from_float(coord.y().id))
        m[].image_write((self.image_type, self.var_id), ucoord, color.to_spv())

    @always_inline
    def write_u(self, coord_x: GpuU32, coord_y: GpuU32, color: GpuVec4):
        var m = _mod(self.mod_addr)
        var ucoord = m[].uvec2(coord_x.id, coord_y.id)
        m[].image_write((self.image_type, self.var_id), ucoord, color.to_spv())


# ============================================================================
# 5. Intrinsic Math & Selection Functions (Slang / HLSL Style)
# ============================================================================

@always_inline
def dot(a: GpuVec3, b: GpuVec3) -> GpuF32:
    return a.dot(b)


@always_inline
def dot2(a: GpuVec2, b: GpuVec2) -> GpuF32:
    return a.dot(b)


@always_inline
def cross(a: GpuVec3, b: GpuVec3) -> GpuVec3:
    return a.cross(b)


@always_inline
def normalize(v: GpuVec3) -> GpuVec3:
    return v.normalized()


@always_inline
def length(v: GpuVec3) -> GpuF32:
    return v.length()


@always_inline
def fma_f(a: GpuF32, b: GpuF32, c: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].fma(a.id, b.id, c.id))


@always_inline
def abs_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].fabs(a.id))


@always_inline
def sqrt_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].fsqrt(a.id))


@always_inline
def rsqrt_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].inverse_sqrt(a.id))


@always_inline
def sin_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].fsin(a.id))


@always_inline
def cos_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].fcos(a.id))


@always_inline
def exp2_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].exp2(a.id))


@always_inline
def log2_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].log2(a.id))


@always_inline
def pow_f(a: GpuF32, b: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].fpow(a.id, b.id))


@always_inline
def pow_f(a: GpuF32, b: Float32) -> GpuF32:
    var m = _mod(a.mod_addr)
    return GpuF32(a.mod_addr, m[].fpow(a.id, m[].float(b)))


@always_inline
def floor_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].ffloor(a.id))


@always_inline
def fract_f(a: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].fract(a.id))


@always_inline
def min_f(a: GpuF32, b: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].nmin(a.id, b.id))


@always_inline
def min_f(a: GpuF32, b: Float32) -> GpuF32:
    var m = _mod(a.mod_addr)
    return GpuF32(a.mod_addr, m[].nmin(a.id, m[].float(b)))


@always_inline
def max_f(a: GpuF32, b: GpuF32) -> GpuF32:
    return GpuF32(a.mod_addr, _mod(a.mod_addr)[].nmax(a.id, b.id))


@always_inline
def max_f(a: GpuF32, b: Float32) -> GpuF32:
    var m = _mod(a.mod_addr)
    return GpuF32(a.mod_addr, m[].nmax(a.id, m[].float(b)))


@always_inline
def clamp_f(x: GpuF32, lo: GpuF32, hi: GpuF32) -> GpuF32:
    return GpuF32(x.mod_addr, _mod(x.mod_addr)[].nclamp(x.id, lo.id, hi.id))


@always_inline
def clamp_f(x: GpuF32, lo: Float32, hi: Float32) -> GpuF32:
    var m = _mod(x.mod_addr)
    return GpuF32(x.mod_addr, m[].nclamp(x.id, m[].float(lo), m[].float(hi)))


@always_inline
def saturate(x: GpuF32) -> GpuF32:
    return clamp_f(x, Float32(0.0), Float32(1.0))


@always_inline
def min_u(a: GpuU32, b: GpuU32) -> GpuU32:
    return GpuU32(a.mod_addr, _mod(a.mod_addr)[].umin(a.id, b.id))


@always_inline
def max_u(a: GpuU32, b: GpuU32) -> GpuU32:
    return GpuU32(a.mod_addr, _mod(a.mod_addr)[].umax(a.id, b.id))


@always_inline
def mix(a: GpuF32, b: GpuF32, t: GpuF32) -> GpuF32:
    return fma_f(b - a, t, a)


@always_inline
def mix_vec3(a: GpuVec3, b: GpuVec3, t: GpuF32) -> GpuVec3:
    return GpuVec3(
        mix(a.x(), b.x(), t),
        mix(a.y(), b.y(), t),
        mix(a.z(), b.z(), t),
    )


@always_inline
def select_f(cond: GpuBool, yes: GpuF32, no: GpuF32) -> GpuF32:
    return GpuF32(cond.mod_addr, _mod(cond.mod_addr)[].select_f(cond.id, yes.id, no.id))


@always_inline
def select_u(cond: GpuBool, yes: GpuU32, no: GpuU32) -> GpuU32:
    return GpuU32(cond.mod_addr, _mod(cond.mod_addr)[].select_u(cond.id, yes.id, no.id))


@always_inline
def select_vec3(cond: GpuBool, yes: GpuVec3, no: GpuVec3) -> GpuVec3:
    return GpuVec3(
        select_f(cond, yes.x(), no.x()),
        select_f(cond, yes.y(), no.y()),
        select_f(cond, yes.z(), no.z()),
    )


# ============================================================================
# 6. Top-Level `ShaderDsl` Builder (Heap-Pinned `AaaSpirvModule` Context)
# ============================================================================

struct ShaderDsl(Movable):
    """High-level Pure-Mojo GPU Shader Builder emitting SPIR-V 1.5."""
    var _box: List[AaaSpirvModule]
    var mod_addr: Int

    def __init__(
        out self,
        exec_model: UInt32 = MODEL_GL_COMPUTE,
        group_width: Int = 8,
        group_height: Int = 8,
        push_words: Int = 16,
        enable_bindless: Bool = False,
        enable_ray_query: Bool = False,
        early_depth: Bool = True,
        enable_subgroups: Bool = False,
        enable_storage_image_formats: Bool = False,
    ):
        self._box = List[AaaSpirvModule](capacity=1)
        self._box.append(
            AaaSpirvModule(
                exec_model=exec_model,
                group_width=group_width,
                group_height=group_height,
                push_words=push_words,
                enable_bindless=enable_bindless,
                enable_ray_query=enable_ray_query,
                early_depth=early_depth,
                enable_subgroups=enable_subgroups,
                enable_storage_image_formats=enable_storage_image_formats,
            )
        )
        self.mod_addr = Int(self._box.unsafe_ptr())

    @always_inline
    def raw_module(mut self) -> Pointer[AaaSpirvModule, MutAnyOrigin]:
        return _mod(self.mod_addr)

    @always_inline
    def f32(mut self, val: Float32) -> GpuF32:
        return GpuF32(self.mod_addr, _mod(self.mod_addr)[].float(val))

    @always_inline
    def u32(mut self, val: UInt32) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].uint(val))

    @always_inline
    def vec2(mut self, x: GpuF32, y: GpuF32) -> GpuVec2:
        return GpuVec2(x, y)

    @always_inline
    def vec3(mut self, x: GpuF32, y: GpuF32, z: GpuF32) -> GpuVec3:
        return GpuVec3(x, y, z)

    @always_inline
    def vec3_const(mut self, x: Float32, y: Float32, z: Float32) -> GpuVec3:
        return GpuVec3(self.f32(x), self.f32(y), self.f32(z))

    @always_inline
    def vec4(mut self, x: GpuF32, y: GpuF32, z: GpuF32, w: GpuF32) -> GpuVec4:
        return GpuVec4(x, y, z, w)

    @always_inline
    def var_f32(mut self, initial: GpuF32) -> GpuVarF32:
        return GpuVarF32(self.mod_addr, _mod(self.mod_addr)[].fvar(initial.id))

    @always_inline
    def var_f32_const(mut self, initial: Float32) -> GpuVarF32:
        var init_v = self.f32(initial)
        return GpuVarF32(self.mod_addr, _mod(self.mod_addr)[].fvar(init_v.id))

    @always_inline
    def var_u32(mut self, initial: GpuU32) -> GpuVarU32:
        return GpuVarU32(self.mod_addr, _mod(self.mod_addr)[].uvar(initial.id))

    @always_inline
    def var_vec3(mut self, initial: GpuVec3) -> GpuVarVec3:
        return GpuVarVec3(
            self.var_f32(initial.x()),
            self.var_f32(initial.y()),
            self.var_f32(initial.z()),
        )

    @always_inline
    def buffer[StrideWords: Int = 1](mut self, binding: Int, set: Int = 0) -> BufferRef[StrideWords]:
        return BufferRef[StrideWords](self.mod_addr, _mod(self.mod_addr)[].buffer(binding, set))

    @always_inline
    def sampled_tex2d(mut self, binding: Int, set: Int = 0) -> SampledTex2D:
        var pair = _mod(self.mod_addr)[].sampled_image_2d(binding, set, is_uint=False, is_cube=False)
        return SampledTex2D(self.mod_addr, pair[0], pair[1])

    @always_inline
    def sampled_cube(mut self, binding: Int, set: Int = 0) -> SampledCube:
        var pair = _mod(self.mod_addr)[].sampled_image_2d(binding, set, is_uint=False, is_cube=True)
        return SampledCube(self.mod_addr, pair[0], pair[1])

    @always_inline
    def bindless_tex2d(mut self, binding: Int, array_len: Int, set: Int = 0) -> BindlessTex2D:
        var pair = _mod(self.mod_addr)[].bindless_sampled_array_2d(binding, array_len, set)
        return BindlessTex2D(self.mod_addr, pair[0], pair[1])

    @always_inline
    def global_id_x(mut self) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].invocation_axis(0))

    @always_inline
    def global_id_y(mut self) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].invocation_axis(1))

    @always_inline
    def global_id_z(mut self) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].invocation_axis(2))

    @always_inline
    def push_u32(mut self, index: Int) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].push_word(index))

    @always_inline
    def push_f32(mut self, index: Int) -> GpuF32:
        return GpuF32(self.mod_addr, _mod(self.mod_addr)[].push_float(index))

    @always_inline
    def if_then(mut self, cond: GpuBool):
        _mod(self.mod_addr)[].begin_if(cond.id)

    @always_inline
    def else_branch(mut self):
        _mod(self.mod_addr)[].begin_else()

    @always_inline
    def end_if(mut self):
        _mod(self.mod_addr)[].end_if()

    @always_inline
    def begin_loop(mut self):
        _mod(self.mod_addr)[].begin_loop()

    @always_inline
    def break_if(mut self, cond: GpuBool):
        _mod(self.mod_addr)[].break_if(cond.id)

    @always_inline
    def continue_if(mut self, cond: GpuBool):
        _mod(self.mod_addr)[].continue_if(cond.id)

    @always_inline
    def end_loop(mut self):
        _mod(self.mod_addr)[].end_loop()

    @always_inline
    def return_if(mut self, cond: GpuBool):
        _mod(self.mod_addr)[].return_if(cond.id)

    @always_inline
    def shared_array_f32[Size: Int](mut self) -> GpuSharedArrayF32[Size]:
        var pair = _mod(self.mod_addr)[].workgroup_array_f32(Size)
        return GpuSharedArrayF32[Size](self.mod_addr, pair[0], pair[1])

    @always_inline
    def shared_array_u32[Size: Int](mut self) -> GpuSharedArrayU32[Size]:
        var pair = _mod(self.mod_addr)[].workgroup_array_u32(Size)
        return GpuSharedArrayU32[Size](self.mod_addr, pair[0], pair[1])

    @always_inline
    def workgroup_barrier(mut self):
        _mod(self.mod_addr)[].workgroup_barrier()

    @always_inline
    def memory_barrier(mut self, mem_scope: UInt32 = 2, mem_semantics: UInt32 = 776):
        _mod(self.mod_addr)[].memory_barrier(mem_scope, mem_semantics)

    @always_inline
    def subgroup_elect(mut self) -> GpuBool:
        return GpuBool(self.mod_addr, _mod(self.mod_addr)[].subgroup_elect())

    @always_inline
    def subgroup_ballot(mut self, pred: GpuBool) -> GpuU32:
        var b4 = _mod(self.mod_addr)[].subgroup_ballot(pred.id)
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].extract_u(b4, 0))

    @always_inline
    def subgroup_add_f32(mut self, val: GpuF32) -> GpuF32:
        return GpuF32(self.mod_addr, _mod(self.mod_addr)[].subgroup_fadd(val.id))

    @always_inline
    def subgroup_add_u32(mut self, val: GpuU32) -> GpuU32:
        return GpuU32(self.mod_addr, _mod(self.mod_addr)[].subgroup_iadd(val.id))

    @always_inline
    def storage_image_2d(mut self, binding: Int, set: Int = 0, format: UInt32 = IMAGE_FORMAT_RGBA16F) -> StorageImage2D:
        var pair = _mod(self.mod_addr)[].storage_image_2d(binding, set, format)
        return StorageImage2D(self.mod_addr, pair[0], pair[1])

    def finish(mut self) -> List[UInt32]:
        var words = _mod(self.mod_addr)[].finish()
        var anchor = Int(self._box.unsafe_ptr())
        _ = anchor
        return words^

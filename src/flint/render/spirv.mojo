"""A SPIR-V emitter for compute shaders, in the spirit of platform/amd_isa.mojo.

Shaders are written as Mojo calls that append instructions, with no SDK and no
compiler in between. A module targets SPIR-V 1.3 and Vulkan 1.2 with only the
`Shader` capability. Everything is scalar: 32-bit unsigned integers, 32-bit
floats and booleans. Storage buffers are flat arrays of uint, so a shader
addresses memory the way the amdgpu shaders do, in words from a binding's start.

Values are ids, as in SPIR-V itself: `s.fadd(a, b)` emits an instruction and
returns its result. There is one integer type, because SPIR-V's signed
operations do not care about the signedness of the type. Mutable values live in
function variables (`uvar`, `fvar`, `load_u`, `load_f`, `store`), which keeps the
emitter free of phi nodes; the driver promotes them.

Control flow is structured, which SPIR-V requires:

    s.begin_if(condition) ... s.begin_else() ... s.end_if()
    s.begin_loop()
    s.break_if(done)
    s.continue_if(skip)
    s.end_loop()
    s.return_if(condition)
"""
from std.collections import Dict

comptime SPIRV_MAGIC: UInt32 = 0x07230203
comptime SPIRV_VERSION_1_3: UInt32 = 0x00010300

comptime OP_EXT_INST_IMPORT: UInt32 = 11
comptime OP_EXT_INST: UInt32 = 12
comptime OP_MEMORY_MODEL: UInt32 = 14
comptime OP_ENTRY_POINT: UInt32 = 15
comptime OP_EXECUTION_MODE: UInt32 = 16
comptime OP_CAPABILITY: UInt32 = 17
comptime OP_TYPE_VOID: UInt32 = 19
comptime OP_TYPE_BOOL: UInt32 = 20
comptime OP_TYPE_INT: UInt32 = 21
comptime OP_TYPE_FLOAT: UInt32 = 22
comptime OP_TYPE_VECTOR: UInt32 = 23
comptime OP_TYPE_ARRAY: UInt32 = 28
comptime OP_TYPE_RUNTIME_ARRAY: UInt32 = 29
comptime OP_TYPE_STRUCT: UInt32 = 30
comptime OP_TYPE_POINTER: UInt32 = 32
comptime OP_TYPE_FUNCTION: UInt32 = 33
comptime OP_CONSTANT: UInt32 = 43
comptime OP_FUNCTION: UInt32 = 54
comptime OP_FUNCTION_END: UInt32 = 56
comptime OP_VARIABLE: UInt32 = 59
comptime OP_LOAD: UInt32 = 61
comptime OP_STORE: UInt32 = 62
comptime OP_ACCESS_CHAIN: UInt32 = 65
comptime OP_DECORATE: UInt32 = 71
comptime OP_MEMBER_DECORATE: UInt32 = 72
comptime OP_CONVERT_F_TO_U: UInt32 = 109
comptime OP_CONVERT_F_TO_S: UInt32 = 110
comptime OP_CONVERT_S_TO_F: UInt32 = 111
comptime OP_CONVERT_U_TO_F: UInt32 = 112
comptime OP_BITCAST: UInt32 = 124
comptime OP_ATOMIC_I_ADD: UInt32 = 234
comptime OP_I_ADD: UInt32 = 128
comptime OP_F_ADD: UInt32 = 129
comptime OP_I_SUB: UInt32 = 130
comptime OP_F_SUB: UInt32 = 131
comptime OP_I_MUL: UInt32 = 132
comptime OP_F_MUL: UInt32 = 133
comptime OP_F_DIV: UInt32 = 136
comptime OP_LOGICAL_OR: UInt32 = 166
comptime OP_LOGICAL_AND: UInt32 = 167
comptime OP_LOGICAL_NOT: UInt32 = 168
comptime OP_SELECT: UInt32 = 169
comptime OP_I_EQUAL: UInt32 = 170
comptime OP_I_NOT_EQUAL: UInt32 = 171
comptime OP_U_GREATER_THAN_EQUAL: UInt32 = 174
comptime OP_U_LESS_THAN: UInt32 = 176
comptime OP_F_ORD_LESS_THAN: UInt32 = 184
comptime OP_F_ORD_GREATER_THAN: UInt32 = 186
comptime OP_F_ORD_GREATER_THAN_EQUAL: UInt32 = 190
comptime OP_SHIFT_RIGHT_LOGICAL: UInt32 = 194
comptime OP_SHIFT_RIGHT_ARITHMETIC: UInt32 = 195
comptime OP_SHIFT_LEFT_LOGICAL: UInt32 = 196
comptime OP_BITWISE_OR: UInt32 = 197
comptime OP_BITWISE_XOR: UInt32 = 198
comptime OP_BITWISE_AND: UInt32 = 199
comptime OP_LOOP_MERGE: UInt32 = 246
comptime OP_SELECTION_MERGE: UInt32 = 247
comptime OP_LABEL: UInt32 = 248
comptime OP_BRANCH: UInt32 = 249
comptime OP_BRANCH_CONDITIONAL: UInt32 = 250
comptime OP_RETURN: UInt32 = 253

comptime GLSL_FRACT: UInt32 = 10
comptime GLSL_INVERSE_SQRT: UInt32 = 32
comptime GLSL_U_MIN: UInt32 = 38
comptime GLSL_S_MIN: UInt32 = 39
comptime GLSL_U_MAX: UInt32 = 41
comptime GLSL_S_MAX: UInt32 = 42
comptime GLSL_FMA: UInt32 = 50
comptime GLSL_N_MIN: UInt32 = 79
comptime GLSL_N_MAX: UInt32 = 80
comptime GLSL_N_CLAMP: UInt32 = 81

comptime CAPABILITY_SHADER: UInt32 = 1
comptime ADDRESSING_LOGICAL: UInt32 = 0
comptime MEMORY_GLSL450: UInt32 = 1
comptime MODEL_GL_COMPUTE: UInt32 = 5
comptime MODE_LOCAL_SIZE: UInt32 = 17

comptime STORAGE_INPUT: UInt32 = 1
comptime STORAGE_FUNCTION: UInt32 = 7
comptime STORAGE_PUSH_CONSTANT: UInt32 = 9
comptime STORAGE_BUFFER: UInt32 = 12

comptime DECORATION_BLOCK: UInt32 = 2
comptime DECORATION_ARRAY_STRIDE: UInt32 = 6
comptime DECORATION_BUILT_IN: UInt32 = 11
comptime DECORATION_BINDING: UInt32 = 33
comptime DECORATION_DESCRIPTOR_SET: UInt32 = 34
comptime DECORATION_OFFSET: UInt32 = 35
comptime BUILT_IN_GLOBAL_INVOCATION_ID: UInt32 = 28


def string_words(text: String) -> List[UInt32]:
    """A nul-terminated UTF-8 string packed little-endian into words."""
    var bytes = text.as_bytes()
    var words = List[UInt32](length=(len(bytes) + 4) // 4, fill=0)
    for index in range(len(bytes)):
        words[index // 4] |= UInt32(bytes[index]) << UInt32((index % 4) * 8)
    return words^


def emit(mut target: List[UInt32], opcode: UInt32, *operands: UInt32):
    target.append(opcode | (UInt32(len(operands) + 1) << 16))
    for operand in operands:
        target.append(operand)


struct LoopLabels(Copyable, Movable):
    var header: UInt32
    var next: UInt32  # the continue target
    var merge: UInt32

    def __init__(out self, header: UInt32, next: UInt32, merge: UInt32):
        self.header = header
        self.next = next
        self.merge = merge


struct IfLabels(Copyable, Movable):
    var merge: UInt32
    var otherwise: UInt32

    def __init__(out self, merge: UInt32, otherwise: UInt32):
        self.merge = merge
        self.otherwise = otherwise


struct SpirvModule(Movable):
    """One compute entry point called main, under construction."""
    var bound: UInt32
    var decorations: List[UInt32]
    var globals: List[UInt32]
    var variables: List[UInt32]
    var code: List[UInt32]
    var constants: Dict[Int, UInt32]
    var loops: List[LoopLabels]
    var ifs: List[IfLabels]
    var type_void: UInt32
    var type_bool: UInt32
    var type_uint: UInt32
    var type_float: UInt32
    var type_uvec3: UInt32
    var type_function: UInt32
    var pointer_buffer_uint: UInt32
    var pointer_function_uint: UInt32
    var pointer_function_float: UInt32
    var pointer_push_uint: UInt32
    var pointer_input_uvec3: UInt32
    var buffer_pointer: UInt32
    var glsl: UInt32
    var main: UInt32
    var invocation: UInt32
    var push: UInt32
    var group_width: UInt32
    var group_height: UInt32

    def __init__(out self, group_width: Int, group_height: Int, push_words: Int):
        self.bound = 1
        self.decorations = List[UInt32]()
        self.globals = List[UInt32]()
        self.variables = List[UInt32]()
        self.code = List[UInt32]()
        self.constants = Dict[Int, UInt32]()
        self.loops = List[LoopLabels]()
        self.ifs = List[IfLabels]()
        self.type_void = 0
        self.type_bool = 0
        self.type_uint = 0
        self.type_float = 0
        self.type_uvec3 = 0
        self.type_function = 0
        self.pointer_buffer_uint = 0
        self.pointer_function_uint = 0
        self.pointer_function_float = 0
        self.pointer_push_uint = 0
        self.pointer_input_uvec3 = 0
        self.buffer_pointer = 0
        self.glsl = 0
        self.main = 0
        self.invocation = 0
        self.push = 0
        self.group_width = UInt32(group_width)
        self.group_height = UInt32(group_height)
        self.declare_types(push_words)
        var first = self.fresh()
        emit(self.code, OP_LABEL, first)

    def fresh(mut self) -> UInt32:
        var id = self.bound
        self.bound += 1
        return id

    def declare_types(mut self, push_words: Int):
        self.glsl = self.fresh()
        self.main = self.fresh()
        self.type_void = self.fresh()
        self.type_function = self.fresh()
        self.type_bool = self.fresh()
        self.type_uint = self.fresh()
        self.type_float = self.fresh()
        self.type_uvec3 = self.fresh()
        emit(self.globals, OP_TYPE_VOID, self.type_void)
        emit(self.globals, OP_TYPE_FUNCTION, self.type_function, self.type_void)
        emit(self.globals, OP_TYPE_BOOL, self.type_bool)
        emit(self.globals, OP_TYPE_INT, self.type_uint, 32, 0)
        emit(self.globals, OP_TYPE_FLOAT, self.type_float, 32)
        emit(self.globals, OP_TYPE_VECTOR, self.type_uvec3, self.type_uint, 3)

        # One buffer type for every binding: struct { uint words[]; }.
        var runtime = self.fresh()
        var block = self.fresh()
        self.buffer_pointer = self.fresh()
        self.pointer_buffer_uint = self.fresh()
        emit(self.globals, OP_TYPE_RUNTIME_ARRAY, runtime, self.type_uint)
        emit(self.globals, OP_TYPE_STRUCT, block, runtime)
        emit(self.globals, OP_TYPE_POINTER, self.buffer_pointer, STORAGE_BUFFER, block)
        emit(self.globals, OP_TYPE_POINTER, self.pointer_buffer_uint, STORAGE_BUFFER, self.type_uint)
        emit(self.decorations, OP_DECORATE, runtime, DECORATION_ARRAY_STRIDE, 4)
        emit(self.decorations, OP_DECORATE, block, DECORATION_BLOCK)
        emit(self.decorations, OP_MEMBER_DECORATE, block, 0, DECORATION_OFFSET, 0)

        self.pointer_function_uint = self.fresh()
        self.pointer_function_float = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, self.pointer_function_uint, STORAGE_FUNCTION, self.type_uint)
        emit(self.globals, OP_TYPE_POINTER, self.pointer_function_float, STORAGE_FUNCTION, self.type_float)

        self.pointer_input_uvec3 = self.fresh()
        self.invocation = self.fresh()
        emit(self.globals, OP_TYPE_POINTER, self.pointer_input_uvec3, STORAGE_INPUT, self.type_uvec3)
        emit(self.globals, OP_VARIABLE, self.pointer_input_uvec3, self.invocation, STORAGE_INPUT)
        emit(self.decorations, OP_DECORATE, self.invocation, DECORATION_BUILT_IN, BUILT_IN_GLOBAL_INVOCATION_ID)

        if push_words > 0:
            var array = self.fresh()
            var push_block = self.fresh()
            var push_pointer = self.fresh()
            self.pointer_push_uint = self.fresh()
            self.push = self.fresh()
            var length = self.uint(UInt32(push_words))
            emit(self.globals, OP_TYPE_ARRAY, array, self.type_uint, length)
            emit(self.globals, OP_TYPE_STRUCT, push_block, array)
            emit(self.globals, OP_TYPE_POINTER, push_pointer, STORAGE_PUSH_CONSTANT, push_block)
            emit(self.globals, OP_TYPE_POINTER, self.pointer_push_uint, STORAGE_PUSH_CONSTANT, self.type_uint)
            emit(self.globals, OP_VARIABLE, push_pointer, self.push, STORAGE_PUSH_CONSTANT)
            emit(self.decorations, OP_DECORATE, array, DECORATION_ARRAY_STRIDE, 4)
            emit(self.decorations, OP_DECORATE, push_block, DECORATION_BLOCK)
            emit(self.decorations, OP_MEMBER_DECORATE, push_block, 0, DECORATION_OFFSET, 0)

    # Constants. Both kinds share a dictionary, told apart by bit 32.

    def uint(mut self, value: UInt32) -> UInt32:
        var key = Int(value)
        var known = self.constants.get(key)
        if known:
            return known.value()
        var id = self.fresh()
        emit(self.globals, OP_CONSTANT, self.type_uint, id, value)
        self.constants[key] = id
        return id

    def float_bits(mut self, bits: UInt32) -> UInt32:
        var key = Int(bits) | (1 << 32)
        var known = self.constants.get(key)
        if known:
            return known.value()
        var id = self.fresh()
        emit(self.globals, OP_CONSTANT, self.type_float, id, bits)
        self.constants[key] = id
        return id

    def float(mut self, value: Float32) -> UInt32:
        var copy = value
        return self.float_bits(Pointer(to=copy).unsafe_bitcast[UInt32]()[unsafe_offset=0])

    # Resources.

    def buffer(mut self, binding: Int) -> UInt32:
        var id = self.fresh()
        emit(self.globals, OP_VARIABLE, self.buffer_pointer, id, STORAGE_BUFFER)
        emit(self.decorations, OP_DECORATE, id, DECORATION_DESCRIPTOR_SET, 0)
        emit(self.decorations, OP_DECORATE, id, DECORATION_BINDING, UInt32(binding))
        return id

    def load_word(mut self, buffer: UInt32, index: UInt32) -> UInt32:
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, pointer)
        return id

    def load_float(mut self, buffer: UInt32, index: UInt32) -> UInt32:
        return self.bitcast_float(self.load_word(buffer, index))

    def store_word(mut self, buffer: UInt32, index: UInt32, value: UInt32):
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        emit(self.code, OP_STORE, pointer, value)

    def atomic_add_word(mut self, buffer: UInt32, index: UInt32, value: UInt32) -> UInt32:
        """Add to a word at device scope with relaxed ordering; returns the old value."""
        var pointer = self.fresh()
        var zero = self.uint(0)
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_buffer_uint, pointer, buffer, zero, index)
        var device_scope = self.uint(1)
        var relaxed = self.uint(0)
        var id = self.fresh()
        emit(self.code, OP_ATOMIC_I_ADD, self.type_uint, id, pointer, device_scope, relaxed, value)
        return id

    def push_word(mut self, index: Int) -> UInt32:
        var pointer = self.fresh()
        var zero = self.uint(0)
        var word = self.uint(UInt32(index))
        emit(self.code, OP_ACCESS_CHAIN, self.pointer_push_uint, pointer, self.push, zero, word)
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, pointer)
        return id

    def push_float(mut self, index: Int) -> UInt32:
        return self.bitcast_float(self.push_word(index))

    def invocation_axis(mut self, axis: Int) -> UInt32:
        var vector = self.fresh()
        emit(self.code, OP_LOAD, self.type_uvec3, vector, self.invocation)
        var id = self.fresh()
        emit(self.code, 81, self.type_uint, id, vector, UInt32(axis))  # OpCompositeExtract
        return id

    # Mutable values.

    def uvar(mut self, initial: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.variables, OP_VARIABLE, self.pointer_function_uint, id, STORAGE_FUNCTION)
        self.store(id, initial)
        return id

    def fvar(mut self, initial: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.variables, OP_VARIABLE, self.pointer_function_float, id, STORAGE_FUNCTION)
        self.store(id, initial)
        return id

    def load_u(mut self, variable: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_uint, id, variable)
        return id

    def load_f(mut self, variable: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_LOAD, self.type_float, id, variable)
        return id

    def store(mut self, variable: UInt32, value: UInt32):
        emit(self.code, OP_STORE, variable, value)

    # Arithmetic.

    def binary(mut self, opcode: UInt32, type: UInt32, a: UInt32, b: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, opcode, type, id, a, b)
        return id

    def unary(mut self, opcode: UInt32, type: UInt32, a: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, opcode, type, id, a)
        return id

    def ext1(mut self, operation: UInt32, type: UInt32, a: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a)
        return id

    def ext2(mut self, operation: UInt32, type: UInt32, a: UInt32, b: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a, b)
        return id

    def ext3(mut self, operation: UInt32, type: UInt32, a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_EXT_INST, type, id, self.glsl, operation, a, b, c)
        return id

    def iadd(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_ADD, self.type_uint, a, b)

    def isub(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_SUB, self.type_uint, a, b)

    def imul(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_MUL, self.type_uint, a, b)

    def bit_and(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_AND, self.type_uint, a, b)

    def bit_or(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_OR, self.type_uint, a, b)

    def bit_xor(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_BITWISE_XOR, self.type_uint, a, b)

    def shl(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_LEFT_LOGICAL, self.type_uint, a, b)

    def shr(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_RIGHT_LOGICAL, self.type_uint, a, b)

    def sar(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_SHIFT_RIGHT_ARITHMETIC, self.type_uint, a, b)

    def umin(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_U_MIN, self.type_uint, a, b)

    def umax(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_U_MAX, self.type_uint, a, b)

    def smin(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_S_MIN, self.type_uint, a, b)

    def smax(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.ext2(GLSL_S_MAX, self.type_uint, a, b)

    def fadd(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ADD, self.type_float, a, b)

    def fsub(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_SUB, self.type_float, a, b)

    def fmul(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_MUL, self.type_float, a, b)

    def fdiv(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_DIV, self.type_float, a, b)

    def fma(mut self, a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        return self.ext3(GLSL_FMA, self.type_float, a, b, c)

    def fract(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_FRACT, self.type_float, a)

    def inverse_sqrt(mut self, a: UInt32) -> UInt32:
        return self.ext1(GLSL_INVERSE_SQRT, self.type_float, a)

    def nmax(mut self, a: UInt32, b: UInt32) -> UInt32:
        """Maximum that ignores a NaN operand, as v_max_num_f32 does."""
        return self.ext2(GLSL_N_MAX, self.type_float, a, b)

    def nmin(mut self, a: UInt32, b: UInt32) -> UInt32:
        """Minimum that ignores a NaN operand, as v_min_num_f32 does."""
        return self.ext2(GLSL_N_MIN, self.type_float, a, b)

    def nclamp(mut self, a: UInt32, low: UInt32, high: UInt32) -> UInt32:
        return self.ext3(GLSL_N_CLAMP, self.type_float, a, low, high)

    def float_from_uint(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_U_TO_F, self.type_float, a)

    def float_from_int(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_S_TO_F, self.type_float, a)

    def uint_from_float(mut self, a: UInt32) -> UInt32:
        """Truncates. The operand must already be in range: see `saturate_uint`."""
        return self.unary(OP_CONVERT_F_TO_U, self.type_uint, a)

    def int_from_float(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_CONVERT_F_TO_S, self.type_uint, a)

    def saturate_uint(mut self, a: UInt32) -> UInt32:
        """Like v_cvt_u32_f32: NaN and negatives give 0. A conversion out of range is
        undefined in SPIR-V, so the clamp is what makes this portable."""
        return self.uint_from_float(self.nclamp(a, self.float(0), self.float(4294967040.0)))

    def bitcast_float(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_BITCAST, self.type_float, a)

    def bitcast_uint(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_BITCAST, self.type_uint, a)

    def select_u(mut self, condition: UInt32, yes: UInt32, no: UInt32) -> UInt32:
        var id = self.fresh()
        emit(self.code, OP_SELECT, self.type_uint, id, condition, yes, no)
        return id

    # Comparisons give booleans.

    def ieq(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_EQUAL, self.type_bool, a, b)

    def ine(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_I_NOT_EQUAL, self.type_bool, a, b)

    def ult(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_U_LESS_THAN, self.type_bool, a, b)

    def uge(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_U_GREATER_THAN_EQUAL, self.type_bool, a, b)

    def flt(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_LESS_THAN, self.type_bool, a, b)

    def fgt(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_GREATER_THAN, self.type_bool, a, b)

    def fge(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_F_ORD_GREATER_THAN_EQUAL, self.type_bool, a, b)

    def logical_and(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_LOGICAL_AND, self.type_bool, a, b)

    def logical_or(mut self, a: UInt32, b: UInt32) -> UInt32:
        return self.binary(OP_LOGICAL_OR, self.type_bool, a, b)

    def logical_not(mut self, a: UInt32) -> UInt32:
        return self.unary(OP_LOGICAL_NOT, self.type_bool, a)

    # Structured control flow.

    def label(mut self, id: UInt32):
        emit(self.code, OP_LABEL, id)

    def begin_if(mut self, condition: UInt32):
        var yes = self.fresh()
        var otherwise = self.fresh()
        var merge = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, merge, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, yes, otherwise)
        self.label(yes)
        self.ifs.append(IfLabels(merge, otherwise))

    def begin_else(mut self):
        var top = self.ifs[len(self.ifs) - 1].copy()
        emit(self.code, OP_BRANCH, top.merge)
        self.label(top.otherwise)
        self.ifs[len(self.ifs) - 1].otherwise = 0

    def end_if(mut self):
        var top = self.ifs.pop()
        emit(self.code, OP_BRANCH, top.merge)
        if top.otherwise != 0:
            self.label(top.otherwise)
            emit(self.code, OP_BRANCH, top.merge)
        self.label(top.merge)

    def begin_loop(mut self):
        var header = self.fresh()
        var body = self.fresh()
        var next = self.fresh()
        var merge = self.fresh()
        emit(self.code, OP_BRANCH, header)
        self.label(header)
        emit(self.code, OP_LOOP_MERGE, merge, next, 0)
        emit(self.code, OP_BRANCH, body)
        self.label(body)
        self.loops.append(LoopLabels(header, next, merge))

    def leave_if(mut self, condition: UInt32, target: UInt32):
        var carry_on = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, carry_on, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, target, carry_on)
        self.label(carry_on)

    def break_if(mut self, condition: UInt32):
        self.leave_if(condition, self.loops[len(self.loops) - 1].merge)

    def continue_if(mut self, condition: UInt32):
        self.leave_if(condition, self.loops[len(self.loops) - 1].next)

    def end_loop(mut self):
        var top = self.loops.pop()
        emit(self.code, OP_BRANCH, top.next)
        self.label(top.next)
        emit(self.code, OP_BRANCH, top.header)
        self.label(top.merge)

    def return_if(mut self, condition: UInt32):
        var leave = self.fresh()
        var carry_on = self.fresh()
        emit(self.code, OP_SELECTION_MERGE, carry_on, 0)
        emit(self.code, OP_BRANCH_CONDITIONAL, condition, leave, carry_on)
        self.label(leave)
        emit(self.code, OP_RETURN)
        self.label(carry_on)

    def finish(mut self) -> List[UInt32]:
        """Close the entry point and lay the module out in the order SPIR-V requires."""
        emit(self.code, OP_RETURN)
        emit(self.code, OP_FUNCTION_END)
        var words = List[UInt32]()
        words.append(SPIRV_MAGIC)
        words.append(SPIRV_VERSION_1_3)
        words.append(0)
        words.append(self.bound)
        words.append(0)
        emit(words, OP_CAPABILITY, CAPABILITY_SHADER)
        var import_name = string_words("GLSL.std.450")
        words.append(OP_EXT_INST_IMPORT | (UInt32(len(import_name) + 2) << 16))
        words.append(self.glsl)
        words.extend(import_name^)
        emit(words, OP_MEMORY_MODEL, ADDRESSING_LOGICAL, MEMORY_GLSL450)
        var entry_name = string_words("main")
        # SPIR-V 1.3 lists only Input and Output variables on the entry point.
        words.append(OP_ENTRY_POINT | (UInt32(len(entry_name) + 4) << 16))
        words.append(MODEL_GL_COMPUTE)
        words.append(self.main)
        words.extend(entry_name^)
        words.append(self.invocation)
        emit(words, OP_EXECUTION_MODE, self.main, MODE_LOCAL_SIZE, self.group_width, self.group_height, 1)
        words.extend(self.decorations.copy())
        words.extend(self.globals.copy())
        emit(words, OP_FUNCTION, self.type_void, self.main, 0, self.type_function)
        # A function's variables sit at the top of its first block.
        words.append(self.code[0])
        words.append(self.code[1])
        words.extend(self.variables.copy())
        for index in range(2, len(self.code)):
            words.append(self.code[index])
        return words^

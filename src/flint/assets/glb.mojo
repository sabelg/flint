from flint.platform.os import read_bytes


struct GlbInfo(Copyable, ImplicitlyCopyable, Movable):
    var valid: Bool
    var animation_count: Int
    var image_count: Int
    var embedded_image_count: Int
    var skin_count: Int
    var mesh_count: Int
    var json_size: Int

    def __init__(out self):
        self.valid = False
        self.animation_count = 0
        self.image_count = 0
        self.embedded_image_count = 0
        self.skin_count = 0
        self.mesh_count = 0
        self.json_size = 0


def u32_le(data: List[UInt8], offset: Int) -> Int:
    if offset < 0 or offset + 4 > len(data):
        return -1
    return Int(data[offset]) | (Int(data[offset + 1]) << 8) | (Int(data[offset + 2]) << 16) | (Int(data[offset + 3]) << 24)


def is_space(value: UInt8) -> Bool:
    return value == 32 or value == 9 or value == 10 or value == 13


def skip_string(data: List[UInt8], start: Int, end: Int) -> Int:
    var index = start + 1
    var escaped = False
    while index < end:
        var value = data[index]
        if escaped:
            escaped = False
        elif value == 92:
            escaped = True
        elif value == 34:
            return index + 1
        index += 1
    return end


def string_equals(data: List[UInt8], start: Int, end: Int, expected: String) -> Bool:
    if end <= start + 1:
        return False
    var length = end - start - 2
    if length != expected.byte_length():
        return False
    for index in range(length):
        if data[start + 1 + index] != expected.as_bytes()[index]:
            return False
    return True


# Find a root-level glTF JSON key and return the start of its value.
def root_value(data: List[UInt8], start: Int, end: Int, key: String) -> Int:
    var index = start
    var curly = 0
    var square = 0
    while index < end:
        var value = data[index]
        if value == 34:
            var string_end = skip_string(data, index, end)
            if curly == 1 and square == 0:
                var cursor = string_end
                while cursor < end and is_space(data[cursor]):
                    cursor += 1
                if cursor < end and data[cursor] == 58 and string_equals(data, index, string_end, key):
                    cursor += 1
                    while cursor < end and is_space(data[cursor]):
                        cursor += 1
                    return cursor
            index = string_end
            continue
        if value == 123:
            curly += 1
        elif value == 125:
            curly -= 1
        elif value == 91:
            square += 1
        elif value == 93:
            square -= 1
        index += 1
    return -1


def array_bounds(data: List[UInt8], start: Int, end: Int) -> List[Int]:
    var result = List[Int]()
    if start < 0 or start >= end or data[start] != 91:
        return result^
    var square = 0
    var curly = 0
    var index = start
    while index < end:
        var value = data[index]
        if value == 34:
            index = skip_string(data, index, end)
            continue
        if value == 91:
            square += 1
        elif value == 93:
            square -= 1
            if square == 0 and curly == 0:
                result.append(start)
                result.append(index + 1)
                return result^
        elif value == 123:
            curly += 1
        elif value == 125:
            curly -= 1
        index += 1
    return result^


def object_count(data: List[UInt8], bounds: List[Int]) -> Int:
    if len(bounds) != 2:
        return 0
    var count = 0
    var square = 0
    var curly = 0
    var index = bounds[0]
    while index < bounds[1]:
        var value = data[index]
        if value == 34:
            index = skip_string(data, index, bounds[1])
            continue
        if value == 91:
            square += 1
        elif value == 93:
            square -= 1
        elif value == 123:
            if square == 1 and curly == 0:
                count += 1
            curly += 1
        elif value == 125:
            curly -= 1
        index += 1
    return count


def token_count(data: List[UInt8], bounds: List[Int], token: String) -> Int:
    if len(bounds) != 2:
        return 0
    var count = 0
    var index = bounds[0]
    while index < bounds[1]:
        if data[index] == 34:
            var string_end = skip_string(data, index, bounds[1])
            if string_equals(data, index, string_end, token):
                count += 1
            index = string_end
        else:
            index += 1
    return count


def inspect_glb(path: String) -> GlbInfo:
    var info = GlbInfo()
    var data = read_bytes(path)
    if len(data) < 28:
        return info^
    if data[0] != 103 or data[1] != 108 or data[2] != 84 or data[3] != 70:
        return info^
    if u32_le(data, 4) != 2 or u32_le(data, 8) != len(data):
        return info^
    var json_length = u32_le(data, 12)
    if json_length <= 0 or u32_le(data, 16) != 0x4E4F534A or 20 + json_length > len(data):
        return info^
    var json_end = 20 + json_length
    info.json_size = json_length
    var animations = array_bounds(data, root_value(data, 20, json_end, "animations"), json_end)
    var images = array_bounds(data, root_value(data, 20, json_end, "images"), json_end)
    var skins = array_bounds(data, root_value(data, 20, json_end, "skins"), json_end)
    var meshes = array_bounds(data, root_value(data, 20, json_end, "meshes"), json_end)
    info.animation_count = object_count(data, animations)
    info.image_count = object_count(data, images)
    info.embedded_image_count = token_count(data, images, "bufferView")
    info.skin_count = object_count(data, skins)
    info.mesh_count = object_count(data, meshes)
    # A shared clip file has no meshes.
    info.valid = (info.mesh_count > 0 or info.animation_count > 0) and info.embedded_image_count == info.image_count
    return info^


def select_lod(distance: Float32, current: Int, quality: Int = 1) -> Int:
    var first: Float32 = 16
    var second: Float32 = 34
    if quality <= 0:
        first = 10
        second = 22
    elif quality >= 2:
        first = 22
        second = 48
    var hysteresis: Float32 = 4
    var level = max(0, min(2, current))
    if level == 0:
        if distance > second:
            return 2
        if distance > first:
            return 1
    elif level == 1:
        if distance > second:
            return 2
        if distance < first - hysteresis:
            return 0
    else:
        if distance < second - hysteresis:
            return 1
    return level


struct AnimationPlayback(Copyable, ImplicitlyCopyable, Movable):
    var clip: Int
    var time: Float32

    def __init__(out self):
        self.clip = -1
        self.time = 0

    def update(mut self, clip: Int, dt: Float32, frame_count: Int, loop: Bool = True) -> Int:
        if frame_count <= 0:
            return 0
        if clip != self.clip:
            self.clip = clip
            self.time = 0
        else:
            self.time += dt
        var frame = Int(self.time * 30)
        if loop:
            return frame % frame_count
        return min(frame_count - 1, frame)

    def advance(mut self, clip: Int, dt: Float32, duration: Float32, loop: Bool = True) -> Float32:
        """Continuous clip time in seconds; no snapping to 30 fps frames."""
        if clip != self.clip:
            self.clip = clip
            self.time = 0
        else:
            self.time += dt
        if duration <= 0:
            return 0
        if loop:
            while self.time >= duration:
                self.time -= duration
            return self.time
        return min(self.time, duration)

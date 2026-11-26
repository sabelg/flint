"""Small strict JSON DOM used by the native glTF runtime."""

from std.math import isfinite

comptime MAX_EXPONENT: Int = 400


struct JsonNode(Copyable, Movable):
    var kind: Int
    var text: String
    var number: Float64
    var boolean: Bool
    var children: List[Int]
    var keys: List[String]

    def __init__(out self, kind: Int = 0):
        self.kind = kind
        self.text = ""
        self.number = 0
        self.boolean = False
        self.children = List[Int]()
        self.keys = List[String]()


struct JsonDocument(Copyable, Movable):
    var nodes: List[JsonNode]
    var root: Int
    var valid: Bool
    var index: Int

    def __init__(out self):
        self.nodes = List[JsonNode]()
        self.root = -1
        self.valid = False
        self.index = 0

    def object_get(self, node: Int, key: String) -> Int:
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 5:
            return -1
        for index in range(len(self.nodes[node].keys)):
            if self.nodes[node].keys[index] == key:
                return self.nodes[node].children[index]
        return -1

    def array_size(self, node: Int) -> Int:
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 4:
            return 0
        return len(self.nodes[node].children)

    def at(self, node: Int, index: Int) -> Int:
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 4 or index < 0 or index >= len(self.nodes[node].children):
            return -1
        return self.nodes[node].children[index]

    def text(self, node: Int, fallback: String = "") -> String:
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 3:
            return fallback
        return self.nodes[node].text

    def number(self, node: Int, fallback: Float64 = 0) -> Float64:
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 2:
            return fallback
        return self.nodes[node].number

    def number_text(self, node: Int) -> String:
        """A number as written in the source, or empty."""
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 2:
            return String()
        return self.nodes[node].text

    def i64(self, node: Int, fallback: Int64 = 0) -> Int64:
        """Parses a signed 64-bit integer from the exact raw JSON number slice without Float64 precision loss."""
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 2:
            return fallback
        var raw = self.nodes[node].text.as_bytes()
        if len(raw) == 0:
            return fallback
        var neg = raw[0] == 45  # '-'
        var start = 1 if neg else 0
        var is_int_lit = start < len(raw)
        if is_int_lit:
            for i in range(start, len(raw)):
                if raw[i] < 48 or raw[i] > 57:
                    is_int_lit = False
                    break
        if is_int_lit:
            comptime I64_MAX_DIV10: UInt64 = UInt64(922337203685477580)
            var max_last = UInt64(8) if neg else UInt64(7)
            var acc = UInt64(0)
            for i in range(start, len(raw)):
                var d = UInt64(raw[i] - 48)
                if acc > I64_MAX_DIV10 or (acc == I64_MAX_DIV10 and d > max_last):
                    return fallback
                acc = acc * UInt64(10) + d
            if neg:
                if acc == (UInt64(1) << 63):
                    return Int64.MIN
                return -Int64(acc)
            return Int64(acc)
        var num = self.nodes[node].number
        if not isfinite(num) or num < -9223372036854774784.0 or num > 9223372036854774784.0:
            return fallback
        return Int64(num)

    def integer(self, node: Int, fallback: Int = -1) -> Int:
        return Int(self.i64(node, Int64(fallback)))

    def u64(self, node: Int, fallback: UInt64 = 0) -> UInt64:
        """Parses an unsigned 64-bit integer from the exact raw JSON number slice without Float64 precision loss."""
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 2:
            return fallback
        var raw = self.nodes[node].text.as_bytes()
        if len(raw) == 0:
            return fallback
        var all_digits = True
        for i in range(len(raw)):
            if raw[i] < 48 or raw[i] > 57:
                all_digits = False
                break
        if all_digits:
            var acc = UInt64(0)
            for i in range(len(raw)):
                var d = UInt64(raw[i] - 48)
                if acc > UInt64(1844674407370955161) or (acc == UInt64(1844674407370955161) and d > UInt64(5)):
                    return fallback
                acc = acc * UInt64(10) + d
            return acc
        var num = self.nodes[node].number
        if not isfinite(num) or num < 0.0 or num > 18446744073709551615.0:
            return fallback
        return UInt64(num)

    def boolean(self, node: Int, fallback: Bool = False) -> Bool:
        if node < 0 or node >= len(self.nodes) or self.nodes[node].kind != 1:
            return fallback
        return self.nodes[node].boolean

    def skip_space(mut self, data: List[UInt8]):
        while self.index < len(data) and (data[self.index] == 32 or data[self.index] == 9 or data[self.index] == 10 or data[self.index] == 13):
            self.index += 1

    def parse_string(mut self, data: List[UInt8]) -> String:
        var result = List[UInt8]()
        if self.index >= len(data) or data[self.index] != 34:
            self.valid = False
            return ""
        self.index += 1
        while self.index < len(data):
            var value = data[self.index]
            self.index += 1
            if value == 34:
                return String(unsafe_from_utf8=result)
            if value < 32:
                self.valid = False
                return ""
            if value != 92:
                result.append(value)
                continue
            if self.index >= len(data):
                break
            var escaped = data[self.index]
            self.index += 1
            if escaped == 34 or escaped == 92 or escaped == 47:
                result.append(escaped)
            elif escaped == 98:
                result.append(8)
            elif escaped == 102:
                result.append(12)
            elif escaped == 110:
                result.append(10)
            elif escaped == 114:
                result.append(13)
            elif escaped == 116:
                result.append(9)
            elif escaped == 117:
                if self.index + 4 > len(data):
                    self.valid = False
                    return ""
                var code = 0
                for _ in range(4):
                    var digit = data[self.index]
                    self.index += 1
                    code = code * 16
                    if digit >= 48 and digit <= 57:
                        code += Int(digit - 48)
                    elif digit >= 65 and digit <= 70:
                        code += Int(digit - 55)
                    elif digit >= 97 and digit <= 102:
                        code += Int(digit - 87)
                    else:
                        self.valid = False
                        return ""
                if code < 128:
                    result.append(UInt8(code))
                elif code < 2048:
                    result.append(UInt8(192 | (code >> 6)))
                    result.append(UInt8(128 | (code & 63)))
                else:
                    result.append(UInt8(224 | (code >> 12)))
                    result.append(UInt8(128 | ((code >> 6) & 63)))
                    result.append(UInt8(128 | (code & 63)))
            else:
                self.valid = False
                return ""
        self.valid = False
        return ""

    def parse_number(mut self, data: List[UInt8]) -> Int:
        var start = self.index
        if self.index < len(data) and data[self.index] == 45:
            self.index += 1
        var digits = 0
        while self.index < len(data) and data[self.index] >= 48 and data[self.index] <= 57:
            self.index += 1
            digits += 1
        if digits == 0:
            self.valid = False
            return -1
        if self.index < len(data) and data[self.index] == 46:
            self.index += 1
            var fraction_digits = 0
            while self.index < len(data) and data[self.index] >= 48 and data[self.index] <= 57:
                self.index += 1
                fraction_digits += 1
            if fraction_digits == 0:
                self.valid = False
                return -1
        if self.index < len(data) and (data[self.index] == 101 or data[self.index] == 69):
            self.index += 1
            if self.index < len(data) and (data[self.index] == 43 or data[self.index] == 45):
                self.index += 1
            var exponent_digits = 0
            while self.index < len(data) and data[self.index] >= 48 and data[self.index] <= 57:
                self.index += 1
                exponent_digits += 1
            if exponent_digits == 0:
                self.valid = False
                return -1
        var cursor = start
        var sign = 1.0
        if data[cursor] == 45:
            sign = -1
            cursor += 1
        var value = 0.0
        while cursor < len(data) and data[cursor] >= 48 and data[cursor] <= 57:
            value = value * 10 + Float64(data[cursor] - 48)
            cursor += 1
        if cursor < len(data) and data[cursor] == 46:
            cursor += 1
            var place = 0.1
            while cursor < len(data) and data[cursor] >= 48 and data[cursor] <= 57:
                value += Float64(data[cursor] - 48) * place
                place *= 0.1
                cursor += 1
        if cursor < len(data) and (data[cursor] == 101 or data[cursor] == 69):
            cursor += 1
            var exponent_sign = 1
            if cursor < len(data) and (data[cursor] == 43 or data[cursor] == 45):
                if data[cursor] == 45:
                    exponent_sign = -1
                cursor += 1
            var exponent = 0
            while cursor < len(data) and data[cursor] >= 48 and data[cursor] <= 57:
                # Past this a double is already infinity or zero.
                exponent = min(MAX_EXPONENT, exponent * 10 + Int(data[cursor] - 48))
                cursor += 1
            for _ in range(exponent):
                if exponent_sign > 0:
                    value *= 10
                else:
                    value *= 0.1
        var node = JsonNode(2)
        # The source text too, for readers that need the exact decimal
        # (gameplay/prefabs.mojo rounds it once, straight to Float32).
        node.text = String(unsafe_from_utf8=data[start:self.index])
        node.number = value * sign
        self.nodes.append(node^)
        return len(self.nodes) - 1

    def parse_value(mut self, data: List[UInt8], depth: Int = 0) -> Int:
        if depth > 128:
            self.valid = False
            return -1
        self.skip_space(data)
        if self.index >= len(data):
            self.valid = False
            return -1
        var value = data[self.index]
        if value == 34:
            var text = self.parse_string(data)
            var node = JsonNode(3)
            node.text = text
            self.nodes.append(node^)
            return len(self.nodes) - 1
        if value == 123:
            self.index += 1
            var node_index = len(self.nodes)
            self.nodes.append(JsonNode(5))
            self.skip_space(data)
            if self.index < len(data) and data[self.index] == 125:
                self.index += 1
                return node_index
            while self.valid and self.index < len(data):
                var key = self.parse_string(data)
                self.skip_space(data)
                if self.index >= len(data) or data[self.index] != 58:
                    self.valid = False
                    return -1
                self.index += 1
                var child = self.parse_value(data, depth + 1)
                if child < 0:
                    return -1
                self.nodes[node_index].keys.append(key)
                self.nodes[node_index].children.append(child)
                self.skip_space(data)
                if self.index < len(data) and data[self.index] == 44:
                    self.index += 1
                    self.skip_space(data)
                    continue
                if self.index < len(data) and data[self.index] == 125:
                    self.index += 1
                    return node_index
                self.valid = False
                return -1
            self.valid = False
            return -1
        if value == 91:
            self.index += 1
            var node_index = len(self.nodes)
            self.nodes.append(JsonNode(4))
            self.skip_space(data)
            if self.index < len(data) and data[self.index] == 93:
                self.index += 1
                return node_index
            while self.valid and self.index < len(data):
                var child = self.parse_value(data, depth + 1)
                if child < 0:
                    return -1
                self.nodes[node_index].children.append(child)
                self.skip_space(data)
                if self.index < len(data) and data[self.index] == 44:
                    self.index += 1
                    continue
                if self.index < len(data) and data[self.index] == 93:
                    self.index += 1
                    return node_index
                self.valid = False
                return -1
            self.valid = False
            return -1
        if value == 116 and self.index + 4 <= len(data):
            for offset in range(4):
                if data[self.index + offset] != "true".as_bytes()[offset]:
                    self.valid = False
            if self.valid:
                self.index += 4
                var node = JsonNode(1)
                node.boolean = True
                self.nodes.append(node^)
                return len(self.nodes) - 1
        if value == 102 and self.index + 5 <= len(data):
            for offset in range(5):
                if data[self.index + offset] != "false".as_bytes()[offset]:
                    self.valid = False
            if self.valid:
                self.index += 5
                self.nodes.append(JsonNode(1))
                return len(self.nodes) - 1
        if value == 110 and self.index + 4 <= len(data):
            for offset in range(4):
                if data[self.index + offset] != "null".as_bytes()[offset]:
                    self.valid = False
            if self.valid:
                self.index += 4
                self.nodes.append(JsonNode(0))
                return len(self.nodes) - 1
        if value == 45 or (value >= 48 and value <= 57):
            return self.parse_number(data)
        self.valid = False
        return -1


def parse_json(data: List[UInt8]) -> JsonDocument:
    var document = JsonDocument()
    document.valid = True
    document.root = document.parse_value(data)
    document.skip_space(data)
    if document.root < 0 or document.index != len(data):
        document.valid = False
    return document^


def json_u64(doc: JsonDocument, node: Int, fallback: UInt64 = 0) -> UInt64:
    return doc.u64(node, fallback)


def json_i64(doc: JsonDocument, node: Int, fallback: Int64 = 0) -> Int64:
    return doc.i64(node, fallback)

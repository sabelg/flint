"""Button glyphs for prompts and rich text (package 3.12).

One table maps every action the game prompts for to its keyboard key and
its controller button, as bound in src/gameplay/game.mojo (where it reads
keys and pad buttons for the player's moves). Rich text names an action
(`[btn=jump]`) and the glyph follows the last device used: a keycap with the
key's label for the keyboard, a round face button (A/B/X/Y in their Xbox
colours, or the PlayStation shapes), a pill for shoulders and triggers, a
cross for the d-pad and a ring for the sticks.

The drawing itself is in `SoftwareFramebuffer.draw_button_glyph`
(render/rasterizer.mojo); this module only knows names, labels and widths.
"""

comptime ACTION_NONE: Int = -1
comptime ACTION_JUMP: Int = 0
comptime ACTION_PUNCH: Int = 1
comptime ACTION_TEAR: Int = 2
comptime ACTION_THROW: Int = 3
comptime ACTION_SLAM: Int = 4
comptime ACTION_USE: Int = 5
# (sing, Echo's song, left with her in package 3.21; the ids after it moved down one)
comptime ACTION_ROAR: Int = 6
comptime ACTION_SNIFF: Int = 7
comptime ACTION_SKILLS: Int = 8
comptime ACTION_PAUSE: Int = 9
comptime ACTION_CONFIRM: Int = 10
comptime ACTION_BACK: Int = 11
comptime ACTION_FORM1: Int = 12
comptime ACTION_FORM2: Int = 13
comptime ACTION_FORM3: Int = 14
comptime ACTION_FORM4: Int = 15
comptime ACTION_MOVE: Int = 16
comptime ACTION_COUNT: Int = 17

# Controller buttons as drawn.
comptime BUTTON_A: Int = 0
comptime BUTTON_B: Int = 1
comptime BUTTON_X: Int = 2
comptime BUTTON_Y: Int = 3
comptime BUTTON_LB: Int = 4
comptime BUTTON_RB: Int = 5
comptime BUTTON_LT: Int = 6
comptime BUTTON_RT: Int = 7
comptime BUTTON_DPAD_UP: Int = 8
comptime BUTTON_DPAD_RIGHT: Int = 9
comptime BUTTON_DPAD_DOWN: Int = 10
comptime BUTTON_DPAD_LEFT: Int = 11
comptime BUTTON_LEFT_STICK: Int = 12
comptime BUTTON_LEFT_STICK_PRESS: Int = 13
comptime BUTTON_START: Int = 14
comptime BUTTON_BACK: Int = 15

# Shapes.
comptime SHAPE_KEYCAP: Int = 0
comptime SHAPE_FACE: Int = 1
comptime SHAPE_PILL: Int = 2
comptime SHAPE_DPAD: Int = 3
comptime SHAPE_STICK: Int = 4


def action_names() -> List[String]:
    return [
        String("jump"), String("punch"), String("tear"), String("throw"), String("slam"), String("use"),
        String("roar"), String("sniff"), String("skills"), String("pause"), String("confirm"),
        String("back"), String("form1"), String("form2"), String("form3"), String("form4"), String("move"),
    ]


def action_id(name: String) -> Int:
    """The action named in `[btn=NAME]`, or ACTION_NONE."""
    var names = action_names()
    for index in range(len(names)):
        if names[index] == name:
            return index
    return ACTION_NONE


def key_label(action: Int) -> String:
    """The keyboard key bound to an action, as printed on the keycap."""
    if action == ACTION_JUMP:
        return "Space"
    if action == ACTION_PUNCH:
        return "J"
    if action == ACTION_TEAR:
        return "F"
    if action == ACTION_THROW:
        return "G"
    if action == ACTION_SLAM:
        return "K"
    if action == ACTION_USE:
        return "E"
    if action == ACTION_ROAR:
        return "L"
    if action == ACTION_SNIFF:
        return "X"
    if action == ACTION_SKILLS:
        return "U"
    if action == ACTION_PAUSE or action == ACTION_BACK:
        return "Esc"
    if action == ACTION_CONFIRM:
        return "Enter"
    if action >= ACTION_FORM1 and action <= ACTION_FORM4:
        return String(action - ACTION_FORM1 + 1)
    if action == ACTION_MOVE:
        return "WASD"
    return "?"


def pad_button(action: Int) -> Int:
    """The controller button bound to an action in src/gameplay/game.mojo."""
    if action == ACTION_JUMP or action == ACTION_CONFIRM:
        return BUTTON_A
    if action == ACTION_BACK:
        return BUTTON_B
    if action == ACTION_PUNCH:
        return BUTTON_X
    if action == ACTION_USE:
        return BUTTON_Y
    if action == ACTION_TEAR:
        return BUTTON_LB
    if action == ACTION_THROW:
        return BUTTON_RB
    if action == ACTION_SLAM:
        return BUTTON_LT
    if action == ACTION_ROAR:
        return BUTTON_RT
    if action == ACTION_SNIFF:
        return BUTTON_LEFT_STICK_PRESS
    if action == ACTION_SKILLS:
        return BUTTON_BACK
    if action == ACTION_PAUSE:
        return BUTTON_START
    if action == ACTION_FORM1:
        return BUTTON_DPAD_UP
    if action == ACTION_FORM2:
        return BUTTON_DPAD_RIGHT
    if action == ACTION_FORM3:
        return BUTTON_DPAD_DOWN
    if action == ACTION_FORM4:
        return BUTTON_DPAD_LEFT
    return BUTTON_LEFT_STICK


def pad_shape(button: Int) -> Int:
    if button < 0:
        return SHAPE_STICK
    if button <= BUTTON_Y:
        return SHAPE_FACE
    if button <= BUTTON_RT or button == BUTTON_START or button == BUTTON_BACK:
        return SHAPE_PILL
    if button <= BUTTON_DPAD_LEFT:
        return SHAPE_DPAD
    return SHAPE_STICK


def pad_label(button: Int, sony: Bool) -> String:
    """Text printed on a controller button; empty where the shape says it
    (the PlayStation face buttons and the d-pad)."""
    if button < 0:
        return ""
    if button <= BUTTON_Y:
        if sony:
            return ""
        var letters = [String("A"), String("B"), String("X"), String("Y")]
        return letters[button]
    if button == BUTTON_LB:
        return "L1" if sony else "LB"
    if button == BUTTON_RB:
        return "R1" if sony else "RB"
    if button == BUTTON_LT:
        return "L2" if sony else "LT"
    if button == BUTTON_RT:
        return "R2" if sony else "RT"
    if button == BUTTON_START:
        return "Options" if sony else "Menu"
    if button == BUTTON_BACK:
        return "Create" if sony else "View"
    if button == BUTTON_LEFT_STICK_PRESS:
        return "L3" if sony else "LS"
    if button == BUTTON_LEFT_STICK:
        return "L"
    return ""


def face_color(button: Int, sony: Bool) -> UInt32:
    """Fill of a round face button (Xbox), or the symbol colour (PlayStation)."""
    if sony:
        if button == BUTTON_A:
            return 0x84AAFF   # cross
        if button == BUTTON_B:
            return 0xF06E6E   # circle
        if button == BUTTON_X:
            return 0xE79BD6   # square
        return 0x5ED3B5       # triangle
    if button == BUTTON_A:
        return 0x56B054
    if button == BUTTON_B:
        return 0xD64E46
    if button == BUTTON_X:
        return 0x3F7FD8
    return 0xE8B53A


def glyph_shape(action: Int, pad: Bool) -> Int:
    if not pad:
        return SHAPE_KEYCAP
    return pad_shape(pad_button(action))


def glyph_label(action: Int, pad: Bool, sony: Bool, keys: List[String] = List[String]()) -> String:
    """`keys` (package 3.15): the keycap labels of remapped keys by action,
    empty where the default stands (gameplay/bindings.mojo)."""
    if not pad:
        if action >= 0 and action < len(keys) and keys[action].byte_length() > 0:
            return keys[action]
        return key_label(action)
    return pad_label(pad_button(action), sony)


def glyph_width(shape: Int, label_width: Float32, height: Float32) -> Float32:
    """Width of a glyph `height` tall whose label measures `label_width` at
    the glyph's label size: keycaps and pills grow with their label (at least
    square), face buttons, sticks and the d-pad are as wide as they are tall."""
    if shape == SHAPE_KEYCAP or shape == SHAPE_PILL:
        return max(height, label_width + height * 0.6)
    return height

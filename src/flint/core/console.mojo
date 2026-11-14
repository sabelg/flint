"""The developer console: one line of text that reads and sets console variables.

The grave key opens it. While it is open `NativeRuntime.update_console` gives it
each key pressed and then mutes the input, so the simulation sees nothing held
and a recording made meanwhile stores that silence. Key codes are the engine's
fixed physical mapping, and variable names need no shift, so it has no notion of
layouts or capitals.

    debug.lights        flip a flag, or show a number
    debug.range 80      set a value
    list                every variable and its value
    list light          those with this in the name
    log                 every log category's level
    log gpu debug       one category's level (trace debug info warn error fatal off follow)
    crash segv          crash on purpose, to try the crash handler (also: crash abort)
    Tab                 finish a name, as far as it is unique
"""
from std.ffi import external_call
from flint.core.anchor import load
from flint.core.log import LOG_CATEGORIES, apply_log_settings, category_level, category_name, level_name
from flint.core.cvars import ConsoleVars, CVAR_COUNT, cvar_entry, parse_value
from flint.core.memory import memory_summary

comptime KEY_ESCAPE: Int = 256
comptime KEY_ENTER: Int = 257
comptime KEY_TAB: Int = 258
comptime KEY_BACKSPACE: Int = 259
comptime LOG_LINES: Int = 12
comptime LINE_BYTES: Int = 96


def typed_byte(key: Int) -> Int:
    """The ASCII byte this key types, or -1. Key codes are ASCII for these keys,
    with letters as capitals."""
    if key >= 65 and key <= 90:
        return key + 32
    if (key >= 48 and key <= 57) or key == 32 or key == 45 or key == 46:
        return key
    return -1


struct DevConsole(Copyable, Movable):
    var open: Bool
    var line: String
    var log: List[String]

    def __init__(out self):
        self.open = False
        self.line = String()
        self.log = List[String]()

    def say(mut self, text: String):
        self.log.append(text)
        if len(self.log) > LOG_LINES:
            _ = self.log.pop(0)

    def key(mut self, key: Int, mut cvars: ConsoleVars):
        """One key pressed while the console is open."""
        if key == KEY_ESCAPE:
            self.open = False
        elif key == KEY_ENTER:
            self.submit(cvars)
        elif key == KEY_TAB:
            self.complete()
        elif key == KEY_BACKSPACE:
            var length = self.line.byte_length()
            if length > 0:
                var kept = String(self.line[byte=0:length - 1])
                self.line = kept^
        else:
            var byte = typed_byte(key)
            if byte >= 0 and self.line.byte_length() < LINE_BYTES:
                self.line += chr(byte)

    def complete(mut self):
        """Extend the line to the longest name every match shares."""
        var prefix = self.line
        var shared = String()
        var matches = 0
        for index in range(CVAR_COUNT):
            var name = cvar_entry(index).name
            if not name.startswith(prefix):
                continue
            if matches == 0:
                shared = name
            else:
                var keep = 0
                var limit = min(shared.byte_length(), name.byte_length())
                while keep < limit and shared.as_bytes()[keep] == name.as_bytes()[keep]:
                    keep += 1
                var kept = String(shared[byte=0:keep])
                shared = kept^
            matches += 1
        if matches > 0:
            self.line = shared
        if matches == 1:
            self.line += " "

    def submit(mut self, mut cvars: ConsoleVars):
        var words = List[String]()
        for word in self.line.split(" "):
            if word.byte_length() > 0:
                words.append(String(word))
        var typed = self.line
        self.line = String()
        if len(words) == 0:
            return
        self.say("> " + typed)
        if words[0] == "list":
            self.list(cvars, words[1] if len(words) > 1 else String())
            return
        if words[0] == "log":
            if len(words) == 3:
                if apply_log_settings(words[1] + "=" + words[2]) == 0:
                    self.say("log takes a category or level or echo, then a level")
                    return
            var shown = String()
            for category in range(LOG_CATEGORIES):
                shown += category_name(category) + "=" + level_name(category_level(category)) + " "
            self.say(shown)
            return
        if words[0] == "mem":
            # The memory tags in one line each, largest first (core/memory.mojo).
            for line in memory_summary(6):
                self.say(line)
            return
        if words[0] == "crash":
            # For trying the crash handler in the real game (docs/PLAYTEST.md 0.9).
            if len(words) == 2 and words[1] == "segv":
                self.say("crashing: " + String(load(8)))
            elif len(words) == 2 and words[1] == "abort":
                external_call["abort", NoneType]()
            else:
                self.say("crash segv or crash abort: ends the game with a crash report")
            return
        var index = cvars.find(words[0])
        if index < 0:
            self.say("no variable " + words[0])
            return
        if len(words) > 1:
            var value = parse_value(words[1])
            if value is None:
                self.say("not a number: " + words[1])
                return
            cvars.set(index, value.value())
        elif cvar_entry(index).flag:
            cvars.set(index, Float32(0) if cvars.on(index) else Float32(1))
        self.say(cvars.describe(index))

    def list(mut self, cvars: ConsoleVars, part: String):
        var shown = 0
        for index in range(CVAR_COUNT):
            var entry = cvar_entry(index)
            if part in entry.name:
                self.say(cvars.describe(index) + "   " + entry.help)
                shown += 1
        if shown == 0:
            self.say("nothing matches " + part)

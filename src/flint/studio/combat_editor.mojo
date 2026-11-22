"""Combat timeline interaction in viewport pixels. Each marker has its own lane."""
from flint.anim.combat_timing import CombatTiming
from flint.platform.input import InputState


struct CombatTimeline(Movable):
    var left: Int
    var right: Int
    var top: Int
    var lane: Int
    var selected: Int
    var dragging: Int
    var changed: Bool

    def __init__(out self, left: Int, right: Int, top: Int, lane: Int):
        self.left = left
        self.right = right
        self.top = top
        self.lane = lane
        self.selected = 0
        self.dragging = -1
        self.changed = False

    def x(self, timing: CombatTiming, frame: Float32) -> Int:
        if timing.end <= 0 or self.right <= self.left:
            return self.left
        return self.left + Int(
            min(Float32(timing.end), max(Float32(0), frame))
            / Float32(timing.end)
            * Float32(self.right - self.left)
            + 0.5
        )

    def row(self, index: Int) -> Int:
        return self.top + self.lane * index

    def input(
        mut self, mut timing: CombatTiming, inp: InputState, x: Int, y: Int
    ) -> Int:
        """Return the frame to preview, or -1. Pointer capture lasts through release."""
        self.changed = False
        if inp.key_pressed(258):
            self.selected = (self.selected + 1) % 3
        var nudge = Int(inp.key_pressed(262)) - Int(inp.key_pressed(263))
        var target = -1
        if nudge:
            target = timing.marker(self.selected) + nudge
        if (
            self.lane > 0
            and self.right > self.left
            and inp.mouse_pressed(0)
            and x >= self.left - 6
            and x <= self.right + 6
            and y >= self.top - self.lane // 2
            and y < self.row(3) + self.lane // 2
        ):
            self.dragging = (y - self.top + self.lane // 2) // self.lane
            if self.dragging < 3:
                self.selected = self.dragging
        if not inp.mouse_down(0) and not inp.mouse_pressed(0):
            self.dragging = -1
        if self.dragging >= 0:
            if self.right <= self.left or timing.end <= 0:
                self.dragging = -1
            else:
                target = max(
                    0,
                    min(
                        timing.end,
                        Int(
                            Float32(x - self.left)
                            * Float32(timing.end)
                            / Float32(self.right - self.left)
                            + 0.5
                        ),
                    ),
                )
                if self.dragging == 3:
                    return target
        if nudge or self.dragging >= 0:
            var before = timing.marker(self.selected)
            timing.move_marker(self.selected, target)
            self.changed = timing.marker(self.selected) != before
            return timing.marker(self.selected)
        return -1

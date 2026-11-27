"""Pre-shaped event captions for the Sunspire bridge encounter."""
from std.math import sin
from flint.app import EngineApp, Resource
from flint.assets.server import AssetServer
from flint.audio.server import AudioServer
from flint.ecs import comp_hash64
from flint.render.font import SdfFont
from flint.render.render_world import ScalableRenderer
from flint.text.catalog import MessageCatalog
from flint.text.rich_text import RichShapedLayout, layout_rich_shaped, parse_rich
from flint.ui.context import UiContext
from flint.ui.draw_list import UiDrawList


struct BridgeCaptions(Resource, Movable):
    comptime RESOURCE_ID: UInt64 = comp_hash64("example.sunspire.BridgeCaptions")
    var font: SdfFont
    var slot: Int
    var messages: List[String]
    var layouts: List[RichShapedLayout]
    var layout_width: Int
    var size: Float32
    var enabled: Bool
    var line: Int
    var seconds: Float32
    var bell_id: Int

    def __init__(out self, mut render: ScalableRenderer, catalog: MessageCatalog, size: Float32, bell_id: Int, var font: SdfFont) raises:
        self.font = font^
        self.messages = List[String]()
        self.layouts = List[RichShapedLayout]()
        self.layout_width = 0
        self.size = size
        self.bell_id = bell_id
        self.enabled = True
        self.line = 0
        self.seconds = 4
        for key in ["arrival", "greeting", "fall", "recovery"]:
            var text = catalog.message(key)
            _ = self.font.shape_and_ensure(text, size)
            self.messages.append(text)
        self.slot = render.register_ui_font(self.font)
        if self.slot < 0:
            raise Error("Could not upload bridge caption font")

    def show(mut self, line: Int):
        self.line = line
        self.seconds = 3

    def draw(mut self, mut draw_list: UiDrawList, width: Int, height: Int):
        if not self.enabled or self.seconds <= 0 or width < 32:
            return
        if width != self.layout_width:
            self.layouts.clear()
            for message in self.messages:
                self.layouts.append(layout_rich_shaped(parse_rich(message), self.font, self.font, self.size, 0,
                                                      Float32(width - 24), 0, False, False))
            self.layout_width = width
        ref layout = self.layouts[self.line]
        var x = max(Float32(12), (Float32(width) - layout.width) * 0.5)
        var y = Float32(height) - layout.height - 18
        draw_list.add_rect(x - 8, y - 4, layout.width + 16, layout.height + 8, 0xDD181818, 4)
        for i in range(len(layout.runs)):
            ref run = layout.runs[i]
            UiContext._emit_shaped_glyphs(draw_list, self.slot, self.font, run.glyphs, x + run.x,
                                          y + run.baseline, run.px, 0xFFFFFFFF)


def setup_captions(mut app: EngineApp, locale: String, size: Float32) raises:
    ref assets = app.resources.get_ptr_mut[AssetServer]().value()[]
    var h = assets.load_raw("bridge_messages.json")
    var bytes = assets.raws.get_ptr(h)
    if not bytes:
        raise Error("Missing bridge message catalog")
    var catalog = MessageCatalog(bytes.value()[].bytes, locale)
    _ = assets.raws.release(h)
    var pcm = List[Int16](capacity=12000)
    for i in range(12000):
        var t = Float32(i) / 48000
        pcm.append(Int16(sin(5530*t) * (1 - 4*t) * 6200))
    var bell = app.resources.get_ptr_mut[AudioServer]().value()[].register_pcm16_asset(pcm)
    ref render = app.resources.get_ptr_mut[ScalableRenderer]().value()[]
    var font_handle = assets.load_font("fonts/DejaVuSans.ttf")
    var loaded = assets.fonts.get_ptr(font_handle)
    if not loaded or not loaded.value()[].font.has_source:
        raise Error("Missing bridge caption font outlines")
    var font = loaded.value()[].font.copy()
    _ = assets.fonts.release(font_handle)
    app.insert_resource(BridgeCaptions(render, catalog, size, bell, font^))

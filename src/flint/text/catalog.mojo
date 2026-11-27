"""Cookable UTF-8 message tables keyed by locale and message ID."""
from flint.assets.json import JsonDocument, parse_json


struct MessageCatalog(Copyable, Movable):
    var document: JsonDocument
    var selected: Int
    var language: Int
    var fallback: Int

    def __init__(out self, bytes: List[UInt8], locale: String, fallback_locale: String = "en") raises:
        self.document = parse_json(bytes)
        if not self.document.valid or self.document.nodes[self.document.root].kind != 5:
            raise Error("Invalid message catalog")
        self.fallback = self.document.object_get(self.document.root, fallback_locale)
        if self.fallback < 0 or self.document.nodes[self.fallback].kind != 5:
            raise Error("Missing fallback locale: " + fallback_locale)
        self.selected = self.document.object_get(self.document.root, locale)
        var dash = locale.find("-")
        self.language = self.document.object_get(self.document.root, String(locale[byte=0:dash])) if dash > 0 else -1

    def message(self, key: String) -> String:
        for table in [self.selected, self.language, self.fallback]:
            var item = self.document.object_get(table, key)
            if item >= 0 and self.document.nodes[item].kind == 3:
                return self.document.text(item)
        return key

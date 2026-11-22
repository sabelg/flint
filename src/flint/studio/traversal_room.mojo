"""Placement data shared across instances of a traversal encounter."""
from std.ffi import external_call
from flint.anim.choreography import _scalar, _vec, _vec_json
from flint.anim.procedural import Placement
from flint.assets.json import parse_json
from flint.assets.json_write import json_quote
from flint.core.math import Vec3
from flint.platform.os import (
    read_bytes,
    write_synced,
    durable_rename,
    last_errno,
    remove_file,
    parent_directory,
    fsync_path,
)


struct TraversalRoom(Copyable, Movable):
    var lip: Placement
    var landing: Vec3
    var takeoff: Vec3
    var defender: Vec3
    var motion: String
    var combat: String

    def __init__(out self, path: String) raises:
        var doc = parse_json(read_bytes(path))
        if (
            not doc.valid
            or doc.number(doc.object_get(doc.root, "version")) != 1
        ):
            raise Error("Expected a version 1 traversal room: " + path)
        self.lip = Placement(
            _vec(doc, "lip"), yaw=_scalar(doc, doc.object_get(doc.root, "yaw"))
        )
        self.takeoff = _vec(doc, "takeoff")
        self.landing = _vec(doc, "landing")
        self.defender = _vec(doc, "defender")
        self.motion = doc.text(doc.object_get(doc.root, "motion"))
        self.combat = doc.text(doc.object_get(doc.root, "combat"))
        if not self.motion or not self.combat or self.lip.position.y <= 0:
            raise Error(
                "Room requires a raised lip and motion and combat presets"
            )

    def ground_point(self, local: Vec3) -> Vec3:
        var p = self.lip.to_world(local)
        p.y = local.y
        return p

    def save(self, path: String) raises:
        var result = String('{\n  "version": 1,\n  "motion": ') + json_quote(
            self.motion
        )
        result += (
            ',\n  "combat": '
            + json_quote(self.combat)
            + ',\n  "lip": '
            + _vec_json(self.lip.position)
        )
        result += (
            ',\n  "yaw": '
            + String(self.lip.yaw)
            + ',\n  "landing": '
            + _vec_json(self.landing)
        )
        result += (
            ',\n  "takeoff": '
            + _vec_json(self.takeoff)
            + ',\n  "defender": '
            + _vec_json(self.defender)
            + "\n}\n"
        )
        var bytes = List[UInt8]()
        for byte in result.as_bytes():
            bytes.append(byte)
        var temporary = (
            path + ".tmp." + String(external_call["getpid", Int32]())
        )
        if not write_synced(temporary, bytes) or not durable_rename(
            temporary, path
        ):
            raise Error("Could not save traversal room: " + path)

    def save_copy(self, stem: String) raises -> String:
        """Publish the completed copy under the first unused sibling name."""
        var temporary = (
            stem + ".copy.tmp." + String(external_call["getpid", Int32]())
        )
        self.save(temporary)
        var index = 2
        while True:
            var dest = stem + "-" + String(index) + ".room.json"
            if (
                external_call["link", Int32](
                    temporary.as_c_string_span(), dest.as_c_string_span()
                )
                == 0
            ):
                _ = remove_file(temporary)
                if not fsync_path(parent_directory(dest), True):
                    raise Error("Could not sync room copy: " + dest)
                return dest
            if last_errno() != 17:  # EEXIST
                _ = remove_file(temporary)
                raise Error("Could not create room copy: " + dest)
            index += 1

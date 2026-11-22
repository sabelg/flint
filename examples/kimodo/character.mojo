"""Native skinned-character playback for the motion take viewport."""
from flint.anim.skinning import SKIN_LBS, skin_mesh
from flint.anim.soma_retarget import retarget_soma
from flint.assets.glb_native import load_native_glb
from flint.assets.image import write_png
from flint.assets.model import NativeModel
from flint.core.math import Vec3
from flint.ml.motionbricks import MbClip
from flint.render.simd_canvas import Camera3D, SimdCanvas


struct CharacterPreview(Movable):
    var model: NativeModel
    var colors: List[List[UInt32]]
    var clip: Int
    var hips: Int
    var height: Float32

    def __init__(out self):
        self.model = NativeModel()
        self.colors = []
        self.clip = -1
        self.hips = -1
        self.height = 1

    def load(mut self, path: String) raises:
        var model = load_native_glb(path)
        if not model.skins:
            raise Error("Expected a skinned GLB at " + path)
        self.model = model^
        self.clip = -1
        self.colors.clear()
        for mesh in self.model.meshes:
            var colors = List[UInt32]()
            for k in range(0, len(mesh.indices) - 2, 3):
                var color = mesh.base_color
                if (
                    mesh.texture >= 0
                    and mesh.texture < len(self.model.textures)
                    and len(mesh.texcoords) == mesh.vertex_count() * 2
                ):
                    ref tex = self.model.textures[mesh.texture]
                    if tex.width > 0 and tex.height > 0:
                        var u: Float32 = 0
                        var v: Float32 = 0
                        for i in range(3):
                            var j = mesh.indices[k + i] * 2
                            u += mesh.texcoords[j]
                            v += mesh.texcoords[j + 1]
                        var x = min(
                            tex.width - 1,
                            max(0, Int(u * Float32(tex.width) / 3)),
                        )
                        var y = min(
                            tex.height - 1,
                            max(0, Int(v * Float32(tex.height) / 3)),
                        )
                        color = tex.pixels[y * tex.width + x]
                colors.append(color)
            self.colors.append(colors^)

    def bake(
        mut self,
        take: MbClip,
        aliases: Dict[String, String] = Dict[String, String](),
        stage_origin: Bool = False,
    ) raises:
        var baked = retarget_soma(take, self.model, aliases, stage_origin)
        self.hips = baked.channels[0].node
        if self.clip < 0:
            self.clip = len(self.model.animations)
            self.model.animations.append(baked^)
        else:
            self.model.animations[self.clip] = baked^
        # Resting stature sets a fixed camera distance throughout crouches and jumps.
        var rest_top: Float32 = 0
        var rest_bottom: Float32 = 0
        for mesh in self.model.meshes:
            if mesh.skin >= 0:
                for k in range(1, len(mesh.positions), 3):
                    rest_top = max(rest_top, mesh.positions[k])
                    rest_bottom = min(rest_bottom, mesh.positions[k])
        self.height = max(Float32(0.1), rest_top - rest_bottom)

    def draw(mut self, mut canvas: SimdCanvas, time: Float32):
        self.model.sample_clip(self.clip, time)
        var hip = self.model.global_position(self.hips)
        var center = Vec3(hip.x, self.height * 0.49, hip.z)
        var cam = Camera3D(
            canvas.width,
            canvas.height,
            center + Vec3(0.65, 0.20, 1.85) * self.height,
            center,
            37,
        )
        self.draw_camera(canvas, cam)

    def draw_camera(mut self, mut canvas: SimdCanvas, cam: Camera3D):
        for k in range(-16, 17):
            var a = Float32(k) * self.height * 0.25
            canvas.draw_line_3d(
                cam,
                Vec3(a, 0, -4 * self.height),
                Vec3(a, 0, 4 * self.height),
                0x354651,
            )
            canvas.draw_line_3d(
                cam,
                Vec3(-4 * self.height, 0, a),
                Vec3(4 * self.height, 0, a),
                0x354651,
            )
        for m in range(len(self.model.meshes)):
            if self.model.meshes[m].skin < 0:
                continue
            var vertices = skin_mesh(self.model, m, SKIN_LBS)
            var pts = List[Vec3](capacity=len(vertices) // 8)
            for j in range(0, len(vertices), 8):
                pts.append(Vec3(vertices[j], vertices[j + 1], vertices[j + 2]))
            ref ix = self.model.meshes[m].indices
            for k in range(0, len(ix) - 2, 3):
                canvas.draw_tri_3d_lit(
                    cam,
                    pts[ix[k]],
                    pts[ix[k + 1]],
                    pts[ix[k + 2]],
                    self.colors[m][k // 3],
                )


def character_sheet(
    mut preview: CharacterPreview, clip: MbClip, path: String
) raises:
    var tile = SimdCanvas(400, 360)
    var strip = SimdCanvas(1200, 360)
    for i in range(3):
        tile.clear()
        var f = (clip.num_frames() - 1) * i // 2
        preview.draw(tile, Float32(f) / clip.sample_rate)
        tile.draw_text(12, 12, "Frame " + String(f + 1), 0xE8ECEE, 2)
        for y in range(360):
            for x in range(400):
                strip.pixels[y * 1200 + i * 400 + x] = tile.pixels[y * 400 + x]
    if not write_png(path, strip.pixels, 1200, 360):
        raise Error("Cannot write character preview " + path)

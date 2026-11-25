"""Model-level front end of Flint's geometry kit (`flint.assets.geometry_kit`).

`process_model` applies one kit operation to every triangle mesh of a `NativeModel` in place
(nodes, skins, materials and animations are kept) and returns a JSON report with before/after
numbers. Used by `examples/geometry_kit` and by Flint Studio's MCP tool `assets.process_mesh`.

Operations:
  - "stats":    topology report only.
  - "lod":      QEM decimation to `ratio` of the triangles (attributes + skin weights kept).
  - "meshlets": meshlet statistics (64 verts / 124 tris clusters with cones) - no change.
  - "unwrap":   automatic UV atlas (`resolution`, `padding`).
  - "remesh":   isotropic remesh to `target_vertices` (total over the model, split by area).

`hausdorff_relative` is the worst per-mesh symmetric Hausdorff distance over the model's bounding
diagonal; `hausdorff_worst_part_relative` is the same distance over that part's own diagonal.
"""
from std.math import max, min
from flint.core.math import Vec3
from flint.assets.model import NativeModel
from flint.assets.mesh_kit import hausdorff_distance, mesh_bounds, mesh_topology, surface_area
from flint.assets.meshlet import simplify_mesh_qem
from flint.assets.mesh_lod import build_meshlets, validate_meshlets
from flint.assets.uv_unwrap import UnwrapConfig, unwrap_mesh, uv_report
from flint.procedural.remesh import RemeshConfig, isotropic_remesh, pair_quads, triangle_quality


struct KitOptions(Copyable, Movable):
    var op: String
    var ratio: Float32
    var lock_boundaries: Bool
    var target_vertices: Int
    var iterations: Int
    var resolution: Int
    var padding: Int
    var measure: Bool        # Hausdorff after lod / remesh

    def __init__(out self, op: String = "stats"):
        self.op = op
        self.ratio = 0.5
        self.lock_boundaries = False
        self.target_vertices = 0
        self.iterations = 6
        self.resolution = 1024
        self.padding = 2
        self.measure = True


def _f(x: Float32) -> String:
    return String(x)


def process_model(mut model: NativeModel, opts: KitOptions) raises -> String:
    var op = opts.op
    if op != "stats" and op != "lod" and op != "meshlets" and op != "unwrap" and op != "remesh":
        raise Error("unknown geometry-kit op: " + op)
    var total_area: Float64 = 0
    var lo = Vec3(3.0e38, 3.0e38, 3.0e38)
    var hi = Vec3(-3.0e38, -3.0e38, -3.0e38)
    for m in range(len(model.meshes)):
        total_area += surface_area(model.meshes[m])
        if model.meshes[m].vertex_count() > 0:
            var b = mesh_bounds(model.meshes[m])
            lo = Vec3(min(lo.x, b[0].x), min(lo.y, b[0].y), min(lo.z, b[0].z))
            hi = Vec3(max(hi.x, b[1].x), max(hi.y, b[1].y), max(hi.z, b[1].z))
    var diag = max(Float32(1e-12), (hi - lo).length())
    var tris_in = 0
    var tris_out = 0
    var verts_in = 0
    var verts_out = 0
    var worst_h: Float32 = 0
    var worst_mesh_rel: Float32 = 0   # vs each mesh's own diagonal (tiny parts dominate)
    var watertight_in = 0
    var watertight_out = 0
    var meshes = 0
    var extra = String("")
    var meshlets = 0
    var meshlet_refs: Float64 = 0
    var all_valid = True
    var charts = 0
    var overlaps = 0
    var texel_overlaps = 0
    var worst_l2: Float32 = 0
    var quads = 0
    var quad_tris = 0
    var cv_in: Float64 = 0      # area-free averages over meshes, weighted by triangle count
    var cv_out: Float64 = 0
    var ang_in: Float64 = 0
    var ang_out: Float64 = 0
    var v6_in: Float64 = 0
    var v6_out: Float64 = 0
    for m in range(len(model.meshes)):
        if model.meshes[m].triangle_count() == 0:
            continue
        meshes += 1
        var src = model.meshes[m].copy()
        var t_in = mesh_topology(src)
        tris_in += t_in.triangles
        verts_in += t_in.vertices
        if t_in.watertight():
            watertight_in += 1
        if op == "lod":
            var target = max(8, Int(Float32(src.triangle_count()) * opts.ratio))
            var r = simplify_mesh_qem(src, target, opts.lock_boundaries)
            model.meshes[m] = r[0].copy()
        elif op == "remesh":
            var cfg = RemeshConfig()
            cfg.iterations = opts.iterations
            var share = surface_area(src) / max(total_area, 1e-30)
            cfg.target_vertices = max(16, Int(Float64(opts.target_vertices if opts.target_vertices > 0 else verts_in_model(model)) * share))
            model.meshes[m] = isotropic_remesh(src, cfg)
            var qa = triangle_quality(src)
            var qb = triangle_quality(model.meshes[m])
            var wa = Float64(src.triangle_count())
            var wb = Float64(model.meshes[m].triangle_count())
            cv_in += Float64(qa.edge_cv) * wa
            ang_in += Float64(qa.mean_min_angle_deg) * wa
            v6_in += Float64(qa.valence6_ratio) * wa
            cv_out += Float64(qb.edge_cv) * wb
            ang_out += Float64(qb.mean_min_angle_deg) * wb
            v6_out += Float64(qb.valence6_ratio) * wb
            var q = pair_quads(model.meshes[m])
            quads += len(q.quads) // 4
            quad_tris += len(q.triangles) // 3
        elif op == "unwrap":
            var cfg = UnwrapConfig()
            cfg.resolution = opts.resolution
            cfg.padding = opts.padding
            var u = unwrap_mesh(src, cfg)
            var rep = uv_report(u.mesh, u.chart_of_triangle, u.resolution)
            charts += rep.charts
            overlaps += rep.overlapping_pairs
            texel_overlaps += rep.texel_overlaps
            worst_l2 = max(worst_l2, rep.max_chart_l2)
            model.meshes[m] = u.mesh.copy()
        elif op == "meshlets":
            var ms = build_meshlets(src)
            meshlets += ms.count()
            meshlet_refs += Float64(len(ms.vertices))
            if not validate_meshlets(src, ms):
                all_valid = False
        var t_out = mesh_topology(model.meshes[m])
        tris_out += t_out.triangles
        verts_out += t_out.vertices
        if t_out.watertight():
            watertight_out += 1
        if opts.measure and (op == "lod" or op == "remesh"):
            var h = hausdorff_distance(src, model.meshes[m], 4000)
            worst_h = max(worst_h, h.max_error())
            worst_mesh_rel = max(worst_mesh_rel, h.relative())
    if op == "meshlets":
        extra = ',"meshlets":' + String(meshlets) + ',"vertex_refs_per_triangle":' + String(Float32(meshlet_refs / Float64(max(1, tris_in)))) + ',"valid":' + ("true" if all_valid else "false")
    elif op == "unwrap":
        extra = ',"charts":' + String(charts) + ',"uv_overlap_pairs":' + String(overlaps) + ',"texel_overlaps":' + String(texel_overlaps) + ',"max_chart_l2_stretch":' + _f(worst_l2)
    elif op == "remesh":
        var ti = Float64(max(1, tris_in))
        var to = Float64(max(1, tris_out))
        extra = (',"quads":' + String(quads) + ',"leftover_triangles":' + String(quad_tris)
                 + ',"edge_cv_in":' + _f(Float32(cv_in / ti)) + ',"edge_cv_out":' + _f(Float32(cv_out / to))
                 + ',"mean_min_angle_in":' + _f(Float32(ang_in / ti)) + ',"mean_min_angle_out":' + _f(Float32(ang_out / to))
                 + ',"valence6_in":' + _f(Float32(v6_in / ti)) + ',"valence6_out":' + _f(Float32(v6_out / to)))
    if opts.measure and (op == "lod" or op == "remesh"):
        extra += (',"hausdorff":' + _f(worst_h) + ',"hausdorff_relative":' + _f(worst_h / diag)
                  + ',"hausdorff_worst_part_relative":' + _f(worst_mesh_rel))
    return ('{"ok":true,"op":"' + op + '","meshes":' + String(meshes) + ',"triangles_in":' + String(tris_in)
            + ',"triangles_out":' + String(tris_out) + ',"vertices_in":' + String(verts_in) + ',"vertices_out":' + String(verts_out)
            + ',"watertight_in":' + String(watertight_in) + ',"watertight_out":' + String(watertight_out) + extra + "}")


def verts_in_model(model: NativeModel) -> Int:
    var n = 0
    for m in range(len(model.meshes)):
        n += model.meshes[m].vertex_count()
    return n

"""Batch-one sparse convolution and paired spatial sampling for Hi3D 512."""
from flint.core.jobs import JobPool
from flint.ml.ops import matmul_2d
from flint.ml.tensor import DT_F32, DT_I32, Tensor


struct SparseGrid(Movable):
    var xyz: Tensor
    var row: List[Int]
    var sparse_rows: Dict[Int, Int]
    var edge: Int

    def __init__(out self, var xyz: Tensor, edge: Int) raises:
        if (
            edge < 1
            or edge > 512
            or xyz.dtype != DT_I32
            or len(xyz.shape) != 2
            or xyz.shape[1] != 3
            or xyz.shape[0] < 1
            or len(xyz.i) != xyz.shape[0] * 3
        ):
            raise Error("Invalid sparse coordinate array")
        self.edge = edge
        self.sparse_rows = Dict[Int, Int]()
        self.row = List[Int](length=edge * edge * edge if edge <= 64 else 0, fill=-1)
        self.xyz = xyz^
        for i in range(self.xyz.shape[0]):
            var x = self.xyz.i[i * 3]
            var y = self.xyz.i[i * 3 + 1]
            var z = self.xyz.i[i * 3 + 2]
            if min(x, min(y, z)) < 0 or max(x, max(y, z)) >= edge:
                raise Error("Coordinate outside Hi3D grid")
            var k = (x * edge + y) * edge + z
            if self.get(x, y, z) >= 0:
                raise Error("Duplicate Hi3D coordinate")
            if edge <= 64:
                self.row[k] = i
            else:
                self.sparse_rows[k] = i

    def get(self, x: Int, y: Int, z: Int) -> Int:
        if min(x, min(y, z)) < 0 or max(x, max(y, z)) >= self.edge:
            return -1
        var key = (x * self.edge + y) * self.edge + z
        return self.row[key] if self.edge <= 64 else self.sparse_rows.get(key, -1)


struct SparseParents(Movable):
    var grid: SparseGrid
    var parent: List[Int]
    var divisor: List[Int]

    def __init__(out self, fine: SparseGrid) raises:
        var edge = (fine.edge + 1) // 2
        var counts = List[Int](length=edge * edge * edge, fill=0)
        var keys = List[Int]()
        for i in range(fine.xyz.shape[0]):
            var x = fine.xyz.i[3 * i] // 2
            var y = fine.xyz.i[3 * i + 1] // 2
            var z = fine.xyz.i[3 * i + 2] // 2
            var k = (x * edge + y) * edge + z
            keys.append(k)
            counts[k] += 1
        var coords = List[Int]()
        self.divisor = List[Int]()
        for k in range(len(counts)):
            if counts[k] > 0:
                # torch.scatter_reduce(mean) includes the initial zero row.
                self.divisor.append(counts[k] + 1)
                counts[k] = len(self.divisor) - 1
                coords.append(k // (edge * edge))
                coords.append((k // edge) % edge)
                coords.append(k % edge)
        self.parent = List[Int](capacity=len(keys))
        for k in keys:
            self.parent.append(counts[k])
        self.grid = SparseGrid(Tensor.ints(DT_I32, [len(self.divisor), 3], coords^), edge)

    def down(self, x: Tensor) -> Tensor:
        var cols = x.shape[1]
        var out = Tensor.zeros([len(self.divisor), cols])
        for i in range(len(self.parent)):
            for j in range(cols):
                out.f[self.parent[i] * cols + j] += x.f[i * cols + j]
        for i in range(len(self.divisor)):
            for j in range(cols):
                out.f[i * cols + j] = Float32(Float16(out.f[i * cols + j] / Float32(self.divisor[i])))
        return out^

    def up(self, x: Tensor) -> Tensor:
        var cols = x.shape[1]
        var out = Tensor.zeros([len(self.parent), cols])
        for i in range(len(self.parent)):
            for j in range(cols):
                out.f[i * cols + j] = x.f[self.parent[i] * cols + j]
        return out^


def sparse_conv3(mut pool: JobPool, grid: SparseGrid, x: Tensor, w: Tensor, bias: Tensor) raises -> Tensor:
    """Cross-correlate occupied rows using the checkpoint's [out,x,y,z,in] kernel."""
    var n = grid.xyz.shape[0]
    if len(x.shape) != 2 or x.shape[0] != n or len(w.shape) != 5 or len(bias.shape) != 1:
        raise Error("Invalid sparse convolution rank or row count")
    var ci = x.shape[1]
    var co = bias.shape[0]
    if (
        w.shape[0] != co
        or w.shape[1] != 3
        or w.shape[2] != 3
        or w.shape[3] != 3
        or w.shape[4] != ci
        or ci <= 0
        or co <= 0
    ):
        raise Error("Invalid sparse convolution weights")
    if (
        x.dtype != DT_F32
        or w.dtype != DT_F32
        or bias.dtype != DT_F32
        or len(x.f) != x.numel()
        or len(w.f) != w.numel()
        or len(bias.f) != co
    ):
        raise Error("Invalid sparse convolution tensor storage")
    var out = Tensor.zeros([n, co])
    var kernel = Tensor.zeros([co, ci])
    var batch = min(n, 2048)
    var gathered = Tensor.zeros([batch, ci])
    var values = Tensor.zeros([batch, co])
    var rows = List[Int](length=batch, fill=0)
    for offset in range(27):
        for o in range(co):
            for j in range(ci):
                kernel.f[o * ci + j] = w.f[(o * 27 + offset) * ci + j]
        var dx = offset // 9 - 1
        var dy = (offset // 3) % 3 - 1
        var dz = offset % 3 - 1
        for first in range(0, n, batch):
            var hits = 0
            for i in range(first, min(n, first + batch)):
                var k = grid.get(
                    grid.xyz.i[3 * i] + dx, grid.xyz.i[3 * i + 1] + dy, grid.xyz.i[3 * i + 2] + dz
                )
                if k >= 0:
                    rows[hits] = i
                    for j in range(ci):
                        gathered.f[hits * ci + j] = x.f[k * ci + j]
                    hits += 1
            if hits == 0:
                continue
            matmul_2d(pool, gathered.fptr(), kernel.fptr(), values.fptr(), 0, hits, ci, co, True)
            for k in range(hits):
                for o in range(co):
                    out.f[rows[k] * co + o] += values.f[k * co + o]
    for i in range(n):
        for o in range(co):
            out.f[i * co + o] += bias.f[o]
    return out^

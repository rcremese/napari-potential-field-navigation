import taichi as ti
import taichi.math as tm
import numpy as np


def convert_binary_map_to_obstacles(
    binary_map,
    origin=tm.vec3(0, 0, 0),
    scale=tm.vec3(1, 1, 1),
):
    obstacles = []
    indices = np.nonzero(binary_map)
    for i, j, k in zip(*indices):
        obstacles.append(
            Box3D(
                origin + tm.vec3(i, j, k) * scale,
                origin + tm.vec3(i + 1, j + 1, k + 1) * scale,
            )
        )
    return obstacles


@ti.dataclass
class Box2D:
    min: tm.vec2
    max: tm.vec2

    def width(self):
        return self.max[0] - self.min[0]

    def height(self):
        return self.max[1] - self.min[1]

    @ti.func
    def contains(self, pos: ti.math.vec2) -> bool:
        clamped_pos = tm.clamp(pos, self.min, self.max)
        return clamped_pos.x == pos.x and clamped_pos.y == pos.y


@ti.dataclass
class Box3D:
    min: tm.vec3
    max: tm.vec3

    def width(self):
        return self.max[0] - self.min[0]

    def height(self):
        return self.max[1] - self.min[1]

    def depth(self):
        return self.max[2] - self.min[2]

    @ti.func
    def contains(self, pos: ti.math.vec3) -> bool:
        clamped_pos = tm.clamp(pos, self.min, self.max)
        return (
            clamped_pos.x == pos.x
            and clamped_pos.y == pos.y
            and clamped_pos.z == pos.z
        )

    @ti.func
    def collides_with(self, other) -> bool:
        return not (
            self.min[0] > other.max[0]
            or self.max[0] < other.min[0]
            or self.min[1] > other.max[1]
            or self.max[1] < other.min[1]
            or self.min[2] > other.max[2]
            or self.max[2] < other.min[2]
        )

import taichi as ti
import taichi.math as tm


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

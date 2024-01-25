from napari.qt.threading import thread_worker
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)
vector_field = np.mgrid[0:1:0.1, 0:1:0.1, 0:1:0.1]
print(vector_field.shape)
vector_field_ti = ti.Vector.field(
    3, dtype=ti.f32, shape=vector_field.shape[1:]
)

# np.moveaxis(vector_field, 0, -1)
vector_field_ti.from_numpy(vector_field)
print(vector_field_ti.ndim)
print(vector_field_ti[0, 5, 0])

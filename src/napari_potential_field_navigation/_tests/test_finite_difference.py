import napari_potential_field_navigation._finite_difference as fd
import numpy as np


def test_create_laplace_matrix_2D():
    """Write a test that checks the shape of the laplacian matrix and the number of non-zero elements."""
    nx = 10
    ny = 10
    dx = 1.0
    dy = 1.0
    laplace = fd.create_laplacian_matrix_2d(nx, ny, dx, dy)
    assert laplace.shape == (nx * ny, nx * ny)
    assert laplace.nnz == 460
    assert laplace[5, 5] == -2 / dx**2 - 2 / dy**2
    assert laplace[5, 6] == 1 / dx**2
    assert laplace[5, 4] == 1 / dx**2


def test_poisson_solve_2D():
    """Write a test that checks the solution of the Poisson equation."""
    nx = 10
    ny = 10
    dx = 1.0
    dy = 1.0
    f = np.zeros((nx, ny))
    f[5, 5] = 1.0
    u = fd.poisson_solve_2D(f, dx, dy)
    X, Y = np.mgrid[0:nx, 0:ny]
    dist = np.sqrt((X - 5) ** 2 + (Y - 5) ** 2)
    green_fct = 0.5 * np.log(dist) / np.pi
    assert np.allclose(u, green_fct)


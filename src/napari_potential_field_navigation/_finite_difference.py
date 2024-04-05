import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import Tuple
import numpy as np


def create_gradient_matrix_2d(nx, ny, dx, dy) -> Tuple[sp.lil_array]:
    # Create 1D finite difference matrices for x and y directions (centered differences)
    Dx = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(nx, nx), format="csc") / (
        2 * dx
    )
    Dy = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(ny, ny), format="csc") / (
        2 * dy
    )
    # Apply the boundary conditions
    Dx[0, 0] = -1 / dx
    Dx[0, 1] = 1 / dx
    Dx[-1, -1] = 1 / dx
    Dx[-1, -2] = -1 / dx

    Dy[0, 0] = -1 / dy
    Dy[0, 1] = 1 / dy
    Dy[-1, -1] = 1 / dy
    Dy[-1, -2] = -1 / dy

    # Kronecker product to obtain 2D gradient matrices
    Gx = sp.lil_array(sp.kron(sp.eye(ny), Dx))
    Gy = sp.lil_array(sp.kron(Dy, sp.eye(nx)))
    return Gx, Gy


def create_gradient_matrix_3d(nx, ny, nz, dx, dy, dz) -> Tuple[sp.lil_array]:
    # Create 1D finite difference matrices for x and y directions (centered differences)
    Dx = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(nx, nx), format="csc") / (
        2 * dx
    )
    Dy = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(ny, ny), format="csc") / (
        2 * dy
    )
    Dz = sp.diags([-1, 0, 1], [-1, 0, 1], shape=(nz, nz), format="csc") / (
        2 * dz
    )
    # Apply the boundary conditions
    Dx[0, 0] = -1 / dx
    Dx[0, 1] = 1 / dx
    Dx[-1, -1] = 1 / dx
    Dx[-1, -2] = -1 / dx

    Dy[0, 0] = -1 / dy
    Dy[0, 1] = 1 / dy
    Dy[-1, -1] = 1 / dy
    Dy[-1, -2] = -1 / dy

    Dz[0, 0] = -1 / dz
    Dz[0, 1] = 1 / dz
    Dz[-1, -1] = 1 / dz
    Dz[-1, -2] = -1 / dz

    # Kronecker product to obtain 2D gradient matrices
    Gx = sp.lil_array(sp.kron(sp.eye(ny * nz), Dx))
    Gy = sp.lil_array(sp.kron(sp.eye(nz), sp.kron(Dy, sp.eye(nx))))
    Gz = sp.lil_array(sp.kron(Dz, sp.eye(nx * ny)))
    return Gx, Gy, Gz


def create_div_matrix_2d(nx, ny, dx, dy) -> sp.lil_matrix:
    Gx, Gy = create_gradient_matrix_2d(nx, ny, dx, dy)
    return sp.hstack((Gx, Gy))


def create_div_matrix_3d(nx, ny, nz, dx, dy, dz) -> sp.lil_matrix:
    Gx, Gy, Gz = create_gradient_matrix_3d(nx, ny, nz, dx, dy, dz)
    return sp.hstack((Gx, Gy, Gz))


def create_laplacian_matrix_2d(
    nx: int, ny: int, dx: float, dy: float
) -> sp.lil_array:
    Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nx, nx)) / dx**2
    Dyy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(ny, ny)) / dy**2
    return sp.lil_array(sp.kronsum(Dyy, Dxx))


def create_laplacian_matrix_3d(
    nx: int, ny: int, nz: int, dx: float, dy: float, dz: float
) -> sp.lil_array:
    # Compute the 2D laplacian matrix
    laplace_2d = create_laplacian_matrix_2d(nx, ny, dx, dy)

    Dzz = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nz, nz)) / dz**2
    return sp.lil_array(sp.kronsum(Dzz, laplace_2d))


def poisson_solve_2D(rhs: np.ndarray, dx: float, dy: float) -> np.ndarray:
    nx, ny = rhs.shape
    laplace = create_laplacian_matrix_2d(nx, ny, dx, dy)
    rhs_vec = rhs.flatten()
    u_vec: np.ndarray = splinalg.spsolve(laplace, rhs_vec)
    return u_vec.reshape((nx, ny))


def poisson_solve_3D(
    rhs: np.ndarray, dx: float, dy: float, dz: float
) -> np.ndarray:
    nx, ny, nz = rhs.shape
    laplace = create_laplacian_matrix_3d(nx, ny, nz, dx, dy, dz)
    rhs_vec = rhs.flatten()
    laplace[rhs_vec > 0, :] = 0
    laplace[rhs_vec > 0, rhs_vec > 0] = 1
    u_vec: np.ndarray = splinalg.spsolve(laplace, rhs_vec)
    return u_vec.reshape((nx, ny, nz))

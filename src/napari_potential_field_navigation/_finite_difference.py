import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from typing import Tuple
import numpy as np
import tqdm


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
    return sp.kronsum(Dyy, Dxx, format="csr")


def create_laplacian_matrix_3d(
    nx: int, ny: int, nz: int, dx: float, dy: float, dz: float
) -> sp.lil_array:
    # Compute the 2D laplacian matrix
    laplace_2d = create_laplacian_matrix_2d(nx, ny, dx, dy)

    Dzz = sp.diags([1, -2, 1], [-1, 0, 1], shape=(nz, nz)) / dz**2
    return sp.kronsum(Dzz, laplace_2d, format="csr")


# Definition of functions to convert between indices and subscripts
def ind2sub_2D(array_shape, ind) -> Tuple[int]:
    rows = ind / array_shape[1]
    cols = ind % array_shape[1]
    return np.array([rows, cols], dtype="int")


def sub2ind_2D(array_shape, rows, cols) -> int:
    return int(rows * array_shape[1] + cols)


def ind2sub_3D(array_shape, ind) -> Tuple[int]:
    rows = ind / (array_shape[1] * array_shape[2])
    cols = (ind / array_shape[2]) % array_shape[1]
    slices = ind % array_shape[2]
    return np.array([rows, cols, slices], dtype="int")


def sub2ind_3D(array_shape, rows, cols, slices) -> int:
    return int(
        rows * array_shape[1] * array_shape[2] + cols * array_shape[2] + slices
    )


def poisson_solve_2D(rhs: np.ndarray, dx: float, dy: float) -> np.ndarray:
    nx, ny = rhs.shape
    laplace = create_laplacian_matrix_2d(nx, ny, dx, dy)
    rhs_vec = rhs.flatten()
    u_vec: np.ndarray = splinalg.spsolve(laplace, rhs_vec)
    return u_vec.reshape((nx, ny))


def create_poisson_system(
    boundary_conditions: np.ndarray, spacing: Tuple[float] = None
) -> Tuple[np.ndarray]:
    """Solve the poisson equation in 2D or 3D using the finite difference method.

    Args:
        boundary_conditions (np.ndarray): Right-hand side of the poisson equation defined as a 2D or 3D array.
        Non-zeros values in the array correspond to dirichlet boundary conditions
        spacing (Tuple[float]): Spacing of the grid in each dimension. If None, set to (1.0, 1.0).

    Returns:
        Tuple[np.ndarray]: Matrix A and vector b of the linear system Ax = b to solve
    """
    assert (
        boundary_conditions.ndim == 2 or boundary_conditions.ndim == 3
    ), "The input array must be 2D or 3D"
    ## Set spacing to one if not provided
    if spacing is None:
        spacing = boundary_conditions.ndim * (1.0,)
    assert boundary_conditions.ndim == len(
        spacing
    ), "The spacing must match the number of dimensions"
    ## Create the laplacian matrix and apply the boundary conditions for 2D and 3D
    laplace_matrix = splinalg.LaplacianNd(
        boundary_conditions.shape,
        boundary_conditions="dirichlet",
        dtype=np.float32,
    ).tosparse()
    if boundary_conditions.ndim == 2:
        # laplace_matrix = create_laplacian_matrix_2d(
        #     *boundary_conditions.shape, *spacing
        # )
        laplace_matrix, rhs = apply_laplace_dirichlet_2D(
            laplace_matrix, boundary_conditions
        )
    elif boundary_conditions.ndim == 3:
        # laplace_matrix = create_laplacian_matrix_3d(
        #     *boundary_conditions.shape, *spacing
        # )
        laplace_matrix, rhs = apply_laplace_dirichlet_3D(
            laplace_matrix, boundary_conditions
        )
    else:
        raise ValueError("The input array must be 2D or 3D")

    return laplace_matrix, rhs


def apply_laplace_dirichlet_2D(
    laplace_matrix: sp.spmatrix, boundary_conditions: np.ndarray
):
    assert laplace_matrix.ndim == 2, "The laplacian matrix must be 2D"
    assert boundary_conditions.ndim == 2, "The boundary conditions must be 2D"

    res = boundary_conditions.shape
    max_index = np.prod(res)
    assert laplace_matrix.shape == (
        max_index,
        max_index,
    ), "Shape mismatch between the finit difference matrix and the boundary_conditions map"

    rhs = np.zeros(max_index, dtype=np.float32)

    rows, cols, slices = np.nonzero(boundary_conditions)
    for i, j, k in tqdm.tqdm(zip(rows, cols, slices), total=len(rows)):
        index = sub2ind_3D(res, i, j, k)

        laplace_matrix[index, index] = 1
        rhs[index] = boundary_conditions[i, j]

        ind_o = sub2ind_2D(res, i + 1, j)
        ind_e = sub2ind_2D(res, i - 1, j)
        ind_n = sub2ind_2D(res, i, j + 1)
        ind_s = sub2ind_2D(res, i, j - 1)

        for neighbour_ind in [ind_o, ind_e, ind_n, ind_s]:
            # Set the row to the identity vector and apply the colums to rhs
            if neighbour_ind >= 0 and neighbour_ind < max_index:
                laplace_matrix[index, neighbour_ind] = 0
                rhs[neighbour_ind] -= (
                    laplace_matrix[neighbour_ind, index] * rhs[index]
                )
                laplace_matrix[neighbour_ind, index] = 0
    return laplace_matrix, rhs


def apply_laplace_dirichlet_3D(
    laplace_matrix: sp.spmatrix, boundary_conditions: np.ndarray
):
    """Apply the Dirichlet boundary conditions to the linear system.

    Args:
        A (sp.spmatrix): Sparse matrix representing the laplacian in the linear system Ax = b
        binary_map (np.ndarray): binary map of the domain, where 1 correspond to spaces to apply the boundary conditions
        rhs (np.ndarray): right-hand side of the linear system
        value (float, optional): Value of the Dirichlet boundary conditions. Defaults to 0.0.
    """
    res = boundary_conditions.shape
    max_index = np.prod(res)
    assert laplace_matrix.shape == (
        max_index,
        max_index,
    ), "Shape mismatch between the finit difference matrix and the boundary_conditions map"

    rhs = np.zeros(max_index, dtype=np.float32)

    rows, cols, slices = np.nonzero(boundary_conditions)
    for i, j, k in tqdm.tqdm(zip(rows, cols, slices), total=len(rows)):
        index = sub2ind_3D(res, i, j, k)

        laplace_matrix[index, index] = 1
        rhs[index] = boundary_conditions[i, j, k]

        ind_o = sub2ind_3D(res, i + 1, j, k)
        ind_e = sub2ind_3D(res, i - 1, j, k)
        ind_n = sub2ind_3D(res, i, j + 1, k)
        ind_s = sub2ind_3D(res, i, j - 1, k)
        ind_u = sub2ind_3D(res, i, j, k + 1)
        ind_d = sub2ind_3D(res, i, j, k - 1)

        for neighbour_ind in [ind_o, ind_e, ind_n, ind_s, ind_u, ind_d]:
            # Set the row to the identity vector and apply the colums to rhs
            if neighbour_ind >= 0 and neighbour_ind < max_index:
                laplace_matrix[index, neighbour_ind] = 0
                rhs[neighbour_ind] -= (
                    laplace_matrix[neighbour_ind, index] * rhs[index]
                )
                laplace_matrix[neighbour_ind, index] = 0
    return laplace_matrix, rhs

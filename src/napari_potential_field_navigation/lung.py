import itk
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import scipy.ndimage as ndi
import numpy as np
import napari
import time


import taichi as ti
from matplotlib.animation import FuncAnimation
from napari_potential_field_navigation.simulations import (
    FreeNavigationSimulation,
)
from napari_potential_field_navigation.geometries import Box2D
from napari_potential_field_navigation.fields import VectorField2D


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


def create_gradient_matrix_3d(nx, ny, nz, dx, dy, dz):
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
    Gx = sp.kron(sp.eye(ny * nz), Dx, format="csc")
    Gy = sp.kron(sp.eye(nz), sp.kron(Dy, sp.eye(nx)), format="csc")
    Gz = sp.kron(Dz, sp.eye(nx * ny), format="csc")
    return Gx, Gy, Gz


def main():
    label_path = (
        Path(__file__)
        .parents[2]
        .joinpath("sample_datas", "label.nii.gz")
        .resolve(strict=True)
    )
    itk_label = itk.imread(label_path)

    steps = itk_label["spacing"]
    label_view = itk.array_view_from_image(itk_label)
    label = label_view[ndi.find_objects(label_view)[0]]

    res = label.shape
    print(f"Res: {res}, Steps: {steps}")

    tic = time.perf_counter()
    laplace_mat = create_laplacian_matrix_3d(*res, *steps)
    # Gx, Gy, Gz = create_gradient_matrix_3d(*res, *steps)
    toc = time.perf_counter()

    print(f"Laplacian matrix creation: {toc - tic:.2f} seconds")

    frontier = ndi.binary_dilation(label, iterations=1) & ~label
    valid_idx = (label.flatten() > 0) | (frontier.flatten() > 0)
    tic = time.perf_counter()

    restricted_laplace_mat = laplace_mat[valid_idx, :][:, valid_idx]
    # restricted_Gx = Gx[valid_idx, :][:, valid_idx]
    # restricted_Gy = Gy[valid_idx, :][:, valid_idx]
    # restricted_Gz = Gz[valid_idx, :][:, valid_idx]
    print(restricted_laplace_mat.shape, restricted_laplace_mat.dtype)
    # print(restricted_Gx.shape, restricted_Gx.dtype)

    toc = time.perf_counter()
    print(f"Restricted laplacian matrix creation: {toc - tic:.2f} seconds")
    source = np.zeros_like(label)
    source[frontier.flatten()] = 100
    source[-1, :, :] = 1
    b = source.flatten()[valid_idx]
    print(b.sum())

    tic = time.perf_counter()
    x, info = splinalg.cg(restricted_laplace_mat, b, maxiter=1000)
    tac = time.perf_counter()
    print(f"CG solve: {tac - tic:.2f} seconds")
    print(f"CG info: {info}")

    solution = np.zeros_like(label).flatten()
    solution[valid_idx] = x
    solution = solution.reshape(res)

    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(label[-1, :, :], cmap="gray")
    ax[0].plot(53, 128, "ro")
    # ax[1].imshow(restricted_laplace_mat[:1000, :1000].todense())
    ax[1].imshow(frontier[-1, :, :], cmap="gray")
    ax[2].imshow(solution[-1, :, :], cmap="inferno")
    plt.show()

    viewer = napari.view_labels(
        np.array(label, dtype=int), name="label", scale=steps
    )
    viewer.add_image(
        solution, name="solution", colormap="inferno", scale=steps
    )
    napari.run()


def autodiff_animation(
    simulation: FreeNavigationSimulation,
    nb_iter: int = 100,
    lr: float = 0.1,
    clip: float = 0,
):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    # Set up title
    ax[0].set_title("Iteration : 0 - Time : 0")

    # Set the simulation
    simulation.reset()
    simulation.run()
    # Plot the initial vector field
    x_grid, y_grid = simulation.vector_field.meshgrid
    x_step, y_step = simulation.vector_field.step_sizes
    vect = simulation.vector_field.values
    quiver = ax[0].quiver(
        x_grid,
        y_grid,
        vect[:, :, 0],
        vect[:, :, 1],
        # angles="xy",
        # scale_units="xy",
        scale=20,
        color="black",
    )
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].set_title("Gradient update")

    quiver2 = ax[1].quiver(
        x_grid,
        y_grid,
        np.zeros_like(x_grid),
        np.zeros_like(y_grid),
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
    )
    # Plot the target at the target location
    tx, ty = simulation.target.to_numpy()
    ax[0].plot(tx, ty, "go", markersize=15, label="Target")
    ax[1].plot(tx, ty, "go", markersize=15, label="Target")
    # Plot the initial position
    x, y = simulation.positions[:, :, 0]
    (point,) = ax[0].plot(x, y, "ro", linestyle="--")
    fig.legend()

    def update(frame):
        step = frame % simulation.nb_steps
        if frame % simulation.nb_steps == 0 and frame != 0:
            old_vect = simulation.vector_field.values
            simulation._optimization_step(clip, lr=lr)
            new_vect = simulation.vector_field.values
            quiver.set_UVC(new_vect[:, :, 0], new_vect[:, :, 1])
            quiver2.set_UVC(
                new_vect[:, :, 0] - old_vect[:, :, 0],
                new_vect[:, :, 1] - old_vect[:, :, 1],
            )
        title = f"Iteration : {frame // simulation.nb_steps} - Time : {step * simulation._dt:.2f}"
        if frame // simulation.nb_steps > 0:
            title += f" - L2 loss : {simulation.loss.to_numpy():.3f}"
        ax[0].set_title(title)

        x = simulation.positions[:, :step, 0]
        y = simulation.positions[:, :step, 1]
        point.set_data(x, y)
        return quiver, point

    nb_frames = simulation.nb_steps * nb_iter
    return FuncAnimation(
        fig, update, frames=range(nb_frames), interval=50, blit=False
    )


def animate_autodiff_process():
    step = 10
    domain = Box2D([-1, -1], [1, 1])
    x, y = np.meshgrid(np.linspace(-1, 1, step), np.linspace(-1, 1, step))
    v = -np.stack([y, x], axis=-1)

    ti.init(arch=ti.gpu)
    vector_field = VectorField2D(v, domain)
    positions = np.array([[0.5, 0.5]])
    target = np.array([-0.65, 0.65])
    simulation = FreeNavigationSimulation(
        positions, target, vector_field, diffusivity=0.01, t_max=1, dt=0.05
    )

    anim = autodiff_animation(simulation, nb_iter=20, lr=1, clip=0)
    anim.save("one_autodiff.gif", fps=10, writer="ffmpeg")
    plt.show()


if __name__ == "__main__":
    main()

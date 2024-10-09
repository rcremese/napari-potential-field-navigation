import taichi as ti
import taichi.math as tm
import numpy as np

import napari_potential_field_navigation.geometries as geometries
from napari_potential_field_navigation.fields import (
    VectorField2D,
    VectorField3D,
    DistanceField,
    lerp,
)

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path

EPS = 1e-6


@ti.dataclass
class State2D:
    pos: tm.vec2
    vel: tm.vec2


@ti.dataclass
class State3D:
    pos: tm.vec3
    vel: tm.vec3


@ti.data_oriented
class NavigationSimulation(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self._init_pos: ti.Field = None
        self._positions: ti.Field = None
        self.target: ti.Vector = None
        self._noise: ti.Field = None
        self._nb_steps: int = None
        self._nb_walkers: int = None
        self._dim: int = None

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def collision_handling(self, n: int, t: int):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self):
        raise NotImplementedError

    @abstractmethod
    def optimize(self, max_iter: int = 1000, lr: float = 1e-3):
        raise NotImplementedError

    @abstractmethod
    def update_time(self, t_max: float, dt: float):
        raise NotImplementedError

    @abstractmethod
    def update_positions(self, positions: np.ndarray):
        raise NotImplementedError


class FreeNavigationSimulation(NavigationSimulation):
    def __init__(
        self,
        positions: np.ndarray,
        target: np.ndarray,
        vector_field: Union[VectorField2D, VectorField3D],
        t_max: float = 100.0,
        dt: float = 0.1,
        diffusivity: float = 1.0,
    ):
        ## Initialisation of the vector field
        if isinstance(vector_field, VectorField2D):
            self._dim = 2
        elif isinstance(vector_field, VectorField3D):
            self._dim = 3
        else:
            raise TypeError("Vector field must be either 2D or 3D")
        self.vector_field = vector_field

        ## Initialisation of the walkers positions
        self.update_positions(positions)

        ## Initialisation of the target point
        if target.shape != (self._dim,):
            raise ValueError(
                f"Expected target to be a {self._dim}D-array. Get {target.shape} array"
            )
        if self.dim == 2:
            self.target = tm.vec2(target)
        elif self.dim == 3:
            self.target = tm.vec3(target)

        ## Simulation specific parameters
        self.update_time(t_max, dt)

        if diffusivity < 0.0:
            raise ValueError(
                f"Expected diffusivity to be positive. Get {diffusivity}"
            )
        self.diffusivity = diffusivity

        # Definition of the loss
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)

    ## Public methods
    def update_positions(self, positions: np.ndarray):
        if self._dim == 2 and not (
            positions.ndim == 2 and positions.shape[1] == 2
        ):
            raise ValueError(
                f"Expected positions to be a (n, 2)-array of position vectors. Get {positions.shape}"
            )
        if self._dim == 3 and not (
            positions.ndim == 2 and positions.shape[1] == 3
        ):
            raise ValueError(
                f"Expected positions to be a (n, 3)-array of position vectors. Get {positions.shape}"
            )
        self._nb_walkers = positions.shape[0]
        self._init_pos: ti.Field = ti.Vector.field(
            self._dim, dtype=ti.f32, shape=(self._nb_walkers,)
        )
        self._init_pos.from_numpy(positions.astype(np.float32))

    def update_time(self, t_max: float, dt: float):
        if not (t_max > 0.0 and dt > 0.0):
            raise ValueError(
                f"Expected tmax and dt to be positive. Get {t_max}, {dt}"
            )
        if dt > t_max:
            raise ValueError(
                f"Expected dt to be smaller than tmax. Get {dt} > {t_max}"
            )
        self._dt = dt
        self._t_max = t_max
        self._nb_steps = np.ceil(t_max / dt).astype(int)

        ## Taichi field definition
        self._positions: ti.Field = ti.Vector.field(
            n=self._dim,
            dtype=ti.f32,
            shape=(self._nb_walkers, self._nb_steps),
            needs_grad=True,
        )
        self._noise: ti.Field = ti.Vector.field(
            self._dim,
            dtype=ti.float32,
            shape=(self._nb_walkers, self._nb_steps),
            needs_grad=False,
        )

    def reset(self):
        if self._dim == 2:
            self._reset_2d()
        elif self._dim == 3:
            self._reset_3d()
        else:
            raise ValueError("Simulation dimension must be either 2 or 3")

    def run(self):
        for t in range(1, self._nb_steps):
            self.step(t, self.diffusivity)

    def optimize(
        self,
        max_iter: int = 100,
        lr: float = 1e-3,
        clip_value: float = 0.0,
    ):
        assert (
            isinstance(max_iter, int) and max_iter > 0
        ), f"Expected max_iter to be a positive integer. Get {max_iter}"
        assert (
            isinstance(lr, (float, int)) and lr > 0.0
        ), f"Expected lr to be a positive float. Get {lr}"
        if clip_value > 0.0:
            assert (
                isinstance(clip_value, (float, int)) and clip_value > 0.0
            ), f"Expected clip_value to be a positive float. Get {clip_value}"
            clip_value = float(clip_value)

        for i in range(max_iter):
            self._optimization_step(clip_value, lr)
            print("Iter=", i, "Loss=", self.loss[None])

    def _optimization_step(self, clip_value: float, lr: float):
        self.reset()
        with ti.ad.Tape(self.loss):
            self.run()
            self.compute_loss(self._nb_steps - 1)
        print("Iter=", iter, "Loss=", self.loss[None])
        self._update_force_field(lr)
        if clip_value > 0.0:
            self.vector_field.norm_clip(clip_value)

    ### Taichi kernels
    @ti.kernel
    def _reset_2d(self):
        for n in ti.ndrange(self._nb_walkers):
            self._positions[n, 0] = self._init_pos[n]
            for t in ti.ndrange(self._nb_steps):
                self._noise[n, t] = tm.vec2(ti.randn(), ti.randn())
        for i, j in self.vector_field._values:
            self.vector_field._values.grad[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def _reset_3d(self):
        for n in ti.ndrange(self._nb_walkers):
            self._positions[n, 0] = self._init_pos[n]
            for t in ti.ndrange(self._nb_steps):
                self._noise[n, t] = tm.vec3(ti.randn(), ti.randn(), ti.randn())
        for i, j, k in self.vector_field._values:
            self.vector_field._values.grad[i, j, k] = tm.vec3(0.0, 0.0, 0.0)

    @ti.kernel
    def step(self, t: int, diffusivity: ti.f32):
        for n in ti.ndrange(self._nb_walkers):
            self._positions[n, t] = (
                self._positions[n, t - 1]
                + self._dt * self.vector_field.at(self._positions[n, t - 1])
                + tm.sqrt(2 * self._dim * self._dt * diffusivity)
                * self._noise[n, t]
            )
            self.collision_handling(n, t)

    @ti.func
    def collision_handling(self, n: int, t: int):
        """Check all collision for a walker n at time t and apply changes.

        Args:
            n (int): walker index
            t (int): timestep index
        """
        ## Domain collisions
        if not self.vector_field.contains(self._positions[n, t]):
            self._domain_collision(n, t)

    @ti.kernel
    def compute_loss(self, t: int):
        for n in range(self._nb_walkers):
            self.loss[None] += (
                tm.length(self._positions[n, t] - self.target) ** 2
            ) / self._nb_walkers

    @ti.kernel
    def clip_grad(self, clip_value: float):
        for I in ti.grouped(self.vector_field._values):
            if tm.length(self.vector_field._values.grad[I]) > clip_value:
                self.vector_field._values.grad[I] = (
                    clip_value
                    * self.vector_field._values.grad[I]
                    / tm.length(self.vector_field._values.grad[I])
                )

    ## Private methods
    @ti.func
    def _domain_collision(self, n: int, t: int):
        """Check the collision of a walker at time t with the domain boundaries.

        Args:
            n (int): index of the walker
            t (int): index of the time step
        """
        for i in ti.static(range(self._dim)):
            if self._positions[n, t][i] < self.vector_field._bounds.min[i]:
                self._positions[n, t][i] = self.vector_field._bounds.min[i]
            elif self._positions[n, t][i] > self.vector_field._bounds.max[i]:
                self._positions[n, t][i] = self.vector_field._bounds.max[i]

    @ti.kernel
    def _update_force_field(self, lr: float):
        for I in ti.grouped(self.vector_field._values):
            # remove nan that are not handled by the optimizer
            if tm.isnan(self.vector_field._values.grad[I]).sum() != 0:
                self.vector_field._values.grad[I] = 0.0
            self.vector_field._values[I] -= (
                lr * self.vector_field._values.grad[I]
            )

    ## Properties
    @property
    def positions(self) -> np.ndarray:
        """Positions of all the walkers at each time step.

        Returns:
            np.ndarray: array of dimensions (nb_walkers, nb_steps, spatial_dim)
        """
        return self._positions.to_numpy()

    @property
    def trajectories(self) -> np.ndarray:
        """Trajectories of all the walkers with added ids and time steps. Useful for napari.Tracks plotting

        Returns:
            np.ndarray: array of dimensions (nb_walkers * nb_steps, ID + time + spatial_dim)
        """
        positions = self.positions.reshape(
            (self.nb_walkers * self.nb_steps, self.dim)
        )
        ids = np.repeat(np.arange(self.nb_walkers), self.nb_steps)
        times = np.tile(np.arange(self.nb_steps), self.nb_walkers)
        return np.column_stack((ids, times, positions))

    @property
    def mean_trajectory(self) -> np.ndarray:
        """Mean trajectory of all the walkers with added ids and time steps. Useful for napari.Tracks plotting

        Returns:
            np.ndarray: array of dimensions (nb_steps, ID + time + spatial_dim)
        """
        mean_positions = np.mean(self.positions, axis=0)
        ids = np.zeros(self.nb_steps, dtype=int)
        times = np.arange(self.nb_steps, dtype=int)
        return np.column_stack((ids, times, mean_positions))

    @property
    def nb_walkers(self) -> int:
        return self._nb_walkers

    @property
    def nb_steps(self) -> int:
        return self._nb_steps

    @property
    def dim(self) -> int:
        return self._dim


@ti.func
def compute_cosine_similarity(v1: tm.vec3, v2: tm.vec3) -> ti.f32:
    return tm.dot(v1, v2) / (tm.length(v1) * tm.length(v2) + EPS)


@ti.func
def compute_angle_3D(v1: tm.vec3, v2: tm.vec3) -> ti.f32:
    cross = tm.cross(v1, v2)
    return tm.atan2(tm.length(cross), tm.dot(v1, v2) + EPS)


class DomainNavigationSimulation(FreeNavigationSimulation):
    def __init__(
        self,
        positions: np.ndarray,
        target: np.ndarray,
        vector_field: Union[VectorField2D, VectorField3D],
        distance_field: DistanceField,
        t_max: float = 100,
        dt: float = 0.1,
        diffusivity: float = 1,
    ):
        super().__init__(
            positions, target, vector_field, t_max, dt, diffusivity
        )
        assert self.dim == 3, "Domain navigation simulation must be 3D"
        assert isinstance(
            distance_field, DistanceField
        ), "Domain must be a 3D binary map"
        self._distance_field = distance_field
        self._distance_field_grad = distance_field.spatial_gradient()
        ## Initialise the loss
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)
        self.distance_loss = ti.field(ti.f32, shape=(), needs_grad=True)
        self.bend_loss = ti.field(ti.f32, shape=(), needs_grad=True)
        self.obstacle_loss = ti.field(ti.f32, shape=(), needs_grad=True)

    def reset(self):
        super().reset()
        self.reset_loss()

    @ti.kernel
    def reset_loss(self):
        self.loss[None] = 0.0
        self.loss.grad[None] = 1.0
        self.distance_loss[None] = 0.0
        self.distance_loss.grad[None] = 1.0
        self.bend_loss[None] = 0.0
        self.bend_loss.grad[None] = 1.0
        self.obstacle_loss[None] = 0.0
        self.obstacle_loss.grad[None] = 1.0

    @ti.func
    def collision_handling(self, n: int, t: int):
        """Check all collision for a walker n at time t and apply changes.

        Args:
            n (int): walker index
            t (int): timestep index
        """
        ## Frontier of the domain collisions
        if not self.vector_field.contains(self._positions[n, t]):
            self._domain_collision(n, t)
        ## Free space navigation
        if self._distance_field.at(self._positions[n, t]) == 0.0:
            self._positions[n, t] = self._positions[n, t - 1]

    @ti.kernel
    def compute_loss(
        self,
        distance_weight: float,
        bend_weight: float,
        obstacle_weight: float,
    ):
        self.loss[None] = (
            distance_weight * self.distance_loss[None]
            + bend_weight * self.bend_loss[None]
            + obstacle_weight * self.obstacle_loss[None]
        ) / self._nb_walkers

    @ti.kernel
    def compute_distance_loss(self, t_max: int):
        for n in range(self._nb_walkers):
            # diff = self._positions[n, t_max] - self.target
            ti.atomic_add(
                self.distance_loss[None],
                tm.length(self._positions[n, t_max] - self.target) ** 2,
                # tm.dot(diff, diff),
            )

    @ti.kernel
    def compute_bend_loss(self, min_diff: float):
        for n in range(self._nb_walkers):
            for t in range(self._nb_steps - 1):
                F_t1 = self.vector_field.at(self._positions[n, t])
                F_t2 = self.vector_field.at(self._positions[n, t + 1])
                bending = compute_angle_3D(F_t1, F_t2)
                # cos = compute_cosine_similarity(F_t1, F_t2)
                # ti.atomic_add(self.bend_loss[None], (1 - cos) ** 2)
                ti.atomic_add(self.bend_loss[None], bending**2)

                # f_diff = F_t2 - F_t1
                # if tm.length(f_diff) < min_diff:
                #     ti.atomic_add(self.bend_loss[None], 0.0)
                # else:
                #     bending = tm.atan2(tm.length(f_diff), tm.dot(F_t1, f_diff))
                #     ti.atomic_add(self.bend_loss[None], ti.abs(bending))

    @ti.kernel
    def compute_obstacle_loss(self, min_diff: float, collision_length: float):
        for n in range(self._nb_walkers):
            for t in range(self._nb_steps - 1):
                f1 = self.vector_field.at(self._positions[n, t])
                f2 = self._distance_field_grad.at(self._positions[n, t])
                angle = compute_angle_3D(f1, f2)
                # cos = compute_cosine_similarity(f1, f2)
                obstacle_dist = self._distance_field.at(self._positions[n, t])
                ti.atomic_add(
                    self.obstacle_loss[None],
                    angle**2
                    * tm.exp(
                        -0.5 * obstacle_dist**2 / collision_length**2
                    ),
                )
                # if tm.length(f2) == 0.0 or tm.length(f1) == 0.0:
                #     ti.atomic_add(self.obstacle_loss[None], 0.0)
                # else:
                #     f1n = tm.normalize(f1)
                #     f2n = tm.normalize(f2)
                #     dist = self._distance_field.at(self._positions[n, t])

                #     f_diff = f2n - f1n
                #     if tm.length(f_diff) < min_diff:
                #         ti.atomic_add(self.obstacle_loss[None], 0.0)
                #     else:
                #         bending = tm.atan2(
                #             tm.length(f_diff), tm.dot(f1n, f_diff)
                #         )

                #         ti.atomic_add(
                #             self.obstacle_loss[None],
                #             ti.abs(bending)
                #             * tm.exp(-0.5 * dist**2 / collision_length),
                #         )


class ClutteredNavigationSimulation(FreeNavigationSimulation):
    def __init__(
        self,
        positions: np.ndarray,
        target: np.ndarray,
        vector_field: VectorField2D,
        obstacles: List[geometries.Box2D],
        t_max: float = 100,
        dt: float = 0.1,
        diffusivity: float = 1,
    ):
        super().__init__(positions, vector_field, t_max, dt, diffusivity)
        # Initialisation of the obstacles
        if not isinstance(obstacles, (list, tuple)):
            raise TypeError(
                f"Expected obstacles to be a list or tuple of boxes. Get {type(obstacles)}"
            )

        self.nb_obstacles = len(obstacles)
        if self._dim == 2:
            if not all(isinstance(obs, geometries.Box2D) for obs in obstacles):
                raise TypeError(
                    f"Expected obstacles to be a list of 2D boxes. Get {obstacles}"
                )
            self._obstacles = geometries.Box2D.field(self.nb_obstacles)

        if self._dim == 3:
            if not all(isinstance(obs, geometries.Box3D) for obs in obstacles):
                raise TypeError(
                    f"Expected obstacles to be a list of 3D boxes. Get {obstacles}"
                )
            self._obstacles = geometries.Box3D.field(self.nb_obstacles)

        for i, obs in enumerate(obstacles):
            self._obstacles.min[i] = obs.min
            self._obstacles.max[i] = obs.max

    def collision_handling(self, n: int, t: int):
        """Check all collision for a walker n at time t and apply changes.

        Args:
            n (int): walker index
            t (int): timestep index
        """
        ## Domain collisions
        if not self.vector_field.contains(self._positions[n, t]):
            self._domain_collision(n, t)

        for o in ti.ndrange(self.nb_obstacles):
            if self._obstacles[o].contains(self._positions[n, t]):
                self._obstacle_collision(n, t, o)

    @ti.func
    def _obstacle_collision(self, n: int, t: int, o: int):
        for i in ti.static(range(self._dim)):
            ## Collision on the min border
            if (self._positions[n, t - 1][i] < self._obstacles[o].min[i]) and (
                self._positions[n, t][i] >= self._obstacles[o].min[i]
            ):
                toi = (
                    self._obstacles[o].min[i] - self._positions[n, t - 1][i]
                ) / (self._positions[n, t][i] - self._positions[n, t - 1][i])
                self._positions[n, t][i] = lerp(
                    self._positions[n, t - 1][i], self._positions[n, t][i], toi
                ) - ti.abs(
                    self._obstacles[o].min[i] - self._positions[n, t][i]
                )

            ## Collision on the max border
            elif (
                self._positions[n, t - 1][i] > self._obstacles[o].max[i]
            ) and (self._positions[n, t][i] <= self._obstacles[o].max[i]):
                toi = (
                    self._obstacles[o].max[i] - self._positions[n, t - 1][i]
                ) / (self._positions[n, t][i] - self._positions[n, t - 1][i])
                self._positions[n, t][i] = lerp(
                    self._positions[n, t - 1][i], self._positions[n, t][i], toi
                ) + ti.abs(
                    self._obstacles[o].max[i] - self._positions[n, t][i]
                )


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    from napari_potential_field_navigation.fields import ScalarField2D
    from napari_potential_field_navigation.geometries import Box2D

    ti.init(arch=ti.cpu)

    space = np.linspace(-1, 1, 100)
    # values = space[:, None] * space[None, :]
    values = np.linalg.norm(np.mgrid[-1:1:100j, -1:1:100j], axis=0)
    vector = np.stack(np.gradient(values), dtype=np.float32)

    bounds = Box2D(tm.vec2([-1, -1]), tm.vec2([1, 1]))

    field = ScalarField2D(values, bounds)
    vector_field = -field.spatial_gradient()

    positions = np.array(
        [[-0.5, -0.75], [0.5, 0.5], [-0.75, 0.5], [0.75, -0.5]],
        dtype=np.float32,
    )
    simulation = FreeNavigationSimulation(
        vector_field=vector_field,
        positions=positions,
        t_max=10,
        dt=0.1,
        diffusivity=0.001,
    )
    simulation.reset()
    simulation.run()

    colors = ["red", "orange", "yellow", "brown"]
    plt.imshow(values, origin="lower", extent=[-1, 1, -1, 1])
    plt.quiver(
        vector_field.meshgrid[0],
        vector_field.meshgrid[1],
        vector_field.values[:, :, 0],
        vector_field.values[:, :, 1],
        color="k",
        # angles="xy",
        # scale_units="xy",
        # scale=1,
    )

    for trajectory, color in zip(simulation.positions, colors):
        plt.plot(trajectory[:, 0], trajectory[:, 1], color=color)
    plt.show()

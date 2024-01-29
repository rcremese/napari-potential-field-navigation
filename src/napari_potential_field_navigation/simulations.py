import taichi as ti
import taichi.math as tm
import numpy as np

import napari_potential_field_navigation.geometries as geometries
from napari_potential_field_navigation.fields import (
    VectorField2D,
    VectorField3D,
    lerp,
)

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path


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
        self._noise: ti.Field = None

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
    def optimize(
        self, target_pos: np.ndarray, max_iter: int = 1000, lr: float = 1e-3
    ):
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
        vector_field: Union[VectorField2D, VectorField3D],
        # obstacles: List[geometries.Box2D] = None,
        t_max: float = 100.0,
        dt: float = 0.1,
        diffusivity: float = 1.0,
    ):
        ## Initialisation of the vector field
        if isinstance(vector_field, VectorField2D):
            self.dim = 2
        elif isinstance(vector_field, VectorField3D):
            self.dim = 3
        else:
            raise TypeError("Vector field must be either 2D or 3D")
        self.vector_field = vector_field

        ## Initialisation of the walkers positions
        self.update_positions(positions)

        ## Initialisation of the force field
        if not isinstance(vector_field, VectorField2D):
            raise TypeError(
                f"Expected force_field to be a VectorField. Get {type(vector_field)}"
            )
        self.vector_field = vector_field

        ## Simulation specific parameters
        self.update_time(t_max, dt)

        if diffusivity < 0.0:
            raise ValueError(
                f"Expected diffusivity to be positive. Get {diffusivity}"
            )
        self.diffusivity = diffusivity
        # self._dt = dt
        # self._t_max = t_max
        # self._nb_steps = np.ceil(t_max / dt).astype(int)
        # self.diffusivity = diffusivity
        # ## Taichi field definition
        # self._positions: ti.Field = ti.Vector.field(
        #     n=self.dim,
        #     dtype=ti.f32,
        #     shape=(self.nb_walkers, self._nb_steps),
        #     needs_grad=True,
        # )
        # self._noise: ti.Field = ti.Vector.field(
        #     self.dim,
        #     dtype=ti.float32,
        #     shape=(self.nb_walkers, self._nb_steps),
        #     needs_grad=False,
        # )
        # Definition of the loss
        self.loss = ti.field(ti.f32, shape=(), needs_grad=True)
        self.target: np.ndarray = None

    ## Public methods
    def update_positions(self, positions: np.ndarray):
        if self.dim == 2 and not (
            positions.ndim == 2 and positions.shape[1] == 2
        ):
            raise ValueError(
                f"Expected positions to be a (n, 2)-array of position vectors. Get {positions.shape}"
            )
        if self.dim == 3 and not (
            positions.ndim == 2 and positions.shape[1] == 3
        ):
            raise ValueError(
                f"Expected positions to be a (n, 3)-array of position vectors. Get {positions.shape}"
            )
        self.nb_walkers = positions.shape[0]
        self._init_pos: ti.Field = ti.Vector.field(
            self.dim, dtype=ti.f32, shape=(self.nb_walkers,)
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
            n=self.dim,
            dtype=ti.f32,
            shape=(self.nb_walkers, self._nb_steps),
            needs_grad=True,
        )
        self._noise: ti.Field = ti.Vector.field(
            self.dim,
            dtype=ti.float32,
            shape=(self.nb_walkers, self._nb_steps),
            needs_grad=False,
        )

    def run(self):
        for t in range(1, self._nb_steps):
            self.step(t)

    def optimize(
        self, target_pos: np.ndarray, max_iter: int = 100, lr: float = 1e-3
    ):
        assert isinstance(target_pos, np.ndarray) and target_pos.shape == (
            self.dim,
        ), f"Expected target_pos to be a {self.dim}D-array. Get {target_pos.shape}"
        assert (
            isinstance(max_iter, int) and max_iter > 0
        ), f"Expected max_iter to be a positive integer. Get {max_iter}"
        assert (
            isinstance(lr, (float, int)) and lr > 0.0
        ), f"Expected lr to be a positive float. Get {lr}"

        self.target = tm.vec2(*target_pos)
        for iter in range(max_iter):
            self.reset()
            with ti.ad.Tape(self.loss):
                self.run()
                self.compute_loss(self._nb_steps - 1)
            print("Iter=", iter, "Loss=", self.loss[None])
            self._update_force_field(lr)

    def reset(self):
        if self.dim == 2:
            self._reset_2d()
        elif self.dim == 3:
            self._reset_3d()
        else:
            raise ValueError("Simulation dimension must be either 2 or 3")

    ### Taichi kernels
    @ti.kernel
    def _reset_2d(self):
        for n in ti.ndrange(self.nb_walkers):
            self._positions[n, 0] = self._init_pos[n]
            for t in ti.ndrange(self._nb_steps):
                self._noise[n, t] = tm.vec2(ti.randn(), ti.randn())
        for i, j in self.vector_field._values:
            self.vector_field._values.grad[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def _reset_3d(self):
        for n in ti.ndrange(self.nb_walkers):
            self._positions[n, 0] = self._init_pos[n]
            for t in ti.ndrange(self._nb_steps):
                self._noise[n, t] = tm.vec3(ti.randn(), ti.randn(), ti.randn())
        for i, j, k in self.vector_field._values:
            self.vector_field._values.grad[i, j, k] = tm.vec3(0.0, 0.0, 0.0)

    @ti.kernel
    def step(self, t: int):
        for n in ti.ndrange(self.nb_walkers):
            self._positions[n, t] = (
                self._positions[n, t - 1]
                + self._dt * self.vector_field.at(self._positions[n, t - 1])
                + tm.sqrt(2 * self.dim * self._dt * self.diffusivity)
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
        for n in range(self.nb_walkers):
            self.loss[None] += (
                tm.length(self._positions[n, t] - self.target)
            ) / self.nb_walkers

    ## Private methods
    @ti.func
    def _domain_collision(self, n: int, t: int):
        """Check the collision of a walker at time t with the domain boundaries.

        Args:
            n (int): index of the walker
            t (int): index of the time step
        """
        for i in ti.static(range(2)):
            if self._positions[n, t][i] < self.vector_field._bounds.min[i]:
                self._positions[n, t][i] = self.vector_field._bounds.min[i]
            elif self._positions[n, t][i] > self.vector_field._bounds.max[i]:
                self._positions[n, t][i] = self.vector_field._bounds.max[i]

    @ti.kernel
    def _update_force_field(self, lr: float):
        for I in ti.grouped(self.vector_field._values):
            self.vector_field._values[I] -= (
                lr * self.vector_field._values.grad[I]
            )

    @property
    def positions(self) -> np.ndarray:
        return self._positions.to_numpy()

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def t_max(self) -> float:
        return self._t_max


class ClutteredNavigationSimulation2D(FreeNavigationSimulation):
    def __init__(
        self,
        positions: np.ndarray,
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
        if self.dim == 2:
            if not all(isinstance(obs, geometries.Box2D) for obs in obstacles):
                raise TypeError(
                    f"Expected obstacles to be a list of 2D boxes. Get {obstacles}"
                )
            self._obstacles = geometries.Box2D.field(self.nb_obstacles)

        if self.dim == 3:
            if not all(isinstance(obs, geometries.Box3D) for obs in obstacles):
                raise TypeError(
                    f"Expected obstacles to be a list of 3D boxes. Get {obstacles}"
                )
            self._obstacles = geometries.Box3D.field(self.nb_obstacles)

        for i, obs in enumerate(obstacles):
            self._obstacles.min[i] = obs.min
            self._obstacles.max[i] = obs.max

    def collision_handling(self, n: int, t: int):
        super().collision_handling(n, t)
        for o in ti.ndrange(self.nb_obstacles):
            if self._obstacles[o].contains(self._positions[n, t]):
                self._obstacle_collision(n, t, o)

    @ti.func
    def _obstacle_collision(self, n: int, t: int, o: int):
        for i in ti.static(range(self.dim)):
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

            # Collision on the max border
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

                self._positions[n, t].vel[i] = -self._positions[n, t].vel[i]


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

import taichi as ti
import taichi.math as tm
import numpy as np
from scipy.ndimage import distance_transform_edt
from napari_potential_field_navigation.fields import DistanceField

from napari_potential_field_navigation.simulations import (
    DomainNavigationSimulation,
)

class FocusedWalkers(DomainNavigationSimulation):
    """
    FocusedWalkers feature decreasing diffusion as they get closer to the
    target. This strategy makes it possible to start with high noise level
    while still targeting locations in tight volumes. In addition, a walker
    will stay around the target once reached.
    """
    def __init__(self, *args, **kwargs):
        domain = kwargs.pop("domain")
        super().__init__(*args, **kwargs)
        self._local_diameter: DistanceField = DistanceField(
            local_scale_transform(domain),
            self.vector_field._bounds,
        )

    @ti.kernel
    def compute_distance_loss(self, maturity: float):
        for n in range(self._nb_walkers):
            self.distance_loss[None] += (
                self._min_normalized_squared_distance_to_target[n]
            )

    def update_time(self, t_max: float, dt: float):
        super().update_time(t_max, dt)
        self._normalized_squared_distance_to_target: ti.Field = ti.field(
            dtype=ti.f32,
            shape=(self._nb_walkers, self._nb_steps),
            needs_grad=True,
        )
        self._min_normalized_squared_distance_to_target: ti.Field = ti.field(
            dtype=ti.f32,
            shape=(self._nb_walkers, ),
            needs_grad=True,
        )
        self._initial_squared_distance_to_target: ti.Field = ti.field(
            dtype=ti.f32,
            shape=(self._nb_walkers, ),
            needs_grad=False,
        )

    # augmented copies of DomainNavigationSimulation implementations
    @ti.kernel
    def _reset_2d(self):
        for n in ti.ndrange(self._nb_walkers):
            self._positions[n, 0] = self._init_pos[n]
            for t in ti.ndrange(self._nb_steps):
                self._noise[n, t] = tm.vec2(ti.randn(), ti.randn())
            # addition
            dr = self._init_pos[n] - self.target
            self._initial_squared_distance_to_target[n] = tm.dot(dr, dr)
            self._normalized_squared_distance_to_target[n, 0] = 1.
            self._min_normalized_squared_distance_to_target[n] = 1.
            #
        for i, j in self.vector_field._values:
            self.vector_field._values.grad[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def _reset_3d(self):
        for n in ti.ndrange(self._nb_walkers):
            self._positions[n, 0] = self._init_pos[n]
            for t in ti.ndrange(self._nb_steps):
                self._noise[n, t] = tm.vec3(ti.randn(), ti.randn(), ti.randn())
            # addition
            dr = self._init_pos[n] - self.target
            self._initial_squared_distance_to_target[n] = tm.dot(dr, dr)
            self._normalized_squared_distance_to_target[n, 0] = 1.
            self._min_normalized_squared_distance_to_target[n] = 1.
            #
        for i, j, k in self.vector_field._values:
            self.vector_field._values.grad[i, j, k] = tm.vec3(0.0, 0.0, 0.0)

    @ti.func
    def scale_diffusivity( self, diffusivity: ti.f32, distance: ti.f32):
        return diffusivity * distance

    @ti.kernel
    def step(self, t: int, diffusivity: ti.f32):
        for n in ti.ndrange(self._nb_walkers):
            p = self._positions[n, t - 1]

            diffusivity_n = self.scale_diffusivity(
                diffusivity,
                tm.sqrt(self._normalized_squared_distance_to_target[n, t - 1]),
            )

            random_displacement = (
                tm.sqrt(2 * self._dim * self._dt * diffusivity_n)
                * self._noise[n, t]
            )

            # adjusting the diffusion helps, especially with making the walkers
            # stop by the target, but thresholding the norm of the actual
            # random displacement is necessary to reduce the rate of
            # cross-boundary jumps
            rd_norm = tm.length(random_displacement)
            rd_norm_max = self._local_diameter.at(p) / 2
            if rd_norm_max < rd_norm:
                random_displacement *= rd_norm_max / rd_norm

            self._positions[n, t] = (
                p + self._dt * self.vector_field.at(p) + random_displacement
            )
            self.collision_handling(n, t)

            dr = self._positions[n, t] - self.target
            d2 = self._normalized_squared_distance_to_target[n, t] = (
                tm.dot(dr, dr) / self._initial_squared_distance_to_target[n]
            )
            ti.atomic_min(
                self._min_normalized_squared_distance_to_target[n], d2
            )

    @property
    def mean_trajectory(self) -> np.ndarray:
        mean_positions = mean_trajectory(
            self.positions,
            self._init_pos.to_numpy()[0,:],
            self.target.to_numpy(),
            self._min_normalized_squared_distance_to_target.to_numpy(),
        )
        ids = np.zeros(self.nb_steps, dtype=int)
        times = np.arange(self.nb_steps, dtype=int)
        return np.column_stack((ids, times, mean_positions))


def set_simulation_default_values_for_lung(container):
    container._time_slider.value = 2_000
    container._speed_slider.value = 1.
    container._diffusivity_slider.value = 5.
    container._agent_count.value = 2_000
    container._lr_slider.value = .5
    container._nb_epochs_box.value = 200 # without diffusion scaling
    # empirical rule: add 100 epochs for each unit of diffusion difference
    # between min and max
    container._nb_epochs_box.value = 500
    container._diffusion_decrease.value = "Linear"
    container._diffusion_max.value = 5.
    container._diffusion_min.value = 2.
    container._goal_distance.value = 1.
    container._bending_constraint.value = 0.
    container._obstacle_distance.value = 0.

def reset_simulation_default_values(container):
    container._time_slider.value = 100
    container._speed_slider.value = 0
    container._diffusivity_slider.value = 0
    container._agent_count.value = 1
    container._nb_epochs_box.value = 10
    container._lr_slider.value = 0.1
    container._diffusion_decrease.value = "None"
    container._diffusion_max.value = 0
    container._diffusion_min.value = 0
    container._goal_distance.value = 1.0
    container._bending_constraint.value = 1.0
    container._obstacle_distance.value = 1.0

def local_scale_transform(volume):
    """
    Similar to scipy.ndimage.distrance_transform_edt, but looks at the opposite
    direction as well. It takes the distance between the first background
    elements in both directions as a measurement of the local available “room”.

    This is used to automatically adjust the local diffusivity.
    """
    nearest = distance_transform_edt(
        volume,
        return_distances=False,
        return_indices=True,
    )
    local = np.stack(
        np.meshgrid(
            *[np.arange(d) for d in volume.shape],
            indexing='ij',
        ),
        axis=0,
    )
    opposite = 2 * local - nearest
    step = (local - nearest).astype(float)
    norm = np.sqrt((step * step).sum(axis=0))
    norm[norm <= 0] = 1
    step /= norm
    #
    active = volume == 1
    ndim = opposite.shape[0]
    n = active.sum()
    while  0 < n:
        outofbounds = np.zeros(n, dtype=bool)
        update = []
        for i in range(ndim):
            r = opposite[i, ...][active].astype(float)
            dr = step[i, ...][active]
            # step
            search_head = np.round(r + dr).astype(int)
            update.append(search_head)
            outofbounds |= search_head < 0
            outofbounds |= volume.shape[i] <= search_head
        # discard out-of-bounds heads
        active[active] = ~outofbounds
        update = [head[~outofbounds] for head in update]
        # discard heads in the background
        foreground = volume[tuple(update)] == 1
        active[active] = foreground
        update = np.stack([head[foreground] for head in update], axis=0)
        # update opposite positions
        for i in range(ndim):
            broadcast_active = np.zeros(opposite.shape, dtype=active.dtype)
            broadcast_active[i] = active
            opposite[broadcast_active] = update[i]
        n = active.sum()
    # return Euclidean distance between nearest and opposite positions
    diameter = opposite - nearest
    diameter = np.sqrt((diameter * diameter).sum(axis=0))
    return diameter

def mean_trajectory(
    positions,
    origin,
    destination,
    squared_distance=None,
    squared_radius=None,
    chunk_size=100,
    downsampling_step=10,
):
    time_average = np.mean(positions, axis=0) # for later usage
    ### 1. compute a path using a moving space window
    ## format the input arrays
    origin = origin.flatten()
    destination = destination.flatten()
    ## select the 10% best trajectories and make precalculations for L2
    # distances with the following standard vectorized formula:
    # ||x-y||^2 = -2 * <x,y> + ||x||^2 + transpose(||y||^2)
    q2 = (destination * destination).sum(axis=-1)
    if squared_distance is None:
        terminals = positions[:, -1, :]
        p2 = (terminals * terminals).sum(axis=-1)
        d2 = -2 * np.inner(terminals, destination) + p2 + q2
    else:
        d2 = squared_distance
    # positions' shape is (nb_walkers, nb_steps, space_dims)
    n, t, d = positions.shape
    n_ = n // 10
    best_trajectories = np.argsort(d2)[:n_]
    positions_ = positions[best_trajectories, 1:, :]
    t_ = t - 1
    assert positions_.shape == (n_, t_, d)
    positions_ = positions_.reshape((n_ * t_, -1))
    p2 = (positions_ * positions_).sum(axis=-1)
    ## initialize
    moving_average = origin
    r2 = np.dot(moving_average, moving_average)
    d2_t = d2_t0 = np.inf # will be the distance to the target
    radius2_scale = 1.
    not_visited = np.ones(p2.shape, dtype=bool) # greedy-search indicator
    path = []
    ## determine the search radius
    if squared_radius is None:
        max_jump = destination - origin
        max_radius2 = np.dot(max_jump, max_jump)
        radius2 = max_radius2 / (10 * 10)
        # # search for nearest neighbors of the origin point among the first
        # # time steps
        # m = 10 * n_
        # d2 = -2 * np.inner(positions_[:m], moving_average) + p2[:m] + r2
        # i = np.argsort(d2)
        # m_ = max(10, n_ // 4) # another arbitrary threshold that implicitly
        # # determines the search radius (squared).
        # # discard points that did not move from (or came back to) the origin
        # if d2[i[0]] == 0:
        #     m_ += (d2 == 0).sum()
        #     m_ = min(m_, m)
        # i = i[:m_]
        # radius2 = d2[i[-1]]
        # print(f'mean_trajectory: search radius = {np.sqrt(radius2)}')
        # moving_average = np.mean(positions_[i, :], axis=0) # first point
        # not_visited[i] = False
        # # squared distance to target
        # r2 = (moving_average * moving_average).sum(axis=-1)
        # d2_t = -2 * np.dot(destination, moving_average) + q2 + r2
        # # initialize
        # path = [moving_average]
    else:
        radius2 = squared_radius
    ## build up the path
    while np.any(not_visited) and radius2 < d2_t:
        d2 = (
            -2 * np.inner(positions_[not_visited, :], moving_average)
            + p2[not_visited] + r2
        )
        # # scale the radius wrt the distance to the target
        # if not np.isinf(d2_t):
        #     radius2_scale = 1.
        #     radius2_scale = d2_t / d2_t0
        # 3 times the (scaled) radius; this again is arbitrary
        I = d2 <= 9 * radius2_scale * radius2
        if not np.any(I): break
        i = not_visited.nonzero()[0][I]
        moving_average = np.mean(positions_[i, :], axis=0)
        not_visited[i] = False
        path.append(moving_average)
        # squared distance to target
        r2 = (moving_average * moving_average).sum(axis=-1)
        d2_t = -2 * np.dot(destination, moving_average) + q2 + r2
        if np.isinf(d2_t0):
            d2_t0 = d2_t # distance to target of the first path point
    ## add a last path point near the target, without the greedy indicator
    if radius2 < d2_t:
        d2 = -2 * np.inner(positions_, destination) + p2 + q2
        I = d2 <= radius2
        moving_average = np.mean(positions_[I, :], axis=0)
        path.append(moving_average)
    else:
        print('mean_trajectory: target not reached')
    ### 2. resample the path over time ]0,t];
    # note we exclude 0 for a practical concern and
    time_progress = np.arange(0, t + 1, downsampling_step)[1:] / t
    space_progress = np.diff(np.stack(path, axis=0), axis=0)
    space_progress = (space_progress * space_progress).sum(axis=-1)
    space_progress = np.r_[0., np.cumsum(space_progress)]
    space_progress /= space_progress[-1]
    # deal with numerical precision errors
    space_progress[0] = 0.
    space_progress[-1] = 1.
    assert time_progress[-1] <= space_progress[-1]
    new_path = [path[0]] # because we excluded 0 from time_progress
    for p in time_progress:
        k = (p <= space_progress).nonzero()[0][0]
        assert 0 < k # because we excluded 0 from time_progress
        # interpolate within segment [k-1,k]
        p0, p1 = space_progress[k - 1] , space_progress[k]
        point = (
            (p1 - p) / (p1 - p0) * path[k - 1]
            + (p - p0) / (p1 - p0) * path[k]
        )
        new_path.append(point)
    path = np.stack(new_path, axis=0)
    ### 3. adjust the mean positions to the actual positions on a
    # nearest-neighbor basis; here we could use all the trajectories but in
    # practice this might take too much resources (memory or compute time)
    q2 = (path * path).sum(axis=-1)
    t__ = path.shape[0]
    p2 = np.atleast_2d(p2).T # column vector
    m = n // 2 # another arbitrary threshold on the number of points
    new_path = []
    for k in range(t__ // chunk_size + 1):
        k *= chunk_size
        if k == t__: break
        l = min(k + chunk_size, t__)
        chunk = path[k:l, :]
        q2k = np.atleast_2d(q2[k:l]) # row vector
        d2 = -2 * np.inner(positions_, chunk) + p2 + q2k
        i = np.argsort(d2, axis=0)[0:m, :]
        for j in range(i.shape[1]):
            new_mean = np.mean(positions_[i[:, j], :], axis=0)
            new_path.append(new_mean)
    path = np.stack(new_path, axis=0)
    ### 4. upsample to compensate for the previous downsampling
    if 1 < downsampling_step:
        new_path = []
        grid = np.linspace(0., 1., downsampling_step + 1)
        for k in range(1, path.shape[0]):
            p0, p1 = path[k - 1, :], path[k, :]
            segment = np.outer(grid[::-1], p0) + np.outer(grid, p1)
            # we could avoid dropping the last point of the last segment,
            # however, as path points are interval bounds, we have to drop 1
            # point anyway
            segment = segment[:-1, :]
            new_path.append(segment)
        path = np.concatenate(new_path, axis=0)
    return path


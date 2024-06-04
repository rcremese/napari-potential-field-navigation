import napari
import numpy as np
import pandas as pd


def create_trajectories(nb_source, nb_agent, t_max):
    space = np.linspace(0, 1, nb_source)
    random_points = np.random.rand(nb_agent * t_max, 3)
    random_points[0] = 0.0
    trajectories = np.zeros((nb_agent * nb_source * t_max, 3))

    product = nb_agent * t_max
    for i in range(nb_source):
        trajectories[i * product : (i + 1) * product] = (
            np.array([space[i], space[i], 0]) + random_points
        )

    return trajectories


nb_agents, nb_source, t_max = 10, 5, 100

positions = create_trajectories(nb_source, nb_agents, t_max)
# space = np.linspace(0, 10, nb_agents)
# random_values = np.random.rand(nb_agents * t_max, 3)
# positions = np.stack([space, space, np.zeros_like(space)], axis=1)

ids = np.repeat(np.arange(nb_agents), t_max * nb_source).astype(int)
times = np.tile(np.arange(t_max), nb_agents * nb_source)
source = np.repeat(np.arange(nb_source), t_max * nb_agents).astype(int)
trajectories = pd.DataFrame(
    np.column_stack([ids, times, positions]),
    columns=["id", "time", "x", "y", "z"],
)
trajectories["source"] = source

mean_traj = trajectories.groupby(["source", "time"]).mean()
mean_traj["id"] = np.repeat(np.arange(nb_source), t_max)
mean_traj["time"] = np.tile(np.arange(t_max), nb_source)

viewer = napari.Viewer()
viewer.add_tracks(
    trajectories[["id", "time", "x", "y", "z"]],
    properties=trajectories["source"],
    color_by="source",
    colormaps_dict={0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "purple"},
)
viewer.add_tracks(
    mean_traj[["id", "time", "x", "y", "z"]],
    # color_by="source",
    # colormaps_dict={0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "purple"},
)
napari.run()

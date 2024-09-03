import napari
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TestCreateTrajectories:
    def setup_method(self):
        self.nb_agent = 10
        self.nb_sources = 2
        self.t_max = 100
        time = np.tile(np.arange(self.t_max), self.nb_agent * self.nb_sources)
        init_pos = np.tile(
            10 * np.arange(self.nb_sources), self.nb_agent * self.t_max
        )
        self.trajectories = pd.DataFrame(
            {
                "id": np.repeat(
                    np.arange(self.nb_agent), self.t_max * self.nb_sources
                ),
                "time": time,
                "source": np.repeat(
                    np.arange(self.nb_sources), self.nb_agent * self.t_max
                ),
                "x": init_pos
                + time
                + np.random.randn(
                    self.nb_agent * self.t_max * self.nb_sources
                ),
                "y": init_pos
                + time
                + np.random.randn(
                    self.nb_agent * self.t_max * self.nb_sources
                ),
                "z": init_pos
                + time
                + np.random.randn(
                    self.nb_agent * self.t_max * self.nb_sources
                ),
            }
        )

    def test_create_trajectories(self):
        mean_traj = self.mean_trajectories()
        assert isinstance(mean_traj, pd.DataFrame)
        assert mean_traj.shape[0] == self.t_max * self.nb_sources
        assert mean_traj.columns.tolist() == [
            "id",
            "time",
            "x",
            "y",
            "z",
        ]

        viewer = napari.Viewer()
        viewer.add_tracks(
            self.trajectories[["id", "time", "x", "y", "z"]],
            properties=self.trajectories["source"],
            color_by="source",
            colormaps_dict={
                0: "red",
                1: "green",
            },
        )
        viewer.add_tracks(
            mean_traj[["id", "time", "x", "y", "z"]],
            properties=mean_traj["id"],
            color_by="id",
            colormaps_dict={
                0: "red",
                1: "green",
            },
        )
        napari.run()

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # for source in range(self.nb_sources):
        #     traj_data = self.trajectories[
        #         self.trajectories["source"] == source
        #     ][["x", "y", "z"]].values
        #     ax.plot(
        #         traj_data[:, 0],
        #         traj_data[:, 1],
        #         traj_data[:, 2],
        #         label=f"Source {source}",
        #     )

        #     mean_traj_source = mean_traj[mean_traj["id"] == source][
        #         ["x", "y", "z"]
        #     ].values
        #     ax.plot(
        #         mean_traj_source[:, 0],
        #         mean_traj_source[:, 1],
        #         mean_traj_source[:, 2],
        #         label=f"Mean Source {source}",
        #         linestyle="--",
        #     )

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.legend()
        # plt.show()

    def mean_trajectories(self):
        mean_traj = pd.DataFrame(columns=["id", "time", "x", "y", "z"])
        for source in range(self.nb_sources):
            samples = {"x": [], "y": [], "z": []}
            for time in range(self.t_max):
                traj_data = self.trajectories[
                    (self.trajectories["source"] == source)
                    & (self.trajectories["time"] == time)
                ][["x", "y", "z"]].values
                kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
                    traj_data
                )
                sample = kde.sample(1)[0]
                samples["x"].append(sample[0])
                samples["y"].append(sample[1])
                samples["z"].append(sample[2])

            mean_traj_source = pd.DataFrame(
                {
                    "id": np.repeat(source, self.t_max),
                    "time": range(self.t_max),
                    "x": samples["x"],
                    "y": samples["y"],
                    "z": samples["z"],
                }
            )
            mean_traj = pd.concat(
                [mean_traj, mean_traj_source], ignore_index=True
            )
        return mean_traj


test_case = TestCreateTrajectories()
test_case.setup_method()
test_case.test_create_trajectories()

from typing import TYPE_CHECKING

import magicgui.widgets as widgets
import napari.utils.notifications as notifications
import numpy as np
import scipy.ndimage as ndi
import scipy.ndimage as ndi
from napari.qt.threading import thread_worker
import taichi as ti

from napari_potential_field_navigation.fields import ScalarField3D
from napari_potential_field_navigation.geometries import Box3D
from napari_potential_field_navigation.simulations import (
    FreeNavigationSimulation,
)

if TYPE_CHECKING:
    import napari


class IoContainer(widgets.Container):
    """Contains all informations about the input datas"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self._viewer = viewer
        # Image
        self._image_reader = widgets.FileEdit(label="Image path")
        self._image_reader.changed.connect(self._read_image)

        # Label
        self._label_reader = widgets.FileEdit(label="Label path")
        self._label_reader.changed.connect(self._read_label)

        self._crop_checkbox = widgets.PushButton(
            text="Crop image",
            tooltip="Crop the image and the labels to a bounding box containing all labels > 0. Helps reduce the computation time.",
        )
        self._crop_checkbox.changed.connect(self._crop_image)
        self._lock_checkbox = widgets.CheckBox(text="Lock")
        self._lock_checkbox.changed.connect(self._lock)

        self._checkbox_container = widgets.Container(
            widgets=[self._crop_checkbox, self._lock_checkbox],
            layout="horizontal",
        )
        self.extend(
            [
                widgets.Label(label="Data selection"),
                self._image_reader,
                self._label_reader,
                self._checkbox_container,
            ]
        )

    def _read_image(self):
        if "Image" in self._viewer.layers:
            self._viewer.layers.remove("Image")
        self._viewer.open(
            self._image_reader.value,
            plugin="napari-itk-io",
            layer_type="image",
            name="Image",
        )

        # Update of the layer stack
        if "Label" in self._viewer.layers:
            idx = self._viewer.layers.index("Label")
            self._viewer.layers.move(idx, -1)

    def _read_label(self):
        if "Label" in self._viewer.layers:
            self._viewer.layers.remove("Label")
        labels = self._viewer.open(
            self._label_reader.value,
            plugin="napari-itk-io",
            layer_type="image",
            name="Label_temp",
            visible=False,
        )
        for label in labels:
            data = label.data.astype(int)
            self._viewer.add_labels(
                data,
                scale=label.scale,
                metadata=label.metadata,
                translate=label.translate,
                name="Label",
                blending="additive",
                visible=True,
            )
            self._viewer.layers.remove(label)

        self._viewer.layers["Label"].editable = False

    def _crop_image(self) -> None:
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before croping the image."
            )
            return
        slices = ndi.find_objects(self._viewer.layers["Label"].data)
        # Take into account the shift of origin
        starting_index = [slide.start for slide in slices[0]]
        new_origin = np.array(
            self._viewer.layers["Label"].data_to_world(starting_index)
        )

        self._viewer.layers["Label"].data = self._viewer.layers["Label"].data[
            slices[0]
        ]
        self._viewer.layers["Label"].translate = new_origin
        if "Image" in self._viewer.layers:
            self._viewer.layers["Image"].data = self._viewer.layers[
                "Image"
            ].data[slices[0]]
            self._viewer.layers["Image"].translate = new_origin

    def _lock(self):
        notifications.show_info(
            "The image locking procedure is not yet available."
        )
        raise NotImplementedError


class PointContainer(widgets.Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(layout="horizontal")
        self._viewer = viewer
        self._source_selection = widgets.PushButton(text="Select goal")
        self._source_selection.changed.connect(self._select_source)

        self._positions_selection = widgets.PushButton(text="Select positions")
        self._positions_selection.changed.connect(self._select_positions)

        self._goal_layer = None
        self._position_layer = None

        self.extend(
            [
                widgets.Label(label="Point cloud selection"),
                self._source_selection,
                self._positions_selection,
            ]
        )

    def _select_source(self):
        if "Goal" in self._viewer.layers:
            self._viewer.layers.remove("Goal")
        self._goal_layer = self._viewer.add_points(
            name="Goal",
            edge_color="lime",
            face_color="transparent",
            symbol="disc",
            ndim=3,
        )
        self._goal_layer.mouse_drag_callbacks.append(self._on_add_point)

        print("Select source")
        self._viewer.layers.selection = [self._goal_layer]
        self._goal_layer.mode = "add"

    def _select_positions(self):
        if "Initial positions" not in self._viewer.layers:
            self._position_layer = self._viewer.add_points(
                name="Initial positions",
                edge_color="#0055ffff",
                face_color="transparent",
                symbol="disc",
                ndim=3,
            )

        print("Select positions")
        self._viewer.layers.selection = [self._position_layer]
        self._position_layer.mode = "add"

    def _on_add_point(self, layer, event):
        if layer.mode == "add" and layer.editable:
            layer.add(event.position)
            layer.editable = False
            self._source_selection.text = "Edit goal"

    @property
    def goal_position(self) -> np.ndarray:
        if self._goal_layer is None:
            raise ValueError("There is no goal layer in the viewer")
        return self._goal_layer.data[0]

    @property
    def initial_positions(self) -> np.ndarray:
        if self._position_layer is None:
            raise ValueError("No initial positions selected")
        return self._position_layer.data


class ApfContainer(widgets.Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._ratio_slider = widgets.FloatSlider(
            min=0, max=1, value=0.5, label="Attractive / Repulsive ratio"
        )
        self._resolution_combobox = widgets.ComboBox(
            label="Potential field resolution",
            choices=["1x", "2x", "4x", "8x", "16x"],
            value="1x",
        )
        # self._attractive_weight_slider = widgets.FloatSlider(
        #     min=1,
        #     max=1000,
        #     step=1,
        #     value=1,
        #     label="Attractive weight (unit)",
        # )
        # self._attractive_weight_slider.changed.connect(self._plot_apf)
        self._repulsive_weight_slider = widgets.FloatSlider(
            min=1,
            max=1000,
            step=1,
            value=1,
            label="Repulsive weight (unit)",
        )
        # self._repulsive_weight_slider.changed.connect(self._plot_apf)
        self._repulsive_radius_slider = widgets.FloatSlider(
            min=0.1, max=100, value=1, label="Repulsive radius (cm)"
        )
        self._weight_container = widgets.Container(
            widgets=[
                widgets.Label(label="APF parameters"),
                self._ratio_slider,
                self._resolution_combobox,
                # self._attractive_weight_slider,
                self._repulsive_weight_slider,
                self._repulsive_radius_slider,
            ],
            layout="vertical",
        )
        self._weight_container.changed.connect(self._plot_apf)
        self._compute_apf_box = widgets.PushButton(text="Compute APF")
        self._compute_worker = self._compute_apf()
        self._compute_worker.returned.connect(self._plot_apf)
        self._compute_apf_box.changed.connect(self._compute_worker.start)

        self.extend(
            [
                self._weight_container,
                self._compute_apf_box,
            ]
        )

        self._attractive_field = None
        self._bounds = None
        self._distance_field = None

    @thread_worker
    def _compute_apf(self) -> bool:
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the APF."
            )
            return False
        if "Goal" not in self._viewer.layers:
            notifications.show_error(
                "No goal found. Please select a goal before computing the APF."
            )
            return False
        self._compute_apf_box.text = "Computing APF..."
        self._compute_apf_box.enabled = False

        if "APF" in self._viewer.layers:
            self._viewer.layers.remove("APF")
            # self._viewer.layers.remove("Initial Vector Field")
        label_layer = self._viewer.layers["Label"]

        starting = np.array(label_layer.translate)
        spacing = np.array(label_layer.scale)
        ending = starting + spacing * label_layer.data.shape
        self._bounds = Box3D(starting, ending)

        self._attractive_field = self._compute_attractive_field(
            label_layer, self._viewer.layers["Goal"].data[0]
        )

        self._attractive_field[label_layer.data == 0] = 0
        self._distance_field = ndi.distance_transform_edt(
            label_layer.data, sampling=label_layer.scale
        )
        self._compute_apf_box.text = "Update APF"
        self._compute_apf_box.enabled = True

        return True

    def _plot_apf(self, compute_success: bool = True) -> bool:
        if not compute_success:
            notifications.show_error(
                "An error occured during the computation of the APF."
            )
            return False

        artificial_potential_field = np.where(
            self._viewer.layers["Label"].data, self.potential_field.values, 0
        )
        try:
            self._viewer.layers["APF"].data = artificial_potential_field
        except KeyError:
            self._viewer.add_image(
                artificial_potential_field,
                name="APF",
                colormap="inferno",
                blending="additive",
                scale=self._viewer.layers["Label"].scale,
                translate=self._viewer.layers["Label"].translate,
                metadata=self._viewer.layers["Label"].metadata,
            )
        return True

    @staticmethod
    def _compute_attractive_field(
        label_layer: "napari.layers.Labels", goal_position: np.ndarray
    ) -> np.ndarray:
        assert goal_position.shape == (3,), "Goal position must be 3D vector"
        starting = np.array(label_layer.translate)
        spacing = np.array(label_layer.scale)
        ending = starting + spacing * label_layer.data.shape
        spacial_grid = np.mgrid[
            starting[0] : ending[0] : spacing[0],
            starting[1] : ending[1] : spacing[1],
            starting[2] : ending[2] : spacing[2],
        ]

        attractive_field = 0.5 * np.linalg.norm(
            np.stack(
                [
                    spacial_grid[0] - goal_position[0],
                    spacial_grid[1] - goal_position[1],
                    spacial_grid[2] - goal_position[2],
                ]
            ),
            axis=0,
        )
        return attractive_field

    @property
    def potential_field(self) -> ScalarField3D:
        if self._attractive_field is None or self._distance_field is None:
            notifications.show_info(
                "No exising Artificial Potential Field found. Click on compute APF to generate one."
            )
            return None

        collision_radius = self._repulsive_radius_slider.value
        repulsive_field = np.zeros_like(self._distance_field)
        valid_indices = (self._distance_field <= collision_radius) & (
            self._distance_field > 0
        )
        repulsive_field[valid_indices] = (
            0.5
            * (
                (collision_radius - self._distance_field[valid_indices])
                / (collision_radius * self._distance_field[valid_indices])
            )
            ** 2
        )
        repulsive_field = np.where(
            self._distance_field > 0, repulsive_field, 1e20
        )
        ratio = self._ratio_slider.value
        artificial_potential_field = (
            (1 - ratio) * self._attractive_field
            + self._repulsive_weight_slider.value * ratio * repulsive_field
        )
        return ScalarField3D(artificial_potential_field, self._bounds)


class SimulationContainer(widgets.Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        apf_container: ApfContainer,
    ):
        super().__init__()
        self._viewer = viewer
        self._apf_container = apf_container
        self._time_slider = widgets.FloatSlider(
            min=0.1,
            max=1000,
            value=100,
            label="Simulation final time (s)",
        )
        self._timestep_slider = widgets.FloatSlider(
            min=0.01,
            max=1,
            value=1,
            step=0.01,
            label="Simulation time step (s)",
        )
        self._speed_slider = widgets.FloatSlider(
            min=0,
            max=10,
            value=0,
            step=0.1,
            label="Maximal speed (cm/s)",
        )
        self._diffusivity_slider = widgets.FloatSlider(
            min=0,
            max=10,
            value=0,
            step=0.1,
            label="Agent diffusivity (cm^2/s)",
        )

        self._agent_count = widgets.SpinBox(
            label="Number of agents", min=1, max=100, value=1
        )
        self._start_button = widgets.PushButton(text="Run simulation")
        self._start_button.changed.connect(self._run_simulation)

        button_container = widgets.Container(
            widgets=[
                self._agent_count,
                self._start_button,
            ],
            layout="horizontal",
        )

        self.extend(
            [
                widgets.Label(label="Simulation parameters"),
                self._time_slider,
                self._timestep_slider,
                self._speed_slider,
                self._diffusivity_slider,
                button_container,
            ]
        )
        ## Optimization widgets
        self._nb_epochs_box = widgets.SpinBox(
            label="Epochs", min=1, max=1000, step=10, value=100
        )
        self._lr_slider = widgets.FloatSpinBox(
            min=0.001, max=10, value=0.1, label="Learning rate"
        )

        self._run_optimization_button = widgets.PushButton(
            text="Run optimization"
        )
        self._run_optimization_button.changed.connect(self._run_optimization)

        self.extend(
            [
                widgets.Label(label="Optimization parameters"),
                self._nb_epochs_box,
                self._lr_slider,
                self._run_optimization_button,
            ]
        )

        self.simulation = None

    def _run_simulation(self) -> bool:
        # self._start_button.text = "Running simulation..."
        # self._start_button.enabled = False

        if not self._initialize_simulation():
            notifications.show_error(
                "The simulation could not be initialized."
            )
            return False
        self.simulation.reset()
        self.simulation.run()
        self._plot_trajectories("Initial trajectories")
        return True

    def _plot_trajectories(self, name: str) -> bool:
        if self.simulation is None:
            notifications.show_error("The simulation is not initialized.")
            return False
        if name.capitalize() in self._viewer.layers:
            self._viewer.layers.remove(name.capitalize())
        self._viewer.add_tracks(
            self.simulation.trajectories,
            name=name.capitalize(),
        )
        return True

    # def _update_simulation(self) -> bool:
    #     if self.simulation is None:
    #         notifications.show_error(
    #             "The simulation could is not initialized."
    #         )
    #         return False
    #     initial_positions = np.repeat(
    #         self._viewer.layers["Initial positions"].data,
    #         self._agent_count.value,
    #         axis=0,
    #     )

    #     self.simulation.diffusivity = self.diffusivity
    #     self.simulation.update_positions(initial_positions)
    #     self.simulation.update_time(self.tmax, self.dt)
    #     return True

    def _initialize_simulation(self) -> bool:
        potential_field = self._apf_container.potential_field
        if potential_field is None:
            notifications.show_error(
                "No potential field found. Please compute the APF before running the simulation."
            )
            return False
        vector_field = -potential_field.spatial_gradient()
        if self._speed_slider.value > 0:
            vector_field.norm_clip(self._speed_slider.value)

        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before running the simulation."
            )
            return False

        if "Initial positions" not in self._viewer.layers:
            notifications.show_error(
                "No initial positions found. Please select initial positions before running the simulation."
            )
            return False

        initial_positions = np.repeat(
            self._viewer.layers["Initial positions"].data,
            self._agent_count.value,
            axis=0,
        )
        if "Goal" not in self._viewer.layers:
            notifications.show_error(
                "No goal found. Please select a goal before running the simulation."
            )
            return False
        goal = self._viewer.layers["Goal"].data[0]
        assert goal.shape == (3,), "Goal position must be a 3D vector"

        self.simulation = FreeNavigationSimulation(
            initial_positions,
            goal,
            vector_field,
            t_max=self.tmax,
            dt=self.dt,
            diffusivity=self.diffusivity,
        )
        return True

    def _run_optimization(self):
        if not self._initialize_simulation():
            notifications.show_error(
                "The simulation could not be initialized."
            )
            return False
        self.simulation.optimize(
            max_iter=self._nb_epochs_box.value, lr=self._lr_slider.value
        )
        self._plot_trajectories("Optimized trajectories")
        return True

    @property
    def dt(self) -> float:
        return self._timestep_slider.value

    @property
    def tmax(self) -> float:
        return self._time_slider.value

    @property
    def vmax(self) -> float:
        return self._speed_slider.value

    @property
    def nb_agents(self) -> int:
        return self._agent_count.value

    @property
    def diffusivity(self) -> float:
        return self._diffusivity_slider.value


class OptimizationContainer(widgets.Container):
    def __init__(
        self, viewer: "napari.viewer.Viewer", sim_widget: SimulationContainer
    ):
        super().__init__(layout="horizontal")
        self._viewer = viewer
        self._sim_widget = sim_widget
        ## Optimization widgets
        self._nb_epochs_box = widgets.SpinBox(
            label="Epochs", min=1, max=1000, step=10, value=100
        )
        self._lr_slider = widgets.FloatSpinBox(
            min=0.001, max=10, value=0.1, label="Learning rate"
        )

        self._run_optimization_button = widgets.PushButton(
            text="Run optimization"
        )
        self._run_optimization_button.changed.connect(self._run_optimization)
        param_container = widgets.Container(
            widgets=[
                widgets.Label(label="Optimization parameters"),
                self._nb_epochs_box,
                self._lr_slider,
            ],
        )

        self.extend(
            [
                param_container,
                self._run_optimization_button,
            ]
        )

    def _run_optimization(self):
        notifications.show_info(f"nb_agents = {self._sim_widget.nb_agents}")
        raise NotImplementedError


class DiffApfWidget(widgets.Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        try:
            ti.init(arch=ti.gpu)
        except RuntimeError:
            notifications.show_warning("No GPU found. Using CPU.")
            ti.init(arch=ti.cpu)
        self._viewer = viewer
        self._io_container = IoContainer(self._viewer)
        self._point_container = PointContainer(self._viewer)
        self._apf_container = ApfContainer(self._viewer)
        self._simulation_container = SimulationContainer(
            self._viewer, self._apf_container
        )
        self._optimization_container = OptimizationContainer(
            self._viewer, self._simulation_container
        )
        self.extend(
            [
                self._io_container,
                self._point_container,
                self._apf_container,
                self._simulation_container,
                # self._optimization_container,
            ]
        )

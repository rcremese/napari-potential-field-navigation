from pathlib import Path
from typing import TYPE_CHECKING, Union
from enum import Enum
import logging

from abc import ABC, abstractmethod
import magicgui.widgets as widgets
import napari.layers
import napari.utils.notifications as notifications
import numpy as np
import scipy.ndimage as ndi
import scipy.ndimage as ndi
from napari.qt.threading import thread_worker
import taichi as ti
import taichi.math as tm
import scipy.sparse.linalg as splinalg
from sklearn.neighbors import KernelDensity
import pandas as pd

from napari_potential_field_navigation.fields import (
    ScalarField3D,
    VectorField3D,
    SimpleVectorField3D,
    DistanceField,
)
from napari_potential_field_navigation._a_star import (
    astar,
    wavefront_generation,
)
from napari_potential_field_navigation._finite_difference import (
    create_poisson_system,
)

from napari_potential_field_navigation.geometries import Box3D
from napari_potential_field_navigation.simulations import (
    DomainNavigationSimulation,
)
from napari_potential_field_navigation.focused_walkers import (
    FocusedWalkers,
    set_simulation_default_values_for_lung,
    reset_simulation_default_values,
)
import csv

from napari_potential_field_navigation._use_case import (
    UseCase,
    use_case_check_point,
)
import napari


class MethodSelection(Enum):
    APF = "Artificial Potential Field"
    WAVEFRONT = "Wavefront method"
    A_STAR = "A*"


class IoContainer(widgets.Container):
    """Contains all informations about the input datas"""

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        super().__init__()
        self._viewer: napari.Viewer = viewer
        # Image
        self._image_reader = widgets.FileEdit(label="Image path")
        self._image_reader.changed.connect(self._read_image)
        self._viewer.layers.events.inserted.connect(
            self._handle_layers_from_other_readers
        )

        # Label
        self._label_reader = widgets.FileEdit(label="Label path")
        self._label_reader.changed.connect(self._read_label)
        io_container = widgets.Container(
            widgets=[self._image_reader, self._label_reader],
            layout="horizontal",
        )

        # Autocropping (for now hard-coded)
        self._autocrop = True

        if not self._autocrop:
            # Manual cropping
            self._crop_checkbox = widgets.PushButton(
                text="Crop image",
                tooltip="Crop the image and the labels to a bounding box containing all labels > 0. Helps reduce the computation time.",
            )
            self._crop_checkbox.changed.connect(self._crop_image)
            io_container = widgets.Container(
                widgets=[io_container, self._crop_checkbox],
            )

        self.extend([widgets.Label(label="Data selection"), io_container])

    @use_case_check_point("Image")
    def _read_image(self, image_path):
        if "Image" in self._viewer.layers:
            self._viewer.layers.remove("Image")
        self._viewer.open(
            image_path,
            plugin="napari-itk-io",
            layer_type="image",
            name="Image",
        )

        # Update of the layer stack
        if "Label" in self._viewer.layers:
            idx = self._viewer.layers.index("Label")
            self._viewer.layers.move(idx, -1)

    @use_case_check_point("Label")
    def _read_label(self, label_path):
        if "Label" in self._viewer.layers:
            self._viewer.layers.remove("Label")
        labels = self._viewer.open(
            label_path,
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
        self._viewer.layers["Label"].metadata["origin"] = self._viewer.layers[
            "Label"
        ].translate

        if self._autocrop and "Image" in self._viewer.layers:
            self._crop_image()

    def _crop_image(self) -> None:
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No labels found. Please select a label file before cropping the image."
            )
            return
        logging.info("Cropping the volume to the label bounding box")
        ## Perform a crop of the image based on the label bounding box + 1 pixel
        slices = ndi.find_objects(
            ndi.binary_dilation(self._viewer.layers["Label"].data)
        )
        # Take into account the shift of origin
        starting_index = [slide.start for slide in slices[0]]
        new_origin = np.array(
            self._viewer.layers["Label"].data_to_world(starting_index)
        )

        self._viewer.layers["Label"].data = self._viewer.layers["Label"].data[
            slices[0]
        ]
        ## TODO : uncomment to get the image at the right resolution
        self._viewer.layers["Label"].translate = new_origin
        if "Image" in self._viewer.layers:
            self._viewer.layers["Image"].data = self._viewer.layers[
                "Image"
            ].data[slices[0]]
            ## TODO : uncomment to get the image at the right resolution
            self._viewer.layers["Image"].translate = new_origin

    @classmethod
    def _is_misnamed_image_layer(cls, layer):
        return (isinstance(layer, napari.layers.image.image.Image)
            and layer.name not in ("Image", "Label_temp", "Density")
            and not layer.name.endswith(" field")
        )

    @use_case_check_point(["Image", "Label"])
    def _handle_layers_from_other_readers(self, event):
        # the newly inserted layer is the last one in the layer list
        layer = event.source[-1]
        if self._is_misnamed_image_layer(layer):
            # the layer has been added using the viewer or File menu, i.e.
            # by copy-paste, drag-n-drop, Ctrl+O or File > Open File(s)...
            if "Image" in self._viewer.layers:
                self._viewer.layers.remove("Image")
            layer.name = "Image"

            if self._autocrop and "Label" in self._viewer.layers:
                self._crop_image()

            # any new Image layer is moved first (below all the other layers)
            nlayers = len(self._viewer.layers)
            if nlayers > 1:
                self._viewer.layers.move(nlayers - 1, 0)


class PointContainer(widgets.Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._source_selection = widgets.PushButton(text="Select goal")
        self._source_selection.changed.connect(self._select_source)

        self._positions_selection = widgets.PushButton(text="Select positions")
        self._positions_selection.changed.connect(self._select_positions)
        selection_container = widgets.Container(
            widgets=[self._source_selection, self._positions_selection],
            layout="horizontal",
        )
        self._goal_layer = None
        self._position_layer = None

        self.extend(
            [
                widgets.Label(label="Endpoint selection"),
                selection_container,
            ]
        )

    @use_case_check_point("Goal")
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

    @use_case_check_point("Initial positions")
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
            pos = event.position
            if len(pos) == 4:
                # time has been appended as first coordinate; remove it
                pos = pos[1:]
            layer.add(pos)
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


class InitFieldContainer(widgets.Container, ABC):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer: napari.Viewer = viewer
        self._compute_button = widgets.PushButton(
            text=f"Compute {self.method_name} field"
        )
        self._compute_button.changed.connect(self.compute)
        self._save_file = widgets.FileEdit(
            label=f"Save {self.method_name} field", mode="w"
        )
        self._save_file.changed.connect(self.save)
        self._load_file = widgets.FileEdit(
            label=f"Load {self.method_name} field"
        )
        self._load_file.changed.connect(self.load)
        self._plot_vectors_check = widgets.CheckBox(
            label="Plot vector field", value=False
        )
        load_and_save_container = widgets.Container(
            widgets=[self._save_file, self._load_file], layout="horizontal"
        )
        compute_container = widgets.Container(
            widgets=[self._plot_vectors_check, self._compute_button],
            layout="horizontal",
        )
        self._plot_vectors_check.changed.connect(self.visualize)
        ## Add the label domain selector
        self._label_domain_selector = widgets.SpinBox(
            description="Domain selection",
            min=0,
            max=10,
            value=1,
            step=1,
            tooltip="Select the domain in which to compute the field based on the label values",
        )
        self._show_all_labels = widgets.CheckBox(
            value=True, text="Show all labels"
        )
        self._show_all_labels.changed.connect(self._on_label_domain_change)
        self._label_domain_selector.changed.connect(
            self._on_label_domain_change
        )
        label_container = widgets.Container(
            widgets=[self._label_domain_selector, self._show_all_labels],
            layout="horizontal",
        )

        self.extend(
            [
                widgets.Label(label=f"{self.method_name} field computation"),
                load_and_save_container,
                compute_container,
                label_container,
            ]
        )

        self._field: np.ma.MaskedArray = None

    def compute(self):
        raise NotImplementedError

    def load(self, path: Union[str, Path]) -> bool:
        try:
            path = Path(path).resolve(strict=True)
        except FileNotFoundError:
            notifications.show_error(
                f"File {path} not found ! Please provide a valid path."
            )
            return False
        with np.load(path) as data:
            self._field = np.ma.masked_array(data["field"], mask=data["mask"])
        return True

    def save(self, path: Union[str, Path]) -> bool:
        path = Path(self._save_file.value).resolve()
        with path.open("wb") as file:
            np.savez_compressed(
                file,
                field=self._field.data,
                mask=self._field.mask,
            )
        return True

    def _on_label_domain_change(self) -> bool:
        assert "Label" in self._viewer.layers, "No label found"
        if (
            self._label_domain_selector.value
            not in self._viewer.layers["Label"].data
        ):
            notifications.show_error(
                f"The selected label is not in the label map. Choose one of {np.unique(self._viewer.layers['Label'].data)}"
            )
            return False
        label_layer: napari.layers.Labels = self._viewer.layers["Label"]
        label_layer.visible = True
        if self._label_domain_selector.value == 0:
            label_layer.visible = self._show_all_labels.value
            return True
        label_layer.selected_label = self._label_domain_selector.value
        label_layer.show_selected_label = not self._show_all_labels.value
        return True

    def visualize(self, plot_vectors=False) -> bool:
        field = self.field
        assert field.ndim == 3, "The field must be 3D"
        assert isinstance(
            field, np.ma.MaskedArray
        ), "The field must be a masked array"
        ## Check if the field is not None
        if field is None:
            notifications.show_error("No field found.")
            return False
        ## Remove the previous field if it exists
        if self.method_name.capitalize() + " field" in self._viewer.layers:
            self._viewer.layers.remove(
                self.method_name.capitalize() + " field"
            )
        ## Plot the scalar field
        self._viewer.add_image(
            np.where(field.mask, field.min(), field.data),
            name=self.method_name.capitalize() + " field",
            colormap="inferno",
            blending="additive",
            scale=self._viewer.layers["Label"].scale,
            translate=self._viewer.layers["Label"].translate,
            metadata=self._viewer.layers["Label"].metadata,
        )
        ## If only the scalar field is requested return
        if not plot_vectors:
            return True

        ## Code to plot also the vector field
        # TODO : Add spatial information
        if "Vector field" in self._viewer.layers:
            self._viewer.layers.remove("Vector field")
        vector_field = self.vector_field
        vector_field.normalize()
        vector_field = vector_field.values

        x, y, z = np.mgrid[
            0 : field.shape[0], 0 : field.shape[1], 0 : field.shape[2]
        ]

        valid_map = ~field.mask

        ## Downscale the vector field to avoid too many vectors
        def highest_power_of_two(n):
            return n & -n

        vec_power_of_two = np.vectorize(highest_power_of_two)
        downscale_factors = vec_power_of_two(field.shape)

        downscale_map = np.zeros_like(field, dtype=bool)
        downscale_map[
            :: downscale_factors[0],
            :: downscale_factors[1],
            :: downscale_factors[2],
        ] = True
        valid_map = valid_map & downscale_map

        data = np.zeros((valid_map.sum(), 2, 3))
        data[:, 0, 0] = x[valid_map]
        data[:, 0, 1] = y[valid_map]
        data[:, 0, 2] = z[valid_map]
        data[:, 1] = vector_field[valid_map]

        self._viewer.add_vectors(
            data,
            ndim=3,
            name="Vector field",
            scale=self._viewer.layers["Label"].scale,
            translate=self._viewer.layers["Label"].translate,
            metadata=self._viewer.layers["Label"].metadata,
        )
        return True

    @property
    @abstractmethod
    def method_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def field(self) -> np.ma.MaskedArray:
        raise NotImplementedError

    @property
    @abstractmethod
    def vector_field(self) -> VectorField3D:
        raise NotImplementedError


class WavefrontContainer(InitFieldContainer):
    @use_case_check_point
    def compute(self) -> bool:
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the wavefront."
            )
            return False
        if "Goal" not in self._viewer.layers:
            notifications.show_error(
                "No goal found. Please select a goal before computing the wavefront."
            )
            return False
        assert (
            len(self._viewer.layers["Goal"].data) == 1
        ), "Only one goal is allowed"
        label_layer = self._viewer.layers["Label"]
        goal_idx = label_layer.world_to_data(
            self._viewer.layers["Goal"].data[0]
        )
        goal_idx = tuple([round(idx) for idx in goal_idx])
        if label_layer.data[goal_idx] == 0:
            notifications.show_error("The goal must be in a free space.")
            return False
        ## Need to upsample the data to allow for gradient computation inside the volume
        self._field = wavefront_generation(
            ndi.binary_dilation(label_layer.data),
            goal_idx,
        )
        self.visualize(plot_vectors=self._plot_vectors_check.value)
        return True

    @property
    def field(self) -> np.ma.MaskedArray:
        return self._field

    @property
    def vector_field(self) -> VectorField3D:
        if self._field is None:
            notifications.show_error("No wavefront field found.")
            return None
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the wavefront."
            )
            return None
        label_layer = self._viewer.layers["Label"]
        starting = np.array(label_layer.translate)
        spacing = np.array(label_layer.scale)
        ending = starting + spacing * label_layer.data.shape
        bounds = Box3D(starting, ending)

        dx, dy, dz = np.gradient(self.field, *spacing, edge_order=2)
        invalid_grad = dx.mask | dy.mask | dz.mask
        dx[invalid_grad] = 0
        dy[invalid_grad] = 0
        dz[invalid_grad] = 0
        return SimpleVectorField3D(-np.stack([dx, dy, dz], axis=-1), bounds)

    @property
    def method_name(self) -> str:
        return "Wavefront"


# TODO : integrate domain selection in the AStar computation
class AStarContainer(InitFieldContainer):
    @use_case_check_point
    def compute(self) -> bool:
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the wavefront."
            )
            return False
        if "Goal" not in self._viewer.layers:
            notifications.show_error(
                "No goal found. Please select a goal before computing the wavefront."
            )
            return False
        if "Initial positions" not in self._viewer.layers:
            notifications.show_error(
                "No initial positions found. Please select initial positions before computing the wavefront."
            )
            return False
        if (
            self._label_domain_selector.value
            not in self._viewer.layers["Label"].data
        ):
            notifications.show_error(
                f"The selected label is not in the label map. Choose one of {np.unique(self._viewer.layers['Label'].data)}"
            )
            return False
        assert (
            len(self._viewer.layers["Goal"].data) == 1
        ), "Only one goal is allowed"
        label_layer = self._viewer.layers["Label"]
        ## Check if the goal is in a free space
        goal_idx = label_layer.world_to_data(
            self._viewer.layers["Goal"].data[0]
        )
        goal_idx = tuple([round(idx) for idx in goal_idx])
        ## Check if the initial position is in a free space
        init_pos_idx = label_layer.world_to_data(
            self._viewer.layers["Initial positions"].data[0]
        )
        init_pos_idx = tuple([round(idx) for idx in init_pos_idx])

        if (
            label_layer.data[goal_idx] == 0
            or label_layer.data[init_pos_idx] == 0
        ):
            notifications.show_error(
                "The goal and initial positions must be in the free space."
            )
            return False
        path = astar(label_layer.data.astype(bool), init_pos_idx, goal_idx)
        if path is False:
            notifications.show_error("The A* algorithm failed.")
            return False
        path.append(init_pos_idx)
        self._path = path
        logging.info(f"Initial path found with length {len(path)}")

        cost_map = np.ma.masked_array(
            np.zeros(label_layer.data.shape, dtype=np.float32),
            mask=~label_layer.data.astype(bool),
            fill_value=0,
        )
        ## Set the values of the path as the distance to the goal
        for i, p in enumerate(path):
            cost_map[p] = len(path) - i
        ## Create laplace matrix and the bc vector to solve the poisson equation
        laplace_mat, rhs = create_poisson_system(
            cost_map, spacing=label_layer.scale
        )
        ## Solve the system on a subset of the map
        logging.info("Start solving the poisson equation")
        valid_indices = label_layer.data.flat != 0
        A = laplace_mat[valid_indices, :][:, valid_indices]
        b = rhs[valid_indices]
        x, info = splinalg.cg(A, b)
        if info != 0:
            logging.error(f"CG did not converge. Info code : {info}")
            return False
        ## Set the values of the solution to the cost map
        cost_map.flat[valid_indices] = x
        logging.info("Field estimation succeded ! Plotting the solution...")

        self._field = cost_map
        self.visualize(plot_vectors=self._plot_vectors_check.value)
        return True

    def visualize(self, plot_vectors=False) -> TYPE_CHECKING:
        super().visualize(plot_vectors)
        ## Visualise the initial path !
        if "Path" in self._viewer.layers:
            self._viewer.layers.remove("Path")
        self._viewer.add_points(
            self._path,
            size=1,
            face_color="green",
            scale=self._viewer.layers["Label"].scale,
            translate=self._viewer.layers["Label"].translate,
            name="Path",
        )

    @property
    def method_name(self) -> str:
        return "A*"

    @property
    def field(self) -> np.ma.MaskedArray:
        return self._field

    @property
    def vector_field(self) -> VectorField3D:
        if self._field is None:
            notifications.show_error("No wavefront field found.")
            return None
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the wavefront."
            )
            return None
        label_layer = self._viewer.layers["Label"]
        starting = np.array(label_layer.translate)
        spacing = np.array(label_layer.scale)
        ending = starting + spacing * label_layer.data.shape
        bounds = Box3D(starting, ending)

        dx, dy, dz = np.gradient(self.field.data, *spacing, edge_order=2)
        ## Be sure to point toward the goal in the trajectory
        # path_inv goes from the start to the goal
        path_inv = self._path[::-1]
        for i, pos in enumerate(path_inv[:-1]):
            next_pos = path_inv[i + 1]
            dx[pos] = np.array(next_pos[0]) - np.array(pos[0])
            dy[pos] = np.array(next_pos[1]) - np.array(pos[1])
            dz[pos] = np.array(next_pos[2]) - np.array(pos[2])

        return SimpleVectorField3D(np.stack([dx, dy, dz], axis=-1), bounds)


class LaplaceContainer(InitFieldContainer):
    @use_case_check_point
    def compute(self):
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the APF."
            )
            return False
        label_layer = self._viewer.layers["Label"]
        if "Goal" not in self._viewer.layers:
            notifications.show_error(
                "No goal found. Please select a goal before computing the APF."
            )
            return False
        if (
            self._label_domain_selector.value
            not in self._viewer.layers["Label"].data
        ):
            notifications.show_error(
                f"The selected label is not in the label map. Choose one of {np.unique(self._viewer.layers['Label'].data)}"
            )
            return False

        if "Laplace" in self._viewer.layers:
            self._viewer.layers.remove("APF")
        ## Start the computation
        self._compute_button.text = "Computing Laplace field..."
        ## Construct a dilated label map for PDE solve
        domain = label_layer.data == self._label_domain_selector.value
        dilated_domain = ndi.binary_dilation(
            domain, structure=np.ones((3, 3, 3))
        )
        ## First define the field as a masked array
        self._field = np.ma.array(
            np.zeros_like(dilated_domain),
            mask=~dilated_domain,
            fill_value=-np.inf,
            dtype=np.float32,
        )
        goal_position = label_layer.world_to_data(
            self._viewer.layers["Goal"].data[0]
        )
        goal_position = tuple([round(idx) for idx in goal_position])
        # Select the index of the goal position
        # goal_index = sub2ind_3D(label_layer.data.shape, *goal_position)
        boundary_condition = np.zeros_like(dilated_domain)
        boundary_condition[goal_position] = 1
        ## Create the laplacian matrix
        laplace_mat, rhs = create_poisson_system(
            boundary_condition, spacing=label_layer.scale
        )
        ## Solve the system on a subset of the map
        logging.info("Start solving the poisson equation")
        valid_indices = dilated_domain.flat != 0
        A = laplace_mat[valid_indices, :][:, valid_indices]
        b = rhs[valid_indices]
        x = splinalg.spsolve(A, b)
        # x, info = splinalg.cg(A, b)
        # if info != 0:
        #     logging.error(f"CG did not converge. Info code : {info}")
        #     return False
        ## Set the values of the solution to the cost map
        min_value = np.min(x[x > 0])
        self._field.flat[valid_indices] = np.where(
            x > min_value, np.log(x), np.log(min_value)
        )
        logging.info("Field estimation succeded ! Plotting the solution...")

        self.visualize(plot_vectors=self._plot_vectors_check.value)
        return True

    @property
    def method_name(self) -> str:
        return "Laplace"

    @property
    def field(self) -> np.ma.MaskedArray:
        return self._field

    @property
    def vector_field(self) -> VectorField3D:
        if self._field is None:
            notifications.show_error("No wavefront field found.")
            return None
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the wavefront."
            )
            return None
        label_layer = self._viewer.layers["Label"]
        starting = np.array(label_layer.translate)
        spacing = np.array(label_layer.scale)
        ending = starting + spacing * label_layer.data.shape
        bounds = Box3D(starting, ending)

        dx, dy, dz = np.gradient(self.field, *spacing, edge_order=2)
        valid_grad = dx.mask | dy.mask | dz.mask
        dx[valid_grad] = 0
        dy[valid_grad] = 0
        dz[valid_grad] = 0
        grad = np.stack([dx, dy, dz], axis=-1)
        return SimpleVectorField3D(grad.data, bounds)


# TODO : integrate domain selection in the APF computation
class ApfContainer(InitFieldContainer):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(viewer)

        self._resolution_combobox = widgets.ComboBox(
            label="Potential field resolution",
            choices=["1x", "2x", "4x", "8x", "16x"],
            value="1x",
            enabled=False,
        )
        self._attractive_weight_box = widgets.FloatSpinBox(
            min=1,
            max=100,
            step=1,
            value=1,
            label="Attractive weight (unit)",
        )
        self._repulsive_weight_box = widgets.FloatSpinBox(
            min=1,
            max=1000,
            step=1,
            value=1,
            label="Repulsive weight (unit)",
        )
        self._repulsive_radius_box = widgets.FloatSpinBox(
            min=0.1, max=100, value=1, label="Repulsive radius (cm)"
        )
        self._weight_container = widgets.Container(
            widgets=[
                self._resolution_combobox,
                self._attractive_weight_box,
                self._repulsive_weight_box,
                self._repulsive_radius_box,
            ],
            layout="vertical",
        )
        self._weight_container.changed.connect(self.update_apf)
        # self._compute_apf_box = widgets.PushButton(text="Compute APF")

        # self._compute_worker = self._compute_apf()
        # self._compute_worker.returned.connect(self._plot_apf)
        # self._compute_apf_box.changed.connect(self._compute_worker.start)

        self.insert(1, self._weight_container)

        self._attractive_field = None
        self._distance_field = None
        self._bounds = None

    @use_case_check_point
    def compute(self) -> bool:
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the APF."
            )
            return False
        label_layer = self._viewer.layers["Label"]
        if "Goal" not in self._viewer.layers:
            notifications.show_error(
                "No goal found. Please select a goal before computing the APF."
            )
            return False

        if "APF" in self._viewer.layers:
            self._viewer.layers.remove("APF")
        ## Start the computation
        self._compute_button.text = "Computing APF..."
        # self._compute_button.enabled = False

        ## First define the field as a masked array
        self._field = np.ma.array(
            np.zeros_like(label_layer.data),
            mask=~label_layer.data.astype(bool),
            fill_value=np.inf,
            dtype=np.float32,
        )

        goal_position = self._viewer.layers["Goal"].data[0]
        ## Compute the attractive field
        spacing = np.array(label_layer.scale)
        starting = np.array(label_layer.translate) + spacing / 2
        ending = starting + spacing * label_layer.data.shape - spacing / 2
        spacial_grid = np.mgrid[
            starting[0] : ending[0] : spacing[0],
            starting[1] : ending[1] : spacing[1],
            starting[2] : ending[2] : spacing[2],
        ]

        self._attractive_field = 0.5 * (
            (spacial_grid[0] - goal_position[0]) ** 2
            + (spacial_grid[1] - goal_position[1]) ** 2
            + (spacial_grid[2] - goal_position[2]) ** 2
        )

        ## Compute the distance field using the extended label data in order to have gradient values in the domain
        self._distance_field = ndi.distance_transform_edt(
            ndi.binary_dilation(label_layer.data), sampling=label_layer.scale
        )
        if not self.update_apf():
            notifications.show_error(
                "An error occured during the update of the APF."
            )

        self._compute_button.text = "Update APF"
        self._compute_button.enabled = True

        return True

    def update_apf(self) -> bool:
        if (self._distance_field is None) or (self._attractive_field is None):
            notifications.show_error(
                "No existing Artificial Potential Field found. Click on compute APF to generate one."
            )
            return False

        ## Compute the repulsive field based on collision radius
        ## Valid values are the ones inside the object and within the collision radius
        collision_radius = self._repulsive_radius_box.value
        repulsive_field = np.ma.masked_array(
            np.zeros_like(self._distance_field),
            mask=(self._distance_field == 0)
            | (self._distance_field > collision_radius),
            dtype=np.float32,
        )

        repulsive_field[~repulsive_field.mask] = (
            0.5
            * (
                (
                    collision_radius
                    - self._distance_field[~repulsive_field.mask]
                )
                / (
                    collision_radius
                    * self._distance_field[~repulsive_field.mask]
                )
            )
            ** 2
        )

        ## Set the values of the repulsive field to infinity if the distance is 0
        self._field[~self._field.mask] = (
            self._attractive_weight_box.value
            * self._attractive_field[~self._field.mask]
        )
        self._field[~repulsive_field.mask] += (
            self._repulsive_weight_box.value
            * repulsive_field[~repulsive_field.mask]
        )

        ## Visualise the result
        if not self.visualize(self._plot_vectors_check.value):
            notifications.show_error(
                "An error occured during the visualization of the APF."
            )
            return False

        return True

    @property
    def method_name(self) -> str:
        return "APF"

    @property
    def field(self) -> np.ma.MaskedArray:
        return self._field

    @property
    def vector_field(self) -> VectorField3D:
        if self._field is None:
            notifications.show_error("No wavefront field found.")
            return None
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before computing the wavefront."
            )
            return None
        label_layer = self._viewer.layers["Label"]
        starting = np.array(label_layer.translate)
        spacing = np.array(label_layer.scale)
        ending = starting + spacing * label_layer.data.shape
        bounds = Box3D(starting, ending)

        dx, dy, dz = np.gradient(self.field, *spacing, edge_order=2)
        dx[dx.mask] = 0
        dy[dy.mask] = 0
        dz[dz.mask] = 0
        return SimpleVectorField3D(-np.stack([dx, dy, dz], axis=-1), bounds)
        # return VectorField3D(-np.stack([dx, dy, dz], axis=-1), bounds)


# TODO : integrate domain selection in the simulation
class SimulationContainer(widgets.Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        field_container: InitFieldContainer,
    ):
        super().__init__()
        self._viewer = viewer
        self._field_container = field_container
        self._time_slider = widgets.FloatSpinBox(
            min=1,
            max=1_000_000_000,
            value=100,
            step=1,
            label="Simulation final time (s)",
        )
        self._timestep_slider = widgets.FloatSpinBox(
            min=0.01,
            max=1,
            value=1,
            step=0.01,
            label="Simulation time step (s)",
        )
        self._speed_slider = widgets.FloatSpinBox(
            min=0,
            max=1_000,
            value=0,
            step=0.1,
            label="Maximal speed (cm/s)",
        )
        self._diffusivity_slider = widgets.FloatSpinBox(
            min=0,
            max=1_000,
            value=0,
            step=1e-3,
            label="Agent diffusivity (cm^2/s)",
        )
        ## Button to start the simulation
        self._agent_count = widgets.SpinBox(
            label="Number of agents", min=1, max=1_000_000, value=1
        )
        self._reset_button = widgets.ComboBox(
            label="Vector field init",
            value="None",
            choices=["None", "Reset", "Keep best", "Keep lastest"],
        )
        self._reset_button.enabled = False
        self._viewer.layers.events.inserted.connect(
            self._update_reset_button_on_field_init
        )
        ## Widget container
        simu_param_container = widgets.Container(
            widgets=[
                self._time_slider,
                self._timestep_slider,
                self._speed_slider,
                self._diffusivity_slider,
                self._reset_button,
            ],
        )
        # self._reset_button.changed.connect(self._initialize_simulation)
        self._start_button = widgets.PushButton(text="Run simulation")
        self._start_button.changed.connect(self._run_simulation)

        start_container = widgets.Container(
            widgets=[
                self._agent_count,
                self._start_button,
            ],
            layout="horizontal",
        )

        self.extend(
            [
                widgets.Label(label="Simulation parameters"),
                simu_param_container,
                start_container,
            ]
        )
        ## Optimization widgets
        ## Classical optimization parameters
        self._nb_epochs_box = widgets.SpinBox(
            label="Epochs", min=1, max=1_000_000, step=10, value=10
        )
        self._lr_slider = widgets.FloatSpinBox(
            min=1e-4, max=10, value=0.1, step=1e-4, label="Learning rate"
        )
        self._clip_value_slider = widgets.FloatSpinBox(
            min=0,
            max=100,
            value=0,
            step=1e-2,
            label="Clip value",
            tooltip="Gradient clipping value",
        )
        optim_param_container = widgets.Container(
            widgets=[
                self._nb_epochs_box,
                self._lr_slider,
                self._clip_value_slider,
            ],
            layout="horizontal",
        )

        ## Diffusion decrease parameters
        self._diffusion_decrease = widgets.ComboBox(
            label="Diffusion decrease",
            choices=["None", "Linear", "Exponential"],
            value="None",
        )
        self._diffusion_max = widgets.FloatSpinBox(
            label="Diffusion max",
            min=0,
            max=10,
            step=1e-3,
            value=0,
        )
        self._diffusion_min = widgets.FloatSpinBox(
            label="Diffusion min",
            min=0,
            max=10,
            step=1e-3,
            value=0,
        )
        diffusion_container = widgets.Container(
            widgets=[
                self._diffusion_decrease,
                self._diffusion_max,
                self._diffusion_min,
            ],
            layout="horizontal",
        )
        ## Optimization weights parameters
        self._goal_distance = widgets.FloatSpinBox(
            label="Goal weight",
            min=0,
            max=100,
            value=1.0,
            tooltip="Weight of the goal distance in the loss function ||x_i - x_goal||^2",
        )
        self._obstacle_distance = widgets.FloatSpinBox(
            label="Obstacle weight",
            min=0,
            max=10_000,
            value=1.0,
            tooltip="Weight of the obstacle distance in the loss function exp(-||x_i - x_obs||^2)",
        )
        self._bending_constraint = widgets.FloatSpinBox(
            label="Bending weight",
            min=0,
            max=10_000,
            value=1.0,
            tooltip="Weight of the bending constraint in the loss function <F(x_{i+1}), F(x_i)>",
        )
        self._collision_lenght = widgets.FloatSpinBox(
            label="Collision length",
            min=0.1,
            max=10_000,
            value=1.0,
            tooltip="Length of the obstacle collision constraint",
        )
        weights_container = widgets.Container(
            widgets=[
                self._goal_distance,
                self._bending_constraint,
                self._obstacle_distance,
                self._collision_lenght,
            ],
            layout="horizontal",
        )
        ## Button to run the optimization & save the results
        self._run_optimization_button = widgets.PushButton(
            text="Run optimization"
        )
        self._run_optimization_button.changed.connect(self._run_optimization)
        ## Buttons to save the results
        # Export trajectories in the image space
        self._exporter = widgets.FileEdit(
            label="Export trajectories",
            mode="w",
            tooltip="Export the trajectories in image coordinate in a csv file",
        )
        self._exporter.changed.connect(self._export_trajectories)
        ## Export the losses in csv
        self._loss_exporter = widgets.FileEdit(
            label="Export loss",
            mode="w",
            tooltip="Losses for the current optimization in a csv file",
        )
        self._loss_exporter.changed.connect(self._export_loss)
        # Export all the generated datas
        self._save_all_button = widgets.FileEdit(
            label="Save all (.npz)",
            mode="w",
            tooltip="Save all the generated datas in a npz file",
        )
        self._save_all_button.changed.connect(self._save_all)
        save_container = widgets.Container(
            widgets=[
                self._exporter,
                self._loss_exporter,
                self._save_all_button,
            ],
            layout="horizontal",
        )

        self._density_button = widgets.PushButton(text="Estimate density")
        self._density_button.changed.connect(self._compute_density)
        self._bandwidth_slider = widgets.FloatSpinBox(
            label="Bandwidth", min=0.1, max=1, value=0.1, step=1e-1
        )
        density_container = widgets.Container(
            widgets=[self._density_button, self._bandwidth_slider],
            layout="horizontal",
        )
        self.extend(
            [
                widgets.Label(label="Optimization parameters"),
                optim_param_container,
                diffusion_container,
                weights_container,
                widgets.Container(widgets=[self._run_optimization_button]),
                density_container,
                widgets.Label(label="Save results"),
                save_container,
            ]
        )

        self.simulation = None
        self.losses = None
        self._optimized_vector_field = None

        if self._reset_button.value == "None":
            set_simulation_default_values_for_lung(self)

    def _run_simulation(self) -> bool:
        # self._start_button.text = "Running simulation..."
        # self._start_button.enabled = False
        if not self._initialize_simulation():
            notifications.show_error(
                "The simulation could not be initialized."
            )
            return False

        # self.simulation.diffusivity = self._diffusivity_slider.value
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

        trajectories = self.trajectories
        self._viewer.add_tracks(
            trajectories[["trajectory id", "frame index", "x", "y", "z"]],
            properties=trajectories["source"],
            color_by="source",
            name=name.capitalize(),
        )
        if self.nb_agents == 1:
            return True
        ## Plot the mean trajectory
        layer_name = f"Mean {name.lower()}"
        if layer_name in self._viewer.layers:
            self._viewer.layers.remove(layer_name)
        self._viewer.add_tracks(
            self.mean_trajectories[
                ["trajectory id", "frame index", "x", "y", "z"]
            ],
            color_by="track_id",
            colormap="hsv",
            blending="translucent",
            name=layer_name,
            tail_width=20,
            tail_length=self.simulation._nb_steps,
        )
        return True

    def _export_trajectories(self):
        if self.simulation is None:
            notifications.show_error("The simulation is not initialized.")
            return False
        if self._exporter.value == "":
            notifications.show_error("No filename provided.")
            return False

        trajectories = self.trajectories
        label_layer = self._viewer.layers["Label"]

        export_trajectories = pd.DataFrame(
            columns=["trajectory id", "frame index", "x", "y", "z"]
        )
        export_trajectories["trajectory id"] = trajectories["trajectory id"]
        export_trajectories["frame index"] = trajectories["frame index"]
        export_trajectories[["z", "y", "x"]] = (
            trajectories[["x", "y", "z"]] - label_layer.metadata["origin"]
        ) / label_layer.metadata["spacing"]
        filepath = self._exporter.value.with_suffix(".csv")
        export_trajectories.to_csv(filepath, index=False)

        if self.nb_agents == 1:
            return True
        ## Export the mean trajectories
        mean_trajectories = self.mean_trajectories
        export_trajectories = pd.DataFrame(
            columns=["trajectory id", "frame index", "x", "y", "z"]
        )
        export_trajectories["trajectory id"] = mean_trajectories[
            "trajectory id"
        ]
        export_trajectories["frame index"] = mean_trajectories["frame index"]
        export_trajectories[["z", "y", "x"]] = (
            mean_trajectories[["x", "y", "z"]] - label_layer.metadata["origin"]
        ) / label_layer.metadata["spacing"]
        filepath = self._exporter.value.with_suffix(".csv")
        filepath = filepath.with_stem(filepath.stem + "_mean")
        export_trajectories.to_csv(filepath, index=False)
        return True

    def _export_loss(self):
        if self.losses is None:
            notifications.show_error("No losses found.")
            return False
        if self._loss_exporter.value == "":
            notifications.show_error("No filename provided.")
            return False

        losses = pd.DataFrame(self.losses)
        filepath = self._loss_exporter.value.with_suffix(".csv")
        losses.to_csv(filepath, index=False)
        return True

    def _initialize_simulation(self) -> bool:
        if self._reset_button.value == "None":
            vector_field = self.empty_vector_field()
        elif self._reset_button.value == "Reset":
            vector_field = self._field_container.vector_field
            ## Normalize the vector field (because it's too slow otherwise)
            vector_field.normalize()
        elif self._reset_button.value == "Keep best":
            vector_field = self._optimized_vector_field
        elif self._reset_button.value == "Keep lastest":
            vector_field = self.simulation.vector_field

        if vector_field is None:
            notifications.show_error(
                "No initial field found. Please compute the field before running the simulation."
            )
            return False

        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before running the simulation."
            )
            return False
        label_layer = self._viewer.layers["Label"]
        if (
            self._field_container._label_domain_selector.value
            not in label_layer.data
        ):
            notifications.show_error(
                f"The selected label is not in the label map. Choose one of {np.unique(label_layer.data)}"
            )
            return False
        domain = (
            label_layer.data
            == self._field_container._label_domain_selector.value
        )

        distance_field = DistanceField(
            ndi.distance_transform_edt(domain, sampling=label_layer.scale),
            vector_field.bounds,
        )
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

        self.simulation = FocusedWalkers(
            initial_positions,
            goal,
            vector_field,
            distance_field=distance_field,
            domain=label_layer.data,
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
        if self._diffusion_decrease.value == "Linear":
            diffusions = np.linspace(
                self._diffusion_max.value,
                self._diffusion_min.value,
                self._nb_epochs_box.value,
            )
        elif self._diffusion_decrease.value == "Exponential":
            diffusions = np.logspace(
                np.log10(self._diffusion_max.value),
                np.log10(self._diffusion_min.value),
                self._nb_epochs_box.value,
            )
        else:
            diffusions = np.repeat(self.diffusivity, self._nb_epochs_box.value)
        max_iter = self._nb_epochs_box.value
        lr = self._lr_slider.value
        clip_grad = self._clip_value_slider.value
        clip_force = self._speed_slider.value
        goal_weight = self._goal_distance.value
        obstacle_weight = self._obstacle_distance.value
        bend_weight = self._bending_constraint.value
        collision_length = self._collision_lenght.value

        best_loss = np.inf
        self.losses = {
            "total": [],
            "distance": [],
            "bending": [],
            "obstacle": [],
        }
        best_vector_field = self.simulation.vector_field

        for i in range(max_iter):
            self.simulation.reset()
            self.simulation.diffusivity = diffusions[i]
            with ti.ad.Tape(self.simulation.loss):
                self.simulation.run()
                if goal_weight > 0.0:
                    self.simulation.compute_distance_loss(
                        self.simulation.nb_steps - 1
                    )
                if bend_weight > 0.0:
                    self.simulation.compute_bend_loss(min_diff=1e-6)
                if obstacle_weight > 0.0:
                    self.simulation.compute_obstacle_loss(
                        min_diff=1e-6, collision_length=collision_length
                    )
                self.simulation.compute_loss(
                    distance_weight=goal_weight,
                    bend_weight=bend_weight,
                    obstacle_weight=obstacle_weight,
                )
            self.losses["distance"].append(self.simulation.distance_loss[None])
            self.losses["bending"].append(self.simulation.bend_loss[None])
            self.losses["obstacle"].append(self.simulation.obstacle_loss[None])
            self.losses["total"].append(self.simulation.loss[None])

            print(
                f"Iter={i}, Loss={self.simulation.loss[None]}",
                f"\nDistance={self.simulation.distance_loss[None]/self.simulation.nb_walkers} Bending={self.simulation.bend_loss[None]} Obstacle={self.simulation.obstacle_loss[None]}",
            )
            if self.simulation.loss[None] < best_loss:
                best_loss = self.simulation.loss[None]
                best_vector_field = np.copy(
                    self.simulation.vector_field.values
                )
            if clip_grad > 0.0:
                self.simulation.clip_grad(clip_grad)
                # self.simulation.vector_field.norm_clip(clip_grad)
            self.simulation._update_force_field(lr)
            if clip_force > 0.0:
                self.simulation.vector_field.norm_clip(clip_force)
        self._optimized_vector_field = SimpleVectorField3D(
            best_vector_field, self.simulation.vector_field.bounds
        )
        self._plot_trajectories("Optimized trajectories")
        self._plot_final_vector_field("Optimized vector field")
        ## Allow to use the optimized vector field for the simulation
        self._reset_button.enabled = True
        return True

    def _compute_density(self):
        assert self.simulation is not None, "No simulation found."
        if "Density" in self._viewer.layers:
            self._viewer.layers.remove("Density")
        kde = KernelDensity(
            kernel="gaussian", bandwidth=self._bandwidth_slider.value
        )
        kde.fit(self.trajectories[["x", "y", "z"]])
        x, y, z = self.simulation.vector_field.meshgrid
        valid_positions = (
            self._viewer.layers["Label"].data
            == self._field_container._label_domain_selector.value
        )
        positions = np.stack(
            [
                x[valid_positions],
                y[valid_positions],
                z[valid_positions],
            ],
            axis=1,
        )
        temp_density = np.exp(kde.score_samples(positions))
        density = np.zeros_like(valid_positions, dtype=np.float32)
        density[valid_positions] = temp_density

        self._viewer.add_image(
            density,
            name="Density",
            colormap="inferno",
            blending="additive",
            scale=self._viewer.layers["Label"].scale,
            translate=self._viewer.layers["Label"].translate,
            metadata=self._viewer.layers["Label"].metadata,
        )

    def _save_all(self, path: Union[str, Path] = None) -> bool:
        if self._optimized_vector_field is None:
            notifications.show_error(
                "No optimized vector field found. Please run the optimization before saving the data."
            )
            return False
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No label found. Please select a label file before saving the data."
            )
            return False
        if "Image" not in self._viewer.layers:
            notifications.show_error(
                "No image found. Please select an image file before saving the data."
            )
            return False
        if "Goal" not in self._viewer.layers:
            notifications.show_error(
                "No goal found. Please select a goal before saving the data."
            )
            return False
        if "Initial positions" not in self._viewer.layers:
            notifications.show_error(
                "No initial positions found. Please select initial positions before saving the data."
            )
            return False
        if "Initial trajectories" not in self._viewer.layers:
            notifications.show_error(
                "No initial trajectories found. Please run the simulation before saving the data."
            )
            return False
        if "Optimized trajectories" not in self._viewer.layers:
            notifications.show_error(
                "No optimized trajectories found. Please run the optimization before saving the data."
            )
            return False

        path = Path(self._save_all_button.value).resolve()
        label_layer = self._viewer.layers["Label"]

        np.savez_compressed(
            path,
            method=self._field_container.method_name,
            image=self._viewer.layers["Image"].data,
            goal=label_layer.world_to_data(
                self._viewer.layers["Goal"].data[0]
            ),
            init_positions=label_layer.world_to_data(
                self._viewer.layers["Initial positions"].data[0]
            ),
            scalar_field=self._field_container.field.data,
            mask=self._field_container.field.mask,
            vector_field=self._field_container.vector_field.values,
            init_traj=np.array(
                self._viewer.layers["Initial trajectories"].data
            ),
            optimized_vector_field=self.simulation.vector_field.values,
            optimized_trajectories=np.array(
                self._viewer.layers["Optimized trajectories"].data
            ),
            spacing=label_layer.scale,
            origin=label_layer.translate,
            **self.losses,
        )
        notifications.show_info(f"Data saved at {path}")
        return True

    def _plot_final_vector_field(
        self, name: str = "Optimized vector field"
    ) -> bool:
        if self._optimized_vector_field is None:
            notifications.show_error(
                "The optimization code did not run. Can not plot vector field"
            )
            return False
        if name.capitalize() in self._viewer.layers:
            self._viewer.layers.remove(name.capitalize())
        vector_field = self._optimized_vector_field.values

        x, y, z = np.mgrid[
            0 : vector_field.shape[0],
            0 : vector_field.shape[1],
            0 : vector_field.shape[2],
        ]

        valid_map = self._viewer.layers["Label"].data.astype(bool)

        ## Downscale the vector field to avoid too many vectors
        data = np.zeros((valid_map.sum(), 2, 3))
        data[:, 0, 0] = x[valid_map]
        data[:, 0, 1] = y[valid_map]
        data[:, 0, 2] = z[valid_map]
        data[:, 1] = vector_field[valid_map]

        self._viewer.add_vectors(
            data,
            ndim=3,
            name=name.capitalize(),
            scale=self._viewer.layers["Label"].scale,
            translate=self._viewer.layers["Label"].translate,
            metadata=self._viewer.layers["Label"].metadata,
            visible=False,
        )
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

    @property
    def nb_sources(self) -> int:
        return self._viewer.layers["Initial positions"].data.shape[0]

    @property
    def trajectories(self) -> pd.DataFrame:
        if self.simulation is None:
            notifications.show_error("No simulation found.")
            return None
        sim_trajectories = self.simulation.trajectories
        trajectories = pd.DataFrame(
            sim_trajectories,
            columns=["trajectory id", "frame index", "x", "y", "z"],
        )
        trajectories["source"] = np.repeat(
            np.arange(self.nb_sources),
            self.simulation.nb_steps * self.nb_agents,
        ).astype(int)
        return trajectories

    @property
    def mean_trajectories(self) -> pd.DataFrame:
        if self.simulation is None:
            notifications.show_error("No simulation found.")
            return None
        mean_traj = self.simulation.mean_trajectory
        mean_traj = pd.DataFrame(
            mean_traj,
            columns=["trajectory id", "frame index", "x", "y", "z"],
        )
        return mean_traj

    def empty_vector_field(self) -> VectorField3D:
        if "Label" not in self._viewer.layers:
            notifications.show_error(
                "No labels found. Please select a label file before running the optimization."
            )
            return None
        label_layer = self._viewer.layers["Label"]
        starting = np.array(label_layer.translate)
        spacing = np.array(label_layer.scale)
        ending = starting + spacing * label_layer.data.shape
        bounds = Box3D(starting, ending)

        data = np.zeros((*label_layer.data.shape, 3), dtype=np.float32)
        x, y, z = np.nonzero(label_layer.data)
        for w in range(3):
            data[x, y, z, w] = 1e-8 * np.random.randn(len(x))

        return SimpleVectorField3D(data, bounds)

    def _field_init_event(self, event):
        layer = event.source[-1]
        return (layer.name.endswith(" field")
            and layer.name != "Optimized vector field"
            and self._reset_button.value == "None"
        )

    def _update_reset_button_on_field_init(self, event):
        # if _reset_button is "None" and initialization has just completed,
        # change _reset_button's value to previous default "Reset", and thus
        # preserve the original behavior.
        if self._field_init_event(event):
            self._reset_button.value = "Reset"
            logging.info(
                "Resetting simulation and optimization parameters to original defaults."
            )
            reset_simulation_default_values(self)

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

        # self._method_container = AStarContainer(self._viewer)
        # self._method_container = WavefrontContainer(self._viewer)
        # self._method_container = ApfContainer(self._viewer)
        self._method_container = LaplaceContainer(self._viewer)

        self._use_case = DefaultUseCase(type(self._method_container))

        self._simulation_container = SimulationContainer(
            self._viewer, self._method_container
        )
        self.extend(
            [
                self._io_container,
                self._point_container,
                self._method_container,
                self._simulation_container,
            ]
        )
        self.max_width = 700

    def extend(self, components):
        if hasattr(self, "_use_case"):
            self._use_case.involve(components)
        super().extend(components)


class DefaultUseCase(UseCase):
    """
    See PR
    [!5](https://github.com/rcremese/napari-potential-field-navigation/pull/5/files)
    for an explaination of the dependencies between the containers.
    """

    def __init__(self, InitFieldContainer: type):
        super().__init__()
        # Reminder: self._requirements[downstream] = upstream
        # Upstream requirements can be methods or layers
        self._requirements[PointContainer] = [
            "Image",
            "Label",
        ]
        self._requirements[InitFieldContainer] = [
            "Goal",
        ]
        self._requirements[SimulationContainer] = [
            "Initial positions",
            #InitFieldContainer.compute,
        ]
        if InitFieldContainer is AStarContainer:
            self._requirements[InitFieldContainer].append(
                "Initial positions",
            )

    def _disable(self, container: widgets.Container):
        container.enabled = False

    def _enable(self, container: widgets.Container):
        container.enabled = True

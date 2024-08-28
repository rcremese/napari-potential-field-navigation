# napari-potential-field-navigation

[![License Mozilla Public License 2.0](https://img.shields.io/pypi/l/napari-potential-field-navigation.svg?color=green)](https://github.com/rcremese/napari-potential-field-navigation/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-potential-field-navigation.svg?color=green)](https://pypi.org/project/napari-potential-field-navigation)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-potential-field-navigation.svg?color=green)](https://python.org)
[![tests](https://github.com/rcremese/napari-potential-field-navigation/workflows/tests/badge.svg)](https://github.com/rcremese/napari-potential-field-navigation/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-potential-field-navigation)](https://napari-hub.org/plugins/napari-potential-field-navigation)

A simple plugin for generating and visualizing trajectories in napari with a focus on lung navigation in CT scans.

[![Graphical User Interface](https://i.ibb.co/SxtqsrD/screencast2.gif)](https://i.ibb.co/Kj9hHjr/screencast1.gif)

The path finding algorithm it features stands out for its original potential field approach with simulation-based optimization.

## Installation

We recommend you install `napari-potential-field-navigation` in a Python virtual environment.

Note that, although installing `napari-potential-field-navigation` will also install the `napari` package,
napari may not work out-of-the-box on every systems due to its non-Python dependencies.
If you need to install napari, please refer to [napari's documentation](https://napari.org/stable/tutorials/fundamentals/installation.html).

You can install `napari-potential-field-navigation` via [pip]:

    pip install napari-potential-field-navigation



To install the latest development version:

    pip install git+https://github.com/rcremese/napari-potential-field-navigation.git#testing_lung


## Usage

If napari and napari-potential-field-navigation have been installed in a virtual environment, first activate the virtual environment.

Napari with the napari-potential-field-navigation plugin loaded can be launched with:

    napari -w napari-potential-field-navigation


This opens up a napari window.

The intended workflow includes the following steps:

1. data loading,
2. trajectory endpoint selection,
3. potential/vector field initialization,
4. trajectory simulation,
5. vector field optimization,
6. output data export.

### Data loading

Both an image volume file and a label file are required.

We recommend you use the two file selection widgets at the top of the right-side panel.

The image file can alternatively be loaded using the native controls in napari, such as the menu (“File” > “Open file(s)...”) and the viewer (drag-n-drop or copy-paste from a file browser, or Ctrl+O).
In this particular case, you will be prompted for the IO plugin to use to load the file. Select “napari-itk-io” and optionally make it the default plugin.

Label files can only be loaded using the “Label path” widget.

Once both an image file and a label file are loaded, the image is automatically cropped.

### Trajectory endpoint selection

A single target point can be defined, while multiple starting points can be defined.

The target point is defined with the “Select goal” button.

The starting points are defined with the “Select positions” button.

If target and starting points are to be made more clearly distinguishable in the viewer, edit the styling parameters (in particular the “face color”) in the top-left panel.

Note that there is no support for simulations with intermediate goal points. Such simulations may be constructed with as many simulations as goals, connecting the resulting trajectories.

### Vector field initialization

To be simulated, trajectories are modeled as random walks driven by a vector field. This vector field is optimized so that trajectories are led towards the target point (or “goal”), and it needs to be initialized first.

napari-potential-field-navigation currently features a Laplace field procedure to initialize the vector field in the shape of a potential field. The “Laplace field computation” section of the right panel allows launching the initialization procedure, clicking on the “Compute Laplace field” button.

Alternatively, a previously saved potential field can be loaded with the “Load Laplace field” widget.

Note that initialization only requires the target point to be defined.

It is also possible to compute an initial potential field for a specific target point, and change the target point afterwards.

On completing the initialization, the “Image” layer is updated with a colored volume that should exhibit a smooth gradient.

This volume is better visualized in 3D. napari features a “Toggle 2D/3D view” button in the bottom left. Alternatively, you can switch between the 2D and 3D views pressing Ctrl+Y.

### Trajectory simulation

A designated number of trajectories can be generated clicking on the “Run simulation” button, after adjusting parameters in the “Simulation parameters” section.

This can be done before and/or after the optimization step. However, since the optimization procedure involves running simulations at each optimization step, the simulation parameters also have an impact on the optimization and it is worth adjusting their values.

Note that, as an pre-optimization step, the parameters should adjusted so that some trajectories reach the goal, while exploring space as much as possible. Indeed, only the explored parts of the vector field can be optimized.

Once generated, the trajectories are shown in a dedicated napari layer. Hide some of the other layers to improve readability.

To animate the trajectories, a slider appears at the bottom, with a play/pause button on the left of the slider. In the 2D mode, two sliders are shown, one to navigate the image slices, the other one to represent time.

### Vector field optimization

The vector field optimization step modifies the vector field so that the resulting (mean) trajectories exhibit less bending. Note however that the individual trajectories will always exhibit a random walk.

Running the optimization is done clicking on the “Run optimization” button. On completion, several additional napari layers are added to represent the optimized vector field as well as some trajectories generated with the optimized field.

The “Optimization parameters” section features additional parameters. Hover the labels to get a description.

### Output data export

The optimized trajectories can be saved with the “Export trajectories” widget.


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [Mozilla Public License 2.0] license,
"napari-potential-field-navigation" is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/rcremese/napari-potential-field-navigation/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

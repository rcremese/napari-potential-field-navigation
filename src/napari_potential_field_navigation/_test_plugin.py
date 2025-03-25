import napari

from napari_potential_field_navigation._widget import DiffApfWidget


def main():
    viewer = napari.Viewer()
    widget = DiffApfWidget(viewer)
    viewer.window.add_dock_widget(widget, name="first_plugin", area="right")

    napari.run()


if __name__ == "__main__":
    main()

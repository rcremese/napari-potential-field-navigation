import napari

from napari_potential_field_navigation._widget import IoContainer


def main():
    with napari.gui_qt():
        viewer = napari.Viewer()
        widget = IoContainer(viewer)
        dw1 = viewer.window.add_dock_widget(
            widget, name="first_plugin", area="right"
        )
        dw1.NoDockWidgetFeatures = 1
        napari.run()


if __name__ == "__main__":
    main()

import napari

from napari_potential_field_navigation._widget import DiffApfWidget


def main():
    viewer = napari.Viewer()
    widget = DiffApfWidget(viewer)
    viewer.window.add_dock_widget(widget, name="first_plugin", area="right")

    napari.run()


if __name__ == "__main__":
    main()
## Test model selection with stacked widgets
# from PyQt5.QtWidgets import (
#     QApplication,
#     QWidget,
#     QVBoxLayout,
#     QComboBox,
#     QStackedWidget,
#     QLabel,
# )


# class AdaptableWindow(QWidget):
#     def __init__(self):
#         super().__init__()

#         # Create a combo box for selecting the method
#         self.comboBox = QComboBox()
#         self.comboBox.addItems(["Method 1", "Method 2", "Method 3"])

#         # Create a stacked widget for holding the method widgets
#         self.stackedWidget = QStackedWidget()

#         # Add the method widgets to the stacked widget
#         self.stackedWidget.addWidget(QLabel("Method 1 selected"))
#         self.stackedWidget.addWidget(QLabel("Method 2 selected"))
#         self.stackedWidget.addWidget(QLabel("Method 3 selected"))

#         # Create a layout and add the combo box and stacked widget to it
#         layout = QVBoxLayout()
#         layout.addWidget(self.comboBox)
#         layout.addWidget(self.stackedWidget)
#         self.setLayout(layout)

#         # Connect the combo box's currentIndexChanged signal to the changeMethod slot
#         self.comboBox.currentIndexChanged.connect(self.changeMethod)

#     def changeMethod(self, index):
#         # Change the visible widget in the stacked widget to the selected method
#         self.stackedWidget.setCurrentIndex(index)


# if __name__ == "__main__":
#     app = QApplication([])
#     window = AdaptableWindow()
#     window.show()
#     app.exec_()

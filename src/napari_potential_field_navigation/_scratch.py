import napari
from skimage import data
import numpy as np

hubble_image = data.hubble_deep_field()

tracks_data = np.asarray(
    [
        [1, 0, 236, 0],
        [1, 1, 236, 100],
        [1, 2, 236, 200],
        [1, 3, 236, 500],
        [1, 4, 236, 1000],
        [2, 0, 436, 0],
        [2, 1, 436, 100],
        [2, 2, 436, 200],
        [2, 3, 436, 500],
        [2, 4, 436, 1000],
        [3, 0, 636, 0],
        [3, 1, 636, 100],
        [3, 2, 636, 200],
        [3, 3, 636, 500],
        [3, 4, 636, 1000],
    ]
)
track_confidence = np.array(5 * [0.9] + 5 * [0.3] + 5 * [0.1])
properties = {"time": tracks_data[:, 1], "confidence": track_confidence}

viewer = napari.view_image(hubble_image)
viewer.add_tracks(tracks_data, properties=properties)
napari.run()

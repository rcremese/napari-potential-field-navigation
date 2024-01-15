"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy
from napari_itk_io._reader import reader_function


def make_sample_data(dir_path: str):
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    return [(numpy.random.rand(512, 512), {})]


def open_samples():
    """Open image and label samples"""
    dirpath = (
        Path(__file__).parents[2].joinpath("sample_datas").resolve(strict=True)
    )
    image_layers = reader_function(dirpath.joinpath("image.nii.gz"))
    image_layers[0][1]["name"] = "Image"

    temp_layers = reader_function(dirpath.joinpath("label.nii.gz"))
    labels = temp_layers[0][0].astype(int)
    labels_metadata = temp_layers[0][1]
    labels_metadata.pop("channel_axis", None)
    labels_metadata.pop("rgb", None)
    labels_metadata["name"] = "Label"
    label_layers = [(labels, labels_metadata, "labels")]
    image_layers.extend(label_layers)
    return image_layers

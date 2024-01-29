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
    labels_metadata["blending"] = "translucent_no_depth"
    label_layers = [(labels, labels_metadata, "labels")]
    image_layers.extend(label_layers)
    return image_layers

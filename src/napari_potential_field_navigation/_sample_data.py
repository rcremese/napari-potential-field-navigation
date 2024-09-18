"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from napari_itk_io._reader import reader_function


def lung_sample():
    dirpath = (
        Path(__file__)
        .parents[2]
        .joinpath("sample_datas", "Lung")
        .resolve(strict=True)
    )
    return open_samples(
        dirpath.joinpath("image.nii.gz"), dirpath.joinpath("label.nii.gz")
    )


def liver_sample():
    dirpath = (
        Path(__file__)
        .parents[2]
        .joinpath("sample_datas", "Liver")
        .resolve(strict=True)
    )
    return open_samples(
        dirpath.joinpath("image.nii.gz"), dirpath.joinpath("label.nii.gz")
    )


def open_samples(image_path: Path, label_path: Path):
    """Open image and label samples"""
    assert image_path.exists(), f"File not found: {image_path}"
    assert label_path.exists(), f"File not found: {label_path}"

    image_layers = reader_function(image_path)
    image_metadata = image_layers[0][1]
    image_metadata["name"] = "Image"

    temp_layers = reader_function(label_path)
    labels = temp_layers[0][0].astype(int)
    labels_metadata = temp_layers[0][1]
    labels_metadata.pop("channel_axis", None)
    labels_metadata.pop("rgb", None)

    # Automatic cropping based on label values
    nonzero = np.where(labels != 0)
    min_xyz = (nonzero[0].min(), nonzero[1].min(), nonzero[2].min())
    max_xyz = (nonzero[0].max(), nonzero[1].max(), nonzero[2].max())
    crop_slice = tuple(
        slice(min, max + 1) for min, max in zip(min_xyz, max_xyz)
    )
    # Take into account the shift of origin
    new_origin = (
        labels_metadata["translate"]
        + np.array(min_xyz) * labels_metadata["scale"]
    )
    labels_metadata["translate"] = new_origin
    image_metadata["translate"] = new_origin
    ## Crop image and labels
    cropped_image = image_layers[0][0][crop_slice]
    cropped_labels = labels[crop_slice]
    ## Add cropped labels and image to viewer
    labels_metadata["name"] = "Label"
    labels_metadata["blending"] = "translucent_no_depth"
    label_layers = [(cropped_labels, labels_metadata, "labels")]
    image_layers[0] = (cropped_image, image_metadata, "image")
    image_layers.extend(label_layers)
    return image_layers

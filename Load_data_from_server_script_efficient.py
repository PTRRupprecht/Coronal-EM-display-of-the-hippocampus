from tifffile import imwrite
import webknossos as wk
import numpy as np

DATASET_NAME = "MCMV_2_3_CA1_LH_v1"
LAYER_NAME = "color"
MAG = wk.Mag("4-4-1")  # Correct magnification
TARGET_Z = 3583# 2979  # Target absolute Z coordinate

with wk.webknossos_context(token="523hGlOELWfrD6VI3VjR-w", url="https://webknossos.brain.mpg.de"):
    dataset = wk.Dataset.open_remote(
        dataset_name_or_url=DATASET_NAME,
        organization_id="Connectomics_Department",
        webknossos_url="https://webknossos.brain.mpg.de",
    )

    mag_view = dataset.get_layer(LAYER_NAME).get_mag(MAG)
    bbox = mag_view.bounding_box

    # Compute local Z index
    local_z = int(TARGET_Z / MAG.z)

    # Make sure we align to the full chunk (which is 2 slices in Z)
    # Find the starting Z that aligns to 2
    aligned_z_start = (local_z // 2) * 2

    slice_bbox = wk.BoundingBox(
        topleft=(bbox.topleft.x, bbox.topleft.y, aligned_z_start),
        size=(bbox.size.x, bbox.size.y, 2),  # size Z=2
        axes=("x", "y", "z"),
    )

    slice_bbox = slice_bbox.align_with_mag(MAG)

    # Now we have a box of 2 slices
    slice_view = mag_view.get_view(size=slice_bbox.size, absolute_offset=slice_bbox.topleft)

    # Read it
    slice_data = slice_view.read()




    # If there are channels, take the first
    if slice_data.ndim == 4:
        slice_data = slice_data[0]

    # slice_data shape: (X, Y, Z)

    # Now pick the correct Z-plane
    z_index_in_block = local_z - aligned_z_start  # Either 0 or 1
    selected_slice = slice_data[:, :, 0]

    # Save
    imwrite(f"Slice_{MAG.z}_z{TARGET_Z}.tiff", selected_slice.T)  # transpose if necessary


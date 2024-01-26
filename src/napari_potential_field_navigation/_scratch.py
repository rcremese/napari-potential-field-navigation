import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt

image_path = Path("F:/JBM lab/20221007 c19 test.tif").resolve(strict=True)
with tifffile.TiffFile(image_path) as tif:
    nb_slice = len(tif.pages)

    central_page = tif.pages[nb_slice // 2]
    image = central_page.asarray()

    values = np.zeros(nb_slice)
    for i, page in enumerate(tif.pages):
        print(page.is_mask)
        values[i] = page.asarray().mean()
        print(f"Slide {i} - mean value : {values[i]}")

print(image.shape)
print(values)
plt.imshow(image, cmap="gray")
plt.hist(values, bins=100)
plt.show()

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# from monai.data.itk_torch_bridge import (
#     itk_image_to_metatensor,
# )
# im = itk.imread(
#     r"D:\Datas\Medical\Lung\ATM_Challenge_2022\labelsTr\ATM_003_0000.nii"
# )
# label = itk_image_to_metatensor(im)
# print(label)

from PIL import Image
import numpy as np
# # 创建一个纯红色的图像
# img_path="E:\Thesis2025\\all_clip_256\\filter2560\images\GF2_PMS1__L1A0000564539-MSS1_image_0_1536.tif"
# img = Image.open(img_path)

# # 获取像素值
# pixel = img.getpixel((0, 0))
# print("Pixel value:", pixel)

# 读取 .npy 文件
loaded_image = np.load("E:\Thesis2025\SWED\SWED\\train\\filter_images\S2A_MSIL2A_20170409T105651_N0204_R094_T30UYC_20170409T110529_image_8_37.npy")
loaded_mask=np.load("E:\Thesis2025\SWED\SWED\\train\\filter_masks\S2A_MSIL2A_20170409T105651_N0204_R094_T30UYC_20170409T110529_chip_8_37.npy")
print("Loaded image shape:", loaded_mask.shape)
print("Pixel value:", loaded_mask[0, 0])# 保存为 .npy 文件
loaded_image=loaded_image[:,:,[3,2,1]]
# 查看像素值
print("Loaded image shape:", loaded_image.shape)
print("Pixel value:", loaded_image[0, 0])# 保存为 .npy 文件



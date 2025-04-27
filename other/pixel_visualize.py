import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal
import os
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True 
Image.MAX_IMAGE_PIXELS = None
def show_image_with_probability_and_nodata(original_image, image, pixel_value, nodata_mask=None):
    """
    显示图像、概率图和无数据区域，并在光标处显示概率值
    """
    def on_mouse_move(event):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < pixel_value.shape[1] and 0 <= y < pixel_value.shape[0]:
                prob_value = pixel_value[y, x]
                if nodata_mask is not None and nodata_mask[y, x] > 0:
                    ax.set_title(f"Pixel at ({x}, {y}): nodata")
                else:
                    ax.set_title(f"Pixel at ({x}, {y}): {prob_value:.4f}")
                fig.canvas.draw_idle()
    
    # 调整图像大小
    original_image = original_image.resize((256, 256))
    image = image.resize((256, 256))
    
    # 确保概率图也是正确的大小
    probability_map = np.array(pixel_value)
    if probability_map.shape[:2] != (256, 256):
        probability_map = np.resize(probability_map, (256, 256))
    
    # 调整无数据掩码大小（如果有）
    if nodata_mask is not None and nodata_mask.shape[:2] != (256, 256):
        nodata_mask = np.resize(nodata_mask, (256, 256))
    
    # 创建图形
    if nodata_mask is not None:
        fig = plt.figure(figsize=(15, 5))
        
        # 原始图像
        ax_original = fig.add_subplot(1, 3, 1)
        ax_original.imshow(original_image)
        ax_original.set_title("original")
        ax_original.axis('off')
        
        # 预测图像和概率叠加
        ax = fig.add_subplot(1, 3, 2)
        image = image.convert('L')  # 转为灰度图
        ax.imshow(image, cmap='gray')
        prob_img = ax.imshow(probability_map, cmap='viridis', alpha=0.5)
        fig.colorbar(prob_img, ax=ax, label="probability")
        ax.set_title("prediction & probability")
        
        # 显示无数据掩码
        ax_mask = fig.add_subplot(1, 3, 3)
        ax_mask.imshow(nodata_mask, cmap='Reds', alpha=0.7)
        ax_mask.set_title("nodata")
        ax_mask.axis('off')
    else:
        fig = plt.figure(figsize=(12, 6))
        
        # 原始图像
        ax_original = fig.add_subplot(1, 2, 1)
        ax_original.imshow(original_image)
        ax_original.set_title("original")
        ax_original.axis('off')
        
        # 预测图像和概率叠加
        ax = fig.add_subplot(1, 2, 2)
        image = image.convert('L')  # 转为灰度图
        ax.imshow(image, cmap='gray')
        prob_img = ax.imshow(probability_map, cmap='viridis', alpha=0.5)
        fig.colorbar(prob_img, ax=ax, label="probability")
        ax.set_title("prediction & probability")
    
    # 添加鼠标移动事件
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载图像、预测结果和无数据掩码
    original_image_path = "E:\imagedata\\new\s2test\\enhanced_s2_202305060000000000-0000040704.tif"
    original_image = Image.open(original_image_path)
    
    image_path ="E:\\imagedata\\2305-06\\combined_image_dl_v1.tif"
    image = Image.open(image_path)
    
    pred_array_path = "E:\imagedata\\2305-06\combined_array_dl_v1.npy"
    probability_map = np.load(pred_array_path)
    
    #尝试加载无数据掩码（如果存在）
    nodata_mask_path = "E:\\imagedata\\2305-06\\combined_image_nodata_dl_v1.tif"
    nodata_mask = None
    if os.path.exists(nodata_mask_path):
        try:
            nodata_ds = gdal.Open(nodata_mask_path, gdal.GA_ReadOnly)
            if nodata_ds:
                nodata_mask = nodata_ds.GetRasterBand(1).ReadAsArray()
                nodata_ds = None
                print(f"已加载无数据掩码: {nodata_mask_path}")
        except Exception as e:
            print(f"加载无数据掩码出错: {str(e)}")
    
    show_image_with_probability_and_nodata(original_image, image, probability_map, nodata_mask)
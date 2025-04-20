import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
def show_image_with_probability(original_image,image, probability_map):
    """
    显示图像和概率图，并在光标处显示概率值
    """
    def on_mouse_move(event):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < probability_map.shape[1] and 0 <= y < probability_map.shape[0]:
                prob_value = probability_map[y, x]
                ax.set_title(f"Probability at ({x}, {y}): {prob_value:.4f}")
                fig.canvas.draw_idle()
    original_image = original_image.resize((256, 256))
    image = image.resize((256, 256))
    probability_map = np.array(probability_map)
    probability_map = np.resize(probability_map, (256, 256))
    fig, ax = plt.subplots()
    ax_original = fig.add_subplot(1, 2, 1)
    ax_original.imshow(original_image)
    ax_original.set_title("Original Image")
    ax_original.axis('off')
    ax = fig.add_subplot(1, 2, 2)

    image = image.convert('L')  # Convert image to grayscale
    ax.imshow(image, cmap='gray')
    fig.colorbar(ax.imshow(probability_map, cmap='viridis', alpha=0.5), ax=ax, label="Probability")
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    plt.show()

# 示例用法
if __name__ == "__main__":
    # 创建一个随机图片和概率图
    original_image_path = "E:\\imagedata\\2305-06\\clipped_images\\test\\s2_20230506_enhanced_image_13312_4608.tif"
    original_image = Image.open(original_image_path)  # 随机 RGB 图片
    image_path="E:\\imagedata\\2305-06\\clipped_images\\prediction\\pred_trans_s2_20230506_enhanced_image_13312_4608.tif"
    image = Image.open(image_path)  # 随机 RGB 图片
    pred_array_path = "E:\\imagedata\\2305-06\\clipped_images\\prediction\\pred_array_s2_20230506_enhanced_image_13312_4608.npy"
    probability_map = np.load(pred_array_path)# 随机概率图

    show_image_with_probability(original_image,image, probability_map)
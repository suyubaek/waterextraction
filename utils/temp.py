import os
import cv2
import numpy as np

folder = '/Users/songyufei1/Documents/look_code_syf/water-extract-dl/data/water_seg'  # 替换为你的文件夹路径

for fname in os.listdir(folder):
    if fname.lower().endswith('.png'):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"{fname}: 图像读取失败")
            continue
        # 检查像素是否全为0
        if np.all(img == 0):
            print(f"{fname}: 全为0，删除该png及同名tif")
            os.remove(img_path)
            tif_name = os.path.splitext(fname)[0] + '.tif'
            tif_path = os.path.join(folder, tif_name)
            if os.path.exists(tif_path):
                os.remove(tif_path)
                print(f"{tif_name}: 已删除")
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 读取源图像和目标图像
root_dir = 'F:\\Picture04\\'
chip = 'chip1'
#day = '4'
#source_dir = root_dir + chip + '\\' + day + '\\'
#output_dir = root_dir + chip + '\\' + day + "-matched\\"

source_dir = root_dir + chip + '\\' +  '\\'
output_dir = root_dir + chip + '-matched\\'

os.makedirs(output_dir, exist_ok=True)

target_img = cv2.imread('F:\\Picture04\\5.png', cv2.IMREAD_GRAYSCALE)

target_hist, bins2 = np.histogram(target_img, bins=256, range=[0, 256])
target_cdf = target_hist.cumsum()
target_cdf = (target_cdf / target_cdf[-1]).astype(np.float32)

# 遍历源目录中的所有图片
for filename in os.listdir(source_dir):
    source_path = os.path.join(source_dir, filename)

    # 确保文件是图片
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        continue

    # 读取源图像
    source_img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

    # 计算源图像的直方图
    source_hist, _ = np.histogram(source_img, bins=256, range=[0, 256])
    source_cdf = source_hist.cumsum()
    source_cdf = (source_cdf / source_cdf[-1]).astype(np.float32)

    # 使用累积直方图进行直方图匹配
    matched_values = np.interp(source_cdf, target_cdf, np.arange(256)).astype(np.uint8)
    matched_img = matched_values[source_img]

    # 保存匹配后的图像
    matched_path = os.path.join(output_dir, filename)
    #cv2.imwrite(matched_path, matched_img)
    cv2.imwrite(matched_path, source_img)
    print(f"Processed and saved: {matched_path}")

print(f"All images have been processed and saved to {output_dir}")
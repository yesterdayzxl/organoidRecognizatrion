import cv2
import numpy as np
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def laplacian_pyramid_fusion(images):
    # 确保所有图片尺寸相同
    h, w = images[0].shape[:2]
    images = [cv2.resize(img, (w, h)) for img in images]

    # 创建拉普拉斯金字塔
    laplacian_pyramids = []

    for img in images:
        gaussian_pyramid = [img]
        for _ in range(5):  # 生成五层金字塔
            gaussian_pyramid.append(cv2.pyrDown(gaussian_pyramid[-1]))

        laplacian_pyramid = []
        for i in range(len(gaussian_pyramid) - 1):
            next_gaussian = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            laplacian = cv2.subtract(gaussian_pyramid[i], next_gaussian)
            laplacian_pyramid.append(laplacian)

        laplacian_pyramids.append(laplacian_pyramid)

    # 融合拉普拉斯金字塔
    fused_pyramid = []
    for i in range(len(laplacian_pyramids[0])):
        max_layer = np.max([laplacian_pyramids[j][i] for j in range(len(laplacian_pyramids))], axis=0)
        fused_pyramid.append(max_layer)

    # 从融合的拉普拉斯金字塔重建图像
    fused_image = fused_pyramid[-1]
    for i in range(len(fused_pyramid) - 2, -1, -1):
        # 确保目标尺寸正确
        target_size = (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0])
        fused_image = cv2.pyrUp(fused_image, dstsize=target_size)

        # 确保尺寸匹配
        if fused_image.shape != fused_pyramid[i].shape:
            fused_image = cv2.resize(fused_image, (fused_pyramid[i].shape[1], fused_pyramid[i].shape[0]))

        fused_image = cv2.add(fused_image, fused_pyramid[i])

    return fused_image

folder_path = 'G:\\z-stack'  # 修改为你的文件夹路径
images = load_images_from_folder(folder_path)
if images:
    fused_image = laplacian_pyramid_fusion(images)
    cv2.imwrite('fused_image.png', fused_image)
    cv2.imshow('Fused Image', fused_image)
    cv2.imwrite("pyramid.png", fused_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("没有找到图片。")

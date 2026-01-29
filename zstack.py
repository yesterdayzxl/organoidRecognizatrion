import cv2
import numpy as np
import glob


# 读取图像文件 (PNG格式)
def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(f"{folder}/*.png"):  # 寻找PNG图像
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images


# Z-Stack算法 - 最大强度投影
def max_intensity_projection(images):
    # 假设所有图像的大小相同
    stacked_image = np.max(np.array(images), axis=0)
    return stacked_image

# Z-Stack算法 - 平均强度投影
def mean_intensity_projection(images):
    # 将所有图像在 Z 轴方向求平均
    stacked_image = np.mean(np.array(images), axis=0).astype(np.uint8)
    return stacked_image


# Z-Stack算法 - 中值强度投影
def median_intensity_projection(images):
    # 取每个像素点的中值
    stacked_image = np.median(np.array(images), axis=0).astype(np.uint8)
    return stacked_image

def generate_gaussian_pyramid(image, levels):
    """生成高斯金字塔"""
    gaussian_pyramid = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    return gaussian_pyramid

def generate_laplacian_pyramid(gaussian_pyramid):
    """生成拉普拉斯金字塔"""
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        next_gaussian = cv2.pyrUp(gaussian_pyramid[i + 1])
        if next_gaussian.shape != gaussian_pyramid[i].shape:
            next_gaussian = cv2.resize(next_gaussian, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        laplacian = cv2.subtract(gaussian_pyramid[i], next_gaussian)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])  # 顶层的高斯金字塔
    return laplacian_pyramid

def blend_pyramids(lap_pyr_A, lap_pyr_B, mask_pyr):
    """融合拉普拉斯金字塔"""
    fused_pyramid = []
    for la, lb, mask in zip(lap_pyr_A, lap_pyr_B, mask_pyr):
        fused = mask * la + (1 - mask) * lb
        fused_pyramid.append(fused)
    return fused_pyramid

#拉普拉斯金字塔融合
def laplacian_pyramid_fusion(images):
    # 假设所有输入图像大小相同
    num_images = len(images)
    depth = 4  # 金字塔的层数，可以根据需求调整
    gaussian_pyramids = []
    laplacian_pyramids = []

    # 构建所有图像的高斯金字塔
    for img in images:
        gaussian_pyramid = [img]
        for _ in range(depth):
            img = cv2.pyrDown(img)
            gaussian_pyramid.append(img)
        gaussian_pyramids.append(gaussian_pyramid)

    # 构建所有图像的拉普拉斯金字塔
    for i in range(num_images):
        laplacian_pyramid = []
        for j in range(depth, 0, -1):
            expanded = cv2.pyrUp(gaussian_pyramids[i][j])
            # 确保尺寸一致
            if expanded.shape != gaussian_pyramids[i][j - 1].shape:
                expanded = cv2.resize(expanded,
                                      (gaussian_pyramids[i][j - 1].shape[1], gaussian_pyramids[i][j - 1].shape[0]))
            laplacian = cv2.subtract(gaussian_pyramids[i][j - 1], expanded)
            laplacian_pyramid.append(laplacian)
        laplacian_pyramids.append(laplacian_pyramid)

    # 初始化融合后的拉普拉斯金字塔
    fused_pyramid = []

    # 对每个金字塔层进行融合
    for level in range(depth):
        fused_layer = np.zeros_like(laplacian_pyramids[0][level])
        for i in range(num_images):
            fused_layer = cv2.add(fused_layer, laplacian_pyramids[i][level])  # 简单相加融合策略
        fused_pyramid.append(fused_layer // num_images)  # 取平均值

    # 对底层高斯图像进行融合
    fused_gaussian = np.zeros_like(gaussian_pyramids[0][depth])
    for i in range(num_images):
        fused_gaussian = cv2.add(fused_gaussian, gaussian_pyramids[i][depth])
    fused_gaussian //= num_images  # 取平均值

    # 重建图像，从拉普拉斯金字塔和底层高斯图像开始
    fused_image = fused_gaussian
    for level in range(depth - 1, -1, -1):
        # 使用 pyrUp 前先确保尺寸匹配
        fused_image = cv2.pyrUp(fused_image)
        if fused_image.shape != fused_pyramid[level].shape:
            fused_image = cv2.resize(fused_image, (fused_pyramid[level].shape[1], fused_pyramid[level].shape[0]))
        fused_image = cv2.add(fused_image, fused_pyramid[level])

    return fused_image

#基于焦点检测的图像融合
def focus_stacking(images):
    focus_measure = [cv2.Laplacian(img, cv2.CV_64F) for img in images]
    focus_stack = np.argmax(np.array(focus_measure), axis=0)
    stacked_image = np.zeros_like(images[0])

    # 根据聚焦度最高的图像选择对应像素值
    for i in range(len(images)):
        stacked_image[focus_stack == i] = images[i][focus_stack == i]

    return stacked_image

#最大拉普拉斯融合（Max Laplacian Fusion）
def max_laplacian_fusion(images):
    laplacians = [cv2.Laplacian(img, cv2.CV_64F) for img in images]
    max_laplacian_idx = np.argmax(laplacians, axis=0)
    stacked_image = np.zeros_like(images[0])

    # 根据最大拉普拉斯值选择对应的像素
    for i in range(len(images)):
        stacked_image[max_laplacian_idx == i] = images[i][max_laplacian_idx == i]

    return stacked_image


# 保存结果图像 (PNG格式)
def save_image(image, filename):
    cv2.imwrite(filename, image)


# 示例代码
if __name__ == "__main__":
    folder_path = r'G:\MHS\2' # zstack 文件夹路径
    images = load_images_from_folder(folder_path)

    if len(images) > 0:
        # 执行Z-Stack算法
        result_image1 = max_intensity_projection(images)
        result_image2 = mean_intensity_projection(images)
        result_image3 = median_intensity_projection(images)
        result_image4 = laplacian_pyramid_fusion(images)
        result_image5 = focus_stacking(images)
        result_image6 = max_laplacian_fusion(images)

        # 保存结果 (PNG格式)
        save_image(result_image1, 'z_stack_result1.png')
        save_image(result_image2, 'z_stack_result2.png')
        save_image(result_image3, 'z_stack_result3.png')
        save_image(result_image4, 'z_stack_result4.png')
        save_image(result_image5, 'z_stack_result5.png')
        save_image(result_image6, 'z_stack_result6.png')
        print("Z-Stack result saved as 'z_stack_result.png'")
    else:
        print("No images found in the folder.")

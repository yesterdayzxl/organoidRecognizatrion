from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO('G:\\Projects\\CellImaging\\活细胞成像\\Project\\runs\\segment\\train51\\weights\\best.pt')

# Define path to the directory containing image files
source = 'G:\\2\\41.png'

# 指定输出路径
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取待预测文件名称，用于保存同名文件
def get_last_part_of_string(path):
    return os.path.basename(path)

# Hex to BGR
def hex_to_bgr(hex):
    return tuple(int(hex[i:i + 2], 16) for i in (4, 2, 0))

# 颜色，同plotting.py的设置
hexs = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231",
    "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
    "2C99A8", "00C2FF", "344593", "6473FF", "0018EC",
    "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"
)
colors = [hex_to_bgr(h) for h in hexs]

# 开始预测
results = model(source=source, save=True, show=True, retina_masks=True)  # list of Results objects

for result in results:
    image_path = result.path
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

    # 将灰度图扩展到16位范围
    gray_image_16bit = (gray_image.astype(np.uint16) * 256)  # 将灰度值扩展到 0-65535

    boxes = result.boxes  # Boxes 对象，用于边界框输出
    masks = result.masks  # Masks 对象，用于分割掩码输出
    names = result.names  # 获取类别名称字典

    for box, mask in zip(boxes, masks):
        for cls, contour, conf in zip(box.cls, mask.xy, box.conf):
            confidence = conf.item()  # 获取置信度
            if confidence > 0.7:
                class_id = int(cls.item())  # 获取张量的值并转换为整数
                color = colors[class_id % len(colors)]  # 获取颜色
                contour = np.array(contour, dtype=np.int32)  # 确保轮廓是整数类型
                area = cv2.contourArea(contour)  # 计算轮廓面积
                class_name = names[class_id]  # 获取类别名称


                # 计算轮廓的中心
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = 0, 0

                # 创建掩码以提取感兴趣区域
                mask_img = np.zeros_like(gray_image_16bit, dtype=np.uint16)  # 确保掩码与灰度图大小和数据类型一致
                cv2.drawContours(mask_img, [contour], -1, 1, -1)  # 填充轮廓区域，值设为1以方便后续操作

                # 计算掩码区域的灰度值
                masked_gray = gray_image_16bit * mask_img  # 使用掩码提取灰度区域

                # 在掩码内计算灰度的平均值和标准差
                mean_val, std_val = cv2.meanStdDev(gray_image_16bit,
                mask=(mask_img > 0).astype(np.uint8))  # OpenCV需要uint8掩码

                # 绘制掩码轮廓
                cv2.drawContours(image, [contour], -1, color, 10)

                 # 在图像上绘制面积、类名和置信度
                text = (f'{class_name} {area:.2f} Conf: {confidence:.2f} '
                        f'Mean: {mean_val[0][0]:.2f} Std: {std_val[0][0]:.2f}')

                cv2.putText(image, text, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 保存图像
    gray_output_path = os.path.join(output_dir, 'gray_' + get_last_part_of_string(image_path))
    annotated_output_path = os.path.join(output_dir, get_last_part_of_string(image_path))
    cv2.imwrite(gray_output_path, gray_image_16bit)
    cv2.imwrite(annotated_output_path, image)
    print(f'Saved annotated image: {annotated_output_path}')
    print(f'Saved 16-bit gray image: {gray_output_path}')
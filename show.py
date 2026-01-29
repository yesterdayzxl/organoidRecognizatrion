from ultralytics import YOLO
import cv2
import os
import numpy as np
import onnx
# Load a pretrained YOLOv8n model
model = YOLO('G:\\活细胞成像\\Project\\runs\\segment\\train43\\weights\\best.pt')
#model = onnx.load('G:\\活细胞成像\\Project\\runs\\segment\\train26\\weights\\best.onnx')
# Define path to the directory containing image files
source_dir = 'G:\\活细胞成像\\Project\\ImagesToTest'

# 指定输出路径
output_dir = 'output_images00'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 获取待预测文件名称，用于保存同名文件
def get_last_part_of_string(path):
    return os.path.basename(path)


# hex to BGR
def hex_to_bgr(hex):
    return tuple(int(hex[i:i + 2], 16) for i in (4, 2, 0))


# 颜色，同 plotting.py 的设置
hexs = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231",
    "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
    "2C99A8", "00C2FF", "344593", "6473FF", "0018EC",
    "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"
)
colors = [hex_to_bgr(h) for h in hexs]

# 遍历目录中的所有图像文件并进行推理
for root, _, files in os.walk(source_dir):
    for file in files:
        # 仅处理图片文件
        if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            source = os.path.join(root, file)

            # 开始预测
            results = model(source=source, save=True, retina_masks=True)  # list of Results objects

            for result in results:
                # 如果没有检测到任何对象，跳过该图像
                if not result.boxes or not result.masks:
                    print(f"No detections in: {source}")
                    continue

                image_path = result.path
                image = cv2.imread(image_path)
                boxes = result.boxes  # Boxes 对象，用于边界框输出
                masks = result.masks  # Masks 对象，用于分割掩码输出
                names = result.names  # 获取类别名称字典

                for box, mask in zip(boxes, masks):
                    for cls, contour in zip(box.cls, mask.xy):
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

                        # 绘制掩码轮廓
                        cv2.drawContours(image, [contour], -1, color, 20)

                        # 在图像上绘制面积和类名
                        text = f'{class_name} {area:.2f}'
                        cv2.putText(image, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 保存图像
                output_path = os.path.join(output_dir, get_last_part_of_string(image_path))
                cv2.imwrite(output_path, image)
                print(f'Saved: {output_path}')

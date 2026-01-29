import os
import cv2
import glob
import numpy as np


def check_labels(txt_labels, images_dir):
    txt_files = glob.glob(os.path.join(txt_labels, "*.txt"))
    for txt_file in txt_files:
        filename = os.path.splitext(os.path.basename(txt_file))[0]

        # 支持各种图像格式，使用通配符查找
        pic_path = glob.glob(os.path.join(images_dir, filename + ".*"))
        if not pic_path:
            print(f"Image for {filename} not found!")
            continue

        pic_path = pic_path[0]  # 使用找到的第一个图像
        img = cv2.imread(pic_path)
        if img is None:
            print(f"Failed to read image {pic_path}")
            continue

        height, width, _ = img.shape

        with open(txt_file, 'r') as file_handle:
            cnt_info = file_handle.readlines()
        new_cnt_info = [line_str.replace("\n", "").split(" ") for line_str in cnt_info]

        color_map = {"0": (0, 255, 255)}  # 可以根据需要扩展
        for new_info in new_cnt_info:
            print(new_info)
            s = []
            for i in range(1, len(new_info), 2):
                b = [float(tmp) for tmp in new_info[i:i + 2]]
                s.append([int(b[0] * width), int(b[1] * height)])
            cv2.polylines(img, [np.array(s, np.int32)], True, color_map.get(new_info[0], (0, 0, 255)))

        cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
        cv2.imshow('img2', img)
        cv2.waitKey()


# 使用示例
check_labels('G:\\活细胞成像\\Project\\dataset\\labels', 'G:\\活细胞成像\\Project\\dataset\\images')

import os
import glob
import re


def rename_tif_files_simple(directory_path):
    """
    简化版重命名函数
    """

    # 获取所有tif文件
    pattern = os.path.join(directory_path, "*.tif")
    tif_files = glob.glob(pattern)

    for file_path in tif_files:
        filename = os.path.basename(file_path)

        # 匹配 "字母+数字.tif" 格式
        if len(filename) >= 5 and filename.endswith('.tif'):
            base_name = filename[:-4]  # 去掉.tif后缀

            # 检查第一个字符是否是字母，其余是数字
            if base_name[0].isalpha() and base_name[1:].isdigit():
                letter = base_name[0].upper()
                number_part = base_name[1:]

                # 字母转数字（A=1, B=2, ...）
                letter_number = ord(letter) - ord('A') + 1

                # 构建新文件名
                new_filename = f"{letter_number}{number_part}D4.tif"
                new_file_path = os.path.join(directory_path, new_filename)

                # 重命名
                try:
                    os.rename(file_path, new_file_path)
                    print(f"重命名: {filename} -> {new_filename}")
                except OSError as e:
                    print(f"失败: {filename} -> {e}")


# 使用示例
if __name__ == "__main__":
    # 直接指定目录路径
    directory = r"F:\Picture04\D4"  # 改为您的目录路径
    rename_tif_files_simple(directory)
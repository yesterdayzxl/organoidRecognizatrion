import os
import shutil


def copy_first_tiff_files(source_dir, destination_dir):
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # 遍历源目录的所有一级子目录
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        if os.path.isdir(subdir_path):  # 检查是否为目录
            # 获取子目录中所有的文件，并筛选出tiff文件
            tiff_files = [f for f in os.listdir(subdir_path) if
                          f.lower().endswith('.tiff') or f.lower().endswith('.tif')]
            if tiff_files:
                # 取第一张tiff文件
                first_tiff = tiff_files[0]
                source_file_path = os.path.join(subdir_path, first_tiff)
                destination_file_path = os.path.join(destination_dir, first_tiff)

                # 复制文件到目标文件夹
                shutil.copy(source_file_path, destination_file_path)
                print(f"复制 {source_file_path} 到 {destination_file_path}")


# 使用方法
source_directory = 'G:\\活细胞成像\\康和达-类器官成型过程+分析 - 副本\\20240117_20240119_080824\\'
destination_directory = 'G:\\活细胞成像\\康和达-类器官成型过程+分析 - 副本\\20240117_20240119_080824\\Img'
copy_first_tiff_files(source_directory, destination_directory)

import os
import shutil

# 定义原始目录和目标目录
source_directory = r"G:\Picture01"
destination_directory =  r"G:\Picture03"

# 确保目标目录存在
os.makedirs(destination_directory, exist_ok=True)

# 遍历chip1到chip7目录
for chip_folder in range(1, 8):  # chip1 to chip7
    chip_source_path = os.path.join(source_directory, f"chip{chip_folder}")
    chip_dest_path = os.path.join(destination_directory, f"chip{chip_folder}")
    os.makedirs(chip_dest_path, exist_ok=True)  # 为每个chip创建新目录

    # 遍历n=0,1,2,3,4目录
    for n in [0, 1, 2, 3, 4]:
        n_source_path = os.path.join(chip_source_path, str(n))
        if not os.path.exists(n_source_path):
            continue  # 如果n目录不存在，则跳过
        for file_name in os.listdir(n_source_path):
            if file_name.endswith(".png"):
                # 提取文件名中的ab部分
                ab_part = file_name.split(".")[0]  # 去掉扩展名
                # 构建新的文件名
                new_file_name = f"{ab_part}D{n}.png"
                # 构建完整路径
                source_file_path = os.path.join(n_source_path, file_name)
                destination_file_path = os.path.join(chip_dest_path, new_file_name)
                # 复制并重命名文件
                shutil.copy(source_file_path, destination_file_path)

print("图片已重命名并保存到 Picture03。")

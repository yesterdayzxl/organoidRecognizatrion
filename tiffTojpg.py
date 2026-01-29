from PIL import Image
import os


def convert_images_to_jpg(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件扩展名是否为 .tiff 或 .png
        if filename.lower().endswith(('.tiff', '.tif', '.png')):
            file_path = os.path.join(directory, filename)

            # 打开图像文件
            with Image.open(file_path) as img:
                # 将图像转换为 RGB 模式 (JPEG 不支持 alpha 通道)
                img = img.convert('RGB')

                # 构造新的文件名
                new_filename = os.path.splitext(filename)[0] + '.jpg'
                new_file_path = os.path.join(directory, new_filename)

                # 保存为 JPEG 格式
                img.save(new_file_path, 'JPEG', quality=95)
                print(f"已将 {filename} 转换为 {new_filename}")

# 使用方法
directory_path = 'G:\\活细胞成像\\Project\\dataset\\images'
convert_images_to_jpg(directory_path)

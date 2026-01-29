import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
import matplotlib.pyplot as plt
import os

# 文件路径
file_path = r'G:\Picture03\results.xlsx'
output_path = r'G:\Picture03\results-plot.xlsx'

# 加载 Excel 文件
xls = pd.ExcelFile(file_path)

# 临时保存图像的文件夹
temp_plot_dir = r'G:\Picture03\results-plot'
os.makedirs(temp_plot_dir, exist_ok=True)


def process_and_embed(sheet_name, workbook):
    # Load data
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 保留 Group 和 Day 相同，Confidence 最大的行
    df = df.loc[df.groupby(['Group', 'Day'])['Confidence'].idxmax()]

    # 按 Group 和 Day 排序
    df_grouped = df.sort_values(['Group', 'Day'])

    # 创建折线图 - Area vs Day
    area_plot_path = os.path.join(temp_plot_dir, f'{sheet_name}_area_vs_day.png')
    plt.figure(figsize=(16, 10))
    for group, data in df_grouped.groupby('Group'):
        plt.plot(data['Day'], data['Area'], marker='o', label=f'Group {group}')
    plt.title('Area vs Day')
    plt.xlabel('Day')
    plt.ylabel('Area')
    plt.legend()
    plt.grid()
    plt.savefig(area_plot_path)
    plt.close()

    # 创建折线图 - Gray Mean vs Day
    gray_mean_plot_path = os.path.join(temp_plot_dir, f'{sheet_name}_gray_mean_vs_day.png')
    plt.figure(figsize=(16, 10))
    for group, data in df_grouped.groupby('Group'):
        plt.plot(data['Day'], data['Gray Mean'], marker='o', label=f'Group {group}')
    plt.title('Gray Mean vs Day')
    plt.xlabel('Day')
    plt.ylabel('Gray Mean')
    plt.legend()
    plt.grid()
    plt.savefig(gray_mean_plot_path)
    plt.close()


    # 创建折线图 - Gray Std vs Day
    gray_stddev_plot_path = os.path.join(temp_plot_dir, f'{sheet_name}_gray_stddev_vs_day.png')
    plt.figure(figsize=(16, 10))
    for group, data in df_grouped.groupby('Group'):
        plt.plot(data['Day'], data['Gray StdDev'], marker='o', label=f'Group {group}')
    plt.title('Gray StdDev vs Day')
    plt.xlabel('Day')
    plt.ylabel('Gray StdDev')
    plt.legend()
    plt.grid()
    plt.savefig(gray_stddev_plot_path)
    plt.close()


    # Embed the plots into the sheet
    sheet = workbook[sheet_name]

    # 添加 Area vs Day 图
    img_area = Image(area_plot_path)
    img_area.anchor = 'J2'  # 指定图片嵌入的单元格位置
    sheet.add_image(img_area)

    # 添加 Gray Mean vs Day 图
    img_gray_mean = Image(gray_mean_plot_path)
    img_gray_mean.anchor = 'J20'  # 指定图片嵌入的单元格位置
    sheet.add_image(img_gray_mean)

    print(f"Plots embedded for {sheet_name}")


# 打开工作簿
workbook = load_workbook(file_path)

# 对所有 chipXresult 表进行处理并嵌入图表
for sheet in xls.sheet_names:
    if 'chip' in sheet:
        process_and_embed(sheet, workbook)

# 保存新的 Excel 文件
workbook.save(output_path)
print(f"Excel with embedded plots saved to {output_path}")

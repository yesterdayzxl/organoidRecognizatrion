import pandas as pd

# 加载Excel文件
file_path = r'G:\Picture03\results-plot.xlsx'  # 修改为你的文件路径
excel_data = pd.ExcelFile(file_path)

# 初始化一个列表来存储所有工作表的聚合结果
aggregated_results = []

# 遍历每个工作表
for sheet_name in excel_data.sheet_names:
    # 加载工作表到DataFrame
    df = excel_data.parse(sheet_name)

    # 按 'Day' 分组，并计算 'Area', 'gray mean', 'gray stddev' 的均值
    group_means = df.groupby('Day')[['Area', 'Gray Mean', 'Gray StdDev']].mean().reset_index()
    group_means['Sheet'] = sheet_name  # 添加工作表名称，用来标识结果来自哪个工作表

    # 将每个工作表的聚合数据添加到结果列表
    aggregated_results.append(group_means)

# 将所有工作表的结果合并为一个DataFrame
combined_results = pd.concat(aggregated_results, ignore_index=True)

# 将合并后的结果保存到新的Excel文件中
output_path = r'G:\Picture03\results-average.xlsx'  # 输出文件路径
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # 保持原始工作表内容
    for sheet in excel_data.sheet_names:
        excel_data.parse(sheet).to_excel(writer, sheet_name=sheet, index=False)

    # 将聚合结果写入新的工作表
    combined_results.to_excel(writer, sheet_name='Aggregated Results', index=False)

print(f"聚合结果已保存到 {output_path}")

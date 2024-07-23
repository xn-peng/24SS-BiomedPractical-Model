import pandas as pd
import os

# 读取 Excel 文件
file_path = 'D:/Deutschland/Data and Computer Science/24SS/Bio Practical/BMP Slices/output-256/TestSet_demographics.xlsx'
df = pd.read_excel(file_path)

# 初始化一个列表，用于存放字典
dict_list = []

# 定义文件夹路径
base_folder = 'D:/Deutschland/Data and Computer Science/24SS/Bio Practical/BMP Slices/output-256'

# 遍历 DataFrame 中的每一行，创建字典并添加到列表中
for index, row in df.iterrows():
    rand_id = row['RandID']
    lesion_folder = os.path.join(base_folder, f"{rand_id}_Lesion")
    
    # 检查文件夹是否存在且是否为空
    if os.path.exists(lesion_folder) and os.listdir(lesion_folder):
        lesion_value = 1
    else:
        lesion_value = 0
    
    # 创建记录字典
    record = {
        "RandID": rand_id,
        "Age": row['Age'],
        "Sex": row['Sex'],
        "TSI": row['TSI'],
        "ScanManufacturer": row['ScanManufacturer'],
        "Lesion": lesion_value,  # 根据文件夹内容设置 Lesion
        "Location": None  # 添加 Location 键值对
    }
    dict_list.append(record)

# 将字典列表转换为 DataFrame
dict_df = pd.DataFrame(dict_list)

# 定义输出 CSV 文件路径
output_csv_path = 'D:/Deutschland/Data and Computer Science/24SS/Bio Practical/BMP Slices/output-256/TestSet_demographics_with_lesion.csv'

# 将 DataFrame 保存为 CSV 文件
dict_df.to_csv(output_csv_path, index=False)

# 打印 CSV 文件路径
print(f"CSV 文件已保存至: {output_csv_path}")
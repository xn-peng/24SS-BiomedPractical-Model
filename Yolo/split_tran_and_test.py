import os
import random

folder_path = r'D:/Deutschland/Data and Computer Science/24SS/Bio Practical/BMP Slices/output-256'

# 获取所有包含"T1"的子文件夹的路径
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir() and 'T1' in f.name]

# 随机打乱文件夹顺序
random.shuffle(subfolders)

# 计算训练集和测试集的分割点
split_point = int(0.8 * len(subfolders))

# 分割成训练集和测试集
train_set = subfolders[:split_point]
test_set = subfolders[split_point:]

# 将路径保存到train.txt
with open('train.txt', 'w') as f:
    for folder in train_set:
        f.write(folder + '\n')

# 将路径保存到test.txt
with open('test.txt', 'w') as f:
    for folder in test_set:
        f.write(folder + '\n')
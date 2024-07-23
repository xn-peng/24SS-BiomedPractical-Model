import os
import random
import shutil

# 定义文件夹路径
datasets_dir = r"D:/Deutschland/Data and Computer Science/24SS/Bio Practical/Yolo/datasets"
labels_dir = os.path.join(datasets_dir, 'labels')
output_dir = os.path.join(datasets_dir, 'datasets')

train_images_dir = os.path.join(output_dir, 'train', 'images')
train_labels_dir = os.path.join(output_dir, 'train', 'labels')
val_images_dir = os.path.join(output_dir, 'val', 'images')
val_labels_dir = os.path.join(output_dir, 'val', 'labels')

# 创建目标文件夹
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 收集所有图片路径
all_image_paths = []
for root, _, files in os.walk(datasets_dir):
    for file in files:
        if file.endswith('.png'):
            # 拼接完整路径并替换路径中的反斜杠为正斜杠
            image_path = os.path.join(root, file).replace("\\", "/")
            all_image_paths.append(image_path)

# 随机打乱所有图片路径
random.shuffle(all_image_paths)

# 计算训练集和测试集的分割点
split_point = int(len(all_image_paths) * 0.8)

# 分割训练集和测试集
train_image_paths = all_image_paths[:split_point]
test_image_paths = all_image_paths[split_point:]

# 定义函数来复制图片和对应的标签并记录相对路径
def copy_files(image_paths, image_dest, label_dest, txt_file):
    with open(txt_file, 'w') as f:
        for image_path in image_paths:
            # 提取ID
            folder_name = os.path.basename(os.path.dirname(image_path))
            id_part = folder_name.split('_')[1]

            # 修改图片文件名
            image_file_name = os.path.basename(image_path)
            new_image_file_name = f"{id_part}_{image_file_name}"
            
            # 复制图片文件
            shutil.copy2(image_path, os.path.join(image_dest, new_image_file_name))
            
            # 记录相对路径
            relative_image_path = os.path.relpath(os.path.join(image_dest, new_image_file_name), output_dir).replace("\\", "/")
            f.write(f"{relative_image_path}\n")
            
            # 找到对应的标签文件并修改文件名后复制
            label_file_name = image_file_name.replace('.png', '.txt')
            new_label_file_name = f"{id_part}_{label_file_name}"
            label_file_path = os.path.join(labels_dir, os.path.basename(os.path.dirname(image_path)), label_file_name)
            
            if os.path.exists(label_file_path):
                shutil.copy2(label_file_path, os.path.join(label_dest, new_label_file_name))

# 复制训练集的图片和标签，并生成train.txt
copy_files(train_image_paths, train_images_dir, train_labels_dir, os.path.join(output_dir, 'train.txt'))

# 复制测试集的图片和标签，并生成val.txt
copy_files(test_image_paths, val_images_dir, val_labels_dir, os.path.join(output_dir, 'val.txt'))

print("Dataset has been organized into train and val directories with train.txt and val.txt.")
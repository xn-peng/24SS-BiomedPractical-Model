import os
import shutil

prepared_data_dir = 'PreparedData'
merged_data_dir = 'Merged'
merged_t1_dir = os.path.join(merged_data_dir, 'T1')
merged_lesion_dir = os.path.join(merged_data_dir, 'Lesion')

# Get directory
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "PreparedData"))
train_t1_dir = os.path.join(data_dir, 'train', 'T1')
train_lesion_dir = os.path.join(data_dir, 'train', 'Lesion')
val_t1_dir = os.path.join(data_dir, 'val', 'T1')
val_lesion_dir = os.path.join(data_dir, 'val', 'Lesion')


os.makedirs(merged_t1_dir, exist_ok=True)
os.makedirs(merged_lesion_dir, exist_ok=True)

# Copy
def copy_files(src_dir, dest_dir):
    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)

# T1
copy_files(train_t1_dir, merged_t1_dir)
copy_files(val_t1_dir, merged_t1_dir)

# Lesion
copy_files(train_lesion_dir, merged_lesion_dir)
copy_files(val_lesion_dir, merged_lesion_dir)

print("Done")

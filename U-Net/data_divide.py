import os
import random
import shutil

data_dir = 'Merged'
t1_dir = os.path.join(data_dir, 'T1')
lesion_dir = os.path.join(data_dir, 'Lesion')

t1_files = [f for f in os.listdir(t1_dir) if os.path.isfile(os.path.join(t1_dir, f))]
lesion_files = [f for f in os.listdir(lesion_dir) if os.path.isfile(os.path.join(lesion_dir, f))]

assert len(t1_files) == len(lesion_files), "T1 and Lesion file counts do not match."

combined_files = list(zip(t1_files, lesion_files))
random.shuffle(combined_files)

total_files = len(combined_files)
train_count = 272
val_count = 58
test_count = total_files - train_count - val_count

train_files = combined_files[:train_count]
val_files = combined_files[train_count:train_count + val_count]
test_files = combined_files[train_count + val_count:]

print(f"Number of train: {len(train_files)}")
print(f"Number of val: {len(val_files)}")
print(f"Number of test: {len(test_files)}")

prepared_data_dir = 'PreparedData'
new_train_t1_dir = os.path.join(prepared_data_dir, 'new_train', 'T1')
new_train_lesion_dir = os.path.join(prepared_data_dir, 'new_train', 'Lesion')
new_val_t1_dir = os.path.join(prepared_data_dir, 'new_val', 'T1')
new_val_lesion_dir = os.path.join(prepared_data_dir, 'new_val', 'Lesion')
new_test_t1_dir = os.path.join(prepared_data_dir, 'test', 'T1')
new_test_lesion_dir = os.path.join(prepared_data_dir, 'test', 'Lesion')

os.makedirs(new_train_t1_dir, exist_ok=True)
os.makedirs(new_train_lesion_dir, exist_ok=True)
os.makedirs(new_val_t1_dir, exist_ok=True)
os.makedirs(new_val_lesion_dir, exist_ok=True)
os.makedirs(new_test_t1_dir, exist_ok=True)
os.makedirs(new_test_lesion_dir, exist_ok=True)

def copy_files(file_pairs, src_dirs, dest_dirs):
    for t1_file, lesion_file in file_pairs:
        t1_src_path = os.path.join(src_dirs[0], t1_file)
        lesion_src_path = os.path.join(src_dirs[1], lesion_file)
        t1_dest_path = os.path.join(dest_dirs[0], t1_file)
        lesion_dest_path = os.path.join(dest_dirs[1], lesion_file)
        shutil.copy(t1_src_path, t1_dest_path)
        shutil.copy(lesion_src_path, lesion_dest_path)


copy_files(train_files, [t1_dir, lesion_dir], [new_train_t1_dir, new_train_lesion_dir])


copy_files(val_files, [t1_dir, lesion_dir], [new_val_t1_dir, new_val_lesion_dir])


copy_files(test_files, [t1_dir, lesion_dir], [new_test_t1_dir, new_test_lesion_dir])


def write_files_to_txt(file_pairs, txt_file_path):
    with open(txt_file_path, 'w') as file:
        for idx, (t1_file, lesion_file) in enumerate(file_pairs):
            file.write(f"{idx+1}: {t1_file}, {lesion_file}\n")


write_files_to_txt(train_files, os.path.join(prepared_data_dir, 'train_files.txt'))


write_files_to_txt(val_files, os.path.join(prepared_data_dir, 'val_files.txt'))


write_files_to_txt(test_files, os.path.join(prepared_data_dir, 'test_files.txt'))

print("Done")

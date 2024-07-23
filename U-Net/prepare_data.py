import os
import shutil
import random

def prepare_data(data_dir, output_dir, test_size=0.2):
    t1_dir = os.path.join(data_dir, "T1")
    lesion_dir = os.path.join(data_dir, "Lesion")

    t1_files = [f for f in os.listdir(t1_dir) if f.endswith(".nii.gz")]
    random.shuffle(t1_files)

    split_idx = int(len(t1_files) * (1 - test_size))
    train_files = t1_files[:split_idx]
    val_files = t1_files[split_idx:]

    # Create the file
    train_t1_dir = os.path.join(output_dir, "train", "T1")
    train_lesion_dir = os.path.join(output_dir, "train", "Lesion")
    val_t1_dir = os.path.join(output_dir, "val", "T1")
    val_lesion_dir = os.path.join(output_dir, "val", "Lesion")

    os.makedirs(train_t1_dir, exist_ok=True)
    os.makedirs(train_lesion_dir, exist_ok=True)
    os.makedirs(val_t1_dir, exist_ok=True)
    os.makedirs(val_lesion_dir, exist_ok=True)

    # Copy
    for file_name in train_files:
        shutil.copy(os.path.join(t1_dir, file_name), os.path.join(train_t1_dir, file_name))
        rand_id = file_name.split('_')[1]
        lesion_file = f"scan_{rand_id}_Lesion.nii.gz"
        shutil.copy(os.path.join(lesion_dir, lesion_file), os.path.join(train_lesion_dir, lesion_file))

    for file_name in val_files:
        shutil.copy(os.path.join(t1_dir, file_name), os.path.join(val_t1_dir, file_name))
        rand_id = file_name.split('_')[1]
        lesion_file = f"scan_{rand_id}_Lesion.nii.gz"
        shutil.copy(os.path.join(lesion_dir, lesion_file), os.path.join(val_lesion_dir, lesion_file))

    # Save txt file
    with open(os.path.join(output_dir, 'validation_files.txt'), 'w') as f:
        for file_name in val_files:
            f.write(f"{file_name}\n")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "Data"))
    output_dir = os.path.abspath(os.path.join(current_dir, "PreparedData"))

    prepare_data(data_dir, output_dir)

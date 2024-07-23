import os
import shutil

os.makedirs('Data/T1', exist_ok=True)
os.makedirs('Data/Lesion', exist_ok=True)

source_folder = '../Dataset/Aims-Tbi'

files = os.listdir(source_folder)

for file in files:
    if 'T1' in file:
        shutil.move(os.path.join(source_folder, file), 'Data/T1/' + file)
    elif 'Lesion' in file:
        shutil.move(os.path.join(source_folder, file), 'Data/Lesion/' + file)

print("T1 files:", os.listdir('Data/T1'))

print("Lesion files:", os.listdir('Data/Lesion'))

import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data.astype(np.float32)  # float32

def resample(image, target_shape):
    factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, factors, order=1)  # Linear interpolation

class BrainLesionDataset(Dataset):
    def __init__(self, t1_files, demographics_file, target_shape=(128, 128, 128)):
        self.t1_files = t1_files
        self.target_shape = target_shape
        self.demographics_data = pd.read_csv(demographics_file)
        self.demographics_data['RandID'] = self.demographics_data['RandID'].str.replace('scan_', '')
        self.demographics_data.set_index('RandID', inplace=True)

    def __len__(self):
        return len(self.t1_files)

    def __getitem__(self, idx):
        t1_file = self.t1_files[idx]
        rand_id = os.path.basename(t1_file).split('_')[1]

        t1_data = load_nifti(t1_file)
        t1_data = (t1_data - np.min(t1_data)) / (np.max(t1_data) - np.min(t1_data))
        t1_data = resample(t1_data, self.target_shape)

        demographics_info = self.demographics_data.loc[rand_id]
        extra_features = torch.tensor(demographics_info[['Age', 'Sex', 'TSI']].values.astype(np.float32), dtype=torch.float32)
        label = torch.tensor(demographics_info['Lesion'], dtype=torch.float32)

        return torch.tensor(t1_data, dtype=torch.float32).unsqueeze(0), extra_features, label

def load_data(data_dir, demographics_file, batch_size=2):
    train_t1_dir = os.path.join(data_dir, "train", "T1")
    val_t1_dir = os.path.join(data_dir, "val", "T1")

    train_t1_files = [os.path.join(train_t1_dir, f) for f in os.listdir(train_t1_dir) if f.endswith(".nii.gz")]
    val_t1_files = [os.path.join(val_t1_dir, f) for f in os.listdir(val_t1_dir) if f.endswith(".nii.gz")]

    train_dataset = BrainLesionDataset(train_t1_files, demographics_file)
    val_dataset = BrainLesionDataset(val_t1_files, demographics_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

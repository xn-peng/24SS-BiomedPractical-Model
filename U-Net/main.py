import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model import UNet
from data_loader import load_nifti, resample
from tqdm import tqdm
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BrainLesionDataset(Dataset):
    def __init__(self, t1_files, lesion_files, demographics_file):
        self.t1_files = t1_files
        self.lesion_files = lesion_files
        self.demographics_file = demographics_file

    def __len__(self):
        return len(self.t1_files)

    def __getitem__(self, idx):
        t1_file = self.t1_files[idx]
        lesion_file = self.lesion_files[idx]

        t1_data = load_nifti(t1_file)
        lesion_data = load_nifti(lesion_file)

        # Normalization
        t1_data = (t1_data - np.min(t1_data)) / (np.max(t1_data) - np.min(t1_data))

        # Reshape
        t1_data = resample(t1_data, (256, 256, 256))
        lesion_data = resample(lesion_data, (256, 256, 256))

        rand_id = os.path.basename(t1_file).split('_')[1]
        demographics_data = pd.read_excel(self.demographics_file)
        demographics_data['RandID'] = demographics_data['RandID'].str.replace('scan_', '')
        demographics_data.set_index('RandID', inplace=True)
        demographics_info = demographics_data.loc[rand_id]


        extra_features = torch.tensor(demographics_info[['Age', 'Sex', 'TSI']].values.astype(np.float32), dtype=torch.float32)

        return torch.tensor(t1_data, dtype=torch.float32).unsqueeze(0), \
               torch.tensor(lesion_data, dtype=torch.float32).unsqueeze(0), \
               extra_features

def load_data(data_dir, demographics_file, batch_size=2):
    train_t1_dir = os.path.join(data_dir, "train", "T1")
    train_lesion_dir = os.path.join(data_dir, "train", "Lesion")
    val_t1_dir = os.path.join(data_dir, "val", "T1")
    val_lesion_dir = os.path.join(data_dir, "val", "Lesion")

    train_t1_files = [os.path.join(train_t1_dir, f) for f in os.listdir(train_t1_dir) if f.endswith(".nii.gz")]
    train_lesion_files = [os.path.join(train_lesion_dir, f) for f in os.listdir(train_lesion_dir) if f.endswith(".nii.gz")]
    val_t1_files = [os.path.join(val_t1_dir, f) for f in os.listdir(val_t1_dir) if f.endswith(".nii.gz")]
    val_lesion_files = [os.path.join(val_lesion_dir, f) for f in os.listdir(val_lesion_dir) if f.endswith(".nii.gz")]

    train_dataset = BrainLesionDataset(train_t1_files, train_lesion_files, demographics_file)
    val_dataset = BrainLesionDataset(val_t1_files, val_lesion_files, demographics_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        logging.info(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info(f"Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch
    else:
        logging.info(f"No checkpoint found at '{filename}'")
        return 0

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, device='cpu', checkpoint_file='checkpoint.pth.tar'):
    start_epoch = load_checkpoint(checkpoint_file, model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(total=len(train_loader.dataset), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for t1, lesion, extra_features in train_loader:
            t1 = t1.to(device)
            lesion = lesion.to(device)
            extra_features = extra_features.to(device)

            optimizer.zero_grad()
            outputs = model(t1, extra_features)
            loss = criterion(outputs, lesion)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.update(len(t1))

        progress_bar.set_postfix(loss=train_loss/len(train_loader))
        progress_bar.close()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for t1, lesion, extra_features in val_loader:
                t1 = t1.to(device)
                lesion = lesion.to(device)
                extra_features = extra_features.to(device)

                outputs = model(t1, extra_features)
                loss = criterion(outputs, lesion)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_file)

    torch.save(model.state_dict(), 'final_model.pth')
    print('Model saved as final_model.pth')

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "PreparedData"))
    demographics_file = os.path.join(data_dir, '../Data/TestSet_demographics.xlsx')

    print(f"Current directory: {current_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Demographics file: {demographics_file}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.exists(demographics_file):
        raise FileNotFoundError(f"Demographics file not found: {demographics_file}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')


    model = UNet().to(device)
    logging.info(model)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 1
    num_epochs = 2

    train_loader, val_loader = load_data(data_dir, demographics_file, batch_size)

    # Train
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device=device, checkpoint_file='checkpoint.pth.tar')

if __name__ == '__main__':
    main()

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import UNet
from data_loader import load_nifti, resample

# cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # remove 'module.' prefix
        new_state_dict[k] = v
    return new_state_dict


# Load the model
print("Loading model...")
model = UNet().to(device)
checkpoint = torch.load('result/new/final_model.pth', map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.")


# Preprocess T1 image
def load_and_preprocess_image(nifti_path):
    if not os.path.exists(nifti_path):
        print(f"File not found: {nifti_path}")
        return None
    t1_data = load_nifti(nifti_path)
    t1_data = (t1_data - np.min(t1_data)) / (np.max(t1_data) - np.min(t1_data))  # 归一化
    t1_data = resample(t1_data, (128, 128, 128))  # 重新采样
    t1_tensor = torch.tensor(t1_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128, 128)
    return t1_tensor.to(device)


def load_and_preprocess_features(age, sex, tsi):
    extra_features = torch.tensor([age, sex, tsi], dtype=torch.float32).unsqueeze(0)  # 转换为二维张量，形状为 (1, num_features)
    return extra_features.to(device)


# Lesion predict
def predict_lesion(nifti_path, age, sex, tsi, threshold=1e-3):
    t1_tensor = load_and_preprocess_image(nifti_path)
    if t1_tensor is None:
        return None, None
    extra_features = load_and_preprocess_features(age, sex, tsi)

    with torch.no_grad():
        outputs = model(t1_tensor, extra_features)
        preds = (outputs > threshold).float()

    num_lesions = preds.sum().item()
    return num_lesions > 0, preds.cpu().numpy()


def evaluate_model(test_data_path, demographics_path, threshold=1e-3):
    test_demographics = pd.read_csv(demographics_path)

    print("Columns in the CSV file:", test_demographics.columns)
    print("RandID values in the CSV file:", test_demographics['RandID'].tolist())

    y_true = []
    y_pred = []

    for file_name in os.listdir(test_data_path):
        if not file_name.endswith('_T1.nii.gz'):
            continue

        file_id = file_name.split('_')[1]  # 去掉'scan_'前缀，并获取ID部分
        file_id = file_id.split('.')[0]  # 移除扩展名
        file_id_full = f'scan_{file_id.zfill(4)}'  # 保留'scan_'前缀并确保ID有四位数
        nifti_path = os.path.join(test_data_path, file_name).replace('\\', '/')
        print(f"Processing file: {nifti_path}")

        # 从人口统计数据中获取对应的年龄、性别和TSI
        row = test_demographics[test_demographics['RandID'] == file_id_full]
        if row.empty:
            print(f"No demographic data found for ID: {file_id_full}")
            continue

        age, sex, tsi, lesion = row.iloc[0]['Age'], row.iloc[0]['Sex'], row.iloc[0]['TSI'], row.iloc[0]['Lesion']

        has_lesion, _ = predict_lesion(nifti_path, age, sex, tsi, threshold)

        # Use label
        y_true.append(lesion)
        y_pred.append(int(has_lesion))
        print(f"ID: {file_id_full}, Predicted Lesion: {has_lesion}")

    if len(y_true) == 0 or len(y_pred) == 0:
        print("No predictions were made. Exiting.")
        return None

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{cm}")

    return accuracy, precision, recall, f1, cm


def plot_metrics(metrics, save_path="metrics.png"):
    if metrics is None:
        print("No metrics to plot.")
        return

    accuracy, precision, recall, f1, cm = metrics

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot confusion matrix
    cax = ax[0].matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax, ax=ax[0])
    for (i, j), value in np.ndenumerate(cm):
        ax[0].text(j, i, f'{value}', ha='center', va='center')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')
    ax[0].set_title('Confusion Matrix')

    # Plot precision, recall, f1
    ax[1].bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [accuracy, precision, recall, f1],
              color=['blue', 'orange', 'green', 'red'])
    ax[1].set_ylim([0, 1])
    ax[1].set_title('Metrics')
    ax[1].set_ylabel('Score')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Metrics plot saved to {save_path}")


def main():
    test_data_path = 'PreparedData/train/T1'
    demographics_path = 'Data/TestSet_demographics_with_lesionl.csv'
    threshold = 0.5

    metrics = evaluate_model(test_data_path, demographics_path, threshold)
    plot_metrics(metrics)


if __name__ == "__main__":
    main()

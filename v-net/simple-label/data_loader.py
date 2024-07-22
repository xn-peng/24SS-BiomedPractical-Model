import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom

def load_nifti(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data.astype(np.float32)  # 变更数据类型为 float32

def resample(image, target_shape):
    factors = [t / s for t, s in zip(target_shape, image.shape)]
    return zoom(image, factors, order=1)  # 线性插值

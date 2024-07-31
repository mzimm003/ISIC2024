import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np

import pandas as pd
from PIL import Image

import h5py
import io
from pathlib import Path

from torch import nn

class PreProcess(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.processes = {}

    def forward(self, x):
        return nn.Sequential(*list(self._modules.values()))(x)

class PPInput(PreProcess):
    def __init__(
        self,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.normalize = nn.Norm

class SkinLesions(Dataset):
    def __init__(self, annotations_file, img_file, img_dir, transform=None, target_transform=None):
        self.metadata = pd.read_csv(annotations_file)
        self.image_data = h5py.File(img_file, "r")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image = np.array(self.image_data[self.metadata.loc[idx, "isic_id"]])
        image = np.array(Image.open(io.BytesIO(image)),dtype=np.uint8)
        label = self.metadata.loc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def main():
    SkinLesions("train-metadata.csv", "train-image.hdf5", "train-image")


if __name__ == "__main__":
    main()
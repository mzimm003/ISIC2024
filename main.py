import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import pandas as pd
from PIL import Image

import h5py
import io
from pathlib import Path

from torch import nn

class Rescale(nn.Module):
    def __init__(
            self,
            minimum=0,
            maximum=255,
            *args,
            **kwargs) -> None:
        """
        Rescale inputs of a preset range to a range from 0 to 1.

        Args:
            minimum: the smallest possible value of expected inputs.
            maximum: the largest possible value of expected inputs.
        """
        super().__init__(*args, **kwargs)
        self.min = minimum
        self.max = maximum
    
    def forward(self, x):
        return (x - self.min) / (self.max - self.min)

class PreProcess(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for proc in self._modules.values():
            x = proc(x)
        return x

class PPInput(PreProcess):
    def __init__(
        self,
        rescale_images = True,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rescale = Rescale() if rescale_images else None

class PPTarget(PreProcess):
    def __init__(
        self,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)

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
    data = SkinLesions(
        "train-metadata.csv",
        "train-image.hdf5",
        "train-image",
        transform=PPInput(),
        target_transform=PPTarget())
    LDA = LinearDiscriminantAnalysis()
    pass

if __name__ == "__main__":
    main()
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image

import h5py
import io

from typing import (
    Union,
)
from pathlib import Path

from isic.registry import Registry

class SkinLesions(Dataset):
    def __init__(
            self,
            annotations_file:Union[str, Path],
            img_file:Union[str, Path] = None,
            img_dir:Union[str, Path] = None,
            img_transform:nn.Module = None,
            annotation_transform:nn.Module = None,
            annotations_only:bool = False,
            label_desc:str = 'target'):
        self.metadata = pd.read_csv(annotations_file, low_memory=False)
        
        self.label_desc = label_desc
        self.annotations_only = annotations_only
        self.image_data = None
        self.img_dir = None
        self.img_listing = None

        if not self.annotations_only:
            self.image_data = h5py.File(img_file, "r")
            self.img_dir = img_dir
            self.img_listing = self.metadata.loc[:, "isic_id"]

        self.img_transform = img_transform
        self.annotation_transform = annotation_transform
        if self.annotation_transform:
            self.metadata = self.annotation_transform(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = None
        label = None
        annotations = self.metadata.loc[idx]
        anno_is_df = isinstance(annotations, pd.DataFrame)

        if self.annotations_only:
            data = torch.tensor(annotations.drop(self.label_desc, axis=anno_is_df).values)
        else:
            image = np.array(self.image_data[self.img_listing[idx]])
            image = np.array(Image.open(io.BytesIO(image)),dtype=np.uint8)
            if self.img_transform:
                image = self.img_transform(image)
            data = (
                torch.tensor(image),
                torch.tensor(annotations.drop(self.label_desc, axis=anno_is_df).values)
                )

        label = torch.tensor(annotations[self.label_desc])
        return data, label

class DatasetReg(Registry):
    SkinLesions = SkinLesions
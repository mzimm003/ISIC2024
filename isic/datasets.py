import torch
from torch import nn
from torch.utils.data import Dataset as DatasetTorch
import numpy as np
import pandas as pd
from PIL import Image

import h5py
import io

from typing import (
    Union,
    Tuple,
    List
)
from pathlib import Path

from isic.registry import Registry

class Dataset(DatasetTorch):
    """
    A basis for database creation and dataset serving.

    To prevent memory problems with multiprocessing, class provides the utility 
    functions:

    * strings_to_mem_safe_val_and_offset
    * mem_safe_val_and_offset_to_string
    * string_to_sequence
    * sequence_to_string
    * pack_sequences
    * unpack_sequence

    Provided by https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519.
    These should be used in lieu of lists or dicts in the data retrieval process
    (e.g. for data labels).

    See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662 
    for a summary of the issue.
    """

    AUTONAME = "complete_generated_dataset"
    LBL_FILE_COUNT = "total_label_files"
    LBL_COUNT = "total_labels"
    CURR_FILE_IDX = "current_file"
    LBL_COUNT_BY_FILE = "_label_counts"
    LBL_BINS = "_cum_label_counts"
    def __init__(
            self,
            seed:int=None,
            ) -> None:
        super().__init__()
        self.seed = seed

    def getName(self):
        return self.__class__.__name__
    
    # --- UTILITY FUNCTIONS ---
    @staticmethod
    def strings_to_mem_safe_val_and_offset(strings: List[str]) -> Tuple[np.ndarray,np.ndarray]:
        """
        Utility function.
        """
        seqs = [Dataset.string_to_sequence(s) for s in strings]
        return Dataset.pack_sequences(seqs)
    
    @staticmethod
    def mem_safe_val_and_offset_to_string(v, o, index:int) -> Tuple[np.ndarray,np.ndarray]:
        '''
        Utility function.

        In case labels represented by file_idx are no longer in memory, use arbitrary index from existing file.
        Either the current idx, or 0 if current index is beyond existing labels. This will be rare, and impact a
        small fraction of a percent of data points, and otherwise still supplies a valid data point. Bandaid necessary
        to allow multiprocessing of a partitioned dataset.
        '''
        index = index if index < len(o) else 0
        seq = Dataset.unpack_sequence(v, o, index)
        return Dataset.sequence_to_string(seq)
    
    @staticmethod
    def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
        """
        Utility function.
        """
        return np.array([ord(c) for c in s], dtype=dtype)

    @staticmethod
    def sequence_to_string(seq: np.ndarray) -> str:
        """
        Utility function.
        """
        return ''.join([chr(c) for c in seq])

    @staticmethod
    def pack_sequences(seqs: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility function.
        """
        values = np.concatenate(seqs, axis=0)
        offsets = np.cumsum([len(s) for s in seqs])
        return values, offsets

    @staticmethod
    def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
        """
        Utility function.
        """
        off1 = offsets[index]
        if index > 0:
            off0 = offsets[index - 1]
        elif index == 0:
            off0 = 0
        else:
            raise ValueError(index)
        return values[off0:off1]

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
        metadata = pd.read_csv(annotations_file, low_memory=False)
        
        self.label_desc = label_desc
        self.annotations_only = annotations_only
        self.img_file = None
        self.img_dir = None
        self.img_listing = None

        if not self.annotations_only:
            self.img_file = img_file
            self.img_dir = img_dir
            self.img_listing_v, self.img_listing_o = (
                Dataset.strings_to_mem_safe_val_and_offset(metadata.loc[:, "isic_id"]))

        self.img_transform = img_transform
        self.annotation_transform = annotation_transform
        if self.annotation_transform:
            metadata = self.annotation_transform(metadata)
        self.annotations = torch.tensor(metadata.drop(self.label_desc, axis=1).values)
        self.labels = torch.tensor(metadata[self.label_desc])

    def __get_img_listing(self, idx):
        return Dataset.mem_safe_val_and_offset_to_string(
            self.img_listing_v,
            self.img_listing_o,
            idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = None
        label = None
        annotations = self.annotations[idx]
        label = self.labels[idx]

        if self.annotations_only:
            data = annotations
        else:
            image_data = h5py.File(self.img_file, "r")
            listing = self.__get_img_listing(idx)
            image = np.array(image_data[listing])
            image = np.array(Image.open(io.BytesIO(image)),dtype=np.uint8)
            if self.img_transform:
                image = self.img_transform(image)
            data = (
                torch.tensor(image),
                annotations
                )

        return data, label

class DatasetReg(Registry):
    SkinLesions = SkinLesions
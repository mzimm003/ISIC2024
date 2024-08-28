import torch
from torch import nn
from torch.utils.data import Dataset as DatasetTorch
from torch.utils.data import Subset as SubsetTorch

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from PIL import Image

import h5py
import io

from typing import (
    Union,
    Tuple,
    List,
    Callable
)
from pathlib import Path
import math

from isic.registry import Registry

def train_test_split(
        *arrays:ArrayLike,
        train_size:Union[int,float]=None,
        test_size:Union[int, float]=None,
        shuffle:bool=True,
        stratify:Union[List[ArrayLike],None]=None,
        seed=None,
        ):
    assert isinstance(test_size, type(train_size))
    rng = np.random.default_rng(seed)
    train_idxs = test_idxs = None
    for i, array in enumerate(arrays):
        assert stratify is None or len(stratify[i]) == len(array)
        idxs = np.arange(len(array))

        if shuffle:
            idxs = rng.permutation(idxs)

        if isinstance(test_size, float):
            assert test_size + train_size <= 1.
            train_size = int(len(array)*train_size)
            test_size = int(math.ceil(len(array)*test_size))
        
        assert (train_size + test_size) <= len(array)

        if stratify is None:
            train_idxs = idxs[:train_size]
            test_idxs = idxs[train_size:train_size+test_size]
        else:
            train_idxs = []
            test_idxs = []
            strat = stratify[i][idxs]
            class_id, class_count = np.unique(strat, return_counts=True)
            class_ratios = class_count/len(strat)
            for c, rat in zip(class_id, class_ratios):
                mask = strat == c
                population = idxs[mask]
                train_sample_size = int(train_size*rat)
                train_idxs.extend(
                    population[:train_sample_size])
                test_sample_size = int(math.ceil(test_size*rat))
                test_idxs.extend(
                    population[train_sample_size:train_sample_size+test_sample_size])
            train_idxs = rng.permutation(train_idxs)
            test_idxs = rng.permutation(test_idxs)
            
        yield train_idxs, test_idxs

class DataHandlerGenerator:
    pipeline:list[tuple[str, Callable]]
    def __init__(
            self,
            **inputs):
        self.pipeline = []
        for key, inp in inputs.items():
            setattr(self, key, inp)
        for i in range(len(self.pipeline)):
            inps = self.pipeline[i][0]
            mod = self.pipeline[i][1]
            if not isinstance(self.pipeline[i][0],list):
                assert isinstance(self.pipeline[i][0],str)
                inps = [self.pipeline[i][0]]
            if isinstance(self.pipeline[i][0],str) or isinstance(self.pipeline[i][0],Path):
                mod = Registry.load_model(mod)
            self.pipeline[i] = (inps, mod)
    def get_data_handler(self, data):
        inputs = {
            inp:getattr(data, val)
            for inp, val in self.__dict__.items() if inp != "pipeline"
        }
        return DataHandler(
            pipeline=self.pipeline,
            **inputs)

class DataHandler:
    loss:torch.Tensor
    output:torch.Tensor
    output_label:torch.Tensor
    target:torch.Tensor
    last_lr:float
    pipeline:list[tuple[str, Callable]]
    def __init__(
            self,
            target=None,
            pipeline=None,
            **inputs,
            ):
        self.pipeline = pipeline
        self.target = target
        self.inputs = inputs

    def process_data(self, model:nn.Module, dtype=None, trunc_batch=None):
        param_ref = next(iter(model.parameters()))
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                if dtype:
                    value = value.to(dtype=dtype)
                if trunc_batch:
                    value = value[:trunc_batch,...]
                setattr(self, key, value.to(device=param_ref.device))
            elif key == 'inputs' and isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        if dtype:
                            v = v.to(dtype=dtype)
                        if trunc_batch:
                            v = v[:trunc_batch,...]
                        self.inputs[k] = v.to(device=param_ref.device)
        self.run_pipeline()
    
    def get_inputs(self):
        return self.inputs
    
    def set_last_lr(self, last_lr):
        self.last_lr = last_lr

    def get_last_lr(self):
        return self.last_lr

    def set_loss(self, loss):
        self.loss = loss

    def set_model_output(self, output):
        self.output = output
        self.output_label = self.output.max(-1).indices

    def run_pipeline(self):
        # Pipeline exists to modify input data prior to providing to training
        # model, e.g. piping through a feature reduction model. To that end, 
        # it is assumed inputs of the pipeline should be replaced by the
        # outputs.
        for inp, mod in self.pipeline:
            inp = {k:self.inputs[k] for k in inp}
            if len(inp) > 1:
                raise NotImplementedError
            out, = mod(**inp)
            self.inputs[list(inp.keys())[0]] = out

class SimpleCustomBatch:
    IMG="img"
    FET="fet"
    TGT="tgt"
    def __init__(self, data):
        transposed_data = list(zip(*data))
        transposed_inps = list(zip(*transposed_data[0]))
        self.img = torch.stack(transposed_inps[0], 0)
        self.fet = torch.stack(transposed_inps[1], 0)
        self.tgt = None
        if isinstance(transposed_data[1][0], str):
            self.tgt = transposed_data[1]
        else:
            self.tgt = torch.stack(transposed_data[1], 0).long()

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.fet = self.fet.pin_memory()
        if not isinstance(self.tgt, tuple):
            self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

class Subset(SubsetTorch):
    dataset:'Dataset'
    def getLabels(self):
        return self.dataset.labels[self.indices]

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
    labels:Union[None, torch.Tensor]
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
            label_desc:str = None,
            ret_id_as_label:bool = False):
        metadata = pd.read_csv(annotations_file, low_memory=False)
        
        self.label_desc = label_desc
        self.annotations_only = annotations_only
        self.img_file = None
        self.img_dir = None
        self.img_listing = None
        self.ret_id_as_label = ret_id_as_label

        if not self.annotations_only:
            self.img_file = img_file
            self.img_dir = img_dir
            self.img_listing_v, self.img_listing_o = (
                Dataset.strings_to_mem_safe_val_and_offset(metadata.loc[:, "isic_id"]))

        self.img_transform = img_transform
        self.annotation_transform = annotation_transform
        if self.annotation_transform:
            metadata = self.annotation_transform(metadata)
        self.annotations = None
        self.labels = None
        if self.label_desc:
            self.annotations = torch.tensor(
                metadata.drop(self.label_desc, axis=1).values,
                dtype=torch.float32)
            self.labels = torch.tensor(
                metadata[self.label_desc],
                dtype=torch.float32)
        else:
            self.annotations = torch.tensor(metadata.values, dtype=torch.float32)
            self.labels = None

    def __get_img_listing(self, idx):
        return Dataset.mem_safe_val_and_offset_to_string(
            self.img_listing_v,
            self.img_listing_o,
            idx)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = None
        label = torch.tensor(-1, dtype=torch.float32)
        annotations = self.annotations[idx]
        if self.label_desc:
            label = self.labels[idx]

        if self.annotations_only:
            data = annotations
        else:
            image_data = h5py.File(self.img_file, "r")
            listing = self.__get_img_listing(idx)
            if self.ret_id_as_label:
                label = listing
            image = np.array(image_data[listing])
            image = np.array(Image.open(io.BytesIO(image)),dtype=np.uint8)
            if self.img_transform:
                image = self.img_transform(image)
            data = (
                torch.tensor(image, dtype=torch.float32),
                annotations
                )

        return data, label

class SkinLesionsSmall(SkinLesions):
    def __init__(
            self,
            annotations_file: str | Path,
            img_file: str | Path = None,
            img_dir: str | Path = None,
            img_transform: nn.Module = None,
            annotation_transform: nn.Module = None,
            annotations_only: bool = False,
            label_desc: str = None,
            ret_id_as_label: bool = False,
            size:int = 2048):
        super().__init__(annotations_file, img_file, img_dir, img_transform, annotation_transform, annotations_only, label_desc, ret_id_as_label)
        rng = np.random.default_rng()
        if not self.labels is None:
            unq_lbls = self.labels.unique()
            lbl_masks = self.labels==unq_lbls[:,None]
            not_lbl_counts = (self.labels!=unq_lbls[:,None]).float().sum(-1)[:,None]
            weights = (lbl_masks*not_lbl_counts).sum(0).double()
            weights = weights/weights.sum()
            mask = rng.choice(len(self.labels), size, replace=False, p=weights)
            self.labels = self.labels[mask]
            self.annotations = self.annotations[mask]
        else:
            mask = rng.choice(len(self.annotations), size, replace=False)
            self.annotations = self.annotations[mask]

class DatasetReg(Registry):
    SkinLesions = SkinLesions
    SkinLesionsSmall = SkinLesionsSmall
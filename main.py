import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.decomposition import (
    PCA
)
from sklearn.preprocessing import (
    OrdinalEncoder
)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime

import pandas as pd
from PIL import Image

import os
import h5py
import io
from pathlib import Path
from typing import (
    List,
    Union,
    Dict,
    Any,
    Type
)

from quickscript.scripts import Script, ScriptChooser

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

class Select(nn.Module):
    def __init__(
            self,
            selections:List[str] = None,
            exclusions:List[str] = None,
            do_not_include:bool = False,
            exclude_uninformative:bool = True,
            *args,
            **kwargs) -> None:
        """
        Filter what is returned.

        Selections and exclusions are mutually exclusive options, only one
        should be used. Currently, selections and exclusions work only for
        pandas dataframes. Do_not_include will omit the entire object (any
        object) and return None instead, for purposes of saving memory.

        Args:
            selections: Specify what labels should be included.
            exclusions: Specify what labels should not be included.
            do_not_include: Omit entire input and instead return None.
            exclude_uninformative: Removes catagories for which all data is the 
              same or NaN.
        """
        super().__init__(*args, **kwargs)
        self.selections = selections
        self.exclusions = exclusions
        assert self.selections is None or self.exclusions is None
        self.do_not_include = do_not_include
        self.exclude_uninformative = exclude_uninformative
    
    def forward(self, x:Union[pd.DataFrame, pd.Series]):
        if self.do_not_include:
            return None
        else:
            sel = self.selections if self.selections else x._info_axis
            if self.exclusions:
                sel = sel.drop(self.exclusions)
            x = x[sel]
            if self.exclude_uninformative:
                uninf_mask = (x.loc[0] == x).all()
                x = x.drop(uninf_mask.index[uninf_mask], axis=1)
                x = x.dropna(axis=1, how='all')
            return x

class OrdinalEncoding(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs) -> None:
        """
        Transform text into an integer category.

        Args:
        """
        super().__init__(*args, **kwargs)
    
    def forward(self, x:Union[pd.DataFrame, pd.Series]):
        mask = x.dtypes == 'object'
        oe = OrdinalEncoder(dtype=int, encoded_missing_value=-1).fit_transform(x.loc[:,mask])
        x.loc[:,mask] = oe
        return x
    
class FillNaN(nn.Module):
    def __init__(
            self,
            selections:List[str] = None,
            fill_value:Union[int, List[int]] = -1,
            *args,
            **kwargs) -> None:
        """
        Transform text into an integer category.

        Args:
            selections: Specify what labels should be included.
            fill_value: Value with which to replace NaNs. Either a list of
              values the same size as selections, where each value will fill
              respectively, or a single value to be applied for all selections.
        """
        super().__init__(*args, **kwargs)
        self.selections = selections if selections else []
        self.fill_value = fill_value
        if not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value]*len(self.selections)
    
    def forward(self, x:Union[pd.DataFrame, pd.Series]):
        for i, selection in enumerate(self.selections):
            mask = x.loc[:,selection].isnull()
            x.loc[mask,selection] = self.fill_value[i]
        return x

class PreProcess(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for proc in self._modules.values():
            x = proc(x)
        return x

class PPPicture(PreProcess):
    def __init__(
        self,
        rescale_images:bool = True,
        omit:bool = False,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rescale = Rescale() if rescale_images else None
        self.select = Select(do_not_include=omit) if omit else None

class PPLabels(PreProcess):
    def __init__(
        self,
        selections = None,
        exclusions = None,
        ordinal_encoding:bool = True,
        fill_nan_selections = None,
        fill_nan_values = None,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.select = Select(selections=selections, exclusions=exclusions)
        self.one_hot = OrdinalEncoding() if ordinal_encoding else None
        fill_nan_config = {'selections':fill_nan_selections}
        if fill_nan_values:
            fill_nan_config['fill_value'] = fill_nan_values
        self.fill_nan = FillNaN(**fill_nan_config)

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

        if not self.annotations_only:
            self.image_data = h5py.File(img_file, "r")
            self.img_dir = img_dir

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
            data = annotations.drop(self.label_desc, axis=anno_is_df)
        else:
            image = np.array(self.image_data[self.metadata.loc[idx, "isic_id"]])
            image = np.array(Image.open(io.BytesIO(image)),dtype=np.uint8)
            if self.img_transform:
                image = self.img_transform(image)
            data = (image, annotations.drop(self.label_desc, axis=anno_is_df))

        label = annotations[self.label_desc]
        return data, label

class Utilities:
    class FEATURE_REDUCERS:
        LDA = LinearDiscriminantAnalysis # Reduces to fewer than number of classifications, only suitable for problems of multiple classes
        PCA = PCA
    class DATASETS:
        SkinLesions = SkinLesions

class TrainingScript(Script):
    data: Type[Dataset]
    save_path: Union[str, Path]
    def save_model(self, model):
        inp, target = next(iter(self.data))
        onx = convert_sklearn(model, initial_types=[("X", FloatTensorType([None, inp.shape[-1]]))])
        with open(self.save_path / "{}.onnx".format(str(model)), "wb") as f:
            f.write(onx.SerializeToString())

class ServingScript(Script):
    data: Type[Dataset]
    save_path: Union[str, Path]
    def load_model(self, load_path):
        class Model:
            def __init__(slf) -> None:
                slf.model = onnxruntime.InferenceSession(load_path)
            def __call__(slf, x:pd.DataFrame, *args: Any, **kwds: Any) -> Any:
                return slf.model.run(None, {"X":x.astype(np.float32).to_numpy()})
        return Model()

class FeatureReductionForTraining(TrainingScript):
    def __init__(
            self,
            dataset:Union[str, Utilities.DATASETS, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            feature_reducer:Union[str, Utilities.FEATURE_REDUCERS] = None,
            feature_reducer_kwargs:Dict[str, Any] = None,
            save_path:Union[str, Path] = None,
            **kwargs) -> None:
        """
        Args:
            dataset: The dataset class to be used.
            dataset_kwargs: Configuration for the dataset.
            feature_reducer: The feature reducer class to be used.
            feature_reducer_kwargs: Configuration for the feature reducer.
            save_path: Specify a path to which feature reducer should be saved.
        """
        super().__init__(**kwargs)
        self.data = dataset
        self.ds_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.feature_reducer = feature_reducer
        self.fr_kwargs = feature_reducer_kwargs if feature_reducer_kwargs else {}
        self.save_path = save_path
    
    def setup(self):
        self.data = (
            getattr(Utilities.DATASETS, self.data)
            if isinstance(self.data, str)
            else self.data)(**self.ds_kwargs)
        self.feature_reducer = (
            getattr(Utilities.FEATURE_REDUCERS, self.feature_reducer)
            if isinstance(self.feature_reducer, str)
            else self.feature_reducer)(**self.fr_kwargs)
        self.save_path = Path(self.save_path)

    def run(self):
        inp, tgt = self.data[:]
        self.feature_reducer.fit(inp, tgt)
        self.save_model(self.feature_reducer)

class Main(Script):
    def __init__(self, **kwargs) -> None:
        """
        Args:
        """
        super().__init__(**kwargs)
    
    def setup(self):
        pass

    def run(self):
        script = FeatureReductionForTraining(
            dataset=Utilities.DATASETS.SkinLesions,
            dataset_kwargs=dict(
                annotations_file="train-metadata.csv",
                img_file="train-image.hdf5",
                img_dir="train-image",
                img_transform=PPPicture(omit=True),
                annotation_transform=PPLabels(
                    exclusions=[
                        "isic_id",
                        "patient_id",
                        "attribution",
                        "copyright_license",
                        "lesion_id",
                        "iddx_full",
                        "iddx_1",
                        "iddx_2",
                        "iddx_3",
                        "iddx_4",
                        "iddx_5",
                        "mel_mitotic_index",
                        "mel_thick_mm",
                        "tbp_lv_dnn_lesion_confidence",
                        ],
                    fill_nan_selections=[
                        "age_approx",
                    ],
                    fill_nan_values=[-1, 0]
                    ),
                annotations_only=True
            ),
            feature_reducer = "PCA",
            feature_reducer_kwargs={
                "n_components":.9999
            },
            save_path="./models/feature_reduction"
            )
        script.setup()
        script.run()

if __name__ == "__main__":
    ScriptChooser().complete_run()
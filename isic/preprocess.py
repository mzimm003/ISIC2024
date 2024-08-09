from torch import nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    OrdinalEncoder
)
from typing import (
    List,
    Union,
)

class RescaleColor(nn.Module):
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

class Pad(nn.Module):
    def __init__(
            self,
            width=300,
            height=300,
            mode='edge',
            pass_larger_images:bool = False,
            *args,
            **kwargs) -> None:
        """
        Add pixels to images to provide a consistent size.

        Args:
            width: The desired total pixel width.
            height: The desired total pixel height.
            mode: The means by which the image should be padded.
        """
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.mode = mode
        self.pass_larger_images = pass_larger_images
    
    def forward(self, x):
        h,w,c = x.shape
        pad_h = self.height-h
        pad_w = self.width-w
        if self.pass_larger_images:
            pad_h = max(pad_h,0)
            pad_w = max(pad_w,0)
        return np.pad(
            x,
            ((0,pad_h),(0,pad_w),(0,0)),
            mode=self.mode)
    
class Crop(nn.Module):
    def __init__(
            self,
            width=125,
            height=125,
            seed=None,
            *args,
            **kwargs) -> None:
        """
        Take away pixels from images to provide a consistent size.

        Args:
            width: The desired total pixel width.
            height: The desired total pixel height.
            seed: Seed for rng to create reproducible results if desired.
        """
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed=seed)
    
    def forward(self, x):
        h,w,c = x.shape
        start_x = self.rng.integers(w-self.width)
        start_y = self.rng.integers(h-self.height)
        return x[start_x:start_x+self.width, start_y:start_y+self.height, :]

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
            dtype=np.int64,
            *args,
            **kwargs) -> None:
        """
        Transform text into an integer category.

        Args:
            dtype: Desired dtype of replacement values
        """
        super().__init__(*args, **kwargs)
        self.dtype = dtype
    
    def forward(self, x:Union[pd.DataFrame, pd.Series]):
        mask = x.dtypes == 'object'
        oe = OrdinalEncoder(dtype=self.dtype, encoded_missing_value=-1).fit_transform(x.loc[:,mask])
        x.loc[:,mask] = oe
        x[mask.index[mask]] = x[mask.index[mask]].astype(self.dtype)
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
        pad_mode:str = None,
        pad_width=200,
        pad_height=200,
        pass_larger_images:bool = False,
        crop:bool = False,
        crop_width=125,
        crop_height=125,
        seed:int = None,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rescale = RescaleColor() if rescale_images else None
        self.select = Select(do_not_include=omit) if omit else None
        self.pad = Pad(
            width=pad_width,
            height=pad_height,
            mode=pad_mode,
            pass_larger_images=pass_larger_images) if pad_mode else None
        self.crop = Crop(width=crop_width, height=crop_height, seed=seed) if crop else None

class PPLabels(PreProcess):
    def __init__(
        self,
        selections = None,
        exclusions = None,
        exclude_uninformative = True,
        ordinal_encoding:bool = True,
        fill_nan_selections = None,
        fill_nan_values = None,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.select = Select(selections=selections, exclusions=exclusions, exclude_uninformative=exclude_uninformative)
        self.one_hot = OrdinalEncoding() if ordinal_encoding else None
        fill_nan_config = {'selections':fill_nan_selections}
        if fill_nan_values:
            fill_nan_config['fill_value'] = fill_nan_values
        self.fill_nan = FillNaN(**fill_nan_config)

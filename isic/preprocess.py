from torch import nn
import pandas as pd
from sklearn.preprocessing import (
    OrdinalEncoder
)
from typing import (
    List,
    Union,
)

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

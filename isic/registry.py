from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.decomposition import (
    PCA,
    FastICA
)

import onnx
import onnxruntime

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import (
    SGD,
    Adam,
)
from torch.optim.lr_scheduler import (
    CyclicLR
)

from typing import (
    Any,
    TypeVar,
    TypeAlias,
    Union,
    List,
    Dict
)
from enum import Enum

T = TypeVar("T")
IterOptional:TypeAlias = Union[T, List[T]]

class Registry(Enum):
    @classmethod
    def initialize(cls, obj, kwargs):
        if isinstance(obj, str):
            obj = getattr(cls, obj)
        if isinstance(obj, Registry):
            obj = obj.value
        return obj(**kwargs)
    
    @staticmethod
    def load_model(load_path, cuda=True):
        onnx.checker.check_model(onnx.load(load_path))
        class Model:
            type_to_np = {
                "tensor(double)":np.float64,
                "tensor(float)":np.float32,
                "tensor(float16)":np.float16
            }
            type_to_torch = {
                "tensor(double)":torch.float64,
                "tensor(float)":torch.float32,
                "tensor(float16)":torch.float16
            }
            def __init__(slf) -> None:
                slf.model = onnxruntime.InferenceSession(load_path, providers = [
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider',
                    ] if cuda else ['CPUExecutionProvider'])
                slf.binding = slf.model.io_binding()
            def __call__(slf, *args: Any, **kwds:torch.Tensor) -> Any:
                # Assume positional args are fed for expected keywords
                if not kwds and args:
                    expected_kwds = [k.name for k in slf.model.get_inputs()]
                    for i, key in enumerate(expected_kwds):
                        kwds[key] = args[i]
                
                val_samp = None
                for arg in slf.model.get_inputs():
                    value = kwds[arg.name].to(dtype=Model.type_to_torch[arg.type]).contiguous()
                    val_samp = value
                    slf.binding.bind_input(
                        name=arg.name,
                        device_type='cuda' if cuda else 'cpu',
                        device_id=0,
                        element_type=Model.type_to_np[arg.type],
                        shape=value.shape,
                        buffer_ptr=value.data_ptr(),
                    )
                outs = []
                for op in slf.model.get_outputs():
                    out_shape = (val_samp.shape[0], *op.shape[1:])
                    out = torch.empty(
                        out_shape,
                        dtype=val_samp.dtype,
                        device=val_samp.device)
                    slf.binding.bind_output(
                        name=op.name,
                        device_type='cuda' if cuda else 'cpu',
                        device_id=0,
                        element_type=Model.type_to_np[op.type],
                        shape=out_shape,
                        buffer_ptr=out.data_ptr(),
                    )
                    outs.append(out)
                slf.model.run_with_iobinding(slf.binding)
                return outs
        return Model()

class FeatureReducersReg(Registry):
    LDA = LinearDiscriminantAnalysis # Reduces to fewer than number of classifications, only suitable for problems of multiple classes
    PCA = PCA
    ICA = FastICA

class ActivationReg(Registry):
    relu = nn.ReLU
    sig = nn.Sigmoid

class OptimizerReg(Registry):
    SGD = SGD
    adam = Adam

    @classmethod
    def initialize(cls, obj, parameters, kwargs):
        kwargs['params'] = parameters
        return super().initialize(obj, kwargs)

# def BatchBasedLRScheduler(cls):
#     class BatchBasedLRScheduler(cls):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#     return BatchBasedLRScheduler

# def EpochBasedLRScheduler(cls):
#     class EpochBasedLRScheduler(cls):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#     return EpochBasedLRScheduler

# class BaseScheduler:
#     pass

class LRSchedulerReg(Registry):
    #TODO, this can be better somehow, meant to serve Trainer in Main.
    #  See "step_lr_scheduler"
    CyclicLR = CyclicLR
    # BatchBasedLRScheduler = BatchBasedLRScheduler(BaseScheduler)
    # EpochBasedLRScheduler = EpochBasedLRScheduler(BaseScheduler)

    @classmethod
    def initialize(cls, obj, optimizer, kwargs):
        kwargs['optimizer'] = optimizer
        return super().initialize(obj, kwargs)

class CriterionReg(Registry):
    MSE = nn.MSELoss
    cross_entropy = nn.CrossEntropyLoss
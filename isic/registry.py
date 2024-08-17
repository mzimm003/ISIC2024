from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.decomposition import (
    PCA,
    FastICA
)

import onnxruntime

import pandas as pd
import numpy as np
from torch import nn
from torch.optim import (
    SGD,
    Adam
)

from typing import (
    Any
)
from enum import Enum

class Registry(Enum):
    @classmethod
    def initialize(cls, obj, kwargs):
        if isinstance(obj, str):
            obj = getattr(cls, obj)
        if isinstance(obj, Registry):
            obj = obj.value
        return obj(**kwargs)
    
    @staticmethod
    def load_model(load_path, cuda=False):
        class Model:
            def __init__(slf) -> None:
                slf.model = onnxruntime.InferenceSession(load_path, providers = [
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider',
                    ] if cuda else ['CPUExecutionProvider'])
            def __call__(slf, *args: Any, **kwds: Any) -> Any:
                return slf.model.run(None, kwds)[0]
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

class CriterionReg(Registry):
    MSE = nn.MSELoss
    cross_entropy = nn.CrossEntropyLoss
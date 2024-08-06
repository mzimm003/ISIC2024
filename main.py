import torch
import torch.utils
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Optimizer

from sklearn.model_selection import KFold
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import numpy as np

import json
from pathlib import Path
from typing import (
    List,
    Union,
    Dict,
    Any,
    Type,
    Callable
)

from quickscript.scripts import Script, ScriptChooser

from isic.preprocess import(
    PPPicture,
    PPLabels
)

from isic.registry import (
    Registry,
    FeatureReducersReg,
    ActivationReg,
    OptimizerReg,
    CriterionReg)
from isic.models import ModelReg
from isic.datasets import DatasetReg

class TrainingScript(Script):
    data: Type[Dataset]
    save_path: Union[str, Path]
    def save_model(self, model):
        inp, target = next(iter(self.data))
        onx = convert_sklearn(model, initial_types=[("X", FloatTensorType([None, inp.shape[-1]]))])
        save_file = self.save_path / "{}/model.onnx".format(str(model))
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)
        with open(save_file, "wb") as f:
            f.write(onx.SerializeToString())

    def save_results(self, model, results):
        save_file = self.save_path / "{}/results.json".format(str(model))
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)
        with open(save_file, "wb") as f:
            json.dump(results, f)
    
    def process_data(self, data, model:nn.Module):
        param_ref = next(iter(model.parameters()))
        inp = data.inp.to(device=param_ref.device, dtype=param_ref.dtype)
        target = data.tgt.to(device=param_ref.device, dtype=param_ref.dtype)
        return inp, target
    
    def process_output(self, output):
        return output
    
    def infer_itr(
            self,
            model:nn.Module,
            optimizer:nn.Module,
            criterion:Union[nn.Module, Callable],
            inp,
            target,
            return_acc=False,
            return_prec=False,
            return_recall=False):
        ret = {}
        if model.training:
            optimizer.zero_grad()
        
        output = model(inp)
        
        loss:torch.Tensor = criterion(output, target)

        if model.training:
            loss.backward()
            optimizer.step()
        ret['loss'] = loss.item()

        result = self.process_output(output)
        if return_acc:
            ret['acc'] = ((result.int() == target.int()).sum()/target.numel()).item()
        if return_prec:
            ret['prec'] = (target.int()[result.int() == 1].sum()/(result.int() == 1).sum()).item()
        if return_recall:
            ret['recall'] = (target.int()[result.int() == 1].sum()/(target.int() == 1).sum()).item()
        return ret

class FeatureReductionForTraining(TrainingScript):
    def __init__(
            self,
            dataset:Union[str, DatasetReg, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            feature_reducer:Union[str, FeatureReducersReg] = None,
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
        self.data = DatasetReg.initialize(
            self.data, self.ds_kwargs)
        self.feature_reducer = FeatureReducersReg.initialize(
            self.feature_reducer, self.fr_kwargs)
        self.save_path = Path(self.save_path)

    def run(self):
        inp, tgt = self.data[:]
        self.feature_reducer.fit(inp, tgt)
        self.save_model(self.feature_reducer)

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

class ClassifierTraining(TrainingScript):
    def __init__(
            self,
            dataset:Union[str, DatasetReg, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            classifier:Union[str, ModelReg] = None,
            classifier_kwargs:Dict[str, Any] = None,
            optimizer:Union[OptimizerReg, Type[Optimizer]]= None,
            optimizer_kwargs:Dict[str, Any] = None,
            criterion:Union[CriterionReg, Type[torch.nn.modules.loss._Loss]] = None,
            criterion_kwargs:Dict[str, Any] = None,
            save_path:Union[str, Path] = None,
            k_fold_splits:int = 4,
            batch_size:int = 64,
            shuffle:bool = True,
            num_workers:int = None,
            **kwargs) -> None:
        """
        Args:
            dataset: The dataset class to be used.
            dataset_kwargs: Configuration for the dataset.
            classifier: The classifier class to be used.
            classifier_kwargs: Configuration for the classifier.
            save_path: Specify a path to which classifier should be saved.
            k_fold_splits: Number of folds used for dataset in training.
        """
        super().__init__(**kwargs)
        self.data = dataset
        self.ds_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.kf = KFold(n_splits=k_fold_splits)
        self.dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_wrapper,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=2
            )

        self.classifier:nn.Module = classifier
        self.classifier_kwargs = classifier_kwargs if classifier_kwargs else {}
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs else {}
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs if criterion_kwargs else {}
        self.save_path = save_path
    
    def setup(self):
        self.data = DatasetReg.initialize(
            self.data, self.ds_kwargs)
        self.classifier = ModelReg.initialize(
            self.classifier, self.classifier_kwargs)
        self.classifier.to(
            device=(torch.device('cuda') if torch.cuda.is_available() else 'cpu'))
        self.optimizer = ModelReg.initialize(
            self.optimizer, self.optimizer_kwargs)
        self.criterion = ModelReg.initialize(
            self.criterion, self.criterion_kwargs)
        self.save_path = Path(self.save_path)

    def run(self):
        metrics = {}
        metrics["k_fold_train_loss"] = []
        metrics["k_fold_val_loss"] = []
        metrics["k_fold_val_acc"] = []
        for i, (train_fold, val_fold) in enumerate(self.kf.get_n_splits(self.data)):
            train_loader = DataLoader(
                Subset(self.data, train_fold),
                ** self.dl_kwargs
            )
            val_loader = DataLoader(
                Subset(self.data, val_fold),
                ** self.dl_kwargs
            )

            self.classifier.train()
            metrics["k_fold_train_loss"].append(0)
            for data in train_loader:
                inp, tgt = self.process_data(data, self.classifier)

                result = self.infer_itr(
                    model=self.classifier,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    inp=inp,
                    target=tgt
                )
                metrics["k_fold_train_loss"][i] += result['loss']
            metrics["k_fold_train_loss"][i] = (
                metrics["k_fold_train_loss"][i] / len(train_loader))
            
            self.classifier.eval()
            metrics["k_fold_val_loss"].append(0)
            metrics["k_fold_val_acc"].append(0)
            for data in val_loader:
                inp, tgt = self.process_data(data, self.classifier)

                result = self.infer_itr(
                    model=self.classifier,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    inp=inp,
                    target=tgt,
                    return_acc=True
                )
                metrics["k_fold_val_loss"][i] += result['loss']
                metrics["k_fold_val_acc"][i] += result['loss']
            metrics["k_fold_val_loss"][i] = (
                metrics["k_fold_val_loss"][i] / len(train_loader))
            metrics["k_fold_val_acc"][i] = (
                metrics["k_fold_val_acc"][i] / len(train_loader))

        # Summarize training epoch
        metrics["mean_train_loss"] = np.mean(metrics["k_fold_train_loss"])
        metrics["mean_val_loss"] = np.mean(metrics["k_fold_val_loss"])
        metrics["mean_val_acc"] = np.mean(metrics["k_fold_val_acc"])
        self.save_model(self.classifier)
        self.save_results(self.classifier, metrics)

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
            dataset=DatasetReg.SkinLesions,
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
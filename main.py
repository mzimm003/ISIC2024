import torch
import torch.utils
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Optimizer

from sklearn.model_selection import KFold
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import numpy as np

import os
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
from functools import partial

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
    def save_model(self, model, data_input_sample):
        if not isinstance(data_input_sample, tuple):
            data_input_sample = (data_input_sample,)
        onx = None
        if isinstance(model, nn.Module):
            onx = torch.onnx.dynamo_export(model, *data_input_sample)
        else:
            init_types = [("X{}".format(i), FloatTensorType([None, inp_i.shape[-1]]))
                        for i, inp_i in enumerate(data_input_sample)]
            onx = convert_sklearn(model, initial_types=init_types)
        save_file = self.save_path / "{}/model.onnx".format(self.get_model_name(model))
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)
        if isinstance(model, nn.Module):
            onx.save(str(save_file))
        else:
            with open(save_file, "wb") as f:
                f.write(onx.SerializeToString())

    def save_results(self, model, results):
        save_file = self.save_path / "{}/results.json".format(self.get_model_name(model))
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)
        with open(save_file, "w") as f:
            json.dump(results, f)
    
    def get_model_name(self, model):
        name = model.name() if hasattr(model, "name") else str(model)
        return name

    def process_data(self, data, model:nn.Module):
        param_ref = next(iter(model.parameters()))
        img = data.img.to(device=param_ref.device, dtype=param_ref.dtype)
        fet = data.fet.to(device=param_ref.device, dtype=param_ref.dtype)
        target = data.tgt.to(device=param_ref.device)
        return img, fet, target
    
    def process_output(self, output:torch.Tensor):
        return output.max(-1).indices
    
    def infer_itr(
            self,
            model:nn.Module,
            optimizer:nn.Module,
            criterion:Union[nn.Module, Callable],
            img,
            fet,
            target,
            return_acc=False,
            return_prec=False,
            return_recall=False,
            metrics_as_totals=False):
        ret = {}
        if model.training:
            optimizer.zero_grad()
        
        output = model(img, fet)
        
        loss:torch.Tensor = criterion(output, target)

        if model.training:
            loss.backward()
            optimizer.step()
        ret['loss'] = loss.item()

        result = self.process_output(output)
        if return_acc:
            num = (result.int() == target.int()).sum()
            denom = target.numel()
            if metrics_as_totals:
                ret['acc'] = (num.item(), denom)
            else:
                ret['acc'] = (num/denom).item()
        if return_prec:
            num = target.int()[result.int() == 1].sum()
            denom = (result.int() == 1).sum()
            if metrics_as_totals:
                ret['prec'] = (num.item(), denom.item())
            else:
                ret['prec'] = (num/denom).item()
        if return_recall:
            num = target.int()[result.int() == 1].sum()
            denom = (target.int() == 1).sum()
            if metrics_as_totals:
                ret['recall'] = (num.item(), denom.item())
            else:
                ret['recall'] = (num/denom).item()
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
        self.save_model(self.feature_reducer, inp)

class SimpleCustomBatch:
    def __init__(self, data, feature_reducer):
        transposed_data = list(zip(*data))
        transposed_inps = list(zip(*transposed_data[0]))
        self.img = torch.stack(transposed_inps[0], 0)
        self.fet = torch.stack(transposed_inps[1], 0)
        self.fet = torch.tensor(feature_reducer(self.fet.numpy()))
        self.tgt = torch.stack(transposed_data[1], 0).long()

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.img = self.img.pin_memory()
        self.fet = self.fet.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch, feature_reducer):
    return SimpleCustomBatch(batch, feature_reducer)

class ClassifierTraining(TrainingScript):
    def __init__(
            self,
            dataset:Union[str, DatasetReg, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            feature_reducer_path:Union[str, Path] = None,
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
            num_workers:int = 0,
            **kwargs) -> None:
        """
        Args:
            dataset: The dataset class to be used.
            dataset_kwargs: Configuration for the dataset.
            feature_reducer_path: Path to feature reducing model, if desired.
            classifier: The classifier class to be used.
            classifier_kwargs: Configuration for the classifier.
            optimizer: The optimizer class to be used.
            optimizer_kwargs : Configuration for the optimizer.
            criterion: The criterion class to be used.
            criterion_kwargs : Configuration for the criterion.
            save_path: Specify a path to which classifier should be saved.
            k_fold_splits: Number of folds used for dataset in training.
            batch_size: Number of data points included in training batches.
            shuffle: Whether to shuffle data points.
            num_workers: For multiprocessing.
        """
        super().__init__(**kwargs)
        self.data = dataset
        self.ds_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.feature_reducer_path = feature_reducer_path
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
        if self.feature_reducer_path:
            self.feature_reducer = Registry.load_model(self.feature_reducer_path)
            self.dl_kwargs['collate_fn'] = partial(
                self.dl_kwargs['collate_fn'],
                feature_reducer=self.feature_reducer
            )
        else:
            self.dl_kwargs['collate_fn'] = partial(
                self.dl_kwargs['collate_fn'],
                feature_reducer=nn.Identity()
            )
        self.classifier = ModelReg.initialize(
            self.classifier, self.classifier_kwargs)
        device = (torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        print("Expected device:{}".format(device))
        self.classifier.to(
            device=device)
        self.optimizer = OptimizerReg.initialize(
            self.optimizer, self.classifier.parameters(), self.optimizer_kwargs)
        self.criterion = CriterionReg.initialize(
            self.criterion, self.criterion_kwargs)
        self.save_path = Path(self.save_path)

    def run(self):
        metrics = {}
        metrics["k_fold_train_loss"] = []
        metrics["k_fold_val_loss"] = []
        metrics["k_fold_val_acc"] = []
        metrics["k_fold_val_prec"] = []
        metrics["k_fold_val_prec_num"] = []
        metrics["k_fold_val_prec_denom"] = []
        metrics["k_fold_val_recall"] = []
        metrics["k_fold_val_recall_num"] = []
        metrics["k_fold_val_recall_denom"] = []
        data_inp_sample = None
        for i, (train_fold, val_fold) in enumerate(self.kf.split(self.data)):
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
                img, fet, tgt = self.process_data(data, self.classifier)
                data_inp_sample = (img, fet)

                result = self.infer_itr(
                    model=self.classifier,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    img=img,
                    fet=fet,
                    target=tgt,
                )
                metrics["k_fold_train_loss"][i] += result['loss']
            metrics["k_fold_train_loss"][i] = (
                metrics["k_fold_train_loss"][i] / len(train_loader))
            
            self.classifier.eval()
            metrics["k_fold_val_loss"].append(0)
            metrics["k_fold_val_acc"].append(0)
            metrics["k_fold_val_prec"].append(0)
            metrics["k_fold_val_recall"].append(0)
            metrics["k_fold_val_prec_num"].append(0)
            metrics["k_fold_val_prec_denom"].append(0)
            metrics["k_fold_val_recall_num"].append(0)
            metrics["k_fold_val_recall_denom"].append(0)
            for data in val_loader:
                img, fet, tgt = self.process_data(data, self.classifier)

                result = self.infer_itr(
                    model=self.classifier,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    img=img,
                    fet=fet,
                    target=tgt,
                    return_acc=True,
                    return_prec=True,
                    return_recall=True,
                    metrics_as_totals=True
                )
                metrics["k_fold_val_loss"][i] += result['loss']
                metrics["k_fold_val_acc"][i] += result['acc'][0]/result['acc'][1]
                metrics["k_fold_val_prec_num"][i] += result['prec'][0]
                metrics["k_fold_val_prec_denom"][i] += result['prec'][1]
                metrics["k_fold_val_recall_num"][i] += result['recall'][0]
                metrics["k_fold_val_recall_denom"][i] += result['recall'][1]
            metrics["k_fold_val_loss"][i] = (
                metrics["k_fold_val_loss"][i] / len(val_loader))
            metrics["k_fold_val_acc"][i] = (
                metrics["k_fold_val_acc"][i] / len(val_loader))
            metrics["k_fold_val_prec"][i] = (
                np.float64(metrics["k_fold_val_prec_num"][i]) / metrics["k_fold_val_prec_denom"][i])
            metrics["k_fold_val_recall"][i] = (
                np.float64(metrics["k_fold_val_recall_num"][i]) / metrics["k_fold_val_recall_denom"][i])

        # Summarize training epoch
        metrics["mean_train_loss"] = np.mean(metrics["k_fold_train_loss"])
        metrics["mean_val_loss"] = np.mean(metrics["k_fold_val_loss"])
        metrics["mean_val_acc"] = np.mean(metrics["k_fold_val_acc"])
        metrics["mean_val_prec"] = np.mean(metrics["k_fold_val_prec"])
        metrics["mean_val_recall"] = np.mean(metrics["k_fold_val_recall"])
        self.save_model(self.classifier, data_inp_sample)
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
        script = ClassifierTraining(
            dataset=DatasetReg.SkinLesions,
            dataset_kwargs=dict(
                annotations_file="train-metadata.csv",
                img_file="train-image.hdf5",
                img_dir="train-image",
                img_transform=PPPicture(pad_mode='edge'),
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
                    fill_nan_values=[-1, 0],
                    ),
            ),
            feature_reducer_path="./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            classifier=ModelReg.Classifier,
            classifier_kwargs=dict(
                activation=ActivationReg.relu,
            ),
            optimizer=OptimizerReg.adam,
            criterion=CriterionReg.cross_entropy,
            save_path="./models/classifier",
            num_workers=os.cpu_count()-1,
            )
        # script = FeatureReductionForTraining(
        #     dataset=DatasetReg.SkinLesions,
        #     dataset_kwargs=dict(
        #         annotations_file="train-metadata.csv",
        #         img_file="train-image.hdf5",
        #         img_dir="train-image",
        #         img_transform=PPPicture(omit=True),
        #         annotation_transform=PPLabels(
        #             exclusions=[
        #                 "isic_id",
        #                 "patient_id",
        #                 "attribution",
        #                 "copyright_license",
        #                 "lesion_id",
        #                 "iddx_full",
        #                 "iddx_1",
        #                 "iddx_2",
        #                 "iddx_3",
        #                 "iddx_4",
        #                 "iddx_5",
        #                 "mel_mitotic_index",
        #                 "mel_thick_mm",
        #                 "tbp_lv_dnn_lesion_confidence",
        #                 ],
        #             fill_nan_selections=[
        #                 "age_approx",
        #             ],
        #             fill_nan_values=[-1, 0]
        #             ),
        #         annotations_only=True
        #     ),
        #     feature_reducer = "PCA",
        #     feature_reducer_kwargs={
        #         "n_components":.9999
        #     },
        #     save_path="./models/feature_reduction"
        #     )
        script.setup()
        script.run()

if __name__ == "__main__":
    ScriptChooser().complete_run()
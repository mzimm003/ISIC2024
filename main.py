from ray.train._internal.storage import StorageContext
from ray.tune.logger import Logger
import sklearn.model_selection
import torch
import torch.utils
from torch import nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Optimizer

from sklearn.model_selection import StratifiedKFold
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import numpy as np
import pandas as pd
import sklearn

import os
import json
from pathlib import Path
from collections import namedtuple
from typing import (
    List,
    Union,
    Dict,
    Any,
    Type,
    Callable
)
from typing_extensions import override
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
from isic.datasets import DatasetReg, train_test_split, collate_wrapper

import ray
from ray import tune
import ray.air as air

def trainable_wrap(
        script:Type["TrainingScript"] = None,
        num_cpus:int=1,
        num_gpus:float=0.):
    class TrainableWrapper(tune.Trainable):
        @override
        def setup(self, config:dict):
            self.script = script(**config)
            self.script.setup()
        @override
        def step(self):
            return self.script.run()
        @override
        def save_checkpoint(self, checkpoint_dir: str) -> Union[Dict, None]:
            data_samp = self.script.get_data_sample()
            for i, mod in self.script.get_models():
                self.script.save_model(mod, data_samp, i, save_dir=checkpoint_dir)
    return tune.with_resources(TrainableWrapper, resources={"CPU":num_cpus, "GPU":num_gpus})

class TrainingScript(Script):
    data: Type[Dataset]
    save_path: Union[str, Path]
    training_manager: 'TrainingManager'

    def get_models(self)-> tuple:
        return self.training_manager.models.items()
    
    def get_data_sample(self)-> dict:
        dataset = self.training_manager.datasets['train'][0]
        *inp, _ = self.process_data(
            next(iter(dataset)),
            self.training_manager.models[0])
        return tuple(inp)
    
    def save_model(self, model, data_input_sample, suffix="", save_dir=None):
        if not isinstance(data_input_sample, tuple):
            data_input_sample = (data_input_sample,)
        onx = None
        save_path = Path(save_dir) if save_dir else self.save_path
        save_file = save_path / "{}{}/model.onnx".format(self.get_model_name(model), suffix)
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)

        if isinstance(model, nn.Module):
            # export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
            onx = torch.onnx.export(
                model,
                data_input_sample,
                input_names=["img","fet"],
                f = save_file,
                dynamic_axes={
                    "img": {0: "batch"},
                    "fet": {0: "batch"},
                }
            )
            # onx.save(str(save_file))
                # export_options=export_options)
        else:
            init_types = [("X{}".format(i), FloatTensorType([None, inp_i.shape[-1]]))
                        for i, inp_i in enumerate(data_input_sample)]
            onx = convert_sklearn(model, initial_types=init_types)
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

class TrainingManager:
    def __init__(
        self,
        data,
        dl_kwargs:dict,
        num_splits:int,
        shuffle:bool,
        model:Union[str, ModelReg] = None,
        model_kwargs:Dict[str, Any] = None,
        optimizer:Union[OptimizerReg, Type[Optimizer]]= None,
        optimizer_kwargs:Dict[str, Any] = None,
        criterion:Union[CriterionReg, Type[torch.nn.modules.loss._Loss]]= None,
        criterion_kwargs:Dict[str, Any] = None,
        ):
        self.num_splits = num_splits
        self.shuffle = shuffle
        self.data = data
        self.dl_kwargs = dl_kwargs
        
        self.model = model
        self.model_kwargs = model_kwargs
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion = CriterionReg.initialize(
            criterion, criterion_kwargs)

        self.device = (torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        print("Expected device:{}".format(self.device))

        self.datasets = {}
        self.models = {}
        self.optimizers = {}
        self.create_models()
        print("Models created.")
        self.create_dataloaders()
        print("Dataloaders created.")

        self.criterion.to(device=next(self.models[0].parameters()).device)
    
    def __getitem__(self, index):
        if not 0 <= index < len(self.models):
            raise IndexError
        TrainElements = namedtuple("TrainElements",[
            "train_data","val_data","model","optimizer","criterion"])
        return TrainElements(
            self.datasets["train"][index],
            self.datasets["validation"][index],
            self.models[index],
            self.optimizers[index],
            self.criterion
            )

    def create_dataloaders(self):
        self.datasets['train'] = {}
        self.datasets['validation'] = {}
        split_idxs = None
        if self.num_splits == 1:
            split_idxs = train_test_split(
                self.data,
                train_size=0.8,
                test_size=0.2,
                shuffle=self.shuffle,
                stratify=[self.data.labels]
                )
            print("Split done.")
        else:
            split_idxs = StratifiedKFold(
                n_splits=self.num_splits,
                shuffle=self.shuffle,
                ).split(self.data, self.data.labels)
        for i, (train_fold, val_fold) in enumerate(split_idxs):
            print("Start dataloaders.")
            self.datasets['train'][i] = DataLoader(
                Subset(self.data, train_fold),
                ** self.dl_kwargs
                )
            print("Train dataloader done.")
            self.datasets['validation'][i] = DataLoader(
                Subset(self.data, val_fold),
                ** self.dl_kwargs
                )
            print("Val dataloader done.")
        
    def create_models(self):
        for i in range(self.num_splits):
            model = ModelReg.initialize(
                self.model, self.model_kwargs)
            model.to(device=self.device)
            self.models[i] = model
            self.optimizers[i] = OptimizerReg.initialize(
                self.optimizer, model.parameters(), self.optimizer_kwargs)
                
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
        self.k_fold_splits = k_fold_splits
        self.training_manager = None
        self.dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_wrapper,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None
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
        print("Dataset initialized.")
        if self.feature_reducer_path:
            self.feature_reducer = Registry.load_model(self.feature_reducer_path)
            self.dl_kwargs['collate_fn'] = partial(
                self.dl_kwargs['collate_fn'],
                feature_reducer=self.feature_reducer
            )
        else:
            self.dl_kwargs['collate_fn'] = partial(
                self.dl_kwargs['collate_fn'],
                feature_reducer=lambda X0: nn.Identity()(X0)
            )
        self.training_manager = TrainingManager(
            data=self.data,
            dl_kwargs=self.dl_kwargs,
            num_splits=self.k_fold_splits,
            shuffle=self.dl_kwargs["shuffle"],
            model=self.classifier,
            model_kwargs=self.classifier_kwargs,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            criterion=self.criterion,
            criterion_kwargs=self.criterion_kwargs,
        )
        print("Training manager initialized.")
        self.save_path = Path(self.save_path)
        print("Completed setup.")

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
        print("Start epoch.")
        for i, (train_data, val_data, classifier, optimizer, criterion) in enumerate(self.training_manager):
            print("Training manager iterated.")
            classifier.train()
            metrics["k_fold_train_loss"].append(0)
            for data in train_data:
                img, fet, tgt = self.process_data(data, classifier)

                result = self.infer_itr(
                    model=classifier,
                    optimizer=optimizer,
                    criterion=criterion,
                    img=img,
                    fet=fet,
                    target=tgt,
                )
                metrics["k_fold_train_loss"][i] += result['loss']
            metrics["k_fold_train_loss"][i] = (
                metrics["k_fold_train_loss"][i] / len(train_data))
            
            classifier.eval()
            metrics["k_fold_val_loss"].append(0)
            metrics["k_fold_val_acc"].append(0)
            metrics["k_fold_val_prec"].append(0)
            metrics["k_fold_val_recall"].append(0)
            metrics["k_fold_val_prec_num"].append(0)
            metrics["k_fold_val_prec_denom"].append(0)
            metrics["k_fold_val_recall_num"].append(0)
            metrics["k_fold_val_recall_denom"].append(0)
            for data in val_data:
                img, fet, tgt = self.process_data(data, classifier)

                result = self.infer_itr(
                    model=classifier,
                    optimizer=optimizer,
                    criterion=criterion,
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
                metrics["k_fold_val_loss"][i] / len(val_data))
            metrics["k_fold_val_acc"][i] = (
                metrics["k_fold_val_acc"][i] / len(val_data))
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
        return metrics

class Notebook(Script):
    def __init__(
            self,
            dataset:Union[str, DatasetReg, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            feature_reducer_path:Union[str, Path] = None,
            classifier_path:Union[str, Path] = None,
            save_path:Union[str, Path] = None,
            batch_size:int = 64,
            shuffle:bool = False,
            num_workers:int = 0,
            **kwargs) -> None:
        """
        Args:
            dataset: The dataset class to be used.
            dataset_kwargs: Configuration for the dataset.
            feature_reducer_path: Path to feature reducing model, if desired.
            classifier_path: Path to classifier model.
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
        self.classifier_path = classifier_path
        self.dl_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_wrapper,
            pin_memory=True,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None
            )
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
        self.classifier = Registry.load_model(self.classifier_path)
        self.save_path = Path(self.save_path)

    def run(self):
        test_loader = DataLoader(
            self.data,
            ** self.dl_kwargs
        )

        sub = []

        for data in test_loader:
            img = data.img
            fet = data.fet

            result = self.classifier(img=img.numpy().astype(np.float32), fet=fet.numpy().astype(np.float32))
            mal_confidence = torch.tensor(result).softmax(1)[:,1]
            sub.extend(list(zip(data.tgt, mal_confidence.numpy())))
        
        sub = pd.DataFrame(sub, columns=['isic_id','target'])
        sub.to_csv(self.save_path/"submission.csv", index=False)

class Main(Script):
    def __init__(
            self,
            debug:bool=False,
            script:str="train_class_ray",
            **kwargs) -> None:
        """
        Args:
            debug: Flag to run script for debugging.
            script: Premade script configuration to be run.
        """
        super().__init__(**kwargs)
        self.debug = debug
        self.script = script

    def createSubmission(self):
        script = Notebook(
            dataset=DatasetReg.SkinLesions,
            dataset_kwargs=dict(
                annotations_file="test-metadata.csv",
                img_file="test-image.hdf5",
                img_transform=PPPicture(pad_mode='edge', pass_larger_images=True, crop=True),
                annotation_transform=PPLabels(
                    exclusions=[
                        "isic_id",
                        "patient_id",
                        "attribution",
                        "copyright_license",
                        "image_type"
                        ],
                    exclude_uninformative=False,
                    fill_nan_selections=[
                        "age_approx",
                    ],
                    fill_nan_values=[-1, 0],
                    ),
                ret_id_as_label=True,
            ),
            feature_reducer_path="./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            classifier_path="./models/classifier/test/model.onnx",
            save_path=".",
            num_workers=os.cpu_count()-1 if not self.debug else 0,
            )
        script.setup()
        script.run()

    def trainFeatureReducer(self):
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
                annotations_only=True,
                label_desc='target',
            ),
            feature_reducer = "PCA",
            feature_reducer_kwargs={
                "n_components":.9999
            },
            save_path="./models/feature_reduction"
            )
        script.setup()
        script.run()

    def trainClassifierRay(self):
        num_trials = 4
        num_cpus = 1 if self.debug else os.cpu_count()
        num_gpus = 0 if self.debug else torch.cuda.device_count()
        cpu_per_trial = num_cpus//num_trials
        gpu_per_trial = num_gpus/num_trials
        annotations_file=Path("train-metadata.csv").resolve()
        img_file=Path("train-image.hdf5").resolve()
        img_dir=Path("train-image").resolve()
        feature_reducer_paths=[
            "./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            "./models/feature_reduction/PCA(n_components=0.99)/model.onnx",
            "./models/feature_reduction/PCA(n_components=0.8)/model.onnx",
            None
            ]
        save_path=Path("./models/classifier").resolve()
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            local_mode=self.debug,
            storage="/opt/ray/results"
            )
        tuner = tune.Tuner(
            trainable_wrap(
                ClassifierTraining,
                num_cpus=cpu_per_trial,
                num_gpus=gpu_per_trial),
            run_config=air.RunConfig(
                name="TransformerClassifier",
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5),
                stop={"training_iteration": 20}),
            param_space=dict(
                dataset=DatasetReg.SkinLesions,
                dataset_kwargs=dict(
                    annotations_file=annotations_file,
                    img_file=img_file,
                    img_dir=img_dir,
                    img_transform=PPPicture(pad_mode='edge', pass_larger_images=True, crop=True),
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
                    label_desc='target',),
                feature_reducer_path=tune.grid_search(
                    [Path(pth).resolve() if pth else pth for pth in feature_reducer_paths]),
                classifier=ModelReg.Classifier,
                classifier_kwargs=dict(
                    activation=ActivationReg.relu,),
                optimizer=OptimizerReg.adam,
                optimizer_kwargs=dict(
                    lr=tune.grid_search([0.00005])
                ),
                criterion=CriterionReg.cross_entropy,
                criterion_kwargs=dict(
                    weight=torch.tensor([393, 400666])/401059),
                save_path=save_path,
                k_fold_splits=1,
                batch_size=128,
                shuffle=True,
                num_workers=cpu_per_trial-1,
            )
        )
        tuner.fit()
        ray.shutdown()
        print("Done")

    def trainClassifier(self):
        script = ClassifierTraining(
            dataset=DatasetReg.SkinLesions,
            dataset_kwargs=dict(
                annotations_file="train-metadata.csv",
                img_file="train-image.hdf5",
                img_dir="train-image",
                img_transform=PPPicture(pad_mode='edge', pass_larger_images=True, crop=True),
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
                label_desc='target',
                balance_augment=True,
            ),
            feature_reducer_path="./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            classifier=ModelReg.Classifier,
            classifier_kwargs=dict(
                activation=ActivationReg.relu,
            ),
            optimizer=OptimizerReg.adam,
            criterion=CriterionReg.cross_entropy,
            batch_size=2,
            save_path="./models/classifier",
            num_workers=os.cpu_count()-1 if not self.debug else 0,
            )
        script.setup()
        script.run()

    def setup(self):
        premades = {
            "submit":self.createSubmission,
            "train_feat_red":self.trainFeatureReducer,
            "train_class":self.trainClassifier,
            "train_class_ray":self.trainClassifierRay,
        }
        self.script = premades[self.script]

    def run(self):
        """
        """
        self.script()

if __name__ == "__main__":
    ScriptChooser().complete_run()
from ray.train._internal.storage import StorageContext
from ray.tune.logger import Logger
import sklearn.model_selection
import torch
import torch.utils
from torch import nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from sklearn.model_selection import StratifiedKFold
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType,DoubleTensorType

import math
import numpy as np
import pandas as pd
import sklearn

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
from dataclasses import dataclass
from typing_extensions import override
from functools import partial
import warnings

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
    LRSchedulerReg,
    CriterionReg,
    IterOptional
    )
from isic.models import ModelReg
from isic.datasets import (
    DatasetReg,
    DataHandlerGenerator,
    DataHandler,
    Subset,
    train_test_split,
    SimpleCustomBatch,
    collate_wrapper
)
from isic.callbacks import Callback, CallbackReg

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
            for i, (mod, data_samp) in enumerate(self.script.get_models_for_onnx_save()):
                self.script.save_model(mod, data_samp, i, save_dir=checkpoint_dir)
    return tune.with_resources(TrainableWrapper, resources={"CPU":num_cpus, "GPU":num_gpus})

class TrainingScript(Script):
    data: Type[Dataset]
    save_path: Union[str, Path]
    training_manager: 'TrainingManager'
    callback:Callback

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = (torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def __isclose(input:torch.Tensor, other:torch.Tensor, equal_nan=False):
        """
        Based on the proposal found here: https://github.com/pytorch/pytorch/issues/41651,
        the need to measure whether a torch model and onnx model behave effectively
        the same without necessarily casting to greater floating point precisions,
        and the description of floating point values here: https://stackoverflow.com/questions/872544/what-range-of-numbers-can-be-represented-in-a-16-32-and-64-bit-ieee-754-syste,
        this method provides a measure of closeness dynamically based on dtype.
        """
        assert input.dtype == other.dtype
        E = int(math.log2(other.abs().max()))
        epsilon_adj = {
            torch.float16:-10,
            torch.float32:-23,
            torch.float64:-52,
        }
        epsilon = 2**(E-epsilon_adj[other.dtype])

        return torch.isclose(input=input,
                             other=other,
                             rtol=epsilon*20,
                             atol=epsilon*5,
                             equal_nan=equal_nan)

    def save_model(self, model, data_input_sample, suffix="", save_dir=None, validate_model=True):
        validation_output = None
        if validate_model:
            if isinstance(model, nn.Module):
                validation_output = model(**data_input_sample)
            else:
                validation_output = model.transform(data_input_sample)
                validation_output = torch.from_numpy(validation_output).float()

        onx = None
        save_path = Path(save_dir) if save_dir else self.save_path
        save_file = save_path / "{}{}/model.onnx".format(self.get_model_name(model), suffix)
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)

        if isinstance(model, nn.Module):
            torch.onnx.export(
                model,
                tuple(data_input_sample.values()),
                input_names=list(data_input_sample.keys()),
                f = save_file,
                dynamic_axes={k: {0: "batch"} for k in data_input_sample.keys()}
            )
        else:
            init_types = [("fet", FloatTensorType([None, data_input_sample.shape[-1]]))]
            onx = convert_sklearn(model, initial_types=init_types)
            with open(save_file, "wb") as f:
                f.write(onx.SerializeToString())
        if validate_model:
            res_mod = None
            res_output = None
            if isinstance(model, nn.Module):
                res_mod = Registry.load_model(save_file)
                res_output, = res_mod(**data_input_sample)
            else:
                res_mod = Registry.load_model(save_file, cuda=False)
                res_output, = res_mod(data_input_sample)
            if not (validation_output.max(-1).indices==res_output.max(-1).indices).all():
                warnings.warn(
                    "Output decisions of training model and saved model do not match.",
                    UserWarning)
            if not self.__isclose(validation_output, res_output).all():
                warnings.warn(
                    "Output logits of training model and saved model do not match.",
                    UserWarning)

    def save_results(self, model, results):
        save_file = self.save_path / "{}/results.json".format(self.get_model_name(model))
        if not save_file.parent.exists():
            save_file.parent.mkdir(parents=True)
        with open(save_file, "w") as f:
            json.dump(results, f)
    
    def get_model_name(self, model):
        name = model.name() if hasattr(model, "name") else str(model)
        return name
    
    def get_models_for_onnx_save(self, dtype=None):
        return self.training_manager.get_models_for_onnx_save(dtype=dtype)

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

@dataclass
class TrainSplits:
    train_data:DataLoader
    val_data:DataLoader
    trainers:list['Trainer']

class TrainingManager:
    def __init__(
        self,
        data,
        dl_kwargs:dict,
        num_splits:int,
        balance_training_set:bool,
        shuffle:bool,
        trainer_class:type['Trainer'],
        pipelines:IterOptional[list[tuple[str, Callable]]] = None,
        models:IterOptional[Union[str, ModelReg]] = None,
        models_kwargs:IterOptional[Dict[str, Any]] = None,
        optimizers:IterOptional[Union[OptimizerReg, Type[Optimizer]]]= None,
        optimizers_kwargs:IterOptional[Dict[str, Any]] = None,
        lr_schedulers:IterOptional[Union[LRSchedulerReg, Type[LRScheduler]]]= None,
        lr_schedulers_kwargs:IterOptional[Dict[str, Any]] = None,
        criterion:Union[CriterionReg, Type[torch.nn.modules.loss._Loss]]= None,
        criterion_kwargs:Dict[str, Any] = None,
        ):
        self.num_splits = num_splits
        self.balance_training_set = balance_training_set
        self.shuffle = shuffle
        self.data = data
        self.dl_kwargs = dl_kwargs
        self.trainer_class = trainer_class
        self.device = (torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        print("Expected device:{}".format(self.device))
        
        self.pipelines = pipelines
        self.models:IterOptional[nn.Module] = models
        if not isinstance(self.models, list):
            self.models = [self.models]
        self.models_kwargs = models_kwargs if models_kwargs else [{}]*len(self.models)
        if not isinstance(self.models_kwargs, list):
            self.models_kwargs = [self.models_kwargs]*len(self.models)
        self.optimizers = optimizers
        if not isinstance(self.optimizers, list):
            self.optimizers = [self.optimizers]*len(self.models)
        self.optimizers_kwargs = optimizers_kwargs if optimizers_kwargs else [{}]*len(self.models)
        if not isinstance(self.optimizers_kwargs, list):
            self.optimizers_kwargs = [self.optimizers_kwargs]*len(self.models)
        self.lr_schedulers = lr_schedulers
        if not isinstance(self.lr_schedulers, list):
            self.lr_schedulers = [self.lr_schedulers]*len(self.models)
        self.lr_schedulers_kwargs = lr_schedulers_kwargs if lr_schedulers_kwargs else [{}]*len(self.models)
        if not isinstance(self.lr_schedulers_kwargs, list):
            self.lr_schedulers_kwargs = [self.lr_schedulers_kwargs]*len(self.models)
        self.criterion = CriterionReg.initialize(
            criterion, criterion_kwargs)
        self.criterion.to(device=self.device)
        self.splits:dict[int, TrainSplits] = {}
        self.create_splits()
        assert len(self) == 1
        for split in self:
            for trainer_x in split.trainers:
                model_match_count = 0
                optimizer_match_count = 0
                for trainer_y in split.trainers:
                    if trainer_x.model == trainer_y.model:
                        model_match_count += 1
                    if trainer_x.optimizer == trainer_y.optimizer:
                        optimizer_match_count += 1
                assert model_match_count == 1
                assert optimizer_match_count == 1

    
    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError
        return self.splits[index]

    def create_splits(self):
        for i, (training_data, validation_data) in enumerate(self.create_dataloaders()):
            self.splits[i] = TrainSplits(
                train_data=training_data,
                val_data=validation_data,
                trainers=list(self.create_trainers()))

    def create_dataloaders(self):
        split_idxs = None
        if self.num_splits == 1:
            split_idxs = train_test_split(
                self.data,
                train_size=0.8,
                test_size=0.2,
                shuffle=self.shuffle,
                stratify=[self.data.labels]
                )
        else:
            split_idxs = StratifiedKFold(
                n_splits=self.num_splits,
                shuffle=self.shuffle,
                ).split(self.data, self.data.labels)
            
        for train_fold, val_fold in split_idxs:
            print("Start dataloaders.")
            train_dl_kwargs = self.dl_kwargs.copy()
            train_data = Subset(self.data, train_fold)
            val_data = Subset(self.data, val_fold)
            if self.balance_training_set:
                unq_lbls = train_data.getLabels().unique()
                lbl_masks = train_data.getLabels()==unq_lbls[:,None]
                not_lbl_counts = (train_data.getLabels()!=unq_lbls[:,None]).sum(-1)[:,None]
                weights = (lbl_masks*not_lbl_counts).sum(0)
                train_dl_kwargs["sampler"] = (
                    torch.utils.data.WeightedRandomSampler(weights, len(train_data)))
                train_dl_kwargs["shuffle"] = False
            yield (DataLoader(train_data,**train_dl_kwargs),
                   DataLoader(val_data,**self.dl_kwargs))
        
    def create_trainers(self):
        for j, m in enumerate(self.models):
            model = ModelReg.initialize(m, self.models_kwargs[j]).to(device=self.device)
            opt = OptimizerReg.initialize(
                    self.optimizers[j], model.parameters(), self.optimizers_kwargs[j])
            lr_sch = self.lr_schedulers[j]
            if lr_sch:
                lr_sch = LRSchedulerReg.initialize(
                    lr_sch, opt, self.lr_schedulers_kwargs[j])
            yield self.trainer_class(
                model=model,
                optimizer=opt,
                lr_scheduler=lr_sch,
                criterion=self.criterion,
                pipeline=self.pipelines[j])

    def get_models_for_onnx_save(self, dtype=None)-> tuple:
        for i, split in self.splits.items():
            data_sample = next(iter(split.train_data))
            for trainer in split.trainers:
                d_h = trainer.generate_data_handler(data_sample)
                d_h.process_data(trainer.model, dtype=dtype, trunc_batch=8)
                yield trainer.model.eval().to(dtype=dtype), d_h.get_inputs()
    
    def step_lr_schedulers(self):
        for train_elements in self:
            for trainer in train_elements.trainers:
                trainer.step_lr_scheduler(endOfEpoch=True)

class Trainer:
    def __init__(
            self,
            model:nn.Module,
            optimizer:Optimizer,
            criterion:torch.nn.modules.loss._Loss,
            lr_scheduler:LRScheduler=None,
            pipeline=None):
        self.model = model
        param_ref = next(iter(model.parameters()))
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.device = param_ref.device
        self.dh_gen:DataHandlerGenerator = self.map_data_handler(pipeline)
    
    def map_data_handler(self, pipeline):
        return DataHandlerGenerator(pipeline=pipeline if pipeline else [])
    
    def generate_data_handler(self, data):
        return self.dh_gen.get_data_handler(data)
    
    def step_lr_scheduler(self, endOfEpoch=False):
        if self.lr_scheduler:
            if ((endOfEpoch and
                not self.lr_scheduler.__class__.__name__ == LRSchedulerReg.CyclicLR.name) or
                (not endOfEpoch and
                self.lr_scheduler.__class__.__name__ == LRSchedulerReg.CyclicLR.name)
                ):
                self.lr_scheduler.step()

    def infer_itr(
            self,
            data
            ):
        data_handler = self.generate_data_handler(data)
        data_handler.process_data(self.model)

        if self.model.training:
            self.optimizer.zero_grad()
        
        with torch.autocast(device_type=self.device.type):
            data_handler.set_model_output(self.model(**data_handler.get_inputs()))
            
            data_handler.set_loss(self.criterion(data_handler.output, data_handler.target))

        if self.model.training:
            data_handler.loss.backward()
            self.optimizer.step()
            self.step_lr_scheduler()
        data_handler.set_last_lr(self.lr_scheduler.get_last_lr()[0])
        return data_handler

class ISICTrainer(Trainer):    
    @override
    def map_data_handler(self, pipeline):
        return DataHandlerGenerator(
            img=SimpleCustomBatch.IMG,
            fet=SimpleCustomBatch.FET,
            target=SimpleCustomBatch.TGT,
            pipeline=pipeline if pipeline else [])

class MultiClassifierTraining(TrainingScript):
    def __init__(
            self,
            dataset:Union[str, DatasetReg, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            pipelines:IterOptional[list[tuple[str, Callable]]] = None,
            classifiers:IterOptional[str | ModelReg] = None,
            classifiers_kwargs:IterOptional[Dict[str, Any]] = None,
            optimizers:IterOptional[OptimizerReg | Type[Optimizer]]= None,
            optimizers_kwargs:IterOptional[Dict[str, Any]] = None,
            lr_schedulers:IterOptional[Union[LRSchedulerReg, Type[LRScheduler]]]= None,
            lr_schedulers_kwargs:IterOptional[Dict[str, Any]] = None,
            criterion:Union[CriterionReg, Type[torch.nn.modules.loss._Loss]] = None,
            criterion_kwargs:Dict[str, Any] = None,
            save_path:Union[str, Path] = None,
            balance_training_set = False,
            k_fold_splits:int = 4,
            batch_size:int = 64,
            shuffle:bool = True,
            num_workers:int = 0,
            callback:type[Callback] = None,
            callback_kwargs:Dict[str, Any] = None,
            **kwargs) -> None:
        """
        Args:
            dataset: The dataset class to be used.
            dataset_kwargs: Configuration for the dataset.
            pipelines: Models to pass data through prior to training model.
            classifiers: The classifier class to be used.
            classifiers_kwargs: Configuration for the classifier.
            optimizers: The optimizer class to be used.
            optimizers_kwargs : Configuration for the optimizer.
            lr_schedulers: The learning rate scheduler class to be used, if any.
            lr_schedulers_kwargs : Configuration for the learning rate scheduler.
            criterion: The criterion class to be used.
            criterion_kwargs : Configuration for the criterion.
            save_path: Specify a path to which classifier should be saved.
            balance_training_set: Create a sampling scheme so that each class
              is trained on equally often.
            k_fold_splits: Number of folds used for dataset in training.
            batch_size: Number of data points included in training batches.
            shuffle: Whether to shuffle data points.
            num_workers: For multiprocessing.
            callback: Class of processes interjected into training run.
            callback_kwargs: Configuration for callback.
        """
        super().__init__(**kwargs)
        self.data = dataset
        self.ds_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.balance_training_set = balance_training_set
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

        self.pipelines = pipelines
        self.classifiers:IterOptional[nn.Module] = classifiers
        if not isinstance(self.classifiers, list):
            self.classifiers = [self.classifiers]
        self.classifiers_kwargs = classifiers_kwargs if classifiers_kwargs else [{}]*len(self.classifiers)
        if not isinstance(self.classifiers_kwargs, list):
            self.classifiers_kwargs = [self.classifiers_kwargs]*len(self.classifiers)
        self.optimizers = optimizers
        if not isinstance(self.optimizers, list):
            self.optimizers = [self.optimizers]*len(self.classifiers)
        self.optimizers_kwargs = optimizers_kwargs if optimizers_kwargs else [{}]*len(self.classifiers)
        if not isinstance(self.optimizers_kwargs, list):
            self.optimizers_kwargs = [self.optimizers_kwargs]*len(self.classifiers)
        self.lr_schedulers = lr_schedulers
        if not isinstance(self.optimizers, list):
            self.lr_schedulers = [self.lr_schedulers]*len(self.classifiers)
        self.lr_schedulers_kwargs = lr_schedulers_kwargs if lr_schedulers_kwargs else [{}]*len(self.classifiers)
        if not isinstance(self.lr_schedulers_kwargs, list):
            self.lr_schedulers_kwargs = [self.lr_schedulers_kwargs]*len(self.classifiers)
        self.criterion = criterion
        self.criterion_kwargs = criterion_kwargs if criterion_kwargs else {}
        self.save_path = save_path
        self.callback = callback if callback else CallbackReg.ClassifierTrainingCallback
        self.callback_kwargs = callback_kwargs if callback_kwargs else {}

    def setup(self):
        self.callback = CallbackReg.initialize(
            self.callback, self.callback_kwargs)
        self.data = DatasetReg.initialize(
            self.data, self.ds_kwargs)
        print("Dataset initialized.")
        self.training_manager = TrainingManager(
            data=self.data,
            dl_kwargs=self.dl_kwargs,
            num_splits=self.k_fold_splits,
            balance_training_set=self.balance_training_set,
            shuffle=self.dl_kwargs["shuffle"],
            trainer_class=ISICTrainer,
            pipelines=self.pipelines,
            models=self.classifiers,
            models_kwargs=self.classifiers_kwargs,
            optimizers=self.optimizers,
            optimizers_kwargs=self.optimizers_kwargs,
            lr_schedulers=self.lr_schedulers,
            lr_schedulers_kwargs=self.lr_schedulers_kwargs,
            criterion=self.criterion,
            criterion_kwargs=self.criterion_kwargs,
        )
        print("Training manager initialized.")
        self.save_path = Path(self.save_path)
        print("Completed setup.")

    def run(self):
        self.callback.on_run_begin(self)
        for train_elements in self.training_manager:
            self.callback.on_fold_begin(self)
            for trainer in train_elements.trainers:
                trainer.model.train()
            for data in train_elements.train_data:
                self.callback.on_train_batch_begin(self)
                for trainer in train_elements.trainers:
                    self.callback.on_model_select(self)
                    self.callback.on_inference_end(
                        self,
                        data_handler=trainer.infer_itr(data))
                
            for trainer in train_elements.trainers:
                trainer.model.eval()
            for data in train_elements.val_data:
                self.callback.on_val_batch_begin(self)
                for trainer in train_elements.trainers:
                    self.callback.on_model_select(self)
                    self.callback.on_inference_end(
                        self,
                        data_handler=trainer.infer_itr(data))
        self.training_manager.step_lr_schedulers()
        return self.callback.get_epoch_metrics()

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
            balance_training_set = False,
            k_fold_splits:int = 4,
            batch_size:int = 64,
            shuffle:bool = True,
            num_workers:int = 0,
            **kwargs) -> None:
        """
        Args:
            dataset: The dataset class to be used.
            dataset_kwargs: Configuration for the dataset.
            classifier: The classifier class to be used.
            classifier_kwargs: Configuration for the classifier.
            optimizer: The optimizer class to be used.
            optimizer_kwargs : Configuration for the optimizer.
            criterion: The criterion class to be used.
            criterion_kwargs : Configuration for the criterion.
            save_path: Specify a path to which classifier should be saved.
            balance_training_set: Create a sampling scheme so that each class
              is trained on equally often.
            k_fold_splits: Number of folds used for dataset in training.
            batch_size: Number of data points included in training batches.
            shuffle: Whether to shuffle data points.
            num_workers: For multiprocessing.
        """
        super().__init__(**kwargs)
        self.data = dataset
        self.ds_kwargs = dataset_kwargs if dataset_kwargs else {}
        self.balance_training_set = balance_training_set
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
        self.training_manager = TrainingManager(
            data=self.data,
            dl_kwargs=self.dl_kwargs,
            num_splits=self.k_fold_splits,
            balance_training_set=self.balance_training_set,
            shuffle=self.dl_kwargs["shuffle"],
            models=self.classifier,
            models_kwargs=self.classifier_kwargs,
            optimizers=self.optimizer,
            optimizers_kwargs=self.optimizer_kwargs,
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

            result, = self.classifier(img=img, fet=fet)
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
            classifier_path="./models/classifier/test/model.onnx",
            save_path=".",
            num_workers=os.cpu_count()-1 if not self.debug else 0,
            )
        script.setup()
        script.run()

    def trainFeatureReducer(self):
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        script = FeatureReductionForTraining(
            dataset=DatasetReg.SkinLesions,
            dataset_kwargs=dict(
                annotations_file=annotations_file,
                img_file=img_file,
                img_dir=img_dir,
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
        num_trials = 2
        num_cpus = 1 if self.debug else os.cpu_count()
        num_gpus = 0 if self.debug else torch.cuda.device_count()
        cpu_per_trial = num_cpus//num_trials
        gpu_per_trial = num_gpus/num_trials
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        feature_reducer_paths=[
            "./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
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
                    img_transform=PPPicture(
                        pad_mode='edge',
                        pass_larger_images=True,
                        crop=True,
                        random_brightness=True,
                        random_contrast=True,
                        random_flips=True),
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
                classifier=ModelReg.Classifier,
                classifier_kwargs=dict(
                    activation=ActivationReg.relu,
                    feature_reducer_path=tune.grid_search(
                        [Path(pth).resolve() if pth else pth for pth in feature_reducer_paths]),
                    ),
                optimizer=OptimizerReg.adam,
                optimizer_kwargs=dict(
                    lr=tune.grid_search([0.00005])
                ),
                criterion=CriterionReg.cross_entropy,
                criterion_kwargs=dict(
                    weight=torch.tensor([393/401059, 400666/401059])
                ),
                save_path=save_path,
                balance_training_set=False,
                k_fold_splits=1,
                batch_size=256,
                shuffle=True,
                num_workers=cpu_per_trial-1,
            )
        )
        tuner.fit()
        ray.shutdown()
        print("Done")


    def LRRangeTestRay(self):
        num_trials = 1
        num_cpus = 1 if self.debug else os.cpu_count()
        num_gpus = 0 if self.debug else torch.cuda.device_count()
        cpu_per_trial = num_cpus//num_trials
        gpu_per_trial = num_gpus/num_trials
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        feature_reducer_paths=[
            "./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            None,
            ]
        save_path=Path("./models/classifier").resolve()

        ds = DatasetReg.SkinLesions
        ds_kwargs = dict(
            annotations_file=annotations_file,
            img_file=img_file,
                    img_dir=img_dir,
            img_transform=PPPicture(
                pad_mode='edge',
                pass_larger_images=True,
                        crop=True,
                random_brightness=True,
                random_contrast=True,
                random_flips=True),
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
            label_desc='target',)
        BATCHSIZE = 512
        EPOCHS = 10
        ds_len = len(DatasetReg.initialize(ds, ds_kwargs))
        
        ray.init(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            local_mode=self.debug,
            storage="/opt/ray/results"
            )
        tuner = tune.Tuner(
            trainable_wrap(
                MultiClassifierTraining,
                num_cpus=cpu_per_trial,
                num_gpus=gpu_per_trial),
            run_config=air.RunConfig(
                name="TransformerClassifier",
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5),
                stop={"training_iteration": EPOCHS}),
            param_space=dict(
                dataset=ds,
                dataset_kwargs=ds_kwargs,
                pipelines=[[('fet', Path(pth).resolve())] if pth else pth
                    for pth in feature_reducer_paths],
                classifiers=[ModelReg.Classifier,ModelReg.Classifier],
                classifiers_kwargs=dict(
                    activation=ActivationReg.relu),
                optimizers=OptimizerReg.adam,
                optimizers_kwargs=dict(
                    lr=tune.grid_search([0.00005])
                ),
                lr_schedulers=LRSchedulerReg.CyclicLR,
                lr_schedulers_kwargs=dict(
                    base_lr=0.000001,
                    max_lr=0.0001,
                    step_size_up=(ds_len//BATCHSIZE)*EPOCHS
                ),
                criterion=CriterionReg.cross_entropy,
                criterion_kwargs=dict(
                    weight=torch.tensor([393/401059, 400666/401059])
                ),
                save_path=save_path,
                balance_training_set=False,
                k_fold_splits=1,
                batch_size=BATCHSIZE,
                shuffle=True,
                num_workers=cpu_per_trial-1,
                callback=CallbackReg.LRRangeTestCallback
            )
        )
        tuner.fit()
        ray.shutdown()
        print("Done")

    def LRRangeTest(self):
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        feature_reducer_paths=[
            "./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            None,
            ]
        save_path=Path("./models/classifier").resolve()

        ds = DatasetReg.SkinLesionsSmall
        ds_kwargs = dict(
            annotations_file=annotations_file,
            img_file=img_file,
                    img_dir=img_dir,
            img_transform=PPPicture(
                pad_mode='edge',
                pass_larger_images=True,
                        crop=True,
                random_brightness=True,
                random_contrast=True,
                random_flips=True),
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
            label_desc='target',)
        BATCHSIZE = 1024
        EPOCHS = 10
        ds_len = len(DatasetReg.initialize(ds, ds_kwargs))
        script = MultiClassifierTraining(
                dataset=ds,
                dataset_kwargs=ds_kwargs,
                pipelines=[[('fet', Path(pth).resolve())] if pth else pth
                    for pth in feature_reducer_paths],
                classifiers=[ModelReg.Classifier,ModelReg.Classifier],
                classifiers_kwargs=dict(
                    activation=ActivationReg.relu),
                optimizers=OptimizerReg.adam,
                optimizers_kwargs=dict(
                    lr=0.00005
                ),
                lr_schedulers=LRSchedulerReg.CyclicLR,
                lr_schedulers_kwargs=dict(
                    base_lr=0.000001,
                    max_lr=0.0005,
                    step_size_up=(ds_len//BATCHSIZE)*EPOCHS
                ),
                criterion=CriterionReg.cross_entropy,
                criterion_kwargs=dict(
                    weight=torch.tensor([393/401059, 400666/401059])
                ),
                save_path=save_path,
                balance_training_set=False,
                k_fold_splits=1,
                batch_size=BATCHSIZE,
                shuffle=True,
                num_workers=os.cpu_count()-1 if not self.debug else 0,
                callback=CallbackReg.LRRangeTestCallback
            )
        script.setup()
        results = script.run()

        for i, (mod, data_samp) in enumerate(script.get_models_for_onnx_save()):
            script.save_model(mod, data_samp, i)
            script.save_results(mod, results)

    def trainClassifiersRay(self):
        num_trials = 2
        num_cpus = 1 if self.debug else os.cpu_count()
        num_gpus = 0 if self.debug else torch.cuda.device_count()
        BATCHSIZE=256
        EPOCHS=100
        lr_sched_params = {
            False:dict(
                    base_lr=0.000001,
                    max_lr=0.00003,
                    step_size_up=(401059/BATCHSIZE)*4
                ),
            True:dict(
                    base_lr=0.000025,
                    max_lr=0.00005,
                    step_size_up=(401059/BATCHSIZE)*4
                ),
                }
        cpu_per_trial = num_cpus//num_trials
        gpu_per_trial = num_gpus/num_trials
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        feature_reducer_paths=[
            "./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            None,
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
                MultiClassifierTraining,
                num_cpus=cpu_per_trial,
                num_gpus=gpu_per_trial),
            run_config=air.RunConfig(
                name="TransformerClassifier",
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5),
                stop={"training_iteration": EPOCHS}),
            param_space=dict(
                dataset=DatasetReg.SkinLesions,
                dataset_kwargs=dict(
                    annotations_file=annotations_file,
                    img_file=img_file,
                    img_dir=img_dir,
                    img_transform=PPPicture(
                        pad_mode='edge',
                        pass_larger_images=True,
                        crop=True,
                        random_brightness=True,
                        random_contrast=True,
                        random_flips=True),
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
                pipelines=tune.grid_search([[[('fet', Path(pth).resolve())] if pth else pth]
                    for pth in feature_reducer_paths]),
                classifiers=ModelReg.Classifier,
                classifiers_kwargs=dict(
                    activation=ActivationReg.relu),
                optimizers=OptimizerReg.adam,
                optimizers_kwargs=dict(
                    lr=tune.grid_search([0.00005])
                ),
                lr_schedulers=LRSchedulerReg.CyclicLR,
                lr_schedulers_kwargs=tune.sample_from(lambda spec: lr_sched_params[spec.config.pipelines is None]),
                criterion=CriterionReg.cross_entropy,
                criterion_kwargs=dict(
                    weight=torch.tensor([393/401059, 400666/401059])
                ),
                save_path=save_path,
                balance_training_set=False,
                k_fold_splits=1,
                batch_size=BATCHSIZE,
                shuffle=True,
                num_workers=cpu_per_trial-1,
            )
        )
        tuner.fit()
        ray.shutdown()
        print("Done")

    def trainClassifiers(self):
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        feature_reducer_paths=[
            "./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            None
            ]
        script = MultiClassifierTraining(
            dataset=DatasetReg.SkinLesionsSmall,
                dataset_kwargs=dict(
                    annotations_file=annotations_file,
                    img_file=img_file,
                    img_dir=img_dir,
                    img_transform=PPPicture(
                        pad_mode='edge',
                        pass_larger_images=True,
                        crop=True,
                        random_brightness=True,
                        random_contrast=True,
                        random_flips=True),
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
                pipelines=[[('fet', Path(pth).resolve())] if pth else pth
                    for pth in feature_reducer_paths],
                classifiers=[ModelReg.Classifier,ModelReg.Classifier],
                classifiers_kwargs=dict(
                    activation=ActivationReg.relu),
                optimizers=OptimizerReg.adam,
                optimizers_kwargs=dict(
                    lr=0.00005
                ),
                criterion=CriterionReg.cross_entropy,
                criterion_kwargs=dict(
                    weight=torch.tensor([393/401059, 400666/401059])
                ),
                save_path="./models/classifier",
                balance_training_set=False,
                k_fold_splits=1,
                batch_size=256,
                shuffle=True,
                num_workers=os.cpu_count()-1 if not self.debug else 0,
            )
        script.setup()
        results = script.run()

        for i, (mod, data_samp) in enumerate(script.get_models_for_onnx_save()):
            script.save_model(mod, data_samp, i)
            script.save_results(mod, results)

    def trainClassifier(self):
        annotations_file=Path("/home/user/datasets/isic-2024-challenge/train-metadata.csv").resolve()
        img_file=Path("/home/user/datasets/isic-2024-challenge/train-image.hdf5").resolve()
        img_dir=Path("/home/user/datasets/isic-2024-challenge/train-image").resolve()
        script = ClassifierTraining(
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
                label_desc='target',
            ),
            classifier=ModelReg.Classifier,
            classifier_kwargs=dict(
                activation=ActivationReg.relu,
                feature_reducer_path="./models/feature_reduction/PCA(n_components=0.9999)/model.onnx",
            ),
            optimizer=OptimizerReg.adam,
            criterion=CriterionReg.cross_entropy,
            batch_size=2,
            save_path="./models/classifier",
            balance_training_set=True,
            num_workers=os.cpu_count()-1 if not self.debug else 0,
            )
        script.setup()
        script.run()

    def setup(self):
        premades = {
            "submit":self.createSubmission,
            "train_feat_red":self.trainFeatureReducer,
            "train_class":self.trainClassifier,
            "train_classes":self.trainClassifiers,
            "train_class_ray":self.trainClassifierRay,
            "train_classes_ray":self.trainClassifiersRay,
            "lr_range_test":self.LRRangeTest,
            "lr_range_test_ray":self.LRRangeTestRay,
        }
        self.script = premades[self.script]

    def run(self):
        """
        """
        self.script()

if __name__ == "__main__":
    ScriptChooser().complete_run()
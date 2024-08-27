from typing import List

import numpy as np
import torch

from isic.datasets import DataHandler

class Metric:
    def __init__(self, model_num, fold_num) -> None:
        self.model_num = model_num
        self.fold_num = fold_num
        self.numerator = 0.
        self.denominator = 0.
    def include(self, num, den, data_handler:DataHandler=None):
        self.numerator += num
        self.denominator += den
    def get_result(self):
        return np.float64(self.numerator)/self.denominator
    def __str__(self):
        return self.__class__.__name__

class Rate(Metric):
    def include(self, num, den=1):
        super().include(num, den)

class Loss(Rate):
    def include(self, data_handler:DataHandler):
        super().include(data_handler.loss.item())

class Ratio(Metric):
    pass

class Accuracy(Ratio):
    def include(self, data_handler: DataHandler):
        num_matches = (
            data_handler.output_label.int() == data_handler.target.int()).sum()
        num_total = data_handler.target.numel()
        super().include(
            num_matches.item(),
            num_total)

class Precision(Ratio):
    def include(self, data_handler: DataHandler):
        num_true_positives=(
            data_handler.target.int()[data_handler.output_label.int() == 1].sum())
        num_positive_inferences = (data_handler.output_label.int() == 1).sum()
        super().include(
            num_true_positives.item(),
            num_positive_inferences.item()
            )

class Recall(Ratio):
    def include(self, data_handler: DataHandler):
        num_true_positives=(
            data_handler.target.int()[data_handler.output_label.int() == 1].sum())
        num_positive_targets = (data_handler.target.int() == 1).sum()
        super().include(
            num_true_positives.item(),
            num_positive_targets.item()
            )

class Callback:
    def __init__(self) -> None:
        pass

    def on_run_begin(self, training_manager):
        pass

    def on_fold_begin(self):
        pass

    def on_inference_end(self, data_handler:DataHandler):
        pass
    
    def on_train_batch_begin(self):
        pass

    def on_val_batch_begin(self):
        pass

    def on_model_select(self):
        pass

    def get_epoch_metrics(self):
        pass

class ClassifierTrainingCallback(Callback):
    TRAINING_METRICS = [Loss]
    VALIDATION_METRICS = [Loss, Accuracy, Precision, Recall]
    INFERENCE_MODES = ["training","validation"]
    def process_output(self, output:torch.Tensor):
        return output.max(-1).indices
    
    def get_epoch_metrics(self):
        ret = {}
        def get_res(x:Metric):
            return x.get_result()
        get_res = np.vectorize(get_res)
        train_mets = get_res(self.training_metrics)
        for f in self.folds:
            for mod in self.models:
                for i, met in enumerate(self.training_metrics[f][mod]):
                    res = train_mets[f][mod][i]
                    ret["fold{}_model{}_train_{}".format(f,mod,met)] = res
        val_mets = get_res(self.validation_metrics)
        for f in self.folds:
            for mod in self.models:
                for i, met in enumerate(self.validation_metrics[f][mod]):
                    res = val_mets[f][mod][i]
                    ret["fold{}_model{}_val_{}".format(f,mod,met)] = res
        
        for f in self.folds:
            for i, met in enumerate(self.TRAINING_METRICS):
                ret["mean_fold{}_train_{}".format(f, met.__name__)] = np.mean(train_mets[f,:,i])
            for i, met in enumerate(self.VALIDATION_METRICS):
                ret["mean_fold{}_val_{}".format(f, met.__name__)] = np.mean(val_mets[f,:,i])

        for mod in self.models:
            for i, met in enumerate(self.TRAINING_METRICS):
                ret["mean_model{}_train_{}".format(f, met.__name__)] = np.mean(train_mets[:,mod,i])
            for i, met in enumerate(self.VALIDATION_METRICS):
                ret["mean_model{}_val_{}".format(f, met.__name__)] = np.mean(val_mets[:,mod,i])
        
        for i, met in enumerate(self.TRAINING_METRICS):
            ret["mean_train_{}".format(met.__name__)] = np.mean(train_mets[:,:,i])
        for i, met in enumerate(self.VALIDATION_METRICS):
            ret["mean_val_{}".format(met.__name__)] = np.mean(val_mets[:,:,i])
        return ret
    
    def on_run_begin(self, training_manager):
        self.models = range(len(training_manager[0].trainers))
        self.folds = range(len(training_manager))
        self.training_metrics:List[List[List[Metric]]] = [[[
            met(i, j) for met in ClassifierTrainingCallback.TRAINING_METRICS]
            for i in self.models]
            for j in self.folds]
        self.validation_metrics:List[List[List[Metric]]] = [[[
            met(i, j) for met in ClassifierTrainingCallback.VALIDATION_METRICS]
            for i in self.models]
            for j in self.folds]
        self.model = -1
        self.fold = -1
        self.inference_mode = ClassifierTrainingCallback.INFERENCE_MODES[0]

    def on_fold_begin(self):
        self.fold += 1
    
    def on_model_select(self):
        self.model += 1
    
    def on_train_batch_begin(self):
        self.model = -1
        self.inference_mode = ClassifierTrainingCallback.INFERENCE_MODES[0]

    def on_val_batch_begin(self):
        self.model = -1
        self.inference_mode = ClassifierTrainingCallback.INFERENCE_MODES[1]

    def on_inference_end(self, data_handler:DataHandler):
        metric_set = (self.training_metrics
                      if self.inference_mode == ClassifierTrainingCallback.INFERENCE_MODES[0]
                      else self.validation_metrics)
        for met in metric_set[self.fold][self.model]:
            met.include(data_handler=data_handler)
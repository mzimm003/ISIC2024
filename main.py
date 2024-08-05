from torch.utils.data import Dataset, DataLoader

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


from pathlib import Path
from typing import (
    List,
    Union,
    Dict,
    Any,
    Type
)

from quickscript.scripts import Script, ScriptChooser

from isic.preprocess import(
    PPPicture,
    PPLabels
)

from isic.registry import (
    Registry,
    FeatureReducersReg,
    ActivationReg,)
from isic.models import ModelReg
from isic.datasets import DatasetReg

class TrainingScript(Script):
    data: Type[Dataset]
    save_path: Union[str, Path]
    def save_model(self, model):
        inp, target = next(iter(self.data))
        onx = convert_sklearn(model, initial_types=[("X", FloatTensorType([None, inp.shape[-1]]))])
        with open(self.save_path / "{}.onnx".format(str(model)), "wb") as f:
            f.write(onx.SerializeToString())

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

class ClassifierTraining(TrainingScript):
    def __init__(
            self,
            dataset:Union[str, DatasetReg, Type[Dataset]] = None,
            dataset_kwargs:Dict[str, Any] = None,
            classifier:Union[str, ModelReg] = None,
            classifier_kwargs:Dict[str, Any] = None,
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
        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs if classifier_kwargs else {}
        self.save_path = save_path
    
    def setup(self):
        self.data = DatasetReg.initialize(
            self.data, self.ds_kwargs)
        self.classifier = ModelReg.initialize(
            self.classifier, self.classifier_kwargs)
        self.save_path = Path(self.save_path)

    def run(self):
        inp, tgt = self.data[:]
        self.classifier.fit(inp, tgt)
        self.save_model(self.classifier)

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
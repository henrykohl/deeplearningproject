from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader


@dataclass # Lec 2
class DataIngestionArtifact:
    train_file_path: str

    test_file_path: str

@dataclass # Lec 3
class DataTransformationArtifact:
    transformed_train_object: DataLoader

    transformed_test_object: DataLoader

    train_transform_file_path: str

    test_transform_file_path: str

@dataclass # Lec 4
class ModelTrainerArtifact:
    trained_model_path: str
    
@dataclass # Lec 5
class ModelEvaluationArtifact:
    model_accuracy: float

@dataclass
class ModelPusherArtifact:
    bentoml_model_name: str

    bentoml_service_name: str
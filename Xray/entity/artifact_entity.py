from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader


@dataclass # Lec 2
class DataIngestionArtifact:
    train_file_path: str # 訓練資料存放路徑

    test_file_path: str # 測試資料存放路徑

@dataclass # Lec 3
class DataTransformationArtifact:
    transformed_train_object: DataLoader # 訓練資料轉換後的 DataLoader

    transformed_test_object: DataLoader # 測試資料轉換後的 DataLoader

    train_transform_file_path: str # 轉型模式 train_transform 的檔案(路徑)

    test_transform_file_path: str # 轉型模式 test_transform 的檔案(路徑)

@dataclass # Lec 4
class ModelTrainerArtifact:
    trained_model_path: str # 訓練後模型檔案(路徑)
    
@dataclass # Lec 5
class ModelEvaluationArtifact:
    model_accuracy: float  # 測試資料後的準確性結果

@dataclass
class ModelPusherArtifact:
    bentoml_model_name: str

    bentoml_service_name: str
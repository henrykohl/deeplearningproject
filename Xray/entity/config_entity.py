import os
from dataclasses import dataclass

from torch import device

from Xray.constant.training_pipeline import *

@dataclass # Lec 2
class DataIngestionConfig: 
    def __init__(self):
        self.s3_data_folder: str = S3_DATA_FOLDER

        self.bucket_name: str = BUCKET_NAME

        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)

        self.data_path: str = os.path.join(
            self.artifact_dir, "data_ingestion", self.s3_data_folder
        )

        self.train_data_path: str = os.path.join(self.data_path, "train")

        self.test_data_path: str = os.path.join(self.data_path, "test")

@dataclass # Lec 3
class DataTransformationConfig: 
    def __init__(self):
        self.color_jitter_transforms: dict = {
            "brightness": BRIGHTNESS,
            "contrast": CONTRAST,
            "saturation": SATURATION,
            "hue": HUE,
        }

        self.RESIZE: int = RESIZE

        self.CENTERCROP: int = CENTERCROP

        self.RANDOMROTATION: int = RANDOMROTATION

        self.normalize_transforms: dict = {
            "mean": NORMALIZE_LIST_1,
            "std": NORMALIZE_LIST_2,
        }

        self.data_loader_params: dict = {
            "batch_size": BATCH_SIZE,
            "shuffle": SHUFFLE,
            "pin_memory": PIN_MEMORY,
        }

        self.artifact_dir: str = os.path.join(
            ARTIFACT_DIR, TIMESTAMP, "data_transformation"
        )

        self.train_transforms_file: str = os.path.join(
            self.artifact_dir, TRAIN_TRANSFORMS_FILE
        ) # "artifacts" + TIMESTAMP + "data_transformation" + "train_transforms.pkl"

        self.test_transforms_file: str = os.path.join(
            self.artifact_dir, TEST_TRANSFORMS_FILE
        ) # "artifacts" + TIMESTAMP + "data_transformation" + "test_transforms.pkl"

@dataclass # Lec 4
class ModelTrainerConfig: 
    def __init__(self):
        self.artifact_dir: int = os.path.join(ARTIFACT_DIR, TIMESTAMP, "model_training") ## (應該是 str)

        self.trained_bentoml_model_name: str = "xray_model"

        self.trained_model_path: int = os.path.join( 
            self.artifact_dir, TRAINED_MODEL_NAME
        ) ## (應該是 str) 訓練模型檔案(路徑)

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY # "xray_train_transforms"

        self.epochs: int = EPOCH

        self.optimizer_params: dict = {"lr": 0.01, "momentum": 0.8}

        self.scheduler_params: dict = {"step_size": STEP_SIZE, "gamma": GAMMA}

        self.device: device = DEVICE

@dataclass # Lec 5
class ModelEvaluationConfig: 
    def __init__(self):
        self.device: device = DEVICE  # configuration() 中設定所需

        self.test_loss: int = 0       # test_net() 中用來儲存 loss 的總值

        self.test_accuracy: int = 0   # test_net() 中用來儲存 predictions 的正確數量

        self.total: int = 0           # test_net() 中用來儲存  labels 的 長度

        self.total_batch: int = 0     # test_net() 中用來儲存  batch 的 數量

        self.optimizer_params: dict = {"lr": 0.01, "momentum": 0.8} ## Evaluation 似乎沒用到

# Model Pusher Configurations
@dataclass # Lec 6
class ModelPusherConfig:
    def __init__(self):
        self.bentoml_model_name: str = BENTOML_MODEL_NAME # "xray_model"

        self.bentoml_service_name: str = BENTOML_SERVICE_NAME # "xray_service"

        self.train_transforms_key: str = TRAIN_TRANSFORMS_KEY # "xray_train_transforms"

        self.bentoml_ecr_image: str = BENTOML_ECR_URI # "xray_bento_image"
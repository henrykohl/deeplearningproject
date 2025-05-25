# deeplearningproject

Lung X-ray Classification Data link: https://drive.google.com/file/d/1pfIAlurfeqFTbirUZ5v_vapIoGPgRiXY/view?usp=sharing

Workflows constants config_enity artifact_enity components pipeline main How to setup: conda create -n lungs python=3.8 -y conda activate lungs pip install -r requirements.txt setup AWS CLI link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html aws configure AWS_ACCESS_KEY_ID=\*\*\*

AWS_SECRET_ACCESS_KEY= \*\*\*

AWS_REGION = us-east-1

BentoML demo repo: https://github.com/entbappy/bentoml-demo

[GitHub -- for Lecture 1 & 2](https://github.com/sunnysavita10/Chest-XRay-Classification/)

[GitHub -- for Lecture 3](https://github.com/entbappy/Lung-Xray-Classification/)

---

# Lecture 1 -- [Deep Learning Essentials](https://www.youtube.com/watch?v=essGb4QDXEU)

## Create a new respository (26:30)

- Respository name\* : `deeplearningproject`

- Description (optional) : `     `

- Keep Public

- Add a README file

- Add .gitignore : `.gitignore template Python`

- Choose a license : `License: MIT License`

## Open a a terminal in VSCode (33:00)

```bash
ls -a
# rm -rf .git ## if exists

git init
# git remote -v ## show the remote repository if we have connected with any remote repository
git remote add origin GITHUB的網址
git remote -v
git branch
git pull origin main
```

- In my case: using `Codespaces`, SKIP the above GIT operations

## MLOPS (42:30)

- ML

  > DVC
  >
  > MLflow
  >
  > Airflow
  >
  > Azure

- DL
  > Vision -- CNN
  >
  > NLP

## Complete infrastructure (48:25)

- Training pipeline

  > Data ingestion (S3)
  >
  > Processing
  >
  > Model training
  >
  > Push Model
  >
  > Model evaluation

- Prediction pipeline

## Content in the project (50:45)

1. DVC

2. MLFLOW

3. Bentoml

4. Docker

5. Testcases

- AWS
  > 1. S3
  >
  > 2. ECR
  >
  > 3. EC2
  >
  > 4. APP runner

## Implementation (57:00)

- `/.github` : going to write it on my entire configuration inside this folder. Going to keep my entire workflows related configuration (1:07:08)
  > `/workflows` : the continuous integration related to this continuous deployment (1:07:40)
  >
  > > `main.yaml` : write my all the configuration in this file (1:07:49)
  > >
  > > `ci.yaml`  (1:08:04)

* `/Xray` (58:00)
  > `/cloud_storage`: all related to the cloud, to the S3 (58:25)
  >
  > > `__init__.py` (1:01:10)
  > >
  > > `s3_operation.py` (1:00:55)
  >
  > `/components`: data ingestion, data transformation, pre-processing, model training, model pusher (58:50)
  >
  > > `data_ingestion.py` (1:01:30)
  >
  > > `data_transformation.py` (1:01:45)
  > >
  > > `model_evaluation.py` (1:02:20)
  > >
  > > `model_training.py` (1:02:00)
  > >
  > > `model_pusher.py` (1:02:35)
  >
  > `/entity` : all the configuration, (entity) representing component only (59:10)
  >
  > > `artifact_entity.py` (1:05:25)
  > >
  > > `config_entity.py` (1:05:15)
  >
  > `/pipeline` (59:30)
  >
  > > `training_pipeline.py` (1:05:50)
  >
  > `logger.py` (59:50)
  >
  > `exception.py` (59:55)
  >
  > `__init__.py`: recognize this folder as a package (1:10:15)

- `/test`: those test cases we will write down inside the test folder (1:08:40)

  > `/integrationtest` (1:08:59)
  >
  > > `__init__.py` (1:09:10)
  >
  > `/unittest` (1:08:48)
  >
  > > `__init__.py` (1:09:22)

- `requirements.txt` : for production whenever we are going to deploy it so well. (1:06:25)

- `setup.py` (1:06:34)
  whenever you are importing your file inside your component, you can install it in your virtual environment as a package for that. this setup.py will help you.

* `bentofile.yaml` : write down the entire configuration related to the Bento ml only

write down the configuration in the form of key and value. (1:09:56)

* push this complete folder to Github (1:10:42)
```bash
git add .
git config --global user.name "使用者名稱" (useless)
git config --global user.email "使用者電子郵件" (useless)
git commit -m "folder updated"
git branch -M main  # 若無，則會產生錯誤 (1:15:00)
git push origin main
```

---

- `/experiment` (1:19:30)
  > `experiment.ipynb` (1:19:40)

* `tox.ini` : do one thing like test our case. write down the configuration over here. (check the MLOps Lecture) (1:20:23)

* `setup.cfg` : if you want to  publish this one as your package so you can mention the configuration inside the setup.cfg. (1:21:30)

* `requirements_dev.txt` : write down the specific library just for development environment (1:21:50)

* `init_setup.sh` : write down our Shell Script just for the automating the entire thing environment creation or like this environment installation and all. We can write down each and every command over here inside this file. (1:22:55)

* post the recent changes in Github (1:23:25)
```bash
git add .
git commit -m "more file added"
git push -u origin main
```

---

## Create a virtual environment and Activate (1:28:30)

```bash
conda create -p env python=3.8 -y
#不要使用`-n`, it's not going to create in your present directory it will create in a default location
# `-p`: path, currect idrectory
source activate ./env
```

- 完成 `requirements.txt` (1:31:10)

```sh
bentoml # used in this project
torchvision # download the pytorch module into the virtual environment
joblib
pip-chill # automatically it will you a specific version (like some specific version of a torch widget) -
# - match the version with the current python version to the specific library
tqdm # get the progress bar whenever you want the progress bar regarding to any sort of a process
wincertstore # when you are going to connect with the server now so the SSL is required

-e .
```

- 完成 `requirements_dev.txt` (1:35:18)

```sh
bentoml 
torchvision 
joblib
pip-chill 
tqdm 
wincertstore 

dvc # do that data management by using this DVC
mlflow # check each and every experiment by using this mlflow
ipykernel
pandas
numpy
seaborn

pytest==7.1.3 # write down different cases and check into different environment
tox==3.25.1 # provide you the environment and by using pytest write down the different use cases.
flake8==5.0.4 # for linting
mypy==0.971 # for linting
black==22.8.0 for linting

-e . # setup.py will be executed. this dot is for this setup.py. `e` means execute. `.` means what you need to find out the package in the current directory. 
# 如果 setup.py 是空的或不存在, `-e .` 要 mask 掉)
```

* (1:39:15)

```bash
pip install -r requirements_dev.txt
# `-e .` 要 mask 掉，因為 setup.py 此時為空

pip list
# 檢查哪些 package 被安裝

git status
# 察看目前 git 狀態

git add .
git commit -m "requirement added"
git push (-f) origin main
# `-f` 需要時，可以用
```

---

- 完成 `setup.py` (1:49:00)

```python
from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
name="Xray",
version="0.0.1",
author="henry kohl",
author_email="u860218@gmail.com",
install_requires=get_requirements(r"//workspaces//deeplearningproject//requirements_dev.txt"),
package=find_packages()
)
```


```bash
git add .
git commit -m "setup file updated"
git push (-f) origin main
```

---

# Lecture 2 -- [Data Ingestion with S3](https://www.youtube.com/watch?v=3uXCAI3MOZ8)

* In local site,

> ```bash
> git clone https://github.com/henrykohl/deeplearningproject.git
> ```

* `Xray` 中 `components`, `entity`, `pipeline` 下都需要有 `__init__.py`

* 在此 Lecture 實做時，安裝 packages 建立 virtual environment，會使用到 setup.py (此時不是空白)，所以 `requirements_dev.txt`的 `-e .` 要存在才可

> ```bash
> conda create -p venv python=3.8 -y
> conda activate venv的途徑
> pip install -r requirements_dev.txt
> ```

* 注意 -- `requirements_dev.txt` 此時與 Lecture 1 中的 `requirements_dev.txt` 不同在於 六個 packages 需要指定版本。此 Lecture 的操作是在 Local 端~ 而 Lecture 1 是在 iNeuron 上 (本實做在 Codespace)。

> ```sh
> bentoml==1.0.25
> # bentoml==1.0.10
> joblib==1.2.0
> pip-chill==1.0.1
> torchvision==0.14.1
> tqdm==4.64.1
> wincertstore==0.2
>```

* 安裝完畢，可用 `pip list` 查看， **Xray** 的版本顯示為 `0.0.1`

> ``` bash
> git add .
> git commit -m "requirement updated"
> git push -m origin main
> ```

## Implementation

- 完成 `/Xray/exception.py`

* 部分完成 `/Xray/cloud_storage/s3_operation.py`

```python
import os
import sys

from Xray.exception import XRayException


class S3Operation:
    def sync_folder_to_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            pass

        except Exception as e:
            raise XRayException(e, sys)

    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            pass

        except Exception as e:
            raise XRayException(e, sys)
```

* 部分完成 `/Xray/components/data_ingestion.py`

```python
import sys

from Xray.cloud_storage.s3_operation import S3Operation
from Xray.constant.training_pipeline import *
from Xray.entity.artifact_entity import DataIngestionArtifact
from Xray.entity.config_entity import DataIngestionConfig
from Xray.exception import XRayException
from Xray.logger import logging

class DataIngestion:
    def __init__(self):

    def get_data_from_s3(self) -> None:
        try:
            pass

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_ingestion(self):
        try:
            pass

        except Exception as e:
            raise XRayException(e, sys)
```

- `/constant`

  > `__init__.py`
  >
  > `/training_pipeline`
  >
  > > `__init__.py` 完成 Data Ingestion Constants 的部分
  > >
  > > > 其中 `BUCKET NAME` 必需與 AWS S3 的 名稱 一致，且 global 範圍中 唯一

- `/Xray/entity/config_entity.py` 完成 class DataIngestionConfig 的部分

- `/Xray/entity/artifact_entity.py` 完成 class DataIngestionArtifact 的部分

```bash
git add .
git commit -m "code skeleton updated"
git push -u origin main
```

- `/Xray/pipeline/training_pipeline.py` （主要）完成 start_data_ingestion

```python
import sys
from Xray.components.data_ingestion import DataIngestion
from Xray.entity.artifact_entity import DataIngestionArtifact
from Xray.entity.config_entity import DataIngestionConfig

from Xray.exception import XRayException
from Xray.logger import logging

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:

            logging.info("Getting the data from s3 bucket")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config,
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Got the train_set and test_set from s3")

            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise XRayException(e, sys)

if __name__ == "__main__":
  train_pipeline=TrainPipeline()

  train_pipeline.start_data_ingestion()s
```

```bash
git add .
git commit -m "training config updated"
git push origin main
```

- Training_Pipeline -- Pipeline
  > (1)> **Data_ingestion** --(2)--> (1)> **Data_transformation** --(2)--> **Model_training** -> **Model_evaluation**
  >
  > > (1) config, (2) artifact
  >
  > entity, logger, exception, constant, cloud_storage

* 完整完成 `/Xray/cloud_storage/s3_operation.py`

- 完整完成 `/Xray/components/data_ingestion.py`

* 完成 `logger.py`

```bash
git add .
git commit -m "data ingestion code updated"
git push origin main
```

- AWS S3 (1:48:50)
  > IAM (1:52:50)
  >
  > > 建立使用者名稱: `liveyoutubesession`
  >
  > > 按下 `Next`
  >
  > > Set permissions -- Permissions options: `Attach policies directly`
  > >
  > > Permissions policies: `AdministratorAccess`
  > >
  > > 按下 `Next`
  > >
  > > 按下 `Create user`
  >
  > > 在帳號 `liveyoutubesession` 的 Summary 頁面選擇 `Security credentials`
  > >
  > > Access keys (1) -- 按下 `Create access key`
  > >
  > > Access key best practices & alternatives -- Use case 選擇 `Command Line Interface (CLI)`
  > >
  > > 按下 `Next`
  > >
  > > 按下 `Create access key`
  >
  > > 按下 `Download .csv file` -- 檔案 `liveyoutubesession_accessKeys.csv` 會用到 [file link](https://drive.google.com/file/d/1Ri6KPbtMiigfCosuJfOHQHMdoZe_DDJu/view?usp=drive_link)

* 在 VS Code 的 terminal 中，切換到 virtual environment 之下

```bash
aws configure
# 輸入 AWS Access Key ID
# 輸入 AWS Secret Access Key
# 輸入: us-east-1 (在Default region name )
# 按下Enter (在Default output format[None])
```

- Upload 資料夾 [data](https://drive.google.com/file/d/1pfIAlurfeqFTbirUZ5v_vapIoGPgRiXY/view) 到 S3
  > 先下載 data 後是一個 ZIP 檔，需要先解壓縮後再把資料夾 data 上傳

* 執行

```bash
python Xray/pipeline/training_pipeline.py
```

Lecture 使用以上的方式執行，但實際測試時，會找不到 Xray 此 package，所以改使用以下方式，才成功執行 Data Ingestion

```bash
python -m Xray.pipeline.training_pipeline
```

---

- 擷取 DataIngestion 功能，可在 Colab 等平台運作

```PowerShell
!pip install aws configure --quiet
!pip install awscli --quiet
!aws configuration
```

```python
from dataclasses import dataclass
import os
import sys

class S3Operation:
    def sync_folder_from_s3(self, folder: str, bucket_name: str, bucket_folder_name: str) -> None:
        try:
            command: str = (
                f"aws s3 sync s3://{bucket_name}/{bucket_folder_name}/ {folder} "
            )

            os.system(command)

        except OSError as e:
            raise Exception(e)


from datetime import datetime


TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACT_DIR: str = "artifacts"
BUCKET_NAME: str = "xraylungimgsproj"
S3_DATA_FOLDER: str = "data"


s3 = S3Operation()

@dataclass
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

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.s3 = S3Operation()

    def get_data_from_s3(self) -> None:

        self.s3.sync_folder_from_s3(
            folder=self.data_ingestion_config.data_path,
            bucket_name=self.data_ingestion_config.bucket_name,
            bucket_folder_name=self.data_ingestion_config.s3_data_folder,
        )

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        self.get_data_from_s3()
        data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
            train_file_path=self.data_ingestion_config.train_data_path,
            test_file_path=self.data_ingestion_config.test_data_path,
        )
        return data_ingestion_artifact


data_ingestion_config = DataIngestionConfig()

def start_data_ingestion() -> DataIngestionArtifact:
    data_ingestion = DataIngestion(
        data_ingestion_config = data_ingestion_config,
    )
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    return data_ingestion_artifact


start_data_ingestion()
```

---

# Lecture 3 -- [Data Transformation Components](https://www.youtube.com/watch?v=6-uZFeyfiCE)

- End to End DL project Implementation
  > 1. Introduction of the project / How to setup environment and requirements
  >
  > 2. Data Ingestion component --> S3 bucket (AWS)
  >
  > 3. Data Transformation

## workflow

- constants
- config_entity
- artifact_entity
- components
- pipeline
- main

## How to setup

```bash
conda create -n lungs python=3.8 -y
```

```bash
conda activate lungs
```

在 Lecture 3 時 virtual environment 的名稱是 lungs, 對照 Lecture 1 使用

```bash
pip install -r requirements.txt
```

```bash
setup AWS CLI
link: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html
```

```bash
aws configure
```

```bash
AWS_ACCESS_KEY_ID=*****************************

AWS_SECRET_ACCESS_KEY=***************************

AWS_REGION = us-east-1
```

```bash
git add .
git commit -m "readme updated"
git push origin main
```

## Data Transformation (->)

> 1. Augmentation
> > 1. Brightness
> >
> > 2. Saturation
> >
> > 3. Size
> >
> > 4. Normalization
> >
> > 5. Centercrop
> >
> > 6. Random Rotation

- (->) : train-data.pkl & test-data.pkl

  > In `model training component`, we will load these particular two data (->).

- Start with model training, it's just a pipeline:
  > Data Ingestion -> download data from S3
  >
  > Data Transformation -> Augmentation (saving data in a pkl format)
  >
  > Model Trainer -> load the data & start the model trainer component
  >
  > Model Evaluation -> test your model on top of the dataest you are having
  >
  > Model Pusher -> push this model to the S3 bucket

* `/Xray/constant/training_pipeline/__init__.py` 完成 data trasnforamtion 的部分

* `/Xray/entity/config_entity.py` 完成 class DataTransformationConfig 的部分

* `/Xray/entity/artifact_entity.py` 完成 class DataTransformationArtifact 的部分

* 完成 `/Xray/components/data_transformation.py`

* `/Xray/pipeline/training_pipeline.py` 

  > 新增 `def start_data_transformation` 的部分
  >
  > 修改 `def run_pipeline`

* 執行 data ingestion (Lecture 2) + data transformation (Lecture 3) 功能

```bash
python main.py
```

- Git Commit: "data transformation added"

---

# Lecture 4 -- [Model Trainer & Bento ML](https://www.youtube.com/watch?v=Aahc28-f4hc)

- Agenda & Review
  > 1. Introduction and Project setup (Lecture 1)
  >
  > 2. Data Ingestion -> S3 (Lecture 2)
  >
  > 3. Data Transformation -> Augmentation (Lecture 3)
  >
  > 4. Model Trainer (Lecture 4)
  >
  > 5. BentoML Demo -> MLOPs tools (Lecture 4)
  >    > perform -- model serving, application packaging and production grid deployment

* Component Architecture

  > |Data Ingestion|
  >
  > > --> Train Data
  > >
  > > --> Test Data
  >
  > |Data Transformation|
  >
  > > --> Transformed Train Data
  > >
  > > --> Transformed Test Data
  >
  > |Model Trainer| (\*)
  >
  > > Trained Model
  >
  > |Model Evaluation| (\*)
  >
  > > Metrices
  >
  > |Model Pusher| (\*)
  >
  > > --> S3 bucket (push the model we have trained to the S3)
  > > (\*) : BentoML as MLOPS tool

- `/Xray/constant/training_pipeline/__init__.py` (新增 model trainer constants 的部分)

* `/Xray/entity/config_entity.py` (新增 class ModelTrainerConfig 的部分)

* `/Xray/entity/artifact_entity.py` (新增 class ModelTrainerArtifact 的部分)

* 參考 [Training with PyTorch](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)

* `/Xray/ml`

  > `__init__.py`
  >
  > `/model`
  >
  > > `__init__.py`
  > >
  > > 完成 `arch.py`

* 完成 `/Xray/components/model_training.py`

  > 重點在
  >
  > ```python
  > bentoml.pytorch.save_model(
  >
  >                name=self.model_trainer_config.trained_bentoml_model_name,
  >                model=model,
  >                custom_objects={
  >                    self.model_trainer_config.train_transforms_key: train_transforms_obj
  >                },
  >            )
  > ```
  >
  > 見參考 8

* 「1」Training Pipeline (Review)

  > 1. Data Ingestion
  >
  > 2. Data Transformation
  >
  > 3. Model Trainer
  >
  > 4. Model Evaluation
  >
  > 5. Model Pusher

* 「2」Prediction Pipeline

  > 1. load model
  >
  > 2. prediction

* 「3」|Flask| --> API --> |Web Interface|

  > user --> |Web Interface| --> prediction

* 「1」+「2」+「3」是針對 localhost 運行，一般使用者無法存取，所以需要 development~~

* |CICD AWS|

  > Docker --> ECR --> EC2

* |Bento ML|

  > Model Serving
  >
  > Application Packaging --> bento.yaml --> Bento Image
  >
  > > AWS
  > > Azure
  > > GCP

* Bento Demo (57:00) -- [Bento Demo Repo](https://github.com/entbappy/bentoml-demo/)

  > 執行命令: `conda activate env` (env 需要先建立)
  >
  > `requirements.txt`
  >
  > > 執行命令: `pip install -r requirements.txt`
  >
  > `bento_train.py`
  >
  > > 執行命令: `python bento_train.py`
  >
  > `bento_cmd.txt`
  >
  > `bento_test.py`
  >
  > > 執行命令: `python bento_test.py`
  >
  > `service.py` (1:10:18)
  >
  > `bentoml_cmd.txt`
  >
  > > 執行命令: `bentoml serve service.py:service --reload`
  >
  > `bentofile.yaml` (1:14:29)
  >
  > 執行命令: `bentoml build`

* `/Xray/pipeline/training_pipeline.py` 完成 start_model_trainer 的部分

* Git Commit: "model trainer added" (1:28:00)

* Bento Demo 實做 Repo 與 notes ，參見 [My BentoML Demo -- README](https://github.com/henrykohl/bentoml-demo/)

---

# Lecture 5 -- [Model Evaluation & Prediction Pipeline](https://www.youtube.com/watch?v=09aayk0s9B4)

- Recap
  > (1) Data Ingestion [S3] -> Artifact
  >
  > (2) Data transformation
  >
  > (3) Model training -- executed by Pytorch & CNN
  >
  > (4) Model Evaluation (focused by this lecture)
  >
  > (5) Model Pusher

* [Project Architecture](https://app.whiteboard.microsoft.com/me/whiteboards/3af4b83f-c715-41cd-82de-a8cd5efde3b3)

* [CNN Architecture]() (1:37:25 )

- 完整完成 `/experiment/Experiment.ipynb`

- 完整完成 `/Xray/components/model_evaluation.py`

* `/Xray/entity/config_entity.py` (新增 class ModelEvaluationConfig 的部分)

* `/Xray/entity/artifact_entity.py` (新增 class ModelEvaluationArtifact 的部分)

* `/Xray/pipeline/training_pipeline.py` 完成 start_model_evaluation 的部分

---
# Lecture 6 -- [Evaluate and Deploy Deep Learning Models to the Cloud](https://www.youtube.com/watch?v=jk_YAsI9z5w)

* Recap (20:15)

> <pre>
> |Data|: images -- 1. NORMAL 2. PNEUMONIA \
>    |__Training (*訓) \
>    |__Validation/testing (*測)
</pre>


> |Data Ingestion| --> |Data Transformation| --> |Model Train|(*訓) --> |Model evaluation|(*測) --> |Model Pusher|

* 復盤 `/Xray/components/model_evaluation.py` (30:50)

* 復盤 `/Xray/components/model_training.py`
> 特別關注 **bentoml** 的部分

* 完整完成 `/Xray/components/model_pusher.py` (1:01:00)
> (1:02:34) 開始解說

* 建立完成 `/test.py` (1:06:55)

>  執行 `python test.py` (requirements_dev.txt 必要存在)

* 完整完成 `/bentofile.yaml`

* AWS ECR （1:10:00）

>  在 AWS 的 ECR 建立一個 repository: `xray_bento_image`

* 建立 `/Xray/ml/model/model_service.py`

* BentoML 教學 (1:14:10)
> [BentoML github](https://github.com/bentoml/BentoML)
>
> [BentoML](https://bentoml.com/)

* 解說 `/Xray/ml/model/model_service.py` (1:21:00)

> * 建立所需的 API

* 解說 `/bentofile.yaml` (1:27:50)

* `/Xray/pipeline/training_pipeline.py` 完成 start_model_pusher 的部分

* 運行 `main.py` (1:32:50)

> - 在 terminal 執行 `source activate ./venv` 以進入 virtual environment
>
> - 執行 `python main.py` -- Lecture 使用 bentoml 1.0.10 或 1.0.25 都會產生 KeyError: 'name' 的錯誤

* 查詢 PyPi BentoML 相關資訊 (1:37:00) 

* [bug: KeyError: 'name'](https://github.com/bentoml/BentoML/issues/4500)(1:40:55)

* 用 Streamlit 取代 BentoML (1:47:30)
> - Github repository -- [Xray-lung-classifier](https://github.com/sunnysavita10/Xray-lung-classifier)
>
> 1. 檢視 `/xray/components/model_pusher.py`
>
> > * 訓練後模型的路徑 `/model/model.pt`
> 
> > * just upload MODEL file to s3
>
> 2. 檢視 `/app.py`
>
> 3. 檢視 `/xray/cloud_storage/s3_ops.py`
> > * 使用 boto3 來連接 s3
>
> 4. 建立 virtual environment: `conda create -p venv python=3.8 -y`
>
> 5. 執行 `source activate ./venv`
>
> 6. 執行 `pip install -r requirments.txt`
>
> 7. 執行 `streamlit run app.py`，後開啟 web APP (2:00:50)
>
> 8. google 一張 'pneumonia lungs x ray' 的 jpeg 圖，將其下載，
>
> 9. 將 jpeg 檔上傳至 web APP -- 檔案將存到 `/images/input.jpeg`

* Project saved in Github [deeplearningproject](https://github.com/henrykohl/deeplearningproject)
> * `git remote -v`
>
> * `git add .`
>
> * `git commit -m "latest changes: pusher evaluation and bentoml"`
>
> * `git push -u origin main`

* Project saved in Github [deeplearningupdatedwithstreamlit]
> * Create a new repository -- Repository name: `eeplearningupdatedwithstreamlit`
>
> * `Public` --> `save`
>
> * `git remote -v` --> 發現 no repository
>
> * `rm -rf .git`
>
> * `git init`
>
> * `git add .`
>
> * `git commit -m "updated code"`
>
> * `git remote add origin https://github.com/ehnrykohl/deeplearningupdatedwithstreamlit`
>
> * `git push -u origin main`

---

# 補充

- `colabscript.ipynb` 可以實現將此 project (Lec 1 ~ Lec 3) 在 Colab 上運行（可使用 free GPU）

## Lecture 1 ~ 4 實做細節（依據 Codespaces）

> Lecture 1 使用的 `requirements.txt` ，而沒有用到 `setup.py`

```sh
bentoml
torchvison
joblib
tqdm
pip-chill
wincertstore

-e .
```

> Lecture 2 使用的 `requirements_dev.txt` 和 `setup.py`

```sh
bentoml
torchvision
joblib
tqdm
pip-chill
wincertstore

dvc
mlflow
ipykernel
pandas
numpy
seaborn

pytest==7.1.3
tox==3.25.1
black==22.8.0
flake8==5.0.4
mypy==0.971

#-e .
```

```python
## setup.py
from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(

name="Xray",
version="0.0.1",
author="henry kohl",
author_email="u860218@gmail.com",
# install_requires=get_requirements(r"C:\\Users\\sunny\\deeplearningproject\\requirements_dev.txt"),
install_requires=get_requirements(r"//workspaces//deeplearningproject//requirements_dev.txt"),
package=find_packages()

)
```

> Lecture 3 & 4 使用不同的 `requirements.txt` 和 `setup.py`

```sh
# requirements.txt
bentoml==1.0.25
joblib==1.2.0
pip-chill==1.0.1
torchvision==0.14.1
tqdm==4.64.1
wincertstore==0.2
-e .
```

```python
## setup.py
from setuptools import find_packages, setup

setup(
    name = 'xray',
    version= '0.0.0',
    author= 'Henry Hsu',
    author_email= 'u860218@gmail.com',
    packages= find_packages(),
    install_requires = []

)
```

## 非常有用的 Saas Cloud dev environment for python

> - Codespaces (成功完成 Xray project)
>
> - Codesandbox (成功完成 Xray project)
>
> - Gitpod (成功完成 Xray project)
>
> - Codeanywhere (成功完成 Xray project)
>
> - Replit (未能成功)
>
> - RunCode (未實做)

---

- 開啟以上任一 Saas Cloud editor 並完成 project import (from Github) 之後，首要的工作為進行環境設定。
  > 先開啟 terminal 以執行後續步驟

### Codespaces

步驟就是前面 Lecture 1 ~ Lecture 4 的方式

- 1. 先安裝 **Conda**

- 2. 用 conda 建立 virtual environment

- 3. 安裝 aws CLI -- 在 .tmp 目錄下執行 (要使用 bash shell)

### Codesandbox

完全同 Codespaces 的步驟

### Codeanywhere

- 1. 先安裝 **Conda**

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda3/miniconda.sh  # (wget 要使用 -O)
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda list # 測試
python --version ## 是 Python 3.12.9
```

- 2. 用 conda 建立 virtual environment

```bash
conda create -p env python==3.8.0 -y
conda activate ./env
python --version ## 是 Python 3.8.0

conda activate ./env ## 用 conda 啟動 env
```

- 3. 安裝 aws CLI -- 在 .tmp 目錄下執行 (要使用 bash shell)

```bash
mkdir .tmp && cd $_
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

aws configuration ## 測試 aws CLI 是否安裝成功
aws configure ## 設定 aws CLI
```

```
輸入 AWS Access Key ID
輸入 AWS Secret Access Key
Default region name 輸入: us-east-1
Default output format[None]: 按下 Enter
```

### Gitpod (啟動很慢)

- 1.  先安裝 **Conda** (Codespaces 不需要)

  > ```bash
  > mkdir .tmp
  > cd .tmp
  > wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
  > bash Anaconda3-2020.07-Linux-x86_64.sh
  > ```
  >
  > 最後一步驟「Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no] [no] >>> yes」
  > 接著要啟動一個 bash shell，有兩個方式
  >
  > > 方式 1: 在原 terminal 輸入，`eval "$(/home/gitpod/anaconda3/bin/conda shell.bash hook)"`
  >
  > > 方式 2: 開啟一個新的 bash shell terminal
  >
  > 在 terminal 輸入 `conda --version` 或 `conda list` 檢測 conda 是否安裝成功

- 2.  用 conda 建立 virtual environment

  > 切換到 project 工作目錄
  >
  > ```bash
  > conda create -p env python==3.8 -y
  > conda create --name env python==3.8 -y ## 另法
  > # conda create -p env python=3.8 -y ## 當啟動 env 後，有時候 python 版本依然是最新版，非版本3.8
  > ```
  >
  > 執行 `conda activate ./env` 看是否成功，再輸入 `python --version` 查看 env 環境下，安裝的 python 版本是否正確

- 3.  安裝 aws CLI

  > 用 conda 啟動 env 後，在 .tmp 目錄下 執行
  >
  > ```bash
  > curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  > unzip awscliv2.zip
  > sudo ./aws/install
  > ```
  >
  > 執行 `aws configuration` 看是否 aws CLI 安裝成功
  >
  > 設定 aws，執行 `aws configure`
  >
  > ```
  > 輸入 AWS Access Key ID
  > 輸入 AWS Secret Access Key
  > Default region name 輸入: us-east-1
  > Default output format[None]: 按下 Enter
  > ```

* Conda 安裝完後，在新開啟的 terminal (bash shell) 中， python 版本就是 3.8.3 (不啟動 virtual environment)

### Replit (未能完成)

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda/miniconda.sh  # (wget 要使用 -O)
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda list # 測試
python --version ## 是 Python 3.12.9

conda create -p env python==3.8.0 -y
conda activate ./env
python --version ## 是 Python 3.8.0

conda activate ./env
aws configuration # 選擇 awscli 或 awscli2 (在Replit中好似不能自行下載awscli後進行安裝)
```

- 建立 `requirements.txt` 與 `setup.py`後，執行 `pip install -r requirements.txt`

但由於 storage limitation, 安裝到一半就出現空間不足的 Error，無法完全成功安裝所有 packages

### RunCode (未測試)

似乎免費空間比 Replit 大

---

- 在 project 工作目錄
  > 建立 `requirements.txt` 與 `setup.py`
  >
  > 執行 `pip install -r requirements.txt`
  >
  > 執行 `python main.py`

# Tech issues

- 參 1 [Installating AWS CLI on Windows 7](https://github.com/aws/aws-cli/issues/7659)

- 參 2 [AWS SSO + Codespaces](https://gist.github.com/pahud/ba133985e1cf3531c09b5ea553a72739)

- 參 3 [Python import 簡易教學](https://medium.com/@alan81920/c98e8e2553d3)

- 參 4 [Python 相對匯入與絕對匯入](https://brainynight.github.io/notes/python-import/)

- 參 5 [Python——在不同層目錄 import 模塊的方法](https://blog.csdn.net/weixin_41605937/article/details/124909644)s

* 參 6 [Meaning of -b and -p in bash script.sh](https://stackoverflow.com/questions/58303251/meaning-of-b-and-p-in-bash-script-sh-b-p-directory)

* 參 7 [Compare CodeSandbox vs. Codeanywhere vs. Codespaces vs. Gitpod](https://slashdot.org/software/comparison/CodeSandbox-vs-Codeanywhere-vs-Codespaces-vs-Gitpod/)
  > 共同特徵 -- online IDE for full stack project?

* 參 8 [BentoML Pytorch Documentation](https://docs.bentoml.com/en/latest/reference/bentoml/frameworks/pytorch.html#bentoml.pytorch.save_model)

* 參 9 [BentoML Documentation](https://docs.bentoml.com/en/latest/index.html)

/usr/local/bin/aws

/usr/local/aws-cli/v2/current
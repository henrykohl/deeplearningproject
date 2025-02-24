# deeplearningproject

Lung X-ray Classification Data link: https://drive.google.com/file/d/1pfIAlurfeqFTbirUZ5v_vapIoGPgRiXY/view?usp=sharing

Workflows constants config_enity artifact_enity components pipeline main How to setup: conda create -n lungs python=3.8 -y conda activate lungs pip install -r requirements.txt setup AWS CLI link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html aws configure AWS_ACCESS_KEY_ID=\*\*\*

AWS_SECRET_ACCESS_KEY= \*\*\*

AWS_REGION = us-east-1

BentoML demo repo: https://github.com/entbappy/bentoml-demo

[GitHub -- for Lecture 1 & 2](https://github.com/sunnysavita10/Chest-XRay-Classification/)

[GitHub -- for Lecture 3](https://github.com/entbappy/Lung-Xray-Classification/)

# Lecture 1 -- [Deep Learning Essentials](https://www.youtube.com/watch?v=essGb4QDXEU)

## Create a new respository

- Respository name\* : `deeplearningproject`

- Description (optional) : `     `

- Keep Public

- Add a README file

- Add .gitignore : `.gitignore template Python`

- Choose a license : `License: MIT License`

## Open a VSCode

```
# rm -rf .git ## if exists

git init
# git remote -v ## show the remote repository if we have connected with any remote repository
git remote add origin GITHUB的網址
git remote -v
git branch
git pull origin main
```

- In my case: using `Codespaces`, SKIP the above GIT operations

--

## Implementation

- `/.github` : going to write it on my entire configuration inside this folder. going to keep my entire workflows related configuration
  > `/workflows` : the continuous integration related to this continuous deployment
  >
  > > `main.yaml` : write my all the configuration in this file
  > >
  > > `ci.yaml` :

* `/Xray`
  > `/cloud_storage`: all related to the cloud, to the S3
  >
  > > `__init__.py`
  > >
  > > `s3_operation.py`
  >
  > `/components`: data ingestion, data transformation, pre-processing, model training, model pusher
  >
  > > `data_ingestion.py`
  >
  > > `data_transformation.py`
  > >
  > > `model_evaluation.py`
  > >
  > > `model_training.py`
  > >
  > > `model_pusher.py`
  >
  > `/entity` : all the configuration, (entity) representing component only
  >
  > > `artifact_entity.py`
  > >
  > > `config_entity.py`
  >
  > `/pipeline`
  >
  > > `training_pipeline.py`
  >
  > `logger.py`
  >
  > `exception.py`
  >
  > `__init__.py`: recognize this folder as a package

- `/test`: those test cases we will write down inside the test folder

  > `/integration`
  >
  > > `__init__.py`
  >
  > `/unittest`
  >
  > > `__init__.py`

- `requirements.txt`

- `setup.py`
  whenever you are importing your file inside your component, you can install it in your virtual environment as a package for that. this setup.py will help you.

* `bentofile.yaml` : write down the entire configuration related to the Bento ml only

write down the configuration in the form of key and value.

```
git add .
git config --global user.name "使用者名稱" (useless)
git config --global user.email "使用者電子郵件" (useless)
git commit -m "folder updated"
git branch -M main  # 若無，則會產生錯誤 (1:15:00)
git push origin main
```

---

- `/experiment`
  > `experiment.ipynb`

* `tox.ini` : do one thing like test our case (check the MLOps Lecture)

* `setup.cfg` : publish this one as your package so you can mention the configuration inside the setup.cfg.

* `requirements_dev.txt` : write down the specific library for development environment

* `init_setup.sh` : write down our Shell Script for the automating the entire thing environment creation or like environment installation and all. We can write down each and every command over here inside this file.

```
git add .
git commit -m "more file added"
git push -u origin main
```

---

## Create a virtual environment

```
conda create -p env python=3.8 -y
#不要使用`-n`, it's not going to create in your present directory it will create in a default location
# `-p`: path, currect idrectory
source activate ./env
```

- 完成 `requirements.txt`

> pytest : write down different cases and check into different environment
>
> tox : provide you the environment and by using pytest write down the different use cases.
>
> flake8 : for linting
>
> mypy : for linting
>
> black : for linting
>
> -e . : setup.py will be executed. this dot is for this setup.py. `e` means execute. `.` means what you need to find out the package in the current directory.
> dvc : do that data management by using this DVC
>
> mlflow: check each and every experiment by using this mlflow
>
> ipykernel
>
> pandas
>
> numpy
>
> seaborn

```
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

- 完成 `setup.py`

```
git add .
git commit -m "setup file updated"
git push (-f) origin main
```

## MLOPS

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

## Complete infrastructure

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

## Content in the project

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

## Description

- `.github` : going to write it on my entire configuration inside this folder. Going to keep our entire workflows related to configuration.
  > `workflows` : write our all the configuration in this file.
  >
  > `ci.yaml` :

* `bentofile.yaml`: write down the entire configuration related to the Bento yaml only.

* `tox.ini` : like test our cases. write down the configuration over here.

* `setup.cfg` : if you want to publish this one as a package, you can mention the confiuration inside this setup.cfg.

* `requirements_dev.txt` : write down the specific library just for development environment.

  > `dvc` -- do data management
  >
  > `mlflow` -- track each and every experiment by using this mlflow

* `requirements.txt` : for production whenever we are going to deploy it so well.

* `init_setup.sh` : write our shell script just for automating the entire environment creation like this requirement installation

# Lecture 2 -- [Data Ingestion with S3](https://www.youtube.com/watch?v=3uXCAI3MOZ8)

In local site,

> ```
> git clone https://github.com/henrykohl/deeplearningproject.git
> ```

`Xray` 中 `components`, `entity`, `pipeline` 下都需要有 `__init__.py`

在此 Lecture 實做時，安裝 packages 建立 virtual environment，會使用到 setup.py (此時不是空白)，所以 `requirements_dev.txt`的 `-e .` 要存在才可

```
conda create -p venv python=3.8 -y
conda activate venv的途徑
pip install -r requirements_dev.txt
```

注意 -- `requirements_dev.txt` 此時與 Lecture 1 中的 `requirements_dev.txt` 不同在於 六個 packages 需要指定版本。此 Lecture 的操作是在 Local 端~ 而 Lecture 1 是在 iNeuron 上 (本實做在 Codespace)。

```
bentoml==1.0.25
#bentoml==1.0.10
joblib==1.2.0
pip-chill==1.0.1
torchvision==0.14.1
tqdm==4.64.1
wincertstore==0.2
```

安裝完畢，可用 `pip list` 查看， **Xray** 的版本顯示為 `0.0.1`

```
git add .
git commit -m "requirement updated"
git push -m origin main
```

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

- 部分完成 `/Xray/components/data_ingestion.py`

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

```
git add .
git commit -m "code skelaton updated"
git push -u origin main
```

- 完成部分 `/Xray/pipeline/training_pipeline.py`

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

```
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

```
aws configure
輸入 AWS Access Key ID
輸入 AWS Secret Access Key
Default region name 輸入: us-east-1
Default output format[None]: 按下Enter
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

AWS_REGION = us-esat-1
```

```bash
git add .
git commit -m "readme updated"
git push origin main
```

## Data Transformation (->)

> 1. Augmentation
>    > 1. Brightness
>    >
>    > 2. Saturation
>    >
>    > 3. Size
>    >
>    > 4. Normalization
>    >
>    > 5. Centercrop
>    >
>    > 6. Random Rotation

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

  > 新增 `def start_data_transformation`
  >
  > 修改 `def run_pipeline`

* 執行 data ingestion (Lecture 2) + data transformation (Lecture 3) 功能

```bash
python main.py
```

- Git Commit: "data transformation added"

## Tech issues

- 參 1 [Installating AWS CLI on Windows 7](https://github.com/aws/aws-cli/issues/7659)

- 參 2 [AWS SSO + Codespaces](https://gist.github.com/pahud/ba133985e1cf3531c09b5ea553a72739)

- 參 3 [Python import 簡易教學](https://medium.com/@alan81920/c98e8e2553d3)

- 參 4 [Python 相對匯入與絕對匯入](https://brainynight.github.io/notes/python-import/)

- 參 5 [Python——在不同層目錄 import 模塊的方法](https://blog.csdn.net/weixin_41605937/article/details/124909644)

/usr/local/bin/aws

/usr/local/aws-cli/v2/current

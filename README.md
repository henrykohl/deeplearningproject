# deeplearningproject

Lung X-ray Classification Data link: https://drive.google.com/file/d/1pfIAlurfeqFTbirUZ5v_vapIoGPgRiXY/view?usp=sharing

Workflows constants config_enity artifact_enity components pipeline main How to setup: conda create -n lungs python=3.8 -y conda activate lungs pip install -r requirements.txt setup AWS CLI link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html aws configure AWS_ACCESS_KEY_ID=\*\*\*

AWS_SECRET_ACCESS_KEY= \*\*\*

AWS_REGION = us-east-1 BentoML demo repo: https://github.com/entbappy/bentoml-demo

# Lecture 1 -- [Deep Learning Essentials](https://www.youtube.com/watch?v=essGb4QDXEU)

```
git init
git remote -v
git remote add origin 網址 github
git remote -v
git branch
git pull origin main
```

---

```
git add .
git config --global user.name "使用者名稱" (useless)
git config --global user.email "使用者電子郵件" (useless)
git commit -m "folder updated"
git branch -M main
git push origin main
```

---

```
git add .
git commit -m "more file added"
git push -u origin main
```

---

```
conda create -p env python=3.8 -y
#不要使用`-n`

source activate ./env

pip install -r requirements_dev.txt
# `-e .` 要 mask 掉，因為 setup.py 此時為空

pip list
# 檢查哪些 package 被安裝

git status
# 察看目前 git 狀態
```

...

```
git add .
git commit -m "requirement added"
git push (-f) origin main
# `-f` 需要時，可以用
```

...

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

```
conda create -p venv python=3.8 -y
conda activate venv的途徑
pip install -r requirements_dev.txt
```

注意 -- `requirements_dev.txt` 此時與 Lecture 1 中的 `requirements_dev.txt` 不同在於 六個 packages 需要指定版本。此 Lecture 的操作是在 Local 端~ 而 Lecture 1 是在 iNeuron 上 (本實做在 Codespace)

```
bentoml==1.0.25
#bentoml==1.0.10
joblib==1.2.0
pip-chill==1.0.1
torchvision==0.14.1
tqdm==4.64.1
wincertstore==0.2
```

- Training_Pipeline -- Pipeline
  > (1)> **Data_ingestion** --(2)--> (1)> **Data_transformation** -(2)-> **Model_training** -> **Model_evaluation**
  >
  > > (1) config, (2) artifact
  >
  > entity, logger, exception, constant, cloud_storage

# Lecture 3 -- [Data Transformation Components](https://www.youtube.com/watch?v=6-uZFeyfiCE)

##workflow

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

## Tech issues

- [Installating AWS CLI on Windows 7](https://github.com/aws/aws-cli/issues/7659)

- [Python import 簡易教學](https://medium.com/@alan81920/c98e8e2553d3)

- [Python 相對匯入與絕對匯入](https://brainynight.github.io/notes/python-import/)

- [Python——在不同層目錄 import 模塊的方法](https://blog.csdn.net/weixin_41605937/article/details/124909644)

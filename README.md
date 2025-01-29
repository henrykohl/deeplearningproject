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

# Lecture 3 -- [Data Transformation Components](https://www.youtube.com/watch?v=6-uZFeyfiCE)

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

# deeplearningproject

Lung X-ray Classification Data link: https://drive.google.com/file/d/1pfIAlurfeqFTbirUZ5v_vapIoGPgRiXY/view?usp=sharing

Workflows constants config_enity artifact_enity components pipeline main How to setup: conda create -n lungs python=3.8 -y conda activate lungs pip install -r requirements.txt setup AWS CLI link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html aws configure AWS_ACCESS_KEY_ID=\*\*\*

AWS_SECRET_ACCESS_KEY= \*\*\*

AWS_REGION = us-east-1 BentoML demo repo: https://github.com/entbappy/bentoml-demo

git init

git remote -v

git remote add origin 網址 github

git remote -v

git branch

git pull origin main

...

git add .

git config --global user.name "使用者名稱"

git config --global user.email "使用者電子郵件"

git commit -m "folder updated"

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

import os
import sys

from Xray.entity.artifact_entity import ModelPusherArtifact
from Xray.entity.config_entity import ModelPusherConfig
from Xray.exception import XRayException
from Xray.logger import logging


class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig):
        self.model_pusher_config = model_pusher_config

    def build_and_push_bento_image(self):
        logging.info("Entered build_and_push_bento_image method of ModelPusher class")

        try:
            logging.info("Building the bento from bentofile.yaml")

            os.system("bentoml build") # 構建打包 bento

            logging.info("Built the bento from bentofile.yaml")

            logging.info("Creating docker image for bento")

            os.system(
                f"bentoml containerize {self.model_pusher_config.bentoml_service_name}:latest -t 136566696263.dkr.ecr.us-east-1.amazonaws.com/{self.model_pusher_config.bentoml_ecr_image}:latest"
            )
            # 將 bentos 容器化为Docker 镜像，並使用模型和 bento 管理服務 來大規模管理bentos。
            # 136566696263.dkr.ecr.us-east-1.amazonaws.com 來自 AWS -- ECR 所建立的 repository
            ## 136566696263 是 建立者 aws_account_id
            ## us-east-1 是 映像建立的region
            ## {self.model_pusher_config.bentoml_service_name} 是 repository 儲存庫名稱
            ## latest 是 tag

            logging.info("Created docker image for bento")

            logging.info("Logging into ECR")

            os.system(
                "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 136566696263.dkr.ecr.us-east-1.amazonaws.com"
            )
            # 執行 aws ecr get-login-password 命令。將身分驗證字符傳遞給 docker login 命令時，使用 AWS 的值作為使用者名稱並指定您要驗證的 Amazon ECR 登錄檔 URI。

            logging.info("Logged into ECR")

            logging.info("Pushing bento image to ECR")

            os.system(
                f"docker push 136566696263.dkr.ecr.us-east-1.amazonaws.com/{self.model_pusher_config.bentoml_ecr_image}:latest"
            )
            # 推送映像
            ## 136566696263.dkr.ecr.us-east-1.amazonaws.com 來自 AWS -- ECR 所建立的 repository

            logging.info("Pushed bento image to ECR")

            logging.info(
                "Exited build_and_push_bento_image method of ModelPusher class"
            )

        except Exception as e:
            raise XRayException(e, sys)
        


    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This method initiates model pusher.

        Output      :   Model pusher artifact
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            self.build_and_push_bento_image() # 建立與推送 映像 至 AWS ECR

            model_pusher_artifact = ModelPusherArtifact(
                bentoml_model_name=self.model_pusher_config.bentoml_model_name, # "xray_model"
                bentoml_service_name=self.model_pusher_config.bentoml_service_name, # "xray_service"
            )

            logging.info("Exited the initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise XRayException(e, sys)
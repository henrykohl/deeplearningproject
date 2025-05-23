import sys

from Xray.cloud_storage.s3_operation import S3Operation
from Xray.constant.training_pipeline import *
from Xray.entity.artifact_entity import DataIngestionArtifact
from Xray.entity.config_entity import DataIngestionConfig
from Xray.exception import XRayException
from Xray.logger import logging

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config 
        # (the configuration of component)
        # we are passing this data injection config from where we are passing you will get to know in some time
        # actually we are passing from the training once we will run the training now 
        # so automatically this data injection will call and we are passing the configuration 
        # so we get the entire configuration over here

        self.s3 = S3Operation()
        # (the operation)

    def get_data_from_s3(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method of Data ingestion class")

            self.s3.sync_folder_from_s3(
                folder=self.data_ingestion_config.data_path,   # 下載資料的存放路徑
                bucket_name=self.data_ingestion_config.bucket_name, # bucket 名稱
                bucket_folder_name=self.data_ingestion_config.s3_data_folder, # bucket中的資料夾名
            )

            logging.info("Exited the get_data_from_s3 method of Data ingestion class")

        except Exception as e:
            raise XRayException(e, sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info(
            "Entered the initiate_data_ingestion method of Data ingestion class"
        )

        try:
            self.get_data_from_s3() # 從 s3 獲取資料

            data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path,
            )

            logging.info(
                "Exited the initiate_data_ingestion method of Data ingestion class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise XRayException(e, sys)
import sys
from Xray.components.data_ingestion import DataIngestion           # Lec 2 
# from Xray.entity.artifact_entity import DataIngestionArtifact      # Lec 2 (old)
# from Xray.entity.config_entity import DataIngestionConfig          # Lec 2 (old)
from Xray.components.data_transformation import DataTransformation # Lec 3
from Xray.components.model_training import ModelTrainer            # Lec 4
from Xray.components.model_evaluation import ModelEvaluation       # Lec 5
# from Xray.components.model_pusher import ModelPusher
from Xray.exception import XRayException
from Xray.logger import logging

from Xray.entity.artifact_entity import (
    DataIngestionArtifact,                                         # Lec 2 (new)
    DataTransformationArtifact,                                    # Lec 3
    ModelTrainerArtifact,                                          # Lec 4
    ModelEvaluationArtifact,                                       # Lec 5
    # ModelPusherArtifact
    )

from Xray.entity.config_entity import (
    DataIngestionConfig,                                           # Lec 2 (new)
    DataTransformationConfig,                                      # Lec 3
    ModelTrainerConfig,                                            # Lec 4
    ModelEvaluationConfig,                                         # Lec 5
    # ModelPusherConfig
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()             # Lec 2
        self.data_transformation_config = DataTransformationConfig()   # Lec 3
        self.model_trainer_config = ModelTrainerConfig()               # Lec 4
        self.model_evaluation_config=ModelEvaluationConfig()           # Lec 5
        # self.model_pusher_config = ModelPusherConfig()
        
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """_summary_ : 用於 Lec 2

        Raises:
            XRayException: 略

        Returns:
            DataIngestionArtifact: 略
        """
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

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        """_summary_ : 用於 Lec 3

        Args:
            data_ingestion_artifact (DataIngestionArtifact): _description_

        Raises:
            XRayException: _description_

        Returns:
            DataTransformationArtifact: _description_
        """
        logging.info("Entered the start_data_transformation method of TrainPipeline class")

        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
            )

            ## 包含四個單元 -- training DataLoader、test DataLoader、training data 的轉型模式、test data 的轉型模式
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )   

            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        logging.info("Entered the start_model_trainer method of TrainPipeline class")

        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info("Exited the start_model_trainer method of TrainPipeline class")

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)

    def start_model_evaluation(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelEvaluationArtifact:
        logging.info("Entered the start_model_evaluation method of TrainPipeline class")

        try:
            model_evaluation = ModelEvaluation(
                data_transformation_artifact=data_transformation_artifact, # 來自 Lec 3
                model_trainer_artifact=model_trainer_artifact, # 來自 Lec 4
                model_evaluation_config=self.model_evaluation_config, 
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            logging.info(
                "Exited the start_model_evaluation method of TrainPipeline class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)
    
    # def start_model_pusher(self) -> ModelPusherArtifact:
    #     logging.info("Entered the start_model_pusher method of TrainPipeline class")

    #     try:
    #         model_pusher = ModelPusher(model_pusher_config=self.model_pusher_config)

    #         model_pusher_artifact = model_pusher.initiate_model_pusher()

    #         logging.info("Exited the start_model_pusher method of TrainPipeline class")

    #         return model_pusher_artifact

    #     except Exception as e:
    #         raise XRayException(e, sys)

    def run_pipeline(self) -> None:
        logging.info("Entered the run_pipeline method of TrainPipeline class")

        try:
            # Lec 2
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()      
            
            # Lec 3
            data_transformation_artifact: DataTransformationArtifact =  self.start_data_transformation( 
                data_ingestion_artifact=data_ingestion_artifact ## 來自 Lec 2
            )                     
                
            # Lec 4
            model_trainer_artifact: ModelTrainerArtifact = self.start_model_trainer(         
                data_transformation_artifact=data_transformation_artifact ## 來自 Lec 3
            )
            
            # Lec 5
            model_evaluation_artifact: ModelEvaluationArtifact = self.start_model_evaluation(
                model_trainer_artifact=model_trainer_artifact, ## 來自 Lec 4
                data_transformation_artifact=data_transformation_artifact, ## 來自 Lec 3
            )
            
            # Lec 6
            # model_pusher_artifact = self.start_model_pusher()

            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise XRayException(e, sys)

if __name__ == "__main__":
  train_pipeline=TrainPipeline()

  train_pipeline.start_data_ingestion()
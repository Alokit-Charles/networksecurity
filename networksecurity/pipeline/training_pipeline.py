import os
import sys

from networksecurity.logging.logger import logging 
from networksecurity.exception.exception import NetworkSecurityException

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTranformationConfig,
    ModelTrainerConfig
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from networksecurity.constants.training_pipeline import TRAINING_BUCKET_NAME
from networksecurity.cloud.s3_syncer import S3Sync


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Inside start_data_ingestion")
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Started Data Ingestion.")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifact : DataIngestionArtifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed. Artifact : {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_validation(self, data_ingestion_artifact : DataIngestionArtifact) -> DataValidationArtifact:
        try:
            self.data_validation_config = DataValidationConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Start Data Validation.")
            data_validation  = DataValidation(data_ingestion_artifact = data_ingestion_artifact, data_validation_config = self.data_validation_config)
            data_validation_artifact : DataValidationArtifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation Completed. Artifact : {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_data_transformation(self, data_validation_artifact : DataValidationArtifact) -> DataTransformationArtifact:
        try:
            self.data_transformation_config = DataTranformationConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Start Data Transformation.")
            data_transformation = DataTransformation(data_validation_artifact = data_validation_artifact, data_transformation_config= self.data_transformation_config)
            data_transformation_artifact : DataTransformationArtifact = data_transformation.inititate_data_transformation()
            logging.info(f"Data Transformation Completed. Artifact : {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def start_model_trainer(self, data_transformation_artifact : DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Start Model Trainer.")
            model_trainer = ModelTrainer(data_transformation_artifact = data_transformation_artifact,model_trainer_config = self.model_trainer_config)
            model_trainer_artifact : ModelTrainerArtifact = model_trainer.inititate_model_trainer()
            logging.info(f"Model Trainer Completed. Artifact : {model_trainer_artifact}")
            return model_trainer_artifact 
        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.artifact_dir, aws_bucket_url = aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def sync_saved_model_dir_to_s3(self):
            try:
                aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/finalized_model/{self.training_pipeline_config.timestamp}"
                self.s3_sync.sync_folder_to_s3(folder = self.training_pipeline_config.model_dir, aws_bucket_url = aws_bucket_url)
            except Exception as e:
                raise NetworkSecurityException(e, sys)
            
    def run_pipeline(self):
        try:
            logging.info("Inside run_pipeline of Training Pipeline.")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact = data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact = data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact = data_transformation_artifact)
            logging.info("Exited run_pipeline of Training Pipeline.")

            logging.info("Syncing Files To AWS")
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
            logging.info("Files Synced to AWS")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
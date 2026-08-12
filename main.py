from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, \
      DataValidationConfig, DataTranformationConfig, ModelTrainerConfig


import sys

if __name__ == '__main__':
    try :
        ## Data Ingestion
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        
        logging.info("Initiate Data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed.")
        print(data_ingestion_artifact)


        ## Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        
        logging.info("Inititate Data Validation.")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Complete.")
        print(data_validation_artifact)

        ## Transformation
        logging.info("Initiate Data Transformation.")
        data_transformation_config = DataTranformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.inititate_data_transformation()
        logging.info("Data Transformation Completed.")
        print(data_transformation_artifact)

        ## Model Training
        logging.info("Initiate Model Training.")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config = model_trainer_config, data_transformation_artifact= data_transformation_artifact)
        model_trainer_artifact = model_trainer.inititate_model_trainer()
        logging.info("Model Training Completed. Model training artifact created.")
        print(data_transformation_artifact)
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)
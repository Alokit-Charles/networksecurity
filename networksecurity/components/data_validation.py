from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from scipy.stats import ks_2samp
import pandas as pd
import os, sys

class DataValidation:
    def __init__(self, data_ingestion_artifact : DataIngestionArtifact,
                  data_validation_config: DataValidationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_number_of_columns(self, dataframe: pd.DataFrame, file_path: str)-> bool:
        try:
            number_of_columns = len(self._schema_config['columns'])
            logging.info(f"Required number of columns : {number_of_columns}")
            logging.info(f"DataFrame {file_path} has {len(dataframe.columns)} columns.")

            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def validate_numerical_column(self, dataframe: pd.DataFrame, file_path : str = "<file path>")->bool:
        try:
            other_cols = len(dataframe.select_dtypes(exclude= 'number').columns)
            logging.info(f"Numeric columns found in {file_path}: {list(dataframe.select_dtypes(include= 'number').columns)}")
            return other_cols == 0
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold = 0.05)-> bool:
        ## 95% confidence interval
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_distribution = ks_2samp(d1, d2)
                if threshold <= is_same_distribution.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({
                    column : {"p_value" : float(is_same_distribution.pvalue),
                              "drift_status" : is_found}
                }) 
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            # create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok= True)
            write_yaml_file(drift_report_file_path, report)
            return status
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def initiate_data_validation(self, )-> DataValidationArtifact:
        try:
            trian_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the data
            train_dataframe = DataValidation.read_data(trian_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            ## validate number of columns
            status = self.validate_number_of_columns(dataframe= train_dataframe, file_path = trian_file_path)
            if not status:
                error_message = f"Train dataframe does not contain all columns/ missing columns.\n"
            status = self.validate_number_of_columns(dataframe= test_dataframe, file_path = test_file_path)
            if not status:
                error_message = f"Test dataframe does not contain all columns/ missing columns.\n"
            
            ## validate numeric columns
            status = self.validate_numerical_column(dataframe= train_dataframe, file_path= trian_file_path)
            if not status:
                error_message = f"Train dataframe contains non numerical columns.\n"
            status = self.validate_numerical_column(dataframe= test_dataframe, file_path = test_file_path)
            if not status:
                error_message = f"Train dataframe contains non numerical columns.\n"
        
            ## check datadrift

            status = self.detect_dataset_drift(base_df= train_dataframe, current_df= test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok= True)
            
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index= False
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index= False
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status= status, 
                valid_train_file_path  = self.data_validation_config.valid_train_file_path,
                invalid_train_file_path= None,
                valid_test_file_path= self.data_validation_config.valid_test_file_path,
                invalid_test_file_path= None,
                drift_report_file_path= self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys, error_message = error_message)
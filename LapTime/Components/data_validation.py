import os
import sys
import pandas as pd
import numpy as np
from laptime.entity.config_entity import DataValidationConfig
from laptime.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from laptime.exception.exception import LapTimeException   
from laptime.logging.logger import logging
from laptime.utils.main_utils.utils import read_yaml_file,write_yaml_file
from laptime.constant.constants import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self.__schema_conifg=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise LapTimeException(e,sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise LapTimeException(e,sys)
    
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self.__schema_conifg["numerical_columns"])+len(self.__schema_conifg["categorical_columns"])+len(self.__schema_conifg["binary_columns"])
            if number_of_columns==dataframe.shape[1]:
                logging.info(f"Number of columns in dataframe {dataframe.shape[1]} matches with schema {number_of_columns}")
                return True
            else:
                logging.info(f"Number of columns in dataframe {dataframe.shape[1]} does not match with schema {number_of_columns}")
                return False
        except Exception as e:
            raise LapTimeException(e,sys)
        
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:                    
                    is_found=True
                    status=False
            report.update({column:{"p_value":float(is_same_dist.pvalue),"drift_status":is_found}})
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
        except Exception as e:
                raise LapTimeException(e,sys)
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            logging.info("Reading base dataframe")
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)

            logging.info("Validating number of columns in training dataset")
            status=self.validate_number_of_columns(train_dataframe)
            if not status:
                raise LapTimeException(f"Number of columns in training dataset does not match with schema")
            status=self.validate_number_of_columns(test_dataframe)
            if not status:
                raise LapTimeException(f"Number of columns in test dataset does not match with schema")
            logging.info("Detecting dataset drift")

            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path-=os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.validated_train_file_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.validated_test_file_path),exist_ok=True)

            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,index=False)

            datavalidationartifact=DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return datavalidationartifact
        except Exception as e:
            raise LapTimeException(e,sys)
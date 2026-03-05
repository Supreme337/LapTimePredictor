import os 
import sys 
import pandas as pd 
import numpy as np 
from typing import List
from laptime.entity.config_entity import DataTransformationConfig
from laptime.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from laptime.constant.constants import TARGET_COLUMN,COLUMNS_TO_DROP,TIRE_TYPE_COLUMNS,NA_COLUMNS
from laptime.logging.logger import logging
from laptime.exception.exception import LapTimeException
from laptime.utils.main_utils.utils import save_numpy_array_data,save_object
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact):
        self.data_transformation_config=data_transformation_config
        self.data_validation_artifact=data_validation_artifact

    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        df=pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}, shape={df.shape}")
        return df
    
    def feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        df['TotalLaps']=df.groupby(['Year','Round'])['LapNumber'].transform('max')
        df['LapPct']=df['LapNumber']/df['TotalLaps']
        df['StintLap']=(df.groupby(['Year','Round','driverId_x','Stint']).cumcount()+1)
        df['IsPitLap']=df['PitInTime'].notna().astype(int)
        df['HasPitOut']=df['PitOutTime'].notna().astype(int)
        df=df.drop(columns=['TotalLaps','LapNumber'])
        return df
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            train_df=self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=self.read_data(self.data_validation_artifact.valid_test_file_path)
        
            train_df=self.feature_engineering(train_df)
            test_df=self.feature_engineering(test_df)

            x_train=train_df.drop(columns=[TARGET_COLUMN]+COLUMNS_TO_DROP)
            y_train=train_df[TARGET_COLUMN]

            x_test=test_df.drop(columns=[TARGET_COLUMN]+COLUMNS_TO_DROP)
            y_test=test_df[TARGET_COLUMN]

            bool_cols=x_train.select_dtypes(include=['bool']).columns
            x_train[bool_cols]=x_train[bool_cols].astype(int)
            x_test[bool_cols]=x_test[bool_cols].astype(int)

            numeric_cols=x_train.columns.tolist()
            numeric_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median"))])
            preprocessor=ColumnTransformer([("num",numeric_pipeline,numeric_cols)])
            x_train_transformed=preprocessor.fit_transform(x_train)
            x_test_transformed=preprocessor.transform(x_test)

            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=train_df)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path,array=test_df)
            save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor)


            data_transformation_artifact=DataTransformationArtifact(transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                                                                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                                                                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path)
            logging.info(f"Data transformation completed successfully, artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise LapTimeException(e,sys)

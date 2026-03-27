import os 
import sys 
import pandas as pd 
import numpy as np 
from typing import List
from laptime.entity.config_entity import DataTransformationConfig
from laptime.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from laptime.entity.config_entity import ModelTrainerConfig
from laptime.constant.constants import TARGET_COLUMN,COLUMNS_TO_DROP,TIRE_TYPE_COLUMNS,NA_COLUMNS
from laptime.logging.logger import logging
from laptime.exception.exception import LapTimeException
from laptime.utils.main_utils.utils import save_numpy_array_data,save_object
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,data_validation_artifact:DataValidationArtifact,model_trainer_config:ModelTrainerConfig):
        self.data_transformation_config=data_transformation_config
        self.data_validation_artifact=data_validation_artifact
        self.model_trainer_config=model_trainer_config

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
        df=df.drop(columns=['TotalLaps','LapNumber','PitInTime','PitOutTime','country','driverId_x'])
        return df
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            train_df=self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=self.read_data(self.data_validation_artifact.valid_test_file_path)
        
            train_df=self.feature_engineering(train_df)
            test_df=self.feature_engineering(test_df)

            x_train=train_df.drop(columns=[TARGET_COLUMN]+COLUMNS_TO_DROP,axis=1)
            y_train=train_df[TARGET_COLUMN]

            x_test=test_df.drop(columns=[TARGET_COLUMN]+COLUMNS_TO_DROP,axis=1)
            y_test=test_df[TARGET_COLUMN]

            categorical_cols=['Driver']

            bool_cols=x_train.select_dtypes(include=['bool']).columns
            x_train[bool_cols]=x_train[bool_cols].astype(int)
            x_test[bool_cols]=x_test[bool_cols].astype(int)

            numeric_cols=[col for col in x_train.columns if col not in categorical_cols]
            numeric_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median"))])

            categorical_pipeline=Pipeline([
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))])

            preprocessor=ColumnTransformer([
                ("num",numeric_pipeline,numeric_cols),
                ("cat",categorical_pipeline,categorical_cols)],remainder='drop',verbose_feature_names_out=False)
            
            x_train_transformed=preprocessor.fit_transform(x_train)
            x_test_transformed=preprocessor.transform(x_test)

            final_model_path=os.path.join(os.getcwd(),"final_model")
            os.makedirs(final_model_path,exist_ok=True)

            encoder=preprocessor.named_transformers_["cat"].named_steps["encoder"]
            driver_mapping=dict(zip(encoder.categories_[0],range(len(encoder.categories_[0]))))
            mapping_path=os.path.join(final_model_path,"driver_mapping.pkl")
            save_object(mapping_path, driver_mapping)
            logging.info(f"Driver mapping saved at {mapping_path}")

            feature_names=preprocessor.get_feature_names_out()
            save_object(os.path.join(final_model_path,"feature_names.pkl"),feature_names)
            logging.info(f"Feature names saved at {os.path.join(final_model_path,'feature_names.pkl')}")

            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=x_train_transformed)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path,array=x_test_transformed)
            save_numpy_array_data(file_path=self.data_transformation_config.y_train_file_path,array=y_train)
            save_numpy_array_data(file_path=self.data_transformation_config.y_test_file_path,array=y_test)
            save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor)

            data_transformation_artifact=DataTransformationArtifact(transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                                                                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                                                                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                                                                    y_train_file_path=self.data_transformation_config.y_train_file_path,
                                                                    y_test_file_path=self.data_transformation_config.y_test_file_path,
                                                                    feature_names_file_path=os.path.join(final_model_path,"feature_names.pkl"))
            logging.info(f"Data transformation completed successfully, artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise LapTimeException(e,sys)

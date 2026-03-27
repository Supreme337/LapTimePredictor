import os
import sys
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv
from laptime.logging.logger import logging
from laptime.exception.exception import LapTimeException
from laptime.entity.config_entity import DataIngestionConfig
from laptime.entity.artifact_entity import DataIngestionArtifact
from typing import List
from sklearn.model_selection import train_test_split

load_dotenv()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise LapTimeException(e,sys)

    def export_collection_as_dataframe(self):
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]
            metadata=collection.find_one({"dataset_name":"f1_tire_strategy_dataset"})
            file_path=metadata["file_path"]
            df=pd.read_csv(file_path)
            if df.empty:
                raise ValueError(f"The MongoDB collection '{collection_name}' in database '{database_name}' is empty.")
            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"],axis=1,inplace=True)
            return df
        except Exception as e:
            raise LapTimeException(e,sys)
    
    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise LapTimeException(e,sys)
    
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Train Test Split Performed on Dataset")
            
            dir_path=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info("Exporting train and test file path")

            train_set.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)

        except Exception as e:
            raise LapTimeException(e,sys)
    
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            dataframe=self.export_collection_as_dataframe()
            self.export_data_into_feature_store(dataframe=dataframe)
            self.split_data_as_train_test(dataframe=dataframe)
            data_ingestion_artifact=DataIngestionArtifact(train_file_path=self.data_ingestion_config.train_file_path,
                                                        test_file_path=self.data_ingestion_config.test_file_path,
                                                        feature_store_file_path=self.data_ingestion_config.feature_store_file_path)
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise LapTimeException(e,sys)
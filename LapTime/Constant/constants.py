import pandas as pd
import numpy as np
import os
import sys

TARGET_COLUMN = "LapTimeSeconds"
ARTIFACT_DIR:str="Artifacts"
FILE_NAME:str="processed_f1_tire_strategy_dataset.csv"
PIPELINE_NAME:str="LapTime"

SCHEMA_FILE_PATH=os.path.join("data_schema","schema.yaml")

TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"

TRAINING_BUCKET_NAME="laptime"

MODEL_FILE_NAME:str="model.pkl"
PREPROCESSING_OBJECT_FILE_NAME:str="preprocessing.pkl"
SAVED_MODEL_DIR:str="final_models"

COLUMNS_TO_DROP=['lap','milliseconds','driverId_y','year','round','name_y','name_x','Driver','LapTime','location','alt'
                 ,'TrackName','race_Id','RaceProgress','Compound_UNKNOWN']
TIRE_TYPE_COLUMNS=['Compound_SOFT','Compound_MEDIUM','Compound_HARD','Compound_INTERMEDIATE','Compound_WET']
NA_COLUMNS=[('TireAge','Stint')]

"""Data Ingestion Constants"""
DATA_INGESTION_COLLECTION_NAME:str="dataset_metadata"
DATA_INGESTION_DATABASE_NAME:str="F1"
DATA_INGESTION_DIR_NAME:str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str="feature_store"
DATA_INGESTION_INGESTED_DIR:str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2

"""Data Validation Constants"""
DATA_VALIDATION_DIR_NAME:str="data_validation"
DATA_VALIDATION_VALID_DIR:str="valid"
DATA_VALIDATION_INVALID_DIR:str="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str="report.yaml"
PREPROCESSING_OBJECT_FILE_NAME:str="preprocessing.pkl" 

"""Data Transformation Constants"""
DATA_TRANSFORMATION_DIR:str="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str="transformed_object"
DATA_TRANSFORMATION_TRAIN_FILE_PATH:str='train.npy'
DATA_TRANSFORMATION_TEST_FILE_PATH:str="test.npy"
DATA_TRANSFORMATION_TRANSFORMED_OBJET_FILE_PATH:str="preprocessing.pkl"

"""Model Trainer Constants"""
MODEL_TRAINER_DIR_NAME="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR="trained_model"
MODEL_FILE_NAME="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE=0.7
MODEL_TRAINER_OVERFITTING_THRESHOLD=0.1
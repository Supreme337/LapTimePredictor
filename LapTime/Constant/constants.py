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

MODEL_FILE_NAME:str="model.pkl"
PREPROCESSING_OBJECT_FILE_NAME:str="preprocessing.pkl"
SAVED_MODEL_DIR:str="final_models"

"""Data Ingestion Constants"""
DATA_INGESTION_COLLECTION_NAME:str="LapTime"
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


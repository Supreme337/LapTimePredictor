import pymongo
import pandas as pd
import dotenv
dotenv.load_dotenv()
import os
import json 
from tqdm import tqdm
from datetime import datetime

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
DATABASE_NAME="F1"
COLLECTION_NAME="LapTime"
CSV_FILE_PATH="/Users/malik/Desktop/LapTime/dataset/f1_tire_strategy_dataset.csv"

client = pymongo.MongoClient(MONGO_DB_URL)
db=client[DATABASE_NAME]
collection=db[COLLECTION_NAME]
metadata_collection=db["dataset_metadata"]
file_path="dataset/f1_tire_strategy_dataset.csv"

df=pd.read_csv(file_path)  
metadata={
    "dataset_name": "f1_tire_strategy_dataset",
    "version": "v1",
    "file_path": file_path,
    "rows": df.shape[0],
    "features": df.shape[1],
    "target_column": "LapTimeSeconds",
    "created_at": datetime.utcnow()
}
metadata_collection.insert_one(metadata)
print("Metadata stored successfully")
import pymongo
import pandas as pd
import dotenv
dotenv.load_dotenv()
import os
import json 

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
DATABASE_NAME="F1"
COLLECTION_NAME="LapTime"
CSV_FILE_PATH="/Users/malik/Desktop/LapTime/dataset/f1_tire_strategy_dataset.csv"

client=pymongo.MongoClient(MONGO_DB_URL)
db=client[DATABASE_NAME]
collection=db[COLLECTION_NAME]

df=pd.read_csv(CSV_FILE_PATH)
records=json.loads(df.to_json(orient="records"))
collection.insert_many(records)
print("Successfully inserted data in MongoDB")
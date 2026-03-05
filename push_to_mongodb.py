import pymongo
import pandas as pd
import dotenv
dotenv.load_dotenv()
import os
import json 
from tqdm import tqdm

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
DATABASE_NAME="F1"
COLLECTION_NAME="LapTime"
CSV_FILE_PATH="/Users/malik/Desktop/LapTime/dataset/f1_tire_strategy_dataset.csv"

client=pymongo.MongoClient(MONGO_DB_URL,serverSelectionTimeoutMS=60000,socketTimeoutMS=60000,connectTimeoutMS=60000,maxPoolSize=50)
db=client[DATABASE_NAME]
collection=db[COLLECTION_NAME]

df=pd.read_csv(CSV_FILE_PATH)
records=json.loads(df.to_json(orient="records"))
batch_size = 1000 

for i in tqdm(range(0, len(records), batch_size)):
    batch = records[i:i + batch_size]
    collection.insert_many(batch)
print("Successfully inserted data in MongoDB")
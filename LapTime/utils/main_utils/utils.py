import yaml
import pickle
import os,sys
import numpy as np
from laptime.exception.exception import LapTimeException
from laptime.logging.logger import logging

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise LapTimeException(e,sys)

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise LapTimeException(e,sys)
    
def save_object(file_path:str,obj:object)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise LapTimeException(e,sys)

def load_object(file_path:str)->object:
    try:
        if not os.path.exist(file_path):
            raise Exception(f"the file:{file_path} doesn't exist")
        with open(file_path,"rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise LapTimeException(e,sys)
    
def save_numpy_array_data(file_path:str,array:np.ndarray):
    try:
        dir_path=os.path.dirname(file_path) 
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file:
            np.save(file,array)
    except Exception as e:
        raise LapTimeException(e,sys)
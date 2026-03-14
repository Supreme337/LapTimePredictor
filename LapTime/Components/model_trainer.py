import pandas as pd
import numpy as np
import os
import sys 
import tempfile
from urllib.parse import urlparse   
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor,plot_importance    
from laptime.exception.exception import LapTimeException
from laptime.constant.constants import TARGET_COLUMN
from laptime.logging.logger import logging
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from laptime.utils.ml_utils.regression_metric import get_regression_score
from laptime.entity.config_entity import ModelTrainerConfig
from laptime.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from laptime.utils.main_utils.utils import load_object,save_object
import mlflow
import dagshub
dagshub.init(repo_owner="Supreme337",repo_name="Laptime",mlflow=True)

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise LapTimeException(e,sys)
        
    def train_model(self,x_train,y_train,x_test,y_test):
        model=XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            random_state=42,
            sub_sample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            verbose=False)
        model.fit(x_train,y_train,eval_set=[(x_test,y_test)])
        plot_importance(model,max_num_features=20)
        plt.savefig("xgb_importance.png")
        return model
    

    def log_mlflow(self,model,train_metrics,test_metrics):
        with mlflow.start_run(run_name="XGBoost_Trainer"):
            mlflow.log_param("n_estimators",1000)
            mlflow.log_param("learning_rate",0.05)
            mlflow.log_param("max_depth",8)
            mlflow.log_param("sub_sample",0.8)
            mlflow.log_param("colsample_bytree",0.8)
            mlflow.log_metric("train_mae",train_metrics["mae"])
            mlflow.log_metric("train_mse",train_metrics["mse"])
            mlflow.log_metric("train_r2_score",train_metrics["r2_score"])
            mlflow.log_metric("train_rmse",train_metrics["rmse"])
            mlflow.log_metric("test_mae",test_metrics["mae"])
            mlflow.log_metric("test_mse",test_metrics["mse"])   
            mlflow.log_metric("test_r2_score",test_metrics["r2_score"])
            mlflow.log_metric("test_rmse",test_metrics["rmse"])
            mlflow.log_artifact("xgb_importance.png")

            tracking_uri=mlflow.get_tracking_uri()
            tracking_scheme=urlparse(tracking_uri).scheme
            if "dagshub" in tracking_uri.lower() or tracking_scheme in ["http","https"]:
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path=os.path.join(temp_dir,"model.pkl")
                    joblib.dump(model,model_path)
                    mlflow.log_artifact(model_path,"model")
                    logging.info(f"Model artifact logged to MLflow at {model_path}")    
            else:
                mlflow.sklearn.log_model(model,"model",registered_model_name="XGBoost_Trainer")
                logging.info("Model artifact logged to MLflow using mlflow.sklearn.log_model")

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Model Trainer Initiated")
            preprocessor=load_object(self.data_transformation_artifact.transformed_object_file_path)   

            train_df=np.load(self.data_transformation_artifact.transformed_train_file_path,allow_pickle=True)
            test_df=np.load(self.data_transformation_artifact.transformed_test_file_path,allow_pickle=True)

            x_train=train_df[:,:-1]
            y_train=train_df[:,-1]
            x_test=test_df[:,:-1]
            y_test=test_df[:,-1]
            
            x_train=preprocessor.transform(x_train)
            x_test=preprocessor.transform(x_test)

            logging.info("Training the model")
            model=self.train_model(x_train,y_train)
            logging.info("Evaluating the model")
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)
            train_metrics=get_regression_score(y_train,y_train_pred)
            test_metrics=get_regression_score(y_test,y_test_pred)
            logging.info("Model metrics calculated")
            self.log_mlflow(model,train_metrics,test_metrics)
            if (test_metrics["r2_score"]<self.model_trainer_config.overfitting_threshold):
                raise Exception(f"Model accuracy {test_metrics['r2_score']} is below the expected threshold {self.model_trainer_config.overfitting_threshold}")
            
            final_model_path=os.path.join(self.model_trainer_config.model_trainer_dir,"final_model")
            os.makedirs(final_model_path,exist_ok=True)
            model_file_path=os.path.join(final_model_path,"model.pkl")
            save_object(model_file_path,model)
            logging.info(f"Model saved at {model_file_path}")

            preprocessor_file_path=os.path.join(final_model_path,"preprocessor.pkl")
            save_object(preprocessor_file_path,preprocessor)
            logging.info(f"Preprocessor saved at {preprocessor_file_path}")

            model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=model_file_path,
                                                        preprocessor_object_file_path=preprocessor_file_path,
                                                        train_metrics=train_metrics,
                                                        test_metrics=test_metrics)
            return model_trainer_artifact
        except Exception as e:
            raise LapTimeException(e,sys)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from laptime.exception.exception import LapTimeException
from laptime.entity.artifact_entity import RegressionMetricArtifact
import numpy as np
import os 
import sys

def get_regression_score(y_true,y_pred)->RegressionMetricArtifact:
    try:
        model_mean_absolute_error=mean_absolute_error(y_true,y_pred)
        model_mean_squared_error=mean_squared_error(y_true,y_pred)
        model_r2_score=r2_score(y_true,y_pred)
        model_root_mean_squared_error=np.sqrt(model_mean_squared_error)
        return RegressionMetricArtifact(
            mean_absolute_error=model_mean_absolute_error,
            mean_squared_error=model_mean_squared_error,
            r2_score=model_r2_score,
            root_mean_squared_error=model_root_mean_squared_error)
    except Exception as e:
        raise LapTimeException(e,sys)
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str
    feature_store_file_path:str

@dataclass
class DataValidationArtifact:
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path:str
    invalid_test_file_path:str
    drift_report_file_path:str
    validation_status:bool

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path:str
    transformed_test_file_path:str
    transformed_object_file_path:str
    y_train_file_path:str
    y_test_file_path:str
    feature_names_file_path:str
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str
    preprocessor_object_file_path:str
    train_metrics:float
    test_metrics:float

@dataclass
class RegressionMetricArtifact:
    mean_absolute_error:float
    mean_squared_error:float
    r2_score:float
    root_mean_squared_error:float
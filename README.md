## F1 LapTime Predictor

A Machine Learning-powered web application that predicts Formula 1 lap times using historical race data, driver performance metrics, and tyre types.

## Overview

The F1 Lap Time Predictor is an end-to-end machine learning project designed to analyze Formula 1 race data and predict lap times with high accuracy. The project demonstrates the complete ML lifecycle — from data ingestion and preprocessing to model training, evaluation, and deployment.
The application allows users to input race-related parameters and receive predicted lap times through an interactive web interface.

## Features
	•	 Historical Formula 1 data analysis
	•	 Automated data ingestion and preprocessing pipeline
	•	 Machine Learning model training and evaluation
	•	 Feature engineering using:
  	•	Stint information
  	•	Driver statistics
  	•	Race variables
  	•	Tire and lap information
	•	 High-performance prediction using XGBoost
	•	 Flask-based web application for deployment
	•	 Experiment tracking with DagsHub
	•	 Modular and scalable ML pipeline architecture

## Tech Stack

### **Languages & Frameworks**
	•	Python
	•	Flask

### **Machine Learning**
	•	Scikit-learn
	•	XGBoost
	•	Pandas
	•	NumPy

### **Visualization & Analysis**
	•	Matplotlib
	•	Seaborn

### **MLOps & Tracking**
	•	DagsHub

### **Deployment**
	•	Flask
	•	HTML/CSS
	•	Tailwind CSS
  	•	Render

## Machine Learning Pipeline

### **1. Data Ingestion**
	•	Collects and loads historical Formula 1 datasets
	•	Handles train-test splitting
	•	Stores processed datasets

### **2. Data Transformation**
	•	Cleans missing values
	•	Performs feature engineering
	•	Encodes categorical variables
	•	Standardizes numerical features

### **3. Data Validation**
    • Validates dataset structure and schema
    • Checks for missing or inconsistent values
    • Ensures correct data types for all features
    • Detects duplicate or corrupted records
    • Verifies required columns before model training
    • Prevents invalid data from entering the pipeline
    • Improves reliability and consistency of predictions

### **4. Model Training**
	•	Trains multiple regression models
	•	Evaluates performance metrics
	•	Selects the best-performing model

### **5. Prediction Pipeline**
	•	Accepts user inputs
	•	Processes input data
	•	Generates lap time predictions

## Model Performance

The project uses XGBoost Regressor for prediction due to its strong performance on structured/tabular datasets.

### **Evaluation Metrics**
	•	R² Score
	•	Mean Absolute Error (MAE)
	•	Root Mean Squared Error (RMSE)

## Web Application

The Flask web application provides a simple interface where users can:
	•	Enter race parameters
	•	Submit prediction requests
	•	View predicted lap times instantly

## Screenshots

<img width="1431" height="735" alt="Screenshot 2026-05-20 at 12 24 36 PM" src="https://github.com/user-attachments/assets/a1ecd8b2-c638-47af-b033-6d0820435b9f" />
<img width="1431" height="735" alt="Screenshot 2026-05-20 at 12 24 56 PM" src="https://github.com/user-attachments/assets/3d6c8c94-3bd4-4d58-a63e-a3713f71e917" />
<img width="1431" height="735" alt="Screenshot 2026-05-20 at 12 25 34 PM" src="https://github.com/user-attachments/assets/878be5b2-f1c2-4315-ad68-46ea3350a278" />
<img width="1431" height="735" alt="Screenshot 2026-05-20 at 12 25 56 PM" src="https://github.com/user-attachments/assets/c964a8da-f57e-4dee-8636-08734f4a7fe3" />

## License

This project is licensed under the MIT License.

## Author

Developed by Harsh Malik


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
#os.add_dll_directory("C://Users/Administrator/Documents/Data Science Projects/Demo-ML-Project/venv\DLLs")

from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    AdaBoostRegressor,
    GradientBoostingRegressor
    )

from sklearn.svm import SVR
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso
    )
from sklearn.model_selection import (
    RandomizedSearchCV, 
    train_test_split)

#from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

from dataclasses import dataclass
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('Artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainingConfig()
        
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        logging.info("split training and test data set")
        
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1]
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbour Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train,
                                               X_test=X_test, y_test=y_test, models= models)
        except:
            pass
    
    

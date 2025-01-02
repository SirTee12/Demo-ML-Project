import numpy as np
import pandas as pd
import sys
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
from src.exception import CustomException

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('Artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainingConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        logging.info("split training and test data set")
        
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1])
            
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
                                               X_test=X_test, y_test=y_test, 
                                               models= models)
            
            ## get the best model score
            best_model_score = max(sorted(model_report.values()))
            
            # get the best model name form dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best Model found on both training and test data")
            
            save_object(
                file_path=self.model_trainer.trained_model_file_path,
                obj= best_model
            )
            
            predicted  = best_model.predict(X_test)
            
            r2_model_score = r2_score(X_test, predicted)
            return r2_model_score
            
        except Exception as e:
            raise CustomException(e, sys)
        
    
    

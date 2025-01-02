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
s
#from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

from dataclasses import dataclass
from src.utils import save_object

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('Artifacts', 'model.pkl')
    



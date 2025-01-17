import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path = os.path.join('Artifacts', 'preprocessor.pkl')
    
class DataTransfromation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            numeric_col = ['writing_score', 'reading_score']
            category_col = [
                'gender', 'race_ethnicity',
                'parental_level_of_education', 'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('Scaler', StandardScaler(with_mean=False))
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('Encoder', OneHotEncoder()),
                    ('Scaler', StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f'Categorical columns: {category_col}')
            logging.info(f'Numerical column: {numeric_col}')
            
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_col),
                    ('cat_pipeline', cat_pipeline, category_col)
                ]
            )
            
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('read train and test data completed')
            logging.info('obtaining preprocessor object')
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_col_name = 'math_score'
            numeric_col = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_col_name], axis = 1)
            target_feature_train_df = train_df[target_col_name]
            
            input_feature_test_df = test_df.drop(columns=[target_col_name], axis = 1)
            target_feature_test_df = test_df[target_col_name]
            
            logging.info(
                f'applying preprocessing object on training and testing dataframe'
                )
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info('data transformation complete')
            logging.info(f'saved preprocessing object')
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
            )
        
        except Exception as e:
            raise CustomException(e, sys)
            
import sys
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path:str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    def make_pipeline(self):
        logging.info("Making the pipeline for Preprocessing")
        try:
            numerical_col = ['reading_score', 'writing_score']
            categorical_col = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            numerical_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='mean')),
                ('Scaling', StandardScaler())
            ])

            categorical_pipe = Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('Encoding', OneHotEncoder())
            ])

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipe', numerical_pipe, numerical_col),
                    ('categorical_pipe', categorical_pipe, categorical_col)
                ]
            )
            logging.info('Pipeline has created successfully!')

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def init_transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Training and Testing data has stored')
            preprocessor_obj = self.make_pipeline()

            depend_col = 'math_score'
            independ_train_col = train_df.drop(depend_col, axis = 1)
            depend_train_col = train_df[depend_col]

            independ_test_col = test_df.drop(depend_col, axis =1)
            depend_test_col = test_df[depend_col]

            processed_train_col = preprocessor_obj.fit_transform(independ_train_col)
            processed_test_col = preprocessor_obj.transform(independ_test_col)
            logging.info('Preprocessing has done on training and testing columns')

            logging.info('Joining processed independant columns and target column')
            train_arr = np.c_[processed_train_col, np.array(depend_train_col)]
            test_arr = np.c_[processed_test_col, np.array(depend_test_col)]

            logging.info('Creating pickle file of the object')
            save_obj(self.data_transform_config.preprocess_obj_file_path, preprocessor_obj)

            return (train_arr, test_arr, self.data_transform_config.preprocess_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)


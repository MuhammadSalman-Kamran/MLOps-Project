import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_obj, train_and_evaluate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

@dataclass
class ModelTrainConfig:
    trained_model_path:str = os.path.join('artifacts', 'model.pkl')

class ModelTraining:
    def __init__(self) -> None:
        self.model_train_config = ModelTrainConfig()
    
    def init_training(self, train_arr, test_arr):
        try:
            logging.info('Model Training phase is starting')
            logging.info('Splitting of training and testing to dependant and indepandant variables')
            x_train, y_train, x_test, y_test = (train_arr[:,:-1], train_arr[:,-1], test_arr[:, :-1], test_arr[:,-1])
            model = LinearRegression()
            
            

            logging.info('Receiving the model and model_score')
            score, model = train_and_evaluate(x_train, y_train, x_test, y_test, model)
            logging.info('Training part has completed successfully!')
            logging.info('Saving the model into directory')
            save_obj(self.model_train_config.trained_model_path, model)
            return score

        except Exception as e:
            raise CustomException(e, sys)
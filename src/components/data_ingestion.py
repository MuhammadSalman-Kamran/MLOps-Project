import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    sample_data_path:str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def ingestion_data(self):
        logging.info('Ingestion of the data is starting')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Data has stored in the DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.sample_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.sample_data_path, index=False, header=True)

            logging.info('Split of Training and Testing')
            training_data, testing_data = train_test_split(df, test_size=0.2, random_state=42)

            training_data.to_csv(self.ingestion_config.train_data_path, index = False, header= True)
            logging.info('Training file exported successfully')

            testing_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info('Testing File exported successfully')

            logging.info('Ingestion has completed successfully!')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    obj.ingestion_data()
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from src.utils import load_obj

class Prediction:
    def __init__(self) -> None:
        pass

    def prediction(self, input):
        model_file_path = 'artifacts/model.pkl'
        preprocessor_file_path = 'artifacts/preprocessor.pkl'
        model = load_obj(model_file_path)
        preprocessor = load_obj(preprocessor_file_path)
        processed_data = preprocessor.transform(input)
        prediction = model.predict(processed_data)
        return prediction

class CustomDataClass:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def data_as_df(self):
        return pd.DataFrame([[self.gender, self.race_ethnicity, self.parental_level_of_education,self.lunch,self.test_preparation_course,self.reading_score,self.writing_score]], columns = ['gender', 'race_ethnicity', 'parental_level_of_education','lunch','test_preparation_course','reading_score','writing_score'])
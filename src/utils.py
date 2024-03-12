import pickle
import os
from sklearn.metrics import r2_score

def save_obj(file_path, object):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pickle.dump(object, open(file_path, 'wb'))

def load_obj(path):
    return pickle.load(open(path, 'rb'))

def train_and_evaluate(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    model_score =  r2_score(y_test, prediction)
    return model_score, model
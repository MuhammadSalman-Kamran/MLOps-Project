import pickle
import os

def save_obj(file_path, object):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pickle.dump(object, open(file_path, 'wb'))
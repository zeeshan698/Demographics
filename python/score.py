import json
import sys
import joblib

from azureml.core.model import Model
import numpy as np

def init():

    global model
    model_path = Model.get_model_path('demodrug')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data']).astype('float32')
        result  =model.predict(np.array([data]))
        return result.tolist()
    except Exception as e:
        result = str(e)
        return result

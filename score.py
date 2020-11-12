
import json
import sys
import joblib

from azureml.core.model import Model
import numpy as np

def init():

    global path
    model_path = Model.get_model_path('demodrug')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        result  = model.predict(data)
        return result.tolist()
    except Exception as e:
        result = str(e)
        return error

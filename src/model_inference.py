import pickle
import math

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def convert_log_to_units(prediction):
    return int(math.exp(prediction))

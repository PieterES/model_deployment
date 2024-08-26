from flask import Flask, render_template, request
from src.data_prep import prepare_data
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('models/forecasting_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        form_data = {key: request.form[key] for key in request.form}
    data = prepare_data(form_data)
    print(data)
    prediction = model.predict(data)
    return render_template("index.html", prediction=prediction, form_data=data)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 13:34:47 2023

@author: aditidadariya
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    print(type(model))
    prediction = model.predict(features)
    
    if prediction == [0]:
        output = "Not Approved"
    elif prediction == [1]:
        output = "Approved"
    return render_template("index.html", prediction_text = "The prediction of term deposit is {}".format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    
    #For direct API calls through request
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    if prediction == [0]:
        output = "Not Approved"
    elif prediction == [1]:
        output = "Approved"
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    
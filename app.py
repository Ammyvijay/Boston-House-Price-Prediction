from django.shortcuts import render
import numpy as np
import pickle
import math
from flask import Flask,request,app,jsonify,url_for,render_template

app=Flask(__name__)

## Load the model
regmodel = pickle.load(open('boston_regression_model.pkl','rb'))
scamodel = pickle.load(open('boston_scaler_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    # print(np.array(list(data.values())).reshape(1,-1))
    new_data=scamodel.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(math.floor(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scamodel.transform(np.array(list(data)).reshape(1,-1))
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int("3000"),debug=True)


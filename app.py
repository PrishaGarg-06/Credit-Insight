from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

app = Flask(__name__)

from data_utils import X_train, y_train

model = pickle.load(open('dt.pkl', 'rb'))

calibrated_dt = CalibratedClassifierCV(model, method='sigmoid')
calibrated_dt.fit(X_train, y_train)

@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


@app.route('/predict', methods=['POST'])
def predict():
    data1 = float(request.form['a'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data6 = float(request.form['f'])
    data7 = float(request.form['g'])
    data8 = float(request.form['h'])
    data9 = float(request.form['i'])
    data10 = float(request.form['j'])
    data11 = float(request.form['k'])
    
    					 		
    df = pd.DataFrame({
        'Age': [data1],
        'Income': [data3],
        'Family': [data4],
        'CCAvg': [data5],
        'Education': [data6],
        'Mortgage': [data7],
        'Securities Account': [data8],
        'CD Account': [data9],
        'Online': [data10],
        'CreditCard': [data11]
    })
    
    pred = calibrated_dt.predict_proba(df)[0][1]    
    prediction_result = "Approved" if pred >= 0.5 else "Rejected"
    predictions = {'Prediction': prediction_result}
    print("Prediction value:", predictions['Prediction'])
    return render_template('final.html', data=predictions) 


if __name__ == "__main__":
    app.run(debug=False)
    

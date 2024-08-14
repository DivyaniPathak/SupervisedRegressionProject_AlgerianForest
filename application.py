from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

##import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_scaled_data=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_scaled_data)

        return render_template('home.html',output=result[0])

    else:
        return render_template('home.html')


def index():
    return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")


    '''
    GET and post , we will create a page
    1. via get [ what you get from app, very first  (a platform to enter value ]
    inside this page. all the fields will be there, once we enter this field, once we press the submit button
    
    2. via post [what app  post for you , at a later time (output or prediction) ] 
    after pressing submit button, post will come into picture
    it(app) should interect in the backend with our model  and give prediction.
    
    
    and in code, if we want to find out , if it is GET or POST, we need to write like this
    if request.method=="POST"
    
    and in code [post part], to get values from form entered by user, write like this
    request.form.get('Region')
    
    '''
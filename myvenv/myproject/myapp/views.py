from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib



# Create your views here.
def index(request):
    return render(request,'index.html')
# predict page
def predict(request):

    return render(request,'predictor.html')


def result(request):

    data=pd.read_csv(r"C:\Users\DELL\Desktop\Ml project\EMPLOYEES LEVEL PREDICTION update.csv")
    data=data.drop(["NAME"],axis=1)
    data=data.set_index("NUMBER")
    x=data.drop(["LEVEL OF AN EMPLOYEE"],axis=1)

    le_data=LabelEncoder()
    x["QUALIFICATION"]=le_data.fit_transform(x["QUALIFICATION"])
    x["SKILLS AND ABILITIES"]=le_data.fit_transform(x["SKILLS AND ABILITIES"])
    x["WORK PERFORMANCE"]=le_data.fit_transform(x["WORK PERFORMANCE"])
    x["STATUS DURING THE TRAINING PERIOD"]=le_data.fit_transform(x["STATUS DURING THE TRAINING PERIOD"])

    y=data["LEVEL OF AN EMPLOYEE"]

    scaler=MinMaxScaler()
    x=scaler.fit_transform(x)

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)

    model=tree.DecisionTreeClassifier()
    model.fit(xtrain,ytrain)
    model.score(xtest,ytest)

    joblib.dump(model,'joblib_model')
    new_model=joblib.load('joblib_model')

   
    v1 = request.GET['qualification']
    v2 = request.GET['abilities']
    v3 = request.GET['seniority']
    v4 = request.GET['performance']
    v5 = request.GET['months']
    v6 = request.GET['status']
    pred=model.predict([[v1,v2,v3,v4,v5,v6]])
    if pred[0]=='class A':
        pred='Class ' + ' ' + 'A'
    elif pred[0]=='class B':
        pred='Class ' + ' ' + 'B'
    elif pred[0]=='class C':
        pred='Class ' + ' ' + 'C'
    else:
        pred='Class ' + ' ' + 'D'




    return render(request,'predictor.html',{'result':pred})
   

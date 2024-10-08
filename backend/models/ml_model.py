import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


import joblib
import json
import sys

# Add these path definitions before your functions
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_joblib_gr')
CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'insurance.csv')
data = pd.read_csv(CSV_PATH)
data.head()
data.tail()
data.shape

print("Number of rows=",data.shape[0])
print("Nuumber of columns=",data.shape[1])

data.info()
data.isnull()
data.isnull().sum()
data.describe(include = 'all')
data['sex'].unique()


data['sex'] = data['sex'].map({'female':0,'male':1})
data.head()


data['smoker'] =  data['smoker'].map({'yes':1,'no':0})
data.head()

data['region'].unique()
data['region'] = data['region'].map({'southwest':1,'southeast':2, 'northwest':3, 'northeast':4})
data.head()

data.columns

X = data.drop(['charges'],axis=1)
y = data['charges']
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=42)
X
y_train


lr = LinearRegression()
lr.fit(X_train,y_train)
svm = SVR()
svm.fit(X_train,y_train)
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
gr = GradientBoostingRegressor()
gr.fit(X_train,y_train)


y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = rf.predict(X_test)
y_pred4 = gr.predict(X_test)

df1 = pd.DataFrame({'Actual':y_test,'lr':y_pred1,'svm':y_pred2,'rf':y_pred3,'gr':y_pred4})
df1

import matplotlib.pyplot as plt

plt.subplot(221)
plt.plot(df1['Actual'].iloc[0:11],label="Actual")
plt.plot(df1['lr'].iloc[0:11],label="lr")
plt.legend

plt.subplot(222)
plt.plot(df1['Actual'].iloc[0:11],label="Actual")
plt.plot(df1['svm'].iloc[0:11],label="svm")
plt.legend

plt.subplot(223)
plt.plot(df1['Actual'].iloc[0:11],label="Actual")
plt.plot(df1['rf'].iloc[0:11],label="rf")
plt.legend

plt.subplot(224)
plt.plot(df1['Actual'].iloc[0:11],label="Actual")
plt.plot(df1['gr'].iloc[0:11],label="gr")

plt.tight_layout()
plt.legend

from sklearn import metrics
score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)
print(score1,score2,score3,score4)

s1 = metrics.mean_absolute_error(y_test,y_pred1)
s2 = metrics.mean_absolute_error(y_test,y_pred2)
s3 = metrics.mean_absolute_error(y_test,y_pred3)
s4 = metrics.mean_absolute_error(y_test,y_pred4)
print(s1,s2,s3,s4)


data = {'age':25,
       'sex':1,'bmi':20,
       'children':1,
       'smoker':0,
       'region':1}
df = pd.DataFrame(data,index=[0])
df

new_pred = gr.predict(df)
print(new_pred)

gr = GradientBoostingRegressor()
gr.fit(X,y)

import joblib
joblib.dump(gr,'model_joblib_gr')

model = joblib.load('model_joblib_gr')
# model.predict(df)

def preprocess_input(data):
    """ Convert input dictionary to the required format for the model. """
    df = pd.DataFrame(data, index=[0])
    return df

def predict_insurance(data):
    """ Make a prediction using the preprocessed data. """
    input_data = preprocess_input(data)
    prediction = model.predict(input_data)
    return prediction[0]

# from tkinter import *
# import joblib

# def show_entry():
#     p1 = float(e1.get())
#     p2 = float(e2.get())
#     p3 = float(e3.get())
#     p4 = float(e4.get())
#     p5 = float(e5.get())
#     p6 = float(e6.get())
    
#     model = joblib.load('model_joblib_gr')
#     result = model.predict([[p1,p2,p3,p4,p5,p6]])
    
#     Label(master, text = "Insurance Cost").grid(row = 7)
#     Label(master, text=result).grid(row = 8)
    


# master  =Tk()
# master.title("Insurance Cost Prediction")
# label = Label(master,text = "Insurance Cost Prediction",bg = "black",fg = "white").grid(row=0,columnspan=2)

# Label(master,text = "Enter Your Age").grid(row=1)
# Label(master,text = "Male or Female [1/0]").grid(row=2)
# Label(master,text = "Enter Your BMI Value").grid(row=3)
# Label(master,text = "Enter Your Number of Children").grid(row=4)
# Label(master,text = "Smoker Yes/No [1/0]").grid(row=5)
# Label(master,text = "Region [1-4]").grid(row=6)

# e1 = Entry(master)
# e2 = Entry(master)
# e3 = Entry(master)
# e4 = Entry(master)
# e5 = Entry(master)
# e6 = Entry(master)

# e1.grid(row = 1,column = 1)
# e2.grid(row = 2,column = 1)
# e3.grid(row = 3,column = 1)
# e4.grid(row = 4,column = 1)
# e5.grid(row = 5,column = 1)
# e6.grid(row = 6,column = 1)

# Button(master,text = "Predict",command=show_entry).grid()
# master.mainloop()


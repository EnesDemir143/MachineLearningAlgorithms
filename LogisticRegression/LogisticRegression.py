#!/usr/bin/env python
# coding: utf-8

# In[196]:


import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from LogisticRegression.config import DB_CONFIG



# In[188]:


def sigmoid(x):
    temp = 1/(1+np.exp(-x))
    return temp


# In[189]:


def cost_ofLogistic_Regression(x,y,w,b,lambda_):
    m,n=x.shape
    cost=0.
    epsilon=1e-15


    z = np.dot(x,w) +b
    f_wb = sigmoid(z)
    cost = -np.mean(y * np.log(f_wb+epsilon) + 
                (1-y) * np.log(1-f_wb+epsilon))


    reg_cost = np.sum(w**2)
    reg_cost = (reg_cost*lambda_)/(2*m)

    Total_cost = cost + reg_cost        
    return Total_cost


# In[190]:


def gradiant_Drivarative(x,y,w,b,lambda_):
    
    m,n = x.shape

    z = np.dot(x,w) + b
    f_wb = sigmoid(z)

    erorr = (f_wb - y) 
    dw = np.dot(erorr,x) + (lambda_ * w)
    dw = np.sum(dw)/m

    db = np.sum(erorr)/m

    return dw,db


# In[191]:


def gradient_Descent(x,y,w,b,iteration,learning_rate,lambda_):
    
    j_history = [] 

    for i in range(iteration):

        dw , db = gradiant_Drivarative(x,y,w,b,lambda_)

        w = w -learning_rate*dw
        b = b - learning_rate*db

        cost = cost_ofLogistic_Regression(x,y,w,b,lambda_)
        j_history.append(cost)

        if len(j_history) > 1 and abs(j_history[-2] - j_history[-1] < 1e-9):
            break
        
    return w,b,j_history



# In[198]:


def train_logistic_regression(): 

    engine = create_engine(f'mysql+mysqlconnector://{DB_CONFIG["user"]}:{DB_CONFIG["password"]}@{DB_CONFIG["host"]}/{DB_CONFIG["database"]}')

    # Veriyi okuma
    df = pd.read_sql('SELECT * FROM diabetes_data', engine)

    x=df.drop(columns="Outcome").to_numpy()
    y=df["Outcome"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    x_mean = np.mean(X_train, axis=0)
    std_x = np.std(X_train, axis=0)
    X_train = (X_train - x_mean) / std_x
    X_test = (X_test - x_mean) / std_x  


    w = np.random.randn(X_train[0].shape[0])*0.01
    b = 0.0

    learning_rate = 0.04
    lambda_= 0.9

    iterations = 1000

    final_w , final_b ,J_HİST= gradient_Descent(X_train,y_train,w,b,iterations,learning_rate,lambda_)

    return final_w,final_b,X_test,y_test


# In[ ]:


def predict(X,w,b):

    z = np.dot(X,w) + b
    y_prob =sigmoid(z)

    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred


# In[ ]:


def accuracy(y_true,y_pred):
    return np.sum(y_pred==y_true) / len(y_true)


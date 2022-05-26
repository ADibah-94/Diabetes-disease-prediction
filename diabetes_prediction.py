# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:36:03 2022

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#%% train.py

# EDA
# Step 1) Load Data

df = pd.read_csv('diabetes.csv')

# Step 2) Data Inspection

df.head()
df.info()
df.describe().T

# Step 3) Clean Data

ii_imputer = IterativeImputer()
df = ii_imputer.fit_transform(df)

pd.DataFrame(df).describe().T
pd.DataFrame(df).boxplot()

bool_series = pd.DataFrame(df).duplicated()
sum(bool_series==True)

df = pd.DataFrame(df).drop_duplicates()

# Step 4) Features Selection

X = df.iloc[:,0:8]
y = df.iloc[:,8]

# Step 5) Preprocessing of data
# MinMax Scaling

mms_scaler = MinMaxScaler()
scaled_data = mms_scaler.fit_transform(X)
SCALER_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','mms_scaler.pkl')
pickle.dump(mms_scaler, open(SCALER_SAVE_PATH,'wb'))

# To save the maximum and minimum value

ohe_scaler = OneHotEncoder(sparse=False)
one_hot_scaler = ohe_scaler.fit_transform(np.expand_dims(y, axis=-1))
OHE_SCALER_SAVE_PATH = os.path.join(os.getcwd(),'saved_model','ohe_scaler.pkl')
pickle.dump(ohe_scaler, open(OHE_SCALER_SAVE_PATH,'wb'))

X_train,X_test,y_train,y_test = train_test_split(scaled_data ,one_hot_scaler, test_size=0.3)
#%% Model Creation

model = Sequential()
model.add(Dense(128,activation=('relu'), input_shape=(8,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

#%% Compile & Model fitting

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(X_train,y_train, 
          epochs=50, 
          validation_data=(X_test,y_test))

#%% Model Evaluation

# Preallocation of of memory approach

predicted_advanced = np.empty([len(X_test), 2])
for index, test in enumerate(X_test):
    predicted_advanced[index,:] =  model.predict(np.expand_dims(test,axis=0))


#%% Model Analysis

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

#%% Model deployment

# Machine learning
# import pickle
# pickle.dump(model,open(SAVE_PATH))

# Deep learning
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model','model.h5')
model.save(MODEL_PATH)
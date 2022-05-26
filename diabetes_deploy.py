# -*- coding: utf-8 -*-
"""
Created on Fri May 13 11:27:23 2022

@author: HP

"""

import pickle
import os
from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') # GPU or CPU

for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#%% Paths
OHE_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model','ohe_scaler.pkl')
MMS_SCALER_PATH = os.path.join(os.getcwd(), 'saved_model','mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'saved_model','model.h5')

#%% Loading of setting or models
ohe_scaler = pickle.load(open(PATH),'rb')
mms_scaler = pickle.load(open(PATH),'rb')

# Machine learning 
# import pickle
# model = pickle.load(open(PATH))

# Deep learning 
model = load_model(PATH)
model.summary()

diabetes_chance = {0:'negative', 1:'positive'}
 

#%% Test Deployment 

patient_info = np.array([5,116,74,0,0,25.6,0.201,30])  # true label 0
patient_info_scaled = mms_scaler.transform(np.expand_dims(patient_info,axis=0))

outcome = model.predict(patient_info_scaled)
print(np.argmax(outcome))
print(diabetes_chance[np.argmax(outcome)])


# #%% Another approach
# if np.argmax(outcome) ==1:
#    outcome = [0,1]
#    print(ohe_scaler.inverse_transform(np.expand_dims([1,0],axis=0)))
# else:
#    outcome = [1,0]
#    print(ohe_scaler.inverse_transform(np.expand_dims([1,0],axis=0)))
    
#%% Build your app using streamlit

with st.form('Diabetes Preciction form'):
    st.write("Patient's info")
    pregnancies = int(st.number_input('Insert Times of Pregnancies'))
    glucose = st.number_input('Glucose')
    bp = st.number_input('Blood Pressure')
    skin_thick = st.number_input('Skin Thickness')
    insulin_level = st.number_input('Insulin Level')
    bmi = st.number_input('bmi')
    diabetes = st.number_input('diabetes')
    age = int(st.number_input('age'))

    submitted = st.form_submit_button('submit')
    
    if submitted == True:
        patient_info = np.array([pregnancies,glucose,bp,skin_thick,
                                 insulin_level,bmi])
        patient_info_scaled = mms.scaler.transform(np.expand_dims(patient_info,
                                                                  axis=0))
        
        outcome = model.predict(patient_info_scaled)
    
        st.write(diabetes_chance[np.argmax(outcome)])
    
        if np.argmax(outcome)==1:
            st.warning('You going to get diabetes soon, GOOD LUCK')
        else:
            st.snow()
            st.succes('YEAH, you are diabetic free')
            
            




























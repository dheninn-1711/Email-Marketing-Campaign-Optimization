# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:25:31 2025

@author: dheni
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/dheni/OneDrive/Desktop/EmailMarketingCampCodeFIles/trained_model1.sav', 'rb'))

def email_prediction(input_data):

  # changing the input_data to numpy array
  input_data_as_numpy_array = np.asarray(input_data)

  # reshape the array as we are predicting for one instance
  input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

  prediction = loaded_model.predict(input_data_reshaped)
  print(prediction)

  if (prediction[0] == 0):
    return 'The person did not open mail'
  else:
    return 'The person opened the mail'

def main():

    # giving a title
    st.title('Email Marketing Camapaign Web App')
    
    # Add an image (e.g., a logo)
    st.image("C:/Users/dheni/OneDrive/Pictures/Email.jpg", use_column_width=True)

    # getting the input data from the user

    Customer_Age = st.text_input('Customer Age')
    Emails_Opened = st.text_input('Emails Opened')
    Emails_Clicked = st.text_input('Emails Clicked')
    Purchase_History = st.text_input('Purchase History')
    Time_Spent_On_Website = st.text_input('Time Spent On Website')
    Days_Since_Last_Open = st.text_input('Days Since Last Open')
    Customer_Engagement_Score = st.text_input('Customer Engagement Score')
    Clicked_Previous_Emails = st.text_input('Clicked Previous Emails')
    Device_Type = st.text_input('Device Type')

    # code for Prediction
    Predicting = ''
    
    # creating a button for Prediction
    if st.button('Email Prediction Result'):
      Predicting = email_prediction([Customer_Age, Emails_Opened, Emails_Clicked, Purchase_History,
                                     Time_Spent_On_Website, Days_Since_Last_Open, Customer_Engagement_Score,
                                     Clicked_Previous_Emails, Device_Type])
    st.success(Predicting)

if __name__=='__main__':
  main()
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open("C:/Users/dheni/OneDrive/Desktop/EmailMarketingCampCodeFIles/trained_model1.sav", 'rb'))

input_data = (33,6,1,1069.8,4.9,23,60.5,0,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person did not open mail')
else:
  print('The person opened the mail')
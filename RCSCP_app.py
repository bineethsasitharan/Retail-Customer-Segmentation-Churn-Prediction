# -*- coding: utf-8 -*-

# RCSCP_app deployment

# importing required libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# loading pickle file
pickle_model = open("segment_and_churn_prediction.pkl","rb") # machine learning model pickle
classifier = pickle.load(pickle_model)

pickle_std = open("standardization.pkl","rb") # standardization pickle
stand = pickle.load(pickle_std)

# function to calculate RFM Score
def calculate_rfm_score(Recency, Frequency, Monetary):
    
    R_Score = pd.cut([Recency], bins=[-1, 60, 180, 900, 2000], labels=[4, 3, 2, 1]).astype('int64') 
    F_Score = pd.cut([Frequency], bins=[0, 10, 50, 100, 1000], labels=[1, 2, 4, 8]).astype('int64') 
    M_Score = pd.cut([Monetary], bins=[0, 10000, 100000, 800000, 10000000], labels=[1, 3, 6, 10]).astype('int64')
        
    RFM_Score = R_Score + F_Score + M_Score 
    
    return RFM_Score, R_Score, F_Score, M_Score

# function to do standardization
def standard(Recency,Frequency,Monetary,RFM_Score,R_Next_3Months):
    
    # getting the minimum and maximum values 
    R_mean = stand['R_mean']
    R_std = stand['R_std']
    
    F_mean = stand['F_mean']
    F_std = stand['F_std']
    
    M_mean = stand['M_mean']
    M_std = stand['M_std']
    
    RFM_mean = stand['RFM_mean']
    RFM_std = stand['RFM_std']
    
    R_N3M_mean = stand['R_N3M_mean']
    R_N3M_std = stand['R_N3M_std']
    
    # processing standardization
    Recency_stand = (Recency - R_mean)/ R_std
    Frequency_stand = (Frequency - F_mean)/ F_std
    Monetary_stand = (Monetary - M_mean)/ M_std
    RFM_Score_stand = (RFM_Score - RFM_mean)/ RFM_std
    R_Next_3Months_stand = (R_Next_3Months - R_N3M_mean)/ R_N3M_std
    
    return Recency_stand,Frequency_stand,Monetary_stand,RFM_Score_stand,R_Next_3Months_stand


# function to predict target variables
def predict(Recency_stand,Frequency_stand,Monetary_stand,RFM_Score_stand,R_Next_3Months_stand):

    X = np.hstack([Recency_stand, Frequency_stand, Monetary_stand, RFM_Score_stand, R_Next_3Months_stand])
    X = X.reshape(1, -1)
    prediction = classifier.predict(X)
    print(prediction)
    return prediction
       
# function main
def main():
    
    # front end interfac
    st.markdown(
        """
        <h1 style="color:white; text-align:center"> Retail Customer Segmentation and Churn Prediction </h1>
        <div style="background-color:gray; padding:10px">
        <h2 style="color:black; text-align:center"> Machine Learning RCSCP Web App </h2>
        </div>
        """,
        unsafe_allow_html=True)
    
    # getting input values
    CustomerID = st.number_input("CustomerID*",step=1)
    Recency = st.number_input("Recency*",step=1)
    Frequency = st.number_input("Frequency*",step=1)
    Monetary = st.number_input("Monetary*", step=1)
    R_Next_3Months = st.number_input("R_Next_3Months(No-0, Yes-1)* -->[Transaction in the next 3 months \
                                     following the RFM analysis]",step=1)
    result=""
     
    # warning info
    if Frequency == 0 or Monetary == 0 :
        st.warning("Please enter values for Recency, Frequency, Monetary and Return in 3months")
        if st.button("Calculate RFM Score and Predict"):
            pass
    
    # prediction
    else:
        if st.button("Calculate RFM Score and Predict"):
            
            # calling calculate RFM score function
            RFM_Score, R_Score, F_Score, M_Score = calculate_rfm_score(Recency, Frequency, Monetary)
            
            # calling standardization function
            Recency_stand,Frequency_stand,Monetary_stand,RFM_Score_stand,R_Next_3Months_stand = standard(Recency,Frequency,Monetary,RFM_Score,R_Next_3Months)
            
            # calling prediction function
            result = predict(Recency_stand,Frequency_stand,Monetary_stand,RFM_Score_stand,R_Next_3Months_stand)
            
            # printing scores
            st.text(f"Recency Score: {R_Score[0]}")
            st.text(f"Frequency Score: {F_Score[0]}") 
            st.text(f"Monetary Score: {M_Score[0]}")
            st.text(f"RFM Score: {RFM_Score[0]}")
            
            # printing predicted values
            st.success(f"CustomerID: {CustomerID}, Segment: {result[0,0]}, Churn: {result[0,1]}")

 
if __name__=='__main__':
    main()

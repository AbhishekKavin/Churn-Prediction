import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

with open('one_hot_encoder_geography.pkl','rb') as f:
    one_hot_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl','rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)


st.title("Customer Churn Prediction")

geography = st.selectbox("Select Geography", one_hot_encoder_geo.categories_[0])
gender = st.selectbox("Select Gender", label_encoder_gender.classes_)
age = st.slider("Select Age", 18, 92)
balance = st.number_input("Enter Balance")
credit_score = st.number_input("Enter Credit Score")
estimated_salary = st.number_input("Enter Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active Member", [0,1])

input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = one_hot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.DataFrame(input_data)
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
pred_proba = prediction[0][0]

if pred_proba > 0.5:
    st.write(f"The customer is likely to churn with a probability of {pred_proba:.2f}")
else:
    st.write(f"The customer is unlikely to churn with a probability of {pred_proba:.2f}")
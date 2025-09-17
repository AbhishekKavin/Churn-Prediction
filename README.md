# Customer Churn Prediction App

## Overview

This project is a machine learning application that predicts customer churn for a banking service. The application uses a neural network model built with TensorFlow/Keras to analyze customer data and predict whether a customer is likely to leave the bank.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Machine Learning Model**: Neural network with 2 hidden layers (64 and 32 neurons)
- **Data Preprocessing**: Automated handling of categorical variables and feature scaling
- **Real-time Predictions**: Instant churn probability calculations

## Project Structure
├── app.py # Streamlit web application
├── model.h5 # Trained neural network model
├── experiments.ipynb # Jupyter notebook for model development
├── predictions.ipynb # Jupyter notebook for testing predictions
├── scaler.pkl # Fitted StandardScaler for feature normalization
├── label_encoder_gender.pkl # Label encoder for Gender feature
├── one_hot_encoder_geography.pkl # One-hot encoder for Geography feature
└── Churn_Modelling.csv # Original dataset (not included in repo)

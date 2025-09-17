# Churn-Prediction

Overview
This project is a machine learning application that predicts customer churn for a banking service. The application uses a neural network model built with TensorFlow/Keras to analyze customer data and predict whether a customer is likely to leave the bank.

Features
Interactive Web Interface: Built with Streamlit for easy user interaction

Machine Learning Model: Neural network with 2 hidden layers (64 and 32 neurons)

Data Preprocessing: Automated handling of categorical variables and feature scaling

Real-time Predictions: Instant churn probability calculations

Project Structure
text
├── app.py                 # Streamlit web application
├── model.h5              # Trained neural network model
├── experiments.ipynb     # Jupyter notebook for model development
├── predictions.ipynb     # Jupyter notebook for testing predictions
├── scaler.pkl            # Fitted StandardScaler for feature normalization
├── label_encoder_gender.pkl      # Label encoder for Gender feature
├── one_hot_encoder_geography.pkl # One-hot encoder for Geography feature
└── Churn_Modelling.csv   # Original dataset (not included in repo)
Installation
Clone the repository:

bash
git clone <repository-url>
cd customer-churn-prediction
Create a virtual environment and install dependencies:

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Required packages (if requirements.txt is not available):

bash
pip install streamlit pandas numpy scikit-learn tensorflow
Usage
Run the Streamlit application:

bash
streamlit run app.py
Open your web browser and navigate to the local URL provided (typically http://localhost:8501)

Input customer data using the interactive form:

Select Geography (France, Germany, Spain)

Select Gender (Male, Female)

Adjust sliders for Age and Tenure

Enter numerical values for Balance, Credit Score, and Estimated Salary

Select product and membership options

View the prediction results showing churn probability

Model Details
Architecture
Input Layer: 12 features

Hidden Layer 1: 64 neurons with ReLU activation

Hidden Layer 2: 32 neurons with ReLU activation

Output Layer: 1 neuron with Sigmoid activation

Training
Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy

Early Stopping: Patience of 5 epochs

Validation Split: 20%

Data Preprocessing
Categorical Variables:

Gender: Label encoded (Female: 0, Male: 1)

Geography: One-hot encoded (France, Germany, Spain)

Numerical Variables: Standard scaled (mean=0, std=1)

Dropped Columns: RowNumber, CustomerId, Surname

File Descriptions
app.py: Main Streamlit application with user interface

experiments.ipynb: Jupyter notebook containing data exploration, preprocessing, and model training

predictions.ipynb: Jupyter notebook for testing the model with sample data

model.h5: Saved trained neural network model

scaler.pkl: Fitted StandardScaler for consistent feature scaling

label_encoder_gender.pkl: Fitted LabelEncoder for Gender column

one_hot_encoder_geography.pkl: Fitted OneHotEncoder for Geography column

Dataset
The model was trained on the "Churn_Modelling.csv" dataset containing:

10,000 customer records

14 features including demographic, financial, and behavioral data

Target variable: Exited (1 = churned, 0 = retained)

Performance
The model achieves approximately 87% accuracy on the validation set with early stopping preventing overfitting.

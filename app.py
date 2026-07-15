import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('wine_model.pkl')

st.title("🍷 Wine Quality Predictor")
st.write("Enter the wine properties below to predict its quality:")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", value=7.0)
volatile_acidity = st.number_input("Volatile Acidity", value=0.3)
citric_acid = st.number_input("Citric Acid", value=0.3)
residual_sugar = st.number_input("Residual Sugar", value=2.0)
chlorides = st.number_input("Chlorides", value=0.05)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=30.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=100.0)
density = st.number_input("Density", value=0.99)
pH = st.number_input("pH", value=3.2)
sulphates = st.number_input("Sulphates", value=0.5)
alcohol = st.number_input("Alcohol", value=10.0)
wine_type = st.selectbox("Wine Type", ["White", "Red"])

# Prediction button
if st.button("Predict Quality"):
    # 1. Create the base input data
    input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                                chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
                                density, pH, sulphates, alcohol, 1 if wine_type == "White" else 0]],
                              columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                                       'density', 'pH', 'sulphates', 'alcohol', 'type_white'])
    
    # 2. Align input with model's expected features
    # This automatically fixes the order and ensures only required columns are sent
    try:
        # Reorder to match model.feature_names_in_
        input_data = input_data[model.feature_names_in_]
        
        prediction = model.predict(input_data)
        st.success(f"The predicted quality of the wine is: {prediction[0]}")
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("The model expects these columns:", model.feature_names_in_)
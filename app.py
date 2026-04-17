import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("ML Model Deployment")

st.write("Upload a CSV file to get predictions")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Input Data")
    st.write(data)

    # Prediction
    if st.button("Predict"):
        try:
            predictions = model.predict(data)
            st.subheader("Predictions")
            st.write(predictions)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

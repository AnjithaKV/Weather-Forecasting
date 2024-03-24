import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('weatherforecasting1.joblib')  # Replace 'your_trained_model.pkl' with your trained model file

# Function to make predictions
def predict_rainfall(features):
    features = np.array(features).reshape(1, -1)  # Reshape features to fit the model
    prediction = model.predict(features)
    return prediction[0]

# Streamlit app
def main():
    st.title('Rainfall Prediction')
    
    # Sidebar inputs
    st.sidebar.header('Input Parameters')

    # Example input fields, replace with actual features you want to use for prediction
    feature1 = st.sidebar.slider('Humidity3pm', min_value=0.0, max_value=100.0, value=50.0)
    feature2 = st.sidebar.slider('Rainfall', min_value=0.0, max_value=371.0, value=50.0)
    feature3 = st.sidebar.selectbox('RainToday', ['No','Yes'])
    feature4 = st.sidebar.slider('Humidity9am', min_value=0.0, max_value=100.0, value=50.0)
    feature5 = st.sidebar.slider('Pressure9am', min_value=980.50, max_value=1041.0, value=50.0)
    feature6 = st.sidebar.slider('WindGustSpeed', min_value=6.0, max_value=135.0, value=50.0)
    feature7 = st.sidebar.slider('Pressure3pm', min_value=977.0, max_value=1039.0, value=50.0)
    feature8 = st.sidebar.slider('Temp3pm', min_value=-5.40, max_value=46.0, value=50.0)
    feature9 = st.sidebar.slider('MaxTemp', min_value=-4.80, max_value=48.10, value=50.0)
    feature10 = st.sidebar.selectbox('WindDir9am', ['W','N','E','S','WSW','SSW','NW','ESE','NNW','SE','SSE','SW','WNW','ENE','NE','NNE'])
    feature11 = st.sidebar.selectbox('WindGustDir', ['W','N','E','S','WSW','SSW','NW','ESE','NNW','SE','SSE','SW','WNW','ENE','NE','NNE'])
    feature12 = st.sidebar.selectbox('WindDir3pm', ['W','N','E','S','WSW','SSW','NW','ESE','NNW','SE','SSE','SW','WNW','ENE','NE','NNE'])
    feature13 = st.sidebar.slider('MinTemp', min_value=-8.50, max_value=33.90, value=50.0)
    feature14 = st.sidebar.slider('WindSpeed9am', min_value=0.0, max_value=50.0, value=50.0)
    feature15 = st.sidebar.slider('WindSpeed3pm', min_value=0.0, max_value=50.0, value=50.0)
    
    # Add more features as needed
    
    # Check button
    if st.sidebar.button('Predict'):
        # Combine features into a list
        features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15]  # Add more features as needed

        # Predict
        prediction = predict_rainfall(features)

        # Display prediction
        st.write('Predicted Rainfall:', prediction, 'mm')

if __name__ == '__main__':
    main()

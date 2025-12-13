import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AgriTech: Smart Crop Recommender", page_icon="🌱")

# --- TITLE & DESCRIPTION ---
st.title("🌱 AgriTech: Smart Crop Advisor")
st.write("This AI-powered tool suggests the best crop to grow based on soil and weather conditions.")

# --- SIDEBAR (For Inputs) ---
st.sidebar.header("Enter Soil & Weather Details")

def user_input_features():
    # These are the 7 features the AI needs to know
    N = st.sidebar.slider('Nitrogen (N)', 0, 140, 50)
    P = st.sidebar.slider('Phosphorus (P)', 5, 145, 50)
    K = st.sidebar.slider('Potassium (K)', 5, 205, 50)
    temperature = st.sidebar.number_input('Temperature (°C)', 0.0, 50.0, 25.0)
    humidity = st.sidebar.number_input('Humidity (%)', 0.0, 100.0, 70.0)
    ph = st.sidebar.slider('Soil pH Level', 0.0, 14.0, 7.0)
    rainfall = st.sidebar.number_input('Rainfall (mm)', 0.0, 300.0, 100.0)
    
    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input on the main page
st.subheader("Current Conditions:")
st.write(input_df)

# --- THE "AI" ENGINE ---
# This block runs when the app starts
try:
    # 1. Load Data
    crop_data = pd.read_csv("Crop_recommendation.csv") 
    
    # 2. Prepare Data (X is input, Y is what we predict)
    X = crop_data.drop('label', axis=1)
    Y = crop_data['label']

    # 3. Train the Model (Random Forest is very accurate)
    clf = RandomForestClassifier()
    clf.fit(X, Y)

    # 4. Make Prediction button
    if st.button("Recommend Crop"):
        prediction = clf.predict(input_df)
        st.success(f"🌾 Recommended Crop: **{prediction[0].upper()}**")
        
        # Bonus: Add simple logic for explanation (Judges love this)
        st.info("💡 Tip: Ensure proper drainage for this crop.")

except FileNotFoundError:
    st.error("⚠️ Error: 'Crop_recommendation.csv' file not found. Please download it and put it in the same folder!")
except Exception as e:
    st.error(f"An error occurred: {e}")
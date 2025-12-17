import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AnnDaata AI", page_icon="🌾")

# --- 2. TRANSLATIONS (Static Dictionary) ---
translations = {
    "English": {
        "title": "AnnDaata AI", "subtitle": "Smart Crop Advisor 🌾",
        "input_section": "Enter Field Details", "predict_button": "Recommend Crop",
        "result_text": "Best Crop to Plant:", "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
        "temp": "Temperature", "hum": "Humidity", "ph": "Soil pH", "rain": "Rainfall"
    },
    "Hindi": {
        "title": "अन्नदाता AI", "subtitle": "स्मार्ट फसल सलाहकार 🌾",
        "input_section": "खेत का विवरण दर्ज करें", "predict_button": "फसल का सुझाव दें",
        "result_text": "सुझाई गई फसल:", "N": "नाइट्रोजन", "P": "फॉस्फोरस", "K": "पोटेशियम",
        "temp": "तापमान", "hum": "नमी", "ph": "pH स्तर", "rain": "वर्षा"
    },
    "Punjabi": {
        "title": "ਅੰਨਦਾਤਾ AI", "subtitle": "ਫਸਲ ਸਲਾਹਕਾਰ 🌾",
        "input_section": "ਖੇਤੀ ਦਾ ਵੇਰਵਾ", "predict_button": "ਫਸਲ ਲੱਭੋ",
        "result_text": "ਵਧੀਆ ਫਸਲ:", "N": "ਨਾਈਟ੍ਰੋਜਨ", "P": "ਫਾਸਫੋਰਸ", "K": "ਪੋਟਾਸ਼ੀਅਮ",
        "temp": "ਤਾਪਮਾਨ", "hum": "ਨਮੀ", "ph": "pH ਪੱਧਰ", "rain": "ਮੀਂਹ"
    }
}

# --- 3. UI LAYOUT ---
lang_choice = st.sidebar.radio("Language", ["English", "Hindi", "Punjabi"])
t = translations[lang_choice]

st.title(t['title'])
st.header(t['subtitle'])

st.subheader(t['input_section'])
col1, col2 = st.columns(2)
with col1:
    N = st.slider(t['N'], 0, 140, 50)
    P = st.slider(t['P'], 5, 145, 50)
    K = st.slider(t['K'], 5, 205, 50)
    ph = st.slider(t['ph'], 0.0, 14.0, 7.0)
with col2:
    temperature = st.number_input(t['temp'], 0.0, 50.0, 25.0)
    humidity = st.number_input(t['hum'], 0.0, 100.0, 70.0)
    rainfall = st.number_input(t['rain'], 0.0, 300.0, 100.0)

# --- 4. PREDICTION LOGIC ---
if st.button(t['predict_button']):
    try:
        crop_data = pd.read_csv("Crop_recommendation.csv")
        X = crop_data.drop('label', axis=1)
        Y = crop_data['label']
        clf = RandomForestClassifier()
        clf.fit(X, Y)
        
        input_df = pd.DataFrame({'N': [N], 'P': [P], 'K': [K], 'temperature': [temperature], 'humidity': [humidity], 'ph': [ph], 'rainfall': [rainfall]})
        prediction = clf.predict(input_df)
        
        st.success(f"{t['result_text']} {prediction[0].upper()}")
        
    except FileNotFoundError:
        st.error("Error: Dataset not found.")













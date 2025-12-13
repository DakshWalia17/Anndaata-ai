import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AnnDaata AI", page_icon="🌾")

# --- TRANSLATION DICTIONARY ---
translations = {
    "English": {
        "title": "🌾 AnnDaata AI: Smart Crop Advisor",
        "desc": "This AI-powered tool suggests the best crop to grow based on soil and weather conditions.",
        "sidebar_title": "Enter Soil & Weather Details",
        "predict_button": "Recommend Crop",
        "result_text": "Recommended Crop:",
        "success_msg": "Ensure proper drainage for this crop.",
        "N": "Nitrogen",
        "P": "Phosphorus",
        "K": "Potassium",
        "temp": "Temperature (°C)",
        "hum": "Humidity (%)",
        "ph": "Soil pH Level",
        "rain": "Rainfall (mm)",
        "analysis": "📊 Soil Analysis",
        "chart_title": "Soil Nutrients Levels"
    },
    "Hindi": {
        "title": "🌾 अन्नदाता AI: स्मार्ट फसल सलाहकार",
        "desc": "यह एआई टूल मिट्टी और मौसम की स्थिति के आधार पर सबसे अच्छी फसल का सुझाव देता है।",
        "sidebar_title": "मिट्टी और मौसम का विवरण",
        "predict_button": "फसल का सुझाव दें",
        "result_text": "सुझाई गई फसल:",
        "success_msg": "इस फसल के लिए उचित जल निकासी सुनिश्चित करें।",
        "N": "नाइट्रोजन",
        "P": "फॉस्फोरस",
        "K": "पोटेशियम",
        "temp": "तापमान (°C)",
        "hum": "नमी (%)",
        "ph": "मिट्टी का pH स्तर",
        "rain": "वर्षा (mm)",
        "analysis": "📊 मिट्टी का विश्लेषण",
        "chart_title": "मिट्टी के पोषक तत्व"
    },
    "Punjabi": {
        "title": "🌾 ਅੰਨਦਾਤਾ AI: ਫਸਲ ਸਲਾਹਕਾਰ",
        "desc": "ਇਹ AI ਟੂਲ ਮਿੱਟੀ ਅਤੇ ਮੌਸਮ ਦੇ ਅਧਾਰ ਤੇ ਵਧੀਆ ਫਸਲ ਦਾ ਸੁਝਾਅ ਦਿੰਦਾ ਹੈ।",
        "sidebar_title": "ਮਿੱਟੀ ਅਤੇ ਮੌਸਮ ਦਾ ਵੇਰਵਾ",
        "predict_button": "ਫਸਲ ਦੀ ਸਿਫਾਰਸ਼ ਕਰੋ",
        "result_text": "ਸਿਫਾਰਸ਼ ਕੀਤੀ ਫਸਲ:",
        "success_msg": "ਇਸ ਫਸਲ ਲਈ ਉਚਿਤ ਪਾਣੀ ਦੀ ਨਿਕਾਸੀ ਯਕੀਨੀ ਬਣਾਓ।",
        "N": "ਨਾਈਟ੍ਰੋਜਨ",
        "P": "ਫਾਸਫੋਰਸ",
        "K": "ਪੋਟਾਸ਼ੀਅਮ",
        "temp": "ਤਾਪਮਾਨ (°C)",
        "hum": "ਨਮੀ (%)",
        "ph": "ਮਿੱਟੀ ਦਾ pH ਪੱਧਰ",
        "rain": "ਮੀਂਹ (mm)",
        "analysis": "📊 ਮਿੱਟੀ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ",
        "chart_title": "ਮਿੱਟੀ ਦੇ ਪੌਸ਼ਟਿਕ ਤੱਤ"
    }
}

# --- LANGUAGE SELECTOR ---
lang_choice = st.sidebar.radio("Language / भाषा / ਭਾਸ਼ਾ", ["English", "Hindi", "Punjabi"])
t = translations[lang_choice]

# --- MAIN APP UI ---
st.title(t['title'])
st.write(t['desc'])

st.sidebar.header(t['sidebar_title'])

def user_input_features():
    # Input sliders
    N = st.sidebar.slider(t['N'], 0, 140, 50)
    P = st.sidebar.slider(t['P'], 5, 145, 50)
    K = st.sidebar.slider(t['K'], 5, 205, 50)
    temperature = st.sidebar.number_input(t['temp'], 0.0, 50.0, 25.0)
    humidity = st.sidebar.number_input(t['hum'], 0.0, 100.0, 70.0)
    ph = st.sidebar.slider(t['ph'], 0.0, 14.0, 7.0)
    rainfall = st.sidebar.number_input(t['rain'], 0.0, 300.0, 100.0)
    
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

input_df = user_input_features()

# Display inputs
st.subheader(t['analysis'])
st.write(input_df)

# --- AI ENGINE ---
try:
    crop_data = pd.read_csv("Crop_recommendation.csv") 
    X = crop_data.drop('label', axis=1)
    Y = crop_data['label']
    clf = RandomForestClassifier()
    clf.fit(X, Y)

    if st.button(t['predict_button']):
        prediction = clf.predict(input_df)
        
        # 1. Show Text Result
        st.success(f"{t['result_text']} **{prediction[0].upper()}**")
        st.info(t['success_msg'])
        
        # 2. Show Visual Chart (Now with translated labels!)
        st.write("---") # A divider line
        st.subheader(t['chart_title'])
        
        # Creating a dynamic chart based on language
        chart_data = pd.DataFrame({
            'Nutrient': [t['N'], t['P'], t['K']],
            'Value': [input_df['N'][0], input_df['P'][0], input_df['K'][0]]
        })
        
        # Display the Bar Chart
        st.bar_chart(chart_data.set_index('Nutrient'))

except FileNotFoundError:
    st.error("⚠️ Error: 'Crop_recommendation.csv' file not found.")


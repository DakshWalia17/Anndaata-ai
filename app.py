import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AnnDaata AI", page_icon="🌾")

# --- CONFIGURE GENAI (THE CHATBOT) ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
except:
    st.error("⚠️ Google API Key not found. Please set it in Streamlit Secrets.")

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
        "chart_title": "Soil Nutrients Levels",
        "ai_advice": "🤖 Ask AI Agronomist",
        "ai_btn": "Get Farming Guide for",
        "ai_prompt": "Give me a practical farming guide for growing {} in India. Keep it short (5 bullet points). Language: English."
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
        "chart_title": "मिट्टी के पोषक तत्व",
        "ai_advice": "🤖 AI कृषि विशेषज्ञ से पूछें",
        "ai_btn": "के लिए खेती की गाइड प्राप्त करें",
        "ai_prompt": "मुझे भारत में {} उगाने के लिए एक व्यावहारिक खेती गाइड दें। इसे छोटा रखें (5 बुलेट पॉइंट)। भाषा: हिंदी।"
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
        "chart_title": "ਮਿੱਟੀ ਦੇ ਪੌਸ਼ਟਿਕ ਤੱਤ",
        "ai_advice": "🤖 AI ਖੇਤੀ ਮਾਹਰ ਤੋਂ ਪੁੱਛੋ",
        "ai_btn": "ਲਈ ਖੇਤੀ ਗਾਈਡ ਪ੍ਰਾਪਤ ਕਰੋ",
        "ai_prompt": "ਮੈਨੂੰ ਭਾਰਤ ਵਿੱਚ {} ਉਗਾਉਣ ਲਈ ਇੱਕ ਵਿਹਾਰਕ ਖੇਤੀ ਗਾਈਡ ਦਿਓ। ਇਸਨੂੰ ਛੋਟਾ ਰੱਖੋ (5 ਬਿੰਦੂ)। ਭਾਸ਼ਾ: ਪੰਜਾਬੀ।"
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
    N = st.sidebar.slider(t['N'], 0, 140, 50)
    P = st.sidebar.slider(t['P'], 5, 145, 50)
    K = st.sidebar.slider(t['K'], 5, 205, 50)
    temperature = st.sidebar.number_input(t['temp'], 0.0, 50.0, 25.0)
    humidity = st.sidebar.number_input(t['hum'], 0.0, 100.0, 70.0)
    ph = st.sidebar.slider(t['ph'], 0.0, 14.0, 7.0)
    rainfall = st.sidebar.number_input(t['rain'], 0.0, 300.0, 100.0)
    
    data = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
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

    # --- SESSION STATE LOGIC (THE FIX) ---
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    # When user clicks "Recommend", save the result in memory
    if st.button(t['predict_button']):
        prediction = clf.predict(input_df)
        st.session_state.prediction = prediction[0].upper()

    # If we have a result in memory, show it
    if st.session_state.prediction:
        predicted_crop = st.session_state.prediction
        
        # 1. Show Text Result
        st.success(f"{t['result_text']} **{predicted_crop}**")
        st.info(t['success_msg'])
        
        # 2. Show Visual Chart
        st.write("---") 
        st.subheader(t['chart_title'])
        chart_data = pd.DataFrame({
            'Nutrient': [t['N'], t['P'], t['K']],
            'Value': [input_df['N'][0], input_df['P'][0], input_df['K'][0]]
        })
        st.bar_chart(chart_data.set_index('Nutrient'))
        
        # 3. GEN AI SECTION
        st.write("---")
        st.subheader(t['ai_advice'])
        
        # The AI Button
        if st.button(f"{t['ai_btn']} {predicted_crop}"):
            with st.spinner("Asking Google Gemini AI..."):
                try:
                    prompt = t['ai_prompt'].format(predicted_crop)
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Connection Failed: {e}")

except FileNotFoundError:
    st.error("⚠️ Error: 'Crop_recommendation.csv' file not found.")



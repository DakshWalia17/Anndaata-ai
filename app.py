import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai

# --- PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(
    page_title="AnnDaata AI",
    page_icon="🌾",
    layout="centered"
)

# --- CUSTOM CSS (THE MAKEUP) ---
# This makes the buttons green and adds a subtle background fade
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    div.stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #1b5e20;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURE GENAI ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"⚠️ API Key Error: {e}")

# --- TRANSLATIONS ---
translations = {
    "English": {
        "title": "AnnDaata AI",
        "subtitle": "Smart Crop Advisor 🌾",
        "desc": "AI-powered precision farming for maximum yield.",
        "sidebar_title": "🌱 Soil & Weather Data",
        "predict_button": "Recommend Crop",
        "result_text": "Best Crop to Plant:",
        "success_msg": "Ideal for your soil conditions.",
        "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
        "temp": "Temperature (°C)", "hum": "Humidity (%)", "ph": "Soil pH", "rain": "Rainfall (mm)",
        "analysis": "📊 Input Summary",
        "chart_title": "Soil Nutrient Profile",
        "ai_advice": "🤖 Ask AI Agronomist",
        "ai_btn": "Get Farming Guide for",
        "ai_prompt": "Give me a practical farming guide for growing {} in India. Keep it short (5 bullet points). Language: English."
    },
    "Hindi": {
        "title": "अन्नदाता AI",
        "subtitle": "स्मार्ट फसल सलाहकार 🌾",
        "desc": "अधिकतम उपज के लिए एआई-संचालित सटीक खेती।",
        "sidebar_title": "🌱 मिट्टी और मौसम",
        "predict_button": "फसल का सुझाव दें",
        "result_text": "सुझाई गई फसल:",
        "success_msg": "आपकी मिट्टी की स्थिति के लिए आदर्श।",
        "N": "नाइट्रोजन", "P": "फॉस्फोरस", "K": "पोटेशियम",
        "temp": "तापमान", "hum": "नमी", "ph": "pH स्तर", "rain": "वर्षा",
        "analysis": "📊 इनपुट सारांश",
        "chart_title": "पोषक तत्व प्रोफ़ाइल",
        "ai_advice": "🤖 AI कृषि विशेषज्ञ",
        "ai_btn": "के लिए गाइड प्राप्त करें",
        "ai_prompt": "मुझे भारत में {} उगाने के लिए एक व्यावहारिक खेती गाइड दें। इसे छोटा रखें (5 बुलेट पॉइंट)। भाषा: हिंदी।"
    },
    "Punjabi": {
        "title": "ਅੰਨਦਾਤਾ AI",
        "subtitle": "ਫਸਲ ਸਲਾਹਕਾਰ 🌾",
        "desc": "ਵੱਧ ਝਾੜ ਲਈ AI ਅਧਾਰਤ ਖੇਤੀ।",
        "sidebar_title": "🌱 ਮਿੱਟੀ ਅਤੇ ਮੌਸਮ",
        "predict_button": "ਫਸਲ ਲੱਭੋ",
        "result_text": "ਵਧੀਆ ਫਸਲ:",
        "success_msg": "ਤੁਹਾਡੀ ਮਿੱਟੀ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ।",
        "N": "ਨਾਈਟ੍ਰੋਜਨ", "P": "ਫਾਸਫੋਰਸ", "K": "ਪੋਟਾਸ਼ੀਅਮ",
        "temp": "ਤਾਪਮਾਨ", "hum": "ਨਮੀ", "ph": "pH ਪੱਧਰ", "rain": "ਮੀਂਹ",
        "analysis": "📊 ਵੇਰਵਾ",
        "chart_title": "ਪੌਸ਼ਟਿਕ ਤੱਤ",
        "ai_advice": "🤖 AI ਖੇਤੀ ਮਾਹਰ",
        "ai_btn": "ਲਈ ਗਾਈਡ ਲਵੋ",
        "ai_prompt": "ਮੈਨੂੰ ਭਾਰਤ ਵਿੱਚ {} ਉਗਾਉਣ ਲਈ ਇੱਕ ਵਿਹਾਰਕ ਖੇਤੀ ਗਾਈਡ ਦਿਓ। ਇਸਨੂੰ ਛੋਟਾ ਰੱਖੋ (5 ਬਿੰਦੂ)। ਭਾਸ਼ਾ: ਪੰਜਾਬੀ।"
    }
}

lang_choice = st.sidebar.radio("Language", ["English", "Hindi", "Punjabi"])
t = translations[lang_choice]

# --- UI HEADER WITH LOGO ---
col1, col2 = st.columns([1, 4])
with col1:
    # Make sure 'logo.png' is in your GitHub repo!
    try:
        st.image("logo.png", width=100)
    except:
        st.write("🌾") # Fallback if logo missing
with col2:
    st.title(t['title'])
    st.markdown(f"**{t['subtitle']}**")

st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header(t['sidebar_title'])
def user_input_features():
    N = st.sidebar.slider(t['N'], 0, 140, 50)
    P = st.sidebar.slider(t['P'], 5, 145, 50)
    K = st.sidebar.slider(t['K'], 5, 205, 50)
    temperature = st.sidebar.number_input(t['temp'], 0.0, 50.0, 25.0)
    humidity = st.sidebar.number_input(t['hum'], 0.0, 100.0, 70.0)
    ph = st.sidebar.slider(t['ph'], 0.0, 14.0, 7.0)
    rainfall = st.sidebar.number_input(t['rain'], 0.0, 300.0, 100.0)
    return pd.DataFrame({'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}, index=[0])

input_df = user_input_features()

# --- MAIN LAYOUT ---
# Using columns to show Input Data vs Results side-by-side on large screens
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(t['analysis'])
    st.write(input_df)
    st.caption(t['desc'])

# --- AI ENGINE ---
try:
    crop_data = pd.read_csv("Crop_recommendation.csv") 
    X = crop_data.drop('label', axis=1)
    Y = crop_data['label']
    clf = RandomForestClassifier()
    clf.fit(X, Y)

    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    # Centered Predict Button
    with col_left:
        if st.button(t['predict_button'], use_container_width=True):
            prediction = clf.predict(input_df)
            st.session_state.prediction = prediction[0].upper()

    # --- RESULT SECTION ---
    if st.session_state.prediction:
        predicted_crop = st.session_state.prediction
        
        with col_right:
            st.success(f"{t['result_text']} **{predicted_crop}**")
            
            # Chart
            chart_data = pd.DataFrame({
                'Nutrient': ['Nitrogen', 'Phosphorus', 'Potassium'],
                'Value': [input_df['N'][0], input_df['P'][0], input_df['K'][0]]
            })
            st.bar_chart(chart_data.set_index('Nutrient'), color="#2e7d32")

        # --- GEN AI SECTION (Full Width) ---
        st.markdown("---")
        st.subheader(t['ai_advice'])
        
        if st.button(f"{t['ai_btn']} {predicted_crop}", type="primary"):
            with st.spinner("🤖 AnnDaata AI is thinking..."):
                try:
                    prompt = t['ai_prompt'].format(predicted_crop)
                    response = model.generate_content(prompt)
                    st.markdown(f"""
                    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2e7d32;">
                        {response.text}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI Error: {e}")

except FileNotFoundError:
    st.error("⚠️ Error: 'Crop_recommendation.csv' not found.")








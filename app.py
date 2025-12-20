import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
from gtts import gTTS
import io

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AnnDaata AI", page_icon="🌾", layout="wide")

# --- 2. CONFIGURATION ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # Hum 'gemini-pro' use karenge jo text ke liye best aur stable hai
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"Configuration Error: {e}")

# --- 3. HELPER FUNCTION (TEXT AI) ---
def get_ai_response(prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "⚠️ Network Busy. Unable to fetch AI advice at the moment."

# --- 4. LANGUAGE DATA ---
translations = {
    "English": {
        "title": "AnnDaata AI 2.0",
        "schemes_title": "💰 Kisan Dhan (Govt Schemes)",
        "find_schemes_btn": "Find Schemes for Me",
        "state_label": "Select State",
        "land_label": "Land Size (Acres)",
        "soil_header": "🌱 Soil & Crop Health",
        "weather_header": "🌦️ Weather Conditions",
        "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "ph": "pH Level",
        "temp": "Temperature (°C)", "hum": "Humidity (%)", "rain": "Rainfall (mm)",
        "predict_btn": "Recommend Best Crop",
        "result_header": "Best Crop to Grow:",
        "ask_ai_btn": "Ask AI How to Grow",
        "success": "High Yield Probability"
    },
    "Hindi": {
        "title": "अन्नदाता AI 2.0",
        "schemes_title": "💰 किसान धन (सरकारी योजनाएं)",
        "find_schemes_btn": "मेरे लिए योजनाएं खोजें",
        "state_label": "राज्य चुनें",
        "land_label": "जमीन (एकड़)",
        "soil_header": "🌱 मिट्टी और फसल",
        "weather_header": "🌦️ मौसम की जानकारी",
        "N": "नाइट्रोजन (N)", "P": "फॉस्फोरस (P)", "K": "पोटेशियम (K)", "ph": "pH स्तर",
        "temp": "तापमान (°C)", "hum": "नमी (%)", "rain": "वर्षा (mm)",
        "predict_btn": "सबसे अच्छी फसल जानें",
        "result_header": "सुझाई गई फसल:",
        "ask_ai_btn": "AI से खेती का तरीका पूछें",
        "success": "अधिक मुनाफे की संभावना"
    },
    "Punjabi": {
        "title": "ਅੰਨਦਾਤਾ AI 2.0",
        "schemes_title": "💰 ਕਿਸਾਨ ਧਨ (ਸਰਕਾਰੀ ਸਕੀਮਾਂ)",
        "find_schemes_btn": "ਸਕੀਮਾਂ ਲੱਭੋ",
        "state_label": "ਰਾਜ ਚੁਣੋ",
        "land_label": "ਜ਼ਮੀਨ (ਏਕੜ)",
        "soil_header": "🌱 ਮਿੱਟੀ ਦੀ ਸਿਹਤ",
        "weather_header": "🌦️ ਮੌਸਮ",
        "N": "ਨਾਈਟ੍ਰੋਜਨ (N)", "P": "ਫਾਸਫੋਰਸ (P)", "K": "ਪੋਟਾਸ਼ੀਅਮ (K)", "ph": "pH ਪੱਧਰ",
        "temp": "ਤਾਪਮਾਨ (°C)", "hum": "ਨਮੀ (%)", "rain": "ਮੀਂਹ (mm)",
        "predict_btn": "ਵਧੀਆ ਫਸਲ ਲੱਭੋ",
        "result_header": "ਸਿਫਾਰਸ਼ ਕੀਤੀ ਫਸਲ:",
        "ask_ai_btn": "AI ਗਾਈਡ ਲਵੋ",
        "success": "ਵਧੇਰੇ ਮੁਨਾਫੇ ਦੀ ਸੰਭਾਵਨਾ"
    }
}

crop_map = {
    'rice': {'hi': 'चावल (Rice)', 'pun': 'ਚੌਲ (Rice)'},
    'maize': {'hi': 'मक्का (Maize)', 'pun': 'ਮੱਕੀ (Maize)'},
    'chickpea': {'hi': 'चना (Chickpea)', 'pun': 'ਛੋਲੇ (Chickpea)'},
    'kidneybeans': {'hi': 'राजमा (Kidney Beans)', 'pun': 'ਰਾਜਮਾ (Kidney Beans)'},
    'pigeonpeas': {'hi': 'अरहर/तुअर (Pigeon Peas)', 'pun': 'ਅਰਹਰ (Pigeon Peas)'},
    'mothbeans': {'hi': 'मोठ (Moth Beans)', 'pun': 'मोठ (Moth Beans)'},
    'mungbean': {'hi': 'मूंग (Mung Bean)', 'pun': 'ਮੂੰਗੀ (Mung Bean)'},
    'blackgram': {'hi': 'उड़द (Black Gram)', 'pun': 'ਮਾਂਹ (Black Gram)'},
    'lentil': {'hi': 'मसूर (Lentil)', 'pun': 'ਮਸੂਰ (Lentil)'},
    'pomegranate': {'hi': 'अनार (Pomegranate)', 'pun': 'अनार (Pomegranate)'},
    'banana': {'hi': 'केला (Banana)', 'pun': 'ਕੇਲਾ (Banana)'},
    'mango': {'hi': 'आम (Mango)', 'pun': 'ਅੰਬ (Mango)'},
    'grapes': {'hi': 'अंगूर (Grapes)', 'pun': 'ਅੰਗੂਰ (Grapes)'},
    'watermelon': {'hi': 'तरबूज (Watermelon)', 'pun': 'ਤਰਬੂਜ (Watermelon)'},
    'muskmelon': {'hi': 'खरबूजा (Muskmelon)', 'pun': 'ਖਰਬੂਜਾ (Muskmelon)'},
    'apple': {'hi': 'सेब (Apple)', 'pun': 'ਸੇਬ (Apple)'},
    'orange': {'hi': 'संतरा (Orange)', 'pun': 'ਸੰਤਰਾ (Orange)'},
    'papaya': {'hi': 'पपीता (Papaya)', 'pun': 'ਪਪੀਤਾ (Papaya)'},
    'coconut': {'hi': 'नारियल (Coconut)', 'pun': 'ਨਾਰੀਅਲ (Coconut)'},
    'cotton': {'hi': 'कपास (Cotton)', 'pun': 'ਕਪਾਹ (Cotton)'},
    'jute': {'hi': 'जूट (Jute)', 'pun': 'ਪਟਸਨ (Jute)'},
    'coffee': {'hi': 'कॉफी (Coffee)', 'pun': 'ਕੌਫੀ (Coffee)'}
}

c1, c2 = st.columns([1, 4])
with c1: st.title("🌾")
with c2: 
    st.title("AnnDaata AI 2.0")
    lang_choice = st.radio("", ["English", "Hindi", "Punjabi"], horizontal=True)

t = translations[lang_choice] 

# ==========================================
# 1. CROP PREDICTION (Random Forest + AI)
# ==========================================
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader(t['soil_header'])
    N = st.slider(t['N'], 0, 140, 50)
    P = st.slider(t['P'], 5, 145, 50)
    K = st.slider(t['K'], 5, 205, 50)
with col2:
    st.subheader(t['weather_header'])
    temp = st.number_input(t['temp'], 0.0, 50.0, 25.0)
    hum = st.number_input(t['hum'], 0.0, 100.0, 70.0)
    rain = st.number_input(t['rain'], 0.0, 300.0, 100.0)
    ph = st.slider(t['ph'], 0.0, 14.0, 7.0)

try:
    df = pd.read_csv("Crop_recommendation.csv")
    X = df.drop('label', axis=1)
    Y = df['label']
    clf = RandomForestClassifier()
    clf.fit(X, Y)
except:
    st.warning("⚠️ Using Default Logic (CSV not found).")

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if st.button(t['predict_btn'], use_container_width=True, type="primary"):
    try:
        pred = clf.predict([[N, P, K, temp, hum, ph, rain]])
        st.session_state.prediction = pred[0]
    except:
        st.session_state.prediction = "rice"

if st.session_state.prediction:
    raw_crop = st.session_state.prediction.lower()
    if lang_choice == "Hindi":
        display_crop = crop_map.get(raw_crop, {}).get('hi', raw_crop.title())
    elif lang_choice == "Punjabi":
        display_crop = crop_map.get(raw_crop, {}).get('pun', raw_crop.title())
    else:
        display_crop = raw_crop.title()

    st.success(f"{t['result_header']} {display_crop} 🌾")
    
    if st.button(f"{t['ask_ai_btn']} {display_crop}"):
        with st.spinner("AI Agronomist is thinking..."):
            prompt = f"Give a practical farming guide for {raw_crop} in {lang_choice}. Keep it short (4 bullet points)."
            response_text = get_ai_response(prompt)
            st.info(response_text)
            try:
                tts_lang = 'hi' if lang_choice != 'English' else 'en'
                tts = gTTS(text=response_text, lang=tts_lang, slow=False)
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                st.audio(audio_bytes, format='audio/mp3')
            except:
                pass

# ==========================================
# 2. KISAN DHAN - GOVT SCHEMES (AI Text)
# ==========================================
st.markdown("---")
st.header(t['schemes_title'])
st.write("Find financial support & subsidies / आर्थिक मदद खोजें")

kc1, kc2 = st.columns(2)
with kc1:
    user_state = st.selectbox(t['state_label'], ["Punjab", "Haryana", "UP", "Maharashtra", "Other"])
with kc2:
    land_size = st.number_input(t['land_label'], 1.0, 100.0, 2.5)

if st.button(t['find_schemes_btn'], use_container_width=True):
    with st.spinner("Searching Govt Database..."):
        scheme_prompt = f"List 3 govt schemes for a farmer in {user_state} with {land_size} acres. Focus on subsidies. Output Language: {lang_choice}. Keep it short."
        response_text = get_ai_response(scheme_prompt)
        st.warning(response_text)

st.markdown('<div style="text-align:center; padding:20px; color:grey;">Made with ❤️ by Team Debuggers</div>', unsafe_allow_html=True)

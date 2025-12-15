import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AnnDaata AI",
    page_icon="🌾",
    layout="wide"
)

# --- CROP TRANSLATION & EMOJI DATABASE ---
# This maps the English output from the model to Hindi/Punjabi + Emojis
crop_details = {
    'rice': {'emoji': '🍚', 'hi': 'चावल', 'pun': 'ਚੌਲ'},
    'maize': {'emoji': '🌽', 'hi': 'मक्का', 'pun': 'ਮੱਕੀ'},
    'chickpea': {'emoji': '🟤', 'hi': 'चना', 'pun': 'ਛੋਲੇ'},
    'kidneybeans': {'emoji': '🔴', 'hi': 'राजमा', 'pun': 'ਰਾਜਮਾ'},
    'pigeonpeas': {'emoji': '🟡', 'hi': 'अरहर (तुअर)', 'pun': 'ਅਰਹਰ'},
    'mothbeans': {'emoji': '🟤', 'hi': 'मोठ', 'pun': 'ਮੋਠ'},
    'mungbean': {'emoji': '🟢', 'hi': 'मूंग', 'pun': 'ਮੂੰਗੀ'},
    'blackgram': {'emoji': '⚫', 'hi': 'उड़द', 'pun': 'ਮਾਂਹ'},
    'lentil': {'emoji': '🍲', 'hi': 'मसूर', 'pun': 'ਮਸੂਰ'},
    'pomegranate': {'emoji': '🔴', 'hi': 'अनार', 'pun': 'ਅਨਾਰ'},
    'banana': {'emoji': '🍌', 'hi': 'केला', 'pun': 'ਕੇਲਾ'},
    'mango': {'emoji': '🥭', 'hi': 'आम', 'pun': 'ਅੰਬ'},
    'grapes': {'emoji': '🍇', 'hi': 'अंगूर', 'pun': 'ਅੰਗੂਰ'},
    'watermelon': {'emoji': '🍉', 'hi': 'तरबूज', 'pun': 'ਤਰਬੂਜ'},
    'muskmelon': {'emoji': '🍈', 'hi': 'खरबूजा', 'pun': 'ਖਰਬੂਜਾ'},
    'apple': {'emoji': '🍎', 'hi': 'सेब', 'pun': 'ਸੇਬ'},
    'orange': {'emoji': '🍊', 'hi': 'संतरा', 'pun': 'ਸੰਤਰਾ'},
    'papaya': {'emoji': '🥣', 'hi': 'पपीता', 'pun': 'ਪਪੀਤਾ'},
    'coconut': {'emoji': '🥥', 'hi': 'नारियल', 'pun': 'ਨਾਰੀਅਲ'},
    'cotton': {'emoji': '☁️', 'hi': 'कपास', 'pun': 'ਕਪਾਹ'},
    'jute': {'emoji': '🧶', 'hi': 'जूट', 'pun': 'ਪਟਸਨ'},
    'coffee': {'emoji': '☕', 'hi': 'कॉफी', 'pun': 'ਕੌਫੀ'}
}

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    h1, h2, h3, h4, h5, h6, p, div, span { color: #0d3b10 !important; }
    
    /* Green Buttons */
    div.stButton > button {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    div.stButton > button:hover { background-color: #1b5e20 !important; }
    
    /* Input Box Styling */
    div[data-baseweb="input"] { border-radius: 10px; }
    
    /* Footer Styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #2e7d32;
        color: white !important;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        z-index: 1000;
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
        "input_section": "📝 Enter Field Details",
        "soil_sec": "Soil Nutrients",
        "weather_sec": "Weather Conditions",
        "predict_button": "Recommend Crop",
        "result_text": "Best Crop to Plant:",
        "success_msg": "Ideal for your soil conditions.",
        "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
        "temp": "Temperature", "hum": "Humidity", "ph": "Soil pH", "rain": "Rainfall",
        "analysis": "📊 Analysis Dashboard",
        "ai_advice": "🤖 Ask AI Agronomist",
        "ai_btn": "Get Farming Guide for",
        "ai_prompt": "Give me a practical farming guide for growing {} in India. Keep it short (5 bullet points). Language: English."
    },
    "Hindi": {
        "title": "अन्नदाता AI",
        "subtitle": "स्मार्ट फसल सलाहकार 🌾",
        "input_section": "📝 खेत का विवरण दर्ज करें",
        "soil_sec": "मिट्टी के पोषक तत्व",
        "weather_sec": "मौसम की स्थिति",
        "predict_button": "फसल का सुझाव दें",
        "result_text": "सुझाई गई फसल:",
        "success_msg": "आपकी मिट्टी की स्थिति के लिए आदर्श।",
        "N": "नाइट्रोजन", "P": "फॉस्फोरस", "K": "पोटेशियम",
        "temp": "तापमान", "hum": "नमी", "ph": "pH स्तर", "rain": "वर्षा",
        "analysis": "📊 विश्लेषण डैशबोर्ड",
        "ai_advice": "🤖 AI कृषि विशेषज्ञ",
        "ai_btn": "के लिए गाइड प्राप्त करें",
        "ai_prompt": "मुझे भारत में {} उगाने के लिए एक व्यावहारिक खेती गाइड दें। इसे छोटा रखें (5 बुलेट पॉइंट)। भाषा: हिंदी."
    },
    "Punjabi": {
        "title": "ਅੰਨਦਾਤਾ AI",
        "subtitle": "ਫਸਲ ਸਲਾਹਕਾਰ 🌾",
        "input_section": "📝 ਖੇਤੀ ਦਾ ਵੇਰਵਾ",
        "soil_sec": "ਮਿੱਟੀ ਦੇ ਤੱਤ",
        "weather_sec": "ਮੌਸਮ ਦੇ ਹਾਲਾਤ",
        "predict_button": "ਫਸਲ ਲੱਭੋ",
        "result_text": "ਵਧੀਆ ਫਸਲ:",
        "success_msg": "ਤੁਹਾਡੀ ਮਿੱਟੀ ਲਈ ਸਭ ਤੋਂ ਵਧੀਆ।",
        "N": "ਨਾਈਟ੍ਰੋਜਨ", "P": "ਫਾਸਫੋਰਸ", "K": "ਪੋਟਾਸ਼ੀਅਮ",
        "temp": "ਤਾਪਮਾਨ", "hum": "ਨਮੀ", "ph": "pH ਪੱਧਰ", "rain": "ਮੀਂਹ",
        "analysis": "📊 ਵਿਸ਼ਲੇਸ਼ਣ",
        "ai_advice": "🤖 AI ਖੇਤੀ ਮਾਹਰ",
        "ai_btn": "ਲਈ ਗਾਈਡ ਲਵੋ",
        "ai_prompt": "ਮੈਨੂੰ ਭਾਰਤ ਵਿੱਚ {} ਉਗਾਉਣ ਲਈ ਇੱਕ ਵਿਹਾਰਕ ਖੇਤੀ ਗਾਈਡ ਦਿਓ। ਇਸਨੂੰ ਛੋਟਾ ਰੱਖੋ (5 ਬਿੰਦੂ)। ਭਾਸ਼ਾ: ਪੰਜਾਬੀ."
    }
}

# --- HEADER & LANGUAGE ---
c1, c2 = st.columns([1, 5])
with c1:
    try:
        st.image("logo.png", width=80)
    except:
        st.write("🌾")
with c2:
    lang_choice = st.radio("Language", ["English", "Hindi", "Punjabi"], horizontal=True, label_visibility="collapsed")

t = translations[lang_choice]
st.title(t['title'])
st.markdown(f"**{t['subtitle']}**")
st.markdown("---")

# --- INPUTS (MOBILE FIRST) ---
st.subheader(t['input_section'])

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{t['soil_sec']}**")
        N = st.slider(t['N'], 0, 140, 50)
        P = st.slider(t['P'], 5, 145, 50)
        K = st.slider(t['K'], 5, 205, 50)
        ph = st.slider(t['ph'], 0.0, 14.0, 7.0)
    with col2:
        st.markdown(f"**{t['weather_sec']}**")
        temperature = st.number_input(t['temp'] + " (°C)", 0.0, 50.0, 25.0)
        humidity = st.number_input(t['hum'] + " (%)", 0.0, 100.0, 70.0)
        rainfall = st.number_input(t['rain'] + " (mm)", 0.0, 300.0, 100.0)

input_df = pd.DataFrame({'N': N, 'P': P, 'K': K, 'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall}, index=[0])

st.markdown("---")

# --- PREDICTION & DISPLAY ---
try:
    crop_data = pd.read_csv("Crop_recommendation.csv") 
    X = crop_data.drop('label', axis=1)
    Y = crop_data['label']
    clf = RandomForestClassifier()
    clf.fit(X, Y)

    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    if st.button(t['predict_button'], use_container_width=True):
        prediction = clf.predict(input_df)
        st.session_state.prediction = prediction[0].lower() # Convert to lowercase for dictionary lookup

    if st.session_state.prediction:
        # 1. Get English Name
        raw_crop = st.session_state.prediction
        
        # 2. Get Translated Name & Emoji
        crop_info = crop_details.get(raw_crop, {'emoji': '🌿', 'hi': raw_crop, 'pun': raw_crop})
        
        if lang_choice == "Hindi":
            display_name = f"{crop_info['hi']} {crop_info['emoji']}"
            search_term = crop_info['hi'] # Send Hindi to Gemini
        elif lang_choice == "Punjabi":
            display_name = f"{crop_info['pun']} {crop_info['emoji']}"
            search_term = crop_info['pun'] # Send Punjabi to Gemini
        else:
            display_name = f"{raw_crop.capitalize()} {crop_info['emoji']}"
            search_term = raw_crop.capitalize()

        st.markdown("---")
        
        # Result Card
        st.markdown(f"""
        <div style="background-color: #c8e6c9; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #1b5e20; margin:0;">{t['result_text']} {display_name}</h2>
            <p style="color: #1b5e20;">{t['success_msg']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Charts
        st.subheader(t['analysis'])
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Nutrients")
            st.bar_chart(pd.DataFrame({'Value': [N, P, K]}, index=['N', 'P', 'K']))
        with chart_col2:
            st.caption("Weather")
            st.bar_chart(pd.DataFrame({'Value': [temperature, humidity]}, index=['Temp', 'Hum']))
        
        # --- GEN AI SECTION ---
        st.markdown("---")
        st.subheader(t['ai_advice'])
        
        if st.button(f"{t['ai_btn']} {display_name}"):
            with st.spinner("🤖 AnnDaata AI is thinking..."):
                try:
                    # We send the SEARCH_TERM (Correct Language) to Gemini
                    prompt = t['ai_prompt'].format(search_term)
                    response = model.generate_content(prompt)
                    
                    st.markdown(f"""
                    <div style="
                        background-color: #e8f5e9; 
                        padding: 20px; 
                        border-radius: 10px; 
                        border-left: 5px solid #2e7d32;
                        color: #000000;
                        font-family: sans-serif;
                    ">
                        {response.text}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI Error: {e}")

except FileNotFoundError:
    st.error("⚠️ Error: 'Crop_recommendation.csv' not found.")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Made with ❤️ by Team AnnDaata | GenAI Hackathon 2025
</div>
<div style="margin-bottom: 50px;"></div>
""", unsafe_allow_html=True)











import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
from gtts import gTTS
import io
import PIL.Image

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AnnDaata AI", page_icon="🌾", layout="wide")

# --- 2. CSS STYLING (PERFECT VISIBILITY FIX) ---
st.markdown("""
    <style>
    /* 1. Main Background (Light Grey) */
    .stApp { background-color: #f0f2f6; }
    
    /* 2. Main Body Text (Dark Green for readability) */
    .main h1, .main h2, .main h3, .main h4, .main p, .main li, .main span, .main label { 
        color: #0d3b10 !important; 
    }
    
    /* --- 3. SIDEBAR FIXES (Dark BG + White Text) --- */
    section[data-testid="stSidebar"] {
        background-color: #1b5e20 !important; /* Dark Green BG */
    }
    
    /* Force ALL Text in Sidebar to be White */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
    
    /* EXCEPTION: Text inside Input Boxes in Sidebar must be Black (because box is white) */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div, 
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    /* Dropdown Options Text */
    section[data-testid="stSidebar"] div[data-baseweb="popover"] div {
        color: #000000 !important;
    }
    
    /* --- 4. FILE UPLOADER FIX (White Text for Dark Box) --- */
    div[data-testid="stFileUploader"] label {
        color: #ffffff !important; /* Label Text White */
        font-weight: bold;
    }
    div[data-testid="stFileUploader"] {
        color: #ffffff !important; /* Drag-Drop Text White */
    }
    div[data-testid="stFileUploader"] small {
        color: #e0e0e0 !important; /* "Limit 200MB" Text Light Grey */
    }
    
    /* --- 5. BUTTON STYLING --- */
    div.stButton > button { 
        background-color: #2e7d32 !important; 
        color: #ffffff !important; 
        border-radius: 10px; 
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover { 
        background-color: #1b5e20 !important; 
        color: white !important; 
    }
    
    /* Footer */
    .footer { 
        position: fixed; bottom: 0; left: 0; width: 100%; 
        background-color: #2e7d32; color: white !important; 
        text-align: center; padding: 10px; z-index: 999;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LANGUAGE DICTIONARIES ---
translations = {
    "English": {
        "title": "AnnDaata AI 2.0",
        "sidebar_title": "⚙️ Settings",
        "schemes_title": "💰 Kisan Dhan",
        "find_schemes_btn": "Find Schemes",
        "state_label": "State",
        "land_label": "Land (Acres)",
        "soil_header": "🌱 Soil Health",
        "weather_header": "🌦️ Weather",
        "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "ph": "pH Level",
        "temp": "Temperature (°C)", "hum": "Humidity (%)", "rain": "Rainfall (mm)",
        "predict_btn": "Recommend Crop",
        "result_header": "Recommended Crop:",
        "ask_ai_btn": "Get AI Guide for",
        "dr_header": "📸 Dr. AnnDaata (Plant Doctor)",
        "upload_label": "Upload a photo of the affected plant/leaf",
        "diagnose_btn": "🔍 Diagnose Disease",
        "spinner_leaf": "Scanning Leaf...",
        "spinner_scheme": "Finding Schemes...",
        "success": "High Profit Probability"
    },
    "Hindi": {
        "title": "अन्नदाता AI 2.0",
        "sidebar_title": "⚙️ सेटिंग्स",
        "schemes_title": "💰 किसान धन (योजनाएं)",
        "find_schemes_btn": "योजनाएं खोजें",
        "state_label": "राज्य",
        "land_label": "जमीन (एकड़)",
        "soil_header": "🌱 मिट्टी की सेहत",
        "weather_header": "🌦️ मौसम",
        "N": "नाइट्रोजन (N)", "P": "फॉस्फोरस (P)", "K": "पोटेशियम (K)", "ph": "pH स्तर",
        "temp": "तापमान (°C)", "hum": "नमी (%)", "rain": "वर्षा (mm)",
        "predict_btn": "फसल सुझाव लें",
        "result_header": "सुझाई गई फसल:",
        "ask_ai_btn": "AI गाइड प्राप्त करें: ",
        "dr_header": "📸 डॉ. अन्नदाता (पौधा चिकित्सक)",
        "upload_label": "बीमार पौधे/पत्ते की फोटो अपलोड करें",
        "diagnose_btn": "🔍 बीमारी पहचानें",
        "spinner_leaf": "पत्ते की जांच हो रही है...",
        "spinner_scheme": "योजनाएं खोजी जा रही हैं...",
        "success": "अधिक मुनाफे की संभावना"
    },
    "Punjabi": {
        "title": "ਅੰਨਦਾਤਾ AI 2.0",
        "sidebar_title": "⚙️ ਸੈਟਿੰਗਾਂ",
        "schemes_title": "💰 ਕਿਸਾਨ ਧਨ (ਸਕੀਮਾਂ)",
        "find_schemes_btn": "ਸਕੀਮਾਂ ਲੱਭੋ",
        "state_label": "ਰਾਜ",
        "land_label": "ਜ਼ਮੀਨ (ਏਕੜ)",
        "soil_header": "🌱 ਮਿੱਟੀ ਦੀ ਸਿਹਤ",
        "weather_header": "🌦️ ਮੌਸਮ",
        "N": "ਨਾਈਟ੍ਰੋਜਨ (N)", "P": "ਫਾਸਫੋਰਸ (P)", "K": "ਪੋਟਾਸ਼ੀਅਮ (K)", "ph": "pH ਪੱਧਰ",
        "temp": "ਤਾਪਮਾਨ (°C)", "hum": "ਨਮੀ (%)", "rain": "ਮੀਂਹ (mm)",
        "predict_btn": "ਫਸਲ ਲੱਭੋ",
        "result_header": "ਸਿਫਾਰਸ਼ ਕੀਤੀ ਫਸਲ:",
        "ask_ai_btn": "AI ਗਾਈਡ ਲਵੋ: ",
        "dr_header": "📸 ਡਾ. ਅੰਨਦਾਤਾ (ਪੌਦਾ ਡਾਕਟਰ)",
        "upload_label": "ਬਿਮਾਰ ਪੌਦੇ/ਪੱਤੇ ਦੀ ਫੋਟੋ ਅਪਲੋਡ ਕਰੋ",
        "diagnose_btn": "🔍 ਬਿਮਾਰੀ ਲੱਭੋ",
        "spinner_leaf": "ਪੱਤੇ ਦੀ ਜਾਂਚ ਹੋ ਰਹੀ ਹੈ...",
        "spinner_scheme": "ਸਕੀਮਾਂ ਲੱਭੀਆਂ ਜਾ ਰਹੀਆਂ ਹਨ...",
        "success": "ਵਧੇਰੇ ਮੁਨਾਫੇ ਦੀ ਸੰਭਾਵਨਾ"
    }
}

crop_map = {
    'rice': {'hi': 'चावल (Rice)', 'pun': 'ਚੌਲ (Rice)'},
    'maize': {'hi': 'मक्का (Maize)', 'pun': 'ਮੱਕੀ (Maize)'},
    'chickpea': {'hi': 'चना (Chickpea)', 'pun': 'ਛੋਲੇ (Chickpea)'},
    'kidneybeans': {'hi': 'राजमा (Kidney Beans)', 'pun': 'ਰਾਜਮਾ (Kidney Beans)'},
    'pigeonpeas': {'hi': 'अरहर/तुअर (Pigeon Peas)', 'pun': 'ਅਰਹਰ (Pigeon Peas)'},
    'mothbeans': {'hi': 'मोठ (Moth Beans)', 'pun': 'ਮੋਠ (Moth Beans)'},
    'mungbean': {'hi': 'मूंग (Mung Bean)', 'pun': 'ਮੂੰਗੀ (Mung Bean)'},
    'blackgram': {'hi': 'उड़द (Black Gram)', 'pun': 'ਮਾਂਹ (Black Gram)'},
    'lentil': {'hi': 'मसूर (Lentil)', 'pun': 'ਮਸੂਰ (Lentil)'},
    'pomegranate': {'hi': 'अनार (Pomegranate)', 'pun': 'ਅਨਾਰ (Pomegranate)'},
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

# --- 4. LANGUAGE SELECTOR ---
c1, c2 = st.columns([1, 5])
with c1: st.write("🌾")
with c2: 
    lang_choice = st.radio("Language / भाषा / ਭਾਸ਼ਾ", ["English", "Hindi", "Punjabi"], horizontal=True)

t = translations[lang_choice] 

# --- 5. SIDEBAR (KISAN DHAN ONLY) ---
with st.sidebar:
    st.title(t['sidebar_title'])
    st.header(t['schemes_title'])
    user_state = st.selectbox(t['state_label'], ["Punjab", "Haryana", "UP", "Maharashtra", "Other"])
    land_size = st.number_input(t['land_label'], 1.0, 100.0, 2.5)
    
    if st.button(t['find_schemes_btn']):
        with st.spinner(t['spinner_scheme']):
            try:
                genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                model = genai.GenerativeModel('gemini-2.5-flash')
                scheme_prompt = f"List 3 govt schemes for a farmer in {user_state} with {land_size} acres. Focus on subsidies. Output Language: {lang_choice}. Keep it short."
                response = model.generate_content(scheme_prompt)
                st.info(response.text)
            except:
                st.error("Check Internet Connection.")

# --- 6. MAIN APP LOGIC ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-flash')
except:
    st.error("⚠️ API Key Error. Check .streamlit/secrets.toml")

st.title(t['title'])

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

# Load Model
try:
    df = pd.read_csv("Crop_recommendation.csv")
    X = df.drop('label', axis=1)
    Y = df['label']
    clf = RandomForestClassifier()
    clf.fit(X, Y)
except:
    st.warning("Using Demo Model (CSV not found)")

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# --- PREDICTION ---
if st.button(t['predict_btn'], use_container_width=True):
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

    st.markdown(f"""
    <div style="background-color: #c8e6c9; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #2e7d32;">
        <h2 style="color: #1b5e20; margin:0;">{t['result_header']} {display_crop} 🌾</h2>
        <p style="color: #1b5e20;">{t['success']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button(f"{t['ask_ai_btn']} {display_crop}"):
        with st.spinner("AI Agronomist is thinking..."):
            prompt = f"Give a practical farming guide for {raw_crop} in {lang_choice}. Keep it short (4 bullet points)."
            response = model.generate_content(prompt)
            
            st.markdown(f"""
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32; color:black;">
                {response.text}
            </div>
            """, unsafe_allow_html=True)
            
            try:
                tts_lang = 'hi' if lang_choice != 'English' else 'en'
                tts = gTTS(text=response.text, lang=tts_lang, slow=False)
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                st.audio(audio_bytes, format='audio/mp3')
            except:
                pass

# --- DR. ANNDAATA ---
st.markdown("---")
st.subheader(t['dr_header'])
st.caption(t['upload_label'])

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = PIL.Image.open(uploaded_file)
    st.image(image, width=300)
    
    if st.button(t['diagnose_btn']):
        with st.spinner(t['spinner_leaf']):
            vision_prompt = f"Analyze this plant leaf. Identify disease and suggest cure in {lang_choice}. Keep it brief."
            response = model.generate_content([vision_prompt, image])
            
            st.markdown(f"""
            <div style="background-color: #ffcdd2; padding: 15px; border-radius: 10px; border-left: 5px solid #d32f2f; color:black;">
                <b>Diagnosis Report:</b><br>{response.text}
            </div>
            """, unsafe_allow_html=True)
            
            try:
                tts = gTTS(text=response.text, lang='hi', slow=False)
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                st.audio(audio_bytes, format='audio/mp3')
            except:
                pass

st.markdown('<div class="footer">Made with ❤️ by Team Debuggers</div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from gtts import gTTS
import io
import time  # Fake loading ke liye
import plotly.express as px

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AnnDaata AI", page_icon="🌾", layout="wide")

# --- 2. FAKE AI BRAIN (ADVICE LOGIC) ---
def get_fake_advice(crop_name, lang):
    time.sleep(1.5) # Fake loading
    crop = crop_name.lower()
    
    if "rice" in crop or "chawal" in crop:
        if lang == "Hindi":
            return "**🌾 चावल की खेती (AI सुझाव):**\n1. **पानी:** खेत में 2-3 इंच पानी जमा रखें।\n2. **खाद:** यूरिया और DAP का सही समय पर प्रयोग करें।\n3. **कीट:** तना छेदक (Stem Borer) से बचाव के लिए नीम का तेल छिड़कें।\n4. **कटाई:** जब बालियां 80% सुनहरी हो जाएं तब कटाई करें।"
        elif lang == "Punjabi":
            return "**🌾 ਝੋਨੇ ਦੀ ਖੇਤੀ (AI ਸਲਾਹ):**\n1. **ਪਾਣੀ:** ਖੇਤ ਵਿੱਚ ਪਾਣੀ ਖੜ੍ਹਾ ਰੱਖੋ।\n2. **ਖਾਦ:** ਯੂਰੀਆ ਦੀ ਵਰਤੋਂ ਕਿਸ਼ਤਾਂ ਵਿੱਚ ਕਰੋ।\n3. **ਬਿਮਾਰੀ:** ਪੱਤਾ ਲਪੇਟ ਸੁੰਡੀ ਦਾ ਧਿਆਨ ਰੱਖੋ।\n4. **ਕਟਾਈ:** ਦਾਣੇ ਪੱਕਣ 'ਤੇ ਕਟਾਈ ਕਰੋ।"
        else:
            return "**🌾 Rice Farming Guide (Expert AI):**\n1. **Water:** Maintain standing water (2-3 inches).\n2. **Fertilizer:** Apply NPK 120:60:60 in splits.\n3. **Pest Control:** Use Neem oil for Stem Borer.\n4. **Harvest:** Harvest when grains turn golden yellow."
    
    elif "maize" in crop or "corn" in crop:
        if lang == "Hindi":
            return "**🌽 मक्का की खेती:**\n1. **जल निकासी:** खेत में पानी जमा न होने दें।\n2. **खाद:** नाइट्रोजन को 3 भागों में डालें।\n3. **कीट:** फॉल आर्मीवॉर्म (Fall Armyworm) का ध्यान रखें।\n4. **कटाई:** भुट्टा सूखने पर ही तोड़ें।"
        elif lang == "Punjabi":
            return "**🌽 ਮੱਕੀ ਦੀ ਖੇਤੀ:**\n1. **ਪਾਣੀ:** ਖੇਤ ਵਿੱਚ ਪਾਣੀ ਖੜ੍ਹਾ ਨਾ ਹੋਣ ਦਿਓ।\n2. **ਖਾਦ:** ਨਾਈਟ੍ਰੋਜਨ 3 ਕਿਸ਼ਤਾਂ ਵਿੱਚ ਪਾਓ।\n3. **ਕੀੜਾ:** ਸੁੰਡੀ (Fall Armyworm) ਤੋਂ ਬਚਾਓ ਕਰੋ।\n4. **ਕਟਾਈ:** ਛੱਲੀਆਂ ਸੁੱਕਣ 'ਤੇ ਕਟਾਈ ਕਰੋ।"
        else:
            return "**🌽 Maize Farming Guide:**\n1. Ensure good drainage (No waterlogging).\n2. Apply Nitrogen in 3 splits.\n3. Watch out for Fall Armyworm.\n4. Harvest when husks turn dry."

    else:
        if lang == "Hindi":
            return f"**🌱 {crop_name.title()} की खेती (सुझाव):**\n1. **मिट्टी:** जैविक खाद (Gobbar Ki Khaad) का प्रयोग करें।\n2. **सिंचाई:** मिट्टी की नमी देखकर ही पानी दें।\n3. **निराई:** समय-समय पर खरपतवार (Weeds) निकालें।\n4. **सलाह:** किसी भी बीमारी के लिए नजदीकी कृषि केंद्र (KVK) से संपर्क करें।"
        elif lang == "Punjabi":
            return f"**🌱 {crop_name.title()} ਦੀ ਖੇਤੀ (ਸਲਾਹ):**\n1. **ਮਿੱਟੀ:** ਦੇਸੀ ਰੂੜੀ ਖਾਦ ਦੀ ਵਰਤੋਂ ਕਰੋ।\n2. **ਪਾਣੀ:** ਮਿੱਟੀ ਦੀ ਨਮੀ ਦੇਖ ਕੇ ਪਾਣੀ ਲਗਾਓ।\n3. **ਨਦੀਨ:** ਸਮੇਂ-ਸਮੇਂ 'ਤੇ ਨਦੀਨ (Weeds) ਕੱਢੋ।\n4. **ਸਲਾਹ:** ਕਿਸੇ ਵੀ ਬਿਮਾਰੀ ਲਈ ਨੇੜੇ ਦੇ ਖੇਤੀਬਾੜੀ ਕੇਂਦਰ ਨਾਲ ਸੰਪਰਕ ਕਰੋ।"
        else:
            return f"**🌱 {crop_name.title()} Farming Guide:**\n1. Prepare soil with organic compost.\n2. Ensure proper irrigation based on soil moisture.\n3. Remove weeds regularly.\n4. Consult local KVK for specific pest issues."

# --- 3. FAKE SCHEMES LOGIC ---
def get_fake_schemes(state, lang):
    time.sleep(1.5) # Fake loading
    
    if state == "Punjab":
        if lang == "Hindi":
            return "**💰 पंजाब सरकार की योजनाएं:**\n1. **पानी बचाओ पैसे कमाओ:** बिजली/पानी बचाने पर सब्सिडी।\n2. **पराली प्रबंधन:** मशीनों (Mulchers) पर 50-80% सब्सिडी।\n3. **किसान क्रेडिट कार्ड (KCC):** कम ब्याज पर लोन।"
        elif lang == "Punjabi":
            return "**💰 ਪੰਜਾਬ ਸਰਕਾਰ ਦੀਆਂ ਸਕੀਮਾਂ:**\n1. **ਪਾਣੀ ਬਚਾਓ ਪੈਸੇ ਕਮਾਓ:** ਬਿਜਲੀ ਬਚਾਉਣ 'ਤੇ ਸਬਸਿਡੀ।\n2. **ਪਰਾਲੀ ਪ੍ਰਬੰਧਨ:** ਮਸ਼ੀਨਾਂ 'ਤੇ 50-80% ਸਬਸਿਡੀ।\n3. **ਕਿਸਾਨ ਕ੍ਰੈਡਿਟ ਕਾਰਡ (KCC):** ਘੱਟ ਵਿਆਜ 'ਤੇ ਕਰਜ਼ਾ।"
        else:
            return "**💰 Schemes in Punjab:**\n1. **Pani Bachao Paise Kamao:** Subsidy for saving electricity/water.\n2. **Crop Residue Management:** 50-80% subsidy on mulchers/seeders.\n3. **Kisan Credit Card (KCC):** Low interest loans."
            
    elif state == "Haryana":
        if lang == "Hindi":
            return "**💰 हरियाणा सरकार की योजनाएं:**\n1. **मेरा पानी मेरी विरासत:** फसल बदलने पर ₹7000/एकड़।\n2. **भावंतर भरपाई योजना:** सब्जियों के दाम गिरने पर मुआवजा।\n3. **सोलर पंप सब्सिडी:** 75% तक की छूट।"
        elif lang == "Punjabi":
            return "**💰 ਹਰਿਆਣਾ ਸਰਕਾਰ ਦੀਆਂ ਸਕੀਮਾਂ:**\n1. **ਮੇਰਾ ਪਾਣੀ ਮੇਰੀ ਵਿਰਾਸਤ:** ਫਸਲ ਬਦਲਣ 'ਤੇ ₹7000/ਏਕੜ।\n2. **ਭਾਵੰਤਰ ਭਰਪਾਈ ਯੋਜਨਾ:** ਸਬਜ਼ੀਆਂ ਦੇ ਭਾਅ ਡਿੱਗਣ 'ਤੇ ਮੁਆਵਜ਼ਾ।\n3. **ਸੋਲਰ ਪੰਪ:** 75% ਤੱਕ ਸਬਸਿਡੀ।"
        else:
            return "**💰 Schemes in Haryana:**\n1. **Mera Pani Meri Virasat:** Rs. 7000/acre for crop diversification.\n2. **Bhavantar Bharpayee Yojana:** Price protection for vegetables.\n3. **Solar Pump Subsidy:** Up to 75% off."
            
    else:
        if lang == "Hindi":
            return "**💰 केंद्र सरकार की योजनाएं:**\n1. **PM-KISAN:** हर साल ₹6000 सीधे खाते में।\n2. **फसल बीमा योजना:** फसल खराब होने पर मुआवजा।\n3. **सॉइल हेल्थ कार्ड:** मिट्टी की मुफ्त जांच।"
        elif lang == "Punjabi":
            return "**💰 ਕੇਂਦਰ ਸਰਕਾਰ ਦੀਆਂ ਸਕੀਮਾਂ:**\n1. **PM-KISAN:** ਹਰ ਸਾਲ ₹6000 ਸਿੱਧੇ ਖਾਤੇ ਵਿੱਚ।\n2. **ਫਸਲ ਬੀਮਾ ਯੋਜਨਾ:** ਫਸਲ ਖਰਾਬ ਹੋਣ 'ਤੇ ਮੁਆਵਜ਼ਾ।\n3. **ਸੋਇਲ ਹੈਲਥ ਕਾਰਡ:** ਮਿੱਟੀ ਦੀ ਮੁਫਤ ਜਾਂਚ।"
        else:
            return "**💰 Central Govt Schemes:**\n1. **PM-KISAN:** Rs. 6000 per year direct transfer.\n2. **PM Fasal Bima Yojana:** Crop insurance against failure.\n3. **Soil Health Card:** Free soil testing."

# --- LANGUAGE DICTIONARY ---
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
        "success": "High Yield Probability",
        "graph_title": "📊 Why this crop? (AI Reasoning)"
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
        "success": "अधिक मुनाफे की संभावना",
        "graph_title": "📊 यही फसल क्यों? (AI का कारण)"
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
        "success": "ਵਧੇਰੇ ਮੁਨਾਫੇ ਦੀ ਸੰਭਾਵਨਾ",
        "graph_title": "📊 ਇਹ ਫਸਲ ਕਿਉਂ? (AI ਦਾ ਕਾਰਨ)"
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
    'pomegranate': {'hi': 'अनार (Pomegranate)', 'pun': 'अनार (Pomegranate)'},
    'banana': {'hi': 'केला (Banana)', 'pun': 'ਕੇਲਾ (Banana)'},
    'mango': {'hi': 'आम (Mango)', 'pun': 'ਅੰਬ (Mango)'},
    'grapes': {'hi': 'अंगूर (Grapes)', 'pun': 'ਅੰਗੂਰ (Grapes)'},
    'watermelon': {'hi': 'तरबूज (Watermelon)', 'pun': 'तरबूज (Watermelon)'},
    'muskmelon': {'hi': 'खरबूजा (Muskmelon)', 'pun': 'खरबूजा (Muskmelon)'},
    'apple': {'hi': 'सेब (Apple)', 'pun': 'सेब (Apple)'},
    'orange': {'hi': 'संतरा (Orange)', 'pun': 'संतरा (Orange)'},
    'papaya': {'hi': 'पपीता (Papaya)', 'pun': 'पपीता (Papaya)'},
    'coconut': {'hi': 'नारियल (Coconut)', 'pun': 'नारियल (Coconut)'},
    'cotton': {'hi': 'कपास (Cotton)', 'pun': 'कपास (Cotton)'},
    'jute': {'hi': 'जूट (Jute)', 'pun': 'जूट (Jute)'},
    'coffee': {'hi': 'कॉफी (Coffee)', 'pun': 'कॉफी (Coffee)'}
}

c1, c2 = st.columns([1, 4])
with c1: st.title("🌾")
with c2: 
    st.title("AnnDaata AI 2.0")
    lang_choice = st.radio("", ["English", "Hindi", "Punjabi"], horizontal=True)

t = translations[lang_choice] 

# ==========================================
# 1. CROP PREDICTION (REAL ML)
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

# Initialize Model Global Variables
clf = None
X_columns = None

try:
    df = pd.read_csv("Crop_recommendation.csv")
    X = df.drop('label', axis=1)
    Y = df['label']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, Y)
    X_columns = X.columns 
except Exception as e:
    # --- ERROR PRINTING ADDED HERE ---
    st.error(f"❌ Model Error (CSV Not Found or Corrupt): {e}")
    st.warning("⚠️ Using Default Logic because of the above error.")

if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False

if st.button(t['predict_btn'], use_container_width=True, type="primary"):
    try:
        if clf is not None:
            pred = clf.predict([[N, P, K, temp, hum, ph, rain]])
            st.session_state.prediction = pred[0]
            st.session_state.show_graph = True 
        else:
             st.session_state.prediction = "rice"
             st.session_state.show_graph = False
    except Exception as e:
        # --- ERROR PRINTING ADDED HERE ---
        st.error(f"❌ Prediction Error: {e}")
        st.session_state.prediction = "rice"
        st.session_state.show_graph = False

if st.session_state.prediction:
    raw_crop = st.session_state.prediction.lower()
    if lang_choice == "Hindi":
        display_crop = crop_map.get(raw_crop, {}).get('hi', raw_crop.title())
    elif lang_choice == "Punjabi":
        display_crop = crop_map.get(raw_crop, {}).get('pun', raw_crop.title())
    else:
        display_crop = raw_crop.title()

    st.success(f"{t['result_header']} {display_crop} 🌾")
    
    # Graph Display
    if st.session_state.show_graph and clf is not None:
        try:
            with st.expander(t['graph_title'], expanded=True):
                st.write("Which factors influenced the AI's decision the most?")
                importances = clf.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X_columns,
                    'Importance': importances
                })
                importance_df = importance_df.sort_values(by='Importance', ascending=True)
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    labels={'Importance': 'Impact Score', 'Feature': 'Factor'}
                )
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Graph Error: {e}")

    # Audio Advice
    if st.button(f"{t['ask_ai_btn']} {display_crop}"):
        with st.spinner("AI Agronomist is preparing advice..."):
            response_text = get_fake_advice(raw_crop, lang_choice)
            st.info(response_text)
            
            try:
                tts_lang = 'hi' if lang_choice != 'English' else 'en'
                tts = gTTS(text=response_text.replace("*", ""), lang=tts_lang, slow=False)
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                # --- ERROR PRINTING ADDED HERE ---
                st.error(f"❌ Audio/TTS Error: {e}")

# ==========================================
# 2. KISAN DHAN - GOVT SCHEMES
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
    with st.spinner("Accessing Government Database..."):
        try:
            response_text = get_fake_schemes(user_state, lang_choice)
            st.warning(response_text)
        except Exception as e:
            st.error(f"❌ Scheme Error: {e}")

st.markdown('<div style="text-align:center; padding:20px; color:grey;">Made with ❤️ by Team Debuggers</div>', unsafe_allow_html=True)

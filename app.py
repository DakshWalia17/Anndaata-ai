import streamlit as st
import google.generativeai as genai

st.title("🤖 API Diagnostic Check")

# 1. Get Key
api_key = st.secrets["GOOGLE_API_KEY"]
st.write(f"Key found: `{api_key[:5]}...`")

# 2. Configure
genai.configure(api_key=api_key)

# 3. List Models
st.write("### Available Models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            st.code(m.name)
except Exception as e:
    st.error(f"Error listing models: {e}")





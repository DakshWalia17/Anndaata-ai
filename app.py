import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
from gtts import gTTS
import io
import PIL.Image

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="AnnDaata AI", page_icon="🌾", layout="wide")

# --- 2. CSS STYLING (Standard Green Theme - FIXED VISIBILITY) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #f0f2f6; }
    
    /* All Headings & Text - Dark Green */
    h1, h2, h3, h4, h5, h6, p, li, span, label, .stMarkdown { 
        color: #0d3b10 !important; 
    }
    
    /* Sidebar Text Fix */
    section[data-testid="stSidebar"] * { 
        color: #0d3b10 !important; 
    }
    
    /* Buttons - Green Background, White Text */
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
    
    /* File Uploader Text Fix */
    div[data-testid="stFileUploader"] label {
        color: #0d3b10 !important;
        font-weight: bold;
    }
    
    /* Input Fields Labels */
    div[data-baseweb="input"] label, div[data-baseweb="slider"] label {
        color: #0d3b10 !important;
    }

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
        "sidebar_title": "⚙️ ਸੈ
















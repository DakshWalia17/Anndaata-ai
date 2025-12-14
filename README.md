# 🌾 AnnDaata AI (Smart Crop Advisor)

**AnnDaata AI** is a "Hybrid AI" agricultural tool designed to help Indian farmers maximize their yield. It combines **Predictive Machine Learning** to recommend crops and **Generative AI** to provide personalized farming guides in local languages.

## 🚀 Key Features

### 1. 🤖 AI Agronomist (Generative AI)
* **Powered by:** Google Gemini 2.5 Flash.
* **Function:** Generates instant, practical farming guides (fertilizer schedules, watering needs, disease management).
* **Multilingual:** Speaks **English, Hindi, and Punjabi** fluently to assist local farmers.

### 2. 🧠 Smart Crop Prediction (Predictive AI)
* **Powered by:** Random Forest Classifier (Scikit-Learn).
* **Function:** Analyzes soil parameters (N, P, K, pH) and weather data to recommend the most profitable crop.
* **Accuracy:** Trained on high-quality agricultural datasets.

### 3. 🇮🇳 Vernacular Interface
* The entire UI adapts to the farmer's preferred language, making technology accessible to rural users.

---

## 🛠️ Tech Stack
* **Frontend:** Streamlit (Python)
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Generative AI:** Google Gemini API (`google-generativeai`)
* **Deployment:** Streamlit Community Cloud

---

## ⚙️ How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/DakshWalia17/Anndaata-ai.git](https://github.com/DakshWalia17/Anndaata-ai.git)
    cd Anndaata-ai
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Key**
    * Get your free key from [Google AI Studio](https://aistudio.google.com/).
    * Create a file named `.streamlit/secrets.toml`.
    * Add your key:
        ```toml
        GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
        ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 🔮 Future Roadmap
* Add computer vision to detect plant diseases from photos.
* Integrate real-time weather forecasting API.
* Add voice-to-text support for illiterate farmers.

---
**Built for the GenAI Hackathon - Chandigarh 2025** 🚀

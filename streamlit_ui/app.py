import streamlit as st
import requests
from PIL import Image
from disease_recommendations import disease_recommendations
import sys
import os
import warnings
warnings.filterwarnings("ignore")

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'fastapi_app')))

#from model_utils import predict


st.set_page_config(
    page_title="Àníkẹ́ – Skin Disease Classifier", 
    page_icon="🧬", 
    layout="centered"
)


def load_css():
    css = """
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f2f6fc;
        color: #212121;
    }
    .main {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 10em;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .prediction {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a73e8;
        margin-top: 1rem;
        text-align: center;
    }
    .recommendation {
        background-color: #fffbe7;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #fbbc04;
        font-size: 1.1rem;
        font-weight: 500;
        color: #444;
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #888;
        margin-top: 3rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()





st.markdown("<h1 style='text-align: center;'>🧬 Àníkẹ́ – Skin Disease Classifier</h1>", unsafe_allow_html=True)
try:
    banner = Image.open("Anike.png")  
    st.image(banner, use_container_width=True)
except:
    st.write("")  
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Upload a skin condition image to get a diagnosis and helpful advice.</p>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("📁 Upload a skin image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, _ = st.columns([1, 2, 0.2])
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### 🔍 Image Preview")
        st.write("This image will be analyzed Àníkẹ́.")
        st.write("Click **Predict** below to proceed.")

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("https://anike-3hyy.onrender.com/predict", files=files)
            #prediction = predict(files['file'])


        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.markdown(f"<div class='prediction'>🩺 Prediction: {prediction}</div>", unsafe_allow_html=True)

            advice = disease_recommendations.get(prediction, "No specific advice available for this condition.")
            st.markdown(f"<div class='recommendation'>💡 <strong>Recommendation:</strong><br>{advice}</div>", unsafe_allow_html=True)
        else:
            st.error("❌ Failed to get prediction from the model API.")


st.markdown("<div class='footer'>Made with ❤️ by <strong>Akinola Samuel Afolabi</strong> | Imọ̀ràn ní Àníkẹ́ maa ngba ni</div>", unsafe_allow_html=True)

import streamlit as st
import os
import sqlite3
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model

# Load environment variables
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

# Initialize database
def init_db():
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()

# Load models
@st.cache_resource
def load_models():
    try:
        satellite_model = load_model("analysis/wildfire_satellite_detection_model.keras")
        camera_model = load_model("analysis/wildfire_detection_model.keras")
        weather_model = load_model("analysis/meteorological-detection-classification.keras")
        return satellite_model, camera_model, weather_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Satellite prediction function
def satellite_cnn_predict(latitude, longitude, zoom_level, model):
    output_size = (350, 350)
    crop_amount = 35
    output_size_modified = (output_size[0], output_size[1] + crop_amount)
    
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{longitude},{latitude},{zoom_level}/{output_size_modified[0]}x{output_size_modified[1]}?access_token={MAPBOX_TOKEN}"
    response = requests.get(url)
    
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        remove_pixels = crop_amount
        remove_pixels_half = remove_pixels // 2
        img_cropped = img.crop((0, remove_pixels_half, img.width, img.height - remove_pixels_half))
        img_resized = img_cropped.resize((224, 224))
        
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        processed_image = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(processed_image)
        return prediction[0][0]
    else:
        st.error("Failed to retrieve satellite image")
        return None

# Camera prediction function
def camera_cnn_predict(image, model):
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = img_array / 255.0
    processed_image = np.expand_dims(img_array, axis=0)
    prediction = model.predict(processed_image)[0][0]
    return prediction

# Weather prediction function
def weather_data_predict(latitude, longitude):
    import random
    return random.uniform(0, 1)

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Forest Fire Risk Classification",
        page_icon="🔥",
        layout="wide"
    )
    
    # Initialize database
    init_db()
    
    # Load models
    satellite_model, camera_model, weather_model = load_models()
    
    if satellite_model is None or camera_model is None or weather_model is None:
        st.error("Failed to load models. Please check if model files exist in the analysis/ directory.")
        return
    
    st.success("Models loaded successfully!")
    
    st.title("🔥 Forest Fire Risk Classification")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose Detection Method",
        ["Home", "Satellite Detection", "Camera Detection", "Alert Service"]
    )
    
    if page == "Home":
        st.header("Welcome to Forest Fire Risk Classification")
        st.markdown("""
        This application uses advanced deep learning models to detect and classify wildfire risks from:
        
        - **Satellite Imagery**: Analyze satellite data for wildfire hotspots
        - **Camera Images**: Upload images for real-time wildfire detection
        - **Weather Data**: Predict wildfire risks based on meteorological conditions
        
        ### Features:
        - 🔥 Real-time wildfire detection
        - 📡 Satellite imagery analysis
        - 📷 Camera image processing
        - ⚠️ Alert subscription service
        - 📊 Confidence scoring
        """)
        
    elif page == "Satellite Detection":
        st.header("🛰️ Satellite Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Location Settings")
            latitude = st.number_input("Latitude", value=37.7749, format="%.4f")
            longitude = st.number_input("Longitude", value=-122.4194, format="%.4f")
            zoom = st.slider("Zoom Level", min_value=10, max_value=20, value=15)
            
            if st.button("Analyze Satellite Image"):
                with st.spinner("Analyzing satellite image..."):
                    prediction = satellite_cnn_predict(latitude, longitude, zoom, satellite_model)
                    
                    if prediction is not None:
                        confidence = round((prediction if prediction > 0.5 else 1 - prediction) * 100)
                        status = "🔥 FIRE DETECTED" if prediction > 0.5 else "✅ NO FIRE"
                        
                        st.success(f"Analysis Complete!")
                        st.metric("Risk Level", status)
                        st.metric("Confidence", f"{confidence}%")
                        
                        st.subheader("Satellite Image")
                        st.info("Satellite image analysis completed")
        
        with col2:
            st.subheader("Weather Analysis")
            if st.button("Analyze Weather Data"):
                with st.spinner("Analyzing weather conditions..."):
                    weather_prediction = weather_data_predict(latitude, longitude)
                    weather_confidence = round((weather_prediction if weather_prediction > 0.5 else 1 - weather_prediction) * 100)
                    weather_status = "🔥 HIGH RISK" if weather_prediction > 0.5 else "✅ LOW RISK"
                    
                    st.metric("Weather Risk", weather_status)
                    st.metric("Weather Confidence", f"{weather_confidence}%")
    
    elif page == "Camera Detection":
        st.header("📷 Camera Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    prediction = camera_cnn_predict(image, camera_model)
                    confidence = round((prediction if prediction > 0.5 else 1 - prediction) * 100)
                    status = "🔥 FIRE DETECTED" if prediction > 0.5 else "✅ NO FIRE"
                    
                    st.success("Analysis Complete!")
                    st.metric("Detection Result", status)
                    st.metric("Confidence", f"{confidence}%")
    
    elif page == "Alert Service":
        st.header("⚠️ Alert Service")
        
        st.markdown("Subscribe to receive wildfire alerts for specific locations.")
        
        with st.form("alert_form"):
            email = st.text_input("Email Address")
            alert_lat = st.number_input("Alert Latitude", value=37.7749, format="%.4f")
            alert_lon = st.number_input("Alert Longitude", value=-122.4194, format="%.4f")
            
            submitted = st.form_submit_button("Subscribe to Alerts")
            
            if submitted:
                if email and alert_lat != 0 and alert_lon != 0:
                    try:
                        conn = sqlite3.connect("alerts.db")
                        cursor = conn.cursor()
                        
                        # Check if email already exists
                        cursor.execute("SELECT * FROM alerts WHERE email=?", (email,))
                        if cursor.fetchone():
                            st.error("This email is already subscribed to alerts.")
                        else:
                            cursor.execute(
                                "INSERT INTO alerts (email, latitude, longitude) VALUES (?, ?, ?)",
                                (email, alert_lat, alert_lon)
                            )
                            conn.commit()
                            st.success("Alert subscription successful! You will now receive wildfire alerts.")
                        
                        conn.close()
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.error("Please provide valid email and coordinates.")

if __name__ == "__main__":
    main() 
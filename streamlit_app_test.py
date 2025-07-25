import streamlit as st
import os
import sys

# Page configuration
st.set_page_config(
    page_title="Forest Fire Risk Classification System",
    page_icon="🔥",
    layout="wide"
)

# Simple test app
def main():
    st.title("🔥 Forest Fire Risk Classification System - Test")
    
    st.write("This is a test version to check if the app loads properly.")
    
    # Check if we can import the required modules
    st.subheader("Module Import Test")
    
    try:
        import pandas as pd
        st.success("✅ pandas imported successfully")
    except Exception as e:
        st.error(f"❌ pandas import failed: {str(e)}")
    
    try:
        import numpy as np
        st.success("✅ numpy imported successfully")
    except Exception as e:
        st.error(f"❌ numpy import failed: {str(e)}")
    
    try:
        import tensorflow as tf
        st.success("✅ tensorflow imported successfully")
    except Exception as e:
        st.error(f"❌ tensorflow import failed: {str(e)}")
    
    try:
        from PIL import Image
        st.success("✅ PIL imported successfully")
    except Exception as e:
        st.error(f"❌ PIL import failed: {str(e)}")
    
    # Check file structure
    st.subheader("File Structure Test")
    
    files_to_check = [
        "streamlit_app.py",
        "src/app.py",
        "src/satellite_functions.py",
        "src/camera_functions.py",
        "src/meteorological_functions.py",
        "analysis/wildfire_satellite_detection_model.keras",
        "analysis/wildfire_detection_model.keras",
        "analysis/meteorological-detection-classification.keras",
        "analysis/std_scaler_weather.pkl"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            st.success(f"✅ {file_path} exists")
        else:
            st.error(f"❌ {file_path} missing")
    
    # Check environment variables
    st.subheader("Environment Variables Test")
    
    mapbox_token = os.getenv("MAPBOX_TOKEN")
    if mapbox_token:
        st.success("✅ MAPBOX_TOKEN is set")
    else:
        st.warning("⚠️ MAPBOX_TOKEN is not set")
    
    mailersend_key = os.getenv("MAILERSEND_KEY")
    if mailersend_key:
        st.success("✅ MAILERSEND_KEY is set")
    else:
        st.warning("⚠️ MAILERSEND_KEY is not set")
    
    # Test basic functionality
    st.subheader("Basic Functionality Test")
    
    if st.button("Test Button"):
        st.success("✅ Button click works!")
    
    # Test file upload
    uploaded_file = st.file_uploader("Test file upload", type=['txt'])
    if uploaded_file is not None:
        st.success("✅ File upload works!")
    
    # Test number input
    test_number = st.number_input("Test number input", value=0)
    st.write(f"Number input value: {test_number}")

if __name__ == "__main__":
    main() 
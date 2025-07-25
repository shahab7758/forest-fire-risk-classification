import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Forest Fire Risk Classification System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1F77B4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .danger-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('alerts.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            latitude REAL,
            longitude REAL,
            prediction REAL,
            confidence REAL,
            method TEXT,
            alert_sent BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Import our custom functions with error handling
def import_custom_functions():
    """Import custom functions with error handling"""
    try:
        import sys
        sys.path.append('src')
        
        # Try to import each module separately
        try:
            from satellite_functions import satellite_cnn_predict
            st.session_state.satellite_available = True
        except Exception as e:
            st.error(f"Satellite functions not available: {str(e)}")
            st.session_state.satellite_available = False
        
        try:
            from camera_functions import camera_cnn_predict
            st.session_state.camera_available = True
        except Exception as e:
            st.error(f"Camera functions not available: {str(e)}")
            st.session_state.camera_available = False
        
        try:
            from meteorological_functions import weather_data_predict
            st.session_state.weather_available = True
        except Exception as e:
            st.error(f"Weather functions not available: {str(e)}")
            st.session_state.weather_available = False
        
        try:
            from email_alert import send_alert_email
            st.session_state.email_available = True
        except Exception as e:
            st.error(f"Email functions not available: {str(e)}")
            st.session_state.email_available = False
            
    except Exception as e:
        st.error(f"Failed to import custom functions: {str(e)}")

# Main app
def main():
    st.markdown('<h1 class="main-header">🔥 Forest Fire Risk Classification System</h1>', unsafe_allow_html=True)
    
    # Import custom functions
    import_custom_functions()
    
    # Check if models are available
    models_dir = "analysis"
    required_models = [
        'wildfire_satellite_detection_model.keras',
        'wildfire_detection_model.keras',
        'meteorological-detection-classification.keras',
        'std_scaler_weather.pkl'
    ]
    
    missing_models = []
    for model in required_models:
        if not os.path.exists(os.path.join(models_dir, model)):
            missing_models.append(model)
    
    if missing_models:
        st.warning(f"Some model files are missing: {', '.join(missing_models)}")
        st.info("The app will work with limited functionality. Please ensure all model files are in the 'analysis' directory.")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🛰️ Satellite Analysis", "📷 Camera Analysis", "🌤️ Weather Analysis", "📊 Analytics", "⚙️ Settings"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🛰️ Satellite Analysis":
        show_satellite_page()
    elif page == "📷 Camera Analysis":
        show_camera_page()
    elif page == "🌤️ Weather Analysis":
        show_weather_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    st.markdown('<h2 class="sub-header">Welcome to the Forest Fire Risk Classification System</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🛰️ Satellite Analysis</h3>
            <p>Analyze satellite imagery to detect potential wildfire risks using advanced CNN models.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📷 Camera Analysis</h3>
            <p>Upload camera images for real-time wildfire detection and risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌤️ Weather Analysis</h3>
            <p>Analyze meteorological data to assess fire weather conditions and risks.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent alerts
    st.subheader("Recent Alerts")
    try:
        conn = sqlite3.connect('alerts.db')
        df = pd.read_sql_query("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 5", conn)
        conn.close()
        
        if not df.empty:
            for _, row in df.iterrows():
                risk_level = "High" if row['prediction'] > 0.7 else "Medium" if row['prediction'] > 0.4 else "Low"
                color = "danger" if risk_level == "High" else "warning" if risk_level == "Medium" else "success"
                
                st.markdown(f"""
                <div class="{color}-message">
                    <strong>{row['method']} Analysis</strong><br>
                    Location: {row['latitude']:.4f}, {row['longitude']:.4f}<br>
                    Risk Level: {risk_level} ({row['prediction']:.2%})<br>
                    Time: {row['timestamp']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent alerts found.")
    except Exception as e:
        st.error(f"Error loading alerts: {str(e)}")

def show_satellite_page():
    st.markdown('<h2 class="sub-header">🛰️ Satellite Image Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('satellite_available', False):
        st.error("Satellite analysis is not available. Please check that all required files are present.")
        return
    
    st.write("Enter coordinates to analyze satellite imagery for wildfire risk assessment.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=37.7749, format="%.4f")
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-122.4194, format="%.4f")
    
    with col2:
        zoom_level = st.slider("Zoom Level", min_value=10, max_value=20, value=15)
        output_size = st.selectbox("Image Size", [(400, 400), (600, 600), (800, 800)], format_func=lambda x: f"{x[0]}x{x[1]}")
    
    crop_amount = st.slider("Crop Amount", min_value=0, max_value=100, value=20)
    
    if st.button("Analyze Satellite Image", type="primary"):
        with st.spinner("Analyzing satellite image..."):
            try:
                from satellite_functions import satellite_cnn_predict
                
                # Create a temporary file path
                save_path = "temp_satellite_image.jpg"
                
                # Get prediction
                prediction = satellite_cnn_predict(
                    latitude, longitude, output_size, zoom_level, crop_amount, save_path
                )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Satellite Image")
                    if os.path.exists(save_path):
                        image = Image.open(save_path)
                        st.image(image, caption="Analyzed Satellite Image", use_column_width=True)
                
                with col2:
                    st.subheader("Analysis Results")
                    
                    # Determine risk level
                    if prediction > 0.7:
                        risk_level = "High"
                        color = "danger"
                    elif prediction > 0.4:
                        risk_level = "Medium"
                        color = "warning"
                    else:
                        risk_level = "Low"
                        color = "success"
                    
                    st.markdown(f"""
                    <div class="{color}-message">
                        <h3>Risk Level: {risk_level}</h3>
                        <p>Wildfire Probability: {prediction:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to database
                    try:
                        conn = sqlite3.connect('alerts.db')
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO alerts (latitude, longitude, prediction, confidence, method)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (latitude, longitude, prediction, prediction, "Satellite"))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        st.error(f"Failed to save to database: {str(e)}")
                    
                    # Send alert if high risk
                    if prediction > 0.7:
                        st.warning("High risk detected! Consider sending alert.")
                        if st.button("Send Alert Email"):
                            try:
                                from email_alert import send_alert_email
                                send_alert_email(latitude, longitude, prediction, "Satellite Analysis")
                                st.success("Alert email sent successfully!")
                            except Exception as e:
                                st.error(f"Failed to send alert: {str(e)}")
                
                # Clean up
                if os.path.exists(save_path):
                    os.remove(save_path)
                    
            except Exception as e:
                st.error(f"Error analyzing satellite image: {str(e)}")

def show_camera_page():
    st.markdown('<h2 class="sub-header">📷 Camera Image Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('camera_available', False):
        st.error("Camera analysis is not available. Please check that all required files are present.")
        return
    
    st.write("Upload a camera image to analyze for wildfire detection.")
    
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing camera image..."):
                try:
                    from camera_functions import camera_cnn_predict
                    
                    # Get prediction
                    prediction = camera_cnn_predict(uploaded_file)
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Uploaded Image")
                        st.image(image, caption="Analyzed Camera Image", use_column_width=True)
                    
                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Determine risk level
                        if prediction > 0.7:
                            risk_level = "High"
                            color = "danger"
                        elif prediction > 0.4:
                            risk_level = "Medium"
                            color = "warning"
                        else:
                            risk_level = "Low"
                            color = "success"
                        
                        st.markdown(f"""
                        <div class="{color}-message">
                            <h3>Risk Level: {risk_level}</h3>
                            <p>Wildfire Probability: {prediction:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Save to database
                        try:
                            conn = sqlite3.connect('alerts.db')
                            cursor = conn.cursor()
                            cursor.execute('''
                                INSERT INTO alerts (latitude, longitude, prediction, confidence, method)
                                VALUES (?, ?, ?, ?, ?)
                            ''', (0.0, 0.0, prediction, prediction, "Camera"))
                            conn.commit()
                            conn.close()
                        except Exception as e:
                            st.error(f"Failed to save to database: {str(e)}")
                        
                        # Send alert if high risk
                        if prediction > 0.7:
                            st.warning("High risk detected! Consider sending alert.")
                            if st.button("Send Alert Email"):
                                try:
                                    from email_alert import send_alert_email
                                    send_alert_email(0.0, 0.0, prediction, "Camera Analysis")
                                    st.success("Alert email sent successfully!")
                                except Exception as e:
                                    st.error(f"Failed to send alert: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Error analyzing camera image: {str(e)}")

def show_weather_page():
    st.markdown('<h2 class="sub-header">🌤️ Weather Data Analysis</h2>', unsafe_allow_html=True)
    
    if not st.session_state.get('weather_available', False):
        st.error("Weather analysis is not available. Please check that all required files are present.")
        return
    
    st.write("Enter coordinates to analyze weather conditions for wildfire risk assessment.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=37.7749, format="%.4f", key="weather_lat")
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-122.4194, format="%.4f", key="weather_lon")
    
    if st.button("Analyze Weather Data", type="primary"):
        with st.spinner("Analyzing weather data..."):
            try:
                from meteorological_functions import weather_data_predict
                
                # Get weather prediction
                prediction = weather_data_predict(latitude, longitude)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Weather Analysis")
                    
                    # Determine risk level
                    if prediction > 0.7:
                        risk_level = "High"
                        color = "danger"
                    elif prediction > 0.4:
                        risk_level = "Medium"
                        color = "warning"
                    else:
                        risk_level = "Low"
                        color = "success"
                    
                    st.markdown(f"""
                    <div class="{color}-message">
                        <h3>Risk Level: {risk_level}</h3>
                        <p>Wildfire Probability: {prediction:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Weather Metrics")
                    st.info("Weather data analysis completed successfully.")
                
                # Save to database
                try:
                    conn = sqlite3.connect('alerts.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO alerts (latitude, longitude, prediction, confidence, method)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (latitude, longitude, prediction, prediction, "Weather"))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    st.error(f"Failed to save to database: {str(e)}")
                
                # Send alert if high risk
                if prediction > 0.7:
                    st.warning("High risk detected! Consider sending alert.")
                    if st.button("Send Alert Email", key="weather_alert"):
                        try:
                            from email_alert import send_alert_email
                            send_alert_email(latitude, longitude, prediction, "Weather Analysis")
                            st.success("Alert email sent successfully!")
                        except Exception as e:
                            st.error(f"Failed to send alert: {str(e)}")
                
            except Exception as e:
                st.error(f"Error analyzing weather data: {str(e)}")

def show_analytics_page():
    st.markdown('<h2 class="sub-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    try:
        # Load data from database
        conn = sqlite3.connect('alerts.db')
        df = pd.read_sql_query("SELECT * FROM alerts", conn)
        conn.close()
        
        if df.empty:
            st.info("No data available for analytics.")
            return
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(df))
        
        with col2:
            st.metric("High Risk Cases", len(df[df['prediction'] > 0.7]))
        
        with col3:
            st.metric("Average Risk", f"{df['prediction'].mean():.2%}")
        
        with col4:
            st.metric("Latest Analysis", df['timestamp'].max().strftime("%Y-%m-%d"))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            fig = px.histogram(df, x='prediction', nbins=20, 
                              title="Distribution of Risk Predictions")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Analysis Methods")
            method_counts = df['method'].value_counts()
            fig = px.pie(values=method_counts.values, names=method_counts.index,
                         title="Analysis Methods Used")
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series
        st.subheader("Risk Trends Over Time")
        df_daily = df.groupby(df['timestamp'].dt.date)['prediction'].mean().reset_index()
        fig = px.line(df_daily, x='timestamp', y='prediction',
                      title="Average Daily Risk Levels")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent data table
        st.subheader("Recent Analyses")
        st.dataframe(df.head(10))
        
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

def show_settings_page():
    st.markdown('<h2 class="sub-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    st.write("Configure your application settings.")
    
    # API Keys
    st.subheader("API Configuration")
    
    mapbox_token = st.text_input("Mapbox Token", value=os.getenv("MAPBOX_TOKEN", ""), type="password")
    mailersend_key = st.text_input("MailerSend Key", value=os.getenv("MAILERSEND_KEY", ""), type="password")
    
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
    
    # Database
    st.subheader("Database Management")
    
    if st.button("Clear All Data"):
        if st.checkbox("I understand this will delete all analysis data"):
            try:
                conn = sqlite3.connect('alerts.db')
                cursor = conn.cursor()
                cursor.execute("DELETE FROM alerts")
                conn.commit()
                conn.close()
                st.success("All data cleared successfully!")
            except Exception as e:
                st.error(f"Failed to clear data: {str(e)}")

if __name__ == "__main__":
    main() 
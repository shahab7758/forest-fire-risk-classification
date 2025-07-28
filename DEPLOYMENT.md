# Deployment Guide

## Streamlit Cloud Deployment

This application can be deployed on Streamlit Cloud. Here's how to set it up:

### 1. Repository Structure
Make sure your repository has the following structure:
```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/config.toml   # Streamlit configuration
├── analysis/                 # Model files
│   ├── wildfire_satellite_detection_model.keras
│   ├── wildfire_detection_model.keras
│   └── meteorological-detection-classification.keras
├── src/                      # Flask app (for local development)
│   ├── app.py               # Flask application
│   ├── satellite_functions.py
│   ├── camera_functions.py
│   └── meteorological_functions.py
└── alerts.db                # Database file (will be created automatically)
```

### 2. Environment Variables
Set up the following environment variables in Streamlit Cloud:
- `MAPBOX_TOKEN`: Your Mapbox API token for satellite imagery

### 3. Deployment Steps
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to: `app.py`
5. Add your environment variables
6. Deploy!

### 4. Local Development

#### Streamlit Version (for Streamlit Cloud)
```bash
streamlit run app.py
```

#### Flask Version (for local development)
```bash
python3 src/app.py
```

### 5. Troubleshooting
- Make sure all model files are included in the repository
- Ensure the MAPBOX_TOKEN environment variable is set
- Check that all dependencies are listed in requirements.txt
- For Flask version, ensure all src/ files are present

## Features Available

### Streamlit Version (app.py)
- ✅ Interactive web interface
- ✅ Satellite image analysis
- ✅ Camera image upload and analysis
- ✅ Weather data analysis
- ✅ Alert subscription service
- ✅ Real-time predictions
- ✅ Confidence scoring
- ✅ Flask API documentation

### Flask Version (src/app.py)
- ✅ RESTful API endpoints
- ✅ Background task scheduling
- ✅ Email alert system
- ✅ Database management
- ✅ Multiple detection methods
- ✅ Web interface with templates

## File Descriptions

### app.py (Main for Streamlit Cloud)
- Streamlit-based web application
- Interactive interface for all detection methods
- Model loading and prediction functions
- Database management
- API documentation

### src/app.py (Flask Version)
- Flask-based web application
- RESTful API endpoints
- Background task scheduling
- Email alert system
- Template-based web interface 
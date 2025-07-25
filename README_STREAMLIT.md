# Forest Fire Risk Classification System - Streamlit Deployment

This is the Streamlit version of the Forest Fire Risk Classification System, designed for easy deployment on Streamlit Cloud.

## Features

- 🛰️ **Satellite Image Analysis**: Analyze satellite imagery for wildfire detection
- 📷 **Camera Image Analysis**: Upload and analyze camera images for real-time detection
- 🌤️ **Weather Data Analysis**: Analyze meteorological conditions for fire risk assessment
- 📊 **Analytics Dashboard**: View trends and statistics of all analyses
- 📧 **Email Alerts**: Automated alert system for high-risk detections

## Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository

1. Make sure your code is in a GitHub repository
2. Ensure all model files are included in the repository
3. Verify that `streamlit_app.py` is in the root directory

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the path to your Streamlit app: `streamlit_app.py`
6. Click "Deploy"

### Step 3: Configure Environment Variables

In your Streamlit Cloud app settings, add these environment variables:

```
MAPBOX_TOKEN=your_mapbox_token_here
MAILERSEND_KEY=your_mailersend_key_here
```

### Step 4: Deploy

Click "Deploy" and wait for the build to complete.

## Local Development

To run the app locally:

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run streamlit_app.py
```

The app will be available at `http://localhost:8501`

## File Structure

```
├── streamlit_app.py              # Main Streamlit application
├── requirements_streamlit.txt     # Python dependencies
├── .streamlit/                   # Streamlit configuration
│   └── config.toml
├── src/                          # Source code
│   ├── satellite_functions.py
│   ├── camera_functions.py
│   ├── meteorological_functions.py
│   └── email_alert.py
├── analysis/                     # Model files
│   ├── wildfire_satellite_detection_model.keras
│   ├── wildfire_detection_model.keras
│   └── meteorological-detection-classification.keras
└── alerts.db                     # SQLite database
```

## Usage

1. **Home Page**: Overview of the system and recent alerts
2. **Satellite Analysis**: Enter coordinates to analyze satellite imagery
3. **Camera Analysis**: Upload images for real-time detection
4. **Weather Analysis**: Analyze weather conditions for fire risk
5. **Analytics**: View trends and statistics
6. **Settings**: Configure API keys and manage data

## API Keys Required

- **Mapbox Token**: For satellite imagery access
- **MailerSend Key**: For email alerts

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure all model files are in the `analysis/` directory
2. **API Key Errors**: Verify environment variables are set correctly
3. **Memory Issues**: The models are large, ensure adequate memory allocation

### Support

For issues with the Streamlit deployment, check:
- Streamlit Cloud logs in your app dashboard
- GitHub repository for the latest code
- Environment variable configuration

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- The `.env` file is excluded from git via `.gitignore` 
# Model Files Setup

Due to GitHub's file size limitations (100MB max), the large model files have been excluded from this repository. You need to download them separately for the application to work.

## Required Model Files

Place these files in the `analysis/` directory:

1. **wildfire_satellite_detection_model.keras** (130.56 MB)
   - Used for satellite image analysis
   - CNN model trained on satellite imagery

2. **wildfire_detection_model.keras** (96.64 MB)
   - Used for camera image analysis
   - CNN model trained on camera images

3. **meteorological-detection-classification.keras** (927 KB)
   - Used for weather data analysis
   - Neural network model for meteorological predictions

4. **std_scaler_weather.pkl** (1.2 KB)
   - Standard scaler for weather data preprocessing

## How to Get the Model Files

### Option 1: Download from Original Source
If you have access to the original model files, simply copy them to the `analysis/` directory.

### Option 2: Train New Models
You can train new models using the Jupyter notebooks in the `analysis/` directory:
- `wildfire-satellite-detection.ipynb`
- `wildfire-camera-detection.ipynb`
- `meteorological-detection-classification.ipynb`

### Option 3: Use Alternative Models
You can replace the models with your own trained models, ensuring they have the same input/output format.

## File Structure After Setup

```
analysis/
├── wildfire_satellite_detection_model.keras
├── wildfire_detection_model.keras
├── meteorological-detection-classification.keras
├── std_scaler_weather.pkl
├── wildfire-satellite-detection.ipynb
├── wildfire-camera-detection.ipynb
├── meteorological-detection-classification.ipynb
└── small datasets/
    └── forestfire-classification.csv
```

## Verification

After placing the model files, you can verify the setup by running:

```bash
# Test the Flask app
cd src && python3 -c 'from app import init_db; init_db()'

# Test the Streamlit app
streamlit run streamlit_app.py
```

## Deployment Notes

For Streamlit Cloud deployment:
1. The model files will need to be included in your deployment
2. Consider using external storage (Google Drive, AWS S3, etc.) for the large models
3. Update the model loading paths in the code if using external storage

## Troubleshooting

If you encounter model loading errors:
1. Verify all model files are in the `analysis/` directory
2. Check file permissions
3. Ensure the model files are not corrupted
4. Verify the model loading paths in the code match your file structure 
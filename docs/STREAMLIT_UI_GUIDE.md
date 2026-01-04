# 🎯 Billboard Display System - Streamlit UI

## 📺 Overview

Professional billboard-style UI for displaying targeted ads in shopping malls with intelligent recommendation system.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- streamlit>=1.29.0
- pillow>=10.0.0
- (other dependencies already in requirements.txt)

### 2. Prepare Ad Images

**Image Location:** `ad_images/` directory

**Naming Convention:** Images must be named with their Product ID (PID)
```
ad_images/
├── HIRHV4SV2LFAQJDM.jpg
├── BZT1XKCN3XL8YXXF.png
└── ... (more images)
```

**Image Specifications:**
- Format: `.jpg`, `.png`, `.jpeg`, or `.webp`
- Dimensions: 1920x1080 (recommended) or 1280x720 (minimum)
- Aspect Ratio: 16:9 (landscape)
- File Size: Under 5MB recommended

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Features

### 🎬 Display Ads Mode
- Upload CSV with target criteria or use default file
- Auto-play mode with smooth transitions
- Navigation: Previous/Next ad controls
- Real-time weather integration
- Match score display (X/5 categories)
- Landscape image display with billboard frame

### 🤖 Predict Categories Mode
- Enter new ad title
- AI predicts: Age Group, Gender, Mood, Weather
- Option to save predictions to database
- Visual results in card format

### 🌤️ Test Weather Mode
- Check weather for any city
- Real-time weather data via OpenWeatherMap
- Temperature, condition, and category display

### 📊 Statistics Mode
- Database overview (total ads, unique titles)
- Distribution charts for all categories
- Age groups, genders, moods, weather, ad types

## 🎨 UI Design Features

### Billboard-Style Display
- Professional gradient backgrounds
- Landscape image frames (16:9)
- Large, readable text for distance viewing
- Smooth transitions between ads
- Color-coded status indicators

### Layout
- **Main Area (60%):** Ad image in landscape frame
- **Info Panel (40%):** Ad details, target audience, criteria
- **Sidebar:** System controls and status
- **Header:** Branded title with gradient

### Color Scheme
- Primary: Purple gradient (#667eea → #764ba2)
- Success: Green (#27ae60)
- Warning: Orange (#f39c12)
- Info: Gray (#ecf0f1)

## 📁 Project Structure

```
AdResearch_Classification_Model/
├── app.py                     # Main Streamlit application
├── manage_ad_images.py        # Helper script for image management
├── ad_images/                 # Ad images directory (IMPORTANT!)
│   ├── README.md             # Image placement instructions
│   ├── {PID}.jpg             # Ad images named by PID
│   └── ...
├── data/
│   └── example_target_values.csv
├── models/                    # Trained ML models
├── src/                       # Core modules
│   ├── classifier.py
│   ├── recommendation_engine.py
│   └── weather_service.py
└── Classification model dataset.csv
```

## 🔧 Image Management Helper

Use the helper script to manage ad images:

```bash
python manage_ad_images.py
```

Options:
1. **Extract all PIDs to file** - Get list of all Product IDs
2. **Check for missing images** - Find which PIDs need images
3. **Show sample PIDs with titles** - See PID-to-title mapping
4. **Create rename script template** - Batch rename existing images
5. **Run all checks** - Complete analysis

## 📝 CSV Format for Target Values

```csv
target_age_group,target_gender,target_mood,ad_type
Kids,Male,Happy,generic ads
18-39,Female,neutral,premium ads
65+,Male,Happy,attention grabbing ads
```

**Required columns:**
- `target_age_group`: Kids, 10-18, 18-39, 40-64, 65+
- `target_gender`: Male, Female
- `target_mood`: Happy, neutral, Angry

**Optional columns:**
- `ad_type`: generic ads, attention grabbing ads, premium ads

## 🎛️ System Controls

### Sidebar Navigation
- **🎬 Display Ads:** Main ad display mode
- **🤖 Predict Categories:** Classify new ad titles
- **🌤️ Test Weather:** Check weather service
- **📊 Statistics:** View database stats
- **🚪 EXIT:** Close application

### Display Controls
- **⏮️ Previous Ad:** Go to previous ad
- **⏭️ Next Ad:** Go to next ad
- **Ad Number:** Jump to specific ad
- **🔄 Auto-play:** Enable automatic progression (3 sec)

## 🌐 Weather Integration

The system fetches real-time weather and adjusts ad recommendations.

**Setup Weather API:**
1. Get free API key from [OpenWeatherMap](https://openweathermap.org/api)
2. Create `.env` file in project root:
   ```
   OPENWEATHER_API_KEY=your_api_key_here
   ```
3. Restart the application

**Without API key:** System uses default "sunny" weather.

## 📊 Match Score System

The system scores ads based on how many criteria match:

**With ad_type:** X/5 match
- Age Group (1 point)
- Gender (1 point)
- Mood (1 point)
- Weather (1 point)
- Ad Type (1 point)

**Without ad_type:** X/4 match
- Age Group (1 point)
- Gender (1 point)
- Mood (1 point)
- Weather (1 point)

## 🔍 Troubleshooting

### Images Not Displaying
1. Check images are in `ad_images/` directory
2. Verify filename matches PID exactly (case-sensitive)
3. Confirm image format is .jpg, .png, .jpeg, or .webp
4. Run `python manage_ad_images.py` → Option 2 to check missing images

### Model Not Loading
1. Ensure `models/` directory contains:
   - ad_classifier_model.pkl
   - vectorizer.pkl
   - label_encoders.pkl
   - model_metadata.pkl
2. If missing, train model using `train_model_colab.py`

### Ads Database Not Loading
1. Check `Classification model dataset.csv` exists in root directory
2. Verify CSV has required columns: pid, ad_title, target_age_group, target_gender, target_mood, target_weather
3. Check for CSV formatting errors

### Weather Service Not Working
1. Check `.env` file exists with valid API key
2. Verify internet connection
3. System will use "sunny" as default if weather unavailable

## 🎯 Best Practices for Shopping Mall Display

### Image Quality
- Use high-resolution images (1920x1080 minimum)
- Ensure good lighting and contrast
- Avoid cluttered or busy images
- Use professional product photography

### Display Settings
- Enable auto-play mode for continuous display
- Set 3-5 second intervals between ads
- Use fullscreen mode (F11 in browser)
- Disable browser UI for clean display

### Content Management
- Update CSV file regularly with new target criteria
- Add seasonal ad_types (holiday, summer, winter)
- Monitor match scores - aim for 4/5 or 5/5 matches
- Remove outdated ads from database

## 🚀 Deployment Options

### Local Network (Recommended for Shopping Mall)
1. Run on dedicated display computer
2. Access via local network: `http://[computer-ip]:8501`
3. Connect to large display screens

### Cloud Deployment
- Deploy to Streamlit Cloud (free)
- Deploy to Heroku, AWS, or Azure
- Use for remote management

### Kiosk Mode
```bash
# Run in kiosk mode (fullscreen, no browser UI)
streamlit run app.py --server.headless=true
```

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review `ad_images/README.md` for image setup
3. Run `manage_ad_images.py` for diagnostics

## 🎉 Tips for Best Results

1. **Prepare images in advance** - Use `manage_ad_images.py` to identify missing images
2. **Test with sample data** - Use default CSV before creating custom targets
3. **Monitor match scores** - Low scores indicate missing ads for certain criteria
4. **Use weather integration** - More relevant ads based on current conditions
5. **Update regularly** - Add new ads and images to keep content fresh

---

**Enjoy your smart billboard display system! 🎯📺**

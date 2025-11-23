# 🎯 Ad Classification & Recommendation System

A smart billboard advertisement suggestion system for shopping malls that uses machine learning to classify ads and recommend the best-fit advertisements based on target demographics and real-time weather conditions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

## 🌟 Overview

This system provides two main functionalities:

1. **Ad Classification**: Uses ML to automatically predict target categories (age group, gender, mood, weather) for new ad titles
2. **Ad Recommendation**: Matches target demographics with stored ads and displays the best-fit advertisement

The system is designed for shopping mall billboard displays, automatically rotating ads every 3 seconds based on target criteria.

## ✨ Features

- 🤖 **Machine Learning Classification**: Predicts target categories for new ads using trained Random Forest model
- 🎯 **Smart Recommendation**: Matches ads with target demographics using similarity scoring
- 🌤️ **Weather Integration**: Fetches real-time weather via OpenWeatherMap API
- 📊 **CSV Processing**: Reads target values row-by-row for automated ad display
- ⏱️ **Timed Display**: Shows each ad for 3 seconds (configurable)
- 📈 **Match Scoring**: Displays how well each ad matches target criteria (0-4 score)
- 💻 **Terminal-Based**: No GUI required - runs entirely in terminal

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Application (main.py)                │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
┌───────▼──────┐ ┌──▼────────┐ ┌▼──────────────┐
│ Classifier   │ │  Weather  │ │ Recommendation│
│   Module     │ │  Service  │ │    Engine     │
└──────────────┘ └───────────┘ └───────────────┘
        │              │               │
        │              │               │
┌───────▼──────┐ ┌────▼─────┐  ┌──────▼────────┐
│ ML Models    │ │ Weather  │  │  Ads Database │
│ (.pkl files) │ │   API    │  │   (CSV)       │
└──────────────┘ └──────────┘  └───────────────┘
```

### Workflow

```
Start → Load Ads Database → Read Target CSV Row → Fetch Weather 
  → Build Target Profile → Find Best Ad → Display Ad (3s) 
  → More Rows? → Yes (loop) / No (End)
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for weather API)

### Step 1: Clone or Download the Project

```bash
cd /path/to/AdResearch_Classification_Model
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Weather API key:
   ```
   WEATHER_API_KEY=your_actual_api_key_here
   DEFAULT_CITY=YourCity
   DEFAULT_COUNTRY=YourCountry
   ```

   Get a free API key at: https://openweathermap.org/api

## 🎓 Model Training

The machine learning model must be trained in Google Colab before using the system.

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `train_model_colab.py` or copy its contents into a new notebook

### Step 2: Run Training Script

1. Execute all cells in the notebook
2. Upload your `Classification model dataset.csv` when prompted
3. Wait for training to complete (may take 5-10 minutes)

### Step 3: Download Model Files

After training, download these 4 files:
- ✅ `ad_classifier_model.pkl`
- ✅ `vectorizer.pkl`
- ✅ `label_encoders.pkl`
- ✅ `model_metadata.pkl`

### Step 4: Place Model Files

Move the downloaded files to your project's `models/` directory:

```
AdResearch_Classification_Model/
└── models/
    ├── ad_classifier_model.pkl    ← Place here
    ├── vectorizer.pkl              ← Place here
    ├── label_encoders.pkl          ← Place here
    └── model_metadata.pkl          ← Place here
```

### Expected Model Performance

- Overall Accuracy: ~85-95% (depends on dataset)
- Target Categories:
  - Age Group: Kids, 10-18, 18-39, 40-64, 65+
  - Gender: Male, Female
  - Mood: Happy, Angry, Sad, Neutral
  - Weather: sunny, rainy, cold

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Required: Weather API Key
WEATHER_API_KEY=your_openweathermap_api_key

# Optional: Default Location
DEFAULT_CITY=London
DEFAULT_COUNTRY=UK
```

### Display Duration

To change ad display duration, modify in `main.py`:

```python
self.display_duration = 3  # Change to desired seconds
```

## 🚀 Usage

### Mode 1: Process Target CSV File

Display ads based on target values from CSV:

```bash
python main.py data/example_target_values.csv
```

**CSV Format:**
```csv
pid,ad_title,target_age_group,target_gender,target_mood,target_weather
TARGET001,New Product,18-39,Female,happy,sunny
TARGET002,Another Ad,40-64,Male,neutral,rainy
```

**Note**: `target_weather` can be left empty - system will use current weather from API.

### Mode 2: Predict Categories for New Ad

Classify a new ad title:

```bash
python main.py --predict "Women's Fashion Leggings"
```

Output:
```
🔮 Predicting categories for: Women's Fashion Leggings

✅ Prediction Results:
   👥 Age Group: 18-39
   👤 Gender: Female
   😊 Mood: neutral
   🌤️  Weather: sunny
```

### Mode 3: Interactive Mode

Run without arguments for interactive menu:

```bash
python main.py
```

Interactive Options:
1. Display ads from target CSV file
2. Predict categories for a new ad title
3. Test weather service
4. View database statistics
5. Exit

### Help

```bash
python main.py --help
```

## 📁 Project Structure

```
AdResearch_Classification_Model/
├── main.py                              # Main application entry point
├── train_model_colab.py                 # Google Colab training script
├── requirements.txt                     # Python dependencies
├── .env                                 # Environment variables (create from .env.example)
├── .env.example                         # Example environment file
├── .gitignore                          # Git ignore rules
├── README.md                           # This file
│
├── src/                                # Source code modules
│   ├── classifier.py                   # ML classification module
│   ├── weather_service.py             # Weather API integration
│   └── recommendation_engine.py       # Ad recommendation logic
│
├── models/                             # Trained model files (from Colab)
│   ├── ad_classifier_model.pkl        # Main ML model
│   ├── vectorizer.pkl                 # Text vectorizer
│   ├── label_encoders.pkl             # Label encoders
│   └── model_metadata.pkl             # Model metadata
│
├── data/                               # Data files
│   └── example_target_values.csv      # Example target values
│
├── docs/                               # Documentation
│   ├── ARCHITECTURE.md                # System architecture
│   ├── API_REFERENCE.md               # API documentation
│   └── WORKFLOW.md                    # Workflow diagrams
│
└── Classification model dataset.csv    # Main ads database
```

## 📚 API Reference

### AdClassifier

```python
from src.classifier import AdClassifier

classifier = AdClassifier(model_dir="models")
classifier.load_model()

# Predict categories
prediction = classifier.predict("Women's Leggings")
# Returns: {'target_age_group': '18-39', 'target_gender': 'Female', ...}
```

### WeatherService

```python
from src.weather_service import WeatherService

weather = WeatherService()

# Get categorized weather
category = weather.get_categorized_weather(city="London")
# Returns: "sunny", "rainy", or "cold"

# Get detailed info
info = weather.get_detailed_weather_info(city="London")
# Returns: {'city': 'London', 'temperature': 15.5, 'category': 'sunny', ...}
```

### AdRecommendationEngine

```python
from src.recommendation_engine import AdRecommendationEngine

engine = AdRecommendationEngine()
engine.load_ads_database()

# Find best ad
target = {
    'target_age_group': '18-39',
    'target_gender': 'Female',
    'target_mood': 'neutral',
    'target_weather': 'sunny'
}

ad = engine.find_best_ad(target)
# Returns: {'pid': '...', 'ad_title': '...', 'match_score': 4, ...}
```

## 🔧 Troubleshooting

### Model Files Not Found

**Error**: `❌ Error: Missing model files`

**Solution**: Train the model using `train_model_colab.py` in Google Colab and download the files to `models/` directory.

### Weather API Error

**Error**: `⚠️ Weather API key not configured`

**Solution**: Add your API key to `.env` file:
```bash
WEATHER_API_KEY=your_actual_key_here
```

Get free key at: https://openweathermap.org/api

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### CSV Format Error

**Error**: `❌ Error: Missing required columns`

**Solution**: Ensure your target CSV has these columns:
- `target_age_group`
- `target_gender`
- `target_mood`
- `target_weather` (optional - can be empty)

### Low Model Accuracy

**Problem**: Model predictions are inaccurate

**Solution**:
1. Ensure you trained on the full dataset
2. Check dataset quality and balance
3. Retrain with more data if available
4. Adjust model parameters in `train_model_colab.py`

## 📊 Dataset Format

### Ads Database CSV

The main ads database (`Classification model dataset.csv`) should have:

```csv
flipkart_with_targets
pid,ad_title,target_age_group,target_gender,target_mood,target_weather
PROD001,Women's Leggings,18-39,Female,neutral,sunny
PROD002,Kids Toy Car,Kids,Male,happy,sunny
...
```

### Target Values CSV

Target values for ad display:

```csv
pid,ad_title,target_age_group,target_gender,target_mood,target_weather
TARGET001,Request 1,18-39,Female,happy,
TARGET002,Request 2,40-64,Male,neutral,rainy
```

**Note**: Leave `target_weather` empty to use current weather from API.

## 🤝 Contributing

This is a research project. For questions or issues, please refer to the documentation files in the `docs/` directory.

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- Dataset: Flipkart product listings
- Weather Data: OpenWeatherMap API
- ML Framework: scikit-learn
- Text Processing: TF-IDF Vectorization

---

**Last Updated**: November 2025

For detailed technical documentation, see:
- `docs/ARCHITECTURE.md` - System architecture details
- `docs/API_REFERENCE.md` - Complete API reference
- `docs/WORKFLOW.md` - Workflow diagrams and processes
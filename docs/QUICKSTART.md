# Quick Start Guide

Get your Ad Classification & Recommendation System up and running in minutes!

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (1 minute)

```bash
cd AdResearch_Classification_Model
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Environment (30 seconds)

```bash
cp .env.example .env
# Edit .env and add your Weather API key
# Get free key at: https://openweathermap.org/api
```

### Step 3: Train Model (See COLAB_GUIDE.md)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `train_model_colab.py`
3. Run all cells (~10 minutes)
4. Download 4 model files to `models/` directory

### Step 4: Test System (30 seconds)

```bash
python main.py --predict "Women's Fashion Leggings"
```

**Expected output:**
```
✅ Model loaded successfully!
🔮 Predicting: Women's Fashion Leggings
   Age Group: 18-39
   Gender: Female
   Mood: neutral
   Weather: sunny
```

### Step 5: Run Ad Display System

```bash
python main.py data/example_target_values.csv
```

---

## 📚 Documentation Quick Links

### For Users
- **[README.md](../README.md)** - Complete user guide and features
- **[COLAB_GUIDE.md](COLAB_GUIDE.md)** - Step-by-step model training

### For Developers
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and components
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- **[WORKFLOW.md](WORKFLOW.md)** - Process workflows and diagrams

---

## 🎯 Common Tasks

### Display Ads from CSV
```bash
python main.py your_target_values.csv
```

### Predict Ad Categories
```bash
python main.py --predict "Your Ad Title Here"
```

### Interactive Mode
```bash
python main.py
# Select from menu options
```

### Get Help
```bash
python main.py --help
```

---

## 📁 Project Structure Overview

```
AdResearch_Classification_Model/
├── main.py                          # Run this!
├── train_model_colab.py            # Upload to Google Colab
├── requirements.txt                # Dependencies
├── .env                            # Your config (API keys)
│
├── src/                            # Core modules
│   ├── classifier.py               # ML predictions
│   ├── weather_service.py         # Weather API
│   └── recommendation_engine.py   # Ad matching
│
├── models/                         # Put downloaded .pkl files here
│   ├── ad_classifier_model.pkl
│   ├── vectorizer.pkl
│   ├── label_encoders.pkl
│   └── model_metadata.pkl
│
├── data/
│   └── example_target_values.csv  # Sample file
│
└── docs/                           # Documentation
    ├── ARCHITECTURE.md
    ├── API_REFERENCE.md
    ├── WORKFLOW.md
    └── COLAB_GUIDE.md
```

---

## ⚡ Quick Reference

### Target Categories

| Category | Values |
|----------|--------|
| Age Group | Kids, 10-18, 18-39, 40-64, 65+ |
| Gender | Male, Female |
| Mood | Happy, Angry, Sad, Neutral |
| Weather | sunny, rainy, cold |

### CSV Format

```csv
pid,ad_title,target_age_group,target_gender,target_mood,target_weather
REQ001,Product 1,18-39,Female,happy,sunny
REQ002,Product 2,40-64,Male,neutral,
```

**Note:** Leave `target_weather` empty to use current weather from API.

---

## 🔧 Troubleshooting

### Model files not found?
→ Train model in Google Colab (see COLAB_GUIDE.md)

### Weather API not working?
→ Add API key to `.env` file

### Import errors?
→ Run `pip install -r requirements.txt`

### Low accuracy?
→ Check dataset quality, retrain with more data

---

## 📞 Next Steps

1. ✅ Read [README.md](../README.md) for complete features
2. ✅ Train model using [COLAB_GUIDE.md](COLAB_GUIDE.md)
3. ✅ Configure weather API in `.env`
4. ✅ Test with example CSV: `python main.py data/example_target_values.csv`
5. ✅ Create your own target CSV and run system
6. ✅ Explore API reference for custom integrations

---

## 💡 Tips

- **Start Simple**: Use interactive mode to test features
- **Test First**: Try with example CSV before creating your own
- **Monitor Accuracy**: Check model performance regularly
- **Update Often**: Retrain model monthly with new data
- **Save Logs**: Keep track of which ads perform best

---

**Need Help?** See detailed documentation in `docs/` folder or README.md

**Last Updated**: November 2025

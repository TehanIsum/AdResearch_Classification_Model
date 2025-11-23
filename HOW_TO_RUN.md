# 🚀 HOW TO RUN THIS PROJECT

## Quick Start (5 Minutes)

### Step 1: Open Terminal
```bash
# Navigate to project directory
cd /Users/tehanisum/Documents/AdResearch_Classification_Model
```

### Step 2: Activate Virtual Environment
```bash
# Activate the Python virtual environment
source .venv/bin/activate
```

### Step 3: Run the Application
```bash
# Start the system
python main.py
```

### Step 4: Choose an Option
```
The interactive menu will appear:

======================================================================
🎮 INTERACTIVE MODE
======================================================================

Options:
1. Display ads from target CSV file
2. Predict categories for a new ad title
3. Test weather service
4. View database statistics
5. Exit

Enter your choice (1-5):
```

---

## 📋 Detailed Instructions

### Option 1: Display Ads from CSV File

**What it does:** Reads target values from a CSV file and displays matching ads

**How to use:**
```bash
# Start the application
python main.py

# Choose option 1
Enter your choice (1-5): 1

# Enter CSV file path (or press Enter for example)
Enter path to target CSV file (or press Enter for example): [Press Enter]

# System will:
# - Read each row from CSV
# - Fetch current weather
# - Find best matching ad
# - Display ad for 3 seconds
# - Move to next row
```

**CSV Format Required:**
```csv
target_age_group,target_gender,target_mood,target_weather
18-39,Female,neutral,sunny
Kids,Male,Happy,sunny
40-64,Male,neutral,cold
```

**Example:**
```bash
cd /Users/tehanisum/Documents/AdResearch_Classification_Model
source .venv/bin/activate
python main.py
# Choose: 1
# Press: Enter (uses default example)
```

---

### Option 2: Predict Categories for New Ad

**What it does:** Takes an ad title and predicts its target categories

**How to use:**
```bash
# Start the application
python main.py

# Choose option 2
Enter your choice (1-5): 2

# Enter an ad title
Enter ad title to classify: Women's Fashion Dress

# System will predict:
# - Age Group (Kids, 10-18, 18-39, 40-64, 65+)
# - Gender (Male, Female)
# - Mood (Happy, Angry, Sad, neutral)
# - Weather (sunny, rainy, cold)
```

**Example:**
```bash
cd /Users/tehanisum/Documents/AdResearch_Classification_Model
source .venv/bin/activate
python main.py
# Choose: 2
# Type: Kids Toy Car Racing Set
# Press: Enter

# Output:
# Age Group: Kids
# Gender: Male
# Mood: neutral
# Weather: sunny
```

---

### Option 3: Test Weather Service

**What it does:** Fetches current weather from API

**How to use:**
```bash
# Start the application
python main.py

# Choose option 3
Enter your choice (1-5): 3

# System will:
# - Try to fetch weather from OpenWeatherMap API
# - Display current weather in Colombo, Sri Lanka
# - If no API key: Shows default (sunny)
```

**Note:** Requires API key in `.env` file (optional)

---

### Option 4: View Database Statistics

**What it does:** Shows statistics about loaded ads

**How to use:**
```bash
# Start the application
python main.py

# Choose option 4
Enter your choice (1-5): 4

# System shows:
# - Total number of ads (20,000)
# - Distribution by age group
# - Distribution by gender
# - Distribution by mood
# - Distribution by weather
```

---

### Option 5: Exit

**What it does:** Closes the application

```bash
Enter your choice (1-5): 5
```

---

## 🎯 Common Use Cases

### Use Case 1: Quick Prediction Test

**Scenario:** You want to quickly test what categories a new ad would get

```bash
# 1. Navigate to project
cd /Users/tehanisum/Documents/AdResearch_Classification_Model

# 2. Activate environment
source .venv/bin/activate

# 3. Run application
python main.py

# 4. Choose option 2
2

# 5. Enter your ad title
Women's Summer Sandals

# 6. See predictions
Age Group: 18-39
Gender: Female
Mood: neutral
Weather: sunny

# 7. Exit
5
```

---

### Use Case 2: Process Multiple Target Values

**Scenario:** You have a CSV with target values and want to see recommended ads

**Step 1: Create your CSV file**
```bash
# Create a file: my_targets.csv
cat > my_targets.csv << 'EOF'
target_age_group,target_gender,target_mood,target_weather
18-39,Female,neutral,sunny
Kids,Male,Happy,sunny
40-64,Male,neutral,cold
EOF
```

**Step 2: Run the application**
```bash
# Start application
python main.py

# Choose option 1
1

# Enter your CSV path
my_targets.csv

# System will process each row and display matching ads
```

---

### Use Case 3: Classify Multiple Ads

**Scenario:** You want to classify several new ad titles

```bash
# Method 1: Interactive (one at a time)
python main.py
# Choose: 2
# Enter: Women's Dress
# Choose: 2 (again)
# Enter: Kids Toy
# Choose: 5 (exit)

# Method 2: Using Python script
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from src.classifier import AdClassifier

classifier = AdClassifier()
classifier.load_model()

ads = [
    "Women's Fashion Dress",
    "Kids Toy Car",
    "Men's Winter Jacket",
    "Teen Backpack"
]

for ad in ads:
    result = classifier.predict(ad)
    print(f"\n{ad}:")
    print(f"  Age: {result['target_age_group']}")
    print(f"  Gender: {result['target_gender']}")
    print(f"  Mood: {result['target_mood']}")
    print(f"  Weather: {result['target_weather']}")
EOF
```

---

## 🔧 Advanced Usage

### Run as Python Module

```python
# Import the system
from main import AdDisplaySystem

# Initialize
system = AdDisplaySystem()
system.initialize()

# Classify an ad
result = system.classifier.predict("Women's Fashion Dress")
print(result)

# Find best ad for target
target = {
    'target_age_group': '18-39',
    'target_gender': 'Female',
    'target_mood': 'neutral',
    'target_weather': 'sunny'
}
best_ad = system.recommendation_engine.find_best_ad(target)
print(best_ad)
```

---

### Use Individual Components

#### 1. Classification Only
```python
from src.classifier import AdClassifier

# Load and use classifier
classifier = AdClassifier()
classifier.load_model()

# Predict categories
result = classifier.predict("Women's Summer Dress")
print(result)
# Output: {
#   'target_age_group': '18-39',
#   'target_gender': 'Female',
#   'target_mood': 'neutral',
#   'target_weather': 'sunny'
# }
```

#### 2. Recommendation Only
```python
from src.recommendation_engine import AdRecommendationEngine

# Load recommendation engine
engine = AdRecommendationEngine()
engine.load_ads_database()

# Find best ad
target = {
    'target_age_group': '18-39',
    'target_gender': 'Female',
    'target_mood': 'neutral',
    'target_weather': 'sunny'
}

best_ad = engine.find_best_ad(target)
print(f"Best ad: {best_ad['ad_title']}")
print(f"Match score: {best_ad['match_score']}/4")
```

#### 3. Weather Service Only
```python
from src.weather_service import WeatherService

# Initialize weather service
weather = WeatherService()

# Get current weather
current = weather.get_current_weather()
print(f"Weather: {current.get('weather_category', 'sunny')}")
```

---

## 🛠️ Troubleshooting

### Issue 1: "No such file or directory"

**Error:**
```
cd: no such file or directory: /Users/tehanisum/Documents/AdResearch_Classification_Model
```

**Solution:**
```bash
# Find your project location
cd ~/Documents
ls -la | grep AdResearch

# Or use full path from Finder
cd /path/to/your/project
```

---

### Issue 2: "Command not found: python"

**Error:**
```
zsh: command not found: python
```

**Solution:**
```bash
# Use python3 instead
python3 main.py

# Or create alias
alias python=python3
```

---

### Issue 3: Virtual Environment Not Activated

**Error:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
# Make sure to activate virtual environment
source .venv/bin/activate

# You should see (.venv) in prompt:
# (.venv) user@computer AdResearch_Classification_Model %

# Then run:
python main.py
```

---

### Issue 4: Model Files Not Found

**Error:**
```
❌ Error: Model files not found in models/
```

**Solution:**
```bash
# Check if model files exist
ls -la models/

# Should show:
# ad_classifier_model.pkl
# vectorizer.pkl
# label_encoders.pkl
# model_metadata.pkl

# If missing, they need to be downloaded from Google Colab training
```

---

### Issue 5: Weather API Not Working

**Error:**
```
⚠️ Weather service not available - will use default 'sunny'
```

**Solution:**
```bash
# This is NORMAL if you haven't set up Weather API key
# System will use default 'sunny' - this is expected

# To add API key (optional):
# 1. Get key from: https://openweathermap.org/api
# 2. Edit .env file:
echo "WEATHER_API_KEY=your_key_here" >> .env
```

---

## 📊 Expected Output Examples

### Example 1: Successful Classification

```
======================================================================
🎯 AD CLASSIFICATION & RECOMMENDATION SYSTEM
======================================================================

🔧 Initializing system components...

📦 Loading trained model...
✅ Model loaded successfully!
   - Overall accuracy: 95.75%
   - Features: 9557

📦 Loading ads database from Classification model dataset.csv...
✅ Loaded 20000 ads from database

✅ System initialized successfully!

======================================================================
🎮 INTERACTIVE MODE
======================================================================

Enter your choice (1-5): 2

Enter ad title to classify: Women's Summer Dress

📊 CLASSIFICATION RESULTS:
====================================
Age Group:  18-39
Gender:     Female
Mood:       neutral
Weather:    sunny
====================================
```

---

### Example 2: Successful Recommendation

```
Enter your choice (1-5): 1

Enter path to target CSV file: [Press Enter]

📋 Processing target values CSV...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Row 1/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Target Values:
   Age Group: 18-39
   Gender:    Female
   Mood:      neutral
   Weather:   sunny

🔍 Finding best matching ad...

✅ Best Match Found!

======================================================================
📺 DISPLAYING AD
======================================================================

🎯 AD TITLE: Alisha Solid Women's Cycling Shorts
📋 Product ID: SRTEH2FF9KEDEFGF
⭐ Match Score: 4/4 categories matched

📊 TARGET AUDIENCE:
   👥 Age Group: 18-39
   👤 Gender: Female
   😊 Mood: neutral
   🌤️  Weather: sunny

⏱️  Displaying for 3 seconds...
```

---

## 🎯 Performance Tips

### Tip 1: Keep Virtual Environment Activated

```bash
# Instead of activating every time, keep terminal open
source .venv/bin/activate

# Run multiple times without re-activating
python main.py
python main.py
python main.py
```

### Tip 2: Use History

```bash
# Use up arrow to recall previous commands
# Or use Ctrl+R to search history

# Example:
# Press Ctrl+R
# Type: python main
# Press Enter
```

### Tip 3: Create Alias

```bash
# Add to ~/.zshrc
alias run-ads='cd /Users/tehanisum/Documents/AdResearch_Classification_Model && source .venv/bin/activate && python main.py'

# Then just type:
run-ads
```

---

## 📝 Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│  🚀 QUICK START COMMANDS                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Navigate:                                           │
│     cd /Users/tehanisum/Documents/...                   │
│                                                          │
│  2. Activate:                                           │
│     source .venv/bin/activate                           │
│                                                          │
│  3. Run:                                                │
│     python main.py                                      │
│                                                          │
│  OPTIONS:                                               │
│  ├─ 1: Process CSV file                                │
│  ├─ 2: Classify single ad                              │
│  ├─ 3: Test weather                                    │
│  ├─ 4: View statistics                                 │
│  └─ 5: Exit                                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## ✅ Pre-Flight Checklist

Before running, make sure:

- [x] ✅ In correct directory: `/Users/tehanisum/Documents/AdResearch_Classification_Model`
- [x] ✅ Virtual environment activated: `source .venv/bin/activate`
- [x] ✅ Model files present in `models/` directory
- [x] ✅ Dataset CSV file present: `Classification model dataset.csv`
- [ ] ⏳ Weather API key set (optional): in `.env` file

---

## 🎉 You're Ready!

### One-Line Command to Run:

```bash
cd /Users/tehanisum/Documents/AdResearch_Classification_Model && source .venv/bin/activate && python main.py
```

### Or Step-by-Step:

```bash
# Step 1
cd /Users/tehanisum/Documents/AdResearch_Classification_Model

# Step 2
source .venv/bin/activate

# Step 3
python main.py
```

---

**Happy advertising! 🎯📺**

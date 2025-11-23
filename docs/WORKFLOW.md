# Workflow Documentation

## System Workflows

This document describes the various workflows and processes in the Ad Classification & Recommendation System.

## 1. Initial Setup Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Environment Setup                                   │
│ - Install Python 3.8+                                       │
│ - Clone/download project                                    │
│ - Create virtual environment                                │
│ - Install dependencies (pip install -r requirements.txt)    │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Configure Environment                               │
│ - Copy .env.example to .env                                 │
│ - Add Weather API key                                       │
│ - Set default location (optional)                           │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Train ML Model (Google Colab)                       │
│ - Open train_model_colab.py in Colab                        │
│ - Upload dataset CSV                                        │
│ - Run all cells                                             │
│ - Download 4 model files                                    │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 4: Deploy Model Files                                  │
│ - Place files in models/ directory:                         │
│   • ad_classifier_model.pkl                                 │
│   • vectorizer.pkl                                          │
│   • label_encoders.pkl                                      │
│   • model_metadata.pkl                                      │
└───────────────────────┬─────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 5: Test System                                         │
│ - Run: python main.py                                       │
│ - Select interactive mode option 4 (database stats)         │
│ - Verify ads database loaded                                │
│ - Test weather service (option 3)                           │
│ - Test ad prediction (option 2)                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. Model Training Workflow (Google Colab)

```
START
  ↓
┌─────────────────────────────────────────┐
│ 1. Install Dependencies                 │
│    - scikit-learn, pandas, nltk, joblib │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 2. Upload Dataset                        │
│    - Classification model dataset.csv    │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 3. Data Preprocessing                    │
│    - Load CSV                            │
│    - Remove duplicates                   │
│    - Clean text                          │
│    - Check class distributions           │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 4. Feature Engineering                   │
│    - TF-IDF Vectorization                │
│      • max_features: 5000                │
│      • ngram_range: (1, 3)               │
│    - Label Encoding (4 categories)       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 5. Train-Test Split                      │
│    - 80% training                        │
│    - 20% testing                         │
│    - Stratified by age group             │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 6. Model Training                        │
│    - Multi-Output Random Forest          │
│    - 200 estimators                      │
│    - Max depth: 30                       │
│    - Training time: 5-10 minutes         │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 7. Model Evaluation                      │
│    - Calculate accuracy per category     │
│    - Generate classification reports     │
│    - Compute overall metrics             │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 8. Test Predictions                      │
│    - Sample ad titles                    │
│    - Verify predictions make sense       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 9. Save Model Files                      │
│    - Serialize all components            │
│    - Package metadata                    │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 10. Download Files                       │
│     - ad_classifier_model.pkl            │
│     - vectorizer.pkl                     │
│     - label_encoders.pkl                 │
│     - model_metadata.pkl                 │
└─────────────────────────────────────────┘
  ↓
END
```

## 3. Ad Display Workflow (Main Use Case)

```
START: python main.py target_values.csv
  ↓
┌─────────────────────────────────────────┐
│ 1. System Initialization                │
│    - Load ads database (CSV)            │
│    - Initialize weather service         │
│    - Check API key configuration        │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 2. Load Target Values CSV               │
│    - Read CSV file                      │
│    - Validate columns                   │
│    - Count rows to process              │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 3. Fetch Current Weather                │
│    - API call to OpenWeatherMap         │
│    - Parse response                     │
│    - Categorize (sunny/rainy/cold)      │
│    - Cache for session                  │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 4. Process First Row                    │
│    - Read target values                 │
│    - Fill missing weather with current  │
│    - Build target profile               │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 5. Find Best Matching Ad                │
│    - Calculate match scores (0-4)       │
│    - Select highest scoring ad          │
│    - Retrieve ad details                │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 6. Display Ad in Terminal               │
│    - Format ad information              │
│    - Show match score                   │
│    - Display target vs actual           │
│    - Countdown timer (3 seconds)        │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 7. Check for More Rows                  │
│    - Any rows remaining?                │
└────────┬───────┴───────┬────────────────┘
         │ YES           │ NO
         ↓               ↓
    Go to Step 4    Display completion
                        message
                          ↓
                         END
```

## 4. Ad Classification Workflow (Prediction)

```
START: python main.py --predict "Ad Title"
  ↓
┌─────────────────────────────────────────┐
│ 1. Initialize Classifier                │
│    - Create AdClassifier instance       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 2. Load Model Files                     │
│    - Load ad_classifier_model.pkl       │
│    - Load vectorizer.pkl                │
│    - Load label_encoders.pkl            │
│    - Load model_metadata.pkl            │
│    - Validate all files present         │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 3. Preprocess Ad Title                  │
│    - Convert to lowercase              │
│    - Remove special characters          │
│    - Clean whitespace                   │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 4. Vectorize Text                       │
│    - Apply TF-IDF transformation        │
│    - Generate feature vector            │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 5. Make Prediction                      │
│    - Feed to multi-output classifier    │
│    - Get encoded predictions (4 values) │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 6. Decode Predictions                   │
│    - Age group: encoded → label         │
│    - Gender: encoded → label            │
│    - Mood: encoded → label              │
│    - Weather: encoded → label           │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ 7. Display Results                      │
│    - Show original ad title             │
│    - Show predicted categories          │
│    - Format output                      │
└─────────────────────────────────────────┘
  ↓
END
```

## 5. Interactive Mode Workflow

```
START: python main.py
  ↓
┌─────────────────────────────────────────┐
│ System Initialization                   │
│ - Load ads database                     │
│ - Check weather service                 │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Display Menu                            │
│ 1. Display ads from CSV                 │
│ 2. Predict categories for ad            │
│ 3. Test weather service                 │
│ 4. View database statistics             │
│ 5. Exit                                 │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ User Selection                          │
└──┬──┬──┬──┬──┬───────────────────────────┘
   │  │  │  │  │
   1  2  3  4  5
   │  │  │  │  │
   │  │  │  │  └──────────► EXIT
   │  │  │  │
   │  │  │  └─────► Show database stats
   │  │  │           - Total ads
   │  │  │           - Distribution by category
   │  │  │           └─► Back to menu
   │  │  │
   │  │  └────────► Test weather
   │  │              - Enter city (optional)
   │  │              - Fetch & display weather
   │  │              - Show categorization
   │  │              └─► Back to menu
   │  │
   │  └───────────► Predict ad categories
   │                - Enter ad title
   │                - Load model (if needed)
   │                - Show predictions
   │                └─► Back to menu
   │
   └──────────────► Display ads from CSV
                    - Enter CSV path
                    - Run display workflow
                    └─► Back to menu
```

## 6. Error Handling Workflow

```
┌─────────────────────────────────────────┐
│ Operation Attempted                     │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Check Prerequisites                     │
│ - Files exist?                          │
│ - API keys configured?                  │
│ - Data valid?                           │
└─────┬──────────────────────┬────────────┘
      │ OK                   │ ERROR
      ↓                      ↓
┌─────────────┐    ┌─────────────────────┐
│ Continue    │    │ Identify Error Type │
│ Operation   │    └──────────┬──────────┘
└─────────────┘               ↓
                    ┌─────────────────────┐
                    │ Log Error Message   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Check if Recoverable│
                    └──┬──────────────┬───┘
                       │ YES          │ NO
                       ↓              ↓
            ┌──────────────┐  ┌──────────────┐
            │ Use Fallback │  │ Display Error│
            │ - Default    │  │ - User Action│
            │   values     │  │   Required   │
            │ - Continue   │  │ - Exit       │
            └──────────────┘  └──────────────┘
```

## 7. Weather Integration Workflow

```
┌─────────────────────────────────────────┐
│ Weather Needed                          │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Check API Key                           │
└─────┬──────────────────────┬────────────┘
      │ Configured           │ Missing
      ↓                      ↓
┌─────────────────┐  ┌───────────────────┐
│ Make API Call   │  │ Use Default       │
└────────┬────────┘  │ weather = "sunny" │
         ↓           └───────────────────┘
┌─────────────────────────────────────────┐
│ Parse API Response                      │
│ - Weather main (rain, snow, etc.)       │
│ - Description                           │
│ - Temperature                           │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Categorize Weather                      │
│                                         │
│ IF rain/drizzle/thunderstorm:           │
│   → "rainy"                             │
│ ELSE IF snow/ice OR temp < 10°C:        │
│   → "cold"                              │
│ ELSE:                                   │
│   → "sunny"                             │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Cache Weather Category                  │
│ (for current session)                   │
└─────────────────────────────────────────┘
```

## 8. Ad Matching Algorithm

```
Input: Target Values
  ↓
┌─────────────────────────────────────────┐
│ Initialize: best_score = 0              │
│             best_ad = None              │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ FOR EACH ad in database:                │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ Calculate Match Score                   │
│ score = 0                               │
│                                         │
│ IF ad.age_group == target.age_group:    │
│   score += 1                            │
│                                         │
│ IF ad.gender == target.gender:          │
│   score += 1                            │
│                                         │
│ IF ad.mood == target.mood:              │
│   score += 1                            │
│                                         │
│ IF ad.weather == target.weather:        │
│   score += 1                            │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ IF score > best_score:                  │
│   best_score = score                    │
│   best_ad = current_ad                  │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│ More ads to check?                      │
└─────┬────────────────┬──────────────────┘
      │ YES            │ NO
      ↓                ↓
  Loop back      ┌─────────────────┐
                 │ Return best_ad  │
                 │ with best_score │
                 └─────────────────┘
```

## 9. Daily Operations Workflow

```
Morning Setup:
  ↓
1. System Check
   - Verify services running
   - Check API quota
   - Validate data files
   ↓
2. Prepare Target CSV
   - Export from system
   - Format correctly
   - Save to data/
   ↓
3. Run Display System
   python main.py data/today_targets.csv
   ↓
4. Monitor Output
   - Check match scores
   - Verify ads displaying
   - Note any errors
   ↓
5. Log Results
   - Ads displayed
   - Match rates
   - Any issues
```

## 10. Maintenance Workflow

```
Weekly:
  - Review API usage
  - Check model performance
  - Update target values format
  
Monthly:
  - Analyze ad performance
  - Consider model retraining
  - Update ads database
  
Quarterly:
  - Retrain model with new data
  - Update dependencies
  - Review system metrics
  
Yearly:
  - Major model refresh
  - Technology stack update
  - Performance optimization
```

---

**Last Updated**: November 2025

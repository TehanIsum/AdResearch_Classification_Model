# 🎓 Complete Training Guide for Ad Classification Model

## 📋 Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Google Colab Setup](#google-colab-setup)
4. [Step-by-Step Training Process](#step-by-step-training-process)
5. [Understanding the Output Files](#understanding-the-output-files)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## 🎯 Overview

This guide explains how to train a **Multi-Output Classification Model** that predicts 4 target attributes for advertisements based on the ad title:

| Target Attribute | Possible Values |
|-----------------|----------------|
| **Age Group** | Kids, 10-18, 18-39, 40-64, 65+ |
| **Gender** | Male, Female |
| **Mood** | Happy, Angry, Sad, Neutral |
| **Weather** | sunny, rainy, cold |

**Example Prediction:**
```
Input:  "Women's Summer Beach Dress"
Output: Age: 18-39, Gender: Female, Mood: Happy, Weather: sunny
```

---

## ✅ Prerequisites

### Required Items
- [ ] **Google Account** (for Google Colab access)
- [ ] **Dataset CSV file** with these columns:
  - `ad_title`: The advertisement title (text)
  - `target_age_group`: Target age group (categorical)
  - `target_gender`: Target gender (categorical)
  - `target_mood`: Target mood (categorical)
  - `target_weather`: Target weather (categorical)
- [ ] **Internet connection** (for Colab and downloads)

### Recommended Dataset
Use `Classification_model_dataset_FINAL_BALANCED.csv` for best results:
- ✅ Pre-balanced weather classes (31-34% each)
- ✅ 8,340 training samples
- ✅ No imbalance issues

---

## 🚀 Google Colab Setup

### Step 1: Access Google Colab

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/)
2. Sign in with your Google account

### Step 2: Upload Training Script

**Option A: Direct Upload**
1. Click **File → Upload notebook**
2. Select `train_model_colab.py`
3. Rename it to `train_model_colab.ipynb` if prompted

**Option B: Open from GitHub**
1. Click **File → Open notebook**
2. Select **GitHub** tab
3. Enter repository URL
4. Select `train_model_colab.py`

**Option C: Create New Notebook**
1. Click **File → New notebook**
2. Copy entire contents of `train_model_colab.py`
3. Paste into code cell

### Step 3: Connect to Runtime

1. Click **Runtime → Change runtime type**
2. Select:
   - **Runtime type**: Python 3
   - **Hardware accelerator**: None (CPU is sufficient)
   - **Runtime shape**: Standard
3. Click **Save**
4. Click **Connect** button (top-right)

---

## 📚 Step-by-Step Training Process

### STEP 1: Setup and Installation

**What happens:**
- Installs required Python packages
- Downloads NLTK language resources

**Code:**
```python
!pip install scikit-learn pandas numpy nltk joblib -q
```

**Expected Output:**
```
Installing required packages...
Downloading NLTK data...
✅ Setup complete!
```

**Duration:** ~30 seconds

---

### STEP 2: Upload Dataset

**What happens:**
- Opens file upload dialog
- Loads your CSV file into Colab

**Interactive Action Required:**
1. Click **Choose Files** button
2. Select your dataset CSV
3. Wait for upload (progress bar appears)

**Expected Output:**
```
📤 Please upload your dataset CSV file
✅ Loaded: Classification_model_dataset_FINAL_BALANCED.csv
```

**Duration:** 10-60 seconds (depends on file size)

---

### STEP 3: Data Loading and Preprocessing

**What happens:**
- Reads CSV into memory
- Validates required columns exist
- Removes duplicate entries
- Cleans text (lowercase, remove special chars)
- Analyzes class distributions

**Code Explanation:**
```python
# Load data
df = pd.read_csv(dataset_name)

# Remove duplicates (keeps same ad with different targets)
df = df.drop_duplicates(subset=['ad_title', 'target_age_group', 
                                 'target_gender', 'target_mood', 
                                 'target_weather'])

# Clean text
df['ad_title_clean'] = df['ad_title'].apply(clean_text)
```

**Expected Output:**
```
📊 Dataset shape: (8340, 6)
📊 Total rows loaded: 8340 samples

🧹 Removed 145 TRUE duplicate rows

📊 Weather distribution AFTER deduplication:
   sunny: 2678 (32.1%)
   rainy: 2601 (31.2%)
   cold: 2916 (35.0%)

✅ Weather distribution is reasonably balanced
```

**Key Metrics to Check:**
- Total samples: Should be 5,000+
- Weather distribution: Each class should be 25-40%
- ⚠️ If one class > 60%, dataset is imbalanced

**Duration:** ~5-10 seconds

---

### STEP 4: Feature Engineering

**What happens:**
- Converts text to numerical features using TF-IDF
- Encodes categorical labels to numbers

**TF-IDF Vectorization:**
```python
vectorizer = TfidfVectorizer(
    max_features=10000,      # Keep top 10,000 words
    ngram_range=(1, 3),      # Use 1-3 word phrases
    min_df=3,                # Word appears in 3+ docs
    max_df=0.85,             # Word in max 85% of docs
)
```

**Example Transformation:**
```
Input:  "Women's Summer Dress"
Output: [0.0, 0.0, 0.3, 0.0, 0.7, 0.0, ...] (10,000 numbers)
         ↑               ↑        ↑
      "the"          "summer"  "dress"
    (common)      (important) (important)
```

**Label Encoding:**
```python
Before: ["Happy", "Sad", "Happy", "Neutral"]
After:  [0, 1, 0, 2]

Mapping saved for later:
  Happy → 0
  Sad → 1
  Neutral → 2
  Angry → 3
```

**Expected Output:**
```
🔤 Creating TF-IDF features...
✅ Feature matrix shape: (8340, 10000)
   (samples × features)

🏷️  Encoding target labels...
   target_age_group: 5 classes → ['10-18' '18-39' '40-64' '65+' 'Kids']
   target_gender: 2 classes → ['Female' 'Male']
   target_mood: 4 classes → ['Angry' 'Happy' 'Neutral' 'Sad']
   target_weather: 3 classes → ['cold' 'rainy' 'sunny']
```

**Duration:** ~10-15 seconds

---

### STEP 5: Train-Test Split

**What happens:**
- Divides data into training (80%) and testing (20%)
- Ensures balanced distribution (stratification)

**Code:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y_encoded, 
    test_size=0.2,           # 20% for testing
    random_state=42,          # Reproducible results
    stratify=y_encoded['target_age_group']  # Balanced split
)
```

**Expected Output:**
```
📊 Training set size: 6672 samples (80%)
📊 Testing set size: 1668 samples (20%)
```

**Why This Matters:**
- **Training set**: Model learns patterns from this data
- **Testing set**: Evaluates performance on unseen data
- **Prevents overfitting**: Model can't memorize test data

**Duration:** < 1 second

---

### STEP 6: Model Training

**What happens:**
- Trains 4 Random Forest models (one per target)
- Each forest contains 200 decision trees
- Takes the longest time

**Architecture:**
```
Input Features (10,000 TF-IDF values)
        ↓
┌─────────────────────────┐
│ Random Forest 1 (200 trees) │ → Predicts Age Group
│ Random Forest 2 (200 trees) │ → Predicts Gender
│ Random Forest 3 (200 trees) │ → Predicts Mood
│ Random Forest 4 (200 trees) │ → Predicts Weather
└─────────────────────────┘
        ↓
4 Predictions Simultaneously
```

**Configuration:**
```python
base_classifier = RandomForestClassifier(
    n_estimators=200,        # 200 trees per forest
    max_depth=30,            # Max tree depth
    min_samples_split=10,    # Min samples to split
    class_weight='balanced', # Handle imbalanced classes
    n_jobs=-1,               # Use all CPU cores
)
```

**Expected Output:**
```
🤖 Training Multi-Output Random Forest Classifier...
   Optimized for large dataset (20,000 samples)...
   This may take 10-15 minutes...

[Parallel(n_jobs=-1)]: Using backend ThreadingBackend
[Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  2.5min
[Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  8.3min
[Parallel(n_jobs=-1)]: Done 200 out of 200 | elapsed:  9.1min finished

✅ Model training complete!
```

**Duration:** 5-15 minutes (depends on dataset size)

**Progress Indicators:**
- Watch for parallel task updates
- CPU usage will spike (normal)
- Memory usage: 1-3 GB

---

### STEP 7: Model Evaluation

**What happens:**
- Makes predictions on test set
- Calculates accuracy for each target
- Shows detailed performance metrics

**Metrics Explained:**

| Metric | Meaning | Good Score |
|--------|---------|-----------|
| **Accuracy** | Overall correct predictions | > 70% |
| **Precision** | Of predictions for class X, how many were correct? | > 0.7 |
| **Recall** | Of actual class X items, how many did we find? | > 0.7 |
| **F1-Score** | Harmonic mean of precision & recall | > 0.7 |

**Expected Output:**
```
📊 MODEL PERFORMANCE:

============================================================
📌 TARGET_AGE_GROUP
============================================================
Accuracy: 0.8523 (85.23%)

Detailed Report:
              precision    recall  f1-score   support

        Kids       0.88      0.91      0.89       320
       10-18       0.83      0.79      0.81       335
       18-39       0.85      0.87      0.86       340
       40-64       0.84      0.83      0.83       338
         65+       0.86      0.86      0.86       335

    accuracy                           0.85      1668
   macro avg       0.85      0.85      0.85      1668
weighted avg       0.85      0.85      0.85      1668

============================================================
📌 TARGET_GENDER
============================================================
Accuracy: 0.9235 (92.35%)

[Similar output for Gender, Mood, Weather...]

============================================================
📊 OVERALL AVERAGE ACCURACY: 0.8342 (83.42%)
============================================================
```

**How to Interpret:**
- **> 80% accuracy**: Excellent performance
- **70-80% accuracy**: Good performance
- **< 70% accuracy**: Needs improvement (check data quality)

**Duration:** ~10-20 seconds

---

### STEP 8: Sample Predictions

**What happens:**
- Tests model on example ad titles
- Shows real-world predictions

**Expected Output:**
```
🧪 Testing predictions on sample ad titles:

📝 Ad Title: FDT Women's Leggings
   Age Group: 18-39
   Gender: Female
   Mood: Happy
   Weather: cold

📝 Ad Title: Kids Toy Car Racing Set
   Age Group: Kids
   Gender: Male
   Mood: Happy
   Weather: sunny

📝 Ad Title: Men's Formal Business Shirt
   Age Group: 40-64
   Gender: Male
   Mood: Neutral
   Weather: sunny

📝 Ad Title: Senior Citizen Walking Stick
   Age Group: 65+
   Gender: Male
   Mood: Neutral
   Weather: sunny

📝 Ad Title: Teen Fashion Backpack
   Age Group: 10-18
   Gender: Female
   Mood: Happy
   Weather: sunny
```

**Validation Checklist:**
- ✅ "Women's" → Female
- ✅ "Kids Toy" → Kids age group
- ✅ "Senior Citizen" → 65+ age group
- ✅ Results make logical sense

**Duration:** < 1 second

---

### STEP 9: Save Model Files

**What happens:**
- Serializes trained model components
- Saves to Colab storage
- Creates 4 .pkl files

**Files Created:**

| File | Size | Contents |
|------|------|----------|
| `ad_classifier_model.pkl` | 50-200 MB | 4 trained Random Forests (800 trees total) |
| `vectorizer.pkl` | 5-20 MB | TF-IDF vocabulary & parameters |
| `label_encoders.pkl` | 1 KB | Category → number mappings |
| `model_metadata.pkl` | 1 KB | Training statistics & accuracy scores |

**Expected Output:**
```
💾 Saving model files...
✅ Saved: ad_classifier_model.pkl
✅ Saved: vectorizer.pkl
✅ Saved: label_encoders.pkl
✅ Saved: model_metadata.pkl
```

**Duration:** ~5-10 seconds

---

### STEP 10: Download Files

**What happens:**
- Triggers browser downloads for all 4 files
- Files save to your Downloads folder

**Interactive Action Required:**
1. Browser shows 4 download prompts
2. Click **Save** or **Keep** for each file
3. Wait for all downloads to complete

**Expected Output:**
```
📥 Downloading files...

[Browser download prompts appear]

============================================================
✅ TRAINING COMPLETE!
============================================================

📁 IMPORTANT: Place the downloaded files in your project:

   Your Project Root/
   └── models/
       ├── ad_classifier_model.pkl    ← Place here
       ├── vectorizer.pkl              ← Place here
       ├── label_encoders.pkl          ← Place here
       └── model_metadata.pkl          ← Place here

🚀 Next Steps:
   1. Download all 4 files using the browser download prompt
   2. Move them to the 'models/' folder in your project
   3. Run the main application: python main.py

📊 Model Summary:
   - Training samples: 6672
   - Test accuracy: 83.42%
   - Features: 10000
```

**Duration:** 10-60 seconds (depends on file size & internet speed)

---

## 📦 Understanding the Output Files

### 1. ad_classifier_model.pkl

**Purpose:** Core prediction engine

**Contains:**
- 4 trained Random Forest models
- 800 decision trees total (200 per target)
- Tree structures, split points, leaf values

**File Structure (simplified):**
```python
{
  'age_group_model': RandomForest(200 trees),
  'gender_model': RandomForest(200 trees),
  'mood_model': RandomForest(200 trees),
  'weather_model': RandomForest(200 trees)
}
```

**Why Critical:**
- Without this, you can't make predictions
- Contains ALL learned patterns from training
- Takes longest to train (10+ minutes)

**Usage in Production:**
```python
import joblib
model = joblib.load('models/ad_classifier_model.pkl')
predictions = model.predict(text_features)
```

---

### 2. vectorizer.pkl

**Purpose:** Converts text to numerical features

**Contains:**
- Vocabulary dictionary (10,000 words/phrases)
- IDF (Inverse Document Frequency) scores
- Configuration parameters

**Example Vocabulary:**
```python
{
  "women": 1523,
  "summer": 2847,
  "dress": 987,
  "women summer": 1524,
  "summer dress": 2848,
  ...  (10,000 total entries)
}
```

**Why Critical:**
- Model trained on SPECIFIC vocabulary
- New text must use SAME vocabulary
- Different vectorizer = wrong predictions

**Example Problem:**
```python
# ❌ WRONG - Creates new vocabulary
vectorizer_new = TfidfVectorizer()
features = vectorizer_new.fit_transform(["New ad title"])
predictions = model.predict(features)  # INCORRECT!

# ✅ CORRECT - Uses trained vocabulary
vectorizer = joblib.load('models/vectorizer.pkl')
features = vectorizer.transform(["New ad title"])
predictions = model.predict(features)  # CORRECT!
```

**Usage in Production:**
```python
vectorizer = joblib.load('models/vectorizer.pkl')
text_features = vectorizer.transform(["Women's Winter Coat"])
```

---

### 3. label_encoders.pkl

**Purpose:** Converts between categories and numbers

**Contains:**
- 4 LabelEncoder objects (one per target)
- Bidirectional mappings

**Example Encodings:**
```python
{
  'target_age_group': LabelEncoder(
    classes=['Kids', '10-18', '18-39', '40-64', '65+']
  ),
  'target_gender': LabelEncoder(
    classes=['Female', 'Male']
  ),
  'target_mood': LabelEncoder(
    classes=['Angry', 'Happy', 'Neutral', 'Sad']
  ),
  'target_weather': LabelEncoder(
    classes=['cold', 'rainy', 'sunny']
  )
}
```

**Encoding Process:**
```python
# Training (encode):
"Happy" → 1
"Sad" → 2

# Prediction (decode):
1 → "Happy"
2 → "Sad"
```

**Why Critical:**
- Model outputs numbers (0, 1, 2, ...)
- Users need readable labels ("Happy", "Sad", ...)
- Must use SAME encoding used during training

**Usage in Production:**
```python
label_encoders = joblib.load('models/label_encoders.pkl')

# Model returns: [2, 0, 1, 2]
# Decode to readable labels:
age = label_encoders['target_age_group'].inverse_transform([2])  # "18-39"
gender = label_encoders['target_gender'].inverse_transform([0])   # "Female"
mood = label_encoders['target_mood'].inverse_transform([1])       # "Happy"
weather = label_encoders['target_weather'].inverse_transform([2]) # "sunny"
```

---

### 4. model_metadata.pkl

**Purpose:** Documentation and validation

**Contains:**
- Training statistics
- Accuracy scores
- Feature counts
- Class names

**Full Structure:**
```python
{
  'target_columns': [
    'target_age_group',
    'target_gender',
    'target_mood',
    'target_weather'
  ],
  'classes': {
    'target_age_group': ['Kids', '10-18', '18-39', '40-64', '65+'],
    'target_gender': ['Female', 'Male'],
    'target_mood': ['Angry', 'Happy', 'Neutral', 'Sad'],
    'target_weather': ['cold', 'rainy', 'sunny']
  },
  'accuracy_scores': {
    'target_age_group': 0.8523,
    'target_gender': 0.9235,
    'target_mood': 0.7845,
    'target_weather': 0.7156
  },
  'overall_accuracy': 0.8189,
  'vocabulary_size': 10000,
  'n_features': 10000,
  'n_samples_trained': 6672
}
```

**Why Useful:**
- **Documentation**: Quick performance reference
- **Validation**: Verify model loaded correctly
- **Debugging**: Identify low-performing targets
- **Monitoring**: Track model version/quality

**Usage in Production:**
```python
metadata = joblib.load('models/model_metadata.pkl')

# Show model info to users
print(f"Model Accuracy: {metadata['overall_accuracy']*100:.2f}%")
print(f"Trained on {metadata['n_samples_trained']} samples")

# Validation check
if metadata['vocabulary_size'] != 10000:
    raise ValueError("Incorrect model version!")
```

---

## 🔧 Troubleshooting

### Issue 1: Low Accuracy (< 70%)

**Possible Causes:**
- Imbalanced dataset
- Insufficient training data
- Noisy/inconsistent labels

**Solutions:**
1. Check class distributions (STEP 3 output)
2. Use balanced dataset: `Classification_model_dataset_FINAL_BALANCED.csv`
3. Increase training samples to 10,000+
4. Clean dataset (remove inconsistent labels)

---

### Issue 2: "Missing Column" Error

**Error Message:**
```
ValueError: Missing required columns: ['target_weather']
```

**Solution:**
1. Open CSV in Excel/Google Sheets
2. Verify column names exactly match:
   - `ad_title`
   - `target_age_group`
   - `target_gender`
   - `target_mood`
   - `target_weather`
3. Check for typos or extra spaces
4. Ensure first row is header

---

### Issue 3: Training Takes Too Long (> 30 minutes)

**Possible Causes:**
- Very large dataset (> 50,000 samples)
- Low-end CPU

**Solutions:**
1. Reduce `n_estimators` from 200 to 100
2. Reduce `max_features` from 10000 to 5000
3. Use Google Colab Pro (faster CPUs)

**Edit this line:**
```python
# Before:
base_classifier = RandomForestClassifier(n_estimators=200, ...)

# After:
base_classifier = RandomForestClassifier(n_estimators=100, ...)
```

---

### Issue 4: Download Files Don't Work

**Symptoms:**
- Browser shows no download prompt
- Files don't save

**Solutions:**
1. **Check browser permissions:**
   - Allow pop-ups for colab.research.google.com
   - Enable multiple downloads

2. **Manual download:**
   - Click folder icon (left sidebar)
   - Right-click each .pkl file
   - Select "Download"

3. **Alternative method:**
   ```python
   # Add to end of notebook
   !zip model_files.zip *.pkl
   files.download('model_files.zip')
   ```

---

### Issue 5: "Memory Error" During Training

**Error Message:**
```
MemoryError: Unable to allocate array
```

**Solutions:**
1. **Reduce dataset size:**
   ```python
   df = df.sample(n=10000, random_state=42)  # Use 10,000 samples
   ```

2. **Reduce features:**
   ```python
   vectorizer = TfidfVectorizer(max_features=5000, ...)  # Instead of 10000
   ```

3. **Use Colab with High RAM:**
   - Runtime → Change runtime type
   - Select "High-RAM" option (Colab Pro required)

---

## 🎯 Best Practices

### 1. Dataset Quality

✅ **Do:**
- Use balanced datasets (each class 20-40%)
- Have at least 5,000 samples
- Remove true duplicates
- Verify label consistency

❌ **Don't:**
- Use severely imbalanced data (one class > 70%)
- Include conflicting labels (same ad, different targets)
- Have missing values

---

### 2. Training Configuration

✅ **Do:**
- Use `class_weight='balanced'` for imbalanced data
- Set `random_state=42` for reproducibility
- Monitor accuracy scores for all targets
- Test on sample predictions before deployment

❌ **Don't:**
- Skip validation step
- Ignore low accuracy warnings
- Deploy without testing

---

### 3. File Management

✅ **Do:**
- Download all 4 .pkl files
- Keep files together in `models/` folder
- Version control your trained models
- Document training date and accuracy

❌ **Don't:**
- Mix files from different training runs
- Rename .pkl files
- Delete model_metadata.pkl

---

### 4. Production Deployment

✅ **Do:**
```python
# Correct: Load all components
model = joblib.load('models/ad_classifier_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
metadata = joblib.load('models/model_metadata.pkl')

# Use them together
features = vectorizer.transform([ad_title])
predictions = model.predict(features)
decoded = {
    col: label_encoders[col].inverse_transform([pred])[0]
    for col, pred in zip(metadata['target_columns'], predictions[0])
}
```

❌ **Don't:**
```python
# Wrong: Create new vectorizer
vectorizer = TfidfVectorizer()  # ❌ Won't match training!
features = vectorizer.fit_transform([ad_title])
```

---

## 📝 Training Checklist

Use this checklist each time you train a model:

- [ ] Google Colab connected
- [ ] Dataset uploaded (balanced version recommended)
- [ ] STEP 1: Packages installed successfully
- [ ] STEP 2: Dataset loaded (check row count)
- [ ] STEP 3: Class distributions reasonable (20-40% each)
- [ ] STEP 4: Feature matrix shape correct
- [ ] STEP 5: Train/test split completed
- [ ] STEP 6: Training finished (no errors)
- [ ] STEP 7: Accuracy scores acceptable (> 70%)
- [ ] STEP 8: Sample predictions make sense
- [ ] STEP 9: All 4 .pkl files saved
- [ ] STEP 10: All 4 .pkl files downloaded
- [ ] Files moved to `models/` folder
- [ ] Tested in production (`python main.py`)

---

## 🚀 Next Steps After Training

1. **Validate Downloaded Files:**
   ```bash
   ls -lh models/
   # Should show 4 .pkl files
   ```

2. **Test Production System:**
   ```bash
   python main.py
   ```

3. **Document Model Version:**
   - Create `models/VERSION.txt`
   - Include: training date, accuracy, dataset used

4. **Backup Model Files:**
   - Copy to cloud storage (Google Drive, Dropbox)
   - Keep versioned backups

---

## 📞 Support

If you encounter issues not covered in this guide:

1. Check error messages in STEP output
2. Verify dataset format matches requirements
3. Try with balanced dataset
4. Review troubleshooting section

---

**Last Updated:** November 2025
**Version:** 1.0

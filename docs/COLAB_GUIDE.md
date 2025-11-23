# Google Colab Training Guide

Complete step-by-step guide for training the Ad Classification Model using Google Colab.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Opening Google Colab](#opening-google-colab)
3. [Setting Up the Notebook](#setting-up-the-notebook)
4. [Running the Training](#running-the-training)
5. [Downloading Model Files](#downloading-model-files)
6. [Deploying to Your Project](#deploying-to-your-project)
7. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

Before starting, ensure you have:

- ✅ Google Account (for Google Colab access)
- ✅ `Classification model dataset.csv` file ready
- ✅ Basic understanding of the project structure
- ✅ Local project folder set up

---

## 2. Opening Google Colab

### Option A: Upload Training Script

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Select `train_model_colab.py` from your project
4. Wait for upload to complete

### Option B: Create New Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → New notebook**
3. Open `train_model_colab.py` from your local project
4. Copy all content
5. Paste into the Colab notebook

### Option C: Run as Python Script

1. Create new Colab notebook
2. Copy entire content of `train_model_colab.py`
3. Paste into a single code cell
4. Proceed with training

---

## 3. Setting Up the Notebook

### Step 3.1: Verify Runtime

1. Click **Runtime** in the top menu
2. Select **Change runtime type**
3. Set **Runtime type** to **Python 3**
4. Set **Hardware accelerator** to **None** (CPU is sufficient)
5. Click **Save**

### Step 3.2: Check File Structure

Your Colab notebook should have the training script ready. The script will:
- Install all required packages
- Guide you through data upload
- Train the model automatically
- Download model files at the end

---

## 4. Running the Training

### Step 4.1: Run Setup and Installation

**What happens:**
- Installs: scikit-learn, pandas, numpy, nltk, joblib
- Downloads NLTK data for text processing
- Takes ~1-2 minutes

**Look for:**
```
✅ Setup complete!
```

### Step 4.2: Upload Dataset

**What you'll see:**
```
📤 Please upload 'Classification model dataset.csv' file
```

**What to do:**
1. Click the **Choose Files** button
2. Select `Classification model dataset.csv` from your computer
3. Wait for upload (may take 1-2 minutes for large files)

**Confirmation:**
```
✅ Loaded: Classification model dataset.csv
```

### Step 4.3: Data Loading and Preprocessing

**What happens:**
- Loads CSV data
- Removes duplicates
- Cleans text data
- Shows dataset statistics

**What you'll see:**
```
📊 Dataset shape: (20000, 6)
🧹 Removed X duplicate ad titles
📈 Class Distributions:
```

**Review the output:**
- Check total number of rows
- Verify class distributions look reasonable
- Note any warnings about missing data

### Step 4.4: Feature Engineering

**What happens:**
- Creates TF-IDF features from ad titles
- Encodes target labels numerically
- Prepares data for ML model

**What you'll see:**
```
🔤 Creating TF-IDF features...
✅ Feature matrix shape: (20000, 5000)
🏷️  Encoding target labels...
```

**Important:** Note the number of features created.

### Step 4.5: Train-Test Split

**What happens:**
- Splits data into 80% training, 20% testing
- Ensures balanced distribution

**What you'll see:**
```
📊 Training set size: 16000 samples
📊 Testing set size: 4000 samples
```

### Step 4.6: Model Training

**⏱️ Duration:** 5-10 minutes

**What happens:**
- Trains Random Forest classifier with 200 trees
- Shows progress during training

**What you'll see:**
```
🤖 Training Multi-Output Random Forest Classifier...
   This may take a few minutes...

[Parallel(n_jobs=-1)]: Done X tasks...
```

**⚠️ Important:**
- Don't close the browser tab
- Don't let your computer sleep
- Keep Colab tab active

**Confirmation:**
```
✅ Model training complete!
```

### Step 4.7: Model Evaluation

**What happens:**
- Tests model on unseen data
- Calculates accuracy for each category
- Shows detailed performance metrics

**What you'll see:**
```
📊 MODEL PERFORMANCE:

============================
📌 TARGET_AGE_GROUP
============================
Accuracy: 0.8523 (85.23%)

Detailed Report:
              precision    recall  f1-score   support
Kids              0.89      0.87      0.88       234
10-18             0.83      0.85      0.84       456
18-39             0.86      0.88      0.87      2345
...
```

**What to check:**
- **Overall accuracy**: Should be > 80% for good performance
- **Per-category accuracy**: Check each category
- **Precision/Recall**: Look for balanced scores

**Example Good Performance:**
```
📊 OVERALL AVERAGE ACCURACY: 0.8745 (87.45%)
```

**Example Poor Performance (need to investigate):**
```
📊 OVERALL AVERAGE ACCURACY: 0.5234 (52.34%)
```

### Step 4.8: Sample Predictions

**What happens:**
- Tests model on example ad titles
- Verifies predictions make sense

**What you'll see:**
```
🧪 Testing predictions on sample ad titles:

📝 Ad Title: FDT Women's Leggings
   Age Group: 18-39
   Gender: Female
   Mood: neutral
   Weather: sunny
```

**What to check:**
- Do predictions make logical sense?
- Are genders predicted correctly?
- Are age groups reasonable?

### Step 4.9: Save Model Files

**What happens:**
- Saves all model components to files
- Prepares for download

**What you'll see:**
```
💾 Saving model files...
✅ Saved: ad_classifier_model.pkl
✅ Saved: vectorizer.pkl
✅ Saved: label_encoders.pkl
✅ Saved: model_metadata.pkl
```

---

## 5. Downloading Model Files

### Step 5.1: Automatic Download

**What happens:**
- Browser automatically downloads all 4 files
- Files appear in your Downloads folder

**Files to download:**
1. ✅ `ad_classifier_model.pkl` (~50-80 MB)
2. ✅ `vectorizer.pkl` (~5-10 MB)
3. ✅ `label_encoders.pkl` (~1 KB)
4. ✅ `model_metadata.pkl` (~1 KB)

### Step 5.2: Manual Download (if automatic fails)

1. Look at the left sidebar in Colab
2. Click the **Files** icon (folder icon)
3. Find each `.pkl` file
4. Right-click → **Download**
5. Repeat for all 4 files

### Step 5.3: Verify Downloads

Check your Downloads folder for all 4 files:

```bash
# On macOS/Linux
ls -lh ~/Downloads/*.pkl

# Expected output:
# ad_classifier_model.pkl  (50-80 MB)
# vectorizer.pkl           (5-10 MB)
# label_encoders.pkl       (~1 KB)
# model_metadata.pkl       (~1 KB)
```

---

## 6. Deploying to Your Project

### Step 6.1: Locate Your Project

```bash
cd /path/to/AdResearch_Classification_Model
```

### Step 6.2: Move Files to models/ Directory

#### On macOS/Linux:

```bash
# Move files from Downloads to project
mv ~/Downloads/ad_classifier_model.pkl models/
mv ~/Downloads/vectorizer.pkl models/
mv ~/Downloads/label_encoders.pkl models/
mv ~/Downloads/model_metadata.pkl models/
```

#### On Windows:

```powershell
# Move files from Downloads to project
Move-Item $HOME\Downloads\ad_classifier_model.pkl models\
Move-Item $HOME\Downloads\vectorizer.pkl models\
Move-Item $HOME\Downloads\label_encoders.pkl models\
Move-Item $HOME\Downloads\model_metadata.pkl models\
```

#### Manual Move:

1. Open your Downloads folder
2. Open your project's `models/` folder in another window
3. Drag and drop all 4 `.pkl` files

### Step 6.3: Verify Deployment

```bash
# Check files are in place
ls -lh models/

# Expected output:
# ad_classifier_model.pkl
# vectorizer.pkl
# label_encoders.pkl
# model_metadata.pkl
```

### Step 6.4: Test the Model

```bash
# Test model loading
python main.py --predict "Women's Fashion Leggings"
```

**Expected output:**
```
📦 Loading trained model...
✅ Model loaded successfully!
   - Overall accuracy: 87.45%
   - Features: 5000

🔮 Predicting categories for: Women's Fashion Leggings

✅ Prediction Results:
   👥 Age Group: 18-39
   👤 Gender: Female
   😊 Mood: neutral
   🌤️  Weather: sunny
```

---

## 7. Troubleshooting

### Issue: "Runtime disconnected"

**Cause:** Colab session timed out or browser closed

**Solution:**
1. Refresh the page
2. Restart runtime: **Runtime → Restart runtime**
3. Re-run all cells from the beginning
4. Re-upload dataset

### Issue: "Out of memory"

**Cause:** Dataset too large for free Colab tier

**Solutions:**
1. Reduce dataset size (sample fewer rows)
2. Reduce `max_features` in TF-IDF (try 3000 instead of 5000)
3. Use Colab Pro for more RAM

### Issue: Low accuracy (<70%)

**Possible causes:**
- Dataset quality issues
- Imbalanced classes
- Too few training samples

**Solutions:**
1. Check dataset for errors or duplicates
2. Balance classes (add more minority class samples)
3. Increase training data size
4. Adjust model parameters:
   ```python
   n_estimators=300  # More trees
   max_depth=50      # Deeper trees
   ```

### Issue: Downloads not starting

**Solution 1: Manual download**
- Use Files sidebar → right-click → Download

**Solution 2: Use Google Drive**
```python
# Add at end of training script
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy('ad_classifier_model.pkl', '/content/drive/MyDrive/')
shutil.copy('vectorizer.pkl', '/content/drive/MyDrive/')
shutil.copy('label_encoders.pkl', '/content/drive/MyDrive/')
shutil.copy('model_metadata.pkl', '/content/drive/MyDrive/')
```

### Issue: "Model file not found" after deployment

**Cause:** Files not in correct location

**Solution:**
```bash
# Verify files exist
ls -la models/

# Check file permissions
chmod 644 models/*.pkl

# Test model loading
python -c "from src.classifier import AdClassifier; c = AdClassifier(); print(c.load_model())"
```

### Issue: Import errors in Colab

**Cause:** Packages not installed

**Solution:**
```python
# Run in first cell
!pip install scikit-learn pandas numpy nltk joblib --quiet

# Verify installation
import sklearn
import pandas
import nltk
import joblib
print("All packages installed!")
```

---

## 📊 Expected Performance Metrics

### Good Model Performance

| Category | Accuracy | Status |
|----------|----------|--------|
| Age Group | > 85% | ✅ Excellent |
| Gender | > 90% | ✅ Excellent |
| Mood | > 80% | ✅ Good |
| Weather | > 85% | ✅ Excellent |
| **Overall** | **> 85%** | ✅ **Good** |

### Acceptable Performance

| Category | Accuracy | Status |
|----------|----------|--------|
| Age Group | 75-85% | ⚠️ Acceptable |
| Gender | 85-90% | ✅ Good |
| Mood | 70-80% | ⚠️ Acceptable |
| Weather | 75-85% | ⚠️ Acceptable |
| **Overall** | **75-85%** | ⚠️ **Acceptable** |

### Poor Performance (Needs Investigation)

| Category | Accuracy | Status |
|----------|----------|--------|
| Age Group | < 75% | ❌ Poor |
| Gender | < 85% | ❌ Poor |
| Mood | < 70% | ❌ Poor |
| Weather | < 75% | ❌ Poor |
| **Overall** | **< 75%** | ❌ **Needs Improvement** |

---

## 📝 Training Checklist

Use this checklist when training:

- [ ] Opened Google Colab
- [ ] Uploaded train_model_colab.py or copied script
- [ ] Verified runtime settings (Python 3, CPU)
- [ ] Ran setup and installation cell
- [ ] Uploaded Classification model dataset.csv
- [ ] Verified dataset loaded correctly
- [ ] Completed preprocessing step
- [ ] Completed feature engineering
- [ ] Model training finished successfully
- [ ] Reviewed accuracy metrics (> 80% overall)
- [ ] Tested sample predictions
- [ ] All 4 files saved successfully
- [ ] Downloaded all 4 .pkl files
- [ ] Moved files to models/ directory
- [ ] Tested model in local project
- [ ] Model predictions working correctly

---

## 🎓 Tips for Better Results

1. **Dataset Quality**
   - Remove obvious errors before training
   - Ensure consistent formatting
   - Balance classes if possible

2. **Training Time**
   - Expect 5-10 minutes for full training
   - Don't interrupt the process
   - Keep Colab tab active

3. **Model Tuning**
   - Start with default parameters
   - Adjust only if accuracy < 75%
   - Test changes incrementally

4. **Save Your Work**
   - Download files immediately after training
   - Keep backup copies
   - Note model version and accuracy

5. **Regular Retraining**
   - Retrain monthly with new data
   - Archive old models with dates
   - Compare performance over time

---

**Last Updated**: November 2025

For additional help, see:
- `README.md` - General project documentation
- `docs/ARCHITECTURE.md` - System architecture
- `docs/API_REFERENCE.md` - API documentation

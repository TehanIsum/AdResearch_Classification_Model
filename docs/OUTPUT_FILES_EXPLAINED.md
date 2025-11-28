# 📦 Output Files Deep Dive

## 📋 Table of Contents
1. [Overview](#overview)
2. [ad_classifier_model.pkl](#1-ad_classifier_modelpkl)
3. [vectorizer.pkl](#2-vectorizerpkl)
4. [label_encoders.pkl](#3-label_encoderspkl)
5. [model_metadata.pkl](#4-model_metadatapkl)
6. [File Dependencies](#file-dependencies)
7. [Loading and Using Files](#loading-and-using-files)
8. [Troubleshooting](#troubleshooting)

---

## Overview

After training, the model produces **4 essential .pkl files** that work together to make predictions. All 4 files are required for the production system to function.

### Quick Summary

| File | Size | Purpose | Can Skip? |
|------|------|---------|-----------|
| `ad_classifier_model.pkl` | 50-200 MB | Makes predictions | ❌ No - Core model |
| `vectorizer.pkl` | 5-20 MB | Converts text to features | ❌ No - Required for input |
| `label_encoders.pkl` | < 1 KB | Decodes predictions | ❌ No - Required for output |
| `model_metadata.pkl` | < 1 KB | Documentation & validation | ⚠️ Optional but recommended |

---

## 1. ad_classifier_model.pkl

### What It Contains

This file stores the **trained Multi-Output Classifier** with 4 Random Forest models:

```python
{
  'estimators_': [
    RandomForestClassifier_1,  # Predicts target_age_group
    RandomForestClassifier_2,  # Predicts target_gender
    RandomForestClassifier_3,  # Predicts target_mood
    RandomForestClassifier_4   # Predicts target_weather
  ]
}
```

Each Random Forest contains:
- **200 decision trees**
- Tree structures (nodes, splits, thresholds)
- Leaf values (predictions)
- Feature importance scores

**Total:** 800 decision trees (200 × 4 targets)

### Internal Structure (Simplified)

```python
RandomForestClassifier {
  trees: [
    DecisionTree_1 {
      nodes: 1523
      depth: 28
      splits: {
        feature_34: threshold=0.42,
        feature_156: threshold=0.78,
        ...
      }
    },
    DecisionTree_2 { ... },
    ...
    DecisionTree_200 { ... }
  ]
}
```

### Why We Need It

**Primary Function:** Core prediction engine

**What it does:**
1. Takes numerical features (from vectorizer)
2. Passes through 800 decision trees
3. Returns 4 numerical predictions

**Example Flow:**
```python
# Input features
features = [0.0, 0.3, 0.0, 0.7, ...]  # 10,000 numbers

# Model processes
predictions = model.predict([features])
# Output: [[2, 0, 1, 2]]
#         Age: 2, Gender: 0, Mood: 1, Weather: 2
```

### File Size Breakdown

**Typical Size:** 50-200 MB

**Size depends on:**
- `n_estimators`: More trees = larger file
- `max_depth`: Deeper trees = more nodes = larger file
- `n_features`: More features considered = larger file
- Dataset size: More training samples = more complex trees

**Example:**
```
Configuration A:
- n_estimators=100
- max_depth=20
- Result: ~30 MB

Configuration B (our model):
- n_estimators=200
- max_depth=30
- Result: ~100 MB

Configuration C:
- n_estimators=500
- max_depth=50
- Result: ~400 MB
```

### How It Was Created

```python
# Training code
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib

# Create base classifier
base_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    random_state=42,
    n_jobs=-1
)

# Wrap for multi-output
model = MultiOutputClassifier(base_clf)

# Train
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'ad_classifier_model.pkl')
```

### Loading in Production

```python
import joblib

# Load model
model = joblib.load('models/ad_classifier_model.pkl')

# Use for predictions
predictions = model.predict(features)
```

### What Happens if Missing?

```python
❌ Error: FileNotFoundError: No such file or directory: 'ad_classifier_model.pkl'
🚫 System Cannot: Make any predictions
💡 Solution: Re-train model or restore from backup
```

---

## 2. vectorizer.pkl

### What It Contains

This file stores the **TfidfVectorizer** with:

1. **Vocabulary Dictionary** (10,000 words/phrases → indices)
2. **IDF Values** (importance scores for each word)
3. **Configuration Parameters** (ngram settings, min/max df, etc.)

### Internal Structure

```python
TfidfVectorizer {
  vocabulary_: {
    "women": 9523,
    "summer": 8234,
    "dress": 2847,
    "women summer": 9524,
    "summer dress": 8235,
    "women summer dress": 9525,
    ... (10,000 total entries)
  },
  
  idf_: [
    1.234,  # IDF score for feature 0
    2.456,  # IDF score for feature 1
    0.789,  # IDF score for feature 2
    ...     # (10,000 total scores)
  ],
  
  parameters: {
    max_features: 10000,
    ngram_range: (1, 3),
    min_df: 3,
    max_df: 0.85,
    lowercase: True,
    ...
  }
}
```

### Why We Need It

**Critical Function:** Converts text to numerical features

**Problem it solves:**
- Model was trained on **specific vocabulary**
- New text must use **exact same vocabulary**
- Different vocabulary = wrong predictions

**Example:**

```python
# ❌ WRONG: Create new vectorizer
vectorizer_new = TfidfVectorizer()
vectorizer_new.fit(["New ad title"])

# This creates DIFFERENT vocabulary:
# {"new": 0, "ad": 1, "title": 2}

features = vectorizer_new.transform(["Women's dress"])
predictions = model.predict(features)  # INCORRECT!
```

```python
# ✅ CORRECT: Use trained vectorizer
vectorizer = joblib.load('models/vectorizer.pkl')

# Uses SAME vocabulary from training
features = vectorizer.transform(["Women's dress"])
predictions = model.predict(features)  # CORRECT!
```

### Vocabulary Example

**Training Data:**
```
1. "Women's summer dress"
2. "Men's winter coat"
3. "Kids toy car"
...
8,340 ad titles
```

**Resulting Vocabulary (top 20 shown):**
```python
{
  "women": 0,
  "men": 1,
  "kids": 2,
  "summer": 3,
  "winter": 4,
  "dress": 5,
  "coat": 6,
  "toy": 7,
  "car": 8,
  "women summer": 9,
  "summer dress": 10,
  "men winter": 11,
  "winter coat": 12,
  "kids toy": 13,
  "toy car": 14,
  ...
  "women summer dress": 9998,
  "kids toy car": 9999
}
```

### IDF Values Example

```python
# Common words (appear in many documents) → LOW IDF
"the": 0.02
"and": 0.05
"for": 0.08

# Meaningful words → MEDIUM IDF
"summer": 0.45
"winter": 0.52
"dress": 0.48

# Rare/distinctive words → HIGH IDF
"luxury": 1.23
"premium": 1.45
"handcrafted": 1.67
```

### Transformation Example

**Input Text:**
```python
"Women's Summer Dress"
```

**Step 1: Tokenization & Vocabulary Lookup**
```python
"women" → index 0
"summer" → index 3
"dress" → index 5
"women summer" → index 9
"summer dress" → index 10
"women summer dress" → index 9998
```

**Step 2: Calculate TF-IDF**
```python
# Result: Sparse array of 10,000 values
[
  0.0,    # Feature 0: Not present
  0.0,    # Feature 1: Not present
  0.0,    # Feature 2: Not present
  0.342,  # Feature 3: "summer" (TF-IDF score)
  0.0,    # Feature 4: Not present
  0.521,  # Feature 5: "dress" (TF-IDF score)
  ...
  0.0,    # Feature 9997: Not present
  0.234,  # Feature 9998: "women summer dress"
  0.0     # Feature 9999: Not present
]
```

**Output:** 10,000-dimensional feature vector

### File Size

**Typical Size:** 5-20 MB

**Size depends on:**
- `max_features`: 10,000 features
- Vocabulary entries: ~1 KB each
- IDF array: ~80 KB (10,000 × 8 bytes)

### How It Was Created

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Create vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.85
)

# Fit on training data (learns vocabulary)
vectorizer.fit(training_texts)

# Save
joblib.dump(vectorizer, 'vectorizer.pkl')
```

### Loading in Production

```python
import joblib

# Load vectorizer
vectorizer = joblib.load('models/vectorizer.pkl')

# Transform new text
features = vectorizer.transform(["New ad title"])
```

### What Happens if Missing?

```python
❌ Error: FileNotFoundError: No such file or directory: 'vectorizer.pkl'
🚫 System Cannot: Convert text to features
💡 Solution: Re-train model or restore from backup
```

---

## 3. label_encoders.pkl

### What It Contains

This file stores **4 LabelEncoder objects** (one per target attribute):

```python
{
  'target_age_group': LabelEncoder(
    classes_=['10-18', '18-39', '40-64', '65+', 'Kids']
  ),
  'target_gender': LabelEncoder(
    classes_=['Female', 'Male']
  ),
  'target_mood': LabelEncoder(
    classes_=['Angry', 'Happy', 'Neutral', 'Sad']
  ),
  'target_weather': LabelEncoder(
    classes_=['cold', 'rainy', 'sunny']
  )
}
```

### Internal Structure (Per Encoder)

```python
LabelEncoder {
  classes_: ['Class1', 'Class2', 'Class3'],  # Original labels
  
  # Internal mappings (created automatically)
  encoding_map: {
    'Class1': 0,
    'Class2': 1,
    'Class3': 2
  },
  
  decoding_map: {
    0: 'Class1',
    1: 'Class2',
    2: 'Class3'
  }
}
```

### Why We Need It

**Critical Function:** Translates between numbers and readable labels

**Problem it solves:**
- Model outputs **numbers** (0, 1, 2, ...)
- Users need **readable labels** ("Happy", "Sad", ...)
- Must use **exact same encoding** as training

**Example:**

```python
# Model prediction (numbers)
predictions = model.predict(features)
# Output: [[2, 0, 1, 2]]

# Without label encoder
print(predictions)  # [[2, 0, 1, 2]]  ❌ Not user-friendly

# With label encoder
age = label_encoders['target_age_group'].inverse_transform([2])
gender = label_encoders['target_gender'].inverse_transform([0])
mood = label_encoders['target_mood'].inverse_transform([1])
weather = label_encoders['target_weather'].inverse_transform([2])

print(f"Age: {age[0]}")       # Age: 18-39 ✅
print(f"Gender: {gender[0]}") # Gender: Female ✅
print(f"Mood: {mood[0]}")     # Mood: Happy ✅
print(f"Weather: {weather[0]}")# Weather: sunny ✅
```

### Encoding Examples

#### Age Group Encoder
```python
# Fit during training
age_groups = ['Kids', '10-18', '18-39', '40-64', '65+', 'Kids', '18-39']
le_age = LabelEncoder()
le_age.fit(age_groups)

# Encoding map (alphabetical order)
'10-18' → 0
'18-39' → 1
'40-64' → 2
'65+' → 3
'Kids' → 4

# Transform (encode)
le_age.transform(['Kids', '18-39', '65+'])
# Output: [4, 1, 3]

# Inverse transform (decode)
le_age.inverse_transform([4, 1, 3])
# Output: ['Kids', '18-39', '65+']
```

#### Gender Encoder
```python
# Encoding map
'Female' → 0
'Male' → 1
```

#### Mood Encoder
```python
# Encoding map
'Angry' → 0
'Happy' → 1
'Neutral' → 2
'Sad' → 3
```

#### Weather Encoder
```python
# Encoding map
'cold' → 0
'rainy' → 1
'sunny' → 2
```

### Complete Prediction Flow

```python
# 1. User input
ad_title = "Women's Summer Dress"

# 2. Vectorize text
features = vectorizer.transform([ad_title])

# 3. Model predicts (returns numbers)
predictions = model.predict(features)
# predictions = [[1, 0, 1, 2]]

# 4. Decode to readable labels
result = {
    'age_group': label_encoders['target_age_group'].inverse_transform([predictions[0][0]])[0],
    'gender': label_encoders['target_gender'].inverse_transform([predictions[0][1]])[0],
    'mood': label_encoders['target_mood'].inverse_transform([predictions[0][2]])[0],
    'weather': label_encoders['target_weather'].inverse_transform([predictions[0][3]])[0]
}

# 5. Display to user
print(result)
# {
#   'age_group': '18-39',
#   'gender': 'Female',
#   'mood': 'Happy',
#   'weather': 'sunny'
# }
```

### File Size

**Typical Size:** < 1 KB

**Very small because:**
- Only stores class names (strings)
- 4 encoders total
- Minimal metadata

### How It Was Created

```python
from sklearn.preprocessing import LabelEncoder
import joblib

# Create encoders for each target
label_encoders = {}

for col in ['target_age_group', 'target_gender', 'target_mood', 'target_weather']:
    le = LabelEncoder()
    le.fit(y_train[col])  # Learn classes from training data
    label_encoders[col] = le

# Save
joblib.dump(label_encoders, 'label_encoders.pkl')
```

### Loading in Production

```python
import joblib

# Load encoders
label_encoders = joblib.load('models/label_encoders.pkl')

# Decode predictions
age = label_encoders['target_age_group'].inverse_transform([2])[0]
```

### What Happens if Missing?

```python
❌ Error: FileNotFoundError: No such file or directory: 'label_encoders.pkl'
🚫 System Cannot: Decode numerical predictions to readable labels
💔 User Sees: Numbers instead of categories (confusing!)
💡 Solution: Re-train model or restore from backup
```

---

## 4. model_metadata.pkl

### What It Contains

This file stores **training statistics and documentation**:

```python
{
  'target_columns': [
    'target_age_group',
    'target_gender',
    'target_mood',
    'target_weather'
  ],
  
  'classes': {
    'target_age_group': ['10-18', '18-39', '40-64', '65+', 'Kids'],
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
  'n_samples_trained': 6672,
  
  'training_date': '2025-11-25',
  'model_version': '1.0',
  'dataset_name': 'Classification_model_dataset_FINAL_BALANCED.csv'
}
```

### Why We Need It

**Functions:**
1. **Documentation:** Quick reference for model performance
2. **Validation:** Verify model loaded correctly
3. **Debugging:** Identify which targets perform poorly
4. **Monitoring:** Track model versions over time
5. **User Information:** Display model capabilities

### Use Cases

#### 1. Display Model Info to Users
```python
import joblib

metadata = joblib.load('models/model_metadata.pkl')

print("=" * 50)
print("AD CLASSIFICATION MODEL")
print("=" * 50)
print(f"Overall Accuracy: {metadata['overall_accuracy']*100:.2f}%")
print(f"Trained on: {metadata['n_samples_trained']} samples")
print(f"Training Date: {metadata['training_date']}")
print("\nTarget Accuracies:")
for target, accuracy in metadata['accuracy_scores'].items():
    print(f"  {target}: {accuracy*100:.2f}%")
```

**Output:**
```
==================================================
AD CLASSIFICATION MODEL
==================================================
Overall Accuracy: 81.89%
Trained on: 6672 samples
Training Date: 2025-11-25

Target Accuracies:
  target_age_group: 85.23%
  target_gender: 92.35%
  target_mood: 78.45%
  target_weather: 71.56%
```

#### 2. Validation Check
```python
metadata = joblib.load('models/model_metadata.pkl')

# Verify correct model version
assert metadata['vocabulary_size'] == 10000, "Wrong model version!"
assert metadata['n_features'] == 10000, "Feature mismatch!"

# Verify all targets present
expected_targets = ['target_age_group', 'target_gender', 'target_mood', 'target_weather']
assert metadata['target_columns'] == expected_targets, "Missing targets!"

print("✅ Model validation passed")
```

#### 3. Performance Monitoring
```python
metadata = joblib.load('models/model_metadata.pkl')

# Check if any target performs poorly
for target, accuracy in metadata['accuracy_scores'].items():
    if accuracy < 0.70:
        print(f"⚠️ Warning: {target} accuracy is low ({accuracy*100:.2f}%)")
        print(f"   Consider retraining with more balanced data")
```

#### 4. Model Comparison
```python
# Load metadata from different model versions
metadata_v1 = joblib.load('models/v1/model_metadata.pkl')
metadata_v2 = joblib.load('models/v2/model_metadata.pkl')

print("Model Version Comparison:")
print(f"V1 Accuracy: {metadata_v1['overall_accuracy']*100:.2f}%")
print(f"V2 Accuracy: {metadata_v2['overall_accuracy']*100:.2f}%")

if metadata_v2['overall_accuracy'] > metadata_v1['overall_accuracy']:
    print("✅ V2 is better - deploy it!")
else:
    print("⚠️ V1 is still better - keep using it")
```

### File Size

**Typical Size:** < 1 KB

**Very small because:**
- Only text and numbers (no model weights)
- JSON-like dictionary structure

### How It Was Created

```python
import joblib
from datetime import datetime

# Collect metadata after training
metadata = {
    'target_columns': target_columns,
    'classes': {col: le.classes_.tolist() for col, le in label_encoders.items()},
    'vocabulary_size': len(vectorizer.vocabulary_),
    'n_features': X_vectorized.shape[1],
    'n_samples_trained': X_train.shape[0],
    'accuracy_scores': {col: acc for col, acc in zip(target_columns, overall_accuracy)},
    'overall_accuracy': np.mean(overall_accuracy),
    'training_date': datetime.now().strftime('%Y-%m-%d'),
    'model_version': '1.0'
}

# Save
joblib.dump(metadata, 'model_metadata.pkl')
```

### Loading in Production

```python
import joblib

# Load metadata
metadata = joblib.load('models/model_metadata.pkl')

# Access information
print(f"Model Accuracy: {metadata['overall_accuracy']}")
print(f"Possible Age Groups: {metadata['classes']['target_age_group']}")
```

### What Happens if Missing?

```python
⚠️ Warning: File not found (optional)
✅ System Still Works: Can make predictions
❌ Missing: Documentation, validation, monitoring
💡 Recommendation: Include for production systems
```

---

## File Dependencies

### Dependency Graph

```
User Input: "Women's Summer Dress"
        ↓
┌───────────────────────┐
│   vectorizer.pkl       │ ← Converts text to features
│   (REQUIRED)           │
└───────────────────────┘
        ↓
Features: [0.0, 0.3, 0.7, ...]
        ↓
┌───────────────────────┐
│ ad_classifier_model.pkl│ ← Makes predictions
│   (REQUIRED)           │
└───────────────────────┘
        ↓
Predictions: [2, 0, 1, 2]
        ↓
┌───────────────────────┐
│  label_encoders.pkl    │ ← Decodes to labels
│   (REQUIRED)           │
└───────────────────────┘
        ↓
Results: Age=18-39, Gender=Female, Mood=Happy, Weather=sunny
        ↓
┌───────────────────────┐
│ model_metadata.pkl     │ ← Shows accuracy, validation
│   (OPTIONAL)           │
└───────────────────────┘
```

### File Interaction Matrix

| File | Depends On | Used By | Can Work Without? |
|------|-----------|---------|-------------------|
| `vectorizer.pkl` | - | `ad_classifier_model.pkl` | ❌ No |
| `ad_classifier_model.pkl` | `vectorizer.pkl` | `label_encoders.pkl` | ❌ No |
| `label_encoders.pkl` | `ad_classifier_model.pkl` | User Interface | ❌ No |
| `model_metadata.pkl` | - | Monitoring/Display | ✅ Yes |

---

## Loading and Using Files

### Complete Production Example

```python
import joblib
import pandas as pd

# ============================================
# 1. LOAD ALL COMPONENTS (Do this once at startup)
# ============================================

print("Loading model components...")

# Load model
model = joblib.load('models/ad_classifier_model.pkl')
print("✅ Model loaded")

# Load vectorizer
vectorizer = joblib.load('models/vectorizer.pkl')
print("✅ Vectorizer loaded")

# Load encoders
label_encoders = joblib.load('models/label_encoders.pkl')
print("✅ Label encoders loaded")

# Load metadata (optional)
metadata = joblib.load('models/model_metadata.pkl')
print("✅ Metadata loaded")
print(f"   Model Accuracy: {metadata['overall_accuracy']*100:.2f}%")

# ============================================
# 2. PREDICTION FUNCTION (Use for each prediction)
# ============================================

def classify_ad(ad_title):
    """
    Classifies an ad title and returns predictions
    
    Args:
        ad_title (str): The advertisement title
        
    Returns:
        dict: Predicted target attributes
    """
    # Clean text
    clean_title = ad_title.lower().strip()
    
    # Vectorize
    features = vectorizer.transform([clean_title])
    
    # Predict
    predictions = model.predict(features)[0]
    
    # Decode
    result = {
        'age_group': label_encoders['target_age_group'].inverse_transform([predictions[0]])[0],
        'gender': label_encoders['target_gender'].inverse_transform([predictions[1]])[0],
        'mood': label_encoders['target_mood'].inverse_transform([predictions[2]])[0],
        'weather': label_encoders['target_weather'].inverse_transform([predictions[3]])[0]
    }
    
    return result

# ============================================
# 3. MAKE PREDICTIONS (Use as many times as needed)
# ============================================

# Single prediction
result = classify_ad("Women's Summer Beach Dress")
print("\n" + "="*50)
print("PREDICTION RESULT")
print("="*50)
print(f"Age Group: {result['age_group']}")
print(f"Gender: {result['gender']}")
print(f"Mood: {result['mood']}")
print(f"Weather: {result['weather']}")

# Batch predictions
ad_titles = [
    "Women's Summer Dress",
    "Men's Winter Coat",
    "Kids Toy Car",
    "Teen Fashion Backpack"
]

print("\n" + "="*50)
print("BATCH PREDICTIONS")
print("="*50)

for title in ad_titles:
    result = classify_ad(title)
    print(f"\n{title}")
    print(f"  → {result}")
```

### Output

```
Loading model components...
✅ Model loaded
✅ Vectorizer loaded
✅ Label encoders loaded
✅ Metadata loaded
   Model Accuracy: 81.89%

==================================================
PREDICTION RESULT
==================================================
Age Group: 18-39
Gender: Female
Mood: Happy
Weather: sunny

==================================================
BATCH PREDICTIONS
==================================================

Women's Summer Dress
  → {'age_group': '18-39', 'gender': 'Female', 'mood': 'Happy', 'weather': 'sunny'}

Men's Winter Coat
  → {'age_group': '40-64', 'gender': 'Male', 'mood': 'Neutral', 'weather': 'cold'}

Kids Toy Car
  → {'age_group': 'Kids', 'gender': 'Male', 'mood': 'Happy', 'weather': 'sunny'}

Teen Fashion Backpack
  → {'age_group': '10-18', 'gender': 'Female', 'mood': 'Happy', 'weather': 'sunny'}
```

---

## Troubleshooting

### Issue 1: FileNotFoundError

**Error:**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'models/ad_classifier_model.pkl'
```

**Solutions:**
1. Check file location:
   ```bash
   ls -la models/
   ```
2. Verify all 4 files present:
   - ✓ `ad_classifier_model.pkl`
   - ✓ `vectorizer.pkl`
   - ✓ `label_encoders.pkl`
   - ✓ `model_metadata.pkl`
3. Check file path in code (use absolute paths if needed)

---

### Issue 2: Incorrect Predictions

**Symptoms:**
- All predictions same (e.g., always "Male")
- Random-seeming predictions
- Poor accuracy

**Possible Causes:**

**Cause 1: Wrong Vectorizer**
```python
# ❌ WRONG: Creating new vectorizer
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform([ad_title])
```

**Solution:**
```python
# ✅ CORRECT: Load trained vectorizer
vectorizer = joblib.load('models/vectorizer.pkl')
features = vectorizer.transform([ad_title])
```

**Cause 2: Mixed Model Files**
```python
# Files from different training runs
models/
├── ad_classifier_model.pkl (from Training 1)
├── vectorizer.pkl (from Training 2)  ← Mismatch!
├── label_encoders.pkl (from Training 1)
└── model_metadata.pkl (from Training 3)
```

**Solution:** Ensure all 4 files are from the SAME training run

---

### Issue 3: Large File Size

**Problem:**
`ad_classifier_model.pkl` is 500 MB (too large)

**Solutions:**

**Option 1: Reduce Trees**
```python
# Change from:
n_estimators=200

# To:
n_estimators=100  # Halves file size
```

**Option 2: Reduce Tree Depth**
```python
# Change from:
max_depth=30

# To:
max_depth=20  # Reduces file size ~30%
```

**Option 3: Reduce Features**
```python
# Change from:
max_features=10000

# To:
max_features=5000  # Reduces file size ~20%
```

---

### Issue 4: Slow Loading Time

**Problem:**
Loading files takes 10+ seconds

**Solutions:**

**Option 1: Use `compress` (smaller files, faster loading)**
```python
# When saving:
joblib.dump(model, 'model.pkl', compress=3)  # Compression level 0-9
```

**Option 2: Load once at startup (not per request)**
```python
# ✅ GOOD: Load at startup
model = joblib.load('models/ad_classifier_model.pkl')  # Once

# Then use many times
for ad in ads:
    result = model.predict(...)  # Fast

# ❌ BAD: Load per request
for ad in ads:
    model = joblib.load('models/ad_classifier_model.pkl')  # Slow!
    result = model.predict(...)
```

---

### Issue 5: Version Mismatch

**Error:**
```python
ValueError: unsupported pickle protocol: 5
```

**Cause:** Model saved with Python 3.8+, loaded with Python 3.7-

**Solution:**
1. Upgrade Python to 3.8+
2. Or save with compatible protocol:
   ```python
   joblib.dump(model, 'model.pkl', protocol=4)  # Python 3.4+
   ```

---

## Best Practices

### 1. File Organization

```
project/
├── models/
│   ├── production/          ← Current production model
│   │   ├── ad_classifier_model.pkl
│   │   ├── vectorizer.pkl
│   │   ├── label_encoders.pkl
│   │   └── model_metadata.pkl
│   ├── v1.0/               ← Versioned backups
│   │   ├── ad_classifier_model.pkl
│   │   ├── vectorizer.pkl
│   │   ├── label_encoders.pkl
│   │   └── model_metadata.pkl
│   └── v1.1/
│       └── ...
└── src/
    └── classifier.py
```

### 2. Versioning

Create `models/VERSION.txt`:
```
Model Version: 1.1
Training Date: 2025-11-25
Dataset: Classification_model_dataset_FINAL_BALANCED.csv
Samples: 8,340
Overall Accuracy: 81.89%
Age Group Accuracy: 85.23%
Gender Accuracy: 92.35%
Mood Accuracy: 78.45%
Weather Accuracy: 71.56%
```

### 3. Backup Strategy

```bash
# After training, backup immediately
cp -r models/production models/backup_2025-11-25

# Or use version control
git add models/production/*
git commit -m "Model v1.1: 81.89% accuracy"
```

### 4. Loading Pattern

```python
class AdClassifier:
    def __init__(self, model_dir='models/production'):
        """Load all components once"""
        self.model = joblib.load(f'{model_dir}/ad_classifier_model.pkl')
        self.vectorizer = joblib.load(f'{model_dir}/vectorizer.pkl')
        self.label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')
        self.metadata = joblib.load(f'{model_dir}/model_metadata.pkl')
    
    def predict(self, ad_title):
        """Make predictions (reuse loaded components)"""
        features = self.vectorizer.transform([ad_title])
        predictions = self.model.predict(features)[0]
        
        return {
            'age_group': self.label_encoders['target_age_group'].inverse_transform([predictions[0]])[0],
            'gender': self.label_encoders['target_gender'].inverse_transform([predictions[1]])[0],
            'mood': self.label_encoders['target_mood'].inverse_transform([predictions[2]])[0],
            'weather': self.label_encoders['target_weather'].inverse_transform([predictions[3]])[0]
        }

# Use it
classifier = AdClassifier()  # Load once
result = classifier.predict("Women's Dress")  # Use many times
```

---

**Last Updated:** November 2025
**Version:** 1.0

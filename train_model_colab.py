"""
==================================================print("\n" + "=" * 80)
print("STEP 2: Upload Your Dataset")
print("=" * 80)
print("\n📤 Please upload one of these dataset files:")
print("   - 'Classification_model_dataset_balanced.csv' (RECOMMENDED)")
print("   - 'updated_classification_dataset.csv' (Most balanced)")
print("   - 'Classification model dataset.csv' (Original, but imbalanced)")
print("\n   Click the folder icon on the left sidebar, then upload button\n")

from google.colab import files
uploaded = files.upload()

# Load the dataset
dataset_name = list(uploaded.keys())[0]
print(f"\n✅ Loaded: {dataset_name}")

# Warn if using imbalanced dataset
if "balanced" not in dataset_name.lower() and "updated" not in dataset_name.lower():
    print("\n⚠️  WARNING: You're using an imbalanced dataset!")
    print("   This may lead to poor predictions for minority classes (rainy, cold)")
    print("   Consider using 'Classification_model_dataset_balanced.csv' instead")====================
AD CLASSIFICATION MODEL TRAINING - Google Colab Notebook
==================================================================================

This notebook trains a multi-output classification model to predict:
- target_age_group: Kids, 10-18, 18-39, 40-64, 65+
- target_gender: Male, Female
- target_mood: Happy, Angry, Sad, Neutral
- target_weather: sunny, rainy, cold

IMPORTANT: After training, you need to download these files to your local project:

📁 FILES TO DOWNLOAD FROM COLAB:
  1. ad_classifier_model.pkl        → Place in: models/ad_classifier_model.pkl
  2. vectorizer.pkl                 → Place in: models/vectorizer.pkl
  3. label_encoders.pkl             → Place in: models/label_encoders.pkl
  4. model_metadata.pkl             → Place in: models/model_metadata.pkl

==================================================================================
"""

# ============================================================================
# STEP 1: Setup and Installation
# ============================================================================

print("Installing required packages...")
!pip install scikit-learn pandas numpy nltk joblib -q

import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
print("\nDownloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("✅ Setup complete!\n")

# ============================================================================
# STEP 2: Upload Your Dataset
# ============================================================================

print("=" * 80)
print("STEP 2: Upload Your Dataset")
print("=" * 80)
print("\n📤 Please upload your dataset CSV file")
print("\n   🌟 RECOMMENDED (Best Results):")
print("   - 'Classification_model_dataset_PROPERLY_BALANCED.csv'")
print("     → Pre-balanced weather classes (31-34% each)")
print("     → 8,340 training samples")
print("     → No imbalance issues!")
print("\n   📊 Alternative Options:")
print("   - 'Classification model dataset.csv' (25,200 rows)")
print("     → Will auto-balance during training")
print("   - Your own custom dataset")
print("     → Must have required columns")
print("\n   Click the folder icon on the left sidebar, then upload button\n")

from google.colab import files
uploaded = files.upload()

# Load the dataset
dataset_name = list(uploaded.keys())[0]
print(f"\n✅ Loaded: {dataset_name}")

# ============================================================================
# STEP 3: Data Loading and Preprocessing
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Data Loading and Preprocessing")
print("=" * 80)

# Read the CSV (no skiprows needed - has proper header)
df = pd.read_csv(dataset_name)
print(f"\n📊 Dataset shape: {df.shape}")
print(f"📊 Total rows loaded: {len(df)} samples")
print(f"\n🔍 Column names found: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nLast few rows:")
print(df.tail())

# Check if 'ad_title' column exists, if not try to identify the correct column
if 'ad_title' not in df.columns:
    print("\n⚠️ WARNING: 'ad_title' column not found!")
    print(f"Available columns: {df.columns.tolist()}")
    # Try to find the ad title column (usually the second column)
    if len(df.columns) >= 2:
        title_col = df.columns[1]
        print(f"Using column '{title_col}' as ad_title")
        df = df.rename(columns={title_col: 'ad_title'})

# Verify required columns exist
required_cols = ['ad_title', 'target_age_group', 'target_gender', 'target_mood', 'target_weather']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}\nAvailable columns: {df.columns.tolist()}")

# Check for missing values
print(f"\n🔍 Missing values:")
print(df.isnull().sum())

# IMPROVED: Remove only TRUE duplicates (same ad_title AND same target values)
# This preserves entries where the same product targets different audiences
print(f"\n🔍 Analyzing duplicates...")
initial_rows = len(df)

# Check weather distribution BEFORE deduplication
print(f"\n📊 Weather distribution BEFORE deduplication:")
weather_before = df['target_weather'].value_counts()
for weather, count in weather_before.items():
    percentage = (count / len(df)) * 100
    print(f"   {weather}: {count} ({percentage:.1f}%)")

# Remove only TRUE duplicates (exact same row including all target columns)
df = df.drop_duplicates(subset=['ad_title', 'target_age_group', 'target_gender', 'target_mood', 'target_weather'])
removed_count = initial_rows - len(df)
print(f"\n🧹 Removed {removed_count} TRUE duplicate rows")
print(f"   (Kept entries with same ad_title but different target values)")

# Check weather distribution AFTER deduplication
print(f"\n📊 Weather distribution AFTER deduplication:")
weather_after = df['target_weather'].value_counts()
for weather, count in weather_after.items():
    percentage = (count / len(df)) * 100
    print(f"   {weather}: {count} ({percentage:.1f}%)")

# Warning if severely imbalanced
max_percentage = (weather_after.max() / len(df)) * 100
min_percentage = (weather_after.min() / len(df)) * 100
if max_percentage > 60:
    print(f"\n⚠️  WARNING: Weather data is imbalanced!")
    print(f"   Max class: {max_percentage:.1f}%, Min class: {min_percentage:.1f}%")
    print(f"   This may lead to biased predictions toward the majority class.")
else:
    print(f"\n✅ Weather distribution is reasonably balanced")

# Clean text data
def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

df['ad_title_clean'] = df['ad_title'].apply(clean_text)

# Display class distributions
print("\n📈 Final Class Distributions:")
print("\nAge Groups:")
print(df['target_age_group'].value_counts())
print("\nGender:")
print(df['target_gender'].value_counts())
print("\nMood:")
print(df['target_mood'].value_counts())
print("\nWeather:")
print(df['target_weather'].value_counts())

# ============================================================================
# STEP 4: Feature Engineering
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Feature Engineering")
print("=" * 80)

# Prepare features (X) and labels (y)
X = df['ad_title_clean'].values
y = df[['target_age_group', 'target_gender', 'target_mood', 'target_weather']]

print(f"\n📝 Number of samples: {len(X)}")
print(f"📝 Number of target categories: {y.shape[1]}")

# Text vectorization using TF-IDF
print("\n🔤 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=10000,      # Increased for larger dataset
    ngram_range=(1, 3),      # Use unigrams, bigrams, and trigrams
    min_df=3,                # Minimum document frequency (adjusted for 20k samples)
    max_df=0.85,             # Maximum document frequency
    strip_accents='unicode',
    lowercase=True
)

X_vectorized = vectorizer.fit_transform(X)
print(f"✅ Feature matrix shape: {X_vectorized.shape}")
print(f"   (samples × features)")

# Encode labels
print("\n🏷️  Encoding target labels...")
label_encoders = {}
y_encoded = pd.DataFrame()

for col in y.columns:
    le = LabelEncoder()
    y_encoded[col] = le.fit_transform(y[col])
    label_encoders[col] = le
    print(f"   {col}: {len(le.classes_)} classes → {le.classes_}")

# ============================================================================
# STEP 5: Train-Test Split
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Train-Test Split")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded['target_age_group']  # Stratify by age group
)

print(f"\n📊 Training set size: {X_train.shape[0]} samples")
print(f"📊 Testing set size: {X_test.shape[0]} samples")

# ============================================================================
# STEP 6: Model Training
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Model Training")
print("=" * 80)
print("\n🤖 Training Multi-Output Random Forest Classifier...")
print("   Optimized for large dataset (20,000 samples)...")
print("   This may take 10-15 minutes...\n")

# Create the multi-output classifier - OPTIMIZED FOR 20,000 SAMPLES
base_classifier = RandomForestClassifier(
    n_estimators=200,        # More trees for large dataset
    max_depth=30,            # Deeper trees for complex patterns
    min_samples_split=10,    # Higher to prevent overfitting
    min_samples_leaf=4,      # Higher for large dataset
    random_state=42,
    n_jobs=-1,
    verbose=1,
    class_weight='balanced',  # Handle imbalanced classes
    max_features='sqrt'       # Use sqrt for better performance
)

model = MultiOutputClassifier(base_classifier, n_jobs=-1)

# Train the model
model.fit(X_train, y_train)

print("\n✅ Model training complete!")

# ============================================================================
# STEP 7: Model Evaluation
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Model Evaluation")
print("=" * 80)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate each target category
target_columns = ['target_age_group', 'target_gender', 'target_mood', 'target_weather']

print("\n📊 MODEL PERFORMANCE:\n")
overall_accuracy = []

for idx, col in enumerate(target_columns):
    accuracy = accuracy_score(y_test.iloc[:, idx], y_pred[:, idx])
    overall_accuracy.append(accuracy)
    
    print(f"\n{'=' * 60}")
    print(f"📌 {col.upper()}")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nDetailed Report:")
    
    # Get unique labels present in test set
    unique_labels = np.unique(y_test.iloc[:, idx])
    target_names_subset = [label_encoders[col].classes_[i] for i in unique_labels]
    
    print(classification_report(
        y_test.iloc[:, idx], 
        y_pred[:, idx],
        labels=unique_labels,
        target_names=target_names_subset,
        zero_division=0
    ))

print(f"\n{'=' * 60}")
print(f"📊 OVERALL AVERAGE ACCURACY: {np.mean(overall_accuracy):.4f} ({np.mean(overall_accuracy)*100:.2f}%)")
print(f"{'=' * 60}")

# ============================================================================
# STEP 8: Test Predictions on Sample Data
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Sample Predictions")
print("=" * 80)

# Test with sample ad titles
sample_titles = [
    "FDT Women's Leggings",
    "Kids Toy Car Racing Set",
    "Men's Formal Business Shirt",
    "Senior Citizen Walking Stick",
    "Teen Fashion Backpack"
]

print("\n🧪 Testing predictions on sample ad titles:\n")

for title in sample_titles:
    # Preprocess
    clean_title = clean_text(title)
    title_vec = vectorizer.transform([clean_title])
    
    # Predict
    prediction = model.predict(title_vec)[0]
    
    # Decode predictions
    decoded_pred = {
        col: label_encoders[col].inverse_transform([prediction[idx]])[0]
        for idx, col in enumerate(target_columns)
    }
    
    print(f"📝 Ad Title: {title}")
    print(f"   Age Group: {decoded_pred['target_age_group']}")
    print(f"   Gender: {decoded_pred['target_gender']}")
    print(f"   Mood: {decoded_pred['target_mood']}")
    print(f"   Weather: {decoded_pred['target_weather']}")
    print()

# ============================================================================
# STEP 9: Save Model Files
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Save Model Files")
print("=" * 80)

# Save all model components
print("\n💾 Saving model files...")

joblib.dump(model, 'ad_classifier_model.pkl')
print("✅ Saved: ad_classifier_model.pkl")

joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Saved: vectorizer.pkl")

joblib.dump(label_encoders, 'label_encoders.pkl')
print("✅ Saved: label_encoders.pkl")

# Save metadata
metadata = {
    'target_columns': target_columns,
    'classes': {col: le.classes_.tolist() for col, le in label_encoders.items()},
    'vocabulary_size': len(vectorizer.vocabulary_),
    'n_features': X_vectorized.shape[1],
    'n_samples_trained': X_train.shape[0],
    'accuracy_scores': {col: acc for col, acc in zip(target_columns, overall_accuracy)},
    'overall_accuracy': np.mean(overall_accuracy)
}

joblib.dump(metadata, 'model_metadata.pkl')
print("✅ Saved: model_metadata.pkl")

# ============================================================================
# STEP 10: Download Files
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Download Files to Your Computer")
print("=" * 80)

print("\n📥 Downloading files...\n")

files.download('ad_classifier_model.pkl')
files.download('vectorizer.pkl')
files.download('label_encoders.pkl')
files.download('model_metadata.pkl')

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)

print("""
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
   - Training samples: {X_train.shape[0]}
   - Test accuracy: {np.mean(overall_accuracy)*100:.2f}%
   - Features: {X_vectorized.shape[1]}
   
""")

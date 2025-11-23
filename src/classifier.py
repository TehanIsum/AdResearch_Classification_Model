"""
Ad Classification Module
=========================
This module handles loading the trained ML model and predicting
target categories for new ad titles.

Categories predicted:
- target_age_group: Kids, 10-18, 18-39, 40-64, 65+
- target_gender: Male, Female
- target_mood: Happy, Angry, Sad, Neutral
- target_weather: sunny, rainy, cold
"""

import os
import joblib
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class AdClassifier:
    """
    Machine Learning classifier for predicting ad target categories.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the classifier by loading trained model components.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.label_encoders = None
        self.metadata = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """
        Load all model components from disk.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            model_path = os.path.join(self.model_dir, 'ad_classifier_model.pkl')
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
            
            # Check if all files exist
            required_files = [model_path, vectorizer_path, encoders_path, metadata_path]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                print("❌ Error: Missing model files:")
                for file in missing_files:
                    print(f"   - {file}")
                print("\n💡 Please train the model first using Google Colab.")
                print("   See train_model_colab.py for instructions.")
                return False
            
            # Load model components
            print("📦 Loading trained model...")
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoders = joblib.load(encoders_path)
            self.metadata = joblib.load(metadata_path)
            
            self.is_loaded = True
            print("✅ Model loaded successfully!")
            print(f"   - Overall accuracy: {self.metadata['overall_accuracy']*100:.2f}%")
            print(f"   - Features: {self.metadata['n_features']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for prediction.
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def predict(self, ad_title: str) -> Optional[Dict[str, str]]:
        """
        Predict target categories for a given ad title.
        
        Args:
            ad_title: The ad title text
            
        Returns:
            Dict with predicted categories or None if error
        """
        if not self.is_loaded:
            print("❌ Model not loaded. Call load_model() first.")
            return None
        
        try:
            # Clean the text
            clean_title = self.clean_text(ad_title)
            
            if not clean_title:
                print("⚠️  Warning: Empty ad title after cleaning")
                return None
            
            # Vectorize
            title_vec = self.vectorizer.transform([clean_title])
            
            # Predict
            prediction = self.model.predict(title_vec)[0]
            
            # Decode predictions
            result = {}
            for idx, col in enumerate(self.metadata['target_columns']):
                encoded_value = prediction[idx]
                decoded_value = self.label_encoders[col].inverse_transform([encoded_value])[0]
                result[col] = decoded_value
            
            return result
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
            return None
    
    def predict_batch(self, ad_titles: list) -> list:
        """
        Predict target categories for multiple ad titles.
        
        Args:
            ad_titles: List of ad title texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for title in ad_titles:
            pred = self.predict(title)
            results.append(pred)
        return results
    
    def get_model_info(self) -> Optional[Dict]:
        """
        Get model metadata and performance information.
        
        Returns:
            Dict with model information or None if not loaded
        """
        if not self.is_loaded:
            return None
        
        return {
            'overall_accuracy': self.metadata['overall_accuracy'],
            'accuracy_by_category': self.metadata['accuracy_scores'],
            'classes': self.metadata['classes'],
            'n_features': self.metadata['n_features'],
            'n_samples_trained': self.metadata['n_samples_trained']
        }


# Example usage
if __name__ == "__main__":
    # Test the classifier
    classifier = AdClassifier()
    
    if classifier.load_model():
        # Test predictions
        test_titles = [
            "FDT Women's Leggings",
            "Kids Toy Car Racing Set",
            "Men's Formal Business Shirt",
            "Senior Walking Aid"
        ]
        
        print("\n" + "="*60)
        print("Testing Predictions")
        print("="*60)
        
        for title in test_titles:
            print(f"\n📝 Ad Title: {title}")
            prediction = classifier.predict(title)
            
            if prediction:
                print(f"   Age Group: {prediction['target_age_group']}")
                print(f"   Gender: {prediction['target_gender']}")
                print(f"   Mood: {prediction['target_mood']}")
                print(f"   Weather: {prediction['target_weather']}")

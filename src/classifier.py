"""
Ad Classification Module

This module handles loading the trained ML model and predicting
target categories for new ad titles.
"""

import os
import joblib
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class AdClassifier:
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.label_encoders = None
        self.metadata = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        try:
            model_path = os.path.join(self.model_dir, 'ad_classifier_model.pkl')
            vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            metadata_path = os.path.join(self.model_dir, 'model_metadata.pkl')
            
            required_files = [model_path, vectorizer_path, encoders_path, metadata_path]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                print("Error: Missing model files:")
                for file in missing_files:
                    print(f"  {file}")
                print("\nPlease train the model first using Google Colab.")
                print("See train_model_colab.py for instructions.")
                return False
            
            print("Loading trained model...")
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoders = joblib.load(encoders_path)
            self.metadata = joblib.load(metadata_path)
            
            self.is_loaded = True
            print("Model loaded successfully!")
            print(f"  Training samples: {self.metadata['n_samples_trained']}")
            print(f"  Overall accuracy: {self.metadata['overall_accuracy']*100:.2f}%")
            print(f"  Weather accuracy: {self.metadata['accuracy_scores']['target_weather']*100:.2f}%")
            print(f"  Features: {self.metadata['n_features']}")
            
            if self.metadata['n_samples_trained'] < 5000:
                print(f"\nWARNING: Model trained on only {self.metadata['n_samples_trained']} samples!")
                print("For better accuracy, consider retraining with a larger balanced dataset.")
                print("See train_model_colab.py for instructions.")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        text = str(text).lower()
        text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
        text = ' '.join(text.split())
        
        return text
    
    def predict(self, ad_title: str) -> Optional[Dict[str, str]]:
        if not self.is_loaded:
            print("Model not loaded. Call load_model() first.")
            return None
        
        try:
            clean_title = self.clean_text(ad_title)
            
            if not clean_title:
                print("Warning: Empty ad title after cleaning")
                return None
            
            title_vec = self.vectorizer.transform([clean_title])
            prediction = self.model.predict(title_vec)[0]
            
            result = {}
            for idx, col in enumerate(self.metadata['target_columns']):
                encoded_value = prediction[idx]
                decoded_value = self.label_encoders[col].inverse_transform([encoded_value])[0]
                result[col] = decoded_value
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def predict_batch(self, ad_titles: list) -> list:
        results = []
        for title in ad_titles:
            pred = self.predict(title)
            results.append(pred)
        return results
    
    def get_model_info(self) -> Optional[Dict]:
        if not self.is_loaded:
            return None
        
        return {
            'overall_accuracy': self.metadata['overall_accuracy'],
            'accuracy_by_category': self.metadata['accuracy_scores'],
            'classes': self.metadata['classes'],
            'n_features': self.metadata['n_features'],
            'n_samples_trained': self.metadata['n_samples_trained']
        }


if __name__ == "__main__":
    classifier = AdClassifier()
    
    if classifier.load_model():
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
            print(f"\nAd Title: {title}")
            prediction = classifier.predict(title)
            
            if prediction:
                print(f"  Age Group: {prediction['target_age_group']}")
                print(f"  Gender: {prediction['target_gender']}")
                print(f"  Mood: {prediction['target_mood']}")
                print(f"  Weather: {prediction['target_weather']}")

"""
Ad Classification and Recommendation System

Main application for the shopping mall billboard ad suggestion system.
"""

import sys
import os
import time
import csv
import random
import string
import pandas as pd
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.classifier import AdClassifier
from src.weather_service import WeatherService
from src.recommendation_engine import AdRecommendationEngine


class AdDisplaySystem:
    
    def __init__(self):
        self.classifier = AdClassifier()
        self.weather_service = WeatherService()
        self.recommendation_engine = AdRecommendationEngine()
        self.display_duration = 3
        
    def initialize(self) -> bool:
        print("="*70)
        print("AD CLASSIFICATION & RECOMMENDATION SYSTEM")
        print("="*70)
        print("\nInitializing system components...\n")
        
        try:
            self.classifier.load_model()
        except Exception as e:
            print(f"Classifier model not loaded: {e}")
            print("Classification features will be limited")
        
        if not self.recommendation_engine.load_ads_database():
            return False
        
        print()
        
        if not self.weather_service.check_api_key():
            print("Weather service not available - will use default 'sunny'")
        
        print("\nSystem initialized successfully!\n")
        return True
    
    def display_ad(self, ad: dict, target_values: dict):
        print("\n" + "="*70)
        print("DISPLAYING AD")
        print("="*70)
        
        print(f"\nAD TITLE: {ad['ad_title']}")
        print(f"Product ID: {ad['pid']}")
        print(f"Match Score: {ad['match_score']}/{ad['max_possible_score']} categories matched")
        
        print(f"\nTARGET AUDIENCE:")
        print(f"  Age Group: {ad['target_age_group']}")
        print(f"  Gender: {ad['target_gender']}")
        print(f"  Mood: {ad['target_mood']}")
        print(f"  Weather: {ad['target_weather']}")
        
        print(f"\nREQUESTED CRITERIA:")
        print(f"  Age Group: {target_values.get('target_age_group', 'N/A')}")
        print(f"  Gender: {target_values.get('target_gender', 'N/A')}")
        print(f"  Mood: {target_values.get('target_mood', 'N/A')}")
        print(f"  Weather: {target_values.get('target_weather', 'N/A')}")
        
        print("\n" + "="*70)
        
        for i in range(self.display_duration, 0, -1):
            print(f"Displaying for {i} more second(s)...", end='\r')
            time.sleep(1)
        
        print(" " * 50)
    
    def process_target_csv(self, csv_path: str):
        try:
            if not os.path.exists(csv_path):
                print(f"Error: File not found - {csv_path}")
                return
            
            print(f"\nReading target values from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            required_cols = ['target_age_group', 'target_gender', 'target_mood']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Error: Missing required columns in CSV: {missing_cols}")
                print(f"Required columns: {', '.join(required_cols)}")
                return
            
            print(f"Found {len(df)} target value rows to process\n")
            
            current_weather = "sunny"
            if self.weather_service.check_api_key():
                print("Fetching current weather...")
                weather_info = self.weather_service.get_detailed_weather_info()
                if weather_info:
                    current_weather = weather_info['category']
                    print(f"  Location: {weather_info['city']}, {weather_info['country']}")
                    print(f"  Condition: {weather_info['description']}")
                    print(f"  Temperature: {weather_info['temperature']}°C")
                    print(f"  Category: {current_weather}")
            
            print(f"\nStarting ad display sequence...\n")
            time.sleep(2)
            
            for idx, row in df.iterrows():
                print(f"\n{'='*70}")
                print(f"PROCESSING ROW {idx + 1} of {len(df)}")
                print(f"{'='*70}")
                
                target_values = {
                    'target_age_group': str(row.get('target_age_group', '18-39')).strip(),
                    'target_gender': str(row.get('target_gender', 'Male')).strip(),
                    'target_mood': str(row.get('target_mood', 'neutral')).strip(),
                    'target_weather': current_weather
                }
                
                print("\nFinding best matching ad...")
                best_ad = self.recommendation_engine.find_best_ad(target_values)
                
                if best_ad:
                    self.display_ad(best_ad, target_values)
                else:
                    print("No matching ad found for these criteria")
                
                if idx < len(df) - 1:
                    print("\nPreparing next ad...\n")
                    time.sleep(1)
            
            print("\n" + "="*70)
            print("All ads displayed successfully!")
            print("="*70)
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
    
    def predict_ad_categories(self, ad_title: str, save_to_dataset: bool = False) -> Optional[dict]:
        if not self.classifier.is_loaded:
            print("\nLoading ML classification model...")
            if not self.classifier.load_model():
                return None
        
        print(f"\nPredicting categories for: {ad_title}")
        prediction = self.classifier.predict(ad_title)
        
        if prediction:
            # Remove internal metadata fields
            prediction.pop('_confidence', None)
            prediction.pop('_vocabulary_match', None)
            
            print("\nPrediction Results:")
            print(f"  Age Group: {prediction['target_age_group']}")
            print(f"  Gender: {prediction['target_gender']}")
            print(f"  Mood: {prediction['target_mood']}")
            print(f"  Weather: {prediction['target_weather']}")
            
            if save_to_dataset:
                self.save_to_dataset(ad_title, prediction)
        
        return prediction
    
    def save_to_dataset(self, ad_title: str, prediction: dict) -> bool:
        try:
            
            dataset_path = "Classification model dataset.csv"
            
            pid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
            
            row_data = [
                pid,
                ad_title,
                prediction['target_age_group'],
                prediction['target_gender'],
                prediction['target_mood'],
                prediction['target_weather']
            ]
            
            with open(dataset_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
            
            print(f"\nSaved to dataset!")
            print(f"  Product ID: {pid}")
            print(f"  File: {dataset_path}")
            
            return True
            
        except Exception as e:
            print(f"\nError saving to dataset: {str(e)}")
            return False
    
    def interactive_mode(self):
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        print("\nOptions:")
        print("1. Display ads from target CSV file")
        print("2. Predict categories for a new ad title")
        print("3. Test weather service")
        print("4. View database statistics")
        print("5. Exit")
        
        while True:
            print("\n" + "-"*70)
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                csv_path = input("Enter path to target CSV file (or press Enter for example): ").strip()
                if not csv_path:
                    csv_path = "data/example_target_values.csv"
                self.process_target_csv(csv_path)
                
            elif choice == '2':
                ad_title = input("Enter ad title: ").strip()
                if ad_title:
                    prediction = self.predict_ad_categories(ad_title)
                    
                    if prediction:
                        save_choice = input("\nSave this prediction to the dataset? (y/n): ").strip().lower()
                        if save_choice in ['y', 'yes']:
                            self.save_to_dataset(ad_title, prediction)
                    
            elif choice == '3':
                city = input("Enter city name (or press Enter for default): ").strip()
                if city:
                    weather_info = self.weather_service.get_detailed_weather_info(city)
                else:
                    weather_info = self.weather_service.get_detailed_weather_info()
                    
                if weather_info:
                    print(f"\nLocation: {weather_info['city']}, {weather_info['country']}")
                    print(f"Temperature: {weather_info['temperature']}°C")
                    print(f"Condition: {weather_info['description']}")
                    print(f"Category: {weather_info['category']}")
                    
            elif choice == '4':
                stats = self.recommendation_engine.get_database_stats()
                if stats:
                    print(f"\nDatabase Statistics:")
                    print(f"  Total Ads: {stats['total_ads']}")
                    print(f"  Unique Ads: {stats['unique_ads']}")
                    print(f"\n  Age Groups: {stats['age_groups']}")
                    print(f"  Genders: {stats['genders']}")
                    print(f"  Moods: {stats['moods']}")
                    print(f"  Weather: {stats['weather']}")
                    
            elif choice == '5':
                print("\nGoodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-5.")


def main():
    system = AdDisplaySystem()
    
    if not system.initialize():
        print("\nSystem initialization failed. Exiting.")
        return
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("\n" + "="*70)
            print("USAGE")
            print("="*70)
            print("\n1. Process target CSV file:")
            print("   python main.py <path_to_target_csv>")
            print("\n2. Predict categories for ad title:")
            print("   python main.py --predict \"Your Ad Title Here\"")
            print("\n3. Interactive mode:")
            print("   python main.py")
            print("="*70)
            
        elif sys.argv[1] == '--predict':
            if len(sys.argv) > 2:
                ad_title = ' '.join(sys.argv[2:])
                system.predict_ad_categories(ad_title)
            else:
                print("Error: Please provide an ad title to predict")
                print("Usage: python main.py --predict \"Your Ad Title\"")
                
        else:
            csv_path = sys.argv[1]
            system.process_target_csv(csv_path)
    else:
        system.interactive_mode()


if __name__ == "__main__":
    main()

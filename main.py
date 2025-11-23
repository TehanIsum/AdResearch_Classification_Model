"""
Ad Classification and Recommendation System
============================================
Main application for the shopping mall billboard ad suggestion system.

This system:
1. Reads target values from CSV (row by row)
2. Fetches current weather using API
3. Finds best matching ad from database
4. Displays ad for 3 seconds
5. Repeats for next row

Also supports:
- Predicting categories for new ad titles using ML model
"""

import sys
import os
import time
import pandas as pd
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.classifier import AdClassifier
from src.weather_service import WeatherService
from src.recommendation_engine import AdRecommendationEngine


class AdDisplaySystem:
    """
    Main system for displaying billboard ads based on target values.
    """
    
    def __init__(self):
        """Initialize the ad display system."""
        self.classifier = AdClassifier()
        self.weather_service = WeatherService()
        self.recommendation_engine = AdRecommendationEngine()
        self.display_duration = 3  # seconds
        
    def initialize(self) -> bool:
        """
        Initialize all components of the system.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("="*70)
        print("🎯 AD CLASSIFICATION & RECOMMENDATION SYSTEM")
        print("="*70)
        print("\n🔧 Initializing system components...\n")
        
        # Load classifier model
        try:
            self.classifier.load_model()
        except Exception as e:
            print(f"⚠️  Classifier model not loaded: {e}")
            print("   Classification features will be limited")
        
        # Load recommendation engine
        if not self.recommendation_engine.load_ads_database():
            return False
        
        print()
        
        # Check weather service (optional)
        if not self.weather_service.check_api_key():
            print("⚠️  Weather service not available - will use default 'sunny'")
        
        print("\n✅ System initialized successfully!\n")
        return True
    
    def display_ad(self, ad: dict, target_values: dict):
        """
        Display an ad in the terminal with formatting.
        
        Args:
            ad: Ad information dictionary
            target_values: Target values that triggered this ad
        """
        print("\n" + "="*70)
        print("📺 DISPLAYING AD")
        print("="*70)
        
        print(f"\n🎯 AD TITLE: {ad['ad_title']}")
        print(f"📋 Product ID: {ad['pid']}")
        print(f"⭐ Match Score: {ad['match_score']}/{ad['max_possible_score']} categories matched")
        
        print(f"\n📊 TARGET AUDIENCE:")
        print(f"   👥 Age Group: {ad['target_age_group']}")
        print(f"   👤 Gender: {ad['target_gender']}")
        print(f"   😊 Mood: {ad['target_mood']}")
        print(f"   🌤️  Weather: {ad['target_weather']}")
        
        print(f"\n🎬 REQUESTED CRITERIA:")
        print(f"   👥 Age Group: {target_values.get('target_age_group', 'N/A')}")
        print(f"   👤 Gender: {target_values.get('target_gender', 'N/A')}")
        print(f"   😊 Mood: {target_values.get('target_mood', 'N/A')}")
        print(f"   🌤️  Weather: {target_values.get('target_weather', 'N/A')}")
        
        print("\n" + "="*70)
        
        # Display for specified duration
        for i in range(self.display_duration, 0, -1):
            print(f"⏱️  Displaying for {i} more second(s)...", end='\r')
            time.sleep(1)
        
        print(" " * 50)  # Clear the countdown line
    
    def process_target_csv(self, csv_path: str):
        """
        Process target values from CSV file row by row.
        
        Args:
            csv_path: Path to CSV file with target values
        """
        try:
            if not os.path.exists(csv_path):
                print(f"❌ Error: File not found - {csv_path}")
                return
            
            print(f"\n📂 Reading target values from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Verify required columns
            required_cols = ['target_age_group', 'target_gender', 'target_mood']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️  Warning: Missing columns in CSV: {missing_cols}")
                print("   Will use default values for missing columns")
            
            print(f"✅ Found {len(df)} target value rows to process\n")
            
            # Get current weather
            current_weather = "sunny"  # Default
            if self.weather_service.check_api_key():
                print("🌤️  Fetching current weather...")
                weather_info = self.weather_service.get_detailed_weather_info()
                if weather_info:
                    current_weather = weather_info['category']
                    print(f"   Location: {weather_info['city']}, {weather_info['country']}")
                    print(f"   Condition: {weather_info['description']}")
                    print(f"   Temperature: {weather_info['temperature']}°C")
                    print(f"   Category: {current_weather}")
            
            print(f"\n🚀 Starting ad display sequence...\n")
            time.sleep(2)
            
            # Process each row
            for idx, row in df.iterrows():
                print(f"\n{'='*70}")
                print(f"📍 PROCESSING ROW {idx + 1} of {len(df)}")
                print(f"{'='*70}")
                
                # Build target values
                target_values = {
                    'target_age_group': row.get('target_age_group', '18-39'),
                    'target_gender': row.get('target_gender', 'Male'),
                    'target_mood': row.get('target_mood', 'neutral'),
                    'target_weather': row.get('target_weather', current_weather)
                }
                
                # If weather not in CSV, use current weather
                if pd.isna(row.get('target_weather')):
                    target_values['target_weather'] = current_weather
                
                # Find best matching ad
                print("\n🔍 Finding best matching ad...")
                best_ad = self.recommendation_engine.find_best_ad(target_values)
                
                if best_ad:
                    self.display_ad(best_ad, target_values)
                else:
                    print("❌ No matching ad found for these criteria")
                
                # Pause between ads (except for last one)
                if idx < len(df) - 1:
                    print("\n⏸️  Preparing next ad...\n")
                    time.sleep(1)
            
            print("\n" + "="*70)
            print("✅ All ads displayed successfully!")
            print("="*70)
            
        except Exception as e:
            print(f"❌ Error processing CSV: {str(e)}")
    
    def predict_ad_categories(self, ad_title: str) -> Optional[dict]:
        """
        Predict categories for a new ad title using ML model.
        
        Args:
            ad_title: The ad title to classify
            
        Returns:
            Dict with predicted categories or None if error
        """
        # Load model if not already loaded
        if not self.classifier.is_loaded:
            print("\n🤖 Loading ML classification model...")
            if not self.classifier.load_model():
                return None
        
        print(f"\n🔮 Predicting categories for: {ad_title}")
        prediction = self.classifier.predict(ad_title)
        
        if prediction:
            print("\n✅ Prediction Results:")
            print(f"   👥 Age Group: {prediction['target_age_group']}")
            print(f"   👤 Gender: {prediction['target_gender']}")
            print(f"   😊 Mood: {prediction['target_mood']}")
            print(f"   🌤️  Weather: {prediction['target_weather']}")
        
        return prediction
    
    def interactive_mode(self):
        """
        Run system in interactive mode for testing.
        """
        print("\n" + "="*70)
        print("🎮 INTERACTIVE MODE")
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
                    self.predict_ad_categories(ad_title)
                    
            elif choice == '3':
                city = input("Enter city name (or press Enter for default): ").strip()
                if city:
                    weather_info = self.weather_service.get_detailed_weather_info(city)
                else:
                    weather_info = self.weather_service.get_detailed_weather_info()
                    
                if weather_info:
                    print(f"\n🌍 Location: {weather_info['city']}, {weather_info['country']}")
                    print(f"🌡️  Temperature: {weather_info['temperature']}°C")
                    print(f"☁️  Condition: {weather_info['description']}")
                    print(f"🎯 Category: {weather_info['category']}")
                    
            elif choice == '4':
                stats = self.recommendation_engine.get_database_stats()
                if stats:
                    print(f"\n📊 Database Statistics:")
                    print(f"   Total Ads: {stats['total_ads']}")
                    print(f"   Unique Ads: {stats['unique_ads']}")
                    print(f"\n   Age Groups: {stats['age_groups']}")
                    print(f"   Genders: {stats['genders']}")
                    print(f"   Moods: {stats['moods']}")
                    print(f"   Weather: {stats['weather']}")
                    
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")


def main():
    """Main entry point for the application."""
    system = AdDisplaySystem()
    
    # Initialize system
    if not system.initialize():
        print("\n❌ System initialization failed. Exiting.")
        return
    
    # Check for command line arguments
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
                print("❌ Error: Please provide an ad title to predict")
                print("   Usage: python main.py --predict \"Your Ad Title\"")
                
        else:
            # Assume it's a CSV file path
            csv_path = sys.argv[1]
            system.process_target_csv(csv_path)
    else:
        # Run in interactive mode
        system.interactive_mode()


if __name__ == "__main__":
    main()

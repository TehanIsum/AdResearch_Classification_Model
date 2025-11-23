"""
Weather API Integration Module
===============================
This module fetches current weather data and maps it to target weather categories.

Weather Categories:
- sunny: Clear, sunny conditions
- rainy: Rain, drizzle, thunderstorm
- cold: Snow, cold temperatures, winter conditions
"""

import os
import requests
from typing import Optional, Dict
from dotenv import load_dotenv


class WeatherService:
    """
    Service for fetching and categorizing weather data.
    """
    
    def __init__(self):
        """
        Initialize weather service with API credentials.
        """
        load_dotenv()
        self.api_key = os.getenv('WEATHER_API_KEY', '')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.default_city = os.getenv('DEFAULT_CITY', 'London')
        self.default_country = os.getenv('DEFAULT_COUNTRY', 'UK')
        
    def check_api_key(self) -> bool:
        """
        Check if API key is configured.
        
        Returns:
            bool: True if API key exists
        """
        if not self.api_key or self.api_key == 'your_openweathermap_api_key_here':
            print("\n⚠️  Weather API key not configured!")
            print("   Please add your API key to the .env file:")
            print("   WEATHER_API_KEY=your_actual_api_key")
            print("\n   Get a free API key at: https://openweathermap.org/api")
            return False
        return True
    
    def get_weather_by_city(self, city: str, country: str = "") -> Optional[Dict]:
        """
        Fetch weather data for a specific city.
        
        Args:
            city: City name
            country: Country code (optional)
            
        Returns:
            Dict with weather data or None if error
        """
        if not self.check_api_key():
            return None
        
        try:
            # Build location query
            location = f"{city},{country}" if country else city
            
            # API request
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'  # Use Celsius
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching weather data: {str(e)}")
            return None
    
    def get_current_location_weather(self) -> Optional[Dict]:
        """
        Fetch weather data for the default location.
        
        Returns:
            Dict with weather data or None if error
        """
        return self.get_weather_by_city(self.default_city, self.default_country)
    
    def categorize_weather(self, weather_data: Dict) -> str:
        """
        Categorize weather data into target weather categories.
        
        Args:
            weather_data: Raw weather data from API
            
        Returns:
            str: Weather category (sunny, rainy, cold)
        """
        if not weather_data:
            return "sunny"  # Default fallback
        
        try:
            # Extract weather information
            weather_main = weather_data['weather'][0]['main'].lower()
            weather_desc = weather_data['weather'][0]['description'].lower()
            temp = weather_data['main']['temp']
            
            # Categorization logic
            # Priority: rainy > cold > sunny
            
            # Check for rainy conditions
            rainy_conditions = ['rain', 'drizzle', 'thunderstorm', 'shower']
            if any(cond in weather_main or cond in weather_desc for cond in rainy_conditions):
                return "rainy"
            
            # Check for cold conditions (snow or temperature below 10°C)
            cold_conditions = ['snow', 'sleet', 'ice', 'freezing']
            if any(cond in weather_main or cond in weather_desc for cond in cold_conditions):
                return "cold"
            
            if temp < 10:
                return "cold"
            
            # Default to sunny for clear/cloudy/other conditions
            return "sunny"
            
        except (KeyError, IndexError) as e:
            print(f"⚠️  Error parsing weather data: {str(e)}")
            return "sunny"  # Default fallback
    
    def get_categorized_weather(self, city: Optional[str] = None, 
                                country: Optional[str] = None) -> str:
        """
        Get categorized weather for a location.
        
        Args:
            city: City name (optional, uses default if not provided)
            country: Country code (optional)
            
        Returns:
            str: Weather category (sunny, rainy, cold)
        """
        if city:
            weather_data = self.get_weather_by_city(city, country or "")
        else:
            weather_data = self.get_current_location_weather()
        
        return self.categorize_weather(weather_data)
    
    def get_detailed_weather_info(self, city: Optional[str] = None,
                                  country: Optional[str] = None) -> Optional[Dict]:
        """
        Get detailed weather information including category.
        
        Args:
            city: City name (optional)
            country: Country code (optional)
            
        Returns:
            Dict with detailed weather info or None if error
        """
        if city:
            weather_data = self.get_weather_by_city(city, country or "")
        else:
            weather_data = self.get_current_location_weather()
        
        if not weather_data:
            return None
        
        try:
            return {
                'city': weather_data['name'],
                'country': weather_data['sys']['country'],
                'temperature': weather_data['main']['temp'],
                'feels_like': weather_data['main']['feels_like'],
                'description': weather_data['weather'][0]['description'],
                'main': weather_data['weather'][0]['main'],
                'humidity': weather_data['main']['humidity'],
                'category': self.categorize_weather(weather_data)
            }
        except (KeyError, IndexError) as e:
            print(f"❌ Error parsing weather data: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    weather_service = WeatherService()
    
    print("="*60)
    print("Weather Service Test")
    print("="*60)
    
    # Test current location weather
    print("\n📍 Fetching weather for default location...")
    weather_info = weather_service.get_detailed_weather_info()
    
    if weather_info:
        print(f"\n🌍 Location: {weather_info['city']}, {weather_info['country']}")
        print(f"🌡️  Temperature: {weather_info['temperature']}°C")
        print(f"🌡️  Feels like: {weather_info['feels_like']}°C")
        print(f"☁️  Condition: {weather_info['description']}")
        print(f"💧 Humidity: {weather_info['humidity']}%")
        print(f"\n🎯 Target Weather Category: {weather_info['category']}")
    else:
        print("\n⚠️  Could not fetch weather data.")
        print("   Using default category: sunny")

"""
Weather API Integration Module

This module fetches current weather data and maps it to target weather categories.
"""

import os
import requests
from typing import Optional, Dict
from dotenv import load_dotenv


class WeatherService:
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('WEATHER_API_KEY', '')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.default_city = os.getenv('DEFAULT_CITY', 'London')
        self.default_country = os.getenv('DEFAULT_COUNTRY', 'UK')
        
    def check_api_key(self) -> bool:
        if not self.api_key or self.api_key == 'your_openweathermap_api_key_here':
            print("\nWeather API key not configured!")
            print("Please add your API key to the .env file:")
            print("WEATHER_API_KEY=your_actual_api_key")
            print("\nGet a free API key at: https://openweathermap.org/api")
            return False
        return True
    
    def get_weather_by_city(self, city: str, country: str = "") -> Optional[Dict]:
        if not self.check_api_key():
            return None
        
        try:
            location = f"{city},{country}" if country else city
            
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {str(e)}")
            return None
    
    def get_current_location_weather(self) -> Optional[Dict]:
        return self.get_weather_by_city(self.default_city, self.default_country)
    
    def categorize_weather(self, weather_data: Dict) -> str:
        if not weather_data:
            return "sunny"
        
        try:
            weather_main = weather_data['weather'][0]['main'].lower()
            weather_desc = weather_data['weather'][0]['description'].lower()
            temp = weather_data['main']['temp']
            
            rainy_conditions = ['rain', 'drizzle', 'thunderstorm', 'shower', 'overcast']
            if any(cond in weather_main or cond in weather_desc for cond in rainy_conditions):
                return "rainy"
            
            cold_conditions = ['snow', 'sleet', 'ice', 'freezing']
            if any(cond in weather_main or cond in weather_desc for cond in cold_conditions):
                return "cold"
            
            if temp < 10:
                return "cold"
            
            return "sunny"
            
        except (KeyError, IndexError) as e:
            print(f"Error parsing weather data: {str(e)}")
            return "sunny"
    
    def get_categorized_weather(self, city: Optional[str] = None, 
                                country: Optional[str] = None) -> str:
        if city:
            weather_data = self.get_weather_by_city(city, country or "")
        else:
            weather_data = self.get_current_location_weather()
        
        return self.categorize_weather(weather_data)
    
    def get_detailed_weather_info(self, city: Optional[str] = None,
                                  country: Optional[str] = None) -> Optional[Dict]:
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
            print(f"Error parsing weather data: {str(e)}")
            return None


if __name__ == "__main__":
    weather_service = WeatherService()
    
    print("="*60)
    print("Weather Service Test")
    print("="*60)
    
    print("\nFetching weather for default location...")
    weather_info = weather_service.get_detailed_weather_info()
    
    if weather_info:
        print(f"\nLocation: {weather_info['city']}, {weather_info['country']}")
        print(f"Temperature: {weather_info['temperature']}°C")
        print(f"Feels like: {weather_info['feels_like']}°C")
        print(f"Condition: {weather_info['description']}")
        print(f"Humidity: {weather_info['humidity']}%")
        print(f"\nTarget Weather Category: {weather_info['category']}")
    else:
        print("\nCould not fetch weather data.")
        print("Using default category: sunny")

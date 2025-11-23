"""
Ad Classification & Recommendation System
==========================================

Core modules for the billboard advertisement suggestion system.

Modules:
    - classifier: ML-based ad category prediction
    - weather_service: Weather API integration and categorization
    - recommendation_engine: Ad matching and recommendation logic
"""

__version__ = "1.0.0"
__author__ = "AdResearch Team"

from .classifier import AdClassifier
from .weather_service import WeatherService
from .recommendation_engine import AdRecommendationEngine

__all__ = [
    'AdClassifier',
    'WeatherService',
    'AdRecommendationEngine'
]

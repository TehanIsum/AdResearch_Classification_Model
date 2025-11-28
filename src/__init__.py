
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

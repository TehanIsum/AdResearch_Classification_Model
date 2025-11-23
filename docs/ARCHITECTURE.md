# System Architecture

## Overview

The Ad Classification & Recommendation System is built with a modular architecture consisting of three main components that work together to classify ads and recommend the best-fit advertisements.

## Component Architecture

### 1. Classification Module (`src/classifier.py`)

**Purpose**: Load and use trained ML model to predict target categories for new ad titles.

**Key Classes**:
- `AdClassifier`: Main classifier class

**Methods**:
- `load_model()`: Loads trained model components from disk
- `predict(ad_title)`: Predicts target categories for a single ad
- `predict_batch(ad_titles)`: Predicts for multiple ads
- `clean_text(text)`: Preprocesses text input

**Model Components**:
- `ad_classifier_model.pkl`: Multi-output Random Forest classifier
- `vectorizer.pkl`: TF-IDF vectorizer for text features
- `label_encoders.pkl`: Label encoders for each target category
- `model_metadata.pkl`: Model performance metrics and metadata

**ML Pipeline**:
```
Raw Text → Clean Text → TF-IDF Vectorization → Multi-Output Classifier → Predictions
```

### 2. Weather Service Module (`src/weather_service.py`)

**Purpose**: Fetch real-time weather data and categorize it into target weather categories.

**Key Classes**:
- `WeatherService`: Weather API integration

**Methods**:
- `get_weather_by_city(city, country)`: Fetch weather for specific location
- `get_current_location_weather()`: Fetch weather for default location
- `categorize_weather(weather_data)`: Map weather to categories (sunny/rainy/cold)
- `get_detailed_weather_info()`: Get comprehensive weather information

**Weather Categorization Logic**:
```
API Response → Extract Conditions & Temperature → Categorize:
  - Rainy: Rain, drizzle, thunderstorm, shower
  - Cold: Snow, ice, freezing, or temp < 10°C
  - Sunny: Default (clear, cloudy, etc.)
```

**External Dependency**:
- OpenWeatherMap API (http://api.openweathermap.org/data/2.5/weather)

### 3. Recommendation Engine Module (`src/recommendation_engine.py`)

**Purpose**: Match target values with stored ads and find best-fit advertisements.

**Key Classes**:
- `AdRecommendationEngine`: Ad matching and recommendation

**Methods**:
- `load_ads_database()`: Load ads from CSV database
- `find_best_ad(target_values)`: Find single best matching ad
- `find_top_n_ads(target_values, n)`: Find top N matches
- `calculate_match_score(target, ad)`: Score ad-target similarity (0-4)
- `get_database_stats()`: Get database statistics

**Matching Algorithm**:
```
For each ad in database:
  score = 0
  if ad.age_group == target.age_group: score += 1
  if ad.gender == target.gender: score += 1
  if ad.mood == target.mood: score += 1
  if ad.weather == target.weather: score += 1
  
Return ad with highest score
```

### 4. Main Application (`main.py`)

**Purpose**: Orchestrate all components and provide user interface.

**Key Classes**:
- `AdDisplaySystem`: Main application controller

**Core Workflow**:
```
1. Initialize System
   ├── Load ads database
   ├── Check weather service
   └── Prepare for model loading (lazy)

2. Process Target CSV
   ├── Read CSV row by row
   ├── Fetch current weather (if needed)
   ├── Build target profile
   ├── Find best matching ad
   ├── Display ad for 3 seconds
   └── Repeat for next row

3. Alternative: Predict Categories
   ├── Load ML model (if not loaded)
   ├── Clean ad title
   └── Predict and display categories
```

## Data Flow

### Ad Display Flow
```
Target CSV → Read Row → Build Target Profile
                              ↓
                    Weather API (if needed)
                              ↓
                    Recommendation Engine
                              ↓
                    Calculate Match Scores
                              ↓
                    Select Best Ad
                              ↓
                    Display for 3 seconds
                              ↓
                    Next Row (loop)
```

### Classification Flow
```
New Ad Title → Clean Text → TF-IDF Vectorization → ML Model → Predictions
                                                                    ↓
                                                    [age, gender, mood, weather]
```

## Technology Stack

### Machine Learning
- **scikit-learn**: Random Forest Classifier, TF-IDF Vectorization
- **Multi-Output Classification**: Predicts 4 categories simultaneously
- **Text Processing**: NLTK for tokenization and preprocessing

### Data Processing
- **pandas**: CSV handling, data manipulation
- **numpy**: Numerical operations

### API Integration
- **requests**: HTTP client for Weather API
- **python-dotenv**: Environment variable management

### Serialization
- **joblib**: Model persistence and loading

## Model Architecture

### Training Architecture
```
Dataset (20,000+ ads)
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization (5000 features, n-grams: 1-3)
    ↓
Train-Test Split (80-20)
    ↓
Multi-Output Random Forest
├── Estimator 1: Age Group (5 classes)
├── Estimator 2: Gender (2 classes)
├── Estimator 3: Mood (4 classes)
└── Estimator 4: Weather (3 classes)
    ↓
Evaluation & Serialization
```

### Model Parameters
- **Algorithm**: Random Forest with Multi-Output Classifier
- **N Estimators**: 200 trees
- **Max Depth**: 30
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Features**: 5000 TF-IDF features
- **N-grams**: 1-3 (unigrams, bigrams, trigrams)

## System Requirements

### Hardware
- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB RAM, 5GB disk space
- **Training**: Google Colab (free tier sufficient)

### Software
- **Python**: 3.8+
- **Operating System**: macOS, Linux, Windows
- **Internet**: Required for weather API

### Dependencies
- Core ML: scikit-learn, pandas, numpy
- NLP: nltk
- API: requests, python-dotenv
- Utilities: joblib, tqdm

## Performance Characteristics

### Model Performance
- **Training Time**: 5-10 minutes (Google Colab)
- **Inference Time**: <100ms per prediction
- **Model Size**: ~50-100MB (all files combined)
- **Accuracy**: 85-95% (varies by category)

### System Performance
- **Database Load Time**: 1-2 seconds
- **Ad Recommendation Time**: <50ms
- **Weather API Call**: 200-500ms
- **Display Rate**: 1 ad per 3 seconds

## Security Considerations

### Environment Variables
- API keys stored in `.env` file
- `.env` excluded from version control
- Environment validation on startup

### Data Privacy
- No user data collected
- Weather API: Only location data
- All processing done locally

### API Rate Limits
- OpenWeatherMap Free Tier: 60 calls/minute, 1,000,000 calls/month
- System caches weather for session duration

## Scalability

### Current Limitations
- Single-threaded processing
- In-memory database loading
- Sequential ad display

### Potential Improvements
- Database indexing for faster lookups
- Batch processing for multiple simultaneous displays
- Model serving with REST API
- Real-time model updates
- Multi-location weather caching

## Error Handling

### Graceful Degradation
1. **Model Not Found**: System prompts user to train model
2. **Weather API Failure**: Falls back to "sunny" default
3. **Missing CSV Columns**: Uses defaults for missing values
4. **Invalid Ad Title**: Returns None, logs warning

### Error Recovery
- All components validate inputs
- Exceptions caught and logged
- System continues operation when possible

## Extensibility

### Adding New Categories
1. Update dataset with new category column
2. Retrain model in Google Colab
3. Update label encoders
4. Modify UI to display new category

### Supporting New Weather APIs
1. Implement new weather service class
2. Inherit from base interface
3. Update configuration
4. Maintain same categorization logic

### Custom Matching Algorithms
1. Modify `calculate_match_score()` in recommendation engine
2. Implement weighted scoring
3. Add machine learning for match prediction

## Testing Strategy

### Unit Testing
- Classifier: Mock model files, test predictions
- Weather: Mock API responses, test categorization
- Recommendation: Test scoring algorithm

### Integration Testing
- End-to-end CSV processing
- API integration with test keys
- Model loading and prediction pipeline

### Manual Testing
- Interactive mode for quick testing
- Example CSV for validation
- Sample predictions for verification

## Deployment

### Local Deployment
1. Install dependencies
2. Configure environment
3. Train and download model
4. Run `python main.py`

### Production Considerations
- Use production API keys
- Implement logging
- Monitor performance
- Set up error alerting
- Schedule model retraining

---

**Last Updated**: November 2025

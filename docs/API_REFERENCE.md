# API Reference

Complete API documentation for the Ad Classification & Recommendation System.

## Table of Contents

- [Classifier Module](#classifier-module)
- [Weather Service Module](#weather-service-module)
- [Recommendation Engine Module](#recommendation-engine-module)
- [Main Application](#main-application)

---

## Classifier Module

Location: `src/classifier.py`

### Class: `AdClassifier`

Machine Learning classifier for predicting ad target categories.

#### Constructor

```python
AdClassifier(model_dir: str = "models")
```

**Parameters:**
- `model_dir` (str): Directory containing model files. Default: `"models"`

**Example:**
```python
from src.classifier import AdClassifier

classifier = AdClassifier(model_dir="models")
```

---

#### Method: `load_model()`

Load all model components from disk.

```python
load_model() -> bool
```

**Returns:**
- `bool`: `True` if successful, `False` otherwise

**Raises:**
- Prints error messages if files are missing or corrupted

**Example:**
```python
if classifier.load_model():
    print("Model loaded successfully")
else:
    print("Failed to load model")
```

**Required Files:**
- `ad_classifier_model.pkl`
- `vectorizer.pkl`
- `label_encoders.pkl`
- `model_metadata.pkl`

---

#### Method: `predict()`

Predict target categories for a given ad title.

```python
predict(ad_title: str) -> Optional[Dict[str, str]]
```

**Parameters:**
- `ad_title` (str): The ad title text to classify

**Returns:**
- `Dict[str, str]`: Dictionary with predicted categories:
  - `target_age_group`: Kids, 10-18, 18-39, 40-64, 65+
  - `target_gender`: Male, Female
  - `target_mood`: Happy, Angry, Sad, Neutral
  - `target_weather`: sunny, rainy, cold
- `None`: If error occurs

**Example:**
```python
prediction = classifier.predict("Women's Fashion Leggings")
print(prediction)
# Output:
# {
#     'target_age_group': '18-39',
#     'target_gender': 'Female',
#     'target_mood': 'neutral',
#     'target_weather': 'sunny'
# }
```

---

#### Method: `predict_batch()`

Predict target categories for multiple ad titles.

```python
predict_batch(ad_titles: list) -> list
```

**Parameters:**
- `ad_titles` (list): List of ad title strings

**Returns:**
- `list`: List of prediction dictionaries (same format as `predict()`)

**Example:**
```python
titles = [
    "Women's Fashion Leggings",
    "Kids Toy Car Set",
    "Men's Business Shirt"
]

predictions = classifier.predict_batch(titles)
for title, pred in zip(titles, predictions):
    print(f"{title}: {pred['target_age_group']}, {pred['target_gender']}")
```

---

#### Method: `clean_text()`

Clean and preprocess text for prediction.

```python
clean_text(text: str) -> str
```

**Parameters:**
- `text` (str): Raw text to clean

**Returns:**
- `str`: Cleaned text (lowercase, no special chars, normalized spaces)

**Example:**
```python
raw = "Women's @Fashion# Leggings!!!"
clean = classifier.clean_text(raw)
print(clean)  # Output: "womens fashion leggings"
```

---

#### Method: `get_model_info()`

Get model metadata and performance information.

```python
get_model_info() -> Optional[Dict]
```

**Returns:**
- `Dict`: Model information including:
  - `overall_accuracy`: Average accuracy across all categories
  - `accuracy_by_category`: Per-category accuracy scores
  - `classes`: Available classes for each category
  - `n_features`: Number of features used
  - `n_samples_trained`: Number of training samples
- `None`: If model not loaded

**Example:**
```python
info = classifier.get_model_info()
print(f"Overall Accuracy: {info['overall_accuracy']:.2%}")
print(f"Categories: {list(info['accuracy_by_category'].keys())}")
```

---

## Weather Service Module

Location: `src/weather_service.py`

### Class: `WeatherService`

Service for fetching and categorizing weather data.

#### Constructor

```python
WeatherService()
```

Automatically loads configuration from `.env` file.

**Environment Variables:**
- `WEATHER_API_KEY`: OpenWeatherMap API key
- `DEFAULT_CITY`: Default city name
- `DEFAULT_COUNTRY`: Default country code

**Example:**
```python
from src.weather_service import WeatherService

weather = WeatherService()
```

---

#### Method: `check_api_key()`

Check if API key is configured.

```python
check_api_key() -> bool
```

**Returns:**
- `bool`: `True` if API key is valid, `False` otherwise

**Example:**
```python
if weather.check_api_key():
    # Proceed with API calls
    pass
else:
    print("Please configure WEATHER_API_KEY in .env")
```

---

#### Method: `get_weather_by_city()`

Fetch weather data for a specific city.

```python
get_weather_by_city(city: str, country: str = "") -> Optional[Dict]
```

**Parameters:**
- `city` (str): City name
- `country` (str): Country code (optional, e.g., "UK", "US")

**Returns:**
- `Dict`: Raw weather data from OpenWeatherMap API
- `None`: If error occurs

**Example:**
```python
weather_data = weather.get_weather_by_city("London", "UK")
if weather_data:
    print(f"Temperature: {weather_data['main']['temp']}°C")
    print(f"Condition: {weather_data['weather'][0]['description']}")
```

**API Response Structure:**
```python
{
    'name': 'London',
    'sys': {'country': 'UK'},
    'main': {
        'temp': 15.5,
        'feels_like': 14.2,
        'humidity': 72
    },
    'weather': [{
        'main': 'Rain',
        'description': 'light rain'
    }]
}
```

---

#### Method: `get_current_location_weather()`

Fetch weather data for the default location.

```python
get_current_location_weather() -> Optional[Dict]
```

**Returns:**
- `Dict`: Weather data (same format as `get_weather_by_city()`)
- `None`: If error occurs

**Example:**
```python
weather_data = weather.get_current_location_weather()
```

---

#### Method: `categorize_weather()`

Categorize weather data into target weather categories.

```python
categorize_weather(weather_data: Dict) -> str
```

**Parameters:**
- `weather_data` (Dict): Raw weather data from API

**Returns:**
- `str`: Weather category:
  - `"sunny"`: Clear, cloudy, or other non-rainy/cold conditions
  - `"rainy"`: Rain, drizzle, thunderstorm
  - `"cold"`: Snow, ice, or temperature < 10°C

**Categorization Logic:**
1. **Rainy**: If weather contains rain, drizzle, thunderstorm, shower
2. **Cold**: If weather contains snow, ice, freezing OR temp < 10°C
3. **Sunny**: Default for all other conditions

**Example:**
```python
weather_data = weather.get_weather_by_city("London", "UK")
category = weather.categorize_weather(weather_data)
print(f"Weather category: {category}")  # Output: "rainy" or "sunny" or "cold"
```

---

#### Method: `get_categorized_weather()`

Get categorized weather for a location (one-step function).

```python
get_categorized_weather(city: Optional[str] = None, 
                       country: Optional[str] = None) -> str
```

**Parameters:**
- `city` (str, optional): City name. If not provided, uses default.
- `country` (str, optional): Country code

**Returns:**
- `str`: Weather category ("sunny", "rainy", "cold")

**Example:**
```python
# Use default location
category = weather.get_categorized_weather()

# Specify location
category = weather.get_categorized_weather(city="Paris", country="FR")
print(f"Weather: {category}")
```

---

#### Method: `get_detailed_weather_info()`

Get detailed weather information including category.

```python
get_detailed_weather_info(city: Optional[str] = None,
                         country: Optional[str] = None) -> Optional[Dict]
```

**Parameters:**
- `city` (str, optional): City name
- `country` (str, optional): Country code

**Returns:**
- `Dict`: Detailed weather information:
  ```python
  {
      'city': 'London',
      'country': 'UK',
      'temperature': 15.5,
      'feels_like': 14.2,
      'description': 'light rain',
      'main': 'Rain',
      'humidity': 72,
      'category': 'rainy'  # Categorized
  }
  ```
- `None`: If error occurs

**Example:**
```python
info = weather.get_detailed_weather_info(city="New York", country="US")
if info:
    print(f"📍 {info['city']}, {info['country']}")
    print(f"🌡️  {info['temperature']}°C")
    print(f"☁️  {info['description']}")
    print(f"🎯 Category: {info['category']}")
```

---

## Recommendation Engine Module

Location: `src/recommendation_engine.py`

### Class: `AdRecommendationEngine`

Engine for matching target values with available ads.

#### Constructor

```python
AdRecommendationEngine(ads_database_path: str = "Classification model dataset.csv")
```

**Parameters:**
- `ads_database_path` (str): Path to CSV file with ads database

**Example:**
```python
from src.recommendation_engine import AdRecommendationEngine

engine = AdRecommendationEngine()
# or
engine = AdRecommendationEngine("path/to/custom_ads.csv")
```

---

#### Method: `load_ads_database()`

Load the ads database from CSV.

```python
load_ads_database() -> bool
```

**Returns:**
- `bool`: `True` if successful, `False` otherwise

**Required CSV Columns:**
- `pid`: Product/Ad ID
- `ad_title`: Ad title text
- `target_age_group`: Age group
- `target_gender`: Gender
- `target_mood`: Mood
- `target_weather`: Weather

**Example:**
```python
if engine.load_ads_database():
    print(f"Loaded {len(engine.ads_df)} ads")
else:
    print("Failed to load database")
```

---

#### Method: `calculate_match_score()`

Calculate how well an ad matches target values.

```python
calculate_match_score(target: Dict[str, str], ad: pd.Series) -> int
```

**Parameters:**
- `target` (Dict): Target values dictionary
- `ad` (pd.Series): Ad data as pandas Series

**Returns:**
- `int`: Match score (0-4, higher is better)
  - +1 for each matching category

**Example:**
```python
target = {
    'target_age_group': '18-39',
    'target_gender': 'Female',
    'target_mood': 'neutral',
    'target_weather': 'sunny'
}

# For internal use - called by find_best_ad()
score = engine.calculate_match_score(target, ad_row)
```

---

#### Method: `find_best_ad()`

Find the best matching ad for given target values.

```python
find_best_ad(target_values: Dict[str, str]) -> Optional[Dict]
```

**Parameters:**
- `target_values` (Dict): Target criteria with keys:
  - `target_age_group`
  - `target_gender`
  - `target_mood`
  - `target_weather`

**Returns:**
- `Dict`: Best matching ad:
  ```python
  {
      'pid': 'PROD123',
      'ad_title': 'Women\'s Leggings',
      'target_age_group': '18-39',
      'target_gender': 'Female',
      'target_mood': 'neutral',
      'target_weather': 'sunny',
      'match_score': 4,  # 0-4
      'max_possible_score': 4
  }
  ```
- `None`: If error occurs or no ads found

**Example:**
```python
target = {
    'target_age_group': '18-39',
    'target_gender': 'Female',
    'target_mood': 'happy',
    'target_weather': 'sunny'
}

best_ad = engine.find_best_ad(target)
if best_ad:
    print(f"Best Ad: {best_ad['ad_title']}")
    print(f"Match Score: {best_ad['match_score']}/4")
```

---

#### Method: `find_top_n_ads()`

Find top N matching ads for given target values.

```python
find_top_n_ads(target_values: Dict[str, str], n: int = 5) -> List[Dict]
```

**Parameters:**
- `target_values` (Dict): Target criteria
- `n` (int): Number of top ads to return. Default: 5

**Returns:**
- `List[Dict]`: List of ad dictionaries (same format as `find_best_ad()`)

**Example:**
```python
target = {
    'target_age_group': '18-39',
    'target_gender': 'Female',
    'target_mood': 'neutral',
    'target_weather': 'sunny'
}

top_ads = engine.find_top_n_ads(target, n=3)
for i, ad in enumerate(top_ads, 1):
    print(f"{i}. {ad['ad_title']} (Score: {ad['match_score']}/4)")
```

---

#### Method: `get_database_stats()`

Get statistics about the ads database.

```python
get_database_stats() -> Optional[Dict]
```

**Returns:**
- `Dict`: Database statistics:
  ```python
  {
      'total_ads': 20000,
      'unique_ads': 15000,
      'age_groups': {
          '18-39': 12000,
          '40-64': 5000,
          'Kids': 2000,
          ...
      },
      'genders': {
          'Male': 11000,
          'Female': 9000
      },
      'moods': {
          'neutral': 15000,
          'happy': 3000,
          ...
      },
      'weather': {
          'sunny': 18000,
          'rainy': 1500,
          'cold': 500
      }
  }
  ```
- `None`: If database not loaded

**Example:**
```python
stats = engine.get_database_stats()
if stats:
    print(f"Total Ads: {stats['total_ads']}")
    print(f"Age Groups: {stats['age_groups']}")
```

---

## Main Application

Location: `main.py`

### Class: `AdDisplaySystem`

Main system for displaying billboard ads based on target values.

#### Constructor

```python
AdDisplaySystem()
```

**Example:**
```python
from main import AdDisplaySystem

system = AdDisplaySystem()
```

---

#### Method: `initialize()`

Initialize all components of the system.

```python
initialize() -> bool
```

**Returns:**
- `bool`: `True` if successful, `False` otherwise

**Initialization Steps:**
1. Load ads database
2. Check weather service configuration
3. Display system status

**Example:**
```python
if system.initialize():
    print("System ready")
else:
    print("Initialization failed")
```

---

#### Method: `display_ad()`

Display an ad in the terminal with formatting.

```python
display_ad(ad: dict, target_values: dict)
```

**Parameters:**
- `ad` (dict): Ad information from recommendation engine
- `target_values` (dict): Target values that triggered this ad

**Side Effects:**
- Prints formatted ad to terminal
- Displays countdown timer (3 seconds default)
- Clears countdown after completion

**Example:**
```python
ad = {
    'pid': 'PROD123',
    'ad_title': 'Women\'s Leggings',
    'match_score': 4,
    ...
}

target = {
    'target_age_group': '18-39',
    ...
}

system.display_ad(ad, target)
# Displays ad for 3 seconds
```

---

#### Method: `process_target_csv()`

Process target values from CSV file row by row.

```python
process_target_csv(csv_path: str)
```

**Parameters:**
- `csv_path` (str): Path to CSV file with target values

**CSV Format:**
```csv
pid,ad_title,target_age_group,target_gender,target_mood,target_weather
TARGET001,Request 1,18-39,Female,happy,
TARGET002,Request 2,40-64,Male,neutral,rainy
```

**Behavior:**
1. Reads CSV file
2. Fetches current weather (if weather column empty)
3. Processes each row sequentially
4. Displays ad for 3 seconds
5. Continues to next row

**Example:**
```python
system.process_target_csv("data/targets.csv")
# Processes all rows and displays ads
```

---

#### Method: `predict_ad_categories()`

Predict categories for a new ad title using ML model.

```python
predict_ad_categories(ad_title: str) -> Optional[dict]
```

**Parameters:**
- `ad_title` (str): The ad title to classify

**Returns:**
- `dict`: Predicted categories (same format as `AdClassifier.predict()`)
- `None`: If error occurs

**Example:**
```python
prediction = system.predict_ad_categories("Women's Fashion Leggings")
if prediction:
    print(f"Age Group: {prediction['target_age_group']}")
    print(f"Gender: {prediction['target_gender']}")
```

---

#### Method: `interactive_mode()`

Run system in interactive mode for testing.

```python
interactive_mode()
```

**Menu Options:**
1. Display ads from target CSV file
2. Predict categories for a new ad title
3. Test weather service
4. View database statistics
5. Exit

**Example:**
```python
system.interactive_mode()
# Displays menu and waits for user input
```

---

## Command Line Interface

### Basic Usage

```bash
# Interactive mode
python main.py

# Process CSV file
python main.py path/to/target_values.csv

# Predict categories
python main.py --predict "Ad Title Here"

# Help
python main.py --help
```

### Exit Codes

- `0`: Success
- `1`: Initialization failed
- `2`: Invalid arguments

---

## Data Types

### Target Values Dictionary

```python
{
    'target_age_group': str,  # "Kids", "10-18", "18-39", "40-64", "65+"
    'target_gender': str,     # "Male", "Female"
    'target_mood': str,       # "Happy", "Angry", "Sad", "Neutral"
    'target_weather': str     # "sunny", "rainy", "cold"
}
```

### Ad Dictionary

```python
{
    'pid': str,                      # Product/Ad ID
    'ad_title': str,                 # Ad title text
    'target_age_group': str,         # Age group
    'target_gender': str,            # Gender
    'target_mood': str,              # Mood
    'target_weather': str,           # Weather
    'match_score': int,              # 0-4
    'max_possible_score': int        # Always 4
}
```

---

## Error Codes

### Classifier Errors
- `"Model not loaded"`: Call `load_model()` first
- `"Missing model files"`: Train model in Google Colab
- `"Empty ad title"`: Provide non-empty string

### Weather Service Errors
- `"API key not configured"`: Add key to `.env`
- `"Error fetching weather"`: Check internet connection
- `"Error parsing weather data"`: API response format changed

### Recommendation Engine Errors
- `"Ads database not loaded"`: Call `load_ads_database()` first
- `"File not found"`: Check CSV path
- `"Missing required columns"`: Verify CSV format

---


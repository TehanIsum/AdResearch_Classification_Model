from src.classifier import AdClassifier

# Initialize and load model
classifier = AdClassifier()
print("Loading classification model...\n")
classifier.load_model()

print("\n" + "="*70)
print("AD CLASSIFICATION MODEL")
print("="*70)
print("\nEnter ad title to predict the target categories")

while True:
    # Get user input
    ad_title = input("Enter ad title: ").strip()
    
    # Check if user wants to exit
    if ad_title.lower() in ['exit', 'quit', 'q']:
        break
    
    # Skip empty input
    if not ad_title:
        print("Please enter a valid ad title.\n")
        continue
    
    # Predict categories
    print(f"\nPredicting categories for: {ad_title}")
    result = classifier.predict(ad_title)
    
    if result:
        # Remove internal fields
        result.pop('_confidence', None)
        result.pop('_vocabulary_match', None)
        
        print("\nPrediction Results:")
        print(f"  Age Group: {result['target_age_group']}")
        print(f"  Gender: {result['target_gender']}")
        print(f"  Mood: {result['target_mood']}")
        print(f"  Weather: {result['target_weather']}")
    else:
        print("Error: Could not predict categories")
    
    print("\n" + "-"*70 + "\n")

"""
Billboard Ad Display System - Streamlit UI
Professional interface for shopping mall billboard displays
"""

import streamlit as st
import pandas as pd
import time
import os
import sys
from pathlib import Path
from PIL import Image
import random
import base64
from io import BytesIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.classifier import AdClassifier
from src.weather_service import WeatherService
from src.recommendation_engine import AdRecommendationEngine


# Page configuration
st.set_page_config(
    page_title="Smart Billboard Display",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Disable Streamlit's default transitions/animations
st.markdown("""
<style>
    /* Remove all transitions and animations */
    * {
        transition: none !important;
        animation: none !important;
        opacity: 1 !important;
    }
    
    /* Remove Streamlit's default fade-in */
    .main .block-container {
        animation: none !important;
        opacity: 1 !important;
    }
    
    /* Instant rerun without fade */
    section[data-testid="stSidebar"],
    .main {
        transition: none !important;
        opacity: 1 !important;
    }
    
    /* Ensure all content is fully opaque */
    .element-container,
    .stMarkdown,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"],
    div[data-testid="column"] {
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for billboard-style display
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
        padding-top: 0 !important;
    }
    
    /* Remove top padding from block container */
    .block-container {
        padding-top: 5rem !important;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Top button bar */
    .top-buttons {
        display: flex;
        justify-content: center;
        gap: 15px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Exit button */
    .exit-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: #e74c3c;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Billboard frame */
    .billboard-frame {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin: 20px auto;
        max-width: 1400px;
    }
    
    /* Ad display area */
    .ad-container {
        background: #f8f9fa;
        border: 3px solid #ddd;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        min-height: 400px;
    }
    
    /* Ad title styling */
    .ad-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 20px 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Match score */
    .match-score {
        font-size: 1.8em;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
        margin: 15px 0;
    }
    
    /* Audience info */
    .audience-info {
        background: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 32px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Image frame */
    .image-frame {
        border: 5px solid #34495e;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        background: #2c3e50;
        padding: 10px;
    }
    
    /* Status indicators */
    .status-success {
        background: #27ae60;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        display: inline-block;
        margin: 10px 0;
    }
    
    .status-warning {
        background: #f39c12;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        display: inline-block;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


class BillboardUI:
    def __init__(self):
        self.ad_images_dir = "ad_images"  # Directory for ad images
        
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.current_ad_index = 0
            st.session_state.auto_play = True  # Auto-play enabled by default
            st.session_state.current_mode = "display"  # default mode
            st.session_state.last_valid_ad = None  # Store last valid ad
            st.session_state.last_target_values = None  # Store last target values
            # Create instances once and store in session state
            st.session_state.classifier = AdClassifier()
            st.session_state.weather_service = WeatherService()
            st.session_state.recommendation_engine = AdRecommendationEngine()
        
        # Use instances from session state
        self.classifier = st.session_state.classifier
        self.weather_service = st.session_state.weather_service
        self.recommendation_engine = st.session_state.recommendation_engine
            
    def initialize_system(self):
        """Initialize all system components"""
        if not st.session_state.initialized:
            with st.spinner("Initializing Billboard System..."):
                # Load classifier
                try:
                    self.classifier.load_model()
                    st.session_state.classifier_loaded = True
                except:
                    st.session_state.classifier_loaded = False
                
                # Load ads database
                if self.recommendation_engine.load_ads_database():
                    st.session_state.ads_loaded = True
                else:
                    st.session_state.ads_loaded = False
                
                # Check weather service
                st.session_state.weather_available = self.weather_service.check_api_key()
                
                st.session_state.initialized = True
                
        return st.session_state.initialized
    
    def get_ad_image_path(self, pid: str) -> str:
        """Get the image path for an ad by PID"""
        # Check for various image formats
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            img_path = os.path.join(self.ad_images_dir, f"{pid}{ext}")
            if os.path.exists(img_path):
                return img_path
        return None
    
    def display_ad_with_image(self, ad: dict, target_values: dict = None):
        """Display ad with image in billboard style"""
        
        # Get image path first
        img_path = self.get_ad_image_path(ad['pid'])
        
        # Prepare image HTML
        if img_path and os.path.exists(img_path):
            try:
                # Convert image to base64 to embed in HTML
                img = Image.open(img_path)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Complete billboard in single HTML block - no gaps!
                st.markdown(f"""
                <div style='max-width: 95%; 
                            margin: 0 auto 15px;
                            border: 5px solid #333; 
                            border-radius: 15px; 
                            overflow: hidden; 
                            background: white; 
                            padding: 0;
                            line-height: 0;'>
                    <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                                padding: 8px; 
                                margin: 0;
                                line-height: 0;
                                display: block;'>
                        <marquee behavior="scroll" direction="left" scrollamount="8" style='color: white; font-size: 1.1em; font-weight: bold; line-height: 1.2;'>
                            WELCOME TO OUR SHOPPING MALL - MEGA SALE NOW ON! UP TO 70% OFF ON SELECTED ITEMS - SHOP NOW AND SAVE BIG!
                        </marquee>
                    </div>
                    <div style='width: 100%; max-height: 600px; padding: 0; line-height: 0; display: block; overflow: hidden;'>
                        <img src="data:image/png;base64,{img_str}" style="width: 100%; height: auto; max-height: 600px; object-fit: contain; display: block; margin: 0; padding: 0;">
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                # Image exists but couldn't load
                st.markdown("""
                <div style='max-width: 95%; 
                            margin: 0 auto 15px;
                            border: 5px solid #333; 
                            border-radius: 15px; 
                            overflow: hidden; 
                            background: white; 
                            padding: 0;
                            line-height: 0;'>
                    <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                                padding: 8px; 
                                margin: 0;
                                line-height: 0;
                                display: block;'>
                        <marquee behavior="scroll" direction="left" scrollamount="8" style='color: white; font-size: 1.1em; font-weight: bold; line-height: 1.2;'>
                            WELCOME TO OUR SHOPPING MALL - MEGA SALE NOW ON! UP TO 70% OFF ON SELECTED ITEMS - SHOP NOW AND SAVE BIG!
                        </marquee>
                    </div>
                    <div style='width: 100%;
                                min-height: 250px;
                                display: flex; 
                                align-items: center; 
                                justify-content: center;
                                background: #f8f9fa;
                                padding: 0;'>
                        <p style='color: #999; font-size: 1.2em; font-weight: bold; margin: 0;'>Ad Image Not Found</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # No image found
            st.markdown("""
            <div style='max-width: 95%; 
                        margin: 0 auto 15px;
                        border: 5px solid #333; 
                        border-radius: 15px; 
                        overflow: hidden; 
                        background: white; 
                        padding: 0;
                        line-height: 0;'>
                <div style='background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); 
                            padding: 8px; 
                            margin: 0;
                            line-height: 0;
                            display: block;'>
                    <marquee behavior="scroll" direction="left" scrollamount="8" style='color: white; font-size: 1.1em; font-weight: bold; line-height: 1.2;'>
                        WELCOME TO OUR SHOPPING MALL - MEGA SALE NOW ON! UP TO 70% OFF ON SELECTED ITEMS - SHOP NOW AND SAVE BIG!
                    </marquee>
                </div>
                <div style='width: 100%;
                            min-height: 250px;
                            display: flex; 
                            align-items: center; 
                            justify-content: center;
                            background: #f8f9fa;
                            padding: 0;'>
                    <p style='color: #999; font-size: 1.2em; font-weight: bold; margin: 0;'>Ad Image Not Found</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Ad Title below frame
        st.markdown(f'<div style="text-align: center; padding: 10px; font-size: 1.8em; font-weight: bold; color: #2c3e50; margin: 0; opacity: 1 !important;">{ad["ad_title"]}</div>', unsafe_allow_html=True)
        
        # Details section below image
        st.markdown('<hr style="opacity: 1 !important; border: 1px solid #ccc;">', unsafe_allow_html=True)
        
        # Wrap columns in a container with full opacity
        st.markdown('<div style="opacity: 1 !important;">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### AD DETAILS")
            st.markdown(f"**Product ID:** `{ad['pid']}`")
            if 'match_score' in ad and 'max_possible_score' in ad:
                score_percentage = (ad['match_score'] / ad['max_possible_score']) * 100
                st.markdown(f"**Match Score:** {ad['match_score']}/{ad['max_possible_score']} ({score_percentage:.0f}%)")
        
        with col2:
            st.markdown("### TARGET AUDIENCE")
            st.write(f"**Age:** {ad.get('target_age_group', 'N/A')}")
            st.write(f"**Gender:** {ad.get('target_gender', 'N/A')}")
            st.write(f"**Mood:** {ad.get('target_mood', 'N/A')}")
            st.write(f"**Weather:** {ad.get('target_weather', 'N/A')}")
            if 'ad_type' in ad and ad['ad_type']:
                st.write(f"**Type:** {ad['ad_type']}")
        
        with col3:
            if target_values:
                st.markdown("### REQUESTED CRITERIA")
                st.write(f"**Age:** {target_values.get('target_age_group', 'N/A')}")
                st.write(f"**Gender:** {target_values.get('target_gender', 'N/A')}")
                st.write(f"**Mood:** {target_values.get('target_mood', 'N/A')}")
                st.write(f"**Weather:** {target_values.get('target_weather', 'N/A')}")
                if 'target_ad_type' in target_values and target_values.get('target_ad_type'):
                    st.write(f"**Type:** {target_values['target_ad_type']}")
        
        # Close opacity container
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_placeholder_image(self):
        """Display placeholder when no image available"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    height: 400px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    border-radius: 15px;'>
            <div style='text-align: center; color: white;'>
                <h2>Image Not Available</h2>
                <p>Please add image to ad_images/ folder</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def show_ad_display_mode(self):
        """Ad Display Mode - Show ads from CSV with automatic switching"""
        
        # Use default file automatically
        default_path = "data/example_target_values.csv"
        
        if not os.path.exists(default_path):
            st.error(f"Default file not found: {default_path}")
            return
        
        try:
            # Read CSV
            df = pd.read_csv(default_path)
            
            # Get current weather
            current_weather = "sunny"
            if st.session_state.weather_available:
                weather_info = self.weather_service.get_detailed_weather_info()
                if weather_info:
                    current_weather = weather_info['category']
            
            # Get current row
            idx = st.session_state.current_ad_index
            if idx >= len(df):
                idx = 0
                st.session_state.current_ad_index = 0
            
            row = df.iloc[idx]
            
            # Display current row info
            st.info(f"Target Row {idx + 1}/{len(df)} | Age: {row.get('target_age_group', 'N/A')} | Gender: {row.get('target_gender', 'N/A')} | Mood: {row.get('target_mood', 'N/A')} | Ad Type: {row.get('ad_type', 'N/A')}")
            
            # Build target values exactly like main.py does
            target_values = {
                'target_age_group': str(row.get('target_age_group', '18-39')).strip(),
                'target_gender': str(row.get('target_gender', 'Male')).strip(),
                'target_mood': str(row.get('target_mood', 'neutral')).strip(),
                'target_weather': current_weather
            }
            
            # Add ad_type if present
            if 'ad_type' in row.index and pd.notna(row['ad_type']) and str(row['ad_type']).strip():
                target_values['target_ad_type'] = str(row['ad_type']).strip()
            
            # Check if ads database is loaded
            if not st.session_state.ads_loaded:
                st.error("Ads database not loaded properly. Please restart the application.")
                return
            
            # Verify recommendation engine is loaded
            if not self.recommendation_engine.is_loaded:
                st.error("Recommendation engine lost its data. Reloading...")
                if self.recommendation_engine.load_ads_database():
                    st.session_state.ads_loaded = True
                else:
                    st.error("Failed to reload ads database.")
                    return
            
            # Find matching ad
            best_ad = self.recommendation_engine.find_best_ad(target_values, debug=False)
            
            if best_ad:
                # Store this ad as the last valid ad
                st.session_state.last_valid_ad = best_ad
                st.session_state.last_target_values = target_values
                # Display the matching ad
                self.display_ad_with_image(best_ad, target_values)
            else:
                # No matching ad found - keep showing previous ad if it exists
                if st.session_state.last_valid_ad:
                    st.warning("No Matching Ad Found - Showing Previous Ad")
                    self.display_ad_with_image(st.session_state.last_valid_ad, st.session_state.last_target_values)
                else:
                    # No previous ad to show
                    st.warning("No Matching Ad Found")
                    st.markdown("### REQUESTED CRITERIA")
                    st.markdown('<div class="audience-info">', unsafe_allow_html=True)
                    st.write(f"**Age Group:** {target_values.get('target_age_group', 'N/A')}")
                    st.write(f"**Gender:** {target_values.get('target_gender', 'N/A')}")
                    st.write(f"**Mood:** {target_values.get('target_mood', 'N/A')}")
                    st.write(f"**Weather:** {target_values.get('target_weather', 'N/A')}")
                    if 'target_ad_type' in target_values:
                        st.write(f"**Ad Type:** {target_values['target_ad_type']}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Auto-refresh immediately for instant transitions
            time.sleep(0.5)
            
            # Update index for next iteration and loop back to start
            st.session_state.current_ad_index = (idx + 1) % len(df)
            st.rerun()
                
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            time.sleep(5)
            st.rerun()
    
    def show_prediction_mode(self):
        """Prediction Mode - Classify new ad titles"""
        st.markdown("## PREDICT CATEGORIES")
        st.markdown("---")
        
        if not st.session_state.classifier_loaded:
            st.error("Classifier not loaded. Please check model files in 'models/' directory.")
            return
        
        # Input
        ad_title = st.text_input(
            "Enter Ad Title",
            placeholder="e.g., Men's leather jacket for winter",
            key="predict_input"
        )
        
        predict_btn = st.button("Predict Categories", type="primary", use_container_width=True)
        
        if predict_btn and ad_title:
            with st.spinner("Analyzing ad title..."):
                prediction = self.classifier.predict(ad_title)
            
            if prediction:
                # Remove internal metadata
                prediction.pop('_confidence', None)
                prediction.pop('_vocabulary_match', None)
                
                st.success(f"Prediction completed for: {ad_title}")
                st.info("Predicted categories have been processed successfully!")
    
    def show_weather_mode(self):
        """Weather Testing Mode"""
        st.markdown("## WEATHER SERVICE TEST")
        st.markdown("---")
        
        if not st.session_state.weather_available:
            st.warning("Weather API key not configured. Using default 'sunny'.")
            st.info("Add your OpenWeatherMap API key to enable weather features.")
            return
        
        # City input
        city = st.text_input(
            "Enter City Name",
            value="Colombo",
            placeholder="e.g., New York, London, Tokyo"
        )
        
        if st.button("Get Weather", type="primary", use_container_width=True):
            with st.spinner(f"Fetching weather data for {city}..."):
                weather_info = self.weather_service.get_detailed_weather_info(city)
            
            if weather_info:
                st.markdown("---")
                
                # Weather display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    st.metric("Location", f"{weather_info['city']}, {weather_info['country']}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    st.metric("Temperature", f"{weather_info['temperature']}°C")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="info-card">', unsafe_allow_html=True)
                    st.metric("Category", weather_info['category'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.info(f"Current Condition: {weather_info['description']}")
            else:
                st.error(f"Could not fetch weather data for {city}")
    
    def run(self):
        """Main application"""
        
        # Header - Small and at top
        st.markdown("""
        <div style='text-align: center; padding: 5px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 5px; margin-bottom: 5px;'>
            <h3 style='color: white; margin: 0; font-size: 1em;'>SMART BILLBOARD DISPLAY</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize system
        self.initialize_system()
        
        # Top navigation buttons
        col1, col2, col3, spacer = st.columns([1, 1, 1, 3])
        
        with col1:
            if st.button("Display Ads", use_container_width=True, type="primary" if st.session_state.current_mode == "display" else "secondary"):
                st.session_state.current_mode = "display"
                st.rerun()
        
        with col2:
            if st.button("Predict", use_container_width=True, type="primary" if st.session_state.current_mode == "predict" else "secondary"):
                st.session_state.current_mode = "predict"
                st.rerun()
        
        with col3:
            if st.button("Weather", use_container_width=True, type="primary" if st.session_state.current_mode == "weather" else "secondary"):
                st.session_state.current_mode = "weather"
                st.rerun()
        
        # Exit button (bottom right)
        st.markdown("""
        <style>
        .exit-btn-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main content based on mode
        if st.session_state.current_mode == "display":
            self.show_ad_display_mode()
        elif st.session_state.current_mode == "predict":
            self.show_prediction_mode()
        elif st.session_state.current_mode == "weather":
            self.show_weather_mode()
        
        # Exit button in bottom right corner
        col_exit1, col_exit2 = st.columns([6, 1])
        with col_exit2:
            if st.button("EXIT", type="secondary", use_container_width=True):
                st.success("Goodbye!")
                time.sleep(1)
                st.stop()


def main():
    app = BillboardUI()
    app.run()


if __name__ == "__main__":
    main()

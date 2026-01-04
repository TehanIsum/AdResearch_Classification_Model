#!/bin/bash

# Quick Start Script for Streamlit Billboard UI

echo "======================================================================"
echo "🎯 BILLBOARD DISPLAY SYSTEM - QUICK START"
echo "======================================================================"
echo ""

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Streamlit not installed"
    echo "   Installing required packages..."
    pip install streamlit pillow
else
    echo "✅ Streamlit installed"
fi

python3 -c "import PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Pillow not installed"
    echo "   Installing Pillow..."
    pip install pillow
else
    echo "✅ Pillow installed"
fi

echo ""
echo "======================================================================"
echo "📁 Checking project structure..."
echo "======================================================================"
echo ""

# Check for required directories
if [ -d "ad_images" ]; then
    echo "✅ ad_images/ directory exists"
    image_count=$(find ad_images -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" -o -name "*.webp" \) 2>/dev/null | wc -l | xargs)
    echo "   Found $image_count images"
    
    if [ "$image_count" -eq 0 ]; then
        echo "   ⚠️  No images found. Please add images to ad_images/"
        echo "   💡 See ad_images/README.md for instructions"
    fi
else
    echo "❌ ad_images/ directory not found"
    echo "   Creating directory..."
    mkdir -p ad_images
fi

# Check for model files
if [ -d "models" ]; then
    echo "✅ models/ directory exists"
    if [ -f "models/ad_classifier_model.pkl" ]; then
        echo "   ✅ Model files found"
    else
        echo "   ⚠️  Model files not found"
        echo "   💡 Train model using train_model_colab.py"
    fi
else
    echo "⚠️  models/ directory not found"
fi

# Check for database
if [ -f "Classification model dataset.csv" ]; then
    echo "✅ Ads database found"
else
    echo "❌ Classification model dataset.csv not found"
fi

echo ""
echo "======================================================================"
echo "🚀 STARTING STREAMLIT APP"
echo "======================================================================"
echo ""
echo "Opening in your browser at: http://localhost:8501"
echo ""
echo "Controls:"
echo "  • Use sidebar to navigate between modes"
echo "  • Press Ctrl+C to stop the server"
echo ""
echo "======================================================================"
echo ""

# Run streamlit
streamlit run app.py

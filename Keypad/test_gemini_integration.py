#!/usr/bin/env python3
"""
Test script for Gemini integration
This script tests the Gemini API integration without running the full Flask app.
"""

import os
import sys
import json
from PIL import Image
import numpy as np

# Add current directory to path
sys.path.append('.')

def test_gemini_imports():
    """Test if all required imports work"""
    try:
        import google.generativeai as genai
        from PIL import Image, ImageDraw, ImageFont
        print("✓ All required imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_api_key():
    """Test if API key is configured"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("❌ No valid API key found")
        print("Please run: python setup_gemini.py")
        return False
    
    print(f"✓ API key found: {api_key[:10]}...")
    return True

def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello, this is a test of the Gemini API.")
        
        if response and response.text:
            print("✓ Gemini API connection successful")
            print(f"Response: {response.text[:100]}...")
            return True
        else:
            print("❌ No response from Gemini API")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False

def test_image_analysis():
    """Test image analysis with a sample image"""
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        import google.generativeai as genai
        api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        prompt = "Describe this image briefly."
        response = model.generate_content([prompt, test_image])
        
        if response and response.text:
            print("✓ Image analysis test successful")
            print(f"Analysis: {response.text[:100]}...")
            return True
        else:
            print("❌ Image analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Image analysis test failed: {e}")
        return False

def test_thermal_analysis_function():
    """Test the thermal analysis function from main_app.py"""
    try:
        # Import the function from main_app
        from main_app import analyze_thermal_image_with_gemini, create_annotated_thermal_image
        
        # Create a test image file
        test_image = Image.new('RGB', (200, 200), color='blue')
        test_image_path = 'test_thermal_image.jpg'
        test_image.save(test_image_path)
        
        # Test the analysis function
        result_path, analysis_data = analyze_thermal_image_with_gemini(test_image_path)
        
        if result_path and analysis_data:
            print("✓ Thermal analysis function works")
            print(f"Analysis data keys: {list(analysis_data.keys())}")
            
            # Clean up test file
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
            if os.path.exists(result_path):
                os.remove(result_path)
            
            return True
        else:
            print("❌ Thermal analysis function failed")
            return False
            
    except Exception as e:
        print(f"❌ Thermal analysis function test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Gemini Integration Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Import Test", test_gemini_imports),
        ("API Key Test", test_api_key),
        ("Connection Test", test_gemini_connection),
        ("Image Analysis Test", test_image_analysis),
        ("Thermal Analysis Function Test", test_thermal_analysis_function),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The Gemini integration is working correctly.")
        print("You can now run the main application with: python main_app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("Make sure to run: python setup_gemini.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

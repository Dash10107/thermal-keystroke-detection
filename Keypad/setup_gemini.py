#!/usr/bin/env python3
"""
Setup script for Gemini API integration
This script helps configure the Gemini API key for the thermal imaging application.
"""

import os
import sys

def setup_gemini_api():
    """
    Interactive setup for Gemini API key
    """
    print("=" * 60)
    print("Gemini API Setup for Thermal Imaging Application")
    print("=" * 60)
    print()
    print("To use the Gemini AI thermal analysis features, you need to:")
    print("1. Get a Gemini API key from: https://ai.google.dev/")
    print("2. Set it as an environment variable or enter it below")
    print()
    
    # Check if API key is already set
    current_key = os.getenv('GEMINI_API_KEY')
    if current_key and current_key != 'your-api-key-here':
        print(f"✓ Gemini API key is already configured: {current_key[:10]}...")
        choice = input("Do you want to update it? (y/n): ").lower().strip()
        if choice != 'y':
            print("Setup complete!")
            return True
    
    print("Enter your Gemini API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Setup cancelled.")
        return False
    
    # Validate API key format (basic check)
    if len(api_key) < 20:
        print("❌ API key seems too short. Please check your key.")
        return False
    
    # Set environment variable for current session
    os.environ['GEMINI_API_KEY'] = api_key
    
    # Create .env file for future use
    env_content = f"GEMINI_API_KEY={api_key}\n"
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✓ API key saved to .env file")
    except Exception as e:
        print(f"⚠️  Could not save to .env file: {e}")
        print("You'll need to set the environment variable manually:")
        print(f"export GEMINI_API_KEY={api_key}")
    
    print()
    print("✓ Gemini API setup complete!")
    print("You can now run the application with thermal analysis features.")
    return True

def test_gemini_connection():
    """
    Test the Gemini API connection
    """
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'your-api-key-here':
            print("❌ No valid API key found")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test with a simple request
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello, this is a test.")
        
        if response and response.text:
            print("✓ Gemini API connection successful!")
            return True
        else:
            print("❌ Gemini API connection failed - no response")
            return False
            
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("Run: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Setting up Gemini API for Thermal Imaging Application...")
    print()
    
    if setup_gemini_api():
        print()
        print("Testing API connection...")
        if test_gemini_connection():
            print()
            print("🎉 Setup complete! You can now run the application.")
            print("Start the app with: python main_app.py")
        else:
            print()
            print("⚠️  Setup completed but API test failed.")
            print("Please check your API key and try again.")
    else:
        print()
        print("❌ Setup failed. Please try again.")
        sys.exit(1)

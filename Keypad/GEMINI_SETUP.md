# Gemini AI Integration Setup

This application now includes Gemini 2.5 Flash API integration for advanced thermal image analysis. The AI can detect heat patterns, identify pressed keys, and analyze UI elements in thermal images.

## Features

- **Heat Pattern Detection**: Identifies areas with high, medium, and low heat intensity
- **Pressed Key Detection**: Detects recently pressed keys on devices
- **UI Element Recognition**: Identifies visible UI elements due to heat signatures
- **Security Analysis**: Provides security recommendations for thermal imaging attacks
- **Visual Annotations**: Overlays analysis results directly on the thermal image

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

### 3. Configure API Key

#### Option A: Using the Setup Script (Recommended)

```bash
python setup_gemini.py
```

This interactive script will:
- Guide you through the setup process
- Test your API key
- Create a `.env` file for automatic loading

#### Option B: Manual Configuration

Create a `.env` file in the Keypad directory:

```bash
echo "GEMINI_API_KEY=your-actual-api-key-here" > .env
```

#### Option C: Environment Variable

Set the environment variable directly:

**Windows:**
```cmd
set GEMINI_API_KEY=your-actual-api-key-here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY=your-actual-api-key-here
```

### 4. Run the Application

```bash
python main_app.py
```

## Usage

1. Upload a thermal image through the web interface
2. The application will now show four analysis results:
   - Original image
   - Traditional UI detection
   - UI obfuscation
   - **NEW**: Gemini AI thermal analysis with annotations

3. The Gemini analysis includes:
   - Heat intensity mapping (Red=High, Orange=Medium, Yellow=Low)
   - Pressed key detection (Green boxes)
   - UI element identification (Blue boxes)
   - Detailed analysis results in text format

## Analysis Legend

- **Red boxes**: High heat intensity areas
- **Orange boxes**: Medium heat intensity areas  
- **Yellow boxes**: Low heat intensity areas
- **Green boxes**: Detected pressed keys
- **Blue boxes**: Identified UI elements

## Troubleshooting

### API Key Issues
- Ensure your API key is valid and active
- Check that the key has proper permissions
- Verify the key is correctly set in the environment

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.7+)

### Analysis Failures
- The Gemini API may occasionally fail due to rate limits or service issues
- Check your internet connection
- Verify the image format is supported (JPG, PNG)

## API Costs

- Gemini 2.5 Flash has usage-based pricing
- Check current pricing at [Google AI Pricing](https://ai.google.dev/pricing)
- Monitor your usage in the Google AI Studio dashboard

## Security Notes

- Never commit your API key to version control
- Use environment variables or `.env` files for configuration
- The `.env` file is already in `.gitignore` to prevent accidental commits

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify your API key is working with the setup script
3. Ensure all dependencies are properly installed
4. Check the Gemini API status page for service issues

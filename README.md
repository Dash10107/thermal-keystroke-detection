# Thermal Imaging Project

This project presents a deep learning framework to defend against thermal imaging attacks—a major privacy risk where attackers use residual heat on surfaces (like keyboards or screens) to deduce sensitive information.

## Motivation
Thermal cameras can reveal recently pressed keys or accessed interfaces by detecting heat signatures left behind. Such side-channel attacks can compromise passwords, PINs, or sensitive usage patterns. Our work demonstrates real-time detection and obfuscation to protect user privacy in these scenarios.

## Solution Overview
We leverage state-of-the-art deep learning models to analyze thermal images or video:
- **Faster R-CNN and YOLO**: Used for visual object detection, key area highlighting, and obfuscation.
- **Real-Time Demo**: The framework includes live processing for both user interface detection and immediate obfuscation of sensitive regions.
- **Gemini AI (optional)**: For advanced analysis, the Gemini API can help segment and classify thermal hotspots, even suggesting which keyboard keys were most recently pressed.

## Project Structure
- **/Faster R-CNN**: Implementation for UI/key detection and obfuscation using the Faster R-CNN model.
- **/YOLO**: Implementation for UI/key detection using YOLOv5/YOLOv8 models.
- **/Keypad**: A Flask web demo for uploading thermal images or live video, showing real-time detection, AI-based analysis, and privacy protection visualizations.
- **/datasets**: Sample datasets and COCO-style annotations used for benchmarking and training.

## Try It Yourself
1. Clone the repo and see the respective `/Faster R-CNN` and `/YOLO` folders for model-specific instructions.
2. Launch the `/Keypad` Flask app to try image/video upload and real-time privacy demos.
3. The app will display your thermal image, visually highlight detected/obfuscated regions, and compress all text for minimal user distraction.


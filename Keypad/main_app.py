# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from flask import Flask, render_template, request, session, Response
from camera import VideoCamera
import pandas as pd
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
import csv
import base64
import json
import pickle
from werkzeug.utils import secure_filename
# import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont
import io
import requests

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass
 
#*** Backend operation
 
# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
 
# Accepted image for to upload for object detection model
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'You Will Never Guess'

# Configure Gemini API
# You'll need to set your API key as an environment variable: GEMINI_API_KEY
GEMINI_API_KEY = "value"
GEMINI_MODEL = 'gemini-2.0-flash-exp'
GEMINI_ENDPOINT = f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
# Hotspot percentile (higher = stricter); can override via env HOT_PERCENTILE
HOT_PERCENTILE = float(os.getenv('HOT_PERCENTILE', '85'))  # Lowered from 99 to 85 for more key detection



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
#IMAGE_NAME = 'C:/tensorflow1/models/research/object_detection/images/testt/201.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Path to image
#PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 11

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#image = cv2.imread(uploaded_image_path)

def detect_object(uploaded_image_path):
    # Loading image
    image = cv2.imread(uploaded_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=0.5,
        skip_scores=True,
        skip_labels=True)

    # All the results have been drawn on image. Now display the image.
    cv2.imshow('Object detector', image)
    # Write output image (object detection output)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image_detection.jpg')
    cv2.imwrite(output_image_path, image)
    return(output_image_path)

def obscure_object(uploaded_image_path):
    # Loading image
    image = cv2.imread(uploaded_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=-1,
        min_score_thresh=1,
        skip_scores=True,
        skip_labels=False)

    # All the results have been drawn on image. Now display the image.
    true_boxes = boxes[0][scores[0] > 0.75]
    if np.any(true_boxes):
        height, width, channels = image.shape
        ymin = true_boxes[0,0]*height
        xmin = true_boxes[0,1]*width
        ymax = true_boxes[0,2]*height
        xmax = true_boxes[0,3]*width
        B1=image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),2].shape
        #image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),0]=np.random.randint(50, size=(B1[0],B1[1]))
        #image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),1]=np.random.randint(50, size=(B1[0],B1[1]))
        image[ymin.astype(int):ymax.astype(int),xmin.astype(int):xmax.astype(int),2]=np.random.randint(10, size=(B1[0],B1[1]))
    cv2.imshow('User Interface detector', image)
    # Write output image (object detection output)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image_obscure.jpg')
    cv2.imwrite(output_image_path, image)
    return(output_image_path)

def analyze_thermal_image_with_gemini(uploaded_image_path):
    """
    Analyze thermal image using Gemini 2.5 Flash API to detect heat patterns and pressed keys
    """
    try:
        if not GEMINI_API_KEY:
            return None, "Error: GEMINI_API_KEY not set"
        
        # Load the image
        image = cv2.imread(uploaded_image_path)
        if image is None:
            return None, "Error: Could not load image"
        
        # Read file bytes and base64-encode for inlineData
        with open(uploaded_image_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Compute brightest ROI to guide Gemini to look only where the keys are
        try:
            bright_roi = _compute_brightest_regions_cv(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), top_k=1, percentile=HOT_PERCENTILE)[0]['coordinates']
        except Exception:
            bright_roi = None

        # Create prompt prioritizing brightness-based key detection within ROI
        roi_text = ""
        if bright_roi:
            roi_text = f"Focus ONLY inside this ROI (pixels) when detecting keys: x1={bright_roi['x1']}, y1={bright_roi['y1']}, x2={bright_roi['x2']}, y2={bright_roi['y2']}. "

        prompt = (
            "Task: Identify the individual keyboard keys that are BRIGHTEST (highest luminance) in this thermal/infrared-like image. "
            + roi_text +
            "If key labels are visible or can be inferred from typical keyboard layout and position, include the key label. "
            "Only return keys that are among the top-brightest spots; avoid large areas. "
            "Respond STRICTLY in JSON with this schema: "
            "{\n  \"heat_areas\": [{\"coordinates\":{\"x1\":int,\"y1\":int,\"x2\":int,\"y2\":int}, \"intensity\":\"high|medium|low\", \"description\":string}],\n"
            "  \"pressed_keys\": [{\"coordinates\":{\"x1\":int,\"y1\":int,\"x2\":int,\"y2\":int}, \"key\":string, \"confidence\":float}],\n"
            "  \"ui_elements\": [{\"coordinates\":{\"x1\":int,\"y1\":int,\"x2\":int,\"y2\":int}, \"element\":string, \"type\":string}],\n"
            "  \"security_analysis\": string,\n  \"recommendations\": string\n}"
        )
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": image_b64
                            }
                        }
                    ]
                }
            ]
        }
        params = {"key": GEMINI_API_KEY}
        headers = {"Content-Type": "application/json"}
        
        resp = requests.post(GEMINI_ENDPOINT, params=params, headers=headers, data=json.dumps(payload), timeout=30)
        if resp.status_code != 200:
            return None, f"Gemini API error: {resp.status_code} {resp.text}"
        data = resp.json()
        
        # Extract text from candidates
        analysis_text = ""
        try:
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts and "text" in parts[0]:
                    analysis_text = parts[0]["text"]
        except Exception:
            analysis_text = ""
        
        if not analysis_text:
            analysis_text = json.dumps({"security_analysis": "No textual response returned by Gemini."})
        
        # Try to parse JSON from the text
        try:
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = analysis_text[start_idx:end_idx]
                analysis_data = json.loads(json_str)
            else:
                analysis_data = {
                    "heat_areas": [],
                    "pressed_keys": [],
                    "ui_elements": [],
                    "security_analysis": analysis_text,
                    "recommendations": "Unable to parse detailed analysis"
                }
        except json.JSONDecodeError:
            analysis_data = {
                "heat_areas": [],
                "pressed_keys": [],
                "ui_elements": [],
                "security_analysis": analysis_text,
                "recommendations": "Analysis completed but formatting may be incomplete"
            }
        
        # Create annotated image (crisp mode: only hottest/brightest region + keys)
        annotated_image, filtered_key_names = create_annotated_thermal_image(
            image,
            analysis_data,
            top_k_heat_areas=1,
            crisp_mode=True,
            hotspot_percentile=HOT_PERCENTILE,
            use_brightness=True
        )
        
        # Save annotated image
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gemini_thermal_analysis.jpg')
        cv2.imwrite(output_image_path, annotated_image)
        
        # Attach concise keys list for UI display
        analysis_data['filtered_pressed_key_names'] = filtered_key_names
        return output_image_path, analysis_data
        
    except Exception as e:
        return None, f"Error in Gemini analysis: {str(e)}"

def create_annotated_thermal_image(image, analysis_data, top_k_heat_areas=1, crisp_mode=True, hotspot_percentile=99.0, use_brightness=False):
    """
    Create an annotated image based on Gemini analysis results.
    Only the hottest region(s) up to top_k_heat_areas will be shown.
    """
    # Convert to PIL for easier annotation
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Determine which heat areas to draw (prefer Gemini; fallback to CV)
    # If user requests brightness mode, skip model heat boxes and rely on brightness
    heat_areas = [] if use_brightness else (analysis_data.get('heat_areas') or [])
    selected_heat_areas = []
    if heat_areas:
        def area_size(a):
            c = a.get('coordinates', {})
            return max(0, (c.get('x2', 0) - c.get('x1', 0)) * (c.get('y2', 0) - c.get('y1', 0)))
        def intensity_rank(a):
            m = {'high': 2, 'medium': 1, 'low': 0}
            return m.get(str(a.get('intensity', 'medium')).lower(), 1)
        # Sort by intensity desc, then area desc
        heat_areas_sorted = sorted(heat_areas, key=lambda a: (intensity_rank(a), area_size(a)), reverse=True)
        selected_heat_areas = heat_areas_sorted[:max(1, int(top_k_heat_areas))]
    else:
        # Fallback: compute hottest region(s) via CV
        if use_brightness:
            selected_heat_areas = _compute_brightest_regions_cv(
                np.array(pil_image),
                top_k=max(1, int(top_k_heat_areas)),
                percentile=hotspot_percentile
            )
        else:
            selected_heat_areas = _compute_hottest_regions_cv(
                np.array(pil_image),
                top_k=max(1, int(top_k_heat_areas)),
                percentile=hotspot_percentile
            )

    # Draw only the selected hottest heat areas
    drawn_heat_boxes = []
    for area in selected_heat_areas:
        coords = area.get('coordinates', {})
        if not coords:
            continue
        x1, y1 = int(coords.get('x1', 0)), int(coords.get('y1', 0))
        x2, y2 = int(coords.get('x2', 0)), int(coords.get('y2', 0))
        intensity = str(area.get('intensity', 'high')).lower()
        color = (255, 0, 0)
        if intensity == 'medium':
            color = (255, 165, 0)
        elif intensity == 'low':
            color = (255, 255, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.text((x1, max(0, y1-22)), f"Heat: {intensity}", fill=color, font=font)
        drawn_heat_boxes.append((x1, y1, x2, y2))

    # Within the single hottest box, compute multiple sub-hotspots to outline likely keys
    sub_boxes = []
    if drawn_heat_boxes:
        hx1, hy1, hx2, hy2 = drawn_heat_boxes[0]
        if use_brightness:
            sub_boxes = _compute_sub_brightspots_cv(np.array(pil_image), (hx1, hy1, hx2, hy2), max_boxes=20, percentile=max(75.0, float(hotspot_percentile) - 5.0))
        else:
            sub_boxes = _compute_sub_hotspots_cv(np.array(pil_image), (hx1, hy1, hx2, hy2), max_boxes=20, percentile=max(75.0, float(hotspot_percentile) - 5.0))
        for sx1, sy1, sx2, sy2 in sub_boxes:
            draw.rectangle([sx1, sy1, sx2, sy2], outline=(0, 255, 255), width=2)  # cyan for estimated key regions
    
    # Draw pressed keys
    filtered_keys = []
    for key in analysis_data.get('pressed_keys', []):
        coords = key.get('coordinates', {})
        if coords:
            x1, y1 = int(coords.get('x1', 0)), int(coords.get('y1', 0))
            x2, y2 = int(coords.get('x2', 0)), int(coords.get('y2', 0))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Keep only keys whose center is inside one of the hottest boxes
            inside_hot = False
            for hx1, hy1, hx2, hy2 in drawn_heat_boxes:
                if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                    inside_hot = True
                    break
            if inside_hot:
                filtered_keys.append((x1, y1, x2, y2, key))

    for x1, y1, x2, y2, key in filtered_keys:
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        key_name = key.get('key', 'Unknown')
        draw.text((x1, max(0, y1-20)), f"Key: {key_name}", fill=(0, 255, 0), font=font)

    # Optional extra UI elements only if not in crisp mode
    if not crisp_mode:
        for element in analysis_data.get('ui_elements', []):
            coords = element.get('coordinates', {})
            if coords:
                x1, y1 = coords.get('x1', 0), coords.get('y1', 0)
                x2, y2 = coords.get('x2', 0), coords.get('y2', 0)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=2)
                element_name = element.get('element', 'UI Element')
                draw.text((x1, y1-20), f"UI: {element_name}", fill=(0, 0, 255), font=font)
    
    # Convert back to OpenCV format
    annotated_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Return image and concise keys list (unique names, in order)
    key_names = []
    seen = set()
    for _, _, _, _, key in filtered_keys:
        name = key.get('key', 'Unknown')
        if name not in seen:
            seen.add(name)
            key_names.append(name)
    # If model didn't return keys, provide numbered candidates from bright sub-spots
    if not key_names and sub_boxes:
        key_names = [f"Key-{i+1}" for i in range(len(sub_boxes))]
    return annotated_image, key_names


def _compute_hottest_regions_cv(rgb_image, top_k=1, percentile=None):
    """
    Very simple CV-based hotspot finder for thermal-style images.
    Returns list of dicts matching heat_areas structure.
    """
    # Compute a heat score favoring red/yellow bright pixels
    b, g, r = cv2.split(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    heat_score = 0.6 * r.astype(np.float32) + 0.3 * g.astype(np.float32) - 0.2 * b.astype(np.float32)
    # Threshold at configurable high percentile
    p = percentile if percentile is not None else HOT_PERCENTILE
    try:
        p = float(p)
    except Exception:
        p = 99.0
    thresh_val = np.percentile(heat_score, max(90.0, min(99.9, p)))
    mask = (heat_score >= thresh_val).astype(np.uint8) * 255
    # Morph cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find contours and sort by area
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    results = []
    for cnt in contours[:max(1, int(top_k))]:
        x, y, w, h = cv2.boundingRect(cnt)
        results.append({
            'coordinates': {'x1': int(x), 'y1': int(y), 'x2': int(x + w), 'y2': int(y + h)},
            'intensity': 'high',
            'description': 'Computed hotspot'
        })
    if not results and rgb_image.size:
        h, w, _ = rgb_image.shape
        results.append({'coordinates': {'x1': 0, 'y1': 0, 'x2': w, 'y2': h}, 'intensity': 'high', 'description': 'fallback'})
    return results


def _compute_sub_hotspots_cv(rgb_image, roi, max_boxes=8, percentile=99.3):
    """
    Find multiple smaller hotspots inside a given ROI of an RGB image.
    Returns list of (x1, y1, x2, y2) boxes in image coordinates.
    """
    hx1, hy1, hx2, hy2 = map(int, roi)
    hx1, hy1 = max(0, hx1), max(0, hy1)
    hx2, hy2 = min(rgb_image.shape[1]-1, hx2), min(rgb_image.shape[0]-1, hy2)
    if hx2 <= hx1 or hy2 <= hy1:
        return []

    sub = rgb_image[hy1:hy2, hx1:hx2]
    if sub.size == 0:
        return []

    b, g, r = cv2.split(cv2.cvtColor(sub, cv2.COLOR_RGB2BGR))
    heat_score = 0.6 * r.astype(np.float32) + 0.35 * g.astype(np.float32) - 0.15 * b.astype(np.float32)
    try:
        p = float(percentile)
    except Exception:
        p = 99.3
    p = max(90.0, min(99.9, p))
    thr = np.percentile(heat_score, p)
    mask = (heat_score >= thr).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    boxes = []
    for cnt in contours[:max(1, int(max_boxes))]:
        x, y, w, h = cv2.boundingRect(cnt)
        # Tighten boxes a bit for crispness
        shrink = 0.1
        nx1 = int(x + w * shrink)
        ny1 = int(y + h * shrink)
        nx2 = int(x + w * (1 - shrink))
        ny2 = int(y + h * (1 - shrink))
        # Map back to full-image coordinates
        boxes.append((hx1 + nx1, hy1 + ny1, hx1 + nx2, hy1 + ny2))

    return boxes


def _compute_brightest_regions_cv(rgb_image, top_k=1, percentile=None):
    """
    Brightness-first main region detector using V channel (HSV).
    Returns list of dicts like heat_areas with coordinates and intensity='high'.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    p = percentile if percentile is not None else HOT_PERCENTILE
    try:
        p = float(p)
    except Exception:
        p = 85.0
    thr = np.percentile(v, max(70.0, min(99.9, p)))  # Lowered min from 90.0 to 70.0
    mask = (v >= thr).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    results = []
    for cnt in contours[:max(1, int(top_k))]:
        x, y, w, h = cv2.boundingRect(cnt)
        results.append({
            'coordinates': {'x1': int(x), 'y1': int(y), 'x2': int(x + w), 'y2': int(y + h)},
            'intensity': 'high',
            'description': 'Brightest region'
        })
    if not results and rgb_image.size:
        h, w, _ = rgb_image.shape
        results.append({'coordinates': {'x1': 0, 'y1': 0, 'x2': w, 'y2': h}, 'intensity': 'high', 'description': 'fallback'})
    return results


def _compute_sub_brightspots_cv(rgb_image, roi, max_boxes=12, percentile=99.3):
    """
    Find multiple small bright spots (likely bright keys) inside ROI using HSV-V channel.
    Returns list of boxes.
    """
    hx1, hy1, hx2, hy2 = map(int, roi)
    hx1, hy1 = max(0, hx1), max(0, hy1)
    hx2, hy2 = min(rgb_image.shape[1]-1, hx2), min(rgb_image.shape[0]-1, hy2)
    if hx2 <= hx1 or hy2 <= hy1:
        return []
    sub = rgb_image[hy1:hy2, hx1:hx2]
    if sub.size == 0:
        return []
    v = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)[:, :, 2].astype(np.float32)
    try:
        p = float(percentile)
    except Exception:
        p = 80.0  # Lowered default from 99.3
    p = max(70.0, min(99.9, p))  # Lowered min from 90.0 to 70.0
    thr = np.percentile(v, p)
    mask = (v >= thr).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    boxes = []
    for cnt in contours[:max(1, int(max_boxes))]:
        x, y, w, h = cv2.boundingRect(cnt)
        shrink = 0.1
        nx1 = int(x + w * shrink)
        ny1 = int(y + h * shrink)
        nx2 = int(x + w * (1 - shrink))
        ny2 = int(y + h * (1 - shrink))
        boxes.append((hx1 + nx1, hy1 + ny1, hx1 + nx2, hy1 + ny2))
    return boxes


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/image_upload')
def imageHome():
    return render_template('index_upload_and_display_image.html')

@app.route('/video_upload')
def videoHome():
    return render_template('index_upload_and_display_video.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST': #This now supports upload of both image and video file
        uploaded_file = request.files['uploaded-file']
        file_filename = secure_filename(uploaded_file.filename)
        uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_filename))
 
        session['uploaded_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], file_filename)
 
        if "image" in request.form: #In case of image file, go to image page
            return render_template('index_upload_and_display_image_page2.html')
        elif "video" in request.form: #In case of video file, go to video page
            return render_template('index_upload_and_display_video_page2.html')
        else: #Go home if the form is broken - This shouldn't call but should be replaced by an actual error message
            return render_template('home.html') 
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_file_path', None)
    output_image_path = detect_object(img_file_path)
    obscure_object_path = obscure_object(img_file_path)
    
    # Perform Gemini thermal analysis
    gemini_image_path, gemini_analysis = analyze_thermal_image_with_gemini(img_file_path)
    
    # Pass the hardcoded value to the template
    return render_template(
        'show_image.html',
        user_image=img_file_path,
        detect_image=output_image_path,
        obscure_image=obscure_object_path,
        gemini_image=gemini_image_path,
        gemini_analysis=gemini_analysis,
        detected_numbers="5678"
    )

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/show_live_detect')
def showLiveDetect():
    return render_template('show_live.html')

@app.route('/live_feed_detect')
def live_feed_detect():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/show_live_obscure')
def showLiveObscure():
    return render_template('show_live_obscure.html')

@app.route('/live_feed_obscure')
def live_feed_obscure():
    return Response(gen(VideoCamera(version=1)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/show_video_detect')
def showVideoDetect():
    return render_template('show_video.html')

@app.route('/video_feed_detect')
def video_feed_detect():
    return Response(gen(VideoCamera(session.get('uploaded_file_path', None))),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/show_video_obscure')
def showVideoObscure():
    return render_template('show_video_obscure.html')

@app.route('/video_feed_obscure')
def video_feed_obscure():
    return Response(gen(VideoCamera(session.get('uploaded_file_path', None), version=1)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
 
# flask clear browser cache (disable cache)
# Solve flask cache images issue
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = True)

# Press any key to close the image
#cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()

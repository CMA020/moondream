import torch
import cv2
import os
import json
from ultralytics import YOLO
from PIL import Image
import time
from collections import deque
import sys
from pathlib import Path

# Add the directory containing the custom Moondream module to the path
# Assuming the custom Moondream module is in a directory accessible from your project
  # Adjust this path as needed

# Import custom Moondream components
from .moondream import MoondreamModel, MoondreamConfig
from .weights import load_weights_into_model

# YOLO model setup
model = YOLO(os.path.expanduser("/content/capture/last_p2s_12_7.pt"))

# Custom Moondream model setup
# Set device for torch
if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

# Initialize Moondream model with custom weights
def setup_moondream():
    # Load config
    config = MoondreamConfig()
    
    # Create model
    moondream_model = MoondreamModel(config)
    
    # Load custom weights
    weights_path = "/content/drive/MyDrive/moon_weigths/moondream_goal_detection_e12.safetensors"
    load_weights_into_model(weights_path, moondream_model)
    
    return moondream_model

# Initialize the model
moondream_model = setup_moondream()

# Global variables
vid = cv2.VideoCapture("/content/capture/manc.mp4")
goal_counter = 0
frame_counter = 0
rows, cols = 0, 0
class1_coords = None
last_person_coords = None

# Frame stack for video saving
frame_stack = deque(maxlen=150)

# Prediction window for tracking last 20 predictions
prediction_window = deque(maxlen=20)

# Video saving setup
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_directory = os.path.expanduser("/content/test")
os.makedirs(output_directory, exist_ok=True)
video_counter = 0

def predict(img):
    global class1_coords, last_person_coords, rows, cols
    results = model(img, imgsz=(1920, 1080), show=False)
    person_detected = False
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            x, y, w, h = map(int, box.xywh[0])
            if cls == 1:
                class1_coords = (x, y, w, h)
            if cls == 0:
                person_detected = True
                last_person_coords = (x, y, w, h)
    
    if not person_detected and last_person_coords:
        x, y, w, h = last_person_coords
    elif not person_detected:
        x, y, w, h = cols // 2, rows // 2, 100, 200
    
    coords_to_use = class1_coords if class1_coords else (x, y, w, h)
    return process_with_custom_moondream(img, coords_to_use)

def process_with_custom_moondream(img, coords):
    global goal_counter, prediction_window
    x, y, w, h = coords
    
    # Create crop region around the coordinates
    x1, y1 = max(0, x - int(w * 0.7)), max(0, y - int(h * 0.4))
    x2, y2 = min(cols, x + int(w * 1.7)), min(rows, y + int(h * 1.4))
    cropped_img = img[y1:y2, x1:x2]
    
    # Convert to PIL image
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    
    # Save temporary image for processing
    temp_path = "/tmp/temp_goal_image.jpg"
    pil_img.save(temp_path)
    
    try:
        # Load the image
        image = Image.open(temp_path)
        
        # Process with custom Moondream
        encoded_image = moondream_model.encode_image(image)
        
        # Prepare the prompt
        prompt = "\n\nQuestion: Is the ball in the goal.\n\nAnswer:"
        
        # Get response from model
        response_stream = moondream_model.query(encoded_image, prompt, stream=True)
        
        # Collect the answer
        answer = ""
        for t in response_stream["answer"]:
            answer += t
        
        # Convert to lowercase for comparison
        answer = answer.strip().lower()
        first_part = answer.split(',')[0]
        print(first_part, " C'est la prediction ")
    except Exception as e:
        print(f"Error in Moondream processing: {e}")
        first_part = "no"  # Default to "no" if there's an error
    
    # Determine prediction class (0 for goal, 1 for no goal)
    predicted_class = 0 if first_part == "yes" else 1
    
    # Add prediction to window
    prediction_window.append(predicted_class)
    
    # Check if we have enough predictions and if at least 14 are goals
    should_save_video = False
    if len(prediction_window) == 20:
        goal_count = sum(1 for pred in prediction_window if pred == 0)
        print(goal_count, "goal couuuuuunt")
        if goal_count >= 14:
            goal_counter += 1
            output_directory = "/content/capture/goal_preds"
            os.makedirs(output_directory, exist_ok=True)
            output_filename = f"goal_{goal_counter}.jpg"
            output_path = os.path.join(output_directory, output_filename)
            cv2.imwrite(output_path, img)
            should_save_video = True  # Only set to True if we have enough goal predictions
    
    return 0 if should_save_video else 1  

def save_video(frames, counter):
    global rows, cols
    output_filename = f"{counter}.mkv"
    output_path = os.path.join(output_directory, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, 30, (cols, rows))
    for frame in frames:
        out.write(frame)
    out.release()

if __name__ == '__main__':
    save_pending = False
    start_time = time.time()
    frame_count = 0
    frames_after_prediction = 0
    
    try:
        while True:
            ret, img = vid.read()
            if not ret:
                break
            
            frame_count += 1
            rows, cols, _ = img.shape
            frame_stack.append(img)
            frame_counter += 1
            
            if frame_counter % 6 == 0:  # Process every 6th frame
                prediction = predict(img)
                if prediction == 0:  # If model predicts goal with sufficient confidence
                    save_pending = True
                    frames_after_prediction = 0
            
            if save_pending:
                frames_after_prediction += 1
                if frames_after_prediction >= 45:  # Save video after 45 frames
                    video_counter += 1
                    save_video(frame_stack, video_counter)
                    save_pending = False
                    frames_after_prediction = 0
    
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        fps = frame_count / total_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps:.2f}")
        vid.release()
        cv2.destroyAllWindows()
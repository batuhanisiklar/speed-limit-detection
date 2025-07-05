import cv2
import pytesseract
import numpy as np
import os
import sys

# Constants
"""Your Tesseract path"""
TESSERACT_PATH = r"D:\Software\Tesseract\tesseract.exe"
VIDEO_PATH = "video3.mp4"
WINDOW_NAME = "Speed Sign Detection"
ROI_WINDOW_NAME = "Detected Sign"

# Check if Tesseract is installed at the given path
def check_tesseract_path(tess_path):
    if not os.path.isfile(tess_path):
        print(f"ERROR: Tesseract executable not found at '{tess_path}'.")
        print("Please install Tesseract OCR or update the TESSERACT_PATH variable.")
        print("You can download Tesseract from: https://github.com/tesseract-ocr/tesseract")
        sys.exit(1)

check_tesseract_path(TESSERACT_PATH)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def detect_speed_signs(frame):
    # Resize image (for faster processing)
    scale_factor = 0.5
    frame_resized = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
    
    # Red color ranges
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create red mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Simple noise reduction (fast and effective)
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Circle detection
    circles = cv2.HoughCircles(
        red_mask, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=30, 
        param1=50, 
        param2=25,
        minRadius=10, 
        maxRadius=50  # Smaller radius in resized image
    )
    
    detected_signs = []
    
    if circles is not None and hasattr(circles, "__getitem__"):
        circles = np.uint16(np.around(circles))
        # Defensive: ensure circles[0] exists and is iterable
        if len(circles) > 0 and hasattr(circles[0], "__iter__"):
            for circle in circles[0]:
                # Adjust coordinates to original image size
                center_x = int(circle[0] / scale_factor)
                center_y = int(circle[1] / scale_factor)
                radius = int(circle[2] / scale_factor)
                
                # Rectangle ROI around circle
                x = max(0, center_x - radius)
                y = max(0, center_y - radius)
                w = min(2 * radius, frame.shape[1] - x)
                h = min(2 * radius, frame.shape[0] - y)
                
                # Extract ROI
                if x+w > frame.shape[1] or y+h > frame.shape[0] or w <= 0 or h <= 0:
                    continue
                    
                roi = frame[y:y+h, x:x+w]
                
                # Check ROI size
                if roi.size == 0:
                    continue
                    
                detected_signs.append({
                    'center': (center_x, center_y),
                    'radius': radius,
                    'roi': roi,
                    'bbox': (x, y, w, h)
                })
    
    return detected_signs

def extract_speed_text(roi):
    if roi is None or roi.size == 0:
        return None
        
    # Resize ROI
    roi_resized = cv2.resize(roi, (100, 100))
    
    # Grayscale
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    
    # Contrast enhancement (simple and fast)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    
    # Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # OCR configuration - only get digits
    config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
    try:
        text = pytesseract.image_to_string(thresh, config=config).strip()
    except pytesseract.pytesseract.TesseractNotFoundError as e:
        print("TesseractNotFoundError:", e)
        print("Please check your Tesseract installation and path.")
        return None
    except Exception as e:
        print("OCR error:", e)
        return None
    
    # Number validation
    if text.isdigit():
        speed = int(text)
        # Validate speed range (typical speed limit signs)
        if 5 <= speed <= 200:
            return speed
    return None

def main():
    # Video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return
    
    # Create main window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)
    
    # FPS and video control
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    skip_frames = 1  # Process every 2nd frame (for speed)
    frame_count = 0
    
    # Last detected speed value
    last_detected_speed = None
    last_roi = None
    
    # ROI window state
    roi_window_created = False
    
    # Maximum signs to process per frame (for speed)
    max_signs_per_frame = 2
    
    # Previously detected speed values
    detected_speeds = set()
    
    # Speed detection cooldown
    cooldown_frames = 0
    cooldown_duration = 50  # Wait 10 frames after detection
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break
        
        frame_count += 1
        
        # Frame skipping (for speed)
        if frame_count % skip_frames != 0:
            # Show main image and keep last ROI
            cv2.imshow(WINDOW_NAME, frame)
            
            # Continue showing last detected sign
            if last_detected_speed is not None and last_roi is not None:
                if not roi_window_created:
                    cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(ROI_WINDOW_NAME, 300, 300)
                    roi_window_created = True
                
                cv2.imshow(ROI_WINDOW_NAME, last_roi)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        output_frame = frame.copy()
        
        # If cooldown is active, don't detect
        if cooldown_frames > 0:
            cooldown_frames -= 1
            signs = []
        else:
            signs = detect_speed_signs(frame)[:max_signs_per_frame]  # Only process first max_signs_per_frame signs
        
        # Was a sign detected?
        new_speed_detected = False
        current_roi = None
        current_speed = None
        
        for sign in signs:
            center = sign['center']
            radius = sign['radius']
            roi = sign['roi']
            
            # Read speed value
            speed = extract_speed_text(roi)
            
            if speed is not None:
                # Draw green circle
                cv2.circle(output_frame, center, radius, (0, 255, 0), 2)
                
                # Show speed text
                text = f"{speed} km/h"
                cv2.putText(
                    output_frame, 
                    text, 
                    (center[0] - 40, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Check if this speed was detected before
                if speed not in detected_speeds:
                    new_speed_detected = True
                    current_speed = speed
                    
                    # Prepare ROI
                    resized_roi = cv2.resize(roi, (300, 300))
                    cv2.putText(
                        resized_roi,
                        f"{speed} km/h",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    current_roi = resized_roi
                    
                    # Save speed value
                    detected_speeds.add(speed)
                    
                    # Start cooldown when new speed is detected
                    cooldown_frames = cooldown_duration
                
                # Always update last detected speed and ROI
                last_detected_speed = speed
                
                # Update last ROI (even if same speed value)
                resized_roi = cv2.resize(roi, (300, 300))
                cv2.putText(
                    resized_roi,
                    f"{speed} km/h",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                last_roi = resized_roi
        
        # Show cooldown status on screen
        if cooldown_frames > 0:
            cv2.putText(
                output_frame,
                f"Detection cooldown: {cooldown_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Show main image
        cv2.imshow(WINDOW_NAME, output_frame)
        
        # If a new speed is detected
        if new_speed_detected and current_roi is not None:
            if not roi_window_created:
                cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(ROI_WINDOW_NAME, 300, 300)
                roi_window_created = True
            
            # Show the detected sign in a separate window
            cv2.imshow(ROI_WINDOW_NAME, current_roi)
        # If a previously detected speed exists, continue showing it
        elif last_detected_speed is not None and last_roi is not None:
            if not roi_window_created:
                cv2.namedWindow(ROI_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(ROI_WINDOW_NAME, 300, 300)
                roi_window_created = True
            
            cv2.imshow(ROI_WINDOW_NAME, last_roi)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
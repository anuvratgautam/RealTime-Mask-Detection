import cv2
import numpy as np
import tensorflow as tf
import os

# Define the model filename
MODEL_FILE = 'face_mask_model.keras'

# Load the trained face mask classification model
print(f"Loading model: {MODEL_FILE}...")
if not os.path.exists(MODEL_FILE):
    print(f"Error: Model file '{MODEL_FILE}' not found.")
    print("Please make sure it's in the same directory as this script.")
    exit()

try:
    # Load the model you trained in Colab
    model = tf.keras.models.load_model(MODEL_FILE)
    print("Mask Classifier loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the pre-trained OpenCV face detector (Haar Cascade)
# This uses the XML file that comes built-in with the 'opencv-python' package
print("Loading OpenCV face detector...")
try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    if face_cascade.empty():
        print("Error: Could not load Haar Cascade file from OpenCV.")
        exit()
    else:
        print("Face Detector loaded successfully.")
except Exception as e:
    print(f"Error loading Haar Cascade: {e}")
    exit()


def predict_mask(face_roi, model):
    """
    Takes a face image (ROI) and returns the label and confidence score.
    """
    # 1. Resize image to what the model expects (128x128)
    face_img = cv2.resize(face_roi, (128, 128))
    
    # 2. Convert from BGR (OpenCV's format) to RGB (TensorFlow's format)
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # 3. Add a "batch" dimension (e.g., shape 1, 128, 128, 3)
    # The model expects a batch of images, even if it's just one.
    img_array = np.expand_dims(face_img_rgb, axis=0)

    # 4. Run prediction
    # The Rescaling(1./255) layer is part of the saved model,
    # so we can feed it pixel values from 0-255 directly.
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # The model outputs a single number:
    # < 0.5 means 'with_mask' (class 0)
    # > 0.5 means 'without_mask' (class 1)
    
    if prediction < 0.5:
        label = "With Mask"
        score = (1 - prediction) # Confidence
    else:
        label = "No Mask"
        score = prediction # Confidence

    return label, score


print("\nStarting webcam feed...")
print("Press 'q' in the video window to quit.")

# Initialize webcam. 0 is usually the built-in one.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read one frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Convert to grayscale (Haar Cascade works better on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect all faces in the frame
    # scaleFactor=1.1, minNeighbors=4 is a good balance
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Extract the face Region of Interest (ROI) from the *color* frame
        face_roi = frame[y:y+h, x:x+w]
        
        # Safety check: if the face is at the edge, the ROI can be empty
        if face_roi.size == 0:
            continue

        try:
            # Get the mask prediction
            label, score = predict_mask(face_roi, model)
            
            # Set the display color (Green for mask, Red for no mask)
            color = (0, 255, 0) if label == "With Mask" else (0, 0, 255)
            
            # Create the label text with the confidence score
            label_text = f'{label} ({score*100:.0f}%)'
            
            # Draw the rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw the label text above the rectangle
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
        except Exception as e:
            # This can happen if the face is too small or at an odd angle
            # We print 'pass' to just skip the frame instead of crashing
            pass

    # Display the final frame in a window
    cv2.imshow('Face Mask Detector (Press "q" to quit)', frame)

    # Check for the 'q' key to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Shutting down...")
cap.release()
cv2.destroyAllWindows()
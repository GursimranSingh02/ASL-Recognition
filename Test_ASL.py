# Description: This script uses the trained model to classify American Sign Language gestures in real-time using webcam.
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv.VideoCapture(0)

# Initialize hand detector (detects only 1 hand)
detector = HandDetector(maxHands=1)

# Load trained classification model and labels
classifier = Classifier("keras_model.h5", "labels.txt")

# Parameters for image processing
offset = 20  # Margin around the detected hand
imgSize = 300  # Size of the square image

# Labels for gesture classification
labels = ["A", "B", "C", "Calm down", "D", "E", "F", "G", "H", "Hello", "I", "I hate you", "I love you", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "Stop", "T", "U", "V", "W", "X", "Y"]

while True:
    # Capture frame from webcam
    success, img = cap.read()
    imgOutput = img.copy()

    # Detect hands in the frame
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]  # Only process the first detected hand
        
        # Bounding box information (x, y are top-left coordinates, w and h are width and height)
        x, y, w, h = hand['bbox']

        # Creating a white background image (300x300 pixels)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop the hand region from the frame with offset
        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        # Calculate aspect ratio of the cropped image
        aspectRatio = h / w

        # If height is greater than width, resize while maintaining aspect ratio
        if aspectRatio > 1:
            k = imgSize / h  # Scale factor
            wCal = math.ceil(k * w)  # Scaled width
            imgResize = cv.resize(imgCrop, (wCal, imgSize))  # Resize image
            wGap = math.ceil((imgSize - wCal) / 2)  # Centering horizontally
            imgWhite[:, wGap : wCal + wGap] = imgResize  # Place on white image
            prediction, index = classifier.getPrediction(imgWhite)  # Predict gesture

        # If width is greater than height, resize while maintaining aspect ratio
        else:
            k = imgSize / w  # Scale factor
            hCal = math.ceil(k * h)  # Scaled height
            imgResize = cv.resize(imgCrop, (imgSize, hCal))  # Resize image
            hGap = math.ceil((imgSize - hCal) / 2)  # Centering vertically
            imgWhite[hGap : hCal + hGap, :] = imgResize  # Place on white image
            prediction, index = classifier.getPrediction(imgWhite)  # Predict gesture

        # Draw a rectangle for the label display
        cv.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 225, y - offset), (255, 4, 255), cv.FILLED)

        # Display predicted label on the output image
        cv.putText(imgOutput, labels[index], (x + 20, y - 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Draw bounding box around detected hand
        cv.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (252, 4, 252), 4)

    # Show the output image
    cv.imshow('Image', imgOutput)

    # Exit when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv.destroyAllWindows()

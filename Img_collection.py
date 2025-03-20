# Description: This file is used to capture the images of the hand gestures and save it in the folder.
# The images are captured in the white background and the hand gesture is placed in the center of the image.
import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

folder = 'Data/Class_Name'
counter = 0

while True:
    sucess, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]                     # hands[0] means only single hand

        # bounded box information of hand    
        # w, h is width and height of image
        x, y, w, h = hand['bbox']   

        # Creating a image with white back-ground
        # Creating a matrix of ones of size 300 * 300 * 3
        # Color value range from 0 to 255 i.e., 8 bit value 
        # uint is unsigned integer of 8 bits
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255        

        # y is starting height, y+h is ending heigth & x is starting width, x+w is ending width
        # offset value is added to increase the boundry of cropped image
        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]   

        # placing the cropped image on the white image
        # Placing image crop matrix inside the image white matrix
        imageCropShape = imgCrop.shape

        aspecRatio = h/w 
        # Resizing an image if the height is max
        # In this case, height is fixed 
        if aspecRatio > 1:
            k = imgSize / h                                 # k is constant
            wCal = math.ceil(k * w)                         # wCal is calculated width
            imgResize = cv.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)          # To make the image at the center
            imgWhite[:, wGap : wCal + wGap] = imgResize

        # In this case, width is fixed
        else:
            k = imgSize / w                                 # k is constant
            hCal = math.ceil(k * h)                         # hCal is calculated height
            imgResize = cv.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)          # To make the image at the center
            imgWhite[hGap : hCal + hGap,:] = imgResize



        #cv.imshow("ImageCrop",imgCrop)
        cv.imshow("ImageWhite",imgWhite)

    cv.imshow('Image', img)
    key = cv.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
# real time background subtraction - detects moving objects from the background
# https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
# To redo - get the current RGB values from the mouse 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)

#lower = np.array([0, 48, 80], dtype = "uint8")
lower = np.array([0, 15, 15], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(frame)

    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 4)
    
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    cv2.imshow('a', skinMask)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
	
    # Display the original frame
    #cv2.imshow('frame',frame)
    # Display the foreground mask
    cv2.imshow('FG MASK FRAME', fgmask)
    cv2.imshow('SKIN MASK', np.hstack([frame, skin]))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

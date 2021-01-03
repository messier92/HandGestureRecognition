# Background subtraction code for hand gesture recognition project by GOH HAN LONG, EUGENE
# Use this version

# THE BASIC IDEA:
# Perform background subtraction using a hybrid method
# 1 - Average frame differencing (subtracts the current frame from the previous average frames)
# 2 - Skin color thresholding (specifies a colour in the frame to detect - light sensitive)
# 3 - Face removal (uses haar cascade to detect the position of the face and blocks it out
# 3a - Perhaps it might be better to split the current frame into 3 parts instead and flat out make the middle portion (where the face is) be blank
# After both hands (and arms are detected) - get the POSITION of the convex hull and compare their y axis

# To get the exact gesture can try - https://github.com/Gogul09/gesture-recognition/blob/master/recognize.py

# Next steps -> Do UDP to send the information over when driving (get the centroid of the left and right hull convex and compute the difference)

# Organize imports 
import cv2 
import numpy as np
from numpy import asarray
import PIL
import time
import random as rng
import socket
import sys

# Initialize empty lists
myImageAverageList = []
calibrated_img = []
calibrated_img_face_removed = []

# Set send IP adress and port
UDP_IP = "127.0.0.1"
UDP_PORT_Rec = 8052
UDP_PORT_Unity = 8055

print("Receiving on Port:" + str(UDP_PORT_Rec))
print("Sending to IP:" + UDP_IP + ":" + str(UDP_PORT_Unity))

# Set socket to send udp messages and bind port
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind(('',UDP_PORT_Rec));

# Capture frames from a camera 
cap = cv2.VideoCapture(0) 

# Set the skin-colour threshold (Getting the light levels right really is a big headache as I'm testing it in 3 different places - but feel free to change the skin colour threshold yourself)
lower = np.array([0, 25, 25], dtype = "uint8")
upper = np.array([20, 200, 200], dtype = "uint8")

# Specify the 4 points where you want the black rectangle to be
facearray = np.array([[[230,100],[230,1200],[450,1200],[450,100]]], dtype=np.int32 )

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop runs if capturing has been initialized.  
while(1): 
    # Reads frames from a camera  
    _, img = cap.read()

    # Copy a fresh image for the final output
    copy_contour_img = img.copy()
    copy_hull_img = img.copy()
    faceremovedimg = img.copy()

    # Detect and remove the face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x,y,w,h) in faces:
        #facearray = np.array([[[x,y],[x,y+h+400],[x+w+50,y+h+400],[x+w+50,y]]], dtype=np.int32 )
        faceremovedimg = cv2.fillPoly(img, facearray, 0)

    # Append the first t images to the list
    myImageAverageList.append(faceremovedimg)

    # Blend the first t images together to create a composite image for frame differencing
    for idx, img in enumerate(myImageAverageList):
        if idx == 0:
            first_img = img
            continue
        else:
            second_img = img
            second_weight = 1/(idx+1)
            first_weight = 1 - second_weight
            calibrated_img = cv2.addWeighted(first_img, 0.5, second_img, 0.5,0)

    # Once the blended image is produced...                                           
    if (len(calibrated_img) == 480):

            # Change the image from BGR to HSV
            converted = cv2.cvtColor(faceremovedimg, cv2.COLOR_BGR2HSV)
            converted = cv2.GaussianBlur(converted,(5,5),cv2.BORDER_DEFAULT)

            # Create the skin mask
            skinMask = cv2.inRange(converted, lower, upper)

            #cv2.imshow("Skin Mask", skinMask)

            # Frame DIfferencing - Find the difference between the calibrated image and the current frame
            differential_img = cv2.absdiff(calibrated_img, faceremovedimg)

            # Apply thresholding on differential_img to keep the gray areas only
            differential_img_blurred = cv2.medianBlur(differential_img,5)
            differential_img_grey = cv2.cvtColor(differential_img_blurred, cv2.COLOR_BGR2GRAY)
            ret,thresh1 = cv2.threshold(differential_img_grey,18,255,cv2.THRESH_BINARY)

            #cv2.imshow("Thresholded 1", thresh1)

            # Should not use BITWISE_AND because the worst one will affect the final quality
            # Use addWeighted to combine the SkinMask and the BackgroundSubtractedMas
            final_result =  cv2.addWeighted(thresh1, 0.5, skinMask, 0.5,0)
            kernel_dilate = np.ones((35,35),np.uint8)
            final_result_dilation = cv2.dilate(final_result,kernel_dilate,iterations = 11)
            kernel_erode = np.ones((3,3),np.uint8)
            final_result_erosion = cv2.erode(final_result,kernel_erode,iterations = 1)
            final_result_erosion_face_removed = cv2.fillPoly(final_result_erosion, facearray, 0)

            # do post-processing (dilation, remove blur, etc...)
            cv2.imshow("Final Result", final_result_erosion_face_removed)

            # Get the contours from the binary frame-differenced image
            contours, hierarchy = cv2.findContours(final_result_erosion_face_removed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize convex hull list
            hull_list = []
            areas_contours_list = []
            for contour in contours:
                # Get all the contour areas
                area = cv2.contourArea(contour)

                # Only accept contour areas that are more than 19000
                if (area >= 19000):
                    areas_contours_list.append(contour)
                    hull = cv2.convexHull(contour)
                    hull_list.append(hull)

                    hull_area = cv2.contourArea(hull)
                    solidity = area/hull_area                    

                    # Use Solidity for the best measure - you can't use hull or contour area because it will increase along with your arm and is position-based
                    # i.e. you can have the same gesture but show more of your arm
                    # ... unless we fix the location where your hand is supposed to be

                    if (len(hull_list) >=2):
                        
                        hull1textstring = str(hull_list[0][0][0][0]) + "," + str(hull_list[0][0][0][1])
                        solidity_image = cv2.putText(copy_hull_img,hull1textstring, (int(hull_list[0][0][0][0]),int(hull_list[0][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,80,255), 3, cv2.LINE_AA)

                        hull2textstring = str(hull_list[1][0][0][0]) + "," + str(hull_list[1][0][0][1])
                        solidity_image = cv2.putText(copy_hull_img,hull2textstring, (int(hull_list[1][0][0][0]),int(hull_list[1][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,80,255), 3, cv2.LINE_AA)

                        if ((hull_list[0][0][0][0]) < (hull_list[1][0][0][0])):
                            send_data = 'Turn right'
                            try:
                                sock.sendto(send_data.encode(), (UDP_IP, UDP_PORT_Unity))
                                print("\n\n 1. Client Sent : ", send_data, "\n\n")
                            except socket.error:
                                print("Error!")
                        else:
                            send_data = 'Turn left'
                            try:
                                sock.sendto(send_data.encode(), (UDP_IP, UDP_PORT_Unity))
                                print("\n\n 1. Client Sent : ", send_data, "\n\n")
                            except socket.error:
                                print("Error!")

                contours_image = cv2.drawContours(copy_contour_img, areas_contours_list, -1, (0, 255, 0), 2)
                contours_hull_image = cv2.drawContours(copy_hull_img, hull_list, -1, (0, 255, 0), 2)

            cv2.imshow("Hull contour", contours_hull_image)


    # Wait for Esc key to stop the program  
    k = cv2.waitKey(30) & 0xff
    if k == 27:  
        break
  
# Close the window  
cap.release()  
    
# De-allocate any associated memory usage  
cv2.destroyAllWindows() 

# Close the socket
sock.close()

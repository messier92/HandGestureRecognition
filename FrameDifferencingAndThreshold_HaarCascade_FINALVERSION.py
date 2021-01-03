# Background subtraction code for hand gesture recognition project by GOH HAN LONG, EUGENE

# THE BASIC IDEA:
# Perform background subtraction using a hybrid method
# 1 - Average frame differencing (subtracts the current frame from the previous average frames)
# 2 - Skin color thresholding (specifies a colour in the frame to detect - light sensitive)
# 3 - Face removal (uses haar cascade to detect the position of the face and blocks it out
# 3a - Perhaps it might be better to split the current frame into 3 parts instead and flat out make the middle portion (where the face is) be blank
# After both hands (and arms are detected) - get the POSITION of the convex hull and compare their y axis

# To get the exact gesture can try - https://github.com/Gogul09/gesture-recognition/blob/master/recognize.py

# Organize imports 
import cv2 
import numpy as np
from numpy import asarray
import PIL
import time
import random as rng
import socket
import sys
from sklearn.metrics import pairwise
from pynput.keyboard import Key, Controller

# Initialize empty lists
myImageAverageList = []
calibrated_img = []
calibrated_img_face_removed = []

# Set send IP adress and port
UDP_IP = "127.0.0.1"
UDP_PORT_Rec = 8052
UDP_PORT_Unity = 8055

keyboard = Controller()

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

time_start = 0
time_end = 20

# Specify the 4 points where you want the black rectangle to be
facearray = np.array([[[230,100],[230,1200],[450,1200],[450,100]]], dtype=np.int32 )

# Loop runs if capturing has been initialized.  
while(1): 
    # Reads frames from a camera  
    _, img = cap.read()

    # Copy a fresh image for the final output
    copy_contour_img = img.copy()
    copy_hull_img = img.copy()
    copy_circular_roi_img = img.copy()

    if (time_start < time_end):
        time_start += 1
        print("Calibrating..." + str(time_start))
        # Append the first t images to the list
        myImageAverageList.append(img)

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
    if (len(myImageAverageList) == time_end):
            #cv2.imshow('Calibrated Image for Frame Diffencing', calibrated_img)

            # Change the image from BGR to HSV
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            converted = cv2.GaussianBlur(converted,(5,5),cv2.BORDER_DEFAULT)

            # Create the skin mask
            skinMask = cv2.inRange(converted, lower, upper)

            #cv2.imshow("Skin Mask", skinMask)

            # Frame Differencing - Find the difference between the calibrated image and the current frame
            differential_img = cv2.absdiff(img, calibrated_img)

            #cv2.imshow('differential_img', differential_img)

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
            #cv2.imshow("Final Result", final_result_erosion_face_removed)

            # Get the contours from the binary frame-differenced image
            contours, hierarchy = cv2.findContours(final_result_erosion_face_removed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize convex hull list
            hull_list = []
            contours_list = []
            areas_contours_list = []
            area_circle_list = []
            for contour in contours:
                # Get all the contour areas
                area = cv2.contourArea(contour)
    
                # Only accept contour areas that are more than 19000
                if (area >= 19000):
                    
                    contours_list.append(contour)
                    areas_contours_list.append(area)
                    hull = cv2.convexHull(contour)
                    hull_list.append(hull)
                    
                    # find the most extreme points in the convex hull
                    extreme_top    = tuple(hull[hull[:, :, 1].argmin()][0])
                    extreme_bottom = tuple(hull[hull[:, :, 1].argmax()][0])
                    extreme_left   = tuple(hull[hull[:, :, 0].argmin()][0])
                    extreme_right  = tuple(hull[hull[:, :, 0].argmax()][0])

                    # find the center of the palm
                    cX = int((extreme_left[0] + extreme_right[0]) / 2)
                    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

                    # find the maximum euclidean distance between the center of the palm
                    # and the most extreme points of the convex hull
                    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
                    maximum_distance = distance[distance.argmax()]

                    # calculate the radius of the circle with 80% of the max euclidean distance obtained
                    radius = int(0.8 * maximum_distance)

                    # find the area of the circle
                    area_circle = np.pi*radius*radius
                    area_circle_list.append(area_circle)
                    
                    # find the circumference of the circle
                    circumference = 2 * np.pi * radius

                    # take out the circular region of interest which has the palm and the fingers
                    circular_roi = np.zeros(final_result_erosion_face_removed.shape[:2], dtype="uint8")

                    # draw the circular ROI
                    solidity_image = cv2.circle(copy_hull_img, (cX, cY), radius, 255, 1)

                    if (len(hull_list) >=2):
                        keyboard.press('w')

                        # you can use areas_contours_list to determine if the hand is open or closed 
                        #print(areas_contours_list)

                        ## For better accuracy, also make sure that the hands are in the upper quadrants of the image 
                        if (np.sum(area_circle_list) > 200000):
                            send_data = "Break!"
             
                            solidity_image = cv2.putText(copy_hull_img,"Break", (300,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,80,255), 6, cv2.LINE_AA)
                            try:
                                sock.sendto(send_data.encode(), (UDP_IP, UDP_PORT_Unity))
                                print("\n\n 1. Client Sent : ", send_data, "\n\n")
                            except socket.error:
                                print("Error!")
                        else:
                            hull1textstring = str(hull_list[0][0][0][0]) + "," + str(hull_list[0][0][0][1])
                            solidity_image = cv2.putText(copy_hull_img,hull1textstring, (int(hull_list[0][0][0][0]),int(hull_list[0][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,80,255), 3, cv2.LINE_AA)

                            hull2textstring = str(hull_list[1][0][0][0]) + "," + str(hull_list[1][0][0][1])
                            solidity_image = cv2.putText(copy_hull_img,hull2textstring, (int(hull_list[1][0][0][0]),int(hull_list[1][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,80,255), 3, cv2.LINE_AA)

                            if ((hull_list[0][0][0][0]) < (hull_list[1][0][0][0])):
                                send_data = "Turn right!"
                                #keyboard.release('a')
                                #keyboard.press('d')
                                solidity_image = cv2.putText(copy_hull_img,"Right!", (300,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,80,255), 6, cv2.LINE_AA)
                                try:
                                    sock.sendto(send_data.encode(), (UDP_IP, UDP_PORT_Unity))
                                    print("\n\n 1. Client Sent : ", send_data, "\n\n")
                                    #time.sleep(1)
                                except socket.error:
                                    print("Error!")
                            else:
                                send_data = "Turn left!"
                                #keyboard.release('d')
                                #keyboard.press('a')
                                solidity_image = cv2.putText(copy_hull_img,"Left!", (300,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,80,255), 6, cv2.LINE_AA)
                                try:
                                    sock.sendto(send_data.encode(), (UDP_IP, UDP_PORT_Unity))
                                    print("\n\n 1. Client Sent : ", send_data, "\n\n")
                                    #time.sleep(1)
                                except socket.error:
                                    print("Error!")

                contours_image = cv2.drawContours(copy_contour_img, contours_list, -1, (0, 255, 0), 2)
                contours_hull_image = cv2.drawContours(copy_hull_img, hull_list, -1, (0, 255, 0), 2)

            #cv2.imshow("Contour", contours_image)
            cv2.imshow("Hull", contours_hull_image)


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

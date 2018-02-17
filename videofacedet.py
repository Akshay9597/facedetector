import cv2
import time
video = cv2.VideoCapture(0)
#print(video.isOpened()) #should return true
first_frame = None
while True:
	check, frame = video.read()#check returns true if working properly and frame has the images
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	#converting to gray is better
	gray = cv2.GaussianBlur(gray,(21,21),0) #noise removal making image more clear
	
	if first_frame is None:
		first_frame = gray
		continue
	
	delta_frame = cv2.absdiff(first_frame,gray)	#difference
	
	thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]	#converting to white and black using threshold
	#THRESH_BINARY is a method. there are many which returns a list where we are using second element [1]
		
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)	#remove holes
	
	(_,cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	#draw all external countours 
	#approximation method used by opencv for retrieving the countours
	
	for contour in cnts:
		if cv2.contourArea(contour) < 1000:	#area should be large enough
			continue
		(x,y,w,h) = cv2.boundingRect(contour)	#Draw the rectangle
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),3)	#Green rectangle
		
		
	cv2.imshow("delta",delta_frame)
	cv2.imshow("Capturing your face", gray)
	cv2.imshow("Threshold frame",thresh_frame)
	cv2.imshow("Original",frame)
	key = cv2.waitKey(1)	#waiting for you to press any key
	if(key == ord('q')):
		break
	
	
video.release()	#
cv2.destroyAllWindows()	#close all opened windows

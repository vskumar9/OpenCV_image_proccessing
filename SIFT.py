import cv2 as cv
import numpy as np

img= cv.resize(cv.imread("sanjeev.jpg"),(1000,600))
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#APPLY SIFT detector
sift=cv.xfeatures2d.SIFT_create()
keypoint,descriptos=sift.detectAndCompute(img,None)

#marking the key point on the image using circles 
img=cv.drawKeypoints(gray,keypoint,img)
cv.imshow("key point image",img)
cv.waitKey(0)
cv.destroyAllWindows()

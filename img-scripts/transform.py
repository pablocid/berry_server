import numpy as np
import cv2
import imutils
from perspective import four_point_transform
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import cv2
import uuid
import sys, getopt
# from skimage.filter import threshold_adaptive
import urllib
from urllib2 import urlopen
from cStringIO import StringIO
from skimage import io
import math
import base64

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

def create_opencv_image_from_url(url, cv2_img_flag=0):
    request = urlopen(url)
    img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 

#print sys.stdin.readline();

#image = cv2.imread("bayas.jpg")
#image = url_to_image("bayas.jpg")

def analyzer(url):
	image = io.imread(url)
	ratio = image.shape[0] / 600.0
	orig = image.copy()
	image = imutils.resize(image, height = 600)
	
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)
	
	# show the original image and the edge detected image
	# print "STEP 1: Edge Detection"
	# cv2.imshow("Image", image)
	# cv2.imshow("Edged", edged)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	_ ,cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	
		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break
	
	# show the contour (outline) of the piece of paper
	# print "STEP 2: Find contours of paper"
	# cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	# cv2.imshow("Outline", image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	if 'screenCnt' in locals():
    		hola = "hola"
	else:
    		print "001"
		return

	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	
	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect
	#warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	# warped = threshold_adaptive(warped, 251, offset = 10)
	#warped = warped.astype("uint8") * 255
	
	# show the original and scanned images
	# print "STEP 3: Apply perspective transform"
	# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
	# cv2.waitKey(0)

	#height, width = warped.shape[:2]


	#orig = warped[10:590, 10:590]



	# load the image, convert it to grayscale, and blur it slightly
	image = warped[5:490, 5:770].copy() #cv2.imread(args["image"])
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=2)
	edged = cv2.erode(edged, None, iterations=2)


	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	
	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable
	(cnts, _) = contours.sort_contours(cnts)
	pixelsPerMetric = None

	orig = image.copy()
	# loop over the contours individually
	mCoord = []

	for c in cnts:
		# if the contour is not sufficiently large, ignore it
		if cv2.contourArea(c) < 20:
			continue
	
		# compute the rotated bounding box of the contour
		
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="float")
	
		# order the points in the contour such that they appear
		# in top-left, top-right, bottom-right, and bottom-left
		# order, then draw the outline of the rotated bounding
		# box
		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)
	
		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 3, (0, 0, 255), -1)

			# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates
		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)
	
		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
	
		# draw the midpoints on the image
		cv2.circle(orig, (int(tltrX), int(tltrY)), 3, (255, 0, 0), -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 3, (255, 0, 0), -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 3, (255, 0, 0), -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 3, (255, 0, 0), -1)
	
		# draw lines between the midpoints
		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 1)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 1)


		# compute the Euclidean distance between the midpoints
		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		
		# if the pixels per metric has not been initialized, then
		# compute it as the ratio of pixels to supplied metric
		# (in this case, inches)
		if pixelsPerMetric is None:
			pixelsPerMetric = (dA + dB)/2

		# compute the size of the object
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric
		area = cv2.contourArea(c) /(pixelsPerMetric * pixelsPerMetric)
		(x,y),radius = cv2.minEnclosingCircle(c)
		circleVsArea = cv2.contourArea(c) / (math.pi * radius * radius)
		rectangleVsArea = cv2.contourArea(c) / (dA*dB)
		ellipseVsArea = cv2.contourArea(c) / (math.pi * dA * dB)

		mean = (dimA + dimB)/2
		mCoord.append([dimA, dimB, mean, area,rectangleVsArea, circleVsArea, ellipseVsArea])
		# draw the object sizes on the image
		cv2.putText(orig, "{:.1f} cm".format(dimA),
			(int(tltrX - 10), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 1)
		cv2.putText(orig, "{:.1f} cm".format(dimB),
			(int(trbrX + 5), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 1)

	
	
	# tempName = "test.jpg"
	# cv2.imwrite(tempName, orig);
	# return 
	namePic = "/tmp/"+str(uuid.uuid1())+".jpg"
	cv2.imwrite(namePic, orig);
	print namePic
	print mCoord[:]
	return mCoord

analyzer(str(sys.argv[1]))


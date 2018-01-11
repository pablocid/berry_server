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


def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def findRectangle(frame):

	template=cv2.imread("reference.jpg")

	result = cv2.matchTemplate(frame.copy(), template, cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	_, w, h = template.shape[::-1]

	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)


	warped = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	warped = cv2.GaussianBlur(warped, (7, 7), 0)

	edged = cv2.Canny(warped, 30, 150)
	edged = cv2.dilate(edged, None, iterations=2)
	edged = cv2.erode(edged, None, iterations=2)

	_ ,cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	area = 1
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			area = cv2.contourArea(c)
			break
	
	return math.sqrt(area)


def simplest_cb(img, percent):
    #assert img.shape[2] == 3a
    #assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        #assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        #assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

        #print "Lowval: ", low_val
        #print "Highval: ", high_val

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 

def analyzer(url):
	image = cv2.imread(url)
	
	ratio = image.shape[0] / 600.0
	orig = image.copy()
	image = imutils.resize(image, height = 600)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)



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
	

	if 'screenCnt' in locals():
    		hola = "hola"
	else:
    		print "001"
		return

	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	
	image = simplest_cb(warped, 1)
	orig = image.copy()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	edged = cv2.Canny(gray, 30, 150)
	edged = cv2.dilate(edged, None, iterations=2)
	edged = cv2.erode(edged, None, iterations=2)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	(cnts, _) = contours.sort_contours(cnts)
	pixelsPerMetric = findRectangle(image)

	mCoord = []
	
	for c in cnts:
	
		if cv2.contourArea(c) < 25:
			continue

		approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
		area = cv2.contourArea(c)

		if ((len(approx) < 10) ):
    				continue

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

		# compute the size of the object
		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric
		area = cv2.contourArea(c) /(pixelsPerMetric **2)
		(x,y),radius = cv2.minEnclosingCircle(c)
		circleVsArea = cv2.contourArea(c) / (math.pi * radius * radius)
		rectangleVsArea = cv2.contourArea(c) / (dA*dB)
		ellipseVsArea = cv2.contourArea(c) / (math.pi * dA * dB /4)

		mean = (dimA + dimB)/2
		mCoord.append([dimA, dimB, mean, area,rectangleVsArea, circleVsArea, ellipseVsArea])
		# draw the object sizes on the image
		cv2.putText(orig, "{:.1f}cm".format(dimB),
			(int(tltrX - 10), int(tltrY - 5)), cv2.FONT_HERSHEY_SIMPLEX,
			0.45, (0, 0, 255), 1)
		cv2.putText(orig, "{:.1f}cm".format(dimA),
			(int(trbrX + 5), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.45, (0, 0, 255), 1)
	
	
	# tempName = "test.jpg"
	# cv2.imwrite(tempName, orig);
	# return

	namePic = "/tmp/"+str(uuid.uuid1())+".jpg"
	cv2.imwrite(namePic, orig);
	print namePic
	print mCoord[:]
	return mCoord

analyzer(str(sys.argv[1]))


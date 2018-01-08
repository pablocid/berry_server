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

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    #assert img.shape[2] == 3
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
 

#print sys.stdin.readline();

#image = cv2.imread("bayas.jpg")
#image = url_to_image("bayas.jpg")

def autocrop(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

def analyzer(url):
	image = cv2.imread(url)

	#image = simplest_cb(image, 1)
	
	ratio = image.shape[0] / 600.0
	orig = image.copy()
	image = imutils.resize(image, height = 600)
	
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)

	# tempName = "test.jpg"
	# cv2.imwrite(tempName, edged)
	# return 
	
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
	image = warped#[5:480, 5:660].copy() #cv2.imread(args["image"])
	#image = autocrop(warped, 100)
	
	image = simplest_cb(image, 1)

	#############
	# img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
 	# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,0.5,40,param1=80,param2=25,minRadius=10,maxRadius=100)
	
 	# circles = np.uint16(np.around(circles))
 	# for i in circles[0,:]:
 	#     # draw the outer circle
 	#     cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
 	#     # draw the center of the circle
 	#     cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)
	
 	# tempName = "test.jpg"
	# cv2.imwrite(tempName, image)
	# return 

	#############

	# Add black border in case that page is touching an image border		
	#image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])		
	orig = image.copy()

	# tempName = "test.jpg"
	# cv2.imwrite(tempName, image)
	# return 

	
	# clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(1,1))
	# lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
	# l, a, b = cv2.split(lab)
	# l2 = clahe.apply(l)
	# lab = cv2.merge((l2,a,b))  # merge channels
	# image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# tempName = "test.jpg"
	# cv2.imwrite(tempName, image);
	# return 
	
	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges
	edged = cv2.Canny(gray, 30, 150)
	edged = cv2.dilate(edged, None, iterations=2)
	edged = cv2.erode(edged, None, iterations=2)

	# tempName = "test.jpg"
	# cv2.imwrite(tempName, edged);
	# return 


	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	
	# sort the contours from left-to-right and initialize the
	# 'pixels per metric' calibration variable
	(cnts, _) = contours.sort_contours(cnts)
	pixelsPerMetric = None

	


	# loop over the contours individually
	mCoord = []
	grapes = []
	reference = 0

	for c in cnts:
		# if the contour is not sufficiently large, ignore it
		
		print cv2.contourArea(c)
		if cv2.contourArea(c) < 25:
			continue

		if cv2.contourArea(c) > 2000:
    			continue

		approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
		area = cv2.contourArea(c)

		if ((len(approx) < 4) ):
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
	
	
	tempName = "test.jpg"
	cv2.imwrite(tempName, orig);
	return

	namePic = "/tmp/"+str(uuid.uuid1())+".jpg"
	cv2.imwrite(namePic, orig);
	print namePic
	print mCoord[:]
	return mCoord

analyzer(str(sys.argv[1]))


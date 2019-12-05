import imutils
import cv2
import numpy as np
import argparse
from imutils.perspective import four_point_transform
from imutils import contours

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Input Image")
args = vars(ap.parse_args())

CorrectAns = {0:1, 1:4, 2:0, 3:3, 4:1}

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#cv2.imshow("blurred", blurred)
edged = cv2.Canny(blurred, 75, 200)
#edged = cv2.Canny(blurred)
#cv2.imshow("edged", edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# print(len(cnts))
# for c in cnts:
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	#cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# 	cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
docCnt = None
if len(cnts) > 0:
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	for c in cnts:
		perimeter = cv2.arcLength(c, True)
		#print(perimeter)
		approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
		#print(approx)
		if len(approx) == 4:
			docCnt = approx
			break

imageTransformed = four_point_transform(image, docCnt.reshape(4, 2))
grayTransformed = four_point_transform(gray, docCnt.reshape(4, 2))

# need to repeat similar process above
thresh = cv2.threshold(grayTransformed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

#print(len(cnts))
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	#print(w, h)
	if w>20 and w<50 and h>20 and h<50:
		questionCnts.append(c)
		#cv2.drawContours(imageTransformed, [c], -1, (0, 0, 255), 2)

questionCnts, _ = contours.sort_contours(questionCnts, method="top-to-bottom")
correct = 0

for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
	cnts, _ = contours.sort_contours(questionCnts[i:i+5])
	bubbled = None
	#print(i)
	for (j, c) in enumerate(cnts):
		mask = np.zeros(thresh.shape, dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)
		# cv2.imshow("thresh", thresh)
		# cv2.imshow("mask", mask)
		# break
	color = (0, 0, 255)
	ans = CorrectAns[q]
	if ans == bubbled[1]:
		color = (0, 255, 0)
		correct += 1
	cv2.drawContours(imageTransformed, [cnts[ans]], -1, color, 3)

score = (correct / (len(questionCnts)/5)) * 100.0
cv2.putText(imageTransformed, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

cv2.imshow("Original Image", image)
cv2.imshow("Result Image", imageTransformed)

#cv2.imshow("Gray", grayTransformed)
#cv2.imshow("GrayOtsu", thresh)
cv2.waitKey(0)



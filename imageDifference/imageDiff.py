#from skimage.measure import compare_ssim
from skimage.metrics._structural_similarity import structural_similarity
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="first image")
ap.add_argument("-s", "--second", required=True, help="second image")
args = vars(ap.parse_args())

image1 = cv2.imread(args["first"])
image2 = cv2.imread(args["second"])

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#score, diff = compare_ssim(gray1, gray2, full=True)
score, diff = structural_similarity(gray1, gray2, full=True)
#print(score)
diff = (diff * 255).astype("uint8")

#cv2.imshow("diff", diff)

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#cv2.imshow("diff after threshold", thresh)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#print(cnts[0][0][0])
for c in cnts:
	x, y, w, h = cv2.boundingRect(c)
	cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 0, 255), 2)
	cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Original", image1)
cv2.imshow("Modified", image2)
cv2.waitKey(0)




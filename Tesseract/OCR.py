import cv2
import argparse
import pytesseract

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image file path")
args = vars(ap.parse_args())

img = cv2.imread(args["image"], cv2.IMREAD_COLOR)

#text = pytesseract.image_to_string(img, config='-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(img, config='-l jpn --oem 1 --psm 3')

file = open("ocr_jpn.txt","a")
print(text)
file.writelines(text)
file.close()
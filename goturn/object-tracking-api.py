import cv2
import sys
import time

tracker = cv2.TrackerKCF_create() # KCF fast and accurate, CSRT high accurate but slower, MOSSE faster but low accuracy

# For video file
video = cv2.VideoCapture("./chaplin.mp4")

# for video stream / webcam
# video = cv2.VideoStream(src=0).start()
# time.sleep(2.0)

ok, frame = video.read()

bbox = cv2.selectROI(frame, False)
print(bbox)
#cv2.destroyWindow(frame)
tracker.init(frame, bbox)

while True:
	ok, frame = video.read()

	timer = cv2.getTickCount()
	ok, bbox = tracker.update(frame)
	fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)

	p1 = (int(bbox[0]), int(bbox[1]))
	p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
	cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

	# Display tracker type on frame
	#cv2.putText(frame, "Bosting Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

	# Display FPS on frame
	#cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

	# Display result
	cv2.imshow("Tracking", frame)

	# Exit if ESC pressed
	k = cv2.waitKey(1) & 0xff
	if k == 'q':
		#video.release() #release for file
		video.stop() # stop for webcam
		break

cv2.destroyAllWindows()
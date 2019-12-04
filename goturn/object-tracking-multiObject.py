import cv2
import time

trackers = cv2.MultiTracker_create() # default is KCF
# For video file
video = cv2.VideoCapture("./goals.mp4")

# for video stream / webcam
# video = cv2.VideoStream(src=0).start()
# time.sleep(2.0)

#ok, frame = video.read()

#bbox = cv2.selectROI(frame, False)
#print(bbox)
#cv2.destroyWindow(frame)
#tracker.init(frame, bbox)
#print('before loop')
while True:
	#print('inside loop')
	time.sleep(0.01)
	ok, frame = video.read()
	if not ok:
		break
	# timer = cv2.getTickCount()
	ok, bbox = trackers.update(frame)
	# fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)
	if not ok:
		break
	# p1 = (int(bbox[0]), int(bbox[1]))
	# p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
	# cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
	for box in bbox:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	# Display tracker type on frame
	#cv2.putText(frame, "Bosting Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

	# Display FPS on frame
	#cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

	# Display result
	cv2.imshow("Tracking", frame)

	# Exit if ESC pressed
	k = cv2.waitKey(1) & 0xff

	if k == ord('s'):
		box = cv2.selectROI("Tracking", frame, False)
		tracker = cv2.TrackerKCF_create()
		trackers.add(tracker, frame, box)

	elif k == ord('q'):
		break
video.release() #release for file
#video.stop() # stop for webcam
cv2.destroyAllWindows()
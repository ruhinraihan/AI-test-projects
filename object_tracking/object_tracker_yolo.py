from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

confidence = 0.8

configPath = "./yolov3.cfg"
weightPath = "./yolov3.weights"

ct = CentroidTracker()
(H, W) = (None, None)

#net = cv2.dnn.readNetFromCaffe(prototxt, model)
net = cv2.dnn.readNetFromDarknet(configPath, weightPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

vs = VideoStream(src=0).start()
time.sleep(2)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=416, height=416)

	(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	rects = []
	# for i in range(0, detections.shape[2]):
	# 	if detections[0, 0, i, 2] > confidence:
	# 		box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
	# 		rects.append(box.astype("int"))
	#
	# 		(startX, startY, endX, endY) = box.astype("int")
	# 		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			if scores[classID] > confidence:
				#box = detection[0:4] #centerX, centerY, boxW, boxH
				box = [detection[0]-detection[2]/2, detection[1]-detection[3]/2, detection[0]+detection[2]/2, detection[1]+detection[3]/2] * np.array([W, H, W, H])
				rects.append(box.astype("int"))
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

	objects = ct.update(rects)

	for(objectID, centroid) in objects.items():
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	cv2.imshow("Object Tracking", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destoyAllWindows()
vs.stop()

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.learningratefinder import LearningRateFinder
from pyimagesearch.clr_callback import CyclicLR
from pyimagesearch import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import sys
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--lr-find", type=int, default= 0,
				help = "Whether or not to find optimal learning rate")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.DATASET_PATH))
data = []
labels = []

for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]
	#print (label)

	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224)) #size is VGG16 requirement

	data.append(image)
	labels.append(label)
	print(label)

print("[INFO] processing data")
data = np.array(data, dtype="float32")
labels = np.array(labels)

print(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=config.TEST_SPLIT, random_state=42)

(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=config.VAL_SPLIT, random_state=84)

aug = ImageDataGenerator(
	rotation_range = 30,
	zoom_range = 0.15,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.15,
	horizontal_flip = True,
	fill_mode = "nearest"
)

baseModel = VGG16(weights = "imagenet", include_top = False, input_tensor=Input(shape= (224, 224, 3)))

headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES),activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

print("[INFO] Compiling Model...")
opt = SGD(lr = config.MIN_LR, momentum = 0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrices=["accuracy"])

if args["lr_find"] > 0:
	print("[INFO] Finding learning rate... ")
	lrf = LearningRateFinder(model)
	lrf.find(
		aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE),
		1e-10, 1e+1,
		stepsPerEpoch=np.ceil((trainX.shape[0] / float(config.BATCH_SIZE))),
		epochs=20,
		batchSize=config.BATCH_SIZE
	)

	lrf.plot_loss()
	plt.savefig(config.LRFIND_PLOT_PATH)

	print("[INFO] learning rate finder complete....")
	print("[INFO] examine plot and adjust learning rate before training")
	sys.exit(0)
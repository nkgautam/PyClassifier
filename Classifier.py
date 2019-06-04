from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
MD = { "vgg16": (VGG16, (224, 224)) }

def loadConvert(image_path, model):
	inShape = MD[model][1]
	preprocess = imagenet_utils.preprocess_input
	image = load_img(image_path, target_size=inShape)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = preprocess(image)
	return image

def classify_image(image_path):
	model = "vgg16"
	img = loadConvert(image_path, model)
	NT = MD[model][0]
	model = NT(weights="imagenet")
	preds = model.predict(img)
	P = imagenet_utils.decode_predictions(preds)
	for (i, (imagenetID, label, prob)) in enumerate(P[0]):
		print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

classify_image("Anaconda3\envs\AnacondaTensor\PythonSrc\parrot.jpg")
import tensorflow as tf
import numpy as np
import cv2
import json
from keras.utils import to_categorical
from PIL import Image


def RGBtoOneHot(rgb, colorDict):
	arr = np.zeros(rgb.shape[:2])  ## rgb shape: (h,w,3); arr shape: (h,w)

	for idx in range(len(colorDict)):
		for color in colorDict[idx].values():
			color = np.array(list(color))

			if idx < len(colorDict):
				if idx == 32:
					print(idx)
					print(np.all(rgb == 70, axis=-1).shape)
					print(rgb.shape)
				arr[np.all(rgb == color, axis=-1)] = idx

	return arr


def onehot_to_rgb(one_hot, color_dict):
	print(f"one hot shape sart")
	print(one_hot.shape)

	one_hot = np.squeeze(one_hot, axis=0)
	one_hot = np.argmax(one_hot, axis=-1)

	output = np.zeros(one_hot.shape + (3,))
	print(f"output shape {output.shape}")

	one_hot = np.expand_dims(one_hot, axis=-1)
	print(f"one hot shape {one_hot.shape}")

	for index in range(len(color_dict)):
		for color in color_dict[index].values():
			color = np.array(list(color))

			if index < len(color_dict):
				output[np.all(one_hot == index, axis=-1)] = color

	output = output.astype(np.uint8)
	output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
	print(output)
	cv2.imshow("winn", output)
	cv2.waitKey(0)


file = open("config.json")
config = json.load(file)
model = tf.keras.models.load_model(r"C:\Users\devjo\Documents\Projects\semantic-segmentation\models\unet.h5")

img = tf.io.read_file("../imgs/N3io5doqqpSZrflnBk2YIA.jpg")
img = tf.io.decode_png(img)
img = tf.image.convert_image_dtype(img, tf.float16)

preds = model.predict(tf.expand_dims(img, axis=0))
onehot_to_rgb(preds, config)


# mask = np.array(Image.open(f"../imgs/N3io5doqqpSZrflnBk2YIA.png"))
#
# mask = RGBtoOneHot(mask, config)
# mask = to_categorical(mask)
# onehot_to_rgb(mask, config)
# print(mask[np.all(mask == np.array([120,10,10]), axis=-1)])
# print(mask.shape)


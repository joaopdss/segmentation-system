import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image


def RGBtoOneHot(rgb, colorDict):
	arr = np.zeros(rgb.shape[:2])  ## rgb shape: (h,w,3); arr shape: (h,w)

	for idx in range(len(colorDict)):
		for color in colorDict[idx].values():
			color = np.array(list(color))

			if idx < len(colorDict):
				arr[np.all(rgb == color, axis=-1)] = idx

	return arr


def onehot_to_rgb(one_hot, color_dict):
	# one_hot = np.array(one_hot)
	# one_hot = np.argmax(one_hot, axis=-1)
	# print(color_dict)
	output = np.zeros(one_hot.shape + (3,))
	# output = np.zeros(one_hot.shape[1:3] + (3,))
	# one_hot = np.transpose(one_hot, [1, 2, 0])

	for index in range(len(color_dict)):
		for color in color_dict[index].values():
			color = np.array(list(color))
			if index == 50:
				print(color)
				print(one_hot)
			if index < len(color_dict):
				output[np.all(one_hot == np.float64(index), axis=-1)] = color
				if index == 50:
					print(np.all(one_hot == np.float64(index), axis=-1))

	cv2.imshow("winn", output)
	cv2.waitKey(0)


file = open("config.json")
config = json.load(file)
model = tf.keras.models.load_model(r"C:\Users\devjo\Documents\Projects\semantic-segmentation\models\unet.h5")

img = tf.io.read_file("../imgs/N3io5doqqpSZrflnBk2YIA.jpg")
img = tf.io.decode_png(img)
img = tf.image.convert_image_dtype(img, tf.float16)

preds = model.predict(tf.expand_dims(img, axis=0))
# onehot_to_rgb(preds, config[0])


mask = np.array(Image.open(f"../imgs/N3io5doqqpSZrflnBk2YIA.png"))
print(mask[0][0])
mask = RGBtoOneHot(mask, config)
onehot_to_rgb(mask, config)
# print(mask[np.all(mask == np.array([120,10,10]), axis=-1)])
# print(mask.shape)


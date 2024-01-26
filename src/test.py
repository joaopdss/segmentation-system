import tensorflow as tf
import numpy as np
import cv2

def onehot_to_rgb(one_hot, color_dict):
	one_hot = np.array(one_hot)
	one_hot = np.argmax(one_hot, axis=-1)
	output = np.zeros(one_hot.shape[1:3]+(3,))

	print(one_hot.shape)

model = tf.keras.models.load_model(r"C:\Users\devjo\Documents\Projects\semantic-segmentation\models\unet.h5")

img = tf.io.read_file("../imgs/_J6Ge9plcquFpJng9IXIAQ.jpg")
img = tf.io.decode_png(img)
img = tf.image.convert_image_dtype(img, tf.float16)

preds = model.predict(tf.expand_dims(img, axis=0))

onehot_to_rgb(preds, 0)



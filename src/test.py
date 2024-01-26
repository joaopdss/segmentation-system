import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("../models/unet.h5")

img = tf.io.read_file("../imgs/_J6Ge9plcquFpJng9IXIAQ.jpg")
img = tf.io.decode_png(img)
img = tf.image.convert_image_dtype(img, tf.float16)

preds = model.predict(tf.expand_dims(img, axis=0))
print(preds[0])
print(np.array(preds[0][150][150]).astype(np.int32))


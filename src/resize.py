import os
import cv2

for image in os.listdir(r"C:\Users\devjo\Documents\Projects\mapillary\validation\v2.0\labels"):
	img = cv2.imread(rf"C:\Users\devjo\Documents\Projects\mapillary\validation\v2.0\labels\{image}")
	if img.shape != (224, 224, 3):
		print("w")
	img = cv2.resize(img, (224, 224))
	cv2.imwrite(rf"C:\Users\devjo\Documents\Projects\mapillary\validation\v2.0\labels\{image}", img)
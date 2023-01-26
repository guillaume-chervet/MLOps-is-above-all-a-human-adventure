from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from shutil import copyfile
from os import makedirs
from pathlib import Path
import numpy as np
import cv2
import time


# load and prepare the image
def load_image(filename):
	start = time.time()
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	end = time.time()
	#print("pillow time: " + str(end - start))
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

def load_image_cv(filename):
	start = time.time()
	local_file = cv2.imread(filename)
	img = cv2.resize(local_file, (224, 224), interpolation=cv2.INTER_CUBIC)
	# convert to array
	# Numpy array
	np_image_data = np.asarray(img)
	# maybe insert float convertion here - see edit remark!
	np_final = np.expand_dims(np_image_data, axis=0)
	end = time.time()
	#print("cv time: " + str(end - start))

	# reshape into a single sample with 3 channels
	img = np_final.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

def rotate_cv_180(img):
	(h, w) = img.shape[:2]
	center = (w / 2, h / 2)
	angle90 = 180
	scale = 1.0
	# Perform the counter clockwise rotation holding at the center
	# 90 degrees
	M = cv2.getRotationMatrix2D(center, angle90, scale)
	rotated90 = cv2.warpAffine(img, M, (h, w))
	return rotated90

def load_image_cv180(filename):
	start = time.time()
	local_file = cv2.imread(filename)
	img = cv2.resize(local_file, (224, 224), interpolation=cv2.INTER_CUBIC)
	img = rotate_cv_180(img)
	# convert to array
	# Numpy array
	np_image_data = np.asarray(img)
	# maybe insert float convertion here - see edit remark!
	np_final = np.expand_dims(np_image_data, axis=0)
	end = time.time()
	#print("cv time: " + str(end - start))

	# reshape into a single sample with 3 channels
	img = np_final.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img

# load an image and predict the class
def run_example():
	makedirs("fails/cat", exist_ok=True)
	makedirs("fails/dog", exist_ok=True)
	makedirs("fails/other", exist_ok=True)
	makedirs("dog_cv", exist_ok=True)
	makedirs("cat_cv", exist_ok=True)
	makedirs("other_cv", exist_ok=True)
	flist = [p for p in Path("dogs-vs-cats/test").iterdir() if p.is_file()]
	# load model
	model = load_model('./final_model.h5')
	diff = []
	for file_path in flist:

		print(file_path)
		# load the image

		# predict the class
		result_pillow = model.predict(load_image(str(file_path)))
		result_cv = model.predict(load_image_cv(str(file_path)))
		switcher = ['Cat', 'Dog', 'Other']
		prediction_pillow = np.argmax(result_pillow[0])
		prediction_cv = np.argmax(result_cv[0])
		if prediction_cv != prediction_pillow:
			copyfile(file_path,  "./fails2/" + Path(file_path).stem + "_cv2-"+switcher[prediction_cv] + "_pillow-" + switcher[prediction_pillow] + Path(file_path).suffix)


	import json
	with open("diff180.json", "w") as out_file:
		json.dump(diff, out_file, indent=6)

run_example()

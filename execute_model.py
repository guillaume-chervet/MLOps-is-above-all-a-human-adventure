from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from shutil import copyfile
from os import makedirs
from pathlib import Path

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img


# load an image and predict the class
def run_example():
	makedirs("cat", exist_ok=True)
	makedirs("dog", exist_ok=True)
	flist = [p for p in Path("./dogs-vs-cats/test").iterdir() if p.is_file()]
	# load model
	model = load_model('./production/model/final_model.h5')
	for file_path in flist:

		print(file_path)
		# load the image
		img = load_image(file_path)

		# predict the class
		result = model.predict(img)

		if result[0]==1:
			copyfile(file_path,  "./dog/" + Path(file_path).stem + Path(file_path).suffix)
		else:
			copyfile(file_path,  "./cat/" + Path(file_path).stem + Path(file_path).suffix)



run_example()

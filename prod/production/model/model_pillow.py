from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
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

BASE_PATH = Path(__file__).resolve().parent

class Model:
	def __init__(self, logging, app_settings):
		self.logger = logging.getLogger(__name__)
		self.model = load_model(str(BASE_PATH / 'final_model.h5'))

	def execute(self, file, filename, settings=None):
		with open(filename, "wb") as stream:
			stream.write(file.getbuffer())
		img = load_image(filename)
		result = self.model.predict(img)
		if result[0] == 1:
			return {"prediction": "Dog", "confidence": 1}
		else:
			return {"prediction": "Cat", "confidence": 1}
from .model_cv import Model as ModelCv
from .model_pillow import Model as ModelPillow



class Model:
	def __init__(self, logging, app_settings):
		self.model_cv = ModelCv(logging, app_settings)
		self.model_pillow = ModelPillow(logging, app_settings)

	def execute(self, file, filename, settings=None):
		print(settings)
		if settings["type"] == "opencv":
			return self.model_cv.execute(file, filename, settings)
		else:
			return self.model_pillow.execute(file, filename, settings)
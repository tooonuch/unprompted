import glob
import random
import os


class Shortcode():
	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Loads an image from the given path and sets it as the initial image for use with img2img."

	def run_atomic(self, pargs, kwargs, context):
		from PIL import Image
		from lib_unprompted.helpers import str_to_pil

		if len(pargs) == 0:
			self.log.info("Getting the current image...")
			return self.Unprompted.current_image()
		else:
			img = str_to_pil(self.Unprompted.parse_advanced(pargs[0], ""))

			if img:
				self.Unprompted.current_image(img)
			# self.Unprompted.shortcode_user_vars["init_images"] = []

			return ""

	def ui(self, gr):
		return [
		    gr.File(label="Image path", file_type="image"),
		]

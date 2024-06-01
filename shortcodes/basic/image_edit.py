class Shortcode():
	def __init__(self, Unprompted):
		import lib_unprompted.helpers as helpers
		self.Unprompted = Unprompted
		self.description = "Performs a wide variety of image operations.<br><br>Wizard UI not yet available. Please refer to the documentation for usage."
		self.destination = "after"
		# Prevent memory address errors
		self.copied_images = []
		self.resample_methods = helpers.pil_resampling_dict

	def run_atomic(self, pargs, kwargs, context):
		import PIL.Image as Image

		image = self.Unprompted.parse_image_kwarg("input")
		if not image:
			return ""

		do_return = self.Unprompted.parse_arg("return", False)
		did_resize = False

		if "copy" in pargs:
			# Note: The copied_images list is used to prevent premature garbage collection
			self.copied_images.append(image.copy())
			image = self.copied_images[-1]

		# Combine pargs and kwargs into a single list, pargs go first
		combined_args = pargs + list(kwargs.keys())

		for this_arg in combined_args:
			if this_arg == "mask":
				mask = self.Unprompted.parse_image_kwarg(this_arg)
				if not mask:
					continue

				if mask.mode != "L":
					mask = mask.convert("L")

				if mask.size != image.size:
					mask = mask.resize(image.size, Image.ANTIALIAS)

				image.putalpha(mask)

				# Clear the RGB information of any pixels with an alpha of 0
				if "keep_masked_rgb" not in pargs:
					pixels = image.load()
					for x in range(image.width):
						for y in range(image.height):
							# Get the current pixel
							r, g, b, a = pixels[x, y]
							if a == 0:
								pixels[x, y] = (0, 0, 0, 0)

			elif this_arg == "mode":
				mode = self.Unprompted.parse_arg("mode", "RGB")
				image = image.convert(mode)
			elif this_arg == "paste":
				paste = self.Unprompted.parse_image_kwarg(this_arg)
				if not paste:
					continue

				if "save" in kwargs:
					paste.save(kwargs["save"] + "_paste.png")

				paste_x = self.Unprompted.parse_arg("paste_x", 0)
				paste_y = self.Unprompted.parse_arg("paste_y", 0)

				# Ensure that paste is RGBA
				if paste.mode != "RGBA":
					paste = paste.convert("RGBA")
				x, y = paste.size

				image.paste(paste, (paste_x, paste_y, x, y), paste)
			elif this_arg == "color_match":
				reference_image = self.Unprompted.parse_image_kwarg(this_arg)
				if not reference_image:
					continue

				if "save" in kwargs:
					reference_image.save(kwargs["save"] + "_reference.png")

				color_match_method = self.Unprompted.parse_arg("color_match_method", "mkl")
				color_match_strength = self.Unprompted.parse_arg("color_match_strength", 1)

				image_cm = self.Unprompted.color_match(reference_image, image, color_match_method, color_match_strength)

				# Blend the color matched image with the original image
				image = Image.blend(image, image_cm, color_match_strength)

				# color_match returns a new image, so we need to prevent it from being garbage collected
				if "return" in pargs:
					self.copied_images.append(image)

				if "save" in kwargs:
					image.save(kwargs["save"] + "_color_match.png")
			elif (this_arg == "width" or this_arg == "height") and not did_resize:
				width = self.Unprompted.parse_arg("width", 0)
				height = self.Unprompted.parse_arg("height", 0)
				min_width = self.Unprompted.parse_arg("min_width", 0)
				min_height = self.Unprompted.parse_arg("min_height", 0)
				unit = self.Unprompted.parse_arg("unit", "px")
				keep_ratio = self.Unprompted.parse_arg("keep_ratio", True)
				technique = self.Unprompted.parse_arg("resize", "scale")
				resample_method = self.resample_methods[self.Unprompted.parse_arg("resample_method", "Lanczos")]
				origin = self.Unprompted.parse_arg("origin", "middle_center")

				if unit == "%":
					width = int(image.width * width)
					height = int(image.height * height)
					min_width = int(image.width * min_width)
					min_height = int(image.height * min_height)

				new_width = image.width
				new_height = image.height

				# If width and height are both specified, resize to those dimensions
				if width and height:
					new_width = width
					new_height = height
				# If only width is specified, resize to that width
				elif width:
					new_width = width
					if keep_ratio:
						new_height = int(image.height * width / image.width)
				# If only height is specified, resize to that height
				elif height:
					new_height = height
					if keep_ratio:
						new_width = int(image.width * height / image.height)

				# Enforce minimum width and height
				if new_height < min_height:
					new_height = min_height
					if keep_ratio:
						new_width = int(image.width * min_height / image.height)
				if new_width < min_width:
					new_width = min_width
					if keep_ratio:
						new_height = int(image.height * min_width / image.width)

				# Resize image if dimensions have changed
				if new_width != image.width or new_height != image.height:
					if technique == "scale":
						image = image.resize((new_width, new_height), resample=resample_method)
					elif technique == "crop":
						# Verify that image is larger than new dimensions
						if image.width < new_width or image.height < new_height:
							self.log.error(f"Image dimensions ({image.width}x{image.height}) are smaller than new dimensions ({new_width}x{new_height}). Skipping crop.")
							continue

						# Determine bounding box based on `origin`
						if origin == "top_left":
							bbox = (0, 0, new_width, new_height)
						elif origin == "top_center":
							bbox = (int((image.width - new_width) / 2), 0, int((image.width - new_width) / 2) + new_width, new_height)
						elif origin == "top_right":
							bbox = (image.width - new_width, 0, image.width, new_height)
						elif origin == "middle_left":
							bbox = (0, int((image.height - new_height) / 2), new_width, int((image.height - new_height) / 2) + new_height)
						elif origin == "middle_center":
							bbox = (int((image.width - new_width) / 2), int((image.height - new_height) / 2), int((image.width - new_width) / 2) + new_width, int((image.height - new_height) / 2) + new_height)
						elif origin == "middle_right":
							bbox = (image.width - new_width, int((image.height - new_height) / 2), image.width, int((image.height - new_height) / 2) + new_height)
						elif origin == "bottom_left":
							bbox = (0, image.height - new_height, new_width, image.height)
						elif origin == "bottom_center":
							bbox = (int((image.width - new_width) / 2), image.height - new_height, int((image.width - new_width) / 2) + new_width, image.height)
						elif origin == "bottom_right":
							bbox = (image.width - new_width, image.height - new_height, image.width, image.height)
						else:
							self.log.error(f"Invalid origin `{origin}`. Skipping crop.")
							continue

						# Crop image
						image = image.crop(bbox)
				else:
					self.log.info("Image dimensions unchanged. Skipping resize.")

				did_resize = True
			elif this_arg == "autotone":
				# from PIL import ImageOps, ImageEnhance
				import numpy as np
				import lib_unprompted.helpers as helpers

				# Reinterpretation of Photoshop's "Auto Tone"
				# Thank you to Gerald Bakker for the following writeup on the algorithm:
				# https://geraldbakker.nl/psnumbers/auto-options.html

				shadows = np.array(helpers.str_to_rgb(self.Unprompted.parse_arg("shadows", "0,0,0")))
				# midtones are only used in other algorithms:
				midtones = helpers.str_to_rgb(self.Unprompted.parse_arg("midtones", "128,128,128"))
				highlights = np.array(helpers.str_to_rgb(self.Unprompted.parse_arg("highlights", "255,255,255")))
				shadow_clip = self.Unprompted.parse_arg("shadow_clip", 0.001)
				highlight_clip = self.Unprompted.parse_arg("highlight_clip", 0.001)

				# Convert the image to a numpy array
				img_array = np.array(image, dtype=np.float32)

				def calculate_adjustment_values(hist, total_pixels, clip_percent):
					clip_threshold = total_pixels * clip_percent
					cumulative_hist = hist.cumsum()

					# Find the first and last indices where the cumulative histogram exceeds the clip thresholds
					lower_bound_idx = np.where(cumulative_hist > clip_threshold)[0][0]
					upper_bound_idx = np.where(cumulative_hist < (total_pixels - clip_threshold))[0][-1]

					return lower_bound_idx, upper_bound_idx

				# Process each channel (R, G, B) separately
				for channel in range(3):
					# Calculate the histogram of the current channel
					hist, _ = np.histogram(img_array[:, :, channel].flatten(), bins=256, range=[0, 255])

					# Total number of pixels
					total_pixels = img_array.shape[0] * img_array.shape[1]

					# Calculate the adjustment values based on clipping percentages
					dark_value, light_value = calculate_adjustment_values(hist, total_pixels, shadow_clip)
					_, upper_light_value = calculate_adjustment_values(hist, total_pixels, highlight_clip)

					# Adjust light_value using upper_light_value for highlights
					light_value = max(light_value, upper_light_value)

					# Avoid division by zero
					if light_value == dark_value:
						continue

					# Scale and clip the channel values
					img_array[:, :, channel] = (img_array[:, :, channel] - dark_value) * (highlights[channel] - shadows[channel]) / (light_value - dark_value) + shadows[channel]
					img_array[:, :, channel] = np.clip(img_array[:, :, channel], 0, 255)

				# Make sure the data type is correct for PIL
				img_array = np.clip(img_array, 0, 255).astype(np.uint8)

				image = Image.fromarray(img_array)
			elif this_arg == "save":
				save_path = kwargs["save"]
				if save_path:
					self.log.debug(f"Saving image to {save_path}...")
					image.save(save_path + ".png")

		if "debug" in pargs:
			for copied_image in self.copied_images:
				copied_image.save("debug_" + str(self.copied_images.index(copied_image)) + ".png")

		if do_return:
			return image
		elif "input" not in kwargs:
			self.Unprompted.current_image(image)
		return ""

	def goodbye(self):
		self.copied_images = []

	def ui(self, gr):
		return [
		    gr.Textbox(label="Path to image (uses current SD image by default) 🡢 input"),
		]

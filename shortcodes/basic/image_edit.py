class Shortcode():

	def __init__(self, Unprompted):
		import lib_unprompted.helpers as helpers
		self.Unprompted = Unprompted
		self.description = "Performs a wide variety of image operations.<br><br>Wizard UI not yet available. Please refer to the documentation for usage."
		self.destination = "after"
		# Prevent memory address errors
		self.copied_images = []
		self.remembered_images = []
		self.resample_methods = helpers.pil_resampling_dict

	def run_atomic(self, pargs, kwargs, context):
		import PIL.Image as Image
		import lib_unprompted.helpers as helpers
		try:
			import torch
		except ImportError:
			self.log.warning("Could not import torch. Some features may not work.")
			pass

		if "unload_cache" in pargs:
			self.remembered_images = []

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
			if this_arg == "upscale":
				from modules import shared

				_models = helpers.ensure(self.Unprompted.parse_arg(this_arg, "None"), list)
				scale = self.Unprompted.parse_arg("scale", 1)
				visibility = self.Unprompted.parse_arg("upscale_alpha", 1.0)
				limit = self.Unprompted.parse_arg("upscale_model_limit", 100)
				keep_res = self.Unprompted.parse_arg("upscale_keep_res", False)

				models = []
				for model in _models:
					for upscaler in shared.sd_upscalers:
						if upscaler.name == model:
							models.append(upscaler)
							break
					if len(models) >= limit:
						self.log.info(f"Upscale model limit satisfied ({limit}). Proceeding...")
						break

				for model in models:
					self.log.info(f"Upscaling {scale}x with {model.name}...")
					image = model.scaler.upscale(image, scale, model.data_path)
					if keep_res:
						image = image.resize(orig_image.size, Image.ANTIALIAS)
			elif this_arg == "mask":
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
			elif this_arg == "remember":
				self.remembered_images.append(image)
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

				# paste_origin = self.Unprompted.parse_arg("paste_origin", "top_left")
				# if paste_origin == "top_right":
				#	paste_x =

				# Ensure that paste is RGBA
				if paste.mode != "RGBA":
					paste = paste.convert("RGBA")
				x, y = paste.size

				image.paste(paste, (paste_x, paste_y, x, y), paste)
			elif this_arg == "color_match":
				if not self.Unprompted.shortcode_install_requirements(f"color_match", ["color-matcher"]):
					continue
				import color_matcher
				from color_matcher.normalizer import Normalizer
				import numpy as np

				reference_image = self.Unprompted.parse_image_kwarg(this_arg)
				if not reference_image:
					continue

				save = ""
				if "save" in kwargs:
					save = kwargs["save"]
					reference_image.save(save + "_reference.png")

				color_match_method = self.Unprompted.parse_arg("color_match_method", "mkl")
				color_match_strength = self.Unprompted.parse_arg("color_match_strength", 1.0)

				# image_cm = self.Unprompted.color_match(reference_image, image, color_match_method, color_match_strength, save=f"{save}_step1.png")
				cm = color_matcher.ColorMatcher()
				img_ref = Normalizer(np.array(reference_image)).uint8_norm()
				img_src = Normalizer(np.array(image)).uint8_norm()

				img_cm = cm.transfer(src=img_src, ref=img_ref, method=color_match_method)
				img_cm = Image.fromarray(Normalizer(img_cm).uint8_norm())

				if "save" in kwargs:
					img_cm.save(kwargs["save"] + "_color_match_step1.png")

				# Mask the color matched image with the original image's alpha channel
				img_cm.putalpha(image.getchannel("A"))

				# Blend the color matched image with the original image
				image = Image.blend(image, img_cm, color_match_strength)

				# color_match returns a new image, so we need to prevent it from being garbage collected
				if "return" in pargs:
					self.copied_images.append(image)

				if "save" in kwargs:
					image.save(kwargs["save"] + "_color_match_step2.png")
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
			elif this_arg == "rotate":
				rotate = self.Unprompted.parse_arg(this_arg, 0)
				resample_method = self.resample_methods[self.Unprompted.parse_arg("resample_method", "Nearest Neighbor")]
				image = image.rotate(rotate, resample=resample_method)
			elif this_arg == "flip_horizontal":
				image = image.transpose(Image.FLIP_LEFT_RIGHT)
			elif this_arg == "flip_vertical":
				image = image.transpose(Image.FLIP_TOP_BOTTOM)
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
			elif this_arg == "add_noise":
				import numpy as np

				img_array = np.array(image)
				row, col, ch = img_array.shape

				amount = self.Unprompted.parse_arg(this_arg, 10)
				monochromatic = self.Unprompted.parse_arg("noise_monochromatic", True)
				noise_type = self.Unprompted.parse_arg("noise_type", "gaussian")
				max_var = self.Unprompted.parse_arg("max_noise_variance", 500)

				if noise_type == "gaussian":
					var = (amount / 100) * max_var  # Calculate variance based on amount
					mean = 0
					sigma = var**0.5
					if monochromatic:
						gauss = np.random.normal(mean, sigma, (row, col, 1))
						gauss = np.repeat(gauss, ch, axis=2)
					else:
						gauss = np.random.normal(mean, sigma, (row, col, ch))
					noisy = img_array + gauss
				elif noise_type == "salt_pepper":
					s_vs_p = 0.5
					amount = amount / 100  # Convert amount to a fraction
					noisy = np.copy(img_array)
					# Salt mode
					num_salt = np.ceil(amount * img_array.size * s_vs_p)
					coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape[:2]]
					if monochromatic:
						noisy[coords[0], coords[1], :] = 1
					else:
						noisy[coords[0], coords[1], np.random.randint(0, ch, int(num_salt))] = 1
					# Pepper mode
					num_pepper = np.ceil(amount * img_array.size * (1.0 - s_vs_p))
					coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape[:2]]
					if monochromatic:
						noisy[coords[0], coords[1], :] = 0
					else:
						noisy[coords[0], coords[1], np.random.randint(0, ch, int(num_pepper))] = 0
				elif noise_type == "poisson":
					vals = len(np.unique(img_array))
					vals = 2**np.ceil(np.log2(vals))  # Determine the appropriate scaling
					if monochromatic:
						noisy = np.random.poisson(img_array[:, :, 0] * vals * (amount / 100.0)) / float(vals)
						noisy = np.repeat(noisy[:, :, np.newaxis], ch, axis=2)
					else:
						noisy = np.random.poisson(img_array * vals * (amount / 100.0)) / float(vals)
				elif noise_type == "speckle":
					gauss = np.random.randn(row, col, ch) * (amount / 100.0)
					if monochromatic:
						gauss = np.random.randn(row, col, 1) * (amount / 100.0)
						gauss = np.repeat(gauss, ch, axis=2)
					noisy = img_array + img_array * gauss
				else:
					self.log.error(f"Invalid noise type `{noise_type}`. Skipping noise addition.")

				# Convert back to PIL
				image = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
			elif this_arg == "brightness":
				from PIL import ImageEnhance

				brightness = self.Unprompted.parse_arg(this_arg, 1.0)
				enhancer = ImageEnhance.Brightness(image)
				image = enhancer.enhance(brightness)
			elif this_arg == "contrast":
				from PIL import ImageEnhance

				contrast = self.Unprompted.parse_arg(this_arg, 1.0)
				enhancer = ImageEnhance.Contrast(image)
				image = enhancer.enhance(contrast)
			elif this_arg == "sharpness":
				from PIL import ImageEnhance

				sharpness = self.Unprompted.parse_arg(this_arg, 1.0)
				enhancer = ImageEnhance.Sharpness(image)
				image = enhancer.enhance(sharpness)
			elif this_arg == "blur":
				from PIL import ImageFilter

				blur_radius = self.Unprompted.parse_arg(this_arg, 1)
				blur_type = self.Unprompted.parse_arg("blur_type", "gaussian")

				if blur_type == "gaussian":
					image = image.filter(ImageFilter.GaussianBlur(blur_radius))
				elif blur_type == "box":
					image = image.filter(ImageFilter.BoxBlur(blur_radius))
				elif blur_type == "unsharp":
					image = image.filter(ImageFilter.UnsharpMask(radius=blur_radius))
			elif this_arg == "intensity":
				from PIL import ImageEnhance

				saturation = self.Unprompted.parse_arg(this_arg, 1.0)
				enhancer = ImageEnhance.Color(image)
				image = enhancer.enhance(saturation)
			elif this_arg == "red" or this_arg == "green" or this_arg == "blue":
				import numpy as np

				shift_value = self.Unprompted.parse_arg(this_arg, 0)
				shift_relative = self.Unprompted.parse_arg("shift_relative", False)

				# Convert the image to a numpy array
				img_array = np.array(image)

				# Separate channels
				r_channel = img_array[:, :, 0].astype(np.float32)
				g_channel = img_array[:, :, 1].astype(np.float32)
				b_channel = img_array[:, :, 2].astype(np.float32)

				if this_arg == "red":
					if shift_relative:
						r_channel = np.clip(r_channel + shift_value, 0, 255)
					else:
						r_channel = np.clip(np.full_like(r_channel, shift_value), 0, 255)
				elif this_arg == "green":
					if shift_relative:
						g_channel = np.clip(g_channel + shift_value, 0, 255)
					else:
						g_channel = np.clip(np.full_like(g_channel, shift_value), 0, 255)
				elif this_arg == "blue":
					if shift_relative:
						b_channel = np.clip(b_channel + shift_value, 0, 255)
					else:
						b_channel = np.clip(np.full_like(b_channel, shift_value), 0, 255)

				# Combine channels back
				img_array[:, :, 0] = r_channel.astype(np.uint8)
				img_array[:, :, 1] = g_channel.astype(np.uint8)
				img_array[:, :, 2] = b_channel.astype(np.uint8)

				# Convert back to PIL
				image = Image.fromarray(img_array)
			elif this_arg == "hue" or this_arg == "value" or this_arg == "saturation":
				import numpy as np
				from PIL import ImageOps

				shift_value = self.Unprompted.parse_arg(this_arg, 0)
				shift_relative = self.Unprompted.parse_arg("shift_relative", False)

				hsv_image = image.convert("HSV")
				hsv_array = np.array(hsv_image)

				# Separate channels
				h_channel = hsv_array[:, :, 0].astype(np.float32)
				s_channel = hsv_array[:, :, 1].astype(np.float32)
				v_channel = hsv_array[:, :, 2].astype(np.float32)

				if this_arg == "hue":
					if shift_relative:
						h_channel = (h_channel + shift_value) % 360
					else:
						h_channel = np.full_like(h_channel, shift_value) % 360
				elif this_arg == "value":
					if shift_relative:
						v_channel = np.clip(v_channel + shift_value, 0, 255)
					else:
						v_channel = np.clip(np.full_like(v_channel, shift_value), 0, 255)
				elif this_arg == "saturation":
					if shift_relative:
						s_channel = np.clip(s_channel + shift_value, 0, 255)
					else:
						s_channel = np.clip(np.full_like(s_channel, shift_value), 0, 255)

				# Combine channels back
				hsv_array[:, :, 0] = h_channel.astype(np.uint8)
				hsv_array[:, :, 1] = s_channel.astype(np.uint8)
				hsv_array[:, :, 2] = v_channel.astype(np.uint8)

				# Convert back to image
				shifted_image = Image.fromarray(hsv_array, mode="HSV")

				# Convert back to original color space
				image = shifted_image.convert("RGB")
			elif this_arg == "colorize":
				import numpy as np
				from PIL import ImageOps

				color = self.Unprompted.parse_arg(this_arg, "255,0,0")
				color = helpers.str_to_rgb(color)

				# Convert the image to a numpy array
				img_array = np.array(image)

				# Convert the RGB image to a grayscale image
				gray = ImageOps.grayscale(image)

				# Colorize the grayscale image
				colorized = ImageOps.colorize(gray, (0, 0, 0), color)
				image = colorized
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
		# ComfyUI support
		elif kwargs["input"] in self.Unprompted.shortcode_user_vars and isinstance(self.Unprompted.shortcode_user_vars[kwargs["input"]], torch.Tensor):
			import lib_unprompted.helpers as helpers
			self.Unprompted.shortcode_user_vars[kwargs["input"]] = helpers.pil_to_tensor(image)

		return ""

	def goodbye(self):
		self.copied_images = []

	def ui(self, gr):
		return [
		    gr.Textbox(label="Path to image (uses current SD image by default) 🡢 input"),
		]

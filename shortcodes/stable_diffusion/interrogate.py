try:
	from modules import shared
except:
	pass  # for unprompted_dry


class Shortcode():
	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Generates a caption for the given image using various technqiues."
		self.model = None
		self.processor = None
		self.last_method = ""
		self.last_model_name = ""
		self.last_caption = ""
		self.last_image = None
		self.tags = None

	def run_atomic(self, pargs, kwargs, context):
		from PIL import Image, ImageChops
		import lib_unprompted.helpers as helpers
		from lib_unprompted.clipxgpt.model.model import Net
		import torch

		image = self.Unprompted.parse_image_kwarg("image")
		if not image:
			return ""

		image = image.convert("RGB")
		# Compare images
		if self.last_image and "bypass_cache" not in pargs:
			diff = ImageChops.difference(image, self.last_image)
			self.last_image = image
			# Images are the same
			if not diff.getbbox():
				self.log.debug("Image is the same as the last one, returning the last caption.")
				return self.last_caption

		self.last_image = image

		method = self.Unprompted.parse_arg("method", "CLIP", arithmetic=False)
		model_name = self.Unprompted.parse_arg("model", "", arithmetic=False)
		prompt = self.Unprompted.parse_arg("text", "")
		question = self.Unprompted.parse_arg("question", "")
		max_tokens = self.Unprompted.parse_arg("max_tokens", 50)
		if question:
			prompt = f"Question: {question} Answer:"

		# Default models per method
		if not model_name:
			if method == "BLIP-2":
				model_name = "Salesforce/blip2-opt-2.7b"
			elif method == "CLIPxGPT":
				model_name = "large_model"
			elif method == "WaifuDiffusion":
				model_name = "SmilingWolf/wd-vit-tagger-v3"

		device = "cuda" if torch.cuda.is_available() else "cpu"
		unload = self.Unprompted.parse_arg("unload", False)

		def get_cached():
			if method != self.last_method or model_name != self.last_model_name or not self.model:
				self.log.info(f"Loading {method} model...")
				return False
			self.log.info(f"Using cached {method} model.")
			return self.model

		if method == "BLIP-2":
			from transformers import AutoProcessor, Blip2ForConditionalGeneration
			model = get_cached()
			if not model:
				#with torch.device(device):
				self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/BLIP-2", low_cpu_mem_usage=True)
				model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, cache_dir=f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/BLIP-2", low_cpu_mem_usage=True)
				model.to(device)

			inputs = self.processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

			generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
			caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

		elif method == "CLIP":
			from modules import shared, lowvram, devices
			self.log.info("Calling the WebUI's standard CLIP interrogator...")
			# caption = shared.interrogator.interrogate(image.convert("RGB"))

			lowvram.send_everything_to_cpu()
			devices.torch_gc()
			shared.interrogator.load()
			caption = shared.interrogator.generate_caption(image.convert("RGB"))
			shared.interrogator.unload()

		elif method == "CLIPxGPT":
			import os
			model = get_cached()
			if not model:
				model = Net(clip_model="openai/clip-vit-large-patch14", text_model="gpt2-medium", ep_len=4, num_layers=5, n_heads=16, forward_expansion=4, dropout=0.08, max_len=40, device=device)
				ckp_file = f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/clipxgpt/{model_name}.pt"
				if not os.path.exists(ckp_file):
					self.log.info("Downloading CLIPxGPT model...")
					helpers.download_file(ckp_file, f"https://drive.google.com/uc?export=download&id=1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG")
				checkpoint = torch.load(ckp_file, map_location=device)
				model.load_state_dict(checkpoint, strict=False)

			model.eval()

			with torch.no_grad():
				caption, _ = model(image, 1.0)  # temperature

		elif method == "WaifuDiffusion":
			from huggingface_hub import hf_hub_download
			from pathlib import Path
			from onnxruntime import InferenceSession
			import pandas as pd
			import numpy as np
			import cv2

			confidence_threshold = self.Unprompted.parse_arg("confidence_threshold", 0.3)

			model = get_cached()

			if not model:
				local_dir = f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/waifudiffusion/"
				# Convert forward slash to underscore
				model_name_sanitized = model_name.replace("/", "_")
				local_dir = f"{local_dir}{model_name_sanitized}/"

				model_path = Path(hf_hub_download(model_name, filename="model.onnx", local_dir=local_dir))
				tags_path = Path(hf_hub_download(model_name, filename="selected_tags.csv", local_dir=local_dir))
				providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

				model = InferenceSession(str(model_path), providers=providers)
				self.tags = pd.read_csv(tags_path)

			# code for converting the image and running the model is taken from the link below
			# thanks, SmilingWolf!
			# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py

			# convert an image to fit the model
			_, height, _, _ = model.get_inputs()[0].shape

			# alpha to white
			image = image.convert("RGBA")
			new_image = Image.new("RGBA", image.size, "WHITE")
			new_image.paste(image, mask=image)
			image = new_image.convert("RGB")
			image = np.asarray(image)

			# PIL RGB to OpenCV BGR
			image = image[:, :, ::-1]

			def make_square(img, target_size):
				old_size = img.shape[:2]
				desired_size = max(old_size)
				desired_size = max(desired_size, target_size)

				delta_w = desired_size - old_size[1]
				delta_h = desired_size - old_size[0]
				top, bottom = delta_h // 2, delta_h - (delta_h // 2)
				left, right = delta_w // 2, delta_w - (delta_w // 2)

				color = [255, 255, 255]
				new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
				return new_im

			def smart_resize(img, size):
				# Assumes the image has already gone through make_square
				if img.shape[0] > size:
					img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
				elif img.shape[0] < size:
					img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
				return img

			image = make_square(image, height)
			image = smart_resize(image, height)
			image = image.astype(np.float32)
			image = np.expand_dims(image, 0)

			# evaluate model
			input_name = model.get_inputs()[0].name
			label_name = model.get_outputs()[0].name
			confidents = model.run([label_name], {input_name: image})[0]

			tags = self.tags[:][["name"]]
			tags["confidents"] = confidents[0]

			# first 4 items are for rating (general, sensitive, questionable, explicit)
			ratings = dict(tags[:4].values)

			# rest are regular tags
			caption = dict(tags[4:].values)

			blacklist_tags = self.Unprompted.parse_arg("blacklist_tags", "")
			if isinstance(blacklist_tags, str):
				blacklist_tags = blacklist_tags.split(self.Unprompted.Config.syntax.delimiter)
			# Replace spaces with underscores
			blacklist_tags = [tag.replace(" ", "_") for tag in blacklist_tags]

			# self.Unprompted.log.debug(blacklist_tags)

			# Create a string with the tags that meet the minimum confidence
			caption = [f"{tag}" for tag, conf in caption.items() if conf >= confidence_threshold and tag not in blacklist_tags]
			caption = ", ".join(caption)

		self.log.debug(f"Caption method {method} returned: {caption}")

		# Cache handling
		self.last_method = method
		self.last_model_name = model_name
		self.last_caption = caption
		if unload:
			self.model = None
			self.processor = None
		elif method != "CLIP":
			self.model = model

		return caption

	def ui(self, gr):
		return [
		    gr.Image(label="Image to perform interrogation on (defaults to SD output) 🡢 image", type="filepath", interactive=True),
		    gr.Radio(label="Interrogation method 🡢 method", value="CLIP", choices=["BLIP-2", "CLIP", "CLIPxGPT"], info="Note: The other methods require large model downloads!"),
		    gr.Text(label="Model name 🡢 model", value="", info="Accepts Hugging Face model strings"),
		    gr.Text(label="Context 🡢 context", value="", info="For BLIP-2, provide contextual information for the interrogation."),
		    gr.Text(label="Question 🡢 question", value="", info="For BLIP-2, ask a question about the image."),
		    gr.Slider(label="Max Tokens 🡢 max_tokens", value=50, min=1, max=100, step=1, info="For BLIP-2, the maximum number of tokens to generate."),
		]

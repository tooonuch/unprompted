class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.image_mask = None
		self.show = False
		self.description = "Creates an image mask from the content for use with inpainting."
		try:
			del self.cached_model
			del self.cached_transform
			del self.cached_model_method
			del self.cached_predictor
		except:
			pass
		self.cached_model = -1
		self.cached_model_b = -1
		self.cached_transform = -1
		self.cached_model_method = ""
		self.cached_predictor = -1
		self.cached_preprocess = -1
		self.copied_images = []

	def run_block(self, pargs, kwargs, context, content):
		self.Unprompted.shortcode_install_requirements("", ["segment_anything"])
		from PIL import ImageChops, Image, ImageOps
		import os
		import torch
		from torchvision import transforms
		from matplotlib import pyplot as plt
		import cv2
		import numpy
		import lib_unprompted.helpers as helpers
		try:
			from modules.images import flatten
			from modules.shared import opts
		except:
			pass
		from torchvision.transforms.functional import pil_to_tensor, to_pil_image

		if "input" in kwargs:
			self.init_image = self.Unprompted.shortcode_user_vars[kwargs["input"]]
			if isinstance(self.init_image, torch.Tensor):
				self.init_image = helpers.tensor_to_pil(self.init_image)
			else:
				self.init_image = self.init_image.copy()
		elif "init_images" not in self.Unprompted.shortcode_user_vars:
			self.log.error("No init_images found...")
			return
		else:
			self.init_image = self.Unprompted.shortcode_user_vars["init_images"][0].copy()

		method = self.Unprompted.parse_arg("method", "clipseg")
		mask_sort_method = self.Unprompted.parse_arg("mask_sort_method", "random")
		reverse_mask_sort = self.Unprompted.parse_arg("reverse_mask_sort", False)
		mask_index = self.Unprompted.parse_arg("mask_index", 0)

		if method == "clipseg":
			mask_width = 512
			mask_height = 512
		else:
			mask_width = self.init_image.size[0]
			mask_height = self.init_image.size[1]

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if device == "cuda":
			torch.cuda.empty_cache()

		if "stamp" in kwargs:
			stamps = self.Unprompted.parse_arg("stamp", "").split(self.Unprompted.Config.syntax.delimiter)

			stamp_x = int(float(self.Unprompted.parse_advanced(kwargs["stamp_x"], context))) if "stamp_x" in kwargs else 0
			stamp_y = int(float(self.Unprompted.parse_advanced(kwargs["stamp_y"], context))) if "stamp_y" in kwargs else 0
			stamp_x_orig = stamp_x
			stamp_y_orig = stamp_y
			stamp_method = self.Unprompted.parse_advanced(kwargs["stamp_method"], context) if "stamp_method" in kwargs else "stretch"

			for stamp in stamps:
				# Checks for file in images/stamps, otherwise assumes absolute path
				stamp_path = f"{self.Unprompted.base_dir}/images/stamps/{stamp}.png"
				if not os.path.exists(stamp_path):
					stamp_path = stamp
				if not os.path.exists(stamp_path):
					self.log.error(f"Stamp not found: {stamp_path}")
					continue

				stamp_img = Image.open(stamp_path).convert("RGBA")

				if stamp_method == "stretch":
					stamp_img = stamp_img.resize((self.init_image.size[0], self.init_image.size[1]))
				elif stamp_method == "center":
					stamp_x = stamp_x_orig + int((mask_width - stamp_img.size[0]) / 2)
					stamp_y = stamp_y_orig + int((mask_height - stamp_img.size[1]) / 2)

				stamp_blur = int(float(self.Unprompted.parse_advanced(kwargs["stamp_blur"], context))) if "stamp_blur" in kwargs else 0
				if stamp_blur:
					from PIL import ImageFilter
					blur = ImageFilter.GaussianBlur(stamp_blur)
					stamp_img = stamp_img.filter(blur)

				self.init_image.paste(stamp_img, (stamp_x, stamp_y), stamp_img)

		brush_mask_mode = self.Unprompted.parse_arg("mode", "add")
		self.show = True if "show" in pargs else False

		mask_blur = self.Unprompted.parse_arg("blur", 0)

		self.legacy_weights = True if "legacy_weights" in pargs else False
		smoothing = self.Unprompted.parse_arg("smoothing", 20)
		smoothing_kernel = None
		if smoothing > 0:
			smoothing_kernel = numpy.ones((smoothing, smoothing), numpy.float32) / (smoothing * smoothing)

		neg_smoothing = self.Unprompted.parse_arg("neg_smoothing", 20)
		neg_smoothing_kernel = None
		if neg_smoothing > 0:
			neg_smoothing_kernel = numpy.ones((neg_smoothing, neg_smoothing), numpy.float32) / (neg_smoothing * neg_smoothing)

		# Pad the mask by applying a dilation or erosion
		mask_padding = int(self.Unprompted.parse_arg("padding", 0))
		neg_mask_padding = int(self.Unprompted.parse_arg("neg_padding", 0))
		padding_dilation_kernel = None
		if (mask_padding != 0):
			padding_dilation_kernel = numpy.ones((abs(mask_padding), abs(mask_padding)), numpy.uint8)

		neg_padding_dilation_kernel = None
		if (neg_mask_padding != 0):
			neg_padding_dilation_kernel = numpy.ones((abs(neg_mask_padding), abs(neg_mask_padding)), numpy.uint8)

		prompts = content.split(self.Unprompted.Config.syntax.delimiter)
		prompt_parts = len(prompts)

		if "negative_mask" in kwargs:
			neg_parsed = self.Unprompted.parse_arg("negative_mask", "")
			if len(neg_parsed) < 1:
				negative_prompts = None
			else:
				negative_prompts = neg_parsed
				negative_prompt_parts = len(negative_prompts)
		else:
			negative_prompts = None

		mask_precision = min(255, self.Unprompted.parse_arg("precision", 100))
		neg_mask_precision = min(255, self.Unprompted.parse_arg("neg_precision", 100))

		def overlay_mask_part(img_a, img_b, mode):
			if (mode == "discard"):
				img_a = ImageChops.darker(img_a, img_b)
			else:
				img_a = ImageChops.lighter(img_a, img_b)
			return (img_a)

		def gray_to_pil(img):
			return (Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)))

		def process_mask_parts(masks, mode, final_img=None, mask_precision=100, mask_padding=0, padding_dilation_kernel=None, smoothing_kernel=None):
			for i, mask in enumerate(masks):

				filename = f"mask_{mode}_{i}.png"

				if method == "clipseg":
					mask_np = torch.sigmoid(mask[0]).cpu().numpy()
					img = (mask_np * 255).astype(numpy.uint8)
				# Ensure the mask is a numpy array
				elif isinstance(mask, torch.Tensor):
					mask_np = mask.detach().cpu().numpy()
					# Convert the mask to an image
					img = (mask_np * 255).astype(numpy.uint8)
				# Catch any other types of masks
				elif not isinstance(mask, numpy.ndarray):
					img = numpy.array(mask)
				else:
					img = mask

				if padding_dilation_kernel is not None:
					if (mask_padding > 0):
						img = cv2.dilate(img, padding_dilation_kernel, iterations=1)
					else:
						img = cv2.erode(img, padding_dilation_kernel, iterations=1)
				if smoothing_kernel is not None:
					img = cv2.filter2D(img, -1, smoothing_kernel)

				if "debug" in pargs:
					Image.fromarray(img).save(filename)

				# Check if we need to convert the mask to a grayscale:
				if len(img.shape) > 2:
					gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				else:
					gray_image = img

				if "debug" in pargs:
					Image.fromarray(gray_image).save("mask_gray_test.png")
				(thresh, bw_image) = cv2.threshold(gray_image, mask_precision, 255, cv2.THRESH_BINARY)

				if (mode == "discard"):
					bw_image = numpy.invert(bw_image)

				# overlay mask parts
				bw_image = gray_to_pil(bw_image)
				if (i > 0 or final_img is not None):
					bw_image = overlay_mask_part(bw_image, final_img, mode)

				final_img = bw_image
			return (final_img)

		def get_mask():
			import torch

			preds = []
			negative_preds = []
			if self.Unprompted.webui != "comfy":
				image_pil = flatten(self.init_image, opts.img2img_background_color)

			if method == "panoptic_sam":
				if not self.Unprompted.shortcode_install_requirements("panoptic_sam", ["timm", "addict", "supervision", "pycocotools"]):
					return False

				try:
					import random
					import numpy as np
					import matplotlib.pyplot as plt
					import torch
					from torch import nn
					import torch.nn.functional as F
					from scipy import ndimage
					from PIL import Image
					from huggingface_hub import hf_hub_download
					# from segments import SegmentsClient
					# from segments.export import colorize
					# from segments.utils import bitmap2file
					# from getpass import getpass

					# Grounding DINO
					import sys
					# Fix imports in GroundingDINO library
					# print(os.path.join(self.Unprompted.base_dir, "lib_unprompted/GroundingDINO"))
					sys.path.append(os.path.join(self.Unprompted.base_dir, "lib_unprompted/GroundingDINO"))
					import lib_unprompted.GroundingDINO.groundingdino.datasets.transforms as T
					from lib_unprompted.GroundingDINO.groundingdino.models import build_model
					from lib_unprompted.GroundingDINO.groundingdino.util import box_ops
					from lib_unprompted.GroundingDINO.groundingdino.util.slconfig import SLConfig
					from lib_unprompted.GroundingDINO.groundingdino.util.utils import clean_state_dict
					from lib_unprompted.GroundingDINO.groundingdino.util.inference import annotate, predict

					# segment anything
					from segment_anything import build_sam, SamPredictor

					# CLIPSeg
					from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
				except:
					self.log.error("Failed to import required libraries for `panoptic_sam.` Please open an issue on GitHub if this persists and try switching to `clipseg` in the meantime.")
					return False

				# if device != "cpu":
				# try:
				# 	from lib_unprompted.GroundingDINO.groundingdino import _C
				# except:
				# 	self.log.warning("Failed to load custom C++ ops. Running on CPU mode Only in groundingdino!")

				if self.cached_model == -1 or self.cached_model_method != method:
					# Use this command for evaluate the Grounding DINO model
					# Or you can download the model by yourself
					ckpt_repo_id = "ShilongLiu/GroundingDINO"
					ckpt_filename = "groundingdino_swinb_cogcoor.pth"
					ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

					def load_model_hf(repo_id, filename, ckpt_config_filename, device):
						cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename, cache_dir=f"{self.Unprompted.base_dir}/models/groundingdino")

						args = SLConfig.fromfile(cache_config_file)
						model = build_model(args)
						args.device = device

						cache_file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=f"{self.Unprompted.base_dir}/models/groundingdino")
						checkpoint = torch.load(cache_file, map_location="cpu")
						log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
						self.log.info("Model loaded from {} \n => {}".format(cache_file, log))
						_ = model.eval()
						model.to(device)
						return model

					groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)
					if device == "cuda":
						groundingdino_model = groundingdino_model.to('cuda:0')
					self.cached_model = groundingdino_model

					# Download SAM
					sam_model_dir = f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/segment_anything"
					os.makedirs(sam_model_dir, exist_ok=True)
					sam_checkpoint = "sam_vit_h_4b8939.pth"
					sam_file = f"{sam_model_dir}/{sam_checkpoint}"
					# Download model weights if we don't have them yet
					if not os.path.exists(sam_file):
						self.log.info("Downloading SAM model weights...")
						helpers.download_file(sam_file, f"https://dl.fbaipublicfiles.com/segment_anything/{sam_checkpoint}")

					sam = build_sam(checkpoint=sam_file)
					sam.to(device=device)
					sam_predictor = SamPredictor(sam)
					self.cached_predictor = sam_predictor

					# Download CLIPSEG

					clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=f"{self.Unprompted.base_dir}/models/clipseg")
					clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined", cache_dir=f"{self.Unprompted.base_dir}/models/clipseg")
					clipseg_model.to(device)
					self.cached_model_b = clipseg_model
					self.cached_preprocess = clipseg_processor

					self.cached_model_method = method
				else:
					self.log.info("Using cached panoptic_sam models...")
					groundingdino_model = self.cached_model
					sam_predictor = self.cached_predictor
					clipseg_processor = self.cached_preprocess
					clipseg_model = self.cached_model_b

				# Helper methods for panoptic_sam
				def load_image_for_dino(image):
					transform = T.Compose([
					    T.RandomResize([800], max_size=1333),
					    T.ToTensor(),
					    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
					])
					dino_image, _ = transform(image, None)
					return dino_image

				def dino_detection(
				    model,
				    image,
				    image_array,
				    category_names,
				    category_name_to_id,
				    box_threshold,
				    text_threshold,
				    device,
				    visualize=False,
				):
					detection_prompt = " . ".join(category_names)
					dino_image = load_image_for_dino(image)
					dino_image = dino_image.to(device)
					with torch.no_grad():
						boxes, logits, phrases = predict(
						    model=model,
						    image=dino_image,
						    caption=detection_prompt,
						    box_threshold=box_threshold,
						    text_threshold=text_threshold,
						    device=device,
						    # remove_combined=True,
						)
					phrases = [phrase for phrase in phrases if phrase]
					category_ids = [category_name_to_id[phrase] for phrase in phrases]

					if visualize:
						annotated_frame = annotate(image_source=image_array, boxes=boxes, logits=logits, phrases=phrases)
						annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
						visualization = Image.fromarray(annotated_frame)
						return boxes, category_ids, visualization
					else:
						return boxes, category_ids, phrases

				def sam_masks_from_dino_boxes(predictor, image_array, boxes, device):
					# box: normalized box xywh -> unnormalized xyxy
					H, W, _ = image_array.shape
					boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
					transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, image_array.shape[:2]).to(device)
					thing_masks, _, _ = predictor.predict_torch(
					    point_coords=None,
					    point_labels=None,
					    boxes=transformed_boxes,
					    multimask_output=False,
					)
					return thing_masks

				def preds_to_semantic_inds(preds, threshold):
					flat_preds = preds.reshape((preds.shape[0], -1))
					# Initialize a dummy "unlabeled" mask with the threshold
					flat_preds_with_treshold = torch.full((preds.shape[0] + 1, flat_preds.shape[-1]), threshold)
					flat_preds_with_treshold[1:preds.shape[0] + 1, :] = flat_preds

					# Get the top mask index for each pixel
					semantic_inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape((preds.shape[-2], preds.shape[-1]))

					return semantic_inds

				def clipseg_segmentation(processor, model, image, category_names, background_threshold, device):
					inputs = processor(
					    text=category_names,
					    images=[image] * len(category_names),
					    padding="max_length",
					    return_tensors="pt",
					).to(device)
					with torch.no_grad():
						outputs = model(**inputs)
					logits = outputs.logits
					if len(logits.shape) == 2:
						logits = logits.unsqueeze(0)
					# resize the outputs
					upscaled_logits = nn.functional.interpolate(
					    logits.unsqueeze(1),
					    size=(image.size[1], image.size[0]),
					    mode="bilinear",
					)
					preds = torch.sigmoid(upscaled_logits.squeeze(dim=1))
					semantic_inds = preds_to_semantic_inds(preds, background_threshold)
					return preds, semantic_inds

				def semantic_inds_to_shrunken_bool_masks(semantic_inds, shrink_kernel_size, num_categories):
					shrink_kernel = np.ones((shrink_kernel_size, shrink_kernel_size))

					bool_masks = torch.zeros((num_categories, *semantic_inds.shape), dtype=bool)
					for category in range(num_categories):
						binary_mask = semantic_inds == category
						shrunken_binary_mask_array = (ndimage.binary_erosion(binary_mask.numpy(), structure=shrink_kernel) if shrink_kernel_size > 0 else binary_mask.numpy())
						bool_masks[category] = torch.from_numpy(shrunken_binary_mask_array)

					return bool_masks

				def clip_and_shrink_preds(semantic_inds, preds, shrink_kernel_size, num_categories):
					# convert semantic_inds to shrunken bool masks
					bool_masks = semantic_inds_to_shrunken_bool_masks(semantic_inds, shrink_kernel_size, num_categories).to(preds.device)

					sizes = [torch.sum(bool_masks[i].int()).item() for i in range(1, bool_masks.size(0))]
					max_size = max(sizes)
					relative_sizes = [size / max_size for size in sizes] if max_size > 0 else sizes

					# use bool masks to clip preds
					clipped_preds = torch.zeros_like(preds)
					for i in range(1, bool_masks.size(0)):
						float_mask = bool_masks[i].float()
						clipped_preds[i - 1] = preds[i - 1] * float_mask

					return clipped_preds, relative_sizes

				def sample_points_based_on_preds(preds, N):
					height, width = preds.shape
					weights = preds.ravel()
					indices = np.arange(height * width)

					# Randomly sample N indices based on the weights
					sampled_indices = random.choices(indices, weights=weights, k=N)

					# Convert the sampled indices into (col, row) coordinates
					sampled_points = [(index % width, index // width) for index in sampled_indices]

					return sampled_points

				def upsample_pred(pred, image_source):
					pred = pred.unsqueeze(dim=0)
					original_height = image_source.shape[0]
					original_width = image_source.shape[1]

					larger_dim = max(original_height, original_width)
					aspect_ratio = original_height / original_width

					# upsample the tensor to the larger dimension
					upsampled_tensor = F.interpolate(pred, size=(larger_dim, larger_dim), mode="bilinear", align_corners=False)

					# remove the padding (at the end) to get the original image resolution
					if original_height > original_width:
						target_width = int(upsampled_tensor.shape[3] * aspect_ratio)
						upsampled_tensor = upsampled_tensor[:, :, :, :target_width]
					else:
						target_height = int(upsampled_tensor.shape[2] * aspect_ratio)
						upsampled_tensor = upsampled_tensor[:, :, :target_height, :]
					return upsampled_tensor.squeeze(dim=1)

				def sam_mask_from_points(predictor, image_array, points):
					points_array = np.array(points)
					# we only sample positive points, so labels are all 1
					points_labels = np.ones(len(points))
					# we don't use predict_torch here cause it didn't seem to work...
					masks, scores, logits = predictor.predict(
					    point_coords=points_array,
					    point_labels=points_labels,
					)
					# max over the 3 segmentation levels
					total_pred = torch.max(torch.sigmoid(torch.tensor(logits)), dim=0)[0].unsqueeze(dim=0)
					# logits are 256x256 -> upsample back to image shape
					upsampled_pred = upsample_pred(total_pred, image_array)
					return upsampled_pred

				# main mask generation function for panoptic_sam
				def generate_panoptic_mask(
				        image,
				        thing_category_names,
				        stuff_category_names,
				        category_name_to_id,
				        dino_model,
				        sam_predictor,
				        clipseg_processor,
				        clipseg_model,
				        device,
				        dino_box_threshold=0.3 * (mask_precision / 100),
				        dino_text_threshold=0.25,
				        segmentation_background_threshold=0.1,
				        shrink_kernel_size=20,
				        num_samples_factor=1000,
				):
					image = image.convert("RGB")
					image_array = np.asarray(image)

					# compute SAM image embedding
					sam_predictor.set_image(image_array)

					# detect boxes for "thing" categories using Grounding DINO
					thing_category_ids = []
					thing_masks = []
					thing_boxes = []
					if len(thing_category_names) > 0:
						thing_boxes, thing_category_ids, _ = dino_detection(
						    dino_model,
						    image,
						    image_array,
						    thing_category_names,
						    category_name_to_id,
						    dino_box_threshold,
						    dino_text_threshold,
						    device,
						)
						if len(thing_boxes) > 0:
							# get segmentation masks for the thing boxes
							thing_masks = sam_masks_from_dino_boxes(sam_predictor, image_array, thing_boxes, device)

					if len(stuff_category_names) > 0:
						# get rough segmentation masks for "stuff" categories using CLIPSeg
						clipseg_preds, clipseg_semantic_inds = clipseg_segmentation(
						    clipseg_processor,
						    clipseg_model,
						    image,
						    stuff_category_names,
						    segmentation_background_threshold,
						    device,
						)
						# remove things from stuff masks
						clipseg_semantic_inds_without_things = clipseg_semantic_inds.clone()
						if len(thing_boxes) > 0:
							combined_things_mask = torch.any(thing_masks, dim=0)
							clipseg_semantic_inds_without_things[combined_things_mask[0]] = 0
						# clip CLIPSeg preds based on non-overlapping semantic segmentation inds (+ optionally shrink the mask of each category)
						# also returns the relative size of each category
						clipsed_clipped_preds, relative_sizes = clip_and_shrink_preds(
						    clipseg_semantic_inds_without_things,
						    clipseg_preds,
						    shrink_kernel_size,
						    len(stuff_category_names) + 1,
						)
						# get finer segmentation masks for the "stuff" categories using SAM
						sam_preds = torch.zeros_like(clipsed_clipped_preds)
						for i in range(clipsed_clipped_preds.shape[0]):
							clipseg_pred = clipsed_clipped_preds[i]
							# for each "stuff" category, sample points in the rough segmentation mask
							num_samples = int(relative_sizes[i] * num_samples_factor)
							if num_samples == 0:
								continue
							points = sample_points_based_on_preds(clipseg_pred.cpu().numpy(), num_samples)
							if len(points) == 0:
								continue
							# use SAM to get mask for points
							pred = sam_mask_from_points(sam_predictor, image_array, points)
							sam_preds[i] = pred
						sam_semantic_inds = preds_to_semantic_inds(sam_preds, segmentation_background_threshold)

					# combine the thing inds and the stuff inds into panoptic inds
					panoptic_inds = (sam_semantic_inds.clone() if len(stuff_category_names) > 0 else torch.zeros(image_array.shape[0], image_array.shape[1], dtype=torch.long))
					ind = len(stuff_category_names) + 1
					for thing_mask in thing_masks:
						# overlay thing mask on panoptic inds
						panoptic_inds[thing_mask.squeeze(dim=0)] = ind
						ind += 1

					return panoptic_inds, thing_category_ids

				# Taken from the segments-ai library
				from io import BytesIO
				import numpy.typing as npt

				def bitmap2file(
				    bitmap: npt.NDArray[np.uint32],
				    is_segmentation_bitmap: bool = True,
				):
					"""Convert a label bitmap to a file with the proper format.

					Args:
						bitmap: A :class:`numpy.ndarray` with :class:`numpy.uint32` dtype where each unique value represents an instance id.
						is_segmentation_bitmap: If this is a segmentation bitmap. Defaults to :obj:`True`.

					Returns:
						A file object.

					Raises:
						:exc:`ValueError`: If the ``dtype`` is not :class:`np.uint32` or :class:`np.uint8`.
						:exc:`ValueError`: If the bitmap is not a segmentation bitmap.
					"""
					bitmaps = []

					# Convert bitmap to np.uint32, if it is not already
					if bitmap.dtype == "uint32":
						pass
					elif bitmap.dtype == "uint8":
						bitmap = np.uint32(bitmap)
					else:
						raise ValueError("Only `np.ndarrays` with `np.uint32` data type can be used.")

					if is_segmentation_bitmap:
						unique_ids = np.unique(bitmap)[1:]  # First index is a composite
						for id in unique_ids:
							mask = np.where(bitmap == id, 255, 0).astype('uint8')
							mask_rgba = np.stack([mask] * 4, axis=-1)  # Create an RGBA image
							mask_rgba[..., 3] = (mask > 0) * 255  # Set alpha channel
							f = BytesIO()
							Image.fromarray(mask_rgba).save(f, "PNG")
							f.seek(0)
							bitmaps.append(f)
					else:
						raise ValueError("Only segmentation bitmaps can be used.")

					return bitmaps

				def run_panoptic(words):
					words = helpers.ensure(words, list)
					these_preds = []
					# thing_category_names = ["car", "person", "bus"]
					stuff_category_names = []  # ["building", "road", "sky", "trees", "sidewalk"]
					category_names = words + stuff_category_names
					category_name_to_id = {category_name: i for i, category_name in enumerate(category_names)}
					panoptic_inds, thing_category_ids = generate_panoptic_mask(
					    image_pil,
					    words,
					    stuff_category_names,  #
					    category_name_to_id,
					    groundingdino_model,
					    sam_predictor,
					    clipseg_processor,
					    clipseg_model,
					    device,
					)

					panoptic_inds_array = panoptic_inds.numpy().astype(np.uint32)
					bitmap_files = bitmap2file(panoptic_inds_array, is_segmentation_bitmap=True)
					for i, bitmap_file in enumerate(bitmap_files):
						mask = Image.open(bitmap_file)
						these_preds.append(mask)

					return these_preds

				preds = run_panoptic(prompts)
				if negative_prompts:
					negative_preds = run_panoptic(negative_prompts)

			elif method == "fastsam":
				self.Unprompted.shortcode_install_requirements(f"fastSAM", ["ultralytics"])
				from ultralytics import YOLO
				from lib_unprompted.fastsam.utils import tools
				import clip
				import numpy as np
				import cv2

				fastsam_better_quality = self.Unprompted.parse_arg("fastsam_better_quality", True)

				fastsam_retina = self.Unprompted.parse_arg("fastsam_retina", False)
				fastsam_model_type = "YOLOv8s"
				fastsam_iou = self.Unprompted.parse_arg("fastsam_iou", 0.9)

				fastsam_conf = self.Unprompted.parse_arg("fastsam_conf", 0.4)
				fastsam_max_det = self.Unprompted.parse_arg("fastsam_max_det", 100)
				fastsam_size = self.Unprompted.parse_arg("fastsam_size", 1024)

				def fast_show_mask(
				    annotation,
				    random_color=False,
				    retinamask=True,
				    target_height=960,
				    target_width=960,
				):
					msak_sum = annotation.shape[0]
					height = annotation.shape[1]
					weight = annotation.shape[2]
					if device != "cpu":
						areas = torch.sum(annotation, dim=(1, 2))
						sorted_indices = torch.argsort(areas, descending=False)
						annotation = annotation[sorted_indices]
						# 找每个位置第一个非零值下标
						index = (annotation != 0).to(torch.long).argmax(dim=0)
						if random_color == True:
							color = torch.rand((msak_sum, 1, 1, 3)).to(annotation.device)
						else:
							color = torch.ones((msak_sum, 1, 1, 3)).to(annotation.device) * torch.tensor([30 / 255, 144 / 255, 255 / 255]).to(annotation.device)
						transparency = torch.ones((msak_sum, 1, 1, 1)).to(annotation.device) * 0.6
						visual = torch.cat([color, transparency], dim=-1)
						mask_image = torch.unsqueeze(annotation, -1) * visual
						# 按index取数，index指每个位置选哪个batch的数，把mask_image转成一个batch的形式
						show = torch.zeros((height, weight, 4)).to(annotation.device)
						h_indices, w_indices = torch.meshgrid(torch.arange(height), torch.arange(weight), indexing="ij")
					else:
						# 将annotation 按照面积 排序
						areas = np.sum(annotation, axis=(1, 2))
						sorted_indices = np.argsort(areas)
						annotation = annotation[sorted_indices]
						index = (annotation != 0).argmax(axis=0)
						if random_color == True:
							color = np.random.random((msak_sum, 1, 1, 3))
						else:
							color = np.ones((msak_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
						transparency = np.ones((msak_sum, 1, 1, 1)) * 0.6
						visual = np.concatenate([color, transparency], axis=-1)
						mask_image = np.expand_dims(annotation, -1) * visual
						show = np.zeros((height, weight, 4))
						h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing="ij")
					indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
					# 使用向量化索引更新show的值
					show[h_indices, w_indices, :] = mask_image[indices]
					show_cpu = show.cpu().numpy()
					if retinamask == False:
						show_cpu = cv2.resize(show_cpu, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
					return show_cpu

				sam_model_dir = f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/fastsam"
				os.makedirs(sam_model_dir, exist_ok=True)
				if fastsam_model_type == "YOLOv8x":
					sam_filename = "FastSAM-x.pt"
				else:
					sam_filename = "FastSAM-s.pt"
				sam_file = f"{sam_model_dir}/{sam_filename}"

				# Download model weights if we don't have them yet
				if not os.path.exists(sam_file):
					self.log.info("Downloading FastSAM model weights...")
					# TODO: The YOLOv8x model is too big to download directly from Gdrive, find another host that supports it. Not particularly urgent as the difference in quality between the two models seems negligible...
					if fastsam_model_type == "YOLOv8x":
						helpers.download_file(sam_file, "https://drive.google.com/uc?export=download&id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv")
					else:
						helpers.download_file(sam_file, f"https://drive.google.com/uc?export=download&id=10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV")

				if self.cached_model == -1 or self.cached_model_method != method:
					model = YOLO(sam_file)
					clip_model, preprocess = clip.load("ViT-B/32", device=device)

					# Cache for future runs
					self.cached_model = model
					self.cached_clip_model = clip_model
					self.cached_preprocess = preprocess
					self.cached_model_method = method
				else:
					self.log.info("Using cached FastSAM model.")
					model = self.cached_model
					preprocess = self.cached_preprocess
					clip_model = self.cached_clip_model

				results = model(image_pil, imgsz=fastsam_size, device=device, retina_masks=fastsam_retina, iou=fastsam_iou, conf=fastsam_conf, max_det=fastsam_max_det)
				results = tools.format_results(results[0], 0)
				cropped_boxes, cropped_images, not_crop, filter_id, annot = tools.crop_image(results, image_pil)

				image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)
				original_h = image.shape[0]
				original_w = image.shape[1]

				def run_fastsam(text_prompts):
					scores = tools.retriev(clip_model, preprocess, cropped_boxes, text_prompts, device=device)
					max_idx = scores.argsort()
					max_idx = max_idx[-1]
					max_idx += sum(np.array(filter_id) <= int(max_idx))
					annotations = annot[max_idx]["segmentation"]
					annotations = np.array([annotations])

					if isinstance(annotations[0], dict):
						annotations = [annotation["segmentation"] for annotation in annotations]

					if fastsam_better_quality == True:
						if isinstance(annotations[0], torch.Tensor):
							annotations = np.array(annotations.cpu())
						for i, mask in enumerate(annotations):
							mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
							annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
					# from PIL import Image
					# preds = Image.fromarray((preds * 255).astype(np.uint8))
					if device == "cpu":
						annotations = np.array(annotations)
					else:
						if isinstance(annotations[0], np.ndarray):
							annotations = torch.from_numpy(annotations)

					return fast_show_mask(
					    annotations,
					    random_color=False,
					    retinamask=fastsam_retina,
					    target_height=original_h,
					    target_width=original_w,
					)

				preds.append(run_fastsam(prompts))
				if negative_prompts:
					negative_preds.append(run_fastsam(negative_prompts))

			elif method == "clip_surgery":
				from lib_unprompted import clip_surgery as clip
				import cv2
				import numpy as np
				from PIL import Image
				from matplotlib import pyplot as plt
				from torchvision.transforms import Compose, Resize, ToTensor, Normalize
				from torchvision.transforms import InterpolationMode
				BICUBIC = InterpolationMode.BICUBIC
				from segment_anything import sam_model_registry, SamPredictor

				# default imagenet redundant features
				redundants = [
				    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
				    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.',
				    'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.'
				]

				if "redundant_features" in kwargs:
					redundants.extend(kwargs["redundant_features"].split(self.Unprompted.Config.syntax.delimiter))
				self.bypass_sam = True if "bypass_sam" in pargs else False

				### Init CLIP and data
				if self.cached_model == -1 or self.cached_model_method != method:
					model, preprocess = clip.load("CS-ViT-B/16", device=device)
					model.eval()
					# Cache for future runs
					self.cached_model = model
					self.cached_transform = preprocess
				else:
					self.log.info("Using cached model(s) for CLIP_Surgery method")
					model = self.cached_model
					preprocess = self.cached_transform

				image = preprocess(image_pil).unsqueeze(0).to(device)
				cv2_img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

				### CLIP Surgery for a single text, without fixed label sets
				with torch.no_grad():
					# CLIP architecture surgery acts on the image encoder
					image_features = model.encode_image(image)
					image_features = image_features / image_features.norm(dim=1, keepdim=True)

					# Prompt ensemble for text features with normalization
					text_features = clip.encode_text_with_prompt_ensemble(model, prompts, device)

					if (negative_prompts):
						negative_text_features = clip.encode_text_with_prompt_ensemble(model, negative_prompts, device)

					# Extract redundant features from an empty string
					redundant_features = clip.encode_text_with_prompt_ensemble(model, [""], device, redundants)

					# no sam
					if self.bypass_sam:

						def reg_inference(text_features):
							preds = []
							# Apply feature surgery for single text
							similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
							similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

							# Draw similarity map
							for b in range(similarity_map.shape[0]):
								for n in range(similarity_map.shape[-1]):
									vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
									preds.append(vis)
							return (preds)

						preds = reg_inference(text_features)
						if (negative_prompts):
							negative_preds = reg_inference(negative_text_features)
					else:
						point_thresh = self.Unprompted.parse_arg("point_threshold", 0.98)
						multimask_output = True if "multimask_output" in pargs else False

						# Init SAM
						if self.cached_predictor == -1 or self.cached_model_method != method:
							sam_model_dir = f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/segment_anything"
							os.makedirs(sam_model_dir, exist_ok=True)
							sam_filename = "sam_vit_h_4b8939.pth"
							sam_file = f"{sam_model_dir}/{sam_filename}"
							# Download model weights if we don't have them yet
							if not os.path.exists(sam_file):
								self.log.info("Downloading SAM model weights...")
								helpers.download_file(sam_file, f"https://dl.fbaipublicfiles.com/segment_anything/{sam_filename}")

							model_type = "vit_h"
							sam = sam_model_registry[model_type](checkpoint=sam_file)
							sam.to(device=device)
							predictor = SamPredictor(sam)

							self.cached_predictor = predictor
						else:
							predictor = self.cached_predictor

						predictor.set_image(np.array(image_pil))
						self.cached_model_method = method

						def sam_inference(text_features):
							preds = []

							# Combine features after removing redundant features and min-max norm
							sm = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]
							sm_norm = (sm - sm.min(0, keepdim=True)[0]) / (sm.max(0, keepdim=True)[0] - sm.min(0, keepdim=True)[0])
							sm_mean = sm_norm.mean(-1, keepdim=True)
							# get positive points from individual maps, and negative points from the mean map
							p, l = clip.similarity_map_to_points(sm_mean, cv2_img.shape[:2], t=point_thresh)
							num = len(p) // 2
							points = p[num:]  # negatives in the second half
							labels = [l[num:]]
							for i in range(sm.shape[-1]):
								p, l = clip.similarity_map_to_points(sm[:, i], cv2_img.shape[:2], t=point_thresh)
								num = len(p) // 2
								points = points + p[:num]  # positive in first half
								labels.append(l[:num])
							labels = np.concatenate(labels, 0)

							# Inference SAM with points from CLIP Surgery
							masks, scores, logits = predictor.predict(point_labels=labels, point_coords=np.array(points), multimask_output=multimask_output)
							mask = masks[np.argmax(scores)]
							mask = mask.astype('uint8')

							vis = cv2_img.copy()
							vis[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)
							vis[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
							preds.append(vis)
							if self.show:
								for idx, mask in enumerate(masks):
									plt.imsave(f"mask{idx}.png", mask)

							return (preds)

						preds = sam_inference(text_features)
						if negative_prompts:
							negative_preds = sam_inference(negative_text_features)
			elif method == "tris":
				import cv2
				import numpy as np
				from lib_unprompted.tris.CLIP import clip as clip
				from PIL import Image
				import torch.nn.functional as F
				import torchvision.transforms as transforms
				from math import floor

				from lib_unprompted.tris.args import get_parser
				from lib_unprompted.tris.model.model_stage2 import TRIS

				# os.environ['CUDA_ENABLE_DEVICES'] = '0'

				bert_tokenizer = self.Unprompted.parse_arg("bert_tokenizer", "clip")
				backbone = self.Unprompted.parse_arg("backbone", "clip-RN50")
				max_query_len = self.Unprompted.parse_arg("max_query_len", 20)
				tris_model = self.Unprompted.parse_arg("tris_model", "stage2_refcocog_google.pth")

				img_size = self.Unprompted.parse_arg("img_size", 320)
				if img_size == -1:
					img_size = None
				# max_length = 20

				if self.cached_model == -1 or self.cached_model_method != method:
					self.log.debug("Loading TRIS model...")
					model = TRIS(bert_tokenizer, backbone, max_query_len)
					model.cuda()

					model_path = f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/tris/{tris_model}"
					checkpoint = torch.load(model_path)
					model.load_state_dict(checkpoint["model"], strict=False)

					self.cached_model = model
					self.cached_model_method = method
				else:
					self.log.info("Using cached TRIS model.")

				def get_transform(size=None):
					if size is None:
						transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
					else:
						transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
					return transform

				def prepare_data(text, max_length=max_query_len):

					word_ids = []
					tokenizer = clip.tokenize

					for text_part in text:
						word_id = tokenizer(text_part).squeeze(0)[:floor(max_length / len(text))]
						word_ids.append(word_id.unsqueeze(0))
					word_ids = torch.cat(word_ids, dim=-1)

					# Pad to max_length if necessary
					if word_ids.shape[1] < max_length:
						pad = torch.zeros((1, max_length - word_ids.shape[1]), dtype=torch.long)
						word_ids = torch.cat([word_ids, pad], dim=1)

					# image_np = np.array(image_pil)
					# h, w, c = image_np.shape
					w, h = image_pil.size

					if self.cached_transform == -1 or self.cached_model_method != method:
						self.cached_transform = get_transform(size=img_size)
					img = self.cached_transform(image_pil)
					# word_ids = torch.tensor(word_ids)

					return img, word_ids, h, w

				def get_norm_cam(cam):
					cam = torch.clamp(cam, min=0)
					cam_t = cam.unsqueeze(0).unsqueeze(0).flatten(2)
					cam_max = torch.max(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
					cam_min = torch.min(cam_t, dim=2).values.unsqueeze(2).unsqueeze(3)
					norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-5)
					norm_cam = norm_cam.squeeze(0).squeeze(0)
					return norm_cam

				img, word_id, h, w = prepare_data(prompts, max_query_len)
				preds = []
				for i in range(word_id.shape[0]):
					single_word_id = word_id[i].unsqueeze(0)

					outputs = self.cached_model(img.unsqueeze(0).cuda(), single_word_id.cuda())

					output = outputs[0]
					pred = F.interpolate(output, (h, w), align_corners=True, mode='bilinear').squeeze(0).squeeze(0)

					normalized_heatmap = get_norm_cam(pred.detach().cpu())
					preds.append(normalized_heatmap)

					if "debug" in pargs:
						map_img = np.uint8(normalized_heatmap * 255)
						heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
						if self.init_image is not None:
							# Convert init image to numpy
							original = np.array(self.init_image)
							original_img = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
							img = cv2.addWeighted(heatmap_img, .6, original_img, 0.4, 0)
						else:
							img = heatmap_img

						img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
						# Convert to PIL
						Image.fromarray(img).save(f"heatmap_{i}.png")

			# CLIPseg method
			else:
				import torchvision.transforms as transforms
				import torch
				from lib_unprompted.stable_diffusion.clipseg.models.clipseg import CLIPDensePredT

				model_dir = f"{self.Unprompted.base_dir}/{self.Unprompted.Config.subdirectories.models}/clipseg"

				os.makedirs(model_dir, exist_ok=True)

				d64_filename = "rd64-uni.pth" if self.legacy_weights else "rd64-uni-refined.pth"
				d64_file = f"{model_dir}/{d64_filename}"
				d16_file = f"{model_dir}/rd16-uni.pth"

				# Download model weights if we don't have them yet
				if not helpers.download_file(d64_file, f"https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files={d64_filename}") or not helpers.download_file(d16_file, "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd16-uni.pth"):
					return ""

				# load model
				if self.cached_model == -1 or self.cached_model_method != method:
					self.log.debug("Loading clipseg model...")
					model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=not self.legacy_weights)

					# non-strict, because we only stored decoder weights (not CLIP weights)
					model.load_state_dict(torch.load(d64_file, map_location=device), strict=False)
					model = model.eval().to(device=device)

					transform = transforms.Compose([
					    transforms.ToTensor(),
					    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
					    transforms.Resize((512, 512)),
					])

					# Cache for future runs
					self.cached_model = model
					self.cached_transform = transform
					self.cached_model_method = method
				else:
					self.log.info("Using cached clipseg model.")
					model = self.cached_model
					transform = self.cached_transform

				img = transform(image_pil).unsqueeze(0)

				# predict
				with torch.no_grad():
					if "image_prompt" in kwargs:
						from PIL import Image
						if self.Unprompted.webui == "comfy":
							img_mask = Image.open(kwargs["image_prompt"])
						else:
							img_mask = flatten(Image.open(kwargs["image_prompt"]), opts.img2img_background_color)
						img_mask = transform(img_mask).unsqueeze(0)
						preds = model(img.to(device=device), img_mask.to(device=device))[0].cpu()
					else:
						preds = model(img.repeat(prompt_parts, 1, 1, 1).to(device=device), prompts)[0].cpu()

					if (negative_prompts):
						negative_preds = model(img.repeat(negative_prompt_parts, 1, 1, 1).to(device=device), negative_prompts)[0].cpu()

			# The below logic applies to all masking methods

			if "image_mask" not in self.Unprompted.shortcode_user_vars:
				self.Unprompted.shortcode_user_vars["image_mask"] = None

			if (brush_mask_mode == "add" and self.Unprompted.shortcode_user_vars["image_mask"] is not None):
				final_img = self.Unprompted.shortcode_user_vars["image_mask"].convert("RGBA").resize((mask_width, mask_height))
			else:
				final_img = None

			# process masking
			if method == "panoptic_sam":
				import cv2
				if len(preds) > 1:
					# Convert all_masks images to cv2 format
					all_masks = [numpy.array(mask) for mask in preds]
					if mask_sort_method == "random":
						import random
						random.shuffle(all_masks)
					elif mask_sort_method == "big-to-small":
						all_masks = sorted(all_masks, key=lambda x: x.shape[0] * x.shape[1], reverse=False)
					elif mask_sort_method == "left-to-right" or mask_sort_method == "top-to-bottom":
						i = 0

						# handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
						if mask_sort_method == "top-to-bottom":
							i = 1

						# construct the list of bounding boxes and sort them from top to bottom
						bounding_boxes = [cv2.boundingRect(cv2.convertScaleAbs(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY))) for c in all_masks]
						(all_masks, bounding_boxes) = zip(*sorted(zip(all_masks, bounding_boxes), key=lambda b: b[1][i]))
					elif mask_sort_method == "in-to-out":
						# Sort masks from the center of the image outwards
						center = (mask_width // 2, mask_height // 2)
						all_masks = sorted(all_masks, key=lambda x: abs(center[0] - x.shape[0] // 2) + abs(center[1] - x.shape[1] // 2))

					if reverse_mask_sort:
						all_masks = all_masks[::-1]

					# For the first mask, replace the transparent background with a black background
					preds = all_masks[min(mask_index, len(all_masks) - 1)]
					preds = cv2.cvtColor(preds, cv2.COLOR_RGBA2RGB)
					preds = [preds]

			final_img = process_mask_parts(preds, "add", final_img, mask_precision, mask_padding, padding_dilation_kernel, smoothing_kernel)

			# process negative masking
			if (brush_mask_mode == "subtract" and self.Unprompted.shortcode_user_vars["image_mask"] is not None):
				self.Unprompted.shortcode_user_vars["image_mask"] = ImageOps.invert(self.Unprompted.shortcode_user_vars["image_mask"])
				self.Unprompted.shortcode_user_vars["image_mask"] = self.Unprompted.shortcode_user_vars["image_mask"].convert("RGBA").resize((mask_width, mask_height))
				final_img = overlay_mask_part(final_img, self.Unprompted.shortcode_user_vars["image_mask"], "discard")
			if (negative_prompts):
				final_img = process_mask_parts(negative_preds, "discard", final_img, neg_mask_precision, neg_mask_padding, neg_padding_dilation_kernel, neg_smoothing_kernel)

			if "size_var" in kwargs:
				img_data = final_img.load()
				# Count number of transparent pixels
				black_pixels = 0
				total_pixels = mask_width * mask_height
				for y in range(mask_height):
					for x in range(mask_width):
						pixel_data = img_data[x, y]
						if (pixel_data[0] == 0 and pixel_data[1] == 0 and pixel_data[2] == 0):
							black_pixels += 1
				subject_size = 1 - black_pixels / total_pixels
				self.Unprompted.shortcode_user_vars[kwargs["size_var"]] = subject_size
			if "aspect_var" in kwargs:
				paste_mask = final_img.resize((self.init_image.width, self.init_image.height))
				paste_mask = ImageOps.colorize(paste_mask.convert("L"), black="black", white="white")
				paste_mask = paste_mask.convert('RGBA')
				mask_data = paste_mask.load()
				width, height = paste_mask.size
				# Convert black pixels to transparent
				for y in range(height):
					for x in range(width):
						if mask_data[x, y] == (0, 0, 0, 255):
							mask_data[x, y] = (0, 0, 0, 0)
				# Crop the image by transparency
				cropped = paste_mask.crop(paste_mask.getbbox())
				# Get the aspect ratio of cropped mask
				aspect_ratio = cropped.size[0] / cropped.size[1]
				self.log.debug(f"Mask aspect ratio: {aspect_ratio}")
				self.Unprompted.shortcode_user_vars[kwargs["aspect_var"]] = aspect_ratio
				if "save" in kwargs:
					cropped.save(f"cropped_mask.png")
			if "mask_var" in kwargs:
				self.Unprompted.shortcode_user_vars[kwargs["mask_var"]] = final_img

			# Inpaint sketch compatibility
			if "sketch_color" in kwargs:
				self.Unprompted.shortcode_user_vars["mode"] = 3

				this_color = kwargs["sketch_color"]
				# Convert to tuple for use with colorize
				if this_color[0].isdigit():
					this_color = tuple(map(int, this_color.split(',')))
				paste_mask = ImageOps.colorize(final_img.convert("L"), black="black", white=this_color)

				# Convert black pixels to transparent
				paste_mask = paste_mask.convert('RGBA')
				mask_data = paste_mask.load()
				width, height = paste_mask.size
				for y in range(height):
					for x in range(width):
						if mask_data[x, y] == (0, 0, 0, 255):
							mask_data[x, y] = (0, 0, 0, 0)

				# Match size just in case
				paste_mask = paste_mask.resize((image_pil.size[0], image_pil.size[1]))

				# Workaround for A1111 not including mask_alpha in p object
				if "sketch_alpha" in kwargs:
					alpha_channel = paste_mask.getchannel('A')
					new_alpha = alpha_channel.point(lambda i: int(float(kwargs["sketch_alpha"])) if i > 0 else 0)
					paste_mask.putalpha(new_alpha)

				# Workaround for A1111 bug, not accepting inpaint_color_sketch param w/ blur
				if (self.Unprompted.shortcode_user_vars["mask_blur"] > 0):
					from PIL import ImageFilter
					blur = ImageFilter.GaussianBlur(self.Unprompted.shortcode_user_vars["mask_blur"])
					paste_mask = paste_mask.filter(blur)
					self.Unprompted.shortcode_user_vars["mask_blur"] = 0

				# Paste mask on
				image_pil.paste(paste_mask, box=None, mask=paste_mask)

				self.Unprompted.shortcode_user_vars["init_images"][0] = image_pil
				# not used by SD, just used to append to our GUI later
				self.Unprompted.shortcode_user_vars["colorized_mask"] = paste_mask

				# Assign webui vars, note - I think it should work this way but A1111 doesn't appear to store some of these in p obj
				# note: inpaint_color_sketch = flattened image with mask on top
				# self.Unprompted.shortcode_user_vars["inpaint_color_sketch"] = image_pil
				# note: inpaint_color_sketch_orig = the init image
				# self.Unprompted.shortcode_user_vars["inpaint_color_sketch_orig"] = self.Unprompted.shortcode_user_vars["init_images"][0]
				# return image_pil

			else:
				if ("mode" not in self.Unprompted.shortcode_user_vars or self.Unprompted.shortcode_user_vars["mode"] != 5):  # 5 =  batch processing
					self.Unprompted.shortcode_user_vars["mode"] = 4  # "mask upload" mode to avoid unnecessary processing

			# Free up memory
			if "unload_model" in pargs:
				self.model = -1
				self.cached_model = -1
				self.cached_model_b = -1
				self.cached_model_method = ""
				self.cached_predictor = -1
				self.cached_preprocess = -1
				self.cached_clip_model = -1

			return final_img

		# Set up processor parameters correctly
		self.image_mask = get_mask()
		if not self.image_mask:
			return ""

		if not mask_blur and ("mask_blur" in self.Unprompted.shortcode_user_vars and self.Unprompted.shortcode_user_vars["mask_blur"] > 0):
			mask_blur = self.Unprompted.shortcode_user_vars["mask_blur"]
			self.Unprompted.shortcode_user_vars["mask_blur"] = 0

		if mask_blur:
			from PIL import ImageFilter
			blur = ImageFilter.GaussianBlur(mask_blur)
			self.image_mask = self.image_mask.filter(blur)

		# Ensure correct size
		self.image_mask = self.image_mask.resize((self.init_image.width, self.init_image.height))

		if "not_img2img" not in pargs:
			if "mode" in self.Unprompted.shortcode_user_vars:
				self.Unprompted.shortcode_user_vars["mode"] = min(5, self.Unprompted.shortcode_user_vars["mode"])
			self.Unprompted.shortcode_user_vars["image_mask"] = self.image_mask

			# Copy code from modules/processing.py, necessary for batch processing
			if "mask" in self.Unprompted.shortcode_user_vars and self.Unprompted.shortcode_user_vars["mask"] is not None:
				self.log.warning("Detected batch tab tensor mask, attempting to update it...")
				latmask = self.image_mask.convert('RGB').resize((self.Unprompted.shortcode_user_vars["init_latent"].shape[3], self.Unprompted.shortcode_user_vars["init_latent"].shape[2]))
				latmask = numpy.moveaxis(numpy.array(latmask, dtype=numpy.float32), 2, 0) / 255
				latmask = latmask[0]
				latmask = numpy.around(latmask)
				latmask = numpy.tile(latmask[None], (4, 1, 1))
				self.Unprompted.shortcode_user_vars["mask"] = torch.asarray(1.0 - latmask).to(device).type(self.Unprompted.main_p.sd_model.dtype)
				self.Unprompted.shortcode_user_vars["nmask"] = torch.asarray(latmask).to(device).type(self.Unprompted.main_p.sd_model.dtype)

			# self.Unprompted.shortcode_user_vars["mmask"]=self.Unprompted.shortcode_user_vars["mask"]
			self.Unprompted.shortcode_user_vars["mask_for_overlay"] = self.image_mask
			self.Unprompted.shortcode_user_vars["latent_mask"] = None  # fixes inpainting full resolution
			arr = {}
			arr["image"] = self.init_image
			arr["mask"] = self.image_mask
			self.Unprompted.shortcode_user_vars["init_img_with_mask"] = arr
			self.Unprompted.shortcode_user_vars["init_mask"] = self.image_mask  # TODO: Not sure if used anymore

		if "save" in kwargs:
			self.image_mask.save(f"{self.Unprompted.parse_advanced(kwargs['save'],context)}.png")

		if "return_image" in pargs:
			if "not_img2img" in pargs:
				return (self.image_mask)
			else:
				import copy
				# Note: The copied_images list is used to prevent premature garbage collection
				self.copied_images.append(copy.deepcopy(self.image_mask))
				# self.copied_images.append(self.image_mask.copy())
				return self.copied_images[-1]
		else:
			return ""

	def after(self, p=None, processed=None):
		from torchvision.transforms.functional import pil_to_tensor, to_pil_image
		from torchvision.utils import draw_segmentation_masks

		if self.image_mask and self.show:
			if "mode" not in self.Unprompted.shortcode_user_vars or self.Unprompted.shortcode_user_vars["mode"] >= 4:
				processed.images.append(self.image_mask)
			else:
				processed.images.append(self.Unprompted.shortcode_user_vars["colorized_mask"])

			overlayed_init_img = draw_segmentation_masks(pil_to_tensor(self.Unprompted.shortcode_user_vars["init_images"][0].convert("RGB")), pil_to_tensor(self.image_mask.convert("L")) > 0)
			processed.images.append(to_pil_image(overlayed_init_img))
			self.image_mask = None
			self.show = False
			return processed

	def goodbye(self):
		self.copied_images = []

	def ui(self, gr):
		o = []

		with gr.Accordion("⚙️ General Settings", open=False):
			o.append(gr.Radio(label="Masking tech method (clipseg is most accurate) 🡢 method", choices=["panoptic_sam", "clipseg", "clip_surgery", "fastsam", "tris"], value="panoptic_sam", interactive=True))
			with gr.Row():
				o.append(gr.Dropdown(label="Mask sort method 🡢 mask_sort_method", choices=["random", "left-to-right", "top-to-bottom", "big-to-small", "in-to-out"], value="left-to-right", interactive=True))
				o.append(gr.Checkbox(label="Reverse mask sort 🡢 reverse_mask_sort", interactive=True, value=False))
				o.append(gr.Number(label="Mask index 🡢 mask_index", value=0, interactive=True))
			o.append(gr.Radio(label="Mask blend mode 🡢 mode", choices=["add", "subtract", "discard"], value="add", interactive=True))
			o.append(gr.Textbox(label="Mask color, enables Inpaint Sketch mode 🡢 sketch_color", max_lines=1, placeholder="e.g. tan or 127,127,127"))
			o.append(gr.Number(label="Mask alpha, must be used in conjunction with mask color 🡢 sketch_alpha", value=0, interactive=True))
			o.append(gr.Number(label="Mask blur, not needed if you use inpaint blur already 🡢 blur", value=0, interactive=True))
			o.append(gr.Textbox(label="Save the mask size to the following variable 🡢 size_var", max_lines=1))
			o.append(gr.Checkbox(label="Show mask in output 🡢 show"))
			o.append(gr.Checkbox(label="Debug mode (saves mask images to root WebUI folder) 🡢 debug"))
			o.append(gr.Checkbox(label="Unload model after inference (for low memory devices) 🡢 unload_model"))
			o.append(gr.Checkbox(label="Use clipseg legacy weights 🡢 legacy_weights"))
		with gr.Row():
			with gr.Accordion("➕ Positive Mask", open=False):
				o.append(gr.Number(label="Precision of selected area 🡢 precision", value=100, interactive=True))
				o.append(gr.Number(label="Padding radius in pixels 🡢 padding", value=0, interactive=True))
				o.append(gr.Number(label="Smoothing radius in pixels 🡢 smoothing", value=20, interactive=True))
			with gr.Accordion("➖ Negative Mask", open=False):
				o.append(gr.Textbox(label="Negative mask prompt 🡢 negative_mask", max_lines=1))
				o.append(gr.Number(label="Negative mask precision of selected area 🡢 neg_precision", value=100, interactive=True))
				o.append(gr.Number(label="Negative mask padding radius in pixels 🡢 neg_padding", value=0, interactive=True))
				o.append(gr.Number(label="Negative mask smoothing radius in pixels 🡢 neg_smoothing", value=20, interactive=True))
		with gr.Accordion("🖼️ Stamp", open=False):
			o.append(gr.Textbox(label="Stamp file(s) 🡢 stamp", max_lines=1, placeholder="Looks for PNG file in unprompted/images/stamps OR absolute path"))
			o.append(gr.Dropdown(label="Stamp method 🡢 stamp_method", choices=["stretch", "center"], value="stretch", interactive=True))
			with gr.Row():
				o.append(gr.Number(label="Stamp X 🡢 stamp_x", value=0, interactive=True))
				o.append(gr.Number(label="Stamp Y 🡢 stamp_y", value=0, interactive=True))

		return o

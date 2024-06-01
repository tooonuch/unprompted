import gradio as gr


class Shortcode():
	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Runs an img2img task inside of an [after] block."

		# using a counter to keep track of the current image index as something isn't working quite
		# right with getting the image index from the shortcode_user_vars
		# this also helps with grid detection as well
		self.counter = -1  # start at -1 so that the first image is 0 when we iterate
		self.image_array = []

	def run_atomic(self, pargs, kwargs, context):
		import modules.img2img
		from modules import sd_samplers
		from modules import scripts

		# this counter is what we are using to track the current image index
		self.counter += 1

		did_error = False

		self.Unprompted.is_enabled = False

		# Synchronize any changes from user vars
		if "no_sync" not in pargs:
			self.Unprompted.update_stable_diffusion_vars(self.Unprompted.main_p)

		if "mask_mode" not in self.Unprompted.shortcode_user_vars:
			self.Unprompted.shortcode_user_vars["mask_mode"] = 0
		init_mask = None
		if "init_mask" in self.Unprompted.shortcode_user_vars:
			init_mask = self.Unprompted.shortcode_user_vars["init_mask"]
		elif "init_mask_inpaint" in self.Unprompted.shortcode_user_vars:
			init_mask = self.Unprompted.shortcode_user_vars["init_mask_inpaint"]

		init_img_with_mask = self.Unprompted.shortcode_user_vars["init_img_with_mask"] if "init_img_with_mask" in self.Unprompted.shortcode_user_vars else None

		for att in self.Unprompted.shortcode_user_vars:
			if att.startswith("controlnet_") or att.startswith("cn_"):
				self.Unprompted.update_controlnet_var(self.Unprompted.main_p, att)

		# this just sets up the starting image for the img2img operation
		# storing a copy of the image array into this class instance as it seems the order of the images
		# change once the processing starts and we end up getting the images in the wrong order
		# compared to the prompt array
		if self.image_array == []:
			try:
				self.image_array = self.Unprompted.main_p.init_images.copy()
			except:
				self.image_array = self.Unprompted.after_processed.images.copy()

		batch_count = self.Unprompted.shortcode_user_vars["n_iter"] if "n_iter" in self.Unprompted.shortcode_user_vars else 1
		batch_size = self.Unprompted.shortcode_user_vars["batch_size"]
		total_images_expected = batch_count * batch_size

		if len(self.image_array) != total_images_expected:
			# when the processed image array contains the grid image, we then will have the batch_count*batch_size + 1 images
			# when there is a grid image it is always the first image, so we can safely remove it from the image array
			self.image_array.pop(0)

		try:
			img2img_images = []
			temp_gr_request = lambda: None
			temp_gr_request.username = "unprompted"

			if self.counter == 0:
				prompt = self.Unprompted.shortcode_user_vars["prompt"]
				negative_prompt = self.Unprompted.shortcode_user_vars["negative_prompt"]
			else:
				prompt = self.Unprompted.after_processed.all_prompts[self.counter]
				negative_prompt = self.Unprompted.after_processed.all_negative_prompts[self.counter]

			image = self.image_array[self.counter]

			# if the ratio is set to 0 then use the width and height, otherwise use the ratio
			ratio = self.Unprompted.parse_arg("ratio", 0.0)

			current_width = image.width
			current_height = image.height

			if ratio != 0 and ratio != 1:
				current_width = int(current_width * ratio)
				current_height = int(current_height * ratio)

			if self.Unprompted.webui == "forge":
				img2img_func = modules.img2img.img2img_function
			else:
				img2img_func = self.img2img_patched

			img2img_result = img2img_func(
			    "unprompted_img2img",  #id_task
			    temp_gr_request,  # gr.request
			    int(self.Unprompted.shortcode_user_vars["mode"]) if "mode" in self.Unprompted.shortcode_user_vars else 0,  #p.mode
			    prompt,
			    negative_prompt,
			    [],  # prompt_styles
			    image,  # init_img
			    None,  # sketch
			    init_img_with_mask,  # p.init_img_with_mask
			    None,  # inpaint_color_sketch
			    None,  # inpaint_color_sketch_orig
			    image,  # p.init_img_inpaint
			    init_mask,  # p.init_mask_inpaint
			    self.Unprompted.shortcode_user_vars["mask_blur"] if "mask_blur" in self.Unprompted.shortcode_user_vars else 0,  # p.mask_blur
			    0.0,  #p.mask_alpha
			    0,  # p.inpainting_fill
			    1,  # n_iter
			    1,  # batch_size
			    self.Unprompted.shortcode_user_vars["cfg_scale"],
			    self.Unprompted.shortcode_user_vars["image_cfg_scale"] if "image_cfg_scale" in self.Unprompted.shortcode_user_vars else None,
			    self.Unprompted.shortcode_user_vars["denoising_strength"] if self.Unprompted.shortcode_user_vars["denoising_strength"] is not None else 1.0,
			    0,  #selected_scale_tab
			    current_height,  #self.Unprompted.shortcode_user_vars["height"],
			    current_width,  #self.Unprompted.shortcode_user_vars["width"],
			    1.0,  #scale_by
			    self.Unprompted.shortcode_user_vars["resize_mode"] if "resize_mode" in self.Unprompted.shortcode_user_vars else 1,
			    self.Unprompted.shortcode_user_vars["inpaint_full_res"] if "inpaint_full_res" in self.Unprompted.shortcode_user_vars else True,  # p.inpaint_full_res
			    self.Unprompted.shortcode_user_vars["inpaint_full_res_padding"] if "inpaint_full_res_padding" in self.Unprompted.shortcode_user_vars else 1,  # p.inpaint_full_res_padding
			    0,  # p.inpainting_mask_invert
			    "",  #p.batch_input_directory
			    "",  #p.batch_output_directory
			    "",  #p.img2img_batch_inpaint_mask_dir
			    [],  # override_settings_texts
			    self.Unprompted.shortcode_user_vars["img2img_batch_use_png_info"] if "img2img_batch_use_png_info" in self.Unprompted.shortcode_user_vars else 0,  # img2img_batch_use_png_info
			    [],  # img2img_batch_png_info_props,
			    "",  # img2img_batch_png_info_dir
			    *self.Unprompted.main_p.script_args)

			# Get the image stored in the first index
			img2img_images.append(img2img_result[0][0])

		except Exception as e:
			self.log.exception("Exception while running the img2img task")
			did_error = True

		self.Unprompted.is_enabled = True

		try:
			if len(img2img_images) < 1:
				self.log.error(f"The returned object does not appear to contain an image: {img2img_images}")
				return ""
		except Exception as e:
			self.log.error("Could not check length of img2img_images")
			pass

		# Add the new image(s) to our main output
		if did_error:
			return False
		elif "return_image" in pargs:
			return img2img_images[0]
		else:
			self.Unprompted.after_processed.images.extend(img2img_images)
			self.Unprompted.shortcode_user_vars["init_images"] = self.Unprompted.after_processed.images
		return ""

	def img2img_patched(self, id_task: str, request: gr.Request, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, mask_blur: int, mask_alpha: float, inpainting_fill: int, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, selected_scale_tab: int, height: int, width: int, scale_by: float, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, img2img_batch_use_png_info: bool, img2img_batch_png_info_props: list, img2img_batch_png_info_dir: str, *args):

		import numpy as np
		from PIL import Image, ImageFilter, ImageEnhance, UnidentifiedImageError
		import gradio as gr

		from modules import images
		from modules.infotext_utils import create_override_settings_dict
		from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
		from modules.shared import opts
		import modules.shared as shared
		import modules.processing as processing
		from modules.ui import plaintext_to_html
		import modules.scripts
		from contextlib import closing

		override_settings = create_override_settings_dict(override_settings_texts)

		if mode == 0:  # img2img
			image = init_img
			mask = None
		elif mode == 1:  # img2img sketch
			image = sketch
			mask = None
		elif mode == 2:  # inpaint
			image, mask = init_img_with_mask["image"], init_img_with_mask["mask"]
			mask = processing.create_binary_mask(mask)
		elif mode == 3:  # inpaint sketch
			image = inpaint_color_sketch
			orig = inpaint_color_sketch_orig or inpaint_color_sketch
			pred = np.any(np.array(image) != np.array(orig), axis=-1)
			mask = Image.fromarray(pred.astype(np.uint8) * 255, "L")
			mask = ImageEnhance.Brightness(mask).enhance(1 - mask_alpha / 100)
			blur = ImageFilter.GaussianBlur(mask_blur)
			image = Image.composite(image.filter(blur), orig, mask.filter(blur))
		elif mode == 4:  # inpaint upload mask
			image = init_img_inpaint
			mask = init_mask_inpaint
		else:
			image = None
			mask = None

		image = images.fix_image(image)
		mask = images.fix_image(mask)

		if selected_scale_tab == 1:
			assert image, "Can't scale by because no image is selected"

			width = int(image.width * scale_by)
			height = int(image.height * scale_by)

		assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

		p = StableDiffusionProcessingImg2Img(
		    sd_model=shared.sd_model,
		    outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
		    outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
		    prompt=prompt,
		    negative_prompt=negative_prompt,
		    styles=prompt_styles,
		    batch_size=batch_size,
		    n_iter=n_iter,
		    cfg_scale=cfg_scale,
		    width=width,
		    height=height,
		    init_images=[image],
		    mask=mask,
		    mask_blur=mask_blur,
		    inpainting_fill=inpainting_fill,
		    resize_mode=resize_mode,
		    denoising_strength=denoising_strength,
		    image_cfg_scale=image_cfg_scale,
		    inpaint_full_res=inpaint_full_res,
		    inpaint_full_res_padding=inpaint_full_res_padding,
		    inpainting_mask_invert=inpainting_mask_invert,
		    override_settings=override_settings,
		)

		if self.Unprompted.shortcode_user_vars["scripts"]:
			p.scripts = self.Unprompted.shortcode_user_vars["scripts"].copy()  # modules.scripts.scripts_img2img
		else:
			p.scripts = modules.scripts.scripts_img2img

		for script in p.scripts.alwayson_scripts:
			if script.name not in [None, "extra options", "refiner", "sampler"]:  # , "seed"
				# Remove incompatible script
				p.scripts.alwayson_scripts.remove(script)
		p.script_args = args

		# Patch in extra needed variables
		p.sampler_name = self.Unprompted.shortcode_user_vars["sampler_name"]
		p.scheduler = self.Unprompted.shortcode_user_vars["scheduler"]
		p.steps = self.Unprompted.shortcode_user_vars["steps"]

		p.user = request.username

		if shared.opts.enable_console_prompts:
			print(f"\nimg2img: {prompt}", file=shared.progress_print_out)

		with closing(p):
			processed = modules.scripts.scripts_img2img.run(p, *args)
			if processed is None:
				processed = process_images(p)

		shared.total_tqdm.clear()

		generation_info_js = processed.js()
		if opts.samples_log_stdout:
			print(generation_info_js)

		if opts.do_not_show_images:
			processed.images = []

		return processed.images, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(processed.comments, classname="comments")

	def after(self, p=None, processed=None):
		self.counter = -1  # reset the counter for the next run
		self.image_array = []  # this array holds the whole images, make sure to clear it after the run
		return ""

	def ui(self, gr):
		return [
		    gr.Slider(label="Img2Img Ratio (if value is other than 1, it is used over the height and width supplied) 🡢 ratio", value=1.0, maximum=3, minimum=0.25, interactive=True, step=0.01),
		]

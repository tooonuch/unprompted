class Shortcode():
	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Optimize a hard prompt using the PEZ algorithm and CLIP encoders, AKA Hard Prompts Made Easy."
		self.learned_prompts = []

	def run_atomic(self, pargs, kwargs, context):
		if not self.Unprompted.shortcode_install_requirements("", ["sentence-transformers==2.2.2"]):
			return ""
		import lib_unprompted.stable_diffusion.hard_prompts_made_easy as pez
		import lib_unprompted.stable_diffusion.pez_open_clip as open_clip
		import torch
		from modules.images import flatten
		from modules.shared import opts
		import argparse

		# optimize prompt
		self.log.debug("Starting prompt optimization")

		# Workaround for conflict between optim_utils.py and batch processing
		if self.Unprompted.shortcode_user_vars["batch_real_index"] == 0:
			imgs = None

			if "init_images" in self.Unprompted.shortcode_user_vars:
				for img in self.Unprompted.shortcode_user_vars["init_images"]:
					imgs.append(flatten(img, opts.img2img_background_color))

			image_path = kwargs["image_path"] if "image_path" in kwargs else ""
			if (len(image_path) > 0):
				from PIL import Image
				image_paths = self.Unprompted.parse_filepath(image_path, return_all=True).split(self.Unprompted.Config.syntax.delimiter)
				imgs = []
				for img in image_paths:
					imgs.append(Image.open(img))

			target_prompt = self.Unprompted.parse_arg("target_prompt", "")

			if not imgs and not target_prompt:
				self.log.error("No valid inputs found!")
				return ""

			prompt_len = int(float(kwargs["prompt_length"])) if "prompt_length" in kwargs else 16
			iterations = int(float(kwargs["iterations"])) if "iterations" in kwargs else 200
			learning_rate = float(kwargs["learning_rate"]) if "learning_rate" in kwargs else 0.1
			weight_decay = float(kwargs["weight_decay"]) if "weight_decay" in kwargs else 0.1
			prompt_bs = int(float(kwargs["prompt_bs"])) if "prompt_bs" in kwargs else 1
			clip_preset = self.Unprompted.parse_arg("clip_preset", "none")

			if clip_preset != "none":
				if clip_preset == "sd1":
					clip_model = "ViT-L-14"
					clip_pretrain = "openai"
				elif clip_preset == "sd2":
					clip_model = "ViT-H-14"
					clip_pretrain = "laion2b_s32b_b79k"
				elif clip_preset == "sdxl":
					clip_model = "ViT-g-14"
					clip_pretrain = "laion2b_s12b_b42k"  # "laion2b_s34b_b88k"
				elif clip_preset == "sdxl_big":
					clip_model = "ViT-bigG-14"
					clip_pretrain = "laion2b_s39b_b160k"
			else:
				clip_model = kwargs["clip_model"] if "clip_model" in kwargs else "ViT-L-14"
				clip_pretrain = kwargs["clip_pretrain"] if "clip_pretrain" in kwargs else "openai"

			print_step = int(float(kwargs["print_step"])) if "print_step" in kwargs else 100
			batch_size = int(float(kwargs["batch_size"])) if "batch_size" in kwargs else 1

			# Set up params with argparse since it's the format used in the original repo
			args = argparse.Namespace()

			setattr(args, "prompt_len", prompt_len)
			setattr(args, "iter", iterations)
			setattr(args, "lr", learning_rate)
			setattr(args, "weight_decay", weight_decay)
			setattr(args, "prompt_bs", prompt_bs)
			setattr(args, "print_step", print_step)
			setattr(args, "batch_size", batch_size)
			setattr(args, "clip_model", clip_model)
			setattr(args, "clip_pretrain", clip_pretrain)

			# load CLIP model
			device = "cuda" if torch.cuda.is_available() else "cpu"
			model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

			for i in range(self.Unprompted.main_p.n_iter * self.Unprompted.main_p.batch_size):
				self.learned_prompts.append(pez.optimize_prompt(model, preprocess, args, device, target_images=imgs, target_prompts=target_prompt))

			if "free_memory" in pargs:
				self.log.debug("Freeing img2pez memory")
				import gc
				model = None
				preprocess = None
				_ = None
				gc.collect()
				with torch.no_grad():
					torch.cuda.empty_cache()
		else:
			self.log.warning("The img2pez library has limited support with WebUI batch processing; all img2pez prompts are generated at the start of the run and its settings cannot be changed mid-run.")

		return self.learned_prompts[self.Unprompted.shortcode_user_vars["batch_real_index"]]

	def cleanup(self):
		self.learned_prompts = []

	def ui(self, gr):
		o = []

		o.append(gr.Text(label="Starting images 🡢 image_path", placeholder="Leave blank to use the initial img2img image. Supports multiple paths."))
		o.append(gr.Text(label="Starting prompt 🡢 target_prompt", placeholder="Prompt to optimize for; can be used in conjunction with images."))
		o.append(gr.Number(label="Output prompt length 🡢 prompt_length", value=75, maximum=75, minimum=1, interactive=True))
		o.append(gr.Number(label="Iterations 🡢 iterations", value=300, interactive=True))
		o.append(gr.Number(label="Learning rate 🡢 learning_rate", value=0.1, interactive=True))
		o.append(gr.Number(label="Weight decay 🡢 weight_decay", value=0.1, interactive=True))
		o.append(gr.Number(label="Batch size 🡢 prompt_bs", value=1, interactive=True))
		o.append(gr.Dropdown(label="CLIP preset 🡢 clip_preset", info="This will override the model and pretrain options below.", choices=["sd1", "sd2", "sdxl_big", "sdxl", "none"], value="sdxl", interactive=True))
		with gr.Accordion("CLIP Overrides", open=False):
			o.append(gr.Dropdown(label="CLIP model 🡢 clip_model", choices=["ViT-L-14", "ViT-H-14"], value="ViT-L-14", interactive=True))
			o.append(gr.Dropdown(label="CLIP pretrain 🡢 clip_pretrain", choices=["openai", "laion2b_s32b_b79k"], value="openai", interactive=True))
		o.append(gr.Checkbox(label="Try freeing CLIP model from memory after training? 🡢 free_memory", value=True))

		return o

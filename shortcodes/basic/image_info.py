class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Returns various types of metadata about the image."
		self.iqa_metrics = {}

	def run_atomic(self, pargs, kwargs, context):
		from PIL import Image
		return_string = ""
		delimiter = self.Unprompted.parse_arg("delimiter", ",")

		image = self.Unprompted.parse_image_kwarg("file")
		if not image:
			return ""

		if "width" in pargs:
			return_string += str(image.width) + delimiter
		if "height" in pargs:
			return_string += str(image.height) + delimiter
		if "aspect_ratio" in pargs:
			return_string += str(image.width / image.height) + delimiter
		if "filename" in pargs:
			from pathlib import Path
			return_string += Path(image.filename).stem + delimiter
		if "filetype" in pargs:
			return_string += (image.format or "None") + delimiter
		if "filesize" in pargs:
			import sys
			return_string += str(sys.getsizeof(image.tobytes())) + delimiter
		if "iqa" in kwargs:
			if self.Unprompted.shortcode_install_requirements(f"image quality assessment", ["pyiqa"]):
				import pyiqa, torch

				metrics = self.Unprompted.parse_alt_tags(kwargs["iqa"], context).split(self.Unprompted.Config.syntax.delimiter)
				for metric_name in metrics:
					if metric_name not in self.iqa_metrics:
						self.log.info(f"Creating IQA metric `{metric_name}`...")
						self.iqa_metrics[metric_name] = pyiqa.create_metric(metric_name, device=torch.device("cuda") if torch.cuda.is_available() else "cpu", as_loss=False)
					else:
						self.log.info(f"Using cached IQA metric `{metric_name}`")

					score = self.iqa_metrics[metric_name](image).cpu().item()

					return_string += str(score) + delimiter
		if "pixel_count" in pargs:
			return_string += str(image.width * image.height) + delimiter

		if "unload_metrics" in pargs:
			self.iqa_metrics = {}

		return (return_string[:-len(delimiter)])

	def ui(self, gr):
		return [
		    gr.Textbox(label="Path to image (uses SD image by default) 🡢 arg_str"),
		    gr.Textbox(label="Image Quality Assessment metric(s) 🡢 iqa"),
		]

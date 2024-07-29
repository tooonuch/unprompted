class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Returns the number of items in a delimited string."

	def run_atomic(self, pargs, kwargs, context):
		_delimiter = self.Unprompted.parse_arg("_delimiter", self.Unprompted.Config.syntax.delimiter)
		_max = self.Unprompted.parse_arg("_max", -1)
		this_obj = self.Unprompted.parse_advanced(pargs[0], context)
		# Support direct array
		if isinstance(this_obj, list):
			return (min(_max if _max != -1 else len(this_obj), len(this_obj)))
		strings = this_obj.split(_delimiter, _max)
		return (len(strings))

	def ui(self, gr):
		return [
		    gr.Textbox(label="The string to evaluate 🡢 arg_str", max_lines=1, placeholder=self.Unprompted.Config.syntax.delimiter),
		    gr.Textbox(label="Delimiter to check for 🡢 _delimiter", max_lines=1, placeholder=self.Unprompted.Config.syntax.delimiter),
		    gr.Number(label="Maximum number to be returned 🡢 _max", value=-1, interactive=True),
		]

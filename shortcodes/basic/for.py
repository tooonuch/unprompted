class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "It's a for loop."

	def preprocess_block(self, pargs, kwargs, context):
		return True

	def run_block(self, pargs, kwargs, context, content):
		final_string = ""
		this_var = ""
		test = pargs[0]
		expr = pargs[1]

		for key, value in kwargs.items():
			if (self.Unprompted.is_system_arg(key)):
				continue  # Skips system arguments
			this_var = key
			self.Unprompted.shortcode_objects["set"].run_block([key], [], context, value)
			# Get the datatype
			datatype = type(self.Unprompted.shortcode_user_vars[key])

		while True:

			if (self.Unprompted.parse_advanced(test, context)):
				if "_raw" in pargs:
					final_string += self.Unprompted.process_string(content, context)
				else:
					final_string += self.Unprompted.process_string(self.Unprompted.sanitize_pre(content, self.Unprompted.Config.syntax.sanitize_block, True), context, False)

				break_type = self.Unprompted.handle_breaks()
				if break_type == self.Unprompted.FlowBreaks.BREAK:
					break

				if this_var:
					self.Unprompted.shortcode_user_vars[this_var] = self.Unprompted.parse_advanced(expr, context)
				else:
					self.Unprompted.parse_arg(expr, datatype=datatype)
			else:
				break

		return final_string

	def ui(self, gr):
		return [
		    gr.Textbox(label="Set a variable 🡢 my_var", max_lines=1, placeholder="1"),
		    gr.Textbox(label="Conditional check 🡢 arg_str", max_lines=1, placeholder="my_var < 10"),
		    gr.Textbox(label="Operation to perform at the end step 🡢 arg_str2", max_lines=1, placeholder="my_var + 1"),
		    gr.Checkbox(label="Print content without sanitizing 🡢 _raw"),
		]

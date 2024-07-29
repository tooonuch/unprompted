class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Foreach loop."

	def preprocess_block(self, pargs, kwargs, context):
		return True

	def run_block(self, pargs, kwargs, context, content):

		if not self.Unprompted.validate_args(self.log.error, min_parg_count=2):
			return ""

		final_string = ""

		idx_var = self.Unprompted.parse_arg("idx", "idx")

		the_var = self.Unprompted.parse_alt_tags(pargs[0], context)
		list_name = self.Unprompted.parse_alt_tags(pargs[1], context)

		if list_name in self.Unprompted.shortcode_user_vars:
			the_list = self.Unprompted.shortcode_user_vars[list_name]
		else:
			self.log.error(f"List variable {list_name} not found.")
			return ""

		for _idx, item in enumerate(the_list):
			self.log.debug(f"Looping through item {_idx} in {list}")
			self.Unprompted.shortcode_user_vars[the_var] = item
			self.Unprompted.shortcode_user_vars[idx_var] = _idx

			final_string += self.Unprompted.process_string(self.Unprompted.sanitize_pre(content, self.Unprompted.Config.syntax.sanitize_block, True), context, False)
			break_type = self.Unprompted.handle_breaks()
			if break_type == self.Unprompted.FlowBreaks.BREAK:
				break

		return final_string

	def ui(self, gr):
		return [
		    gr.Textbox(label="Iterator variable 🡢 str", max_lines=1, placeholder="item"),
		    gr.Textbox(label="List variable 🡢 str", max_lines=1, placeholder="my_list"),
		]

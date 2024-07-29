class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Return the absolute value of the content."

	def run_block(self, pargs, kwargs, context, content):
		try:
			content = abs(float(content))
		except:
			self.log.exception("Unable to get the absolute value of content.")
		return content

	def ui(self, gr):
		pass

class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Skips the remaining code within the current iteration of the loop and move on to the next iteration."

	def run_atomic(self, pargs, kwargs, context):
		return ""

	def ui(self, gr):
		pass

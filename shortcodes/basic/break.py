class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Exits the current loop or function early."

	def run_atomic(self, pargs, kwargs, context):
		# self.log.debug("Exiting early.")
		return ""

	def ui(self, gr):
		pass

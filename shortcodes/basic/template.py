class Shortcode():
	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "This is used by the Wizard to instantiate a custom template UI. It is bypassed by the normal shortcode parser."

	def run_atomic(self, pargs, kwargs, context):
		return ("")

	def ui(self, gr):
		pass

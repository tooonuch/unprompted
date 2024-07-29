class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Returns whether the specified variable exists."

	def run_atomic(self, pargs, kwargs, context):
		_any = "_any" in pargs
		for parg in pargs:
			if parg == "_any":
				continue
			the_var = self.Unprompted.parse_advanced(parg, context)
			if the_var in self.Unprompted.shortcode_user_vars:
				if _any:
					return True
				continue
			else:
				return False

		return True

	def ui(self, gr):
		pass

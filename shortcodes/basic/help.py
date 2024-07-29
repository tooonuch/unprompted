class Shortcode():

	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Returns documentation for the requested shortcode or feature."

	def run_atomic(self, pargs, kwargs, context):
		import os, webbrowser
		help_string = ""
		_online = self.Unprompted.parse_arg("_online", False)

		# Add "help" to pargs for documentation on this shortcode
		if not pargs:
			pargs = ["help"]

		def retrieve_doc(subpath, title=None):
			if not title:
				title = subpath

			if _online:
				# Open documentation in the browser
				webbrowser.open(f"https://github.com/ThereforeGames/unprompted/tree/main/docs/{subpath}.md")
				return f"Documentation for '{title}' opened in browser."
			else:
				doc_file = f"{self.Unprompted.base_dir}/docs/{subpath}.md"
				if os.path.exists(doc_file):
					with open(doc_file, "r", encoding="utf-8") as f:
						return f"# {title}\n\n" + f.read()
				else:
					return f"Documentation for '{title}' not found: {subpath}.md"

		for parg in pargs:
			parg = parg.lower()

			if self.Unprompted.is_system_arg(parg):
				continue
			if parg in self.Unprompted.shortcode_objects:
				help_string += retrieve_doc(f"shortcodes/{parg}", f"{self.Unprompted.Config.syntax.tag_start}{parg}{self.Unprompted.Config.syntax.tag_end}")
			elif parg == "manual":
				help_string += retrieve_doc("MANUAL")
			elif parg == "changelog":
				help_string += retrieve_doc("CHANGELOG")

		# Circumvent newline sanitization rules of Unprompted
		help_string = help_string.replace("\n", "%NEWLINE%")

		return help_string

	def ui(self, gr):
		pass

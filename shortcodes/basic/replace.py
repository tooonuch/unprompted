class Shortcode():
	def __init__(self, Unprompted):
		self.Unprompted = Unprompted
		self.description = "Updates a string using the arguments for replacement logic."

	def preprocess_block(self, pargs, kwargs, context):
		if "_now" in pargs:
			return True
		return False

	def run_block(self, pargs, kwargs, context, content):

		_insensitive = self.Unprompted.shortcode_var_is_true("_insensitive", pargs, kwargs)
		_strict = self.Unprompted.parse_arg("_strict", False)

		if "_load" in kwargs:
			jsons = self.Unprompted.load_jsons(self.Unprompted.parse_advanced(kwargs["_load"], context), context)
			kwargs.update(jsons)

		from_values = []
		to_values = []

		for key, value in kwargs.items():
			if (key == "_from"):
				from_values.extend(self.Unprompted.parse_advanced(value, context).split(self.Unprompted.Config.syntax.delimiter))
				to_values.extend(self.Unprompted.parse_advanced(kwargs["_to"] if "_to" in kwargs else "", context).split(self.Unprompted.Config.syntax.delimiter))

			elif (key[0] != "_"):
				from_values.append(self.Unprompted.parse_advanced(key, context))
				if _strict:
					to_values.append(value)
				else:
					to_values.append(self.Unprompted.parse_advanced(value, context))
			else:
				continue

		for i in range(len(from_values)):
			from_value = from_values[i]
			to_value = to_values[i] if i < len(to_values) else to_values[-1]

			if _insensitive:
				import re
				_count = int(float(kwargs["_count"])) if "_count" in kwargs else 0
				compiled = re.compile(re.escape(from_value), re.IGNORECASE)
				content = compiled.sub(to_value, content, count=_count)
			else:
				if _strict and from_value in content:
					to_value = self.Unprompted.parse_advanced(to_value, context)

				if ("_count" in kwargs):
					content = content.replace(from_value, to_value, self.Unprompted.parse_advanced(kwargs["_count"]))
				else:
					content = content.replace(from_value, to_value)

		if "_now" in pargs:
			return self.Unprompted.process_string(content, context)
		return content

	def ui(self, gr):
		return [
		    gr.Textbox(label="Arbitrary replacement arguments in old=new format 🡢 arg_verbatim", max_lines=1, placeholder='hello="goodbye" red="blue"'),
		    gr.Textbox(label="Original value, with advanced expression support 🡢 _from", max_lines=1),
		    gr.Textbox(label="New value, with advanced expression support 🡢 _to", max_lines=1),
		    gr.Textbox(label="Path to one or more JSON files containing from:to replacement data 🡢 _load", max_lines=1),
		    gr.Number(label="Maximum number of times the replacement may occur 🡢 _count", max_lines=1, value=-1),
		    gr.Checkbox(label="Perform case-insensitive replacements 🡢 _insensitive", value=False),
		    gr.Checkbox(label="Only evaluate expressions on matches 🡢 _strict", value=False),
		]

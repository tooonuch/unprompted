# Unprompted by Therefore Games. All Rights Reserved.
# https://patreon.com/thereforegames
# https://github.com/ThereforeGames/unprompted

# This assumes Unprompted is in the same directory as this script (see below for an alternative import method)
from lib_unprompted.shared import Unprompted

# Main object
Unprompted = Unprompted()

# Note! If Unprompted is located elsewhere (e.g. as a package), you should be able to import it like this instead:

# import inspect, os, sys
# import unprompted.lib_unprompted.shared
# module_path = os.path.dirname(os.path.dirname(inspect.getfile(unprompted.lib_unprompted.shared.Unprompted)))
# Ensure that Unprompted imports are available at all times:
# sys.path.insert(0, f"{module_path}")
# Unprompted = unprompted.lib_unprompted.shared.Unprompted(module_path)


def do_unprompted(string):
	# Reset vars
	Unprompted.shortcode_user_vars = {}

	# TODO: We may want to declare our own log level for the result message
	Unprompted.log.info(Unprompted.start(string))

	# Cleanup routines
	Unprompted.cleanup()
	Unprompted.goodbye()


# Allows user input indefinitely
while True:
	try:
		command = input("(INPUT) Unprompted string:")
		do_unprompted(command)
	except Exception as e:
		Unprompted.log.exception("Exception occurred.")

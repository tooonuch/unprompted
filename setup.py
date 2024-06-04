# Strictly for use outside of the WebUI. This file will allow you to install Unprompted via pip. Example:
# pip install unprompted@git+https://github.com/ThereforeGames/unprompted

from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil


class CustomInstall(install):
	def run(self):
		# Run the standard install
		install.run(self)

		# Move the files and directories into the 'unprompted' directory
		installation_path = os.path.join(self.install_lib, "unprompted")
		os.makedirs(installation_path, exist_ok=True)
		for item in ["shortcodes", "lib_unprompted", "templates", "config.json"]:
			shutil.move(os.path.join(self.install_lib, item), os.path.join(installation_path, item))


setup(
    name="Unprompted",
    version="11.0.2",
    packages=find_packages(),
    package_data={
        '': ['config.json', 'shortcodes/*', 'lib_unprompted/*', 'templates/common/*'],
    },
    cmdclass={
        'install': CustomInstall,
    },
)

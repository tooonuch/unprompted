# Strictly for use outside of the WebUI. This file will allow you to install Unprompted via pip. Example:
# pip install unprompted@git+https://github.com/ThereforeGames/unprompted

from setuptools import setup, find_packages

setup(
    name="Unprompted",
    version="11.0.2",
    packages=find_packages(),
    package_data={
        'unprompted': ['config.json', 'shortcodes/*', 'lib_unprompted/*'],
    },
)

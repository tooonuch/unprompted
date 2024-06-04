# Strictly for use outside of the WebUI. This file will allow you to install Unprompted via pip. Example:
# pip install unprompted@git+https://github.com/ThereforeGames/unprompted

from setuptools import setup

setup(
    name='Unprompted',
    version='11.0.2',
    package_dir={'unprompted': '.'},
    packages=['unprompted.lib_unprompted', 'unprompted.shortcodes', 'unprompted.shortcodes.basic', 'unprompted.shortcodes.stable_diffusion', 'unprompted.templates'],
    package_data={
        '': ['*.json'],
        'unprompted.templates': ['common/*.txt']
    },
)

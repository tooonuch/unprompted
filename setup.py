# Strictly for use outside of the WebUI. This file will allow you to install Unprompted via pip. Example:
# pip install unprompted@git+https://github.com/ThereforeGames/unprompted

from setuptools import setup

setup(
    name='Unprompted',
    version='11.0.2',
    package_dir={'unprompted': '.'},
    packages=['unprompted.lib_unprompted', 'unprompted.shortcodes', 'unprompted.templates'],
    package_data={
        'unprompted': ['config.json'],
        'unprompted.shortcodes': ['basic/*.txt', 'stable_diffusion/*.txt'],
        'unprompted.templates': ['common/*.txt']
    },
)

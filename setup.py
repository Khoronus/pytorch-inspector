#!/usr/bin/env python
import io
import os
import re
from datetime import datetime
from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()

def find_version(*file_paths):
    """Adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/main/setup.py"""
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('pytorch_inspector', '__init__.py')
if VERSION.endswith('dev'):
    VERSION = VERSION + datetime.today().strftime('%Y%m%d')
    
# Read requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    # Metadata
    name='pytorch_inspector',
    version=VERSION,
    python_requires='>=3.7',
    description='pytorch_inspector is a library to inspect torch tensors at run-time.',
    license='Apache-2.0',
    zip_safe=True,
    include_package_data=True,
    packages=find_packages(),
    scripts=[],
    url="https://github.com/Khoronus/pytorch_inspector/",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0",
    ],
    author='Alessandro Moro',
    author_email='alessandromoro.italy@gmail.com',
)

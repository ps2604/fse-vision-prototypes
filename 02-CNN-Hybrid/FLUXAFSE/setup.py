#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="fse-fluxa-cnn-hybrid",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.13.0",
        "numpy",
        "opencv-python",
    ],
)

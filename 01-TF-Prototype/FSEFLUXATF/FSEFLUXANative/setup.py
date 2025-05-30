#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="fse-fluxa-tf-prototype",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy",
        "opencv-python",
        "google-cloud-storage",
    ],
)

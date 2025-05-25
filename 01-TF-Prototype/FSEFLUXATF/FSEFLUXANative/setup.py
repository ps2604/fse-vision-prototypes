from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='auralith_fluxa_fse_native_training',
    version='2.0.0',
    description='FLUXA-FSE Native Training with Pure Float-Native State Elements Architecture and Multi-GPU Support',
    author='Auralith',
    py_modules=['fluxa_fse_native_train', 'fse_native_core'],
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'train-fluxa-fse-native=fluxa_fse_native_train:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
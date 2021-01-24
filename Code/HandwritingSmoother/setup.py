import setuptools
import pywritesmooth as pws

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywritesmooth",
    version=get_version("pywritesmooth/__init__.py"),
    author="YuMei Bennett, Edward Fry, Muchigi Kimari, Ikenna Nwaogu",
    author_email="edwardf@smu.edu",
    description="Transform handwriting into a smoothed version",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdwardAF-IT/Capstone",
    packages=setuptools.find_packages(),
    install_requires=[
        'beautifulsoup4',
        'click',
        'ipython',
        'ipywidgets',
        'lxml',
        'matplotlib',
        'numpy',
        'Pillow',
        'pytorch-lightning',
        'scikit-learn',
        'scipy',
        'svgwrite',
        'torch'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points={
    'console_scripts': [
        'pywritesmooth = pywritesmooth.HandwritingSmoother:main',
    ],
}
)
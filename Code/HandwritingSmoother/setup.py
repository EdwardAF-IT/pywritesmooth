import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pywritesmooth",
    version="0.0.1",
    author="YuMei Bennett, Edward Fry, Muchigi Kimari, Ikenna Nwaogu",
    author_email="edwardf@smu.edu",
    description="Transform handwriting into a smoothed version",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdwardAF-IT/Capstone",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
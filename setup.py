import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GraphSpace", # Replace with your own username
    version="0.0.1",
    author="Anna Calissano",
    author_email="anna.calissano@polimi.it",
    description="GraphSpace: how to perform statistical analysis of a population of graphs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/annacalissano/GraphSpace",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CC - BY NC",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

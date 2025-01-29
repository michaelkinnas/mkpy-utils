from setuptools import setup, find_packages

setup(
    name="mkpyutils",
    version="0.3",
    packages=find_packages(),
    install_requires=['torch>=2.0.0', 'tqdm>=4.67.0'],
    author="Michael Kinnas",
    author_email="michaelkinnas@gmail.com",
    description="A collection of python utilities to help automate machine learning tasks with the PyTorch library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/michaelkinnas/mkpy-utils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    license = "MIT"
)

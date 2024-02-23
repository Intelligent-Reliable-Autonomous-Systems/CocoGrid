import os
from setuptools import setup, find_packages

def read(fname):
    """Reads a file's contents as a string.
    Args:
        fname: Filename.
    Returns:
        File's contents.
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

VERSION = "0.1"
BASE_URL = "https://gitlab.engr.oregonstate.edu/jewettje/minimujo"
INSTALL_REQUIRES = [
    "dm-control>=1.0.16",
    "gymnasium>=0.29.1",
    "labmaze>=1.0.6",
    "minigrid>=2.3.1",
    "numpy>=1.26.4"
]

setup(
    name="minimujo",
    version=VERSION,
    description="A continuous extension of Minigrid gridworld environments to MuJoCo",
    long_description=read("README.md"),
    author="Jeff Jewett",
    author_email="jewettje@oregonstate.edu",
    license="MIT",
    url=BASE_URL,
    packages=find_packages(),
    zip_safe=True,
    install_requires=INSTALL_REQUIRES,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education" "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
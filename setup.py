import setuptools
from pySatlantic import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pySatlantic",
    version=__version__,
    author="Nils Haentjens",
    author_email="nils.haentjens@maine.edu",
    description="Unpack binary messages from Satlantic instruments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OceanOptics/pySatlantic/",
    packages=setuptools.find_packages(),
    install_requires=['numpy'],
    python_requires='>=3.5',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education"
    ]
)
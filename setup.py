"""
CervShort: Domain-Aware Shortcut Disruption for Robust Cervical Cancer
Cytology Classification

Author: Teerapong Panboonyuen (Kao Panboonyuen)
Affiliation: Chulalongkorn University · Khon Kaen University, Thailand
"""

from setuptools import setup, find_packages

setup(
    name="cervshort",
    version="1.0.0",
    author="Teerapong Panboonyuen",
    author_email="teerapong.pa@chula.ac.th",
    description="Domain-Aware Shortcut Disruption for Cervical Cancer Cytology",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kaopanboonyuen/CervShort",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "Pillow>=9.5.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "scipy>=1.11.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

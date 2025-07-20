"""Setup script for OCEAN package."""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ocean",
    version="0.1.0",
    author="OCEAN Implementation Team",
    description="Online Multi-modal Causal structure lEArNiNG for Root Cause Analysis",
    long_description="A PyTorch implementation of the OCEAN model for root cause analysis in microservice systems.",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "ocean-train=ocean.scripts.train:main",
            "ocean-evaluate=ocean.scripts.evaluate:main",
        ],
    },
)
"""
ViTfly最小实现安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vitfly-minimal-airsim",
    version="1.0.0",
    author="ViTfly Team",
    author_email="",
    description="基于Vision Transformer的端到端无人机避障系统 - AirSim版本",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vitfly/vitfly-minimal-airsim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    python_requires=">=3.7,<3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.9.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "analysis": [
            "jupyter>=1.0.0",
            "seaborn>=0.11.0",
            "scipy>=1.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "vitfly=vitfly_main:main",
            "vitfly-nav=vitfly_navigation:main", 
            "vitfly-train=train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
)
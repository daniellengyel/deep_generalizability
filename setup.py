from setuptools import find_packages, setup


setup(
    name="margin_flatness",
    version="1.0.0",
    author="Daniel Lengyel",
    author_email="dl2119@ic.ac.uk",
    description="Test the relationship between margins and flatness. ",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pyyaml",
        "torch",
        "torchvision",
        "tensorboard==2.3.0",
        # "ray[tune]",
        "numpy",
        "matplotlib",
        "pandas",
        "tqdm",
    ],
)

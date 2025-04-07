from setuptools import setup, find_packages

setup(
    name="data2param",
    version="0.0.0",
    author="Bingsong Zhao",
    author_email="bingsong.zhao.psy@gmail.com",
    description="End-to-End Neural Networks for Behavioral Model Fitting and Comparison",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

from setuptools import find_packages, setup
import setuptools

setuptools.setup(
    name="geotool",
    version="0.1.0",
    author="uda-koki",
    author_email="koki190403nen.3@gmail.com",
    description="You can receive the message 'Hello!!!'",
    url="https://github.com/koki190403nen/geotool",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
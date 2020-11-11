# BirdVoxClassify: species classification of bird flight calls

An open-source Python library and command-line tool for classifying bird species from flight calls in audio recordings.

[![PyPI](https://img.shields.io/badge/python-3.5%2C%203.6-blue.svg)]()
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![Coverage Status](https://coveralls.io/repos/github/BirdVox/birdvoxclassify/badge.svg)](https://coveralls.io/github/BirdVox/birdvoxclassify)
[![Build Status](https://travis-ci.org/BirdVox/birdvoxclassify.svg?branch=master)](https://travis-ci.org/BirdVox/birdvoxclassify)
[![Documentation Status](https://readthedocs.org/projects/birdvoxclassify/badge/?version=latest)](http://birdvoxclassify.readthedocs.io/en/latest/?badge=latest)

BirdVoxClassify is a pre-trained deep learning system for classifying bird species from flight calls in short audio recordings.
It relies on per-channel energy normalization (PCEN) for improved robustness to background noise.
It is made available both as a Python library and as a command-line tool for Windows, OS X, and Linux.


# Installation instructions

Dependencies
------------


#### Python Versions
Currently, we support Python 3.6 and Python 3.7. Python 3.8 is not currently supported since TensorFlow 1.x is not supported for Python 3.8.

#### libsndfile (Linux only)
BirdVoxClassify depends on the PySoundFile module to load audio files, which itself depends on the non-Python library libsndfile.
On Windows and Mac OS X, these will be installed automatically via the ``pip`` package manager and you can therefore skip this step.
However, on Linux, `libsndfile` must be installed manually via your platform's package manager.
For Debian-based distributions (such as Ubuntu), this can be done by simply running

    apt-get install libsndfile

For more detailed information, please consult the
[installation instructions of pysoundfile](https://pysoundfile.readthedocs.io/en/0.9.0/#installation>).

#### Note about TensorFlow:
Please note that we explicitly require `tensorflow==1.15.2` to be consistent with [BirdVoxDetect](https://github.com/BirdVox/birdvoxdetect). We plan to move to TF 2.x in the future.


Installing BirdVoxClassify
------------------------
The simplest way to install BirdVoxClassify is by using ``pip``, which will also install the additional required dependencies
if needed.

To install the latest version of BirdVoxClassify from source:

1. Clone or pull the latest version:

        git clone git@github.com:BirdVox/birdvoxclassify.git

2. Install using pip to handle Python dependencies:

        cd birdvoxclassify
        pip install -e .

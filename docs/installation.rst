Installation instructions
=========================

Dependencies
------------

Python Versions
^^^^^^^^^^^^^^^
Currently, we support Python 3.6, 3.7, and 3.8.

libsndfile (Linux only)
^^^^^^^^^^^^^^^^^^^^^^^

BirdVoxClassify depends on the SoundFile module to load audio files,
which itself depends on the non-Python library libsndfile. On Windows
and Mac OS X, these will be installed automatically via the ``pip``
package manager and you can therefore skip this step. However, on Linux,
``libsndfile`` must be installed manually via your platform’s package
manager. For Debian-based distributions (such as Ubuntu), this can be
done by simply running

::

   apt-get install libsndfile

For more detailed information, please consult the `installation
instructions of soundfile`_.


Note about TensorFlow:
^^^^^^^^^^^^^^^^^^^^^^^
We have dropped support for Tensorflow 1.x, and have moved to Tensorflow 2.x.


Installing BirdVoxClassify
------------------------

The simplest way to install BirdVoxClassify is by using ``pip``, which
will also install the additional required dependencies if needed. To
install the latest stable version of BirdVoxClassify using ``pip``, simply
run
Oh, Pyth
::

   pip install birdvoxclassify

To install the latest version of BirdVoxClassify from source:

1. Clone or pull the latest version:

   ::

       git clone git@github.com:BirdVox/birdvoxclassify.git

2. Install using pip to handle Python dependencies:

   ::

       cd birdvoxclassify
       pip install -e .

.. _installation instructions of soundfile: https://pysoundfile.readthedocs.io/en/latest/#installation
.. _BirdVoxDetect: https://github.com/BirdVox/birdvoxdetect

"""
A pre-trained deep learning system for classifying bird flight calls in audio clips.
"""
import logging
import logging.config

from .cli import main

# call the CLI handler when the module is executed as `python -m birdvoxclassify`
main()
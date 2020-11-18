from __future__ import print_function
import logging
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections import Iterable
from pprint import pformat
from six import string_types

import birdvoxclassify
from birdvoxclassify.core import DEFAULT_MODEL_NAME
from birdvoxclassify.birdvoxclassify_exceptions import BirdVoxClassifyError

# The following line circumvent issue #1715 in xgboost
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_file_list(input_list):
    """Parse list of input paths."""
    if not isinstance(input_list, Iterable)\
            or isinstance(input_list, string_types):
        raise BirdVoxClassifyError('input_list must be a non-string iterable')
    file_list = []
    for item in input_list:
        if os.path.isfile(item):
            file_list.append(os.path.abspath(item))
        elif os.path.isdir(item):
            for fname in os.listdir(item):
                path = os.path.join(item, fname)
                if os.path.isfile(path):
                    file_list.append(path)
        else:
            raise BirdVoxClassifyError(
                'Could not find input at path {}'.format(item))

    return file_list


def run(inputs, output_dir=None, output_summary_path=None,
        model_name=DEFAULT_MODEL_NAME, batch_size=512, suffix="",
        logger_level=logging.INFO):
    """Runs classification model on input audio clips"""
    # Set logger level.
    logging.getLogger().setLevel(logger_level)

    if isinstance(inputs, string_types):
        file_list = [inputs]
    elif isinstance(inputs, Iterable):
        file_list = get_file_list(inputs)
    else:
        raise BirdVoxClassifyError('Invalid input: {}'.format(str(inputs)))

    if len(file_list) == 0:
        info_msg = 'birdvoxclassify: No WAV files found in {}. Aborting.'
        logging.info(info_msg.format(str(inputs)))
        sys.exit(-1)

    # Print header
    if output_dir:
        logging.info("birdvoxclassify: Output directory = " + output_dir)

    if not suffix == "":
        logging.info("birdvoxclassify: Suffix string = " + suffix)

    # Process all files in the arguments
    output = birdvoxclassify.process_file(
        file_list,
        output_dir=output_dir,
        output_summary_path=output_summary_path,
        model_name=model_name,
        batch_size=batch_size,
        suffix=suffix,
        logger_level=logger_level)

    logging.info('birdvoxclassify: Printing output.')
    logging.info(pformat(output))
    logging.info('birdvoxclassify: Done.')


def parse_args(args):
    """Parses CLI arguments"""
    parser = ArgumentParser(
        sys.argv[0],
        description=main.__doc__,
        formatter_class=RawDescriptionHelpFormatter)

    parser.add_argument(
        'inputs', nargs='*',
        help='Path or paths to files to process, or path to '
             'a directory of files to process.')

    parser.add_argument(
        '--output-dir', '-o', default=None, dest='output_dir',
        help='Directory to save individual output file(s)')

    parser.add_argument(
        '--output-summary-path', '-O', default=None, dest='output_summary_path',
        help='Directory to save individual output file(s)')

    parser.add_argument(
        '--model-name', '-c', default=DEFAULT_MODEL_NAME,
        dest='model_name',
        help='Name of bird species classifier model to be used.')

    parser.add_argument(
        '--batch-size', '-b', type=positive_int, default=512, dest='batch_size',
        help='Input batch size used by classifier model.'
    )

    parser.add_argument(
        '--suffix', '-s', default="", dest='suffix',
        help='String to append to the output filenames.'
             'The default value is the empty string.')

    parser.add_argument(
        '--quiet', '-q', action='store_true', dest='quiet',
        help='Print less messages on screen.')

    parser.add_argument(
        '--verbose', '-v', action='store_true', dest='verbose',
        help='Print timestamps of classified events.')

    parser.add_argument(
        '--version', '-V', action='store_true', dest='version',
        help='Print version number.')

    args = parser.parse_args(args)

    if args.quiet and args.verbose:
        raise BirdVoxClassifyError(
            'Command-line flags --quiet (-q) and --verbose (-v) '
            'are mutually exclusive.')

    return args


def main():
    """
    Classifies nocturnal flight calls from audio by means of the
    BirdVoxClassify deep learning model.
    """
    args = parse_args(sys.argv[1:])

    if args.version:
        print(birdvoxclassify.version.version)
        return

    if args.quiet:
        logger_level = 30
    elif args.verbose:
        logger_level = 20
    else:
        logger_level = 25

    run(args.inputs,
        output_dir=args.output_dir,
        output_summary_path=args.output_summary_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        suffix=args.suffix,
        logger_level=logger_level)


def positive_int(value):
    """An argparse-like method for accepting only positive number"""
    try:
        fvalue = int(value)
    except (ValueError, TypeError) as e:
        raise BirdVoxClassifyError(
            'Expected a positive int, error message: {}'.format(e))
    if fvalue <= 0:
        raise BirdVoxClassifyError('Expected a positive integer')
    return fvalue

import pytest
import os
import tempfile
import shutil
from birdvoxclassify.cli import (get_file_list, run, parse_args,
                                 main, positive_int)
from birdvoxclassify.version import version
from birdvoxclassify.birdvoxclassify_exceptions import BirdVoxClassifyError
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch


TEST_AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'data/audio')
CHIRP_PATH = os.path.join(TEST_AUDIO_DIR, 'synth_chirp.wav')

MODEL_SUFFIX = "taxonet_tv1hierarchical" \
                       "-2e7e1bbd434a35b3961e315cfe3832fc"
MODEL_NAME = "birdvoxclassify-{}".format(MODEL_SUFFIX)


def test_get_file_list():
    # Make test audio directory
    test_dir = tempfile.mkdtemp()
    test_a_path = os.path.join(test_dir, "a.wav")
    test_b_path = os.path.join(test_dir, "b.wav")
    test_c_path = os.path.join(test_dir, "c.wav")
    test_d_path = os.path.join(test_dir, "d.wav")

    shutil.copy(CHIRP_PATH, test_a_path)
    shutil.copy(CHIRP_PATH, test_b_path)
    shutil.copy(CHIRP_PATH, test_c_path)
    shutil.copy(CHIRP_PATH, test_d_path)

    try:
        # Test for valid list of file paths
        flist = get_file_list([CHIRP_PATH, CHIRP_PATH])
        assert len(flist) == 2
        assert flist[0] == CHIRP_PATH
        assert flist[1] == CHIRP_PATH

        # Test for valid directory
        flist = get_file_list([test_dir])
        assert len(flist) == 4
        flist = sorted(flist)
        assert flist[0] == test_a_path
        assert flist[1] == test_b_path
        assert flist[2] == test_c_path
        assert flist[3] == test_d_path

        # Test for combination of lists and folders
        flist = get_file_list([test_dir, CHIRP_PATH])
        assert len(flist) == 5

        # Make sure nonexistent path raises exception
        pytest.raises(BirdVoxClassifyError, get_file_list,
                      ['/fake/path/to/file'])

        # Make sure providing a string results in an exception
        pytest.raises(BirdVoxClassifyError, get_file_list, "not a list")
    finally:
        shutil.rmtree(test_dir)


def test_parse_args():
    # Test default values
    args = [CHIRP_PATH]
    args = parse_args(args)
    assert args.output_dir is None
    assert args.output_summary_path is None
    assert args.model_name == MODEL_NAME
    assert args.batch_size == 512
    assert args.suffix == ""
    assert args.quiet is False
    assert args.verbose is False

    # Test custom values
    args = [CHIRP_PATH,
            '-o', '/tmp/output/dir',
            '-O', '/tmp/summary.json',
            '-c', MODEL_NAME,
            '-b', '16',
            '-s', 'suffix',
            '-q']
    args = parse_args(args)
    assert args.output_dir == '/tmp/output/dir'
    assert args.output_summary_path == '/tmp/summary.json'
    assert args.model_name == MODEL_NAME
    assert args.batch_size == 16
    assert args.suffix == 'suffix'
    assert args.quiet is True
    assert args.verbose is False

    # Test clash between quiet and verbose
    args = [CHIRP_PATH,
            '-v',
            '-q']
    pytest.raises(BirdVoxClassifyError, parse_args, args)

    # Test failure with invalid batch sizes
    args = [CHIRP_PATH,
            '-b', '-1']
    pytest.raises(BirdVoxClassifyError, parse_args, args)
    args = [CHIRP_PATH,
            '-b', '2.5']
    pytest.raises(BirdVoxClassifyError, parse_args, args)


def test_positive_int():
    i = positive_int(1)
    assert i == 1
    assert type(i) == int
    i = positive_int("1")
    assert i == 1
    i = positive_int("20")
    assert i == 20

    invalid = [-1, "-1", "-20", "1.0", "1.5", "-1.0", "-1.5", None, []]
    for i in invalid:
        pytest.raises(BirdVoxClassifyError, positive_int, i)


def test_main(capsys):
    tempdir = tempfile.mkdtemp()

    arg_list = ['birdvoxclassify',
                CHIRP_PATH,
                '--output-dir', tempdir]
    with patch('sys.argv', arg_list):
        main()
    # Check output file created
    outfile = os.path.join(tempdir, 'synth_chirp.json')
    assert os.path.isfile(outfile)

    arg_list = ['birdvoxclassify',
                CHIRP_PATH,
                '-v',
                '--output-dir', tempdir]
    with patch('sys.argv', arg_list):
        main()
    # Check output file created
    outfile = os.path.join(tempdir, 'synth_chirp.json')
    assert os.path.isfile(outfile)

    arg_list = ['birdvoxclassify',
                CHIRP_PATH,
                '-q',
                '--output-dir', tempdir]
    with patch('sys.argv', arg_list):
        main()
    # Check output file created
    outfile = os.path.join(tempdir, 'synth_chirp.json')
    assert os.path.isfile(outfile)

    arg_list = ['birdvoxclassify', '-V']
    with patch('sys.argv', arg_list):
        main()
    captured = capsys.readouterr()
    assert captured.out == (version + '\n')

    shutil.rmtree(tempdir)


def test_script_main():
    # Duplicate regression test from test_run just to hit coverage
    tempdir = tempfile.mkdtemp()
    arg_list = ['birdvoxclassify',
                CHIRP_PATH,
                '--output-dir', tempdir]
    with patch('sys.argv', arg_list):
        import birdvoxclassify.__main__

    # check output file created
    outfile = os.path.join(tempdir, 'synth_chirp.json')
    assert os.path.isfile(outfile)


def test_run(capsys):
    # Test invalid input
    invalid_inputs = [None, 5, 1.0]
    for i in invalid_inputs:
        pytest.raises(BirdVoxClassifyError, run, i)

    # test empty input folder
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        tempdir = os.path.abspath(tempfile.mkdtemp())
        run([tempdir])
    shutil.rmtree(tempdir)

    # make sure it exited
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == -1

    # Since we are using logger, have to rethink this test
    # make sure it printed a message
    # captured = capsys.readouterr()
    # expected_message = 'birdvoxclassify: No WAV files found in {}. Aborting.\n'
    # expected_message = expected_message.format(str([tempdir]))
    # assert captured.out == expected_message

    # test string input
    string_input = CHIRP_PATH
    tempdir = tempfile.mkdtemp()
    run(string_input, output_dir=tempdir)
    outfile = os.path.join(tempdir, 'synth_chirp.json')
    assert os.path.exists(outfile)
    shutil.rmtree(tempdir)

    # test suffix
    string_input = CHIRP_PATH
    tempdir = tempfile.mkdtemp()
    run(string_input, output_dir=tempdir, suffix="suffix")
    outfile = os.path.join(tempdir, 'synth_chirp_suffix.json')
    assert os.path.exists(outfile)
    shutil.rmtree(tempdir)

    # test list input
    tempdir = tempfile.mkdtemp()
    run([CHIRP_PATH], output_dir=tempdir)
    outfile = os.path.join(tempdir, 'synth_chirp.json')
    assert os.path.exists(outfile)
    shutil.rmtree(tempdir)

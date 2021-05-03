import hashlib
import json
import logging
import os
import pathlib
import shutil
import sys
import tempfile
import pytest
import numpy as np
import soundfile as sf
from scipy.signal.windows import get_window
from numbers import Real

from birdvoxclassify import *
from birdvoxclassify.core import apply_hierarchical_consistency, \
                                 _validate_prediction, \
                                 _validate_batch_pred_list
from birdvoxclassify.birdvoxclassify_exceptions import BirdVoxClassifyError

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_DIR = os.path.join(PROJECT_DIR, "birdvoxclassify")

TEST_AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'data/audio')
CHIRP_PATH = os.path.join(TEST_AUDIO_DIR, 'synth_chirp.wav')

RES_DIR = os.path.join(MODULE_DIR, "resources")
TAX_DIR = os.path.join(RES_DIR, "taxonomy")
MODELS_DIR = os.path.join(RES_DIR, "models")
TAXV1_HIERARCHICAL_PATH = os.path.join(TAX_DIR, "tv1hierarchical.json")
TAXV1_FINE_PATH = os.path.join(TAX_DIR, "tv1fine.json")

MODEL_SUFFIX = "flat-multitask-convnet-v2_" \
               "tv1hierarchical-2e7e1bbd434a35b3961e315cfe3832fc"
MODEL_NAME = "birdvoxclassify-{}".format(MODEL_SUFFIX)


def test_process_file():
    test_output_dir = tempfile.mkdtemp()
    test_audio_dir = tempfile.mkdtemp()
    classifier = load_classifier(MODEL_NAME)
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)
    test_output_summary_path = os.path.join(test_output_dir, "summary.json")
    test_output_path = get_output_path(CHIRP_PATH, '.json',
                                       test_output_dir)
    suffix_test_output_path = get_output_path(CHIRP_PATH,
                                              'suffix.json',
                                              test_output_dir)

    try:
        # Test with defaults
        output = process_file(CHIRP_PATH, model_name=MODEL_NAME)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict

        # Test with selecting best candidates
        output = process_file(CHIRP_PATH, model_name=MODEL_NAME,
                              select_best_candidates=True)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict
            # There should only be one candidate per level
            for level, cand_dict in v.items():
                assert isinstance(level, str)
                assert isinstance(cand_dict, dict)

        # Test with list
        output = process_file([CHIRP_PATH], model_name=MODEL_NAME)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict

        # Test with given classifier and taxonomy
        output = process_file([CHIRP_PATH], classifier=classifier,
                              taxonomy=taxonomy)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict

        # Test output_dir
        output = process_file([CHIRP_PATH], output_dir=test_output_dir,
                              classifier=classifier)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict
        with open(test_output_path, 'r') as f:
            f_output = json.load(f)
        assert next(iter(output.values())) == f_output

        # Test output dir with suffix
        output = process_file([CHIRP_PATH], output_dir=test_output_dir,
                              classifier=classifier, suffix='suffix')
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict
        with open(suffix_test_output_path, 'r') as f:
            f_output = json.load(f)
        assert next(iter(output.values())) == f_output

        # Test output summary file
        output = process_file([CHIRP_PATH],
                              output_summary_path=test_output_summary_path,
                              classifier=classifier)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict
        with open(test_output_summary_path, 'r') as f:
            f_output = json.load(f)
        assert output == f_output

        test_a_path = os.path.join(test_audio_dir, "a.wav")
        test_b_path = os.path.join(test_audio_dir, "b.wav")
        test_c_path = os.path.join(test_audio_dir, "c.wav")
        test_d_path = os.path.join(test_audio_dir, "d.wav")

        shutil.copy(CHIRP_PATH, test_a_path)
        shutil.copy(CHIRP_PATH, test_b_path)
        shutil.copy(CHIRP_PATH, test_c_path)
        shutil.copy(CHIRP_PATH, test_d_path)

        test_audio_list = [test_a_path, test_b_path, test_c_path, test_d_path]

        # Test multiple files
        output = process_file(test_audio_list, classifier=classifier)
        assert type(output) == dict
        assert len(output) == len(test_audio_list)
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict

        # Test with different batch_sizes
        output = process_file(test_audio_list, classifier=classifier, batch_size=2)
        assert type(output) == dict
        assert len(output) == len(test_audio_list)
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict

        # Make sure we create the output dir if it doesn't exist
        shutil.rmtree(test_output_dir)
        output = process_file([CHIRP_PATH], output_dir=test_output_dir,
                              classifier=classifier)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, str)
            assert type(v) == dict
        with open(test_output_path, 'r') as f:
            f_output = json.load(f)
        assert next(iter(output.values())) == f_output
    finally:
        shutil.rmtree(test_output_dir)
        shutil.rmtree(test_audio_dir)


def test_format_pred():
    taxonomy = load_taxonomy(TAXV1_FINE_PATH)

    pred = np.random.random((15,))
    pred /= pred.sum()
    pred_list = [pred]

    output_ids = [
        "1.1.1",
        "1.1.2",
        "1.1.3",
        "1.1.4",
        "1.2.1",
        "1.3.1",
        "1.3.2",
        "1.4.1",
        "1.4.2",
        "1.4.3",
        "1.4.4",
        "1.4.5",
        "1.4.6",
        "1.4.7",
        "other"
    ]

    exp_output = {'fine': {}}
    for idx, ref_id in enumerate(output_ids):
        if ref_id == "other":
            node = {
                "common_name": "other",
                "scientific_name": "other",
                "taxonomy_level_names": "fine",
                "taxonomy_level_aliases": {},
                'child_ids': taxonomy["output_encoding"]["fine"][idx]["ids"]
            }
        else:
            node = get_taxonomy_node(ref_id, taxonomy)

        exp_output['fine'][ref_id] = {'probability': pred[idx]}
        exp_output['fine'][ref_id].update(node)

    output = format_pred(pred_list, taxonomy)
    for ref_id in output_ids:
        assert output['fine'][ref_id]["common_name"] \
            == exp_output['fine'][ref_id]["common_name"]
        assert output['fine'][ref_id]["scientific_name"] \
            == exp_output['fine'][ref_id]["scientific_name"]
        assert output['fine'][ref_id]["taxonomy_level_names"] \
            == exp_output['fine'][ref_id]["taxonomy_level_names"]
        assert output['fine'][ref_id]["taxonomy_level_aliases"] \
            == exp_output['fine'][ref_id]["taxonomy_level_aliases"]
        assert output['fine'][ref_id]["child_ids"] \
            == exp_output['fine'][ref_id]["child_ids"]
        assert np.isclose(output['fine'][ref_id]["probability"],
                          exp_output['fine'][ref_id]["probability"])

    # Test when we have a batch dimension of 1
    pred_list = [pred[np.newaxis, :]]

    output = format_pred(pred_list, taxonomy)
    for ref_id in output_ids:
        assert output['fine'][ref_id]["common_name"] \
               == exp_output['fine'][ref_id]["common_name"]
        assert output['fine'][ref_id]["scientific_name"] \
               == exp_output['fine'][ref_id]["scientific_name"]
        assert output['fine'][ref_id]["taxonomy_level_names"] \
               == exp_output['fine'][ref_id]["taxonomy_level_names"]
        assert output['fine'][ref_id]["taxonomy_level_aliases"] \
               == exp_output['fine'][ref_id]["taxonomy_level_aliases"]
        assert output['fine'][ref_id]["child_ids"] \
               == exp_output['fine'][ref_id]["child_ids"]
        assert np.isclose(output['fine'][ref_id]["probability"],
                          exp_output['fine'][ref_id]["probability"])

    # Make sure we fail when batch dimension is greater than 1
    pred_list = [np.tile(pred, (2, 1))]
    pytest.raises(BirdVoxClassifyError, format_pred, pred_list, taxonomy)

    # Make sure we fail with wrong taxonomy
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)
    pred_list = [pred]
    pytest.raises(BirdVoxClassifyError, format_pred, pred_list, taxonomy)

    # Test the hierarchical case
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)
    fine_pred = np.random.random((15,))
    medium_pred = np.random.random((5,))
    coarse_pred = np.random.random((1,))
    pred_list = [coarse_pred, medium_pred, fine_pred]
    output = format_pred(pred_list, taxonomy)
    exp_output = {}
    for level_idx, (level, encoding_list) \
            in enumerate(taxonomy['output_encoding'].items()):
        exp_output[level] = {}
        for idx, encoding_item in enumerate(encoding_list):
            if len(encoding_item["ids"]) > 1:
                ref_id = "other"
                node = {
                    "common_name": "other",
                    "scientific_name": "other",
                    "taxonomy_level_names": level,
                    "taxonomy_level_aliases": {},
                    'child_ids': encoding_item["ids"]
                }
            else:
                ref_id = encoding_item["ids"][0]
                node = get_taxonomy_node(ref_id, taxonomy)

            if level == 'coarse' and idx == 1:
                exp_output[level][ref_id] = {
                    'probability': 1 - pred_list[level_idx][0]}
            else:
                exp_output[level][ref_id] = {'probability': pred_list[level_idx][idx]}
            exp_output[level][ref_id].update(node)
    for level, encoding_list in taxonomy["output_encoding"].items():
        for encoding_item in encoding_list:
            if len(encoding_item["ids"]) > 1:
                ref_id = "other"
            else:
                ref_id = encoding_item["ids"][0]

            assert output[level][ref_id]["common_name"] \
                   == exp_output[level][ref_id]["common_name"]
            assert output[level][ref_id]["scientific_name"] \
                   == exp_output[level][ref_id]["scientific_name"]
            assert output[level][ref_id]["taxonomy_level_names"] \
                   == exp_output[level][ref_id]["taxonomy_level_names"]
            assert output[level][ref_id]["taxonomy_level_aliases"] \
                   == exp_output[level][ref_id]["taxonomy_level_aliases"]
            assert output[level][ref_id]["child_ids"] \
                   == exp_output[level][ref_id]["child_ids"]
            assert np.isclose(output[level][ref_id]["probability"],
                              exp_output[level][ref_id]["probability"])

    # Make sure real prediction makes it through the pipeline
    audio, sr = sf.read(CHIRP_PATH, dtype='float64')
    classifier = load_classifier(MODEL_NAME)
    pcen = compute_pcen(audio, sr)
    pred = predict(pcen, classifier, logging.INFO)
    output = format_pred(pred, taxonomy)


def test_format_pred_batch():
    taxonomy = load_taxonomy(TAXV1_FINE_PATH)

    pred = np.random.random((15,))
    pred /= pred.sum()

    output_ids = [
        "1.1.1",
        "1.1.2",
        "1.1.3",
        "1.1.4",
        "1.2.1",
        "1.3.1",
        "1.3.2",
        "1.4.1",
        "1.4.2",
        "1.4.3",
        "1.4.4",
        "1.4.5",
        "1.4.6",
        "1.4.7",
        "other"
    ]

    exp_output = {'fine': {}}
    for idx, ref_id in enumerate(output_ids):
        if ref_id == "other":
            node = {
                "common_name": "other",
                "scientific_name": "other",
                "taxonomy_level_names": "fine",
                "taxonomy_level_aliases": {},
                'child_ids': taxonomy["output_encoding"]["fine"][idx]["ids"]
            }
        else:
            node = get_taxonomy_node(ref_id, taxonomy)

        exp_output['fine'][ref_id] = {'probability': pred[idx]}
        exp_output['fine'][ref_id].update(node)

    batch_pred_list = [np.tile(pred, (10, 1))]
    exp_output_batch = [exp_output] * 10

    output_batch = format_pred_batch(batch_pred_list=batch_pred_list,
                               taxonomy=taxonomy)
    for idx, output in enumerate(output_batch):
        for ref_id in output_ids:
            exp_output = exp_output_batch[idx]
            assert output['fine'][ref_id]["common_name"] \
                == exp_output['fine'][ref_id]["common_name"]
            assert output['fine'][ref_id]["scientific_name"] \
                == exp_output['fine'][ref_id]["scientific_name"]
            assert output['fine'][ref_id]["taxonomy_level_names"] \
                == exp_output['fine'][ref_id]["taxonomy_level_names"]
            assert output['fine'][ref_id]["taxonomy_level_aliases"] \
                == exp_output['fine'][ref_id]["taxonomy_level_aliases"]
            assert output['fine'][ref_id]["child_ids"] \
                == exp_output['fine'][ref_id]["child_ids"]
            assert np.isclose(output['fine'][ref_id]["probability"],
                              exp_output['fine'][ref_id]["probability"])

    pytest.raises(BirdVoxClassifyError, format_pred_batch,
                  [np.tile(pred, (10, 1)), np.tile(pred, (5, 1))], taxonomy)


def test_get_taxonomy_node():
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)

    ref_id = "1"
    node = get_taxonomy_node(ref_id, taxonomy)
    assert "id" in node
    assert "common_name" in node
    assert "scientific_name" in node
    assert "taxonomy_level_names" in node
    assert "taxonomy_level_aliases" in node
    assert "child_ids" in node

    assert node["id"] == ref_id
    assert isinstance(node["common_name"], str)
    assert isinstance(node["scientific_name"], str)
    assert node["taxonomy_level_names"] == "coarse"
    assert isinstance(node["taxonomy_level_aliases"], dict)
    assert type(node["child_ids"]) == list
    assert len(node["child_ids"]) >= 1

    ref_id = "1.1"
    node = get_taxonomy_node(ref_id, taxonomy)
    assert "id" in node
    assert "common_name" in node
    assert "scientific_name" in node
    assert "taxonomy_level_names" in node
    assert "taxonomy_level_aliases" in node
    assert "child_ids" in node

    assert node["id"] == ref_id
    assert isinstance(node["common_name"], str)
    assert isinstance(node["scientific_name"], str)
    assert node["taxonomy_level_names"] == "medium"
    assert isinstance(node["taxonomy_level_aliases"], dict)
    assert type(node["child_ids"]) == list
    assert len(node["child_ids"]) >= 1

    ref_id = "1.1.1"
    node = get_taxonomy_node(ref_id, taxonomy)
    assert "id" in node
    assert "common_name" in node
    assert "scientific_name" in node
    assert "taxonomy_level_names" in node
    assert "taxonomy_level_aliases" in node
    assert "child_ids" in node

    assert node["id"] == ref_id
    assert isinstance(node["common_name"], str)
    assert isinstance(node["scientific_name"], str)
    assert node["taxonomy_level_names"] == "fine"
    assert isinstance(node["taxonomy_level_aliases"], dict)
    assert type(node["child_ids"]) == list
    assert len(node["child_ids"]) == 0

    ref_id = "other"
    node = get_taxonomy_node(ref_id, taxonomy)
    assert node == {"id": "other"}

    # Check for invalid inputs
    pytest.raises(BirdVoxClassifyError, get_taxonomy_node, "0", taxonomy)
    pytest.raises(TypeError, get_taxonomy_node, "1", [])
    pytest.raises(BirdVoxClassifyError, get_taxonomy_node, "1",
                  {'taxonomy': []})
    pytest.raises(BirdVoxClassifyError, get_taxonomy_node, "1",
                  {'taxonomy': [{}]})


def test_batch_generator():
    pcen_settings = get_pcen_settings()

    # Test invalid inputs
    with pytest.raises(BirdVoxClassifyError) as e:
        gen = batch_generator(['/invalid/path.wav'], batch_size=512)
        next(gen)
    with pytest.raises(BirdVoxClassifyError) as e:
        gen = batch_generator(['/invalid/path.wav'], batch_size=-1)
        next(gen)
    with pytest.raises(BirdVoxClassifyError) as e:
        gen = batch_generator(['/invalid/path.wav'], batch_size=512.0)
        next(gen)
    with pytest.raises(BirdVoxClassifyError) as e:
        gen = batch_generator([], batch_size=512)
        next(gen)

    # Test empty file
    empty_path = 'empty.wav'
    pathlib.Path(empty_path).touch()

    try:
        with pytest.raises(BirdVoxClassifyError) as e:
            gen = batch_generator([empty_path], batch_size=512)
            next(gen)
    finally:
        os.remove(empty_path)

    gen = batch_generator([CHIRP_PATH]*10, batch_size=10)
    batch, batch_filepaths = next(gen)
    assert type(batch) == np.ndarray
    assert batch.shape == (10, pcen_settings['top_freq_id'],
                           pcen_settings['n_hops'], 1)
    assert len(batch_filepaths) == 10

    gen = batch_generator([CHIRP_PATH], batch_size=10)
    batch, batch_filepaths = next(gen)
    assert type(batch) == np.ndarray
    assert batch.shape == (1, pcen_settings['top_freq_id'],
                           pcen_settings['n_hops'], 1)
    assert len(batch_filepaths) == 1


def test_compute_pcen():
    pcen_settings = get_pcen_settings()

    audio, sr = sf.read(CHIRP_PATH, dtype='float64')
    pcenf64 = compute_pcen(audio, sr)
    assert pcenf64.dtype == np.float32
    assert pcenf64.shape == (pcen_settings['top_freq_id'],
                             pcen_settings['n_hops'], 1)
    pcenf64_r = compute_pcen(audio, sr, input_format=False)
    assert pcenf64_r.dtype == np.float32
    assert pcenf64_r.ndim == 2
    assert pcenf64_r.shape[0] == pcen_settings['top_freq_id']

    audio, sr = sf.read(CHIRP_PATH, dtype='float32')
    pcenf32 = compute_pcen(audio, sr)
    assert pcenf32.dtype == np.float32
    assert pcenf32.shape == (pcen_settings['top_freq_id'],
                             pcen_settings['n_hops'], 1)
    pcenf32_r = compute_pcen(audio, sr, input_format=False)
    assert pcenf32_r.dtype == np.float32
    assert pcenf32_r.ndim == 2
    assert pcenf32_r.shape[0] == pcen_settings['top_freq_id']

    audio, sr = sf.read(CHIRP_PATH, dtype='int16')
    pceni16 = compute_pcen(audio, sr)
    assert pceni16.dtype == np.float32
    assert pceni16.shape == (pcen_settings['top_freq_id'],
                             pcen_settings['n_hops'], 1)
    pceni16_r = compute_pcen(audio, sr, input_format=False)
    assert pceni16_r.dtype == np.float32
    assert pceni16_r.ndim == 2
    assert pceni16_r.shape[0] == pcen_settings['top_freq_id']

    audio, sr = sf.read(CHIRP_PATH, dtype='int32')
    pceni32 = compute_pcen(audio, sr)
    assert pceni32.dtype == np.float32
    assert pceni32.shape == (pcen_settings['top_freq_id'],
                             pcen_settings['n_hops'], 1)
    pceni32_r = compute_pcen(audio, sr, input_format=False)
    assert pceni32_r.dtype == np.float32
    assert pceni32_r.ndim == 2
    assert pceni32_r.shape[0] == pcen_settings['top_freq_id']

    # Make sure PCEN values are similar for different input representations
    assert np.allclose(pcenf64, pcenf32, rtol=1e-5, atol=1e-5)
    assert np.allclose(pcenf64, pceni16, rtol=1e-5, atol=1e-5)
    assert np.allclose(pcenf64, pceni32, rtol=1e-5, atol=1e-5)

    # Make sure that padding is handled with short audio
    short_audio = np.random.random((10,))
    short_pcen = compute_pcen(short_audio, sr)
    assert short_pcen.dtype == np.float32
    assert short_pcen.shape == (pcen_settings['top_freq_id'],
                                pcen_settings['n_hops'], 1)

    # Make sure unsigned ints raise an error
    pytest.raises(BirdVoxClassifyError, compute_pcen,
                  audio.astype('uint32'), sr)


def test_predict():
    classifier = load_classifier(MODEL_NAME)

    audio, sr = sf.read(CHIRP_PATH, dtype='float64')
    pcen = compute_pcen(audio, sr)
    pred = predict(pcen, classifier, logging.INFO)
    assert type(pred) == list
    assert pred[0].shape == (1, 1)
    assert pred[1].shape == (1, 5)
    assert pred[2].shape == (1, 15)

    gen = batch_generator([CHIRP_PATH]*10, batch_size=10)
    batch, batch_filepaths = next(gen)
    pred = predict(batch, classifier, logging.INFO)
    assert type(pred) == list
    assert pred[0].shape == (10, 1)
    assert pred[1].shape == (10, 5)
    assert pred[2].shape == (10, 15)

    # Test invalid inputs
    inv_pcen = compute_pcen(audio, sr, input_format=False)[..., np.newaxis]
    pytest.raises(BirdVoxClassifyError, predict, inv_pcen, classifier, logging.INFO)
    pytest.raises(BirdVoxClassifyError, predict, np.array([1, 2, 3, 4]), classifier,
                  logging.INFO)
    pytest.raises(BirdVoxClassifyError, predict, np.zeros((1, 42, 104, 1)),
                  classifier, logging.INFO)
    pytest.raises(BirdVoxClassifyError, predict, np.zeros((1, 120, 42, 1)),
                  classifier, logging.INFO)
    pytest.raises(BirdVoxClassifyError, predict, np.zeros((1, 120, 104, 42)),
                  classifier, logging.INFO)


def test_get_output_path():
    filepath = '/path/to/file/test.wav'
    output_dir = '/output/dir'

    exp_output_path = '/path/to/file/test.npz'
    output_path = get_output_path(filepath, ".npz", None)
    assert output_path == exp_output_path

    exp_output_path = '/path/to/file/test_suffix.npz'
    output_path = get_output_path(filepath, "suffix.npz", None)
    assert output_path == exp_output_path

    exp_output_path = '/output/dir/test.npz'
    output_path = get_output_path(filepath, ".npz", output_dir)
    assert output_path == exp_output_path

    exp_output_path = '/output/dir/test_suffix.npz'
    output_path = get_output_path(filepath, "suffix.npz", output_dir)
    assert output_path == exp_output_path


def test_get_pcen_settings():
    settings = get_pcen_settings()

    assert type(settings) == dict

    assert 'sr' in settings
    assert isinstance(settings['sr'], Real)
    assert settings['sr'] > 0

    assert 'fmin' in settings
    assert isinstance(settings['fmin'], Real)
    assert settings['fmin'] > 0

    assert 'fmax' in settings
    assert isinstance(settings['fmax'], Real)
    assert settings['fmax'] > settings['fmin']
    assert settings['sr'] / 2.0 >= settings['fmax']

    assert 'hop_length' in settings
    assert isinstance(settings['hop_length'], int)
    assert settings['hop_length'] > 0

    assert 'n_fft' in settings
    assert isinstance(settings['n_fft'], int)
    assert settings['n_fft'] > 0

    assert 'n_mels' in settings
    assert isinstance(settings['n_mels'], int)
    assert settings['n_fft'] > settings['n_mels'] > 0

    assert 'pcen_delta' in settings
    assert isinstance(settings['pcen_delta'], float)
    assert settings['pcen_delta'] > 0

    assert 'pcen_time_constant' in settings
    assert isinstance(settings['pcen_time_constant'], float)
    assert settings['pcen_time_constant'] > 0

    assert 'pcen_norm_exponent' in settings
    assert isinstance(settings['pcen_norm_exponent'], float)
    assert settings['pcen_norm_exponent'] > 0

    assert 'pcen_power' in settings
    assert isinstance(settings['pcen_power'], float)
    assert settings['pcen_power'] > 0

    assert 'top_freq_id' in settings
    assert isinstance(settings['top_freq_id'], int)
    assert settings['top_freq_id'] > 0

    assert 'win_length' in settings
    assert isinstance(settings['win_length'], int)
    assert settings['win_length'] > 0

    assert 'n_hops' in settings
    assert isinstance(settings['n_hops'], int)
    assert settings['n_hops'] > 0

    assert 'window' in settings
    assert isinstance(settings['window'], str)
    # Make sure window is valid
    get_window(settings['window'], 5)


def test_get_model_path():
    test_model_name = "test_model_name"
    exp_model_path = os.path.join(RES_DIR, "models", test_model_name + '.h5')
    model_path = get_model_path(test_model_name)
    assert os.path.abspath(model_path) == os.path.abspath(exp_model_path)

    # Test Python 3.8 handling
    test_model_name = "birdvoxclassify_test_model_name"
    if sys.version_info.major == 3 and sys.version_info.minor == 8:
        exp_test_model_name = "birdvoxclassify-py3pt8_test_model_name"
    else:
        exp_test_model_name = "birdvoxclassify_test_model_name"
    exp_model_path = os.path.join(RES_DIR, "models", exp_test_model_name + '.h5')
    model_path = get_model_path(test_model_name)
    assert os.path.abspath(model_path) == os.path.abspath(exp_model_path)


def test_load_classifier():
    classifier = load_classifier(MODEL_NAME)

    # Test invalid inputs
    invalid_path = get_model_path("invalid-classifier")
    with open(invalid_path, "w") as f:
        f.write("INVALID")

    try:
        pytest.raises(BirdVoxClassifyError, load_classifier, "invalid-classifier")
    finally:
        os.remove(invalid_path)

    pytest.raises(BirdVoxClassifyError, load_classifier, "/invalid/path")


def test_get_taxonomy_path():
    # Make sure that the correct taxonomy path is returned
    taxonomy_version = "v1234"
    test_content = 'hello world!'
    hash_md5 = hashlib.md5()
    hash_md5.update(test_content.encode())

    exp_md5sum = hash_md5.hexdigest()
    exp_taxonomy_path = os.path.join(TAX_DIR, taxonomy_version + ".json")
    with open(exp_taxonomy_path, 'w') as f:
        f.write(test_content)

    model_name = "test-model-name_{}-{}".format(taxonomy_version, exp_md5sum)
    try:
        taxonomy_path = get_taxonomy_path(model_name)
        assert os.path.abspath(taxonomy_path) == os.path.abspath(exp_taxonomy_path)

        # Make sure that an error is raised when md5sum doesn't match
        hash_md5 = hashlib.md5()
        hash_md5.update("different".encode())
        diff_md5sum = hash_md5.hexdigest()
        model_name = "test-model-name_{}-{}".format(taxonomy_version, diff_md5sum)
        pytest.raises(BirdVoxClassifyError, get_taxonomy_path,
                      model_name)
    finally:
        os.remove(exp_taxonomy_path)


def test_validate_batch_pred_list():
    n_examples = 5
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)

    # Test valid batch
    batch_pred_list = []
    for level, encoding_list in taxonomy["output_encoding"].items():
        n_classes = len(encoding_list)
        batch_pred = np.random.random((n_examples, n_classes))
        batch_pred_list.append(batch_pred)
    _validate_batch_pred_list(batch_pred_list)

    # Test invalid batch
    batch_pred_list = []
    for idx, (level, encoding_list) in enumerate(taxonomy["output_encoding"].items()):
        n_classes = len(encoding_list)
        if idx == 0:
            batch_pred = np.random.random((10, n_classes))
        else:
            batch_pred = np.random.random((n_examples, n_classes))
        batch_pred_list.append(batch_pred)
    pytest.raises(BirdVoxClassifyError, _validate_batch_pred_list, batch_pred_list)


def test_validate_prediction():
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)

    # Test valid prediction
    pred_list = []
    for level, encoding_list in taxonomy["output_encoding"].items():
        n_classes = len(encoding_list)
        pred = np.random.random((n_classes,))
        pred_list.append(pred)
    formatted_pred_dict = format_pred(pred_list, taxonomy)
    _validate_prediction(pred_list, taxonomy)
    _validate_prediction([pred[np.newaxis, :] for pred in pred_list], taxonomy)
    _validate_prediction(formatted_pred_dict, taxonomy)

    # Test invalid batches
    pytest.raises(BirdVoxClassifyError, _validate_prediction,
                  pred_list * 2, taxonomy)

    pred_list = []
    for level, encoding_list in taxonomy["output_encoding"].items():
        n_classes = len(encoding_list)
        # Ensure number of classes is different than expected
        pred = np.random.random((n_classes + 5,))
        pred_list.append(pred)
    pytest.raises(BirdVoxClassifyError, _validate_prediction,
                  pred_list, taxonomy)

    # Make sure a real prediction makes it through the pipeline with no problem
    output = process_file(CHIRP_PATH, model_name=MODEL_NAME)
    formatted_pred_dict = [x for x in output.values()][0]
    _validate_prediction(formatted_pred_dict, taxonomy)


def test_get_batch_best_candidates():
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)

    # Non-HC and HC Cand: "1"
    coarse_pred = np.array([0.9])
    # Non-HC and HC Cand: "1.4"
    medium_pred = np.array([0.1, 0.0, 0.0, 0.8, 0.2])
    # Non-HC Cand: "1.1.1", HC Cand: "1.4.3"
    fine_pred = np.array([0.7, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.6,
                          0.0, 0.0, 0.0, 0.0, 0.3])
    pred_list = [coarse_pred, medium_pred, fine_pred]
    formatted_pred_dict = format_pred(pred_list, taxonomy)

    exp_output = {}
    exp_output['coarse'] = {'probability': coarse_pred[0]}
    exp_output['coarse'].update(get_taxonomy_node("1", taxonomy))
    exp_output['medium'] = {'probability': medium_pred[3]}
    exp_output['medium'].update(get_taxonomy_node("1.4", taxonomy))
    exp_output['fine'] = {'probability': fine_pred[0]}
    exp_output['fine'].update(get_taxonomy_node("1.1.1", taxonomy))

    batch_pred_list = [np.tile(pred, (10, 1)) for pred in pred_list]
    batch_formatted_pred_list = [formatted_pred_dict] * 10
    exp_output_batch = [exp_output] * 10

    batch_best_cand_list = get_batch_best_candidates(
        batch_formatted_pred_list=batch_formatted_pred_list)
    batch_best_cand_list_2 = get_batch_best_candidates(
        batch_pred_list=batch_pred_list,
        taxonomy=taxonomy
    )

    # Make sure output is as expected
    for idx, best_cand_dict in enumerate(batch_best_cand_list):
        exp_output = exp_output_batch[idx]
        assert set(exp_output.keys()) == set(best_cand_dict.keys())
        for level in exp_output.keys():
            assert best_cand_dict[level]["common_name"] \
                   == exp_output[level]["common_name"]
            assert best_cand_dict[level]["scientific_name"] \
                   == exp_output[level]["scientific_name"]
            assert best_cand_dict[level]["taxonomy_level_names"] \
                   == exp_output[level]["taxonomy_level_names"]
            assert best_cand_dict[level]["taxonomy_level_aliases"] \
                   == exp_output[level]["taxonomy_level_aliases"]
            assert best_cand_dict[level]["child_ids"] \
                   == exp_output[level]["child_ids"]
            assert np.isclose(best_cand_dict[level]["probability"],
                              exp_output[level]["probability"])

    # Make sure unformatted and formatted batches both produce the same result
    assert len(batch_best_cand_list) == len(batch_best_cand_list_2)
    for idx, best_cand_dict in enumerate(batch_best_cand_list):
        best_cand_dict_2 = batch_best_cand_list_2[idx]
        assert set(best_cand_dict_2.keys()) == set(best_cand_dict.keys())
        for level in best_cand_dict_2.keys():
            assert best_cand_dict[level]["common_name"] \
                   == best_cand_dict_2[level]["common_name"]
            assert best_cand_dict[level]["scientific_name"] \
                   == best_cand_dict_2[level]["scientific_name"]
            assert best_cand_dict[level]["taxonomy_level_names"] \
                   == best_cand_dict_2[level]["taxonomy_level_names"]
            assert best_cand_dict[level]["taxonomy_level_aliases"] \
                   == best_cand_dict_2[level]["taxonomy_level_aliases"]
            assert best_cand_dict[level]["child_ids"] \
                   == best_cand_dict_2[level]["child_ids"]
            assert np.isclose(best_cand_dict[level]["probability"],
                              best_cand_dict_2[level]["probability"])

    # Check invalid inputs
    pytest.raises(BirdVoxClassifyError, get_batch_best_candidates,
                  batch_pred_list=batch_pred_list,
                  batch_formatted_pred_list=batch_formatted_pred_list)
    pytest.raises(BirdVoxClassifyError, get_batch_best_candidates,
                  batch_formatted_pred_list=batch_formatted_pred_list,
                  taxonomy=None, hierarchical_consistency=True)


def test_get_best_candidates():
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)

    # Non-HC and HC Cand: "1"
    coarse_pred = np.array([0.9])
    # Non-HC and HC Cand: "1.4"
    medium_pred = np.array([0.1, 0.0, 0.0, 0.8, 0.2])
    # Non-HC Cand: "1.1.1", HC Cand: "1.4.3"
    fine_pred = np.array([0.7, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.6,
                          0.0, 0.0, 0.0, 0.0, 0.3])
    pred_list = [coarse_pred, medium_pred, fine_pred]
    formatted_pred_dict = format_pred(pred_list, taxonomy)
    out1 = get_best_candidates(formatted_pred_dict=formatted_pred_dict)

    exp_output = {}
    exp_output['coarse'] = {'probability': coarse_pred[0]}
    exp_output['coarse'].update(get_taxonomy_node("1", taxonomy))
    exp_output['medium'] = {'probability': medium_pred[3]}
    exp_output['medium'].update(get_taxonomy_node("1.4", taxonomy))
    exp_output['fine'] = {'probability': fine_pred[0]}
    exp_output['fine'].update(get_taxonomy_node("1.1.1", taxonomy))

    # Make sure output is in expected format
    assert set(out1.keys()) == set(taxonomy["output_encoding"].keys())
    for level, cand_dict in out1.items():
        assert isinstance(cand_dict, dict)
        assert out1[level]["common_name"] == exp_output[level]["common_name"]
        assert out1['fine']["scientific_name"] \
               == exp_output['fine']["scientific_name"]
        assert out1['fine']["taxonomy_level_names"] \
               == exp_output['fine']["taxonomy_level_names"]
        assert out1['fine']["taxonomy_level_aliases"] \
               == exp_output['fine']["taxonomy_level_aliases"]
        assert out1['fine']["child_ids"] \
               == exp_output['fine']["child_ids"]
        assert np.isclose(out1['fine']["probability"],
                          exp_output['fine']["probability"])

    # Make sure passing in pred_list or formatted_pred_dict results in the same
    # output
    out2 = get_best_candidates(pred_list=pred_list, taxonomy=taxonomy)
    for level in out1.keys():
        assert set(out1[level].keys()) == set(out2[level].keys())
        for k in out1[level].keys():
            assert out1[level][k] == out2[level][k]

    # Make sure hierarchical consistency is correctly applied
    out3 = get_best_candidates(formatted_pred_dict=formatted_pred_dict,
                               hierarchical_consistency=True,
                               taxonomy=taxonomy)
    exp_output = {}
    exp_output['coarse'] = {'probability': coarse_pred[0]}
    exp_output['coarse'].update(get_taxonomy_node("1", taxonomy))
    exp_output['medium'] = {'probability': medium_pred[3]}
    exp_output['medium'].update(get_taxonomy_node("1.4", taxonomy))
    exp_output['fine'] = {'probability': fine_pred[9]}
    exp_output['fine'].update(get_taxonomy_node("1.4.3", taxonomy))

    # Make sure output is in expected format
    assert set(out1.keys()) == set(taxonomy["output_encoding"].keys())
    for level, cand_dict in out1.items():
        assert isinstance(cand_dict, dict)
        assert out3[level]["common_name"] == exp_output[level]["common_name"]
        assert out3['fine']["scientific_name"] \
               == exp_output['fine']["scientific_name"]
        assert out3['fine']["taxonomy_level_names"] \
               == exp_output['fine']["taxonomy_level_names"]
        assert out3['fine']["taxonomy_level_aliases"] \
               == exp_output['fine']["taxonomy_level_aliases"]
        assert out3['fine']["child_ids"] \
               == exp_output['fine']["child_ids"]
        assert np.isclose(out3['fine']["probability"],
                          exp_output['fine']["probability"])

    # Test invalid inputs
    pytest.raises(BirdVoxClassifyError, get_best_candidates,
                  pred_list=pred_list, formatted_pred_dict=formatted_pred_dict)
    pytest.raises(BirdVoxClassifyError, get_best_candidates,
                  formatted_pred_dict=formatted_pred_dict,
                  taxonomy=None, hierarchical_consistency=True)
    pytest.raises(BirdVoxClassifyError, get_best_candidates,
                  pred_list=pred_list, taxonomy=None)

    # Make sure a real prediction makes it through the pipeline with no problem
    output = process_file(CHIRP_PATH, model_name=MODEL_NAME)
    formatted_pred_dict = [x for x in output.values()][0]
    out1 = get_best_candidates(formatted_pred_dict=formatted_pred_dict)
    out2 = get_best_candidates(formatted_pred_dict=formatted_pred_dict,
                               hierarchical_consistency=True,
                               taxonomy=taxonomy)


def test_apply_hierarchial_consistency():
    taxonomy = load_taxonomy(TAXV1_HIERARCHICAL_PATH)

    # HC Cand: "1"
    coarse_pred = np.array([0.9])
    # HC Cand: "1.4"
    medium_pred = np.array([0.1, 0.0, 0.0, 0.8, 0.2])
    # HC Cand: "1.4.3"
    fine_pred = np.array([0.7, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.6,
                          0.0, 0.0, 0.0, 0.0, 0.3])
    pred_list = [coarse_pred, medium_pred, fine_pred]
    formatted_pred_dict = format_pred(pred_list, taxonomy)
    out1 = apply_hierarchical_consistency(formatted_pred_dict, taxonomy)

    exp_output = {}
    exp_output['coarse'] = {'probability': coarse_pred[0]}
    exp_output['coarse'].update(get_taxonomy_node("1", taxonomy))
    exp_output['medium'] = {'probability': medium_pred[3]}
    exp_output['medium'].update(get_taxonomy_node("1.4", taxonomy))
    exp_output['fine'] = {'probability': fine_pred[9]}
    exp_output['fine'].update(get_taxonomy_node("1.4.3", taxonomy))

    # Make sure output is in expected format
    assert set(out1.keys()) == set(taxonomy["output_encoding"].keys())
    for level, cand_dict in out1.items():
        assert isinstance(cand_dict, dict)
        assert out1[level]["common_name"] == exp_output[level]["common_name"]
        assert out1['fine']["scientific_name"] \
               == exp_output['fine']["scientific_name"]
        assert out1['fine']["taxonomy_level_names"] \
               == exp_output['fine']["taxonomy_level_names"]
        assert out1['fine']["taxonomy_level_aliases"] \
               == exp_output['fine']["taxonomy_level_aliases"]
        assert out1['fine']["child_ids"] \
               == exp_output['fine']["child_ids"]
        assert np.isclose(out1['fine']["probability"],
                          exp_output['fine']["probability"])

    # HC Cand: "1"
    coarse_pred = np.array([0.9])
    # HC Cand: "1.4"
    medium_pred = np.array([0.1, 0.0, 0.0, 0.8, 0.2])
    # HC Cand: "1.4.3"
    fine_pred = np.array([0.7, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.6,
                          0.0, 0.0, 0.0, 0.0, 0.3])
    pred_list = [coarse_pred, medium_pred, fine_pred]
    formatted_pred_dict = format_pred(pred_list, taxonomy)
    out2 = apply_hierarchical_consistency(formatted_pred_dict, taxonomy)

    exp_output = {}
    exp_output['coarse'] = {'probability': coarse_pred[0]}
    exp_output['coarse'].update(get_taxonomy_node("1", taxonomy))
    exp_output['medium'] = {'probability': medium_pred[3]}
    exp_output['medium'].update(get_taxonomy_node("1.4", taxonomy))
    exp_output['fine'] = {'probability': fine_pred[9]}
    exp_output['fine'].update(get_taxonomy_node("1.4.3", taxonomy))

    # Make sure output is in expected format
    assert set(out2.keys()) == set(taxonomy["output_encoding"].keys())
    for level, cand_dict in out2.items():
        assert isinstance(cand_dict, dict)
        assert out2[level]["common_name"] == exp_output[level]["common_name"]
        assert out2['fine']["scientific_name"] \
               == exp_output['fine']["scientific_name"]
        assert out2['fine']["taxonomy_level_names"] \
               == exp_output['fine']["taxonomy_level_names"]
        assert out2['fine']["taxonomy_level_aliases"] \
               == exp_output['fine']["taxonomy_level_aliases"]
        assert out2['fine']["child_ids"] \
               == exp_output['fine']["child_ids"]
        assert np.isclose(out2['fine']["probability"],
                          exp_output['fine']["probability"])

    # HC Cand: "other"
    coarse_pred = np.array([0.1])
    # HC Cand: "other"
    medium_pred = np.array([0.1, 0.0, 0.0, 0.8, 0.2])
    # HC Cand: "other"
    fine_pred = np.array([0.7, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.6,
                          0.0, 0.0, 0.0, 0.0, 0.3])
    pred_list = [coarse_pred, medium_pred, fine_pred]
    formatted_pred_dict = format_pred(pred_list, taxonomy)
    out3 = apply_hierarchical_consistency(formatted_pred_dict, taxonomy)

    exp_output = {
        "coarse": {
            "probability": 1 - coarse_pred[0],
            "common_name": "other",
            "scientific_name": "other",
            "taxonomy_level_names": "coarse",
            "taxonomy_level_aliases": {},
            'child_ids': taxonomy["output_encoding"]["coarse"][-1]["ids"]
        },
        "medium": {
            "probability": 1 - coarse_pred[0],
            "common_name": "other",
            "scientific_name": "other",
            "taxonomy_level_names": "medium",
            "taxonomy_level_aliases": {},
            'child_ids': taxonomy["output_encoding"]["medium"][-1]["ids"]
        },
        "fine": {
            "probability": 1 - coarse_pred[0],
            "common_name": "other",
            "scientific_name": "other",
            "taxonomy_level_names": "fine",
            "taxonomy_level_aliases": {},
            'child_ids': taxonomy["output_encoding"]["fine"][-1]["ids"]
        }
    }

    # Make sure output is in expected format
    assert set(out3.keys()) == set(taxonomy["output_encoding"].keys())
    for level, cand_dict in out3.items():
        assert isinstance(cand_dict, dict)
        assert out3[level]["common_name"] == exp_output[level]["common_name"]
        assert out3['fine']["scientific_name"] \
               == exp_output['fine']["scientific_name"]
        assert out3['fine']["taxonomy_level_names"] \
               == exp_output['fine']["taxonomy_level_names"]
        assert out3['fine']["taxonomy_level_aliases"] \
               == exp_output['fine']["taxonomy_level_aliases"]
        assert out3['fine']["child_ids"] \
               == exp_output['fine']["child_ids"]
        assert np.isclose(out3['fine']["probability"],
                          exp_output['fine']["probability"])


    # HC Cand: "1"
    coarse_pred = np.array([0.9])
    # HC Cand: "other"
    medium_pred = np.array([0.1, 0.0, 0.0, 0.2, 0.8])
    # HC Cand: "other"
    fine_pred = np.array([0.7, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.6,
                          0.0, 0.0, 0.0, 0.0, 0.3])
    pred_list = [coarse_pred, medium_pred, fine_pred]
    formatted_pred_dict = format_pred(pred_list, taxonomy)
    out4 = apply_hierarchical_consistency(formatted_pred_dict, taxonomy)

    exp_output = {}
    exp_output['coarse'] = {'probability': coarse_pred[0]}
    exp_output['coarse'].update(get_taxonomy_node("1", taxonomy))
    exp_output['medium'] = {
        "probability": 1 - medium_pred[3],
        "common_name": "other",
        "scientific_name": "other",
        "taxonomy_level_names": "medium",
        "taxonomy_level_aliases": {},
        'child_ids': taxonomy["output_encoding"]["medium"][-1]["ids"]
    }
    exp_output['fine'] = {
        "probability": 1 - medium_pred[3],
        "common_name": "other",
        "scientific_name": "other",
        "taxonomy_level_names": "fine",
        "taxonomy_level_aliases": {},
        'child_ids': taxonomy["output_encoding"]["fine"][-1]["ids"]
    }

    # Make sure output is in expected format
    assert set(out4.keys()) == set(taxonomy["output_encoding"].keys())
    for level, cand_dict in out4.items():
        assert isinstance(cand_dict, dict)
        assert out4[level]["common_name"] == exp_output[level]["common_name"]
        assert out4['fine']["scientific_name"] \
               == exp_output['fine']["scientific_name"]
        assert out4['fine']["taxonomy_level_names"] \
               == exp_output['fine']["taxonomy_level_names"]
        assert out4['fine']["taxonomy_level_aliases"] \
               == exp_output['fine']["taxonomy_level_aliases"]
        assert out4['fine']["child_ids"] \
               == exp_output['fine']["child_ids"]
        assert np.isclose(out4['fine']["probability"],
                          exp_output['fine']["probability"])

    # Test with custom level_threshold_dict
    level_threshold_dict = {
        "coarse": 0.99,
        "medium": 0.99,
        "fine": 0.99
    }
    # HC Cand: "other"
    coarse_pred = np.array([0.9])
    # HC Cand: "other"
    medium_pred = np.array([0.1, 0.0, 0.0, 0.8, 0.2])
    # HC Cand: "other"
    fine_pred = np.array([0.7, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.6,
                          0.0, 0.0, 0.0, 0.0, 0.3])
    pred_list = [coarse_pred, medium_pred, fine_pred]
    formatted_pred_dict = format_pred(pred_list, taxonomy)
    out1 = apply_hierarchical_consistency(formatted_pred_dict, taxonomy,
                                          level_threshold_dict=level_threshold_dict)

    exp_output = {}
    exp_output['coarse'] = {
        "probability": 1 - coarse_pred[0],
        "common_name": "other",
        "scientific_name": "other",
        "taxonomy_level_names": "medium",
        "taxonomy_level_aliases": {},
        'child_ids': taxonomy["output_encoding"]["coarse"][-1]["ids"]
    }
    exp_output['medium'] = {
        "probability": 1 - coarse_pred[0],
        "common_name": "other",
        "scientific_name": "other",
        "taxonomy_level_names": "medium",
        "taxonomy_level_aliases": {},
        'child_ids': taxonomy["output_encoding"]["medium"][-1]["ids"]
    }
    exp_output['fine'] = {
        "probability": 1 - coarse_pred[0],
        "common_name": "other",
        "scientific_name": "other",
        "taxonomy_level_names": "fine",
        "taxonomy_level_aliases": {},
        'child_ids': taxonomy["output_encoding"]["fine"][-1]["ids"]
    }

    # Make sure output is in expected format
    assert set(out1.keys()) == set(taxonomy["output_encoding"].keys())
    for level, cand_dict in out1.items():
        assert isinstance(cand_dict, dict)
        assert out1[level]["common_name"] == exp_output[level]["common_name"]
        assert out1['fine']["scientific_name"] \
               == exp_output['fine']["scientific_name"]
        assert out1['fine']["taxonomy_level_names"] \
               == exp_output['fine']["taxonomy_level_names"]
        assert out1['fine']["taxonomy_level_aliases"] \
               == exp_output['fine']["taxonomy_level_aliases"]
        assert out1['fine']["child_ids"] \
               == exp_output['fine']["child_ids"]
        assert np.isclose(out1['fine']["probability"],
                          exp_output['fine']["probability"])

    # Check invalid inputs
    pytest.raises(BirdVoxClassifyError, apply_hierarchical_consistency,
                  formatted_pred_dict, taxonomy, detection_threshold=-1)
    # Check invalid inputs
    pytest.raises(BirdVoxClassifyError, apply_hierarchical_consistency,
                  formatted_pred_dict, taxonomy, level_threshold_dict={
            'coarse': -1,
            'medium': -1,
            'fine': -1
        })
    pytest.raises(BirdVoxClassifyError, apply_hierarchical_consistency,
                  formatted_pred_dict, taxonomy, level_threshold_dict={
            'garply': 0.1
        })

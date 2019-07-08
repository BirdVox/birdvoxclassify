import pytest
import os
import tempfile
import numpy as np
import soundfile as sf
import json
import hashlib
import keras
import logging
import shutil
from scipy.signal.windows import get_window
from six import string_types
from numbers import Real
from birdvoxclassify import *
from birdvoxclassify.birdvoxclassify_exceptions import BirdVoxClassifyError

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
MODULE_DIR = os.path.join(PROJECT_DIR, "birdvoxclassify")

TEST_AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'data/audio')
CHIRP_PATH = os.path.join(TEST_AUDIO_DIR, 'synth_chirp.wav')

RES_DIR = os.path.join(PROJECT_DIR, "resources")
TAX_DIR = os.path.join(RES_DIR, "taxonomy")
MODELS_DIR = os.path.join(RES_DIR, "models")
TAXV1_HIERARCHICAL_PATH = os.path.join(TAX_DIR, "tv1hierarchical.json")
TAXV1_FINE_PATH = os.path.join(TAX_DIR, "tv1fine.json")

MODEL_SUFFIX = "flat-multitask-convnet_tv1hierarchical-a112ec5506b67d95109894a7dbfd186e"
MODEL_NAME = "birdvoxclassify-{}".format(MODEL_SUFFIX)


def test_process_file():
    test_output_dir = tempfile.mkdtemp()
    model = load_model(MODEL_NAME)
    with open(TAXV1_HIERARCHICAL_PATH) as f:
        taxonomy = json.load(f)
    test_output_summary_path = os.path.join(test_output_dir, "summary.json")
    test_output_path = get_output_path(CHIRP_PATH, '.json',
                                       test_output_dir)
    suffix_test_output_path = get_output_path(CHIRP_PATH,
                                              'suffix.json',
                                              test_output_dir)

    try:
        # Test with defaults
        output = process_file(CHIRP_PATH, classifier_name=MODEL_NAME)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict

        # Test with list
        output = process_file([CHIRP_PATH], classifier_name=MODEL_NAME)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict

        # Test with given classifier
        output = process_file([CHIRP_PATH], classifier=model)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict

        # Test with given taxonomy
        output = process_file([CHIRP_PATH], classifier=model, taxonomy=taxonomy)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict

        # Test output_dir
        output = process_file([CHIRP_PATH], output_dir=test_output_dir,
                              classifier=model)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict
        with open(test_output_path, 'r') as f:
            f_output = json.load(f)
        assert next(output.values()) == f_output

        # Test output dir with suffix
        output = process_file([CHIRP_PATH], output_dir=test_output_dir,
                              classifier=model, suffix='suffix')
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict
        with open(suffix_test_output_path, 'r') as f:
            f_output = json.load(f)
        assert next(output.values()) == f_output

        # Test output summary file
        output = process_file([CHIRP_PATH],
                              output_summary_path=test_output_summary_path,
                              classifier=model)
        assert type(output) == dict
        assert len(output) == 1
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict
        with open(test_output_summary_path, 'r') as f:
            f_output = json.load(f)
        assert output == f_output

        # Test multiple files
        process_file([CHIRP_PATH]*10, classifier=model)
        assert type(output) == dict
        assert len(output) == 10
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict

        # Test with different batch_sizes
        process_file([CHIRP_PATH]*10, classifier=model, batch_size=5)
        assert type(output) == dict
        assert len(output) == 10
        for k, v in output.items():
            assert isinstance(k, string_types)
            assert type(v) == dict

        # Make sure we fail if no classifier name is given
        pytest.raises(BirdVoxClassifyError, process_file, CHIRP_PATH)
    finally:
        shutil.rmtree(test_output_dir)


def test_format_pred():
    with open(TAXV1_FINE_PATH) as f:
        taxonomy = json.load(f)

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

    with open(TAXV1_HIERARCHICAL_PATH) as f:
        taxonomy = json.load(f)
    pytest.raises(BirdVoxClassifyError, format_pred, pred_list, taxonomy)


def test_format_pred_batch():
    with open(TAXV1_FINE_PATH) as f:
        taxonomy = json.load(f)

    pred = np.random.random((15,))
    pred /= pred.sum()

    output_ids = [
        "1.4.1",
        "1.1.1",
        "1.4.2",
        "1.4.3",
        "1.4.4",
        "1.1.2",
        "1.4.5",
        "1.3.1",
        "1.4.6",
        "1.4.7",
        "1.2.1",
        "1.1.3",
        "1.3.2",
        "1.1.4",
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
    with open(TAXV1_HIERARCHICAL_PATH) as f:
        taxonomy = json.load(f)

    ref_id = "1"
    node = get_taxonomy_node(ref_id, taxonomy)
    assert "id" in node
    assert "common_name" in node
    assert "scientific_name" in node
    assert "taxonomy_level_names" in node
    assert "taxonomy_level_aliases" in node
    assert "child_ids" in node

    assert node["id"] == ref_id
    assert isinstance(node["common_name"], string_types)
    assert isinstance(node["scientific_name"], string_types)
    assert node["taxonomy_level_names"] == "coarse"
    assert type(node["taxonomy_level_aliases"]) == dict
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
    assert isinstance(node["common_name"], string_types)
    assert isinstance(node["scientific_name"], string_types)
    assert node["taxonomy_level_names"] == "medium"
    assert type(node["taxonomy_level_aliases"]) == dict
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
    assert isinstance(node["common_name"], string_types)
    assert isinstance(node["scientific_name"], string_types)
    assert node["taxonomy_level_names"] == "fine"
    assert type(node["taxonomy_level_aliases"]) == dict
    assert type(node["child_ids"]) == list
    assert len(node["child_ids"]) == 0

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

    gen = batch_generator([CHIRP_PATH]*10, batch_size=10)
    batch = next(gen)
    assert type(batch) == np.ndarray
    assert batch.shape == (10, pcen_settings['top_freq_id'],
                           pcen_settings['n_hops'], 1)

    gen = batch_generator([CHIRP_PATH], batch_size=10)
    batch = next(gen)
    assert type(batch) == np.ndarray
    assert batch.shape == (1, pcen_settings['top_freq_id'],
                           pcen_settings['n_hops'], 1)


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

    # Make sure unsigned ints raise an error
    pytest.raises(BirdVoxClassifyError, compute_pcen,
                  audio.astype('uint32'), sr)


def test_predict():
    model = load_model(MODEL_NAME)

    audio, sr = sf.read(CHIRP_PATH, dtype='float64')
    pcen = compute_pcen(audio, sr)
    pred = predict(pcen, model, logging.INFO)
    assert type(pred) == list
    assert pred[0].shape == (1, 1)
    assert pred[1].shape == (1, 5)
    assert pred[2].shape == (1, 15)

    gen = batch_generator([CHIRP_PATH]*10, batch_size=10)
    batch = next(gen)
    pred = predict(batch, model, logging.INFO)
    assert type(pred) == list
    assert pred[0].shape == (10, 1)
    assert pred[1].shape == (10, 5)
    assert pred[2].shape == (10, 15)

    # Test invalid inputs
    inv_pcen = compute_pcen(audio, sr, input_format=False)[..., np.newaxis]
    pytest.raises(BirdVoxClassifyError, predict, inv_pcen, model, logging.INFO)
    pytest.raises(BirdVoxClassifyError, predict, np.array([1, 2, 3, 4]), model,
                  logging.INFO)
    pytest.raises(BirdVoxClassifyError, predict, np.zeros((1, 42, 104, 1)),
                  model, logging.INFO)
    pytest.raises(BirdVoxClassifyError, predict, np.zeros((1, 120, 42, 1)),
                  model, logging.INFO)
    pytest.raises(BirdVoxClassifyError, predict, np.zeros((1, 120, 104, 42)),
                  model, logging.INFO)


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
    assert isinstance(settings['window'], string_types)
    # Make sure window is valid
    get_window(settings['window'], 5)


def test_get_model_path():
    test_model_name = "test_model_name"
    exp_model_path = os.path.join(RES_DIR, "models", test_model_name + '.h5')
    model_path = get_model_path(test_model_name)
    assert os.path.abspath(model_path) == os.path.abspath(exp_model_path)


def test_load_model():
    model = load_model(MODEL_NAME)
    assert type(model) == keras.models.Model

    # Test invalid inputs
    invalid_path = get_model_path("invalid-classifier")
    with open(invalid_path, "w") as f:
        f.write("INVALID")

    try:
        pytest.raises(BirdVoxClassifyError, load_model, "invalid-classifier")
    finally:
        os.remove(invalid_path)


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

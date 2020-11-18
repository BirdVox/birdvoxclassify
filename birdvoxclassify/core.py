import librosa
import logging
import hashlib
import json
import sys
import numpy as np
import six
import os
import warnings
import traceback
import soundfile as sf
from contextlib import redirect_stderr

with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.simplefilter("ignore")
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    with redirect_stderr(open(os.devnull, "w")):
        from tensorflow import keras

from .birdvoxclassify_exceptions import BirdVoxClassifyError

DEFAULT_MODEL_SUFFIX = "taxonet_tv1hierarchical" \
                       "-2e7e1bbd434a35b3961e315cfe3832fc"
MODEL_PREFIX = 'birdvoxclassify'
DEFAULT_MODEL_NAME = "{}-{}".format(MODEL_PREFIX, DEFAULT_MODEL_SUFFIX)


def process_file(filepaths, output_dir=None, output_summary_path=None,
                 classifier=None, taxonomy=None, batch_size=512, suffix='',
                 logger_level=logging.INFO, model_name=DEFAULT_MODEL_NAME):
    """
    Runs bird species classification model on one or more audio clips.

    Parameters
    ----------
    filepaths : list or str
        Filepath or list of filepaths of audio files for which to run prediction
    output_dir : str or None [default: None]
        Output directory used for outputting per-file prediction JSON files. If
        ``None``, no per-file prediction JSON files are produced.
    output_summary_path : str or None [default: None]
        Output path for summary prediction JSON file for all processed audio
        files. If ``None``, no summary prediction file is produced.
    classifier : keras.models.Model or None [default: None]
        Bird species classification model object. If ``None``, the model
        corresponding to ``model_name`` is loaded.
    taxonomy : dict or None [default: None]
        Taxonomy JSON object. If ``None``, the taxonomy corresponding to
        ``model_name`` is loaded.
    batch_size : int [default: 512]
        Batch size for predictions
    suffix : str [default: ""]
        String to append to filename
    logger_level : int [default: logging.INFO]
        Logger level
    model_name : str [default birdvoxclassify.DEFAULT_MODEL_NAME]
        Name of classifier model. Should be in format
        ``<model id>_<taxonomy version>-<taxonomy md5sum>``

    Returns
    -------
    output_dict : dict[str, dict]
        Output dictionary mapping audio filename to prediction dictionary, in
        the format produced by ``format_pred``.
    """
    # Set logger level.
    logging.getLogger().setLevel(logger_level)

    # Print model.
    logging.info("Loading model: {}".format(model_name))

    # Load the classifier.
    if classifier is None:
        classifier = load_classifier(model_name)

    if taxonomy is None:
        taxonomy_path = get_taxonomy_path(model_name)
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)

    # Create output_dir if necessary.
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if isinstance(filepaths, six.string_types):
        filepaths = [filepaths]

    batch_gen = batch_generator(filepaths, batch_size=batch_size)

    output_dict = {}

    for batch, batch_filepaths in batch_gen:
        batch_pred = predict(batch, classifier, logger_level)

        for idx, filepath in enumerate(batch_filepaths):
            pred = [p[idx] for p in batch_pred]
            pred_dict = format_pred(pred, taxonomy)

            output_dict[filepath] = pred_dict

            if output_dir:
                output_path = get_output_path(filepath,
                                              suffix + '.json',
                                              output_dir)
                with open(output_path, 'w') as f:
                    json.dump(pred_dict, f)

            # Print final messages.
            logging.info("Done with file: {}.".format(filepath))

    if output_summary_path is not None:
        with open(output_summary_path, 'w') as f:
            json.dump(output_dict, f)

    return output_dict


def format_pred(pred_list, taxonomy):
    """
    Formats a list of predictions for a single audio clip into a more
    human-readable JSON object using the given taxonomy object.

    The output will be in the following format:

    .. code-block:: javascript

        {
          <prediction level> : {
            <taxonomy id> : {
              "probability": <float>,
              "common_name": <str>,
              "scientific_name": <str>,
              "taxonomy_level_names": <str>,
              "taxonomy_level_aliases": <dict of aliases>,
              "child_ids": <list of children IDs>
            },
            ...
          },
          ...
        }

    Parameters
    ----------
    pred_list : list[np.ndarray [shape (1, num_labels) or (num_labels,)]
        List of predictions at the taxonomical levels predicted by the model
        for a single example. ``num_labels`` may be different for each of the
        different levels of the taxonomy.
    taxonomy : dict
        Taxonomy JSON object

    Returns
    -------
    pred_dict : dict
        Prediction dictionary object

    """
    if len(pred_list) != len(taxonomy['output_encoding']):
        err_msg = "Taxonomy expects {} outputs but model produced {} outputs."
        raise BirdVoxClassifyError(err_msg.format(
            len(taxonomy['output_encoding']), len(pred_list)
        ))

    pred_dict = {}

    encoding_items = taxonomy['output_encoding'].items()
    for pred, (level, encoding_list) in zip(pred_list, encoding_items):

        pred_dict[level] = {}

        if pred.ndim == 2:
            if pred.shape[0] != 1:
                err_msg = 'Attempted to provide prediction of a batch larger ' \
                          'than 1. Please use `format_pred_batch`.'
                raise BirdVoxClassifyError(err_msg)
            pred = pred.flatten()

        for prob, item in zip(pred, encoding_list):
            if len(item['ids']) == 1:
                ref_id = item['ids'][0]
            else:
                ref_id = "other"

            pred_dict[level][ref_id] = {'probability': float(prob)}

            if ref_id != "other":
                pred_dict[level][ref_id].update(get_taxonomy_node(ref_id,
                                                                  taxonomy))
            else:
                pred_dict[level][ref_id].update({
                    "common_name": "other",
                    "scientific_name": "other",
                    "taxonomy_level_names": level,
                    "taxonomy_level_aliases": {},
                    "child_ids": item['ids']
                })

    return pred_dict


def format_pred_batch(batch_pred_list, taxonomy):
    """
    Formats a list of predictions for a batch of audio clips into a more
    human-readable JSON object using the given taxonomy object. The output will
    be in the form of a list of JSON objects in the format returned by
    ``format_pred``.


    Parameters
    ----------
    batch_pred_list : list[np.ndarray [shape (batch_size, num_labels)] ]
        List of predictions at the taxonomical levels predicted by the model
        for a batch of examples. ``num_labels`` may be different for each of the
        different levels of the taxonomy.
    taxonomy : dict
        Taxonomy JSON object

    Returns
    -------
    pred_dict : list[dict]
        List of JSON dictionary objects

    """
    for level_pred in batch_pred_list:
        if len(level_pred) != len(batch_pred_list[0]):
            err_msg = 'Number of predictions for each level are not consistent.'
            raise BirdVoxClassifyError(err_msg)

    pred_dict_list = []
    for idx in range(len(batch_pred_list[0])):
        pred_list = [p[idx] for p in batch_pred_list]
        pred_dict = format_pred(pred_list, taxonomy)
        pred_dict_list.append(pred_dict)

    return pred_dict_list


def get_taxonomy_node(ref_id, taxonomy):
    """
    Gets node in taxonomy corresponding to the given reference ID (e.g. ``1.4.1``)

    Parameters
    ----------
    ref_id : str
        Taxonomy reference ID
    taxonomy : dict
        Taxonomy JSON object

    Returns
    -------
    node : dict[str, *]
        Taxonomy node, containing information about the entity corresponding to
        the given taxonomy reference ID

    """
    if ref_id == 'other':
        return {"id": "other"}

    # Not the most efficient but shouldn't be too bad
    for item in taxonomy['taxonomy']:
        if "id" not in item:
            raise BirdVoxClassifyError("Taxonomy node does not contain an id")

        if item["id"] == ref_id:
            return item

    err_msg = "Could not find id {} in taxonomy"
    raise BirdVoxClassifyError(err_msg.format(ref_id))


def batch_generator(filepath_list, batch_size=512):
    """
    Returns a generator that, from a list of filepaths, yields batches of PCEN
    images and the corresponding filenames.

    Parameters
    ----------
    filepath_list : list[str]
        (Non-empty) list of filepaths to audio files for which to generate
        batches of PCEN images and the corresponding filenames
    batch_size : int [default: 512]
        Size of yielded batches

    Yields
    ------
    batch : np.ndarray [shape: (batch_size, top_freq_id, n_hops, 1)]
        PCEN batch
    batch_filepaths : list[str]
        List of filepaths corresponding to the clips in the batch

    """
    if batch_size <= 0 or not isinstance(batch_size, int):
        err_msg = 'Batch size must be a positive integer. Got {}'
        raise BirdVoxClassifyError(err_msg.format(batch_size))

    if type(filepath_list) != list or len(filepath_list) == 0:
        raise BirdVoxClassifyError("Must provide non-empty filepath list.")

    batch = []
    batch_filepaths = []
    file_count = 0
    for filepath in filepath_list:
        # Print new line and file name.
        logging.info("-" * 72)
        logging.info("Loading file: {}".format(filepath))

        # Check for existence of the input file.
        if not os.path.exists(filepath):
            raise BirdVoxClassifyError(
                'File "{}" could not be found.'.format(filepath))

        try:
            audio, sr = sf.read(filepath)
        except Exception:
            exc_str = 'Could not open file "{}":\n{}'
            exc_formatted_str = exc_str.format(filepath, traceback.format_exc())
            raise BirdVoxClassifyError(exc_formatted_str)

        pcen = compute_pcen(audio, sr, input_format=True)[np.newaxis, ...]

        batch.append(pcen)
        batch_filepaths.append(filepath)
        file_count += 1

        if file_count == batch_size:
            yield np.vstack(batch), batch_filepaths
            file_count = 0
            batch = []
            batch_filepaths = []

    # Yield final batch
    if file_count > 0:
        yield np.vstack(batch), batch_filepaths

    return


def compute_pcen(audio, sr, input_format=True):
    """
    Computes PCEN (per-channel-energy normalization) for the given audio clip.

    Parameters
    ----------
    audio : np.ndarray [shape: (N,)]
        Audio array
    sr : int
        Sample rate
    input_format : bool [default: True]
        If True, adds an additional channel dimension (of size 1) and ensures
        that a fixed number of PCEN frames (corresponding to
        ``get_pcen_settings()['n_hops']``) is returned. If number of frames is
        greater, the center frames are returned. If the the number of frames is
        less, empty frames are padded.

    Returns
    -------
    pcen : np.ndarray [shape: (top_freq_id, n_hops, 1) or (top_freq_id, num_frames)]
        Per-channel energy normalization processed Mel spectrogram. If
        ``input_format=True``, will be in shape ``(top_freq_id, n_hops, 1)``.
        Otherwise it will be in shape ``(top_freq_id, num_frames)``, where
        ``num_frames`` is the number of PCEN frames for the entire audio clip.

    """
    # Load settings.
    pcen_settings = get_pcen_settings()

    # Standardize type to be float32 [-1, 1]
    if audio.dtype.kind == 'i':
        max_val = max(np.iinfo(audio.dtype).max, -np.iinfo(audio.dtype).min)
        audio = audio.astype('float64') / max_val
    elif audio.dtype.kind == 'f':
        audio = audio.astype('float64')
    else:
        err_msg = 'Invalid audio dtype: {}'
        raise BirdVoxClassifyError(err_msg.format(audio.dtype))

    # Map to the range [-2**31, 2**31[
    audio = (audio * (2**31)).astype('float32')

    # Resample to 22,050 kHz
    if not sr == pcen_settings["sr"]:
        audio = librosa.resample(audio, sr, pcen_settings["sr"])

    # Compute Short-Term Fourier Transform (STFT).
    stft = librosa.stft(
        audio,
        n_fft=pcen_settings["n_fft"],
        win_length=pcen_settings["win_length"],
        hop_length=pcen_settings["hop_length"],
        window=pcen_settings["window"])

    # Compute squared magnitude coefficients.
    abs2_stft = (stft.real*stft.real) + (stft.imag*stft.imag)

    # Gather frequency bins according to the Mel scale.
    # NB: as of librosa v0.6.2, melspectrogram is type-instable and thus
    # returns 64-bit output even with a 32-bit input. Therefore, we need
    # to convert PCEN to single precision eventually. This might not be
    # necessary in the future, if the whole PCEN pipeline is kept type-stable.
    melspec = librosa.feature.melspectrogram(
        y=None,
        S=abs2_stft,
        sr=pcen_settings["sr"],
        n_fft=pcen_settings["n_fft"],
        n_mels=pcen_settings["n_mels"],
        htk=True,
        fmin=pcen_settings["fmin"],
        fmax=pcen_settings["fmax"])

    # Compute PCEN.
    pcen = librosa.pcen(
        melspec,
        sr=pcen_settings["sr"],
        hop_length=pcen_settings["hop_length"],
        gain=pcen_settings["pcen_norm_exponent"],
        bias=pcen_settings["pcen_delta"],
        power=pcen_settings["pcen_power"],
        time_constant=pcen_settings["pcen_time_constant"])

    # Convert to single floating-point precision.
    pcen = pcen.astype('float32')

    # Truncate spectrum to range 2-10 kHz.
    pcen = pcen[:pcen_settings["top_freq_id"], :]

    # Format for input to network
    if input_format:
        # Trim TFR in time to required number of hops.
        pcen_width = pcen.shape[1]
        n_hops = pcen_settings["n_hops"]
        if pcen_width >= n_hops:
            first_col = int((pcen_width - n_hops) / 2)
            last_col = int((pcen_width + n_hops) / 2)
            pcen = pcen[:, first_col:last_col]
        else:
            # Pad if not enough frames
            pad_length = n_hops - pcen_width
            left_pad = pad_length // 2
            right_pad = pad_length - left_pad
            pcen = np.pad(pcen, [(0, 0), (left_pad, right_pad)],
                          mode='constant')

        # Add channel dimension
        pcen = pcen[:, :, np.newaxis]

    # Return.
    return pcen


def predict(pcen, classifier, logger_level=logging.INFO):
    """
    Performs bird species classification on PCEN arrays using the given model.

    Parameters
    ----------
    pcen : np.ndarray [shape (n_mels, n_hops, 1) or (batch_size, n_mels, n_hops, 1)
        PCEN array for a single clip or a batch of clips
    classifier : keras.models.Model
        Bird species classification model object
    logger_level : int [default: logging.INFO]
        Logger level

    Returns
    -------
    pred_list : list[np.ndarray [shape (batch_size or 1, num_labels)] ]
        List of predictions at the taxonomical levels predicted by the model.
        num_labels may be different for each of the different levels of the
        taxonomy. If a single example is given (i.e. there is no batch dimension
        in the input PCEN), ``batch_size = 1``.

    """
    pcen_settings = get_pcen_settings()

    # Add batch dimension if we are classifying a single clip
    if pcen.ndim == 3:
        pcen = pcen[np.newaxis, ...]
    elif pcen.ndim not in (3, 4):
        err_msg = 'Invalid number of PCEN dimension. ' \
                  'Expected 3 or 4, but got {}'
        raise BirdVoxClassifyError(err_msg.format(pcen.ndim))

    if pcen.shape[1] != pcen_settings['top_freq_id']:
        err_msg = 'Invalid number of mel-frequency bins in input PCEN. ' \
                  'Expected {} but got {}.'
        raise BirdVoxClassifyError(err_msg.format(
            pcen_settings['top_freq_id'],
            pcen.shape[1]
        ))

    if pcen.shape[2] != pcen_settings['n_hops']:
        err_msg = 'Invalid number of frames in input PCEN. ' \
                  'Expected {} but got {}.'
        raise BirdVoxClassifyError(err_msg.format(
            pcen_settings['n_hops'],
            pcen.shape[2]
        ))

    if pcen.shape[3] != 1:
        err_msg = 'Invalid number of channels in input PCEN. ' \
                  'Expected 1 but got {}.'
        raise BirdVoxClassifyError(err_msg.format(pcen.shape[3]))

    # Predict
    verbose = (logger_level < 15)
    pred = classifier.predict(pcen, verbose=verbose)
    return pred


def get_output_path(filepath, suffix, output_dir):
    """
    Returns output path to file containing bird species classification
    predictions for a given audio clip file.

    Parameters
    ----------
    filepath : str
        Path to audio file to be processed
    suffix : str
        String to append to filename (including extension)
    output_dir : str or None
        Path to directory where file will be saved.
        If None, will use directory of given filepath.

    Returns
    -------
    output_path : str
        Path to output file
    """
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    if not output_dir:
        output_dir = os.path.dirname(filepath)

    if suffix[0] != '.':
        output_filename = "{}_{}".format(base_filename, suffix)
    else:
        output_filename = base_filename + suffix

    return os.path.join(output_dir, output_filename)


def get_pcen_settings():
    """
    Returns dictionary of Mel spectrogram and PCEN parameters for preparing the
    input to the bird species classification models.

    Returns
    -------
    pcen_settings : dict[str, *]
        Dictionary of Mel spectrogram and PCEN parameters

    """
    pcen_settings = {
        "fmin": 2000,
        "fmax": 11025,
        "hop_length": 32,
        "n_fft": 1024,
        "n_mels": 128,
        "pcen_delta": 10.0,
        "pcen_time_constant": 0.06,
        "pcen_norm_exponent": 0.8,
        "pcen_power": 0.25,
        "sr": 22050.0,
        "top_freq_id": 120,
        "win_length": 256,
        "n_hops": 104,
        "window": "flattop"}
    return pcen_settings


def get_model_path(model_name):
    """
    Returns path to the bird species classification model of the given name.

    Parameters
    ----------
    model_name : str
        Name of classifier model. Should be in format
        ``<model id>_<taxonomy version>-<taxonomy md5sum>``

    Returns
    -------
    model_path : str
        Path to classifier model weights. Should be in format
        ``<BirdVoxClassify dir>/resources/models/<model id>_<taxonomy version>-<taxonomy md5sum>.h5``

    """
    # Python 3.8 requires a different model for compatibility
    if sys.version_info.major == 3 and sys.version_info.minor == 8:
        model_name = model_name.replace(MODEL_PREFIX, MODEL_PREFIX + '-py3pt8')

    path = os.path.join(os.path.dirname(__file__),
                        "resources",
                        "models",
                        model_name + '.h5')
    # Use abspath to get rid of the relative path
    return os.path.abspath(path)


def load_classifier(model_name):
    """
    Loads bird species classification model of the given name.

    Parameters
    ----------
    model_name : str
        Name of classifier model. Should be in format
        ``<model id>_<taxonomy version>-<taxonomy md5sum>``

    Returns
    -------
    classifier : keras.models.Model
        Bird species classification model

    """
    model_path = get_model_path(model_name)

    if not os.path.exists(model_path):
        raise BirdVoxClassifyError(
            'Model "{}" could not be found.'.format(model_name))
    try:
        classifier = keras.models.load_model(model_path, compile=False)
    except Exception:
        exc_str = 'Could not open model "{}":\n{}'
        formatted_trace = traceback.format_exc()
        exc_formatted_str = exc_str.format(model_path, formatted_trace)
        raise BirdVoxClassifyError(exc_formatted_str)

    return classifier


def get_taxonomy_path(model_name):
    """
    Get the path to the taxonomy corresponding to the model of the given name.

    Specifically, with a model name of the format:

    ``<model id>_<taxonomy version>-<taxonomy md5sum>``

    the path to taxonomy file
    ``<BirdVoxClassify dir>/resources/taxonomy/<taxonomy version>.json``
    is returned. The MD5 checksum of this file is compared to <taxonomy md5sum>
    to ensure that the content of the taxonomy file matches the format of the
    output that the model is expected to produce.


    Parameters
    ----------
    model_name : str
        Name of model. Should be in format
        `<model id>_<taxonomy version>-<taxonomy md5sum>`

    Returns
    -------
    taxonomy_path : str
        Path to taxonomy file, which should be in format
        `<BirdVoxClassify dir>/resources/taxonomy/<taxonomy version>.json`

    """
    taxonomy_version, exp_md5sum = model_name.split('_')[1].split('-')
    taxonomy_path = os.path.abspath(
                        os.path.join(
                            os.path.dirname(__file__),
                            "resources",
                            "taxonomy",
                            taxonomy_version + '.json'))

    # Verify the MD5 checksum
    hash_md5 = hashlib.md5()
    with open(taxonomy_path, "rb") as f:
        hash_md5.update(f.read())
    md5sum = hash_md5.hexdigest()

    if exp_md5sum != md5sum:
        err_msg = 'Taxonomy corresponding to model {} has bad checksum. ' \
                  'Expected {} but got {}.'
        raise BirdVoxClassifyError(err_msg.format(
            model_name, exp_md5sum, md5sum
        ))

    return taxonomy_path

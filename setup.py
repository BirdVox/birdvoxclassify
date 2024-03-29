import os
import sys
import gzip
from setuptools import setup, find_packages

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

model_dir = os.path.join('birdvoxclassify', 'resources', 'models')
taxonomy_dir = os.path.join('birdvoxclassify', 'resources', 'taxonomy')
suffixes = [
    'flat-multitask-convnet-v2_tv1hierarchical-3c6d869456b2705ea5805b6b7d08f870',
    'flat-multitask-convnet-v2_tv1hierarchical-2e7e1bbd434a35b3961e315cfe3832fc',
    'taxonet_tv1hierarchical-3c6d869456b2705ea5805b6b7d08f870',
    'taxonet_tv1hierarchical-2e7e1bbd434a35b3961e315cfe3832fc'
]

model_prefix = 'birdvoxclassify'
# Python 3.8 requires a different model for compatibility
if sys.version_info.major == 3 and sys.version_info.minor == 8:
    model_prefix += '-py3pt8'
weight_files = [f'{model_prefix}-{suffix}.h5' for suffix in suffixes]
base_url = 'https://github.com/BirdVox/birdvoxclassify/raw/models/'

if len(sys.argv) > 1 and sys.argv[1] == 'sdist':
    # exclude the weight files in sdist
    weight_files = []
else:
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # in all other cases, download and decompress weight files
    for weight_file in weight_files:
        weight_path = os.path.join(model_dir, weight_file)
        if not os.path.isfile(weight_path):
            compressed_file = weight_file + '.gz'
            compressed_path = os.path.join(model_dir, compressed_file)
            if not os.path.isfile(compressed_file):
                print('Downloading weight file {} ...'.format(compressed_file))
                urlretrieve(base_url + compressed_file, compressed_path)

            print('Decompressing ...')
            with open(weight_path, 'wb') as target:
                try:
                    with gzip.open(compressed_path, 'rb') as source:
                        target.write(source.read())
                except OSError:
                    # Handle symlinks
                    with open(compressed_path) as symlink:
                        # Github raw stores symlinks as text files, so we need
                        # to read it to check the text
                        real_compressed_file = symlink.read()
                    os.remove(compressed_path)
                    msg = '{} is symlink, downloading {} ...'
                    print(msg.format(compressed_file, real_compressed_file))
                    urlretrieve(base_url + real_compressed_file,
                                compressed_path)
                    with gzip.open(compressed_path, 'rb') as source:
                        target.write(source.read())

            print('Decompression complete')
            os.remove(compressed_path)
            print('Removing compressed file')

try:
    import types
    import importlib.machinery
    loader = importlib.machinery.SourceFileLoader(
        'birdvoxclassify.version',
        os.path.join('birdvoxclassify', 'version.py'))
    version = types.ModuleType(loader.name)
    loader.exec_module(version)
except ImportError:
    import types
    import imp
    version = imp.load_source(
        'birdvoxclassify.version',
        os.path.join('birdvoxclassify', 'version.py'))


with open('README.md') as file:
    long_description = file.read()

setup(
    name='birdvoxclassify',
    version=version.version,
    description='Species identification from bird flight call recordings',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BirdVox/birdvoxclassify',
    author='Aurora Cramer, Vincent Lostanlen, Justin Salamon, ' +\
        'Andrew Farnsworth, and Juan Pablo Bello',
    author_email='jtcramer@nyu.edu',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['birdvoxclassify=birdvoxclassify.cli:main'],
    },
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='bioacoustics, audio signal processing, machine learning',
    project_urls={
        'Source': 'https://github.com/BirdVox/birdvoxclassify',
        'Tracker': 'https://github.com/BirdVox/birdvoxclassify/issues'
    },
    install_requires=[
        'scipy>=1.0.0',
        'numpy>=1.16.5',
        'pandas>=0.23',
        'h5py>=2.7.0',
        'SoundFile>=0.9.0',
        'tensorflow>=2.0.0',
        'librosa>=0.6.2',
    ],
    extras_require={
        'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        'tests': [
            'pytest>=5.0.1',
            'mock>=3.0.5'
        ]
    },
    package_data={
        'birdvoxclassify':
            [os.path.join('resources', 'models', fname)
             for fname in weight_files] + \
            [os.path.join('resources', 'taxonomy', fname)
             for fname in os.listdir(taxonomy_dir)]
    }
)

.. _taxonomy:

BirdVoxClassify taxonomy
========================

Introduction
------------

In the context of the `BirdVox <https://wp.nyu.edu/birdvox/>`_ project, we limit the taxonomical scope
to a subset of bird species. This subset is formalized via a taxonomy file which is used to both specify
what species we are interested in as well as describing the output of a particular model using the
taxonomy. In fact, model names refer to the taxonomy they use (as well as the MD5 checksum of the
taxonomy file) in their filenames. For example, the model with ID
``birdvoxclassify-flat-multitask-convnet_tv1hierarchical-3c6d869456b2705ea5805b6b7d08f870`` uses the taxonomy
``tv1hierarchical``, which has an MD5 checksum of ``3c6d869456b2705ea5805b6b7d08f870``. Generally, a model name
is in the format ``<model identifier>_<taxonomy name>-<taxonomy md5 checksum>``.


.. _overview:

Overview
-----------------

The primary taxonomy currently used (``v1hierarchical``) is as such:

- [0] Other - Coarse (Non-passerine)
- [1] Passerines
    - [1.0] Other - Medium (Other Passerine)
    - [1.1] American Sparrow
        - [1.1.0] Other - Fine (Other American Sparrow)
        - [1.1.1] ATSP (American Tree Sparrow)
        - [1.1.2] CHSP (Chipping Sparrow)
        - [1.1.3] SAVS (Savannah Sparrow)
        - [1.1.4] WTSP (White-throated Sparrow)
        - [1.1.X] Unknown - Fine (Unknown American Sparrow)
    - [1.2] Cardinal
        - [1.2.0] Other - Fine (Other Cardinal)
        - [1.2.1] RBGR (Rose-breasted Grosbeak)
        - [1.2.X] Unknown - Fine (Unknown Cardinal)
    - [1.3] Thrush
        - [1.3.0] Other - Fine (Other Thrush)
        - [1.3.1] GCTH (Gray-cheeked Thrush)
        - [1.3.2] SWTH (Swainson's Thrush)
        - [1.3.X] Unknown - Fine (Unknown Thrush)
    - [1.4] Warbler
        - [1.4.0] Other - Fine (Other Warbler)
        - [1.4.1] AMRE (American Redstart)
        - [1.4.2] BBWA (Bay-breasted Warbler)
        - [1.4.3] BTBW (Black-throated Blue Warbler)
        - [1.4.4] CAWA (Canada Warbler)
        - [1.4.5] COYE (Common Yellowthroat)
        - [1.4.6] MOWA (Mourning Warbler)
        - [1.4.7] OVEN (Ovenbird)
        - [1.4.X] Unknown - Fine (Unknown Warbler)
    - [1.X] Unknown - Medium (Unknown Passerine)
- [X] Unknown - Coarse (Unknown bird)

Note that all species have a fully qualified three-digit taxonomy code. Each digit corresponds to
the respective values of each species within our three level taxonomy, corresponding roughly to the standard
taxonomical groupings "order", "family", and "species". N-digit taxonomy codes (e.g. for N < 3) correspond to
higher level concepts in the taxonomy (such as family). Any N-digit taxonomy code ending in a ``0`` corresponds to
an "other" class, simply meaning it is outside our scope of interest. For example, ``1.0`` corresponds to an
"other Passerine" meaning a Passerine that is not a member of the families of interest. Similarly, any N-digit taxonomy
code ending in an ``X`` corresponds to an "unknown" class, meaning that the annotator could not determine the
identity of a particular bird at that level of the taxonomy. For example, ``1.1.X`` corresponds to an American
Sparrow for which the annotator was unsure of the particular species.


.. _taxonomy_format:

Taxonomy format
-----------------

The taxonomy is specified in JSON format. The taxonomy files can be found in
``<BirdVoxClassify dir>/resources/taxonomy/``.

.. code-block:: javascript

    {
      "taxonomy": [
        {
          "id": <str>, // Numeric taxonomic ID
          "common_name": <str>, // Common name
          "scientific_name": <str>, // Latin name
          "taxonomy_level_names": <str>, // Level of taxonomy
          "taxonomy_level_aliases": {

          },
          "child_ids": [
              <str>, // Child taxonomic IDs
              ...
          ]
        },
        ...

      ],
      "output_encoding": {
        <str>: [ // Name of level of taxonomy
          // Ordered list of outputs
          {
            "ids": [
                <str>, // List of taxonomic IDs encapsulated in this output
                ...
            ]
          },
          ...
          {
            "ids": [
                <str>, // Last output should encompass all "other" classes
                ...
            ]
          }
        ],
    }

The ``taxonomy`` field contains nodes of the tree of the taxonomy. each of which contain the N-digit taxonomy reference
ID, identifying information and aliases about the node, and the IDs of children nodes in the taxonomy.
``output_encoding`` specifies the taxonomy IDs associated with each element of an output probability vector produced
by a classifier using this taxonomy. The order of the list associated with each level of the taxonomy
corresponds to the position in an output vector.

.. _output_format:

Output format
-------------

Model output is given as JSON:

.. code-block:: javascript

    {
      <prediction level> : {
        <taxonomy id> : {
          "probability": <float>,
          "common_name": <str>,
          "scientific_name": <str>,
          "taxonomy_level_names": <str>,
          "taxonomy_level_aliases": <dict of aliases>,
          "child_ids": []
        },
        ...
        "other" : {
          "common_name": "other",
          "scientific_name": "other",
          "taxonomy_level_names": level,
          "taxonomy_level_aliases": {},
          "child_ids": <list of children IDs>
        }
      },
      ...
    }

The probabilities at each prediction level. For a summary file, containing predictions for multiple files the output is
given as:

.. code-block:: javascript

    {
      <filename> : {
        <output node>
      },
      ...
    }

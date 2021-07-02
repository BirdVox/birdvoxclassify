.. _changes:

Changelog
---------

v0.3.1
~~~~~~
- Restore deprecated taxonomy files
- Raise DeprecationWarning when deprecated taxonomy files or models that use them are loaded
- Update documentation to notify users of deprecation of these files and how to address it
- Update v0.3.0 changelog

v0.3.0
~~~~~~
- Add functionality for obtaining best candidates from predictions
- Add hierarchical consistency implementation for selecting best candidates
- Drop ``six`` dependency.
- Update taxonomy files so that order-level taxa are in plural form *[v0.3.1 UPDATE: model names with taxonomy md5sum
  ``2e7e1bbd434a35b3961e315cfe3832fc`` or ``beb9234f0e13a34c7ac41db72e85addd`` are not available in this version but
  are restored in v0.3.1 for backwards compatibility. They will no longer be supported starting with v0.4.
  Please use model names with taxonomy md5 checksums ``3c6d869456b2705ea5805b6b7d08f870``
  and ``2f6efd9017669ef5198e48d8ec7dce4c`` (respectively) instead.]*

v0.2.0
~~~~~~
- Drop support for Python 3.5, add support for Python 3.7 and 3.8.
- Deprecate and remove v1 model non-hierarchical
- Add models compatible with Python 3.8
- Make TaxoNet the default model.
- Fix broken dependencies.
- Swap ``keras`` for ``tf.keras`` and require Tensorflow 2.x.

v0.1.1
~~~~~~
- Add TaxoNet model.

v0.1.0
~~~~~~
- First release.

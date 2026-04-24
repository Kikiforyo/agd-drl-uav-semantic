# Makes the ``agents`` directory a Python package. This allows
# importing subpackages like ``agents.ddpg`` from outside of this
# directory. Without an ``__init__`` file, Python would not
# recognize ``agents`` as a package, leading to import errors.
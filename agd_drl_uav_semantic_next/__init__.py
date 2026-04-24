# Makes `agd-drl-uav-semantic` a Python package so that modules can be
# imported relative to this root directory. Without this file, Python
# treats this directory as a regular folder, causing import errors when
# using ``import envs`` or ``import agents`` from scripts located in
# subfolders. Adding this file ensures the package structure is
# recognized and allows training scripts to resolve imports properly.
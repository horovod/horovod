import os
import sys
import sysconfig
from packaging import version
from importlib import metadata
from setuptools import build_meta as _orig

prepare_metadata_for_build_wheel = _orig.__legacy__.prepare_metadata_for_build_wheel
build_wheel = _orig.__legacy__.build_wheel
build_sdist = _orig.__legacy__.build_sdist
get_requires_for_build_sdist = _orig.__legacy__.get_requires_for_build_sdist


def get_requires_for_build_wheel(self, config_settings=None):
    """
        Custom backend to enable PEP517, utilises env variables to define which extra build
         packages we should be installing into the isolated build env.
        These should match the users expected versions installed outside the isolated environment or it will
         cause library mismatch failures.
    """
    new_pkgs = []
    MXNET = "mxnet"
    key_pkg_map = {'HOROVOD_WITH_MXNET': MXNET,
                   'HOROVOD_WITH_PYTORCH': 'torch',
                   'HOROVOD_WITH_TENSORFLOW': 'tensorflow'}
    for key in key_pkg_map.keys():
        try:
            version_string = os.environ[key]
            try:
                version.Version(version_string)
                new_pkgs.append(f"{key_pkg_map[key]}=={version_string}")
            except version.InvalidVersion:
                new_pkgs.append(f"{version_string}")
            if key_pkg_map[key] == MXNET:
                # MxNet has np.bool everywhere which is removed in newer
                # versions...
                new_pkgs.append("numpy==1.20.3")
        except BaseException:
            # Pass for now, elsewhere will alert the user has built this wrong.
            ...

    return _orig.__legacy__.get_requires_for_build_wheel(
        config_settings) + new_pkgs

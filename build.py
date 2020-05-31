#  libfive-python: libfive bindings for Python
#
#  Copyright (C) 2020  Sem Mulder
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this file,
#  You can obtain one at http://mozilla.org/MPL/2.0/.
from Cython.Build import cythonize
from setuptools import Extension


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                [
                    Extension(
                        "libfive._libfive", ["libfive/_libfive.pyx"], libraries=["five"]
                    )
                ],
                language_level=3,
            ),
            "zip_safe": False,
        }
    )

import os

from conans import ConanFile


class ArLegoClassification(ConanFile):
    name = "ar-lego-classification"
    version = "0.0.1"
    license = "Proprietary"
    author = 'Agile Robots AG'
    description = "Lego classification"
    url = 'http://git.ar.int/b.koenig/ar-lego-classification'
    generators = "cmake"
    default_user = 'ar'
    default_channel = 'stable'
    scm = {"revision": "auto", "type": "git", "url": "auto"}
    no_copy_source = True
    settings = "os", "arch", "compiler", "build_type"
    build_requires = ["ar-dev/[^3.x]@ar/stable"]
    requires = ["ar-detection/[^1.x]@ar/stable"]

    pass
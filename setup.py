from setuptools import setup
from distutils.core import setup,Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

import numpy

with open("README.md", 'r') as f:
    long_description = f.read()

cmdclass = {}
ext_modules = []

ext_modules += [
    Extension("_domainGeneration", ["PMMoTo/_domainGeneration.pyx"],include_dirs=['PMMoTo']),
    Extension("_distance", ["PMMoTo/_distance.pyx"],include_dirs=['PMMoTo']),
    Extension("PMMoTo.drainage", ["PMMoTo/drainage.pyx"],include_dirs=['PMMoTo']),
]
cmdclass.update({'build_ext': build_ext})

setup(
    name="PMMoTo",
    version="0.0.1",
    packages=['PMMoTo'],
    author="Timothy M Weigand",
    description="Porous Media Morphological and Topological Analysis Toolkit",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/tmweigand/PMMoTo",
    cmdclass=cmdclass,
    ext_modules=cythonize(ext_modules,annotate=True),
    include_dirs=numpy.get_include(),
    install_requires=[
        'numpy>=1.22.3',
        'mpi4py>=3.1.3'
    ]
)

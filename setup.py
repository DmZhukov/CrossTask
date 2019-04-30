from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'dp',
    ext_modules = cythonize("dp.pyx"),
    include_dirs=[numpy.get_include()]
)

__author__ = 'mikhail'

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("gauss_seidel_cython.pyx"))
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
ext_modules=[
    Extension("build.perishableInventory.wrap",            ["perishableInventory.pyx"])
	]


setup(name          = 'build',
      packages      = find_packages(),
      cmdclass      = {'build_ext': build_ext},
      ext_modules   = ext_modules,
      author        = 'Parshan Pakiman',
      author_email  = 'ppakim2@uic.edu',
      url           = 'https://parshanpakiman.github.io/homepage/',
     )
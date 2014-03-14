# -*- coding: utf-8 -*-
import os
from distutils.core import setup
import codecs


# shamelessly copied from VoroPy
def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname),
                       encoding='utf-8').read()

from krypy import __version__

setup(name='krypy',
      packages=['krypy', 'krypy.recycling'],
      version=__version__,
      description='Krylov subspace methods for linear systems',
      long_description=read('README.md'),
      author='André Gaul',
      author_email='gaul@web-yard.de',
      url='https://github.com/andrenarchy/krypy',
      requires=['numpy (>=1.7)', 'scipy (>=0.12)'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics'
          ],
      )

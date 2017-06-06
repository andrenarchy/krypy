# -*- coding: utf-8 -*-
import os
import codecs

# Use setuptools for these commands (they don't work well or at all
# with distutils).  For normal builds use distutils.
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


# shamelessly copied from VoroPy
def read(fname):
    try:
        content = codecs.open(
            os.path.join(os.path.dirname(__file__), fname),
            encoding='utf-8'
            ).read()
    except Exception:
        content = ''
    return content

setup(name='krypy',
      packages=['krypy', 'krypy.recycling'],
      version='2.1.6',
      description='Krylov subspace methods for linear systems',
      long_description=read('README.rst'),
      author='André Gaul',
      author_email='gaul@web-yard.de',
      url='https://github.com/andrenarchy/krypy',
      install_requires=['numpy (>=1.7)', 'scipy (>=0.13)'],
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

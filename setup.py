# -*- coding: utf-8 -*-

# Use setuptools for these commands (they don't work well or at all
# with distutils).  For normal builds use distutils.
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='krypy',
      packages=['krypy', 'krypy.recycling'],
      version='2.1.7',
      description='Krylov subspace methods for linear systems',
      long_description=open('README.md').read(),
      author='André Gaul',
      author_email='gaul@web-yard.de',
      url='https://github.com/andrenarchy/krypy',
      install_requires=['numpy (>=1.11)', 'scipy (>=0.17)'],
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

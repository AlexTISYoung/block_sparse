from setuptools import setup

setup(name='block_sparse',
      version='1.2.0a1',
      description='Computations with matrices with a (hierarchical) block-sparsity structure',
      url='http://github.com/alexTISYoung/block_sparse',
      author='Alexander I. Young',
      author_email='alextisyoung@gmail.com',
      license='MIT',
      classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',

            # Pick your license as you wish (should match "license" above)
            'License :: OSI Approved :: MIT License',

            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            'Programming Language :: Python :: 2.7',
      ],
      keywords='sparse linear algebra matrices',
      packages=['block_sparse'],
      install_requires=[
            'numpy',
        ],
      zip_safe=False)
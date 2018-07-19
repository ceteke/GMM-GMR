from setuptools import setup

setup(name='gmm_gmr',
      version='0.1',
      description='GMM-GMR for imitation learning',
      url='http://github.com/storborg/funniest',
      author='Cem Eteke',
      author_email='ceteke13@ku.edu.tr',
      license='MIT',
      packages=['gmm_gmr'],
      install_requires=[
          'scikit-learn',
          'dtw'
      ],
      zip_safe=False)
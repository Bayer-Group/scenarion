from setuptools import setup
 
setup(name='scenarion',
      version='0.0.1',
      description='Scenario testing tool for interpreting models',
      author='Jeff Pobst',
      packages=['scenarion'],
      install_requires=[
            'pandas',
            'tqdm',
            'matplotlib',
            'seaborn',
            'scikit-learn>=0.20.0'],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
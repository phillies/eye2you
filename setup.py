from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='eye2you',
      version='0.1.dev0',
      description='fundus image analysis',
      long_description=readme(),
      classifiers=[
          'Development Status :: 2 - Pre-Alpha', 'Intended Audience :: Healthcare Industry',
          'Intended Audience :: Science/Research', 'License :: Other/Proprietary License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Medical Science Apps.'
      ],
      keywords='',
      url='https://github.com/phillies/eye2you',
      author='Philipp Lies',
      author_email='phil@lies.io',
      license='None',
      packages=['eye2you'],
      install_requires=['pandas', 'torch', 'torchvision', 'opencv-python', 'sklearn', 'tqdm', 'yaml'],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'])

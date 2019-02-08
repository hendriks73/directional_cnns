from setuptools import setup, find_packages

import glob

# define scripts to be installed by the PyPI package
scripts = glob.glob('bin/*')

setup(name='directional_cnns',
      version='0.1',
      packages=find_packages(),
      description='Experiments with directional CNNs for key and tempo estimation',
      author='Hendrik Schreiber',
      author_email='hs@tagtraum.com',
      license='Attribution 3.0 Unported (CC BY 3.0)',
      scripts=scripts,
      install_requires=[
          'h5py',
          'scikit-learn',
          'librosa',
          'tensorflow-gpu==1.10.1;platform_system!="Darwin"',
          'tensorflow==1.10.1;platform_system=="Darwin"',
      ],
      include_package_data=True,
      zip_safe=False)

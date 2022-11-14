import os

directory = os.path.abspath(os.path.dirname(__file__))

setup(name='volgar',
      version='0.1',
      description='A scientific computing frameword',
      author='Lilanka Pathirage',
      packages = ['volgar'],
      install_requires=['numpy'],
      python_requires='>=3.8',
      include_package_data=True)
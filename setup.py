from setuptools import setup

setup(
    name='spectrabuster',
    url='https://github.com/eduardosprp/spectrabuster',
    author='Eduardo Lopes Dias',
    author_email='eduardotogpi@usp.br',
    packages=['spectrabuster'],
    install_requires=['numpy', 'seabreeze'],
    version='0.1',
    license='MIT',
    description='Tools for simplifying the procecssing and storing of spectrums acquired using pyseabreeze',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)

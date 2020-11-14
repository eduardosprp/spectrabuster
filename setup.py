from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='spectrabuster',
    url='https://github.com/jladan/package_demo',
    author='Eduardo Lopes Dias',
    author_email='eduardotogpi@usp.br',
    # Needed to actually package something
    packages=['spectrabuster'],
    # Needed for dependencies
    install_requires=['numpy'],['seabreeze']
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='Tools for simplifying the procecssing and storing of spectrums acquired using pyseabreeze',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)

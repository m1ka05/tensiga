from setuptools import setup

setup(
    name='tensiga',
    version='0.1.0',
    author='Michał Ł. Mika',
    author_email='michal@mika.sh',
    packages=['tensiga', 'tensiga.test'],
    scripts=[],
    url='http://pypi.python.org/pypi/tensiga/',
    license='LICENSE',
    description="Implementation of a matrix-free isogeometric Galerkin method for Karhunen-Loeve approximation of random fields using tensor product splines, tensor contraction and interpolation based quadrature",
    long_description=open('README.md').read(),
    install_requires=[
        "pytest",
        "numpy",
        "numba",
        "scipy",
        "scikit-sparse",
        "vtk",
        "pyvista",
    ],
)

# tensiga

Implementation of a matrix-free isogeometric Galerkin method for Karhunen-Loeve approximation of random fields using tensor product splines, tensor contraction and interpolation based quadrature.

## Install

Until I register it as a public pip package, I recommend the following procedure:

1. Clone the repository to your local machine and navigate to the main directory
2. Run ``sudo pip install -e ./`` in order to install the ``tensiga`` package locally on your machine. The installation references to the cloned repository. You can work on this repository and changes will be immediately applied to the installation.

## Examples

Under ``bin/`` you can find some examples to run. Other than that, many modules and routines have sample code in the ``__main__`` function, which can be run directly from the terminal, i.e.

    $ python3 -i tensiga/iga/bfun.py
    $ python3 -i tensiga/iga/bfuns.py
    $ python3 -i tensiga/iga/Bspline.py

Not all examples have been adapted to this new package structure. Some work is needed.

## Docs

The documentation can be generated via

    $ cd docs
    $ make html

This generates documentation at ``build/html/`` which can be conveniently access by starting a local http server, i.e.

    $ cd docs/build/html
    $ python3 -m http.server

## Disclaimer

This is work-in-progress. The repository and the code base will change significantly before publication. I need to

1. Write some more tests
2. Update examples in ``__main__`` sections
3. Write significant documentation
4. Register the package

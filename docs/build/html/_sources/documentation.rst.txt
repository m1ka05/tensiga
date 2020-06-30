Documentation
==============================================================================

This documentation covers the implementation of the new Galerkin method. Examples
for each module can be found in the source code in the main section of the
respective module. To gain an overview over the code base reference
the :ref:`genindex`.

IGA package
----------------

.. autofunction:: tensiga.iga.fspan.fspan(u, p, U)

.. autofunction:: tensiga.iga.auxkv.auxkv(U)

.. autofunction:: tensiga.iga.bfun.bfun(i, u, p, U)

.. autofunction:: tensiga.iga.bfuns.bfuns(i, u, p, U)

.. autofunction:: tensiga.iga.bfunsop.bfunsop(u, p, U)

.. autofunction:: tensiga.iga.bfunsop.bfunsmat(u, p, U)

.. autofunction:: tensiga.iga.dbfun.dbfun(i, n, u, p, U)

.. autofunction:: tensiga.iga.dbfuns.dbfuns(i, n, u, p, U)

.. autofunction:: tensiga.iga.dbfunsop.dbfunsop(i, n, u, p, U)

.. autofunction:: tensiga.iga.kntinsop.kntinsop(i, n, u, p, U)

.. autofunction:: tensiga.iga.ktsinsop.ktsinsop(n, u, p, U)

.. automodule:: tensiga.iga.Bspline
   :members:


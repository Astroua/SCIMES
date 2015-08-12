Spectral Clustering for Molecular Emission Segmentation
=======================================================

.. image:: scimes_logo.png
   :width: 200px
   :align: center

``SCIMES`` idenfifies relevant molecular gas structures within
dendrograms of emission using the spectral clustering paradigm

Documentation
-------------

.. toctree::
   :maxdepth: 2

   installation.rst
   description.rst

Reporting issues and getting help
---------------------------------

Please help us improve this package by reporting issues via `GitHub
<https://github.com/dendrograms/astrodendro/issues>`_. You can also open an
issue if you need help with using the package.

Developers
----------

This package was developed by:

* Thomas Robitaille
* Chris Beaumont
* Adam Ginsburg
* Braden MacDonald
* Erik Rosolowsky

Acknowledgments
---------------

Thanks to the following users for using early versions of this package and
providing valuable feedback:

* Katharine Johnston

Citing astrodendro
------------------

If you make use of this package in a publication, please consider adding the
following acknowledgment:

*This research made use of astrodendro, a Python package to compute dendrograms
of Astronomical data (http://www.dendrograms.org/)*

If you make use of the analysis code (:doc:`catalog`) or read/write FITS files,
please also consider adding an acknowledgment for Astropy (see
`<http://www.astropy.org>`_ for the latest recommended citation).

Public API
----------

.. toctree::
   :maxdepth: 1

   api/astrodendro.dendrogram.Dendrogram
   api/astrodendro.dendrogram.periodic_neighbours
   api/astrodendro.structure.Structure
   api/astrodendro.plot.DendrogramPlotter
   api/astrodendro.analysis
   api/astrodendro.pruning

Step-by-step description of the algorithm
=========================================
This guide provides an intuitive description of the steps followed 
by ``SCIMES`` to identify structures within a dataset. 
The full formalism is reported in <SCIMES paper link>.

The dataset
-----------
``SCIMES`` can be applied to position-position-velocity (PPV) data cubes, position-position-position simulations or position-position images. For this example we use the PPV data cube of
the Orion-Monoceros complex imaged in 12CO(1-0) by 
`Wilson et al. 2005, A&A, 430, 523W <http://adsabs.harvard.edu/abs/2005A%26A...430..523W>`_ whose presents a spatial and spectral resolution of ~ 1 pc (at an average distance to the complex of 450 pc) and 0.65 km/s, respectively.


Building the dendrogram
------------------------
A dendrogram is a tree that represents the hierarchical structure in the data (`Rosolowsky et al. 2008, ApJ, 679, 1338R <http://adsabs.harvard.edu/abs/2008ApJ...679.1338R>`_; see also the description for the `astrodendro algorithm core <https://dendrograms.readthedocs.org/en/latest/algorithm.html>`_). It is composed of two types of structures: branches, which are structures which split into multiple sub-structures, and leaves, which are structures that have no sub-structure (i.e. local maxima). The trunk is a super-structure that has no parent structure, and contains all other structures. In the Orion-Monoceros dataset, branches represent potential GMCs, leaves are essentially clumps within the GMCs, and the trunk can be considered as the full star forming complex. 


How can we identify relevant objects within the dendrogram? We need first to turn the dendrogram into a graph.

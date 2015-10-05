``SCIMES`` tutorial
====================

In this tutorial we will show the steps necessary to obtain the segmentation of the 
Orion-Monoceros dataset (:download:`orion.fits`, public available on 
`<https://www.cfa.harvard.edu/rtdc/CO/NumberedRegions/DHT27/index.html>`_) presented in the ``SCIMES`` paper.

Building the dendrogram and the structure catalog
----------------------------------------
First, we need to compute the dendrogram and its related catalog,
i.e. the inputs of  ``SCIMES``. In this example we are dealing with 
real observations. Therefore, we have to open a FITS file by using,
for example, ``astropy`` :

    >>> from astropy.io import fits
    >>> data = fits.getdata('observation.fits')

Afterward the dendrogram can be computed:

    >>> from astrodendro import Dendrogram
    >>> d = Dendrogram.compute(data)

the astrodendro.dendrogram.Dendrogram class has various tuning 
parameters. To explore them and their meaning, please refer to:
`Computing and exploring dendrograms <https://dendrograms.readthedocs.org/en/latest/using.html>`_.

The dendrogram of Orion-Monoceros data shown in the paper has be
obtained using the following parameters:

    >>> sigma = 0.3 #K, noise level
    >>> ppb = 1.3 #pixels/beam
    >>> from astropy.io import fits
    >>> data = fits.getdata('orion.fits')
    >>> from astrodendro import Dendrogram
    >>> d = Dendrogram.compute(data, min_value=0, min_delta=2*sigma, min_npix=3*ppb)

where ``min_value`` has be set to 0, since the cube was previously
masked with a 2 sigma cut.

``SCIMES`` applies the spectral clustering based on the properties of
all structures within the dendrogram. The dendrogram catalog
can be obtained by the `ppv_catalog() <https://dendrograms.readthedocs.org/en/latest/api/astrodendro.analysis.html#astrodendro.analysis.ppv_catalog>`_ or `pp_catalog() <https://dendrograms.readthedocs.org/en/latest/api/astrodendro.analysis.html#astrodendro.analysis.pp_catalog>`_ functions, for example:

    >>> from astrodendro import ppv_catalog
    >>> from astropy import units as u
    >>> metadata = {}
    >>> metadata['data_unit'] = u.Jy / u.beam
    >>> cat = ppv_catalog(d, metadata)

Further information about the dendrogram catalog functions can be found here: `Making a catalog <https://dendrograms.readthedocs.org/en/latest/catalog.html#making-a-catalog>`_.

Clustering the dendrogram
------------------------
The clustering of the dendrogram is obtained through the 
:class:`~scimes.SpectralCloudstering` class which requires as inputs
the dendrogram and its related catalog:

    >>> from scimes import SpectralCloudstering
    >>> dclust = SpectralCloudstering(d, cat)

By default, the clustering is performed on the aggregate affinity matrix given by
the `element-wise multiplication of the luminosity and the volume
matrix <http://scimes.readthedocs.org/en/latest/algorithm.html#from-the-graph-to-the-affinity-matrix>`_.  If instead you want
to perform the clustering based on volume only, ignoring luminosity, this can be achieved by setting:  

    >>> dclust = SpectralCloudstering(d, cat, cl_luminosity = False)

or if only the luminosity matrix is needed:

    >>> dclust = SpectralCloudstering(d, cat, cl_volume = False)

The :class:`~scimes.SpectralCloudstering` class provides several
optional inputs:

* ``user_k``: the number of clusters expected can be provided as an
  input. In this case, ``SCIMES`` will not make any attempt to guess
  it from the affinity matrix.

* ``user_ams``: if clustering based on a different property than
  volume and luminosity is wanted, this can be obtained by providing a
  user defined affinity matrix. This matrix needs to be ordered according to
  the dendrogram leaves indexing. Several matrices based on various
  properties can be provided all together; ``SCIMES`` aggregates them
  and generates the clustering based on all these properties.

* ``user_scalpars``: the scaling parameters of the affinity matrices
  can be provided as input. The scaling parameters are used to suppress
  some affinity values of the matrix and enhance others by
  rescaling the matrices with a Gaussian kernel. Also, this operation
  normalizes the matrices and prompts the user whether the matrices should be aggregated
  or this step should be skipped, proceeding directly to the clustering. The choice of the scaling parameters
  might influence the final result. If not provided, ``SCIMES``
  estimates them directly from the affinity matrices.

* ``savesingles``: by definition single leaves do not form clusters,
  since clusters are constituted by at least two objects. Therefore, they
  are eliminated by default from the final cluster counts. For some
  applications, as in case of low resolution observations,
  single leaves might represent relevant entities that need to be
  retained. This keyword forces ``SCIMES`` to consider unclustered and
  isolated leaves as independent clusters that will appear in the
  final cluster index catalog.       

Clustering results
--------------
The main output of the algorithm, ``clusters``, is a list of dendrogram
indices representing the relevant structures within the dendrogram according
to the scale of the observation and the affinity criteria used. In the
case of Orion-Monoceros, the properties of the structures are the
equivalent to "Giant Molecular Clouds". Those structures are already
present in the dendrogram. The hierarchy can be accessed
following the instructions on the `astrodendro documentation page  <https://dendrograms.readthedocs.org/en/latest/using.html#exploring-the-dendrogram>`_,
while their properties and statistics are collected in the dendrogram `pp <https://dendrograms.readthedocs.org/en/latest/api/astrodendro.analysis.html#astrodendro.analysis.PPStatistic>`_ or `ppv <https://dendrograms.readthedocs.org/en/latest/api/astrodendro.analysis.html#astrodendro.analysis.PPVStatistic>`_ catalog.
``SCIMES`` provides other outputs that result from the
clustering analysis:

* ``affmats``: numpy arrays containing the affinity matrices produced
  by the algorithm or provided as inputs by the user. The indices of
  those matrices represent the ``leaves`` of the dendrogram permuted
  in order to make the possible matrix block structure emerge. The
  permutation, however, does not influence the following spectral embedding.

* ``escalpars``: list containing the estimated scale parameters
  from the clustering analysis associated with the different input affinity
  matrices. Scaling parameters represent maximal properties (by
  default ``volume`` and ``luminosity``, or ``flux``) that the final
  structures tend to have.

* ``silhouette``: float showing the silhouette of the selected
  clustering configuration. This value ranges between 0 and 1 and
  represents the goodness of the clustering, where values close to 0
  indicate poor clustering, while values close to 1 indicate well
  separated clusters (i.e. good clustering)

``SCIMES`` visualizes the clusters within the dendrogram throught the 
`plot_tree <https://dendrograms.readthedocs.org/en/latest/api/astrodendro.plot.DendrogramPlotter.html#astrodendro.plot.DendrogramPlotter.plot_tree>`_ method of ``astrodendro``. Each cluster is indicated
with a different random color. 

Together, ``SCIMES`` generates the assignment cube of the clouds
through the `get_mask <https://dendrograms.readthedocs.org/en/latest/api/astrodendro.structure.Structure.html#astrodendro.structure.Structure.get_mask>`_ method of ``astrodendro``.  Pixels within a given cloud are labeled with a number related to the index of the dendrogram.

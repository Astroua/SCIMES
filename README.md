About
-----

The "Spectral Clustering for Interstellar Molecular Emission Segmentation" module for Python, ``SCIMES``, uses the spectral clustering approach to identify relevant structures within dendrogram of emission from observational data or simulation.  

Documentation
-------------

-  For information on installing and using ``SCIMES``, please visit [http://scimes.readthedocs.org/](http://scimes.readthedocs.org/)
-  If you are installing this version, use the following command: pip install "git+https://github.com/Astroua/SCIMES.git@new_version"
-  Please cite "Colombo et al. 2015" and "Bazzi et al. 2025b" for credits.
Reporting issues
----------------

Please report issues via the [GitHub issue tracker](https://github.com/Astroua/scimes/issues)

Changes
----------

- Parallel processing works now. SpectralCloudstering automatically specifies `n_jobs=-1` to make use of all cores.
- `tqdm` module is required and used for tracing the progress of SCIMES.
- Uses scikit-learn instead of a deprecated version of sklearn.
  

Credits
-------

This package and the related paper has been written by:

* [Dario Colombo](https://github.com/dcolombo)
* [Erik Rosolowsky](https://github.com/low-sky)
* [Adam Ginsburg](https://github.com/keflavich)
* [Ana Duarte-Cabral](https://github.com/anacabral)
* [Annie Hughes](Annie.Hughes@irap.omp.eu)
* [Zein Bazzi](https://github.com/Zeinbazzi)

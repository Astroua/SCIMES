# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
        import numpy as np
        from matplotlib import pyplot as plt
        from astrodendro import Dendrogram, ppv_catalog
        from astropy import units as u
        from sklearn import metrics
        from sklearn.cluster import spectral_clustering
        from itertools import combinations

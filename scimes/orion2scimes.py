import numpy as np

from astrodendro import Dendrogram, ppv_catalog
from astropy import units as u
from astropy.io import fits

from scimes import SpectralCloudstering

import aplpy

filename = 'orion_12CO'

#%&%&%&%&%&%&%&%&%&%&%&%
#    Make dendrogram
#%&%&%&%&%&%&%&%&%&%&%&%
print 'Make dendrogram from the full cube'
hdu = fits.open(filename+'.fits')[0]
data = hdu.data
hd = hdu.header

# Survey designs
sigma = 0.3 #K, noise level
ppb = 1.3 #pixels/beam
	        
d = Dendrogram.compute(data, min_value=sigma, \
	                min_delta=2*sigma, min_npix=3*ppb, verbose = 1)


# Plot the tree
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111)            
ax.set_yscale('log')
ax.set_xlabel('Structure')
ax.set_ylabel('Flux')

p = d.plotter()
p.plot_tree(ax, color='black')


#%&%&%&%&%&%&%&%&%&%&%&%&%&%
#   Generate the catalog
#%&%&%&%&%&%&%&%&%&%&%&%&%&%
print "Generate a catalog of dendrogram structures"
metadata = {}
metadata['data_unit'] = u.Jy #This should be Kelvin (not yet implemented)!
cat = ppv_catalog(d, metadata)


#%&%&%&%&%&%&%&%&%&%&%&%&%&%
#     Running SCIMES
#%&%&%&%&%&%&%&%&%&%&%&%&%&%
print "Running SCIMES"
dclust = SpectralCloudstering(d, cat, criteria = ['volume'])

print "Visualize the clustered dendrogram"
dclust.showdendro()

print "Produce the assignment cube"
dclust.asgncube(hd)


#%&%&%&%&%&%&%&%&%&%&%&%&%&%
#     Image the result
#%&%&%&%&%&%&%&%&%&%&%&%&%&%
print "Image the results with APLpy"

clusts = dclust.clusters
colors = dclust.colors
hdu = fits.open(filename+'_mom0.fits')[0]
	    
fig = aplpy.FITSFigure(hdu, figsize=(8, 6), convention='wells')
fig.show_colorscale(cmap='gray', vmax=36, stretch = 'sqrt')


count = 0
for c in clusts:

	mask = d[c].get_mask()
	mask_hdu = fits.PrimaryHDU(mask.astype('short'), hdu.header)

	mask_coll = np.amax(mask_hdu.data, axis = 0)
	mask_coll_hdu = fits.PrimaryHDU(mask_coll.astype('short'), hdu.header)
	                
	fig.show_contour(mask_coll_hdu, colors=colors[count], linewidths=1, convention='wells')

	count = count+1
	        
fig.tick_labels.set_xformat('dd')
fig.tick_labels.set_yformat('dd')

fig.add_colorbar()
fig.colorbar.set_axis_label_text(r'[(K km/s)$^{1/2}$]')


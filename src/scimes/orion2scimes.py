import numpy as np

from matplotlib import pyplot as plt
import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astrodendro import Dendrogram#, ppv_catalog
from astrodendro_analysis import ppv_catalog
from astropy import units as u
from astropy.io import fits
import astropy.wcs as wcs

from scimes import SpectralCloudstering

#import aplpy


# Function to remove the 3rd dimension in a 
# spectroscopic cube header
def hd2d(hd):

	# Create bi-dimensional header
	mhd = fits.PrimaryHDU(np.zeros([hd['NAXIS2'],hd['NAXIS1']])).header

	for i in ['1','2']:
		for t in ['CRVAL','CRPIX','CDELT','CTYPE','CROTA','CUNIT']:
			if hd.get(t+i) != None:
				mhd[t+i] = hd[t+i]

	for t in ['BUNIT','BMAJ','BMIN','BPA','RESTFRQ']:
		if hd.get(t) != None:
			mhd[t] = hd[t]

	return mhd


# Function to generate the integrated intensity map
def mom0map(hdu):

	hd = hdu.header
	data = hdu.data

	# Generate moment 0 map in K km/s
	mom0 = np.nansum(data,axis=0)*abs(hd['CDELT3'])/1000.

	return fits.PrimaryHDU(mom0,hd2d(hd))



filename = 'orion_12CO'

#%&%&%&%&%&%&%&%&%&%&%&%
#    Make dendrogram
#%&%&%&%&%&%&%&%&%&%&%&%
print('Make dendrogram from the full cube')
hdu = fits.open('../%s.fits' % filename)[0]
data = hdu.data
hd = hdu.header

# Survey designs
sigma = 0.3 #K, noise level
ppb = 1.3 #pixels/beam
	        
d = Dendrogram.compute(data, min_value=sigma, \
	                min_delta=2*sigma, min_npix=3*ppb, verbose = 1)


#%&%&%&%&%&%&%&%&%&%&%&%&%&%
#   Generate the catalog
#%&%&%&%&%&%&%&%&%&%&%&%&%&%
print("Generate a catalog of dendrogram structures")
metadata = {}
metadata['data_unit'] = u.Jy #This should be Kelvin (not yet implemented)!
cat = ppv_catalog(d, metadata)


#%&%&%&%&%&%&%&%&%&%&%&%&%&%
#     Running SCIMES
#%&%&%&%&%&%&%&%&%&%&%&%&%&%
print("Running SCIMES")
dclust = SpectralCloudstering(d, cat, hd, rms=sigma)


#%&%&%&%&%&%&%&%&%&%&%&%&%&%
#     Image the result
#%&%&%&%&%&%&%&%&%&%&%&%&%&%
print("Visualize the clustered dendrogram")
dclust.showdendro()

print("Visualize collapsed maps of the assignment cubes")
cubes = [dclust.clusters_asgn,\
		dclust.leaves_asgn,\
		dclust.trunks_asgn]
titles = ['Clusters', 'Leaves', 'Trunks']

for cube, title in zip(cubes, titles):

	plt.figure()
	plt.imshow(np.nanmax(cube.data,axis=0),origin='lower',\
			interpolation='nearest',cmap='jet')
	plt.title(title+' assignment map')
	plt.colorbar(label='Structure label')
	plt.xlabel('X [pixel]')
	plt.ylabel('Y [pixel]')


print("Image the results")

from astropy.visualization import simple_norm

clusts = dclust.clusters
colors = dclust.colors
# Create Orion integrated intensity map
mhdu = mom0map(hdu)

norm = simple_norm(mhdu.data, 'sqrt')

w = wcs.WCS(mhdu.header)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1,1,1,projection=w)

im = ax.imshow(mhdu.data,origin='lower',\
	interpolation='nearest',cmap='gray',norm=norm)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=maxes.Axes)
cbar = fig.colorbar(im, cax=cax, orientation = 'vertical')
cbar.set_label(r'(K km/s)$^{1/2}$', fontsize = 12)

ax.set_xlabel(r'$l$ [$^{\circ}$]')
ax.set_ylabel(r'$b$ [$^{\circ}$]')


count = 0
for c in clusts:

	mask = d[c].get_mask()
	mask_coll = np.nanmax(mask,axis=0)

	ax.contour(mask_coll, colors=colors[count], linewidths=2, levels = [0])

	count = count+1


plt.show()
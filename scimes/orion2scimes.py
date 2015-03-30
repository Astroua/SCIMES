import warnings
import os.path

import numpy as np
import math
import aplpy
import random

from matplotlib import pyplot as plt

from astrodendro import Dendrogram, ppv_catalog
from astropy import units as u
from astropy.io import fits
from astropy import wcs
from astropy.table.table import Column
from astropy.table import Table

from sklearn.cluster import spectral_clustering
from skimage.measure import regionprops

from scimes import SpectralCloudstering

from datetime import datetime

from pdb import set_trace as stop



def showdendro( dendro, cores_idx):

    # For the random colors
    r = lambda: random.randint(0,255)
             
    p = dendro.plotter()

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111)
            
    ax.set_yscale('log')
            
    cols = []

    
    # Plot the whole tree

    p.plot_tree(ax, color='black')

    for i in range(len(cores_idx)):

        col = '#%02X%02X%02X' % (r(),r(),r())
        cols.append(col)
        p.plot_tree(ax, structure=dendro[cores_idx[i]], color=cols[i], lw=3)

    ax.set_title("Final clustering configuration")

    ax.set_xlabel("Structure")
    ax.set_ylabel("Flux")


    return



def make_asgn(dendro, data_file, cores_idx = [], tag = '_', collapse = True):

    data = fits.open(data_file)[0]
    
    # Making the assignment cube
    if len(data.shape) == 3:
        asgn = np.zeros(data.shape, dtype = np.int32)

    if len(data.shape) == 4:
        asgn = np.zeros((data.shape[1],data.shape[2],data.shape[3]), dtype = np.int32)    
    
    for i in cores_idx:
        asgn[np.where(d[i].get_mask(shape = asgn.shape))] = i
            
        
    # Write the fits file
    asgn = fits.PrimaryHDU(asgn.astype('short'), data.header)
    
    os.system("rm -rf "+data_file+'_asgn_'+tag+'.fits')
    print "Write "+data_file+'_asgn_'+tag+'.fits'        
    asgn.writeto(data_file+'_asgn_'+tag+'.fits')

    # Collapsed version of the asgn cube
    if collapse:

        asgn_map = np.amax(asgn.data, axis = 0) 

        plt.matshow(asgn_map, origin = "lower")
        plt.colorbar()
        
            
    return





path = './'
#path = '/Volumes/Zeruel_data/ORION/'
#path = '/Users/Dario/Documents/dendrograms/'
filename = os.path.join(path, 'orion')

do_make = True
do_catalog = True
do_load = True


if do_make:

    # Make dendrogram

    do_load = False
    
    print 'Make dendrogram from the full cube'
    hdu = fits.open(path+'orion.fits')[0]
    data = hdu.data
    hd = hdu.header

    if data.ndim==4:
        data = data[0,:,:,:]

        
    #  Calculating pixel per beam
    bmaj = hd.get('BMAJ')
    bmin = hd.get('BMIN')
    cdelt1 = abs(hd.get('CDELT1'))
    cdelt2 = abs(hd.get('CDELT2'))

    if bmaj != None and bmin != None:
        ppbeam = abs((bmaj*bmin)/(cdelt1*cdelt2)*2*math.pi/(8*math.log(2)))
    else:
        ppbeam = 5

    # Getting a very rought estimation of rms   
    free1 = data[22:24,:,:]
    free2 = data[75:79,:,:]
    freeT = np.concatenate((free1,free2), axis=0) 
    rms = np.std(freeT[np.isfinite(freeT)])

    #rms = 0.25
    #ppbeam = 5
    
    d = Dendrogram.compute(data, min_value=2*rms, \
                           min_delta=2*rms, min_npix=3*ppbeam, verbose = 1)

    d.save_to(filename+'_dendrogram.fits')                       


if do_load:

    # Load data and dendrogram
    hdu = fits.open(filename+'.fits')[0]
    data = hdu.data
    hd = hdu.header
    
    if size(shape(data))==4:
        data = data[0,:,:,:]
    
    print 'Load dendrogram file: '+filename
    d = Dendrogram.load_from(filename+'_dendrogram.fits')

    print 'Load catalog file: '+filename
    cat = Table.read(filename+'_catalog.fits')

    #stop()                           
                           

                           

if do_catalog:
    
    # Making the catalog
    print "Making a simple catalog of dendrogram structures"
    metadata = {}
    metadata['data_unit'] = u.Jy #This should be Kelvin!

    cat = ppv_catalog(d, metadata)

    # Centre position
    xcens = cat['x_cen'].data
    ycens = cat['y_cen'].data
    vcens = cat['v_cen'].data
    
    hdm = fits.open(filename+'_distmap.fits')[0]
    distmap = hdm.data
    
    w = wcs.WCS(hd)
    xint, yint, vint = np.meshgrid(np.arange(data.shape[2]),\
                                    np.arange(data.shape[1]),\
                                    np.arange(data.shape[0]),\
                                    indexing='ij')

    ra, dec, vel = w.all_pix2world(xint,yint,vint,0)
    raxis = ra[:,0,0]
    daxis = dec[0,:,0]
    vaxis = vel[0,0,:]
        
        
    if hd.get('CTYPE3') == 'M/S':
        vaxis = vaxis/1000.

    # Pixel (voxel) size     
    dtor = math.pi/180.    

    deltax_pc = np.zeros(len(cat['radius'].data))
    deltay_pc = np.zeros(len(cat['radius'].data))

    for i in range(len(cat['radius'].data)):

        dist = distmap[ycens[i],xcens[i]]
                    
        deltax_pc[i] = abs(raxis[1]-raxis[0])*math.cos(dtor*np.mean(daxis))*dtor*dist
        deltay_pc[i] = abs(daxis[1]-daxis[0])*dtor*dist
            
    deltav_kms = abs(vaxis[1]-vaxis[0])

    # Physical constants
    mh = 1.673534*10**(-24)             # hydrogen mass CGS
    ms = 1.98900*10**33              # solar mass CGS
    pc = 3.0857*10**18                # parsec CGS
    xco = 2*10**20


    # Radius [pc]
    rads = 1.91*np.sqrt(deltax_pc*deltay_pc)*cat['radius'].data

    # Velocity dispersion [km/s]
    sigvs = deltav_kms*cat['v_rms'].data 

    # Luminosity and Luminosity mass
    luminosities = cat['flux'].data*deltav_kms*deltax_pc*deltay_pc    
    mlums = luminosities*(xco*(2.*mh)*(1.36)*(pc*pc)/ms)

    # Virial mass
    mvirs = 1040*sigvs**2*rads

    # Volume
    volumes = np.pi*rads**2*sigvs
    
    
    # Update the catalog
    cat.add_column(Column(name='radius_pc', data=rads))#, units=u.pc))
    cat.add_column(Column(name='sigv_kms', data=sigvs))#, units=u.km/u.s))
    cat.add_column(Column(name='luminosity', data=luminosities))#, units=u.K*u.km/u.s*u.pc**2))
    cat.add_column(Column(name='volume', data=volumes))        
    cat.add_column(Column(name='mlum', data=mlums))#, units=u.msolMass))
    cat.add_column(Column(name='mvir', data=mvirs))#, units=u.msolMass))
        
    # Save the catalog
    os.system('rm -rf '+filename+'_catalog.fits')
    cat.write(filename+'_catalog.fits')

# Calling the clustering procedure
gmcs = SpectralCloudstering(d, cat)

# Make the asgn cube
make_asgn(d, filename+'.fits', cores_idx = gmcs.clusters)

# Show the clustered dendrogram
showdendro(d, gmcs.clusters)

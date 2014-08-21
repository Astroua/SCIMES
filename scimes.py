import warnings
import os.path
import math

import numpy as np

from matplotlib import pyplot as plt

from astrodendro import Dendrogram, ppv_catalog
from astropy import units as u

from sklearn import metrics
from sklearn.cluster import spectral_clustering
from skimage.measure import regionprops

from datetime import datetime
from pdb import set_trace as stop




def mat_smooth(Mat):
    
    #Evaluate sigma

    nonmax = Mat != Mat.max()
    M = Mat[nonmax]

    Mlin = np.sort(M.ravel())
    Mdiff = np.ediff1d(Mlin)

    Mup = [0]+Mdiff.tolist()
    Mdn = Mdiff.tolist()+[0]

    ind_up = np.argsort(Mup)[::-1]
    ind_dn = np.argsort(Mdn)[::-1]

    # Search for the highest distance
    # between the values
    Mlsup = Mlin[ind_up]
    Mlsdn = Mlin[ind_dn]

    # Take the lower value larger distance
    Mlsup_sel = np.sort(Mlsup[0:5])
    Mlsdn_sel = np.sort(Mlsdn[0:5])
        
    sd_dist = (Mlsup[0]+Mlsdn[0])/2.

    print "-- Estimated smoothing parameter =", sd_dist
        
    NM = np.exp(-(Mat**2)/(sd_dist**2))
    NM[range(NM.shape[0]), range(NM.shape[1])] = 0

    stop()
            
    return NM
    




def aff_matrix(allleavidx, dictparents, dictprops):

    print "- Creating affinity matrices"

    num = len(allleavidx)        
    WAs = np.zeros((2,num,num))

    volumes = dictprops['volumes']
    luminosities = dictprops['luminosities']

    
    # Let's save one for loop
    n2 = num**2
    yy = np.outer(np.ones(num, dtype = np.int),range(num))
    xx = np.outer(range(num),np.ones(num, dtype = np.int))
    yy = yy.reshape(n2)
    xx = xx.reshape(n2)                
        
    # Going through the branch
    for lp in range(len(xx)):

        icont = xx[lp]
        jcont = yy[lp]
            
        i_idx = allleavidx[icont]
        imat = allleavidx.index(i_idx)
        
        if icont != jcont:
                
            j_idx = allleavidx[jcont]
            jmat = allleavidx.index(j_idx)
            
            ipars = dictparents[str(i_idx)]
            jpars = dictparents[str(j_idx)]

            # Find shorter list for the comparison
            lpars = min(ipars,jpars)

            # Finding the common parents
            aux_commons = list(set(ipars).intersection(set(jpars)))
            commons = [x for x in lpars if x in aux_commons]
                
            pi_idx = commons[0]
                                        
            # Volume
            wij = volumes[pi_idx]
            WAs[0,imat,jmat] = wij
            WAs[0,jmat,imat] = wij

            # Luminosity
            wij = luminosities[pi_idx]
            WAs[1,imat,jmat] = wij
            WAs[1,jmat,imat] = wij

            
    return WAs



def guessk(Mat, thresh = 0.2):
    
    M = 1*Mat
    M[M < thresh] = 0
    M[M > 0] = 1
    
    guess_clusters = np.zeros(M.shape[0])

    for i in range(M.shape[0]):
        guess_clusters[i] = sum(M[i,:])

    nguess_clusters = []
    trash_clusts = 0
    currn = 0
    
    for i in range(len(guess_clusters)-1):

        curr = guess_clusters[i]
        
        if curr == guess_clusters[i+1]:
            currn = currn+1
        else:

            if curr == 1:
                trash_clusts = trash_clusts+1
            
            currn = currn+1
            nguess_clusters.append(currn)
            currn = 0
        
    
    kguess = len(nguess_clusters)-trash_clusts+1
    
    return kguess



def clust_cleaning(dendro, allleavidx, allclusters, t_brs_idx):


    # Find the lowest level parent common for all clusters
    # to get the cores_idx
    pars_lev = []

    for leaf in allleavidx:
        pars_lev.append(dendro[leaf].parent.level)

    pars_lev = np.asarray(pars_lev)
    allclusters = np.asarray(allclusters)
        
    cores_idx = []
    doubt_idx = []
    doubt_clust = []
      
    for cluster in set(allclusters):

        # Selecting the cluster
        clust_idx = allclusters == cluster

        # Leaves in that cluster
        clust_leaves_idx = np.asarray(allleavidx)[clust_idx] 
        
        # Height of leaf parents into that cluster       
        clust_pars_lev = pars_lev[clust_idx] 
        clust_pars_lev = np.sort(clust_pars_lev)
        clust_pars_lev = clust_pars_lev.tolist()

        index = clust_pars_lev.index(clust_pars_lev[0])
        clust_leaves_idx = clust_leaves_idx.tolist()

        t_brs_idx = np.asarray(t_brs_idx)
        
        # To avoid "two leaves" branches as cores
        if len(dendro[clust_leaves_idx[index]].parent.sorted_leaves()) >= len(clust_leaves_idx) \
          or t_brs_idx[t_brs_idx == dendro[clust_leaves_idx[index]].parent.idx].size > 0:         
            core_candidate = dendro[clust_leaves_idx[index]].parent
        else:
            core_candidate = dendro[clust_leaves_idx[index]].parent.parent
            
                
        # Checking cluster cores.
        # A cluster core is a branch that contains only
        # leaves of that given core. Otherwise move to the upper level
        # and check again.

        size_core_candidate = len(core_candidate.sorted_leaves())
        size_cluster = len(clust_leaves_idx)
        
        count = 0
        while size_core_candidate > size_cluster:

            if count >= len(clust_pars_lev)-1:
                print 'Unassignable cluster', cluster
                break

            index = clust_pars_lev.index(clust_pars_lev[count+1])
            core_candidate = dendro[clust_leaves_idx[index]].parent
                
            size_core_candidate = len(core_candidate.sorted_leaves())
                        
            count = count + 1

        else:
            cores_idx.append(core_candidate.idx)


    return cores_idx        



def cloudstering(dendrogram, catalog, criteria, user_k):    

    # Listing all branches at dendrogram's trunk
    # and get useful information
    
    trunk = dendrogram.trunk
    
    branches = []
    trunk_brs_idx = []

    all_parents = []
    all_leaves = []
    all_leav_names = []
    all_leav_idx = []
    
    for t in trunk:

        if t.is_branch:
                    
            branches.append(t)
            trunk_brs_idx.append(t.idx)

            leaves = t.sorted_leaves()

            for leaf in leaves:
                
                parents = []
                levels = []

                all_leaves.append(leaf)
                all_leav_idx.append(leaf.idx)
                
                all_leav_names.append(str(leaf.idx))
                par = leaf.parent

                while par.idx != t.idx:

                    parents.append(par.idx)
                    par = par.parent
            
                parents.append(t.idx)
                parents.append(len(catalog['radius'].data)) # This is the trunk!
                
                all_parents.append(parents)
                        
    dict_parents = dict(zip(all_leav_names,all_parents))


                
    # Retriving needed properties from the catalog
    volumes = np.pi*catalog['radius_pc'].data**2*catalog['sigv_kms'].data
    luminosities = catalog['luminosity'].data

    t_volume = sum(volumes[trunk_brs_idx])
    t_luminosity = sum(luminosities[trunk_brs_idx])

    volumes = volumes.tolist()
    luminosities = luminosities.tolist()

    volumes.append(t_volume)
    luminosities.append(t_luminosity)


    dict_props = {'volumes':volumes, 'luminosities':luminosities}
    

    # Generating affinity matrices
    AMs = aff_matrix(all_leav_idx, dict_parents, dict_props)


    print "- Start spectral clustering"

    # Selecting the criteria and merging the matrices    
    for cr in criteria:

        print "-- Smoothing ", cr, " matrix"
        
        if criteria.index(cr) == 0:
            AM = mat_smooth(AMs[cr,:,:])                
        else:
            AM = AM*mat_smooth(AMs[cr,:,:])

            
    # Showing the final affinity matrix
    plt.matshow(AM)
    plt.colorbar()
    plt.title('Final Affinity Matrix')        

        
    # Guessing the number of clusters
    # if not provided

    if user_k == 0:   
        kg = guessk(AM)
    else:
        kg = user_k
            
    print '-- Guessed number of clusters =', kg
    
    if kg > 1:

        # Find the best cluster number
        sils = []

        min_ks = max(2,kg-5)
        max_ks = min(kg+20,len(all_leav_idx))
                
        for ks in range(min_ks,max_ks):
                                        
            all_clusters, evecs, _, _ = spectral_clustering(AM, n_clusters=ks, assign_labels = 'kmeans', eigen_solver='arpack')

            sil = metrics.silhouette_score(evecs, np.asarray(all_clusters), metric='euclidean')
            sils.append(sil)
                    
        # Use the best cluster number to generate clusters                    
        best_ks = sils.index(max(sils))+min_ks
        print "-- Best cluster number found through SILHOUETTE (", max(sils),")= ", best_ks

        all_clusters, evecs, _, _ = spectral_clustering(AM, n_clusters=best_ks, assign_labels = 'kmeans', eigen_solver='arpack')
                        
    else:

        print '-- Not necessary to cluster'
        all_clusters = np.zeros(len(leaves), dtype = np.int32)
    
    
    clust_branches = clust_cleaning(dendrogram, all_leav_idx, all_clusters, trunk_brs_idx)

    return clust_branches


    
    
class SpectralCloudstering:
    """
    Apply the spectral clustering to find the best 
    cloud segmentation out from a dendrogram.

    Parameters
    -----------

    dendrogram: 'astrodendro.dendrogram.Dendrogram' instance
    The dendrogram to clusterize

    catalog: 'astropy.table.table.Table' instance
    A catalog containing all properties of the dendrogram
    structures. Generally generated with ppv_catalog module

    cl_volume: bool
    Clusterize the dendrogram using the 'volume' criterium 

    cl_luminosity: bool
    Clusterize the dendrogram using the 'luminosity' criterium        

    user_k: int
    The expected number of clusters, if not provided
    it will be guessed automatically through the eigenvalues
    of the unsmoothed affinity matrix

    Return
    -------

    clusters: list
    The dendrogram branch indexes corresponding to the
    identified clusters 
        
    """

    def __init__(self, dendrogram, catalog, cl_volume = True, cl_luminosity=True, user_k = None):

        self.dendrogram = dendrogram
        self.catalog = catalog
        self.cl_volume = cl_volume
        self.cl_luminosity = cl_luminosity
        self.user_k = user_k or 0

        # Clustering criteria chosen
        self.criteria = []
        if self.cl_volume:
            self.criteria.append(0)
        if self.cl_luminosity:
            self.criteria.append(1)

        
        self.clusters = cloudstering(self.dendrogram, self.catalog, self.criteria, self.user_k)

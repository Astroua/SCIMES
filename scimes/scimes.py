import warnings
import os.path
import math

import numpy as np

from matplotlib import pyplot as plt

from astrodendro import Dendrogram, ppv_catalog
from astropy import units as u
from astropy.stats import median_absolute_deviation as mad

from sklearn import metrics
from sklearn.cluster import spectral_clustering
from skimage.measure import regionprops

from datetime import datetime
from pdb import set_trace as stop




def mat_smooth(Mat, scalpar = 0):
    
    #Using local scaling
    
    if scalpar == 0:

        print "-- Local scaling"

        dr = np.std(Mat, axis=0)
        sigmar = np.tile(dr,(Mat.shape[0],1))
        sigmas = sigmar*sigmar.T                
        
    else:

        print "-- Using provided scaling parameter =", scalpar
        sigmas = scalpar**2
            
    NM = np.exp(-(Mat**2)/sigmas)
    NM[range(NM.shape[0]), range(NM.shape[1])] = 0
            
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
        
        if icont > jcont:
                
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
    np.fill_diagonal(M, 1)
        
    guess_clusters = np.zeros(M.shape[0])

    for i in range(M.shape[0]):
        guess_clusters[i] = sum(M[i,:])

    kguess = 1
    i = 0

    while i < len(guess_clusters)-1:

        curr = int(guess_clusters[i])

        if curr != 1:
            kguess = kguess+1

        i = i + curr
    
    return kguess



def get_idx(struct):

    leav_list_idx = []
    sort_leav = struct.sorted_leaves()

    for l in sort_leav:
        leav_list_idx.append(l.idx)
    
    return leav_list_idx




def clust_cleaning(dendro, allleavidx, allclusters, dictpars, dictchilds):
        
    cores_idx = []
      
    for cluster in set(allclusters):

        # Selecting the cluster
        clust_idx = allclusters == cluster

        # Leaves and levels in that cluster
        clust_leaves_idx = np.asarray(allleavidx)[clust_idx]

        all_par_list = []
        
        for cli in clust_leaves_idx:

            par_list = dictpars[str(cli)]
            par_list = par_list[0:-1] #The lowest, the trunk, is not considered

            all_par_list = all_par_list + par_list

        all_par_list = list(set(all_par_list))
        
        core_clust_num = []
        clust_core_num = []
        
        for i in range(len(all_par_list)):
            
            sel_par = all_par_list[i]
            core_leaves_idx = dictchilds[str(sel_par)]

            # Leaves in the core but not in the cluster
            core_clust = list(set(core_leaves_idx) - set(clust_leaves_idx))
            core_clust_num.append(len(core_clust))

            # Leaves in the cluster but not in the core            
            clust_core = list(set(clust_leaves_idx) & set(core_leaves_idx))
            clust_core_num.append(len(clust_core))

        # The selected core must not contain other leaves than
        # those of the cluster, plus it is the one with the highest
        # number of cluster leaves contained    

        core_clust_num = np.asarray(core_clust_num)
        core_clust_num0 = np.where(core_clust_num == 0)
        
        if len(core_clust_num0[0]) > 0:
        
            all_par_list = np.asarray(all_par_list)
            all_par_list0 = all_par_list[core_clust_num0]
            all_par_list0 = np.asarray(all_par_list0)
            
            clust_core_num = np.asarray(clust_core_num)
            clust_core_num0 = clust_core_num[core_clust_num0]
            clust_core_num0 = np.asarray(clust_core_num0)
            
            max_num = max(clust_core_num0)
            cores = all_par_list0[np.where(clust_core_num0 == max_num)]           

            cores_idx = cores_idx + cores.tolist()
            
        else:

            print "Unassignable cluster ", cluster
            #stop()
            
    return cores_idx        




def cloudstering(dendrogram, catalog, criteria, user_k, user_ams, user_scalpars):    

    # Collecting all connectivity information into more handy lists
    all_structures_idx = range(len(catalog['radius'].data))

    all_leav_names = []
    all_leav_idx = []

    all_brc_names = []
    all_brc_idx = []

    all_parents = []
    all_children = []

    trunk_brs_idx = []
    two_clust_idx = []    
    mul_leav_idx = []
    
    for structure_idx in all_structures_idx:

        s = dendrogram[structure_idx]

        # If structure is a leaf find all the parents
        if s.is_leaf and s.parent != None:

            par = s.parent
            all_leav_names.append(str(s.idx))

            parents = []
            
            while par != None:

                parents.append(par.idx)
                par = par.parent
                
            parents.append(len(catalog['radius'].data)) # This is the trunk!
            all_parents.append(parents)
            
            
        # If structure is a brach find all its leaves
        if s.is_branch:

            all_brc_idx.append(s.idx)
            all_brc_names.append(str(s.idx))
            
            children = []
            
            for leaf in s.sorted_leaves():

                children.append(leaf.idx)
                
            all_children.append(children)

            # Trunk branches
            if s.parent == None:

                trunk_brs_idx.append(s.idx)
                all_leav_idx = all_leav_idx + children

                if s.children[0].is_branch or s.children[1].is_branch:
                    mul_leav_idx = mul_leav_idx + children
                else:
                    two_clust_idx.append(s.idx)
        
    two_clust_idx = np.unique(two_clust_idx).tolist()
    
    dict_parents = dict(zip(all_leav_names,all_parents))            
    dict_children = dict(zip(all_brc_names,all_children))    
    
    # Retriving needed properties from the catalog
    volumes = catalog['volume'].data
    luminosities = catalog['luminosity'].data

    t_volume = sum(volumes[trunk_brs_idx])
    t_luminosity = sum(luminosities[trunk_brs_idx])

    volumes = volumes.tolist()
    luminosities = luminosities.tolist()

    volumes.append(t_volume)
    luminosities.append(t_luminosity)
    
    dict_props = {'volumes':volumes, 'luminosities':luminosities}
     
    # Generating affinity matrices if not provided
    if user_ams == None:
        AMs = aff_matrix(all_leav_idx, dict_parents, dict_props, dendrogram)
    else:
        AMs = user_ams

        
    # Check whether the affinity matrix scaling parameter
    # are provided by the user, if so use them, otherwise
    # calculate them    

    if user_scalpars == None:
        scpars = np.zeros(max(criteria)+1)
    else:
        scpars = user_scalpars

                
    print "- Start spectral clustering"

    # Selecting the criteria and merging the matrices    
    for cr in criteria:

        print "-- Smoothing ", cr, " matrix"
        
        if criteria.index(cr) == 0:
            AM = mat_smooth(AMs[cr,:,:], scalpar = scpars[cr])                
        else:
            AM = AM*mat_smooth(AMs[cr,:,:], scalpar = scpars[cr])

            
    # Showing the final affinity matrix
    plt.matshow(AM)
    plt.colorbar()
    plt.title('Final Affinity Matrix')        

    
    # Making the reduced affinity matrices
    mul_leav_mat = []
    for mli in mul_leav_idx:
        mul_leav_mat.append(all_leav_idx.index(mli))

    mul_leav_mat = np.asarray(mul_leav_mat)
    rAM = AM[mul_leav_mat,:]
    rAM = rAM[:,mul_leav_mat]

    
    # Showing the reduced affinity matrix
    plt.matshow(rAM)
    plt.colorbar()
    plt.title('Reduced Affinity Matrix') 
            
        
    # Guessing the number of clusters
    # if not provided

    if user_k == 0:   
        kg = guessk(rAM)
    else:
        kg = user_k

    print '-- Reduced matrix number of clusters =', kg                    
    print '-- Total guessed number of clusters =', kg+len(two_clust_idx)
    
    if kg > 1:

        # Find the best cluster number
        sils = []

        min_ks = max(2,kg-5)
        max_ks = min(kg+20,len(all_leav_idx)-len(two_clust_idx)-1)
                
        for ks in range(min_ks,max_ks):
                                        
            all_clusters, evecs, _, _ = spectral_clustering(rAM, n_clusters=ks, assign_labels = 'kmeans', eigen_solver='arpack')

            sil = metrics.silhouette_score(evecs, np.asarray(all_clusters), metric='euclidean')
            sils.append(sil)
                    
        # Use the best cluster number to generate clusters                    
        best_ks = sils.index(max(sils))+min_ks
        print "-- Best cluster number found through SILHOUETTE (", max(sils),")= ", best_ks+len(two_clust_idx)

        all_clusters, evecs, _, _ = spectral_clustering(rAM, n_clusters=best_ks, assign_labels = 'kmeans', eigen_solver='arpack')
                        
    else:

        print '-- Not necessary to cluster'
        all_clusters = np.zeros(len(leaves), dtype = np.int32)

                    
    clust_branches = clust_cleaning(dendrogram, mul_leav_idx, all_clusters, dict_parents, dict_children)
    clust_branches = clust_branches + two_clust_idx

    print "-- Final cluster number (after cleaning)", len(clust_branches)
    
    return clust_branches, AMs  


    
    
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

    user_ams: numpy array
    User provided affinity matrix. Whether this is not
    furnish it is automatically generated through the
    volume and/or luminosity criteria

    user_scalpars: float
    User defined scaling parameter(s). Whether those are not
    furnish scaling parameters are automatically estimated. 


    Return
    -------

    clusters: list
    The dendrogram branch indexes corresponding to the
    identified clusters

    affmats: numpy array
    The affinity matrices calculated by the algorithm
        
    """

    def __init__(self, dendrogram, catalog, cl_volume = True, cl_luminosity=True, \
                 user_k = None, user_ams = None, user_scalpars = None):

        self.dendrogram = dendrogram
        self.catalog = catalog
        self.cl_volume = cl_volume
        self.cl_luminosity = cl_luminosity
        self.user_k = user_k or 0
        self.user_ams = user_ams
        self.user_scalpars = user_scalpars

        # Clustering criteria chosen
        self.criteria = []
        if self.cl_volume:
            self.criteria.append(0)
        if self.cl_luminosity:
            self.criteria.append(1)

        
        self.clusters, self.affmats = cloudstering(self.dendrogram, self.catalog, self.criteria, \
                                                   self.user_k, self.user_ams, self.user_scalpars)

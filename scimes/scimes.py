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
    
    #Evaluate sigma

    nonmax = Mat != Mat.max()
    M = Mat#[nonmax]

    Mlin = np.sort(M.ravel())
    Mdiff = np.ediff1d(Mlin)

    Mup = [0]+Mdiff.tolist()
    Mdn = Mdiff.tolist()+[0]

    if scalpar == 0:
    
        Mdiff0 = Mdiff[Mdiff != 0]

        # Find outliers
        outls = Mdiff0[Mdiff0 > np.mean(Mdiff0)+4*np.std(Mdiff0)]
        sd_dist = (Mlin[Mup.index(outls[0])]+Mlin[Mdn.index(outls[0])])/2
    
        print "-- Estimated scaling parameter =", sd_dist

    else:

        print "-- Using provided scaling parameter =", scalpar
        sd_dist = scalpar
            
    NM = np.exp(-(Mat**2)/(sd_dist**2))
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



def get_idx(struct):

    leav_list_idx = []
    sort_leav = struct.sorted_leaves()

    for l in sort_leav:
        leav_list_idx.append(l.idx)
    
    return leav_list_idx




def clust_cleaning(dendro, allleavidx, allclusters, t_brs_idx):


    # Find the lowest level parent common for all clusters
    # to get the cores_idx
    pars_lev = []
    pars_idx = []

    for leaf in allleavidx:
        pars_lev.append(dendro[leaf].parent.height)
        pars_idx.append(dendro[leaf].parent.idx)
        
    pars_lev = np.asarray(pars_lev)
    pars_idx = np.asarray(pars_idx)
    allclusters = np.asarray(allclusters)
        
    cores_idx = []
      
    for cluster in set(allclusters):

        # Selecting the cluster
        clust_idx = allclusters == cluster

        # Leaves in that cluster
        clust_leaves_idx = np.asarray(allleavidx)[clust_idx] 
        
        # Height of leaf parents into that cluster
        # the parent with the lowest height is supposed
        # to be the parent common to all leaves
        # and becomes the core candidate
        clust_pars_idx = np.asarray(pars_idx[clust_idx])       
        clust_pars_lev = pars_lev[clust_idx] 
        clust_pars_lev = np.argsort(clust_pars_lev)

        ord_pars_idx = clust_pars_idx[clust_pars_lev]
        ord_pars_idx = ord_pars_idx.tolist()
        
        t_brs_idx = np.asarray(t_brs_idx)
        core_candidate = dendro[ord_pars_idx[0]]
        
                                 
        # Checking cluster cores.
        # A cluster core is a branch that contains only
        # leaves of that given core. Otherwise move to the upper level
        # and check again.

        
        # First check if the core candidate contains all leaves of the
        # cluster, otherwise go one level down        
                    
        leav_core = get_idx(core_candidate)
        leav_cluster = clust_leaves_idx


        fst_check = list(set(leav_cluster) - set(leav_core))

        if len(fst_check)>0 and \
          t_brs_idx[t_brs_idx == core_candidate.idx].size == 0:
            core_candidate = core_candidate.parent
            leav_core = get_idx(core_candidate)
            
                
        # Difference between the two lists: leaves that
        # are in the core but not in the cluster
        diff_leav = list(set(leav_core) - set(leav_cluster))

        count = 1
        while len(diff_leav) > 0:

            core_candidate = dendro[leav_cluster[count]].parent
            leav_core = get_idx(core_candidate)
            new_cluster = leav_cluster[count:-1]            

            if len(new_cluster) == 0:
                print 'Unassignable cluster', cluster
                break
                    
            diff_leav = list(set(leav_core) - set(new_cluster))
            count = count+1

        else:
            cores_idx.append(core_candidate.idx)

    return cores_idx        




def cloudstering(dendrogram, catalog, criteria, user_k, user_ams, user_scalpars):    

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
        AMs = aff_matrix(all_leav_idx, dict_parents, dict_props)
    else:
        AMs = user_ams
        
    
    # Check whether the affinity matrix scaling parameter
    # are provided by the user, if so use them, otherwise
    # calculate them    

    if user_scalpars == None:
        scpars = np.zeros(len(criteria))
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

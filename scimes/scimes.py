"""
The SCIMES core package
"""
import numpy as np
import random
from itertools import combinations, cycle
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Column
from sklearn.metrics import silhouette_score
from sklearn.manifold import spectral_embedding
from sklearn.cluster.k_means_ import k_means

def mat_smooth(Mat, S2Nmat, s2nlim = 3, scalpar = None, lscal = False):
    """
    Estimate the scaling parameter and rescale
    the affinity matrix through a Gaussian kernel.
    
    Parameters
    -----------

    Mat: numpy array
        The affinity matrix to be rescaled.

    S2Nmat: numpy array
        Signal-to-noise ratio affinity matrix.
        If rms is np.nan S2Nmat is np.nan and
        the scaling parameter is searched between
        the 5 largest gaps.

    s2nlim: int or float
        Signal-to-noise limit above which the
        scaling parameter is calculated
        Needed only if S2Nmat is not np.nan.

    scalpar: float
        User-defined scaling parameter.

    lscal: boll
        Rescale the matrix using a local
        scaling approach.
        
    Return
    -------

    NM: numpy array
        Rescaled affinity matrix.

    sigmas: float
        The estimated scaling parameter.

    """
    
    # Using estimated global scaling    
    if scalpar is None and lscal == False:

        if not np.isnan(np.median(S2Nmat)):

            sMat = Mat[S2Nmat > s2nlim]
            Aff = np.unique(sMat.ravel())[1::]
            psigmas = (Aff+np.roll(Aff,-1))/2
                
            psigmas = psigmas[1:-1]        

            diff = np.roll(Aff,-1)-Aff                
            diff = diff[1:-1]

            sel_diff_ind = np.argmax(diff)

        else:

            Aff = np.unique(Mat.ravel())[1::]
            psigmas = (Aff+np.roll(Aff,-1))/2
                
            psigmas = psigmas[1:-1]        

            diff = np.roll(Aff,-1)-Aff                
            diff = diff[1:-1]

            # Old method
            sel_diff_ind = np.min(np.argsort(diff)[::-1][0:5])

        sigma = psigmas[sel_diff_ind]**2

        # New method:
        # larger difference range
        # taking the affinity value closer to the std of all affinities
        #sel_diff_ind = np.argsort(diff)[::-1][0:10]
        #spsigmas = psigmas[sel_diff_ind]
        #print "-- Affinity standard deviation:", np.std(Aff)
        #sigmas = spsigmas[np.argmin(np.abs(spsigmas - np.std(Aff)))]
        #sigmas = sigmas**2
    

        print("-- Estimated scaling parameter: %f" % np.sqrt(sigma))

    # Using local scaling        
    if scalpar is None and lscal == True:

        print("-- Local scaling")

        dr = np.std(Mat, axis=0)
        sigmar = np.tile(dr,(Mat.shape[0],1))
        sigma = sigmar*sigmar.T


    # Using user-defined scaling parameter
    if scalpar:

        print("-- User defined scaling parameter: %f" % scalpar)
        sigma = scalpar**2
            
    NM = np.exp(-(Mat**2)/sigma)
    NM[range(NM.shape[0]), range(NM.shape[1])] = 0

    return NM, sigma


def aff_matrix(num, trk, allleavidx, allbrcidx, brclevels, dictchildrens, props):

    """
    Generate the affinity matrices.
    
    Parameters
    -----------

    num: int
        Number of non isolated leaves.

    trk: int
        Dummy index of the trunk.

    allbrcidx: list
        List of all branches (parents) 
        indexes within the dendrogram.

    brclevels: list
        Dendrogram levels of all branches.

    dictchildrens: dictionary
        Descendants of all branches within
        the dendrogram.

    props: list of lists
        Properties of all leaf parents and
        ancestors within the dendrogram.
        
    Return
    -------

    WAs: numpy array
        Clustering criteria affinity matrices.

    """
    
    print("- Creating affinity matrices")

    Widx = np.zeros((num,num), dtype='int')+trk
    WAs = np.zeros((len(props),num,num))

    brclevels = np.asarray(brclevels)
    allbrcidx = np.asarray(allbrcidx)
    allleavidx = np.asarray(allleavidx)
    allleavpos = np.arange(len(allleavidx))


    """
    for l in np.unique(brclevels):

        lbrcidx = allbrcidx[brclevels == l]

        for p in lbrcidx:

            pleavidx = np.asarray(dictchildrens[str(p)])
            pleavpos = allleavpos[np.in1d(allleavidx,pleavidx)]

            x,y = np.meshgrid(pleavpos,pleavpos)
            Widx[x,y] = p
    """

    
    allbrcidx = allbrcidx[np.argsort(brclevels)]

    for p in allbrcidx:

        pleavidx = np.asarray(dictchildrens[str(p)])
        pleavpos = allleavpos[np.in1d(allleavidx,pleavidx)]
        x,y = np.meshgrid(pleavpos,pleavpos)
        Widx[x,y] = p

    Widx[np.diag_indices(num)] = 0

    for j,prop in enumerate(props):

        prop = np.asarray(prop)
        Wprop = prop[Widx.ravel()].reshape([num,num])
        Wprop[np.diag_indices(num)] = 0
        WAs[j,:,:] = Wprop

    return WAs



def guessk(Mat, thresh = 0.2):

    """
    Guess the number of clusters by couting
    the connected blocks in the affinity matrix.
    
    Parameters
    -----------

    Mat: numpy array
        The rescaled affinity matrix to guess the
        number of cluster from.

    thresh: float
        The threshold to mask the matrix and count
        the blocks.
        
    Return
    -------

    kguess: int
        Number of guessed clusters.

    """
    
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

    """
    np.fill_diagonal(M, 0)
    D = np.zeros(M.shape)

    for i in range(D.shape[0]):
        D[i,i] = sum(M[i,:])

    Lap = D - M
    eigv = np.abs(np.linalg.eigvals(Lap))

    kguess2 = len(np.where(eigv == 0)[0])    

    """
            
    return kguess




def clust_cleaning(allleavidx, allclusters, dictpars, dictchilds, dictancests, savebranches = False):

    """
    Remove clusters that do not corresponds to
    isolated dendrogram branches.
    
    Parameters
    -----------

    allleavidx: list
        List of all leaf indexes within the
        dendrogram.

    allclusters: list
        List of dendrogram indexes that
        correspond to significant objects
        (i.e. clusters).

    dictpars: dictionary
        Parents and ancestors of all leaves
        within the dendrogram.

    dictchilds: dictionary
        Children of all branches
        within the dendrogram.

    savebranches: bool
        Retain all isolated branches usually discarded
        by the cluster analysis.
        
    Return
    -------

    cores_idx: list
        The final cluster dendrogram indexes.

    """
        
    if savebranches == True:
        print("SAVE_BRANCHES triggered, all isolated branches will be retained")

    cores_idx = []

    _, ucidx = np.unique(allclusters, return_index=True)
    uclusters = allclusters[np.sort(ucidx)]
    rallclusters = np.zeros(len(allclusters), dtype=np.int32)

    for j,ucluster in enumerate(uclusters):
        rallclusters[allclusters == ucluster] = j

    for cluster in set(rallclusters):

        # Selecting the cluster
        clust_idx = rallclusters == cluster

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

            if savebranches == True:

                sel_num = np.zeros(len(all_par_list0))
                for par in all_par_list0:

                    ancests = dictancests[str(par)]
                    cancests = list(set(ancests) & set(all_par_list0))
                    sel_num[all_par_list0 == par] = len(cancests)

                cores = all_par_list0[sel_num <= 1]

            else:

                max_num = clust_core_num0 == max(clust_core_num0)
                cores = all_par_list0[max_num]

            cores_idx = cores_idx + cores.tolist()
            
        else:

            print("Unassignable cluster %i" % cluster)
            
    return cores_idx



def make_asgncube(dendro, asgn_idx, header, collapse = True):

    """
    Create a label cube with only the cluster (cloudster) IDs included.

    Parameters
    ----------
    dendro: 'astrodendro.dendrogram.Dendrogram' instance
        The clusterized dendrogram.

    header : `fits.Header`
        The header of the output assignment cube.  Should be the same
        header that the dendrogram was generated from.
        
    collapse : bool
        Collapsed (2D) version of the assignment cube.
        Requires matplotlib.

    Return
    -------
    asgn = 'astropy.io.fits.PrimaryHDU' instance
        Cube of labels.

    """

    data = dendro.data.squeeze()

    # Making the assignment cube
    asgn = np.zeros(data.shape, dtype=np.int32)-1

    for i in asgn_idx:
        asgn[dendro[i].get_mask(shape = asgn.shape)] = i

    # Write the fits file
    asgn = fits.PrimaryHDU(asgn.astype('short'), header)

    # Collapsed version of the asgn cube
    if collapse:

        asgn_map = np.amax(asgn.data, axis = 0) 

        plt.matshow(asgn_map, origin = "lower")
        cbar = plt.colorbar()
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Structure label', rotation=270)

    return asgn




def cloudstering(dendrogram, catalog, criteria, user_k, user_ams, user_scalpars, user_iter, 
    save_isol_leaves, save_clust_leaves, save_branches, blind, rms, s2nlim, locscal):

    """
    SCIMES main function. It collects parents/children
    of all structrures within the dendrogram, and their
    properties. It calls the affinity matrix-related
    functions (for creation, rescaling, cluster counting),
    and it runs several time the actual spectral clustering
    routine by calculating every time the silhouette of the
    current configuration. Input parameter are passed by the
    SpectralCloudstering class.
    
    Parameters
    -----------

    dendrogram: 'astrodendro.dendrogram.Dendrogram' instance
        The dendrogram to clusterize.

    catalog: 'astropy.table.table.Table' instance
        A catalog containing all properties of the dendrogram
        structures. Generally generated with ppv_catalog module.

    header: 'astropy.io.fits.header.Header' instance
        The header of the fits data the dendrogram was 
        generated from. Necessary to obtain the assignment cubes.

    criteria: list of strings
        Clustering criteria referred to the structure properties
        in the catalog (default ['volume', 'luminosity']).

    user_k: int
        The expected number of clusters, if not provided
        it will be guessed automatically through the eigenvalues
        of the unsmoothed affinity matrix.

    user_ams: numpy array
        User provided affinity matrix. Whether this is not
        furnish it is automatically generated through the
        volume and/or luminosity criteria.

    user_scalpars: list of floats
        User-provided scaling parameters to smooth the
        affinity matrices.

    user_iter: int
        User-provided number of k-means iterations.
    
    save_isol_leaves: bool
        Consider the isolated leaves (without parent) 
        as individual 'clusters'. Useful for low
        resolution data where the beam size
        corresponds to the size of a Giant
        Molecular Cloud.

    save_clust_leaves: bool
        Consider unclustered leaves as individual
        'clusters'. This keyword will not include
        the isolated leaves without parents.

    save_all_leaves: bool
        Trigger both save_isol_leaves and
        save_clust_leaves.

    save_branches: bool
        Retain all isolated branches usually discarded
        by the cluster analysis.

    save_all: bool
        Trigger all save_isol_leaves, 
        save_clust_leaves, and save_branches.        
    
    rms: int or float
        Noise level of the observation. Necessary tolist
        calculate the scaling parameter above a certain
        signal-to-noise ratio.

    s2nlim: int or float
        Signal-to-noise limit above which the
        scaling parameter is calculated. Needed
        only if rms is not np.nan.

    blind: bool
        Show the affinity matrices. 
        Matplotlib required.

    locscaling: bool
        Smooth the affinity matrices using a local
        scaling technique.


    Return
    -------

    clusts: list
        The dendrogram branch indexes corresponding to the
        identified clusters

    catalog: 'astropy.table.table.Table' instance
        The input catalog updated with dendrogram structure
        parent, ancestor, number of leaves, and type 
        ('T', trunks or branches without parent; 'B', branches
        with parent; 'L', leaves). 

    AMs: numpy array
        The affinity matrices calculated by the algorithm
    
    escalpars: list
        Estimated scaling parameters for the different
        affinity matrixes
    
    silhouette: float
        Silhouette of the best cluster configuration

    """

    # Collecting all connectivity and other information into more handy lists
    all_structures_idx = np.arange(len(catalog[criteria[0]].data), dtype='int')

    all_levels = []
    brc_levels = []

    all_leav_names = []
    all_leav_idx = []

    all_brc_names = []
    all_brc_idx = []

    all_parents = []
    all_children = []

    all_struct_names = []
    all_ancestors = []

    all_struct_ancestors = []
    all_struct_parents = []
    all_struct_types = []
    nleaves = []

    trunk_brs_idx = []
    two_clust_idx = []    
    mul_leav_idx = []

    s2ns = []

    for structure_idx in all_structures_idx:

        s = dendrogram[structure_idx]
        all_levels.append(s.level)
        
        s2ns.append(dendrogram[structure_idx].height/rms)

        all_struct_names.append(str(s.idx))
        all_struct_ancestors.append(s.ancestor.idx)
        if s.parent:
            all_struct_parents.append(s.parent.idx)
        else:
            all_struct_parents.append(-1)
        nleaves.append(len(s.sorted_leaves()))

        ancestors = []
        anc = s.parent
        while anc != None:

            ancestors.append(anc.idx)
            anc = anc.parent

        ancestors.append(s.idx)
        all_ancestors.append(ancestors)

        # If structure is a leaf find all the parents
        if s.is_leaf and s.parent != None:

            par = s.parent
            all_leav_names.append(str(s.idx))

            parents = []
            
            while par != None:

                parents.append(par.idx)
                par = par.parent
                
            parents.append(len(catalog[criteria[0]].data)) # This is the trunk!
            all_parents.append(parents)
            
        # If structure is a brach find all its leaves
        if s.is_branch:

            brc_levels.append(s.level)
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

                all_struct_types.append('T')

            else:

                all_struct_types.append('B')
        
        else:

            all_struct_types.append('L')


    two_clust_idx = np.unique(two_clust_idx).tolist()
    
    dict_parents = dict(zip(all_leav_names,all_parents))            
    dict_children = dict(zip(all_brc_names,all_children))    
    dict_ancestors = dict(zip(all_struct_names,all_ancestors))

    all_levels.append(-1)
    all_levels = np.asarray(all_levels)

    # Retriving needed properties from the catalog
    # and adding fake "trunk" properties   
    props = []
    for crit in criteria:
        prop = catalog[crit].data.tolist()
        tprop = sum(catalog[crit].data[trunk_brs_idx])
        prop.append(tprop)
        props.append(prop)
    
    s2ns.append(1)
    props.append(s2ns)


    # Generating affinity matrices if not provided
    if user_ams == None:

        AMs = aff_matrix(len(all_leav_idx), len(catalog[criteria[0]].data), \
            all_leav_idx, all_brc_idx, brc_levels, dict_children, props)

        if blind == False:

            # Showing all affinity matrices
            for i, crit in enumerate(criteria):

                plt.matshow(AMs[i,:,:])
                plt.title('"'+crit+'" affinity matrix', fontsize = 'medium')
                plt.xlabel('leaf index')
                plt.ylabel('leaf index')    
                plt.colorbar()
        
    else:

        AMs = user_ams


    S2Nmat = AMs[-1,:,:]
    AMs = AMs[:-1,:,:]

    # Check if the affinity matrix has more than 2 elements
    # otherwise return everything as clusters ("save_all").
    if AMs.shape[1] <= 2:

        print("--- Not necessary to cluster. 'save_all' keyword triggered")

        all_leaves = []
        for leaf in dendrogram.leaves:
            all_leaves.append(leaf.idx)

        clusts = all_leaves

        return clusts, AMs
        
                
    # Check whether the affinity matrix scaling parameter
    # are provided by the user, if so use them, otherwise
    # calculate them    

    """
    scpars = np.zeros(len(criteria))
    
    if user_scalpars is not None:
        print("- Using user-provided scaling parameters")
        user_scalpars = np.asarray(user_scalpars)
        scpars[0:len(user_scalpars)] = user_scalpars
    """
       
    scpars = np.array(user_scalpars)         

    print("- Start spectral clustering")

    # Selecting the criteria and merging the matrices    
    escalpars = []
    AM = np.ones(AMs[0,:,:].shape)
    for i, crit in enumerate(criteria):

        print("-- Rescaling %s matrix" % crit)
        AMc, sigma = mat_smooth(AMs[i,:,:], S2Nmat, s2nlim = s2nlim, 
            scalpar = scpars[i], lscal = locscal)        
        AM = AM*AMc
        escalpars.append(sigma)
            
    
    # Making the reduced affinity matrices
    mul_leav_mat = []
    for mli in mul_leav_idx:
        mul_leav_mat.append(all_leav_idx.index(mli))

    mul_leav_mat = np.asarray(mul_leav_mat)
    rAM = AM[mul_leav_mat,:]
    rAM = rAM[:,mul_leav_mat]

    if blind == False:
            
        # Showing the final affinity matrix
        plt.matshow(AM)
        plt.colorbar()
        plt.title('Final Affinity Matrix')
        plt.xlabel('leaf index')
        plt.ylabel('leaf index')

      
    # Guessing the number of clusters
    # if not provided

    if user_k == 0:   
        kg = guessk(rAM)
    else:
        kg = user_k-len(two_clust_idx)

    print("-- Guessed number of clusters = %i" % (kg+len(two_clust_idx)))
    
    if kg > 1:

        print("-- Number of k-means iteration: %i" % user_iter)

        # Find the best cluster number
        sils = []

        min_ks = max(2,kg-15)
        max_ks = min(kg+15,rAM.shape[0]-1)
                
        clust_configs = []

        for ks in range(min_ks,max_ks):

            try:
                
                evecs = spectral_embedding(rAM, n_components=ks,
                                        eigen_solver='arpack',
                                        random_state=222,
                                        eigen_tol=0.0, drop_first=False)
                _, all_clusters, _ = k_means(evecs, ks, random_state=222, n_init=user_iter)
                
                sil = silhouette_score(evecs, np.asarray(all_clusters), metric='euclidean')

                clust_configs.append(all_clusters)

            except np.linalg.LinAlgError:

                sil = 0
                
            sils.append(sil)
                    
        # Use the best cluster number to generate clusters                    
        best_ks = sils.index(max(sils))+min_ks
        print("-- Best cluster number found through SILHOUETTE (%f)= %i" % (max(sils), best_ks+len(two_clust_idx)))        
        silhoutte = max(sils)
        
        all_clusters = clust_configs[np.argmax(sils)]
                        
    else:

        print("-- Not necessary to cluster")
        all_clusters = np.zeros(len(all_leaves_idx), dtype = np.int32)

    clust_branches = clust_cleaning(mul_leav_idx, all_clusters, dict_parents, dict_children, dict_ancestors, savebranches = save_branches)
    clusts = clust_branches + two_clust_idx

    print("-- Final cluster number (after cleaning) %i" % len(clusts))
    

    # Calculate the silhouette after cluster cleaning
    #fclusts_idx = np.ones(len(mul_leav_idx))
    fclusts_idx = -1*all_clusters

    i = 1
    for clust in clusts:
        i += 1
        fleavs = dendrogram[clust].sorted_leaves()

        fleavs_idx = []
        for fleav in fleavs:
            fleavs_idx.append(fleav.idx)

        fleavs_idx = np.asarray(fleavs_idx)

        # Find the position of the cluster leaves
        pos = np.where(np.in1d(mul_leav_idx,fleavs_idx))[0]
        fclusts_idx[pos] = i

    oldclusts = np.unique(fclusts_idx[fclusts_idx < 0])

    for oldclust in oldclusts:
        fclusts_idx[fclusts_idx == oldclust] = np.max(fclusts_idx)+1

    evecs = spectral_embedding(rAM, n_components=ks,
                            eigen_solver='arpack',
                            random_state=222,
                            eigen_tol=0.0, drop_first=False)
    sil = silhouette_score(evecs, fclusts_idx, metric='euclidean')

    print("-- Final clustering configuration silhoutte %f" % sil)


    all_struct_types = np.asarray(all_struct_types)
    all_struct_parents = np.asarray(all_struct_parents)

    # Add the isolated leaves to the cluster list, if required
    if save_isol_leaves:

        isol_leaves = all_structures_idx[(all_struct_parents == -1) & (all_struct_types == 'L')]
        clusts = clusts + list(isol_leaves)

        print("SAVE_ISOL_LEAVES triggered. Isolated leaves added.") 
        print("-- Total cluster number %i" % len(clusts))


    # Add the unclustered leaves within clusters to the cluster list, if required
    if save_clust_leaves:

        isol_leaves = all_structures_idx[(all_struct_parents == -1) & (all_struct_types == 'L')]

        all_leaves = []
        for leaf in dendrogram.leaves:
            all_leaves.append(leaf.idx)

        clust_leaves = []
        for clust in clusts:
            for leaf in dendrogram[clust].sorted_leaves():
                clust_leaves.append(leaf.idx)

        unclust_leaves = list(set(all_leaves)-set(clust_leaves + list(isol_leaves)))
        clusts = clusts + unclust_leaves

        print("SAVE_CLUST_LEAVES triggered. Unclustered leaves added.")
        print("-- Total cluster number %i" % len(clusts))
    

    # Update the catalog with new information
    catalog['parent'] = all_struct_parents
    catalog['ancestor'] = all_struct_ancestors
    catalog['n_leaves'] = nleaves
    catalog['structure_type'] = all_struct_types

    return clusts, catalog, AMs, escalpars, silhoutte 


    
    
class SpectralCloudstering(object):
    """
    Apply the spectral clustering to find the best 
    cloud segmentation out from a dendrogram.

    Parameters
    -----------

    dendrogram: 'astrodendro.dendrogram.Dendrogram' instance
        The dendrogram to clusterize.

    catalog: 'astropy.table.table.Table' instance
        A catalog containing all properties of the dendrogram
        structures. Generally generated with ppv_catalog module.

    header: 'astropy.io.fits.header.Header' instance
        The header of the fits data the dendrogram was 
        generated from. Necessary to obtain the assignment cubes.

    criteria: list of strings
        Clustering criteria referred to the structure properties
        in the catalog (default ['volume', 'luminosity']).

    user_k: int
        The expected number of clusters, if not provided
        it will be guessed automatically through the eigenvalues
        of the unsmoothed affinity matrix.

    user_ams: numpy array
        User provided affinity matrix. Whether this is not
        furnish it is automatically generated through the
        volume and/or luminosity criteria.

    user_scalpars: list of floats
        User-provided scaling parameters to smooth the
        affinity matrices.

    user_iter: int
        User-provided number of k-means iterations.
    
    save_isol_leaves: bool
        Consider the isolated leaves (without parent) 
        as individual 'clusters'. Useful for low
        resolution data where the beam size
        corresponds to the size of a Giant
        Molecular Cloud.

    save_clust_leaves: bool
        Consider unclustered leaves as individual
        'clusters'. This keyword will not include
        the isolated leaves without parents.

    save_all_leaves: bool
        Trigger both save_isol_leaves and
        save_clust_leaves.

    save_branches: bool
        Retain all isolated branches usually discarded
        by the cluster analysis.

    save_all: bool
        Trigger all save_isol_leaves, 
        save_clust_leaves, and save_branches.        
    
    rms: int or float
        Noise level of the observation. Necessary to
        calculate the scaling parameter above a certain
        signal-to-noise ratio.

    s2nlim: int or float
        Signal-to-noise limit above which the
        scaling parameter is calculated. Needed
        only if rms is not np.nan.

    blind: bool
        Show the affinity matrices. 
        Matplotlib required.

    locscaling: bool
        Smooth the affinity matrices using a local
        scaling technique. This does not work well ...


    Return
    -------

    clusters: list
        The dendrogram branch indexes corresponding to the
        identified clusters

    catalog: 'astropy.table.table.Table' instance
        The input catalog updated with dendrogram structure
        parent, ancestor, number of leaves, and type 
        ('T', trunks or branches without parent; 'B', branches
        with parent; 'L', leaves). 

    affmats: numpy array
        The affinity matrices calculated by the algorithm
    
    escalpars: list
        Estimated scaling parameters for the different
        affinity matrixes
    
    silhouette: float
        Silhouette of the best cluster configuration

    clusters_asgn: astropy.io.fits.hdu.image.PrimaryHDU
        Assignment cube generated by the 'cluster'-type
        structures.

    trunks_asgn: astropy.io.fits.hdu.image.PrimaryHDU
        Assignment cube generated by the 'trunk'-type
        structures.         

    leaves_asgn: astropy.io.fits.hdu.image.PrimaryHDU
        Assignment cube generated by the 'leaf'-type
        structures.                 
        
    """

    def __init__(self, dendrogram, catalog, header, criteria = ['volume', 'luminosity'],
                 user_k = None, user_ams = None, user_scalpars = None, user_iter = None,
                 save_isol_leaves = False, save_clust_leaves = False, save_branches = False, 
                 save_all_leaves = False, save_all = False, blind = False, rms = np.nan, s2nlim = 3, 
                 locscaling = False):

        self.dendrogram = dendrogram
        self.header = header

        # Checking for user defined criteria existence in the catalog
        for i, crit in enumerate(criteria):
            if (crit not in catalog.colnames) & (crit != 'volume') & (crit != 'luminosity'):
                print("WARNING: %s not in the catalog, removed from the criteria list" % crit)
                criteria.pop(criteria.index(crit))

        if len(criteria) == 0:
            print("WARNING: criteria list empty, running on default criteria list, [volume, luminosity]")
            criteria = ['volume', 'luminosity']

        # Check for default criteria existence in the catalog
        if ('luminosity' not in catalog.colnames) and ('luminosity' in criteria):
            print("WARNING: adding luminosity = flux to the catalog.")
            catalog['luminosity'] = catalog['flux']
        if ('volume' not in catalog.colnames) and ('volume' in criteria):
            print("WARNING: adding volume = pi * radius^2 * v_rms to the catalog.")
            catalog['volume'] = np.pi*catalog['radius']**2*catalog['v_rms']
            
        if len(criteria) > 1:
            print("WARNING: clustering will be performed on the Aggregated matrix")

        if save_all_leaves is True:
            print("SAVE_ALL_LEAVES triggered: isolated leaves and unclustered leaves will be retained")
            save_clust_leaves, save_isol_leaves = True, True

        if save_all is True:
            print("SAVE_ALL triggered: isolated leaves, unclustered leaves, and unclustered branches will be retained")
            save_branches, save_clust_leaves, save_isol_leaves = True, True, True


        self.criteria = criteria
        self.user_k = user_k or 0
        self.user_ams = user_ams
        self.user_scalpars = user_scalpars or [None]*len(criteria)
        self.user_iter = user_iter or 1
        self.locscaling = locscaling        
        self.save_all = save_all
        self.save_isol_leaves = save_isol_leaves
        self.save_clust_leaves = save_clust_leaves
        self.save_branches = save_branches
        self.save_all_leaves = save_all_leaves
        self.save_all = save_all
        self.blind = blind
        self.rms = rms
        self.s2nlim = s2nlim

        # Default colors in case plot_connected_colors is called before showdendro
        self.colors = cycle('rgbcmyk')

        self.clusters, self.catalog, self.affmats, self.escalpars, self.silhouette = cloudstering(self.dendrogram,
                                                                                        catalog,
                                                                                        self.criteria,
                                                                                        self.user_k, 
                                                                                        self.user_ams, 
                                                                                        self.user_scalpars, 
                                                                                        self.user_iter,
                                                                                        self.save_isol_leaves, 
                                                                                        self.save_clust_leaves,
                                                                                        self.save_branches,
                                                                                        self.blind, 
                                                                                        self.rms, 
                                                                                        self.s2nlim,
                                                                                        self.locscaling)
        
        print("Generate assignment cubes...")
        # Automatically generate assignment cubes
        # ... of clusters
        self.clusters_asgn = make_asgncube(self.dendrogram, self.clusters, self.header, collapse = False)
        # ... of trunks
        trunks = np.where(self.catalog['structure_type'] == 'T')[0]
        self.trunks_asgn = make_asgncube(self.dendrogram, trunks, self.header, collapse = False)
        # ... of leaves
        leaves = np.where(self.catalog['structure_type'] == 'L')[0]        
        self.leaves_asgn = make_asgncube(self.dendrogram, leaves, self.header, collapse = False)



    def showdendro(self, cores_idx=[], savefile=None):
        
        """

        Show the clustered dendrogram.
        Every color correspond to a
        different cluster.

        """

        dendro = self.dendrogram
        if len(cores_idx) == 0:
            cores_idx = self.clusters

        # For the random colors
        r = lambda: random.randint(0,255)
                 
        p = dendro.plotter()

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
                
        ax.set_yscale('log')
                
        cols = []
        
        # Plot the whole tree

        p.plot_tree(ax, color='black')

        for i in range(len(cores_idx)):

            col = '#%02X%02X%02X' % (r(),r(),r())
            cols.append(col)
            p.plot_tree(ax, structure=[dendro[cores_idx[i]]], color=cols[i], lw=3)

        ax.set_title("Final clustering configuration")

        ax.set_xlabel("Structure")
        ax.set_ylabel("Flux")

        self.colors = cols

        if savefile:
            fig.savefig(savefile)
        



    def plot_connected_clusters(self, **kwargs):
        from plotting import dendroplot_clusters

        return dendroplot_clusters(self.clusters, self.dendrogram, self.catalog,
                                   colors=self.colors,
                                   **kwargs)

if __name__ == '__main__':
    SpectralCloudstering().run()


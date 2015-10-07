from matplotlib import pyplot as plt

def dendroplot_clusters(clusters,
                        dend,
                        cat,
                        axname1='flux',
                        axname2='v_rms',
                        axscale1=1.,
                        axscale2=1.,
                        colors=itertools.cycle('rgbcmyk'),
                        marker='s',
                        marker2=None,
                        linestyle='-', **kwargs):
    """
    Plot all of the clusters with (partially transparent) lines connecting
    leaves to their parents all the way up to the trunk

    Examples
    --------
    >>> SC = SpectralCloudstering(dend, cat)
    >>> colors = showdendro(SC.dendrogram, SC.clusters)
    >>> dendroplot_clusters(SC.clusters, SC.dendrogram, SC.catalog,
    ...                     colors=colors, axname1='radius')
    """

    axis = plt.gca()

    for cluster, color in zip(clusters, colors):
        leaves = dend[cluster].sorted_leaves()
        last_parent = cluster

        for leaf in leaves:
            xax,yax = ([cat[leaf.idx][axname1]*axscale1],
                       [cat[leaf.idx][axname2]*axscale2])
            #if axname1 in ('v_rms','reff'):
            #    xax *= gcorfactor[leaf.idx]
            #if axname2 in ('v_rms','reff'):
            #    yax *= gcorfactor[leaf.idx]
            axis.plot(xax, yax, marker, color=color, markeredgecolor='none', alpha=0.5)
            obj = leaf.parent
            while obj.parent:
                xax.append(cat[obj.idx][axname1]*axscale1)
                yax.append(cat[obj.idx][axname2]*axscale2)
                if obj.idx == last_parent:
                    break
                obj = obj.parent
            if np.any(np.isnan(yax)):
                ok = ~np.isnan(yax)
                axis.plot(np.array(xax)[ok], np.array(yax)[ok], alpha=0.5,
                          label=leaf.idx, color='b', zorder=5,
                          linestyle=linestyle, marker=marker2, **kwargs)
            else:
                axis.plot(xax, yax, alpha=0.1, label=leaf.idx, color=color,
                          zorder=5, linestyle=linestyle, marker=marker2,
                          **kwargs)

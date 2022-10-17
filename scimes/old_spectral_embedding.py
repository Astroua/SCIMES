"""
Spectral Embedding function from scikit-learn 1.1.1 but with
division by dd replaced by multiplication as in version 0.19.2.

Only eigen_solver="arpack" or "lobpcg" supported.

See these pages for more details of why '* dd' was replaced by '/ dd' in
version 0.20.0.

https://scikit-learn.org/0.20/whats_new.html#version-0-20-0
https://github.com/scikit-learn/scikit-learn/pull/9062
https://github.com/scikit-learn/scikit-learn/issues/8129

Fix proposed by Prof. Tony Wong, University of Illinois, wongt@illinois.edu 
"""

import warnings
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.utils import check_array, check_symmetric, check_random_state
from sklearn.manifold._spectral_embedding import _graph_is_connected, _set_diag
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils.fixes import lobpcg


def spectral_embedding(adjacency, *, 
                       n_components=8, eigen_solver="arpack", random_state=None, 
                       eigen_tol=0.0, norm_laplacian=True, drop_first=True):

    adjacency = check_symmetric(adjacency)
    random_state = check_random_state(random_state)
    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected, spectral embedding may not work as expected."
        )

    laplacian, dd = csgraph_laplacian(
        adjacency, normed=norm_laplacian, return_diag=True
    )

    if eigen_solver == "arpack":
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        try:
            # We are computing the opposite of the laplacian inplace so as
            # to spare a memory allocation of a possibly very large array
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            _, diffusion_map = eigsh(
                laplacian, k=n_components, sigma=1.0, which="LM", tol=eigen_tol, v0=v0
            )
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                # REVERT FROM DIVISION TO MULTIPLICATION
                embedding = embedding * dd
        except RuntimeError:
                # When submatrices are exactly singular, an LU decomposition
                # in arpack fails. We fallback to lobpcg
                eigen_solver = "lobpcg"
                # Revert the laplacian to its opposite to have lobpcg work
                laplacian *= -1

    if eigen_solver == "lobpcg":
        laplacian = check_array(
            laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        )
        if n_nodes < 5 * n_components + 1:
            # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
            # for details see the source code in scipy:
            # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
            # /lobpcg/lobpcg.py#L237
            # or matlab:
            # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to eigh, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.toarray()
            _, diffusion_map = eigh(laplacian, check_finite=False)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                # REVERT FROM DIVISION TO MULTIPLICATION
                embedding = embedding * dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            X = random_state.standard_normal(
                size=(laplacian.shape[0], n_components + 1)
            )
            X[:, 0] = dd.ravel()
            X = X.astype(laplacian.dtype)
            _, diffusion_map = lobpcg(
                laplacian, X, tol=1e-5, largest=False, maxiter=2000
            )
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                # REVERT FROM DIVISION TO MULTIPLICATION
                embedding = embedding * dd
            if embedding.shape[0] == 1:
                raise ValueError

    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T

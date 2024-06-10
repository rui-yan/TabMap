"""
Optimization methods to solve the Gromov-Wasserstein problem.
"""

import numpy as np
import ot
from ot import bregman
from scipy.stats import describe


def create_space_distributions(num_locations, num_features):
    """
    Generates uniform distributions for both target and source spaces.

    Parameters:
    num_locations (int): The number of grid locations (target space).
    num_features (int): The number of features in the dataset (source space).

    Returns:
    u: the uniform distribution across the grid locations of the target space. 
    v: the uniform distribution across the features in the source space.
    """
    u = ot.unif(num_locations)
    v = ot.unif(num_features)
    return u, v


def tensor_product(constC, hC1, hC2, T):
    """
    Computes the tensor for accelerated Gromov-Wasserstein calculations as described in 
    Proposition 1, Equation (6) of the reference [1]. This function facilitates efficient 
    tensor-matrix multiplication for the Gromov-Wasserstein discrepancy.

    Parameters:
    ----------
    constC : ndarray, shape (ns, nt)
        Constant C matrix in Eq. (6), representing cost matrix between source and target measures.
    hC1 : ndarray, shape (ns, ns)
        Transformed source cost matrix h1(C1) as per Eq. (6).
    hC2 : ndarray, shape (nt, nt)
        Transformed target cost matrix h2(C2) as per Eq. (6).
    T : ndarray, shape (ns, nt)
        Coupling matrix between source and target distributions.

    Returns:
    -------
    tens : ndarray, shape (ns, nt)
        The resulting tensor from the operation \mathcal{L}(C1,C2) \otimes T, which represents
        the tensor-matrix multiplication outcome.

    References:
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon, "Gromov-Wasserstein averaging of kernel 
           and distance matrices." International Conference on Machine Learning (ICML). 2016.
    """
    A = -np.dot(hC1, T).dot(hC2.T)
    tens = constC + A
    # tens = A - A.min()
    return tens


class gromov_wass_solver:
    """
    A solver for finding an optimal transport coupling matrix that minimizes the 
    Gromov-Wasserstein discrepancy between two metric spaces.
    """
    def __init__(self, loss_fun='kl_loss', epsilon=0.0, tol=1e-9, seed=42):
        """
        Initializes the Gromov-Wasserstein solver with specified parameters.
        
        Parameters:
        ----------
        loss_fun : str, optional
            The loss function to use. Options include 'kl_loss', 'sqeuclidean', and 'square_loss'. 
            Default is 'kl_loss'.
        epsilon : float, optional
            Entropic regularization parameter. Default is 0.0.
        tol : float, optional
            Tolerance for the stopping criterion. Default is 1e-9.
        seed : int, optional
            Seed for the random number generator to ensure reproducibility. Default is 42.
        """
        self.loss_fun = loss_fun
        self.epsilon = epsilon
        self.tol = tol
        np.random.seed(seed)

    def init_matrix(self, C1, C2, u, v, loss_fun='kl_loss'):
        """
        Prepares the necessary matrices and tensors for efficient Gromov-Wasserstein computations
        based on a specified loss function.

        Parameters:
        ----------
        C1 : ndarray, shape (ns, ns)
            Metric cost matrix for the source space.
        C2 : ndarray, shape (nt, nt)
            Metric cost matrix for the target space.
        u : ndarray, shape (ns,)
            Mass distribution in the source space.
        v : ndarray, shape (nt,)
            Mass distribution in the target space.
        loss_fun : str
            The loss function, either 'kl_loss' or 'square_loss', defining the discrepancy measure.

        Returns:
        -------
        constC : ndarray, shape (ns, nt)
            Constant C matrix combining transformations of both source and target spaces.
        hC1 : ndarray, shape (ns, ns)
            Transformed source cost matrix.
        hC2 : ndarray, shape (nt, nt)
            Transformed target cost matrix.

        The 'kl_loss' function is calculated as:
            L(a, b) = a*log(a/b) - a + b
            Decomposed as:
                f1(a) = a*log(a) - a
                f2(b) = b
                h1(a) = a
                h2(b) = log(b)
        The 'square_loss' function is calculated as:
            L(a, b) = (1/2)*|a - b|^2
            Decomposed as:
                f1(a) = (a^2) / 2
                f2(b) = (b^2) / 2
                h1(a) = a
                h2(b) = b
        """
        if loss_fun == 'kl_loss':
            f1 = lambda a: a * np.log(a + 1e-15) - a
            f2 = lambda b: b
            h1 = lambda a: a
            h2 = lambda b: np.log(b + 1e-15)

        elif loss_fun == 'square_loss':
            f1 = lambda a: (a**2) / 2
            f2 = lambda b: (b**2) / 2
            h1 = lambda a: a
            h2 = lambda b: b

        constC1 = np.dot(np.dot(f1(C1), u.reshape(-1, 1)), np.ones(len(v)).reshape(1, -1))
        constC2 = np.dot(np.ones(len(u)).reshape(-1, 1), np.dot(v.reshape(1, -1), f2(C2).T))
        constC = constC1 + constC2
        hC1 = h1(C1)
        hC2 = h2(C2)
        return constC, hC1, hC2
    
    def solve(self, C1, C2, u, v, maxiter=100, print_every=1, verbose=False):
        """
        Solves the Gromov-Wasserstein optimization problem to find the optimal transport coupling matrix (T)
        between two metric spaces (C1,u) and (C2,v) characterized by their respective cost matrices and mass distributions.
        
        Parameters:
        ----------
        C1 : ndarray, shape (ns, ns)
            Metric cost matrix for the source space, indicating the cost to transport mass within the source.
        C2 : ndarray, shape (nt, nt)
            Metric cost matrix for the target space, indicating the cost to transport mass within the target.
        u : ndarray, shape (ns,)
            Mass distribution vector for the source space.
        v : ndarray, shape (nt,)
            Mass distribution vector for the target space.
        maxiter : int, optional
            Maximum number of iterations to perform. Default is 100.
        print_every : int, optional
            Frequency of iteration updates to print for monitoring. Default is 1.
        verbose : bool, optional
            Flag to enable printing of progress during computation. Default is True.

        Returns:
        -------
        T : ndarray, shape (ns, nt)
            The computed optimal transport matrix that minimizes the Gromov-Wasserstein discrepancy between the two spaces.

        Example usage:
        -------------
        Assuming `solver` is an instance of `gromov_wass_solver`, and `C1`, `C2`, `u`, and `v` are predefined arrays:
            T = solver.solve(C1, C2, u, v)
        """
        # Normalizing the cost matrices
        C1 /= C1.mean()
        C2 /= C2.mean()
        stats_C1 = describe(C1.flatten())
        stats_C2 = describe(C2.flatten())

        if verbose:
            for (k, C, s) in [('C1', C1, stats_C1), ('C2', C2, stats_C2)]:
                print('Stats Distance Matrix {}. mean: {:8.2f}, median: {:8.2f},\
                 in: {:8.2f}, max:{:8.2f}'.format(k, s.mean, np.median(C), s.minmax[0], s.minmax[1]))
        
        C1 = np.asarray(C1, dtype=np.float64)
        C2 = np.asarray(C2, dtype=np.float64)
        
        # Initializes T
        T = np.outer(u, v)
        
        if self.loss_fun in ['square_loss', 'kl_loss']:
            constC, hC1, hC2 = self.init_matrix(C1, C2, u, v, self.loss_fun)
            tens = tensor_product(constC, hC1, hC2, T)
        elif self.loss_fun in ['sqeuclidean']:
            tens = ot.dist(C1, C2, metric=self.loss_fun)
        
        if self.epsilon == 0:
            T = ot.lp.emd(u, v, tens, numItermax=1e7)
        else: # Optimal transport computation with entropic regularization if eps > 0
            it = 0
            err = 1
            while (err > self.tol and it <= maxiter):
                Tprev = T
                tens = tensor_product(constC, hC1, hC2, T)
                T = bregman.sinkhorn(u, v, tens, self.epsilon, numItermax=100000)
                err = np.linalg.norm(T - Tprev)
                if it % print_every == 0 and verbose: 
                    print('{:5d}|{:8e}|'.format(it, err))           
                it += 1

        return T
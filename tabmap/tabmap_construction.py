"""
TabMap construction
"""
import numpy as np
# import gower
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import euclidean

from .utils import OT_solver


class TabMapGenerator:
    """
    TabMap construction: TabMaps visually represent the inter-relationships between data features 
    by mapping them onto a 2D topographic map using optimal transport techniques, such that 
    the strength of two inter-related features correlates with their distance on the map.
    """
    
    def __init__(self, metric='correlation', loss_fun='kl_loss', epsilon=0.0, version='v2.0', num_iter=10):
        """
        Initializes the TabMapGenerator.
        
        Parameters:
        metric (str): Metric used to compute the feature inter-relationships. 
            Default is 'correlation'. Options include 'correlation', 'euclidean', and 'gower'.
        loss_fun (str): Loss function used for computing the optimal transport. 
            Default is 'kl_loss'.Options include 'kl_loss', 'sqeuclidean', and 'square_loss'.
        epsilon (float): Entropic regularization parameter.
            Default is 0.0.
        version (str): Version of the distance matrix calculation algorithm. 
            Default is 'v2.0'. Versions 'v1.0' and 'v2.0' use different methods for computing grid distances.
        num_iter (int): Number of iterations for the optimal transport problem. Default is 10.
        """
        self.metric = metric
        self.loss_fun = loss_fun
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.version = version

    def _create_feature_distance_matrix(self, X):
        """
        Computes the distance matrix among the features using the specified metric.

        Parameters:
        X (ndarray): Data matrix with shape (sample_size, num_features).
        
        Returns:
        ndarray: Pairwise distance matrix among features.
        """
        if self.metric == 'gower':
            return 1 - gower.gower_matrix(X.T)
        else:
            return pairwise_distances(X.T, metric=self.metric)
    
    def _create_pixel_distance_matrix(self, row_num, col_num):
        """
        Computes the Euclidean distance matrix among the pixels in the 2D grid.
        
        Parameters:
        row_num (int): Number of rows in the 2D grid.
        col_num (int): Number of columns in the 2D grid.

        Returns:
        ndarray: Distance matrix with shape (row_num * col_num, row_num * col_num).
        """
        if row_num % 2 == 0:
            nx = row_num / 2
            x = np.linspace(-nx, nx-1, row_num)
        else:
            nx = (row_num - 1) / 2
            x = np.linspace(-nx, nx, row_num)

        if col_num % 2 == 0:
            mx = col_num / 2
            y = np.linspace(-mx, mx - 1, col_num)
        else:
            mx = (col_num - 1) / 2
            y = np.linspace(-mx, mx, col_num)
        xx, yy = np.meshgrid(x, y)
            
        if self.version == 'v1.0':
            distances = np.sqrt(xx**2 + yy**2)
            flat_distances = distances.flatten().reshape(-1, 1)
            return pairwise_distances(flat_distances)

        elif self.version == 'v2.0':
            grid = np.indices((row_num, col_num)).reshape(2, -1).T
            return pairwise_distances(grid)
    
    def fit(self, X, row_num=None, col_num=None, truncate=False):
        """
        Computes the projection matrix from data to a grid layout specified by row_num and col_num.

        Parameters:
        X (ndarray): Data matrix with shape (sample_size, num_features).
        row_num (int, optional): Number of rows in the resulting TabMap. Computed if not provided.
        col_num (int, optional): Number of columns in the resulting TabMap. Computed if not provided.
        truncate (bool): Determines whether to truncate or zero-pad the data to fit the grid.
        
        Returns:
        - self
        """
        num_features = X.shape[1]
        if row_num is None:
            if truncate: # truncate
                row_num = int(np.floor(np.sqrt(num_features))) 
            else: # zero padding
                row_num = int(np.ceil(np.sqrt(num_features)))
        
        if col_num is None:
            col_num = row_num
        
        # Adjust grid size for the features
        grid_points = row_num * col_num
        effective_points = min(grid_points, num_features)
        
        # Compute distance matrix among the features
        feat_dist_mat = self._create_feature_distance_matrix(X)
        
        # Compute distance matrix for the 2D grid
        grid_dist_mat = self._create_pixel_distance_matrix(row_num, col_num)
        
        u, v = OT_solver.create_space_distributions(effective_points, effective_points)

        solver = OT_solver.gromov_wass_solver(loss_fun=self.loss_fun, epsilon=self.epsilon)

        # Define the solver and solve for the optimal transport matrix T_prime
        print('Solving optimization problem...')
        T_prime = solver.solve(feat_dist_mat, 
                         grid_dist_mat[:effective_points, :effective_points], 
                         u, v, maxiter=self.num_iter,
                         print_every=10,
                         verbose=False)
        
        if T_prime.sum() == 0:
            raise ValueError('Optimal transport calculation failed. Adjust epsilon and retry.')
        
        # Scale T_prime to obtain the final permutation matrix T
        T = T_prime * effective_points
        
        # Generate projection matrix ensuring one-to-one assignments
        # Ensure that each column has only one non-zero element
        # And each row has only one non-zero element
        if not np.allclose(np.max(T, axis=0), 1):
            print('Performing linear sum assignment')
            row_indices, col_indices = linear_sum_assignment(-T_prime)
            T = np.zeros_like(T_prime)
            T[row_indices, col_indices] = 1
        assert np.allclose(np.sum(T, axis=0), 1)
        assert np.allclose(np.max(T, axis=0), 1)
        
        self.row_num = row_num
        self.col_num = col_num
        self.project_matrix = T
        
        return self
    
    def transform(self, X):
        """
        Transforms the data into TabMaps using the computed projection matrix.

        Parameters:
        X (ndarray): Data matrix with shape (sample_size, num_features).
        projection_matrix (ndarray): Computed projection matrix for data transformation.
        row_num (int): Number of rows in the TabMap.
        col_num (int): Number of columns in the TabMap.

        Returns:
        ndarray: Transformed TabMaps.
        """
        grid_points = self.row_num * self.col_num
        X_permuted = np.matmul(X, self.project_matrix)
        # Pad the permuted data such that the number of data points can be reshaped into a square
        X_padded = np.pad(X_permuted, ((0, 0), (0, grid_points - X_permuted.shape[1])), 
                             mode='constant', constant_values=0)
        X_tabmap = np.reshape(X_padded, (X.shape[0], self.col_num, self.row_num), order='F')
        X_tabmap = X_tabmap.transpose(0, 2, 1)
        
        return X_tabmap
    
    def fit_transform(self, X, row_num=None, col_num=None, truncate=False):
        """
        Fits the model on the data and transforms it in one step.

        Parameters:
        - X (ndarray): Data matrix with shape (sample_size, num_features).
        - row_num (int, optional): Number of rows in the TabMap. Computed if not provided.
        - col_num (int, optional): Number of columns in the TabMap. Computed if not provided.
        - truncate (bool): Determines whether to truncate or zero-pad data to fit the grid.

        Returns:
        - ndarray: Transformed TabMaps of shape (sample_size, row_num, col_num).
        """
        self.fit(X, row_num, col_num, truncate)
        return self.transform(X)
#from sklearn.base import BaseEstimator
from scipy.stats import cauchy, laplace
#from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

class RFF_perso():

    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.
    ----------
    gamma : float
         Parameter of RBF kernel: exp(-gamma * x^2)
    n_components : int
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.
    """
        
    def __init__(self, gamma=1., n_components=100, random_state=None, kernel = 'rbf'):
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

        
    def fit(self, X, y=None):
        """Fit the model with X.
        Samples random projection according to n_features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the transformer.
        """
        n_features = X.shape[1] 
        #self.random_weights_ = (np.sqrt(2 * self.gamma) * np.random.normal(size=(n_features, self.n_components)))
        if self.kernel == 'rbf':
            self.random_weights_ = (1.0 / self.gamma * np.random.normal(size=(n_features, self.n_components)))
        elif self.kernel == 'laplacian':
            self.random_weights_ = cauchy.rvs(scale = self.gamma, size=(n_features, self.n_components))
        elif self.kernel == 'cauchy':
            self.random_weights_ = laplace.rvs(scale = self.gamma, size = (n_features, self.n_components))
        self.random_offset_ = np.random.uniform(0, 2 * np.pi,
                                                   size=self.n_components)
        
        return self
    
    
    def transform(self,X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        projection = X.dot(self.random_weights_)        ### waiting for changing
        #projection += self.random_offset_               ### be able to change
        #np.cos(projection, projection)
        pro_cos = np.cos(projection)
        pro_sin = np.sin(projection)
        projection = np.concatenate((pro_cos, pro_sin), axis=1)
        projection *= 1.0 / np.sqrt(self.n_components)
        
        return projection
    
    def compute_kernel(self, X):
        """Computes the approximated kernel matrix.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        K : approximated kernel matrix
        """
        projection = self.transform(X)
        K = projection.dot(projection.T)
        
        return K

        
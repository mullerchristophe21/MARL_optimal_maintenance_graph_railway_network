import numpy as np
import pymc as pm
import warnings
import pytensor.tensor as pt

import numbers
import warnings

from collections import Counter
from collections.abc import Sequence
from functools import reduce
from operator import add, mul
from typing import Any, Callable, Optional, Union


from pymc.pytensorf import constant_fold



def pt_jitchol(mat, jitter=0):
    """Run Cholesky decomposition with an increasing jitter,
    until the jitter becomes too large.
    Arguments
    ---------
    mat : (m, m) pt.tensor
        Positive-definite matrix
    jitter : float
        Initial jitter
    """
    try:
        chol = pm.math.linalg.cholesky(mat)
        return chol
    except:
        new_jitter = jitter*10.0 if jitter > 0.0 else 1e-15
        if new_jitter > 1.0:
            raise RuntimeError('Matrix not positive definite even with jitter')
        warnings.warn(
            'Matrix not positive-definite, adding jitter {:e}'
            .format(new_jitter),
            RuntimeWarning)
        return pt_jitchol(mat + new_jitter * pt.eye(mat.shape[-1]), new_jitter)


class GraphMaternKernel(pm.gp.cov.Covariance):
    """The Matern kernel on Graph. Kernel is direct product of Matern Kernel on Graph and some kernel on \R^d

    Attributes
    ----------

    eigenpairs : tuple
        Truncated tuple returned by pm.math.linalg.eigh applied to the Laplacian of the graph.
    nu : float
        Trainable smoothness hyperparameter.
    kappa : float
        Trainable lengthscale hyperparameter.
    sigma_f : float
        Trainable scaling kernel hyperparameter.
    vertex_dim: int
        dimension of \R^d
    point_kernel: pm.gp.cov.Covariance
        kernel on \R^d
    active_dims: slice or list of indices
    """

    def __init__(self, eigenpairs, nu=3, kappa=4, sigma_f=1,
                 vertex_dim=0, point_kernel=None, active_dims=None,
                 jitter=1e-6, ls=1.0):

        eigenvalues, eigenvectors = eigenpairs
        self.eigenvectors = pt.as_tensor(eigenvectors)
        self.eigenvalues = pt.as_tensor(eigenvalues)
        self.num_vertices = len(eigenpairs[0])
        self.vertex_dim = vertex_dim
        self.nu = nu
        self.jitter = jitter
        self.kappa = kappa
        self.sigma_f = sigma_f
        self.ls = ls

        if point_kernel is not None:
            self.point_kernel = point_kernel
            # Increment active_dims by 1 if point_kernel is provided
            if self.point_kernel.active_dims is not None:
                if isinstance(self.point_kernel.active_dims, list):
                    self.active_dims = [dim + 1 for dim in self.point_kernel.active_dims]
                else:
                    self.active_dims = self.point_kernel.active_dims + 1
            else:
                self.active_dims = active_dims
        else:
            self.point_kernel = None
            self.active_dims = active_dims

        super().__init__(input_dim=vertex_dim, active_dims=self.active_dims)

    def eval_S(self, kappa, sigma_f):

        S = pt.pow(self.eigenvalues + 2 * self.nu / kappa / kappa, -self.nu)

        num_vertices = float(self.num_vertices) 

        S = S * (num_vertices / pt.sum(S))  
        S = S * sigma_f

        return S

    def _eval_K_vertex(self, X_id, X2_id):
        if X2_id is None:
            X2_id = X_id

        S = self.eval_S(self.kappa, self.sigma_f)

        K_vertex = (self.eigenvectors[X_id] * S[:]) @ \
            pt.transpose(self.eigenvectors[X2_id])

        return K_vertex

    def __call__(self, X, X2=None):

        if X2 is None:
            X2 = X

        if self.vertex_dim == 0:
            X_id = X[:, 0].astype('int32')
            X2_id = X2[:, 0].astype('int32')

            K = self._eval_K_vertex(X_id, X2_id)

        else:

            X_id = X[:, 0] 
            X_v = X[:, 1:] 

            X2_id = X2[:, 0]
            X2_v = X2[:, 1:] 

            X_id = pt.as_tensor(X_id.astype('int32'))  # Force NumPy conversion before PyTensor
            X2_id = pt.as_tensor(X2_id.astype('int32'))
            
            X_v = pt.as_tensor(X_v)
            X2_v = pt.as_tensor(X2_v)

            K_vertex = self._eval_K_vertex(X_id, X2_id)
            K_point = self.point_kernel(X_v, X2_v)

            K = np.multiply(K_point, K_vertex)
            # K = K_point * K_vertex

        return K

    def diag(self, X):
        if self.vertex_dim == 0:
            X_id = pm.math.cast(X[:, 0], dtype='int32')
            K_diag = pm.math.sum(pm.math.transpose((self.eigenvectors[X_id]) * self.eval_S(self.kappa, self.sigma_f)[None, :]) *
                                   pm.math.transpose(self.eigenvectors[X_id]), axis=0)
        else:
            X_id, X_v = pm.math.cast(X[:, 0], dtype='int32'), X[:, 1:]
            K_diag_vertex = pm.math.sum(pm.math.transpose((self.eigenvectors[X_id]) * self.eval_S(self.kappa, self.sigma_f)[None, :]) *
                                          pm.math.transpose(self.eigenvectors[X_id]), axis=0)
            K_diag_point = self.point_kernel.diag(X_v)
            K_diag = K_diag_point * K_diag_vertex
        return K_diag

    def sample(self, X):
        print("HERE")
        K_chol = pt_jitchol(self(X))
        sample = K_chol.dot(np.random.randn(pm.math.shape(K_chol)[0]))
        return sample
    

    def power_spectral_density(self, omega):
        """Compute the power spectral density of the combined kernel."""
        if self.point_kernel is not None and hasattr(self.point_kernel, 'power_spectral_density'):
            return self.point_kernel.power_spectral_density(omega)
        else:
            raise NotImplementedError("The point kernel does not have a power_spectral_density method.")

    # def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
    #     r"""
    #     The power spectral density for the ExpQuad kernel is:

    #     .. math::

    #        S(\boldsymbol\omega) =
    #            (\sqrt(2 \pi)^D \prod_{i}^{D}\ell_i
    #             \exp\left( -\frac{1}{2} \sum_{i}^{D}\ell_i^2 \omega_i^{2} \right)
    #     """
    #     ls = pt.ones(self.n_dims) * self.ls
    #     c = pt.power(pt.sqrt(2.0 * np.pi), self.n_dims)
    #     exp = pt.exp(-0.5 * pt.dot(pt.square(omega), pt.square(ls)))
    #     return c * pt.prod(ls) * exp

class GraphMaternKernel_numpy(pm.gp.cov.Covariance):
    """The Matern kernel on Graph. Kernel is direct product of Matern Kernel on Graph and some kernel on \R^d

    Attributes
    ----------

    eigenpairs : tuple
        Truncated tuple returned by pm.math.linalg.eigh applied to the Laplacian of the graph.
    nu : float
        Trainable smoothness hyperparameter.
    kappa : float
        Trainable lengthscale hyperparameter.
    sigma_f : float
        Trainable scaling kernel hyperparameter.
    vertex_dim: int
        dimension of \R^d
    point_kernel: pm.gp.cov.Covariance
        kernel on \R^d
    active_dims: slice or list of indices
    """

    def __init__(self, eigenpairs, nu=3, kappa=4, sigma_f=1,
                 vertex_dim=0, point_kernel=None, active_dims=None,
                 jitter=1e-6, n_dims=1, ls=1.0):

        eigenvalues, eigenvectors = eigenpairs
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.num_vertices = len(eigenpairs[0])
        self.vertex_dim = vertex_dim
        self.nu = nu
        self.jitter = jitter
        self.kappa = kappa
        self.sigma_f = sigma_f
        if vertex_dim != 0:
            self.point_kernel = point_kernel
        else:
            self.point_kernel = None
        
        # self.n_dims = n_dims
        self.ls = ls

        # super().__init__(input_dim=(vertex_dim), active_dims=active_dims)

    def eval_S(self, kappa, sigma_f):

        S = np.power(self.eigenvalues + 2 * self.nu / kappa / kappa, -self.nu)

        num_vertices = float(self.num_vertices) 

        S = S * (num_vertices / np.sum(S))  
        S = S * sigma_f

        return S

    def _eval_K_vertex(self, X_id, X2_id):
        if X2_id is None:
            X2_id = X_id

        S = self.eval_S(self.kappa, self.sigma_f)

        K_vertex = (self.eigenvectors[X_id] * S[:]) @ \
            np.transpose(self.eigenvectors[X2_id])

        return K_vertex

    def __call__(self, X, X2=None):

        if X2 is None:
            X2 = X

        if self.vertex_dim == 0:
            X_id = X[:, 0].astype('int32')
            X2_id = X2[:, 0].astype('int32')

            K = self._eval_K_vertex(X_id, X2_id)

        else:

            X_id = X[:, 0] 
            X_v = X[:, 1:] 

            X2_id = X2[:, 0]
            X2_v = X2[:, 1:] 

            X_id = X_id.astype('int32')
            X2_id = X2_id.astype('int32')

            K_vertex = self._eval_K_vertex(X_id, X2_id)
            K_point = self.point_kernel(X_v, X2_v)

            K = np.multiply(K_point, K_vertex)
            # K = K_point * K_vertex

            K = K + self.jitter * np.eye(K.shape[0]) # Add jitter

        return K
    

class GraphMaternKernel_numpy_efficient(pm.gp.cov.Covariance):
    """The Matern kernel on Graph. Kernel is direct product of Matern Kernel on Graph and some kernel on \R^d

    Attributes
    ----------

    eigenpairs : tuple
        Truncated tuple returned by pm.math.linalg.eigh applied to the Laplacian of the graph.
    nu : float
        Trainable smoothness hyperparameter.
    kappa : float
        Trainable lengthscale hyperparameter.
    sigma_f : float
        Trainable scaling kernel hyperparameter.
    vertex_dim: int
        dimension of \R^d
    point_kernel: pm.gp.cov.Covariance
        kernel on \R^d
    active_dims: slice or list of indices
    """

    def __init__(self, eigenpairs, nu=3, kappa=4, sigma_f=1,
                 vertex_dim=0, point_kernel=None, active_dims=None,
                 jitter=1e-6, n_dims=1, ls=1.0):

        eigenvalues, eigenvectors = eigenpairs
        self.eigenvectors = eigenvectors
        self.eigenvalues = eigenvalues
        self.num_vertices = len(eigenpairs[0])
        self.vertex_dim = vertex_dim
        self.nu = nu
        self.jitter = jitter
        self.kappa = kappa
        self.sigma_f = sigma_f
        if vertex_dim != 0:
            self.point_kernel = point_kernel
        else:
            self.point_kernel = None
        
        # self.n_dims = n_dims
        self.ls = ls

        self.saved_S = self.eval_S(self.kappa, self.sigma_f)

        # super().__init__(input_dim=(vertex_dim), active_dims=active_dims)

    def eval_S(self, kappa, sigma_f):

        S = np.power(self.eigenvalues + 2 * self.nu / kappa / kappa, -self.nu)

        num_vertices = float(self.num_vertices) 

        S = S * (num_vertices / np.sum(S))  
        S = S * sigma_f

        return S

    def _eval_K_vertex(self, X_id, X2_id):
        if X2_id is None:
            X2_id = X_id

        # S = self.eval_S(self.kappa, self.sigma_f)
        S = self.saved_S

        K_vertex = (self.eigenvectors[X_id] * S[:]) @ \
            np.transpose(self.eigenvectors[X2_id])

        return K_vertex

    def __call__(self, X, X2=None):

        if X2 is None:
            X2 = X

        if self.vertex_dim == 0:
            X_id = X[:, 0].astype('int32')
            X2_id = X2[:, 0].astype('int32')

            K = self._eval_K_vertex(X_id, X2_id)

        else:

            X_id = X[:, 0] 
            X_v = X[:, 1:] 

            X2_id = X2[:, 0]
            X2_v = X2[:, 1:] 

            X_id = X_id.astype('int32')
            X2_id = X2_id.astype('int32')

            K_vertex = self._eval_K_vertex(X_id, X2_id)
            K_point = self.point_kernel(X_v, X2_v)

            K = np.multiply(K_point, K_vertex)
            # K = K_point * K_vertex

            K = K + self.jitter * np.eye(K.shape[0]) # Add jitter

        return K
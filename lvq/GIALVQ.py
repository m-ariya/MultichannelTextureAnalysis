"""
Globalized with charting IALVQ (with Euclidean distance!)
Charting code is based on MATLAB implementation by Laurens van der Maaten https://lvdmaaten.github.io/drtoolbox/
"""

import numpy as np
from lvq.IALVQ import IALVQ
import scipy.sparse.linalg as sla


def normalize(omega):
    nf = np.sqrt(np.trace(omega.T.dot(omega)))
    omega = omega / nf
    return omega


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


class GIALVQ(IALVQ):

    def __init__(self, cialvq, k=3, alpha=None, normalize_omegas=True):

        super().__init__(cialvq.prototypes_per_class, cialvq.initial_prototypes, cialvq.initial_omegas,
                         cialvq.omega_rank,
                         cialvq.max_iter, cialvq.gtol, cialvq.regularization, cialvq.seed, cialvq.omega_locality,
                         cialvq.filter_bank, cialvq.block_eye, cialvq.norm, cialvq.channel_num,
                         cialvq.correct_imbalance)
        self.nb_prototypes = cialvq.nb_prototypes
        self.omegas_ = cialvq.omegas_
        self.class_weights = cialvq.class_weights
        self.samples = cialvq.samples
        self.w_ = cialvq.w_
        self.c_w_ = cialvq.c_w_
        self.classes_ = cialvq.classes_
        self.nb_samples = cialvq.nb_samples
        self.filter_bank = cialvq.filter_bank
        self.nb_classes = cialvq.nb_classes
        self.classes_ = cialvq.classes_
        self.labels = cialvq.labels
        self.normalize_omegas = normalize_omegas
        if k > self.nb_prototypes:
            raise ValueError("k should not exceed the number of prototypes %d" % self.nb_prototypes)
        self.k = k
        self.alpha = alpha
        self.alphas = None
        self.V = None
        self.chart()

    def map_locally(self, X):
        """
        :param X: X data samples
        :return: a lower dimensional representation associated with each prototype/omega
        """
        m = self.omega_rank
        Z = np.zeros((self.nb_prototypes, X.shape[0], m))
        for i in range(self.nb_prototypes):
            Z[i, :, :] = super().project(X - self.w_[i], self.omegas_[self.omega_index(i)])
        return Z

    def bandwidth(self, X):
        """
        Computes bandwidth of a gaussian. Distances to minority class will be *enlarged* ->
        larger bandwidth (sigma), hence their responsibilities will be larger (if alpha is not fixed)
        :param X: data samples
        :return:
        """
        d = self._compute_distance(X)
        idxs = np.argsort(d, 1)[:, 0:self.k]
        d = np.sort(d, 1)[:, 0:self.k]
        d = d * self.alphas[idxs]  # weighting
        sigma = 1 / self.k * np.sqrt(np.sum(d, 1))
        if self.alpha is not None:
            sigma *= self.alpha
        return sigma

    def responsibilities(self, X):
        """
        Computes responsibilities of each prototype for each sample from X
        :param X: data samples
        :return: responsibilities
        """
        R = np.zeros((self.nb_prototypes, X.shape[0]))
        sigma = self.bandwidth(X)
        for i in range(self.nb_prototypes):  # for every prototype
            R[i] = super().distance(X, self.w_[i], self.omegas_[self.omega_index(i)])
            R[i] = np.exp(-R[i] / sigma)
        col_sums = R.sum(axis=0)
        nonzeros = np.where(col_sums != 0)[0]
        R[:, nonzeros] = R[:, nonzeros] / col_sums[nonzeros]
        return R

    def chart(self):
        # ensuring that all omegas have a unit trace
        if not self.norm and self.normalize_omegas:
            for i in range(len(self.omegas_)):
                self.omegas_[i] = normalize(self.omegas_[i])

        # prepare alphas
        alphas = np.ones(self.nb_prototypes)
        if self.alpha is None:
            idx = 0
            for w in self.class_weights:
                for i in range(self.prototypes_per_class):
                    alphas[idx] = w / np.sum(self.class_weights) / self.prototypes_per_class
                    idx += 1
        self.alphas = alphas

        m = self.omega_rank
        n = self.nb_prototypes
        Z = self.map_locally(self.samples)
        Z = np.reshape(Z.T, (m, self.nb_samples, n))
        Z = np.concatenate((Z, np.ones((1, self.nb_samples, n))))
        Z = Z.transpose(0, 2, 1)
        R = self.responsibilities(self.samples)
        D = np.zeros(((m + 1) * n, (m + 1) * n))
        for i in range(n):
            Ds = np.zeros((m + 1, m + 1))
            for j in range(self.nb_samples):
                Ds = Ds + R[i, j] * np.dot(Z[:, i, j][np.newaxis].T, Z[:, i, j][np.newaxis])
            D[i * (m + 1):(i + 1) * (m + 1), i * (m + 1):(i + 1) * (m + 1)] = Ds
        R = R.reshape((1, n, self.nb_samples))
        U = R * Z
        U = U.reshape((n * (m + 1), self.nb_samples), order='F').T
        if not check_symmetric(D - np.dot(U.T, U)):
            print("NON symmetric")
            return None
        lmbda, V = sla.eigsh(A=D - np.dot(U.T, U), M=np.dot(U.T, U), k=(m + 1), which='SM', maxiter=10000)
        ind = np.argsort(lmbda)
        self.V = V[:, ind[1:]]
        self.w_global = self.project(self.w_)
        self.samples_global = U.dot(self.V)
        return V

    def project(self, X):
        """
        Yield global projections of data after charting
        :param X: any data in the original dimensionality
        :return: globally projected data
        """
        m = self.omega_rank
        n = self.nb_prototypes
        Z = self.map_locally(X)
        Z = np.reshape(Z.T, (m, X.shape[0], n))
        Z = np.concatenate((Z, np.ones((1, X.shape[0], n))))
        Z = Z.transpose(0, 2, 1)
        R = self.responsibilities(X)
        R = R.reshape((1, n, X.shape[0]))
        U = R * Z
        U = U.reshape((n * (m + 1), X.shape[0]), order='F').T
        return U.dot(self.V)

    def predict(self, X, return_dist=False, return_projected=False):
        """
        :param x: data
        :return:
        """
        X = self.project(X)
        nb_samples = X.shape[0]
        distance = np.zeros([self.nb_prototypes, nb_samples])
        for i in range(self.nb_prototypes):
            distance[i] = self.distance(X, self.w_global[i])
        distance = np.transpose(distance)
        if return_dist:
            if return_projected:
                return self.c_w_[distance.argmin(1)], distance, X
            else:
                return self.c_w_[distance.argmin(1)], distance
        else:
            if return_projected:
                return self.c_w_[distance.argmin(1)], X
            else:
                return self.c_w_[distance.argmin(1)]

    def distance(self, X, w):
        return np.sum((X - w) ** 2, 1)

    def dist_to_protos(self, x, iqr=[75, 25]):
        _, distances = self.predict(x, True, False)
        dist_statistics = np.zeros((self.nb_classes, self.nb_classes, 3))
        iqrs = np.zeros((self.nb_classes, self.nb_classes, 2))
        for i in self.classes_:
            x_i = np.where(self.labels == i)[0]  # idxs of samples of class i
            for j in self.classes_:
                w_j = np.where(self.c_w_ == j)[0]  # idxs of prototypes of class j
                dist_statistics[i, j, 0] = np.min(distances[np.ix_(x_i, w_j)])
                dist_statistics[i, j, 1] = np.mean(distances[np.ix_(x_i, w_j)])
                dist_statistics[i, j, 2] = np.max(distances[np.ix_(x_i, w_j)])
                iqrs[i, j, :] = np.percentile(distances[np.ix_(x_i, w_j)], iqr)
        return dist_statistics, iqrs

"""
Image Analysis (with) Learning Vector Quantization (IALVQ)
Code is partially based on sklearn-lvq package: https://sklearn-lvq.readthedocs.io/en/stable/
"""

from __future__ import division
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from scipy.optimize import minimize
import logging

omega_nf = []


def closest(distances, label_equality_table):
    distances = distances.copy()
    distances[label_equality_table] = np.inf
    d_closest = distances.min(1)
    idx_closest = distances.argmin(1)
    return d_closest, idx_closest


class IALVQ(ClassifierMixin):

    def __init__(self, prototypes_per_class=1, initial_prototypes=None, initial_omegas=None, omega_rank=None,
                 max_iter=200, gtol=1e-5, regularization=0.0, seed=None, omega_locality="CW",
                 filter_bank=None, block_eye=False, norm=False, channel_num=3, correct_imbalance=False):
        """
        :param prototypes_per_class: number of prototypes per class
        :param initial_prototypes: array-like,
     shape =  [n_prototypes, n_features + 1], optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype
        :param initial_omegas: starting omegas
        :param omega_rank: omega rank
        :param max_iter: limit of iterations during optimization
        :param gtol: maximum projected gradient, stopping criterion for optimizer
        :param regularization: if > 0, the cost function is regularized
        :param seed: for reproducibility of Omegas and prototypes initialization
        :param omega_locality: "CW" for classwise, else prototype-wise
        :param filter_bank: option filter coefficients
        :param block_eye: if True, block-identity matrix is used (for cases where rank = 1/3 of input dimensionality)
        :param norm: if True, Omegas are normalized to unit trace after each iteration
        :param correct_imbalance: if True, updates are weighted based on a class sizes
        """
        self.norm = norm
        self.block_eye = block_eye
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.initial_prototypes = initial_prototypes
        self.prototypes_per_class = prototypes_per_class
        self.max_iter = max_iter
        self.gtol = gtol
        self.regularization = regularization
        self.initial_omegas = initial_omegas
        self.omega_rank = omega_rank
        self.omega_locality = omega_locality if prototypes_per_class != 1 else "PW"
        self.channel_num = channel_num
        self.correct_imbalance = correct_imbalance

        self.x_val = None
        if filter_bank is None:
            self.filter_bank = np.ones(self.omega_rank)[np.newaxis]
        else:
            self.filter_bank = filter_bank[np.newaxis]

    def predict(self, x, ret_dist=False):
        """Predict class membership index for each input sample.
        This function does classification on an array of
        test vectors X.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """

        if x.shape[1] != self.w_.shape[1]:
            raise ValueError("X has wrong number of features\n"
                             "found=%d\n"
                             "expected=%d" % (self.w_.shape[1], x.shape[1]))
        dist = self._compute_distance(x)
        if ret_dist:
            return self.c_w_[dist.argmin(1)], np.min(dist, axis=1)
        return (self.c_w_[dist.argmin(1)])

    def phi(self, x):
        return x

    def phi_prime(self, x):
        return 1

    def _set_prototypes(self):
        nb_ppc = np.ones([self.nb_classes], dtype='int') * self.prototypes_per_class
        if self.initial_prototypes is None:  # init as means with noise
            self.w_ = np.empty([np.sum(nb_ppc), self.nb_features], dtype=np.double)
            self.c_w_ = np.empty([nb_ppc.sum()], dtype=self.classes_.dtype)
            pos = 0
            for cls in range(self.nb_classes):
                nb_prot = nb_ppc[cls]
                mean = np.mean(
                    self.samples[self.labels == self.classes_[cls], :], 0)
                self.w_[pos:pos + nb_prot] = mean + (
                        self.random_state.rand(nb_prot, self.nb_features) * 2 - 1)
                self.c_w_[pos:pos + nb_prot] = self.classes_[cls]
                pos += nb_prot
        else:
            self.w_ = self.initial_prototypes[:, :-1]
            self.c_w_ = self.initial_prototypes[:, -1].astype(int)

        logging.debug("Prototypes set")

        self.nb_prototypes = self.c_w_.shape[0]
        self.c_ = np.ones((self.c_w_.size, self.c_w_.size))

    def _set_omegas(self):
        self.nb_omegas = self.nb_prototypes if self.omega_locality == "PW" else self.nb_classes
        self.split_indices = np.arange(self.omega_rank, self.omega_rank * self.nb_omegas, self.omega_rank)

        if self.omega_rank is None:
            self.omega_rank = self.nb_features

        if self.initial_omegas is None:
            self.omegas_ = []
            for omega in range(self.nb_omegas):
                if self.block_eye:
                    eye = np.eye(self.nb_features // self.channel_num, self.nb_features // self.channel_num)
                    omega = np.stack([eye for _ in range(self.channel_num)]).ravel().reshape(self.nb_features,
                                                                                             self.nb_features // self.channel_num)
                    omega = omega.T
                    if self.norm:
                        nf = np.sqrt(np.trace(omega.T.dot(omega)))
                        omega = omega / nf
                    self.omegas_.append(omega)
                else:
                    omega = self.random_state.rand(self.omega_rank, self.nb_features) * 2.0 - 1.0
                    if self.norm:
                        nf = np.sqrt(np.trace(omega.T.dot(omega)))
                        omega = omega / nf
                    self.omegas_.append(omega)

        else:
            if not isinstance(self.initial_omegas, list):
                raise ValueError("initial matrices must be a list")
            self.omegas_ = list(self.initial_omegas)
        logging.debug("Omegas set")

    def omega_index(self, prototype_idx):
        """
        Useful for CW Omegas
        :param prototype_idx: index of a prototype
        :return: omega associated with the prototype
        """
        if self.omega_locality == 'CW':
            omega_idx = np.where((self.c_w_[prototype_idx] == self.classes_) == 1)[0][0]
        else:
            omega_idx = prototype_idx
        return omega_idx

    def prototypes_idxs(self, omega_idx):
        """
        Useful for CW Omegas
        :param omega_idx: index of an omega
        :return: prototype(s) associated with an omega
        """
        if self.omega_locality == 'CW':
            idxs = np.where((self.classes_[omega_idx] == self.c_w_) == 1)[0]
        else:
            idxs = [omega_idx]
        return idxs

    def _initialize(self, x, y):
        """
        :param x: train samples
        :param y: train labels
        :return:
        """
        self.samples = np.array(x)
        self.labels = np.array(y)
        self.nb_samples, self.nb_features = self.samples.shape
        self.classes_ = unique_labels(self.labels)
        self.nb_classes = len(self.classes_)
        # set prototypes
        self._set_prototypes()
        # set omegas
        self._set_omegas()

    def _validate_params(self):
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be an positive integer")
        if not isinstance(self.gtol, float) or self.gtol <= 0:
            raise ValueError("gtol must be a positive float")
        if self.omega_rank is not None and self.omega_rank <= 0:
            raise ValueError("rank must be a  positive int")
        if self.regularization < 0:
            raise ValueError('regularization must be a positive float')

        logging.debug("Parameters Validated")

    def fit(self, x, y, x_val=None, y_val=None):
        """Fit the LVQ model to the given training data and parameters using
        l-bfgs-b.

        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
          Training vector, where n_samples in the number of samples and
          n_features is the number of features.
        y : array, shape = [n_samples]
          Target values (integers in classification, real numbers in
          regression)

        Returns
        --------
        self
        """
        if x_val is not None:
            self.x_val = x_val
            self.y_val = y_val
        self._validate_params()
        self.class_weights = 1 / np.bincount(y)

        self._initialize(x, y)
        self.cost_mat = np.repeat([self.class_weights], self.nb_classes, 0).T
        self.cost_mat = self.cost_mat / np.sum(self.cost_mat)
        self._optimize()
        return self

    def _reg_term(self, omega):
        """
        :param omega:
        :return: regularization term for omega
        """
        # for rectangular matrix det(omega.T.dot(omega)) will always be 0, hence  det(omega.dot(omega.T)) is used
        slogdet = np.linalg.slogdet(omega.dot(omega.T))
        return slogdet[1]

    def _g(self, variables, label_equals_prototype,
           lr_relevances=1,
           lr_prototypes=1):
        """
        Gradient of the cost function
        :param variables: omegas and prototypes
        :param label_equals_prototype:
        :param lr_relevances: learning rate
        :param lr_prototypes: learning rate
        :return: gradient
        """

        variables = variables.reshape(variables.size // self.nb_features,
                                      self.nb_features)  # to 2D

        omegas = np.split(variables[self.nb_prototypes:], self.split_indices)

        if self.norm:
            for i, omega in enumerate(omegas):
                omegas[i] = omega / omega_nf[i]

        dist = self._compute_distance(self.samples, variables[:self.nb_prototypes], omegas)

        # distances to closest wrong and correct w for every sample
        dist_k, idx_k = closest(dist, label_equals_prototype)
        dist_j, idx_j = closest(dist, np.invert(label_equals_prototype))

        mu = (dist_j - dist_k) / (dist_j + dist_k)
        mu = np.vectorize(self.phi_prime)(mu)

        g = np.zeros(variables.shape)
        norm_factors = 4 / (dist_j + dist_k) ** 2

        if self.correct_imbalance:
            norm_factors = norm_factors * [self.cost_mat[self.labels[i], self.c_w_[idx_j[i]]] for i in
                                           range(self.nb_samples)]

        gw = []
        for i in range(len(omegas)):
            gw.append(np.zeros(omegas[i].shape))

        for i in range(self.nb_prototypes):
            idx_ij = i == idx_j  # idx of samples for which w_i is  the closest correct
            idx_ik = i == idx_k  # idx of samples for which w_i is  the closest wrong

            omega_idx = self.omega_index(i)

            mu_j = mu[idx_ik] * dist_j[idx_ik] * norm_factors[idx_ik]
            mu_k = mu[idx_ij] * dist_k[idx_ij] * norm_factors[idx_ij]

            d_j = self.samples[idx_ij] - variables[i]  # displacement vectors of samples for which wi is a match
            d_k = self.samples[idx_ik] - variables[i]  # displacement vectors of samples  for which wi is not a match

            d_omega_filter = omegas[omega_idx].T * (self.filter_bank ** 2)
            g[i] = (mu_j.dot(d_k) - mu_k.dot(d_j)).dot(np.dot(d_omega_filter, omegas[omega_idx]))

            gw[omega_idx] -= (d_k * mu_j[np.newaxis].T).dot(d_omega_filter).T.dot(d_k) - \
                             (d_j * mu_k[np.newaxis].T).dot(d_omega_filter).T.dot(d_j)

        if self.block_eye:
            gw = self._enforce_block_eye(gw)

        # update omegas
        if self.regularization > 0:
            regmatrices = np.zeros((len(self.omegas_) * self.omega_rank, self.nb_features))
            for i, omega in enumerate(omegas):
                regmatrices[self.omega_rank * (i + 1) - self.omega_rank: self.omega_rank * (i + 1)] \
                    = self.regularization * 2 * np.linalg.pinv(omega).T
            g[self.nb_prototypes:] = 1 / self.nb_samples * (lr_relevances * np.concatenate(gw) - regmatrices)
        else:
            g[self.nb_prototypes:] = 1 / self.nb_samples * lr_relevances * np.concatenate(gw)

        # update prototypes
        g[:self.nb_prototypes] = 1 / self.nb_samples * lr_prototypes * g[:self.nb_prototypes]

        return g.ravel()

    def _enforce_block_eye(self, omegas):
        """
        Preserves only block-diagonal entries of matrices
        :param omegas:
        :return: block-identity omegas
        """
        eye = np.eye(self.nb_features // self.channel_num, self.nb_features // self.channel_num)
        eye = np.stack([eye for _ in range(self.channel_num)]).ravel().reshape(self.nb_features,
                                                                               self.nb_features // self.channel_num).T
        for i in range(len(omegas)):
            omegas[i] = omegas[i] * eye
        return omegas

    def _f(self, variables, label_equals_prototype):
        """
        Cost function
        :param variables: omegas and prototypes
        :param label_equals_prototype:
        :return: cost function value
        """
        variables = variables.reshape(variables.size // self.nb_features,
                                      self.nb_features)  # to 2D
        omegas = np.split(variables[self.nb_prototypes:], self.split_indices)

        if self.norm:
            omega_nf.clear()
            for i, omega in enumerate(omegas):
                nf = np.sqrt(np.trace(omega.T.dot(omega)))
                omega_nf.append(nf)
                omegas[i] = omega / nf

        dist = self._compute_distance(self.samples, variables[:self.nb_prototypes],
                                      omegas)  # distances from all samples (rows) to all prototypes (cols)

        distwrong, idxwrong = closest(dist, label_equals_prototype)  # idxs of protos for each sample
        distcorrect, pidxcorrect = closest(dist, np.invert(label_equals_prototype))

        distcorrectpluswrong = distcorrect + distwrong
        distcorectminuswrong = distcorrect - distwrong
        mu = distcorectminuswrong / distcorrectpluswrong  # per sample

        if self.correct_imbalance:
            mu = [mu[i] * self.cost_mat[self.labels[i], self.c_w_[idxwrong[i]]] for i in range(self.nb_samples)]

        if self.regularization > 0:
            reg_terms = 0.5 * self.regularization * np.array([self._reg_term(omega) for omega in omegas])
            result = np.vectorize(self.phi)(mu).sum() - np.sum(reg_terms) * 1 / self.nb_samples
        else:
            result = np.vectorize(self.phi)(mu).sum() * 1 / self.nb_samples
        return result

    def _optimize(self):
        variables = np.append(self.w_, np.concatenate(self.omegas_),
                              axis=0)  # First rows are prototypes, following are omegas
        label_equals_prototype = self.labels[
                                     np.newaxis].T == self.c_w_  # table showing which prototypes have same label as sample
        res = minimize(
            fun=lambda vs: self._f(
                vs, label_equals_prototype=label_equals_prototype),
            jac=lambda vs: self._g(
                vs, label_equals_prototype=label_equals_prototype,
                lr_prototypes=1, lr_relevances=1),
            # jac='3-point',
            method='L-BFGS-B',
            x0=variables, options={'gtol': self.gtol,
                                   'maxiter': self.max_iter})

        out = res.x.reshape(res.x.size // self.nb_features, self.nb_features)
        print(res.message, res.nit)
        self.w_ = out[:self.nb_prototypes]
        indices = np.arange(self.omega_rank, self.omega_rank * self.nb_omegas + 1, self.omega_rank)
        self.omegas_ = np.split(out[self.nb_prototypes:], indices[:-1])

        if self.norm:
            for i, omega in enumerate(self.omegas_):
                self.omegas_[i] = self.omegas_[i] / omega_nf[i]

    def _distance(self, x, w, omega):
        """
        Computes Squared Euclidean distance
        :param x: samples
        :param w: prototype
        :param omega:
        :return: distances from all samples to a prototype w using omega
        """
        return np.sum(np.dot(x - w, omega) ** 2, 1)

    def distance(self, x, w, omega):
        """
        To be used for computation between using same Omega
        :param x: data points
        :param w: prototype(s), either one prototoype or w.shape[0] = x.shape[0]
        :param omega: transformation matrix mxn
        :return: distance between x and w using omega (and optionally a filter bank)
        """
        if x.ndim == 1 and w.ndim == 1:
            return np.sum(np.dot(x - w, omega.T * self.filter_bank) ** 2)
        return np.sum(np.dot(x - w, omega.T * self.filter_bank) ** 2, 1)

    def _compute_distance(self, x, w=None, omegas=None):
        """
        Distance from all samples to all prototypes
        :param x: samples
        :param w: prototypes, if None, the current trained prototypes are used
        :param omegas:  if None, the current trained omegas are used
        :return: distances from all samples to all prototypes
        """
        if w is None:
            w = self.w_
        if omegas is None:
            omegas = self.omegas_
        nb_samples = x.shape[0]
        if len(w.shape) == 1:
            nb_prototypes = 1
        else:
            nb_prototypes = w.shape[0]
        distance = np.zeros([nb_prototypes, nb_samples])
        if self.omega_locality == "PW":
            for i in range(nb_prototypes):
                # distances for all samples for one prototype
                distance[i] = self._distance(x, w[i], omegas[i].T * self.filter_bank)
        else:
            for i in range(nb_prototypes):
                omega_idx = self.omega_index(i)
                distance[i] = self._distance(x, w[i], omegas[omega_idx].T * self.filter_bank)

        return np.transpose(distance)

    def project(self, X, omega):
        """
        Projects X with omega
        :param X: data sample(s)
        :param omega:
        :return:
        """
        return np.dot(X, omega.T)

    def project_eigen(self, x, prototype_idx, dims):
        """Projects the data input data X using the relevance matrix of the
        prototype specified by prototype_idx to dimension dim
        Parameters
        ----------
        x : array-like, shape = [n,n_features]
          input data for project
        prototype_idx : int
          index of the prototype
        dims : int
          dimension to project to

        Returns
        --------
        C : array, shape = [n,n_features]
            Returns predicted values.
        """
        nb_prototypes = self.w_.shape[0]
        if len(self.omegas_) != nb_prototypes \
                or self.prototypes_per_class != 1:
            print('project only possible with classwise relevance matrix')
        v, u = np.linalg.eig(
            self.omegas_[prototype_idx].T.dot(self.omegas_[prototype_idx]))
        idx = v.argsort()[::-1]

        return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v[idx][:dims]))))

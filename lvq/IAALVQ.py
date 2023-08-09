"""
Extension of IALVQ with Parametrized Angle-based distance metric
Is based on MATLAB implementation by Kerstin Bunte (ORCID 0000-0002-2930-6172)
"""
from lvq.IALVQ import *


class IAALVQ(IALVQ):

    def __init__(self, prototypes_per_class=1, initial_prototypes=None, initial_omegas=None, omega_rank=None,
                 max_iter=200, gtol=1e-5, regularization=0.0, seed=None, omega_locality="CW",
                 filter_bank=None, block_eye=False, norm=False, beta=1, channel_num=3, correct_imbalance=False):
        """
        :param beta: controls the slope of the transformation function. Small beta leads to near-linear function
        """
        self.beta = beta
        super().__init__(prototypes_per_class, initial_prototypes, initial_omegas, omega_rank,
                         max_iter, gtol, regularization, seed, omega_locality,
                         filter_bank, block_eye, norm, channel_num, correct_imbalance)

    def _compute_distance(self, x, w=None, omegas=None):
        if w is None:
            w = self.w_
        if omegas is None:
            omegas = self.omegas_
        dist, _, _, _, _ = self._distance(x, w, omegas)
        return dist

    def distance(self, x, w, omega):

        """
        To be used for computation between using same Omega
        :param x: data points
        :param w: prototypes
        :param omega: transformation matrix mxn
        :return: distance between x and w using omega (and optionally a filter bank)
        """
        beta = self.beta
        if w.ndim == 1:  # computing to a single prototype
            norms_w = np.linalg.norm(w.dot(omega.T * self.filter_bank))
            norms_x = np.linalg.norm(x.dot(omega.T * self.filter_bank), axis=1)
            xAw = x.dot(omega.T * self.filter_bank ** 2).dot(omega).dot(w)
            cosa = (xAw / (norms_x * norms_w)).T
            return (np.exp(-beta * (cosa - 1)) - 1) / (np.exp(2 * beta) - 1)
        else:
            size = x.shape[0]
            norms_x = np.zeros(size)
            norms_w = np.zeros(size)
            xAw = np.zeros(size)
            for i in range(size):
                norms_w[i] = np.linalg.norm(w[i].dot(omega.T * self.filter_bank))
                norms_x[i] = np.linalg.norm(x[i].dot(omega.T * self.filter_bank))
                xAw[i] = x[i].dot(omega.T * self.filter_bank ** 2).dot(omega).dot(w[i])
            # compute distance
            cosa = (xAw / (norms_x * norms_w)).T
            return (np.exp(-beta * (cosa - 1)) - 1) / (np.exp(2 * beta) - 1)

    def _distance(self, x, w, omegas):
        """
        :param x: samples
        :param w: prototypes
        :param omegas:
        :return: distances (transformed with g), dot product of xAw, cos of xAw, norms of samples and prototypes
        """
        beta = self.beta
        # precompute samples' and prototypes' lengths
        norms_x = np.zeros((self.nb_prototypes, x.shape[0]))
        norms_w = np.zeros((self.nb_prototypes, 1))
        xAw = np.zeros((self.nb_prototypes, x.shape[0]))
        for i in range(self.nb_prototypes):
            omega_idx = self.omega_index(i)
            norms_w[i] = np.linalg.norm(w[i].dot(omegas[omega_idx].T * self.filter_bank))
            norms_x[i] = np.linalg.norm(x.dot(omegas[omega_idx].T * self.filter_bank), axis=1)
            xAw[i] = x.dot(omegas[omega_idx].T * self.filter_bank ** 2).dot(omegas[omega_idx]).dot(w[i])
        # compute distance
        cosa = (xAw / (norms_x * norms_w)).T
        xAw = xAw.T
        dist = (np.exp(-beta * (cosa - 1)) - 1) / (np.exp(2 * beta) - 1)
        return dist, xAw, cosa, norms_x, norms_w

    def _g(self, variables, label_equals_prototype,
           lr_relevances=1,
           lr_prototypes=1):
        """
        Gradient of a cost function
        :param variables: omegas and prototypes
        :param label_equals_prototype:
        :param lr_relevances: learning rate
        :param lr_prototypes: learniing rate
        :return: gradient
        """
        beta = self.beta
        variables = variables.reshape(variables.size // self.nb_features,
                                      self.nb_features)  # to 2D

        omegas = np.split(variables[self.nb_prototypes:], self.split_indices)

        if self.norm:
            for i, omega in enumerate(omegas):
                omegas[i] = omega / omega_nf[i]

        prototypes = variables[0:self.nb_prototypes]

        dist, xAw, cosa, norms_x, norms_w = self._distance(self.samples, prototypes, omegas)

        # distances to closest wrong and correct w for every sample
        dist_k, idx_k = closest(dist, label_equals_prototype)
        dist_j, idx_j = closest(dist, np.invert(label_equals_prototype))

        mu = (dist_j - dist_k) / (dist_j + dist_k)
        mu = np.vectorize(self.phi_prime)(mu)

        g = np.zeros(variables.shape)
        norm_factors = 2 / (dist_j + dist_k) ** 2

        if self.correct_imbalance:
            norm_factors = norm_factors * [self.cost_mat[self.labels[i], self.c_w_[idx_j[i]]] for i in
                                           range(self.nb_samples)]

        ga = []
        for i in range(self.nb_prototypes, len(variables)):
            ga.append(np.zeros(self.nb_features))

        mud_j = mu[idx_k] * dist_k * norm_factors
        mud_k = mu[idx_j] * -1 * dist_j * norm_factors

        dgw_j = -beta / (np.exp(2 * beta) - 1) * np.exp(-beta * cosa[np.arange(0, len(idx_j)), idx_j] + beta)
        dgw_k = -beta / (np.exp(2 * beta) - 1) * np.exp(-beta * cosa[np.arange(0, len(idx_k)), idx_k] + beta)

        for i in range(self.nb_prototypes):
            omega_idx = self.omega_index(i)
            idx_ij = i == idx_j  # idx of samples for which w_i is  the closest correct
            idx_ik = i == idx_k  # idx of samples for which w_i is  the closest wrong

            d_omega_filter = omegas[omega_idx].T * (self.filter_bank ** 2)
            dbw = self.samples[idx_ij].dot(d_omega_filter).dot(omegas[omega_idx]) * norms_w[i] ** 2
            dbw = dbw - (xAw[idx_ij, i][np.newaxis].T * variables[i].dot(d_omega_filter).dot(omegas[omega_idx]))
            dbw = dbw.T / (norms_x[i, idx_ij] * norms_w[i] ** 3)
            gwj = np.sum(mud_j[idx_ij] * dgw_j[idx_ij] * dbw, axis=1)

            dbw = self.samples[idx_ik].dot(d_omega_filter).dot(omegas[omega_idx]) * norms_w[i] ** 2
            dbw = dbw - (xAw[idx_ik, i][np.newaxis].T * variables[i].dot(d_omega_filter).dot(omegas[omega_idx]))
            dbw = dbw.T / (norms_x[i, idx_ik] * norms_w[i] ** 3)
            gwk = np.sum(mud_k[idx_ik] * dgw_k[idx_ik] * dbw, axis=1)
            g[i] = gwk + gwj

        # update Omegas
        for i in range(self.nb_prototypes, len(variables)):  # iterate through rows of omegas
            omega_row = variables[i] * self.filter_bank[:, (i - self.nb_prototypes) % self.filter_bank.shape[1]] ** 2
            w_idxs = self.prototypes_idxs(
                (i - self.nb_prototypes) // self.omega_rank)  # prototypes indices associated with current omega
            for w_idx in w_idxs:
                actw_j = np.where((idx_j == w_idx) == 1)[0]  # indices of samples for which w is closest correct
                actw_k = np.where((idx_k == w_idx) == 1)[0]

                dcosaJdA = self.samples[actw_j] * (variables[w_idx].dot(omega_row)) + variables[w_idx] * \
                           (self.samples[actw_j].dot(omega_row))[np.newaxis].T
                dcosaJdA = dcosaJdA / (norms_x[w_idx, actw_j] * norms_w[w_idx])[np.newaxis].T
                temp = (self.samples[actw_j] * (self.samples[actw_j].dot(omega_row[np.newaxis].T))) / \
                       (norms_x[w_idx, actw_j] ** 3 * norms_w[w_idx])[np.newaxis].T
                temp += (variables[w_idx] * (variables[w_idx].dot(omega_row))[np.newaxis].T) / \
                        (norms_x[w_idx, actw_j] * norms_w[w_idx] ** 3)[np.newaxis].T
                dcosaJdA -= (xAw[actw_j, w_idx][np.newaxis].T * temp)

                dcosaKdA = self.samples[actw_k] * (variables[w_idx].dot(omega_row)) + variables[w_idx] * \
                           (self.samples[actw_k].dot(omega_row))[np.newaxis].T
                dcosaKdA = dcosaKdA / (norms_x[w_idx, actw_k] * norms_w[w_idx])[np.newaxis].T

                temp = (self.samples[actw_k] * (self.samples[actw_k].dot(omega_row[np.newaxis].T))) / \
                       (norms_x[w_idx, actw_k] ** 3 * norms_w[w_idx])[np.newaxis].T

                temp += (variables[w_idx] * (variables[w_idx].dot(omega_row))[np.newaxis].T) / \
                        (norms_x[w_idx, actw_k] * norms_w[w_idx] ** 3)[np.newaxis].T

                dcosaKdA -= (xAw[actw_k, w_idx][np.newaxis].T * temp)

                ga[i - self.nb_prototypes] += np.sum((mud_j[actw_j] * dgw_j[actw_j])[np.newaxis].T * dcosaJdA,
                                                     axis=0) + np.sum(
                    (mud_k[actw_k] * dgw_k[actw_k])[np.newaxis].T * dcosaKdA, axis=0)

        if self.block_eye:
            ga = self._enforce_block_eye(ga)

        # update omegas
        g[self.nb_prototypes:] = lr_relevances * ga
        if self.regularization > 0:
            regmatrices = np.zeros((len(self.omegas_) * self.omega_rank, self.nb_features))
            for i, omega in enumerate(omegas):
                regmatrices[self.omega_rank * (i + 1) - self.omega_rank: self.omega_rank * (i + 1)] \
                    = self.regularization * 2 * np.linalg.pinv(omega).T
            g[self.nb_prototypes:] = g[self.nb_prototypes:] - regmatrices

        g[self.nb_prototypes:] = g[self.nb_prototypes:] * 1 / self.nb_samples
        # update prototypes
        g[:self.nb_prototypes] = 1 / self.nb_samples * lr_prototypes * g[:self.nb_prototypes]

        return g.ravel()

    def _enforce_block_eye(self, omegas):
        """
               Preserves only block-diagonal entries of matrices
               :param omegas:
               :return: block-identity omegas
               """
        j = 0
        for i in range(len(omegas)):
            if i % self.omega_rank == 0:
                j = 0
            eye = np.zeros(self.nb_features)
            for k in range(self.channel_num):
                eye[j + k * self.omega_rank] = 1
            omegas[i] = omegas[i] * eye
            j = j + 1
        return omegas

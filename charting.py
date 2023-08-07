import numpy as np
import scipy
import scipy.sparse.linalg as sla


def map_locally(model, X):
    m=model.omega_rank
    n=model.nb_prototypes
    Z = np.zeros((n, X.shape[0], m))
    for i in range(n):
        Z[i, :, :] = model.project(X-model.w_[i], model.omegas_[model.omega_index(i)])
    return Z

def responsibilities(model, X, alpha=None,k=3):
    R = np.zeros((model.nb_prototypes, X.shape[0]))
    sigma = bandwidth(model,X,alpha,k)
    for i in range(model.nb_prototypes): # for every prototype
        R[i] = model.distance(X, model.w_[i], model.omegas_[model.omega_index(i)])
        R[i] = np.exp(-R[i]/sigma)
    col_sums = R.sum(axis=0)
    nonzeros = np.where(col_sums!=0)[0]
    R[:,nonzeros] = R[:,nonzeros]/col_sums[nonzeros]

    return R


def bandwidth(model, X,alpha=None,k=3):
    if k > model.nb_prototypes:
        raise ValueError("k should not exceed the number of prototypes %d" % model.nb_prototypes)
# distances to minority class will be *enlarged* -> larger bandwidth (sigma), hence their responsibilities will be larger
    d=model._compute_distance(X)
    alphas = np.ones(model.nb_prototypes)
    if alpha is None:
        idx=0
        for w in model.class_weights:
            for i in range(model.prototypes_per_class):
                alphas[idx] = w/np.sum(model.class_weights)/model.prototypes_per_class
                idx +=1
    idxs = np.argsort(d,1)[:,0:k]
    d = np.sort(d,1)[:,0:k]
    d = d * alphas[idxs] # weighting
    sigma =  1 / k * np.sqrt(np.sum(d, 1))
    if alpha is not None:
        sigma *= alpha
    return sigma


def map_globally(model, V, X, alpha=None, k=3):
    m = model.omega_rank
    n = model.nb_prototypes
    Z = map_locally(model, X)
    Z = np.reshape(Z.T, (m, X.shape[0], n))
    Z = np.concatenate((Z, np.ones((1, X.shape[0], n))))
    Z = Z.transpose(0, 2, 1)
    R = responsibilities(model,X,alpha,k)
    R = R.reshape((1, n, X.shape[0]))
    U = R * Z
    U = U.reshape((n * (m + 1), X.shape[0]), order='F').T
    return U.dot(V)

def chart(model, a=None, k=3):
    m = model.omega_rank
    n = model.nb_prototypes
    Z = map_locally(model, model.samples)
    Z = np.reshape(Z.T, (m, model.samples.shape[0], n))
    Z = np.concatenate((Z, np.ones((1, model.samples.shape[0], n))))
    Z = Z.transpose(0, 2, 1)
    R = responsibilities(model, model.samples, a, k)
    D = np.zeros(((m + 1) * n, (m + 1) * n))
    for i in range(n):
        Ds = np.zeros((m + 1, m + 1))
        for j in range(model.samples.shape[0]):
            Ds = Ds + R[i, j] * np.dot(Z[:, i, j][np.newaxis].T, Z[:, i, j][np.newaxis])
        D[i * (m + 1):(i + 1) * (m + 1), i * (m + 1):(i + 1) * (m + 1)] = Ds
    R = R.reshape((1, n, model.samples.shape[0]))
    U = R * Z
    U = U.reshape((n * (m + 1), model.samples.shape[0]), order='F').T
    if not check_symmetric(D - np.dot(U.T,  U)):
        print("NON symmetric")
        return None
    lmbda ,V= sla.eigsh(A=D - np.dot(U.T,  U), M=np.dot(U.T, U), k=(m+1),  which='SM',maxiter=10000)
    ind = np.argsort(lmbda)
    V = V[:, ind[1:]]
    return V, U.dot(V)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)












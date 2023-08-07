import numbers

import matplotlib.pyplot as plt
import numpy as np
import skimage
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
from sklearn import metrics
from sklearn.datasets import make_blobs, make_gaussian_quantiles
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from sklearn.utils import check_random_state, resample

from charting import chart, map_globally
from lvq.GCIALVQ import *
# from lvq.CIALVQ import *
from utils.DataManager import *
# import utils.ResultsManager
import scipy.sparse.linalg as sla

from utils.visualization import show_lambdas

SHOW_DS = 1
TRAIN = 0
GENERATE_DS = 1
GENERATE_DS2 = 0
DISCR = 0
DISCR_CHARTING = 0

LOAD = 1
model = None


def random_dist_torus(a, c, n=200, seed=11):
    """
    Parameters
    c: radius from the center of the hole to the center of the torus
    a: radius of the tube
    n_low: minimum number of points in the volume
    n_high: maximum number of points in the volume
    """
    np.random.seed(seed)
    # consider a cross section of the torus, which is going
    # to be a circle. The variable r represents the radius
    # of the points in this cross section.
    # r = a * np.random.uniform(0, 1, n)
    r = a * np.random.normal(0, 0.5, n)
    # u: angular coordinate of the torus. Here I set it
    # to np.pi, so it will generate only half torus
    u = np.pi * np.random.uniform(0, 1, n)
    # v: circumferential coordinate. Consider a cross section
    # of a torus, which is going to be a circle. v represents
    # the angle around this circle.
    v = 2 * np.pi * np.random.normal(0, 0.1, n)  # z
    return (
        (c + r * np.cos(v)) * np.cos(u),  # x-coordinates
        (c + r * np.cos(v)) * np.sin(u),  # y-coordinates
        r * np.sin(v)  # z-coordinates
    )


def random_blob(n=100, x_std=0.5, y_std=0.5, z_std=0.5, seed=11):
    """
    :param n: number of samples
    :param x_std: std. dev. in x direction
    :param y_std: std. dev. in y direction
    :param z_std: std. dev. in x direction
    :param seed: seed for reproducibility
    :return: a point cloud with the same std. dev. for all features
    """
    np.random.seed(seed)
    x = np.random.normal(0, x_std, n)
    y = np.random.normal(0, y_std, n)
    z = np.random.normal(0, z_std, n)
    return x, y, z


if LOAD:
    dm = DataManager()
    # model = dm.load_model("QF_s=11_reg=0.0_norm=False_be=False_synth.pkl")#4classes.pkl")
    samples = np.load("synth_samples.npy")
    labels = np.load("synth_labels.npy")

    blues = ListedColormap(colormaps['Blues'](np.linspace(0.4, 1, 300)))
    oranges = ListedColormap(colormaps['Oranges'](np.linspace(0.4, 1, 300)))
    purples = ListedColormap(colormaps['Purples'](np.linspace(0.4, 1, 300)))
    colormaps = [blues, oranges, purples]
    ax = 0
    coords_0 = samples[labels == 0]
    ord_0 = coords_0[:, ax].argsort()
    coords_0 = coords_0[ord_0]
    colors_0 = blues(np.arange(0, coords_0.shape[0]))

    coords_1 = samples[labels == 1]
    ord_1 = coords_1[:, ax].argsort()
    coords_1 = coords_1[ord_1]
    colors_1 = oranges(np.arange(0, coords_1.shape[0]))

    coords_2 = samples[labels == 2]
    ord_2 = coords_2[:, ax].argsort()
    coords_2 = coords_2[ord_2]
    colors_2 = purples(np.arange(0, coords_2.shape[0]))

    colors = [colors_0, colors_1, colors_2]
    ords = [ord_0, ord_1, ord_2]

if TRAIN:
    init_w = np.array([[-0.63139788, 0.04169798, -0.07517481, 0],
                       [3.51485576, 3.23199772, -0.0085968, 1, ],
                       [2.09054953, -1.4168473, 0.90416231, 2, ],
                       [-2.6388001, 3.41178327, -0.80259249, 1],
                       [-2.55258194, -0.23754583, -0.6898914, 2]])
    model = CIALVQ(max_iter=200, prototypes_per_class=1, omega_rank=2, seed=11,
                   regularization=0., omega_locality='PW',
                   block_eye=False, norm=False, correct_imbalance=True, initial_prototypes=init_w)
    model.fit(samples, labels)
    print(model.score(samples, labels))
# l=show_lambdas(model.omegas_, class_names=[0,1,2], colors=colors, title=r'$\Lambda=\Omega^T\Omega$')


if GENERATE_DS2:
    overlap = 2
    n0 = 100
    n1 = 300
    n2 = 300
    x0, y0, z0 = random_blob(n=n0, x_std=0.7, y_std=0.7, z_std=0.1)
    y0 = y0 + overlap / 2
    x1, y1, z1 = random_dist_torus(1, 5, n=n1)
    x2, y2, z2 = random_dist_torus(1, 5, n=n2)
    y2 = -y2 + overlap

    samples0 = np.stack((x0, y0, z0), axis=1)
    samples1 = np.stack((x1, y1, z1), axis=1)
    samples3 = samples1[samples1[:, 0] < 0]
    samples1 = samples1[samples1[:, 0] >= 0]

    samples2 = np.stack((x2, y2, z2), axis=1)
    samples4 = samples2[samples2[:, 0] < 0]
    samples2 = samples2[samples2[:, 0] >= 0]

    samples = np.vstack((samples0, samples1, samples2, samples3, samples4))
    labels = np.repeat([0, 1, 2, 3, 4],
                       [n0, samples1.shape[0], samples2.shape[0], samples3.shape[0], samples4.shape[0]], axis=0)
    np.save("synth_samples_4classes", samples)
    np.save("synth_labels_4classes", labels)

if GENERATE_DS:
    overlap = 2
    n0 = 100
    n1 = 300
    n2 = 300
    x0, y0, z0 = random_blob(n=n0, x_std=0.7, y_std=0.7, z_std=0.1)
    y0 = y0 + overlap / 2
    x1, y1, z1 = random_dist_torus(1, 5, n=n1)
    x2, y2, z2 = random_dist_torus(1, 5, n=n2)
    y2 = -y2 + overlap

    samples0 = np.stack((x0, y0, z0), axis=1)
    samples1 = np.stack((x1, y1, z1), axis=1)
    samples2 = np.stack((x2, y2, z2), axis=1)
    samples = np.vstack((samples0, samples1, samples2))
    labels = np.repeat([0, 1, 2], [n0, n1, n2], axis=0)
    #np.save("synth_samples", samples)
    #np.save("synth_labels", labels)

if SHOW_DS:
    # title='bla'
    proj = 3
    init_w = np.array([[-0.63139788, 0.04169798, -0.07517481],
                       [0.48297675, 3.02385593, -0.02782059],
                       [-0.94132948, -1.2087055, 0.88493852]])
    init_w = []
    ax = plt.axes(projection='3d')
    for idx, label in enumerate(np.unique(labels)):
        coords = samples[labels == label]
        coords = coords[ords[idx]]
        ax.scatter3D(xs=coords[:, 0], ys=coords[:, 1], zs=coords[:, 2],
                     label=label, c=colors[idx])
    plt.legend()
    if model is not None:
        for idx, label in enumerate(np.unique(labels)):
            ax.scatter3D(xs=model.w_[model.c_w_ == label, 0], ys=model.w_[model.c_w_ == label, 1],
                         zs=model.w_[model.c_w_ == label, 2], label=label, c=colors[idx][0:model.prototypes_per_class],
                         s=100, linewidths=2, edgecolor='black', marker='^')
    elif len(init_w) != 0:
        cw = [0, 1, 2]
        for idx, label in enumerate(np.unique(labels)):
            ax.scatter3D(xs=init_w[cw == label, 0], ys=init_w[cw == label, 1],
                         zs=init_w[cw == label, 2], label=label, c=colors[idx][0],
                         s=100, linewidths=2, edgecolor='black', marker='^')

    if proj == 2:
        ax.view_init(90, 0)
        ax.set_zticks([])
    elif proj == 3:
        ax.view_init(180, 0, 180)
        ax.set_xticks([])
    ax.set_xlabel(r'$\mathbf{x}$', fontsize='x-large')
    ax.set_ylabel(r'$\mathbf{y}$', fontsize='x-large')
    ax.set_zlabel(r'$\mathbf{z}$', fontsize='x-large')
    # plt.title(title)
    plt.show()
    title = 'synth/init_data180'
    #plt.savefig(title + '.png', bbox_inches='tight', pad_inches=0.0)

if DISCR:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    class_names = ['0', '1', '2']
    plot = None

    for cls in model.classes_:
        unique_labels = np.unique(labels)
        idx = np.where(unique_labels == cls)[0]
        temp = unique_labels[-1]
        unique_labels[-1] = cls
        unique_labels[idx] = temp
        # class_projected_train = model.project(samples, model.omegas_[cls])
        class_projected_train = model.project2(samples, cls, 2)
        for label in unique_labels:
            coords = class_projected_train[labels == label]
            coords = coords[ords[label]]
            plot = axes[cls].scatter(coords[:, 0], coords[:, 1], c=colors[label])

        # w = model.project(model.w_, model.omegas_[cls])
        w = model.project2(model.w_, cls, 2)
        for label in unique_labels:
            coords = w[model.c_w_ == label]
            axes[cls].scatter(coords[:, 0], coords[:, 1], edgecolors='black', linewidths=2,
                              c=colors[label][0:model.prototypes_per_class], marker='^', s=100)
        axes[cls].set_title(class_names[cls], color=colors[cls][99], fontsize='xx-large', fontweight='bold')
        for i in unique_labels:
            axes[i].get_yaxis().set_ticks([])
            axes[i].get_xaxis().set_ticks([])

    plt.suptitle(r"Discriminative Projections with eig($\Lambda$)", fontsize='large')
    plt.tight_layout()
    title = 'synth/discreigen'
# plt.savefig(title + '.png', bbox_inches='tight', pad_inches=0.0)
# plt.show()


if DISCR_CHARTING:
    from lvq.GCIALVQ import GCIALVQ

    alpha = 50
    k = 3
    gcialvq = GCIALVQ(model, k=k, alpha=alpha)
    acc = gcialvq.score(samples, labels) * 100

    x = gcialvq.samples_global
    w = gcialvq.w_global
    for label in np.unique(labels):
        coords = x[labels == label]
        coords = coords[ords[label]]
        plt.scatter(coords[:, 0], coords[:, 1], c=colors[label])
    for label in np.unique(labels):
        coords = w[model.c_w_ == label]
        plt.scatter(coords[:, 0], coords[:, 1], edgecolors='black', linewidths=2,
                    # c=colors[label][0:model.prototypes_per_class], marker='^')
                    c=colors[label][len(model.c_w_ == label)], marker='^', s=100)

    if alpha is None:
        plt.title('acc=' + str(round(acc, 2)) + r' ($\nu$=' + 'automatic' + ', $k$=' + str(k) + ')',
                  fontsize='xx-large')
    else:
        plt.title('acc=' + str(round(acc, 2)) + r' ($\nu$=' + str(alpha) + ', $k$=' + str(k) + ')', fontsize='xx-large')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.savefig('synth/chart_pw' + '.png', bbox_inches='tight', pad_inches=0.0)
    # plt.show()

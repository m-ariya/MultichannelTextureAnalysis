import numpy as np
import skimage
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid


def normalize(omega):
    nf = np.sqrt(np.trace(omega.T.dot(omega)))
    omega = omega / nf
    return omega


def lmbda(omega):
    return omega.T.dot(omega)


def remove_diag(matrix):
    np.fill_diagonal(matrix, 0)
    return matrix


def show_cm(model, x, y, display_labels, title='conf. mat'):
    ConfusionMatrixDisplay.from_estimator(model, x, y, labels=model.classes_,
                                          display_labels=display_labels, normalize='true',
                                          cmap='magma', colorbar=False, text_kw={'fontsize': 16})
    plt.title(title)
    plt.tight_layout()
    plt.show()


def show_lambdas(omegas, class_names, title="", cmap="coolwarm", colors=None, features=None):
    n_omegas = len(omegas)
    lmbdas = []
    for omega in omegas:
        lmbdas.append(remove_diag(lmbda(normalize(omega))))
    vmin = min([np.min(l) for l in lmbdas])
    vmax = max([np.max(l) for l in lmbdas])
    shape = lmbdas[0].shape[0]

    fig, axes = plt.subplots(ncols=n_omegas, sharey=True, sharex=True, figsize=(10, 4.5))
    # cbar_ax = fig.add_axes([.91, 0.13, .03, .64])
    cbar_ax = fig.add_axes([.91, 0.145, .03, .61])
    for i in range(n_omegas):
        sns.heatmap(lmbdas[i], ax=axes[i], square=True, cmap=cmap, cbar=i == 0, cbar_ax=None if i else cbar_ax,
                    vmin=vmin, vmax=vmax, center=0)
        col = 'black' if colors is None else colors[i][99]
        axes[i].set_title(class_names[i], fontdict={'color': col, 'fontsize': 'xx-large', 'fontweight': 'bold'})
        if features is None:
            axes[i].get_xaxis().set_visible(False)
            axes[i].get_yaxis().set_visible(False)
        else:
            axes[i].set_xticklabels(features, fontsize='large')
            axes[i].set_yticklabels(features, fontsize='large')

        axes[i].axhline(y=0, color='k', linewidth=1)
        axes[i].axhline(y=shape, color='k', linewidth=1)
        axes[i].axvline(x=0, color='k', linewidth=1)
        axes[i].axvline(x=shape, color='k', linewidth=1)
    fig.suptitle(title, fontsize='xx-large')
    # fig.subplots_adjust(left=0, top=0.90, right=0.90, bottom=0,wspace=0.05)
    fig.subplots_adjust(left=0.05, top=0.90, right=0.90, bottom=0, wspace=0.05)
    # plt.tight_layout()
    plt.show()
    return lmbdas


def show_prototypes(prototypes, patch_size, names, title='', cs=None):
    m = int(patch_size ** 2)
    protos = []
    for p in prototypes:
        r = p[0:m].reshape((patch_size, patch_size))
        g = p[m:2 * m].reshape((patch_size, patch_size))
        b = p[2 * m:3 * m].reshape((patch_size, patch_size))
        if cs is None:
            protos.append(np.dstack([r, g, b]).astype(int))
        else:
            protos.append(cs(np.dstack([r, g, b])))

    fig, axes = plt.subplots(nrows=1, ncols=len(prototypes), figsize=(10, 3))
    for i, ax in enumerate(axes):
        ax.set_title(names[i])
        ax.imshow(protos[i])
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if 0:
    imgs = []
    ds_path = "../data/raw/roses_prototypes/"
    names = ["healthy_green", "healthy_purple", "egg", "mold_green", "mold_purple"]
    for img_id in names:
        imgs.append(skimage.io.imread((ds_path + img_id + ".jpg"), as_gray=False))

    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 3))
    for i, p in enumerate(imgs):
        axes[i].imshow(p)
        axes[i].set_xlabel(names[i], fontsize=15)
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])

    fig.tight_layout(pad=0., w_pad=0., h_pad=0)
    plt.show()

    exit(0)
    names = ['11.jpg', '5.jpg', '9.jpg', 'egg3.jpg']
    # names = ['0010_1659000563.993358_xyz_rpy_[1.80][0.40][2.25]_[0.05][-0.08][-1.71]_exp_1900_15.jpg_1.jpg', '0047_1659000639.0086837_xyz_rpy_[0.78][-1.54][2.24]_[0.05][-0.04][-1.70]_exp_1752_3.jpg_3.jpg', '0028_1659000604.6240604_xyz_rpy_[1.04][-1.64][2.26]_[0.05][-0.06][-1.71]_exp_1721_6.jpg_2.jpg', '0274_1659001031.3850796_xyz_rpy_[-4.75][2.58][2.32]_[0.02][-0.09][-1.74]_exp_1357_3.jpg_1.jpg']
    for img_id in names:
        imgs.append(skimage.io.imread((ds_path + img_id), as_gray=False))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
    plt.setp(axes.flat)

    for ax in axes[0]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    for i, axx in enumerate(axes):
        for j, ax in enumerate(axx):
            ax.imshow(imgs[2 * j + i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    plt.show()

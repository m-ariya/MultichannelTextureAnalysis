
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



from lvq.GIALVQ import *

from utils.preprocessing import *


from utils.visualization import show_lambdas


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

def show_ds(model, samples, labels, colors, ords, proj=2, init_w=[]):
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
    plt.show()

def show_discr_proj(model, samples, labels, colors, ords, eigen_proj=False):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    class_names = ['0', '1', '2']
    plot = None

    for cls in model.classes_:
        unique_labels = np.unique(labels)
        idx = np.where(unique_labels == cls)[0]
        temp = unique_labels[-1]
        unique_labels[-1] = cls
        unique_labels[idx] = temp
        if eigen_proj:
            class_projected_train = model.project_eigen(samples, cls, 2)
            w = model.project_eigen(model.w_, cls, 2)
        else:
            class_projected_train = model.project(samples, model.omegas_[cls])
            w = model.project(model.w_, model.omegas_[cls])

        for label in unique_labels:
            coords = class_projected_train[labels == label]
            coords = coords[ords[label]]
            plot = axes[cls].scatter(coords[:, 0], coords[:, 1], c=colors[label])


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
    plt.show()
def main():
    samples = np.load("data/synthetic/synth_samples.npy")
    labels = np.load("data/synthetic/synth_labels.npy")
    blues = ListedColormap(plt.colormaps['Blues'](np.linspace(0.4, 1, 300)))
    oranges = ListedColormap(plt.colormaps['Oranges'](np.linspace(0.4, 1, 300)))
    purples = ListedColormap(plt.colormaps['Purples'](np.linspace(0.4, 1, 300)))
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

    model = IALVQ(max_iter=200, prototypes_per_class=1, omega_rank=2, seed=11,
                  regularization=0., omega_locality='PW',
                  block_eye=False, norm=False, correct_imbalance=True)
    model.fit(samples, labels)
    print(model.score(samples, labels))
    show_lambdas(model.omegas_, class_names=[0, 1, 2], colors=colors, title=r'$\Lambda=\Omega^T\Omega$')
    show_ds(model ,samples ,labels, colors, ords, proj=2)
    show_discr_proj(model, samples, labels, colors, ords, eigen_proj=False)

    k=2
    alpha=None
    model_global= GIALVQ(model, k=k, alpha=alpha, normalize_omegas=False)
    acc = model_global.score(samples, labels) * 100

    x = model_global.samples_global
    w = model_global.w_global
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
    plt.show()


if __name__ == "__main__":
    main()

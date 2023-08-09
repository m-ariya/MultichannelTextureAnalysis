import matplotlib
import matplotlib.pyplot as plt

from utils.visualization import *

from scipy.special import softmax

from alpha_trees.AlphaTree import AlphaTree


def softmin(distances, sigma):
    return softmax(-distances / sigma ** 2, 1)


def slide_gialvq(orig, p, model, rgb2x=None):
    image = np.copy(orig)
    if rgb2x is not None:
        image = rgb2x(image)
    image = np.pad(image, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='symmetric')
    patches = np.lib.stride_tricks.sliding_window_view(image, window_shape=(p, p), axis=(0, 1))
    patches = patches.reshape(-1, p * p * 3)
    labels, distances = model.predict(patches, return_dist=True)
    probabilities = distances.reshape((distances.shape[0], -1, model.prototypes_per_class)).min(-1)
    probabilities = softmin(probabilities, 0.001)
    probabilities = probabilities[:, 0]  # for now consider probability only for mold
    distances = np.min(distances, axis=1)
    return labels.reshape(orig.shape[0], orig.shape[1]), distances.reshape(orig.shape[0],
                                                                           orig.shape[1]), probabilities.reshape(
        orig.shape[0], orig.shape[1])


def segment_gialvq(img, gialvq):
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(1, 2),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="5%",
                     cbar_pad=0.15
                     )

    grid[0].imshow(img)
    cmap = matplotlib.colors.ListedColormap(["peachpuff", "salmon", "red"])
    imgc = np.copy(img)
    labels_cialvq_charting, d, prob1 = slide_gialvq(img, int(np.sqrt(gialvq.omega_rank)), gialvq)
    mold_pixels_bool_idxs = labels_cialvq_charting == 0
    idxs_eggs = labels_cialvq_charting == 1
    imgc[idxs_eggs, 2] = 255
    imgc[idxs_eggs, 0] = 0
    imgc[idxs_eggs, 1] = 0
    imgc[mold_pixels_bool_idxs, :] = 255 * cmap(prob1[mold_pixels_bool_idxs])[:, 0:3]
    mappbl = grid[1].imshow(imgc, cmap=cmap)

    cbar = grid[1].cax.colorbar(mappbl, cmap=cmap)
    cbar.ax.set_yticks([0, 255])
    cbar.ax.set_yticklabels(['0', '1'])

    grid[0].set_xticks([])
    grid[1].set_xticks([])
    grid[0].set_yticks([])
    grid[1].set_yticks([])
    plt.axis('off')

    grid[0].set_title("Input", fontsize='large')
    grid[1].set_title("GMLVQ (sliding window)", fontsize='large')
    plt.tight_layout()
    plt.show()


def build_informed_tree(gialvq, x_train, img, patch_sz, preset_alphas=False):
    d, iqrs = gialvq.dist_to_protos(x_train, [75, 25])
    a = AlphaTree(img, patch_sz)
    if preset_alphas:
        alphas = iqrs.flatten()
    else:
        alphas = []
    a.build(alphas, gialvq, labels=None, alpha_start=0)
    return a, iqrs


# roses data set specific function (work in progress)
def cut_tree(atree, iqrs, cls, show_mode='default'):
    if type(cls) is int:
        if cls == 0:
            t = iqrs[0, 0, 0]
            s = "(within mold)"
        elif cls == 1:
            t = iqrs[1, 1, 0]
            s = "(within eggs)"
        else:
            t = iqrs[2, 2, 0]
            s = "(within healthy)"
    else:
        i, j, k = cls
        t = iqrs[i, j, k]
        s = str(t)

    if show_mode == 'default':
        fig, ax = plt.subplots()
        cmap = matplotlib.colormaps['twilight']
        res = atree.filter(t)
        u = np.unique(res).astype(int)

        im = ax.imshow(res, interpolation='none', cmap=ListedColormap(cmap(np.linspace(0.15, 0.95, len(u)))))
        plt.title(r"Cut at $Q_3$ %s" % s)
        plt.yticks([])
        plt.xticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad='2%')
        cb = plt.colorbar(im, cmap=cmap, cax=cax)
        cb.set_ticks([0.5, u[-1] - 0.5])
        cb.ax.set_yticklabels([0, len(u) - 1])
        plt.tight_layout()
        plt.show()
        return res
    elif show_mode == 'alpha':
        fig, ax = plt.subplots()
        cmap = matplotlib.colormaps['Greens']
        res = atree.filter2(t)

        im = ax.imshow(np.ln(res), interpolation='none', cmap=cmap)
        plt.title(r"Cut at $Q_3$ %s" % s)
        plt.yticks([])
        plt.xticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad='2%')
        cb = plt.colorbar(im, cmap=cmap, cax=cax)

        plt.tight_layout()
        plt.show()
        return res

    else:
        fig, ax = plt.subplots()
        res, cls_count = atree.filter3(t)

        res = res.astype(int)
        cls_count = cls_count.astype(int)
        u = np.unique(res)

        r = matplotlib.colormaps['Reds']
        b = matplotlib.colormaps['Blues']
        g = matplotlib.colormaps['Greens']
        classes = [r, b, g]
        class_colors = [c(np.linspace(0.3, 1, n)) for c, n in zip(classes, cls_count) if n > 0]
        colors = np.vstack(class_colors)
        cmap = matplotlib.colors.ListedColormap(colors)
        im = ax.imshow(res, cmap=cmap, interpolation='none')
        plt.title(r"Cut at $Q_3$ %s" % s)
        plt.yticks([])
        plt.xticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad='2%')

        cb = plt.colorbar(im, cmap=cmap, cax=cax)

        cb.set_ticks([0.5, u[-1] - 0.5])
        cb.ax.set_yticklabels([0, len(u) - 1])
        plt.tight_layout()
        plt.show()
        return res, cls_count

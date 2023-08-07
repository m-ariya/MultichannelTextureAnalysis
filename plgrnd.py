import os

import matplotlib.colors
import numpy as np
import skimage
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
from numpy import linspace
from scipy.special import softmax
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from alpha_trees.AlphaTree import AlphaTree
from charting import map_globally, chart
from lvq.GCIALVQ import GCIALVQ
from lvq.CIALVQ import CIALVQ
from segmentation import slide_global, paint_per_pixel, softmin, slide_gcialvq
from utils.DataManager import DataManager
from alpha_trees.AlphaTree import CC_TYPE
from matplotlib.colors import ListedColormap, BoundaryNorm
SEGMENT=0
dm = DataManager()
alpha = None
k=3
CLASSIFY=0



def show_all(img):
    fig = plt.figure()
    grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                     nrows_ncols=(2, 2),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="5%",
                     cbar_pad=0.15,
                     cbar_set_cax=True
                     )

    grid[0].imshow(img)
    cmap = matplotlib.colors.ListedColormap(["peachpuff", "salmon", "red"])
    imgc = np.copy(img)
    labels_cialvq_charting, d, prob1 = slide_gcialvq(img, p, gcialvq)
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
    plt.show()
def build_informed_tree(gcialvq, x_train, img, preset_alphas=True):
    d, iqrs = gcialvq.dist_to_protos(x_train, [75, 25])
    a = AlphaTree(img, 7)
    if preset_alphas:
        alphas = iqrs.flatten()
    else:
        alphas= []
    a.build(alphas, gcialvq, labels=None, alpha_start=0)
    return a, iqrs
def cut_tree(atree, iqrs, cls, show_mode='default'):
    if type(cls) is int:
        if cls==0:
            t = iqrs[0, 0, 0]
            s= "(within mold)"
        elif cls==1:
            t = iqrs[1, 1,0]
            s ="(within eggs)"
        else:
            t= iqrs[2, 2, 0]
            s = "(within healthy)"
    else:
        i,j,k=cls
        t=iqrs[i,j,k]
        s = str(t)

    if show_mode == 'default':
        fig, ax = plt.subplots()
        cmap = matplotlib.colormaps['twilight']
        res= atree.filter(t)
        u=np.unique(res).astype(int)
        print(len(u), u)

        im=ax.imshow(res, interpolation='none', cmap=ListedColormap(cmap(np.linspace(0.15, 0.95, len(u)))))
        plt.title(r"Cut at $Q_3$ %s" % s)
        plt.yticks([])
        plt.xticks([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%",pad='2%')
        cb = plt.colorbar(im, cmap=cmap,  cax=cax)
        cb.set_ticks([0.5,u[-1]-0.5])
        cb.ax.set_yticklabels([0, len(u)-1])
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
        print(cls_count)
        res = res.astype(int)
        cls_count = cls_count.astype(int)
        u = np.unique(res)
        print(len(u),u)

        r = matplotlib.colormaps['Reds']
        b = matplotlib.colormaps['Blues']
        g = matplotlib.colormaps['Greens']
        classes = [r, b, g]
        class_colors = [c(np.linspace(0.3, 1, n)) for c,n in zip(classes, cls_count)  if n > 0]
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



class_names=['mold', 'eggs', 'healthy']
ds_name = "roses_3classes"
s = 11
x_test, y_test = dm.load_data(ds_name + str(s), test=True)
x_train, y_train = dm.load_data(ds_name + str(s), test=False)


gcialvq = dm.load_model("QF_s=11_reg=0.0_norm=False_be=False_charted_.pkl")
#cialvq= dm.load_model("QF_s=11_reg=0.0_norm=False_be=False_.pkl")

img1= skimage.io.imread('/home/mariya/Documents/uni/Thesis/datasets/roses_selection/tosegment_test/moldy.jpg')
img= skimage.io.imread('/home/mariya/Documents/uni/Thesis/datasets/roses_selection/tosegment_test/blurry2cut.jpg')
p=7
show_all(img)







if CLASSIFY:
    path= '/home/mariya/Documents/uni/Thesis/datasets/roses_selection/tosegment_test/'

    for img_id in os.listdir(path):
        img = skimage.io.imread(os.path.join(path, img_id))

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
        labels_cialvq_charting, d, prob1 = slide_gcialvq(img, p, gcialvq)
        mold_pixels_bool_idxs = labels_cialvq_charting == 0
        idxs_eggs = labels_cialvq_charting == 1
        imgc[idxs_eggs, 2] = 255
        imgc[idxs_eggs, 0] = 0
        imgc[idxs_eggs, 1] = 0
        imgc[mold_pixels_bool_idxs, :] = 255 * cmap(prob1[mold_pixels_bool_idxs])[:, 0:3]
        mappbl=grid[1].imshow(imgc, cmap=cmap)

        cbar=grid[1].cax.colorbar(mappbl, cmap=cmap)
        cbar.ax.set_yticks([0,255])
        cbar.ax.set_yticklabels(['0', '1'])

        grid[0].set_xticks([])
        grid[1].set_xticks([])
        grid[0].set_yticks([])
        grid[1].set_yticks([])
        plt.axis('off')

        grid[0].set_title("Input", fontsize='large')
        grid[1].set_title("GMLVQ (sliding window)", fontsize='large')
        plt.tight_layout()
        plt.savefig('./segmentation/' + img_id[:-3] +'png', bbox_inches='tight', pad_inches=0.0)
        plt.clf()


if SEGMENT:
    path = '/home/mariya/Documents/uni/Thesis/datasets/roses_selection/tosegment_test/'

    #for img_id in os.listdir(path):
       # if str(img_id) in [ 'no_mold.jpg']:
            #img = skimage.io.imread(os.path.join(path, img_id))
    atree, iqrs = build_informed_tree(gcialvq, x_train,img, preset_alphas=False)
    res=cut_tree(atree, iqrs, cls=0, show_mode='default')
            #plt.savefig('./segmentation/_within_eggs_no_preset_' + img_id[:-3] + 'png', bbox_inches='tight', pad_inches=0.0)
            #plt.clf()






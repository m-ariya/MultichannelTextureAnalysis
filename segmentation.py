import numpy as np
import skimage.color.colorconv
from matplotlib import pyplot as plt
from scipy.special import softmax

from charting import map_globally
from lvq import CIALVQ
from utils.DataManager import DataManager


def softmin(distances, sigma):
    return softmax(-distances/sigma**2,1)


def slide(orig,  p, model, rgb2x=None):
    image = np.copy(orig)
    if rgb2x is not None:
        image = rgb2x(image)
    image = np.pad(image, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='symmetric')
    patches = np.lib.stride_tricks.sliding_window_view(image, window_shape=(p, p), axis=(0, 1))
    patches = patches.reshape(-1, p * p * 3)
    labels, distances = model.predict(patches, True)
    return labels.reshape(orig.shape[0], orig.shape[1]), distances.reshape(orig.shape[0], orig.shape[1])


def slide_gcialvq(orig,  p, model, rgb2x=None):
    image = np.copy(orig)
    if rgb2x is not None:
        image = rgb2x(image)
    image = np.pad(image, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='symmetric')
    patches = np.lib.stride_tricks.sliding_window_view(image, window_shape=(p, p), axis=(0, 1))
    patches = patches.reshape(-1, p * p * 3)
    #patches = model.project(patches)
    labels, distances = model.predict(patches, True)

    probabilities = distances.reshape((distances.shape[0],-1,model.prototypes_per_class)).min(-1) # sum distances per class
    probabilities = softmin(probabilities,0.001)#softmax(-probabilities,1) # negate, since smaller values should have the largest probabilities
    probabilities = probabilities[:,0] # for now consider probability only for mold
    distances=np.min(distances,axis=1)
    return labels.reshape(orig.shape[0], orig.shape[1]), distances.reshape(orig.shape[0], orig.shape[1]), probabilities.reshape(orig.shape[0], orig.shape[1])

def slide_global(orig,  p, model, V, a, k,rgb2x=None):
    image = np.copy(orig)
    if rgb2x is not None:
        image = rgb2x(image)
    image = np.pad(image, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='symmetric')
    patches = np.lib.stride_tricks.sliding_window_view(image, window_shape=(p, p), axis=(0, 1))
    patches = patches.reshape(-1, p * p * 3)
    patches = map_globally(model, V, patches, a, k)
    w_global = map_globally(model, V, model.w_, a, k)
    labels, distances = model.global_predict(patches, w_global, True)

    probabilities = distances.reshape((distances.shape[0],-1,model.prototypes_per_class)).min(-1) # sum distances per class
    probabilities = softmin(probabilities,0.001)#softmax(-probabilities,1) # negate, since smaller values should have the largest probabilities
    probabilities = probabilities[:,0] # for now consider probability only for mold
    distances=np.min(distances,axis=1)
    return labels.reshape(orig.shape[0], orig.shape[1]), distances.reshape(orig.shape[0], orig.shape[1]), probabilities.reshape(orig.shape[0], orig.shape[1])
def paint_per_pixel(orig,labels,target=0,title="", show=True, save=False,img_name="",target2=1,dists=None):
    img = np.copy(orig)
    idxs_mold = labels == target
    idxs_eggs = labels == target2

    if not show and not save:
        img[idxs_mold, 0] = 255
        img[idxs_mold, 1] = 0
        img[idxs_mold, 2] = 0

        img[idxs_eggs, 2] = 255
        img[idxs_eggs, 0] = 0
        img[idxs_eggs, 1] = 0
        return img
    f, axarr = plt.subplots(1, 2, figsize=(10, 5))
    axarr[0].imshow(img)
    img[idxs_mold, 0] = 255
    img[idxs_eggs, 2] = 255
    img[idxs_eggs, 0] = 0
    img[idxs_eggs, 1] = 0
    return img
    axarr[1].imshow(img)
    axarr[0].get_yaxis().set_ticks([])
    axarr[1].get_yaxis().set_ticks([])
    axarr[0].get_xaxis().set_ticks([])
    axarr[1].get_xaxis().set_ticks([])
    plt.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig( title+ "_slided_stride1_" + img_name)
    if show:
        plt.show()
    plt.clf()
import os

import numpy as np

#from lvq import IAALVQ, GIALVQ
#from lvq.IALVQ import IALVQ
import pickle
import skimage.io
from skimage.transform import resize




def save_model( model, extra=''):
    print(type(model))
    d= 'QF'
    if type(model) is IAALVQ.IAALVQ:
        d= "A" + str(model.beta)
    name = d+ "_s=" + str(model.seed) + "_reg=" + str(model.regularization) + "_norm=" + str(model.norm) + "_be=" + str(model.block_eye) + "_"
    if type(model) is GIALVQ.GIALVQ:
        name = name + "charted_"
    with open("./models/" + name + extra +".pkl", "wb") as f:
        pickle.dump(model, f)

def load_model(name):
    with open("./models/" + name, "rb") as f:
        return pickle.load(f)

def load_images(ds_path, rescale_shape=(128, 128)):
    imgs = []
    labels = []
    for cl, class_dir in enumerate(os.listdir(ds_path)):
        for img_id in os.listdir(os.path.join(ds_path, class_dir)):
            img = skimage.io.imread(os.path.join(ds_path, class_dir, img_id))
            img = resize(img, rescale_shape)
            imgs.append(img)
            labels.append(cl)
    return imgs, labels

def save_preprocessed_data( preprocessed, labels, ds_name, test):
    if test:
        ds_path = os.path.join(".", "data/preprocessed", ds_name, "test")
    else:
        ds_path = os.path.join(".", "data/preprocessed", ds_name, "train")
    os.mkdir(ds_path)
    np.save(os.path.join(ds_path, "samples.npy"), preprocessed)
    np.save(os.path.join(ds_path, "labels.npy"), labels)

def load_data( ds_path, test=False):
    if test:
        samples = np.load(os.path.join("data/preprocessed", ds_path, "test/samples.npy"))
        labels = np.load(os.path.join("data/preprocessed", ds_path, "test/labels.npy"))
    else:
        samples = np.load(os.path.join("data/preprocessed", ds_path, "train/samples.npy"))
        labels = np.load(os.path.join("data/preprocessed", ds_path, "train/labels.npy"))
    return samples, labels
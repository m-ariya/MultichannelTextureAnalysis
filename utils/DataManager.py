import os
import pickle

import numpy as np
import skimage.io
from numpy import copy

from skimage.transform import resize

from sklearn.feature_extraction import image as sk_img

from lvq import CIAALVQ, GCIALVQ
from lvq.CIALVQ import CIALVQ
from utils.visualization import show_prototypes


class DataManager:

    def preprocess(self, ds_train_path, ds_test_path=None, resize_to=False, resize_shape=(128, 128),
                   extract_patches=True, patch_size=15,
                   n_patches=200, rnd_state=11, as_gray=False, ds_name="data"):
        """
        Off-line preprocessor for raw RGB images
        :param ds_test_path: path to the test data (each class should be in separate folder)
        :param ds_train_path: path to the training data (each class should be in separate folder)
        :param n_patches: the number of patches per image (if applicable)
        :param patch_size: the size of a patch (if applicable)
        :param extract_patches: if True random patches are extracted from the image
        :return: saves the processed data and labels in two separate .npy files at "./data/processed/ds"
        """
        os.mkdir(os.path.join("data", ds_name))
        class_names = []
        with open(os.path.join(".", "data", ds_name, 'info.txt'), 'w') as f:
            f.write('Name: %s\n' % ds_name)
            if extract_patches:
                f.write('Patches per image: %d\n' % n_patches)
                f.write('Patch size: %d\n' % patch_size)
                f.write('Random state: %d\n' % rnd_state)
        for test, ds_path in enumerate([ds_train_path, ds_test_path]):
            preprocessed_ = []
            labels_ = []
            label = 0
            for class_dir in os.listdir(ds_path):
                if not test:
                    class_names.append(class_dir)
                for img_id in os.listdir(os.path.join(ds_path, class_dir)):
                    img = skimage.io.imread(os.path.join(ds_path, class_dir, img_id), as_gray=as_gray)
                    if resize_to:
                        img = resize(img, resize_shape)
                    if extract_patches:
                        patches = sk_img.extract_patches_2d(img, (patch_size, patch_size), max_patches=n_patches,
                                                            random_state=rnd_state)
                        for patch in patches:
                            vectorized = self.vectorize(patch, as_gray)
                            preprocessed_.append(vectorized)
                            labels_.append(label)
                    else:
                        vectorized = self.vectorize(img, as_gray)
                        preprocessed_.append(vectorized)
                        labels_.append(label)
                label = label + 1
            preprocessed = np.array(preprocessed_)
            labels = np.array(labels_)
            self.save_preprocessed_data(preprocessed, labels, ds_name, test)
            with open(os.path.join(".", "data", ds_name, 'info.txt'), 'a') as f:
                if not test:
                    f.write('Number of classes: %d\n' % len(np.unique(labels)))
                    for c in range(len(class_names)):
                        f.write('{%d : %s}\n' % (c, class_names[c]))
                    f.write('Train size: %d\n' % len(labels))
                else:
                    f.write('Test size: %d\n' % len(labels))
            if ds_test_path is None:
                return

    def save_preprocessed_data(self, preprocessed, labels, ds_name, test):
        if test:
            ds_path = os.path.join(".", "data", ds_name, "test")
        else:
            ds_path = os.path.join(".", "data", ds_name, "train")
        os.mkdir(ds_path)
        np.save(os.path.join(ds_path, "samples.npy"), preprocessed)
        np.save(os.path.join(ds_path, "labels.npy"), labels)

    def load_data(self, ds_path, test=False):
        if test:
            samples = np.load(os.path.join("data", ds_path, "test/samples.npy"))
            labels = np.load(os.path.join("data", ds_path, "test/labels.npy"))
        else:
            samples = np.load(os.path.join("data", ds_path, "train/samples.npy"))
            labels = np.load(os.path.join("data", ds_path, "train/labels.npy"))
        self.patch_size = int(np.sqrt(samples.shape[1] // 3))
        return samples, labels

    def load_images(self, ds_path, rescale_shape=(128, 128)):
        imgs = []
        labels = []
        for cl, class_dir in enumerate(os.listdir(ds_path)):
            for img_id in os.listdir(os.path.join(ds_path, class_dir)):
                img = skimage.io.imread(os.path.join(ds_path, class_dir, img_id))
                img = resize(img, rescale_shape)
                imgs.append(img)
                labels.append(cl)
        return imgs, labels

    def vectorize(self, img, as_gray=False):
        """
        Off-line preprocessing step
        :param img: pxpx3 image in Fourier/spatial domain
        :return: v
        """
        if not as_gray:
            img = img.transpose(2, 0, 1)
        return img.flatten()

    def preproces_satimage(self, data, train=True):
        h = 4
        n = 3 * 3
        samples = data[:, 0:data.shape[1] - 1]
        labels = data[:, data.shape[1] - 1:]
        new_labels = copy(labels)
        d = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 7: 5}
        for k, v in d.items(): new_labels[labels == k] = v

        col_idx_ord = []
        for band in range(h):
            for i in range(n):
                col_idx_ord.append(band + i * h)
        samples = (samples[:, col_idx_ord])
        if train:
            np.save("data/satimage/samples_train.npy", samples)
            np.save("data/satimage/labels_train.npy", new_labels.ravel().astype(int))
        else:
            np.save("data/satimage/samples_test.npy", samples)
            np.save("data/satimage/labels_test.npy", new_labels.ravel().astype(int))


    def preprocess_roses(self, ds_train_path, ds_test_path=None,
                   extract_patches=True, patch_size=15,
                   n_patches=200, rnd_state=11, as_gray=False, ds_name="data", rgb2x=None):
        """
        extracts only if img is not of the patch size
        """
        os.mkdir(os.path.join("data", ds_name))
        class_names = []
        with open(os.path.join(".", "data", ds_name, 'info.txt'), 'w') as f:
            f.write('Name: %s\n' % ds_name)
            if rgb2x is None:
                f.write('Colour space: rgb\n')
            else:
                f.write('Colour space: %s\n' % str(rgb2x))
            if extract_patches:
                f.write('Patch size: %d\n' % patch_size)
                f.write('Random state: %d\n' % rnd_state)
        for test, ds_path in enumerate([ds_train_path, ds_test_path]):
            preprocessed_ = []
            labels_ = []
            label = 0
            for class_dir in os.listdir(ds_path):
                if not test:
                    class_names.append(class_dir)
                for img_id in os.listdir(os.path.join(ds_path, class_dir)):
                    img = skimage.io.imread(os.path.join(ds_path, class_dir, img_id), as_gray=as_gray)

                    if rgb2x is not None:
                        img = rgb2x(img)

                    if img.shape[0] != patch_size and img.shape[1] !=patch_size:
                        patches = sk_img.extract_patches_2d(img, (patch_size, patch_size), max_patches=n_patches,
                                                            random_state=rnd_state)
                        for patch in patches:
                            vectorized = self.vectorize(patch, as_gray)
                            preprocessed_.append(vectorized)
                            labels_.append(label)
                    else:
                        vectorized = self.vectorize(img, as_gray)
                        preprocessed_.append(vectorized)
                        labels_.append(label)
                label = label + 1
            preprocessed = np.array(preprocessed_)
            labels = np.array(labels_)
            cl_distr=np.bincount(labels)
            self.save_preprocessed_data(preprocessed, labels, ds_name, test)
            with open(os.path.join(".", "data", ds_name, 'info.txt'), 'a') as f:
                if not test:
                    f.write('Number of classes: %d\n' % len(np.unique(labels)))
                    for c in range(len(class_names)):
                        f.write('{%d : %s (%d train patches)}\n' % (c, class_names[c], cl_distr[c]))
                else:
                    for c in range(len(class_names)):
                        f.write('{%d : %s (%d test patches)}\n' % (c, class_names[c], cl_distr[c]))
            if ds_test_path is None:
                return

    def prep_roses_protos(self, ds_path, rgb2x=None):
            protos = []
            for img_id in os.listdir(ds_path):
                if img_id[-3:] == 'jpg':
                    img = skimage.io.imread(os.path.join(ds_path, img_id), as_gray=False)
                    if rgb2x is not None:
                        img = rgb2x(img)
                    vectorized = self.vectorize(img, False)
                    if img_id in [ 'mold_green.jpg', 'mold_purple.jpg']:
                        label = 0
                    else:
                        label = 1
                    vectorized = np.append(vectorized, label)
                    protos.append(vectorized)
            np.save(ds_path + "/prototypes_" + str(rgb2x.__name__)[4:] + ".npy", protos)


    def prep_roses_prototypes_cm(self, ds_path, patch_size,n_patches, rnd_state,  rgb2x=None):
        prototypes = []
        labels=[]
        rng = np.random.RandomState(rnd_state)
        for proto_dir in os.listdir(ds_path):
            if proto_dir in ['green_mold', 'purple_mold']:
                label=0
            elif proto_dir in ['green_eggs', 'purple_eggs']:
                label=1
            elif proto_dir in ['green_healthy', 'purple_healthy']:
                label =2
            else:
                print("error")
                return
            labels.append(label)
            preprocessed_ = []
            for img_id in os.listdir(os.path.join(ds_path, proto_dir)):
                img = skimage.io.imread(os.path.join(ds_path, proto_dir, img_id))
                if rgb2x is not None:
                    img = rgb2x(img)
                if img.shape[0] != patch_size and img.shape[1] !=patch_size:
                    patches = sk_img.extract_patches_2d(img, (patch_size, patch_size), max_patches=n_patches,
                                                        random_state=rnd_state)
                    for patch in patches:
                        vectorized = self.vectorize(patch)
                        preprocessed_.append(vectorized)
                else:
                    vectorized = self.vectorize(img)
                    preprocessed_.append(vectorized)
            print(label, len(preprocessed_))
            proto = np.mean(preprocessed_, 0) + (rng.rand(vectorized.shape[0]) * 2 - 1)
            proto = np.append(proto, label)
            prototypes.append(proto)
        idxs=np.argsort(labels)
        print(idxs)
        prototypes=np.array(prototypes)[idxs]
        labels=np.array(labels)[idxs]
        names= ['mold', 'mold', 'egg', 'egg', 'healthy', 'healthy']
        #show_prototypes(prototypes, 7, names)
        np.save(ds_path + "/prototypes_cm_.npy",np.array(prototypes))
        return prototypes


    def save_model(self, model, extra=''):
        print(type(model))
        d= 'QF'
        if type(model) is CIAALVQ.CIAALVQ:
            d= "A" + str(model.beta)
        name = d+ "_s=" + str(model.seed) + "_reg=" + str(model.regularization) + "_norm=" + str(model.norm) + "_be=" + str(model.block_eye) + "_"
        if type(model) is GCIALVQ.GCIALVQ:
            name = name + "charted_"
        with open("./models/" + name + extra +".pkl", "wb") as f:
            pickle.dump(model, f)

    def load_model(self, name):
        with open("./models/" + name, "rb") as f:
            return pickle.load(f)


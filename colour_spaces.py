import os

import matplotlib.pyplot as plt
import skimage.color
import segmentation
from lvq.CIAALVQ import CIAALVQ
from  segmentation import *
from lvq.CIALVQ import *
from utils.ResultsManager import ResultsManager, show_cm
from utils.visualization import show_lambdas

AVG_ACC = 0
SEGMENT =1
CM=0
LAMBDAS=0
dm = DataManager()


fs=[None,skimage.color.rgb2hsv,  skimage.color.rgb2luv, skimage.color.rgb2ycbcr,   skimage.color.rgb2yuv]

if AVG_ACC:
    dm = DataManager()
    seeds = [11]#[7,11,25,33,1984]
    for cs in fs:
        scores_train = []
        scores_test=[]
        for s in seeds:
            ds_name = "roses_3classes" + str(s)
            if cs is not None:
                ds_name = ds_name + '_' + cs.__name__[4:]
                #dm.preprocess_roses("data/raw/" + 'roses_3classes' + "/train", "data/raw/" + 'roses_3classes' + "/test",
                                    #True, 7, 50, rnd_state=s,
                                    #as_gray=False, ds_name=ds_name, rgb2x=cs)


            x_test, y_test = dm.load_data(ds_name, test=True)
            x_train, y_train = dm.load_data(ds_name, test=False)
            cialvq = CIAALVQ(max_iter=200, prototypes_per_class=2, omega_rank=7 * 7, seed=s,
                            regularization=0.01, omega_locality='CW',
                            block_eye=False, norm=True, correct_imbalance=True,beta=0.1)
            cialvq.fit(x_train, y_train)
            if s==11:
                if cs is not None:
                    dm.save_model(cialvq, extra="rand_2k_3classes" + cs.__name__[4:] )
                else:
                    dm.save_model(cialvq, extra="rand_2k_3classes")
            scores_train.append(cialvq.score(x_train, y_train))
            scores_test.append(cialvq.score(x_test, y_test))
        if cs is not None:
            print(cs.__name__[4:])
        else:
            print("rgb")
        print("train: ", np.mean(scores_train), np.std(scores_train, ddof=1))
        print("test: ",  np.mean(scores_test), np.std(scores_test, ddof=1))
        print("\n")




if CM:
    s=11
    for cs in fs:
        n='rgb'
        ds_name = "roses_3classes" + str(s)
        if cs is None:
            w = np.load("data/raw/roses_prototypes/prototypes.npy")
        else:
            w = np.load("data/raw/roses_prototypes/prototypes_" + cs.__name__[4:] + '.npy')
        name = 'A0.1_s=11_reg=0.0_norm=False_be=False_rand_2k_'
        if cs is not None:
            name = name  + cs.__name__[4:]
            n=cs.__name__[4:]
            ds_name = ds_name + '_' + cs.__name__[4:]
            print(cs.__name__[4:])

        x_test, y_test = dm.load_data(ds_name, test=True)
        x_train, y_train = dm.load_data(ds_name, test=False)
        cialvq = dm.load_model(name + '.pkl')

        show_cm(cialvq, x_test, y_test, display_labels=['mold', 'healthy'], title='Test ' +n )
        plt.savefig("cm_" + n)
        plt.clf()

if LAMBDAS:
    s = 11
    for cs in fs:
        n = 'rgb'
        ds_name = "roses" + str(s)

        name = 'A0.1_s=11_reg=0.0_norm=False_be=False_rand_2k_'
        if cs is not None:
            name = name + cs.__name__[4:]
            n = cs.__name__[4:]
            ds_name = ds_name + '_' + cs.__name__[4:]
            print(cs.__name__[4:])


        cialvq = dm.load_model(name + '.pkl')


        show_lambdas(cialvq.omegas_, class_names=["Mold", "Healthy"],  title= n)
        plt.savefig("lambdas_" + n)
        plt.clf()



if SEGMENT:
    fs = [None, skimage.color.rgb2ycbcr, skimage.color.rgb2hsv, skimage.color.rgb2luv,  skimage.color.rgb2yuv]
    imgs = []
    s = 11
    p = 7
    dm = DataManager()
    dir = '/home/mariya/Documents/uni/Thesis/datasets/roses_selection/tosegment_test/'
    for img_name in os.listdir(dir):
        for cs in fs:
            name = 'A0.1_s=11_reg=0.01_norm=True_be=False_rand_2k_3classes'

            if cs is not None:
                name = name + cs.__name__[4:]
                print(cs.__name__[4:])
            cialvq = dm.load_model(name + '.pkl')


            orig = skimage.io.imread(dir + img_name)

            labels,_ = segmentation.slide(orig,1, p, cialvq, cs)
            slided = segmentation.paint_per_pixel(orig, labels, 0, title="", show=False, save=False, target2=1)
            imgs.append(slided)

        print(img_name)
        cols = ["rgb"]
        for cs in fs[1:]:
            cols.append(cs.__name__[4:])

        fig, axes = plt.subplots(nrows=1, ncols=len(cols) + 1, figsize=(30, 4.8))

        i = 0
        for ax in axes:
            if i == 0:
                ax.set_title("Original", fontsize=20)
                ax.imshow(orig)
                ax.get_yaxis().set_ticks([])
                ax.get_xaxis().set_ticks([])
            else:
                ax.set_title(cols[i - 1], fontsize=20)
                ax.imshow(imgs[i - 1])
                ax.get_yaxis().set_ticks([])
                ax.get_xaxis().set_ticks([])
            i += 1

        #plt.suptitle("Quadratic Form")
        fig.tight_layout()
        fig.savefig("A0.1_reg=0.01_norm=True_colourspaces_" + img_name[:-4] +".png")
        plt.clf()
        img = None
        orig = None
        imgs = []

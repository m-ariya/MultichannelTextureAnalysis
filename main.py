import pickle

import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from skimage.color import rgb2luv, rgb2ycbcr, rgb2yiq, rgb2ydbdr, rgb2ypbpr
from sklearn.preprocessing import StandardScaler

import segmentation
import utils.ResultsManager
from alpha_trees.AlphaTree import *
from lvq.CIAALVQ import *
from lvq.CIALVQ import *
from utils.DataManager import *
from utils.visualization import *
from charting import *

LOAD_MODEL = 1
TRAIN_MODEL = 0
SEGMENT_IMG = 0
CONF_MAT = 1
PREP_DATA = 0
PREP_PROT = 0
GET_SCORE = 1
LOAD_DATA = 1 #if (TRAIN_MODEL or GET_SCORE) else 0
OPT=0
SHOW_LAMBDAS=0
BETA_OPT=0
TREES=0
SLIDE=0

dm = DataManager()
ds_name="roses_3classes"
s=11
if LOAD_DATA:
    x_test, y_test=dm.load_data(ds_name + str(s), test=True)
    x_train, y_train=dm.load_data(ds_name + str(s), test=False)


if OPT:
    dm = DataManager()
    ds_name = "roses_3classes"
    seeds = [7, 11, 25, 33, 1984]
    reg = [0.0, 0.0001, 0.001, 0.01, 0.1]
    norms = [True, False]
    bes=[True, False]
    for r in reg:
        for norm in norms:
            for be in bes:
                scores_train = []
                scores_test = []
                for s in seeds:
                    x_test, y_test = dm.load_data(ds_name + str(s), test=True)
                    x_train, y_train = dm.load_data(ds_name + str(s), test=False)
                    cialvq = CIAALVQ(max_iter=200, prototypes_per_class=2, omega_rank=7 * 7, seed=s,
                                     regularization=r, omega_locality='CW',
                                     block_eye=be, norm=norm, correct_imbalance=True,beta=0.1)
                    cialvq.fit(x_train, y_train)
                    #if s == 11:
                        #dm.save_model(cialvq, extra="rand_2k_3classes")
                    scores_train.append(cialvq.score(x_train, y_train))
                    scores_test.append(cialvq.score(x_test, y_test))
                print("---reg=%f; norm=%d; be=%d---" % (r, norm, be))
                print("train: ", np.mean(scores_train), np.std(scores_train, ddof=1))
                print("test: ", np.mean(scores_test), np.std(scores_test, ddof=1))


if BETA_OPT:
    dm=DataManager()
    ds_name="roses_3classes"
    seeds = [7,11,25,33, 1984]
    betas=[0.1,1,2,3,4]
    for b in betas:
        scores_train=[]
        scores_test=[]
        for s in seeds:
            x_test, y_test=dm.load_data(ds_name + str(s), test=True)
            x_train, y_train=dm.load_data(ds_name + str(s), test=False)
            cialvq = CIAALVQ(max_iter=200, prototypes_per_class=2, omega_rank=7 * 7, seed=s,
                            regularization=0., omega_locality='CW',
                            block_eye=False, norm=False, correct_imbalance=True,beta=b)
            cialvq.fit(x_train, y_train)
            if s==11:
                dm.save_model(cialvq, extra="rand_2k_3classes")
            scores_train.append(cialvq.score(x_train, y_train))
            scores_test.append(cialvq.score(x_test, y_test))
        print("---beta=%f---" % b)
        print("train: ", np.mean(scores_train), np.std(scores_train, ddof=1))
        print("test: ", np.mean(scores_test), np.std(scores_test, ddof=1))



if PREP_DATA:
    dm.preprocess_roses("data/raw/"+ds_name+"/train", "data/raw/"+ds_name+"/test", True, 7, 50, rnd_state=s, as_gray=False, ds_name=ds_name+str(s), rgb2x=None)
if PREP_PROT:
    dm.prep_roses_protos("data/raw/roses_prototypes")



if TRAIN_MODEL:

    x_test, y_test = dm.load_data(ds_name + str(s), test=True)
    x_train, y_train = dm.load_data(ds_name + str(s), test=False)
    cialvq = CIALVQ(max_iter=200, prototypes_per_class=1, omega_rank=7*7, seed=s,
                                regularization=0., omega_locality='CW',
                                block_eye=False, norm=True,correct_imbalance=True)
    cialvq.fit(x_train, y_train)
    dm.save_model(cialvq, "rand_1k_3classes")

if LOAD_MODEL:

    cialvq = dm.load_model('QF_s=11_reg=0.0_norm=False_be=False_rand_2k_3classes.pkl')





if SLIDE:
    p=7
    cialvqA = dm.load_model('A0.1_s=11_reg=0.0_norm=False_be=False_rand_2k_3classes.pkl')
    cialvqQF = dm.load_model('QF_s=11_reg=0.0_norm=False_be=False_rand_2k_3classes.pkl')
    img_name = 'blurry2'
    img = skimage.io.imread(
        '/home/mariya/Documents/uni/Thesis/datasets/roses_selection/tosegment_test/'+img_name+'.jpg')

    if 0:
        l, da = segmentation.slide(img, p=7,  model=cialvqA)
        l, dqf= segmentation.slide(img, p=7, model=cialvqQF)
        f, axarr = plt.subplots(1, 3, figsize=(15, 5))
        axarr[0].imshow(img)
        axarr[1].imshow(da)
        axarr[2].imshow(dqf)

        axarr[0].get_yaxis().set_ticks([])
        axarr[1].get_yaxis().set_ticks([])
        axarr[2].get_yaxis().set_ticks([])

        axarr[0].get_xaxis().set_ticks([])
        axarr[1].get_xaxis().set_ticks([])
        axarr[2].get_xaxis().set_ticks([])

        axarr[0].set_title("Original")
        axarr[1].set_title("Angle")
        axarr[2].set_title("QF")


        plt.suptitle("Distances to the closest prototypes")
        plt.tight_layout()
        plt.show()

   # np.save("blurry2_labels.npy", l)

    #segmentation.paint_per_pixel(img, l, 0, title="reg01_be1", show=True, save=False, target2=1, dists=d,
     #                            img_name=img_name)


    if 1:
        #l, da = segmentation.slide(img, p=7, model=cialvqA)
        l=np.load("blurry2_labels_A.npy")
        a = AlphaTree(img, patch_sz=7)
        alphas = [0.00001, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.05,0.5,1]
        a.build(alphas=[], model=cialvqA,labels=l, alpha_start=0)



    if 1:
        f, axarr = plt.subplots(1, 3, figsize=(15, 5))
        res=a.filter(0.97)
        axarr[0].imshow(a.filter(0.0001))
        axarr[1].imshow(res)
        axarr[2].imshow(a.filter(0.0005))

        axarr[0].get_yaxis().set_ticks([])
        axarr[1].get_yaxis().set_ticks([])
        axarr[2].get_yaxis().set_ticks([])

        axarr[0].get_xaxis().set_ticks([])
        axarr[1].get_xaxis().set_ticks([])
        axarr[2].get_xaxis().set_ticks([])

        axarr[0].set_title("Cut at 0.0001")
        axarr[1].set_title("Cut at 0.0003")
        axarr[2].set_title("Cut at 0.0005")
        plt.suptitle("Angle, alphas={0.00001, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.05,0.5}")
        plt.show()

if GET_SCORE and (LOAD_MODEL or TRAIN_MODEL):
    print(cialvq.score(x_train, y_train))
    print(cialvq.score(x_test, y_test))

if SEGMENT_IMG and (LOAD_MODEL or TRAIN_MODEL):
    dm = DataManager()



if CONF_MAT and (LOAD_MODEL or TRAIN_MODEL):
    utils.ResultsManager.show_cm(cialvq,x_train, y_train, display_labels=['mold','eggs', 'healthy'],title='Train')
    utils.ResultsManager.show_cm(cialvq,x_test, y_test, display_labels=['mold', 'eggs', 'healthy'],title='Test')


if SHOW_LAMBDAS and (LOAD_MODEL or TRAIN_MODEL):
    show_lambdas(cialvq.omegas_, class_names=["Mold", "Healthy"])


if TREES:

    img_toseg = skimage.io.imread('/home/mariya/Documents/uni/Thesis/datasets/roses_selection/tosegment_test/moldy2.jpg')










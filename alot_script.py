from utils.DataManager import DataManager
from lvq.CIAALVQ import CIAALVQ
from lvq.CIALVQ import *
dm = DataManager()

seeds = [1984, 11, 33]
patch_size = 15
ds = "bread2"
if 0:
    n_patches = 200
    for s in seeds:
        dm.preprocess("data/raw/" + ds + "/train", "./data/raw/" + ds + "/test", extract_patches=True, patch_size=patch_size, n_patches=n_patches, rnd_state=s, ds_name=ds+ str(s), resize_to=True, resize_shape=(128, 128))


max_iter = 200
beta = 4
wn = 2
reg = 0.0
norm = False
be = False
omega_rank = 225

for s in [11, 33,1984]:
    ds_name=ds+str(s)
    x_train, y_train = dm.load_data(ds_name)
    x_val, y_val = dm.load_data(ds_name, test=True)

    cialvq = CIAALVQ(max_iter=max_iter, prototypes_per_class=wn, omega_rank=omega_rank, seed=s,
                                 regularization=reg, omega_locality='CW',
                                 beta=beta,
                                 block_eye=be, norm=norm, correct_imbalance=False)


    cialvq.fit(x_train, y_train)
    #dm.save_model(cialvq, extra=ds_name)
    print(cialvq.score(x_train, y_train))
    print(cialvq.score(x_val, y_val))
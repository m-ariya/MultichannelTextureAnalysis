from lvq.GIALVQ import GIALVQ
from lvq.IALVQ import IALVQ

from utils.io_management import *
from utils.preprocessing import *
from utils.segmentation import build_informed_tree, cut_tree, segment_gialvq
from utils.visualization import *


def main():
    ds_name = "roses_3classes"
    s = 11  # seed

    x_test, y_test = load_data(ds_name + str(s), test=True)
    x_train, y_train = load_data(ds_name + str(s), test=False)
    model = IALVQ(max_iter=200, prototypes_per_class=2, omega_rank=7 * 7, seed=s,
                  regularization=0., omega_locality='CW',
                  block_eye=False, norm=False, correct_imbalance=True)
    print("Training...")
    model.fit(x_train, y_train)

    print("Train accuracy: ", model.score(x_train, y_train))
    print("Test accuracy: ", model.score(x_test, y_test))

    # confusion matrices
    show_cm(model, x_train, y_train, display_labels=['mold', 'eggs', 'healthy'], title='Train')
    show_cm(model, x_test, y_test, display_labels=['mold', 'eggs', 'healthy'], title='Test')

    # feature correlation matrices
    show_lambdas(model.omegas_, class_names=['mold', 'eggs', 'healthy'])

    # globalizing model:
    print("Charting...")
    model_global = GIALVQ(model)
    print("Train accuracy: ", model_global.score(x_train, y_train))
    print("Test accuracy: ", model_global.score(x_test, y_test))

    # segmentation
    img = skimage.io.imread('data/to_segment/blurry.jpg')

    print("Segmentation...")
    atree, iqrs = build_informed_tree(model_global, x_train, img, patch_sz=int(np.sqrt(model_global.omega_rank)))
    cut_tree(atree, iqrs, cls=0)

    print("Segmentation (GIALVQ only)...")
    segment_gialvq(img, model_global)


if __name__ == "__main__":
    main()

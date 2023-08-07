import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def show_cm(model, x, y, display_labels,title='conf. mat'):
    ConfusionMatrixDisplay.from_estimator(model, x, y, labels=model.classes_,
                                          display_labels=display_labels, normalize='true',
                                          cmap='magma', colorbar=False,  text_kw={'fontsize':16})
    plt.title(title)
    plt.tight_layout()
    plt.show()

class ResultsManager:

    def __init__(self, ds_name, conf_mat=False, save_protos=False, save_omegas=False, cv=False):
        self.ds_name = ds_name
        self.conf_mat = conf_mat
        self.save_protos = save_protos
        self.save_omegas = save_omegas
        self.cv = cv
        self.train = []
        self.test = []
        path = "./models/" + ds_name
        if not os.path.exists(path):
            os.makedirs(path)
            os.mkdir(path + "/omegas")
            os.mkdir(path + "/l_curves")
            os.mkdir(path + "/prototypes")
            os.mkdir(path + "/cms")
            os.mkdir(path + "/results")
        self.path = path + "/"

    def results(self, x_train, y_train, x_test, y_test, model, model_name, display_labels=[]):
        train_acc = model.score(x_train, y_train)
        test_acc = model.score(x_test, y_test)

        print('classification accuracy train: %f\n' % train_acc)
        print('classification accuracy test: %f' % test_acc)

        self.train.append(train_acc)
        self.test.append(test_acc)

        with open(self.path + "results/" + model_name + '_result.txt', 'w') as f:
            f.write('classification accuracy train: %f\n' % train_acc)
            f.write('classification accuracy test: %f' % test_acc)

        if self.conf_mat:
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, labels=model.classes_,
                                                  display_labels=display_labels, normalize='true',
                                                  cmap='magma', colorbar=True)
            plt.title("Confusion Matrix")
            plt.savefig(self.path + "cms/" + model_name + ".png")
            plt.clf()

        if self.save_omegas:
            np.save(self.path + "omegas/" + model_name, model.omegas_)

        if self.save_protos:
            np.save(self.path + "prototypes/" + model_name, model.w_)

    def cv_results(self):
        if self.cv:
            print("Train accuracy = %f, std = %f" % (np.mean(self.train), np.std(self.train, ddof=1)))
            print("Test accuracy = %f, std = %f" % (np.mean(self.test), np.std(self.test, ddof=1)))

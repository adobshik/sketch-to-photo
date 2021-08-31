import numpy as np


def loss_writer(path, train_loss, val_loss):
    with open(path, "a") as file:
        file.write('Train loss = {}, Val loss = {} \n'.format(str(np.round(train_loss, 3)),
                                                              str(np.round(val_loss, 3))))

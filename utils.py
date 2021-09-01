import numpy as np


def loss_writer(path, train_loss, val_loss):
    ''' Сохраняет в текстовый файл значения функций потерь 
    Args:
    path (str): путь до текстового файла
    train_loss (float): значение функции потерь на тренировочных данных
    val_loss (float): значение функции потерь на валидационных данных
    '''
    with open(path, "a") as file:
        file.write('Train loss = {}, Val loss = {} \n'.format(str(np.round(train_loss, 3)),
                                                              str(np.round(val_loss, 3))))

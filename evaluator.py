import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np


class Evaluator:
    def __init__(self, args_save_load, args, gen, disc):
        self.args_save_load = args_save_load
        self.device = args['device'] if torch.cuda.is_available() else 'cpu'
        self.EXAMPLES_PATH = args_save_load.EXAMPLES_PATH
        self.disc = disc
        self.gen = gen
        self.BCE = nn.BCEWithLogitsLoss()
        self.L1_LOSS = nn.L1Loss()
        self.L1_LAMBDA = args['L1_LAMBDA']

    def save_some_examples(self, val_loader, epoch):
        ''' Сохраняет примеры работы генератора на валидационных данных в EXAMPLES_PATH
        Аrgs: 
        val_loader (объект класса DataLoader): загружает данные из валидационного датасета
        epoch (int): номер текущей эпохи
        '''
        x, y = next(iter(val_loader))
        x, y = x.to(self.device), y.to(self.device)
        self.gen.eval()
        with torch.no_grad():
            y_fake = self.gen(x)
            y_fake = y_fake * 0.5 + 0.5  # Перевод значений в отрезок [0,1]. На выходе из генератора значения находятся в отрезке [-1,1], тк последняя функция активации - гиперболический тангенс
            save_image(y_fake, self.EXAMPLES_PATH + f"/y_gen_{epoch}.png")
            save_image(x, self.EXAMPLES_PATH + f"/input_{epoch}.png")
            save_image(y, self.EXAMPLES_PATH + f"/label_{epoch}.png")

    def one_step(self, x, y):
        ''' Считает значение функций потерь дискриминатора и генератора на валидационных данных за одну итерацию
        Аrgs:
        x (Tensor): контур (эскиз) кота
        y (Tensor): действительное изображение кота
        Return:
        Значения функций потерь генератора и дискриминатора в виде Python float
        '''
        self.disc.eval()
        self.gen.eval()
        with torch.no_grad():
            x = x.to(self.device, dtype=torch.float)  
            y = y.to(self.device, dtype=torch.float)  
            y_fake = self.gen(x)
            D_real = self.disc(x, y)
            D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
            D_fake = self.disc(x, y_fake)
            D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2
            D_fake = self.disc(x, y_fake)
            G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
            L1 = self.L1_LOSS(y_fake, y) * self.L1_LAMBDA
            G_loss = G_fake_loss + L1

        return D_loss.item(), G_loss.item()

    def evaluate(self, val_loader, epoch):
        ''' Считает среднее значение функций потерь дискриминатора и генератора на валидационных данных за одну эпоху
        Аrgs: 
        val_loader (объект класса DataLoader): загружает данные из валидационного датасета
        epoch (int): номер текущей эпохи
        Return: 
        Среднее значение функций потерь дискриминатора и генератора на валидационных данных за одну эпоху в виде Python float
        '''
        d_losses = []
        g_losses = []
        print('Evaluating...')
        for x, y in val_loader:
            d_loss, g_loss = self.one_step(x, y)
            d_losses.append(d_loss)
            g_losses.append(g_loss)
        final_d_loss =  np.array(d_losses).mean()
        final_g_loss =  np.array(g_losses).mean()
        print('Eval d_loss = {}, g_loss = {}'.format(str(np.round(final_d_loss, 3)), str(np.round(final_g_loss, 3))))
        self.save_some_examples(val_loader, epoch)
        return final_d_loss, final_g_loss

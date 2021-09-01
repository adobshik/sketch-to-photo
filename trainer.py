import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import loss_writer
import yaml
from data_generator import CatsDataset
from model_generator import Generator
from model_discriminator import Discriminator
import argparse
import pandas as pd
import numpy as np
from evaluator import Evaluator


class Trainer:
    def __init__(self, args_save_load, args):
        self.args_save_load = args_save_load
        self.device = args['device'] if torch.cuda.is_available() else 'cpu'
        self.LOAD_MODEL = args_save_load.LOAD_MODEL
        self.CHECKPOINT_DISC_LOAD = args_save_load.CHECKPOINT_DISC_LOAD
        self.CHECKPOINT_GEN_LOAD = args_save_load.CHECKPOINT_GEN_LOAD
        self.CHECKPOINT_DISC = args_save_load.CHECKPOINT_DISC
        self.CHECKPOINT_GEN = args_save_load.CHECKPOINT_GEN
        self.LEARNING_RATE = args['LEARNING_RATE']
        self.disc = Discriminator(in_channels=3).to(self.device)
        self.gen = Generator(in_channels=3, features=64).to(self.device)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=self.LEARNING_RATE, betas=(0.5, 0.999),)
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=self.LEARNING_RATE, betas=(0.5, 0.999))
        if self.args_save_load.LOAD_MODEL:
            self.load_checkpoint()

        self.BCE = nn.BCEWithLogitsLoss()
        self.L1_LOSS = nn.L1Loss()
        self.L1_LAMBDA = args['L1_LAMBDA']
        self.NUM_EPOCHS = args['NUM_EPOCHS']
        self.SAVE_MODEL = args['SAVE_MODEL']
        self.D_LOSS_PATH = args['D_LOSS_PATH']
        self.G_LOSS_PATH = args['G_LOSS_PATH']
        self.evaluator = Evaluator(args_save_load, args, self.gen, self.disc)

    def save_checkpoint(self, epoch):
        ''' Сохраняет дискриминатор и генератор в пути CHECKPOINT_DISC и CHECKPOINT_GEN соответственно
        Args:
        epoch (int): номер эпохи
        '''
        print("=> Saving checkpoint")
        checkpoint_disc = {
            "state_dict": self.disc.state_dict(),
            "optimizer": self.opt_disc.state_dict(),
        }
        checkpoint_gen = {
            "state_dict": self.gen.state_dict(),
            "optimizer": self.opt_gen.state_dict(),
        }
        torch.save(checkpoint_disc, self.args_save_load.CHECKPOINT_DISC + '_disc_epoch_' + str(epoch) + '.pth')
        torch.save(checkpoint_gen, self.args_save_load.CHECKPOINT_GEN + '_gen_epoch_' + str(epoch) + '.pth')

    def load_checkpoint(self):
        ''' Загружает дискриминатор и генератор из путей CHECKPOINT_DISC_LOAD и CHECKPOINT_GEN_LOAD соответственно
        '''
        print("=> Loading checkpoint")
        checkpoint_disc = torch.load(self.args_save_load.CHECKPOINT_DISC_LOAD, map_location='cpu')
        checkpoint_gen = torch.load(self.args_save_load.CHECKPOINT_GEN_LOAD, map_location='cpu')
        self.disc.load_state_dict(checkpoint_disc["state_dict"])
        self.gen.load_state_dict(checkpoint_gen["state_dict"])
        self.opt_disc.load_state_dict(checkpoint_disc["optimizer"])
        self.opt_gen.load_state_dict(checkpoint_gen["optimizer"])
        print('Checkpoint was restored!')

    def one_step(self, x, y):
        ''' Одна итерация обучения сети 
        Args:
        x (Tensor): контур (эскиз) кота
        y (Tensor): действительное изображение кота
        Return:
        Значения функций потерь генератора и дискриминатора за одну итерацию в виде Python float
        '''
        self.disc.train()
        self.gen.train()
        x = x.to(self.device, dtype=torch.float)  
        y = y.to(self.device, dtype=torch.float)  

        # Тренировка дискриминатора
        y_fake = self.gen(x)
        D_real = self.disc(x, y)
        D_real_loss = self.BCE(D_real, torch.ones_like(D_real))
        D_fake = self.disc(x, y_fake.detach())
        D_fake_loss = self.BCE(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        self.opt_disc.zero_grad()
        D_loss.backward()
        self.opt_disc.step()

        # Тренировка генератора
        D_fake = self.disc(x, y_fake)
        G_fake_loss = self.BCE(D_fake, torch.ones_like(D_fake))
        L1 = self.L1_LOSS(y_fake, y) * self.L1_LAMBDA
        G_loss = G_fake_loss + L1

        self.opt_gen.zero_grad()
        G_loss.backward()
        self.opt_gen.step()
        return D_loss.item(), G_loss.item()

    def train(self, train_loader, val_loader):
        ''' Обучение сети в течении количества эпох равного NUM_EPOCHS, 
        включая:
        сохранение в файлы D_LOSS_PATH и G_LOSS_PATH соответственно средних значений функций потерь дискриминатора и генератора за каждую эпоху, 
        сохранение моделей каждые 5 эпох.
        Args:
        train_loader (объект класса DataLoader): загружает данные из тренировочного датасета
        val_loader (объект класса DataLoader): загружает данные из валидационного датасета
        '''
        for epoch in range(self.NUM_EPOCHS):
            d_loss_epoch = []
            g_loss_epoch = []
            loop = tqdm(train_loader, leave=True)
            for idx, (x, y) in enumerate(loop):
                d_loss, g_loss = self.one_step(x, y)
                d_loss_epoch.append(d_loss)
                g_loss_epoch.append(g_loss)
                loop.set_postfix(loss_d=np.array(d_loss_epoch).mean(), loss_g=np.array(g_loss_epoch).mean())
            val_d_loss, val_g_loss = self.evaluator.evaluate(val_loader, epoch)
            loss_writer(self.D_LOSS_PATH, train_loss=np.array(d_loss_epoch).mean(), val_loss=val_d_loss)
            loss_writer(self.G_LOSS_PATH, train_loss=np.array(g_loss_epoch).mean(), val_loss=val_g_loss)
            if self.SAVE_MODEL and epoch % 5 == 0 and epoch != 0:  
                self.save_checkpoint(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Введите требуемые параметры и пути.')
    parser.add_argument('--DATASET_FILE_PATH', type=str, help='Путь до csv файла с путями до тренировочных данных')
    parser.add_argument('--LOAD_MODEL', default=False, type=bool, help='Загрузить предобученную модель? True, если да; иначе False.')
    parser.add_argument('--CHECKPOINT_DISC_LOAD', default='metadata/models', type=str, help='Путь до предобученного дискриминатора.')
    parser.add_argument('--CHECKPOINT_GEN_LOAD', default='metadata/models', type=str, help='Путь до предобученного генератора.')
    parser.add_argument('--CHECKPOINT_DISC', default='metadata/models/disc.pth', type=str, help='Путь для сохранения дискриминатора.')
    parser.add_argument('--CHECKPOINT_GEN',  default='metadata/models/gen.pth', type=str, help='Путь для сохранения генератора.')
    parser.add_argument('--EXAMPLES_PATH', default='metadata/examples', type=str, help='Путь для сохранения примеров работы генератора')
    parser.add_argument('--FOLD', default=5, type=int, help='Валидационный фолд')

    args_save_load = parser.parse_args()
    with open(r'config.yaml') as file:
        args = yaml.load(file)
    df = pd.read_csv(args_save_load.DATASET_FILE_PATH)
    fold = args_save_load.FOLD  # значение в столбце "kfold", указывающее на принадлежность экземпляров к валидационным
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_val = df[df.kfold == fold].reset_index(drop=True)
    train_dataset = CatsDataset(imagespath=df_train['cats'].tolist(), maskspath=df_train['masks'].tolist(), augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args['BATCH_SIZE'], shuffle=True, num_workers=args['NUM_WORKERS'])
    val_dataset = CatsDataset(imagespath=df_val['cats'].tolist(),  maskspath=df_val['masks'].tolist(), augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    trainer = Trainer(args_save_load, args)
    trainer.train(train_loader, val_loader)

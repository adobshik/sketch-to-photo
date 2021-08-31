import torch
import argparse
import yaml
from skimage import io, transform
from model_generator import Generator
from torchvision.utils import save_image


class Inference:
    def __init__(self, args_inference, args):
        self.args_inference = args_inference
        self.device = args['device'] if torch.cuda.is_available() else 'cpu'
        self.MODEL_PATH = args_inference.MODEL_PATH
        self.gen = Generator(in_channels=3, features=64).to(self.device)
        self.load_gen()

    def load_gen(self):
        print("=> Loading checkpoint")
        checkpoint_gen = torch.load(self.args_inference.MODEL_PATH, map_location='cpu')
        self.gen.load_state_dict(checkpoint_gen["state_dict"])
        print('Checkpoint was restored!')

    def prepare(self, path_to_image):
        image = io.imread(path_to_image)
        image = transform.resize(image=image, output_shape=(256,256), order=1)
        image = torch.FloatTensor(image)
        if len(image.shape) == 2:
            image = torch.stack([image] * 3)
        else:
            image = image.permute(2, 0, 1)

        image = image[None, :]
        return image

    def inference(self, path_to_image, save_path):
        image = self.prepare(path_to_image)
        self.gen.eval()
        with torch.no_grad():
            image = image.to(self.device)
            image_fake = self.gen(image)
            image_fake = image_fake * 0.5 + 0.5  # Перевод значений в отрезок [0,1]. На выходе из генератора значения находятся в отрезке [-1,1], тк последняя функция активации - гиперболический тангенс
        save_image(image_fake, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Введите требуемые параметры и пути.')
    parser.add_argument('--MODEL_PATH', type=str, help='Путь до предобученного генератора')
    parser.add_argument('--IMAGE_PATH', type=str, help='Путь до картинки (эскиза).')
    parser.add_argument('--SAVE_PATH', type=str, help='Путь для сохранения синтезированной фотореалистической версии.')

    args_inference = parser.parse_args()
    with open(r'config.yaml') as file:
        args = yaml.load(file)

    inference = Inference(args_inference, args)
    inference.inference(path_to_image=args_inference.IMAGE_PATH, save_path=args_inference.SAVE_PATH)

import torch
from torch.utils.data import Dataset
import albumentations as A
from skimage import io, transform


class CatsDataset(Dataset):
    def __init__(self, imagespath, maskspath, augment=None):
        self.imagespath = imagespath
        self.maskspath = maskspath
        self.augment = augment

    def __len__(self):
        return len(self.imagespath)

    def __getitem__(self, idx):
        image = self.imagespath[idx]
        mask = self.maskspath[idx]
        image = io.imread(image)
        mask = io.imread(mask)
        image = transform.resize(image=image, output_shape=(256, 256), order=1)
        mask = transform.resize(image=mask, output_shape=(256, 256), order=1)

        if self.augment:
            aug = A.OneOf([A.HorizontalFlip(p=1),
                           A.RandomSizedCrop(min_max_height=(492, 512), height=512, width=512, p=1),
                           A.Rotate(limit=8, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_REFLECT_101, p=1),
                           A.ElasticTransform(p=1, alpha=70, sigma=120 * 0.05, alpha_affine=120 * 0.03)
                           ], p=0.5)
            augmented = aug(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        #если изображение чернобелое, преобразование в трехмерный тензор, иначе только смена размерностей
        if len(image.shape) == 2:
            image = torch.stack([image] * 3)
        else:
            image = image.permute(2, 0, 1)
            
        if len(mask.shape) == 2:
            mask = torch.stack([mask] * 3)
        else:
            mask = mask.permute(2, 0, 1)
            
        return mask, image

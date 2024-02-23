# References:
    # https://github.com/KimRass/DDPM/blob/main/data.py

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CelebA
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from pathlib import Path


class CelebADS(Dataset):
    def __init__(self, data_dir, split, img_size, hflip):
        self.ds = CelebA(root=data_dir, split=split, download=True)

        transforms = [
            A.HorizontalFlip(p=0.5),
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
        if not hflip:
            transforms = transforms[1:]
        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, _ = self.ds[idx]
        return self.transform(image=np.array(image))["image"]


def get_train_and_val_dls(data_dir, img_size, batch_size, n_cpus):
    train_ds = CelebADS(data_dir=data_dir, split="train", img_size=img_size, hflip=True)
    val_ds = CelebADS(data_dir=data_dir, split="valid", img_size=img_size, hflip=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    return train_dl, val_dl


def get_test_dl(data_dir, img_size, batch_size, n_cpus):
    test_ds = CelebADS(data_dir=data_dir, split="test", img_size=img_size, hflip=False)
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
        num_workers=n_cpus,
    )

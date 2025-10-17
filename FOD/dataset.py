import os
import random
from glob import glob

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from FOD.api import get_total_paths, get_splitted_dataset, get_transforms


def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu').float())
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


class AutoFocusStackDataset(Dataset):
    """
    Dataset for focal stack inputs.
    Each sample = [stack of focus images], depth GT.
    Directory format expected:
        path_images/
            scene_0001/
                focus_00.png
                focus_01.png
                ...
            scene_0002/
                ...
        path_depths/
            scene_0001.png
            scene_0002.png
    """

    def __init__(self, config, dataset_name, split=None):
        self.split = split
        self.config = config

        base_path = config['Dataset']['paths']['path_dataset']
        path_images = os.path.join(base_path, dataset_name, config['Dataset']['paths']['path_images'])
        path_depths = os.path.join(base_path, dataset_name, config['Dataset']['paths']['path_depths'])

        print('########################')
        print("Image path:", path_images)
        print("Depth path:", path_depths)

        # list of subfolders (each contains one focal stack)
        self.stack_dirs = sorted(
            [d for d in glob(os.path.join(path_images, '*')) if os.path.isdir(d)]
        )

        self.paths_depths = get_total_paths(path_depths, config['Dataset']['extensions']['ext_depths'])

        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        assert (len(self.stack_dirs) == len(self.paths_depths)), (
            f"Different number of stacks ({len(self.stack_dirs)}) and depths ({len(self.paths_depths)})"
        )
        assert (
            config['Dataset']['splits']['split_train']
            + config['Dataset']['splits']['split_test']
            + config['Dataset']['splits']['split_val']
            == 1
        ), "Invalid splits (sum must be 1)"

        # split into train/val/test subsets
        self.stack_dirs, self.paths_depths = get_splitted_dataset(
            config, self.split, dataset_name, self.stack_dirs, self.paths_depths
        )

        # Get transforms
        self.transform_image, self.transform_depth = get_transforms(config)

        # probabilities for augmentations
        self.p_flip = config['Dataset']['transforms']['p_flip'] if split == 'train' else 0
        self.p_crop = config['Dataset']['transforms']['p_crop'] if split == 'train' else 0
        self.p_rot = config['Dataset']['transforms']['p_rot'] if split == 'train' else 0
        self.resize = config['Dataset']['transforms']['resize']

    def __len__(self):
        return len(self.stack_dirs)

    def _load_stack(self, stack_dir):
        """Load all focal images in sorted order from a given stack directory"""
        img_paths = sorted(glob(os.path.join(stack_dir, '*')))
        stack = []
        for p in img_paths:
            img = Image.open(p)
            img_t = self.transform_image(img)
            stack.append(img_t)
        # stack into a single tensor [N, C, H, W]
        return torch.stack(stack, dim=0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load all focus images
        img_stack = self._load_stack(self.stack_dirs[idx])

        # Load depth GT
        depth = self.transform_depth(Image.open(self.paths_depths[idx]))

        # --- Data augmentation applied consistently ---
        if random.random() < self.p_flip:
            img_stack = torch.flip(img_stack, dims=[3])  # flip W dimension
            depth = TF.hflip(depth)

        if random.random() < self.p_crop:
            random_size = random.randint(256, self.resize - 1)
            max_size = self.resize - random_size
            left = int(random.random() * max_size)
            top = int(random.random() * max_size)

            img_stack = img_stack[:, :, top:top + random_size, left:left + random_size]
            depth = TF.crop(depth, top, left, random_size, random_size)

            img_stack = torch.stack([
                transforms.Resize((self.resize, self.resize))(img)
                for img in img_stack
            ])
            depth = transforms.Resize((self.resize, self.resize))(depth)

        if random.random() < self.p_rot:
            random_angle = random.random() * 20 - 10  # [-10, 10]
            mask = torch.ones((1, self.resize, self.resize))
            mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)

            img_stack = torch.stack([
                TF.rotate(img, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
                for img in img_stack
            ])
            depth = TF.rotate(depth, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)

            left = torch.argmax(mask[:, 0, :]).item()
            top = torch.argmax(mask[:, :, 0]).item()
            coin = min(left, top)
            size = self.resize - 2 * coin

            img_stack = img_stack[:, :, coin:coin + size, coin:coin + size]
            depth = TF.crop(depth, coin, coin, size, size)

            img_stack = torch.stack([
                transforms.Resize((self.resize, self.resize))(img)
                for img in img_stack
            ])
            depth = transforms.Resize((self.resize, self.resize))(depth)

        return img_stack, depth

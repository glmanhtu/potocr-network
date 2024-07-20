# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import PurePath
from typing import Callable, Optional, Sequence

from torch.utils.data import DataLoader
from torchvision import transforms as T

import pytorch_lightning as pl

from .augment import HorizontalLongest, HeightWidthLongestResize, PadCenterCrop, RandomDistortion, GaussianBlur
from .dataset import LmdbDataset, build_tree_dataset


class SceneTextDataModule(pl.LightningDataModule):
    TEST_BENCHMARK_SUB = ('IIIT5k', 'SVT', 'IC13_857', 'IC15_1811', 'SVTP', 'CUTE80')
    TEST_BENCHMARK = ('IIIT5k', 'SVT', 'IC13_1015', 'IC15_2077', 'SVTP', 'CUTE80')
    TEST_NEW = ('ArT', 'COCOv1.4', 'Uber')
    TEST_ALL = tuple(set(TEST_BENCHMARK_SUB + TEST_BENCHMARK + TEST_NEW))

    def __init__(
        self,
        root_dir: str,
        img_size: Sequence[int],
        max_label_length: int,
        batch_size: int,
        num_workers: int,
        augment: bool,
        rotation: int = 0,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = tuple(img_size)
        self.max_label_length = max_label_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.rotation = rotation
        self.collate_fn = collate_fn
        self._train_dataset = None
        self._val_dataset = None

    @staticmethod
    def get_transform(img_size: tuple[int], augment: bool = False, rotation: int = 0):
        transforms = []
        if augment:
            transforms.extend([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                RandomDistortion(img_size, fill=144, percentage=0.1),
                GaussianBlur(radius_max=0.7)
            ])
        else:
            transforms.append(PadCenterCrop(img_size, pad_if_needed=True, fill=144))
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        return T.Compose(transforms)

    @property
    def train_dataset(self):
        if self._train_dataset is None:
            transform = self.get_transform(self.img_size, self.augment)
            root = PurePath(self.root_dir, 'train')
            self._train_dataset = build_tree_dataset(
                root,
                self.max_label_length,
                transform=transform,
            )
        return self._train_dataset

    @property
    def val_dataset(self):
        if self._val_dataset is None:
            transform = self.get_transform(self.img_size)
            root = PurePath(self.root_dir, 'val')
            self._val_dataset = build_tree_dataset(
                root,
                self.max_label_length,
                transform=transform,
            )
        return self._val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloaders(self, subset):
        transform = self.get_transform(self.img_size, rotation=self.rotation)
        root = PurePath(self.root_dir, 'test')
        datasets = {
            s: LmdbDataset(
                str(root / s),
                self.max_label_length,
                transform=transform,
            )
            for s in subset
        }
        return {
            k: DataLoader(
                v, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, collate_fn=self.collate_fn
            )
            for k, v in datasets.items()
        }

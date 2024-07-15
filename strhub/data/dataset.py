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
import glob
import io
import logging
import unicodedata
from pathlib import Path, PurePath
from typing import Callable, Optional, Union

import lmdb
from PIL import Image

from torch.utils.data import ConcatDataset, Dataset

from strhub.data.utils import CharsetAdapter

log = logging.getLogger(__name__)


def build_tree_dataset(root: Union[PurePath, str], *args, **kwargs):
    try:
        kwargs.pop('root')  # prevent 'root' from being passed via kwargs
    except KeyError:
        pass
    root = Path(root).absolute()
    log.info(f'dataset root:\t{root}')
    datasets = []
    for mdb in glob.glob(str(root / '**/data.mdb'), recursive=True):
        mdb = Path(mdb)
        ds_name = str(mdb.parent.relative_to(root))
        ds_root = str(mdb.parent.absolute())
        dataset = LmdbDataset(ds_root, *args, **kwargs)
        log.info(f'\tlmdb:\t{ds_name}\tnum samples: {len(dataset)}')
        datasets.append(dataset)
    return ConcatDataset(datasets)


class LmdbDataset(Dataset):
    """Dataset interface to an LMDB database.

    It supports both labelled and unlabelled datasets. For unlabelled datasets, the image index itself is returned
    as the label. Unicode characters are normalized by default. Case-sensitivity is inferred from the charset.
    Labels are transformed according to the charset.
    """

    def __init__(
        self,
        root: str,
        charset: str,
        max_label_len: int,
        min_image_dim: int = 0,
        remove_whitespace: bool = True,
        normalize_unicode: bool = True,
        unlabelled: bool = False,
        transform: Optional[Callable] = None,
        prepend_lang_token=False
    ):
        self._env = None
        self.root = root
        self.unlabelled = unlabelled
        self.transform = transform
        self.labels = []
        self.filtered_index_list = []

        with self._create_env() as env, env.begin() as txn:
            self.num_samples = int(txn.get('num-samples'.encode()))

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self):
        return lmdb.open(
            self.root, max_readers=1, readonly=True, create=False, readahead=False, meminit=False, lock=False
        )

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        lmdb_idx = index + 1   # LMDB keys are 1-indexed
        img_key = f'image-{lmdb_idx:09d}'.encode()
        label_key = f'secondary-{lmdb_idx:09d}'.encode()
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
            if not self.unlabelled:
                label = txn.get(label_key).decode()
            else:
                label = index
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

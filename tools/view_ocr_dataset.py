import argparse
import io

import cv2
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class LmdbSimpleDataset(Dataset):
    def __init__(self, root: str):
        self._env = None
        self.root = root

        with self._create_env() as env, env.begin() as txn:
            self.num_samples = int(txn.get('num-samples'.encode()))

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
        label_key = f'secondary-{lmdb_idx:09d}'.encode()
        image_key = f'image-{lmdb_idx:09d}'.encode()
        with self.env.begin() as txn:
            label = txn.get(label_key).decode()
            imgbuf = txn.get(image_key)
        buf = io.BytesIO(imgbuf)
        img = Image.open(buf).convert('RGB')
        np_img = np.array(img)
        return np_img, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser('View dataset script', add_help=False)
    parser.add_argument('--dataset-dir', required=True, type=str)
    args = parser.parse_args()

    dataset = LmdbSimpleDataset(args.dataset_dir)
    for i in range(len(dataset)):
        img, label = dataset[i]

        # Draw label on a white image which has the same shape as img, using cv2.putText, respect \n as new line
        img_with_label = np.ones_like(img) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        font_thickness = 1
        color = (0, 0, 0)
        y0, dy = 15, 15
        for i, line in enumerate(label.split('\n')):
            y = y0 + i * dy
            cv2.putText(img_with_label, line, (15 , y), font, font_scale, color, font_thickness, cv2.LINE_AA)

        # Horizontally concatenate img and img with label
        img_with_label = np.concatenate((img, img_with_label), axis=1)
        cv2.imshow('img_with_label', img_with_label)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

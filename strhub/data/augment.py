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
import random
from functools import partial

import imgaug.augmenters as iaa
import numpy as np
import torch
import torchvision.transforms
from PIL import Image, ImageFilter

from timm.data import auto_augment

from strhub.data import aa_overrides
import torchvision.transforms.functional as F


aa_overrides.apply()

_OP_CACHE = {}


def _get_op(key, factory):
    try:
        op = _OP_CACHE[key]
    except KeyError:
        op = factory()
        _OP_CACHE[key] = op
    return op


def _get_param(level, img, max_dim_factor, min_level=1):
    max_level = max(min_level, max_dim_factor * max(img.size))
    return round(min(level, max_level))


def gaussian_blur(img, radius, **__):
    radius = _get_param(radius, img, 0.02)
    key = 'gaussian_blur_' + str(radius)
    op = _get_op(key, lambda: ImageFilter.GaussianBlur(radius))
    return img.filter(op)


def motion_blur(img, k, **__):
    k = _get_param(k, img, 0.08, 3) | 1  # bin to odd values
    key = 'motion_blur_' + str(k)
    op = _get_op(key, lambda: iaa.MotionBlur(k))
    return Image.fromarray(op(image=np.asarray(img)))


def gaussian_noise(img, scale, **_):
    scale = _get_param(scale, img, 0.25) | 1  # bin to odd values
    key = 'gaussian_noise_' + str(scale)
    op = _get_op(key, lambda: iaa.AdditiveGaussianNoise(scale=scale))
    return Image.fromarray(op(image=np.asarray(img)))


def poisson_noise(img, lam, **_):
    lam = _get_param(lam, img, 0.2) | 1  # bin to odd values
    key = 'poisson_noise_' + str(lam)
    op = _get_op(key, lambda: iaa.AdditivePoissonNoise(lam))
    return Image.fromarray(op(image=np.asarray(img)))


def _level_to_arg(level, _hparams, max):
    level = max * level / auto_augment._LEVEL_DENOM
    return (level,)


_RAND_TRANSFORMS = auto_augment._RAND_INCREASING_TRANSFORMS.copy()
_RAND_TRANSFORMS.remove('SharpnessIncreasing')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.remove('ShearX')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.remove('ShearY')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.remove('TranslateXRel')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.remove('TranslateYRel')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.remove('Rotate')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.remove('Invert')  # remove, interferes with *blur ops
_RAND_TRANSFORMS.extend([
    'GaussianBlur',
    # 'MotionBlur',
    # 'GaussianNoise',
    'PoissonNoise',
])
auto_augment.LEVEL_TO_ARG.update({
    'GaussianBlur': partial(_level_to_arg, max=2),
    'MotionBlur': partial(_level_to_arg, max=10),
    'GaussianNoise': partial(_level_to_arg, max=0.05 * 255),
    'PoissonNoise': partial(_level_to_arg, max=20),
})
auto_augment.NAME_TO_OP.update({
    'GaussianBlur': gaussian_blur,
    'MotionBlur': motion_blur,
    'GaussianNoise': gaussian_noise,
    'PoissonNoise': poisson_noise,
})


def rand_augment_transform(magnitude=5, num_layers=3):
    # These are tuned for magnitude=5, which means that effective magnitudes are half of these values.
    hparams = {
        # 'rotate_deg': 30,
        # 'shear_x_pct': 0.9,
        # 'shear_y_pct': 0.2,
        # 'translate_x_pct': 0.10,
        # 'translate_y_pct': 0.30,
    }
    ra_ops = auto_augment.rand_augment_ops(magnitude, hparams=hparams, transforms=_RAND_TRANSFORMS)
    # Supply weights to disable replacement in random selection (i.e. avoid applying the same op twice)
    choice_weights = [1.0 / len(ra_ops) for _ in range(len(ra_ops))]
    return auto_augment.RandAugment(ra_ops, num_layers, choice_weights)


class HorizontalLongest(torch.nn.Module):
    def __init__(self, enable=True):
        super().__init__()
        self.enable = enable

    def forward(self, img):
        if not self.enable:
            return img
        w, h = img.size
        if w < h:
            img = img.transpose(Image.ROTATE_270)
        return img


class RandomDistortion(torch.nn.Module):
    def __init__(self, img_size, fill=0, percentage=0.1):
        super().__init__()
        self.img_size = img_size
        self.fill = fill
        self.percentage = percentage
        self.resizer = torchvision.transforms.Resize(img_size)

    def forward(self, img):
        w, h = img.size
        w = int(w * np.random.uniform(1.0 - self.percentage, 1.0 + self.percentage))
        h = int(h * np.random.uniform(1.0 - self.percentage, 1.0 + self.percentage))
        img = img.resize((w, h), Image.BILINEAR)
        cropper = torchvision.transforms.RandomCrop(max(w, h), pad_if_needed=True, fill=self.fill)
        img = cropper(img)
        return self.resizer(img)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class HeightWidthLongestResize(torch.nn.Module):
    def __init__(self, max_height, max_width, interpolation=Image.BICUBIC):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.interpolation = interpolation

    def forward(self, img):
        w, h = img.size

        w_factor = self.max_width / w
        h_factor = self.max_height / h
        if w_factor < h_factor:
            new_w = self.max_width
            new_h = int(h * w_factor)
            assert new_h <= self.max_height
        else:
            new_h = self.max_height
            new_w = int(w * h_factor)
            assert new_w <= self.max_width
        return img.resize((new_w, new_h), self.interpolation)


class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img):

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return F.center_crop(img, self.size)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

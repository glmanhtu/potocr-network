import hydra
import numpy as np
import torchvision.transforms
from matplotlib import pyplot as plt
from omegaconf import DictConfig

from strhub.data.augment import UnNormalize
from strhub.data.module import SceneTextDataModule


@hydra.main(config_path='configs', config_name='main', version_base='1.2')
def main(config: DictConfig):
    datamodule: SceneTextDataModule = hydra.utils.instantiate(config.data)
    unnorm = torchvision.transforms.Compose([
        UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        torchvision.transforms.ToPILImage(),
        lambda x: np.asarray(x)
    ])
    dataset = datamodule.train_dataset
    for img, label in dataset:
        img = unnorm(img)
        plt.imshow(img)
        plt.show()



if __name__ == '__main__':
    main()


import argparse
import os

import lmdb
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader


class LmdbLabelDataset(Dataset):
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
        with self.env.begin() as txn:
            label = txn.get(label_key).decode()
        return label


def batch_iterator(dataloader):
    for batch in dataloader:
        yield batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train tokenizer script', add_help=False)
    parser.add_argument('--dataset-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--vocab_size', required=True, type=int)
    args = parser.parse_args()

    batch_size = 100
    save_dir = args.output_dir

    dataset = LmdbLabelDataset(args.dataset_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(batch_iterator(dataloader), vocab_size=args.vocab_size, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
    ])

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_model(args.output_dir)

    print(f'Tokenizer saved to {args.output_dir}')

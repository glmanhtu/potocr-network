import argparse
import os
import re
import statistics

import lmdb
from tokenizers.implementations import ByteLevelBPETokenizer
from torch.utils.data import Dataset


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


def batch_iterator(dataset, special_chars):
    for i in range(0, len(dataset), batch_size):
        result = []
        for j in range(i, min(i + batch_size, len(dataset))):
            label = dataset[j]
            # remove all numbers from label
            label = re.sub(r'\d+', '', label)
            # replace - with space
            label = label.replace('-', ' ')
            label = label.replace('/', ' ')
            # remove special characters
            for special_char in special_chars:
                label = label.replace(special_char, '')

            result.append(label)
        yield result

def evaluate_tokenizer(tokenizer: ByteLevelBPETokenizer, dataset):
    total = 0
    correct = 0
    token_lengths = []
    for i in range(0, len(dataset)):
        item = dataset[i]
        encoded = tokenizer.encode(item)
        token_lengths.append(len(encoded.ids))
        decoded = tokenizer.decode(encoded.ids)
        if item == decoded:
            correct += 1
        total += 1

    t = args.threshold
    print("Token length statistics:")
    print(f"Max token length: {max(token_lengths)}")
    print(f"Min token length: {min(token_lengths)}")
    print(f"Mean token length: {sum(token_lengths) / len(token_lengths)}")
    print(f"Standard deviation: {statistics.stdev(token_lengths)}")
    print(f"Median token length: {statistics.median(token_lengths)}")
    print(f"Number of tokens with length > {t}: {len([l for l in token_lengths if l > t])} / {len(token_lengths)}")
    max_token_length = max(token_lengths)
    return correct / total, max_token_length


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train tokenizer script', add_help=False)
    parser.add_argument('--dataset-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--vocab-size', type=int, default=5000)
    parser.add_argument('--threshold', type=int, default=200)
    args = parser.parse_args()

    batch_size = 100
    save_dir = args.output_dir

    train_dataset = LmdbLabelDataset(os.path.join(args.dataset_dir, 'train'))
    val_dataset = LmdbLabelDataset(os.path.join(args.dataset_dir, 'val'))
    test_dataset = LmdbLabelDataset(os.path.join(args.dataset_dir, 'test'))

    tokenizer = ByteLevelBPETokenizer()
    special_chars = ['(', ')', '"', ':', '?', '!', ';', '/', '-', ',', '.', '%', '+']

    tokenizer.train_from_iterator(batch_iterator(train_dataset, special_chars), vocab_size=args.vocab_size,
                                  special_tokens=["</s>", "<s>", "<pad>", "<unk>"])
    tokenizer.add_tokens([str(i) for i in range(10)])
    tokenizer.add_tokens(special_chars)

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_model(args.output_dir)

    print(f'Tokenizer saved to {args.output_dir}')
    accuracy, max_token_length = evaluate_tokenizer(tokenizer, train_dataset)
    print(f'Train accuracy: {accuracy}, max token length: {max_token_length}\n')
    accuracy, max_token_length = evaluate_tokenizer(tokenizer, val_dataset)
    print(f'Validation accuracy: {accuracy}, max token length: {max_token_length}\n')
    accuracy, max_token_length = evaluate_tokenizer(tokenizer, test_dataset)
    print(f'Test accuracy: {accuracy}, max token length: {max_token_length}\n')


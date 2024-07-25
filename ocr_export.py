import argparse
import os.path

import cv2
import torch
from torch import Tensor

from strhub.models.parseq.system import PARSeq


class DummyTokenizer(torch.nn.Module):
    def __init__(self, eos_id: int, bos_id: int, pad_id: int):
        super().__init__()
        self.eos_id, self.bos_id, self.pad_id = eos_id, bos_id, pad_id



class ParseqOCR(PARSeq):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shape = [384, 384]
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id

    def forward(self, images: Tensor):
        p = self.model.forward(self.bos_id, self.pad_id, self.eos_id, images)
        result = p.softmax(-1)
        batch_probs, batch_tokens = [], []
        for i in range(result.shape[0]):
            probs, tokens = result[i].max(-1)
            batch_probs.append(probs)
            batch_tokens.append(tokens)

        return batch_tokens, batch_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--tokenizer', required=True, type=str)
    parser.add_argument('--device', default='cpu')
    args, _ = parser.parse_known_args()
    kwargs = {
        'tokenizer': args.tokenizer
    }
    model = ParseqOCR.load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)

    img_path = os.path.join('assets', 'test_imgs', 'test-18.jpg')
    bboxes = [[2.3631279e+03, 0.0000000e+00, 4.4395605e+03, 1.6244840e+03],
              [1.0369185e+03, 3.2731348e+02, 2.3202070e+03, 1.2938562e+03]]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    bbox_tensor = torch.tensor([bboxes[1]], dtype=torch.int64)

    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    # model(img_tensor, bbox_tensor)
    model.to_torchscript(args.output_path, example_inputs=(img_tensor, bbox_tensor))

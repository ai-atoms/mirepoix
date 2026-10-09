"""Predict u_640 from image encodings with a trained regression head.

  python scripts/predict_heads.py --encoder convnext_in --layers 3
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

IN_FEATURES = {'convnext_in': 1536, 'efb0_in': 1280, 'efb5_in': 2048, 'res50_in': 2048,
               'res50_mn': 2048, 'vitl16_in': 1024}
MODELS = {'convnext_in': 'convnext_in_3hl', 'efb0_in': 'effnet_b0_in_2hl', 'efb5_in': 'effnet_b5_in_3hl',
          'res50_in': 'resnet50_in_2hl', 'res50_mn': 'resnet50_mn_2hl', 'vitl16_in': 'vitl16_in_3hl'}


def make_head(in_features, n_hidden, out=640):
    layers, d = [], in_features
    for _ in range(n_hidden):
        layers += [nn.Linear(d, 1024), nn.ReLU()]
        d = 1024
    return nn.Sequential(*layers, nn.Linear(d, out))


class Head(nn.Module):
    def __init__(self, in_features, n_hidden):
        super().__init__()
        self.regression_head = make_head(in_features, n_hidden)

    def forward(self, x):
        return self.regression_head(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--encoder', default='convnext_in', choices=MODELS)
    ap.add_argument('--data', default='data/d567b')
    ap.add_argument('--models', default='data/npj_components/models')
    args = ap.parse_args()

    name = MODELS[args.encoder]
    model = Head(IN_FEATURES[args.encoder], int(name.split('_')[-1][0]))
    model.load_state_dict(torch.load(os.path.join(args.models, name, 'best_head.pth'), map_location='cpu'))
    model.eval()

    src = os.path.join(args.data, 'enc_images', args.encoder)
    out = os.path.join(args.data, 'results', args.encoder)
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(src) if f.endswith('.npy'))
    with torch.no_grad():
        for f in files:
            x = torch.from_numpy(np.load(os.path.join(src, f)).astype(np.float32).reshape(1, -1))
            np.save(os.path.join(out, f), model(x)[0].numpy())
    print(f'{len(files)} predictions -> {out}')


if __name__ == '__main__':
    main()

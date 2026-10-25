"""Reproduce the shipped linear classifier from frozen training features.

The feature archive contains only supervised training examples. This command
does not tune thresholds or read evaluation recordings. Image embedding
provenance and the separate validation results are supplied in the model card.
"""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[1]))

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',required=True)
    parser.add_argument('--feature-model',required=True)
    parser.add_argument('--expected-feature-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    digest=hashlib.sha256(Path(args.feature_model).read_bytes()).hexdigest()
    if digest!=args.expected_feature_sha256:raise ValueError('Feature model does not match the training provenance')
    with np.load(args.features,allow_pickle=False) as data:
        x=data['x'].copy();y=data['y'].copy()
    if (x.ndim!=2 or y.shape!=(len(x),) or x.shape[1]!=576 or len(x)<2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isin(y,[0.,1.]).all() or not 0.<float(y.mean())<1.):
        raise ValueError('Expected finite 576-dimensional features with both binary training classes')
    torch.set_num_threads(2);torch.manual_seed(42)
    mean=x.mean(axis=0);scale=np.maximum(x.std(axis=0),.02)
    tx=torch.from_numpy((x-mean)/scale);ty=torch.from_numpy(y)
    model=torch.nn.Linear(x.shape[1],1)
    torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
    sample_weight=torch.where(ty>0,.5/max(float(y.mean()),1e-6),.5/max(1-float(y.mean()),1e-6))
    optimizer=torch.optim.LBFGS(model.parameters(),lr=.5,max_iter=150,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        logits=model(tx).flatten()
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,ty,reduction='none')*sample_weight).mean()+.05*model.weight.square().sum()
        loss.backward();return loss
    optimizer.step(closure)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,mean=mean,scale=scale,weight=model.weight.detach().numpy().flatten(),
        bias=float(model.bias.detach().numpy()[0]),schema_version=np.asarray(1),
        feature_model_sha256=np.asarray(digest),reject_threshold=np.asarray(.35),temporal_confirmations=np.asarray(2))
    print('Refit classifier from',len(y),'training examples:',output)


if __name__=='__main__':main()

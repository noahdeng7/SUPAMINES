#!/usr/bin/env python3
"""Fetch a BetaTetris release checkpoint to use as the distillation teacher.

    python distillation/download_teacher.py              # the default 'normal' model
    python distillation/download_teacher.py --variant aggro

The v1.0.0 models are the ones trained across every tap speed and reaction time, which is
the variant this repo's extension is built for (LINE_CAP=430, rotation enabled).  They are
residual towers, not the transformer the students are, so they load through
RL/legacy_model.py: same observation, same 800 placements, a different network.

The v0.1.0 model predates the multi-tap observation and the 'perfect' model is trained for
a different objective; both are here for completeness but distil from a v1.0.0 file unless
you know you want otherwise.  A checkpoint whose shapes do not match the extension you
built will fail to load, and infer_model_args will say which shapes it wanted.
"""

import argparse
import pathlib
import sys
import urllib.request

BASE = 'https://github.com/BetaTetris/betatetris-tablebase/releases/download'
VARIANTS = {
    # name: (release tag, asset)
    'normal': ('v1.0.0', 'model-v1.0.0-normal.pth'),
    'aggro': ('v1.0.0', 'model-v1.0.0-aggro.pth'),
    'perfect': ('v1.0.0-perfect', 'model-v1.0.0-perfect.pth'),
    'v0.1.0': ('v0.1.0', 'model-v0.1.0-30hz-18f.pth'),
}
MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / 'models'


def download(variant='normal', out_dir=MODELS_DIR, force=False):
    tag, asset = VARIANTS[variant]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / asset
    if target.is_file() and not force:
        print('{} already there ({:.1f} MB); pass --force to fetch it again'.format(
            target, target.stat().st_size / 1e6))
        return target
    url = '{}/{}/{}'.format(BASE, tag, asset)
    print('fetching {}'.format(url))
    tmp = target.with_suffix(target.suffix + '.part')

    def progress(blocks, block_size, total):
        if total <= 0: return
        done = min(blocks * block_size, total)
        sys.stderr.write('\r  {:.1f} / {:.1f} MB'.format(done / 1e6, total / 1e6))
        sys.stderr.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    sys.stderr.write('\n')
    tmp.replace(target)
    print('wrote {} ({:.1f} MB)'.format(target, target.stat().st_size / 1e6))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', default='normal', choices=sorted(VARIANTS))
    parser.add_argument('--out-dir', default=str(MODELS_DIR))
    parser.add_argument('--force', action='store_true', help='re-download over an existing file')
    parser.add_argument('--check', action='store_true',
                        help='load the checkpoint afterwards and print its architecture')
    args = parser.parse_args()

    target = download(args.variant, args.out_dir, args.force)
    if args.check:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'RL'))
        import torch
        from legacy_model import Model, infer_model_args
        state_dict = torch.load(target, weights_only=True, map_location='cpu')
        model_args = infer_model_args(state_dict)
        model = Model(**model_args)
        model.load_state_dict(state_dict)
        print('loads as Model({}) -- {:.2f}M parameters'.format(
            ', '.join('{}={}'.format(k, v) for k, v in model_args.items()),
            model.num_params() / 1e6))


if __name__ == '__main__':
    main()

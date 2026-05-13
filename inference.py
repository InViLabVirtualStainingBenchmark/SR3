import os
import sys
import time
import argparse

import torch
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import EvalDataset
from diffusion import GaussianDiffusion


def get_parser():
    parser = argparse.ArgumentParser(description='SR3 inference — saves predicted images to disk for external evaluation.')
    parser.add_argument('-m', '--model',       type=str,   required=True,          help='Model directory name (e.g. UNet)')
    parser.add_argument('-a', '--args',        type=int,   nargs='+', default=[],  help='Model constructor positional args (e.g. 56)')
    parser.add_argument('-d', '--dataset',     type=str,   required=True,          help='Dataset: BCI | MIST_ER | MIST_HER2 | MIST_Ki67 | MIST_PR')
    parser.add_argument('-o', '--output',      type=str,   default=None,           help='Output dir for predicted images. Default: <model>/test_out_<dataset>')
    parser.add_argument('--ckpt',              type=str,   default=None,           help='Checkpoint path. Default: <model>/pnt_<dataset>/best.pt')
    parser.add_argument('--base',              type=str,   default=None,           help='Dataset base path. Default: $VSC_SCRATCH/datasets')
    parser.add_argument('--steps',             type=int,   default=2000,           help='Diffusion training steps (must match training config)')
    parser.add_argument('--sample_steps',      type=int,   default=100,            help='Diffusion sampling steps')
    parser.add_argument('--batch_size',        type=int,   default=4)
    parser.add_argument('--workers',           type=int,   default=4)
    parser.add_argument('--crop_size',         type=int,   default=256)
    parser.add_argument('--device',            type=str,   default=None,           help='Device override (e.g. cpu, cuda:1). Default: auto')
    return parser.parse_args()


def main():
    args = get_parser()

    # device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')

    # dataset paths
    base = args.base or os.path.join(os.environ.get('VSC_SCRATCH', ''), 'datasets')
    if args.dataset == 'BCI':
        test_he_path  = f'{base}/BCI/HE/test'
        test_ihc_path = f'{base}/BCI/IHC/test'
    elif 'MIST' in args.dataset:
        marker = args.dataset.split('_')[1]
        root   = f'{base}/MIST/{marker}/TrainValAB'
        test_he_path  = f'{root}/valA'
        test_ihc_path = f'{root}/valB'
    else:
        print(f'ERROR: unknown dataset: {args.dataset}')
        sys.exit(1)

    # checkpoint
    ckpt_path = args.ckpt or os.path.join(os.getcwd(), args.model, f'pnt_{args.dataset}', 'best.pt')
    if not os.path.exists(ckpt_path):
        print(f'ERROR: checkpoint not found: {ckpt_path}')
        sys.exit(1)
    print(f'[INFO] checkpoint : {ckpt_path}')

    # output dir
    out_dir = args.output or os.path.join(os.getcwd(), args.model, f'test_out_{args.dataset}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'[INFO] output     : {out_dir}')

    # build model
    module    = __import__(args.model + '.model', fromlist=[args.model])
    model_cls = getattr(module, args.model)
    net = model_cls(*args.args, steps=args.steps)
    net = GaussianDiffusion(net, steps=args.steps, sample_steps=args.sample_steps)

    states = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(states['ema_net'], strict=False)
    net = net.to(device)
    net.eval()
    print(f'[INFO] loaded ema_net (epoch {states.get("epoch", "?")})')

    # dataset + loader
    dataset = EvalDataset(test_he_path, test_ihc_path, args.crop_size)
    loader  = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    print(f'[INFO] test samples: {len(dataset)}')

    # inference
    sample_idx = 0
    total_time = 0.0

    with torch.no_grad():
        for img, _ in tqdm(loader, desc='[Infer]'):
            img = torch.clip(img.to(device), 0.0, 1.0)
            x_T = torch.randn_like(img)

            t0 = time.perf_counter()
            x_0 = net.sample(x_T, img)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            for b in range(x_0.shape[0]):
                fname = dataset.img_names[sample_idx + b]
                torchvision.utils.save_image(x_0[b], os.path.join(out_dir, fname))

            sample_idx += x_0.shape[0]

    runtime_ms_per_img = (total_time / len(dataset)) * 1000
    print(f'\n[Done] {len(dataset)} images saved to {out_dir}')
    print(f'[Time] {total_time:.1f}s total  |  {runtime_ms_per_img:.1f}ms per image')


if __name__ == '__main__':
    main()
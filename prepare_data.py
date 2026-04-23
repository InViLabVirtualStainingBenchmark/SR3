"""
Organizes paired HE/IHC images into the folder structure expected by train.py.

Usage:
  python prepare_data.py \
    --he_src  /path/to/HE/images \
    --ihc_src /path/to/IHC/images \
    --dst     ./data \
    --train 8 --val 4 --test 4

Output layout:
  <dst>/train/he/   <dst>/train/ihc/
  <dst>/val/he/     <dst>/val/ihc/
  <dst>/test/he/    <dst>/test/ihc/

Files are renamed to zero-padded integers (0000.png, 0001.png, ...)
so sorted(listdir()) always produces aligned pairs.
"""
import argparse
import os
import shutil


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--he_src',  required=True, help='Source folder of HE images')
    p.add_argument('--ihc_src', required=True, help='Source folder of IHC images')
    p.add_argument('--dst',     required=True, help='Destination root folder')
    p.add_argument('--train',   type=int, default=8,  help='Number of training pairs (default: 8 = 2 full batches)')
    p.add_argument('--val',     type=int, default=4,  help='Number of validation pairs (default: 4 = covers report_img_idx)')
    p.add_argument('--test',    type=int, default=4,  help='Number of test pairs (default: 4)')
    return p.parse_args()


def collect_images(folder):
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )
    return files


def copy_split(pairs, he_src, ihc_src, he_dst, ihc_dst, ext):
    os.makedirs(he_dst,  exist_ok=True)
    os.makedirs(ihc_dst, exist_ok=True)
    for i, (he_file, ihc_file) in enumerate(pairs):
        name = f'{i:04d}{ext}'
        shutil.copy2(os.path.join(he_src,  he_file),  os.path.join(he_dst,  name))
        shutil.copy2(os.path.join(ihc_src, ihc_file), os.path.join(ihc_dst, name))


def main():
    args = parse_args()

    he_files  = collect_images(args.he_src)
    ihc_files = collect_images(args.ihc_src)

    if len(he_files) != len(ihc_files):
        raise ValueError(
            f'HE folder has {len(he_files)} images, '
            f'IHC folder has {len(ihc_files)} — counts must match.'
        )

    n_needed = args.train + args.val + args.test
    if len(he_files) < n_needed:
        raise ValueError(
            f'Requested {n_needed} images ({args.train} train + {args.val} val + '
            f'{args.test} test) but only {len(he_files)} available.'
        )

    pairs = list(zip(he_files, ihc_files))[:n_needed]
    ext   = os.path.splitext(he_files[0])[1].lower()

    splits = {
        'train': pairs[:args.train],
        'val':   pairs[args.train : args.train + args.val],
        'test':  pairs[args.train + args.val :],
    }

    for split, split_pairs in splits.items():
        he_dst  = os.path.join(args.dst, split, 'he')
        ihc_dst = os.path.join(args.dst, split, 'ihc')
        copy_split(split_pairs, args.he_src, args.ihc_src, he_dst, ihc_dst, ext)
        print(f'{split:5s}: {len(split_pairs)} pairs  →  he:  {he_dst}')
        print(f'{"":5s}  {" " * len(str(len(split_pairs)))}         ihc: {ihc_dst}')


if __name__ == '__main__':
    main()

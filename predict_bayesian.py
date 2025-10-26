import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from bayesian_unet import BayesianUNet
from utils.utils import plot_img_and_mask


# -----------------------------
# MC Dropout helpers
# -----------------------------
# dropout activation and deactivation
def enable_mc_dropout(model):
	""" Function to enable the dropout layers during test-time """
	for m in model.modules():
		if m.__class__.__name__.startswith('Dropout'):
			m.train()


def tensor_to_probs(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Convert logits to probabilities.
    logits: (B, C, H, W) or (B, 1, H, W)
    returns probabilities in [0, 1] with same shape.
    """
    if n_classes > 1:
        return torch.softmax(logits, dim=1)
    else:
        return torch.sigmoid(logits)


# -----------------------------
# Original single-pass predictor (hard mask)
# -----------------------------
def predict_img(
    net,
    full_img: Image.Image,
    device,
    out_threshold: float = 0.5,
    newW: int = 128, 
    newH: int = 128
):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, newW, newH, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]),
            mode='bilinear', align_corners=False
        )
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


# -----------------------------
# MC Dropout predictor (probability maps)
# -----------------------------
def predict_img_mc(
    net,
    full_img: Image.Image,
    device,
    newW: int = 128, 
    newH: int = 128,
    mc_passes: int = 20,
):
    """
    Returns a numpy array of shape:
      - binary (classes=1):        (T, H, W)
      - multiclass (classes=C>1):  (T, C, H, W)
    with values in [0, 1].
    """


    # Preprocess once
    img = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, newW, newH, is_mask=False)
    ).unsqueeze(0).to(device=device, dtype=torch.float32)

    H, W = full_img.size[1], full_img.size[0]
    T = int(mc_passes)
    C = net.n_classes if net.n_classes > 1 else 1

    # Container on CPU
    if C > 1:
        all_probs = np.zeros((T, C, H, W), dtype=np.float32)
    else:
        all_probs = np.zeros((T, H, W), dtype=np.float32)

    # Run T stochastic forward passes
    for t in range(T):
        with torch.no_grad():
            enable_mc_dropout(net)  # keep only dropout stochastic
            logits = net(img)  # (1, C, h, w) or (1,1,h,w)
            logits = F.interpolate(
                logits, (H, W), mode='bilinear', align_corners=False
            )
            probs = tensor_to_probs(logits, net.n_classes)  # (1,C,H,W) or (1,1,H,W)

            if C > 1:
                all_probs[t] = probs.squeeze(0).cpu().numpy()          # (C,H,W)
            else:
                all_probs[t] = probs.squeeze(0).squeeze(0).cpu().numpy()  # (H,W)

    return all_probs


# -----------------------------
# IO helpers
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images (with optional MC Dropout)')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', required=True,
                        help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white (binary)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling in UNet')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes in the model')

    # MC Dropout options
    parser.add_argument('--mc', type=int, default=20,
                        help='Number of MC Dropout passes (T). If <=1, runs once deterministically.')
    parser.add_argument('--save-prob', action='store_true',
                        help='Save probability maps as .npy in addition to PNGs.')
    parser.add_argument('--save-mean-std', action='store_true',
                        help='Also save mean and std maps across MC passes.')
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'
    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def save_png01(arr01: np.ndarray, path: str):
    """Save a [0,1] float array as PNG."""
    arr = np.clip(arr01, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


# -----------------------------
# main
# -----------------------------
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = BayesianUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    # Device selection: prefer CUDA, then MPS, else CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        if args.mc and args.mc > 1:
            # MC Dropout path → probability maps
            probs = predict_img_mc(
                net=net,
                full_img=img,
                device=device,
                mc_passes=args.mc
            )
            # probs shapes:
            #  - binary: (T, H, W)
            #  - multiclass: (T, C, H, W)

            base = os.path.splitext(get_output_filenames(args)[i])[0]

            # Save PNGs for each pass/channel
            if net.n_classes > 1:
                T, C, H, W = probs.shape
                for t in range(T):
                    for c in range(C):
                        save_png01(probs[t, c], f'{base}_mc{t:02d}_c{c}.png')
            else:
                T, H, W = probs.shape
                for t in range(T):
                    save_png01(probs[t], f'{base}_mc{t:02d}.png')

            # Save npy if requested
            if args.save_prob:
                npy_path = f'{base}_probs.npy'
                np.save(npy_path, probs)
                logging.info(f'All probability maps saved to {npy_path}')

            # Optional summaries (mean/std across T)
            if args.save_mean_std:
                if net.n_classes > 1:
                    mean_map = probs.mean(axis=0)  # (C,H,W)
                    std_map = probs.std(axis=0)   # (C,H,W)
                    for c in range(mean_map.shape[0]):
                        save_png01(mean_map[c], f'{base}_mean_c{c}.png')
                        save_png01(std_map[c], f'{base}_std_c{c}.png')
                else:
                    mean_map = probs.mean(axis=0)  # (H,W)
                    std_map = probs.std(axis=0)   # (H,W)
                    save_png01(mean_map, f'{base}_mean.png')
                    save_png01(std_map, f'{base}_std.png')

            if args.viz:
                logging.info(f'Visualizing MC mean for {filename} (close figure to continue)...')
                if net.n_classes == 1:
                    mean_map = probs.mean(axis=0)
                    hard_mask = (mean_map > args.mask_threshold).astype(np.uint8)
                    plot_img_and_mask(img, hard_mask)

        else:
            # Deterministic single forward → hard mask
            mask = predict_img(
                net=net,
                full_img=img,
                out_threshold=args.mask_threshold,
                device=device
            )

            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)


# python predict_bayesian.py -i ./data/imgs_test/ISIC_0000023.jpg -o ./data/test_uq/ISIC_0000023/ISIC_0000023.png --mc 20 --save-prob -m ./checkpoints/checkpoint_epoch20.pth
# python predict_bayesian.py -i ./data/imgs_test/ISIC_0000056.jpg -o ./data/test_uq/ISIC_0000056/ISIC_0000056.png --mc 20 --save-prob -m ./checkpoints/checkpoint_epoch20.pth
# python predict_bayesian.py -i ./data/imgs_test/ISIC_0000174.jpg -o ./data/test_uq/ISIC_0000174/ISIC_0000174.png --mc 20 --save-prob -m ./checkpoints/checkpoint_epoch20.pth



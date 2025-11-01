import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image

from utils.data_loading import BasicDataset
from bayesian_unet import BayesianUNet
from utils.utils import plot_img_and_mask  # kept if used elsewhere


# ------------------------- Core helpers -------------------------

def tensor_to_probs(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert logits to probabilities. Shape-preserving."""
    if n_classes > 1:
        return torch.softmax(logits, dim=1)
    else:
        return torch.sigmoid(logits)


def save_png01(arr01: np.ndarray, path: Path):
    """Save a [0,1] float (H,W) as PNG."""
    arr = np.clip(arr01, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def forward_logits_resized(
    net: torch.nn.Module,
    img_t: torch.Tensor,
    out_hw: tuple[int, int]
) -> torch.Tensor:
    """
    Forward pass -> logits upsampled to (H, W).
    img_t: (1, C, newH, newW)
    returns: (1, C_or_1, H, W)
    """
    logits = net(img_t)
    logits = F.interpolate(logits, size=out_hw, mode='bilinear', align_corners=False)
    return logits


# ------------------------- MC Dropout utilities -------------------------

# dropout activation and deactivation
def enable_mc_dropout(model):
	""" Function to enable the dropout layers during test-time """
	for m in model.modules():
		if m.__class__.__name__.startswith('Dropout'):
			m.train()


# ------------------------- MC Dropout inference -------------------------

def mc_dropout_average_probability(
    net: torch.nn.Module,
    img: Image.Image,
    device: torch.device,
    n_classes: int,
    class_index: int | None,
    mc_samples: int = 50,
    newW: int = 128,
    newH: int = 128,
) -> np.ndarray:
    """
    Run MC Dropout with `mc_samples` forward passes and return averaged (H,W) prob map in [0,1].
    Multi-class: select class_index each pass then average. Binary: average sigmoid.
    """
    # Prepare input once
    H, W = img.size[1], img.size[0]
    img_t = torch.from_numpy(
        BasicDataset.preprocess(None, img, newW, newH, is_mask=False)
    ).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, C, newH, newW)

    # Enable MC dropout behavior
    net.eval()
    enable_mc_dropout(net)

    acc = torch.zeros((1, 1 if n_classes == 1 else n_classes, H, W), dtype=torch.float32)

    with torch.no_grad():
        for _ in range(mc_samples):
            logits = forward_logits_resized(net, img_t, (H, W))  # (1, C_or_1, H, W)
            probs = tensor_to_probs(logits, n_classes)           # same shape
            acc += probs.cpu()

    mean_probs = acc / float(mc_samples)  # (1, C_or_1, H, W)

    if n_classes > 1:
        # choose channel (default to foreground 1 for 2-class setups)
        ci = 1 if (class_index is None) else class_index
        if not (0 <= ci < n_classes):
            raise ValueError(f"class_index={ci} out of range for n_classes={n_classes}")
        out = mean_probs[:, ci:ci+1, :, :].squeeze().numpy().astype(np.float32)  # (H,W)
    else:
        out = mean_probs.squeeze().numpy().astype(np.float32)  # (H,W)

    out = np.clip(out, 0.0, 1.0)
    return out


# ------------------------- IO / CLI -------------------------

def get_args():
    parser = argparse.ArgumentParser(description='Batch predict masks with MC Dropout (UQ) averaging')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Path to the trained model .pth file')
    parser.add_argument('--input-dir', required=True, help='Directory containing input images')
    parser.add_argument('--output-dir', required=True, help='Directory to save averaged probability maps')
    parser.add_argument('--ext', nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'],
                        help='Image extensions to include (case-insensitive)')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling flag in your UNet constructor (if needed)')
    parser.add_argument('--classes', '-c', type=int, default=2,
                        help='Number of classes in the model (1 for sigmoid, >1 for softmax)')
    parser.add_argument('--class-index', type=int, default=1,
                        help='For multi-class (C>1), which class probability to save (0..C-1)')
    parser.add_argument('--recursive', action='store_true', help='Recurse into subdirectories')
    parser.add_argument('--newW', type=int, default=128, help='Network input width')
    parser.add_argument('--newH', type=int, default=128, help='Network input height')
    parser.add_argument('--mc', type=int, default=50, help='Number of MC dropout samples')
    return parser.parse_args()


def list_images(in_dir: Path, exts, recursive: bool):
    exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    pattern = '**/*' if recursive else '*'
    files = [p for p in in_dir.glob(pattern) if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def out_name_for(in_path: Path, out_dir: Path) -> Path:
    return out_dir / f'{in_path.stem}_OUT{in_path.suffix if in_path.suffix.lower() in {".png"} else ".png"}'.replace('..', '.')


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(in_dir, args.ext, args.recursive)
    if not files:
        logging.error(f'No images found in {in_dir} with extensions {args.ext}')
        raise SystemExit(1)

    # Build and load model
    net = BayesianUNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)

    state = torch.load(args.model, map_location=device)
    # Some checkpoints may include extra keys (like 'mask_values'); remove if present
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if isinstance(state, dict) and 'mask_values' in state:
        state.pop('mask_values', None)

    net.load_state_dict(state, strict=True)
    logging.info('Model loaded!')

    # Warn if class-index not meaningful
    if args.classes <= 1 and args.class_index not in (0, 1):
        logging.warning('Binary model detected (classes=1). --class-index is ignored.')

    for f in files:
        try:
            logging.info(f'Predicting with MC Dropout (mc={args.mc}): {f} ...')
            img = Image.open(f).convert('RGB')

            prob_hw = mc_dropout_average_probability(
                net=net,
                img=img,
                device=device,
                n_classes=args.classes,
                class_index=(args.class_index if args.classes > 1 else None),
                mc_samples=args.mc,
                newW=args.newW,
                newH=args.newH,
            )

            out_path = out_name_for(f, out_dir)
            save_png01(prob_hw, out_path)
            logging.info(f'Avg prob saved to {out_path}')

        except Exception as e:
            logging.exception(f'Failed on {f}: {e}')


if __name__ == '__main__':
    main()



# python predict_prob_bayesian.py --input-dir ./data/imgs_test --output-dir ./data/probs_test_bayesian -m ./checkpoints/checkpoint_epoch20.pth
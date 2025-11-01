import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from bayesian_unet import BayesianUNet
from utils.utils import plot_img_and_mask  # (kept if you use it elsewhere; safe to remove if unused)


def tensor_to_probs(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    Convert logits to probabilities.
    logits: (B, C, H, W) for multi-class OR (B, 1, H, W) for binary
    returns probabilities with same shape.
    """
    if n_classes > 1:
        return torch.softmax(logits, dim=1)
    else:
        return torch.sigmoid(logits)


def save_png01(arr01: np.ndarray, path: Path):
    """
    Save a [0,1] float array of shape (H, W) as PNG.
    """
    arr = np.clip(arr01, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)  # (H, W) uint8
    Image.fromarray(arr).save(path)


def predict_img(
    net: torch.nn.Module,
    full_img: Image.Image,
    device: torch.device,
    newW: int = 128,
    newH: int = 128,
    class_index: int | None = None
) -> np.ndarray:
    """
    Run a forward pass and return a (H, W) probability map.
    - For multi-class (C>1), returns the selected class_index probability.
    - For binary (C=1), returns the sigmoid probability.
    """
    net.eval()

    # Preprocess -> (C, newH, newW), float32
    img_t = torch.from_numpy(
        BasicDataset.preprocess(None, full_img, newW, newH, is_mask=False)
    ).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, C, newH, newW)

    with torch.no_grad():
        logits = net(img_t).cpu()  # (1, C, h', w') or (1, 1, h', w')
        # Upsample logits back to original HxW
        logits = F.interpolate(
            logits,
            (full_img.size[1], full_img.size[0]),  # (H, W)
            mode='bilinear',
            align_corners=False
        )

        probs = tensor_to_probs(logits, net.n_classes)  # same shape as logits

        if net.n_classes > 1:
            # choose one channel -> (1, H, W)
            if class_index is None:
                class_index = 1  # default to "foreground" if using 2-class softmax
            if not (0 <= class_index < probs.shape[1]):
                raise ValueError(
                    f"class_index={class_index} is out of range for n_classes={net.n_classes}"
                )
            probs = probs[:, class_index, :, :]  # (1, H, W)
            out = probs.squeeze(0).numpy()       # (H, W)
        else:
            # single logit -> sigmoid -> (1, 1, H, W)
            out = probs.squeeze(0).squeeze(0).numpy()  # (H, W)

    return out  # (H, W) in [0,1]


def get_args():
    parser = argparse.ArgumentParser(description='Batch predict masks for all images in a directory')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Path to the trained model .pth file')
    parser.add_argument('--input-dir', required=True, help='Directory containing input images')
    parser.add_argument('--output-dir', required=True, help='Directory to save predicted masks')
    parser.add_argument('--ext', nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'],
                        help='Image extensions to include (case-insensitive)')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling in UNet')
    parser.add_argument('--classes', '-c', type=int, default=2,
                        help='Number of classes in the model (1 for sigmoid, >1 for softmax)')
    parser.add_argument('--class-index', type=int, default=1,
                        help='For multi-class models (C>1), which class probability to save (0..C-1)')
    parser.add_argument('--recursive', action='store_true',
                        help='Recurse into subdirectories')
    return parser.parse_args()


def list_images(in_dir: Path, exts, recursive: bool):
    exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    pattern = '**/*' if recursive else '*'
    files = [p for p in in_dir.glob(pattern) if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def out_name_for(in_path: Path, out_dir: Path) -> Path:
    return out_dir / f'{in_path.stem}_OUT.png'


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
        state.pop('mask_values')

    net.load_state_dict(state, strict=True)
    logging.info('Model loaded!')

    # Warn if class-index not meaningful
    if args.classes <= 1 and args.class_index != 0 and args.class_index != 1:
        logging.warning('Binary model detected (classes=1). --class-index is ignored.')

    for f in files:
        try:
            logging.info(f'Predicting image {f} ...')
            img = Image.open(f).convert('RGB')

            prob_hw = predict_img(
                net=net,
                full_img=img,
                device=device,
                class_index=(args.class_index if args.classes > 1 else None)
            )

            out_path = out_name_for(f, out_dir)
            save_png01(prob_hw, out_path)

            logging.info(f'Prob saved to {out_path}')

        except Exception as e:
            logging.exception(f'Failed on {f}: {e}')


if __name__ == '__main__':
    main()


# python predict_prob_direct.py --input-dir ./data/imgs_test --output-dir ./data/probs_test_direct -m ./checkpoints/checkpoint_epoch20.pth
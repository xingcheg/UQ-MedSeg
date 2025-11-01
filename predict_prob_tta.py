import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


# ------------------------- D4 (8-way) transforms -------------------------
# Each transform is represented as an ordered list of elementary ops.
# We use center-preserving, exact pixel permutations via numpy (no resampling).
# Ops: ('rot', angle in {0,90,180,270}), ('flip','h') for horizontal flip.

D4_OPS = [
    [],                                     # 0) identity
    [('rot', 90)],                          # 1) rot90
    [('rot', 180)],                         # 2) rot180
    [('rot', 270)],                         # 3) rot270
    [('flip', 'h')],                        # 4) hflip
    [('flip', 'h'), ('rot', 90)],           # 5) hflip -> rot90
    [('flip', 'h'), ('rot', 180)],          # 6) hflip -> rot180
    [('flip', 'h'), ('rot', 270)],          # 7) hflip -> rot270
]


def invert_ops(ops):
    """Inverse of a sequence of ops: reverse order, invert each."""
    inv = []
    for kind, val in reversed(ops):
        if kind == 'rot':
            inv.append(('rot', (360 - val) % 360))
        elif kind == 'flip':
            # horizontal flip is its own inverse
            inv.append(('flip', val))
        else:
            raise ValueError(f'Unknown op: {kind}')
    return inv


def apply_ops_array(arr: np.ndarray, ops):
    """
    Apply ops to a numpy array. Works for (H,W) or (H,W,C).
    Rotations use np.rot90; horizontal flip uses np.flip along axis=1.
    """
    out = arr
    for kind, val in ops:
        if kind == 'rot':
            if val == 0:
                pass
            elif val == 90:
                out = np.rot90(out, k=1, axes=(0, 1))
            elif val == 180:
                out = np.rot90(out, k=2, axes=(0, 1))
            elif val == 270:
                out = np.rot90(out, k=3, axes=(0, 1))
            else:
                raise ValueError('Rotation must be 0/90/180/270')
        elif kind == 'flip':
            if val == 'h':
                out = np.flip(out, axis=1)  # left-right
            else:
                raise ValueError("Only horizontal ('h') flip is supported")
        else:
            raise ValueError(f'Unknown op: {kind}')
    return out


def apply_ops_image(img: Image.Image, ops):
    """Apply D4 ops to a PIL image using exact numpy permutations."""
    arr = np.array(img)
    arr2 = apply_ops_array(arr, ops)
    return Image.fromarray(arr2)


# ------------------------- Shrink/Unshrink (center) -------------------------

def shrink_to_center(img: Image.Image, scale: float = 0.8) -> Image.Image:
    """
    Shrink image by `scale` and paste centered on a black canvas of original size.
    """
    W, H = img.size
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    small = img.resize((newW, newH), resample=Image.BILINEAR)

    canvas = Image.new('RGB', (W, H), color=(0, 0, 0))
    x0 = (W - newW) // 2
    y0 = (H - newH) // 2
    canvas.paste(small, (x0, y0))
    return canvas


def unshrink_from_center(prob_hw: np.ndarray, scale: float = 0.8) -> np.ndarray:
    """
    Inverse of shrink_to_center for a (H,W) float array:
    scale up by 1/scale about the center, then center-crop back to (H,W).
    """
    H, W = prob_hw.shape
    # Enlarge
    inv = 1.0 / scale
    bigW = max(1, int(round(W * inv)))
    bigH = max(1, int(round(H * inv)))

    # PIL in 'F' mode for float; keep range [0,1]
    im = Image.fromarray(prob_hw.astype(np.float32), mode='F')
    big = im.resize((bigW, bigH), resample=Image.BILINEAR)

    # Center-crop to (W,H)
    left = max(0, (bigW - W) // 2)
    upper = max(0, (bigH - H) // 2)
    right = left + W
    lower = upper + H
    big_cropped = big.crop((left, upper, right, lower))
    return np.array(big_cropped, dtype=np.float32)


# ------------------------- Model inference (single pass) -------------------------

def predict_img(
    net: torch.nn.Module,
    full_img: Image.Image,
    device: torch.device,
    newW: int = 128,
    newH: int = 128,
    class_index: int | None = None
) -> np.ndarray:
    """
    Forward pass and return a (H,W) probability map in [0,1].
    Multi-class: select class_index. Binary: sigmoid.
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
            if class_index is None:
                class_index = 1  # default foreground for 2-class setup
            if not (0 <= class_index < probs.shape[1]):
                raise ValueError(
                    f"class_index={class_index} is out of range for n_classes={net.n_classes}"
                )
            out = probs[:, class_index, :, :].squeeze(0).numpy().astype(np.float32)  # (H,W)
        else:
            out = probs.squeeze(0).squeeze(0).numpy().astype(np.float32)  # (H,W)

    return out  # (H, W) in [0,1]


# ------------------------- TTA (16-way) wrapper -------------------------

def tta_average_probability(
    net: torch.nn.Module,
    img: Image.Image,
    device: torch.device,
    class_index: int | None,
    newW: int = 128,
    newH: int = 128,
    shrink_scale: float = 0.8,
) -> np.ndarray:
    """
    16-way TTA:
      - 8 dihedral transforms (D4)
      - same 8 after shrinking to center by `shrink_scale`
    For each, predict, inverse-transform the prob map, and average.
    """
    W, H = img.size
    acc = np.zeros((H, W), dtype=np.float64)
    n = 0

    for ops in D4_OPS:
        # ---- Base 8 (no shrink) ----
        img_t = apply_ops_image(img, ops)
        prob = predict_img(net, img_t, device, newW=newW, newH=newH, class_index=class_index)
        # inverse ops
        inv_prob = apply_ops_array(prob, invert_ops(ops))
        acc += inv_prob
        n += 1

        # ---- Shrunk 8 ----
        img_t_shrink = shrink_to_center(img_t, scale=shrink_scale)
        prob_shrink = predict_img(net, img_t_shrink, device, newW=newW, newH=newH, class_index=class_index)
        # inverse shrink, then inverse ops (order commutes about center, but this is conceptually correct)
        prob_unshrunk = unshrink_from_center(prob_shrink, scale=shrink_scale)
        inv_prob_shrink = apply_ops_array(prob_unshrunk, invert_ops(ops))
        acc += inv_prob_shrink
        n += 1

    avg = (acc / float(n)).astype(np.float32)
    avg = np.clip(avg, 0.0, 1.0)
    return avg


# ------------------------- IO / CLI -------------------------

def get_args():
    parser = argparse.ArgumentParser(description='Batch predict masks for all images in a directory (with 16-way TTA)')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Path to the trained model .pth file')
    parser.add_argument('--input-dir', required=True, help='Directory containing input images')
    parser.add_argument('--output-dir', required=True, help='Directory to save predicted probability maps')
    parser.add_argument('--ext', nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'],
                        help='Image extensions to include (case-insensitive)')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling in UNet (if your model expects this flag)')
    parser.add_argument('--classes', '-c', type=int, default=2,
                        help='Number of classes in the model (1 for sigmoid, >1 for softmax)')
    parser.add_argument('--class-index', type=int, default=1,
                        help='For multi-class models (C>1), which class probability to save (0..C-1)')
    parser.add_argument('--recursive', action='store_true',
                        help='Recurse into subdirectories')
    parser.add_argument('--newW', type=int, default=128, help='Network input width')
    parser.add_argument('--newH', type=int, default=128, help='Network input height')
    parser.add_argument('--shrink', type=float, default=0.8, help='Shrink scale (e.g., 0.8 means 20%% smaller)')
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
        state.pop('mask_values')

    net.load_state_dict(state, strict=True)
    logging.info('Model loaded!')

    # Warn if class-index not meaningful
    if args.classes <= 1 and args.class_index not in (0, 1):
        logging.warning('Binary model detected (classes=1). --class-index is ignored.')

    for f in files:
        try:
            logging.info(f'Predicting with 16-way TTA: {f} ...')
            img = Image.open(f).convert('RGB')

            prob_hw = tta_average_probability(
                net=net,
                img=img,
                device=device,
                class_index=(args.class_index if args.classes > 1 else None),
                newW=args.newW,
                newH=args.newH,
                shrink_scale=args.shrink,
            )

            out_path = out_name_for(f, out_dir)
            save_png01(prob_hw, out_path)
            logging.info(f'Avg prob saved to {out_path}')

        except Exception as e:
            logging.exception(f'Failed on {f}: {e}')


if __name__ == '__main__':
    main()


# python predict_prob_tta.py --input-dir ./data/imgs_test --output-dir ./data/probs_test_tta -m ./checkpoints/checkpoint_epoch20.pth
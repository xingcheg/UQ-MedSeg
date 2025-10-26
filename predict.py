import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.data_loading import BasicDataset
from bayesian_unet import BayesianUNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img: Image.Image,
                device,
                out_threshold=0.5,
                newW: int = 128, 
                newH: int = 128):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, newW, newH, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        # Upsample logits back to original HxW
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear', align_corners=False)
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Batch predict masks for all images in a directory')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Path to the trained model .pth file')
    parser.add_argument('--input-dir', required=True, help='Directory containing input images')
    parser.add_argument('--output-dir', required=True, help='Directory to save predicted masks')
    parser.add_argument('--ext', nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'],
                        help='Image extensions to include (case-insensitive)')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize each image+mask as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Threshold for binary mask when n_classes=1')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling in UNet')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes in the model')
    # Optional: process subfolders too
    parser.add_argument('--recursive', action='store_true', help='Recurse into subdirectories')
    return parser.parse_args()


def list_images(in_dir: Path, exts, recursive: bool):
    exts = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    pattern = '**/*' if recursive else '*'
    files = [p for p in in_dir.glob(pattern) if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def out_name_for(in_path: Path, out_dir: Path) -> Path:
    return out_dir / f'{in_path.stem}_OUT.png'


def mask_to_image(mask: np.ndarray, mask_values):
    # Same behavior as your original helper
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


if __name__ == '__main__':
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
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    for f in files:
        try:
            logging.info(f'Predicting image {f} ...')
            # Ensure 3 channels
            img = Image.open(f).convert('RGB')

            mask = predict_img(
                net=net,
                full_img=img,
                out_threshold=args.mask_threshold,
                device=device
            )

            if not args.no_save:
                out_path = out_name_for(f, out_dir)
                result = mask_to_image(mask, mask_values)
                result.save(out_path)
                logging.info(f'Mask saved to {out_path}')

            if args.viz:
                logging.info(f'Visualizing results for image {f}, close the window to continue...')
                plot_img_and_mask(img, mask)

        except Exception as e:
            logging.exception(f'Failed on {f}: {e}')



# python predict.py --input-dir ./data/imgs_test --output-dir ./data/masks_test_pred -m ./checkpoints/checkpoint_epoch20.pth
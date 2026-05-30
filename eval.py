"""Unified full-reference metrics for TriPS (shared by TriPS-T grid search and root inference).

Tags:
  psnr, ssim, fid,
  lpips_FLAIR / lpips_flair       -> LPIPS(vgg) on full-resolution images
  lpips_FlowDPS / lpips_flowdps   -> LPIPS(vgg) on 224x224-resized images

(Both casings are registered so all callers work with a single eval.py.)

CLI:
  python eval.py --path1 <recon_dir> --path2 <label_dir> --metric psnr ssim lpips_flair
"""
import argparse
from typing import List
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as skimage_ssim
import lpips
from pytorch_fid import fid_score


def tag(name: str):
    def wrapper(func):
        func.tag = name
        return func
    return wrapper


class Factory(object):
    def __init__(self, name: List[str]):
        self.name = name
        methods = {f for f in dir(self) if callable(getattr(self, f)) and hasattr(getattr(self, f), 'tag')}
        self.tagged_method = {getattr(self, f).tag: getattr(self, f) for f in methods}
        self._call_func = self.get_method(name)

    def retrieve(self, input_dir, pred_dir):
        input_path = sorted(list(Path(input_dir).glob('*.png'))) + sorted(list(Path(input_dir).glob('*.jpg')))
        pred_path = sorted(list(Path(pred_dir).glob('*.png'))) + sorted(list(Path(pred_dir).glob('*.jpg')))
        return input_path, pred_path

    def __call__(self, *args, **kwargs):
        return [f(*args, **kwargs) for f in self._call_func]

    def get_method(self, name: List[str]):
        methods = []
        for n in name:
            if n not in self.tagged_method:
                raise ValueError(f'Cannot find {self.__class__.__name__} ({n})')
            methods.append(self.tagged_method[n])
        return methods


class Metric(Factory):
    @tag('psnr')
    def _psnr(self, input_path, pred_path, transform=None, data_range: int = 255, **kwargs):
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = np.array(transform(Image.open(in_f).convert('RGB'))) * data_range
                img2 = np.array(transform(Image.open(pred_f).convert('RGB'))) * data_range
                values.append(psnr(img1, img2, data_range=data_range))
            except Exception:
                continue
        return float(np.mean(values)) if values else float('nan')

    @tag('ssim')
    def _ssim(self, input_path, pred_path, data_range: int = 255, **kwargs):
        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = np.array(Image.open(in_f).convert('RGB'), dtype=np.uint8)
                img2 = np.array(Image.open(pred_f).convert('RGB'), dtype=np.uint8)
                try:
                    score = skimage_ssim(img1, img2, data_range=data_range, channel_axis=-1)
                except TypeError:
                    score = skimage_ssim(img1, img2, data_range=data_range, multichannel=True)
                values.append(float(score))
            except Exception:
                continue
        return float(np.mean(values)) if values else float('nan')

    @tag('fid')
    def _fid(self, pred_path, label_path, **kwargs):
        return fid_score.calculate_fid_given_paths([str(pred_path), str(label_path)], 50, 'cuda', 2048).item()

    # ---- LPIPS (shared core, two resize policies, both name casings) ----
    def _lpips_core(self, input_path, pred_path, resize):
        lpips_fn = lpips.LPIPS(net='vgg').to('cuda').eval()
        ops = []
        if resize:
            ops.append(transforms.Resize((224, 224)))
        ops.append(transforms.ToTensor())
        transform = transforms.Compose(ops)
        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = transform(Image.open(in_f).convert('RGB')).to('cuda')
                img2 = transform(Image.open(pred_f).convert('RGB')).to('cuda')
                values.append(lpips_fn(img1, img2).item())
            except Exception:
                continue
        return float(np.mean(values)) if values else float('nan')

    @tag('lpips_FLAIR')
    def _lpips_FLAIR(self, input_path, pred_path, **kwargs):
        return self._lpips_core(input_path, pred_path, resize=False)

    @tag('lpips_flair')
    def _lpips_flair(self, input_path, pred_path, **kwargs):
        return self._lpips_core(input_path, pred_path, resize=False)

    @tag('lpips_FlowDPS')
    def _lpips_FlowDPS(self, input_path, pred_path, **kwargs):
        return self._lpips_core(input_path, pred_path, resize=True)

    @tag('lpips_flowdps')
    def _lpips_flowdps(self, input_path, pred_path, **kwargs):
        return self._lpips_core(input_path, pred_path, resize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=Path, help='reconstruction dir')
    parser.add_argument('--path2', type=Path, help='ground-truth (label) dir')
    parser.add_argument('--metric', type=str, nargs='+', default=['psnr', 'ssim', 'lpips_flair'])
    args = parser.parse_args()
    metric = Metric(args.metric)
    for m, o in zip(args.metric, metric(args.path1, args.path2)):
        print(f'{m}: {o}')

# import argparse
# from typing import List
# from pathlib import Path
# import torch
# from torchvision import transforms
# import numpy as np
# from PIL import Image
# from skimage.metrics import peak_signal_noise_ratio as psnr
# import lpips
# from pytorch_fid import fid_score
# from pytorch_msssim import ssim
# def tag(name:str):
#     def wrapper(func):
#         func.tag = name
#         return func
#     return wrapper
# class Factory(object):
#     def __init__(self, name: List[str]):
#         self.name = name
#         methods = {func for func in dir(self) if callable(getattr(self, func)) and hasattr(getattr(self, func), 'tag')}
#         self.tagged_method = {getattr(self, func).tag : getattr(self, func) for func in methods}
#         self._call_func = self.get_method(name)
#     def retrieve(self, input_dir, pred_dir):
#         input_path = sorted(list(Path(input_dir).glob('*.png'))) + sorted(list(Path(input_dir).glob('*.jpg')))
#         pred_path = sorted(list(Path(pred_dir).glob('*.png'))) + sorted(list(Path(pred_dir).glob('*.jpg')))
#         return input_path, pred_path
#     def __call__(self, *args, **kwargs):
#         output = []
#         for _func in self._call_func:
#             output.append(_func(*args, **kwargs))
#         return output
#     def get_method(self, name: List[str]):
#         methods = []
#         for n in name:
#             if n not in self.tagged_method:
#                 raise ValueError(f'Cannot find {self.__class__.__name__} ({n})')
#             else:
#                 methods.append(self.tagged_method[n])
#         return methods
# class Metric(Factory):
#     @tag('psnr')
#     def _psnr(self, input_path, pred_path, transform=None, data_range:int=255, **kwargs):
#         if transform is None:
#             transform = transforms.Compose([
#                 transforms.ToTensor()
#             ])
#         values = []
#         img1 = np.array(transform(Image.open(input_path).convert('RGB'))) * data_range
#         img2 = np.array(transform(Image.open(pred_path).convert('RGB'))) * data_range
#         values.append(psnr(img1, img2, data_range=data_range))
#         return np.mean(values)
#     @tag('ssim')
#     def _ssim(self, input_path, pred_path, transform=None, data_range:int=255, **kwargs):
#         if transform is None:
#             transform = transforms.Compose([
#                 transforms.ToTensor()
#             ])
#         values = []
#         img1 = transform(Image.open(input_path).convert('RGB')).unsqueeze(0) * data_range
#         img2 = transform(Image.open(pred_path).convert('RGB')).unsqueeze(0) * data_range
#         values.append(ssim(img1, img2).item())
#         return np.mean(values)
#     @tag('fid')
#     def _fid(self, pred_path, label_path, **kwargs):
#         return fid_score.calculate_fid_given_paths([str(pred_path), str(label_path)], 50, 'cuda', 2048).item()
#     @tag('lpips')
#     def _lpips(self, input_path, pred_path, transform=None, **kwargs):
#         lpips_fn = lpips.LPIPS(net='vgg').to('cuda').eval()
#         if transform is None:
#             transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 # transforms.Resize((256, 256)),
#                 transforms.ToTensor()
#             ])
#         values = []
#         img1 = transform(Image.open(input_path).convert('RGB')).to('cuda')
#         img2 = transform(Image.open(pred_path).convert('RGB')).to('cuda')
#         values.append(lpips_fn(img1, img2).item())
#         return np.mean(values)
#     # patch-wise lpips
#     # @tag('lpips')
#     # def _lpips(self, input_path, pred_path, transform=None, **kwargs):
#     #     lpips_fn = lpips.LPIPS(net='vgg').to('cuda').eval()
#     #     internal_transform = transforms.Compose([
#     #         transforms.ToTensor()
#     #     ])
#     #     img1_full = internal_transform(Image.open(input_path).convert('RGB')).to('cuda')
#     #     img2_full = internal_transform(Image.open(pred_path).convert('RGB')).to('cuda')
#     #     values = []
#     #     patch_size = 256
#     #     num_patches_y = img1_full.shape[1] // patch_size  # 3
#     #     num_patches_x = img1_full.shape[2] // patch_size  # 3
#     #     for i in range(num_patches_y):
#     #         for j in range(num_patches_x):
#     #             y_start = i * patch_size
#     #             x_start = j * patch_size
#     #             y_end = y_start + patch_size
#     #             x_end = x_start + patch_size
#     #             patch1 = img1_full[..., y_start:y_end, x_start:x_end]
#     #             patch2 = img2_full[..., y_start:y_end, x_start:x_end]
#     #             lp_val = lpips_fn(patch1, patch2).item()
#     #             values.append(lp_val)
#     #     return np.mean(values)
import argparse
from typing import List
from pathlib import Path
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from pytorch_fid import fid_score
from pytorch_msssim import ssim
from skimage.metrics import structural_similarity as skimage_ssim
def tag(name:str):
    def wrapper(func):
        func.tag = name
        return func
    return wrapper
class Factory(object):
    def __init__(self, name: List[str]):
        self.name = name
        methods = {func for func in dir(self) if callable(getattr(self, func)) and hasattr(getattr(self, func), 'tag')}
        self.tagged_method = {getattr(self, func).tag : getattr(self, func) for func in methods}
        self._call_func = self.get_method(name)
    def retrieve(self, input_dir, pred_dir):
        input_path = sorted(list(Path(input_dir).glob('*.png'))) + sorted(list(Path(input_dir).glob('*.jpg')))
        pred_path = sorted(list(Path(pred_dir).glob('*.png'))) + sorted(list(Path(pred_dir).glob('*.jpg')))
        return input_path, pred_path
    def __call__(self, *args, **kwargs):
        output = []
        for _func in self._call_func:
            output.append(_func(*args, **kwargs))
        return output
    def get_method(self, name: list[str]):
        methods = []
        for n in name:
            if n not in self.tagged_method:
                raise ValueError(f'Cannot find {self.__class__.__name__} ({n})')
            else:
                methods.append(self.tagged_method[n])
        return methods
class Metric(Factory):
    @tag('psnr')
    def _psnr(self, input_path, pred_path, transform=None, data_range:int=255, **kwargs):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = np.array(transform(Image.open(in_f).convert('RGB'))) * data_range
                img2 = np.array(transform(Image.open(pred_f).convert('RGB'))) * data_range
                values.append(psnr(img1, img2, data_range=data_range))
            except:
                continue
        return np.mean(values)
    # @tag('ssim')
    # def _ssim(self, input_path, pred_path, transform=None, data_range:int=255, **kwargs):
    #     if transform is None:
    #         transform = transforms.Compose([
    #             transforms.ToTensor()
    #         ])
    #     values = []
    #     in_fs, pred_fs = self.retrieve(input_path, pred_path)
    #     for in_f, pred_f in zip(in_fs, pred_fs):
    #         try:
    #             img1 = transform(Image.open(in_f).convert('RGB')).unsqueeze(0) * data_range
    #             img2 = transform(Image.open(pred_f).convert('RGB')).unsqueeze(0) * data_range
    #             # values.append(ssim(img1, img2).item())
    #             values.append(ssim(img1, img2, data_range=data_range, size_average=True).item())
    #         except:
    #             continue
    #     return np.mean(values)
    @tag('ssim')
    def _ssim(self, input_path, pred_path, data_range: int = 255):
        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = np.array(Image.open(in_f).convert("RGB"), dtype=np.uint8)   # (H,W,3), 0..255
                img2 = np.array(Image.open(pred_f).convert("RGB"), dtype=np.uint8) # (H,W,3), 0..255
                try:
                    score = skimage_ssim(img1, img2, data_range=data_range, channel_axis=-1)
                except TypeError:
                    score = skimage_ssim(img1, img2, data_range=data_range, multichannel=True)
                values.append(float(score))
            except Exception as e:
                continue
        return float(np.mean(values)) if values else float("nan")
    @tag('fid')
    def _fid(self, pred_path, label_path, **kwargs):
        return fid_score.calculate_fid_given_paths([str(pred_path), str(label_path)], 50, 'cuda', 2048).item()
    @tag('lpips_FLAIR')
    def _lpips_flair(self, input_path, pred_path, transform=None, **kwargs):
        lpips_fn = lpips.LPIPS(net='vgg').to('cuda').eval()
        if transform is None:
            transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = transform(Image.open(in_f).convert('RGB')).to('cuda')
                img2 = transform(Image.open(pred_f).convert('RGB')).to('cuda')
                values.append(lpips_fn(img1, img2).item())
            except:
                continue
        return np.mean(values)
    @tag('lpips_FlowDPS')
    def _lpips_flowdps(self, input_path, pred_path, transform=None, **kwargs):
        lpips_fn = lpips.LPIPS(net='vgg').to('cuda').eval()
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        values = []
        in_fs, pred_fs = self.retrieve(input_path, pred_path)
        for in_f, pred_f in zip(in_fs, pred_fs):
            try:
                img1 = transform(Image.open(in_f).convert('RGB')).to('cuda')
                img2 = transform(Image.open(pred_f).convert('RGB')).to('cuda')
                values.append(lpips_fn(img1, img2).item())
            except:
                continue
        return np.mean(values)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path1', type=Path)
    parser.add_argument('--path2', type=Path)
    parser.add_argument('--metric', type=str, nargs='+')
    parser.add_argument('--prompt', type=str)
    args = parser.parse_args()
    metric = Metric(args.metric)
    output = metric(args.path1, args.path2, prompt=args.prompt)
    for m, o in zip(args.metric, output):
        print(f'{m}: {o}')
import cv2
import numpy as np
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftshift, ifftshift
import torch

import argparse

from pytorch_msssim import ssim
import math
import pytorch_fid_wrapper as pfw

import lpips

import random
from torchvision import models, transforms
import torch.nn as nn
import os

from PIL import Image



SYNTHETIC = True


from torchvision import models

resize_and_normalize = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.Lambda(lambda x: (x + 1) / 2),  # Rescale from [-1, 1] to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize like VGG input
])

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children()))

    def forward(self, x):
        x = self.features(x)
        #print("Features shape before flatten:", x.shape)  # Debug: check the shape
        if x.shape[-1] != 7 or x.shape[-2] != 7:
            raise ValueError("Feature map size is incorrect, expected 7x7 per channel.")
        x = x.view(x.size(0), -1)
        return x

class VGGClassifier(nn.Module):
    def __init__(self):
        super(VGGClassifier, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.classifier = nn.Sequential(*list(vgg.classifier.children()))

    def forward(self, x):
        return self.classifier(x)
    

def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

def perceptual_loss(img1, img2, feature_extractor, classifier):
    img1 = resize_and_normalize(img1).to('cuda:7')
    img2 = resize_and_normalize(img2).to('cuda:7')
    img1_features = feature_extractor(img1)
    img2_features = feature_extractor(img2)
    
    #print ("img1_features.shape",img1_features.shape)
    img1_fc = classifier(img1_features)
    img2_fc = classifier(img2_features)
    #print ("img1_fc.shape",img1_fc.shape)
    return F.l1_loss(img1_fc, img2_fc)



def calculate_average(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    total_files = len(files)

    sum_df = None
    columns = ['PSNR', 'SSIM', 'FID', 'LPIPS']

    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=None, names=columns)

        if sum_df is None:
            sum_df = df
        else:
            sum_df += df

    average_df = sum_df / total_files

    return average_df

def tv1_loss(x):
    #### here, our input must be {batch, channel, height, width}
    ndims = len(list(x.size()))
    if ndims != 3:
        assert False, "Input must be {channel, height, width}"
    n_pixels = x.size()[0] * x.size()[1] * x.size()[2]
    dh = torch.abs(x[:, 1:, :] - x[:, :-1, :])
    dw = torch.abs(x[:, :, 1:] - x[:, :, :-1])
    tot_var = torch.sum(dh) + torch.sum(dw)
    tot_var = tot_var / n_pixels
    return tot_var

def low_pass(img, size):
    fft_img = torch.fft.fftn(img, dim=(-2, -1))
    fft_img_shifted = torch.fft.fftshift(fft_img, dim=(-2, -1))

    h, w = img.shape[-2:]
    center_h, center_w = h // 2, w // 2

    low_pass_mask = torch.zeros_like(fft_img_shifted)
    low_pass_mask[:, :, center_h-size:center_h+size, center_w-size:center_w+size] = 1

    low_freq_component = fft_img_shifted * low_pass_mask

    norm_factor = torch.max(torch.abs(low_freq_component))
    normalized_low_freq_component = low_freq_component / norm_factor if norm_factor != 0 else low_freq_component


    return normalized_low_freq_component

def low_pass_filter(img, size):
    fft_img = torch.fft.fftn(img, dim=(-2, -1))
    fft_img_shifted = torch.fft.fftshift(fft_img, dim=(-2, -1))

    h, w = img.shape[-2:]
    center_h, center_w = h // 2, w // 2

    low_pass_mask = torch.zeros_like(fft_img_shifted)
    low_pass_mask[:, :, center_h-size:center_h+size, center_w-size:center_w+size] = 1

    low_freq_img = torch.fft.ifftn(torch.fft.ifftshift(fft_img_shifted * low_pass_mask, dim=(-2, -1)), dim=(-2, -1)).real

    return low_freq_img


def high_pass(img, size):

    fft_img = torch.fft.fftn(img, dim=(-2, -1))
    fft_img_shifted = torch.fft.fftshift(fft_img, dim=(-2, -1))

    h, w = img.shape[-2:]
    center_h, center_w = h // 2, w // 2

    high_pass_mask = torch.ones_like(fft_img_shifted)
    high_pass_mask[:, :, center_h-size:center_h+size, center_w-size:center_w+size] = 0

    high_freq_component = fft_img_shifted * high_pass_mask

    norm_factor = torch.max(torch.abs(high_freq_component))
    normalized_high_freq_component = high_freq_component / norm_factor if norm_factor != 0 else high_freq_component

    return normalized_high_freq_component



def high_pass_filter(img, size):
    fft_img = torch.fft.fftn(img, dim=(-2, -1))
    fft_img_shifted = torch.fft.fftshift(fft_img, dim=(-2, -1))

    h, w = img.shape[-2:]
    center_h, center_w = h // 2, w // 2

    high_pass_mask = torch.ones_like(fft_img_shifted)
    high_pass_mask[:, :, center_h-size:center_h+size, center_w-size:center_w+size] = 0

    high_freq_img = torch.fft.ifftn(torch.fft.ifftshift(fft_img_shifted * high_pass_mask, dim=(-2, -1)), dim=(-2, -1)).real

    return high_freq_img

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def apply_edge_mask(kl_dec_mask, percentage):
    height, width = kl_dec_mask.shape
    edge_size_h = int(height * percentage)
    edge_size_w = int(width * percentage)

    if edge_size_h > 0 and edge_size_w > 0:
        kl_dec_mask[:edge_size_h, :] = 0
        kl_dec_mask[-edge_size_h:, :] = 0
        kl_dec_mask[:, :edge_size_w] = 0
        kl_dec_mask[:, -edge_size_w:] = 0

    return kl_dec_mask

def show_metric(img1,img2,device,name):
    
    psnr = round(calculate_psnr(normalize(img1), normalize(img2)), 2)
    ssim = round(calculate_ssim(normalize(img1), normalize(img2)), 3)
    fid = round(calculate_fid(normalize(img1), normalize(img2), device), 2)
    lpips = round(calculate_lpips(normalize(img1), normalize(img2), device), 3)

    print("----------------")
    print("Method= ",name)
    print ("----------------")
    print("PSNR= ",psnr)
    print("SSIM= ", ssim)
    print("FID= ", fid)
    print("LPIPS= ", lpips)
    print ("----------------")

    return psnr,ssim,fid,lpips


def write_results(fname, psnr, ssim, fid, lpips):

    with open(fname, 'w') as file:
        file.write(f"{psnr},{ssim},{fid},{lpips}\n")

    print(f"Results written to {fname}")

def seed_everything(seed=42):

    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def normalize(data):
    return (data-data.min())/(data.max()-data.min())

def normalize_std(data):
    return (data-data.mean())/(data.std())

def conv_fft2(x,y):
    # return torch.real(ifftshift(ifft2(fftshift(fft2(x), dim=(-2,-1)) * fftshift(fft2(y), dim=(-2,-1))), dim=(-2,-1)))
    return torch.abs(ifftshift(ifft2(fftshift(fft2(x), dim=(-2,-1)) * fftshift(fft2(y), dim=(-2,-1))), dim=(-2,-1)))

def conv_psf_fft(x, y):
    
    # Calculate the necessary padding sizes
    pad_left = pad_right = y.size(3) // 2
    pad_top = pad_bottom = y.size(2) // 2

    # Pad the single channel tensor with zeros (constant padding)
    x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    y_padded = F.pad(y, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # Perform the convolution
    output = conv_fft2(x_padded, y_padded)

    # Adjust output_channel size if necessary
    # Assuming output_channel needs to be cropped to match the original size
    output_cropped = center_crop(output, size=(pad_right * 2, pad_bottom * 2))

    # output_cropped = output_cropped / torch.max(output_cropped)
    
    return output_cropped

def transform_centercrop(input_tensor,size):
    cropped_tensor = center_crop(input_tensor, size=(2400,2400))

    # Resize to 512x512
    resized_tensor = resize_tensor(cropped_tensor, size)
    
    return resized_tensor


def transform_padding_centercrop(input_tensor,size):

    cropped_tensor = center_crop(input_tensor, size=(2400,2400))
    if SYNTHETIC:
        cropped_tensor = F.pad(cropped_tensor,(250,250,220,180),"replicate")
    #cropped_tensor = F.pad(cropped_tensor,(250,250,220,180),"replicate")

    # Resize to 512x512
    resized_tensor = resize_tensor(cropped_tensor, size)
    
    return resized_tensor

def transform_padding_centercrop_real(input_tensor,size):

    cropped_tensor = center_crop(input_tensor, size=(2400,2400))
    #cropped_tensor = F.pad(cropped_tensor,(250,250,220,180),"replicate")

    # Resize to 512x512
    resized_tensor = resize_tensor(cropped_tensor, size)
    
    return resized_tensor

def transform_resize_waller_real(input_tensor,size):

    resized_tesnor = input_tensor

    # Calculate padding
    target_size = 512
    height_pad = (target_size - resized_tesnor.size(2)) // 2
    width_pad = (target_size - resized_tesnor.size(3)) // 2
    padded_tensor = F.pad(resized_tesnor, pad=(width_pad, width_pad, height_pad, height_pad), mode = "replicate")
    # breakpoint()
    # padded_tensor = F.pad(cropped_tensor,(250,250,220,180),"replicate")

    # Resize to 512x512
    resized_tensor = resize_tensor(padded_tensor.squeeze(0), size)
    
    return resized_tensor

def increase_brightness(img, value=0.1):

    brighter_img = img + value
    brighter_img = np.clip(brighter_img, 0, 1) 
    return brighter_img

def load_psf(args, psf_file,resize_dim=None,arg_voronoi=1):


    psf = cv2.imread(psf_file, -1).astype(np.float32)
    if arg_voronoi:
        psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
    psf = torch.tensor(psf, dtype=torch.float32)

    if psf_file == './psf_voronoi.png':
        psf [ psf<70] = 0
    else:
        # pass
        psf [ psf<70] = 0
    
    psf = psf/1023.0
    
    psf_crop = center_crop(psf, size=(2400,2400))
    if SYNTHETIC:
        psf_crop = F.pad(psf_crop,(250,250,220,180),"constant",value=0)

    psf = resize_tensor(psf_crop.unsqueeze(0),resize_dim).cuda(args.gpu)

    return psf

def load_psf_real(args, psf_file,resize_dim=None):

    psf = cv2.imread(psf_file, -1).astype(np.float32)
    psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
    psf = torch.tensor(psf, dtype=torch.float32)
    
    psf[psf < 80.0] = 0
    
    psf_crop = center_crop(psf.unsqueeze(0),size=(2400,2400)) # [3,2400,2400]
    psf = resize_tensor(psf_crop,resize_dim).cuda(args.gpu)  # [3,512,512]
    
    #psf = psf/1023.0 
    print ("psf max,min:",psf.max(),psf.min())
    return psf



def load_psf_waller(args, psf_file,resize_dim=None):
    psf = cv2.imread(psf_file, -1).astype(np.float32)
    psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
    psf = torch.tensor(psf, dtype=torch.float32)
    # height_pad = 270
    # width_pad = 480
    # psf = F.pad(psf, pad=(width_pad, width_pad, height_pad, height_pad), mode='constant', value=0)
    psf = resize_tensor(psf.unsqueeze(0),resize_dim).cuda(args.gpu)  # [3,512,512]
    
    #psf = psf/1023.0 
    print ("psf max,min:",psf.max(),psf.min())
    return psf

def load_psf_waller_wiener(args, psf_file,resize_dim=None):
    # Calculate padding
    
    psf = cv2.imread(psf_file, -1).astype(np.float32)
    psf = cv2.cvtColor(psf, cv2.COLOR_BGR2GRAY)
    psf = torch.tensor(psf, dtype=torch.float32)
    # breakpoint()
    psf = resize_tensor(psf.unsqueeze(0),resize_dim).cuda(args.gpu)  # [3,270,480]
    target_size = 480
    height_pad = (target_size - psf.size(1)) // 2 # 105
    padded_psf = F.pad(psf, pad=(0, 0, height_pad, height_pad), mode='constant', value=0)
    resized_padded_psf = resize_tensor(padded_psf,256).cuda(args.gpu)  # [3,270,480]
    
    # psf = psf/255.0 
    # print ("psf max,min:",psf.max(),psf.min())
    return resized_padded_psf



def resize(img, factor):
    num = int(-np.log2(factor))
    for _ in range(num):
        img = 0.25 * (img[::2, ::2] + img[1::2, ::2] +
                      img[::2, 1::2] + img[1::2, 1::2])
    return img



def cyclic_shift_torch(input_tensor, shift_x, shift_y):
    output_tensor = torch.roll(input_tensor, shifts=shift_x, dims=0)
    output_tensor = torch.roll(output_tensor, shifts=shift_y, dims=1) 
    return output_tensor

def to_tensor_no_scaling(pil_image):
    numpy_image = np.array(pil_image)
    tensor_image = torch.tensor(numpy_image, dtype=torch.float32)
    tensor_image = tensor_image.permute(2, 0, 1)
    return tensor_image


def transform_gt(input_tensor):
    # Resize from 500x500 to 1000x1000
    resized_1000 = F.interpolate(input_tensor, size=(1000, 1000), mode='bilinear', align_corners=False)

    # Padding to 2000x2000 (Assuming input_tensor is a 4D Batch x Channels x Height x Width tensor)
    # Calculate padding
    target_size = 2000
    height_pad = (target_size - resized_1000.size(2)) // 2
    width_pad = (target_size - resized_1000.size(3)) // 2
    padded_2000 = F.pad(resized_1000, pad=(width_pad, width_pad, height_pad, height_pad), mode='constant', value=0)

    # Resize from 2000x2000 to 512x512
    resized_512 = F.interpolate(padded_2000, size=(512, 512), mode='bilinear', align_corners=False)

    return resized_512

# waller gt###
def transform_gt_waller(input_tensor):
    # Resize from 500x500 to 1000x1000
    # resized_1000 = F.interpolate(input_tensor, size=(1000, 1000), mode='bilinear', align_corners=False)
    # breakpoint()
    # Padding to 2000x2000 (Assuming input_tensor is a 4D Batch x Channels x Height x Width tensor)
    # Calculate padding
    target_size = 480
    height_pad = (target_size - input_tensor.size(2)) // 2
    width_pad = (target_size - input_tensor.size(3)) // 2
    padded_gt = F.pad(input_tensor, pad=(width_pad, width_pad, height_pad, height_pad), mode='constant', value=0)

    # Resize from 2000x2000 to 512x512
    # breakpoint()
    resized_256 = F.interpolate(padded_gt, size=(256, 256), mode='bilinear', align_corners=False)
        
    return resized_256

#################for size 128(origin 128 + pad 128) ###########################
def transform_gt_256(input_tensor):
    # Resize from 500x500 to 1000x1000
    resized_500 = F.interpolate(input_tensor, size=(500, 500), mode='bilinear', align_corners=False)

    # Padding to 2000x2000 (Assuming input_tensor is a 4D Batch x Channels x Height x Width tensor)
    # Calculate padding
    target_size = 1000
    height_pad = (target_size - resized_500.size(2)) // 2
    width_pad = (target_size - resized_500.size(3)) // 2
    padded_1000 = F.pad(resized_500, pad=(width_pad, width_pad, height_pad, height_pad),mode='constant', value=0) #,mode='circular')

    # Resize from 2000x2000 to 512x512
    resized_256 = F.interpolate(padded_1000, size=(256, 256), mode='bilinear', align_corners=False)
        
    return resized_256

#################for size 128 ###########################

def center_crop(img, center=None, size=(0, 0), mode="crop", **kwargs):
    """
    Crop the input image based on the center and size.

    Parameters:
    - img: Input image (numpy array or torch tensor)
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")

    Returns:
    - output: Cropped image
    """
    if torch.is_tensor(img):
        return center_crop_t(img, center=center, size=size, mode=mode, **kwargs)
    else:
        return center_crop_n(img, center=center, size=size, mode=mode, **kwargs)

def center_crop_n(img, center=None, size=(0, 0), mode="crop"):
    """
    Crop numpy array based on the center and size.

    Parameters:
    - img: Input numpy array
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")

    Returns:
    - output: Cropped numpy array
    """
    img_h, img_w = np.shape(img)[:2]
    crop_h, crop_w = size
    crop_h_half, crop_h_mod = divmod(crop_h, 2)
    crop_w_half, crop_w_mod = divmod(crop_w, 2)

    if center is None:
        crop_center_h = img_h // 2
        crop_center_w = img_w // 2
    else:
        crop_center_h, crop_center_w = center

    if mode == "crop":
        output = img[crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                 crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...]
    elif mode == "same":
        output = np.zeros_like(img)
        output[crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
        crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...] = img[
                                                                                      crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                                                                                      crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod,
                                                                                      ...
                                                                                      ]
    elif mode == "center":
        output = np.zeros_like(img)
        output[img_h - crop_h_half: img_h + crop_h_half + crop_h_mod,
        img_w - crop_w_half: img_w + crop_w_half + crop_w_mod, ...] = img[
                                                                      crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                                                                      crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod,
                                                                      ...
                                                                      ]
    return output

def center_crop_t(img, center=None, size=(0, 0), mode="crop"):
    """
    Crop torch tensor based on the center and size.

    Parameters:
    - img: Input torch tensor
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")

    Returns:
    - output: Cropped torch tensor
    """
    img_h, img_w = img.size()[-2:]
    crop_h, crop_w = size
    crop_h_half, crop_h_mod = divmod(crop_h, 2)
    crop_w_half, crop_w_mod = divmod(crop_w, 2)

    if center is None:
        crop_center_h = img_h // 2
        crop_center_w = img_w // 2
    else:
        crop_center_h, crop_center_w = center

    if mode == "crop":
        output = img[..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                 crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod]
    elif mode == "same":
        output = torch.zeros_like(img)
        output[..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
        crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod] = img[
                                                                                 ...,
                                                                                 crop_center_h - crop_h_half: cropdeconv_psf_center_h + crop_h_half + crop_h_mod,
                                                                                 crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod
                                                                                 ]
    elif mode == "center":
        output = torch.zeros_like(img)
        output[..., img_h - crop_h_half: img_h + crop_h_half + crop_h_mod,
        img_w - crop_w_half: img_w + crop_w_half + crop_w_mod] = img[
                                                                 ...,
                                                                 crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                                                                 crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod
                                                                 ]
    return output

def resize_tensor(input_tensor, size):
    """
    Resize the input tensor to the specified size.
    """
    return F.interpolate(input_tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)


def transform_centercrop(input_tensor,size):
    cropped_tensor = center_crop(input_tensor, size=(2400,2400))

    # Resize to 512x512
    resized_tensor = resize_tensor(cropped_tensor, size)
    
    return resized_tensor

def deconv_psf(x, psf, alpha=0):
    '''
    wiener filtering for monochome raw image
    x     : raw image
    psf   : psf
    alpha : reconstruction hyperparameter (high alpha means more noise)
    '''
    psf_ft = torch.fft.fftn(psf, dim=(-2,-1))
    psf_ft = torch.conj(psf_ft) / (abs(psf_ft)**2 + alpha)
    x_ft = torch.fft.fftn(x, dim=(-2, -1))
    # Wiener deconvolution Operation
    recon = torch.fft.fftshift(torch.fft.ifftn(x_ft * psf_ft, dim=(-2, -1)), dim=(-2, -1))
    recon = torch.real(recon)

    recon = (recon - recon.min()) / (recon.max()-recon.min())

    return recon


def add_gaussian_noise(image, mean=0, std=0.1, device='cuda'):
    noise = torch.randn_like(image, device=device) * std + mean
    noisy_image = image + noise
    return noisy_image



def calculate_psnr(img1, img2):
    # MSE (Mean Squared Error) 계산
    mse = torch.mean((img1 - img2) ** 2)
    # PSNR 계산
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse.item()))
    return psnr

def calculate_ssim(img1, img2):
    ssim_val = ssim(img1, img2, data_range=1.0, size_average=True)
    return ssim_val.item()


def calculate_fid(img1,img2,device):
    pfw.set_config(batch_size=1,device=device)
    fid_score=pfw.fid(img1,img2)
    return fid_score

def calculate_lpips(img1,img2,device,is_alex=False):
    if is_alex:
        loss_fn_vgg = lpips.LPIPS(net='alex',verbose=False).to(device)
    else:
        loss_fn_vgg = lpips.LPIPS(net='vgg',verbose=False).to(device)
    lpips_score = loss_fn_vgg(img1, img2)
    return lpips_score.item()


def crop_and_noise(x_psf, psf, crop_size, noise_size):

    resize_size = x_psf.shape[-1]

    if crop_size:
        cropped_size = int((resize_size- crop_size)/2)
        x_psf = center_crop(x_psf,size=(crop_size,crop_size))
        # x_psf = F.pad(x_psf, (cropped_size, cropped_size, cropped_size, cropped_size), "replicate")
        psf = center_crop(psf,size=(crop_size,crop_size))
        # psf = F.pad(psf, (cropped_size, cropped_size, cropped_size, cropped_size), "replicate")
    if noise_size:
        mean = 0
        noise = torch.randn(x_psf.size()).to(x_psf.device) *noise_size + mean
        x_psf = x_psf + noise
    
    return x_psf,psf

def crop_and_noise_2(x_psf,crop_size,noise_size):

    resize_size = x_psf.shape[-1]

    if crop_size:
        cropped_size = int((resize_size- crop_size)/2)
        x_psf = center_crop(x_psf,size=(crop_size,crop_size))
        x_psf = F.pad(x_psf, (cropped_size, cropped_size, cropped_size, cropped_size), "replicate")
        # psf = center_crop(psf,size=(crop_size,crop_size))
        # psf = F.pad(psf, (cropped_size, cropped_size, cropped_size, cropped_size), "replicate")
    if noise_size:
        mean = 0
        noise = torch.randn(x_psf.size()).to(x_psf.device) *noise_size + mean
        x_psf = x_psf+noise
    
    return x_psf

# input: [1,3,512,512]
def vignetting(input_image,kernel_size):
    # Extracting the height and width of an image
    
    rows, cols = input_image.shape[2:4]

    input_image = input_image.squeeze()

    output = torch.zeros_like(input_image)

    # generating vignette mask using Gaussian 
    # resultant_kernels
    X_resultant_kernel = cv2.getGaussianKernel(cols,kernel_size)
    Y_resultant_kernel = cv2.getGaussianKernel(rows,kernel_size)

    #generating resultant_kernel matrix 
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

    #creating mask and normalising by using np.linalg
    # function
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)

    mask = torch.tensor(mask, dtype=torch.float32)

    mask = mask.to(input_image.device)

    # applying the mask to each channel in the input image
    for i in range(3):
        output[i,:,:] = input_image[i,:,:] * mask

    output = output.unsqueeze(0)

    return output

# input: [1,3,512,512]
def vignetting_inv(input_image,kernel_size):
    # Extracting the height and width of an image
    
    rows, cols = input_image.shape[2:4]

    input_image = input_image.squeeze()

    output = torch.zeros_like(input_image)

    # generating vignette mask using Gaussian 
    # resultant_kernels
    X_resultant_kernel = cv2.getGaussianKernel(cols,kernel_size)
    Y_resultant_kernel = cv2.getGaussianKernel(rows,kernel_size)

    #generating resultant_kernel matrix 
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T

    #creating mask and normalising by using np.linalg
    # function
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)

    mask = torch.tensor(mask, dtype=torch.float32)

    mask = mask.to(input_image.device)

    inverse_mask = 1 / mask 

    inverse_mask = inverse_mask.to(input_image.device)


    # applying the mask to each channel in the input image
    for i in range(3):
        output[i,:,:] = input_image[i,:,:] * inverse_mask

    output = output.unsqueeze(0)

    return output


def count_png_files(directory):

    png_count = 0

    for entry in os.listdir(directory):

        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path) and full_path.endswith('.png'):
            png_count += 1

    return png_count



def save_tensor_as_image_manual(tensor: torch.Tensor, filename: str):

    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)


    tensor = tensor.permute(1, 2, 0)  # shape: [512, 512, 3]

    array_uint8 = (tensor * 255).byte().cpu().numpy()

    img = Image.fromarray(array_uint8)
    img.save(filename)
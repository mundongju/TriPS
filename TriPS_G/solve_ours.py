import argparse
from pathlib import Path
from typing import List
from munch import munchify
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torchvision import transforms
from util import set_seed, get_img_list, process_text
from sd3_sampler_ours_test import get_solver
# from sd3_sampler_tt import get_solver
# from functions_backup.degradation import get_degradation
from eval import Metric
from torchvision.utils import make_grid
import torchvision.transforms as T
# from custom_util import *
from torch.nn import functional as F
from grpo_schedule import coeff_to_schedule, split_phi, ScheduleBounds
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

def load_grpo_schedules_from_ckpt(
    ckpt_path: str,
    NFE: int,
    device: torch.device,
    repr: str = "mean",          # "mean" | "mode" | "sample"
    sample_seed: int = 0,
    eps: float = 1e-6,
):

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "degree" not in ckpt:
        raise KeyError(f"ckpt missing key 'degree': {ckpt_path}")
    if "bounds" not in ckpt:
        raise KeyError(f"ckpt missing key 'bounds': {ckpt_path}")

    degree = int(ckpt["degree"])
    bounds_dict = ckpt["bounds"]
    bounds = ScheduleBounds(**bounds_dict)

    d = degree + 1

    # -----------------------
    # 1) coeff(phi) 복원
    # -----------------------
    phi_vec = None  # [D_total]

    if "policy_state_dict" in ckpt:
        sd = ckpt["policy_state_dict"]
        if ("log_alpha" in sd) and ("log_beta" in sd):
            log_alpha = sd["log_alpha"].float()
            log_beta  = sd["log_beta"].float()
            alpha = log_alpha.exp().clamp_min(eps)
            beta  = log_beta.exp().clamp_min(eps)
            phi_vec = (alpha / (alpha + beta)).clamp(eps, 1.0 - eps)  # default mean
        elif "mu" in sd:
            phi_vec = sd["mu"].float()
        else:
            raise KeyError(f"policy_state_dict has no log_alpha/log_beta or mu: {ckpt_path}")
    else:
        raise KeyError(f"ckpt missing keys for coeff reconstruction: {ckpt_path}")

    # shape check
    expected_D = 3 * d
    if phi_vec.numel() != expected_D:
        raise ValueError(f"phi dim mismatch: got {phi_vec.numel()}, expected {expected_D} (=3*(degree+1)).")

    # -----------------------
    # 2) phi -> schedules
    # -----------------------
    phi = phi_vec.to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, D_total]
    cfg_phi, step_phi, eta_phi = split_phi(phi, d, d, d)

    cfg_coeff = cfg_phi.squeeze(0)
    step_coeff = step_phi.squeeze(0)
    eta_coeff = eta_phi.squeeze(0)

    cfg_schedule = coeff_to_schedule(
        cfg_coeff, NFE, kind="cfg",
        cfg_min=bounds.cfg_min, cfg_max=bounds.cfg_max,
        step_min=bounds.step_min, step_max=bounds.step_max,
        device=device,
    )
    step_schedule = coeff_to_schedule(
        step_coeff, NFE, kind="step",
        cfg_min=bounds.cfg_min, cfg_max=bounds.cfg_max,
        step_min=bounds.step_min, step_max=bounds.step_max,
        device=device,
    )
    eta_schedule = coeff_to_schedule(
        eta_coeff, NFE, kind="eta",
        cfg_min=bounds.cfg_min, cfg_max=bounds.cfg_max,
        step_min=bounds.step_min, step_max=bounds.step_max,
        device=device,
    )

    return cfg_schedule, step_schedule, eta_schedule, bounds, degree

@torch.no_grad
def precompute(args, prompts:List[str], solver) -> List[torch.Tensor]:
    prompt_emb_set = []
    pooled_emb_set = []
    num_samples = args.num_samples if args.num_samples > 0 else len(prompts)
    for prompt in prompts[:num_samples]:
        # prompt_emb, pooled_emb = solver.encode_prompt(prompt, batch_size=1)
        prompt_emb, pooled_emb = solver.encode_prompt(prompt)
        prompt_emb_set.append(prompt_emb)
        pooled_emb_set.append(pooled_emb)
    return prompt_emb_set, pooled_emb_set

def run(args):
    # load solver
    solver = get_solver(args.method)
    solver.seed = args.seed

    ###### suffix remove ############################
    def sanitize_prompt(s: str, suffix: str) -> str:
        s = s.strip()
        if suffix and s.endswith(suffix):
            s = s[: -len(suffix)].strip()
        return s

    # load text prompts
    prompts = process_text(prompt=args.prompt, prompt_file=args.prompt_file)

    # suffix remove (DIV2K : prompt file // FFHQ : prompt)
    if args.prompt_file is not None:
        prompts = [sanitize_prompt(p, args.prompt_suffix_to_remove) for p in prompts]
    
    solver.text_enc_1.to('cuda')
    solver.text_enc_2.to('cuda')
    solver.text_enc_3.to('cuda')

    if args.efficient_memory:
        # precompute text embedding and remove encoders from GPU
        # This will allow us 1) fast inference 2) with lower memory requirement (<24GB)
        with torch.no_grad():
            prompt_emb_set, pooled_emb_set = precompute(args, prompts, solver)
            null_emb, null_pooled_emb = solver.encode_prompt([''])

        del solver.text_enc_1
        del solver.text_enc_2
        del solver.text_enc_3
        torch.cuda.empty_cache()
        prompt_embs = [[x, y] for x, y in zip(prompt_emb_set, pooled_emb_set)]
    else:
        prompt_embs = [[None, None]] * len(prompts)
        null_embs = [None, None]

    print("Prompts are processed.")
    
    solver.vae.to('cuda')
    solver.transformer.to('cuda')

    #-----------------------problem setup------------------------
    device = solver.vae.device
    img_size = args.img_size

    if args.task == 'cs_walshhadamard':
        compress_by = round(1/args.deg_scale)
        from functions.svd_operators import WalshHadamardCS
        A_funcs = WalshHadamardCS(3, img_size, compress_by,
                                    torch.randperm(img_size ** 2, device=device), device)
    elif args.task == 'cs_blockbased':
        cs_ratio = args.deg_scale
        from functions.svd_operators import CS
        A_funcs = CS(3, img_size, cs_ratio, device)
    elif args.task == 'inpainting':
        from functions.svd_operators import Inpainting
        loaded = np.load("exp/inp_masks/FFHQ_mask.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(3, img_size, missing, device)
    elif args.task == 'inpainting_DIV2K':
        from functions.svd_operators import Inpainting
        loaded = np.load("exp/inp_masks/DIV2k_mask.npy")
        mask = torch.from_numpy(loaded).to(device).reshape(-1)
        missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
        missing_g = missing_r + 1
        missing_b = missing_g + 1
        missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        A_funcs = Inpainting(3, img_size, missing, device)
    elif args.task == 'denoising':
        from functions.svd_operators import Denoising
        A_funcs = Denoising(3, img_size, device)
    elif args.task == 'colorization':
        from functions.svd_operators import Colorization
        A_funcs = Colorization(img_size, device)
    elif args.task == 'sr_averagepooling':
        blur_by = int(args.deg_scale)
        if args.operator_imp == 'SVD':
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(3, img_size, blur_by, device)
        else:
            raise NotImplementedError()

    elif args.task == 'sr_bicubic':
        factor = int(args.deg_scale)
        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0
        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        
        if args.operator_imp == 'SVD':
            from functions.svd_operators import SRConv
            A_funcs = SRConv(kernel / kernel.sum(), 3, img_size, device, stride=factor)
        elif args.operator_imp == 'FFT':                
            from functions.fft_operators import Superres_fft, prepare_cubic_filter
            k = prepare_cubic_filter(1/factor)
            kernel = torch.from_numpy(k).float().to(device)
            A_funcs = Superres_fft(kernel / kernel.sum(), 3, img_size, device, stride=factor)
        else:
            raise NotImplementedError()

    elif args.task == 'deblur_uni':
        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(device), 3, img_size, device)
        elif args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
            A_funcs = Deblurring_fft(torch.Tensor([1 / 9] * 9).to(device), 3, img_size, device)
        else:
            raise NotImplementedError()

    elif args.task == 'deblur_gauss':
        # sigma = 50 # better make argument for kernel type
        sigma = args.deg_scale
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))


        size = 61
        ker = []
        for k in range(-size//2, size//2):
            ker.append(pdf(k))
        kernel = torch.Tensor(ker).to(device)

        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(kernel / kernel.sum(), 3, img_size, device)
        elif args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
            A_funcs = Deblurring_fft(kernel / kernel.sum(), 3, img_size, device)
        else:
            raise NotImplementedError()

    elif args.task == 'deblur_aniso':
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)

        if args.operator_imp == 'SVD':
            from functions.svd_operators import Deblurring2D
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), 3, img_size, device)
        elif args.operator_imp == 'FFT':
            # unlike when using 'SVD' mode, here you can implement any 2D kernel that you want (not just seperable kernels)
            from functions.fft_operators import Deblurring_fft
            kernel = torch.matmul(kernel1[:,None],kernel2[None,:])
            A_funcs = Deblurring_fft(kernel / kernel.sum(), 3, img_size, device)
        else:
            raise NotImplementedError()

    elif args.task == 'deblur_motion':
        from functions.motionblur.motionblur import Kernel
        if args.operator_imp == 'FFT':
            from functions.fft_operators import Deblurring_fft
        else:
            raise ValueError("set operator_imp = FFT")

    else:
        raise ValueError("degradation type not supported")

    # solve problem
    tf = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor()
        ])

    # ---- GRPO ckpt --> final schedule ----
    cfg_schedule, step_schedule, eta_schedule, bounds, degree = load_grpo_schedules_from_ckpt(
        args.grpo_ckpt, NFE=args.NFE, device=device
    )
    print(
        f"[GRPO schedule loaded] ckpt={args.grpo_ckpt} | degree={degree} | "
        f"bounds(cfg=[{bounds.cfg_min},{bounds.cfg_max}], step=[{bounds.step_min},{bounds.step_max}])\n"
        f"  cfg:  min/mean/max = {cfg_schedule.min().item():.3f}/{cfg_schedule.mean().item():.3f}/{cfg_schedule.max().item():.3f}\n"
        f"  step: min/mean/max = {step_schedule.min().item():.3f}/{step_schedule.mean().item():.3f}/{step_schedule.max().item():.3f}\n"
        f"  eta:  min/mean/max = {eta_schedule.min().item():.3f}/{eta_schedule.mean().item():.3f}/{eta_schedule.max().item():.3f}"
    )
    pbar = tqdm(get_img_list(args.img_path), desc="Solving")
    for i, path in enumerate(pbar):
        img_name = path.stem
        img_num = int(img_name)
        prompt_idx = img_num - 1

        img = tf(Image.open(path).convert('RGB'))
        img = img.unsqueeze(0).to(solver.vae.device)
        img = img * 2 - 1
        
        if args.task == 'deblur_motion':
            from functions.motionblur.motionblur import Kernel
            if args.operator_imp == 'FFT':
                from functions.fft_operators import Deblurring_fft
            else:
                raise ValueError("set operator_imp = FFT")
            np.random.seed(seed=i * 10)  # for reproducibility of blur kernel for each image
            kernel = torch.from_numpy(Kernel(size=(args.deg_scale, args.deg_scale), intensity=0.5).kernelMatrix)
            A_funcs = Deblurring_fft(kernel / kernel.sum(), 3, args.img_size, solver.transformer.device)
        
        y = A_funcs.A(img)
        y = y + args.noise_std * torch.randn(y.shape, device=y.device, generator=torch.Generator(y.device).manual_seed(args.seed))
        
        out, images_x0t, images_x0y = solver.sample_final(
            measurement=y,
            operator=A_funcs,
            task=args.task,
            prompts=prompts[i] if len(prompts)>1 else prompts[0],
            NFE=args.NFE,
            img_shape=(args.img_size, args.img_size),
            cfg_scale=args.cfg_scale,
            prompt_embs=prompt_embs[i] if len(prompt_embs)>1 else prompt_embs[0],
            null_embs=null_embs,
            inner_steps=args.inner_steps,
            sigma_y=args.noise_std,
            cfg_schedule=cfg_schedule,
            step_schedule=step_schedule,
            eta_schedule=eta_schedule,
        )

        i = img_num

        if args.task in ['sr_bicubic', 'inpainting', 'inpainting_DIV2K', 'cs_walshhadamard', 'cs_blockbased']: # modify
            save_image(A_funcs.At(y).reshape(img.shape),
                    args.workdir.joinpath(f'input1/{str(i).zfill(4)}.png'),
                    normalize=True)
        else:
            save_image(y.reshape(img.shape),
                    args.workdir.joinpath(f'input1/{str(i).zfill(4)}.png'),
                    normalize=True)

        save_image(out,
                args.workdir.joinpath(f'recon/{str(i).zfill(4)}.png'),
                normalize=True)
        save_image(img,
                args.workdir.joinpath(f'label/{str(i).zfill(4)}.png'),
                normalize=True)

        # Store processes
        fname = str(i).zfill(5) + f'.png'
        images = torch.cat(images_x0t, dim=0)
        grid = make_grid(images, nrow=5, normalize=True)
        to_pil = T.ToPILImage()
        grid_img = to_pil(grid)
        grid_img = grid_img.resize((grid_img.width // 2, grid_img.height // 2))
        grid_img.save(args.workdir.joinpath('process_x0t', fname))

        fname = str(i).zfill(5) + f'.png'
        images = torch.cat(images_x0y, dim=0)
        grid = make_grid(images, nrow=5, normalize=True)
        to_pil = T.ToPILImage()
        grid_img = to_pil(grid)
        grid_img = grid_img.resize((grid_img.width // 2, grid_img.height // 2))
        grid_img.save(args.workdir.joinpath('process_x0y', fname))

        if (i+1) == args.num_samples:
            break

    # ============================
    # Evaluation
    # ============================
    metric = Metric(['psnr', 'lpips_flowdps', 'lpips_flair', 'ssim'])
    recon_path_1 = args.workdir.joinpath('recon')
    label_path = args.workdir.joinpath('label')

    psnr_val_1, lpips_val_1, lpips_val_2, ssim_val_1 = metric(recon_path_1, label_path)

    print(f"\n==== 1st stage Evaluation Results ====")
    print(f"PSNR : {psnr_val_1:.4f} dB")
    print(f"LPIPS (flowdps): {lpips_val_1:.4f}")
    print(f"LPIPS (flair): {lpips_val_2:.4f}")
    print(f"SSIM : {ssim_val_1:.4f}")

    result_file = args.workdir.joinpath("eval_results.txt")
    with open(result_file, "a") as f:
        f.write(f"\n==== {i} sample 1st stage ====\n")
        f.write(f"recon path : {recon_path_1}\n")
        f.write(f"PSNR : {psnr_val_1:.4f} dB\n")
        f.write(f"LPIPS (flowdps): {lpips_val_1:.4f}\n")
        f.write(f"LPIPS (flair): {lpips_val_2:.4f}\n")
        f.write(f"SSIM : {ssim_val_1:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # sampling params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--NFE', type=int, default=28)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--img_size', type=int, default=768)
    # workdir params
    parser.add_argument('--workdir', type=Path, default='workdir_TriPS_G_test_demo')
    # data params
    parser.add_argument('--img_path', type=Path)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=-1)
    # problem params
    parser.add_argument('--task', type=str, default='sr_avgpool')
    parser.add_argument('--method', type=str, default='TriPS_G_test')
    # parser.add_argument('--method', type=str, default='flowchef')
    parser.add_argument('--deg_scale', type=int, default=12)
    parser.add_argument('--noise_std', type=float, default=0.03)
    # solver params
    parser.add_argument('--efficient_memory',default=False, action='store_true')
    parser.add_argument('--attn_enforce', type=float, default=1.3)
    parser.add_argument('--inner_steps', type=int, default=3)
    # Added for operator
    parser.add_argument(
        "--operator_imp", type=str, default="FFT", help="SVD | FFT"  # TODO: add CG support
    )


    parser.add_argument('--prompt_suffix_to_remove', type=str, default=', high-resolution, 8k')

    parser.add_argument(
        "--grpo_ckpt",
        type=str,
        required=True,
        help="Path to trained GRPO schedule ckpt (.pt) saved by train_grpo_schedule.py",
    )
    args = parser.parse_args()

    # workdir creation and seed setup
    set_seed(args.seed)
    args.workdir.joinpath('input1').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('recon').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('label').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('process_x0t').mkdir(parents=True, exist_ok=True)
    args.workdir.joinpath('process_x0y').mkdir(parents=True, exist_ok=True)
    # run main script
    run(args)







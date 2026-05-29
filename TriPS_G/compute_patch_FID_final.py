import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor


# =========================
# Dataset: patch cropping
# =========================
class PatchDataset(Dataset):
    """
    samples: List[Tuple[path, x, y, ps]]
    """
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, x, y, ps = self.samples[idx]
        img = Image.open(p).convert("RGB")
        patch = img.crop((x, y, x + ps, y + ps))
        return self.transform(patch)


def build_patch_coords_per_image(img_paths, patch_size, patches_per_image=None, seed=0):
    """
    FLAIR-style pFID:
    - random patch가 아니라, patch_size stride로 image를 grid로 분할(기본: non-overlap).
    - 예: 768x768, patch_size=256 -> (0,256,512) x (0,256,512) = 9 patches/image
    """
    stride = patch_size  # non-overlapping tiling

    coords_per_img = []
    for p in img_paths:
        img = Image.open(p)
        W, H = img.size
        img.close()

        if W < patch_size or H < patch_size:
            raise ValueError(f"Image too small for patch_size={patch_size}: {p} size={W}x{H}")

        # x/y positions: 기본은 stride 간격. 끝이 딱 안 맞으면 마지막은 right/bottom에 붙여서 포함(결정적)
        xs = list(range(0, W - patch_size + 1, stride))
        ys = list(range(0, H - patch_size + 1, stride))
        if xs[-1] != W - patch_size:
            xs.append(W - patch_size)
        if ys[-1] != H - patch_size:
            ys.append(H - patch_size)

        coords = [(x, y) for y in ys for x in xs]
        coords_per_img.append(coords)

    return coords_per_img


def build_patch_samples(img_paths, coords_per_img, patch_size):
    assert len(img_paths) == len(coords_per_img)
    samples = []
    for i, p in enumerate(img_paths):
        for (x, y) in coords_per_img[i]:
            samples.append((p, x, y, patch_size))
    return samples


def align_by_filename(label_paths, recon_dir):
    """
    label_paths 순서(파일명 기준)를 기준으로 recon_dir의 png를 매칭.
    """
    recon_paths = sorted(Path(recon_dir).glob("*.png"))
    recon_map = {p.name: p for p in recon_paths}

    aligned = []
    missing = []
    for lp in label_paths:
        rp = recon_map.get(lp.name, None)
        if rp is None:
            missing.append(lp.name)
        else:
            aligned.append(rp)

    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} recon files in {recon_dir}. examples: {missing[:5]}")
    return aligned


# =========================
# Inception feature extraction
# =========================
@torch.no_grad()
def extract_inception_features_from_patches(patch_samples, device, batch_size=32, num_workers=4):
    weights = torchvision.models.Inception_V3_Weights.DEFAULT
    tf = weights.transforms()  # includes resize to 299 + normalize

    ds = PatchDataset(patch_samples, tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    model = torchvision.models.inception_v3(weights=weights)
    model.eval().to(device)
    extractor = create_feature_extractor(model, return_nodes={"avgpool": "feat"})

    feats = []
    for x in dl:
        x = x.to(device, non_blocking=True)
        out = extractor(x)["feat"]          # [B, 2048, 1, 1]
        out = out.squeeze(-1).squeeze(-1)   # [B, 2048]
        feats.append(out.cpu())
    return torch.cat(feats, dim=0).numpy()  # [Npatch, 2048] float32/float64 later


# =========================
# FID (Fréchet distance)
# =========================
def compute_stats(feats: np.ndarray):
    feats = feats.astype(np.float64)
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    FID = ||mu1-mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrtm(sigma1*sigma2))
    """
    mu1 = mu1.astype(np.float64)
    mu2 = mu2.astype(np.float64)
    sigma1 = sigma1.astype(np.float64)
    sigma2 = sigma2.astype(np.float64)

    diff = mu1 - mu2

    # sqrtm(sigma1*sigma2)
    try:
        from scipy import linalg
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    except Exception:
        # fallback (less stable): eigen on product
        w, v = np.linalg.eig(sigma1.dot(sigma2))
        w = np.maximum(w.real, 0.0)
        covmean = (v @ np.diag(np.sqrt(w)) @ np.linalg.inv(v)).real

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(fid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_dir", type=str, required=True)
    ap.add_argument("--recon_dir", type=str, required=True)

    ap.add_argument("--max_images", type=int, default=1000, help="use first N images after sorting")
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--patches_per_image", type=int, default=16)
    ap.add_argument("--patch_seed", type=int, default=42)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    args = ap.parse_args()

    label_dir = Path(args.label_dir)
    recon_dir = Path(args.recon_dir)
    assert label_dir.exists(), f"label_dir not found: {label_dir}"
    assert recon_dir.exists(), f"recon_dir not found: {recon_dir}"

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    label_paths = sorted(label_dir.glob("*.png"))[:args.max_images]
    if len(label_paths) == 0:
        raise FileNotFoundError(f"No png images found in label_dir: {label_dir}")

    recon_paths = align_by_filename(label_paths, recon_dir)

    # label 기준 coords 고정 생성
    coords_per_img = build_patch_coords_per_image(
    label_paths,
    patch_size=args.patch_size,
    patches_per_image=args.patches_per_image,
    seed=args.patch_seed
    )

    actual_ppi = len(coords_per_img[0])
    print(f"#images={len(label_paths)}, patches/image={actual_ppi}, total patches={len(label_paths)*actual_ppi}")

    label_patch_samples = build_patch_samples(label_paths, coords_per_img, args.patch_size)
    recon_patch_samples = build_patch_samples(recon_paths, coords_per_img, args.patch_size)

    print(f"#images={len(label_paths)}, patches/image={args.patches_per_image}, total patches={len(label_patch_samples)}")
    print("Extracting Inception features for label patches...")
    real_feats = extract_inception_features_from_patches(
        label_patch_samples, device=device, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("Extracting Inception features for recon patches...")
    fake_feats = extract_inception_features_from_patches(
        recon_patch_samples, device=device, batch_size=args.batch_size, num_workers=args.num_workers
    )

    mu_r, sig_r = compute_stats(real_feats)
    mu_f, sig_f = compute_stats(fake_feats)

    pfid = frechet_distance(mu_r, sig_r, mu_f, sig_f)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! : ", label_dir)
    print(f"patch-FID (ps={args.patch_size}, pp={args.patches_per_image}) = {pfid:.6f}")
    


if __name__ == "__main__":
    main()
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
    stride = patch_size  # non-overlapping tiling

    coords_per_img = []
    for p in img_paths:
        img = Image.open(p)
        W, H = img.size
        img.close()

        if W < patch_size or H < patch_size:
            raise ValueError(f"Image too small for patch_size={patch_size}: {p} size={W}x{H}")

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
    return torch.cat(feats, dim=0).numpy()  # [Npatch, 2048]


# =========================
# KID (Kernel Inception Distance): polynomial MMD^2
# =========================
def _poly_kernel(X, Y, degree=3, gamma=None, coef0=1.0):
    """
    k(x,y) = (gamma * <x,y> + coef0)^degree
    - KID의 통상 설정: degree=3, gamma=1/d, coef0=1
    """
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    d = X.shape[1]
    if gamma is None:
        gamma = 1.0 / d
    return (gamma * (X @ Y.T) + coef0) ** degree


def _mmd2_unbiased_poly(X, Y, degree=3, gamma=None, coef0=1.0):
    """
    Unbiased MMD^2 estimator:
      MMD^2 = 1/(m(m-1)) sum_{i!=j} k(xi,xj)
            + 1/(n(n-1)) sum_{i!=j} k(yi,yj)
            - 2/(mn)     sum_{i,j}  k(xi,yj)
    Assumes X,Y are 2D arrays.
    """
    m = X.shape[0]
    n = Y.shape[0]
    if m < 2 or n < 2:
        return np.nan

    Kxx = _poly_kernel(X, X, degree=degree, gamma=gamma, coef0=coef0)
    Kyy = _poly_kernel(Y, Y, degree=degree, gamma=gamma, coef0=coef0)
    Kxy = _poly_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    # remove diagonals for unbiased within-domain terms
    sum_Kxx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sum_Kyy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    sum_Kxy = Kxy.mean()  # 1/(mn) sum

    return float(sum_Kxx + sum_Kyy - 2.0 * sum_Kxy)


def compute_kid(real_feats, fake_feats, subset_size=1000, n_subsets=100, seed=0,
                degree=3, gamma=None, coef0=1.0):
    """
    KID = average of unbiased MMD^2 over random subsets.
    Returns (mean, std, all_subset_scores)
    """
    real_feats = real_feats.astype(np.float64)
    fake_feats = fake_feats.astype(np.float64)

    N = min(len(real_feats), len(fake_feats))
    if N == 0:
        return float("nan"), float("nan"), np.array([], dtype=np.float64)

    s = int(min(subset_size, N))
    rng = np.random.RandomState(seed)

    scores = []
    for _ in range(int(n_subsets)):
        idx_r = rng.choice(len(real_feats), size=s, replace=False) if len(real_feats) >= s else rng.choice(len(real_feats), size=s, replace=True)
        idx_f = rng.choice(len(fake_feats), size=s, replace=False) if len(fake_feats) >= s else rng.choice(len(fake_feats), size=s, replace=True)

        X = real_feats[idx_r]
        Y = fake_feats[idx_f]
        scores.append(_mmd2_unbiased_poly(X, Y, degree=degree, gamma=gamma, coef0=coef0))

    scores = np.asarray(scores, dtype=np.float64)
    return float(scores.mean()), float(scores.std(ddof=1) if len(scores) > 1 else 0.0), scores


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

    # KID specific
    ap.add_argument("--kid_subset_size", type=int, default=1000)
    ap.add_argument("--kid_subsets", type=int, default=100)
    ap.add_argument("--kid_seed", type=int, default=0)

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

    print("Extracting Inception features for label patches...")
    real_feats = extract_inception_features_from_patches(
        label_patch_samples, device=device, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print("Extracting Inception features for recon patches...")
    fake_feats = extract_inception_features_from_patches(
        recon_patch_samples, device=device, batch_size=args.batch_size, num_workers=args.num_workers
    )

    kid_mean, kid_std, _ = compute_kid(
        real_feats, fake_feats,
        subset_size=args.kid_subset_size,
        n_subsets=args.kid_subsets,
        seed=args.kid_seed,
        degree=3, gamma=None, coef0=1.0
    )

    print(
        f"patch-KID (ps={args.patch_size}, subsets={args.kid_subsets}, subset_size={min(args.kid_subset_size, min(len(real_feats), len(fake_feats)))}) "
        f"= {kid_mean:.6f} ± {kid_std:.6f}"
    )


if __name__ == "__main__":
    main()

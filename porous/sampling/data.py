import csv
import io
import os
import zipfile

import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kw):
        return it


COORD_FIELDS = ["patch_id", "x_start", "y_start", "z_start", "x_end", "y_end", "z_end", "porosity"]
PACK_CONVENTION = "bit 1 = pore, 0 = solid; np.unpackbits(packed, axis=1)[:, :D*H*W].reshape(shape)"


def norm3(v):
    return (int(v),) * 3 if np.ndim(v) == 0 else tuple(int(x) for x in np.asarray(v).reshape(-1))


def load_volume(raw_file_path, dims):
    d, h, w = dims
    print(f"  INFO: Loading volume '{raw_file_path}' ...")
    vol = np.fromfile(raw_file_path, dtype=np.uint8).reshape((d, h, w))
    print(f"  INFO: Volume loaded, shape {vol.shape}")
    return vol


def extract_patches_with_coords(volume, n_patches, patch_size, seed=42):
    np.random.seed(seed)
    d, h, w = volume.shape
    results = []
    for i in tqdm(range(n_patches), desc="  Extracting patches"):
        x = np.random.randint(0, d - patch_size)
        y = np.random.randint(0, h - patch_size)
        z = np.random.randint(0, w - patch_size)
        patch = volume[x : x + patch_size, y : y + patch_size, z : z + patch_size]
        porosity = float(np.mean(patch == 0))
        results.append(
            {
                "patch_id": i,
                "x_start": x,
                "y_start": y,
                "z_start": z,
                "x_end": x + patch_size,
                "y_end": y + patch_size,
                "z_end": z + patch_size,
                "porosity": porosity,
            }
        )
    return results


def write_coords_csv(path, results):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COORD_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in COORD_FIELDS})


def save_npz_lzma(path, **arrays):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA) as zf:
        for k, v in arrays.items():
            buf = io.BytesIO()
            np.save(buf, np.asarray(v), allow_pickle=False)
            zf.writestr(k + ".npy", buf.getvalue())


def pack_pore(pore, chunk=256):
    n = len(pore)
    return np.concatenate(
        [np.packbits(pore[i : i + chunk].reshape(len(pore[i : i + chunk]), -1), axis=1) for i in range(0, n, chunk)], axis=0
    )


def save_binary_npz(path, pore):
    save_npz_lzma(path, packed=pack_pore(pore), shape=np.asarray(pore.shape, np.int64), convention=np.asarray(PACK_CONVENTION))
    return path


def save_real_packed(shared_dir, real_patches):
    path = os.path.join(shared_dir, "real_patches_binary.npz")
    if not os.path.exists(path):
        save_binary_npz(path, real_patches == 0)
    return path


def ensure_real_patches(cfg, dataset_name, volume_dims, patch_size):
    shared_dir = os.path.join(cfg.out_dir, dataset_name, "shared")
    os.makedirs(shared_dir, exist_ok=True)
    patches_path = os.path.join(shared_dir, "real_patches.npy")
    csv_path = os.path.join(shared_dir, "real_samples.csv")
    src = cfg.real_patches_npy or (patches_path if os.path.exists(patches_path) else None)
    if src:
        arr = np.load(src)
        want = (cfg.n_samples, patch_size, patch_size, patch_size)
        if arr.ndim == 4 and tuple(arr.shape[1:]) == want[1:] and len(arr) >= cfg.n_samples:
            print(f"  INFO: real patches loaded from {src}: {arr.shape}")
            return arr[: cfg.n_samples]
        print(f"  WARNING: {src} has shape {arr.shape}, need >= {want}; re-extracting")
    if not cfg.volume:
        raise ValueError(
            "a local volume is required: pass --volume PATH (<Name>.raw, uint8, 0 = pore, 1 = solid) "
            "or --real-patches-npy PATH (N, D, H, W) uint8"
        )
    if not os.path.exists(cfg.volume):
        raise FileNotFoundError(cfg.volume)
    print(f"  INFO: extracting {cfg.n_samples} real patches of {patch_size}^3 for '{dataset_name}' ...")
    volume = load_volume(cfg.volume, tuple(volume_dims))
    results = extract_patches_with_coords(volume=volume, n_patches=cfg.n_samples, patch_size=patch_size, seed=cfg.seed)
    patches = np.stack([volume[r["x_start"] : r["x_end"], r["y_start"] : r["y_end"], r["z_start"] : r["z_end"]] for r in results])
    write_coords_csv(csv_path, results)
    if cfg.save_real_patches:
        np.save(patches_path, patches)
        print(f"  INFO: patches saved -> {patches_path}")
    del volume
    return patches

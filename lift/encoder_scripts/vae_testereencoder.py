# VAE encoder test / train script from Jingwen's implementation

from functools import partial
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomChoice
from torchvision.utils import make_grid
from torch.optim import Adam
from tqdm import trange
from torch.nn.functional import mse_loss
import torch
import argparse
import logging
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
import glob
import h5py
import os
import random
import torch.nn.functional as F

flip_transform = RandomChoice([RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)])

class MultiKeyHDF5ImageDataset(Dataset):
    """
    返回：
      {
        "image": Tensor(C,H,W) 归一化到[-1,1],
        "key":   int (对应第几个类别，按 type_names 顺序)
      }
    """
    def __init__(self, h5_files, type_names, data_key=None, target_size=(64, 64), normalize='sd'):
        import os, h5py, numpy as np, torch
        self.h5_files = h5_files
        self.type_names = list(type_names)   # 例如 ['agentview_image', 'robot0_eye_in_hand_image']
        self.type_to_id = {n: i for i, n in enumerate(self.type_names)}
        self.data_key = data_key             # 若为 None，则逐文件在 type_names 中自动检测
        self.target_size = target_size
        self.normalize = normalize

        # 逐文件检测实际存在的键（优先使用 data_key，否则从 type_names 中找）
        self.index_map = []  # (h5_path, type_name_as_key, idx)
        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    k = None
                    if self.data_key is not None and self.data_key in f:
                        k = self.data_key
                    else:
                        for name in self.type_names:
                            if name in f:   # 该文件包含此键
                                k = name
                                break
                    if k is None:
                        continue
                    n = f[k].shape[0]
                for i in range(n):
                    # type_name 就用实际使用的键名 k（需在 type_names 里）
                    if k in self.type_to_id:
                        self.index_map.append((h5_path, k, i))
            except Exception:
                continue

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        import h5py, numpy as np, torch
        h5_path, type_name, local_idx = self.index_map[idx]
        with h5py.File(h5_path, "r") as f:
            img = f[type_name][local_idx]  # 直接用检测到的键名读取

        im = Image.fromarray(img).convert("RGB")
        im = im.resize(self.target_size, resample=Image.BICUBIC)

        arr = np.array(im)
        x = torch.from_numpy(arr).float() / 255.0
        if self.normalize == 'sd':
            x = x * 2.0 - 1.0
        x = x.permute(2, 0, 1)
        return {"image": x, "key": torch.tensor(self.type_to_id[type_name], dtype=torch.long)}

def train_vae(vae_model, data_loaders, model_save_path, log_images_to_path, epochs, learning_rate, image_keys):
    logger = logging.getLogger('my_logger')
    logger.info("Training started")

    vae_model.train()
    optimizer = Adam(vae_model.parameters(), lr=learning_rate)

    data_loader_train = data_loaders['train']
    data_loader_val = data_loaders['val']
    data_loader_test = data_loaders['test']

    tqdm_epoch = trange(epochs)
    steps = 0

    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in tqdm_epoch:
        torch.cuda.empty_cache()
        avg_loss = 0.0
        num_items = 0

        # -------- Train --------
        for batch in data_loader_train:
            x = batch["image"].to('cuda', non_blocking=True)
            x_hat = vae_model(x).sample
            loss = mse_loss(x, x_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            steps += 1

        train_loss = avg_loss / max(1, num_items)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f}")
        logger.info(f"[Epoch {epoch}] Train Loss: {train_loss:.6f}")

        # Save checkpoint
        torch.save(vae_model.state_dict(), f'{model_save_path}/ckpt_vae_{epoch}.pth')
        logger.info(f'Checkpoint saved at step {steps}')

        # -------- Log grid --------
        test_batch = next(iter(data_loader_test))
        test_images = test_batch["image"]
        with torch.no_grad():
            output = vae_model(test_images.to('cuda')).sample.detach().cpu()
            recon = F.interpolate(output, size=test_images.shape[-2:], mode='bilinear', align_corners=False)


            # 每个样本一行：左原图，右重建
            num_show = min(8, test_images.shape[0])  # 可调整展示数量
            orig = test_images[:num_show]
            recon = recon[:num_show]
            pairwise = torch.stack([orig, recon], dim=1).flatten(0, 1)  # [o0,r0,o1,r1,...]
            grid = make_grid(pairwise, nrow=2, normalize=True, value_range=(-1, 1))
            ToPILImage()(grid).save(f'{log_images_to_path}/grid_{epoch}.png')

        # -------- Validation (per-key and overall) --------
        with torch.no_grad():
            K = len(image_keys)
            val_sums = [0.0] * K
            val_counts = [0] * K
            overall_sum = 0.0
            overall_count = 0

            for batch in data_loader_val:
                x = batch["image"].to('cuda', non_blocking=True)
                key_ids = batch["key"]  # (B,)
                x_hat = vae_model(x).sample

                loss_per_pix = F.mse_loss(x, x_hat, reduction='none')  # (B,3,H,W)
                loss_per_sample = loss_per_pix.view(loss_per_pix.size(0), -1).mean(dim=1).cpu()  # (B,)

                for k in range(K):
                    mask = (key_ids == k)
                    if mask.any():
                        val_sums[k] += loss_per_sample[mask].sum().item()
                        val_counts[k] += int(mask.sum().item())

                overall_sum += loss_per_sample.sum().item()
                overall_count += loss_per_sample.numel()

            val_losses = [val_sums[k] / val_counts[k] if val_counts[k] > 0 else float('nan') for k in range(K)]
            overall_val = overall_sum / max(1, overall_count)

            per_key_str = " | ".join([f"{image_keys[k]}: {val_losses[k]:.6f}" for k in range(K)])
            print(f"[Epoch {epoch}] Validation Loss -> {per_key_str} | Overall: {overall_val:.6f}")
            logger.info(f"[Epoch {epoch}] Validation Loss -> {per_key_str} | Overall: {overall_val:.6f}")

            # 以总体验证损失选最优
            if overall_val < best_val_loss:
                best_val_loss = overall_val
                best_epoch = epoch
                torch.save(vae_model.state_dict(), f'{model_save_path}/best_vae.pth')
                print(f"[Epoch {epoch}] New best! Overall Val: {overall_val:.6f}")
                logger.info(f"[Epoch {epoch}] New best! Overall Val: {overall_val:.6f}")

    logger.info(f"Training completed at step {steps}")
    logger.info(f"The best model is Epoch {best_epoch}, overall_val: {best_val_loss:.6f}")
    print(f"Finished! Best model from Epoch {best_epoch}, overall_val: {best_val_loss:.6f}")

    # -------- Test (per-key and overall) --------
    best_model_state = torch.load(f'{model_save_path}/best_vae.pth', map_location='cuda')
    vae_model.load_state_dict(best_model_state)
    vae_model.eval()
    with torch.no_grad():
        K = len(image_keys)
        test_sums = [0.0] * K
        test_counts = [0] * K
        overall_sum = 0.0
        overall_count = 0

        for batch in data_loaders['test']:
            x = batch["image"].to('cuda', non_blocking=True)
            key_ids = batch["key"]
            x_hat = vae_model(x).sample

            loss_per_pix = F.mse_loss(x, x_hat, reduction='none')
            loss_per_sample = loss_per_pix.view(loss_per_pix.size(0), -1).mean(dim=1).cpu()

            for k in range(K):
                mask = (key_ids == k)
                if mask.any():
                    test_sums[k] += loss_per_sample[mask].sum().item()
                    test_counts[k] += int(mask.sum().item())

            overall_sum += loss_per_sample.sum().item()
            overall_count += loss_per_sample.numel()

        test_losses = [test_sums[k] / test_counts[k] if test_counts[k] > 0 else float('nan') for k in range(K)]
        overall_test = overall_sum / max(1, overall_count)
        per_key_str = " | ".join([f"{image_keys[k]}: {test_losses[k]:.6f}" for k in range(K)])
        print(f"BEST Model Test Loss -> {per_key_str} | Overall: {overall_test:.6f}")
        logger.info(f"BEST Model Test Loss -> {per_key_str} | Overall: {overall_test:.6f}")

def mainLoop():
  import os, glob, random, argparse, logging, h5py


def main():
    import os, glob, random, argparse, logging, h5py
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Train a single VAE on multiple HDF5 image keys with MSE loss')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default=os.path.join(script_dir, 'vae_models'))
    parser.add_argument('--validation_image_out_dir', type=str, default=os.path.join(script_dir, 'vae_val_images'))
    parser.add_argument('--log_dir', type=str, default=os.path.join(script_dir, 'logs'))
    parser.add_argument('--h5_dir', type=str, default=os.path.join(script_dir, 'data'))
    parser.add_argument('--image_keys', type=str, nargs='+', default=['camera0_obs', 'camera1_obs'])
    parser.add_argument('--target_size', type=int, nargs=2, default=[64, 64])
    args = parser.parse_args()

    args.save_path = os.path.abspath(os.path.expanduser(args.save_path))
    args.validation_image_out_dir = os.path.abspath(os.path.expanduser(args.validation_image_out_dir))
    args.log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    args.h5_dir = os.path.abspath(os.path.expanduser(args.h5_dir))

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.validation_image_out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    log_file = os.path.join(args.log_dir, 'single_VAE_train.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger('my_logger')

    vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to('cuda')

    all_h5_files = sorted(glob.glob(os.path.join(args.h5_dir, "*.h5")))
    if not all_h5_files:
        raise FileNotFoundError(f"No .h5 files found under {args.h5_dir}")
    random.seed(42)


    matched_files = {k: [p for p in all_h5_files if k in os.path.basename(p)] for k in args.image_keys}
    counts = {k: len(v) for k, v in matched_files.items()}
    logger.info(f"Matched file counts per image key: {counts}")
    
    for k, n in counts.items():
        if n == 0:
            error_msg = (f"No files matched image key '{k}'. "
                        f"Please check filenames under {args.h5_dir} or set --image_keys correctly.")
            print(f"\nERROR: {error_msg}")
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 构建总数据集：让数据集在 type_names 中自动检测每个文件实际的键
    print("\nBuilding dataset from HDF5 files...")
    logger.info("Building dataset from HDF5 files...")
    
    full_dataset = MultiKeyHDF5ImageDataset(
        all_h5_files,
        type_names=args.image_keys,
        data_key=None,
        target_size=tuple(args.target_size),
        normalize='sd'
    )
    
    if len(full_dataset) == 0:
        # 打印一个样本文件的可用键，便于排查
        sample = all_h5_files[0]
        with h5py.File(sample, 'r') as f:
            available = list(f.keys())
        error_msg = (f"Dataset is empty. Please ensure files contain one of these keys: {args.image_keys}\n"
                    f"Sample file {os.path.basename(sample)} has keys: {available}")
        print(f"\nERROR: {error_msg}")
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    print(f"Dataset created with {len(full_dataset)} total images")

    # 分层切分：按实际使用的键名统计
    def stratified_split_by_key(dataset, keys, train_ratio=0.8, val_ratio=0.1, seed=42):
        rng = random.Random(seed)
        by_key = {k: [] for k in keys}
        for i, (_, type_name, _) in enumerate(dataset.index_map):
            if type_name in by_key:
                by_key[type_name].append(i)
        train_idx, val_idx, test_idx = [], [], []
        for k, idxs in by_key.items():
            rng.shuffle(idxs)
            n = len(idxs)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_idx += idxs[:n_train]
            val_idx   += idxs[n_train:n_train + n_val]
            test_idx  += idxs[n_train + n_val:]
        rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
        return train_idx, val_idx, test_idx

    print("\nSplitting dataset into train/val/test (80%/10%/10%)...")
    train_idx, val_idx, test_idx = stratified_split_by_key(
        full_dataset, args.image_keys, train_ratio=0.8, val_ratio=0.1, seed=42
    )

    from torch.utils.data import Subset
    dataset_train = Subset(full_dataset, train_idx)
    dataset_val   = Subset(full_dataset, val_idx)
    dataset_test  = Subset(full_dataset, test_idx)

    # 记录每个划分的每键样本数
    def count_per_key(dataset_subset, keys):
        counts = {k: 0 for k in keys}
        base = dataset_subset.dataset
        for i in dataset_subset.indices:
            _, type_name, _ = base.index_map[i]
            counts[type_name] += 1
        return counts
    
    train_counts = count_per_key(dataset_train, args.image_keys)
    val_counts = count_per_key(dataset_val, args.image_keys)
    test_counts = count_per_key(dataset_test, args.image_keys)
    
    print(f"Train set: {len(dataset_train)} images - {train_counts}")
    print(f"Val set:   {len(dataset_val)} images - {val_counts}")
    print(f"Test set:  {len(dataset_test)} images - {test_counts}")
    
    logger.info(f"Train per-key counts: {train_counts}")
    logger.info(f"Val per-key counts:   {val_counts}")
    logger.info(f"Test per-key counts:  {test_counts}")

    print("\nCreating data loaders...")
    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_dataloader   = DataLoader(dataset_val,   batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_dataloader  = DataLoader(dataset_test,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    data_loaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

    train_vae(
        vae_model=vae_model,
        data_loaders=data_loaders,
        model_save_path=args.save_path,
        log_images_to_path=args.validation_image_out_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        image_keys=args.image_keys
    )

if __name__ == '__main__':
    main()

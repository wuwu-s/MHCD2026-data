"""SAM3 零样本直测 MHCD2022 全量数据 → 生产 mask 标注"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import models  # noqa: E402  仓库的 register/make 与 sam 模型

DEFAULT_IMG_DIR = "/root/autodl-tmp/Military-Camouflage-MHCD2022/JPEGImages"
DEFAULT_SETS_DIR = "/root/autodl-tmp/Military-Camouflage-MHCD2022/ImageSets/Main"
DEFAULT_MODEL = "/root/autodl-tmp/sam3/sam3.pt"
DEFAULT_CONFIG = os.path.join(REPO, "configs", "cod-sam-vit-l.yaml")
DEFAULT_OUT = "/root/autodl-tmp/sam3_masks_mhcd2022"

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def load_sam3(model_path, model):
    """test.py 同款键映射：兼容官方 sam3.pt 与本仓库训练 checkpoint。"""
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cuda:0")
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 去掉 DDP 的 module. 前缀
    state_dict = {k[7:] if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}

    ref = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("detector.backbone."):
            new_k = k.replace("detector.backbone.", "image_encoder.")
        elif "mask_decoder" in k:
            new_k = "mask_decoder." + k.split("mask_decoder.")[-1]
        elif "pe_layer" in k:
            new_k = "pe_layer." + k.split("pe_layer.")[-1]
        elif "no_mask_embed" in k:
            new_k = "no_mask_embed.weight"
        else:
            new_k = k
        if new_k in ref and v.shape != ref[new_k].shape:
            print(f"Skipping {new_k}: shape mismatch "
                  f"({tuple(v.shape)} vs {tuple(ref[new_k].shape)})")
            continue
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model loaded. missing={len(missing)} unexpected={len(unexpected)}")


class ImgSet(Dataset):
    """只读图像，不做 mask 配对（原始数据只有 box 标注）。"""

    def __init__(self, files, transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        return (self.transform(img), os.path.splitext(os.path.basename(path))[0],
                h, w)


def read_splits(sets_dir):
    """ImageSets/Main/*.txt → {stem: split}；split 取 txt 文件名主干。"""
    stem2split = {}
    for txt in sorted(glob.glob(os.path.join(sets_dir, "*.txt"))):
        split = os.path.splitext(os.path.basename(txt))[0]
        with open(txt) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stem = line.split()[0]
                stem2split.setdefault(stem, split)
    return stem2split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--sets-dir", default=DEFAULT_SETS_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="只取其中 model 段（与原实验同一模型规格）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="OOM 时调小（2 或 1）")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("❌ 无可用 GPU（无卡模式？）。请先在 AutoDL 控制台开机/切到有卡模式再运行。")

    with open(args.config) as f:
        config = yaml.safe_load(f)
    inp_size = config["model"]["args"]["inp_size"]

    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_files.extend(glob.glob(os.path.join(args.img_dir, ext)))
    stem2split = read_splits(args.sets_dir)
    if not stem2split:
        print(f"警告: {args.sets_dir} 下没有找到 txt 划分文件，全部图片归入 all/")

    # 断点续跑：已产出的跳过
    todo = []
    done = 0
    for path in sorted(set(img_files)):
        stem = os.path.splitext(os.path.basename(path))[0]
        split = stem2split.get(stem, "all")
        out_path = os.path.join(args.out, split, "gt", stem + "_mask.png")
        if os.path.exists(out_path):
            done += 1
        else:
            todo.append((path, stem, split, out_path))
    print(f"图片总数 {len(img_files)}，已完成 {done}，待处理 {len(todo)}")
    if not todo:
        print("全部完成，无需处理。")
        return

    model = models.make(config["model"]).cuda().eval()
    load_sam3(args.model, model)

    transform = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    ds = ImgSet([p for p, _, _, _ in todo], transform)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=8,
                        shuffle=False, pin_memory=True)

    os.makedirs(args.out, exist_ok=True)
    cursor = 0
    for imgs, stems, hs, ws in tqdm(loader, desc="sam3 zeroshot"):
        imgs = imgs.cuda()
        with torch.no_grad():
            masks = model.infer(imgs)  # [B,1,1008,1008] logits
        binary = (torch.sigmoid(masks) > 0.5).float()
        for i in range(imgs.shape[0]):
            h, w = int(hs[i]), int(ws[i])
            m = binary[i:i + 1]
            if m.shape[-2] != h or m.shape[-1] != w:
                m = F.interpolate(m, size=(h, w), mode="nearest")
            m = (m[0, 0].cpu().numpy() * 255).astype(np.uint8)
            path, stem, split, out_path = todo[cursor + i]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(m, mode="L").save(out_path)
        cursor += imgs.shape[0]

    print("完成。输出目录:", args.out)


if __name__ == "__main__":
    main()

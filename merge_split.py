"""合并 Camouflage Dataset 与 MHCD 并按 8:2 重新随机划分训练/测试集。

MHCD2026 结构约定：
  MHCD2026/train/Image/*.jpg    MHCD2026/train/gt/*_mask.png (uint8 0/255)
  MHCD2026/test/Image/*.jpg     MHCD2026/test/gt/*_mask.png
"""
import os
import random
import shutil
import numpy as np
from PIL import Image

SEED = 42
TRAIN_RATIO = 0.8

MHCD_DIR = "/root/autodl-tmp/MHCD2026"
CAM_DIR = "/root/autodl-tmp/Camouflage Dataset"
OLD_DIR = "/root/autodl-tmp/MHCD2026_old_20260909"
STAGE_DIR = "/root/autodl-tmp/MHCD2026_stage"

random.seed(SEED)

# ---- 1. 收集 MHCD 原有 2971 对（Image 与 gt 按 _mask 后缀配对）----
pairs = []  # (src_img, src_gt, stem)
for split in ("train", "test"):
    img_dir = os.path.join(MHCD_DIR, split, "Image")
    gt_dir = os.path.join(MHCD_DIR, split, "gt")
    for f in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(f)
        assert ext == ".jpg", f
        gt = os.path.join(gt_dir, stem + "_mask.png")
        assert os.path.exists(gt), gt
        pairs.append((os.path.join(img_dir, f), gt, stem))
print(f"MHCD 原有样本: {len(pairs)}")

# ---- 2. 收集 Camouflage Dataset 2600 对（label 为 bool，需转 uint8 0/255）----
cam_img_dir = os.path.join(CAM_DIR, "img")
cam_lab_dir = os.path.join(CAM_DIR, "label")
for f in sorted(os.listdir(cam_img_dir)):
    stem, ext = os.path.splitext(f)
    assert ext == ".jpg", f
    lab = os.path.join(cam_lab_dir, stem + ".png")
    assert os.path.exists(lab), lab
    pairs.append((os.path.join(cam_img_dir, f), lab, stem))
print(f"Camouflage 样本: {len(pairs) - 2971}, 合计: {len(pairs)}")

# ---- 3. 随机打乱并 8:2 划分 ----
random.shuffle(pairs)
n_total = len(pairs)
n_train = int(round(n_total * TRAIN_RATIO))
train_pairs = pairs[:n_train]
test_pairs = pairs[n_train:]
print(f"划分: train {len(train_pairs)} / test {len(test_pairs)} (比例 {len(train_pairs)/n_total:.4f})")

# ---- 4. 写入 staging 目录 ----
def write_split(split_pairs, split):
    for src_img, src_gt, stem in split_pairs:
        img_out = os.path.join(STAGE_DIR, split, "Image", stem + ".jpg")
        gt_out = os.path.join(STAGE_DIR, split, "gt", stem + "_mask.png")
        shutil.copy2(src_img, img_out)
        m = np.array(Image.open(src_gt))
        if m.dtype == bool or m.max() <= 1:  # 二值 mask 统一为 0/255
            m = (m > 0).astype(np.uint8) * 255
        else:
            m = m.astype(np.uint8)
        Image.fromarray(m).save(gt_out)

if os.path.exists(STAGE_DIR):
    shutil.rmtree(STAGE_DIR)
os.makedirs(os.path.join(STAGE_DIR, "train", "Image"))
os.makedirs(os.path.join(STAGE_DIR, "train", "gt"))
os.makedirs(os.path.join(STAGE_DIR, "test", "Image"))
os.makedirs(os.path.join(STAGE_DIR, "test", "gt"))

write_split(train_pairs, "train")
write_split(test_pairs, "test")

# ---- 5. 划分清单（可复现）----
with open(os.path.join(STAGE_DIR, "train.txt"), "w") as fp:
    fp.write("\n".join(stem for _, _, stem in train_pairs) + "\n")
with open(os.path.join(STAGE_DIR, "test.txt"), "w") as fp:
    fp.write("\n".join(stem for _, _, stem in test_pairs) + "\n")
print("staging 写入完成")

# ---- 6. 换入正式目录：旧划分保留为 MHCD2026_old_20260909 ----
if os.path.exists(OLD_DIR):
    raise RuntimeError(f"{OLD_DIR} 已存在，请先处理")
os.rename(MHCD_DIR, OLD_DIR)
os.rename(STAGE_DIR, MHCD_DIR)
print(f"完成：新 MHCD2026 已生成，旧划分保留在 {OLD_DIR}")

"""从训练集里切出验证集（MHCD2026 三分：train/val/test）。

merge_split.py 最初只做 8:2 train/test 划分；验证集是后来从 train 里
按 seed 42 随机切出 VAL_RATIO=0.2 得到的。

动作：
  - 读 train.txt，随机打乱，取 20% 为 val，其余仍为 train
  - 移动三件套：Image/{stem}.jpg、gt/{stem}_mask.png、prompt/{stem}.json
  - 重写 train.txt / val.txt（test.txt 不动）

注意：
  - clip_cache.npz 的行序与划分对应，切分后需用 scripts/build_clip_cache.py
    重建
  - 结果依赖当前 train 的内容，重跑不会与历史 823 张完全一致，
    只保证同一 seed、同一输入状态下可复现
"""
import os
import random
import shutil

SEED = 42
VAL_RATIO = 0.2

DATA_DIR = "/root/autodl-tmp/MHCD2026"


def main():
    train_txt = os.path.join(DATA_DIR, "train.txt")
    val_txt = os.path.join(DATA_DIR, "val.txt")
    if not os.path.exists(train_txt):
        raise RuntimeError(f"未找到 {train_txt}，请先跑 merge_split.py")
    if os.path.exists(val_txt):
        raise RuntimeError(f"{val_txt} 已存在，已切过或需先处理")

    with open(train_txt) as f:
        stems = [l.strip() for l in f if l.strip()]
    random.shuffle(stems)
    n_val = round(len(stems) * VAL_RATIO)
    val_stems = set(stems[:n_val])
    train_stems = stems[n_val:]
    print(f"train {len(train_stems)} / val {len(val_stems)} "
          f"(从 train 切出 {VAL_RATIO:.0%})")

    def move_files(stem, src_split, dst_split):
        for sub, name in (("Image", stem + ".jpg"),
                          ("gt", stem + "_mask.png"),
                          ("prompt", stem + ".json")):
            src = os.path.join(DATA_DIR, src_split, sub, name)
            if not os.path.exists(src):
                continue
            dst = os.path.join(DATA_DIR, dst_split, sub, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

    for stem in sorted(val_stems):
        move_files(stem, "train", "val")

    with open(train_txt, "w") as f:
        f.write("\n".join(train_stems) + "\n")
    with open(val_txt, "w") as f:
        f.write("\n".join(sorted(val_stems)) + "\n")
    print("完成：val 三件套已移入 val/，train.txt/val.txt 已重写")


if __name__ == "__main__":
    main()

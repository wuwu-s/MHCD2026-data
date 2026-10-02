# MHCD2026 数据与生成管线

## 数据在哪

- **数据集下载**（GitHub Release v1.0，打包 tar.gz）：
  https://github.com/wuwu-s/MHCD2026-data/releases/tag/v1.0
  内容：5132 张（train 3284 / val 823 / test 1025），
  `{split}/Image/` + `{split}/gt/` + `{split}/prompt/` 三件套，
  加 `train.txt`/`val.txt`/`test.txt` 划分清单和 `clip_cache.npz`（CLIP 缓存）
- **原始数据**：
  - MHCD2022：`autodl-tmp/Military-Camouflage-MHCD2022/`（VOC 结构，
    JPEGImages 原图 + Annotations 框标注 + ImageSets 划分，**无 mask**）
  - Camouflage Dataset：`autodl-tmp/Camouflage Dataset/`（img + label，label 为 bool png）
## 用法

```bash
bash run_zeroshot.sh            # 输出目录/日志见脚本头部 OUT 常量
```

## 千问提示词生成（Qwen3-VL-8B）

- **环境**：conda env `qwen`（transformers Qwen3VL）
- **模型/输出**：脚本头部 `MODEL_DIR` / `OUT_ROOT` 常量（默认输出
  `{OUT_ROOT}/{split}/prompt/{img_id}.json`）
- 7 字段：target_hypothesis / camouflage_cues / background_distractors /
  positive_prompt / negative_prompt / boundary_clue / confidence；
  含军事硬约束（禁动物幻觉）；
```bash
python run_qwen_camoreason.py --split all
```

## 输出

```
{OUT}/
├── train/gt/{stem}_mask.png     # split 来自 ImageSets/Main 下 txt 文件名
├── val/gt/...
├── test/gt/...
└── all/gt/...                   # 未出现在任何 txt 里的图片
```

命名 `{stem}_mask.png` 与 `merge_split.py` 的约定一致，可直接作为
`{split}/gt/` 与 `{split}/Image/` 配对使用。

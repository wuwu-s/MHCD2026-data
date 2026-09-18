# SAM3 零样本直测 MHCD2022 → mask 标注
## 背景

MHCD2022 原始标注只有 **box 框**（VOC 结构 `Annotations`）。本实验用官方
SAM3（**无任何 prompt 直推**）对数据集**全部图片**做零样本测试，把预测
mask 作为数据集的 mask 标注——即后来 MHCD2026 各 `split/gt/` 下
`_mask.png` 的来源。

## 文件

| 文件 | 作用 |
|---|---|
| `sam3_zeroshot_annotate.py` | 主脚本：读图 → SAM3 直推 → 二值 mask 存盘 |
| `run_zeroshot.sh` | 一键后台启动（含 GPU 检查，日志写 `{out}/run.log`） |
| `run_qwen_camoreason.py` | Qwen3-VL-8B 生成 7 字段伪装推理提示词（v4 军事约束版，断点续跑） |
| `run_qwen_gtbox.py` | GT 红框版：对拒答/失败样本画红框重生成提示词 |
| `merge_split.py` | 合并 MHCD + Camouflage Dataset，seed 42 随机 8:2 划分 train/test（原版 09-09） |
| `split_val.py` | 从 train 里按 seed 42 切出 20% 为 val（移动三件套 + 重写 train.txt/val.txt） |
| `README.md` | 本文档 |

路径与超参均写在脚本头部常量/命令行参数里（模型路径、输出目录、采样参数）。

## 协议（与原实验 test.py + configs/cod-sam-vit-l.yaml 一致）

- 模型规格取自 `../configs/cod-sam-vit-l.yaml` 的 `model` 段：官方 K=4
  mask decoder，`inp_size=1008`
- 图像归一化：ImageNet `(0.485,0.456,0.406)/(0.229,0.224,0.225)`（与仓库
  val wrapper 相同）
- 推理：`model.infer()` —— 空 sparse prompt + `no_mask_embed` 稠密嵌入，
  `multimask_output=False`（单输出 token 0），即"原生 SAM3 零样本直测"
- 后处理：`sigmoid → >0.5 二值化 → NEAREST 还原原始分辨率 → uint8 0/255 PNG`
- 权重加载：与 `test.py` 相同的键映射（兼容官方 `sam3.pt`）

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
  v4 含军事硬约束（禁动物幻觉）；断点续跑（已存在 json 跳过）
- 拒答/失败样本修复：先准备 GT 红框图，再用 `run_qwen_gtbox.py`

```bash
python run_qwen_camoreason.py --split all
# 冒烟：--max-images 5；调试：--debug
```

> 注意：当前 MHCD2026 的 prompt 已全部生成完毕，直接跑基本是 no-op。

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

## 注意事项

- **断点续跑**：已存在的输出自动跳过，中断后重跑即可
- **GPU**：需有卡模式（脚本与 sh 都会检查）；OOM 时用
  `--batch-size 2` 或 `1`
- 推理无随机性（无采样/无 prompt），重跑结果与原始实验一致

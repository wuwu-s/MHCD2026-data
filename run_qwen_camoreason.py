"""CamoReason · Qwen3-VL-8B-Instruct 结构化伪装推理提示生成（2026-09-09 v3）

严格按官方 README quickstart 的用法：
  - processor.apply_chat_template(tokenize=True) 一步出 inputs（含图像），
    不再用 process_vision_info + processor(text=, images=) 拼接（v1/v2 失败疑点）
  - 采样参数按 README VL 推荐：do_sample=True, top_p=0.8, top_k=20, temperature=0.7,
    repetition_penalty=1.0, presence_penalty=1.5
  - dtype="auto", device_map="auto"，与官方示例一致
  - 单张顺序跑、不 padding（用户拍板：正确优先，不用批量）输出: {out_root}/{split}/prompt/{img_id}.json（7 字段结构化 JSON）：
  target_hypothesis / camouflage_cues / background_distractors /
  positive_prompt / negative_prompt / boundary_clue / confidence
断点续跑：已存在 {img_id}.json 的跳过；解析失败仅屏幕告警，不生成任何 txt/log 文件。
"""
import os
import re
import json
import time
import argparse
import traceback

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_DIR = '/root/autodl-tmp/Qwen3-VL-8B'
OUT_ROOT = '/root/autodl-tmp/MHCD2026'
MAX_SIDE = 1024          # 原图长边上限，控制显存
MAX_NEW_TOKENS = 768     # 结构化 JSON 生成上限

# 采样模式下模型偶发回显指令文本再输出 JSON（echo），extract_json 会从尾部救出 JSON；
# 字段不全（如被 768 token 截断）视同解析失败，走严格重试，防止截断 JSON 被静默保存
REQUIRED_KEYS = {'target_hypothesis', 'camouflage_cues', 'background_distractors',
                 'positive_prompt', 'negative_prompt', 'boundary_clue', 'confidence'}

# README VL 推荐超参（greedy=false 即采样）；
# 注: presence_penalty 是 Qwen 服务端参数，transformers generate() 不接受，故省略
GEN_KW = dict(do_sample=True, top_p=0.8, top_k=20, temperature=0.7,
              repetition_penalty=1.0)

SYSTEM = ('You are an expert camouflage analyst. You analyze images with concealed '
          'targets and output structured camouflage-reasoning descriptions in strict '
          'JSON. Never output anything except the JSON.')

FIELD_SPEC = (
    'Analyze how the target(s) are camouflaged in this scene, then output ONLY a '
    'strict JSON object (no markdown fences, no extra text) with exactly these keys:\n'
    '- "target_hypothesis": what the camouflaged target most likely is '
    '(category and appearance; if multiple targets, describe all of them)\n'
    '- "camouflage_cues": object with keys "color_similarity", '
    '"texture_continuity", "broken_boundary", "shape_anomaly", each a short '
    'phrase describing the camouflage strategy\n'
    '- "background_distractors": array of background elements that visually '
    'distract from the target\n'
    '- "positive_prompt": short concept phrase describing the target itself\n'
    '- "negative_prompt": short phrase listing background/distractor concepts '
    'to reject\n'
    '- "boundary_clue": short description of where the target boundary is most '
    'inconsistent or detectable\n'
    '- "confidence": one of "low", "medium", "high"')


# 数据集为纯军事伪装（无人/动物自然伪装图），v4 加入硬约束防止动物类幻觉
MILITARY_CONSTRAINT = (
    'IMPORTANT: This is a MILITARY camouflage scene from a military camouflage '
    'detection dataset. The camouflaged target is ALWAYS a person, vehicle, '
    'equipment, or other man-made object — NEVER an animal, bird, insect, or '
    'wildlife. Do not mention any animal, bird, or wildlife in ANY field.\n')


def make_user_prompt():
    return ('The image contains one or more camouflaged targets.\n'
            + MILITARY_CONSTRAINT + FIELD_SPEC)


def extract_json(text):
    """从模型输出里抠出 JSON 对象（剥 markdown 围栏、找首个平衡大括号）"""
    t = text.strip()
    t = re.sub(r'^```(?:json)?\s*', '', t)
    t = re.sub(r'\s*```$', '', t)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find('{')
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except Exception:
                    return None
    return None


def build_messages(args, split, img_id, strict=False):
    img_path = os.path.join(args.out_root, split, 'Image', f'{img_id}.jpg')
    im = Image.open(img_path).convert('RGB')
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    prompt = make_user_prompt()
    if strict:
        prompt = 'OUTPUT ONLY A VALID JSON OBJECT, NO OTHER TEXT. ' + prompt
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': prompt}]}]


def run_forward(model, processor, messages, debug=False):
    """官方 quickstart 模式：单张、无 padding（用户拍板：一张一张跑，正确优先）"""
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt')
    inputs = inputs.to(model.device)   # 官方模式：跟 device_map="auto" 对齐，不硬编码 "cuda"
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **GEN_KW)
    if debug:
        print(f'    [debug] 输入 {tuple(inputs["input_ids"].shape)} → 输出 {tuple(out.shape)}',
              flush=True)
    # 注意：batch_decode 吃"序列的列表"；1D 切片会被当成逐 token 的序列列表，
    # 取 [0] 只剩第一个 token（v1/v2 "生成 1 字符"的根因）。官方 README 模式：包成列表。
    raw = processor.batch_decode(
        [out[0][inputs['input_ids'].shape[1]:]], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    if debug:
        print(f'    [debug] 生成 {len(raw.strip())} 字符: {raw[:300]!r}', flush=True)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    ap.add_argument('--max-images', type=int, default=0, help='只跑前 N 张（冒烟）')
    ap.add_argument('--out-root', default=OUT_ROOT)
    ap.add_argument('--debug', action='store_true', help='打印原始生成内容与张量形状')
    args = ap.parse_args()

    splits = ['train', 'test'] if args.split == 'all' else [args.split]

    print('=== 加载 Qwen3-VL-8B-Instruct（官方 quickstart 模式）===', flush=True)
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype='auto', device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    processor.tokenizer.padding_side = 'left'   # decoder-only 批量要求左填充
    print(f'模型加载 {time.time()-t0:.0f}s，GPU 显存 {torch.cuda.max_memory_allocated()/1e9:.1f}GB',
          flush=True)

    total_done, n_err = 0, 0
    failed_ids = []
    for split in splits:
        split_txt = os.path.join(args.out_root, f'{split}.txt')
        if os.path.exists(split_txt):
            ids = [l.strip() for l in open(split_txt)]
        else:   # 清单缺失时从 Image/ 目录推导
            ids = sorted(f[:-4] for f in os.listdir(
                os.path.join(args.out_root, split, 'Image'))
                if f.endswith('.jpg'))
        if args.max_images > 0:
            ids = ids[:args.max_images]
        out_dir = os.path.join(args.out_root, split, 'prompt')
        os.makedirs(out_dir, exist_ok=True)

        todo = [img_id for img_id in ids
                if not os.path.exists(os.path.join(out_dir, f'{img_id}.json'))]  # 断点续跑

        for n_done, img_id in enumerate(todo, 1):
            try:
                raw = run_forward(model, processor,
                                  build_messages(args, split, img_id, False),
                                  debug=args.debug)
            except Exception:
                print(f'  {img_id}: 前向异常\n{traceback.format_exc()}', flush=True)
                n_err += 1
                failed_ids.append(img_id)
                continue
            parsed = extract_json(raw)
            if parsed is not None and not REQUIRED_KEYS.issubset(parsed):
                parsed = None   # 字段不全（如 echo 挤爆 token 上限致截断）视同失败
            if parsed is None:
                # 单张严格重试一次（同 README 采样参数）
                try:
                    raw = run_forward(model, processor,
                                      build_messages(args, split, img_id, True),
                                      debug=args.debug)
                    parsed = extract_json(raw)
                    if parsed is not None and not REQUIRED_KEYS.issubset(parsed):
                        parsed = None
                except Exception:
                    pass
            if parsed is None:
                print(f'  {img_id}: JSON 解析失败 | 原始输出: '
                      f'{raw.strip()[:200]!r}', flush=True)
                n_err += 1
                failed_ids.append(img_id)
                continue
            rec = {'img_id': img_id, 'split': split, **parsed}
            json.dump(rec, open(os.path.join(out_dir, f'{img_id}.json'), 'w'),
                      ensure_ascii=False, indent=1)
            total_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                el = (time.time() - t0) / 60
                print(f'  [{split}] {n_done}/{len(todo)} | 累计 {total_done} | '
                      f'异常 {n_err} | {el:.1f}min | '
                      f'均速 {el*60/max(total_done,1):.1f}s/张 | '
                      f'显存 {torch.cuda.max_memory_allocated()/1e9:.1f}GB', flush=True)

    el = (time.time() - t0) / 60
    print(f'完成 {total_done} 张（异常 {n_err}），{el:.1f}min', flush=True)
    if failed_ids:
        print('失败列表（前 50）:', ', '.join(failed_ids[:50]), flush=True)
        print(f'失败总数: {len(failed_ids)}，重跑时断点续跑会自动重试', flush=True)
    print(f'输出: {args.out_root}/{{split}}/prompt/{{img_id}}.json', flush=True)


if __name__ == '__main__':
    main()

"""CamoReason v5 · 带 GT 红框的 Qwen3-VL 结构化提示生成（仅用于拒答文件补跑）

与 run_qwen_camoreason.py 的差异：
  - 输入图来自 {split}/Image_gtbox/（原图 + GT 掩码 bbox 红框）
  - 提示中告知模型：红框内就是伪装目标，必须描述框内内容，禁止回答 no target
  - 其余（军事约束、7 字段 JSON、断点续跑）与 v4 相同
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
MAX_SIDE = 1024
MAX_NEW_TOKENS = 768

REQUIRED_KEYS = {'target_hypothesis', 'camouflage_cues', 'background_distractors',
                 'positive_prompt', 'negative_prompt', 'boundary_clue', 'confidence'}

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

BOX_HINT = (
    'A red rectangle in the image marks the exact location of the camouflaged '
    'target. The target IS inside the red rectangle — it definitely exists. '
    'Look closely at the content inside the red rectangle and describe it. '
    'Never say "no target detected" or that no target exists.')

MILITARY_CONSTRAINT = (
    'IMPORTANT: This is a MILITARY camouflage scene from a military camouflage '
    'detection dataset. The camouflaged target is ALWAYS a person, vehicle, '
    'equipment, or other man-made object — NEVER an animal, bird, insect, or '
    'wildlife. Do not mention any animal, bird, or wildlife in ANY field.')


def make_user_prompt():
    return ('The image contains one or more camouflaged targets.\n'
            + BOX_HINT + '\n' + MILITARY_CONSTRAINT + '\n' + FIELD_SPEC)


def extract_json(text):
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
    img_path = os.path.join(args.out_root, split, 'Image_gtbox', f'{img_id}.jpg')
    im = Image.open(img_path).convert('RGB')
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    prompt = make_user_prompt()
    if strict:
        prompt = 'OUTPUT ONLY A VALID JSON OBJECT, NO OTHER TEXT. ' + prompt
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': prompt}]}]


def run_forward(model, processor, messages, debug=False):
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt')
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, **GEN_KW)
    if debug:
        print(f'    [debug] 输入 {tuple(inputs["input_ids"].shape)} → 输出 {tuple(out.shape)}',
              flush=True)
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

    print('=== 加载 Qwen3-VL-8B-Instruct（GT 红框模式）===', flush=True)
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype='auto', device_map='auto')
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    processor.tokenizer.padding_side = 'left'
    print(f'模型加载 {time.time()-t0:.0f}s，GPU 显存 {torch.cuda.max_memory_allocated()/1e9:.1f}GB',
          flush=True)

    total_done, n_err = 0, 0
    failed_ids = []
    for split in splits:
        ids = []
        for f in sorted(os.listdir(os.path.join(args.out_root, split, 'Image_gtbox'))):
            if f.endswith('.jpg'):
                ids.append(f[:-4])
        if args.max_images > 0:
            ids = ids[:args.max_images]
        out_dir = os.path.join(args.out_root, split, 'prompt')
        os.makedirs(out_dir, exist_ok=True)

        todo = [img_id for img_id in ids
                if not os.path.exists(os.path.join(out_dir, f'{img_id}.json'))]

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
                parsed = None
            if parsed is None:
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
            if n_done % 5 == 0 or n_done == len(todo):
                el = (time.time() - t0) / 60
                print(f'  [{split}] {n_done}/{len(todo)} | 累计 {total_done} | '
                      f'异常 {n_err} | {el:.1f}min | '
                      f'均速 {el*60/max(total_done,1):.1f}s/张', flush=True)

    el = (time.time() - t0) / 60
    print(f'完成 {total_done} 张（异常 {n_err}），{el:.1f}min', flush=True)
    if failed_ids:
        print('失败列表:', ', '.join(failed_ids), flush=True)
    print(f'输出: {args.out_root}/{{split}}/prompt/{{img_id}}.json', flush=True)


if __name__ == '__main__':
    main()

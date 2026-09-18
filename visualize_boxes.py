"""MHCD2022 VOC 框标注可视化：把 Annotations/*.xml 的 bndbox 叠加画在原图上"""

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

DEFAULT_ANN_DIR = "/root/autodl-tmp/Military-Camouflage-MHCD2022/Annotations"
DEFAULT_IMG_DIR = "/root/autodl-tmp/Military-Camouflage-MHCD2022/JPEGImages"
DEFAULT_OUT = "/root/autodl-tmp/Military-Camouflage-MHCD2022/Visualization"

# 每类固定颜色（BGR 风格无要求，这里是 PIL 的 RGB）
CLASS_COLORS = {
    "person": (255, 0, 0),           # 红
    "tank": (0, 200, 0),             # 绿
    "aeroplane": (0, 128, 255),      # 蓝
    "military vehicle": (255, 128, 0),  # 橙
    "warship": (255, 0, 255),        # 品红
}
DEFAULT_COLOR = (255, 255, 0)        # 未知类别用黄色

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


_FONT_CACHE = {}


def load_font(size):
    """优先系统 TTF，找不到退回 PIL 默认字体；按字号缓存。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size=size)
            break
    else:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def parse_xml(xml_path):
    """返回 [(name, xmin, ymin, xmax, ymax), ...]，无 object 时为空列表。"""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or "unknown"
        bb = obj.find("bndbox")
        boxes.append((name,
                      int(float(bb.findtext("xmin"))),
                      int(float(bb.findtext("ymin"))),
                      int(float(bb.findtext("xmax"))),
                      int(float(bb.findtext("ymax")))))
    return boxes


def find_image(img_dir, stem):
    """按 xml 主干名找原图（支持 jpg/jpeg/png）。"""
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(img_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def draw_boxes(img, boxes):
    """在原图上画框 + 类别标签，返回 PIL 图。"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    lw = max(2, round(min(w, h) / 300))  # 线宽随图大小自适应
    font = load_font(max(12, round(min(w, h) / 25)))  # 字号随图大小自适应

    for name, xmin, ymin, xmax, ymax in boxes:
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=lw)

        # 类别标签：文字底下垫同色实心块保证可读
        text = name
        tb = draw.textbbox((xmin, ymin), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = ymin - th - 2 * lw
        if ty < 0:
            ty = ymin + lw  # 框贴顶时标签画进框内
        draw.rectangle([xmin, ty, xmin + tw + 2 * lw, ty + th + 2 * lw],
                       fill=color)
        draw.text((xmin + lw, ty + lw), text, fill=(255, 255, 255),
                  font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default=DEFAULT_ANN_DIR)
    ap.add_argument("--img-dir", default=DEFAULT_IMG_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    xml_files = sorted(glob.glob(os.path.join(args.ann_dir, "*.xml")))
    if not xml_files:
        sys.exit(f"❌ {args.ann_dir} 下没有找到 xml 标注文件")

    os.makedirs(args.out, exist_ok=True)

    done = miss = 0
    stats = {}
    todo = []
    for xml_path in xml_files:
        stem = os.path.splitext(os.path.basename(xml_path))[0]
        out_path = os.path.join(args.out, stem + ".jpg")
        if os.path.exists(out_path):
            done += 1
            continue
        img_path = find_image(args.img_dir, stem)
        if img_path is None:
            miss += 1
            print(f"警告: {stem} 找不到对应原图，跳过")
            continue
        todo.append((xml_path, img_path, out_path))

    print(f"xml 总数 {len(xml_files)}，已完成 {done}，缺原图 {miss}，"
          f"待处理 {len(todo)}")

    for xml_path, img_path, out_path in tqdm(todo, desc="draw boxes"):
        boxes = parse_xml(xml_path)
        img = Image.open(img_path).convert("RGB")
        for name, *_ in boxes:
            stats[name] = stats.get(name, 0) + 1
        draw_boxes(img, boxes).save(out_path, quality=95)

    print("完成。输出目录:", args.out)
    if stats:
        print("各类别框数:", ", ".join(f"{k}×{v}" for k, v in
                                        sorted(stats.items())))


if __name__ == "__main__":
    main()
